# RMO-DPO on HelpSteer2, strict Algorithm 1 implementation

Reference implementation for **Distributionally Robust Multi-Objective Direct Preference Optimization** on HelpSteer2. This version removes the online preference predictor path and trains only from objective-specific preference data, the reference policy, the dual thresholds, and the policy update specified in Algorithm 1.

Configured first experiment:

- **Dataset:** `nvidia/HelpSteer2`
- **Objectives:** helpfulness, correctness, coherence, complexity, verbosity
- **Policy:** `Qwen/Qwen2.5-7B-Instruct`
- **Reference policy:** `Qwen/Qwen2.5-7B-Instruct`
- **No predictor:** no `Qwen2.5-3B` preference predictor is trained or used

## Repository layout

```text
configs/
  helpsteer2_rmo_dpo.yaml               # main strict Algorithm 1 RMO-DPO experiment
  helpsteer2_mo_dpo_baseline.yaml       # no-DRO multi-objective baseline
  helpsteer2_weighted_dpo_baseline.yaml # weighted DPO baseline
scripts/
  prepare_helpsteer2.py                 # builds z=(x, y_a, y_b, b_i) JSONL files
  train_rmo_dpo.py                      # explicit line-by-line Algorithm 1 loop
  evaluate_helpsteer2.py                # validation metrics per objective
  run_helpsteer2_first.sh               # prepare -> train -> evaluate
src/rmo_dpo/
  losses.py                             # Eq. (11), Eq. (12), Eq. (17), Eq. (18), Eq. (22)
  conflict.py                           # Eq. (26) MGDA and Eq. (27)--Eq. (29) Clip
  trainer.py                            # Algorithm 1 helper operations
  data.py                               # HelpSteer2 objective datasets and collation
  models.py                             # Qwen2.5 loading and LoRA/reference handling
```

## Installation

Use Python 3.10+ and a CUDA-capable PyTorch build.

```bash
git clone <this-repo>
cd rmo-dpo-helpsteer2-strict
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
huggingface-cli login
```

## Run HelpSteer2 first

```bash
# 1. Build objective-specific z=(x, y_a, y_b, b_i) preference data.
python scripts/prepare_helpsteer2.py \
  --output_dir data/helpsteer2_pairs \
  --min_score_gap 1 \
  --seed 42

# 2. Train RMO-DPO. No preference predictor step is needed.
python scripts/train_rmo_dpo.py \
  --config configs/helpsteer2_rmo_dpo.yaml

# 3. Evaluate the final adapter.
python scripts/evaluate_helpsteer2.py \
  --config configs/helpsteer2_rmo_dpo.yaml \
  --checkpoint outputs/helpsteer2_rmo_dpo_qwen2p5_7b/final
```

Equivalent launcher:

```bash
bash scripts/run_helpsteer2_first.sh
```

## Algorithm 1 mapping

`scripts/train_rmo_dpo.py` is intentionally written as a direct readable version of Algorithm 1.

| Algorithm 1 line | Code location |
|---|---|
| Initialize policy parameters `theta_0` and dual thresholds `eta_i,0` | `RMODPOTrainer(cfg)` in `scripts/train_rmo_dpo.py` |
| For `t = 0, ..., T-1` | main `for t in trange(...)` loop |
| For each objective `i` | loop over `trainer.objectives` |
| Sample minibatch `B_i,t={(x,y_a,y_b,b_i)}` | `trainer.sample_minibatch(objective)` |
| Compute DPO losses Eq. (12) | `trainer.compute_dpo_losses(...)`; implemented in `losses.dpo_loss_from_logps` |
| Compute adversarial weights Eq. (18) | `trainer.compute_adversarial_weights(...)`; implemented in `losses.raw_dro_weights` |
| Update `eta_i,t` Eq. (17) | `trainer.update_dual_threshold(...)`; implemented in `losses.eta_sgd_update` |
| Clip and renormalize Eq. (22) | `trainer.clip_and_renormalize_weights(...)` |
| Compute `g_Bi(theta_t)` Eq. (21) | `trainer.compute_objective_gradient(...)` via `mean(bar_omega.detach() * loss)` |
| Form `G_t` | `torch.stack(objective_grads, dim=0)` |
| Compute `d_t` Eq. (26)--Eq. (29) | `trainer.compute_update_direction(...)` and `src/rmo_dpo/conflict.py` |
| Update `theta_{t+1}=theta_t-alpha_theta d_t` | `trainer.policy_update(...)`, default `optimizer.name: sgd` |
| Return `pi_theta_T` | save final adapter/model to `outputs/.../final` |

## Equation matching details

### Eq. (11) and Eq. (12): objective-specific DPO

The data collator returns response side `a`, response side `b`, and label `b_i`:

```text
z = (x, y_a, y_b, b_i)
b_i = 1: y_a preferred to y_b
b_i = 0: y_b preferred to y_a
```

`losses.dpo_loss_from_logps` computes:

```text
Delta_theta = [log pi_theta(y_a|x) - log pi_ref(y_a|x)]
            - [log pi_theta(y_b|x) - log pi_ref(y_b|x)]

ell_i = -b_i log sigmoid(beta_i Delta_theta)
        -(1-b_i) log sigmoid(-beta_i Delta_theta)
```

### Eq. (17), Eq. (18), Eq. (22), Eq. (21): objective-wise DRO

For each objective and minibatch:

```text
omega_i,k = (f_i*)'((ell_i(theta; Z_i,k) - eta_i) / lambda_i)
eta_i <- eta_i - alpha_eta * (1 - mean_k omega_i,k)
bar_omega_i,k = min(omega_i,k, omega_max) / (mean_r min(omega_i,r, omega_max) + 1e-12)
g_i = grad_theta mean_k [bar_omega_i,k * ell_i(theta; Z_i,k)]
```

The weights are detached before the gradient call, so the computed gradient is exactly the empirical adversarial gradient estimator in Eq. (21), not a gradient through the weight function.

Supported divergences:

```text
KL:   omega = exp((loss - eta) / lambda)
chi2: omega = relu(1 + (loss - eta) / lambda)
none: omega = 1
```

### Eq. (26)--Eq. (29): conflict resolution

Set `rmo_dpo.conflict: mgda` to use Eq. (26), or `rmo_dpo.conflict: clip` to use Eq. (27)--Eq. (29). The default main config uses `clip` for a user-weighted frontier point.

### Exact line 13 policy step

The main config uses:

```yaml
optimizer:
  name: sgd
  lr: 5.0e-6
  max_grad_norm: null
```

This performs the literal update `theta <- theta - alpha_theta d_t`. Setting `optimizer.name: adamw` is supported as a practical variant, but it is not the strict Algorithm 1 update.

## HelpSteer2 conversion

`prepare_helpsteer2.py` groups HelpSteer2 rows by prompt. For every response pair under the same prompt and every objective, it writes:

```json
{
  "prompt": "...",
  "response_a": "...",
  "response_b": "...",
  "preference_label": 1,
  "objective": "helpfulness"
}
```

Higher attribute scores are treated as preferred by default. Equal-score pairs are skipped. Optional random label noise flips `preference_label`, which matches the paper's label-noise experiment plan.

```bash
python scripts/prepare_helpsteer2.py \
  --output_dir data/helpsteer2_pairs_noisy \
  --noise helpfulness=0.05,correctness=0.30,coherence=0.10,complexity=0.25,verbosity=0.15
```

## Memory notes

The default config uses QLoRA and does not instantiate a second Qwen2.5-7B model for the reference policy. For PEFT/LoRA, the reference policy is evaluated by temporarily disabling the adapter, so the frozen base model acts as `pi_ref`.

Typical starting point:

- 1× A100 40GB or 80GB: use default config.
- 1× 24GB GPU: reduce `max_length`, keep batch size 1, and consider LoRA `r: 16`.

## Baselines

```bash
python scripts/train_rmo_dpo.py --config configs/helpsteer2_weighted_dpo_baseline.yaml
python scripts/train_rmo_dpo.py --config configs/helpsteer2_mo_dpo_baseline.yaml
```
