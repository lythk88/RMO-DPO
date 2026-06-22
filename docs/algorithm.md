# Algorithm 1 compliance notes

This repository implements the manuscript's RMO-DPO training loop directly in `scripts/train_rmo_dpo.py`. The script does not train or call an online preference predictor.

## Data object

The per-objective dataset rows are stored as

```text
Z_i = (x, y_a, y_b, b_i)
```

where `b_i=1` means `response_a` is preferred and `b_i=0` means `response_b` is preferred.

## Loss equations

`src/rmo_dpo/losses.py` implements the DPO margin

```text
Delta_theta(x, y_a, y_b)
  = log pi_theta(y_a|x)/pi_ref(y_a|x)
  - log pi_theta(y_b|x)/pi_ref(y_b|x)
```

and the objective-specific DPO loss

```text
ell_i(theta; z)
  = -b_i log sigmoid(beta_i Delta_theta)
    -(1-b_i) log sigmoid(-beta_i Delta_theta).
```

This is Eq. (11)--Eq. (12), including both label cases.

## DRO equations

For each objective, the training loop computes raw adversarial weights before the eta update:

```text
omega_i,k = (f_i^*)'((ell_i(theta; Z_i,k) - eta_i) / lambda_i)
```

Then it updates the threshold with

```text
eta_i <- eta_i - alpha_eta * (1 - mean_k omega_i,k)
```

Then it clips and normalizes with

```text
bar_omega_i,k = min(omega_i,k, omega_max)
              / (mean_r min(omega_i,r, omega_max) + 1e-12).
```

The objective gradient is computed as

```text
g_i = grad_theta mean_k [bar_omega_i,k * ell_i(theta; Z_i,k)].
```

The `bar_omega` tensor is detached before autograd, giving Eq. (21) rather than differentiating through the adversary.

## Conflict update

The script stacks objective gradients into `G_t` and then calls `combine_gradients`:

- `conflict: mgda` solves Eq. (26).
- `conflict: clip` computes Eq. (27), solves Eq. (28), clips `p_t` against user weights, and computes Eq. (29).

## Policy step

With the default config, `optimizer.name: sgd`, the update is exactly

```text
theta <- theta - alpha_theta d_t.
```

`adamw` is available only as a practical variant.
