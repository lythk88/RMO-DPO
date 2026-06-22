from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOBatchLogps:
    """Sequence log-probabilities for the two responses in z=(x, y_a, y_b, b_i)."""

    a: torch.Tensor
    b: torch.Tensor


@dataclass
class DPOResult:
    """Per-example quantities for Eq. (11) and Eq. (12)."""

    losses: torch.Tensor
    response_a_rewards: torch.Tensor
    response_b_rewards: torch.Tensor
    margins: torch.Tensor
    labels: torch.Tensor
    accuracy: torch.Tensor


def sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return summed log p(response | prompt) for each sequence.

    ``labels`` must be -100 outside the response tokens. Only response tokens
    contribute to the sequence log-probability, matching the DPO convention.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    logps = F.log_softmax(shifted_logits.float(), dim=-1)
    token_logps = torch.gather(logps, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask.float()).sum(dim=-1)


def pair_log_probs(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> DPOBatchLogps:
    """Compute log pi(y_a|x) and log pi(y_b|x) for a minibatch.

    The preferred response is not assumed to be side ``a``. The preference label
    b_i decides the orientation in Eq. (12). For backwards compatibility with an
    older chosen/rejected JSONL schema, chosen/rejected tensors are accepted and
    interpreted as y_a/y_b with b_i=1 by the collator.
    """
    prefix_a = "response_a" if "response_a_input_ids" in batch else "chosen"
    prefix_b = "response_b" if "response_b_input_ids" in batch else "rejected"
    logp_a = sequence_log_probs(
        model,
        batch[f"{prefix_a}_input_ids"],
        batch[f"{prefix_a}_attention_mask"],
        batch[f"{prefix_a}_labels"],
    )
    logp_b = sequence_log_probs(
        model,
        batch[f"{prefix_b}_input_ids"],
        batch[f"{prefix_b}_attention_mask"],
        batch[f"{prefix_b}_labels"],
    )
    return DPOBatchLogps(a=logp_a, b=logp_b)


def dpo_loss_from_logps(
    policy_a_logps: torch.Tensor,
    policy_b_logps: torch.Tensor,
    ref_a_logps: torch.Tensor,
    ref_b_logps: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> DPOResult:
    """Objective-specific DPO loss, exactly Eq. (11)--Eq. (12).

    For z=(x, y_a, y_b, b_i),

        Delta_theta = log pi_theta(y_a|x)/pi_ref(y_a|x)
                    - log pi_theta(y_b|x)/pi_ref(y_b|x)

        ell_i = -b_i log sigmoid(beta_i Delta_theta)
                -(1-b_i) log sigmoid(-beta_i Delta_theta).

    ``labels`` is b_i: 1 means y_a is preferred to y_b; 0 means the reverse.
    """
    labels = labels.to(device=policy_a_logps.device, dtype=policy_a_logps.dtype).view(-1)
    if policy_a_logps.shape != labels.shape:
        raise ValueError("DPO labels must have the same batch shape as log-probabilities.")

    pi_logratio = policy_a_logps - policy_b_logps
    ref_logratio = ref_a_logps - ref_b_logps
    margins = pi_logratio - ref_logratio
    scaled = float(beta) * margins
    losses = -labels * F.logsigmoid(scaled) - (1.0 - labels) * F.logsigmoid(-scaled)

    response_a_rewards = float(beta) * (policy_a_logps - ref_a_logps).detach()
    response_b_rewards = float(beta) * (policy_b_logps - ref_b_logps).detach()
    prediction = (margins.detach() > 0).to(dtype=labels.dtype)
    accuracy = prediction.eq(labels.detach()).float()
    return DPOResult(
        losses=losses,
        response_a_rewards=response_a_rewards,
        response_b_rewards=response_b_rewards,
        margins=margins,
        labels=labels.detach(),
        accuracy=accuracy,
    )


def raw_dro_weights(
    losses: torch.Tensor,
    eta: torch.Tensor | float,
    lam: torch.Tensor | float,
    divergence: str,
    *,
    exp_clip: float = 30.0,
) -> torch.Tensor:
    """Compute omega_i,k = (f_i^*)'((ell_i - eta_i) / lambda_i), Eq. (18).

    Supported f-divergence penalties:
    - none: omega = 1, which recovers non-DRO DPO/MO-DPO.
    - kl:   omega = exp((loss - eta) / lambda), Eq. (23).
    - chi2: omega = [1 + (loss - eta) / lambda]_+, Eq. (25).
    """
    divergence = divergence.lower()
    if divergence in {"none", "off", "false"}:
        return torch.ones_like(losses)
    if isinstance(eta, float):
        eta = torch.tensor(eta, dtype=losses.dtype, device=losses.device)
    if isinstance(lam, float):
        lam = torch.tensor(lam, dtype=losses.dtype, device=losses.device)
    if torch.any(lam <= 0):
        raise ValueError("DRO lambda must be positive.")
    scaled = (losses - eta) / lam
    if divergence == "kl":
        return torch.exp(torch.clamp(scaled, min=-exp_clip, max=exp_clip))
    if divergence in {"chi2", "chisq", "chi-square"}:
        return torch.relu(1.0 + scaled)
    raise ValueError(f"Unknown divergence: {divergence}")


def eta_sgd_update(
    eta: float,
    raw_weights: torch.Tensor,
    eta_lr: float,
    divergence: str,
) -> float:
    """Dual threshold update, Eq. (17).

        eta <- eta - alpha_eta * (1 - mean_k omega_i,k)
    """
    if divergence.lower() in {"none", "off", "false"} or eta_lr == 0:
        return float(eta)
    mean_omega = float(raw_weights.detach().float().mean().cpu())
    return float(eta - eta_lr * (1.0 - mean_omega))


def clip_and_normalize_weights(
    raw: torch.Tensor,
    omega_max: float | None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Clip and renormalize adversarial weights, Eq. (22).

    When ``normalize=True``, this returns

        min(omega, omega_max) / (mean_k min(omega_k, omega_max) + eps).

    If omega_max is None, no clipping is applied. This is the raw-weight variant
    mentioned immediately after Eq. (22).
    """
    weights = raw
    if omega_max is not None:
        weights = torch.clamp(weights, max=float(omega_max))
    if normalize:
        weights = weights / (weights.mean().detach() + eps)
    return weights


def robust_batch_loss(losses: torch.Tensor, normalized_weights: torch.Tensor) -> torch.Tensor:
    """Scalar whose gradient is the empirical robust gradient in Eq. (21).

    The weights are detached so autograd computes

        grad_theta mean_k bar_omega_k * ell_i(theta; Z_k)

    and not gradients through the adversarial weighting rule.
    """
    return (normalized_weights.detach() * losses).mean()


# Backwards-compatible helper retained for older call sites and tests.
def robust_weighted_loss(
    losses: torch.Tensor,
    eta: torch.Tensor | float,
    lam: torch.Tensor | float,
    divergence: str,
    omega_max: float | None,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = raw_dro_weights(losses.detach(), eta, lam, divergence)
    weights = clip_and_normalize_weights(raw.detach(), omega_max, normalize)
    return robust_batch_loss(losses, weights), weights, raw
