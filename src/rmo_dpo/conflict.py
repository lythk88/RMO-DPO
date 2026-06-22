from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class ConflictResult:
    direction: torch.Tensor
    coefficients: torch.Tensor
    aux_weights: torch.Tensor
    info: dict[str, float]


def project_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum x = z}.

    Implementation follows the sorting-based algorithm of Duchi et al. (2008).
    """
    if v.ndim != 1:
        raise ValueError("project_simplex expects a 1-D tensor")
    if z <= 0:
        raise ValueError("simplex radius z must be positive")
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not torch.any(cond):
        return torch.full_like(v, z / v.numel())
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(dtype=v.dtype) + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    return w


def _gram(gradients: torch.Tensor) -> torch.Tensor:
    # gradients shape: [m, d]
    return gradients @ gradients.T


def _safe_weights(weights: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
    weights = weights.to(device=device, dtype=torch.float32).flatten()
    if weights.numel() != m:
        raise ValueError(f"Expected {m} objective weights; got {weights.numel()}")
    if torch.any(weights < 0):
        raise ValueError("Objective weights must be nonnegative.")
    s = weights.sum()
    if s <= 0:
        raise ValueError("At least one objective weight must be positive.")
    return weights / s


def solve_mgda_weights(
    gradients: torch.Tensor,
    rho: float = 1e-4,
    steps: int = 80,
    lr: float | None = None,
) -> torch.Tensor:
    """Solve min_{v in simplex} 0.5 ||G^T v||^2 + rho/2 ||v||^2 by PGD."""
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [num_objectives, num_params]")
    m = gradients.shape[0]
    device = gradients.device
    gram = _gram(gradients.float())
    hessian = gram + float(rho) * torch.eye(m, device=device, dtype=torch.float32)
    try:
        lip = float(torch.linalg.eigvalsh(hessian).max().cpu())
    except RuntimeError:
        lip = float(torch.linalg.matrix_norm(hessian, ord=2).cpu())
    step_size = float(lr) if lr is not None else 1.0 / (lip + 1e-12)
    v = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)
    for _ in range(steps):
        grad = hessian @ v
        v = project_simplex(v - step_size * grad)
    return v


def solve_cagrad_weights(
    gradients: torch.Tensor,
    user_weights: torch.Tensor,
    c: float = 0.4,
    steps: int = 80,
    lr: float | None = None,
) -> torch.Tensor:
    """Solve the CAGrad inner simplex problem used by RMO-DPO-Clip.

    min_p <G^T p, g_a> + c ||g_a|| ||G^T p||, p in simplex.
    The returned p is later clipped against user weights.
    """
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [num_objectives, num_params]")
    m = gradients.shape[0]
    device = gradients.device
    a = _safe_weights(user_weights, m, device)
    gram = _gram(gradients.float())
    ga = a @ gradients.float()
    b = gram @ a  # b_i = <g_i, g_a>
    ga_norm = torch.linalg.vector_norm(ga).clamp_min(1e-12)
    try:
        lip = float(torch.linalg.eigvalsh(gram).max().cpu())
    except RuntimeError:
        lip = float(torch.linalg.matrix_norm(gram, ord=2).cpu())
    step_size = float(lr) if lr is not None else 1.0 / (lip * (1.0 + abs(c)) + 1e-12)
    p = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)
    for _ in range(steps):
        gp = gram @ p
        gp_norm = torch.sqrt(torch.clamp(p @ gp, min=1e-24))
        grad = b + float(c) * ga_norm * gp / gp_norm
        p = project_simplex(p - step_size * grad)
    return p


def combine_gradients(
    gradients: torch.Tensor,
    *,
    mode: Literal["weighted", "mgda", "clip"],
    user_weights: torch.Tensor,
    cagrad_c: float = 0.4,
    mgda_rho: float = 1e-4,
    qp_steps: int = 80,
    qp_lr: float | None = None,
) -> ConflictResult:
    """Combine objective gradients into a single update direction.

    Input shape is [m, d], where each row is g_i. The returned direction has
    shape [d] and is meant to be assigned as the parameter gradient for an
    optimizer step.
    """
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [num_objectives, num_params]")
    gradients = gradients.float()
    m = gradients.shape[0]
    device = gradients.device
    mode = mode.lower()  # type: ignore[assignment]
    a = _safe_weights(user_weights, m, device)

    if mode == "weighted":
        coefficients = a
        direction = coefficients @ gradients
        aux = coefficients
    elif mode == "mgda":
        coefficients = solve_mgda_weights(gradients, rho=mgda_rho, steps=qp_steps, lr=qp_lr)
        direction = coefficients @ gradients
        aux = coefficients
    elif mode == "clip":
        p = solve_cagrad_weights(gradients, a, c=cagrad_c, steps=qp_steps, lr=qp_lr)
        p_tilde = torch.minimum(p, a)
        ga = a @ gradients
        correction = p_tilde @ gradients
        corr_norm = torch.linalg.vector_norm(correction).clamp_min(1e-12)
        ga_norm = torch.linalg.vector_norm(ga)
        scale = float(cagrad_c) * ga_norm / corr_norm
        coefficients = a + scale * p_tilde
        direction = coefficients @ gradients
        aux = p
    else:
        raise ValueError(f"Unknown conflict mode: {mode}")

    gram = _gram(gradients)
    info = {
        "direction_norm": float(torch.linalg.vector_norm(direction).cpu()),
        "mean_grad_norm": float(torch.linalg.vector_norm(gradients, dim=1).mean().cpu()),
        "min_pairwise_cosine": float(_min_pairwise_cosine(gram).cpu()) if m > 1 else 1.0,
    }
    return ConflictResult(direction=direction, coefficients=coefficients, aux_weights=aux, info=info)


def _min_pairwise_cosine(gram: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.clamp(torch.diag(gram), min=1e-24))
    denom = diag[:, None] * diag[None, :]
    cos = gram / denom
    m = gram.shape[0]
    mask = ~torch.eye(m, device=gram.device, dtype=torch.bool)
    return cos[mask].min()
