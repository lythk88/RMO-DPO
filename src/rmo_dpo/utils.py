from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "rmo_dpo", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def cycle(iterable: Iterable[Any]) -> Iterator[Any]:
    while True:
        for x in iterable:
            yield x


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_grads(
    grads: Iterable[torch.Tensor | None],
    params: Iterable[torch.nn.Parameter],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=dtype, device=device))
        else:
            chunks.append(grad.detach().to(device=device, dtype=dtype).reshape(-1))
    if not chunks:
        raise ValueError("No trainable parameters found.")
    return torch.cat(chunks, dim=0)


def assign_flat_grad(params: Iterable[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in params:
        n = param.numel()
        grad = flat_grad[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype)
        param.grad = grad.clone()
        offset += n
    if offset != flat_grad.numel():
        raise ValueError(f"Flat gradient has {flat_grad.numel()} values but assigned {offset}.")


def global_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    for p in params:
        if p.grad is not None:
            sq_sum += float(p.grad.detach().float().pow(2).sum().cpu())
    return sq_sum**0.5


def clip_flat_grad(flat_grad: torch.Tensor, max_norm: float | None) -> tuple[torch.Tensor, float]:
    norm = float(torch.linalg.vector_norm(flat_grad.detach().float()).cpu())
    if max_norm is None or max_norm <= 0 or norm <= max_norm:
        return flat_grad, norm
    return flat_grad * (max_norm / (norm + 1e-12)), norm


def maybe_init_wandb(config: dict[str, Any]) -> Any | None:
    if str(config.get("report_to", "none")).lower() != "wandb":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(project="rmo-dpo-helpsteer2", name=config.get("run_name"), config=config)
