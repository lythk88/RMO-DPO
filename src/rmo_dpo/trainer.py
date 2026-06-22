from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange

from .config import as_list
from .conflict import ConflictResult, combine_gradients
from .data import DPODataCollator, build_objective_datasets, resolve_pair_dir
from .losses import (
    DPOResult,
    clip_and_normalize_weights,
    dpo_loss_from_logps,
    eta_sgd_update,
    pair_log_probs,
    raw_dro_weights,
    robust_batch_loss,
)
from .models import (
    can_reuse_peft_base_as_reference,
    load_policy_checkpoint,
    load_causal_lm_for_lora,
    load_frozen_reference_model,
    load_tokenizer,
    save_policy,
)
from .utils import (
    assign_flat_grad,
    clip_flat_grad,
    first_parameter_device,
    flatten_grads,
    global_grad_norm,
    maybe_init_wandb,
    move_batch_to_device,
    seed_everything,
    setup_logger,
    trainable_parameters,
)


class RMODPOTrainer:
    """Utilities for Algorithm 1 RMO-DPO.

    The public methods intentionally mirror the manuscript's Algorithm 1:
    sample a minibatch, compute Eq. (12), compute Eq. (18), update Eq. (17),
    clip/renormalize Eq. (22), compute Eq. (21), form G_t, compute d_t with
    Eq. (26) or Eq. (27)--Eq. (29), and update theta.
    """

    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        seed_everything(int(config.get("seed", 0)))
        self.logger = setup_logger("rmo_dpo.trainer")
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = maybe_init_wandb(config)

        data_cfg = config["data"]
        model_cfg = config["model"]
        rmo_cfg = config["rmo_dpo"]
        opt_cfg = config["optimizer"]

        self.objectives = list(data_cfg["objectives"])
        self.m = len(self.objectives)
        self.objective_to_idx = {obj: i for i, obj in enumerate(self.objectives)}
        self.training_mode = str(config.get("training_mode", "algorithm1")).lower()
        if self.training_mode not in {"algorithm1", "scalarized_robust"}:
            raise ValueError("training_mode must be 'algorithm1' or 'scalarized_robust'.")
        self.beta = [float(x) for x in as_list(rmo_cfg.get("beta", 0.1), self.m, "beta")]
        self.lam = [float(x) for x in as_list(rmo_cfg.get("lambda", 1.0), self.m, "lambda")]
        self.eta = [float(x) for x in as_list(rmo_cfg.get("eta_init", 0.0), self.m, "eta_init")]
        self.eta_lr = float(rmo_cfg.get("eta_lr", 0.0))
        self.divergence = str(rmo_cfg.get("divergence", "none")).lower()
        self.omega_max = rmo_cfg.get("omega_max")
        self.normalize_weights = bool(rmo_cfg.get("normalize_weights", True))
        self.conflict = str(rmo_cfg.get("conflict", "clip")).lower()
        self.user_weights = torch.tensor(
            [float(x) for x in as_list(rmo_cfg.get("user_weights", 1.0 / self.m), self.m, "user_weights")],
            dtype=torch.float32,
        )
        self.user_weights = self.user_weights / self.user_weights.sum()
        self.cagrad_c = float(rmo_cfg.get("cagrad_c", 0.4))
        self.mgda_rho = float(rmo_cfg.get("mgda_rho", 1e-4))
        self.qp_steps = int(rmo_cfg.get("qp_steps", 80))
        self.qp_lr = rmo_cfg.get("qp_lr")
        self.qp_lr = None if self.qp_lr in {None, "null"} else float(self.qp_lr)
        scalarized_cfg = config.get("scalarized_dro", {})
        self.scalar_lambda = float(scalarized_cfg.get("lambda", 1.0))
        self.scalar_eta = float(scalarized_cfg.get("eta_init", 0.0))
        self.scalar_eta_lr = float(scalarized_cfg.get("eta_lr", self.eta_lr))
        self.scalar_divergence = str(scalarized_cfg.get("divergence", self.divergence)).lower()
        self.scalar_omega_max = scalarized_cfg.get("omega_max", self.omega_max)
        self.scalar_normalize_weights = bool(scalarized_cfg.get("normalize_weights", self.normalize_weights))

        self.tokenizer = load_tokenizer(
            model_cfg["policy_name"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True))
        )
        self.reference_name = model_cfg.get("reference_name", model_cfg["policy_name"])
        self.model = load_causal_lm_for_lora(
            model_cfg["policy_name"],
            use_lora=bool(model_cfg.get("use_lora", True)),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            lora_config=model_cfg.get("lora", {}),
        )
        self.reference_model = None
        self.use_peft_reference = can_reuse_peft_base_as_reference(
            use_lora=bool(model_cfg.get("use_lora", True)),
            policy_name=model_cfg["policy_name"],
            reference_name=self.reference_name,
            model=self.model,
        )
        if not self.use_peft_reference:
            self.reference_model = load_frozen_reference_model(
                self.reference_name,
                load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
                torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
                trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            )
        self.input_device = first_parameter_device(self.model)

        collator = DPODataCollator(
            self.tokenizer,
            max_length=int(data_cfg.get("max_length", 2048)),
            max_prompt_length=int(data_cfg.get("max_prompt_length", 1024)),
            max_response_length=int(data_cfg.get("max_response_length", 1024)),
            system_message=data_cfg.get("system_message"),
        )
        train_pair_dir = resolve_pair_dir(data_cfg, eval_mode=False)
        datasets = build_objective_datasets(train_pair_dir, data_cfg.get("split", "train"), self.objectives)
        batch_size = int(data_cfg.get("per_objective_batch_size", 1))
        num_workers = int(data_cfg.get("dataloader_num_workers", 0))
        self.loaders: dict[str, DataLoader] = {
            obj: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collator,
                drop_last=True,
            )
            for obj, ds in datasets.items()
        }
        self.loader_iters = {obj: iter(loader) for obj, loader in self.loaders.items()}
        self.eval_loaders: dict[str, DataLoader] | None = None
        eval_split = data_cfg.get("eval_split")
        if eval_split:
            try:
                eval_pair_dir = resolve_pair_dir(data_cfg, eval_mode=True)
                eval_datasets = build_objective_datasets(eval_pair_dir, eval_split, self.objectives)
                self.eval_loaders = {
                    obj: DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collator,
                        drop_last=False,
                    )
                    for obj, ds in eval_datasets.items()
                }
            except FileNotFoundError as exc:
                self.logger.warning("Evaluation split disabled: %s", exc)

        self.params = trainable_parameters(self.model)
        if not self.params:
            raise ValueError("No trainable policy parameters. Enable LoRA or unfreeze model parameters.")

        self.policy_lr = float(opt_cfg.get("lr", opt_cfg.get("alpha_theta", 5e-6)))
        self.optimizer_name = str(opt_cfg.get("name", opt_cfg.get("type", "sgd"))).lower()
        self.optimizer: AdamW | None = None
        if self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.params,
                lr=self.policy_lr,
                betas=tuple(float(x) for x in opt_cfg.get("betas", [0.9, 0.95])),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            )
        elif self.optimizer_name not in {"sgd", "manual_sgd", "algorithm1"}:
            raise ValueError("optimizer.name must be 'sgd' for exact Algorithm 1 or 'adamw' as a practical variant.")

        self.max_steps = int(opt_cfg.get("max_steps", 1000))
        self.max_grad_norm = opt_cfg.get("max_grad_norm", None)
        self.max_grad_norm = None if self.max_grad_norm in {None, "null"} else float(self.max_grad_norm)
        self.save_every = int(opt_cfg.get("save_every", 0))
        self.eval_every = int(opt_cfg.get("eval_every", 0))

    def _infer_checkpoint_step(self, checkpoint_dir: str | Path) -> int:
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.name.startswith("checkpoint-"):
            return int(checkpoint_dir.name.split("-", 1)[1])
        if checkpoint_dir.name == "final":
            return self.max_steps
        raise ValueError(f"Cannot infer training step from checkpoint path: {checkpoint_dir}")

    def resume_from_checkpoint(self, checkpoint_dir: str | Path) -> int:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

        if not hasattr(self.model, "peft_config"):
            raise ValueError("Resume is currently implemented for PEFT/LoRA checkpoints only.")
        load_policy_checkpoint(self.model, checkpoint_dir)

        resume_step = self._infer_checkpoint_step(checkpoint_dir)
        state_path = checkpoint_dir / "state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            eta = state.get("eta")
            if eta is not None:
                self.eta = [float(x) for x in eta]
            scalar_eta = state.get("scalar_eta")
            if scalar_eta is not None:
                self.scalar_eta = float(scalar_eta)
            stored_step = state.get("step")
            if stored_step is not None:
                resume_step = int(stored_step)

        if self.optimizer is not None:
            self.logger.warning(
                "Resuming model weights and dual state without optimizer state; optimizer buffers restart fresh."
            )
        self.logger.info("Resumed checkpoint %s at completed step %d", checkpoint_dir, resume_step)
        return resume_step

    # Algorithm 1, line 4.
    def sample_minibatch(self, objective: str) -> dict[str, Any]:
        try:
            batch = next(self.loader_iters[objective])
        except StopIteration:
            self.loader_iters[objective] = iter(self.loaders[objective])
            batch = next(self.loader_iters[objective])
        return move_batch_to_device(batch, self.input_device)

    @torch.no_grad()
    def _reference_logps(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_peft_reference:
            was_training = self.model.training
            self.model.eval()
            try:
                with self.model.disable_adapter():
                    logps = pair_log_probs(self.model, batch)
            finally:
                if was_training:
                    self.model.train()
        elif self.reference_model is not None:
            logps = pair_log_probs(self.reference_model, batch)
        else:
            raise RuntimeError("No reference policy available.")
        return logps.a.detach(), logps.b.detach()

    # Algorithm 1, line 5; Eq. (12), using Eq. (11) internally.
    def compute_dpo_losses(self, objective: str, batch: dict[str, Any]) -> DPOResult:
        i = self.objective_to_idx[objective]
        self.model.train()
        policy_logps = pair_log_probs(self.model, batch)
        ref_a, ref_b = self._reference_logps(batch)
        return dpo_loss_from_logps(
            policy_logps.a,
            policy_logps.b,
            ref_a,
            ref_b,
            labels=batch["preference_label"],
            beta=self.beta[i],
        )

    # Algorithm 1, line 6; Eq. (18).
    def compute_adversarial_weights(self, objective: str, losses: torch.Tensor) -> torch.Tensor:
        i = self.objective_to_idx[objective]
        return raw_dro_weights(
            losses.detach(),
            eta=self.eta[i],
            lam=self.lam[i],
            divergence=self.divergence,
        )

    # Algorithm 1, line 7; Eq. (17).
    def update_dual_threshold(self, objective: str, raw_weights: torch.Tensor) -> tuple[float, float]:
        i = self.objective_to_idx[objective]
        old_eta = self.eta[i]
        self.eta[i] = eta_sgd_update(old_eta, raw_weights, self.eta_lr, self.divergence)
        return old_eta, self.eta[i]

    # Algorithm 1, line 8; Eq. (22).
    def clip_and_renormalize_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        return clip_and_normalize_weights(
            raw_weights.detach(),
            omega_max=self.omega_max,
            normalize=self.normalize_weights,
        )

    # Algorithm 1, line 9; Eq. (21).
    def compute_objective_gradient(
        self,
        losses: torch.Tensor,
        normalized_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robust_loss = robust_batch_loss(losses, normalized_weights)
        grads = torch.autograd.grad(robust_loss, self.params, retain_graph=False, allow_unused=True)
        flat_grad = flatten_grads(grads, self.params, device="cpu")
        return flat_grad, robust_loss.detach()

    # Algorithm 1, line 12; Eq. (26) or Eq. (27)--Eq. (29).
    def compute_update_direction(self, grad_matrix: torch.Tensor) -> ConflictResult:
        return combine_gradients(
            grad_matrix,
            mode=self.conflict,  # type: ignore[arg-type]
            user_weights=self.user_weights,
            cagrad_c=self.cagrad_c,
            mgda_rho=self.mgda_rho,
            qp_steps=self.qp_steps,
            qp_lr=self.qp_lr,
        )

    # Algorithm 1, line 13. Exact default: theta <- theta - alpha_theta d_t.
    def policy_update(self, direction: torch.Tensor) -> tuple[torch.Tensor, float]:
        flat_direction, unclipped_norm = clip_flat_grad(direction, self.max_grad_norm)
        assign_flat_grad(self.params, flat_direction)
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                for p in self.params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-self.policy_lr)
        return flat_direction, unclipped_norm

    def objective_metrics(
        self,
        objective: str,
        dpo: DPOResult,
        robust_loss: torch.Tensor,
        raw_weights: torch.Tensor,
        normalized_weights: torch.Tensor,
        eta_before: float,
        eta_after: float,
    ) -> dict[str, float]:
        return {
            f"train/{objective}_loss": float(dpo.losses.detach().float().mean().cpu()),
            f"train/{objective}_robust_loss": float(robust_loss.detach().float().cpu()),
            f"train/{objective}_acc": float(dpo.accuracy.detach().float().mean().cpu()),
            f"train/{objective}_margin": float(dpo.margins.detach().float().mean().cpu()),
            f"train/{objective}_eta_before": float(eta_before),
            f"train/{objective}_eta": float(eta_after),
            f"train/{objective}_raw_omega_mean": float(raw_weights.detach().float().mean().cpu()),
            f"train/{objective}_omega_max": float(normalized_weights.detach().float().max().cpu()),
            f"train/{objective}_omega_min": float(normalized_weights.detach().float().min().cpu()),
        }

    def scalarized_objective_metrics(
        self,
        objective: str,
        dpo: DPOResult,
        objective_weight: float,
    ) -> dict[str, float]:
        return {
            f"train/{objective}_loss": float(dpo.losses.detach().float().mean().cpu()),
            f"train/{objective}_acc": float(dpo.accuracy.detach().float().mean().cpu()),
            f"train/{objective}_margin": float(dpo.margins.detach().float().mean().cpu()),
            f"train/{objective}_weight": float(objective_weight),
        }

    def finalize_algorithm1_step(
        self,
        step: int,
        metrics: dict[str, float],
        conflict_result: ConflictResult,
        unclipped_norm: float,
    ) -> None:
        metrics["train/grad_norm_unclipped"] = unclipped_norm
        metrics["train/grad_norm_assigned"] = global_grad_norm(self.params)
        metrics["train/lr"] = self.policy_lr
        for k, v in conflict_result.info.items():
            metrics[f"train/conflict_{k}"] = v
        for obj, coef in zip(self.objectives, conflict_result.coefficients.tolist()):
            metrics[f"train/coef_{obj}"] = float(coef)
        for obj, aux in zip(self.objectives, conflict_result.aux_weights.tolist()):
            metrics[f"train/aux_{obj}"] = float(aux)

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        if step == 1 or step % 10 == 0:
            msg = (
                f"step={step} loss="
                + ",".join(
                    f"{obj}:{metrics.get(f'train/{obj}_loss', math.nan):.3f}" for obj in self.objectives
                )
                + f" grad={metrics['train/grad_norm_assigned']:.3f}"
            )
            self.logger.info(msg)
        if self.eval_every and self.eval_loaders is not None and step % self.eval_every == 0:
            eval_metrics = self.evaluate(max_batches=50)
            if self.wandb_run is not None:
                self.wandb_run.log(eval_metrics, step=step)
            self.logger.info("eval step=%d %s", step, eval_metrics)
        if self.save_every and step % self.save_every == 0:
            self.save(self.output_dir / f"checkpoint-{step}", step=step)

    def finalize_scalarized_step(
        self,
        step: int,
        metrics: dict[str, float],
        unclipped_norm: float,
    ) -> None:
        metrics["train/grad_norm_unclipped"] = unclipped_norm
        metrics["train/grad_norm_assigned"] = global_grad_norm(self.params)
        metrics["train/lr"] = self.policy_lr

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        if step == 1 or step % 10 == 0:
            msg = (
                f"step={step} loss="
                + ",".join(
                    f"{obj}:{metrics.get(f'train/{obj}_loss', math.nan):.3f}" for obj in self.objectives
                )
                + f" scalar={metrics['train/scalarized_robust_loss']:.3f}"
                + f" grad={metrics['train/grad_norm_assigned']:.3f}"
            )
            self.logger.info(msg)
        if self.eval_every and self.eval_loaders is not None and step % self.eval_every == 0:
            eval_metrics = self.evaluate(max_batches=50)
            if self.wandb_run is not None:
                self.wandb_run.log(eval_metrics, step=step)
            self.logger.info("eval step=%d %s", step, eval_metrics)
        if self.save_every and step % self.save_every == 0:
            self.save(self.output_dir / f"checkpoint-{step}", step=step)

    def algorithm1_step(self, step: int) -> dict[str, float]:
        """One complete iteration of Algorithm 1, lines 3--13."""
        objective_grads: list[torch.Tensor] = []
        metrics: dict[str, float] = {"step": float(step)}

        for objective in self.objectives:
            batch = self.sample_minibatch(objective)  # line 4
            dpo = self.compute_dpo_losses(objective, batch)  # line 5, Eq. (12)
            raw_weights = self.compute_adversarial_weights(objective, dpo.losses)  # line 6, Eq. (18)
            eta_before, eta_after = self.update_dual_threshold(objective, raw_weights)  # line 7, Eq. (17)
            normalized_weights = self.clip_and_renormalize_weights(raw_weights)  # line 8, Eq. (22)
            flat_grad, robust_loss = self.compute_objective_gradient(dpo.losses, normalized_weights)  # line 9, Eq. (21)
            objective_grads.append(flat_grad)
            metrics.update(
                self.objective_metrics(
                    objective,
                    dpo,
                    robust_loss,
                    raw_weights,
                    normalized_weights,
                    eta_before,
                    eta_after,
                )
            )

        grad_matrix = torch.stack(objective_grads, dim=0)  # line 11
        conflict_result = self.compute_update_direction(grad_matrix)  # line 12
        _, unclipped_norm = self.policy_update(conflict_result.direction)  # line 13
        self.finalize_algorithm1_step(step, metrics, conflict_result, unclipped_norm)
        return metrics

    def scalarized_robust_step(self, step: int) -> dict[str, float]:
        """Single-risk DRO baseline over a pooled, weighted objective mixture."""
        pooled_losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {"step": float(step)}

        for objective in self.objectives:
            i = self.objective_to_idx[objective]
            batch = self.sample_minibatch(objective)
            dpo = self.compute_dpo_losses(objective, batch)
            objective_weight = float(self.user_weights[i].cpu())
            metrics.update(self.scalarized_objective_metrics(objective, dpo, objective_weight))
            pooled_losses.append(dpo.losses * (objective_weight * self.m))

        scalar_losses = torch.cat(pooled_losses, dim=0)
        raw_weights = raw_dro_weights(
            scalar_losses.detach(),
            eta=self.scalar_eta,
            lam=self.scalar_lambda,
            divergence=self.scalar_divergence,
        )
        eta_before = self.scalar_eta
        self.scalar_eta = eta_sgd_update(
            self.scalar_eta,
            raw_weights,
            self.scalar_eta_lr,
            self.scalar_divergence,
        )
        normalized_weights = clip_and_normalize_weights(
            raw_weights.detach(),
            omega_max=self.scalar_omega_max,
            normalize=self.scalar_normalize_weights,
        )
        robust_loss = robust_batch_loss(scalar_losses, normalized_weights)
        grads = torch.autograd.grad(robust_loss, self.params, retain_graph=False, allow_unused=True)
        flat_grad = flatten_grads(grads, self.params, device="cpu")
        _, unclipped_norm = self.policy_update(flat_grad)

        metrics["train/scalarized_robust_loss"] = float(robust_loss.detach().float().cpu())
        metrics["train/scalarized_eta_before"] = float(eta_before)
        metrics["train/scalarized_eta"] = float(self.scalar_eta)
        metrics["train/scalarized_raw_omega_mean"] = float(raw_weights.detach().float().mean().cpu())
        metrics["train/scalarized_omega_max"] = float(normalized_weights.detach().float().max().cpu())
        metrics["train/scalarized_omega_min"] = float(normalized_weights.detach().float().min().cpu())
        self.finalize_scalarized_step(step, metrics, unclipped_norm)
        return metrics

    def train(self, start_step: int = 1) -> None:
        """Run the configured training loop for T=max_steps iterations."""
        if self.training_mode == "scalarized_robust":
            self.logger.info("Starting scalarized robust DPO baseline for %d steps", self.max_steps)
            for step in trange(
                start_step,
                self.max_steps + 1,
                desc="Scalarized robust DPO",
                initial=max(0, start_step - 1),
                total=self.max_steps,
            ):
                self.scalarized_robust_step(step)
        else:
            self.logger.info("Starting Algorithm 1 RMO-DPO for %d steps", self.max_steps)
            for step in trange(
                start_step,
                self.max_steps + 1,
                desc="Algorithm 1 RMO-DPO",
                initial=max(0, start_step - 1),
                total=self.max_steps,
            ):
                self.algorithm1_step(step)
        self.save(self.output_dir / "final", step=self.max_steps)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        if self.eval_loaders is None:
            return {}
        self.model.eval()
        metrics: dict[str, float] = {}
        for objective, loader in self.eval_loaders.items():
            i = self.objective_to_idx[objective]
            losses = []
            accs = []
            margins = []
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = move_batch_to_device(batch, self.input_device)
                policy_logps = pair_log_probs(self.model, batch)
                ref_a, ref_b = self._reference_logps(batch)
                dpo = dpo_loss_from_logps(
                    policy_logps.a,
                    policy_logps.b,
                    ref_a,
                    ref_b,
                    labels=batch["preference_label"],
                    beta=self.beta[i],
                )
                losses.append(dpo.losses.detach().float().cpu())
                accs.append(dpo.accuracy.detach().float().cpu())
                margins.append(dpo.margins.detach().float().cpu())
            if losses:
                metrics[f"eval/{objective}_loss"] = float(torch.cat(losses).mean())
                metrics[f"eval/{objective}_acc"] = float(torch.cat(accs).mean())
                metrics[f"eval/{objective}_margin"] = float(torch.cat(margins).mean())
        obj_losses = [v for k, v in metrics.items() if k.endswith("_loss")]
        obj_accs = [v for k, v in metrics.items() if k.endswith("_acc")]
        if obj_losses:
            metrics["eval/mean_loss"] = float(sum(obj_losses) / len(obj_losses))
            metrics["eval/worst_loss"] = float(max(obj_losses))
        if obj_accs:
            metrics["eval/mean_acc"] = float(sum(obj_accs) / len(obj_accs))
            metrics["eval/worst_acc"] = float(min(obj_accs))
        self.model.train()
        return metrics

    def save(self, output_dir: str | Path, *, step: int | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_policy(self.model, self.tokenizer, str(output_dir))
        torch.save(
            {
                "step": None if step is None else int(step),
                "eta": self.eta,
                "scalar_eta": self.scalar_eta,
                "objectives": self.objectives,
                "config": self.cfg,
            },
            output_dir / "state.pt",
        )
        self.logger.info("Saved checkpoint to %s", output_dir)
