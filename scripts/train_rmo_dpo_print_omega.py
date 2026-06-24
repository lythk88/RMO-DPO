#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import trange

from rmo_dpo.config import load_config
from rmo_dpo.trainer import RMODPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Algorithm 1 RMO-DPO and print renormalized omega weights."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from", default=None, help="Optional checkpoint directory to resume from.")
    parser.add_argument("--print_every_steps", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["output_dir"] = str(Path(args.output_dir))
    cfg["report_to"] = "none"
    if args.run_name:
        cfg["run_name"] = args.run_name

    print(f"config={args.config}", flush=True)
    print(f"output_dir={cfg['output_dir']}", flush=True)
    print(f"per_objective_batch_size={cfg['data'].get('per_objective_batch_size')}", flush=True)
    print(
        f"normalize_weights={cfg['rmo_dpo'].get('normalize_weights')} "
        f"omega_max={cfg['rmo_dpo'].get('omega_max')}",
        flush=True,
    )

    trainer = RMODPOTrainer(cfg)
    print(f"model_device={trainer.input_device}", flush=True)
    print(
        f"max_steps={trainer.max_steps} save_every={trainer.save_every} eval_every={trainer.eval_every}",
        flush=True,
    )

    resume_step = 0
    if args.resume_from:
        resume_step = trainer.resume_from_checkpoint(args.resume_from)
        if resume_step >= trainer.max_steps:
            trainer.logger.info(
                "Checkpoint already reached step %d >= max_steps %d; saving final and exiting.",
                resume_step,
                trainer.max_steps,
            )
            trainer.save(trainer.output_dir / "final", step=resume_step)
            return

    if trainer.training_mode != "algorithm1":
        raise ValueError(f"This omega-print runner expects training_mode=algorithm1; got {trainer.training_mode}.")

    trainer.logger.info(
        "Starting Algorithm 1 RMO-DPO with renormalized omega printing for %d steps from completed step %d",
        trainer.max_steps,
        resume_step,
    )

    for t in trange(
        resume_step,
        trainer.max_steps,
        desc="Algorithm 1 RMO-DPO omega",
        initial=resume_step,
        total=trainer.max_steps,
    ):
        step = t + 1
        objective_grads: list[torch.Tensor] = []
        metrics: dict[str, float] = {"step": float(step)}

        for objective in trainer.objectives:
            batch = trainer.sample_minibatch(objective)
            dpo = trainer.compute_dpo_losses(objective, batch)
            raw_weights = trainer.compute_adversarial_weights(objective, dpo.losses)
            eta_before, eta_after = trainer.update_dual_threshold(objective, raw_weights)
            normalized_weights = trainer.clip_and_renormalize_weights(raw_weights)

            if args.print_every_steps > 0 and step % args.print_every_steps == 0:
                omega = normalized_weights.detach().float().cpu()
                print(
                    f"step={step} objective={objective} "
                    f"renormalized_omega={omega.tolist()} "
                    f"mean={float(omega.mean()):.10f} sum={float(omega.sum()):.10f}",
                    flush=True,
                )

            flat_grad, robust_loss = trainer.compute_objective_gradient(dpo.losses, normalized_weights)
            objective_grads.append(flat_grad)
            metrics.update(
                trainer.objective_metrics(
                    objective,
                    dpo,
                    robust_loss,
                    raw_weights,
                    normalized_weights,
                    eta_before,
                    eta_after,
                )
            )

        grad_matrix = torch.stack(objective_grads, dim=0)
        conflict_result = trainer.compute_update_direction(grad_matrix)
        _, unclipped_norm = trainer.policy_update(conflict_result.direction)
        trainer.finalize_algorithm1_step(step, metrics, conflict_result, unclipped_norm)

    trainer.save(trainer.output_dir / "final", step=trainer.max_steps)
    print(f"DONE saved final checkpoint to {trainer.output_dir / 'final'}", flush=True)


if __name__ == "__main__":
    main()
