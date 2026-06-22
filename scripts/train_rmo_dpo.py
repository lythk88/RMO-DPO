#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from tqdm.auto import trange

from rmo_dpo.config import load_config
from rmo_dpo.trainer import RMODPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RMO-DPO exactly following Algorithm 1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from", default=None, help="Optional checkpoint directory to resume from.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Algorithm 1, line 1: initialize theta_0 and eta_{i,0} from the config.
    trainer = RMODPOTrainer(cfg)
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
        trainer.train(start_step=resume_step + 1)
        return
    trainer.logger.info(
        "Starting Algorithm 1 RMO-DPO for %d steps from completed step %d",
        trainer.max_steps,
        resume_step,
    )

    # Algorithm 1, line 2: for t = 0, 1, ..., T-1.
    for t in trange(
        resume_step,
        trainer.max_steps,
        desc="Algorithm 1 RMO-DPO",
        initial=resume_step,
        total=trainer.max_steps,
    ):
        objective_grads: list[torch.Tensor] = []
        metrics: dict[str, float] = {"step": float(t + 1)}

        # Algorithm 1, line 3: for each objective i in [m].
        for objective in trainer.objectives:
            # line 4: sample minibatch B_{i,t}={(x, y_a, y_b, b_i)} from D_i.
            batch = trainer.sample_minibatch(objective)

            # line 5: compute DPO losses using Eq. (12), with margin Eq. (11).
            dpo = trainer.compute_dpo_losses(objective, batch)

            # line 6: compute adversarial weights omega_{i,k} using Eq. (18).
            raw_weights = trainer.compute_adversarial_weights(objective, dpo.losses)

            # line 7: update eta_{i,t} using Eq. (17), before clipping weights.
            eta_before, eta_after = trainer.update_dual_threshold(objective, raw_weights)

            # line 8: clip and renormalize weights using Eq. (22).
            normalized_weights = trainer.clip_and_renormalize_weights(raw_weights)

            # line 9: compute g_{B_i}(theta_t) using Eq. (21).
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
        # line 10: end for.

        # line 11: form G_t=[g_{B_1}(theta_t), ..., g_{B_m}(theta_t)].
        grad_matrix = torch.stack(objective_grads, dim=0)

        # line 12: compute d_t using Eq. (26) for MGDA or Eq. (27)--Eq. (29) for Clip.
        conflict_result = trainer.compute_update_direction(grad_matrix)

        # line 13: update theta_{t+1}=theta_t - alpha_theta d_t.
        _, unclipped_norm = trainer.policy_update(conflict_result.direction)
        trainer.finalize_algorithm1_step(t + 1, metrics, conflict_result, unclipped_norm)
    # line 14: end for.

    # line 15: return pi_{theta_T}; in code, save the final policy/adapters.
    trainer.save(trainer.output_dir / "final", step=trainer.max_steps)


if __name__ == "__main__":
    main()
