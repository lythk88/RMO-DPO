#!/usr/bin/env python
from __future__ import annotations

import argparse

from rmo_dpo.config import load_config
from rmo_dpo.trainer import RMODPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scalarized robust DPO baseline.")
    parser.add_argument(
        "--config",
        default="configs/noise_sweep/helpsteer2_scalarized_robust_dpo_noise_0.1_baseline_5000.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_mode = str(cfg.get("training_mode", "algorithm1")).lower()

    if training_mode != "scalarized_robust":
        raise ValueError("Scalarized robust DPO baseline requires training_mode=scalarized_robust.")

    trainer = RMODPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
