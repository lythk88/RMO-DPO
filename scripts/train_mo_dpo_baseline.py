#!/usr/bin/env python
from __future__ import annotations

import argparse

from rmo_dpo.config import load_config
from rmo_dpo.trainer import RMODPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-objective DPO baseline without DRO.")
    parser.add_argument(
        "--config",
        default="configs/noise_sweep/helpsteer2_mo_dpo_noise_0.1_baseline_5000.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_mode = str(cfg.get("training_mode", "algorithm1")).lower()
    conflict = str(cfg["rmo_dpo"].get("conflict", "")).lower()
    divergence = str(cfg["rmo_dpo"].get("divergence", "none")).lower()
    eta_lr = float(cfg["rmo_dpo"].get("eta_lr", 0.0))

    if training_mode != "algorithm1":
        raise ValueError("MO-DPO baseline requires training_mode=algorithm1.")
    if conflict != "clip":
        raise ValueError("MO-DPO baseline requires rmo_dpo.conflict=clip.")
    if divergence != "none" or eta_lr != 0.0:
        raise ValueError("MO-DPO baseline must disable DRO with divergence=none and eta_lr=0.0.")

    trainer = RMODPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
