#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_helpsteer2.py \
  --output_dir data/helpsteer2_pairs \
  --min_score_gap 1 \
  --seed 42

python scripts/train_rmo_dpo.py \
  --config configs/helpsteer2_rmo_dpo.yaml

python scripts/evaluate_helpsteer2.py \
  --config configs/helpsteer2_rmo_dpo.yaml \
  --checkpoint outputs/helpsteer2_rmo_dpo_qwen2p5_7b/final
