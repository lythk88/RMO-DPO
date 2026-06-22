#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_pku_saferlhf.py \
  --output_dir data/pku_saferlhf_pairs \
  --seed 42

python scripts/train_rmo_dpo.py \
  --config configs/pku_saferlhf_rmo_dpo.yaml

python scripts/evaluate_helpsteer2.py \
  --config configs/pku_saferlhf_rmo_dpo.yaml \
  --checkpoint outputs/pku_saferlhf_rmo_dpo_qwen2p5_7b/final \
  --output_json outputs/pku_saferlhf_rmo_dpo_qwen2p5_7b/eval_metrics.json
