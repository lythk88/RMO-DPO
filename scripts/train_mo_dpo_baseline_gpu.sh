#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_LIB_DIR="$ROOT_DIR/vendor/nvidia-570/extracted/usr/lib/x86_64-linux-gnu"

if [[ -d "$NVIDIA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

NVIDIA_SMI="$ROOT_DIR/vendor/nvidia-570/extracted/usr/bin/nvidia-smi"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -x "$NVIDIA_SMI" ]]; then
  CUDA_VISIBLE_DEVICES="$(
    "$NVIDIA_SMI" --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F, 'BEGIN {best_idx = ""; best_free = -1}
        {gsub(/ /, "", $1); gsub(/ /, "", $2)}
        $2 + 0 > best_free {best_idx = $1; best_free = $2 + 0}
        END {print best_idx}'
  )"
  export CUDA_VISIBLE_DEVICES
fi

cd "$ROOT_DIR"
source .venv/bin/activate
exec python scripts/train_mo_dpo_baseline.py --config "${1:-configs/noise_sweep/helpsteer2_mo_dpo_noise_0.1_baseline_5000.yaml}"
