#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_LIB_DIR="$ROOT_DIR/vendor/nvidia-570/extracted/usr/lib/x86_64-linux-gnu"

if [[ -d "$NVIDIA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$ROOT_DIR"
source .venv/bin/activate
exec python scripts/run_noise_sweep.py "$@"
