#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs/noise_sweep"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

while [[ ! -d outputs/noise_sweep/noise_0.0/final ]]; do
  sleep 300
done

bash scripts/run_noise_sweep_gpu.sh --skip_prepare --noise_rates 0.1 0.2 0.3 >>"$LOG_DIR/continue_noise_sweep.log" 2>&1
