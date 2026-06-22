#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
WAIT_CONFIG="configs/noise_sweep/helpsteer2_rmo_dpo_noise_0.1.yaml"

wait_for_current_rmo_run() {
  while pgrep -af "scripts/train_rmo_dpo.py --config ${WAIT_CONFIG}" >/dev/null; do
    printf "[%s] waiting for current RMO-DPO run to finish\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep 60
  done
}

run_one() {
  local config_path="$1"
  local checkpoint_dir="$2"
  local metrics_path="$3"

  "$PYTHON_BIN" scripts/train_rmo_dpo.py --config "$config_path"
  "$PYTHON_BIN" scripts/evaluate_helpsteer2.py \
    --config "$config_path" \
    --checkpoint "$checkpoint_dir" \
    --output_json "$metrics_path"
}

wait_for_current_rmo_run

run_one \
  "configs/noise_sweep/helpsteer2_weighted_dpo_noise_0.1_baseline.yaml" \
  "outputs/baselines/noise_0.1/weighted_dpo/final" \
  "outputs/baselines/noise_0.1/weighted_dpo_metrics.json"

run_one \
  "configs/noise_sweep/helpsteer2_mo_dpo_noise_0.1_baseline.yaml" \
  "outputs/baselines/noise_0.1/mo_dpo/final" \
  "outputs/baselines/noise_0.1/mo_dpo_metrics.json"

run_one \
  "configs/noise_sweep/helpsteer2_scalarized_robust_dpo_noise_0.1_baseline.yaml" \
  "outputs/baselines/noise_0.1/scalarized_robust_dpo/final" \
  "outputs/baselines/noise_0.1/scalarized_robust_dpo_metrics.json"
