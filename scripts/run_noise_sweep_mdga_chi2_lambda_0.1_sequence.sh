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

export PYTHONUNBUFFERED=1

cd "$ROOT_DIR"
source .venv/bin/activate

noise_rates=("$@")
if [[ ${#noise_rates[@]} -eq 0 ]]; then
  noise_rates=(0.1 0.2 0.3)
fi

latest_checkpoint_path() {
  local output_dir="$1"
  local latest_step

  if [[ ! -d "$output_dir" ]]; then
    return 0
  fi

  latest_step="$(
    find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' |
      sed 's#.*/checkpoint-##' |
      sort -n |
      tail -n 1
  )"
  if [[ -n "$latest_step" ]]; then
    printf '%s/checkpoint-%s\n' "$output_dir" "$latest_step"
  fi
}

printf '[%s] using CUDA_VISIBLE_DEVICES=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CUDA_VISIBLE_DEVICES:-unset}"

resume_config="configs/noise_sweep_mdga_chi2/helpsteer2_rmo_dpo_noise_0.0_lambda_0.1.yaml"
resume_output_dir="outputs/noise_sweep_mdga_chi2/noise_0.0_lambda_0.1"
resume_final_dir="$resume_output_dir/final"
resume_checkpoint="$(latest_checkpoint_path "$resume_output_dir")"

if [[ -d "$resume_final_dir" ]]; then
  printf '[%s] noise_0.0_lambda_0.1 already complete; skipping clean-data resume\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elif [[ -n "$resume_checkpoint" ]]; then
  printf '[%s] resume noise_0.0_lambda_0.1 from %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$resume_checkpoint"
  python scripts/train_rmo_dpo.py --config "$resume_config" --resume_from "$resume_checkpoint"
else
  printf '[%s] no checkpoint found for noise_0.0_lambda_0.1; skipping clean-data resume\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

for noise_rate in "${noise_rates[@]}"; do
  config_path="configs/noise_sweep_mdga_chi2/helpsteer2_rmo_dpo_noise_${noise_rate}_lambda_0.1.yaml"
  output_dir="outputs/noise_sweep_mdga_chi2/noise_${noise_rate}_lambda_0.1"
  final_dir="$output_dir/final"
  resume_checkpoint="$(latest_checkpoint_path "$output_dir")"

  if [[ -d "$final_dir" ]]; then
    printf '[%s] noise_%s_lambda_0.1 already complete; skipping\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$noise_rate"
    continue
  fi

  if [[ -n "$resume_checkpoint" ]]; then
    printf '[%s] resume noise_%s_lambda_0.1 from %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$noise_rate" "$resume_checkpoint"
    python scripts/train_rmo_dpo.py --config "$config_path" --resume_from "$resume_checkpoint"
  else
    printf '[%s] start noise_%s_lambda_0.1\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$noise_rate"
    bash scripts/train_rmo_dpo_gpu.sh "$config_path"
  fi
done
