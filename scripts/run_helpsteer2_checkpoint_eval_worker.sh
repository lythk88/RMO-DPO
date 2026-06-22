#!/usr/bin/env bash
set -euo pipefail

TASK_FILE=""
GPU_ID=""
LOG_FILE=""
STATUS_FILE=""
RMO_REPO_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task_file)
      TASK_FILE="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --log_file)
      LOG_FILE="$2"
      shift 2
      ;;
    --status_file)
      STATUS_FILE="$2"
      shift 2
      ;;
    --rmo_repo_root)
      RMO_REPO_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_FILE" || -z "$GPU_ID" || -z "$LOG_FILE" || -z "$STATUS_FILE" || -z "$RMO_REPO_ROOT" ]]; then
  echo "Missing required arguments." >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATUS_FILE")"
: >"$LOG_FILE"

rc=1
trap 'printf "%s\n" "$rc" >"$STATUS_FILE"' EXIT

NVIDIA_LIB_DIR="$RMO_REPO_ROOT/vendor/nvidia-570/extracted/usr/lib/x86_64-linux-gnu"
if [[ -d "$NVIDIA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$RMO_REPO_ROOT"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU_ID"

task_count="$(grep -cve '^[[:space:]]*$' "$TASK_FILE" || true)"
echo "[$(date -u +%FT%TZ)] worker starting on GPU $GPU_ID with $task_count task(s)" | tee -a "$LOG_FILE"

if [[ "$task_count" == "0" ]]; then
  echo "No tasks assigned to GPU $GPU_ID" | tee -a "$LOG_FILE"
  rc=0
  exit 0
fi

task_index=0
while IFS=$'\t' read -r mode arg1 arg2 arg3 arg4; do
  if [[ -z "${mode:-}" ]]; then
    continue
  fi
  task_index=$((task_index + 1))
  echo "[$(date -u +%FT%TZ)] task $task_index/$task_count gpu=$GPU_ID mode=$mode target=$arg2" | tee -a "$LOG_FILE"

  case "$mode" in
    rmo_pair)
      python scripts/evaluate_helpsteer2.py \
        --config "$arg1" \
        --checkpoint "$arg2" \
        --split validation \
        --output_json "$arg3" 2>&1 | tee -a "$LOG_FILE"
      cmd_rc=${PIPESTATUS[0]}
      ;;
    raco_jsonl)
      PYTHONPATH="$arg4" python "$arg4/scripts/evaluate_raco_validation.py" \
        --checkpoint "$arg1" \
        --validation_jsonl "$arg2" \
        --output_json "$arg3" 2>&1 | tee -a "$LOG_FILE"
      cmd_rc=${PIPESTATUS[0]}
      ;;
    *)
      echo "Unknown task mode: $mode" | tee -a "$LOG_FILE"
      exit 2
      ;;
  esac

  if [[ "$cmd_rc" -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] task failed with exit code $cmd_rc" | tee -a "$LOG_FILE"
    rc="$cmd_rc"
    exit "$cmd_rc"
  fi
done <"$TASK_FILE"

echo "[$(date -u +%FT%TZ)] worker completed successfully on GPU $GPU_ID" | tee -a "$LOG_FILE"
rc=0
