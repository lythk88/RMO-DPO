#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMO_REPO_ROOT="${RMO_REPO_ROOT:-$ROOT_DIR}"
RACO_REPO_ROOT="${RACO_REPO_ROOT:-/home/namn1/RACO/trl}"
RACO_OUTPUT_ROOT="${RACO_OUTPUT_ROOT:-/home/namn1/RACO/outputs}"
RMO_OUTPUT_ROOT="${RMO_OUTPUT_ROOT:-$RMO_REPO_ROOT/outputs/noise_sweep}"
RACO_VALIDATION_JSONL="${RACO_VALIDATION_JSONL:-$RMO_REPO_ROOT/data/helpsteer2_pairs_raco/validation.jsonl}"
SESSION_NAME="${SESSION_NAME:-helpsteer2-ckpt-eval}"
SESSION_DIR="$RMO_REPO_ROOT/outputs/noise_sweep/logs/$SESSION_NAME"
GPU_LIST="${GPU_LIST:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session_name)
      SESSION_NAME="$2"
      SESSION_DIR="$RMO_REPO_ROOT/outputs/noise_sweep/logs/$SESSION_NAME"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --skip_existing)
      SKIP_EXISTING=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$SESSION_DIR"

if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -r -a GPU_IDS <<<"$GPU_LIST"
else
  mapfile -t GPU_IDS < <(
    for info_file in /proc/driver/nvidia/gpus/*/information; do
      awk -F: '/Device Minor/ {gsub(/[[:space:]]/, "", $2); print $2}' "$info_file"
    done | sort -n
  )
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPUs detected." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  exit 1
fi

ALL_TASKS_FILE="$SESSION_DIR/tasks.all.tsv"
: >"$ALL_TASKS_FILE"

for idx in "${!GPU_IDS[@]}"; do
  : >"$SESSION_DIR/worker-$idx.tasks.tsv"
  rm -f "$SESSION_DIR/worker-$idx.status"
done

task_index=0

for noise_dir in "$RMO_OUTPUT_ROOT"/noise_*; do
  [[ -d "$noise_dir" ]] || continue
  noise_tag="${noise_dir##*/noise_}"
  config_rel="configs/noise_sweep/helpsteer2_rmo_dpo_noise_${noise_tag}.yaml"
  output_dir_rel="outputs/noise_sweep/noise_${noise_tag}/evals_clean_validation"

  while IFS= read -r checkpoint_dir; do
    checkpoint_rel="${checkpoint_dir#$RMO_REPO_ROOT/}"
    checkpoint_name="$(basename "$checkpoint_dir")"
    output_rel="${output_dir_rel}/${checkpoint_name}.json"
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$RMO_REPO_ROOT/$output_rel" ]]; then
      continue
    fi
    worker_index=$((task_index % ${#GPU_IDS[@]}))
    printf "rmo_pair\t%s\t%s\t%s\t-\n" "$config_rel" "$checkpoint_rel" "$output_rel" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
    task_index=$((task_index + 1))
  done < <(find "$noise_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

  final_dir="$noise_dir/final"
  if [[ -d "$final_dir" ]]; then
    output_rel="${output_dir_rel}/final.json"
    if [[ "$SKIP_EXISTING" -ne 1 || ! -f "$RMO_REPO_ROOT/$output_rel" ]]; then
      worker_index=$((task_index % ${#GPU_IDS[@]}))
      printf "rmo_pair\t%s\t%s\t%s\t-\n" "$config_rel" "${final_dir#$RMO_REPO_ROOT/}" "$output_rel" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
      task_index=$((task_index + 1))
    fi
  fi
done

for family_dir in "$RACO_OUTPUT_ROOT"/*; do
  [[ -d "$family_dir" ]] || continue

  run_roots=()
  if [[ -f "$family_dir/adapter_config.json" ]]; then
    run_roots+=("$family_dir")
  fi
  while IFS= read -r nested_root; do
    run_roots+=("$nested_root")
  done < <(find "$family_dir" -maxdepth 1 -mindepth 1 -type d -exec test -f '{}/adapter_config.json' ';' -print | sort -V)

  for run_root in "${run_roots[@]}"; do
    while IFS= read -r checkpoint_dir; do
      checkpoint_name="$(basename "$checkpoint_dir")"
      output_json="$run_root/evals_raco_validation/${checkpoint_name}.json"
      if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_json" ]]; then
        continue
      fi
      worker_index=$((task_index % ${#GPU_IDS[@]}))
      printf "raco_jsonl\t%s\t%s\t%s\t%s\n" "$checkpoint_dir" "$RACO_VALIDATION_JSONL" "$output_json" "$RACO_REPO_ROOT" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
      task_index=$((task_index + 1))
    done < <(find "$run_root" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

    output_json="$run_root/evals_raco_validation/final.json"
    if [[ "$SKIP_EXISTING" -ne 1 || ! -f "$output_json" ]]; then
      worker_index=$((task_index % ${#GPU_IDS[@]}))
      printf "raco_jsonl\t%s\t%s\t%s\t%s\n" "$run_root" "$RACO_VALIDATION_JSONL" "$output_json" "$RACO_REPO_ROOT" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
      task_index=$((task_index + 1))
    fi
  done
done

printf "%s\n" "${#GPU_IDS[@]}" >"$SESSION_DIR/worker_count.txt"
printf "%s\n" "$task_index" >"$SESSION_DIR/task_count.txt"

for idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$idx]}"
  task_file="$SESSION_DIR/worker-$idx.tasks.tsv"
  log_file="$SESSION_DIR/worker-$idx.log"
  status_file="$SESSION_DIR/worker-$idx.status"
  worker_cmd="cd '$RMO_REPO_ROOT' && bash scripts/run_helpsteer2_checkpoint_eval_worker.sh --rmo_repo_root '$RMO_REPO_ROOT' --task_file '$task_file' --gpu '$gpu_id' --log_file '$log_file' --status_file '$status_file'; rc=\$?; echo; echo worker-$idx exit=\$rc; exec bash"
  if [[ "$idx" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "bash -lc \"$worker_cmd\""
  else
    tmux new-window -t "$SESSION_NAME:" -n "gpu${gpu_id}" "bash -lc \"$worker_cmd\""
  fi
done

echo "Launched tmux session: $SESSION_NAME"
echo "Session dir: $SESSION_DIR"
echo "GPU ids: ${GPU_IDS[*]}"
echo "Queued tasks: $task_index"
