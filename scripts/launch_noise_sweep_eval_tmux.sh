#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$ROOT_DIR"
SPLIT="validation"
EVAL_DIR_NAME="evals"
SESSION_NAME="noise-sweep-eval"
PLOT_OUTPUT_DIR="outputs/noise_sweep/posthoc_analysis"
SKIP_EXISTING=0
GPU_LIST=""
NOISE_RATES=("0.1" "0.2" "0.3")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo_root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --eval_dir_name)
      EVAL_DIR_NAME="$2"
      shift 2
      ;;
    --session_name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --plot_output_dir)
      PLOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip_existing)
      SKIP_EXISTING=1
      shift
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --noise_rates)
      shift
      NOISE_RATES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        NOISE_RATES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
SESSION_DIR="$REPO_ROOT/outputs/noise_sweep/logs/$SESSION_NAME"
mkdir -p "$SESSION_DIR"

if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -r -a GPU_IDS <<<"$GPU_LIST"
else
  mapfile -t GPU_IDS < <(
    for info_file in /proc/driver/nvidia/gpus/*/information; do
      awk -F: '/Device Minor/ {gsub(/ /, "", $2); print $2}' "$info_file"
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
rm -f "$SESSION_DIR/plot.status"

task_index=0
for noise_rate in "${NOISE_RATES[@]}"; do
  noise_dir="$REPO_ROOT/outputs/noise_sweep/noise_${noise_rate}"
  config_rel="configs/noise_sweep/helpsteer2_rmo_dpo_noise_${noise_rate}.yaml"
  eval_dir="$noise_dir/$EVAL_DIR_NAME"
  mkdir -p "$eval_dir"

  while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "$checkpoint_dir")"
    checkpoint_rel="${checkpoint_dir#$REPO_ROOT/}"
    output_rel="outputs/noise_sweep/noise_${noise_rate}/${EVAL_DIR_NAME}/${checkpoint_name}.json"
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$REPO_ROOT/$output_rel" ]]; then
      continue
    fi
    worker_index=$((task_index % ${#GPU_IDS[@]}))
    printf "%s\t%s\t%s\n" "$config_rel" "$checkpoint_rel" "$output_rel" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
    task_index=$((task_index + 1))
  done < <(find "$noise_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

  final_dir="$noise_dir/final"
  if [[ -d "$final_dir" ]]; then
    output_rel="outputs/noise_sweep/noise_${noise_rate}/${EVAL_DIR_NAME}/final.json"
    if [[ "$SKIP_EXISTING" -ne 1 || ! -f "$REPO_ROOT/$output_rel" ]]; then
      worker_index=$((task_index % ${#GPU_IDS[@]}))
      printf "%s\t%s\t%s\n" "$config_rel" "${final_dir#$REPO_ROOT/}" "$output_rel" | tee -a "$ALL_TASKS_FILE" >>"$SESSION_DIR/worker-$worker_index.tasks.tsv"
      task_index=$((task_index + 1))
    fi
  fi
done

printf "%s\n" "${#GPU_IDS[@]}" >"$SESSION_DIR/worker_count.txt"
printf "%s\n" "$task_index" >"$SESSION_DIR/task_count.txt"
printf "%s\n" "$SPLIT" >"$SESSION_DIR/split.txt"

for idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$idx]}"
  task_file="$SESSION_DIR/worker-$idx.tasks.tsv"
  log_file="$SESSION_DIR/worker-$idx.log"
  status_file="$SESSION_DIR/worker-$idx.status"
  worker_cmd="cd '$REPO_ROOT' && bash scripts/run_noise_sweep_eval_worker.sh --repo_root '$REPO_ROOT' --task_file '$task_file' --gpu '$gpu_id' --split '$SPLIT' --log_file '$log_file' --status_file '$status_file'; rc=\$?; echo; echo worker-$idx exit=\$rc; exec bash"
  if [[ "$idx" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "bash -lc \"$worker_cmd\""
  else
    tmux new-window -t "$SESSION_NAME:" -n "gpu${gpu_id}" "bash -lc \"$worker_cmd\""
  fi
done

plot_log="$SESSION_DIR/plot.log"
plot_status="$SESSION_DIR/plot.status"
plot_cmd="cd '$REPO_ROOT' && \
  while true; do \
    ready=0; \
    for status_file in '$SESSION_DIR'/worker-*.status; do \
      if [[ -f \"\$status_file\" ]]; then ready=\$((ready + 1)); fi; \
    done; \
    if [[ \"\$ready\" -ge '${#GPU_IDS[@]}' ]]; then break; fi; \
    sleep 30; \
  done; \
  source .venv/bin/activate; \
  python scripts/plot_noise_sweep_posthoc_eval_analysis.py --repo_root '$REPO_ROOT' --eval_dir_name '$EVAL_DIR_NAME' --output_dir '$PLOT_OUTPUT_DIR' --noise_rates ${NOISE_RATES[*]} 2>&1 | tee '$plot_log'; \
  rc=\${PIPESTATUS[0]}; \
  printf '%s\n' \"\$rc\" > '$plot_status'; \
  echo; echo plot exit=\$rc; \
  exec bash"
tmux new-window -t "$SESSION_NAME:" -n "plot" "bash -lc \"$plot_cmd\""

echo "Launched tmux session: $SESSION_NAME"
echo "Session dir: $SESSION_DIR"
echo "GPU ids: ${GPU_IDS[*]}"
echo "Queued tasks: $task_index"
