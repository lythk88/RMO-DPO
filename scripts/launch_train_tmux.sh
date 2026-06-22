#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$ROOT_DIR"
CONFIG_REL=""
SESSION_NAME=""
LOG_DIR="outputs/tmux_runs"
TRAIN_SCRIPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_REL="$2"
      shift 2
      ;;
    --session_name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --train_script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG_REL" || -z "$SESSION_NAME" ]]; then
  echo "Usage: $0 --config <config-path> --session_name <name> [--log_dir <dir>] [--train_script <script>]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
CONFIG_PATH="$REPO_ROOT/$CONFIG_REL"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "$TRAIN_SCRIPT" ]]; then
  case "$CONFIG_REL" in
    *rmo_dpo*)
      TRAIN_SCRIPT="scripts/train_rmo_dpo.py"
      ;;
    *scalarized_robust*)
      TRAIN_SCRIPT="scripts/train_scalarized_robust_dpo_baseline.py"
      ;;
    *weighted_dpo*)
      TRAIN_SCRIPT="scripts/train_weighted_dpo_baseline.py"
      ;;
    *mo_dpo*)
      TRAIN_SCRIPT="scripts/train_mo_dpo_baseline.py"
      ;;
    *)
      TRAIN_SCRIPT="scripts/train_rmo_dpo.py"
      ;;
  esac
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/$LOG_DIR"
RUN_STEM="$(basename "$CONFIG_REL" .yaml)"
LOG_FILE="$REPO_ROOT/$LOG_DIR/${RUN_STEM}.log"
STATUS_FILE="$REPO_ROOT/$LOG_DIR/${RUN_STEM}.status"
rm -f "$STATUS_FILE"

NVIDIA_LIB_DIR="$REPO_ROOT/vendor/nvidia-570/extracted/usr/lib/x86_64-linux-gnu"
TMUX_CMD="cd '$REPO_ROOT' && \
  if [[ -d '$NVIDIA_LIB_DIR' ]]; then export LD_LIBRARY_PATH='$NVIDIA_LIB_DIR'\${LD_LIBRARY_PATH:+:\"\$LD_LIBRARY_PATH\"}; fi; \
  source .venv/bin/activate; \
  python '$TRAIN_SCRIPT' --config '$CONFIG_REL' 2>&1 | tee '$LOG_FILE'; \
  rc=\${PIPESTATUS[0]}; \
  printf '%s\n' \"\$rc\" > '$STATUS_FILE'; \
  echo; echo exit=\$rc; \
  exec bash"

tmux new-session -d -s "$SESSION_NAME" "bash -lc \"$TMUX_CMD\""

echo "Launched tmux session: $SESSION_NAME"
echo "Trainer script: $TRAIN_SCRIPT"
echo "Config: $CONFIG_REL"
echo "Log: $LOG_FILE"
echo "Status: $STATUS_FILE"
