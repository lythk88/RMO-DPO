#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 0 )); then
  SESSION_NAME="$1"
  shift
else
  SESSION_NAME="rmo_mdga_chi2_lambda_0_1_$(date -u +%Y%m%dT%H%M%SZ)"
fi
SEQUENCE_ARGS=("$@")
OUTPUT_DIR="$ROOT_DIR/outputs/noise_sweep_mdga_chi2"

SEQUENCE_ARGS_Q=""
if (( ${#SEQUENCE_ARGS[@]} > 0 )); then
  printf -v SEQUENCE_ARGS_Q ' %q' "${SEQUENCE_ARGS[@]}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/${SESSION_NAME}.log"
STATUS_FILE="$OUTPUT_DIR/${SESSION_NAME}.status"
rm -f "$STATUS_FILE"

TMUX_CMD="cd '$ROOT_DIR' && \
  bash scripts/run_noise_sweep_mdga_chi2_lambda_0.1_sequence.sh$SEQUENCE_ARGS_Q 2>&1 | tee '$LOG_FILE'; \
  rc=\${PIPESTATUS[0]}; \
  printf '%s\n' \"\$rc\" > '$STATUS_FILE'; \
  echo; echo exit=\$rc; \
  exec bash"

tmux new-session -d -s "$SESSION_NAME" "bash -lc \"$TMUX_CMD\""

echo "Launched tmux session: $SESSION_NAME"
echo "Log: $LOG_FILE"
echo "Status: $STATUS_FILE"
if (( ${#SEQUENCE_ARGS[@]} > 0 )); then
  echo "Noise rates: ${SEQUENCE_ARGS[*]}"
fi
