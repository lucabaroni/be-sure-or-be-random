#!/usr/bin/env bash
# Full CoT thinking-sampling sweep over 3 system prompts on Qwen3.5-27B.
#
# Schedule:
#   Wave 1 — GPU0 + GPU1 in parallel (TP=1 each)
#   Wave 2 — remaining prompt on GPU0 (TP=1)
#
# Usage:
#   ./scripts/launch_thinking_sampling_3prompt_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export AB_PAIRS_VARIANT=neutral
export PYTHONUNBUFFERED=1

PY="$ROOT/.venv/bin/python"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/thinking_sweep_${TS}"
mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen3.5-27B"
INSTR='Be sure!:Be random!'

# Shared generation / memory knobs (single-GPU).
# Qwen3.5-27B bf16 ~54GB; leave headroom for long CoTs + logprob scoring.
COMMON_ARGS=(
  --model "$MODEL"
  --instruction-pair "$INSTR"
  --samples 30
  --temperature 1.0
  --max-tokens 6144
  --max-model-len 10240
  --max-num-batched-tokens 2048
  --max-num-seqs 256
  --gpu-memory-utilization 0.92
  --logprob-batch-size 256
  --tensor-parallel-size 1
)

run_one() {
  local gpu="$1"
  local prompt="$2"
  local slug
  slug="$(basename "$prompt" .txt)"
  local logfile="$LOG_DIR/${slug}_gpu${gpu}.log"
  local statusfile="$LOG_DIR/${slug}_gpu${gpu}.status"

  echo "[$(date -u +%H:%M:%S)] START gpu=${gpu} prompt=${slug}" | tee -a "$LOG_DIR/master.log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_thinking_sampling.py \
    --system-prompt "$prompt" \
    "${COMMON_ARGS[@]}" \
    >"$logfile" 2>&1
  local rc=$?
  set -e
  echo "$rc" >"$statusfile"
  if [[ "$rc" -eq 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] DONE  gpu=${gpu} prompt=${slug} rc=0" | tee -a "$LOG_DIR/master.log"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL  gpu=${gpu} prompt=${slug} rc=${rc}" | tee -a "$LOG_DIR/master.log"
  fi
  return "$rc"
}

echo "==============================================" | tee "$LOG_DIR/master.log"
echo "Thinking-sampling 3-prompt sweep" | tee -a "$LOG_DIR/master.log"
echo "Started: $(date -u)" | tee -a "$LOG_DIR/master.log"
echo "Logs:    $LOG_DIR" | tee -a "$LOG_DIR/master.log"
echo "Wave 1:  standard_system (GPU0) + cot_across_trajectory_system (GPU1)" | tee -a "$LOG_DIR/master.log"
echo "Wave 2:  cot_within_and_across_system (GPU0)" | tee -a "$LOG_DIR/master.log"
echo "==============================================" | tee -a "$LOG_DIR/master.log"

# ---- Wave 1: two prompts in parallel ----
run_one 0 standard_system.txt &
PID0=$!
run_one 1 cot_across_trajectory_system.txt &
PID1=$!

set +e
wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?
set -e

echo "[$(date -u +%H:%M:%S)] Wave 1 finished: standard_rc=${RC0} across_rc=${RC1}" | tee -a "$LOG_DIR/master.log"

if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  echo "Wave 1 had failures; still launching Wave 2." | tee -a "$LOG_DIR/master.log"
fi

# ---- Wave 2: remaining prompt on GPU0 ----
echo "[$(date -u +%H:%M:%S)] Wave 2 starting on GPU0" | tee -a "$LOG_DIR/master.log"
set +e
run_one 0 cot_within_and_across_system.txt
RC2=$?
set -e

echo "==============================================" | tee -a "$LOG_DIR/master.log"
echo "Finished: $(date -u)" | tee -a "$LOG_DIR/master.log"
echo "Exit codes: standard=${RC0} across=${RC1} within_and_across=${RC2}" | tee -a "$LOG_DIR/master.log"
echo "Results under: results/neutral/thinking_sampling/Qwen_Qwen3.5-27B/<prompt_slug>/" | tee -a "$LOG_DIR/master.log"
echo "==============================================" | tee -a "$LOG_DIR/master.log"

# Non-zero if any failed.
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 || "$RC2" -ne 0 ]]; then
  exit 1
fi
exit 0
