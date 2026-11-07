#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

# Stop prior monitors / broken launches
pkill -f 'AGENT_LOOP_TICK_mmlu' 2>/dev/null || true
pkill -f 'AGENT_LOOP_TICK_mmlu32' 2>/dev/null || true
sleep 1

VEC="artifacts/neutral/steering_layers/steering_vectors/Qwen_Qwen3-32B/Be_sure_vs_Be_random/layer_56.pt"
if head -c 20 "$VEC" | grep -q 'version https://'; then
  git lfs pull --include="artifacts/neutral/steering_layers/steering_vectors/Qwen_Qwen3-32B/Be_sure_vs_Be_random/**"
fi

set -a
[ -f .env ] && source .env
set +a
export AB_PAIRS_VARIANT=neutral
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

: > logs/mmlu_qwen3_32b_20260721.log
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u - <<'PY' > logs/mmlu_qwen3_32b_20260721.log 2>&1 &
import os
os.environ["AB_PAIRS_VARIANT"] = "neutral"
print("AB_PAIRS_VARIANT=", os.environ.get("AB_PAIRS_VARIANT"), flush=True)
print("starting mmlu_entropy_steering for Qwen/Qwen3-32B", flush=True)

from be_sure_or_random.experiments import (
    run_mmlu_entropy_steering,
    run_mmlu_random_vector_control,
)

entropy_path = run_mmlu_entropy_steering(
    model_name="Qwen/Qwen3-32B",
    vector_certainty_instruction="Be sure!",
    vector_uncertainty_instruction="Be random!",
    steering_strength=2.0,
    steering_conditions=["none", "increase_entropy", "decrease_entropy"],
    fixed_layer=56,
    dataset_name="cais/mmlu",
    dataset_config="all",
    split="test",
    max_examples=None,
    batch_size=16,
    prompt_variant="format_instructed",
)
print("entropy done:", entropy_path, flush=True)

print("starting mmlu_random_vector_control", flush=True)
random_path = run_mmlu_random_vector_control(
    model_name="Qwen/Qwen3-32B",
    vector_certainty_instruction="Be sure!",
    vector_uncertainty_instruction="Be random!",
    steering_strength=2.0,
    fixed_layer=56,
    n_random_vectors=8,
    random_seed=0,
    dataset_name="cais/mmlu",
    dataset_config="all",
    split="test",
    max_examples=None,
    batch_size=16,
    prompt_variant="format_instructed",
)
print("random control done:", random_path, flush=True)
print("ALL DONE", flush=True)
PY

JOBPID=$!
echo "STARTED_32B_PID=$JOBPID" | tee logs/mmlu32_launch_status.txt
sleep 8
ps -p "$JOBPID" -o pid=,etime=,cmd= | tee -a logs/mmlu32_launch_status.txt || echo FAILED_TO_STAY_UP | tee -a logs/mmlu32_launch_status.txt
head -n 30 logs/mmlu_qwen3_32b_20260721.log | tee -a logs/mmlu32_launch_status.txt
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a logs/mmlu32_launch_status.txt

# 10-minute monitor for 32B
nohup bash -c 'while true; do sleep 600; echo "AGENT_LOOP_TICK_mmlu32 {\"prompt\":\"Check MMLU Qwen3-32B logs/mmlu_qwen3_32b_20260721.log and process. Summarize progress. If ALL DONE stop loop and report paths. If failed stop and report error.\"}"; done' > logs/mmlu32_monitor_loop.log 2>&1 &
echo "MONITOR_PID=$!" | tee -a logs/mmlu32_launch_status.txt
