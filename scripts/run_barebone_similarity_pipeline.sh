#!/usr/bin/env bash
# Sequential barebone designated-layer extract (3 models) + similarity matrices.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export AB_PAIRS_VARIANT=neutral
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="logs/barebone_extract_${STAMP}.log"
STATUS="logs/barebone_extract_status.txt"
mkdir -p logs

write_status() {
  {
    echo "updated=$(date -Iseconds)"
    echo "phase=$1"
    echo "detail=$2"
    echo "log=$LOG"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' || true
  } >"$STATUS"
}

{
  echo "================================================================"
  echo "[$(date -Iseconds)] Barebone designated-layer extract start"
  echo "  log=$LOG"
  echo "================================================================"
  write_status "extract" "Qwen3-32B → Qwen3.5-27B → Llama-70B (one model at a time)"

  # Both GPUs visible; only one model loaded at a time (script enforces no CPU offload).
  .venv/bin/python scripts/run_barebone_designated_extract.py

  write_status "similarity_barebone" "computing barebone similarity matrices"
  .venv/bin/python run_steering_vector_similarity.py --template barebone --group steering-extract

  write_status "similarity_standard" "computing standard similarity matrices from existing vectors"
  .venv/bin/python run_steering_vector_similarity.py --template standard --group steering-extract

  write_status "DONE" "barebone extract + standard/barebone similarity matrices ready"
  echo "[$(date -Iseconds)] Pipeline COMPLETE"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
