#!/usr/bin/env bash
# Sequential barebone every-4th-layer extract: one model per process (hard GPU free).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export AB_PAIRS_VARIANT=neutral
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="logs/barebone_full_sweep_${STAMP}.log"
STATUS="logs/barebone_full_sweep_status.txt"
mkdir -p logs

MODELS=(
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3.5-27B"
  "meta-llama/Meta-Llama-3-70B-Instruct"
)

write_status() {
  {
    echo "updated=$(date -Iseconds)"
    echo "phase=$1"
    echo "detail=$2"
    echo "log=$LOG"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' || true
    echo "activations_done=$(find artifacts/neutral/steering_layers/activations/barebone -name '*_activations.pt' 2>/dev/null | wc -l)/21"
    echo "vectors_barebone=$(find artifacts/neutral/steering_layers/steering_vectors/barebone -name 'layer_*.pt' 2>/dev/null | wc -l)"
  } >"$STATUS"
}

{
  echo "================================================================"
  echo "[$(date -Iseconds)] Barebone full-sweep start (sequential, 1 process/model)"
  echo "  log=$LOG"
  echo "================================================================"

  for model in "${MODELS[@]}"; do
    write_status "extract" "$model"
    echo
    echo "[$(date -Iseconds)] === extract $model (fresh process) ==="
    .venv/bin/python scripts/run_barebone_full_sweep.py --include-model "$model"
    echo "[$(date -Iseconds)] === freed after $model ==="
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
  done

  write_status "compute" "compute_all_steering_vectors(barebone)"
  echo
  echo "[$(date -Iseconds)] === compute barebone steering vectors ==="
  .venv/bin/python - <<'PY'
from be_sure_or_random.experiments import compute_all_steering_vectors
paths = compute_all_steering_vectors(template_name="barebone")
print(f"Wrote {len(paths)} vector files")
PY

  write_status "DONE" "barebone full-sweep extract + compute complete"
  echo "[$(date -Iseconds)] Pipeline COMPLETE"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
