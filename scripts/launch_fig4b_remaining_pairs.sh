#!/usr/bin/env bash
# Run remaining figure4b instruction pairs (Be sure! already done; skipped automatically).
#   GPU0: Qwen3.5-27B
#   GPU1: Qwen3-32B, then Qwen3.5-35B-A3B
#   Then both GPUs: Llama-70B
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="logs/fig4b_remaining_pairs_status_${STAMP}.txt"
LOG27="logs/fig4b_remaining_qwen35_27b_${STAMP}.log"
LOG32="logs/fig4b_remaining_qwen3_32b_${STAMP}.log"
LOG35="logs/fig4b_remaining_qwen35_35b_${STAMP}.log"
LOGLAMA="logs/fig4b_remaining_llama70b_${STAMP}.log"
COORD_LOG="logs/fig4b_remaining_coord_${STAMP}.log"

set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a
export PATH="${HOME}/.local/bin:${PATH}"
export AB_PAIRS_VARIANT=neutral
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

PY="${PWD}/.venv/bin/python"

{
  echo "STAMP=$STAMP"
  echo "STARTED_AT=$(date -Is)"
  echo "LOG27=$LOG27"
  echo "LOG32=$LOG32"
  echo "LOG35=$LOG35"
  echo "LOGLAMA=$LOGLAMA"
} | tee "$STATUS_FILE"

# --- GPU0: Qwen3.5-27B ---
screen -dmS fig4b_27b bash -lc "
  set -euo pipefail
  cd '$PWD'
  set -a; [ -f .env ] && source .env; set +a
  export AB_PAIRS_VARIANT=neutral
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
  export CUDA_VISIBLE_DEVICES=0
  echo \"[27b] start \$(date -Is)\" | tee -a '$STATUS_FILE'
  '${PY}' -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model qwen/Qwen3.5-27B \
    2>&1 | tee '$LOG27'
  echo \"[27b] DONE \$(date -Is)\" | tee -a '$STATUS_FILE'
  echo DONE > 'logs/fig4b_remaining_27b_${STAMP}.done'
"

# --- GPU1: Qwen3-32B then Qwen3.5-35B-A3B ---
screen -dmS fig4b_qwen bash -lc "
  set -euo pipefail
  cd '$PWD'
  set -a; [ -f .env ] && source .env; set +a
  export AB_PAIRS_VARIANT=neutral
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
  export CUDA_VISIBLE_DEVICES=1
  echo \"[32b] start \$(date -Is)\" | tee -a '$STATUS_FILE'
  '${PY}' -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model qwen/Qwen3-32B \
    2>&1 | tee '$LOG32'
  echo \"[32b] DONE \$(date -Is)\" | tee -a '$STATUS_FILE'
  echo DONE > 'logs/fig4b_remaining_32b_${STAMP}.done'

  echo \"[35b] start \$(date -Is)\" | tee -a '$STATUS_FILE'
  '${PY}' -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model qwen/Qwen3.5-35B-A3B \
    2>&1 | tee '$LOG35'
  echo \"[35b] DONE \$(date -Is)\" | tee -a '$STATUS_FILE'
  echo DONE > 'logs/fig4b_remaining_35b_${STAMP}.done'
"

# --- Coordinator: after Qwen waves, run Llama on both GPUs ---
screen -dmS fig4b_coord bash -lc "
  set -euo pipefail
  cd '$PWD'
  set -a; [ -f .env ] && source .env; set +a
  export AB_PAIRS_VARIANT=neutral
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
  PY='$PY'
  STAMP='$STAMP'
  STATUS_FILE='$STATUS_FILE'
  LOGLAMA='$LOGLAMA'
  exec > >(tee -a '$COORD_LOG') 2>&1
  echo \"[coord] waiting for Qwen 27B/32B/35B done markers...\"
  while [ ! -f \"logs/fig4b_remaining_27b_\${STAMP}.done\" ]; do sleep 30; done
  while [ ! -f \"logs/fig4b_remaining_32b_\${STAMP}.done\" ]; do sleep 30; done
  while [ ! -f \"logs/fig4b_remaining_35b_\${STAMP}.done\" ]; do sleep 30; done
  echo \"[coord] Qwen waves done; starting Llama-70B on GPU0+1 \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  for _ in \$(seq 1 60); do
    if ! pgrep -af 'run_paper.py' | grep -v fig4b_coord | grep -q .; then
      break
    fi
    sleep 20
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a \"\$STATUS_FILE\"
  export CUDA_VISIBLE_DEVICES=0,1
  \"\$PY\" -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model meta-llama/Meta-Llama-3-70B-Instruct \
    2>&1 | tee \"\$LOGLAMA\"
  echo \"[coord] Llama DONE \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  echo ALL_DONE > \"logs/fig4b_remaining_all_\${STAMP}.done\"
  echo \"[coord] ALL DONE \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
"

sleep 3
screen -ls | tee -a "$STATUS_FILE"
echo "STATUS_FILE=$STATUS_FILE"
