#!/usr/bin/env bash
# Overnight dual-GPU launch:
#   GPU0: prompt-only MMLU on Qwen3.5-27B (3 system prompts)
#   GPU1: figure4b prompt-structure ablations on Qwen3-32B + Qwen3.5-35B-A3B
#   Then GPU0: figure4b for Qwen3.5-27B (after MMLU finishes)
#   Then both GPUs: figure4b for Llama-70B (only if both free / prior jobs done)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="logs/overnight_mmlu_fig4b_status_${STAMP}.txt"
MMLU_LOG="logs/mmlu_prompt_conditions_qwen35_27b_${STAMP}.log"
FIG4B_QWEN_LOG="logs/fig4b_qwen_32b_35b_${STAMP}.log"
FIG4B_27B_LOG="logs/fig4b_qwen35_27b_${STAMP}.log"
FIG4B_LLAMA_LOG="logs/fig4b_llama70b_${STAMP}.log"
COORD_LOG="logs/overnight_coord_${STAMP}.log"

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
RUN=("$PY" -u run_paper.py run --ab-pairs neutral)

{
  echo "STAMP=$STAMP"
  echo "STARTED_AT=$(date -Is)"
  echo "HF_HOME=$HF_HOME"
  echo "MMLU_LOG=$MMLU_LOG"
  echo "FIG4B_QWEN_LOG=$FIG4B_QWEN_LOG"
  echo "FIG4B_27B_LOG=$FIG4B_27B_LOG"
  echo "FIG4B_LLAMA_LOG=$FIG4B_LLAMA_LOG"
} | tee "$STATUS_FILE"

# --- GPU0: MMLU prompt conditions ---
screen -dmS mmlu_prompt bash -lc "
  set -euo pipefail
  cd '$PWD'
  set -a; [ -f .env ] && source .env; set +a
  export AB_PAIRS_VARIANT=neutral
  export HF_HOME='$HF_HOME'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
  export CUDA_VISIBLE_DEVICES=0
  echo \"[mmlu] start \$(date -Is)\" | tee -a '$STATUS_FILE'
  '${PY}' -u run_paper.py run --ab-pairs neutral --figure 7 --group mmlu-prompt-conditions \
    2>&1 | tee '$MMLU_LOG'
  echo \"[mmlu] DONE \$(date -Is)\" | tee -a '$STATUS_FILE'
  echo MMLU_DONE > 'logs/mmlu_prompt_conditions_${STAMP}.done'
"

# --- GPU1: fig4b Qwen 32B + 35B-A3B (skip 27B to avoid download/load clash with MMLU) ---
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
  echo \"[fig4b-qwen] start \$(date -Is)\" | tee -a '$STATUS_FILE'
  '${PY}' -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model qwen/Qwen3-32B \
    --include-model qwen/Qwen3.5-35B-A3B \
    2>&1 | tee '$FIG4B_QWEN_LOG'
  echo \"[fig4b-qwen] DONE \$(date -Is)\" | tee -a '$STATUS_FILE'
  echo FIG4B_QWEN_DONE > 'logs/fig4b_qwen_${STAMP}.done'
"

# --- Coordinator: after MMLU, run fig4b 27B on GPU0; after both Qwen waves, run Llama on 0+1 ---
screen -dmS overnight_coord bash -lc "
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
  FIG4B_27B_LOG='$FIG4B_27B_LOG'
  FIG4B_LLAMA_LOG='$FIG4B_LLAMA_LOG'
  exec > >(tee -a '$COORD_LOG') 2>&1
  echo \"[coord] waiting for MMLU done marker...\"
  while [ ! -f \"logs/mmlu_prompt_conditions_\${STAMP}.done\" ]; do sleep 30; done
  echo \"[coord] MMLU finished; starting fig4b Qwen3.5-27B on GPU0 \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  export CUDA_VISIBLE_DEVICES=0
  \"\$PY\" -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model qwen/Qwen3.5-27B \
    2>&1 | tee \"\$FIG4B_27B_LOG\"
  echo \"[coord] fig4b 27B DONE \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  echo FIG4B_27B_DONE > \"logs/fig4b_27b_\${STAMP}.done\"

  echo \"[coord] waiting for fig4b Qwen 32B/35B done marker...\"
  while [ ! -f \"logs/fig4b_qwen_\${STAMP}.done\" ]; do sleep 30; done
  echo \"[coord] both Qwen waves done; checking GPUs before Llama \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  # Wait until no other python run_paper holds GPUs (best-effort)
  for _ in \$(seq 1 60); do
    if ! pgrep -af 'run_paper.py' | grep -v overnight_coord | grep -q .; then
      break
    fi
    sleep 30
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a \"\$STATUS_FILE\"
  echo \"[coord] starting fig4b Llama-70B on GPU0+1 \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  export CUDA_VISIBLE_DEVICES=0,1
  \"\$PY\" -u run_paper.py run --ab-pairs neutral --figure 4b --group prompt-structure-ablations \
    --include-model meta-llama/Meta-Llama-3-70B-Instruct \
    2>&1 | tee \"\$FIG4B_LLAMA_LOG\"
  echo \"[coord] Llama DONE \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
  echo ALL_DONE > \"logs/overnight_all_\${STAMP}.done\"
  echo \"[coord] ALL DONE \$(date -Is)\" | tee -a \"\$STATUS_FILE\"
"

sleep 3
screen -ls | tee -a "$STATUS_FILE"
echo "STATUS_FILE=$STATUS_FILE"
echo "Screens: mmlu_prompt, fig4b_qwen, overnight_coord"
