# Shared env + helpers for neutral paper launchers.
# Sourced by run_neutral_{local,steering,all}.sh — do not run directly.
#
# HF cache: pin to the large local root disk (not NFS). Override with HF_HOME
# in .env if needed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# Large local disk: / has ~5T free. Keep weights off the NFS mount.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS="${1:-}"
RUN="$PYTHON run_paper.py run --ab-pairs neutral $EXTRA_ARGS"

# Set CLEAN_CACHE=1 before sourcing to wipe hub weights between waves (tight disks only).
CLEAN_CACHE="${CLEAN_CACHE:-0}"
HF_CACHE="$HUGGINGFACE_HUB_CACHE"

clean_all_models() {
    if [[ "$CLEAN_CACHE" != "1" ]]; then
        return 0
    fi
    local before
    before=$(du -sh "${HF_CACHE}" 2>/dev/null | cut -f1 || echo "0")
    echo "  [cache] Clearing downloaded models (was ${before})..."
    rm -rf "${HF_CACHE}"/models--*
}

# Manifest figure-3 local models (exact IDs), excluding Llama-70B (solo dual-GPU wave).
# Paired for roughly balanced GPU0/GPU1 wall time.
FIG3_PAIRS=(
    "qwen/Qwen3-4B"                           "qwen/Qwen3.5-4B"
    "qwen/Qwen3-8B"                           
    "qwen/Qwen3-14B"                          
    "qwen/Qwen3.5-9B"                         "meta-llama/Meta-Llama-3-8B-Instruct"
    "qwen/Qwen3.5-35B-A3B"                    "google/gemma-4-26B-A4B-it"
    "qwen/Qwen3-32B"                          "allenai/Olmo-3.1-32B-Instruct-SFT"
    "qwen/Qwen3.5-27B"                        "allenai/Olmo-3.1-32B-Instruct-DPO"
    "google/gemma-4-31B-it"                   "allenai/Olmo-3.1-32B-Instruct"
)

run_figures_2_3_4() {
    # ── Figure 2 + Figure 4 (non-70B) in parallel ───────────────────────────────
    # Fig4 includes Llama-70B; that model must not run with only one GPU visible.
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figures 2 & 4 — wave 1 (GPU0: fig2; GPU1: fig4 Qwen3.5-27B)"
    echo "══════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=0 $RUN --figure 2 --stage run &
    local PID_FIG2=$!
    CUDA_VISIBLE_DEVICES=1 $RUN --figure 4 --stage run \
        --include-model qwen/Qwen3.5-27B &
    local PID_FIG4A=$!
    wait $PID_FIG2 $PID_FIG4A

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 4 — wave 2 (Qwen3-32B, GPU 0)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 $RUN --figure 4 --stage run \
        --include-model qwen/Qwen3-32B

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 4 — wave 3 (Llama-70B, GPU 0+1)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0,1 $RUN --figure 4 --stage run \
        --include-model meta-llama/Meta-Llama-3-70B-Instruct
    clean_all_models
    echo "→ Figures 2 & 4 done"

    # ── Figure 3: standard + barebone per model (one download / load pass) ─
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 3 — wave 1 (model pairs, GPU 0 / GPU 1)"
    echo "══════════════════════════════════════════════════════════"

    local i M0 M1 PID0 PID1
    for (( i=0; i<${#FIG3_PAIRS[@]}; i+=2 )); do
        M0="${FIG3_PAIRS[$i]}"
        M1="${FIG3_PAIRS[$((i+1))]}"
        echo ""
        echo "  → GPU0: $M0  |  GPU1: $M1"
        CUDA_VISIBLE_DEVICES=0 $RUN --figure 3 --stage run \
            --group local-model-sweep --group local-model-sweep-barebone \
            --include-model "$M0" &
        PID0=$!
        CUDA_VISIBLE_DEVICES=1 $RUN --figure 3 --stage run \
            --group local-model-sweep --group local-model-sweep-barebone \
            --include-model "$M1" &
        PID1=$!
        wait $PID0 $PID1
        clean_all_models
    done
    echo "→ Wave 1 done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 3 — wave 2 (Llama-70B, GPU 0+1)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0,1 $RUN --figure 3 --stage run \
        --group local-model-sweep --group local-model-sweep-barebone \
        --include-model meta-llama/Meta-Llama-3-70B-Instruct
    clean_all_models
    echo "→ Wave 2 done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Local zero-shot / barebone / fig4 complete (figures 2, 3, 4)."
    echo "══════════════════════════════════════════════════════════"
}

run_figure_5() {
    # Extract sequentially: parallel 32B+27B OOM'd (CPU offload / memory contention).
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / prepare / steering-extract"
    echo " Wave 1a — Qwen3-32B (GPU 0)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 $RUN --figure 5 --stage prepare \
        --group steering-extract \
        --include-model Qwen/Qwen3-32B
    clean_all_models
    echo "→ Wave 1a done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / prepare / steering-extract"
    echo " Wave 1b — Qwen3.5-27B (GPU 1)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=1 $RUN --figure 5 --stage prepare \
        --group steering-extract \
        --include-model Qwen/Qwen3.5-27B
    clean_all_models
    echo "→ Wave 1b done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / prepare / steering-extract"
    echo " Wave 2 — Llama-70B (GPU 0+1)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0,1 $RUN --figure 5 --stage prepare \
        --group steering-extract \
        --include-model meta-llama/Meta-Llama-3-70B-Instruct
    clean_all_models
    echo "→ Wave 2 done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / prepare / steering-compute (CPU)"
    echo "══════════════════════════════════════════════════════════"
    $RUN --figure 5 --stage prepare --group steering-compute
    echo "→ Steering vectors done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / run — wave 1 (Qwen3-32B + Qwen3.5-27B, parallel)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 $RUN --figure 5 --stage run \
        --include-model Qwen/Qwen3-32B &
    local PID_RUN_GPU0=$!
    CUDA_VISIBLE_DEVICES=1 $RUN --figure 5 --stage run \
        --include-model Qwen/Qwen3.5-27B &
    local PID_RUN_GPU1=$!
    wait $PID_RUN_GPU0 $PID_RUN_GPU1
    clean_all_models
    echo "→ Wave 1 done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 / run — wave 2 (Llama-70B, GPU 0+1)"
    echo "══════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0,1 $RUN --figure 5 --stage run \
        --include-model meta-llama/Meta-Llama-3-70B-Instruct
    clean_all_models
    echo "→ Wave 2 done"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 5 (steering) complete."
    echo "══════════════════════════════════════════════════════════"
}
