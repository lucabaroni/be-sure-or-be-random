#!/usr/bin/env bash
# Run figures 2, 3, 4 with neutral AB pairs using both GPUs.
# Same schedule as run_neutral_all.sh (figs 2–4); does NOT wipe the HF cache.
#
# Usage:
#   bash run_neutral_local.sh          # run everything
#   bash run_neutral_local.sh --dry-run  # preview without running

set -euo pipefail

CLEAN_CACHE=0
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/neutral_common.sh" "${1:-}"

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
run_figures_2_3_4
