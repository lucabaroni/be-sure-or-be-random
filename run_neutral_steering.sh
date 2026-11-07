#!/usr/bin/env bash
# Run figure 5 (steering) with neutral AB pairs using both GPUs.
# Same schedule as run_neutral_all.sh (fig 5); does NOT wipe the HF cache.
#
# Stages: extract → compute → run (all_layers / cross / no_instruction).
#
# Usage:
#   bash run_neutral_steering.sh           # run everything
#   bash run_neutral_steering.sh --dry-run   # preview without running

set -euo pipefail

CLEAN_CACHE=0
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/neutral_common.sh" "${1:-}"

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
run_figure_5
