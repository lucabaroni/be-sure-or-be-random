#!/usr/bin/env bash
# Regenerate the paper figures and the numbers quoted in the text from the results bundle.
# Run from anywhere; needs the results bundle (bash scripts/download_results.sh). CPU only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python}"
export AB_PAIRS_VARIANT=neutral
export PYTHONPATH="$PWD:$PWD/figures/scripts:$PWD/figures/scripts/camera_ready/final${PYTHONPATH:+:$PYTHONPATH}"

# Section 4.1 and Appendix "Results across models"
$PY figures/scripts/camera_ready/final/fig1_schematic_data.py          # Fig. 1 data (HTML); render with fig1_render_schematic_pdf.py
$PY figures/scripts/camera_ready/final/fig2_grid1x6.py
$PY figures/scripts/camera_ready/final/fig3_multimodel_distributions.py --variant mixed
$PY figures/scripts/camera_ready/final/sweep_kde_grid.py --version submitted
$PY figures/scripts/camera_ready/prelim_sweep_bootstrap.py
$PY scripts/analysis/check_exclusion_rule.py

# Appendix "System-prompt variations and controls"
$PY scripts/analysis/min_vs_std2.py
$PY scripts/analysis/fig_min_vs_std_final.py
$PY figures/scripts/bootstrap_meaningful_vs_control_floor.py --n-boot 10000
$PY figures/scripts/bootstrap_standard_vs_barebone_controls.py --n-boot 10000
$PY figures/scripts/camera_ready/prelim_bootstrap.py
$PY scripts/analysis/control_pairs.py
$PY figures/scripts/camera_ready/prelim_prompt_structure_fill.py

# Section 4.2 and steering appendices (Llama-3-70B at layer 44, Qwen models at layer 56)
$PY figures/scripts/camera_ready/layer44_previews.py                  # dose response, cross-instruction, similarity
$PY scripts/camera_ready_llama_layer44_numbers.py
$PY figures/scripts/camera_ready/layer44_layer_choice.py
$PY figures/scripts/camera_ready/layer44_prob_mass.py

# Sections 4.3 and 4.4, MMLU appendices
$PY figures/scripts/camera_ready/final/fig6_softacc_PREVIEW.py
$PY figures/scripts/camera_ready/fig_sandbagging_bars.py
$PY figures/scripts/camera_ready/prelim_mmlu_prompt_conditions.py
$PY figures/scripts/camera_ready/prelim_probe_random_directions.py

# Reasoning and calibration fine-tuning appendices
$PY figures/scripts/camera_ready/prelim_cot_thinking.py
$PY figures/scripts/camera_ready/prelim_calibration_ft.py
