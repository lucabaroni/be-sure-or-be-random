#!/usr/bin/env python3
"""Camera-ready Fig. 10 (Appendix E): figures/camera_ready/cross_instruction_steering.pdf

Style: the submitted figure (2026-04-17) and the July regeneration
(figures/regenerated_neutral_20260721/paper_named/cross_instruction_steering.pdf, 2026-07-21)
were both produced by
  figures/scripts/generate_supplementary_figure5_fixed_layer_distributions_cross_instruction_commit_2x.py
which calls build_figure() of
  figures/scripts/generate_supplementary_figure5_low_cosine_layer_distributions.py
(unchanged between commit 236b4f7, 2026-04-17, and today). This script calls that same
build_figure with the same arguments; only two things are wrapped:

* Validity rule: the P(option A) histograms (valid_instruction_probs) did not apply the
  P(A)+P(B) > 0.9 rule (the entropy KDEs and Delta H already did, via
  clean_probability_records). It is applied here to all panels. At the designated layers
  there are no invalid records in any of the runs used (checked below), so this changes
  nothing visible.
* Data provenance: every loaded result path is printed and asserted to be a post-fix run
  (results/neutral/, timestamp >= 2026-07-20).

Data: vector "Be sure!"/"Be random!" (code: certainty - uncertainty), test pair
"Commit to one option" (certainty) / "Keep both options equally likely" (uncertainty),
strengths +-2 from results/neutral/steering_all_layers/<model>/vector_Be_sure_vs_Be_random_test_
Commit_to_one_option_vs_Keep_both_options_equally_likely_all_layers_strength*/ at layers
56 (Qwen3.5-27B), 56 (Qwen3-32B), 64 (Llama-3-70B). "no steering" = the zero-shot run of the
same pair (results/neutral/zero_shot_local), exactly as in the submitted figure (no strength-0
steering run existed then either).
Sign: SteeringInjector adds +s*v under the certainty instruction and -s*v under the uncertainty
instruction with v = certainty - uncertainty, i.e. +s*("Be random!" - "Be sure!") under
"Keep both options equally likely": code strength = paper strength, labels unchanged.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/final/cross_instruction_steering.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # figures/scripts
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402

import generate_supplementary_figure5_low_cosine_layer_distributions as base  # noqa: E402
from common import is_valid_probability_record, normalized_option_a_probability  # noqa: E402
from _qc_appendix_de import check_text_overlap  # noqa: E402

assert os.environ.get("AB_PAIRS_VARIANT") == "neutral", "run with AB_PAIRS_VARIANT=neutral"

REPO = HERE.parents[3]
OUT_DIR = REPO / "figures" / "camera_ready"
MIN_DATE = "20260720"
MODEL_NAMES = [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3-32B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]
TEST_PAIR = ("Commit to one option", "Keep both options equally likely")
SUBTITLE = (
    'Prompt instructions: "Keep both options equally likely" vs "Commit to one option"  —  '
    'Steering vector: "Be random!" vs "Be sure!"'
)


def _checked(path: Path | None, what: str) -> Path | None:
    if path is not None:
        stamp = path.parent.name.rsplit("_", 2)[-2]
        assert "/results/neutral/" in str(path) and stamp >= MIN_DATE, f"pre-fix result: {path}"
        print(f"  {what} <- {path.relative_to(REPO)}")
    return path


_orig_latest_result_path = base.latest_result_path
_orig_latest_zero_shot = base.latest_zero_shot_local_result


def latest_result_path_checked(kind, **kwargs):
    # build_figure loads +-1 as well (manifest strengths) but only draws -2, no steering, +2
    return _checked(_orig_latest_result_path(kind, **kwargs), f"{kwargs['model_name']} strength {kwargs.get('steering_strength'):+.0f}")


def latest_zero_shot_checked(model_name, certainty_instruction, uncertainty_instruction):
    return _checked(_orig_latest_zero_shot(model_name, certainty_instruction, uncertainty_instruction), f"{model_name} no steering (zero-shot)")


def valid_instruction_probs(records: list[dict], instruction: str) -> list[float]:
    """As base.valid_instruction_probs, plus the uniform validity rule P(A)+P(B) > 0.9."""
    return [
        normalized_option_a_probability(record)
        for record in records
        if record.get("instruction") == instruction and is_valid_probability_record(record)
    ]


base.latest_result_path = latest_result_path_checked
base.latest_zero_shot_local_result = latest_zero_shot_checked
base.valid_instruction_probs = valid_instruction_probs


def build_figure() -> plt.Figure:
    return base.build_figure(
        max_strength=2.0,
        model_names=MODEL_NAMES,
        test_pair_override=TEST_PAIR,
        instruction_in_labels=False,
        subtitle=SUBTITLE,
    )


def main() -> None:
    fig = build_figure()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = check_text_overlap(fig)
    fig.savefig(OUT_DIR / "cross_instruction_steering.pdf")
    fig.savefig(OUT_DIR / "cross_instruction_steering.png", dpi=150)
    twin_dir = os.environ.get("QC_TYPE3_TWIN_DIR")
    if twin_dir:  # Type 3 copy (as submitted) for a like-for-like pdftotext -bbox comparison only
        with plt.rc_context({"pdf.fonttype": 3}):
            fig.savefig(Path(twin_dir) / "cross_instruction_steering.pdf")
    plt.close(fig)
    print(OUT_DIR / "cross_instruction_steering.pdf")
    print(OUT_DIR / "cross_instruction_steering.png")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
