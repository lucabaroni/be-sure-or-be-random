"""Shared helpers for the camera-ready preliminary appendix figures (R1, R2, R4).

Reuses ``apply_paper_style()`` and the palette of ``figures/scripts/common.py`` and the
condition colours of the existing steering / MMLU paper figures. Figures are drawn at
full paper width (13.5 in, like ``mmlu_entropy_correctness_calibration_with_random_mean.pdf``)
and scaled to ``\\textwidth`` by LaTeX.

Font sizes: the paper PDFs drawn at about 13.5 in use 9 to 14 pt text (measured from the
PDF content streams), so the defaults of ``apply_paper_style()`` (8 to 10 pt) are bumped
to the values used by ``generate_figure7_entropy_correctness_calibration.py``. Set
``PRELIM_FONT=default`` to keep the plain ``apply_paper_style()`` sizes instead.

Sign convention: steering vectors in code are ``mean(Be sure!) - mean(Be random!)``
(certainty minus uncertainty). The paper uses the opposite sign
(``Be random! - Be sure!``), so code "positive" steering is the paper's entropy-decreasing
(negative) steering and vice versa. All figures here are labelled in paper terms.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt  # noqa: E402

from common import GRAY, SLATE, apply_paper_style, display_model_name  # noqa: E402,F401
from be_sure_or_random.utils import get_results_dir  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parents[1]
OUT_DIR = REPO_ROOT / "figures" / "camera_ready_prelim"
TABLE_DIR = OUT_DIR / "tables"
FULL_WIDTH_IN = 13.5

# Same hex values as the existing paper figures (figure 7 MMLU, cross-instruction steering).
NONE_COLOR = "#64748B"      # no steering / baseline prompt
H_UP_COLOR = "#DC2626"      # entropy-increasing ("positive steering" in the paper)
H_DOWN_COLOR = "#2563EB"    # entropy-decreasing ("negative steering" in the paper)
SANDBAG_COLOR = "#CA8A04"   # second entropy-raising prompt; checked for CVD separation vs red/blue
REF_LINE_COLOR = "#9CA3AF"  # reference lines (zero, chance), as in figure 7
INK = "#111827"
MUTED_INK = "#4B5563"

FONT_OVERRIDES = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
}


def results_dir() -> Path:
    path = get_results_dir()
    if path.name != "neutral":
        raise RuntimeError("Run with AB_PAIRS_VARIANT=neutral (expected results/neutral).")
    return path


def apply_prelim_style() -> None:
    apply_paper_style()
    if os.environ.get("PRELIM_FONT", "paper") != "default":
        plt.rcParams.update(FONT_OVERRIDES)
    # TrueType embedding avoids Type 3 font warnings in the camera-ready check.
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def unique_path(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {parent}/{pattern}, found {len(matches)}: {matches}")
    return matches[0]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def save(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    return [pdf_path, png_path]


def write_json(payload: dict[str, Any], stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_table(text: str, stem: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{stem}.tex"
    path.write_text(text)
    return path
