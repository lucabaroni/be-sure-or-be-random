#!/usr/bin/env python3
"""Fig. 2 (qwen35_27b_grid1x6.pdf): Qwen3.5-27B, standard prompt, ΔH KDE for 6 instruction pairs.

Rebuilds the SUBMITTED style (the original generator is lost; its relatives are the tab10
figure3_supplementary_local_sweep_grid2x3.py / *_layout.py scripts, whose styling it shares):
  * one tab10 colour per pair: plt.cm.tab10(np.linspace(0, 1, 6)); KDE line lw 2, fill alpha 0.1
    (common.plot_kde, Scott bandwidth, 256-point grid on [-1, 1]);
  * grey dashed zero line (#9CA3AF, lw 1); dashed median line in the pair colour (lw 1.5,
    full axes height, unclipped); shared y-limit = 1.05 x the largest KDE peak; no y-axis;
  * vertical x-grid at the ticks (#E5E7EB, lw 0.7, alpha 0.7; from common.apply_paper_style);
  * titles 12 pt '"<uncertainty>"\\nvs\\n"<certainty>"' with the submitted line breaks,
    ticks -1/0/1 at 13 pt, x-label r"$\\Delta$Entropy" at 14 pt (mathtext, as submitted);
  * page and axes geometry copied from the submitted PDF (963.36 x 225.26 pt; axes 141.81 x
    106.36 pt, pitch 158.688 pt), so the PDF is drawn at the submitted size without bbox cropping.
Only change vs submitted: data (post-fix neutral run), TrueType fonts (pdf.fonttype 42).

    cd ~/Luca/be-sure-or-random-cr
    AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
        figures/scripts/camera_ready/final/fig2_grid1x6.py
    # style validation: rebuild the submitted figure from the pre-fix runs it was made from
    ... fig2_grid1x6.py --validate-old /tmp/somewhere
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig123_common as fc  # noqa: E402  (sets AB_PAIRS_VARIANT=neutral, sys.path)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    apply_paper_style,
    kde_curve,
    load_zero_shot_result,
    paired_entropy_differences,
    plot_kde,
)

MODEL = "qwen/Qwen3.5-27B"
# (certainty, uncertainty, title) in the submitted panel order; titles exactly as submitted.
PAIRS = [
    ("Be sure!", "Be random!", '"Be random!"\nvs\n"Be sure!"'),
    ("Maximize certainty", "Minimize certainty", '"Minimize certainty"\nvs\n"Maximize certainty"'),
    ("Maximize certainty", "Maximize uncertainty", '"Maximize uncertainty"\nvs\n"Maximize certainty"'),
    ("Commit to one option", "Keep both options equally likely",
     '"Keep both options\nequally likely"\nvs\n"Commit to one option"'),
    ("Reply with confidence", "Reply without confidence",
     '"Reply without\nconfidence"\nvs\n"Reply with\nconfidence"'),
    ("Aim for low entropy", "Aim for high entropy", '"Aim for high entropy"\nvs\n"Aim for low entropy"'),
]
# Geometry of the submitted PDF (points, origin bottom-left).
PAGE_W, PAGE_H = 963.36, 225.259922
AX_X0, AX_PITCH, AX_Y0, AX_W, AX_H = 16.778125, 158.688, 44.778125, 141.8090625, 106.356796875
TITLE_PAD = 4.0
YLIM_HEADROOM = 1.05
X_LABEL = r"$\Delta$Entropy"


def load_diffs(old: bool) -> tuple[list[np.ndarray], list[str]]:
    diffs, sources = [], []
    for cert, unc, _ in PAIRS:
        path = fc.resolve_submitted_run(MODEL, cert, unc) if old else fc.resolve_run(MODEL, cert, unc)
        recs = load_zero_shot_result(path)
        d = np.asarray(paired_entropy_differences(recs, cert, unc), dtype=float)  # validity rule inside
        diffs.append(d)
        sources.append(str(path.relative_to(fc.REPO_ROOT)))
    return diffs, sources


def build_figure(diffs: list[np.ndarray]) -> plt.Figure:
    apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 12, "xtick.labelsize": 13, "axes.labelsize": 14,
                         "pdf.fonttype": 42, "savefig.bbox": None})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(PAIRS)))
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    ymax = max(float(kde_curve(d, -1.0, 1.0)[1].max()) for d in diffs) * YLIM_HEADROOM
    for i, (d, color, (_, _, title)) in enumerate(zip(diffs, colors, PAIRS)):
        ax = fig.add_axes([(AX_X0 + i * AX_PITCH) / PAGE_W, AX_Y0 / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H])
        plot_kde(ax, d, color=color, label="", x_min=-1.0, x_max=1.0, alpha=0.1)
        ax.axvline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, pad=TITLE_PAD)
        ax.axvline(float(np.median(d)), color=color, linestyle="--", linewidth=1.5, zorder=5, clip_on=False)
    return fig


def summary(diffs, sources) -> dict:
    out = {}
    for (cert, unc, _), d, src in zip(PAIRS, diffs, sources):
        out[f"{unc} vs {cert}"] = dict(source=src, n=int(d.size), median=round(float(np.median(d)), 3),
                                       mean=round(float(d.mean()), 3), frac_pos=round(float((d > 0).mean()), 3),
                                       frac_zero=round(float((d == 0).mean()), 3),
                                       frac_neg=round(float((d < 0).mean()), 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-old", metavar="OUTDIR", help="rebuild from the pre-fix runs used in the submission")
    args = ap.parse_args()
    old = args.validate_old is not None
    diffs, sources = load_diffs(old)
    fig = build_figure(diffs)
    problems = fc.text_overlap_report(fig)
    print("text overlap/clipping check:", "PASS" if not problems else "FAIL", *problems, sep="\n  ")
    out_dir = Path(args.validate_old) if old else fc.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "qwen35_27b_grid1x6"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)
    print(json.dumps(summary(diffs, sources), indent=1))
    print("wrote", f"{stem}.pdf", f"{stem}.png")


if __name__ == "__main__":
    main()
