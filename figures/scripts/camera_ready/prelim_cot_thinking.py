"""Preliminary appendix figure (rebuttal R3): entropy control with reasoning enabled.

Qwen3.5-27B, thinking enabled, 30 sampled CoT trajectories per (option pair,
ordering, instruction), instruction pair "Be sure!" / "Be random!", three system
prompts (standard = within-trajectory / logit control, across-trajectory control,
and within- plus across-trajectory control).

Two notions of entropy per (option pair, ordering) item:
  within-trajectory  mean over the 30 trajectories of the binary entropy of the
                     answer-token probabilities after the CoT (renormalised over the
                     two option tokens);
  across-trajectory  binary entropy of the empirical frequency of the sampled final
                     answers over the 30 trajectories.
DeltaH = H("Be random!") - H("Be sure!") per item (n = 200 items per prompt).

Layout: rows = system prompts; columns = within-trajectory H (both instructions),
within-trajectory DeltaH, across-trajectory H (both instructions), across-trajectory
DeltaH. Colours follow the paper's multimodel figure (blue = certainty instruction,
red = uncertainty instruction, purple = DeltaH, dashed lines = medians).

  cd ~/Luca/be-sure-or-random-cr
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/prelim_cot_thinking.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from common import REPO_ROOT  # noqa: E402
# Shared font scale / output helpers for all camera-ready preliminary figures.
from _cr_prelim_style import FULL_WIDTH_IN, OUT_DIR, apply_prelim_style, save  # noqa: E402

STEM = "prelim_cot_thinking"
RUN_ROOT = REPO_ROOT / "results" / "neutral" / "thinking_sampling" / "Qwen_Qwen3.5-27B"
MIN_RUN_DATE = "20260720"

CERTAINTY = "Be sure!"
UNCERTAINTY = "Be random!"

# (system prompt slug, row label)
PROMPTS = [
    ("standard_system", "Standard prompt\n(within-trajectory)"),
    ("cot_across_trajectory_system", "Across-trajectory\nprompt"),
    ("cot_within_and_across_system", "Within- and across-\ntrajectory prompt"),
]

# Paper palette (colours read from multimodel_distributions.pdf / qwen35_27b_grid1x6.pdf).
CERT_BLUE = "#2563EB"
UNC_RED = "#DC2626"
DELTA_PURPLE = "#9333EA"
ZERO_GRAY = "#9CA3AF"
TEXT_GRAY = "#374151"

FIG_WIDTH = FULL_WIDTH_IN  # drawn large and scaled to \textwidth by LaTeX, like the paper figures
MIN_BW = 0.02  # minimum absolute KDE bandwidth so near-degenerate spikes stay resolvable


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def kde(values, x_min: float, x_max: float, points: int = 1001):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    grid = np.linspace(x_min, x_max, points)
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    if data.size < 2 or std < 1e-9:
        centre = float(data[0]) if data.size else 0.0
        dens = np.exp(-0.5 * ((grid - centre) / MIN_BW) ** 2) / (MIN_BW * math.sqrt(2 * math.pi))
        return grid, dens
    est = gaussian_kde(data, bw_method="scott")
    if est.factor * std < MIN_BW:
        est = gaussian_kde(data, bw_method=MIN_BW / std)
    return grid, est(grid)


def draw_kde(ax, values, color, x_min, x_max, *, median=True, zorder=2):
    grid, dens = kde(values, x_min, x_max)
    ax.fill_between(grid, 0, dens, color=color, alpha=0.18, lw=0, zorder=zorder)
    ax.plot(grid, dens, color=color, lw=1.8, zorder=zorder + 1)
    if median:
        ax.axvline(float(np.median(values)), color=color, ls="--", lw=1.4, zorder=zorder + 2)
    return float(dens.max())


def latest_run(slug: str) -> Path:
    runs = sorted(
        d for d in (RUN_ROOT / slug).iterdir() if d.is_dir() and d.name[:8] >= MIN_RUN_DATE
    )
    if not runs:
        raise FileNotFoundError(f"No run >= {MIN_RUN_DATE} under {RUN_ROOT / slug}")
    return runs[-1]


def analyse(slug: str) -> dict:
    run = latest_run(slug)
    config = json.loads((run / "config.json").read_text())
    records = json.loads((run / "results.json").read_text())
    rejected = json.loads((run / "rejected_results.json").read_text()) if (run / "rejected_results.json").exists() else []
    assert config["instruction_pairs"] == [[CERTAINTY, UNCERTAINTY]], config["instruction_pairs"]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["pair_idx"], bool(r["flip_ab"]), r["instruction"])].append(r)

    items = sorted({(k[0], k[1]) for k in groups})
    out = {k: [] for k in ("within_c", "within_u", "across_c", "across_u", "within_d", "across_d")}
    n_traj = []
    for pair_idx, flip in items:
        per = {}
        for instr, tag in ((CERTAINTY, "c"), (UNCERTAINTY, "u")):
            recs = groups[(pair_idx, flip, instr)]
            n_traj.append(len(recs))
            within = [binary_entropy(float(r["answer_token_a_normalized_prob"])) for r in recs]
            freq_a = sum(r["chosen"] == r["token_a"] for r in recs) / len(recs)
            per[f"within_{tag}"] = float(np.mean(within))
            per[f"across_{tag}"] = binary_entropy(freq_a)
        for key, val in per.items():
            out[key].append(val)
        out["within_d"].append(per["within_u"] - per["within_c"])
        out["across_d"].append(per["across_u"] - per["across_c"])

    mass = np.array([r["answer_token_a_prob"] + r["answer_token_b_prob"] for r in records])
    summary = {
        "run": str(run.relative_to(REPO_ROOT)),
        "system_prompt_file": config.get("system_prompt_file"),
        "n_items": len(items),
        "n_trajectories_per_item_instruction": sorted(set(n_traj)),
        "n_records": len(records),
        "n_rejected_resampled": len(rejected),
        "option_mass_min": float(mass.min()),
        "option_mass_mean": float(mass.mean()),
    }
    for key in out:
        vals = np.asarray(out[key])
        summary[key] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "frac_positive": float(np.mean(vals > 0)) if key.endswith("_d") else None,
        }
    return {"values": out, "summary": summary}


def main() -> None:
    apply_prelim_style()
    plt.rcParams.update({"axes.grid": False})  # density panels: no grid, as in the paper KDE figures
    fs_text = plt.rcParams["font.size"]
    fs_label = plt.rcParams["axes.labelsize"]
    fs_title = plt.rcParams["axes.titlesize"]
    analyses = [analyse(slug) for slug, _ in PROMPTS]

    fig = plt.figure(figsize=(FIG_WIDTH, 5.6))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        width_ratios=[1.0, 1.0, 0.16, 1.0, 1.0],
        hspace=0.34,
        wspace=0.22,
        left=0.165,
        right=0.99,
        top=0.78,
        bottom=0.11,
    )
    col_spec = [
        (0, "within", "H"),
        (1, "within", "D"),
        (3, "across", "H"),
        (4, "across", "D"),
    ]
    axes = [[fig.add_subplot(gs[r, c]) for c, _, _ in col_spec] for r in range(len(PROMPTS))]

    ymax = defaultdict(float)
    for r, analysis in enumerate(analyses):
        vals = analysis["values"]
        for j, (_, notion, kind) in enumerate(col_spec):
            ax = axes[r][j]
            if kind == "H":
                m1 = draw_kde(ax, vals[f"{notion}_c"], CERT_BLUE, 0.0, 1.0, median=False)
                m2 = draw_kde(ax, vals[f"{notion}_u"], UNC_RED, 0.0, 1.0, median=False, zorder=4)
                ymax[j] = max(ymax[j], m1, m2)
                ax.set_xlim(0.0, 1.0)
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0", "0.5", "1"])
            else:
                d = vals[f"{notion}_d"]
                ax.axvline(0.0, color=ZERO_GRAY, ls="--", lw=1.0, zorder=1)
                ymax[j] = max(ymax[j], draw_kde(ax, d, DELTA_PURPLE, -1.0, 1.0))
                med = float(np.median(d))
                # Put the label on the side away from the bulk of the density.
                right = abs(med) < 0.2
                ax.text(
                    0.97 if right else 0.03,
                    0.93,
                    f"median\n{med:+.3f}",
                    transform=ax.transAxes,
                    ha="right" if right else "left",
                    va="top",
                    fontsize=fs_text,
                    color=DELTA_PURPLE,
                )
                ax.set_xlim(-1.0, 1.0)
                ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                ax.set_xticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            if r < len(PROMPTS) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Entropy" if kind == "H" else r"$\Delta$Entropy")
        axes[r][0].set_ylabel(PROMPTS[r][1], fontweight="bold", color="black", labelpad=12, fontsize=fs_label, rotation=0, ha="right", va="center")

    # Shared density scale within each column (as in the paper's multimodel figure).
    for j in range(len(col_spec)):
        for r in range(len(PROMPTS)):
            axes[r][j].set_ylim(0.0, ymax[j] * 1.08)

    titles = ["Entropy per instruction", r"$\Delta$Entropy", "Entropy per instruction", r"$\Delta$Entropy"]
    for j, title in enumerate(titles):
        axes[0][j].set_title(title, color=TEXT_GRAY, pad=6)

    # Group headers above the two column pairs.
    for (c0, c1), text in (
        ((0, 1), "Within-trajectory (answer-token probabilities after the CoT)"),
        ((2, 3), "Across-trajectory (answer frequencies over 30 CoTs)"),
    ):
        b0 = axes[0][c0].get_position()
        b1 = axes[0][c1].get_position()
        x_mid = 0.5 * (b0.x0 + b1.x1)
        y = b0.y1 + 0.065
        fig.text(x_mid, y, text, ha="center", va="bottom", fontsize=fs_title, fontweight="bold", color=TEXT_GRAY)
        fig.add_artist(
            Line2D([b0.x0, b1.x1], [y - 0.006, y - 0.006], transform=fig.transFigure, color="#D1D5DB", lw=1.0)
        )

    handles = [
        Line2D([0], [0], color=CERT_BLUE, lw=2.2, label='"Be sure!"'),
        Line2D([0], [0], color=UNC_RED, lw=2.2, label='"Be random!"'),
        Line2D([0], [0], color=DELTA_PURPLE, lw=2.2, label=r'$\Delta$Entropy = H("Be random!") $-$ H("Be sure!")'),
        Line2D([0], [0], color=TEXT_GRAY, lw=1.4, ls="--", label="Median"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=4, frameon=True)

    pdf, png = save(fig, STEM)
    plt.close(fig)

    summary = {slug: a["summary"] for (slug, _), a in zip(PROMPTS, analyses)}
    (OUT_DIR / f"{STEM}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'prompt':32s} {'within dH med':>14s} {'mean':>8s} {'across dH med':>14s} {'mean':>8s}"
          f" {'H_c within':>11s} {'H_u within':>11s} {'H_c across':>11s} {'H_u across':>11s}")
    for slug, s in summary.items():
        print(
            f"{slug:32s} {s['within_d']['median']:>+14.4f} {s['within_d']['mean']:>+8.4f}"
            f" {s['across_d']['median']:>+14.4f} {s['across_d']['mean']:>+8.4f}"
            f" {s['within_c']['mean']:>11.4f} {s['within_u']['mean']:>11.4f}"
            f" {s['across_c']['mean']:>11.4f} {s['across_u']['mean']:>11.4f}"
        )
    for p in (pdf, png, OUT_DIR / f"{STEM}_summary.json"):
        print(p)


if __name__ == "__main__":
    main()
