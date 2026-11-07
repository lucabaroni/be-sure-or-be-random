"""Preliminary appendix figure (rebuttal R7): calibration fine-tuning (Baldelli et al., 2026).

Base Qwen3-14B versus its soft-target calibration fine-tune (models/Qwen3-14B-calib-soft),
standard system prompt, neutral option pairs, 7 meaningful instruction pairs.

Left block: a two-row version of the paper's multimodel figure for the representative
pair "Be sure!" / "Be random!": P(option A) histogram, entropy under each instruction,
and the paired DeltaH = H("Be random!") - H("Be sure!") with its median (dashed).
Right panel: all 7 pairs, mean entropy under the certainty instruction (x) versus under
the uncertainty instruction (y), base -> fine-tuned. The diagonal is "no control"; the
vertical distance above it equals the mean DeltaH.

Entropy is binary entropy in bits after renormalising over the two option tokens;
records with < 0.9 mass on the two options are dropped (none are, see the audit).

  cd ~/Luca/be-sure-or-random-cr
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/prelim_calibration_ft.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from scipy.stats import gaussian_kde, wilcoxon  # noqa: E402

from common import (  # noqa: E402
    REPO_ROOT,
    SLATE,
    clean_probability_records,
    entropy_values_for_instruction,
    load_zero_shot_result,
    normalized_option_a_probability,
    paired_entropy_differences,
)
# Shared font scale / output helpers for all camera-ready preliminary figures.
from _cr_prelim_style import FULL_WIDTH_IN, OUT_DIR, apply_prelim_style, save  # noqa: E402

STEM = "prelim_calibration_ft"
RESULTS = REPO_ROOT / "results" / "neutral" / "zero_shot_local"
MIN_RUN_DATE = "20260720"

ARMS = [
    ("qwen_Qwen3-14B", "Qwen3-14B"),
    ("models_Qwen3-14B-calib-soft", "Qwen3-14B\n+ calib. FT"),
]
REP_PAIR = ("Be sure!", "Be random!")
PAIRS = [
    ("Be sure!", "Be random!"),
    ("Maximize certainty", "Minimize certainty"),
    ("Maximize certainty", "Maximize uncertainty"),
    ("Commit to one option", "Keep both options equally likely"),
    ("Reply with confidence", "Reply without confidence"),
    ("Aim for low entropy", "Aim for high entropy"),
    ("Be deterministic", "Assign equal probability to both options"),
]

CERT_BLUE = "#2563EB"
UNC_RED = "#DC2626"
DELTA_PURPLE = "#9333EA"
ZERO_GRAY = "#9CA3AF"
TEXT_GRAY = "#374151"
BASE_COLOR = SLATE        # "#64748B"
FT_COLOR = "#D85A30"      # common.ORANGE

FIG_WIDTH = FULL_WIDTH_IN
MIN_BW = 0.02


def slug(text: str) -> str:
    return text.replace(" ", "_").replace("!", "").replace("?", "").replace(",", "").replace(".", "")


def latest_run(model_slug: str, cert: str, unc: str) -> Path:
    parent = RESULTS / model_slug
    prefix = f"{slug(cert)}_{slug(unc)}_"
    runs = []
    for d in parent.iterdir():
        stamp = d.name[len(prefix):] if d.name.startswith(prefix) else ""
        if len(stamp) == 15 and stamp[:8].isdigit() and stamp[:8] >= MIN_RUN_DATE and (d / "results.json").exists():
            runs.append((stamp, d / "results.json"))
    if not runs:
        raise FileNotFoundError(f"No run >= {MIN_RUN_DATE} for {model_slug} {cert} / {unc}")
    return max(runs)[1]


def kde(values, x_min: float, x_max: float, points: int = 1001):
    data = np.asarray(values, dtype=float)
    grid = np.linspace(x_min, x_max, points)
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    if data.size < 2 or std < 1e-9:
        centre = float(data[0]) if data.size else 0.0
        return grid, np.exp(-0.5 * ((grid - centre) / MIN_BW) ** 2) / (MIN_BW * math.sqrt(2 * math.pi))
    est = gaussian_kde(data, bw_method="scott")
    if est.factor * std < MIN_BW:
        est = gaussian_kde(data, bw_method=MIN_BW / std)
    return grid, est(grid)


def draw_kde(ax, values, color, x_min, x_max, *, median=False) -> float:
    grid, dens = kde(values, x_min, x_max)
    ax.fill_between(grid, 0, dens, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(grid, dens, color=color, lw=1.8, zorder=3)
    if median:
        ax.axvline(float(np.median(values)), color=color, ls="--", lw=1.4, zorder=4)
    return float(dens.max())


def main() -> None:
    apply_prelim_style()
    plt.rcParams.update({"axes.grid": False})  # density panels: no grid, as in the paper KDE figures
    fs_text = plt.rcParams["font.size"]
    fs_label = plt.rcParams["axes.labelsize"]
    fs_title = plt.rcParams["axes.titlesize"]
    fs_small = plt.rcParams["legend.fontsize"]

    # ---------- data ----------
    rep = {}
    summary = {"representative_pair": list(REP_PAIR), "arms": {}}
    for model_slug, _ in ARMS:
        cert, unc = REP_PAIR
        records = load_zero_shot_result(latest_run(model_slug, cert, unc))
        clean = clean_probability_records(records)
        rep[model_slug] = {
            "p_cert": [normalized_option_a_probability(r) for r in clean if r["instruction"] == cert],
            "p_unc": [normalized_option_a_probability(r) for r in clean if r["instruction"] == unc],
            "h_cert": entropy_values_for_instruction(records, cert),
            "h_unc": entropy_values_for_instruction(records, unc),
            "d": paired_entropy_differences(records, cert, unc),
        }
        per_pair = []
        pooled = []
        for c, u in PAIRS:
            path = latest_run(model_slug, c, u)
            recs = load_zero_shot_result(path)
            hc = entropy_values_for_instruction(recs, c)
            hu = entropy_values_for_instruction(recs, u)
            d = paired_entropy_differences(recs, c, u)
            pooled.extend(d)
            mass = [float(r["option_a_prob"]) + float(r["option_b_prob"]) for r in recs]
            per_pair.append(
                {
                    "certainty": c,
                    "uncertainty": u,
                    "run": str(path.parent.relative_to(REPO_ROOT)),
                    "n_records": len(recs),
                    "n_paired": len(d),
                    "mean_H_cert": float(np.mean(hc)),
                    "mean_H_unc": float(np.mean(hu)),
                    "median_H_cert": float(np.median(hc)),
                    "median_H_unc": float(np.median(hu)),
                    "median_dH": float(np.median(d)),
                    "mean_dH": float(np.mean(d)),
                    "frac_dH_positive": float(np.mean(np.asarray(d) > 0)),
                    "min_option_mass": float(np.min(mass)),
                }
            )
        summary["arms"][model_slug] = {
            "per_pair": per_pair,
            "pooled_median_dH": float(np.median(pooled)),
            "pooled_mean_dH": float(np.mean(pooled)),
            "pooled_n": len(pooled),
        }

    # Paired base-vs-FT test on pooled items (same items in both arms).
    base_pool, ft_pool = [], []
    for c, u in PAIRS:
        items = {}
        for model_slug, _ in ARMS:
            g = {}
            for r in clean_probability_records(load_zero_shot_result(latest_run(model_slug, c, u))):
                key = (r["option_a"], r["option_b"], bool(r.get("flip_ab")))
                g.setdefault(key, {})[r["instruction"]] = r
            items[model_slug] = g
        from common import record_entropy

        for key, recs in items[ARMS[0][0]].items():
            other = items[ARMS[1][0]].get(key)
            if other and c in recs and u in recs and c in other and u in other:
                base_pool.append(record_entropy(recs[u]) - record_entropy(recs[c]))
                ft_pool.append(record_entropy(other[u]) - record_entropy(other[c]))
    w = wilcoxon(np.asarray(base_pool), np.asarray(ft_pool))
    summary["paired_wilcoxon_base_vs_ft"] = {"n": len(base_pool), "statistic": float(w.statistic), "pvalue": float(w.pvalue)}

    # ---------- figure ----------
    fig = plt.figure(figsize=(FIG_WIDTH, 5.3))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.22, 1.45],
        hspace=0.38,
        wspace=0.22,
        left=0.085,
        right=0.99,
        top=0.86,
        bottom=0.12,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
    ax_sum = fig.add_subplot(gs[:, 5])

    bins = np.linspace(0.0, 1.0, 16)
    ymax = [0.0] * 4
    for r, (model_slug, label) in enumerate(ARMS):
        d = rep[model_slug]
        ax0, ax1, ax2, ax3 = axes[r]
        c0, _, _ = ax0.hist(d["p_cert"], bins=bins, color=CERT_BLUE, alpha=0.45, label='"Be sure!"', zorder=2)
        c1, _, _ = ax0.hist(d["p_unc"], bins=bins, color=UNC_RED, alpha=0.45, label='"Be random!"', zorder=2)
        ymax[0] = max(ymax[0], float(np.max(c0)), float(np.max(c1)))
        ax0.set_xlim(0.0, 1.0)
        ax0.set_ylabel("Count")
        ax0.text(-0.50, 0.5, label, transform=ax0.transAxes, rotation=90, ha="center", va="center", fontweight="bold", fontsize=fs_label)
        ax0.grid(True, axis="y")

        ymax[1] = max(ymax[1], draw_kde(ax1, d["h_unc"], UNC_RED, 0.0, 1.0))
        ymax[2] = max(ymax[2], draw_kde(ax2, d["h_cert"], CERT_BLUE, 0.0, 1.0))
        ax3.axvline(0.0, color=ZERO_GRAY, ls="--", lw=1.0, zorder=1)
        ymax[3] = max(ymax[3], draw_kde(ax3, d["d"], DELTA_PURPLE, -1.0, 1.0, median=True))
        med = float(np.median(d["d"]))
        ax3.text(0.03, 0.93, f"median\n{med:+.3f}", transform=ax3.transAxes, ha="left", va="top", fontsize=fs_text, color=DELTA_PURPLE)

        for ax in (ax1, ax2, ax3):
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["0", "0.5", "1"])
        ax3.set_xlim(-1.0, 1.0)
        ax3.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax3.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
        for ax in (ax1, ax2):
            ax.set_xlim(0.0, 1.0)
        if r == 0:
            ax0.legend(loc="upper center", frameon=True, handlelength=1.2)
            for ax in (ax0, ax1, ax2, ax3):
                ax.set_xticklabels([])
        else:
            ax0.set_xlabel("P(option A)")
            ax1.set_xlabel("Entropy")
            ax2.set_xlabel("Entropy")
            ax3.set_xlabel(r"$\Delta$Entropy")

    for j in range(4):
        for r in range(2):
            axes[r][j].set_ylim(0.0, ymax[j] * 1.08)

    # ---------- summary panel: all 7 pairs ----------
    ax = ax_sum
    ax.plot([0, 1], [0, 1], color=ZERO_GRAY, ls="--", lw=1.0, zorder=1)
    base_pts = [(p["median_H_cert"], p["median_H_unc"]) for p in summary["arms"][ARMS[0][0]]["per_pair"]]
    ft_pts = [(p["median_H_cert"], p["median_H_unc"]) for p in summary["arms"][ARMS[1][0]]["per_pair"]]
    for (x0, y0), (x1, y1) in zip(base_pts, ft_pts):
        ax.add_patch(
            FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color="#9CA3AF", lw=0.9,
                            shrinkA=5, shrinkB=6, alpha=0.8, zorder=2)
        )
    ax.scatter(*zip(*base_pts), s=55, color=BASE_COLOR, edgecolor="white", linewidth=0.8, zorder=3, label="Qwen3-14B")
    ax.scatter(*zip(*ft_pts), s=55, marker="s", color=FT_COLOR, edgecolor="white", linewidth=0.8, zorder=3,
               label="+ calibration FT")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.grid(True)
    ax.set_xlabel("Median entropy, certainty instr.")
    ax.set_ylabel("Median entropy, uncertainty instr.")
    ax.set_title("All 7 instruction pairs", color=TEXT_GRAY, pad=8)
    ax.text(0.79, 0.65, r"no control ($\Delta$H = 0)", rotation=45, ha="center", va="center", fontsize=fs_small,
            color="#6B7280")
    ax.legend(loc="lower right", frameon=True, handletextpad=0.4, borderaxespad=0.4)

    titles = ["P(option A)", '"Be random!"', '"Be sure!"', r"$\Delta$Entropy"]
    for j, t in enumerate(titles):
        axes[0][j].set_title(t, color=TEXT_GRAY, pad=6)
    b0 = axes[0][0].get_position()
    b1 = axes[0][3].get_position()
    y = b0.y1 + 0.075
    fig.text(0.5 * (b0.x0 + b1.x1), y, 'Representative pair: "Be random!" vs "Be sure!"', ha="center", va="bottom",
             fontsize=fs_title, fontweight="bold", color=TEXT_GRAY)
    fig.add_artist(Line2D([b0.x0, b1.x1], [y - 0.008, y - 0.008], transform=fig.transFigure, color="#D1D5DB", lw=1.0))

    pdf, png = save(fig, STEM)
    plt.close(fig)
    (OUT_DIR / f"{STEM}_summary.json").write_text(json.dumps(summary, indent=2))

    for model_slug, _ in ARMS:
        s = summary["arms"][model_slug]
        print(f"== {model_slug}: pooled median dH {s['pooled_median_dH']:+.4f} mean {s['pooled_mean_dH']:+.4f} n={s['pooled_n']}")
        for p in s["per_pair"]:
            print(f"   {p['certainty'][:22]:22s} H_c {p['mean_H_cert']:.3f}/{p['median_H_cert']:.3f}"
                  f"  H_u {p['mean_H_unc']:.3f}/{p['median_H_unc']:.3f}  dH med {p['median_dH']:+.3f} mean {p['mean_dH']:+.3f}"
                  f"  n={p['n_paired']} minmass={p['min_option_mass']:.3f}")
    print("paired Wilcoxon base vs FT:", summary["paired_wilcoxon_base_vs_ft"])
    for p in (pdf, png, OUT_DIR / f"{STEM}_summary.json"):
        print(p)


if __name__ == "__main__":
    main()
