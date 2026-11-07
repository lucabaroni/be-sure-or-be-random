"""Camera-ready Fig. 5: steering_dose_response.pdf (no-instruction dose response).

Replicates the submitted figure (paper figures/steering_dose_response.pdf)
on the post-fix neutral results. The submitted figure was produced by
figures/scripts/generate_supplementary_no_instruction_dose_response.py (be-sure-or-random @ 236b4f7)
without its suptitle; colours, line widths, alphas, z-orders, font sizes and the shared KDE ceiling
below are taken from that script and checked against the submitted PDF content stream.

Top row:    entropy KDEs at steering strength 2x (no / positive / negative steering), medians dashed.
Bottom row: median entropy vs steering strength 0x..3x for each direction (0x = no steering).

Sign convention: the stored vectors are certainty - uncertainty ("Be sure!" - "Be random!"), so the code
condition "negative" is the paper's positive steering (entropy-increasing, red) and the code condition
"positive" is the paper's negative steering (entropy-decreasing, blue).

Data: results/neutral/no_instruction_control/<model>/no_instruction_layer<L>_strength<s>_<date>/results.json
(post-fix runs dated >= 2026-07-20). Validity rule: P(A)+P(B) > 0.9.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/final/fig5_steering_dose_response.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util_fig4_5_6_11 import (  # noqa: E402
    RESULTS,
    check_text_overlap,
    compare,
    latest_post_fix,
    save,
    set_truetype,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (  # noqa: E402
    GRAY,
    apply_paper_style,
    clean_probability_records,
    kde_curve,
    plot_kde,
    record_entropy,
)

STEM = "steering_dose_response"

MODELS = [  # (result slug, display name, layer)
    ("Qwen_Qwen3.5-27B", "Qwen3.5-27B", 56),
    ("Qwen_Qwen3-32B", "Qwen3-32B", 56),
    ("meta-llama_Meta-Llama-3-70B-Instruct", "Llama-3-70B", 64),
]
STRENGTHS = [1.0, 2.0, 3.0]
STRENGTH_TOP_ROW = 2.0

POSITIVE_RED = "#DC2626"
NEGATIVE_BLUE = "#2563EB"
NONE_GREY = GRAY  # #6B7280

# (code condition, colour, paper label)
CONDITION_STYLES = (
    ("none", NONE_GREY, "No steering"),
    ("negative", POSITIVE_RED, "Positive steering"),  # Be random! direction -> higher entropy
    ("positive", NEGATIVE_BLUE, "Negative steering"),  # Be sure! direction -> lower entropy
)

# Geometry (constrained layout); tuned so that the tight-cropped page and the axes boxes
# reproduce the submitted PDF (915.6 x 319.5 pt; axes 255.2 x 83.9 pt).
FIGSIZE = (12.6, 4.4)
LAYOUT_RECT = (0, 0, 1, 0.9)
LEGEND_ANCHOR = (0.5, 0.99)


def load_payloads(slug: str, layer: int) -> dict[float, dict]:
    out = {}
    for s in STRENGTHS:
        path = latest_post_fix(RESULTS / "no_instruction_control" / slug, f"no_instruction_layer{layer}_strength{s}")
        with open(path) as fh:
            payload = json.load(fh)
        payload["_path"] = str(path)
        out[s] = payload
    return out


def cond_entropies(payload: dict) -> dict[str, list[float]]:
    return {c: [record_entropy(r) for r in clean_probability_records(recs)] for c, recs in payload["results"].items()}


def build_figure(figsize=FIGSIZE, rect=LAYOUT_RECT, legend_anchor=LEGEND_ANCHOR):
    apply_paper_style()
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })
    set_truetype()

    payloads = {slug: load_payloads(slug, layer) for slug, _, layer in MODELS}
    ents = {slug: {s: cond_entropies(p) for s, p in pl.items()} for slug, pl in payloads.items()}

    # Shared KDE y-ceiling across all models, strengths and conditions (as submitted).
    peak = 0.0
    for slug in ents:
        for s in ents[slug]:
            for vals in ents[slug][s].values():
                _, dens = kde_curve(vals, 0.0, 1.0)
                peak = max(peak, float(np.max(dens, initial=0.0)))
    y_max_kde = peak * 1.1

    fig, axes = plt.subplots(2, len(MODELS), figsize=figsize, constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(rect=rect)
    handles = [Line2D([0], [0], color=c, lw=2, label=l) for _, c, l in CONDITION_STYLES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=legend_anchor, ncol=3,
               frameon=True, framealpha=1.0, edgecolor="#D1D5DB")

    summary = {}
    for col, (slug, name, layer) in enumerate(MODELS):
        # Top row: KDEs at 2x. z-order puts positive steering (red) on top.
        ax = axes[0, col]
        zorder = {"none": 2, "positive": 3, "negative": 4}
        top = ents[slug][STRENGTH_TOP_ROW]
        summary[name] = {"kde_medians_2x": {}}
        for cond, color, label in CONDITION_STYLES:
            vals = top.get(cond, [])
            plot_kde(ax, vals, color, "", 0.0, 1.0, alpha=0.12, zorder=zorder[cond])
            if vals:
                med = float(np.median(vals))
                summary[name]["kde_medians_2x"][label] = round(med, 4)
                ax.axvline(med, color=color, lw=1.2, linestyle="--", alpha=0.8, zorder=zorder[cond])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max_kde)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{name}\nLayer {layer}, steering strength {STRENGTH_TOP_ROW:g}×")
        ax.set_xlabel("Entropy")

        # Bottom row: dose response, anchored at 0x by the no-steering median.
        ax = axes[1, col]
        none_pooled = [v for s in STRENGTHS for v in ents[slug][s].get("none", [])]
        base = float(np.median(none_pooled))
        for cond, color, label in CONDITION_STYLES:
            if cond == "none":
                continue
            pts = [(0.0, base)] + [(s, float(np.median(ents[slug][s][cond]))) for s in STRENGTHS]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, marker="o", linewidth=1.8, markersize=6,
                    zorder={"positive": 3, "negative": 4}[cond])
            summary[name][f"median_vs_strength[{label}]"] = {f"{x:g}x": round(y, 4) for x, y in pts}
        ax.axhline(base, color=NONE_GREY, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0] + STRENGTHS)
        ax.set_xticklabels([f"{int(s)}×" for s in [0] + STRENGTHS])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(False)
        ax.set_xlabel("Steering strength")
        ax.set_ylabel("Median entropy" if col == 0 else "")
    return fig, payloads, ents, summary


def main() -> None:
    fig, payloads, ents, summary = build_figure()
    for slug in payloads:
        for s, p in payloads[slug].items():
            n_total = {c: len(r) for c, r in p["results"].items()}
            n_valid = {c: len(v) for c, v in ents[slug][s].items()}
            print(f"{slug} {s:g}x {Path(p['_path']).parent.name} valid {n_valid} of {n_total}")
    print(json.dumps(summary, indent=1))
    check_text_overlap(fig)
    for p in save(fig, STEM):
        print(p)
    compare(STEM)


if __name__ == "__main__":
    main()
