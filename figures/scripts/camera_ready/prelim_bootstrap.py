#!/usr/bin/env python3
"""Preliminary appendix figure: bootstrap test of instructed entropy control.

For each model and meaningful instruction pair, plots the median dH with a
shared-AB bootstrap 95% CI under the standard and minimal ("barebone") system
prompts, against the noise floor estimated from the minimal-prompt control
battery. Filled markers: pair passes both one-sided tests after BH correction
(median dH > 0 and median dH > floor). Data: analysis_outputs/bootstrap_*.json
(see analysis_outputs/BOOTSTRAP_ENTROPY_CONTROL.md).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GRAY, TEMPLATE_COLORS, apply_paper_style  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cr_prelim_style import FONT_OVERRIDES  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis_outputs"
OUT = ROOT / "figures" / "camera_ready_prelim"

MODELS = [
    ("qwen_Qwen3.5-27B", "Qwen3.5-27B"),
    ("qwen_Qwen3-32B", "Qwen3-32B"),
    ("meta-llama_Meta-Llama-3-70B-Instruct", "Llama-3-70B"),
]
ARMS = [
    ("standard", "Standard", "bootstrap_standardZSL_vs_bareboneControls_noleak10_{}.json", TEMPLATE_COLORS["standard"]),
    ("minimal", "Minimal", "bootstrap_meaningful_vs_control_floor_{}.json", TEMPLATE_COLORS["minimal"]),
]


def short(cert: str, unc: str) -> str:
    return f"\"{unc}\" vs \"{cert}\""


def main() -> None:
    apply_paper_style()
    plt.rcParams.update(FONT_OVERRIDES)
    data = {}
    for slug, _ in MODELS:
        for arm, _, pattern, _ in ARMS:
            data[(slug, arm)] = json.loads((ANALYSIS / pattern.format(slug)).read_text())

    pair_order = [(p["certainty_instruction"], p["uncertainty_instruction"]) for p in data[(MODELS[0][0], "minimal")]["meaningful_pairs"]]
    y = np.arange(len(pair_order))[::-1]
    offset = {"standard": 0.17, "minimal": -0.17}

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13.5, 4.6), sharey=True)
    rows = []
    for ax, (slug, label) in zip(axes, MODELS):
        title_bits = []
        floor = data[(slug, "minimal")]
        ax.axvspan(floor["Cbar_ci_lo"], floor["Cbar_ci_hi"], color=GRAY, alpha=0.15, lw=0, zorder=0)
        ax.axvline(floor["Cbar"], color=GRAY, lw=1.0, ls="--", zorder=1)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        for arm, arm_label, _, color in ARMS:
            pairs = {(p["certainty_instruction"], p["uncertainty_instruction"]): p for p in data[(slug, arm)]["meaningful_pairs"]}
            n_pass = 0
            for yi, key in zip(y, pair_order):
                p = pairs.get(key)
                if p is None:
                    continue
                ok = p["verdict"] == "controlled"
                n_pass += ok
                yy = yi + offset[arm]
                ax.plot([p["T_ci_lo"], p["T_ci_hi"]], [yy, yy], color=color, lw=1.6, solid_capstyle="round", zorder=2)
                ax.scatter([p["T_median_dH"]], [yy], s=34, zorder=3, edgecolor=color, linewidth=1.3,
                           facecolor=color if ok else "white")
                rows.append((label, arm_label, short(*key), p["n_ab"], p["T_median_dH"], p["T_ci_lo"], p["T_ci_hi"],
                             data[(slug, arm)]["Cbar"], p["q_direction"], p["q_excess"], ok))
            title_bits.append(f"{arm_label.lower()} {n_pass}/{len(pair_order)}")
        ax.set_title(f"{label}\n{', '.join(title_bits)} significant", fontsize=10)
        ax.set_xlabel(r"median $\Delta H$ (95% bootstrap CI)")
        ax.set_xlim(-0.25, 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short(c, u) for c, u in pair_order])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="o", color=c, lw=1.6, markersize=6, label=f"{l} system prompt") for _, l, _, c in ARMS]
    handles += [Line2D([], [], marker="o", color=GRAY, lw=0, markerfacecolor="white", markersize=6, label="not significant (BH, both tests)"),
                Patch(facecolor=GRAY, alpha=0.3, label="control noise floor (minimal prompt, 95% CI)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"prelim_bootstrap.{ext}", dpi=200 if ext == "png" else None)

    # booktabs table
    lines = [r"\begin{tabular}{lllrrrcc}", r"\toprule",
             r"Model & Prompt & Instruction pair & $N$ & median $\Delta H$ [95\% CI] & floor & $q_{>0}$ / $q_{>\mathrm{floor}}$ & Pass \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}] & {r[7]:.3f} & {r[8]:.4f} / {r[9]:.4f} & {"yes" if r[10] else "no"} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "bootstrap_entropy_control.tex").write_text("\n".join(lines) + "\n")
    for arm in ("Standard", "Minimal"):
        print(arm, sum(r[10] for r in rows if r[1] == arm), "/", sum(1 for r in rows if r[1] == arm))
    for slug, label in MODELS:
        print(label, "n_ab(min)=", data[(slug, "minimal")]["n_ab_common"], "n_ab(std)=", data[(slug, "standard")]["n_ab_common"],
              "floor(min)=%.3f floor(std-analysis)=%.3f" % (data[(slug, "minimal")]["Cbar"], data[(slug, "standard")]["Cbar"]))


if __name__ == "__main__":
    main()
