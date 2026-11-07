#!/usr/bin/env python3
"""DRAFT main-text figure: prompted sandbagging via uncertainty on MMLU + the steering vector as a probe.

(a) Answer entropy over A-D under the four MMLU system prompts, grouped by model
    (Qwen3.5-27B, Llama-3-8B; data as in Appendix G, prelim_mmlu_prompt_conditions.py).
(b) Soft accuracy under the three instructed prompts; the baseline is a bar above each model group.
(c) Top-1 agreement with the baseline answers.
(d) Probe at layer 56 (Qwen3.5-27B): AUROC of the Be random! - Be sure! steering vector for
    baseline vs. prompted activations, against the mean +- sd of 200 isotropic random directions,
    each allowed its better sign (prelim_probe_random_directions.py data).

Run from the repo root:
  AB_PAIRS_VARIANT=neutral python figures/scripts/camera_ready/fig_sandbagging_main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cr_prelim_style import (  # noqa: E402
    FULL_WIDTH_IN, INK, MUTED_INK, REF_LINE_COLOR, apply_prelim_style, display_model_name, results_dir, save, write_json,
)
from prelim_mmlu_prompt_conditions import CONDITIONS, MODELS, compute_metrics, load_conditions  # noqa: E402

STEM = "fig_sandbagging_bars_draft"
LAYER = 56
TICK = {"original": "Baseline\n(format only)", "maximize_entropy": "Maximize\nentropy",
        "sandbag_via_uncertainty": "Sandbag via\nuncertainty", "maximize_certainty": "Maximize\ncertainty"}
COLOR = {key: color for key, _, _, color in CONDITIONS}
INSTRUCTED = [c for c in CONDITIONS if c[0] != "original"]
PROBE_PROMPTS = ["maximize_entropy", "sandbag_via_uncertainty"]
GAP = 1.0  # extra space between model groups


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_stats() -> dict:
    run = sorted((results_dir() / "camera_ready_checks" / "probe_random_directions").glob("2*/projections.pt"))[-1]
    d = torch.load(run, map_location="cpu", weights_only=False)
    base = d["proj"]["format_instructed"][LAYER].numpy()
    out = {"source": str(run), "k": int(d["k"])}
    for key in PROBE_PROMPTS:
        x = d["proj"][key][LAYER].numpy()
        steer = auroc(-x[:, 0], -base[:, 0])  # paper sign: projection onto Be random! - Be sure!
        rnd = np.array([auroc(x[:, j], base[:, j]) for j in range(1, x.shape[1])])
        fold = np.maximum(rnd, 1 - rnd)  # random directions get their better sign
        out[key] = {"steering": steer, "random_mean": float(fold.mean()), "random_sd": float(fold.std(ddof=1)),
                    "frac_random_ge_steering": float((fold >= steer).mean())}
    return out


def grouped_positions(n_per_group: int, models) -> list[np.ndarray]:
    return [np.arange(n_per_group) + g * (n_per_group + GAP) for g in range(len(models))]


def group_axis(ax, positions, keys, models, ymin_frac=-0.42):
    """Vertical condition labels as ticks; model names centred below each group."""
    ax.set_xticks(np.concatenate(positions))
    ax.set_xticklabels([TICK[k] for _ in models for k in keys], rotation=0, fontsize=10.5)
    for pos, model in zip(positions, models if len(models) > 1 else []):
        ax.text(pos.mean(), ymin_frac, display_model_name(model), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, color=INK)
    ax.set_xlim(positions[0][0] - 0.6, positions[-1][-1] + 0.6)
    ax.grid(axis="x", visible=False)


def entropy_panel(ax, data, models=MODELS, title="(a) Answer entropy", ylabel=True):
    keys = [k for k, *_ in CONDITIONS]
    positions = grouped_positions(len(keys), models)
    for pos, model in zip(positions, models):
        vals = [data[model][k]["entropy"] for k in keys]
        parts = ax.violinplot(vals, positions=pos, widths=0.8, showextrema=False)
        for body, k in zip(parts["bodies"], keys):
            body.set_facecolor(COLOR[k]); body.set_edgecolor("none"); body.set_alpha(0.25)
        for x, v, k in zip(pos, vals, keys):
            q10, q50, q90 = np.quantile(v, [0.1, 0.5, 0.9])
            ax.vlines(x, q10, q90, color=COLOR[k], lw=2.0, zorder=3)
            ax.scatter([x], [q50], color=COLOR[k], s=36, zorder=4)
            ax.scatter([x], [float(np.mean(v))], color="white", edgecolor=COLOR[k], lw=1.3, s=38, zorder=5)
            ax.text(x + 0.1, q50, f"{q50:.2f}", ha="left", va="center", fontsize=9.5, color=INK, zorder=6)
    ax.axhline(2.0, color=REF_LINE_COLOR, lw=1.0, ls=":", zorder=1)
    ax.text(positions[-1][-1] + 0.55, 2.0, "uniform", ha="right", va="bottom", fontsize=9.5, color=MUTED_INK)
    ax.set_ylim(-0.04, 2.2)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    if ylabel:
        ax.set_ylabel("Entropy over A–D (bits)")
    ax.set_title(title)
    group_axis(ax, positions, keys, models)


def soft_accuracy_panel(ax, metrics, models=MODELS, title="(b) Soft accuracy", span=None):
    """Bars hang from the baseline level down to each prompt's soft accuracy (the drop)."""
    keys = [k for k, *_ in INSTRUCTED]
    positions = grouped_positions(len(keys), models)
    for pos, model in zip(positions, models):
        base = 100 * metrics[model]["original"]["soft_acc"]
        ax.hlines(base, pos[0] - 0.5, pos[-1] + 0.5, color=COLOR["original"], lw=1.2, ls="--", zorder=3)
        ax.text(pos[0] - 0.5, base + 1.5, f"baseline ({base:.1f})", ha="left", va="bottom", fontsize=9.5, color=MUTED_INK)
        for x, k in zip(pos, keys):
            y = 100 * metrics[model][k]["soft_acc"]
            ax.bar(x, y - base, bottom=base, width=0.6, facecolor=to_rgba(COLOR[k], 0.3), edgecolor=COLOR[k], linewidth=1.6, zorder=2)
            ax.text(x, min(y, base) - 1.5, f"{y - base:+.1f}".replace("-", "\u2212"), ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Soft accuracy (%)")
    ax.set_title(title)
    group_axis(ax, positions, keys, models)


def agreement_panel(ax, metrics, models=MODELS, title="(c) Same top answer as baseline", span=None):
    """Plain bars from 0 to the share of questions whose top answer matches the baseline."""
    keys = [k for k, *_ in INSTRUCTED]
    positions = grouped_positions(len(keys), models)
    for pos, model in zip(positions, models):
        for x, k in zip(pos, keys):
            y = 100 * metrics[model][k]["top1_agreement"]
            ax.bar(x, y, width=0.6, facecolor=to_rgba(COLOR[k], 0.3), edgecolor=COLOR[k], linewidth=1.6, zorder=2)
            ax.text(x, y + 1.5, f"{y:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Same top answer (%)")
    ax.set_title(title)
    group_axis(ax, positions, keys, models)


def probe_panel(ax, ps, title=f"(d) Probe, layer {LAYER}", show_model=True):
    xs = np.arange(len(PROBE_PROMPTS))
    for x, key in zip(xs, PROBE_PROMPTS):
        r = ps[key]
        ax.errorbar(x - 0.13, r["random_mean"], yerr=r["random_sd"], fmt="o", color=REF_LINE_COLOR, ms=7, capsize=4, lw=1.6, zorder=2)
        ax.scatter(x + 0.13, r["steering"], marker="*", s=220, color=COLOR[key], edgecolor="white", lw=0.6, zorder=3)
        ax.text(x + 0.23, r["steering"], f"{r['steering']:.2f}", va="center", ha="left", fontsize=9.5, color=INK)
    ax.axhline(0.5, color=REF_LINE_COLOR, lw=1.0, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([TICK[k] for k in PROBE_PROMPTS], rotation=0, fontsize=10.5)
    ax.set_xlim(-0.6, len(PROBE_PROMPTS) - 0.4)
    ax.set_ylim(0.15, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Probe AUROC")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if show_model:
        ax.text(0.5, -0.42, display_model_name(MODELS[0]), transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color=INK)
    ax.legend(handles=[Line2D([0], [0], marker="*", color=MUTED_INK, ls="", ms=12, label="Steering vector"),
                       Line2D([0], [0], marker="o", color=REF_LINE_COLOR, ls="", ms=7,
                              label=f"Random directions\n(mean ± sd, n={ps['k']})")],
              frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=9.5)


def common_span(metrics, model):
    """Height (pp) shared by the soft-accuracy and agreement panels, so equal drops look equal."""
    m = metrics[model]
    soft_drop = max(100 * (m["original"]["soft_acc"] - m[k]["soft_acc"]) for k, *_ in INSTRUCTED)
    agree_drop = max(100 * (1 - m[k]["top1_agreement"]) for k, *_ in INSTRUCTED)
    return float(np.ceil((max(soft_drop, agree_drop) + 6) / 5) * 5)


def legend(fig):
    handles = [Patch(facecolor=c, alpha=0.6, edgecolor="none", label=l) for _, l, _, c in CONDITIONS]
    handles += [Line2D([0], [0], marker="o", color=MUTED_INK, ls="", ms=6.5, label="Median (value shown)"),
                Line2D([0], [0], marker="o", color=MUTED_INK, markerfacecolor="white", ls="", ms=6.5, label="Mean"),
                Line2D([0], [0], color=MUTED_INK, lw=2.0, label="10th–90th pct.")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
               handlelength=1.4, columnspacing=1.4)


def main() -> None:
    apply_prelim_style()
    data, _ = load_conditions()
    metrics = {m: compute_metrics(c) for m, c in data.items()}
    ps = probe_stats()
    Q, L = MODELS
    # 1. grouped (both models in each panel)
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH_IN, 4.6), gridspec_kw={"width_ratios": [1.9, 1.15, 1.15, 0.75]}, constrained_layout=True)
    entropy_panel(axes[0], data); soft_accuracy_panel(axes[1], metrics); agreement_panel(axes[2], metrics); probe_panel(axes[3], ps)
    legend(fig); save(fig, STEM); plt.close(fig)
    # 2a/2b. one model per figure (Qwen with probe; Llama alone, e.g. for the appendix)
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH_IN, 3.3), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 0.8]}, constrained_layout=True)
    sp = common_span(metrics, Q)
    entropy_panel(axes[0], data, [Q]); soft_accuracy_panel(axes[1], metrics, [Q], span=sp); agreement_panel(axes[2], metrics, [Q], span=sp); probe_panel(axes[3], ps, show_model=False)
    save(fig, STEM + "_qwen_only"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN * 0.8, 3.3), gridspec_kw={"width_ratios": [1.2, 0.85, 0.85]}, constrained_layout=True)
    sp = common_span(metrics, L)
    entropy_panel(axes[0], data, [L]); soft_accuracy_panel(axes[1], metrics, [L], span=sp); agreement_panel(axes[2], metrics, [L], span=sp)
    save(fig, STEM + "_llama_only"); plt.close(fig)
    # 3. Qwen block left, Llama block right
    fig, axes = plt.subplots(1, 7, figsize=(FULL_WIDTH_IN * 1.25, 4.6), gridspec_kw={"width_ratios": [1.2, 0.8, 0.8, 0.8, 1.2, 0.8, 0.8]}, constrained_layout=True)
    entropy_panel(axes[0], data, [Q], "(a) Answer entropy"); soft_accuracy_panel(axes[1], metrics, [Q], "(b) Soft accuracy")
    agreement_panel(axes[2], metrics, [Q], "(c) Same top answer"); probe_panel(axes[3], ps, f"(d) Probe, layer {LAYER}")
    entropy_panel(axes[4], data, [L], "(e) Answer entropy"); soft_accuracy_panel(axes[5], metrics, [L], "(f) Soft accuracy")
    agreement_panel(axes[6], metrics, [L], "(g) Same top answer")
    legend(fig); save(fig, STEM + "_split"); plt.close(fig)
    write_json({"probe_layer56": ps}, STEM)
    print("wrote all")


if __name__ == "__main__":
    main()
