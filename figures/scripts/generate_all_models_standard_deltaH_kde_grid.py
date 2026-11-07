"""All local models: one subplot per model, 7 pair KDEs overlaid.

Models ordered by decreasing mean median-ΔH across instruction pairs.

  AB_PAIRS_VARIANT=neutral \\
    python figures/scripts/generate_all_models_standard_deltaH_kde_grid.py

Set SYSTEM_PROMPT_VARIANT=barebone to use the paper's minimal prompt results.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    apply_paper_style,
    display_model_name,
    figure_group,
    kde_curve,
    latest_prompt_variant_result,
    latest_zero_shot_local_result,
    load_zero_shot_result,
    paired_entropy_differences,
    plot_kde,
    print_outputs,
    save_figure,
)

X_MIN, X_MAX = -1.0, 1.0
SYSTEM_PROMPT_VARIANT = os.environ.get("SYSTEM_PROMPT_VARIANT", "standard")
if SYSTEM_PROMPT_VARIANT not in {"standard", "barebone"}:
    raise ValueError(
        "SYSTEM_PROMPT_VARIANT must be 'standard' or 'barebone', "
        f"got {SYSTEM_PROMPT_VARIANT!r}"
    )

SHORT_PAIR_LABEL = {
    ("Be sure!", "Be random!"): "Be sure vs random",
    ("Maximize certainty", "Minimize certainty"): "Max vs min certainty",
    ("Maximize certainty", "Maximize uncertainty"): "Max cert vs max uncert",
    ("Commit to one option", "Keep both options equally likely"): "Commit vs keep equal",
    ("Reply with confidence", "Reply without confidence"): "With vs no confidence",
    ("Aim for low entropy", "Aim for high entropy"): "Low vs high entropy",
    ("Be deterministic", "Assign equal probability to both options"): "Be determ. vs equal prob",
}


def _aliases(model_name: str) -> list[str]:
    aliases = [model_name]
    if model_name.startswith("qwen/"):
        aliases.append("Qwen/" + model_name[len("qwen/") :])
    elif model_name.startswith("Qwen/"):
        aliases.append("qwen/" + model_name[len("Qwen/") :])
    return aliases


def resolve_result(model_name: str, certainty: str, uncertainty: str) -> Path | None:
    for alias in _aliases(model_name):
        if SYSTEM_PROMPT_VARIANT == "standard":
            path = latest_zero_shot_local_result(alias, certainty, uncertainty)
        else:
            path = latest_prompt_variant_result(
                alias, certainty, uncertainty, SYSTEM_PROMPT_VARIANT
            )
        if path is not None:
            return path
    return None


def load_model_pair_diffs(
    model_name: str,
    instruction_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for cert, uncert in instruction_pairs:
        path = resolve_result(model_name, cert, uncert)
        if path is None:
            continue
        records = load_zero_shot_result(path)
        diffs = paired_entropy_differences(records, cert, uncert)
        if diffs:
            out[(cert, uncert)] = diffs
    return out


def sort_key(pair_diffs: dict[tuple[str, str], list[float]]) -> float:
    """Mean of per-pair median ΔH; higher = stronger overall entropy control."""
    if not pair_diffs:
        return float("-inf")
    medians = [float(np.median(v)) for v in pair_diffs.values()]
    return float(np.mean(medians))


def build_figure() -> plt.Figure:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )

    group = figure_group("figure3", "local-model-sweep")
    instruction_pairs = [tuple(p) for p in group["matrix"]["instruction_pairs"]]
    models = list(group["matrix"]["models"])

    # Preload + sort by decreasing ΔH
    loaded: list[tuple[str, dict[tuple[str, str], list[float]], float]] = []
    for model_name in models:
        pair_diffs = load_model_pair_diffs(model_name, instruction_pairs)
        score = sort_key(pair_diffs)
        loaded.append((model_name, pair_diffs, score))
    loaded.sort(key=lambda t: t[2], reverse=True)

    pair_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(instruction_pairs)))
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8,
            label=SHORT_PAIR_LABEL.get((c, u), f"{c} vs {u}"),
        )
        for color, (c, u) in zip(pair_colors, instruction_pairs, strict=True)
    ]

    n_models = len(loaded)
    ncols = 4
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.15, nrows * 2.35),
        sharex=True,
        sharey=False,  # per-panel y so near-zero spikes don't crush other models
        constrained_layout=False,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (model_name, pair_diffs, score) in zip(axes_flat, loaded, strict=False):
        panel_ymax = 0.0
        for color, (cert, uncert) in zip(pair_colors, instruction_pairs, strict=True):
            diffs = pair_diffs.get((cert, uncert))
            if not diffs:
                continue
            plot_kde(
                ax,
                diffs,
                color=color,
                label="",
                x_min=X_MIN,
                x_max=X_MAX,
                alpha=0.10,
                linewidth=1.15,
            )
            _, density = kde_curve(diffs, x_min=X_MIN, x_max=X_MAX)
            panel_ymax = max(panel_ymax, float(np.max(density, initial=0.0)))
        ax.axvline(0.0, color="#9CA3AF", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0.0, panel_ymax * 1.08 if panel_ymax > 0 else 1.0)
        ax.set_xticks([-1.0, 0.0, 1.0])
        title = f"{display_model_name(model_name)}  (med ΔH={score:.2f})"
        ax.set_title(title, fontsize=8, color="#374151", pad=4)
        ax.grid(True, axis="both", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx, ax in enumerate(axes_flat):
        if idx >= n_models:
            ax.axis("off")
            continue
        row, col = divmod(idx, ncols)
        if row == nrows - 1 or idx + ncols >= n_models:
            ax.set_xlabel(r"$\Delta H$ (bits)")
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")

    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7.5,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        title="Instruction pair",
        title_fontsize=8,
    )
    prompt_label = (
        "Standard prompt"
        if SYSTEM_PROMPT_VARIANT == "standard"
        else "Barebone (minimal) prompt"
    )
    fig.suptitle(
        f"{prompt_label} · neutral AB pairs · models ordered by decreasing mean median ΔH",
        fontsize=10,
        y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.82, top=0.93, bottom=0.06, wspace=0.18, hspace=0.38)
    return fig


def main() -> None:
    fig = build_figure()
    try:
        stem = f"all_models_{SYSTEM_PROMPT_VARIANT}_deltaH_kde_grid"
        print_outputs(save_figure(fig, stem))
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
