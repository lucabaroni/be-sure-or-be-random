"""Shared Figure 3 supplementary layout (local-model-sweep, zero-shot KDE grid)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (
    apply_paper_style,
    display_model_name,
    figure_group,
    kde_curve,
    latest_result_path,
    load_zero_shot_result,
    paired_entropy_differences,
    plot_kde,
    style_distribution_axis,
)

PLOT_BOX_ASPECT = 0.7
X_AXIS_LABEL = r"$\Delta$Entropy"


def _pair_label(certainty_instruction: str, uncertainty_instruction: str) -> str:
    return f'"{certainty_instruction}" vs "{uncertainty_instruction}"'


def _strip_y_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)


def _draw_median_vline_full_height(ax: plt.Axes, diffs: list[float], color: str) -> None:
    """Vertical dashed line at the median, spanning the full y-axis (after ylim is set)."""
    if not diffs:
        return
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    median = float(np.median(arr))
    ax.axvline(
        median,
        color=color,
        linestyle="--",
        linewidth=1.5,
        zorder=5,
        clip_on=False,
    )


def build_local_sweep_layout_figure(model_name: str) -> plt.Figure:
    """7-pair KDE layout matching supplementary_figure3_qwen35_27b_layout (local-model-sweep)."""
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.titlesize": 17,
        }
    )
    group = figure_group("figure3", "local-model-sweep")
    instruction_pairs = [tuple(pair) for pair in group["matrix"]["instruction_pairs"]]
    primary_pair = instruction_pairs[0]
    secondary_pairs = instruction_pairs[1:]

    fig = plt.figure(figsize=(13.2, 5.1))
    fig_ratio = fig.get_figheight() / fig.get_figwidth()
    top = 0.87
    bottom = 0.13
    left = 0.07
    right = 0.03
    main_gap = 0.034
    col_gap = 0.014
    row_gap = 0.05

    total_height = top - bottom
    small_height = (total_height - 2 * row_gap) / 3
    small_width = small_height * fig_ratio / PLOT_BOX_ASPECT
    left_width = total_height * fig_ratio / PLOT_BOX_ASPECT
    used_width = left + left_width + main_gap + (2 * small_width) + col_gap + right
    if used_width > 1.0:
        raise ValueError(
            f"Figure geometry exceeds available width; reduce spacing or widen figure (used={used_width:.3f})."
        )

    left_ax = fig.add_axes([left, bottom, left_width, total_height])
    right_col1_x = left + left_width + main_gap
    right_col2_x = right_col1_x + small_width + col_gap
    row_bottoms = [bottom + 2 * (small_height + row_gap), bottom + small_height + row_gap, bottom]
    right_axes = [fig.add_axes([right_col1_x, y, small_width, small_height]) for y in row_bottoms]
    right_axes.extend(fig.add_axes([right_col2_x, y, small_width, small_height]) for y in row_bottoms)
    fig.suptitle(display_model_name(model_name))

    pair_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(instruction_pairs)))
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2,
            label=_pair_label(certainty_instruction, uncertainty_instruction),
        )
        for color, (certainty_instruction, uncertainty_instruction) in zip(pair_colors, instruction_pairs, strict=False)
    ]
    shared_ymax = 0.0
    median_vlines: list[tuple[plt.Axes, list[float], str]] = []

    for color, (certainty_instruction, uncertainty_instruction) in zip(pair_colors, instruction_pairs, strict=False):
        path = latest_result_path(
            "zero_shot_local",
            model_name=model_name,
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
        )
        if path is None:
            continue
        records = load_zero_shot_result(path)
        diffs = paired_entropy_differences(records, certainty_instruction, uncertainty_instruction)
        if not diffs:
            continue
        _, density = kde_curve(diffs, x_min=-1.0, x_max=1.0)
        shared_ymax = max(shared_ymax, float(np.max(density, initial=0.0)))
        label = _pair_label(certainty_instruction, uncertainty_instruction)
        if (certainty_instruction, uncertainty_instruction) == primary_pair:
            plot_kde(left_ax, diffs, color=color, label=label, x_min=-1.0, x_max=1.0, alpha=0.1)
            median_vlines.append((left_ax, diffs, color))

    left_ax.axvline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
    style_distribution_axis(
        left_ax,
        X_AXIS_LABEL,
        "",
        (-1.0, 1.0),
    )
    _strip_y_axis(left_ax)
    left_ax.set_box_aspect(PLOT_BOX_ASPECT)

    for (certainty_instruction, uncertainty_instruction), ax, color in zip(
        secondary_pairs, right_axes, pair_colors[1:], strict=False
    ):
        path = latest_result_path(
            "zero_shot_local",
            model_name=model_name,
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
        )
        if path is not None:
            records = load_zero_shot_result(path)
            diffs = paired_entropy_differences(records, certainty_instruction, uncertainty_instruction)
            if diffs:
                _, density = kde_curve(diffs, x_min=-1.0, x_max=1.0)
                shared_ymax = max(shared_ymax, float(np.max(density, initial=0.0)))
                plot_kde(ax, diffs, color=color, label="", x_min=-1.0, x_max=1.0, alpha=0.1)
                median_vlines.append((ax, diffs, color))
        ax.axvline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
        style_distribution_axis(
            ax,
            "",
            "",
            (-1.0, 1.0),
        )
        _strip_y_axis(ax)
        ax.set_box_aspect(PLOT_BOX_ASPECT)
        ax.set_xlabel(X_AXIS_LABEL)

    ylim_top = shared_ymax * 1.05 if shared_ymax > 0 else 1.0
    left_ax.set_ylim(0.0, ylim_top)
    left_ax.set_xticks([-1.0, 0.0, 1.0])

    for ax in right_axes:
        ax.set_ylim(0.0, ylim_top)
        ax.set_xticks([-1.0, 1.0])

    for ax, diffs, color in median_vlines:
        _draw_median_vline_full_height(ax, diffs, color)

    left_ax.legend(
        handles=legend_handles,
        frameon=True,
        loc="upper left",
        borderaxespad=0.35,
        handlelength=1.8,
        labelspacing=0.4,
        fontsize=9,
        framealpha=0.9,
        edgecolor="#D1D5DB",
    )

    return fig
