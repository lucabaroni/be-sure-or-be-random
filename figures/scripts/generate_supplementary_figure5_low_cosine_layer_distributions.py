from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (
    SLATE,
    apply_paper_style,
    display_model_name,
    entropy_values_for_instruction,
    figure_group,
    latest_result_path,
    load_all_layers_result,
    normalized_option_a_probability,
    paired_entropy_differences,
    plot_kde,
    print_outputs,
    save_figure,
    style_distribution_axis,
)
from be_sure_or_random.experiments import (
    compute_condition_cosine_similarities_from_artifact,
    latest_activation_artifact,
    latest_zero_shot_local_result,
)
from be_sure_or_random.utils import choose_min_cosine_layer_from_similarities
from common import load_zero_shot_result

POSITIVE_RED = "#DC2626"
NEGATIVE_BLUE = "#2563EB"


def load_steering_payloads(
    group: dict,
    model_name: str,
    test_pair_override: tuple[str, str] | None = None,
) -> dict[float, dict]:
    vector_certainty_instruction, vector_uncertainty_instruction = group["matrix"]["vector_source"][0]
    if test_pair_override is not None:
        test_certainty_instruction, test_uncertainty_instruction = test_pair_override
    else:
        test_certainty_instruction, test_uncertainty_instruction = group["matrix"]["test_pairs"][0]
    payloads: dict[float, dict] = {}
    for strength in group["matrix"]["strengths"]:
        path = latest_result_path(
            "steering_all_layers",
            model_name=model_name,
            vector_certainty_instruction=vector_certainty_instruction,
            vector_uncertainty_instruction=vector_uncertainty_instruction,
            test_certainty_instruction=test_certainty_instruction,
            test_uncertainty_instruction=test_uncertainty_instruction,
            steering_strength=strength,
        )
        if path is not None:
            payloads[strength] = load_all_layers_result(path)
    return payloads


def load_condition_cosine_similarities(group: dict, model_name: str) -> dict[int, float]:
    certainty_instruction, uncertainty_instruction = group["matrix"]["vector_source"][0]
    path = latest_activation_artifact(model_name, certainty_instruction, uncertainty_instruction)
    if path is None:
        return {}
    return compute_condition_cosine_similarities_from_artifact(
        path, certainty_instruction, uncertainty_instruction
    )


def valid_instruction_probs(records: list[dict], instruction: str) -> list[float]:
    values = []
    for record in records:
        if record.get("instruction") != instruction:
            continue
        try:
            values.append(normalized_option_a_probability(record))
        except Exception:
            continue
    return values


def layer_with_lowest_cosine(group: dict, model_name: str) -> int | None:
    similarities = load_condition_cosine_similarities(group, model_name)
    if not similarities:
        return None
    try:
        return choose_min_cosine_layer_from_similarities(similarities)
    except ValueError:
        return None


def layer_records(payload: dict, layer_idx: int) -> list[dict]:
    return list(payload["results"].get(f"layer_{layer_idx}", []))


def _load_zero_shot_baseline(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    layer_idx: int,
) -> dict | None:
    """Build a synthetic strength-0 payload from the zero-shot (no-steering) run."""
    candidates = [model_name]
    if "/" in model_name:
        head, tail = model_name.split("/", 1)
        candidates.append(f"{head.lower()}/{tail}")
    for candidate in candidates:
        path = latest_zero_shot_local_result(candidate, certainty_instruction, uncertainty_instruction)
        if path is not None:
            records = load_zero_shot_result(path)
            return {"results": {f"layer_{layer_idx}": records}}
    return None


FIXED_LAYERS = {
    "Qwen/Qwen3.5-27B": 56,
    "Qwen/Qwen3-32B": 56,
    "meta-llama/Meta-Llama-3-70B-Instruct": 64,
}

def build_figure(
    max_strength: float = 2.0,
    model_names: list[str] | None = None,
    test_pair_override: tuple[str, str] | None = None,
    instruction_in_labels: bool = True,
    subtitle: str | None = None,
) -> plt.Figure:
    apply_paper_style()
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.grid": False,
    })
    steering_group = figure_group("figure5", "steering-all-layers")
    if model_names is None:
        model_names = [
            "Qwen/Qwen3.5-27B",
            "Qwen/Qwen3-32B",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ]
    n_rows = len(model_names)
    row_height = 7.2 / 3
    fig, axes = plt.subplots(n_rows, 5, figsize=(16.5, row_height * n_rows), constrained_layout=True, squeeze=False, sharex="col")

    # Share y-axes within semantic groups (across all rows)
    prob_axes   = [axes[r, c] for r in range(n_rows) for c in (0, 2)]
    entropy_axes = [axes[r, c] for r in range(n_rows) for c in (1, 3)]
    delta_axes  = [axes[r, 4] for r in range(n_rows)]
    for ax in prob_axes[1:]:
        ax.sharey(prob_axes[0])
    for ax in entropy_axes[1:]:
        ax.sharey(entropy_axes[0])
    for ax in delta_axes[1:]:
        ax.sharey(delta_axes[0])
    main_title = "Control of entropy via instructions and steering"
    if subtitle:
        fig.suptitle(f"{main_title}\n{subtitle}", fontsize=15, fontweight="bold", y=1.12)
    else:
        fig.suptitle(main_title, fontsize=15, fontweight="bold", y=1.01)

    show_strengths = {-max_strength, 0.0, max_strength}
    strength_colors = {
        -max_strength: NEGATIVE_BLUE,
        0.0: SLATE,
        max_strength: POSITIVE_RED,
    }
    prob_bins = np.linspace(0.0, 1.0, 16)
    if test_pair_override is not None:
        certainty_instruction, uncertainty_instruction = test_pair_override
    else:
        certainty_instruction, uncertainty_instruction = steering_group["matrix"]["test_pairs"][0]
    if instruction_in_labels:
        column_xlabels = ["P(A)", "Entropy", "P(A)", "Entropy", "\u0394Entropy"]
    else:
        column_xlabels = [
            "P(option A)",
            "Entropy",
            "P(option A)",
            "Entropy",
            "\u0394Entropy",
        ]

    for row_idx, model_name in enumerate(model_names):
        row_axes = axes[row_idx]
        payloads = load_steering_payloads(steering_group, model_name, test_pair_override)
        selected_layer = FIXED_LAYERS.get(model_name)

        if 0.0 not in payloads and selected_layer is not None:
            baseline = _load_zero_shot_baseline(
                model_name,
                certainty_instruction,
                uncertainty_instruction,
                selected_layer,
            )
            if baseline is not None:
                payloads[0.0] = baseline

        row_axes[0].text(
            -0.18,
            0.5,
            f"{display_model_name(model_name)}\nLayer {selected_layer}",
            transform=row_axes[0].transAxes,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color="#374151",
            rotation=90,
        )

        if selected_layer is None or not payloads:
            for ax in row_axes:
                ax.text(0.5, 0.5, "No run", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
            continue

        for strength in sorted(payloads):
            if strength not in show_strengths:
                continue
            records = layer_records(payloads[strength], selected_layer)
            color = strength_colors[strength]
            label = "no steering" if strength == 0.0 else f"Steering strength: {int(strength)}\u00d7"

            certainty_probs = valid_instruction_probs(records, certainty_instruction)
            certainty_entropies = entropy_values_for_instruction(records, certainty_instruction)
            uncertainty_probs = valid_instruction_probs(records, uncertainty_instruction)
            uncertainty_entropies = entropy_values_for_instruction(records, uncertainty_instruction)
            diff_entropies = paired_entropy_differences(records, certainty_instruction, uncertainty_instruction)

            if uncertainty_probs:
                row_axes[0].hist(
                    uncertainty_probs,
                    bins=prob_bins,
                    density=True,
                    alpha=0.35,
                    color=color,
                    label=label,
                )
            if uncertainty_entropies:
                plot_kde(row_axes[1], uncertainty_entropies, color, label, 0.0, 1.0, alpha=0.08)
            if certainty_probs:
                row_axes[2].hist(
                    certainty_probs,
                    bins=prob_bins,
                    density=True,
                    alpha=0.35,
                    color=color,
                    label=label,
                )
            if certainty_entropies:
                plot_kde(row_axes[3], certainty_entropies, color, label, 0.0, 1.0, alpha=0.08)
            plot_kde(row_axes[4], diff_entropies, color, label, -1.0, 1.0, alpha=0.08)
            if diff_entropies:
                row_axes[4].axvline(float(np.median(diff_entropies)), color=color,
                                    lw=1.8, linestyle="--", alpha=0.9, zorder=5)

        is_bottom = row_idx == n_rows - 1
        is_top = row_idx == 0
        style_distribution_axis(row_axes[0], column_xlabels[0] if is_bottom else "", "", (0.0, 1.0))
        style_distribution_axis(row_axes[1], column_xlabels[1] if is_bottom else "", "", (0.0, 1.0))
        style_distribution_axis(row_axes[2], column_xlabels[2] if is_bottom else "", "", (0.0, 1.0))
        style_distribution_axis(row_axes[3], column_xlabels[3] if is_bottom else "", "", (0.0, 1.0))
        row_axes[4].axvline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
        style_distribution_axis(row_axes[4], column_xlabels[4] if is_bottom else "", "", (-1.0, 1.0))
        if is_top and not instruction_in_labels:
            for col in (0, 1):
                row_axes[col].set_title(f'"{uncertainty_instruction}"', fontsize=13, fontstyle="italic")
            for col in (2, 3):
                row_axes[col].set_title(f'"{certainty_instruction}"', fontsize=13, fontstyle="italic")
            row_axes[4].set_title(
                f'"{uncertainty_instruction}"\nvs\n"{certainty_instruction}"',
                fontsize=13, fontstyle="italic",
            )

        for col_idx, ax in enumerate(row_axes):
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
            if col_idx == 4:
                ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            else:
                ax.set_xticks([0.0, 0.5, 1.0])

    # Collect legend handles from all rows' last column, deduped by label,
    # ordered by strength (-, 0, +).
    label_order = {
        f"Steering strength: -{int(max_strength)}\u00d7": 0,
        "no steering": 1,
        f"Steering strength: {int(max_strength)}\u00d7": 2,
    }
    seen: dict[str, object] = {}
    for row_idx in range(n_rows):
        h_list, l_list = axes[row_idx, 4].get_legend_handles_labels()
        for h, lbl in zip(h_list, l_list):
            if lbl not in seen:
                seen[lbl] = h
    ordered = sorted(seen.items(), key=lambda kv: label_order.get(kv[0], 99))
    labels = [kv[0] for kv in ordered]
    handles = [kv[1] for kv in ordered]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(handles),
        frameon=True,
        framealpha=1.0,
        edgecolor="#D1D5DB",
    )

    return fig


def main() -> None:
    fig = build_figure(max_strength=2.0)
    try:
        print_outputs(save_figure(fig, "new/fixed_layer_distributions_2x"))
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
