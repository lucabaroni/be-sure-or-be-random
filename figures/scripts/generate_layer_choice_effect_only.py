"""Generate the available half of the layer-choice figure from neutral results.

The neutral all-layer steering runs contain the causal entropy effect at every
evaluated layer, but the condition-mean activation artifacts needed for the
cosine-similarity half of the original figure are no longer available.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from common import (
    apply_paper_style,
    display_model_name,
    figure_group,
    layer_mean_shifts,
    latest_result_path,
    load_all_layers_result,
    print_outputs,
    save_figure,
)

MODELS = [
    ("Qwen/Qwen3.5-27B", 56),
    ("Qwen/Qwen3-32B", 56),
    ("meta-llama/Meta-Llama-3-70B-Instruct", 64),
]

STRENGTH_COLORS = {
    -2.0: "#1E3A8A",
    -1.0: "#2563EB",
    1.0: "#DC2626",
    2.0: "#B91C1C",
}


def load_payloads(group: dict, model_name: str) -> dict[float, dict]:
    vector_certainty, vector_uncertainty = group["matrix"]["vector_source"][0]
    test_certainty, test_uncertainty = group["matrix"]["test_pairs"][0]
    payloads: dict[float, dict] = {}
    for strength in group["matrix"]["strengths"]:
        path = latest_result_path(
            "steering_all_layers",
            model_name=model_name,
            vector_certainty_instruction=vector_certainty,
            vector_uncertainty_instruction=vector_uncertainty,
            test_certainty_instruction=test_certainty,
            test_uncertainty_instruction=test_uncertainty,
            steering_strength=strength,
        )
        if path is not None:
            payloads[strength] = load_all_layers_result(path)
    return payloads


def build_figure() -> plt.Figure:
    apply_paper_style()
    group = figure_group("figure5", "steering-all-layers")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True, constrained_layout=True)

    for index, (model_name, designated_layer) in enumerate(MODELS):
        ax = axes[index]
        payloads = load_payloads(group, model_name)
        for strength in sorted(payloads):
            shifts = layer_mean_shifts(payloads[strength])
            layers = sorted(shifts)
            ax.plot(
                layers,
                [shifts[layer] for layer in layers],
                marker="o",
                markersize=3,
                linewidth=1.7,
                color=STRENGTH_COLORS[strength],
                label=f"strength {strength:+.0f}",
            )

        ax.axhline(0.0, color="#9CA3AF", linewidth=0.9, linestyle="--")
        ax.axvline(
            designated_layer,
            color="#4B5563",
            linewidth=1.1,
            linestyle=":",
            label="designated layer" if index == 0 else None,
        )
        ax.set_title(f"{display_model_name(model_name)} · layer {designated_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Mean $\Delta H$ (bits)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle(
        'Layer dependence of entropy steering · neutral AB pairs · "Be sure!" vs "Be random!"',
        fontsize=11,
    )
    return fig


def main() -> None:
    fig = build_figure()
    try:
        print_outputs(save_figure(fig, "paper_named/layer_choice_effect_only"))
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
