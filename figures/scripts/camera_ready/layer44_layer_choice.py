#!/usr/bin/env python3
"""Layer-choice figure (appendix D) rebuilt on the fixed prompt path.

Left column: cosine similarity between mean residual-stream activations under
"Be sure!" and "Be random!" (results/neutral/layer_condition_cosine, recomputed by
scripts/camera_ready_layer_cosine.py). Right column: mean entropy shift of the
Be sure/Be random steering vector applied at each layer (steering_all_layers).
Layout follows the submitted layer_choice.pdf (rows: Llama-3-70B, Qwen3-32B,
Qwen3.5-27B). Missing cosine data is marked as pending.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PURPLE, apply_paper_style, display_model_name, figure_group, layer_mean_shifts  # noqa: E402
from generate_layer_choice_effect_only import STRENGTH_COLORS, load_payloads  # noqa: E402
from _cr_prelim_style import FONT_OVERRIDES, OUT_DIR  # noqa: E402
from be_sure_or_random.utils import slugify_model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODELS = [
    ("meta-llama/Meta-Llama-3-70B-Instruct", 44),
    ("Qwen/Qwen3-32B", 56),
    ("Qwen/Qwen3.5-27B", 56),
]


def main() -> None:
    apply_paper_style()
    plt.rcParams.update(FONT_OVERRIDES)
    group = figure_group("figure5", "steering-all-layers")
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(13.5, 11.0), constrained_layout=True)
    for row, (model_name, layer) in enumerate(MODELS):
        ax_c, ax_e = axes[row]
        name = display_model_name(model_name)
        cos_path = ROOT / "results" / "neutral" / "layer_condition_cosine" / f"{slugify_model_name(model_name)}.json"
        if cos_path.exists():
            rows = json.loads(cos_path.read_text())["layers"]
            ax_c.plot([r["layer"] for r in rows], [r["cos_condition_means"] for r in rows], marker="o", markersize=4, lw=1.8, color=PURPLE)
        else:
            ax_c.set_xlim(-3, 83)
            ax_c.text(0.5, 0.5, "pending (needs gated weights)", ha="center", va="center", transform=ax_c.transAxes, color="#6B7280")
        ax_c.axvline(layer, color="#4B5563", lw=1.1, ls=":")
        ax_c.set_ylim(0.75, 1.01)
        ax_c.set_title(f"{name}: condition cosine")
        ax_c.set_ylabel("Cosine similarity")

        payloads = load_payloads(group, model_name)
        for strength in sorted(payloads):
            shifts = layer_mean_shifts(payloads[strength])
            ls = sorted(shifts)
            ax_e.plot(ls, [shifts[k] for k in ls], marker="o", markersize=4, lw=1.8, color=STRENGTH_COLORS[strength], label=f"strength {strength:+.0f}")
        ax_e.axhline(0.0, color="#9CA3AF", lw=0.9, ls="--")
        ax_e.axvline(layer, color="#4B5563", lw=1.1, ls=":", label=f"designated layer ({layer})")
        ax_e.set_ylim(-1.0, 1.0)
        ax_e.set_title(f"{name}: layer dependence")
        ax_e.set_ylabel(r"Mean $\Delta H$ under steering")
        ax_e.legend(loc="lower left", frameon=False, ncol=2)
        for ax in (ax_c, ax_e):
            ax.set_xlabel("Layer")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "layer44" / "layer_choice_layer44.pdf")
    fig.savefig(OUT_DIR / "layer44" / "layer_choice_layer44.png", dpi=110)
    print("wrote", OUT_DIR / "layer44" / "layer_choice_layer44.pdf")


if __name__ == "__main__":
    main()
