#!/usr/bin/env python3
"""Preview figures with Llama-3-70B steered at layer 44 instead of 64 (not paper figures).

Each figure is built by the camera-ready final script's own build function with only the
Llama-3-70B layer (and, for Fig. 11, the Llama data source) replaced, so style, validity rule
and Qwen panels are unchanged. Outputs go to figures/camera_ready_prelim/layer44/:

  steering_dose_response_layer44.{pdf,png}         Fig. 5 layout (final/fig5_steering_dose_response.py)
  steering_dose_response_llama_64_vs_44.{pdf,png}  same, Llama-3-70B only: layer 64 | layer 44
  cross_instruction_steering_layer44.{pdf,png}     Fig. 10 layout (final/cross_instruction_steering.py)
  cross_instruction_llama_64_vs_44.png             Fig. 10 Llama-3-70B row at layer 64 (top) and 44 (bottom)
  dose_response_entropy_decreasing_zoom.{pdf,png}  entropy-decreasing direction of Fig. 5, zoomed y-axis
                                                   (values from analysis_outputs/llama_layer44/numbers.json)
  steering_vector_similarity_layer44.{pdf,png}     Fig. 11 layout (final/fig11_steering_vector_similarity.py);
                                                   Llama row from results/neutral/steering_vector_similarity/
                                                   llama_layer44_designated_{standard,barebone}/
  layer44_previews_summary.json                    inputs, figure summaries, text-overlap checks

Numbers for the report come from scripts/camera_ready_llama_layer44_numbers.py, not from here.

Run from the repo root (CPU only):
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/layer44_previews.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # figures/scripts/camera_ready
FINAL = HERE / "final"
REPO = HERE.parents[2]
sys.path.insert(0, str(FINAL))
sys.path.insert(0, str(HERE.parent))  # figures/scripts
sys.path.insert(0, str(REPO))
assert os.environ.get("AB_PAIRS_VARIANT") == "neutral", "run with AB_PAIRS_VARIANT=neutral"

import be_sure_or_random  # noqa: E402

# The shared venv also has an editable install of the main worktree (pre-fix results); refuse it.
assert Path(be_sure_or_random.__file__).resolve().is_relative_to(REPO), be_sure_or_random.__file__

import matplotlib.pyplot as plt  # noqa: E402

import cross_instruction_steering as ci  # noqa: E402
import fig11_steering_vector_similarity as f11  # noqa: E402
import fig5_steering_dose_response as f5  # noqa: E402
from _util_fig4_5_6_11 import RESULTS, check_text_overlap, set_truetype  # noqa: E402

LLAMA = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAMA_SLUG = "meta-llama_Meta-Llama-3-70B-Instruct"
LAYER = 44
OUT = REPO / "figures" / "camera_ready_prelim" / "layer44"


def save(fig: plt.Figure, stem: str, pdf_dpi: float | None = None, png_dpi: int = 200) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    set_truetype()
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    fig.savefig(paths[0], **({"dpi": pdf_dpi} if pdf_dpi else {}))
    fig.savefig(paths[1], dpi=png_dpi)
    plt.close(fig)
    return [str(p.relative_to(REPO)) for p in paths]


def dose_response(summary: dict) -> None:
    qwen = [m for m in f5.MODELS if not m[0].startswith("meta-llama")]
    assert len(qwen) == 2 and len(f5.MODELS) == 3
    f5.MODELS = qwen + [(LLAMA_SLUG, "Llama-3-70B", LAYER)]
    fig, payloads, _, fig_summary = f5.build_figure()
    summary["dose_response"] = {
        "inputs": {slug: {f"{s:g}x": p["_path"] for s, p in pl.items()} for slug, pl in payloads.items()},
        "figure_summary": fig_summary,
        "overlap_problems": check_text_overlap(fig),
        "outputs": save(fig, "steering_dose_response_layer44"),
    }
    # build_figure keys its data by slug, so the same model at two layers needs distinct keys.
    original_load = f5.load_payloads
    f5.load_payloads = lambda key, layer: original_load(key.split("@")[0], layer)
    f5.MODELS = [(f"{LLAMA_SLUG}@64", "Llama-3-70B (paper)", 64),
                 (f"{LLAMA_SLUG}@{LAYER}", "Llama-3-70B (candidate)", LAYER)]
    fig, payloads, _, fig_summary = f5.build_figure(figsize=(8.8, 4.4))
    f5.load_payloads = original_load
    for key, layer in ((f"{LLAMA_SLUG}@64", 64), (f"{LLAMA_SLUG}@{LAYER}", LAYER)):
        assert all(f"no_instruction_layer{layer}_" in p["_path"] for p in payloads[key].values()), key
    summary["dose_response_llama_64_vs_44"] = {
        "inputs": {key: {f"{s:g}x": p["_path"] for s, p in pl.items()} for key, pl in payloads.items()},
        "figure_summary": fig_summary,
        "overlap_problems": check_text_overlap(fig),
        "outputs": save(fig, "steering_dose_response_llama_64_vs_44"),
    }


def cross_instruction(summary: dict) -> None:
    assert ci.base.FIXED_LAYERS[LLAMA] == 64
    ci.base.FIXED_LAYERS[LLAMA] = LAYER
    fig = ci.build_figure()
    summary["cross_instruction"] = {
        "layers": dict(ci.base.FIXED_LAYERS),
        "overlap_problems": check_text_overlap(fig),
        "outputs": save(fig, "cross_instruction_steering_layer44", png_dpi=150),
    }


def cross_instruction_llama_64_vs_44(summary: dict) -> None:
    """Llama-3-70B row of Fig. 10 at layer 64 (paper) above the same row at layer 44."""
    from PIL import Image

    rows = []
    for layer in (64, LAYER):
        ci.base.FIXED_LAYERS[LLAMA] = layer
        fig = ci.base.build_figure(max_strength=2.0, model_names=[LLAMA], test_pair_override=ci.TEST_PAIR,
                                   instruction_in_labels=False, subtitle=None)
        rows.append(save(fig, f"_cross_instruction_llama_layer{layer}", png_dpi=150)[1])
    images = [Image.open(REPO / p).convert("RGB") for p in rows]
    stacked = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), "white")
    y = 0
    for image in images:
        stacked.paste(image, (0, y))
        y += image.height
    out = OUT / "cross_instruction_llama_64_vs_44.png"
    stacked.save(out)
    for p in rows:  # keep only the stacked image
        (REPO / p).unlink()
        (REPO / p).with_suffix(".pdf").unlink()
    summary["cross_instruction_llama_64_vs_44"] = {"outputs": [str(out.relative_to(REPO))]}


def dose_response_zoom(summary: dict) -> None:
    """Entropy-decreasing direction of the no-instruction dose response on a zoomed y-axis (numbers.json)."""
    numbers_path = REPO / "analysis_outputs" / "llama_layer44" / "numbers.json"
    dr = json.loads(numbers_path.read_text())["dose_response"]
    styles = {64: ("#6B7280", "--", "Layer 64 (paper)"), LAYER: (f5.NEGATIVE_BLUE, "-", f"Layer {LAYER}")}
    xs = [0, 1, 2, 3]
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, stat, title in zip(axes, ("median_H", "mean_H"), ("Median entropy", "Mean entropy")):
        for layer, (color, ls, label) in styles.items():
            e = dr[f"layer{layer}"]["entropy"]
            ys = [e["0x"]["none"][stat]] + [e[f"{s}x"]["entropy_decreasing"][stat] for s in (1, 2, 3)]
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=7, label=label)
            for x, y in zip(xs[2:], ys[2:]):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9 if layer == 64 else -15),
                            ha="center", fontsize=9, color="#374151")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}×" for x in xs])
        ax.set_xlabel("Steering strength")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.2)
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Llama-3-70B, no instruction, entropy-decreasing steering (paper: negative steering); "
                 "zoomed y-axes", fontsize=11)
    summary["dose_response_zoom"] = {
        "input": str(numbers_path.relative_to(REPO)),
        "overlap_problems": check_text_overlap(fig),
        "outputs": save(fig, "dose_response_entropy_decreasing_zoom"),
    }


def similarity(summary: dict) -> None:
    original_load = f11.load
    inputs = {}

    def load_layer44(template: str):
        payload, path = original_load(template)  # paper matrices (asserts layers 56/56/64)
        new_path = RESULTS / "steering_vector_similarity" / f"llama_layer44_designated_{template}" / "similarity_matrices.json"
        new = json.loads(new_path.read_text())
        assert new["metadata"]["template"] == template
        assert new["metadata"]["fixed_layers_by_model"] == {LLAMA: LAYER}
        assert new["metadata"]["instruction_pairs"] == payload["metadata"]["instruction_pairs"]
        merged = copy.deepcopy(payload)
        merged["results"][LLAMA] = new["results"][LLAMA]
        merged["metadata"]["fixed_layers_by_model"][LLAMA] = LAYER
        inputs[template] = {"qwen": str(Path(path).relative_to(REPO)), "llama_layer44": str(new_path.relative_to(REPO))}
        return merged, path

    f11.load = load_layer44
    f11.MODELS = [m if m[0] != LLAMA else (LLAMA, f"Llama-3-70B\nlayer {LAYER}") for m in f11.MODELS]
    fig, stats, _ = f11.build()
    summary["similarity"] = {
        "inputs": inputs,
        "offdiag_mean": {t: {m.replace("\n", " "): round(float(v.mean()), 4) for m, v in per.items()} for t, per in stats.items()},
        "overlap_problems": check_text_overlap(fig, tight=False),
        "outputs": save(fig, "steering_vector_similarity_layer44", pdf_dpi=f11.PDF_DPI, png_dpi=150),
    }


def main() -> None:
    summary: dict = {"note": "Preview only; Llama-3-70B at layer 44, Qwen models unchanged (layer 56)."}
    dose_response(summary)
    cross_instruction(summary)
    cross_instruction_llama_64_vs_44(summary)
    dose_response_zoom(summary)
    similarity(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer44_previews_summary.json").write_text(json.dumps(summary, indent=1))
    for section in summary.values():
        if isinstance(section, dict):
            for p in section.get("outputs", []):
                print(p)
    print(OUT.relative_to(REPO) / "layer44_previews_summary.json")


if __name__ == "__main__":
    main()
