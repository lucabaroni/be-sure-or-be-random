"""Camera-ready Fig. 11 (appendix): steering_vector_similarity.pdf.

Pairwise cosine similarity between the seven steering vectors at the designated layer
(Qwen3.5-27B L56, Qwen3-32B L56, Llama-3-70B L64), standard vs minimal ("barebone") system prompt.

Style: replicates the submitted PDF (paper figures/steering_vector_similarity.pdf,
2026-05-05, matplotlib 3.10.9). The script that produced it is not in git (the committed
figures/scripts/generate_steering_vector_similarity.py draws a different layout: rows = prompts,
columns = models, 'bwr' colormap, 45-degree single-line labels). The submitted layout was therefore
reverse-engineered from the PDF content stream and is reproduced here with explicit geometry:
  * 3 x 2 grid of 240 x 240 pt heatmaps (rows = models, columns = standard / minimal prompt),
    imshow with RdBu_r, vmin -1, vmax 1, white minor gridlines (1.2 pt), no spines;
  * cell annotations 9 pt, two decimals, white if value >= 0.75 else #111827 (annotate_matrix rule);
  * 3-line quoted tick labels ('"A"\\nvs\\n"B"'), 9 pt; y labels on the left column only,
    x labels (rotated 90 degrees) on the bottom row only;
  * model names 12 pt bold to the right of the right column, bold 12 pt column headers
    "Standard / Minimal\\nsystem prompt", horizontal colour bar on top (ticks -1..1, label 12 pt);
  * PDF rasterised at 100 dpi (as submitted: 334 x 334 px heatmap images).
Positions below are the submitted ones in PDF points (page 952.41 x 1184.49 pt).

Data: results/neutral/steering_vector_similarity/designated_{standard,barebone}_20260721_*/similarity_matrices.json
(the undated designated_{standard,barebone}/ directories are byte-identical copies).
Cosine similarity is sign-invariant to the vector convention only jointly; all seven vectors use the same
certainty - uncertainty convention, so pairwise cosines are identical in the paper's convention.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/final/fig11_steering_vector_similarity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util_fig4_5_6_11 import (  # noqa: E402
    OUT_DIR,
    RESULTS,
    check_text_overlap,
    compare,
    latest_post_fix,
    set_truetype,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STEM = "steering_vector_similarity"

MODELS = [  # (key in similarity_matrices.json, display name)
    ("Qwen/Qwen3.5-27B", "Qwen3.5-27B"),
    ("Qwen/Qwen3-32B", "Qwen3-32B"),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "Llama-3-70B"),
]
TEMPLATES = [("standard", "Standard\nsystem prompt"), ("barebone", "Minimal\nsystem prompt")]

# ---- geometry of the submitted PDF (points) ----
PAGE_W, PAGE_H = 952.410096, 1184.493125
AX_SIZE = 240.0
COL_X0 = [201.809375, 605.5632211538]
ROW_Y0 = [825.809375, 513.809375, 201.809375]
CBAR_RECT = (348.726298, 1130.609375, 349.92, 15.84)
HEADER_BASELINE = 1095.434375  # baseline of the last header line ("system prompt")
MODEL_LABEL_X = 1.04  # axes fraction (9.6 pt right of the axes)
CMAP = "RdBu_r"
PDF_DPI = 100


def load(template: str) -> tuple[dict, Path]:
    path = latest_post_fix(RESULTS / "steering_vector_similarity", f"designated_{template}",
                           filename="similarity_matrices.json")
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["metadata"]["template"] == template
    assert payload["metadata"]["fixed_layers_by_model"] == {
        "Qwen/Qwen3-32B": 56, "Qwen/Qwen3.5-27B": 56, "meta-llama/Meta-Llama-3-70B-Instruct": 64}
    return payload, path


def pair_labels(payload: dict) -> list[str]:
    return [f'"{a}"\nvs\n"{b}"' for a, b in payload["metadata"]["instruction_pairs"]]


def matrix(payload: dict, model: str) -> np.ndarray:
    res = payload["results"][model]
    pairs = payload["metadata"]["instruction_pairs"]
    expected = [f"{a.rstrip('!').replace(' ', '_')}_vs_{b.rstrip('!').replace(' ', '_')}" for a, b in pairs]
    assert res["labels"] == expected, (res["labels"], expected)
    return np.array([[np.nan if v is None else float(v) for v in row] for row in res["matrix"]])


def frac_rect(x0, y0, w, h):
    return (x0 / PAGE_W, y0 / PAGE_H, w / PAGE_W, h / PAGE_H)


def annotate(ax, values: np.ndarray) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center", color="#6B7280", fontsize=9)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v >= 0.75 else "#111827", fontsize=9)


def build():
    plt.rcdefaults()
    set_truetype()
    payloads = {t: load(t) for t, _ in TEMPLATES}
    labels = pair_labels(payloads["standard"][0])
    assert labels == pair_labels(payloads["barebone"][0])
    n = len(labels)

    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#F3F4F6")
    image = None
    stats = {}
    for r, (model, name) in enumerate(MODELS):
        for c, (template, _) in enumerate(TEMPLATES):
            ax = fig.add_axes(frac_rect(COL_X0[c], ROW_Y0[r], AX_SIZE, AX_SIZE))
            values = matrix(payloads[template][0], model)
            image = ax.imshow(np.ma.masked_invalid(values), vmin=-1.0, vmax=1.0, cmap=cmap, aspect="auto")
            annotate(ax, values)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            if r == len(MODELS) - 1:
                ax.set_xticklabels(labels, rotation=90)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(labels if c == 0 else [])
            ax.tick_params(axis="both", labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            if c == len(TEMPLATES) - 1:
                ax.text(MODEL_LABEL_X, 0.5, name, transform=ax.transAxes, ha="left", va="center",
                        fontsize=12, fontweight="bold")
            off = values[~np.eye(n, dtype=bool)]
            stats.setdefault(template, {})[name] = off
    cax = fig.add_axes(frac_rect(*CBAR_RECT))
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticklocation="top")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Cosine similarity", fontsize=12, labelpad=6)
    for c, (_, header) in enumerate(TEMPLATES):
        cx = (COL_X0[c] + AX_SIZE / 2) / PAGE_W
        # for multi-line text, va="baseline" anchors the baseline of the last line
        fig.text(cx, HEADER_BASELINE / PAGE_H, header, ha="center", va="baseline",
                 fontsize=12, fontweight="bold")
    payload_paths = {t: str(p) for t, (_, p) in payloads.items()}
    return fig, stats, payload_paths


def main() -> None:
    fig, stats, paths = build()
    for t, p in paths.items():
        print(f"{t}: {p}")
    summary = {}
    for template, per_model in stats.items():
        allv = np.concatenate(list(per_model.values()))
        summary[template] = {
            "all_models": {"min": float(allv.min()), "max": float(allv.max()), "mean": float(allv.mean()),
                           "n_offdiag_pairs": int(allv.size // 2)},
            **{m: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())} for m, v in per_model.items()},
        }
        print(f"{template:9s} off-diagonal (3 models, 21 pairs each): min {allv.min():.4f}  max {allv.max():.4f}  mean {allv.mean():.4f}")
        for m, v in per_model.items():
            print(f"    {m:12s} min {v.min():.4f} max {v.max():.4f} mean {v.mean():.4f}")
    check_text_overlap(fig, tight=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{STEM}.pdf", OUT_DIR / f"{STEM}.png"
    fig.savefig(pdf, dpi=PDF_DPI)
    fig.savefig(png, dpi=300)
    (OUT_DIR / f"{STEM}_summary.json").write_text(json.dumps({"inputs": paths, "offdiag": summary}, indent=1))
    print(pdf)
    print(png)
    compare(STEM)


if __name__ == "__main__":
    main()
