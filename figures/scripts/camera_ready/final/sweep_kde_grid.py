#!/usr/bin/env python3
"""Camera-ready supplementary_all_models_kde_grid_standard.pdf (Appendix "Results across models").

One panel per model, 7 overlaid KDEs of per-question Delta-entropy (one per instruction pair), standard
system prompt. Two versions (sweep_models.VERSIONS): paper_families (30 models) and all_families (33).

Style rebuilt from the submitted PDF (generator not in the repo; matplotlib 3.10.8, DejaVu Sans):
  * panel 143.84 x 96.897 pt, column pitch 165.416 pt (gap = 0.15 x width), row pitch 140.50 pt
    (gap = 0.45 x height), 6 columns;
  * per pair: common.plot_kde (Scott bandwidth, 256-point grid on [-1, 1]), line lw 1.4, fill alpha 0.12;
    colours and legend order as submitted (Tailwind 600 shades, see PAIR_STYLE);
  * grey dashed zero line (#9CA3AF, lw 0.8); per-panel y-limits (-0.05, 1.05) x the panel's highest
    KDE peak (matplotlib's default 5 % margins, as in the submission); no y-axis, no grid;
  * x ticks -1, -0.5, 0, 0.5, 1 on every panel (labels 7 pt only on the bottom panel of each column),
    x-label "ΔEntropy" 8 pt;
  * panel titles "Name (size)" bold 8 pt, pad 3 pt, orange (#D85A30) local / green (#0F6E56) API;
    sizes written as in the submission ("(1000B)", "(46.7B)"); no size when undisclosed;
  * suptitle "ΔEntropy distributions — Standard prompt" bold 12 pt, baseline 25.435 pt above the grid;
  * legend below the grid: 4 columns, 9 pt, title "Instruction pair  (certainty / uncertainty)" 9 pt,
    handle length 1.6, handle lw 2, frame alpha 0.95, frame top 77.56 pt below the grid;
  * saved with bbox_inches="tight" (pad 0.1 in), as the submission.
The submitted figure was verified to be exactly this recipe: its curves are reproduced to 0.0000 of the
panel height from the April 2026 runs under results/zero_shot_* (default AB pairs, not results/neutral)
for Qwen3.5-27B, DeepSeek-V3.1, DeepSeek-V3.2 and GLM-4.7.

Deliberate changes vs submitted: models, data (post-fix neutral runs, 0.9 validity rule), order
(sweep_models.overall_median_delta_h, descending), more rows; x tick labels and x-label also on the
bottom panel of columns whose last row is empty; suptitle and legend centred on the panel grid (the
submission centred them on a wider canvas, 15.7 pt left of the grid centre); titles wider than a panel
drop the active-parameter token (-A12B, -A22B); TrueType fonts (pdf.fonttype 42).

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/final/sweep_kde_grid.py [--version paper_families|all_families|both]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _util_fig4_5_6_11 import OUT_DIR, REPO, check_text_overlap, compare, save, set_truetype  # noqa: E402
import sweep_models as sm  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import MODEL_META, apply_paper_style, kde_curve, model_source, plot_kde, source_color  # noqa: E402

STEM = "supplementary_all_models_kde_grid_standard"

# (certainty, uncertainty) -> (colour, legend label), in the submitted drawing and legend order.
PAIR_STYLE: dict[tuple[str, str], tuple[str, str]] = {
    ("Be sure!", "Be random!"): ("#2563EB", "Be sure / Be random"),
    ("Aim for low entropy", "Aim for high entropy"): ("#DC2626", "Aim low entropy / Aim high entropy"),
    ("Maximize certainty", "Maximize uncertainty"): ("#16A34A", "Maximize certainty / Maximize uncertainty"),
    ("Maximize certainty", "Minimize certainty"): ("#D97706", "Maximize certainty / Minimize certainty"),
    ("Commit to one option", "Keep both options equally likely"): ("#7C3AED", "Commit to one / Keep equal"),
    ("Reply with confidence", "Reply without confidence"): ("#DB2777", "Reply with confidence / Without confidence"),
    ("Be deterministic", "Assign equal probability to both options"): ("#0891B2", "Be deterministic / Equal probability"),
}
assert set(PAIR_STYLE) == set(sm.PAIRS)

X_MIN, X_MAX = -1.0, 1.0
X_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
X_LABEL = "ΔEntropy"
SUPTITLE = "ΔEntropy distributions — Standard prompt"
LEGEND_TITLE = "Instruction pair  (certainty / uncertainty)"
LINE_W, FILL_ALPHA = 1.4, 0.12
ZERO_LINE = dict(color="#9CA3AF", linewidth=0.8, linestyle="--")
Y_MARGIN = 0.05

# Geometry in points, measured on the submitted PDF.
AX_W, AX_H = 143.84, 96.8971962617
GAP_X, GAP_Y = 0.15 * AX_W, 0.45 * AX_H
NCOLS = 6
TITLE_SIZE, TITLE_PAD = 8, 3.0
TICK_SIZE, XLABEL_SIZE = 7, 8
SUPTITLE_SIZE, SUPTITLE_ABOVE_GRID = 12, 25.435
LEGEND_SIZE, LEGEND_TOP_BELOW_GRID = 9, 77.56
CANVAS_MARGIN = 72.0  # generous blank margin; bbox_inches="tight" crops it as in the submission


def size_label(model: str) -> str:
    size = sm.size_b(model)
    return "" if math.isnan(size) else f" ({size:g}B)"


def panel_titles(models: list[str]) -> dict[str, str]:
    """'Name (size)'; a title wider than the panel drops the active-parameter token(s) '-A<n>B'."""
    fig = plt.figure(figsize=(4, 2), dpi=72)
    renderer = fig.canvas.get_renderer()

    def width(text: str) -> float:
        t = fig.text(0, 0, text, fontsize=TITLE_SIZE, fontweight="bold")
        w = t.get_window_extent(renderer).width  # dpi 72: pixels == points
        t.remove()
        return w

    titles = {}
    for m in models:
        name = MODEL_META[m].display_name
        title = name + size_label(m)
        if width(title) > AX_W:
            title = re.sub(r"-A\d+(\.\d+)?B", "", name) + size_label(m)
        assert width(title) <= AX_W, f"title still too wide: {title}"
        titles[m] = title
    plt.close(fig)
    return titles


def load(models: list[str]) -> dict[str, dict]:
    data = {}
    for m in models:
        pairs = {}
        for cert, unc in sm.PAIRS:
            d = np.asarray(sm.delta_h(m, cert, unc), dtype=float)
            path = sm.result_path(m, cert, unc)
            pairs[(cert, unc)] = dict(diffs=d, path=str(path.relative_to(REPO)) if path else None)
        data[m] = dict(pairs=pairs, overall=sm.overall_median_delta_h(m))
    return data


def build_figure(models: list[str], data: dict[str, dict], titles: dict[str, str]) -> tuple[plt.Figure, list]:
    apply_paper_style()
    set_truetype()
    plt.rcParams.update({"xtick.labelsize": TICK_SIZE, "axes.labelsize": XLABEL_SIZE, "axes.grid": False})

    n = len(models)
    nrows = math.ceil(n / NCOLS)
    grid_w = NCOLS * AX_W + (NCOLS - 1) * GAP_X
    grid_h = nrows * AX_H + (nrows - 1) * GAP_Y
    fig_w = grid_w + 2 * CANVAS_MARGIN
    fig_h = grid_h + 2 * CANVAS_MARGIN + LEGEND_TOP_BELOW_GRID + SUPTITLE_ABOVE_GRID
    grid_x0 = CANVAS_MARGIN
    grid_y0 = CANVAS_MARGIN + LEGEND_TOP_BELOW_GRID  # bottom edge of the last row
    fig = plt.figure(figsize=(fig_w / 72, fig_h / 72))

    # bottom panel of every column (also for columns whose last row is empty)
    bottom_of_col = {c: max(i for i in range(n) if i % NCOLS == c) for c in range(min(NCOLS, n))}
    axes = []
    for i, m in enumerate(models):
        r, c = divmod(i, NCOLS)
        x0 = grid_x0 + c * (AX_W + GAP_X)
        y0 = grid_y0 + (nrows - 1 - r) * (AX_H + GAP_Y)
        ax = fig.add_axes([x0 / fig_w, y0 / fig_h, AX_W / fig_w, AX_H / fig_h])
        peak = 0.0
        for pair, (color, _) in PAIR_STYLE.items():
            d = data[m]["pairs"][pair]["diffs"]
            if d.size == 0:
                continue
            plot_kde(ax, d, color=color, label="", x_min=X_MIN, x_max=X_MAX, alpha=FILL_ALPHA, linewidth=LINE_W)
            peak = max(peak, float(kde_curve(d, X_MIN, X_MAX)[1].max()))
        ax.axvline(0.0, **ZERO_LINE)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-Y_MARGIN * peak, (1 + Y_MARGIN) * peak)
        ax.set_xticks(X_TICKS)
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        is_bottom = bottom_of_col[c] == i
        ax.tick_params(axis="x", labelbottom=is_bottom)
        if is_bottom:
            ax.set_xlabel(X_LABEL)
        ax.set_title(titles[m], fontsize=TITLE_SIZE, fontweight="bold",
                     color=source_color(model_source(m)), pad=TITLE_PAD)
        axes.append(ax)

    cx = (grid_x0 + grid_w / 2) / fig_w
    fig.text(cx, (grid_y0 + grid_h + SUPTITLE_ABOVE_GRID) / fig_h, SUPTITLE, ha="center", va="baseline",
             fontsize=SUPTITLE_SIZE, fontweight="bold")
    handles = [Line2D([0], [0], color=color, linewidth=2, label=label) for color, label in PAIR_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(cx, (grid_y0 - LEGEND_TOP_BELOW_GRID) / fig_h),
               ncol=4, fontsize=LEGEND_SIZE, title=LEGEND_TITLE, title_fontsize=LEGEND_SIZE, handlelength=1.6,
               framealpha=0.95, borderaxespad=0.0)
    return fig, axes


def title_overflow(fig: plt.Figure, axes: list) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for ax in axes:
        tb, ab = ax.title.get_window_extent(renderer), ax.get_window_extent(renderer)
        if tb.x0 < ab.x0 - 0.01 or tb.x1 > ab.x1 + 0.01:
            problems.append(f"title overflows panel: {ax.get_title()!r}")
    print(f"[title width check] {len(axes)} titles, {len(problems)} problems", *problems, sep="\n    ")
    return problems


def summary(version: str, models: list[str], data: dict, titles: dict) -> dict:
    nrows = math.ceil(len(models) / NCOLS)
    out = dict(
        figure=f"{STEM}.pdf", version=version, n_models=len(models), layout=f"{nrows} rows x {NCOLS} columns",
        order_key="sweep_models.overall_median_delta_h (mean over the 7 pairs of the per-pair median ΔH), descending",
        delta_h="H(uncertainty) - H(certainty), bits, per question, items valid (P(A)+P(B) > 0.9) under both",
        pairs=[f"{c} / {u}" for c, u in sm.PAIRS],
        models=[],
    )
    for rank, m in enumerate(models, 1):
        r, c = divmod(rank - 1, NCOLS)
        per_pair = {}
        for cert, unc in sm.PAIRS:
            p = data[m]["pairs"][(cert, unc)]
            d = p["diffs"]
            per_pair[f"{cert} / {unc}"] = dict(
                n=int(d.size), median=round(float(np.median(d)), 4) if d.size else None,
                mean=round(float(d.mean()), 4) if d.size else None,
                frac_zero=round(float((np.abs(d) < 1e-9).mean()), 3) if d.size else None, source=p["path"])
        meds = [v["median"] for v in per_pair.values() if v["median"] is not None]
        out["models"].append(dict(
            rank=rank, row=r + 1, col=c + 1, model=m, display_name=MODEL_META[m].display_name, title=titles[m],
            source=model_source(m), family=sm.family(m),
            size_b=None if math.isnan(sm.size_b(m)) else sm.size_b(m),
            overall_median_delta_h=round(data[m]["overall"], 4),
            n_pairs_median_negative=sum(v < 0 for v in meds), n_pairs_median_positive=sum(v > 0 for v in meds),
            per_pair=per_pair))
    return out


def run(version: str) -> None:
    models = list(sm.VERSIONS[version])
    data = load(models)
    models.sort(key=lambda m: -data[m]["overall"])  # stable: ties keep the VERSIONS order
    titles = panel_titles(models)
    fig, axes = build_figure(models, data, titles)
    problems = check_text_overlap(fig) + title_overflow(fig, axes)
    out_dir = sm.OUTPUT_DIRS[version]
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = out_dir.relative_to(OUT_DIR)
    paths = save(fig, f"{rel}/{STEM}")
    plt.close(fig)
    summ = summary(version, models, data, titles)
    summ["qc_problems"] = problems
    (out_dir / f"{STEM}_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    for p in paths:
        print(p)
    print(out_dir / f"{STEM}_summary.json")
    compare(f"{STEM}__{version}", new_pdf=paths[0], submitted_stem=STEM)
    print(f"[{version}] order:")
    for e in summ["models"]:
        print(f"  {e['rank']:2d} r{e['row']}c{e['col']} {e['overall_median_delta_h']:+.4f} {e['title']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", choices=[*sm.VERSIONS, "both"], default="both")
    args = ap.parse_args()
    for version in (sm.VERSIONS if args.version == "both" else [args.version]):
        run(version)


if __name__ == "__main__":
    main()
