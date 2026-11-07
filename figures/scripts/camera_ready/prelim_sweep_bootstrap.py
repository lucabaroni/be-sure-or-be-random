#!/usr/bin/env python3
"""Preliminary appendix figure: per-model, per-pair median dH with 95% bootstrap CIs (standard prompt).

All 30 models of the paper_families sweep (sweep_models.VERSIONS["submitted"]), ordered as in the
KDE grid (sweep_models.overall_median_delta_h, descending). For each model and instruction pair: median of
the per-item dH = H(uncertainty) - H(certainty), with a percentile 95% CI from 10,000 bootstrap resamples
over items (an item is one option pair in one order; same validity rule and pairing as the sweep figures).
Filled marker: CI entirely above 0; open marker: CI contains 0; filled triangle down: CI entirely below 0.
Robustness: the summary also reports counts for a cluster bootstrap that resamples option pairs (both
orders together).

Run: AB_PAIRS_VARIANT=neutral python figures/scripts/camera_ready/prelim_sweep_bootstrap.py
Writes figures/camera_ready_prelim/prelim_sweep_bootstrap.{pdf,png} and _summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "final"))
sys.path.insert(0, str(HERE))
import sweep_models as sm  # noqa: E402
from sweep_kde_grid import PAIR_STYLE  # noqa: E402
from common import clean_probability_records, record_entropy, source_color, model_source  # noqa: E402
from _cr_prelim_style import OUT_DIR, REF_LINE_COLOR, apply_prelim_style  # noqa: E402

N_BOOT = 10_000
SEED = 20261104
import os  # noqa: E402
VERSION = os.environ.get("SWEEP_VERSION", "submitted")
STEM = "prelim_sweep_bootstrap" + ("" if VERSION == "submitted" else f"_{VERSION}")


def items(model: str, cert: str, unc: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-item dH and the option-pair cluster id, paired exactly as common.paired_entropy_differences."""
    paired: dict = {}
    for r in clean_probability_records(sm.valid_records(model, cert, unc)):
        if r.get("instruction") not in (cert, unc):
            continue
        key = (r["option_a"], r["option_b"], bool(r.get("flip_ab", False)))
        paired.setdefault(key, {})[r["instruction"]] = record_entropy(r)
    dh, cl = [], []
    for (a, b, _), g in paired.items():
        if cert in g and unc in g:
            dh.append(g[unc] - g[cert]); cl.append("/".join(sorted((a, b))))
    ref = sorted(sm.delta_h(model, cert, unc))
    assert np.allclose(sorted(dh), ref), (model, cert)
    _, cid = np.unique(cl, return_inverse=True)
    return np.asarray(dh), cid


def boot_ci(dh: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(dh), size=(N_BOOT, len(dh)))
    meds = np.median(dh[idx], axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def cluster_ci(dh: np.ndarray, cid: np.ndarray, rng) -> tuple[float, float]:
    groups = [dh[cid == k] for k in range(cid.max() + 1)]
    n = len(groups)
    meds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        meds[i] = np.median(np.concatenate([groups[j] for j in rng.integers(0, n, n)]))
    return tuple(np.percentile(meds, [2.5, 97.5]))


def classify(lo: float, hi: float) -> str:
    return "above" if lo > 0 else "below" if hi < 0 else "contains_zero"


def main() -> None:
    apply_prelim_style(); plt.rcParams.update({"axes.labelsize": 15, "ytick.labelsize": 13, "axes.titlesize": 15})
    kde = json.loads((sm.OUTPUT_DIRS[VERSION] / "supplementary_all_models_kde_grid_standard_summary.json").read_text())
    order = [m["model"] for m in sorted(kde["models"], key=lambda m: m["rank"])]
    names = {m["model"]: m["display_name"] for m in kde["models"]}
    assert sorted(order) == sorted(sm.VERSIONS[VERSION]), "KDE summary out of date"
    rng = np.random.default_rng(SEED)

    rows = []
    for m in order:
        for c, u in sm.PAIRS:
            dh, cid = items(m, c, u)
            lo, hi = boot_ci(dh, rng)
            clo, chi = cluster_ci(dh, cid, rng)
            rows.append(dict(model=m, display_name=names[m], certainty=c, uncertainty=u, n_items=int(len(dh)),
                             median=float(np.median(dh)), ci_lo=float(lo), ci_hi=float(hi), status=classify(lo, hi),
                             cluster_ci_lo=float(clo), cluster_ci_hi=float(chi), cluster_status=classify(clo, chi)))

    # Figure: models on x (KDE-grid order), 7 dodged pair markers with vertical CI bars.
    # Geometry in points, matched to Fig. 2 (qwen35_27b_grid1x6.pdf is 963.36 pt wide; its fonts are 12-14 pt).
    PAGE_W, PAGE_H = 963.36, 370.0
    LEFT, RIGHT, BOTTOM, TOP = 82.0, 6.0, 104.0, 84.0   # TOP holds the instruction-pair legend
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 13,
                         "legend.fontsize": 12, "pdf.fonttype": 42, "savefig.bbox": None})
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    ax = fig.add_axes([LEFT / PAGE_W, BOTTOM / PAGE_H, (PAGE_W - LEFT - RIGHT) / PAGE_W, (PAGE_H - BOTTOM - TOP) / PAGE_H])
    pairs = list(sm.PAIRS)
    width = 0.78
    offs = np.linspace(-width / 2, width / 2, len(pairs))
    for r in rows:
        x = order.index(r["model"]) + offs[pairs.index((r["certainty"], r["uncertainty"]))]
        col = PAIR_STYLE[(r["certainty"], r["uncertainty"])][0]
        ax.plot([x, x], [r["ci_lo"], r["ci_hi"]], color=col, lw=1.0, alpha=0.9, solid_capstyle="butt")
        if r["status"] == "above":
            ax.plot(x, r["median"], "o", ms=3.6, color=col, mec=col)
        elif r["status"] == "below":
            ax.plot(x, r["median"], "v", ms=4.2, color=col, mec=col)
        else:
            ax.plot(x, r["median"], "o", ms=3.6, mfc="white", mec=col, mew=0.9)
    ax.axhline(0, color=REF_LINE_COLOR, lw=0.8, ls="--", zorder=0)
    for k in range(1, len(order)):
        ax.axvline(k - 0.5, color="#E5E7EB", lw=0.5, zorder=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[m] for m in order], rotation=50, ha="right", rotation_mode="anchor", color="black")
    ax.set_ylabel("Median $\\Delta$Entropy (bits)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    n = len(rows)
    cnt = {s: sum(r["status"] == s for r in rows) for s in ("above", "contains_zero", "below")}
    full = {(c, u): f'"{u}" vs "{c}"' for c, u in pairs}  # Fig. 2 naming: uncertainty vs certainty
    pair_handles = [Line2D([0], [0], color=PAIR_STYLE[pp][0], marker="o", lw=1.0, ms=4, label=full[pp]) for pp in pairs]
    style_handles = [
        Line2D([0], [0], marker="o", ls="", color="#374151", ms=4.5, label=f"95% CI above 0 ({cnt['above']}/{n})"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="#374151", ms=4.5, label=f"CI contains 0 ({cnt['contains_zero']}/{n})"),
        Line2D([0], [0], marker="v", ls="", color="#374151", ms=5, label=f"95% CI below 0 ({cnt['below']}/{n})"),
    ]
    ax.legend(handles=style_handles, loc="upper right", frameon=True, framealpha=0.95)
    leg1 = fig.legend(handles=pair_handles, loc="upper center", bbox_to_anchor=(0.5 + (LEFT - RIGHT) / 2 / PAGE_W, 1.0),
                      ncol=2, frameon=False, columnspacing=2.0, handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{STEM}.pdf")
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=200)

    # Overlap / clipping check: every text artist inside the page, no two texts overlapping.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text(), t.get_window_extent(rend)) for t in texts]
    tick_names = set(names.values())  # rotated labels: axis-aligned boxes overlap by construction
    W, H = fig.bbox.width, fig.bbox.height
    problems = [f"clipped: {txt!r}" for txt, b in boxes if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][1].overlaps(boxes[b][1]) and not (boxes[a][0] in tick_names and boxes[b][0] in tick_names):
                problems.append(f"overlap: {boxes[a][0]!r} / {boxes[b][0]!r}")
    print("FIG3 CHECK:", "PASS" if not problems else problems)

    def per_model(key):
        out = {}
        for m in order:
            rr = [r for r in rows if r["model"] == m]
            out[names[m]] = {s: sum(r[key] == s for r in rr) for s in ("above", "contains_zero", "below")}
        return out

    pm = per_model("status")
    summary = dict(
        figure=f"figures/camera_ready_prelim/{STEM}.pdf", version=VERSION, prompt="standard", n_models=len(order),
        n_pairs=len(pairs), n_model_pairs=n, n_boot=N_BOOT, seed=SEED, ci="percentile 95%, resampling items (option pair x order)",
        counts=cnt,
        counts_cluster_bootstrap={s: sum(r["cluster_status"] == s for r in rows) for s in ("above", "contains_zero", "below")},
        models_all_7_above=[k for k, v in pm.items() if v["above"] == 7],
        models_majority_above=[k for k, v in pm.items() if v["above"] >= 4],
        models_majority_below=[k for k, v in pm.items() if v["below"] >= 4],
        counts_above_with_median_ge_0_05=sum(r["status"] == "above" and r["median"] >= 0.05 for r in rows),
        per_model=pm, rows=rows,
    )
    (OUT_DIR / f"{STEM}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "per_model")}, indent=1))
    print(json.dumps(pm))


if __name__ == "__main__":
    main()
