#!/usr/bin/env python3
"""Draft appendix figure: the steering vector as a sandbagging probe, against random directions.

(a) Steering effect of the Be sure!/Be random! vector at each layer (Qwen3.5-27B, binary task,
    steering_all_layers): largest |change in mean DeltaH| relative to unsteered over strengths +-1, +-2.
(b) Probe AUROC by layer (MMLU, baseline vs. prompted activations at the last prompt token):
    steering vector (paper sign, AUROC = P(prompt projection > baseline projection)) and the
    2.5-97.5% band of K isotropic random unit directions (sign as drawn).
(c) Layer 56: distribution of random-direction AUROCs folded to max(AUROC, 1 - AUROC), i.e.
    random directions are allowed to choose their best sign; steering-vector AUROC marked.

Data: results/neutral/camera_ready_checks/probe_random_directions/<latest>/projections.pt
      (scripts/camera_ready_probe_random_directions.py) and steering_all_layers.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
from _cr_prelim_style import H_UP_COLOR, SANDBAG_COLOR, MUTED_INK, apply_prelim_style, results_dir, save, write_json  # noqa: E402
from common import figure_group, layer_mean_shifts  # noqa: E402
from generate_layer_choice_effect_only import load_payloads  # noqa: E402

STEM = "prelim_probe_random_directions"
LAYER = 56
PROMPTS = [("maximize_entropy", "Maximize entropy", H_UP_COLOR), ("sandbag_via_uncertainty", "Sandbag via uncertainty", SANDBAG_COLOR)]


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    apply_prelim_style()
    run = sorted((results_dir() / "camera_ready_checks" / "probe_random_directions").glob("2*/projections.pt"))[-1]
    d = torch.load(run, map_location="cpu", weights_only=False)
    layers, proj = d["layers"], d["proj"]
    base = {L: proj["format_instructed"][L].numpy() for L in layers}
    res = {}
    for p, *_ in PROMPTS:
        res[p] = {}
        for L in layers:
            b, x = base[L], proj[p][L].numpy()
            st = auroc(-x[:, 0], -b[:, 0])  # paper sign: projection onto Be random! - Be sure!
            rnd = np.array([auroc(x[:, j], b[:, j]) for j in range(1, x.shape[1])])
            fold = np.maximum(rnd, 1 - rnd)
            res[p][L] = dict(steering=st, rnd_lo=float(np.quantile(rnd, .025)), rnd_hi=float(np.quantile(rnd, .975)),
                             rnd_med=float(np.median(rnd)), fold_med=float(np.median(fold)), fold_q95=float(np.quantile(fold, .95)),
                             fold_max=float(fold.max()), frac_fold_ge_steering=float((fold >= max(st, 1 - st)).mean()), _fold=fold)
    pay = load_payloads(figure_group("figure5", "steering-all-layers"), "Qwen/Qwen3.5-27B")
    shifts = {s: layer_mean_shifts(pay[s]) for s in pay}
    sl = sorted(next(iter(shifts.values())))
    unsteered = next(iter(shifts.values()))[sl[0]]
    effect = [max(abs(shifts[s][L] - unsteered) for s in shifts) for L in sl]

    from matplotlib.patches import Patch
    fig, (a, b) = plt.subplots(2, 1, figsize=(13.5, 5.6), sharex=True, constrained_layout=True,
                               gridspec_kw={"height_ratios": [1, 1.5]})
    a.plot(sl, effect, marker="o", ms=4, lw=1.8, color="#374151")
    a.set_ylabel("Steering effect\n" + r"(max $|\Delta$ mean $\Delta H|$)")
    a.set_title("(a) Steering effect of the vector, binary task")
    rnd_handles = []
    for p_, lab, col in PROMPTS:
        r = res[p_]
        mu = np.array([r[L]["_fold"].mean() for L in layers]); lo = np.array([np.percentile(r[L]["_fold"], 2.5) for L in layers]); hi = np.array([np.percentile(r[L]["_fold"], 97.5) for L in layers])
        b.fill_between(layers, lo, hi, color=col, alpha=0.15, lw=0)
        b.plot(layers, mu, color=col, lw=1.0, ls=":", alpha=0.9)
        b.plot(layers, [r[L]["steering"] for L in layers], marker="o", ms=4, lw=1.8, color=col, label=f"{lab}: steering vector")
        rnd_handles.append(Patch(facecolor=col, alpha=0.3, edgecolor="none", label=f"{lab}: random (mean, 95% range)"))
    b.axhline(0.5, color="#9CA3AF", ls="--", lw=0.9)
    b.set_ylabel("Probe AUROC"); b.set_xlabel("Layer"); b.set_ylim(0, 1.02)
    b.set_title("(b) Probe AUROC on MMLU (prompt vs. baseline)")
    h, l = b.get_legend_handles_labels()
    b.legend(handles=h + rnd_handles, frameon=False, loc="lower left", fontsize=9, ncol=2)
    for ax in (a, b):
        ax.axvline(LAYER, color="#111827", ls=":", lw=1.2)
    paths = save(fig, STEM)
    summ = {"source": str(run), "k": d["k"], "seed": d["seed"], "steering_effect_by_layer": dict(zip(map(str, sl), effect)),
            "probe": {p: {str(L): {k: v for k, v in res[p][L].items() if not k.startswith("_")} for L in layers} for p in res}}
    write_json(summ, STEM)
    print("wrote", paths)
    for p in res:
        for L in layers:
            r = res[p][L]; print(p[:8], L, "steer %.3f  rnd95 [%.2f,%.2f]  fold med %.2f q95 %.2f max %.2f  frac>=steer %.3f" % (r["steering"], r["rnd_lo"], r["rnd_hi"], r["fold_med"], r["fold_q95"], r["fold_max"], r["frac_fold_ge_steering"]))


if __name__ == "__main__":
    main()
