"""Camera-ready Fig. 6: mmlu_entropy_correctness_calibration_with_random_mean.pdf (+ Qwen3-32B appendix version).

Replicates the submitted figure (paper figures/
mmlu_entropy_correctness_calibration_with_random_mean.pdf, built on 2026-05-04 by
figures/scripts/generate_figure7_entropy_correctness_calibration.py @ fb4c892,
build_figure_with_random_mean(statistic="mean")) on the post-fix neutral MMLU results.
All plotting helpers are imported unchanged from that generator; this script only
(1) points it at explicit post-fix result files, (2) applies the option-mass validity rule,
(3) writes TrueType PDFs under the paper file names.

Decision D3: main text = Qwen3.5-27B, Be sure / Be random vector, format_instructed prompt;
appendix = Qwen3-32B with the same vector and prompt.

Validity rule (MMLU analogue of P(A)+P(B) > 0.9): a record counts only if the probability mass on the
four option tokens (label_probability_mass = P(A)+P(B)+P(C)+P(D)) exceeds 0.9. Paired entropy changes
need both the unsteered and the steered record to be valid; calibration uses each condition's valid records.

Sign convention: the stored vector is certainty - uncertainty. The run's "increase_entropy" condition
applies it with strength -2 (= +2 x (uncertainty - certainty), the paper's positive steering) and
"decrease_entropy" with +2 (the paper's negative steering). Labels follow the paper.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/final/fig6_mmlu_entropy_correctness_calibration.py
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
    save,
    set_truetype,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import generate_figure7_entropy_correctness_calibration as G  # noqa: E402
from common import apply_paper_style  # noqa: E402

MASS_THRESHOLD = 0.9
RUN_PREFIX = "cais_mmlu_all_test_maxall_layer56_strength2.0"
VARIANTS = [
    # (stem, result slug, display name)
    ("mmlu_entropy_correctness_calibration_with_random_mean", "Qwen_Qwen3.5-27B", "Qwen3.5-27B"),
    ("mmlu_entropy_correctness_calibration_with_random_mean_qwen3_32b", "Qwen_Qwen3-32B", "Qwen3-32B"),
]
SUBMITTED_STEM = "mmlu_entropy_correctness_calibration_with_random_mean"


def _valid(record: dict) -> bool:
    mass = record.get("label_probability_mass")
    if mass is None:
        mass = sum(float(record[f"prob_{lab}"]) for lab in G.LABELS)
    return bool(np.isfinite(mass) and mass > MASS_THRESHOLD)


def _filter(results: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    kept, dropped = {}, {}
    for cond, recs in results.items():
        kept[cond] = [r for r in recs if _valid(r)]
        dropped[cond] = len(recs) - len(kept[cond])
    return kept, dropped


def _paired(baseline: list[dict], cond: list[dict]) -> tuple[list[dict], list[dict]]:
    """Restrict both lists to examples valid in both (pairs are matched by example_idx downstream)."""
    common = {int(r["example_idx"]) for r in baseline} & {int(r["example_idx"]) for r in cond}
    return ([r for r in baseline if int(r["example_idx"]) in common],
            [r for r in cond if int(r["example_idx"]) in common])


def _soft(recs):
    return float(np.mean([float(r[f"normalized_prob_{r['answer_label']}"]) for r in recs]))


def _plot_soft_accuracy(ax, records, random_records):
    """Soft accuracy (mean P(correct), %): bars start at the no-steering level and end at the steered value."""
    from matplotlib.colors import to_rgba
    base = 100 * _soft(records["none"])
    rbase = 100 * _soft(random_records["none"])
    rnd = np.array([base + 100 * _soft(v) - rbase for k, v in sorted(random_records.items()) if k.startswith("random_")])
    items = [("Positive", 100 * _soft(records["increase_entropy"]), "#DC2626"),
             ("Negative", 100 * _soft(records["decrease_entropy"]), "#2563EB")]
    ax.axhline(base, color="#4B5563", lw=1.2, ls="--", zorder=3)
    ax.text(-0.55, base + 1.5, f"no steering ({base:.1f})", ha="left", va="bottom", fontsize=9, color="#4B5563")
    for x, (lab, y, col) in enumerate(items):
        ax.bar(x, y - base, bottom=base, width=0.6, facecolor=to_rgba(col, 0.3), edgecolor=col, linewidth=1.6, zorder=2)
        d = y - base
        ax.text(x, min(y, base) - 1.5, f"{d:+.1f}".replace("-", "\u2212"), ha="center", va="top", fontsize=10)
    x = len(items)
    m, sd = float(rnd.mean()), float(rnd.std(ddof=1))
    ax.errorbar(x, m, yerr=sd, fmt="o", color="#9CA3AF", ms=6, capsize=4, lw=1.4, zorder=4)
    ax.text(x, min(m - sd, base) - 1.5, (f"{m - base:+.1f}".replace("-", "\u2212") + f" \u00b1 {sd:.1f}"), ha="center", va="top", fontsize=10, color="#4B5563")
    ax.set_xticks(range(x + 1))
    ax.set_xticklabels([lab for lab, *_ in items] + ["Random"])
    ax.set_xlim(-0.6, x + 0.6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Soft accuracy (%)")
    ax.set_title("Soft accuracy")
    ax.grid(axis="x", visible=False)


def build(slug: str) -> tuple[plt.Figure, dict]:
    steer_path = latest_post_fix(RESULTS / "mmlu_entropy_steering" / slug,
                                 f"{RUN_PREFIX}_vecBe_sure_format_instructed")
    rand_path = latest_post_fix(RESULTS / "mmlu_random_vector_control" / slug,
                                f"{RUN_PREFIX}_seed0_n8_vecBe_sure_format_instructed")
    with open(steer_path) as fh:
        payload = json.load(fh)
    with open(rand_path) as fh:
        random_payload = json.load(fh)
    md = payload["metadata"]
    assert md["vector_certainty_instruction"] == "Be sure!" and md["vector_uncertainty_instruction"] == "Be random!"
    assert md["prompt_variant"] == "format_instructed" and md["fixed_layer"] == 56
    assert md["condition_strengths"] == {"none": 0.0, "increase_entropy": -2.0, "decrease_entropy": 2.0}

    records, dropped = _filter(payload["results"])
    random_records, random_dropped = _filter(random_payload["results"])
    baseline = records["none"]

    # --- identical to G.build_figure_with_random_mean(statistic="mean") from here on, on filtered records ---
    delta_groups = {}
    for cond in ("increase_entropy", "decrease_entropy"):
        b, c = _paired(baseline, records[cond])
        delta_groups[cond] = G._delta_entropy_by_baseline_correctness(b, c)
    entropy_deltas = G._pool_correctness_groups(delta_groups)
    calibration = {c: G._calibration_metrics(records[c]) for c in ("none", "increase_entropy", "decrease_entropy")}
    random_baseline = random_records["none"]
    random_deltas = {}
    for cond, recs in random_records.items():
        if cond.startswith("random_"):
            b, c = _paired(random_baseline, recs)
            random_deltas[cond] = G._paired_delta_entropy(b, c)
    random_calibration = {c: G._calibration_metrics(r) for c, r in sorted(random_records.items()) if c.startswith("random_")}

    apply_paper_style()
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })
    set_truetype()
    # Fixed layout (inches): every panel has the same height H; the calibration panel is H x H (square).
    FW, FH = 13.4, 4.2
    H, BOT, LEFT, GAP, RIGHT = 3.1, 0.78, 0.72, 0.78, 0.08
    w_sq = H
    w_rest = FW - LEFT - RIGHT - 3 * GAP - w_sq
    w1 = w2 = w_rest / 2.95
    w4 = w_rest - w1 - w2
    fig = plt.figure(figsize=(FW, FH))
    xs, axes = LEFT, []
    for w in (w1, w2, w_sq, w4):
        axes.append(fig.add_axes([xs / FW, BOT / FH, w / FW, H / FH]))
        xs += w + GAP
    G._plot_delta_entropy(axes[0], delta_groups)
    G._plot_random_mean_delta_panel(axes[1], random_deltas, entropy_deltas, "mean")
    G._plot_calibration(axes[2], calibration, random_calibration)
    _plot_soft_accuracy(axes[3], records, random_records)

    axes[1].set_title("Control: random vectors")
    # Even spacing in panel 2: the generator puts positive/negative/random at 0, 0.2, 0.48.
    for coll in axes[1].collections:
        if coll.get_label() == "Negative steering":
            off = coll.get_offsets(); off[:, 0] = 0.24; coll.set_offsets(off)
    axes[1].set_xticks([0.0, 0.24, 0.48])
    axes[1].set_xticklabels(["Positive\nsteering", "Negative\nsteering", "Random"])
    leg = axes[2].get_legend()
    for t in leg.get_texts():  # put the ECE value on its own line so the legend clears the curves
        t.set_text(t.get_text().replace(" (ECE", "\n(ECE"))
    # --- end of generator body ---

    random_means = [float(np.mean(v)) for _, v in sorted(random_deltas.items())]
    p_correct = {c: float(np.mean([float(r[f"normalized_prob_{r['answer_label']}"]) for r in records[c]]))
                 for c in ("none", "increase_entropy", "decrease_entropy")}
    mean_entropy = {c: float(np.mean([float(r["label_entropy"]) for r in records[c]]))
                    for c in ("none", "increase_entropy", "decrease_entropy")}
    summary = {
        "result_path": str(steer_path),
        "random_result_path": str(rand_path),
        "validity_rule": f"label_probability_mass (sum of P(A..D)) > {MASS_THRESHOLD}",
        "dropped_invalid_records": dropped,
        "dropped_invalid_random_records": random_dropped,
        "n_records": {c: len(r) for c, r in records.items()},
        "delta_entropy": {c: {k: G._summarize(v) for k, v in g.items()} for c, g in delta_groups.items()},
        "delta_entropy_pooled": {c: G._summarize(v) for c, v in entropy_deltas.items()},
        "calibration": {c: {k: v for k, v in m.items() if k != "curve"} for c, m in calibration.items()},
        "random_calibration": {c: {k: v for k, v in m.items() if k != "curve"} for c, m in random_calibration.items()},
        "random_mean_delta_entropy": {
            "per_vector": random_means,
            "mean_across_vectors": float(np.mean(random_means)),
            "sd_across_vectors": float(np.std(random_means, ddof=1)),
        },
        "mean_p_correct": p_correct,
        "mean_label_entropy_bits": mean_entropy,
    }
    return fig, summary


def main() -> None:
    for stem, slug, name in VARIANTS:
        print(f"== {stem} ({name})")
        fig, summary = build(slug)
        summary["model"] = name
        check_text_overlap(fig)
        fig.savefig(OUT_DIR / f"{stem}_softacc_PREVIEW.png", dpi=200)
        fig.savefig(OUT_DIR / f"{stem}_softacc_PREVIEW.pdf")
        print(OUT_DIR / f"{stem}_softacc_PREVIEW.png")

        print(json.dumps({k: summary[k] for k in ("dropped_invalid_records", "dropped_invalid_random_records",
                                                  "calibration", "random_mean_delta_entropy",
                                                  "mean_p_correct", "mean_label_entropy_bits")}, indent=1))
        plt.close(fig)



if __name__ == "__main__":
    main()
