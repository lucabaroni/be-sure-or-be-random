#!/usr/bin/env python3
"""R1 (preliminary): prompted sandbagging via uncertainty on MMLU.

Qwen3.5-27B and Llama-3-8B answer the full MMLU test set (14,042 questions) under four
system prompts: the format-only baseline (``format_instructed``), ``maximize_entropy``,
``sandbag_via_uncertainty`` and ``maximize_certainty``. For each condition we report the
4-way answer entropy H over A-D (bits, probabilities renormalised over the four labels),
hard accuracy (argmax correct), soft accuracy (mean renormalised P(correct), i.e. accuracy
under sampling) and top-1 agreement with the baseline argmax.

Data:
  results/neutral/mmlu_prompt_conditions/<model>/*maxall*/results.json
  Qwen3.5-27B has no format_instructed condition in that run; its baseline is the
  ``none`` condition of the MMLU entropy-steering run (same format_instructed prompt, no
  steering), which is checked to be bit-identical to the format_instructed behaviour
  stored by the steering-vector probe run.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral python figures/scripts/camera_ready/prelim_mmlu_prompt_conditions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cr_prelim_style import (  # noqa: E402
    FULL_WIDTH_IN,
    H_DOWN_COLOR,
    H_UP_COLOR,
    INK,
    MUTED_INK,
    NONE_COLOR,
    REF_LINE_COLOR,
    SANDBAG_COLOR,
    apply_prelim_style,
    display_model_name,
    rel,
    results_dir,
    save,
    unique_path,
    write_json,
    write_table,
)

STEM = "prelim_mmlu_prompt_conditions"
LABELS = ("A", "B", "C", "D")
# (key, legend label, tick label, colour)
CONDITIONS = [
    ("original", "Baseline (format only)", "Baseline", NONE_COLOR),
    ("maximize_entropy", "Maximize entropy", "Max.\nentropy", H_UP_COLOR),
    ("sandbag_via_uncertainty", "Sandbag via uncertainty", "Sandbag", SANDBAG_COLOR),
    ("maximize_certainty", "Maximize certainty", "Max.\ncertainty", H_DOWN_COLOR),
]
TABLE_NAMES = {
    "original": "Baseline",
    "maximize_entropy": "Maximize entropy",
    "sandbag_via_uncertainty": "Sandbag via uncertainty",
    "maximize_certainty": "Maximize certainty",
}
MODELS = ["qwen/Qwen3.5-27B", "meta-llama/Meta-Llama-3-8B-Instruct"]


# ----------------------------------------------------------------------------- data
def _arrays(records: list[dict]) -> dict[str, np.ndarray]:
    records = sorted(records, key=lambda r: int(r["example_idx"]))
    probs = np.asarray([[float(r[f"prob_{x}"]) for x in LABELS] for r in records], dtype=float)
    return {
        "idx": np.asarray([int(r["example_idx"]) for r in records]),
        "answer": np.asarray([LABELS.index(r["answer_label"]) for r in records]),
        "probs": probs,
        "entropy": np.asarray([float(r["label_entropy"]) for r in records], dtype=float),
    }


def load_conditions() -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, str]]]:
    root = results_dir()
    data: dict[str, dict[str, dict]] = {}
    sources: dict[str, dict[str, str]] = {}

    # Qwen3.5-27B: prompt conditions + baseline from the steering run's no-steering condition.
    qwen_pc = unique_path(root / "mmlu_prompt_conditions" / "qwen_Qwen3.5-27B", "cais_mmlu_all_test_maxall_prompts_*/results.json")
    qwen_none = unique_path(
        root / "mmlu_entropy_steering" / "Qwen_Qwen3.5-27B",
        "cais_mmlu_all_test_maxall_layer56_strength2.0_vecBe_sure_format_instructed_*/results.json",
    )
    qwen_probe = unique_path(
        root / "mmlu_steering_vector_probe" / "Qwen_Qwen3.5-27B",
        "cais_mmlu_all_test_maxall_probe_format_instructed+maximize_entropy_*/results.json",
    )
    pc = json.loads(qwen_pc.read_text())
    steer = json.loads(qwen_none.read_text())
    assert steer["metadata"]["prompt_variant"] == "format_instructed"
    baseline = _arrays(steer["results"]["none"])
    probe_baseline = _arrays(json.loads(qwen_probe.read_text())["behavior"]["format_instructed"])
    if not (np.array_equal(baseline["idx"], probe_baseline["idx"]) and np.array_equal(baseline["probs"], probe_baseline["probs"])):
        raise AssertionError("Qwen3.5-27B baseline differs between steering-run 'none' and probe-run format_instructed")
    data["qwen/Qwen3.5-27B"] = {"original": baseline, **{k: _arrays(v) for k, v in pc["results"].items()}}
    sources["qwen/Qwen3.5-27B"] = {
        "original": rel(qwen_none) + " [results.none]",
        "original_crosscheck": rel(qwen_probe) + " [behavior.format_instructed, identical]",
        "prompt_conditions": rel(qwen_pc),
    }

    llama_pc = unique_path(root / "mmlu_prompt_conditions" / "meta-llama_Meta-Llama-3-8B-Instruct", "cais_mmlu_all_test_maxall_prompts_*/results.json")
    pc = json.loads(llama_pc.read_text())
    conds = {k: _arrays(v) for k, v in pc["results"].items()}
    conds["original"] = conds.pop("format_instructed")
    data["meta-llama/Meta-Llama-3-8B-Instruct"] = conds
    sources["meta-llama/Meta-Llama-3-8B-Instruct"] = {"all_conditions": rel(llama_pc), "original": "format_instructed"}

    for model, conds in data.items():
        missing = [c for c, *_ in CONDITIONS if c not in conds]
        if missing:
            raise KeyError(f"{model}: missing conditions {missing}")
        ref = conds["original"]
        for name, arr in conds.items():
            if not (np.array_equal(arr["idx"], ref["idx"]) and np.array_equal(arr["answer"], ref["answer"])):
                raise AssertionError(f"{model}/{name}: example order or answers differ from baseline")
    return data, sources


def _se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def compute_metrics(conds: dict[str, dict]) -> dict[str, dict]:
    base = conds["original"]
    base_top = base["probs"].argmax(1)
    rows = np.arange(base_top.size)
    base_pnorm = base["probs"] / base["probs"].sum(1, keepdims=True)
    base_pcorrect = base_pnorm[rows, base["answer"]]
    out = {}
    for name, arr in conds.items():
        probs = arr["probs"]
        mass = probs.sum(1)
        pnorm = probs / mass[:, None]
        top = probs.argmax(1)
        hard = (top == arr["answer"]).astype(float)
        p_correct = pnorm[rows, arr["answer"]]
        agree = (top == base_top).astype(float)
        delta_h = arr["entropy"] - base["entropy"]
        delta_soft = p_correct - base_pcorrect
        out[name] = {
            "n": int(top.size),
            "mean_H": float(arr["entropy"].mean()),
            "median_H": float(np.median(arr["entropy"])),
            "q10_H": float(np.quantile(arr["entropy"], 0.1)),
            "q90_H": float(np.quantile(arr["entropy"], 0.9)),
            "median_delta_H_vs_baseline": float(np.median(delta_h)),
            "hard_acc": float(hard.mean()),
            "hard_acc_se": _se(hard),
            "soft_acc": float(p_correct.mean()),
            "soft_acc_se": _se(p_correct),
            "soft_acc_raw_unnormalised": float(probs[rows, arr["answer"]].mean()),
            "delta_soft_vs_baseline": float(delta_soft.mean()),
            "delta_soft_vs_baseline_se_paired": _se(delta_soft),
            "top1_agreement": float(agree.mean()),
            "top1_agreement_se": _se(agree),
            "mean_p_baseline_top1": float(pnorm[rows, base_top].mean()),
            "hard_acc_given_agree": float(hard[agree == 1].mean()),
            "soft_acc_given_agree": float(p_correct[agree == 1].mean()),
            "mass_mean": float(mass.mean()),
            "mass_min": float(mass.min()),
            "frac_mass_below_0.9": float(np.mean(mass < 0.9)),
        }
    return out


# ----------------------------------------------------------------------------- plot
def _violin_panel(ax: plt.Axes, conds: dict[str, dict], title: str) -> None:
    positions = np.arange(len(CONDITIONS))
    data = [conds[key]["entropy"] for key, *_ in CONDITIONS]
    parts = ax.violinplot(data, positions=positions, widths=0.78, showextrema=False)
    for body, (_, _, _, color) in zip(parts["bodies"], CONDITIONS):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)
    for pos, values, (_, _, _, color) in zip(positions, data, CONDITIONS):
        q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        ax.vlines(pos, q10, q90, color=color, linewidth=2.0, zorder=3)
        ax.scatter([pos], [q50], color=color, s=40, zorder=4)
        ax.scatter([pos], [float(np.mean(values))], color="white", edgecolor=color, linewidth=1.4, s=44, zorder=5)
        ax.text(pos + 0.09, q50, f"{q50:.2f}", ha="left", va="center", fontsize=10.5, color=INK, zorder=6)
    ax.axhline(2.0, color=REF_LINE_COLOR, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(positions[-1] + 0.45, 2.0, "uniform", ha="right", va="bottom", fontsize=9.5, color=MUTED_INK)
    ax.set_xticks(positions)
    ax.set_xticklabels([tick for _, _, tick, _ in CONDITIONS])
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.set_ylim(-0.04, 2.2)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def _group_offsets(n: int, spacing: float) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2) * spacing


def _accuracy_panel(ax: plt.Axes, metrics: dict[str, dict[str, dict]]) -> None:
    offsets = _group_offsets(len(CONDITIONS), 0.2)
    for g, model in enumerate(MODELS):
        for off, (key, _, _, color) in zip(offsets, CONDITIONS):
            m = metrics[model][key]
            x = g + off
            soft, hard = 100 * m["soft_acc"], 100 * m["hard_acc"]
            ax.plot([x, x], [soft, hard], color=color, linewidth=1.2, alpha=0.6, zorder=2)
            ax.errorbar(x, soft, yerr=196 * m["soft_acc_se"], fmt="o", color=color, markersize=7.5, capsize=0, zorder=4)
            ax.errorbar(x, hard, yerr=196 * m["hard_acc_se"], fmt="D", color=color, markerfacecolor="white",
                        markeredgewidth=1.4, markersize=6.5, capsize=0, zorder=4)
            if key != "original":
                ax.text(x, soft - 1.6, f"{100 * m['delta_soft_vs_baseline']:+.1f}".replace("-", "−"),
                        ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([display_model_name(m) for m in MODELS])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(44, 92)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Soft and hard accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=MUTED_INK, linestyle="", markersize=7.5, label="Soft accuracy"),
            Line2D([0], [0], marker="D", color=MUTED_INK, markerfacecolor="white", markeredgewidth=1.4,
                   linestyle="", markersize=6.5, label="Hard accuracy"),
        ],
        frameon=False,
        loc="lower left",
        handletextpad=0.3,
        borderaxespad=0.2,
    )


def _agreement_panel(ax: plt.Axes, metrics: dict[str, dict[str, dict]]) -> None:
    conds = [c for c in CONDITIONS if c[0] != "original"]
    offsets = _group_offsets(len(conds), 0.3)
    ax.axhline(100, color=REF_LINE_COLOR, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(-0.46, 100.6, "baseline", ha="left", va="bottom", fontsize=9.5, color=MUTED_INK)
    for g, model in enumerate(MODELS):
        for off, (key, _, _, color) in zip(offsets, conds):
            m = metrics[model][key]
            x, y = g + off, 100 * m["top1_agreement"]
            ax.errorbar(x, y, yerr=196 * m["top1_agreement_se"], fmt="o", color=color, markersize=7.5, capsize=0, zorder=4)
            ax.text(x, y - 1.8, f"{y:.1f}", ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([display_model_name(m) for m in MODELS])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(62, 104)
    ax.set_ylabel("Top-1 agreement (%)")
    ax.set_title("Top-1 agreement with baseline")
    ax.grid(axis="x", visible=False)


def build_figure(data: dict, metrics: dict) -> plt.Figure:
    apply_prelim_style()
    fig, axes = plt.subplots(
        1, 4, figsize=(FULL_WIDTH_IN, 4.1), gridspec_kw={"width_ratios": [1.25, 1.25, 1.05, 0.95]}, constrained_layout=True
    )
    _violin_panel(axes[0], data[MODELS[0]], f"{display_model_name(MODELS[0])}: answer entropy")
    _violin_panel(axes[1], data[MODELS[1]], f"{display_model_name(MODELS[1])}: answer entropy")
    axes[0].set_ylabel("Entropy H over A–D (bits)")
    _accuracy_panel(axes[2], metrics)
    _agreement_panel(axes[3], metrics)

    handles = [Patch(facecolor=color, alpha=0.6, edgecolor="none", label=label) for _, label, _, color in CONDITIONS]
    handles += [
        Line2D([0], [0], marker="o", color=MUTED_INK, linestyle="", markersize=6.5, label="Median (value shown)"),
        Line2D([0], [0], marker="o", color=MUTED_INK, markerfacecolor="white", linestyle="", markersize=6.5, label="Mean"),
        Line2D([0], [0], color=MUTED_INK, linewidth=2.0, label="10th–90th pct."),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
               handlelength=1.4, columnspacing=1.4)
    return fig


# ----------------------------------------------------------------------------- table
def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build_table(metrics: dict[str, dict[str, dict]]) -> str:
    lines = [
        "% Generated by figures/scripts/camera_ready/prelim_mmlu_prompt_conditions.py (do not edit by hand).",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Prompted sandbagging via uncertainty on MMLU (14{,}042 test questions).}",
        "$H$ is the entropy (bits) of the answer distribution renormalised over A--D. Hard accuracy is the",
        "accuracy of the top-1 answer; soft accuracy is the mean probability assigned to the correct answer",
        "(the accuracy expected under sampling); $\\Delta$ soft is the change in soft accuracy relative to the",
        "baseline in percentage points. Top-1 agreement is the fraction of questions on which the top-1 answer",
        "matches the baseline top-1 answer.}",
        "\\label{tab:mmlu-prompt-conditions}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{$H$ (bits)} & \\multicolumn{3}{c}{Accuracy (\\%)} & Top-1 \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6}",
        "Condition & Median & Mean & Hard & Soft & $\\Delta$ soft & agr.\\ (\\%) \\\\",
        "\\midrule",
    ]
    for i, model in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\texttt{{{display_model_name(model)}}}}} \\\\")
        for key, *_ in CONDITIONS:
            m = metrics[model][key]
            delta = "--" if key == "original" else f"${100 * m['delta_soft_vs_baseline']:+.1f}$"
            agree = "--" if key == "original" else _pct(m["top1_agreement"])
            lines.append(
                f"\\quad {TABLE_NAMES[key]} & {m['median_H']:.3f} & {m['mean_H']:.3f} & {_pct(m['hard_acc'])} & "
                f"{_pct(m['soft_acc'])} & {delta} & {agree} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    data, sources = load_conditions()
    metrics = {model: compute_metrics(conds) for model, conds in data.items()}
    fig = build_figure(data, metrics)
    paths = save(fig, STEM)
    plt.close(fig)
    table_path = write_table(build_table(metrics), STEM)
    summary_path = write_json(
        {
            "description": "R1 prompted sandbagging on MMLU; entropy in bits over A-D (renormalised), soft accuracy = mean renormalised P(correct)",
            "sources": sources,
            "metrics": metrics,
            "outputs": [rel(p) for p in [*paths, table_path]],
        },
        STEM,
    )
    for model in MODELS:
        print(f"== {display_model_name(model)}")
        for key, *_ in CONDITIONS:
            m = metrics[model][key]
            print(
                f"  {key:24s} medH={m['median_H']:.3f} meanH={m['mean_H']:.3f} hard={100*m['hard_acc']:.2f} "
                f"soft={100*m['soft_acc']:.2f} dsoft={100*m['delta_soft_vs_baseline']:+.2f} agree={100*m['top1_agreement']:.2f} "
                f"P(base top1)={100*m['mean_p_baseline_top1']:.2f}"
            )
    for p in [*paths, table_path, summary_path]:
        print(p)


if __name__ == "__main__":
    main()
