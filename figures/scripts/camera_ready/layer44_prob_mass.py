#!/usr/bin/env python3
"""R4 (preliminary): steering keeps the probability mass on the allowed answer tokens.

For every item we take the raw (unnormalised) next-token probability mass on the allowed
answer tokens: P(option A) + P(option B) in the binary-choice task and P(A)+P(B)+P(C)+P(D)
on MMLU. Three steering experiments, all at the designated layer (56 for both Qwen models,
64 for Llama-3-70B):

  1. No-instruction dose response, strengths 1x, 2x, 3x (200 items per condition):
     results/neutral/no_instruction_control/<model>/no_instruction_layer<L>_strength<s>_*/results.json
  2. Cross-instruction steering ("Be sure!"/"Be random!" vector, test pair "Commit to one option" vs
     "Keep both options equally likely"), strengths 1x, 2x (400 items per run); strength 0 is the
     zero-shot run of the same test pair:
     results/neutral/steering_all_layers/<model>/vector_Be_sure_vs_Be_random_test_Commit_*_strength<s>_*/all_layers_results.json
  3. MMLU entropy steering at 2x for Qwen3.5-27B and Qwen3-32B (14,042 questions per condition):
     results/neutral/mmlu_entropy_steering/<model>/*/results.json

Sign convention. In code the vector is Be sure! - Be random! (certainty minus uncertainty), so
code "positive" steering lowers entropy. Everything here is labelled by the direction the applied
vector pushes entropy, in paper terms: entropy-increasing (+, along Be random! - Be sure!) and
entropy-decreasing (-). No-instruction: "negative" -> entropy-increasing, "positive" ->
entropy-decreasing. MMLU: "increase_entropy" (code strength -2) and "decrease_entropy" (+2).
Cross-instruction: the injector applies +s under the certainty instruction and -s under the
uncertainty instruction (s > 0 reinforces the prompted instruction, the paper's "positive steering");
items are regrouped by the sign actually applied, so each direction pools 200 items from each
instruction.

Probabilities were computed in half precision, so masses can slightly exceed 1. In the figure,
values of 1 - mass at or below 1e-4 (including negative values) are drawn in a band at the floor.

Run from the repo root (CPU only):
  AB_PAIRS_VARIANT=neutral python figures/scripts/camera_ready/prelim_prob_mass.py
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
    apply_prelim_style,
    display_model_name,
    rel,
    results_dir,
    save,
    unique_path,
    write_json,
    write_table,
)
from be_sure_or_random.experiments import latest_zero_shot_local_result  # noqa: E402

STEM = "layer44/prelim_prob_mass_layer44"
MODELS = [
    ("Qwen/Qwen3.5-27B", "Qwen_Qwen3.5-27B", 56),
    ("Qwen/Qwen3-32B", "Qwen_Qwen3-32B", 56),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama_Meta-Llama-3-70B-Instruct", 44),
]
MMLU_MODELS = MODELS[:2]
VEC_CERT, VEC_UNC = "Be sure!", "Be random!"
TEST_CERT, TEST_UNC = "Commit to one option", "Keep both options equally likely"
FLOOR = 1e-4
THRESHOLD = 0.9
DIRS = [("up", "Entropy-increasing steering (+)", H_UP_COLOR), ("down", "Entropy-decreasing steering (−)", H_DOWN_COLOR)]
SETTINGS = [
    ("no_instruction", "No instruction (dose response)", [0.0, 1.0, 2.0, 3.0]),
    ("cross_instruction", "Cross-instruction steering", [0.0, 1.0, 2.0]),
    ("mmlu", "MMLU (mass on A–D)", [0.0, 2.0]),
]


# ----------------------------------------------------------------------------- data
def _pair_mass(records: list[dict]) -> np.ndarray:
    out = []
    for r in records:
        if r.get("error"):
            raise ValueError(f"record with error: {r.get('error')}")
        out.append(float(r["option_a_prob"]) + float(r["option_b_prob"]))
    return np.asarray(out, dtype=float)


def load_no_instruction(root: Path) -> tuple[dict, dict]:
    data, sources = {}, {}
    for model, slug, layer in MODELS:
        none_ref = None
        for s in (1.0, 2.0, 3.0):
            path = unique_path(root / "no_instruction_control" / slug, f"no_instruction_layer{layer}_strength{s}_*/results.json")
            payload = json.loads(path.read_text())
            meta = payload["metadata"]
            assert meta["fixed_layer"] == layer and float(meta["steering_strength"]) == s
            assert meta["vector_certainty_instruction"] == VEC_CERT and meta["vector_uncertainty_instruction"] == VEC_UNC
            res = payload["results"]
            assert all(float(r["effective_strength"]) == s for r in res["positive"])
            assert all(float(r["effective_strength"]) == -s for r in res["negative"])
            none = _pair_mass(res["none"])
            if none_ref is None:
                none_ref = none
                data[(model, 0.0, "none")] = none
            elif not np.array_equal(none, none_ref):
                raise AssertionError(f"{model}: no-steering condition differs across strength runs")
            data[(model, s, "up")] = _pair_mass(res["negative"])
            data[(model, s, "down")] = _pair_mass(res["positive"])
            sources[f"{model} {s:g}x"] = rel(path)
    return data, sources


def load_cross_instruction(root: Path) -> tuple[dict, dict]:
    data, sources = {}, {}
    for model, slug, layer in MODELS:
        grouped: dict[tuple[float, str], list[float]] = {}
        for s in (-2.0, -1.0, 1.0, 2.0):
            path = unique_path(
                root / "steering_all_layers" / slug,
                f"vector_Be_sure_vs_Be_random_test_Commit_to_one_option_vs_Keep_both_options_equally_likely_all_layers_strength{s}_*/all_layers_results.json",
            )
            payload = json.loads(path.read_text())
            meta = payload["metadata"]
            assert float(meta["steering_strength"]) == s
            assert meta["test_certainty_instruction"] == TEST_CERT and meta["test_uncertainty_instruction"] == TEST_UNC
            records = payload["results"][f"layer_{layer}"]
            assert len(records) == 400
            for r in records:
                applied = s if r["instruction"] == TEST_CERT else -s  # code sign: >0 adds certainty direction
                key = (abs(s), "down" if applied > 0 else "up")
                grouped.setdefault(key, []).append(float(r["option_a_prob"]) + float(r["option_b_prob"]))
            sources[f"{model} {s:+g}x"] = rel(path)
        for (s, direction), values in grouped.items():
            data[(model, s, direction)] = np.asarray(values, dtype=float)
        zs = None
        for candidate in (model, model.split("/")[0].lower() + "/" + model.split("/")[1]):
            zs = latest_zero_shot_local_result(candidate, TEST_CERT, TEST_UNC)
            if zs is not None:
                break
        if zs is None:
            raise FileNotFoundError(f"no zero-shot run for {model} {TEST_CERT} / {TEST_UNC}")
        data[(model, 0.0, "none")] = _pair_mass(json.loads(Path(zs).read_text()))
        sources[f"{model} 0x (zero-shot)"] = rel(Path(zs))
    return data, sources


def load_mmlu(root: Path) -> tuple[dict, dict]:
    data, sources = {}, {}
    for model, slug, layer in MMLU_MODELS:
        path = unique_path(root / "mmlu_entropy_steering" / slug, f"cais_mmlu_all_test_maxall_layer{layer}_strength2.0_vecBe_sure_format_instructed_*/results.json")
        payload = json.loads(path.read_text())
        meta = payload["metadata"]
        assert meta["condition_strengths"] == {"none": 0.0, "increase_entropy": -2.0, "decrease_entropy": 2.0}
        for cond, key in (("none", (model, 0.0, "none")), ("increase_entropy", (model, 2.0, "up")), ("decrease_entropy", (model, 2.0, "down"))):
            data[key] = np.asarray([sum(float(r[f"prob_{x}"]) for x in "ABCD") for r in payload["results"][cond]], dtype=float)
            assert data[key].size == 14042
        sources[model] = rel(path)
    return data, sources


def describe(mass: np.ndarray) -> dict:
    return {
        "n": int(mass.size),
        "mean": float(mass.mean()),
        "median": float(np.median(mass)),
        "min": float(mass.min()),
        "max": float(mass.max()),
        "frac_below_0.99": float(np.mean(mass < 0.99)),
        "frac_below_0.9": float(np.mean(mass < THRESHOLD)),
        "count_below_0.9": int(np.sum(mass < THRESHOLD)),
        "frac_above_1": float(np.mean(mass > 1.0)),
    }


# ----------------------------------------------------------------------------- plot
def _strip(ax, x: float, mass: np.ndarray, color: str, rng: np.random.Generator, width: float) -> None:
    gap = 1.0 - mass
    y = np.where(gap > FLOOR, gap, 0.0)
    at_floor = y == 0.0
    # Items at or below the floor are spread inside a band just below 1e-4.
    y[at_floor] = 10 ** rng.uniform(np.log10(FLOOR) - 0.42, np.log10(FLOOR) - 0.08, size=int(at_floor.sum()))
    xs = x + rng.uniform(-width, width, size=mass.size)
    big = mass.size > 2000
    ax.scatter(xs, y, s=3 if big else 9, color=color, alpha=0.12 if big else 0.55, linewidths=0, rasterized=True, zorder=3)


def _panel(ax, data: dict, models: list, strengths: list[float], title: str, rng: np.random.Generator) -> None:
    group_w = len(strengths)
    step = group_w + 0.6
    centers, ticks = [], []
    for g, (model, _, _) in enumerate(models):
        base = g * step
        for j, s in enumerate(strengths):
            x = base + j
            ticks.append(x)
            if s == 0:
                groups = [(x, data[(model, 0.0, "none")], NONE_COLOR)]
            else:
                groups = [(x - 0.2, data[(model, s, "up")], H_UP_COLOR), (x + 0.2, data[(model, s, "down")], H_DOWN_COLOR)]
            for gx, mass, color in groups:
                _strip(ax, gx, mass, color, rng, width=0.15 if s else 0.2)
                count = int(np.sum(mass < THRESHOLD))
                if count:
                    ax.text(gx, 0.62, f"{count}/{mass.size}", ha="center", va="bottom", fontsize=9.5, color=INK, zorder=6)
        centers.append(base + (group_w - 1) / 2)
        if g:
            ax.axvline(base - 0.8, color="#E5E7EB", linewidth=1.0, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([("0" if s == 0 else f"{s:g}×") for _ in models for s in strengths])
    ax.tick_params(axis="x", length=0)
    sec = ax.secondary_xaxis(-0.105)
    sec.set_xticks(centers, labels=[display_model_name(m) for m, _, _ in models])
    sec.tick_params(axis="x", length=0)
    sec.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, (len(models) - 1) * step + group_w - 0.4)
    ax.axhline(1.0 - THRESHOLD, color=INK, linewidth=1.0, linestyle="--", zorder=4)
    ax.axhspan(10 ** (np.log10(FLOOR) - 0.47), 10 ** (np.log10(FLOOR) - 0.03), color="#F3F4F6", zorder=0, linewidth=0)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def build_figure(data_by_setting: dict) -> plt.Figure:
    apply_prelim_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_WIDTH_IN, 4.2), sharey=True,
        gridspec_kw={"width_ratios": [13.2, 10.2, 5.0]}, constrained_layout=True,
    )
    for ax, (key, title, strengths) in zip(axes, SETTINGS):
        models = MMLU_MODELS if key == "mmlu" else MODELS
        _panel(ax, data_by_setting[key], models, strengths, title, rng)
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_ylim(10 ** (np.log10(FLOOR) - 0.5), 1.0)
    ax.set_yticks([10 ** (np.log10(FLOOR) - 0.25), 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["\u2264$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("1 − option-token mass")
    ax.text(-0.45, 1.0 - THRESHOLD, "mass = 0.9", ha="left", va="bottom", fontsize=9.5, color=MUTED_INK)

    handles = [Patch(facecolor=NONE_COLOR, label="No steering")]
    handles += [Patch(facecolor=color, label=label) for _, label, color in DIRS]
    handles += [Line2D([0], [0], color=INK, linestyle="--", linewidth=1.0, label="Mass = 0.9")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    return fig


# ----------------------------------------------------------------------------- table
def _pct_below(stats: dict) -> str:
    pct = 100 * stats["frac_below_0.9"]
    if stats["count_below_0.9"] == 0:
        return "0.0"
    return f"{pct:.2f}" if pct < 0.1 else f"{pct:.1f}"


def _fmt_row(stats: dict) -> str:
    return f"{stats['mean']:.4f} & {stats['min']:.4f} & {_pct_below(stats)}"


def build_table(stats: dict) -> str:
    lines = [
        "% Generated by figures/scripts/camera_ready/prelim_prob_mass.py (do not edit by hand).",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Probability mass on the allowed answer tokens under steering.}",
        "Mass is the raw next-token probability of the two options (binary choice) or of A--D (MMLU).",
        "Entropy-increasing ($+$) steering adds the ``\\texttt{Be random!}''~$-$~``\\texttt{Be sure!}'' vector and",
        "entropy-decreasing ($-$) steering subtracts it, at the designated layer (56 for the Qwen models, 44 for",
        "\\texttt{Llama-3-70B}). Cross-instruction items are grouped by the direction of the applied vector,",
        "pooling both test instructions. Probabilities were computed in half precision, so masses can",
        "slightly exceed 1.}",
        "\\label{tab:prob-mass}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Entropy-increasing ($+$)} & \\multicolumn{3}{c}{Entropy-decreasing ($-$)} \\\\",
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}",
        "Model & Strength & $n$ & Mean & Min & \\% $<0.9$ & Mean & Min & \\% $<0.9$ \\\\",
    ]
    for key, title, strengths in SETTINGS:
        models = MMLU_MODELS if key == "mmlu" else MODELS
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title.replace(chr(8211), '--')}}}}} \\\\")
        for i, (model, _, _) in enumerate(models):
            name = f"\\texttt{{{display_model_name(model)}}}"
            for j, s in enumerate(strengths):
                first = name if j == 0 else ""
                if s == 0:
                    st = stats[key][(model, 0.0, "none")]
                    lines.append(
                        f"{first} & none & {st['n']} & \\multicolumn{{6}}{{c}}{{mean {st['mean']:.4f}, min {st['min']:.4f}, "
                        f"{_pct_below(st)}\\% $<0.9$}} \\\\"
                    )
                else:
                    up, down = stats[key][(model, s, "up")], stats[key][(model, s, "down")]
                    assert up["n"] == down["n"]
                    lines.append(f"{first} & ${s:g}\\times$ & {up['n']} & {_fmt_row(up)} & {_fmt_row(down)} \\\\")
            if i < len(models) - 1:
                lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    root = results_dir()
    no_inst, src_a = load_no_instruction(root)
    cross, src_b = load_cross_instruction(root)
    mmlu, src_c = load_mmlu(root)
    data_by_setting = {"no_instruction": no_inst, "cross_instruction": cross, "mmlu": mmlu}
    stats = {key: {k: describe(v) for k, v in d.items()} for key, d in data_by_setting.items()}

    fig = build_figure(data_by_setting)
    paths = save(fig, STEM)
    plt.close(fig)
    table_path = write_table(build_table(stats), STEM)
    summary = {
        "description": "Raw probability mass on allowed answer tokens; 'up' = entropy-increasing (paper +), 'down' = entropy-decreasing (paper -)",
        "sources": {"no_instruction": src_a, "cross_instruction": src_b, "mmlu": src_c},
        "stats": {key: {f"{m} | {s:g}x | {d}": v for (m, s, d), v in st.items()} for key, st in stats.items()},
        "outputs": [rel(p) for p in [*paths, table_path]],
    }
    summary_path = write_json(summary, STEM)
    for key, st in stats.items():
        print(f"== {key}")
        for (m, s, d), v in sorted(st.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            print(f"  {display_model_name(m):12s} {s:g}x {d:5s} n={v['n']:5d} mean={v['mean']:.5f} min={v['min']:.4f} "
                  f"<0.99={100 * v['frac_below_0.99']:.2f}% <0.9={100 * v['frac_below_0.9']:.2f}% ({v['count_below_0.9']})")
    for p in [*paths, table_path, summary_path]:
        print(p)


if __name__ == "__main__":
    main()
