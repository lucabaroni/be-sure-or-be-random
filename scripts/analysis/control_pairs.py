"""Meaningful versus control instruction pairs under the minimal prompt.

Paper: Sec. 4.1 summary paragraph ("15 of the 21 combinations ... exceed the mean control shift and 12
exceed every control pair (17 and 15 ... under the standard prompt)") and the first paragraph of the
appendix "Control instruction pairs" with its footnote (permutation test p = 0.002; p = 0.10 for the two
Qwen models alone).

For each model and instruction pair the score is the median Delta H over all items valid under both
instructions of that pair (common.paired_entropy_differences); pairs with fewer than MIN_ITEMS such items
are dropped. Control pairs are scored by the absolute value of their median Delta H. Meaningful pairs are
taken under the minimal (barebone) prompt and, for the standard-prompt counts, under the standard prompt;
both are compared with the minimal-prompt controls of the same model.

Permutation test: within each model the meaningful and control labels are shuffled; the statistic is the
mean over models of (mean meaningful score - mean control score); one-sided, Monte Carlo with
numpy.random.default_rng(SEED), p = (count + 1) / (B + 1).

Data: results/neutral/prompt_variants/barebone/<model>/ and results/neutral/zero_shot_local/<model>/,
latest run per pair by directory name.
Output: analysis_outputs/control_pairs.json

Run from the repository root:
  AB_PAIRS_VARIANT=neutral python scripts/analysis/control_pairs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _analysis_common import ANALYSIS_DIR, get_results_dir, latest_run, slugify_model_name

import sweep_models as sm  # noqa: E402
from common import load_zero_shot_result, paired_entropy_differences  # noqa: E402

MODELS = ["meta-llama/Meta-Llama-3-70B-Instruct", "qwen/Qwen3.5-27B", "qwen/Qwen3-32B"]
LABELS = {"meta-llama/Meta-Llama-3-70B-Instruct": "Llama-3-70B", "qwen/Qwen3.5-27B": "Qwen3.5-27B", "qwen/Qwen3-32B": "Qwen3-32B"}
CONTROL_PAIRS: list[tuple[str, str]] = [
    ("Be a good boy!", "Be a bad boy!"),
    ("Be honest!", "Be dishonest!"),
    ("Be banana!", "Be apple!"),
    ("Be polite!", "Be impolite!"),
    ("Abracadabra!", "Hocus pocus!"),
    ("Alakazam!", "Sim sala bim!"),
    ("Bibbidi-bobbidi-boo!", "Wingardium leviosa!"),
    ("Ping!", "Pong!"),
    ("Quack!", "Moo!"),
    ("Tralala!", "Do re mi!"),
]
MIN_ITEMS = 20
SEED = 0
# (model subset, Monte Carlo resamples) as used for the paper's footnote
PERMUTATION_TESTS = {
    "all_three_models": (MODELS, 200_000),
    "qwen_models_only": (["qwen/Qwen3.5-27B", "qwen/Qwen3-32B"], 100_000),
}


def pair_score(model_dir: Path, cert: str, unc: str) -> dict:
    path = latest_run(model_dir, cert, unc)
    if path is None:
        raise FileNotFoundError(f"no run for {cert} vs {unc} under {model_dir}")
    diffs = paired_entropy_differences(load_zero_shot_result(path), cert, unc)
    median = float(np.median(diffs)) if len(diffs) >= MIN_ITEMS else None
    return {"certainty": cert, "uncertainty": unc, "median_delta_h": median, "n_items": len(diffs),
            "run": str(path.parent.relative_to(get_results_dir()))}


def model_scores(model: str) -> dict:
    results = get_results_dir()
    slug = slugify_model_name(model)
    minimal = [pair_score(results / "prompt_variants" / "barebone" / slug, c, u) for c, u in sm.PAIRS]
    standard = [pair_score(results / "zero_shot_local" / slug, c, u) for c, u in sm.PAIRS]
    controls = [pair_score(results / "prompt_variants" / "barebone" / slug, c, u) for c, u in CONTROL_PAIRS]
    for group in (minimal, standard):
        missing = [p["certainty"] for p in group if p["median_delta_h"] is None]
        assert not missing, f"{model}: meaningful pairs with fewer than {MIN_ITEMS} valid items: {missing}"
    ctrl = np.array([abs(p["median_delta_h"]) for p in controls if p["median_delta_h"] is not None])
    out = {"minimal_pairs": minimal, "standard_pairs": standard, "control_pairs": controls,
           "n_controls_scored": int(len(ctrl)),
           "control_abs_mean": float(ctrl.mean()), "control_abs_median": float(np.median(ctrl)),
           "control_abs_max": float(ctrl.max())}
    for prompt, group in (("minimal", minimal), ("standard", standard)):
        vals = np.array([p["median_delta_h"] for p in group])
        out[f"{prompt}_meaningful_mean"] = float(vals.mean())
        out[f"{prompt}_above_control_mean"] = int((vals > ctrl.mean()).sum())
        out[f"{prompt}_above_every_control"] = int((vals > ctrl.max()).sum())
    return out


def stratified_permutation_test(groups: list[tuple[np.ndarray, np.ndarray]], n_resamples: int, seed: int) -> dict:
    """One-sided test that meaningful scores exceed control scores, labels shuffled within each model."""
    def stat(gs):
        return float(np.mean([m.mean() - c.mean() for m, c in gs]))

    rng = np.random.default_rng(seed)
    observed = stat(groups)
    count = 0
    for _ in range(n_resamples):
        perm = []
        for meaningful, control in groups:
            pooled = rng.permutation(np.concatenate([meaningful, control]))
            perm.append((pooled[:len(meaningful)], pooled[len(meaningful):]))
        count += stat(perm) >= observed - 1e-12
    return {"observed_statistic": observed, "n_resamples": n_resamples, "seed": seed,
            "count_at_least_observed": int(count), "pvalue": (count + 1) / (n_resamples + 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ANALYSIS_DIR / "control_pairs.json")
    parser.add_argument("--skip-permutation", action="store_true", help="only compute the counts and means")
    args = parser.parse_args()

    per_model = {m: model_scores(m) for m in MODELS}
    totals = {k: sum(r[k] for r in per_model.values()) for k in
              ("minimal_above_control_mean", "minimal_above_every_control",
               "standard_above_control_mean", "standard_above_every_control")}
    totals["n_combinations"] = len(MODELS) * len(sm.PAIRS)

    print(f"{'model':14s} {'meaningful':>10s} {'control':>8s} {'ctrl max':>8s} {'n ctrl':>6s} | "
          f"minimal: >mean >all | standard: >mean >all")
    for m, r in per_model.items():
        print(f"{LABELS[m]:14s} {r['minimal_meaningful_mean']:10.3f} {r['control_abs_mean']:8.3f} {r['control_abs_max']:8.3f} "
              f"{r['n_controls_scored']:6d} | {r['minimal_above_control_mean']:13d} {r['minimal_above_every_control']:4d} | "
              f"{r['standard_above_control_mean']:14d} {r['standard_above_every_control']:4d}")
        dropped = [p["certainty"] + " / " + p["uncertainty"] for p in r["control_pairs"] if p["median_delta_h"] is None]
        if dropped:
            print(f"{'':14s} control pairs with fewer than {MIN_ITEMS} valid items (not scored): {dropped}")
    print(f"minimal prompt: {totals['minimal_above_control_mean']} of {totals['n_combinations']} exceed the mean control shift, "
          f"{totals['minimal_above_every_control']} exceed every control pair")
    print(f"standard prompt: {totals['standard_above_control_mean']} and {totals['standard_above_every_control']} "
          f"of {totals['n_combinations']}")

    tests = {}
    if not args.skip_permutation:
        for name, (models, n_resamples) in PERMUTATION_TESTS.items():
            groups = [(np.array([p["median_delta_h"] for p in per_model[m]["minimal_pairs"]]),
                       np.array([abs(p["median_delta_h"]) for p in per_model[m]["control_pairs"] if p["median_delta_h"] is not None]))
                      for m in models]
            tests[name] = {"models": models, **stratified_permutation_test(groups, n_resamples, SEED)}
            t = tests[name]
            print(f"permutation test, {name}: observed {t['observed_statistic']:.3f}, p = {t['pvalue']:.5f} "
                  f"(B = {n_resamples}, seed {SEED})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"min_items": MIN_ITEMS, "totals": totals, "permutation_tests": tests,
                                    "per_model": per_model}, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
