"""Shared-AB bootstrap: meaningful median ΔH vs control absolute-median floor.

For each model under results/neutral/prompt_variants/barebone/:

  1. Align all instruction pairs on common AB keys
     (option_a, option_b, flip_ab).
  2. Control floor  Cbar = aggregator_c( |median_i ΔH_c,i| )
     with aggregator in {median, mean, max} (default: median).
  3. For each meaningful pair m:
       T_m = median_i ΔH_m,i          (signed)
       E_m = T_m - Cbar               (excess over floor)
  4. Resample the SAME AB indices for every pair (shared bootstrap),
     recompute T*, Cbar*, E*, and form percentile CIs + one-sided
     bootstrap p-values for H0: T<=0 and H0: E<=0.
  5. BH-correct p-values across meaningful pairs within the model.
  6. Classify each pair as controlled / not_controlled.
  7. Heterogeneity contrast: Var(meaningful medians) - Var(control |medians|).

Writes:
  analysis_outputs/bootstrap_meaningful_vs_control_floor_{model}.json
  analysis_outputs/bootstrap_meaningful_vs_control_floor_summary.csv
  analysis_outputs/bootstrap_meaningful_vs_control_floor_pairs.csv

Run:
  AB_PAIRS_VARIANT=neutral .venv/bin/python \\
    figures/scripts/bootstrap_meaningful_vs_control_floor.py
  AB_PAIRS_VARIANT=neutral .venv/bin/python \\
    figures/scripts/bootstrap_meaningful_vs_control_floor.py --model qwen_Qwen3.5-27B
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    clean_probability_records,
    load_zero_shot_result,
    record_entropy,
)

ROOT = Path(__file__).resolve().parents[2]
BAREBONE_ROOT = ROOT / "results" / "neutral" / "prompt_variants" / "barebone"
OUTPUT_DIR = ROOT / "analysis_outputs"

MEANINGFUL_PAIRS: list[tuple[str, str, str]] = [
    ("Be sure!", "Be random!", "Be_sure_Be_random"),
    ("Maximize certainty", "Minimize certainty", "Maximize_certainty_Minimize_certainty"),
    ("Maximize certainty", "Maximize uncertainty", "Maximize_certainty_Maximize_uncertainty"),
    ("Commit to one option", "Keep both options equally likely", "Commit_to_one_option_Keep_both_options_equally_likely"),
    ("Reply with confidence", "Reply without confidence", "Reply_with_confidence_Reply_without_confidence"),
    ("Aim for low entropy", "Aim for high entropy", "Aim_for_low_entropy_Aim_for_high_entropy"),
    ("Be deterministic", "Assign equal probability to both options", "Be_deterministic_Assign_equal_probability_to_both_options"),
]

# Fixed orderings; absolute median makes direction irrelevant.
CONTROL_PAIRS: list[tuple[str, str, str]] = [
    ("Be a good boy!", "Be a bad boy!", "Be_a_good_boy_Be_a_bad_boy"),
    ("Be honest!", "Be dishonest!", "Be_honest_Be_dishonest"),
    ("Be banana!", "Be apple!", "Be_banana_Be_apple"),
    ("Be polite!", "Be impolite!", "Be_polite_Be_impolite"),
    ("Abracadabra!", "Hocus pocus!", "Abracadabra_Hocus_pocus"),
    ("Alakazam!", "Sim sala bim!", "Alakazam_Sim_sala_bim"),
    ("Bibbidi-bobbidi-boo!", "Wingardium leviosa!", "Bibbidi-bobbidi-boo_Wingardium_leviosa"),
    ("Ping!", "Pong!", "Ping_Pong"),
    ("Quack!", "Moo!", "Quack_Moo"),
    ("Tralala!", "Do re mi!", "Tralala_Do_re_mi"),
]

FloorAgg = Literal["median", "mean", "max"]
ALPHA = 0.05
CI_LEVEL = 0.95


def latest_results(model_dir: Path, slug: str) -> Path | None:
    matches = sorted(model_dir.glob(f"{slug}_*/results.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_diffs_by_key(
    model_dir: Path,
    cert: str,
    unc: str,
    slug: str,
) -> dict[tuple, float] | None:
    path = latest_results(model_dir, slug)
    if path is None:
        return None
    records = load_zero_shot_result(path)
    paired: dict[tuple, dict[str, float]] = {}
    for record in clean_probability_records(records):
        instruction = record.get("instruction")
        if instruction not in {cert, unc}:
            continue
        key = (record["option_a"], record["option_b"], bool(record.get("flip_ab", False)))
        paired.setdefault(key, {})[instruction] = record_entropy(record)
    out = {
        key: grouped[unc] - grouped[cert]
        for key, grouped in paired.items()
        if cert in grouped and unc in grouped
    }
    return out or None


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    total = len(pvalues)
    qvalues = [float("nan")] * total
    finite = [(i, p) for i, p in enumerate(pvalues) if np.isfinite(p)]
    if not finite:
        return qvalues
    descending = sorted(finite, key=lambda ip: ip[1], reverse=True)
    running_min = 1.0
    n = len(finite)
    for position, (index, p) in enumerate(descending):
        rank = n - position
        running_min = min(running_min, p * n / rank)
        qvalues[index] = float(min(1.0, running_min))
    return qvalues


def percentile_ci(samples: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def aggregate_floor(abs_medians: np.ndarray, how: FloorAgg) -> float:
    if abs_medians.size == 0:
        return float("nan")
    if how == "median":
        return float(np.median(abs_medians))
    if how == "mean":
        return float(np.mean(abs_medians))
    return float(np.max(abs_medians))


def short_pair(cert: str, unc: str) -> str:
    return f"{unc} vs {cert}"


def display_slug(model_slug: str) -> str:
    return (
        model_slug.replace("qwen_", "")
        .replace("meta-llama_", "")
        .replace("google_", "")
        .replace("allenai_", "")
        .replace("_", "/")
    )


def bootstrap_model(
    model_slug: str,
    *,
    n_boot: int,
    seed: int,
    floor_agg: FloorAgg,
    min_control_n: int = 100,
) -> dict[str, Any] | None:
    model_dir = BAREBONE_ROOT / model_slug
    if not model_dir.is_dir():
        return None

    meaningful_raw: list[tuple[str, str, str, dict[tuple, float]]] = []
    for cert, unc, slug in MEANINGFUL_PAIRS:
        diffs = load_diffs_by_key(model_dir, cert, unc, slug)
        if diffs is not None:
            meaningful_raw.append((cert, unc, slug, diffs))

    control_raw: list[tuple[str, str, str, dict[tuple, float]]] = []
    dropped_controls: list[dict[str, Any]] = []
    for cert, unc, slug in CONTROL_PAIRS:
        diffs = load_diffs_by_key(model_dir, cert, unc, slug)
        if diffs is None:
            dropped_controls.append({"slug": slug, "n_ab": 0, "reason": "missing"})
            continue
        if len(diffs) < min_control_n:
            # Sparse retention (e.g. many invalid A/B probs) would otherwise
            # shrink the shared AB key set for every pair.
            dropped_controls.append({"slug": slug, "n_ab": len(diffs), "reason": "below_min_control_n"})
            continue
        control_raw.append((cert, unc, slug, diffs))

    if not meaningful_raw or not control_raw:
        return None

    control_common = set.intersection(*(set(d) for *_, d in control_raw))
    # Shared analysis keys: present in every retained control and every meaningful pair.
    # (Meaningful pairs almost always cover the control-common set.)
    common_keys = set.intersection(control_common, *(set(d) for *_, d in meaningful_raw))
    keys = sorted(common_keys)
    n = len(keys)
    if n < 10:
        return None

    m_mat = np.column_stack([[d[k] for k in keys] for *_, d in meaningful_raw])  # (N, M)
    c_mat = np.column_stack([[d[k] for k in keys] for *_, d in control_raw])  # (N, C)
    m_labels = [short_pair(cert, unc) for cert, unc, _, _ in meaningful_raw]
    c_labels = [short_pair(cert, unc) for cert, unc, _, _ in control_raw]
    n_m = m_mat.shape[1]
    n_c = c_mat.shape[1]

    # Observed statistics on the shared AB set (same keys as the bootstrap).
    t_obs = np.median(m_mat, axis=0)
    c_abs_obs = np.abs(np.median(c_mat, axis=0))
    cbar_obs = aggregate_floor(c_abs_obs, floor_agg)
    e_obs = t_obs - cbar_obs

    rng = np.random.default_rng(seed)
    t_boot = np.empty((n_boot, n_m), dtype=float)
    c_abs_boot = np.empty((n_boot, n_c), dtype=float)
    cbar_boot = np.empty(n_boot, dtype=float)
    e_boot = np.empty((n_boot, n_m), dtype=float)
    var_m_boot = np.empty(n_boot, dtype=float)
    var_c_boot = np.empty(n_boot, dtype=float)

    idx = np.arange(n)
    for b in range(n_boot):
        I = rng.choice(idx, size=n, replace=True)
        t_star = np.median(m_mat[I], axis=0)
        c_abs_star = np.abs(np.median(c_mat[I], axis=0))
        cbar_star = aggregate_floor(c_abs_star, floor_agg)
        t_boot[b] = t_star
        c_abs_boot[b] = c_abs_star
        cbar_boot[b] = cbar_star
        e_boot[b] = t_star - cbar_star
        var_m_boot[b] = float(np.var(t_star, ddof=1)) if n_m > 1 else 0.0
        var_c_boot[b] = float(np.var(c_abs_star, ddof=1)) if n_c > 1 else 0.0

    # One-sided bootstrap p-values: fraction of replicates on the null side.
    p_direction = (1.0 + np.sum(t_boot <= 0.0, axis=0)) / (1.0 + n_boot)
    p_excess = (1.0 + np.sum(e_boot <= 0.0, axis=0)) / (1.0 + n_boot)
    q_direction = benjamini_hochberg(p_direction.tolist())
    q_excess = benjamini_hochberg(p_excess.tolist())

    pairs_out: list[dict[str, Any]] = []
    for j in range(n_m):
        t_lo, t_hi = percentile_ci(t_boot[:, j])
        e_lo, e_hi = percentile_ci(e_boot[:, j])
        direction_ok = t_lo > 0.0 and q_direction[j] < ALPHA
        excess_ok = e_lo > 0.0 and q_excess[j] < ALPHA
        verdict = "controlled" if direction_ok and excess_ok else "not_controlled"
        cert, unc, slug, _ = meaningful_raw[j]
        pairs_out.append(
            {
                "kind": "meaningful",
                "certainty_instruction": cert,
                "uncertainty_instruction": unc,
                "slug": slug,
                "label": m_labels[j],
                "n_ab": n,
                "T_median_dH": float(t_obs[j]),
                "T_ci_lo": t_lo,
                "T_ci_hi": t_hi,
                "Cbar": float(cbar_obs),
                "E_excess": float(e_obs[j]),
                "E_ci_lo": e_lo,
                "E_ci_hi": e_hi,
                "p_direction": float(p_direction[j]),
                "q_direction": float(q_direction[j]),
                "p_excess": float(p_excess[j]),
                "q_excess": float(q_excess[j]),
                "direction_ok": bool(direction_ok),
                "excess_ok": bool(excess_ok),
                "verdict": verdict,
            }
        )

    controls_out: list[dict[str, Any]] = []
    for j in range(n_c):
        lo, hi = percentile_ci(c_abs_boot[:, j])
        cert, unc, slug, _ = control_raw[j]
        controls_out.append(
            {
                "kind": "control",
                "certainty_instruction": cert,
                "uncertainty_instruction": unc,
                "slug": slug,
                "label": c_labels[j],
                "n_ab": n,
                "abs_median_dH": float(c_abs_obs[j]),
                "abs_median_ci_lo": lo,
                "abs_median_ci_hi": hi,
            }
        )

    var_m_obs = float(np.var(t_obs, ddof=1)) if n_m > 1 else 0.0
    var_c_obs = float(np.var(c_abs_obs, ddof=1)) if n_c > 1 else 0.0
    het_obs = var_m_obs - var_c_obs
    het_boot = var_m_boot - var_c_boot
    het_lo, het_hi = percentile_ci(het_boot)
    # One-sided: meaningful more dispersed than controls
    p_het = float((1.0 + np.sum(het_boot <= 0.0)) / (1.0 + n_boot))

    cbar_lo, cbar_hi = percentile_ci(cbar_boot)
    n_controlled = sum(1 for row in pairs_out if row["verdict"] == "controlled")

    return {
        "model_slug": model_slug,
        "model": display_slug(model_slug),
        "n_ab_common": n,
        "n_boot": n_boot,
        "seed": seed,
        "floor_agg": floor_agg,
        "min_control_n": min_control_n,
        "dropped_controls": dropped_controls,
        "alpha": ALPHA,
        "ci_level": CI_LEVEL,
        "Cbar": float(cbar_obs),
        "Cbar_ci_lo": cbar_lo,
        "Cbar_ci_hi": cbar_hi,
        "n_meaningful": n_m,
        "n_control": n_c,
        "n_controlled": n_controlled,
        "controlled_frac": f"{n_controlled}/{n_m}",
        "var_meaningful_medians": var_m_obs,
        "var_control_abs_medians": var_c_obs,
        "heterogeneity_diff": het_obs,
        "heterogeneity_ci_lo": het_lo,
        "heterogeneity_ci_hi": het_hi,
        "p_heterogeneity": p_het,
        "meaningful_pairs": pairs_out,
        "control_pairs": controls_out,
    }


def print_model_report(result: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"  {result['model']}  |  barebone  |  floor={result['floor_agg']}(|median ΔH|)")
    dropped = result.get("dropped_controls") or []
    if dropped:
        dropped_txt = ", ".join(f"{d['slug']}(n={d['n_ab']})" for d in dropped)
        print(f"  dropped controls: {dropped_txt}")
    print(
        f"  N_AB={result['n_ab_common']}  B={result['n_boot']}  "
        f"controls_used={result['n_control']}  "
        f"Cbar={result['Cbar']:.4f} "
        f"[{result['Cbar_ci_lo']:.4f}, {result['Cbar_ci_hi']:.4f}]  "
        f"controlled={result['controlled_frac']}"
    )
    print(
        f"  heterogeneity Var(m)-Var(c)={result['heterogeneity_diff']:.4f} "
        f"[{result['heterogeneity_ci_lo']:.4f}, {result['heterogeneity_ci_hi']:.4f}]  "
        f"p={result['p_heterogeneity']:.4g}"
    )
    print("-" * 88)
    header = (
        f"{'pair':<48} {'T':>7} {'T CI':>18} {'E':>7} {'E CI':>18} "
        f"{'q_dir':>7} {'q_exc':>7} {'verdict':<16}"
    )
    print(header)
    for row in result["meaningful_pairs"]:
        t_ci = f"[{row['T_ci_lo']:+.3f},{row['T_ci_hi']:+.3f}]"
        e_ci = f"[{row['E_ci_lo']:+.3f},{row['E_ci_hi']:+.3f}]"
        print(
            f"{row['label']:<48} "
            f"{row['T_median_dH']:+7.3f} {t_ci:>18} "
            f"{row['E_excess']:+7.3f} {e_ci:>18} "
            f"{row['q_direction']:7.3g} {row['q_excess']:7.3g} "
            f"{row['verdict']:<16}"
        )
    print()


def discover_model_slugs() -> list[str]:
    if not BAREBONE_ROOT.is_dir():
        return []
    return sorted(p.name for p in BAREBONE_ROOT.iterdir() if p.is_dir())


def write_outputs(results: list[dict[str, Any]], floor_agg: FloorAgg) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in results:
        path = OUTPUT_DIR / f"bootstrap_meaningful_vs_control_floor_{result['model_slug']}.json"
        path.write_text(json.dumps(result, indent=2))
        written.append(path)

    summary_path = OUTPUT_DIR / "bootstrap_meaningful_vs_control_floor_summary.csv"
    summary_fields = [
        "model",
        "model_slug",
        "floor_agg",
        "n_ab_common",
        "n_boot",
        "Cbar",
        "Cbar_ci_lo",
        "Cbar_ci_hi",
        "n_meaningful",
        "n_control",
        "n_controlled",
        "controlled_frac",
        "var_meaningful_medians",
        "var_control_abs_medians",
        "heterogeneity_diff",
        "heterogeneity_ci_lo",
        "heterogeneity_ci_hi",
        "p_heterogeneity",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k) for k in summary_fields})
    written.append(summary_path)

    pairs_path = OUTPUT_DIR / "bootstrap_meaningful_vs_control_floor_pairs.csv"
    pair_fields = [
        "model",
        "model_slug",
        "floor_agg",
        "kind",
        "label",
        "certainty_instruction",
        "uncertainty_instruction",
        "slug",
        "n_ab",
        "T_median_dH",
        "T_ci_lo",
        "T_ci_hi",
        "Cbar",
        "E_excess",
        "E_ci_lo",
        "E_ci_hi",
        "p_direction",
        "q_direction",
        "p_excess",
        "q_excess",
        "direction_ok",
        "excess_ok",
        "verdict",
        "abs_median_dH",
        "abs_median_ci_lo",
        "abs_median_ci_hi",
    ]
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for result in results:
            base = {
                "model": result["model"],
                "model_slug": result["model_slug"],
                "floor_agg": floor_agg,
            }
            for row in result["meaningful_pairs"]:
                writer.writerow({**base, **{k: row.get(k) for k in pair_fields if k in row}})
            for row in result["control_pairs"]:
                writer.writerow({**base, **{k: row.get(k) for k in pair_fields if k in row}})
    written.append(pairs_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model slug (e.g. qwen_Qwen3.5-27B). Repeatable. Default: all barebone models.",
    )
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--floor",
        choices=("median", "mean", "max"),
        default="median",
        help="Aggregator over control |median ΔH| values (default: median).",
    )
    parser.add_argument(
        "--min-control-n",
        type=int,
        default=100,
        help="Drop control pairs with fewer than this many valid AB items "
        "(avoids one sparse control shrinking the shared key set).",
    )
    return parser.parse_args()


def main() -> None:
    # Neutral barebone results live under results/neutral/.
    os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
    args = parse_args()
    model_slugs = args.model or discover_model_slugs()
    if not model_slugs:
        raise SystemExit(f"No model directories found under {BAREBONE_ROOT}")

    results: list[dict[str, Any]] = []
    for slug in model_slugs:
        print(f"Bootstrapping {slug} ...", flush=True)
        result = bootstrap_model(
            slug,
            n_boot=args.n_boot,
            seed=args.seed,
            floor_agg=args.floor,
            min_control_n=args.min_control_n,
        )
        if result is None:
            print(f"  skip (missing meaningful/control data or too few common AB keys)")
            continue
        print_model_report(result)
        results.append(result)

    if not results:
        raise SystemExit("No models produced bootstrap results.")

    written = write_outputs(results, args.floor)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
