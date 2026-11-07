"""Analysis B: standard meaningful vs barebone control floor (2026-07-20 neutral).

T_m from results/neutral/zero_shot_local (standard system prompt).
Cbar from results/neutral/prompt_variants/barebone control pairs.

Shared-AB bootstrap; see analysis_outputs/BOOTSTRAP_ENTROPY_CONTROL.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_meaningful_vs_control_floor import (  # noqa: E402
    ALPHA,
    CI_LEVEL,
    CONTROL_PAIRS,
    MEANINGFUL_PAIRS,
    aggregate_floor,
    benjamini_hochberg,
    display_slug,
    percentile_ci,
    short_pair,
)
from common import clean_probability_records, load_zero_shot_result, record_entropy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ZSL = ROOT / "results" / "neutral" / "zero_shot_local"
BAREBONE = ROOT / "results" / "neutral" / "prompt_variants" / "barebone"
OUTPUT_DIR = ROOT / "analysis_outputs"

DEFAULT_MODELS = [
    "qwen_Qwen3.5-27B",
    "qwen_Qwen3-32B",
    "meta-llama_Meta-Llama-3-70B-Instruct",
]


def latest(model_dir: Path, slug: str) -> Path | None:
    matches = sorted(model_dir.glob(f"{slug}_*/results.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_diffs(model_dir: Path, cert: str, unc: str, slug: str) -> dict[tuple, float] | None:
    path = latest(model_dir, slug)
    if path is None:
        return None
    paired: dict[tuple, dict[str, float]] = {}
    for record in clean_probability_records(load_zero_shot_result(path)):
        instr = record.get("instruction")
        if instr not in {cert, unc}:
            continue
        key = (record["option_a"], record["option_b"], bool(record.get("flip_ab", False)))
        paired.setdefault(key, {})[instr] = record_entropy(record)
    out = {k: g[unc] - g[cert] for k, g in paired.items() if cert in g and unc in g}
    return out or None


def leak_frac(model_dir: Path, cert: str, unc: str, slug: str) -> float:
    path = latest(model_dir, slug)
    if path is None:
        return 1.0
    by: dict[str, list[float]] = defaultdict(list)
    for r in load_zero_shot_result(path):
        if r.get("instruction") in {cert, unc}:
            by[r["instruction"]].append(float(r["option_a_prob"]) + float(r["option_b_prob"]))
    fracs = []
    for instr in (cert, unc):
        arr = np.asarray(by.get(instr, []), dtype=float)
        fracs.append(float((arr <= 0.9).mean()) if arr.size else 1.0)
    return max(fracs)


def run_model(
    model_slug: str,
    *,
    n_boot: int,
    seed: int,
    leak_thresh: float | None,
    min_control_n: int,
) -> dict[str, Any] | None:
    zsl_dir, bb_dir = ZSL / model_slug, BAREBONE / model_slug
    if not zsl_dir.is_dir() or not bb_dir.is_dir():
        return None

    meaningful_raw: list[tuple[str, str, str, dict[tuple, float], float]] = []
    dropped_m: list[dict[str, Any]] = []
    for cert, unc, slug in MEANINGFUL_PAIRS:
        leak = leak_frac(zsl_dir, cert, unc, slug)
        diffs = load_diffs(zsl_dir, cert, unc, slug)
        if diffs is None:
            dropped_m.append({"slug": slug, "reason": "missing", "leak": leak, "n_ab": 0})
            continue
        if leak_thresh is not None and leak >= leak_thresh:
            dropped_m.append({"slug": slug, "reason": "leak", "leak": leak, "n_ab": len(diffs)})
            continue
        meaningful_raw.append((cert, unc, slug, diffs, leak))

    control_raw: list[tuple[str, str, str, dict[tuple, float], float]] = []
    dropped_c: list[dict[str, Any]] = []
    for cert, unc, slug in CONTROL_PAIRS:
        leak = leak_frac(bb_dir, cert, unc, slug)
        diffs = load_diffs(bb_dir, cert, unc, slug)
        if diffs is None:
            dropped_c.append({"slug": slug, "reason": "missing", "leak": leak, "n_ab": 0})
            continue
        if len(diffs) < min_control_n:
            dropped_c.append({"slug": slug, "reason": "sparse", "leak": leak, "n_ab": len(diffs)})
            continue
        if leak_thresh is not None and leak >= leak_thresh:
            dropped_c.append({"slug": slug, "reason": "leak", "leak": leak, "n_ab": len(diffs)})
            continue
        control_raw.append((cert, unc, slug, diffs, leak))

    if not meaningful_raw or not control_raw:
        return None

    common = set.intersection(
        *(set(d) for *_, d, _ in meaningful_raw),
        *(set(d) for *_, d, _ in control_raw),
    )
    keys = sorted(common)
    n = len(keys)
    if n < 10:
        return None

    m_mat = np.column_stack([[d[k] for k in keys] for *_, d, _ in meaningful_raw])
    c_mat = np.column_stack([[d[k] for k in keys] for *_, d, _ in control_raw])
    n_m, n_c = m_mat.shape[1], c_mat.shape[1]

    t_obs = np.median(m_mat, axis=0)
    c_abs_obs = np.abs(np.median(c_mat, axis=0))
    cbar_obs = aggregate_floor(c_abs_obs, "median")
    e_obs = t_obs - cbar_obs

    rng = np.random.default_rng(seed)
    t_boot = np.empty((n_boot, n_m))
    e_boot = np.empty((n_boot, n_m))
    cbar_boot = np.empty(n_boot)
    var_m_boot = np.empty(n_boot)
    var_c_boot = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        I = rng.choice(idx, size=n, replace=True)
        t_star = np.median(m_mat[I], axis=0)
        c_abs_star = np.abs(np.median(c_mat[I], axis=0))
        cbar_star = aggregate_floor(c_abs_star, "median")
        t_boot[b] = t_star
        cbar_boot[b] = cbar_star
        e_boot[b] = t_star - cbar_star
        var_m_boot[b] = float(np.var(t_star, ddof=1)) if n_m > 1 else 0.0
        var_c_boot[b] = float(np.var(c_abs_star, ddof=1)) if n_c > 1 else 0.0

    p_dir = (1.0 + np.sum(t_boot <= 0.0, axis=0)) / (1.0 + n_boot)
    p_exc = (1.0 + np.sum(e_boot <= 0.0, axis=0)) / (1.0 + n_boot)
    q_dir = benjamini_hochberg(p_dir.tolist())
    q_exc = benjamini_hochberg(p_exc.tolist())

    pairs_out = []
    for j, (cert, unc, slug, _, leak) in enumerate(meaningful_raw):
        t_lo, t_hi = percentile_ci(t_boot[:, j])
        e_lo, e_hi = percentile_ci(e_boot[:, j])
        direction_ok = t_lo > 0.0 and q_dir[j] < ALPHA
        excess_ok = e_lo > 0.0 and q_exc[j] < ALPHA
        pairs_out.append(
            {
                "kind": "meaningful",
                "certainty_instruction": cert,
                "uncertainty_instruction": unc,
                "slug": slug,
                "label": short_pair(cert, unc),
                "leak_frac": leak,
                "n_ab": n,
                "T_median_dH": float(t_obs[j]),
                "T_ci_lo": t_lo,
                "T_ci_hi": t_hi,
                "Cbar": float(cbar_obs),
                "E_excess": float(e_obs[j]),
                "E_ci_lo": e_lo,
                "E_ci_hi": e_hi,
                "p_direction": float(p_dir[j]),
                "q_direction": float(q_dir[j]),
                "p_excess": float(p_exc[j]),
                "q_excess": float(q_exc[j]),
                "direction_ok": bool(direction_ok),
                "excess_ok": bool(excess_ok),
                "verdict": "controlled" if direction_ok and excess_ok else "not_controlled",
            }
        )

    var_m = float(np.var(t_obs, ddof=1)) if n_m > 1 else 0.0
    var_c = float(np.var(c_abs_obs, ddof=1)) if n_c > 1 else 0.0
    het_boot = var_m_boot - var_c_boot
    het_lo, het_hi = percentile_ci(het_boot)
    cbar_lo, cbar_hi = percentile_ci(cbar_boot)
    n_controlled = sum(1 for row in pairs_out if row["verdict"] == "controlled")

    return {
        "model_slug": model_slug,
        "model": display_slug(model_slug),
        "meaningful_source": "zero_shot_local / standard (neutral)",
        "control_source": "prompt_variants/barebone (neutral)",
        "leak_thresh": leak_thresh,
        "min_control_n": min_control_n,
        "n_ab_common": n,
        "n_boot": n_boot,
        "seed": seed,
        "floor_agg": "median",
        "alpha": ALPHA,
        "ci_level": CI_LEVEL,
        "Cbar": float(cbar_obs),
        "Cbar_ci_lo": cbar_lo,
        "Cbar_ci_hi": cbar_hi,
        "n_meaningful": n_m,
        "n_control": n_c,
        "n_controlled": n_controlled,
        "controlled_frac": f"{n_controlled}/{n_m}",
        "var_meaningful_medians": var_m,
        "var_control_abs_medians": var_c,
        "heterogeneity_diff": var_m - var_c,
        "heterogeneity_ci_lo": het_lo,
        "heterogeneity_ci_hi": het_hi,
        "p_heterogeneity": float((1.0 + np.sum(het_boot <= 0.0)) / (1.0 + n_boot)),
        "dropped_meaningful": dropped_m,
        "dropped_controls": dropped_c,
        "meaningful_pairs": pairs_out,
    }


def main() -> None:
    os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leak-thresh", type=float, default=0.10)
    parser.add_argument("--no-leak-filter", action="store_true")
    parser.add_argument("--min-control-n", type=int, default=100)
    args = parser.parse_args()

    leak_thresh = None if args.no_leak_filter else args.leak_thresh
    models = args.model or DEFAULT_MODELS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "all" if leak_thresh is None else f"noleak{int(leak_thresh * 100):02d}"

    for model in models:
        print(f"Bootstrapping {model} ...", flush=True)
        result = run_model(
            model,
            n_boot=args.n_boot,
            seed=args.seed,
            leak_thresh=leak_thresh,
            min_control_n=args.min_control_n,
        )
        if result is None:
            print("  skip")
            continue
        print(
            f"  {result['model']}: controlled={result['controlled_frac']} "
            f"Cbar={result['Cbar']:.4f} N_AB={result['n_ab_common']}"
        )
        for row in result["meaningful_pairs"]:
            print(
                f"    {row['verdict']:<16} T={row['T_median_dH']:+.3f} "
                f"E={row['E_excess']:+.3f}  {row['label']}"
            )
        path = OUTPUT_DIR / f"bootstrap_standardZSL_vs_bareboneControls_{tag}_{model}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
