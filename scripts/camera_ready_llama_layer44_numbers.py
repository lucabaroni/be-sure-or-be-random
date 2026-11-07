#!/usr/bin/env python3
"""Numbers for analysis_outputs/llama_layer44_report.md: Llama-3-70B steering at layer 44 vs 64.

CPU only. Reads existing results and vectors; writes
  analysis_outputs/llama_layer44/numbers.json       every number in the report, with its source file
  analysis_outputs/llama_layer44/*.csv              the report tables
  results/neutral/steering_vector_similarity/llama_layer44_designated_{standard,barebone}/
      similarity_matrices.json + <model>.csv        Llama-3-70B layer-44 matrices (same format as
                                                    run_steering_vector_similarity.py; the layer-64
                                                    designated_* directories are not touched)

Sections
  1. No-instruction dose response (vector "Be sure!"/"Be random!", strengths 1-3):
     results/neutral/no_instruction_control/<model>/no_instruction_layer{44,64}_strength<s>_*/results.json.
     Entropy statistics use valid records (P(A)+P(B) > 0.9), as in the paper figure.
  2. Probability mass P(A)+P(B) on the two option tokens, all 200 records per condition.
  3. Instruction-present sweep (existing): results/neutral/steering_all_layers/<model>/
     vector_Be_sure_vs_Be_random_test_<pair>_all_layers_strength<s>_*/all_layers_results.json at
     layers 44 and 64; strength 0 is the zero-shot run of the test pair (as in the paper figure).
     Delta H = H(uncertainty instruction) - H(certainty instruction), paired by item.
  4. Steering-vector cosine similarity (7 pairs): standard prompt from the per-layer vectors
     (artifacts/neutral/steering_layers/steering_vectors/<model>/<pair>/layer_44.pt), minimal prompt
     from the designated-script vectors (.../steering_vectors/barebone_designated_layer44/...); layer 64
     from the stored paper matrices (results/neutral/steering_vector_similarity/designated_*).

Sign convention (paper terms). Code vectors are mean(Be sure!) - mean(Be random!). No-instruction
runs: code condition "positive" adds +s*v and lowers entropy (paper: negative steering, blue,
"entropy-decreasing"); code "negative" raises entropy (paper: positive steering, red,
"entropy-increasing"). Instruction-present sweep: the injector adds +s*v under the certainty
instruction and -s*v under the uncertainty instruction, so s > 0 reinforces the prompted instruction
(paper strength = code strength) and s < 0 counteracts it.

Run from the repo root:
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python scripts/camera_ready_llama_layer44_numbers.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures" / "scripts"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

import numpy as np  # noqa: E402

import be_sure_or_random  # noqa: E402

# The shared venv also has an editable install of the main worktree (pre-fix results); refuse it.
assert Path(be_sure_or_random.__file__).resolve().is_relative_to(ROOT), be_sure_or_random.__file__

from be_sure_or_random.experiments import latest_zero_shot_local_result  # noqa: E402
from be_sure_or_random.utils import get_results_dir, slugify_model_name  # noqa: E402
from common import (  # noqa: E402
    clean_probability_records,
    is_valid_probability_record,
    paired_entropy_differences,
    record_entropy,
)
from run_steering_vector_similarity import (  # noqa: E402
    compute_model_matrix,
    find_manifest_group,
    load_manifest,
    matrix_from_group,
    write_csv,
)

MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
SLUG = slugify_model_name(MODEL)
LAYERS = (44, 64)
NEW_LAYER, OLD_LAYER = 44, 64
STRENGTHS = (1.0, 2.0, 3.0)
SWEEP_STRENGTHS = (-2.0, -1.0, 1.0, 2.0)
MIN_DATE = 20260720  # post-fix runs only
RESULTS = get_results_dir()
assert RESULTS == ROOT / "results" / "neutral", RESULTS
OUT = ROOT / "analysis_outputs" / "llama_layer44"
SIM_OUT = RESULTS / "steering_vector_similarity"
BAREBONE_L44_TEMPLATE = "barebone_designated_layer44"  # directory under steering_vectors/
TEST_PAIRS = {
    "cross": ("Commit to one option", "Keep both options equally likely"),
    "same": ("Be sure!", "Be random!"),
}
# code condition -> paper label
DIRECTIONS = {"positive": "entropy_decreasing", "negative": "entropy_increasing"}
_DATE = re.compile(r"_(\d{8})_(\d{6})$")


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def unique_post_fix(parent: Path, prefix: str, filename: str) -> Path:
    """The single post-fix run directory parent/<prefix>_<date>_<time>/filename."""
    hits = []
    for d in parent.glob(f"{prefix}_*"):
        m = _DATE.search(d.name)
        if m and d.name[: m.start()] == prefix and int(m.group(1)) >= MIN_DATE and (d / filename).exists():
            hits.append(d / filename)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one post-fix {prefix}_* under {parent}, found {hits}")
    return hits[0]


def r4(x: float) -> float:
    return round(float(x), 4)


def entropy_stats(records: list[dict]) -> dict:
    valid = clean_probability_records(records)
    h = np.array([record_entropy(r) for r in valid])
    return {"n": len(records), "n_valid": len(valid), "mean_H": r4(h.mean()), "median_H": r4(np.median(h))}


def mass_stats(records: list[dict]) -> dict:
    m = np.array([float(r["option_a_prob"]) + float(r["option_b_prob"]) for r in records])
    return {
        "n": int(m.size),
        "mean": r4(m.mean()),
        "min": r4(m.min()),
        "frac_below_0.9": r4((m < 0.9).mean()),
        "n_below_0.9": int((m < 0.9).sum()),
    }


def key(r: dict) -> tuple:
    return (r["option_a"], r["option_b"], bool(r["flip_ab"]))


def monotone(values: list[float], increasing: bool) -> bool:
    pairs = list(zip(values, values[1:]))
    return all(b > a for a, b in pairs) if increasing else all(b < a for a, b in pairs)


# ----------------------------------------------------------------------------- 1 + 2
def dose_response(layer: int) -> dict:
    parent = RESULTS / "no_instruction_control" / SLUG
    out = {"sources": {}, "entropy": {}, "mass": {}, "paired_dH_vs_none": {}}
    none_ref = None
    for s in STRENGTHS:
        path = unique_post_fix(parent, f"no_instruction_layer{layer}_strength{s}", "results.json")
        payload = json.loads(path.read_text())
        meta = payload["metadata"]
        assert meta["fixed_layer"] == layer and float(meta["steering_strength"]) == s
        assert (meta["vector_certainty_instruction"], meta["vector_uncertainty_instruction"]) == TEST_PAIRS["same"]
        assert not meta["normalize_steering_vector"] and not meta["preserve_norm"]
        res = payload["results"]
        assert all(float(r["effective_strength"]) == s for r in res["positive"])
        assert all(float(r["effective_strength"]) == -s for r in res["negative"])
        out["sources"][f"{s:g}x"] = rel(path)
        probs = [(float(r["option_a_prob"]), float(r["option_b_prob"])) for r in res["none"]]
        if none_ref is None:
            none_ref = probs
            out["entropy"]["0x"] = {"none": entropy_stats(res["none"])}
            out["mass"]["0x"] = {"none": mass_stats(res["none"])}
        else:
            assert probs == none_ref, f"layer {layer}: no-steering condition differs across strength runs"
        none_h = {key(r): record_entropy(r) for r in clean_probability_records(res["none"])}
        out["entropy"][f"{s:g}x"], out["mass"][f"{s:g}x"], out["paired_dH_vs_none"][f"{s:g}x"] = {}, {}, {}
        for code, paper in DIRECTIONS.items():
            out["entropy"][f"{s:g}x"][paper] = entropy_stats(res[code])
            out["mass"][f"{s:g}x"][paper] = mass_stats(res[code])
            d = [record_entropy(r) - none_h[key(r)] for r in clean_probability_records(res[code]) if key(r) in none_h]
            out["paired_dH_vs_none"][f"{s:g}x"][paper] = {"n": len(d), "mean": r4(np.mean(d)), "median": r4(np.median(d))}
    out["monotonic"] = {}
    for paper in DIRECTIONS.values():
        for stat in ("median_H", "mean_H"):
            vals = [out["entropy"]["0x"]["none"][stat]] + [out["entropy"][f"{s:g}x"][paper][stat] for s in STRENGTHS]
            out["monotonic"][f"{paper}.{stat}"] = {
                "values_0x_to_3x": vals,
                "monotone": monotone(vals, increasing=(paper == "entropy_increasing")),
            }
    return out


def baseline_reproduction() -> dict:
    """Unsteered records of the layer-44 run (B200) vs the layer-64 run (H100): same prompts, hardware differs."""
    parent = RESULTS / "no_instruction_control" / SLUG
    runs = {}
    for layer in LAYERS:
        path = unique_post_fix(parent, f"no_instruction_layer{layer}_strength1.0", "results.json")
        runs[layer] = {key(r): r for r in json.loads(path.read_text())["results"]["none"]}
    a, b = runs[NEW_LAYER], runs[OLD_LAYER]
    assert a.keys() == b.keys() and all(a[k]["user_prompt"] == b[k]["user_prompt"] for k in a)
    dpa = np.array([abs(float(a[k]["option_a_prob"]) - float(b[k]["option_a_prob"])) for k in a])
    dh = np.array([abs(record_entropy(a[k]) - record_entropy(b[k])) for k in a])
    return {"n": len(a), "identical_prompts": True, "mean_abs_dPA": r4(dpa.mean()), "max_abs_dPA": r4(dpa.max()),
            "mean_abs_dH": r4(dh.mean()), "max_abs_dH": r4(dh.max())}


def barebone_vector_check() -> dict:
    """Designated-script layer-44 minimal-prompt vectors vs the existing per-layer barebone layer_44.pt files."""
    import torch

    base = ROOT / "artifacts" / "neutral" / "steering_layers" / "steering_vectors"
    cos = {}
    for new in sorted((base / BAREBONE_L44_TEMPLATE / SLUG).glob(f"*/layer_{NEW_LAYER}.pt")):
        old = base / "barebone" / SLUG / new.parent.name / new.name
        va = torch.load(new, map_location="cpu", weights_only=False)["steering_vector"].float()
        vb = torch.load(old, map_location="cpu", weights_only=False)["steering_vector"].float()
        cos[new.parent.name] = round(float(torch.nn.functional.cosine_similarity(va, vb, dim=0)), 6)
    assert len(cos) == 7, cos
    return {"min_cosine": min(cos.values()), "per_pair": cos}


# ----------------------------------------------------------------------------- 3
def sweep(layer: int, pair_name: str) -> dict:
    cert, unc = TEST_PAIRS[pair_name]
    test_label = f"{cert.rstrip('!').replace(' ', '_')}_vs_{unc.rstrip('!').replace(' ', '_')}"
    parent = RESULTS / "steering_all_layers" / SLUG
    out = {"test_pair": [cert, unc], "sources": {}, "dH": {}, "mass_min": {}, "n_invalid": {}}
    zero = latest_zero_shot_local_result(MODEL, cert, unc)
    zero = Path(zero)
    assert zero.resolve().is_relative_to(RESULTS) and int(_DATE.search(zero.parent.name).group(1)) >= MIN_DATE, zero
    runs = {0.0: (zero, json.loads(zero.read_text()))}
    for s in SWEEP_STRENGTHS:
        path = unique_post_fix(parent, f"vector_Be_sure_vs_Be_random_test_{test_label}_all_layers_strength{s}", "all_layers_results.json")
        payload = json.loads(path.read_text())
        assert float(payload["metadata"]["steering_strength"]) == s
        assert (payload["metadata"]["test_certainty_instruction"], payload["metadata"]["test_uncertainty_instruction"]) == (cert, unc)
        runs[s] = (path, payload["results"][f"layer_{layer}"])
    for s in sorted(runs):
        path, records = runs[s]
        assert len(records) == 400, (s, len(records))
        d = paired_entropy_differences(records, cert, unc)
        out["sources"][f"{s:+g}"] = rel(path)
        out["dH"][f"{s:+g}"] = {"n_pairs": len(d), "mean": r4(np.mean(d)), "median": r4(np.median(d))}
        out["n_invalid"][f"{s:+g}"] = sum(not is_valid_probability_record(r) for r in records)
        out["mass_min"][f"{s:+g}"] = r4(min(float(r["option_a_prob"]) + float(r["option_b_prob"]) for r in records))
    return out


# ----------------------------------------------------------------------------- 4
def offdiag(matrix: list[list[float | None]]) -> dict:
    a = np.array(matrix, dtype=float)
    v = a[np.triu_indices_from(a, k=1)]
    return {"n_pairs": int(v.size), "mean": r4(np.nanmean(v)), "min": r4(np.nanmin(v)), "max": r4(np.nanmax(v)),
            "n_missing": int(np.isnan(v).sum())}


def similarity() -> dict:
    manifest_path = ROOT / "paper_manifest.yaml"
    _, pairs = matrix_from_group(find_manifest_group(load_manifest(manifest_path), "steering-extract"))
    out = {}
    for template, vec_template in (("standard", "standard"), ("barebone", BAREBONE_L44_TEMPLATE)):
        res44 = compute_model_matrix(MODEL, pairs, fixed_layer=NEW_LAYER, template_name=vec_template)
        dest = SIM_OUT / f"llama_layer44_designated_{template}"
        payload = {
            "metadata": {
                "script": "scripts/camera_ready_llama_layer44_numbers.py",
                "manifest": str(manifest_path),
                "group": "steering-extract",
                "models": [MODEL],
                "instruction_pairs": pairs,
                "use_fixed_layer_overrides": False,
                "fixed_layers_by_model": {MODEL: NEW_LAYER},
                "template": template,
                "vector_dir": f"artifacts/neutral/steering_layers/steering_vectors/"
                f"{'' if vec_template == 'standard' else vec_template + '/'}{SLUG}/<pair>/layer_{NEW_LAYER}.pt",
                "ab_pairs_variant": os.environ["AB_PAIRS_VARIANT"],
            },
            "results": {MODEL: res44},
        }
        if (dest / "similarity_matrices.json").exists():  # idempotent: never overwrite, check instead
            prev = json.loads((dest / "similarity_matrices.json").read_text())
            assert prev["results"][MODEL]["matrix"] == res44["matrix"], f"{dest} differs from a recomputation"
        else:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "similarity_matrices.json").write_text(json.dumps(payload, indent=2))
            write_csv(dest / f"{SLUG}.csv", res44["labels"], res44["matrix"])
        stored = SIM_OUT / f"designated_{template}" / "similarity_matrices.json"
        stored_payload = json.loads(stored.read_text())
        assert stored_payload["metadata"]["fixed_layers_by_model"][MODEL] == OLD_LAYER
        res64 = stored_payload["results"][MODEL]
        assert res64["labels"] == res44["labels"]
        a44, a64 = np.array(res44["matrix"], dtype=float), np.array(res64["matrix"], dtype=float)
        iu = np.triu_indices_from(a44, k=1)
        # Pooled over the three paper models (Qwen models at 56 from the stored matrices), as quoted
        # in the paper text and the Fig. 11 caption.
        qwen = [np.array(stored_payload["results"][m]["matrix"], dtype=float)[iu] for m in stored_payload["metadata"]["models"] if m != MODEL]
        assert len(qwen) == 2
        pooled = {}
        for layer, a in ((NEW_LAYER, a44), (OLD_LAYER, a64)):
            v = np.concatenate(qwen + [a[iu]])
            pooled[f"llama_layer{layer}"] = {"n_pairs": int(v.size), "mean": r4(v.mean()), "min": r4(v.min()), "max": r4(v.max())}
        out[template] = {
            f"layer{NEW_LAYER}": {**offdiag(res44["matrix"]), "source": rel(dest / "similarity_matrices.json")},
            f"layer{OLD_LAYER}": {**offdiag(res64["matrix"]), "source": rel(stored)},
            "pooled_3_models": pooled,
            "corr_offdiag_44_vs_64": r4(np.corrcoef(a44[iu], a64[iu])[0, 1]),
            "labels": res44["labels"],
            "matrix_layer44": [[r4(x) for x in row] for row in res44["matrix"]],
        }
    return out


def write_tables(numbers: dict) -> None:
    with open(OUT / "dose_response.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "strength", "direction", "n", "n_valid", "mean_H", "median_H",
                    "mass_mean", "mass_min", "mass_frac_below_0.9", "paired_dH_mean", "paired_dH_median"])
        for layer in LAYERS:
            dr = numbers["dose_response"][f"layer{layer}"]
            for s, conds in dr["entropy"].items():
                for cond, e in conds.items():
                    m = dr["mass"][s][cond]
                    p = dr["paired_dH_vs_none"].get(s, {}).get(cond, {})
                    w.writerow([layer, s, cond, e["n"], e["n_valid"], e["mean_H"], e["median_H"],
                                m["mean"], m["min"], m["frac_below_0.9"], p.get("mean", ""), p.get("median", "")])
    with open(OUT / "instruction_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_pair", "layer", "strength", "n_pairs", "mean_dH", "median_dH", "n_invalid_records", "mass_min"])
        for pair_name in TEST_PAIRS:
            for layer in LAYERS:
                sw = numbers["instruction_sweep"][pair_name][f"layer{layer}"]
                for s, d in sw["dH"].items():
                    w.writerow([pair_name, layer, s, d["n_pairs"], d["mean"], d["median"], sw["n_invalid"][s], sw["mass_min"][s]])
    with open(OUT / "similarity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["template", "layer", "n_pairs", "mean", "min", "max", "source"])
        for template, d in numbers["similarity"].items():
            for layer in LAYERS:
                x = d[f"layer{layer}"]
                w.writerow([template, layer, x["n_pairs"], x["mean"], x["min"], x["max"], x["source"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {
        "model": MODEL,
        "sign_convention": "paper terms: entropy_decreasing = code condition 'positive' (paper negative steering); "
        "entropy_increasing = code 'negative' (paper positive steering). Sweep strengths: paper = code; "
        "s > 0 reinforces the prompted instruction.",
        "validity_rule": "entropy statistics on records with P(A)+P(B) > 0.9; mass statistics on all records",
        "dose_response": {f"layer{L}": dose_response(L) for L in LAYERS},
        "baseline_reproduction_44_vs_64": baseline_reproduction(),
        "barebone_vector_check": barebone_vector_check(),
        "instruction_sweep": {p: {f"layer{L}": sweep(L, p) for L in LAYERS} for p in TEST_PAIRS},
        "similarity": similarity(),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    write_tables(numbers)
    print(json.dumps({k: v for k, v in numbers.items() if k in ("dose_response",)}, indent=1)[:6000])
    for f in sorted(OUT.iterdir()):
        print(rel(f))


if __name__ == "__main__":
    main()
