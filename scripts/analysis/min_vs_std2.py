"""Standard vs minimal (barebone) system prompt, all sweep models, using the paper's own run selection.

Standard: sweep_models.result_path (what the sweep figures use).
Minimal:  experiments.latest_prompt_variant_result / latest_zero_shot_api_variant_result(template "barebone").
Validity: common.is_valid_probability_record (0.9 rule); dH via common.paired_entropy_differences.
Run with AB_PAIRS_VARIANT=neutral from figures/scripts.
"""
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path("./figures/scripts")
sys.path[:0] = [str(HERE), str(HERE / "camera_ready/final"), str(HERE.parents[1])]
import sweep_models as sm  # noqa: E402
from common import (load_zero_shot_result, paired_entropy_differences, model_source,  # noqa: E402
                    is_valid_probability_record, record_entropy)
from be_sure_or_random.experiments import latest_prompt_variant_result, latest_zero_shot_api_variant_result  # noqa: E402

rng = np.random.default_rng(0)
B = 2000


def boot_ci(x):
    x = np.asarray(x)
    meds = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return np.percentile(meds, [2.5, 97.5])


def date(p):
    m = re.search(r"_(\d{8})_\d{6}", str(p))
    return int(m.group(1)) if m else None


def min_path(model, c, u):
    f = latest_prompt_variant_result if model_source(model) == "local" else latest_zero_shot_api_variant_result
    return f(model, c, u, "barebone")


def stats(path, c, u):
    recs = load_zero_shot_result(path)
    recs = [r for r in recs if r.get("instruction") in (c, u)]
    valid = [r for r in recs if is_valid_probability_record(r)]
    d = paired_entropy_differences(recs, c, u)
    hc = np.median([record_entropy(r) for r in valid if r["instruction"] == c]) if valid else np.nan
    hu = np.median([record_entropy(r) for r in valid if r["instruction"] == u]) if valid else np.nan
    sysp = (recs[0].get("system_prompt") or "")[:60] if recs else ""
    return dict(n=len(d), invalid=1 - len(valid) / max(len(recs), 1), med=float(np.median(d)) if d else np.nan,
                ci=boot_ci(d) if len(d) > 10 else (np.nan, np.nan), hc=hc, hu=hu, sys=sysp, date=date(path))


out = {}
problems = []
for model in sm.VERSIONS["all_families"]:
    name = sm.MODEL_META[model].display_name if hasattr(sm.MODEL_META[model], "display_name") else model
    row = {"std": [], "min": []}
    for c, u in sm.PAIRS:
        ps, pm = sm.result_path(model, c, u), min_path(model, c, u)
        if ps is None or pm is None:
            problems.append(f"{model} {c}: std={ps is not None} min={pm is not None}")
            continue
        if model_source(model) == "local" and date(pm) < sm.POST_FIX_LOCAL:
            problems.append(f"{model} {c}: minimal run pre-fix {pm}")
            continue
        row["std"].append(stats(ps, c, u)); row["min"].append(stats(pm, c, u))
    out[model] = row

print("PROBLEMS:", *problems, sep="\n  ")
print()
sysprompts = {(m, k): {s["sys"] for s in r[k]} for m, r in out.items() for k in ("std", "min")}
odd = {k: v for k, v in sysprompts.items() if len(v) != 1}
print("system-prompt heads (std):", {next(iter(v)) for (m, k), v in sysprompts.items() if k == "std" and v})
print("system-prompt heads (min):", {next(iter(v)) for (m, k), v in sysprompts.items() if k == "min" and v})
print("models with mixed system prompts across pairs:", odd)
print()
hdr = f"{'model':42s} np | {'std':>6s} {'min':>6s} | sig+ std/min | sig- std/min | inval% std/min | dates min"
print(hdr)
rows = []
for model, r in out.items():
    if not r["std"]:
        continue
    k = len(r["std"])
    s = np.mean([x["med"] for x in r["std"]]); m = np.mean([x["med"] for x in r["min"]])
    sp = sum(x["ci"][0] > 0 for x in r["std"]); mp = sum(x["ci"][0] > 0 for x in r["min"])
    sn = sum(x["ci"][1] < 0 for x in r["std"]); mn = sum(x["ci"][1] < 0 for x in r["min"])
    si = 100 * np.mean([x["invalid"] for x in r["std"]]); mi = 100 * np.mean([x["invalid"] for x in r["min"]])
    ds = sorted({x["date"] for x in r["min"]})
    rows.append((model, k, s, m, sp, mp, sn, mn, si, mi))
    print(f"{model:42s} {k:2d} | {s:6.2f} {m:6.2f} | {sp:5d}/{mp:<5d} | {sn:5d}/{mn:<5d} | {si:6.1f}/{mi:<6.1f} | {ds[0]}..{ds[-1]}")

rows_full = [x for x in rows if x[1] == 7]
S = np.array([x[2] for x in rows_full]); M = np.array([x[3] for x in rows_full])
print()
print(f"models with all 7 pairs under both prompts: {len(rows_full)}")
print(f"mean of model scores: standard {S.mean():.3f}, minimal {M.mean():.3f}; median standard {np.median(S):.3f}, minimal {np.median(M):.3f}")
print(f"minimal lower for {(M < S).sum()}/{len(S)}; |diff|<0.03 for {(abs(M - S) < 0.03).sum()}")
print(f"combinations with CI above zero: standard {sum(x[4] for x in rows_full)}, minimal {sum(x[5] for x in rows_full)} of {7 * len(rows_full)}")
print(f"combinations with CI below zero: standard {sum(x[6] for x in rows_full)}, minimal {sum(x[7] for x in rows_full)}")
from scipy.stats import wilcoxon, spearmanr
print("Wilcoxon signed-rank (model scores):", wilcoxon(S, M))
print("Spearman std vs min:", spearmanr(S, M))
top = S >= 0.25
print(f"models with standard >= 0.25: {top.sum()}, minimal mean {M[top].mean():.3f} vs standard {S[top].mean():.3f}")
print(f"models with standard < 0.1: {(S < 0.1).sum()}, minimal mean {M[S < 0.1].mean():.3f} vs standard {S[S < 0.1].mean():.3f}")

for target in ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"):
    print(f"\n{target}: per pair  [std med (CI) Hc Hu] | [min med (CI) Hc Hu]")
    for (c, u), a, b in zip(sm.PAIRS, out[target]["std"], out[target]["min"]):
        print(f"  {c[:22]:22s} {a['med']:+.2f} ({a['ci'][0]:+.2f},{a['ci'][1]:+.2f}) {a['hc']:.2f} {a['hu']:.2f} | "
              f"{b['med']:+.2f} ({b['ci'][0]:+.2f},{b['ci'][1]:+.2f}) {b['hc']:.2f} {b['hu']:.2f}  n={a['n']}/{b['n']}")
json.dump({m: {k: [{kk: (list(map(float, vv)) if kk == "ci" else vv) for kk, vv in x.items()} for x in v] for k, v in r.items()}
           for m, r in out.items()}, open("analysis_outputs/min_vs_std2.json", "w"), default=float)
