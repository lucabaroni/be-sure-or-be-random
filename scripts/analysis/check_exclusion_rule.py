"""Apply the model-exclusion rule of the appendix "Results across models".

Paper: "We exclude models for which more than 20% of items fail the 0.9 threshold ..., which removes
Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-2B."

An item is one question in one option order, (option_a, option_b, flip_ab), within one instruction
pair. It fails when the certainty or the uncertainty record is missing or puts at most 0.9 probability
mass on the two options (common.is_valid_probability_record), so it cannot enter Delta H. The failure
fraction pools the items of the seven meaningful pairs under the standard system prompt; a model is
excluded when it exceeds MAX_FAILED_FRACTION. The per-record failure fraction is printed for reference.

Every local model under results/neutral/zero_shot_local/ with a post-fix run (dated 2026-07-20 or later)
for all seven pairs is checked; fine-tuned checkpoints (models_*) are evaluated separately (appendix
"Calibration fine-tuning") and skipped. The kept models are compared with sweep_models.LOCAL_MODELS.
Output: analysis_outputs/exclusion_rule.json

Run from the repository root:
  AB_PAIRS_VARIANT=neutral python scripts/analysis/check_exclusion_rule.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _analysis_common import ANALYSIS_DIR, POST_FIX_LOCAL, get_results_dir, latest_run, run_stamp, slugify_model_name

import sweep_models as sm  # noqa: E402
from common import is_valid_probability_record, load_zero_shot_result  # noqa: E402

MAX_FAILED_FRACTION = 0.20


def failure_fractions(model_dir: Path) -> dict | None:
    items_total = items_failed = recs_total = recs_failed = 0
    per_pair, runs = [], []
    for cert, unc in sm.PAIRS:
        path = latest_run(model_dir, cert, unc)
        if path is None or int(run_stamp(path)[:8]) < POST_FIX_LOCAL:
            return None
        runs.append(str(path.parent.relative_to(get_results_dir())))
        records = [r for r in load_zero_shot_result(path) if r.get("instruction") in (cert, unc)]
        items: dict[tuple, dict[str, bool]] = {}
        for r in records:
            items.setdefault((r["option_a"], r["option_b"], bool(r.get("flip_ab", False))), {})[r["instruction"]] = (
                is_valid_probability_record(r))
        failed = sum(not (v.get(cert) and v.get(unc)) for v in items.values())
        per_pair.append(failed / len(items))
        items_total += len(items)
        items_failed += failed
        recs_total += len(records)
        recs_failed += sum(not is_valid_probability_record(r) for r in records)
    return {"failed_item_fraction": items_failed / items_total, "n_items": items_total,
            "failed_item_fraction_max_over_pairs": max(per_pair),
            "failed_record_fraction": recs_failed / recs_total, "n_records": recs_total, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ANALYSIS_DIR / "exclusion_rule.json")
    args = parser.parse_args()

    root = get_results_dir() / "zero_shot_local"
    checked, skipped = {}, {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name.startswith("models_"):
            skipped[model_dir.name] = "fine-tuned checkpoint, evaluated separately"
            continue
        row = failure_fractions(model_dir)
        if row is None:
            skipped[model_dir.name] = "no post-fix standard-prompt run for all 7 pairs"
            continue
        row["excluded"] = row["failed_item_fraction"] > MAX_FAILED_FRACTION
        checked[model_dir.name] = row

    print(f"{'model (results dir)':40s} {'failed items':>12s} {'max pair':>9s} {'failed records':>15s}  decision")
    for name, row in sorted(checked.items(), key=lambda kv: -kv[1]["failed_item_fraction"]):
        print(f"{name:40s} {100 * row['failed_item_fraction']:11.1f}% {100 * row['failed_item_fraction_max_over_pairs']:8.1f}% "
              f"{100 * row['failed_record_fraction']:14.1f}%  {'EXCLUDE' if row['excluded'] else 'keep'}")
    for name, why in skipped.items():
        print(f"{name:40s} skipped: {why}")

    excluded = sorted(n for n, r in checked.items() if r["excluded"])
    kept = sorted(n for n, r in checked.items() if not r["excluded"])
    sweep = sorted(slugify_model_name(m) for m in sm.LOCAL_MODELS)
    print(f"\nexcluded (> {MAX_FAILED_FRACTION:.0%} of items fail): {excluded}")
    print(f"kept models equal sweep_models.LOCAL_MODELS: {kept == sweep}")
    if kept != sweep:
        print(f"  kept but not in LOCAL_MODELS: {sorted(set(kept) - set(sweep))}")
        print(f"  in LOCAL_MODELS but not kept: {sorted(set(sweep) - set(kept))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"max_failed_item_fraction": MAX_FAILED_FRACTION, "excluded": excluded, "kept": kept,
                                    "kept_equals_sweep_local_models": kept == sweep, "models": checked,
                                    "skipped": skipped}, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
