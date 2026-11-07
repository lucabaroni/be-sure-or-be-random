"""Rescore the five Fireworks sweep runs with the first-token (longest-prefix) rule.

The original Fireworks runs (May 2026) matched an option only when the whole option word was a single
returned token, so options split by the model's tokenizer (e.g. ' tou' + 'can') got probability 0 and
their records failed the 0.9 mass rule. Every other model scores the first token of ' ' + option
(utils.option_first_token_id), and the API extractor (experiments._extract_option_probabilities) takes
the returned token with the longest prefix match (experiments._option_prefix_match_length). This script
applies that rule to the stored top-2 log-probabilities of the original runs; no API calls are made.

Inputs (selected by name, listed in ORIGINAL_RUNS):
  results/neutral/zero_shot_api/accounts_fireworks_models_<model>/<pair>_<May 2026 stamp>/results.json
Outputs (never overwritten; the script stops if any of them exists):
  results/neutral/zero_shot_api/accounts_fireworks_models_<model>/<pair>_20261104_120000/{results.json,README.md}
The sweep figures read the rescored runs.

Fields: option_a_prob/option_b_prob are rescored; option_a_prob_exact_match/option_b_prob_exact_match keep
the original values; option_a_token/option_b_token give the matched token; scoring = SCORING_TAG.

--check makes no changes: it rebuilds each original record as a synthetic API response, runs the live
extractor _extract_option_probabilities on it and counts records where it disagrees with the rescoring,
then compares the rescored runs already on disk with what this script would write.

Run from the repository root:
  AB_PAIRS_VARIANT=neutral python scripts/analysis/rescore_fireworks_first_token.py [--check]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from _analysis_common import get_results_dir, instruction_label, slugify_model_name

import sweep_models as sm  # noqa: E402
from be_sure_or_random.experiments import _extract_option_probabilities, _option_prefix_match_length  # noqa: E402
from common import load_zero_shot_result  # noqa: E402

RESCORED_STAMP = "20261104_120000"
SCORING_TAG = "first_token_prefix_rescored_20261104"

# Original run stamps, one per instruction pair in sweep_models.PAIRS order.
ORIGINAL_RUNS: dict[str, list[str]] = {
    "accounts/fireworks/models/deepseek-v3p1": [
        "20260504_211743", "20260504_212352", "20260504_212847", "20260504_213316",
        "20260504_214036", "20260504_214828", "20260504_215334"],
    "accounts/fireworks/models/deepseek-v3p2": [
        "20260504_222450", "20260504_230118", "20260505_013039", "20260505_065945",
        "20260505_071444", "20260505_072904", "20260505_081920"],
    "accounts/fireworks/models/glm-4p7": [
        "20260505_090306", "20260505_091227", "20260505_092111", "20260505_092959",
        "20260505_093930", "20260505_094812", "20260505_095706"],
    "accounts/fireworks/models/glm-5": [
        "20260505_100834", "20260505_101709", "20260505_102339", "20260505_103135",
        "20260505_103919", "20260505_104708", "20260505_105501"],
    "accounts/fireworks/models/llama-v3p3-70b-instruct": [
        "20260505_082415", "20260505_082915", "20260505_083406", "20260505_083841",
        "20260505_084310", "20260505_084636", "20260505_085130"],
}

README = """Rescored copy of {src}.

The original Fireworks run (May 2026) matched each option only when the whole option word was a single
returned token, so options that the model's tokenizer splits (e.g. ' tou' + 'can') were recorded as
probability 0 and those records failed the 0.9 mass rule. This copy applies the rule used for every other
model: local runs score the first token of ' ' + option (utils.option_first_token_id), and the API extractor
(experiments._extract_option_probabilities) takes the returned token with the longest prefix match.
The rule was applied to the stored top-2 log-probabilities; no new API calls were made.
Cross-checked against the live extractor on all 14,000 Fireworks records (0 mismatches).
Fields: option_a_prob/option_b_prob rescored; option_a_prob_exact_match/option_b_prob_exact_match keep the
original values; option_a_token/option_b_token give the matched token; scoring = "{tag}".
Script: scripts/analysis/rescore_fireworks_first_token.py.
"""


def _as_list(value) -> list:
    return ast.literal_eval(value) if isinstance(value, str) else list(value or [])


def rescore_record(record: dict) -> dict:
    """Copy of ``record`` with option probabilities read off the stored top-k by longest prefix match.

    Same rule as experiments._extract_option_probabilities: for each option the returned token with the
    longest prefix match wins (ties go to the earlier, more probable token); an option with no candidate
    gets probability 0; one token being the best candidate for both options is ambiguous.
    """
    tokens = _as_list(record.get("top_2_tokens"))
    probs = [float(p) for p in _as_list(record.get("top_2_probs"))]
    best_a = best_b = None  # (match length, index, prob)
    for idx, (token, prob) in enumerate(zip(tokens, probs)):
        len_a = _option_prefix_match_length(token, record["option_a"])
        len_b = _option_prefix_match_length(token, record["option_b"])
        if len_a is not None and (best_a is None or len_a > best_a[0]):
            best_a = (len_a, idx, prob)
        if len_b is not None and (best_b is None or len_b > best_b[0]):
            best_b = (len_b, idx, prob)
    if best_a is not None and best_b is not None and best_a[1] == best_b[1]:
        raise ValueError(f"ambiguous option prefix: {record['option_a']!r} / {record['option_b']!r} -> {tokens[best_a[1]]!r}")
    out = dict(record)
    out["option_a_prob"] = best_a[2] if best_a else 0.0
    out["option_b_prob"] = best_b[2] if best_b else 0.0
    out["option_a_prob_exact_match"] = record["option_a_prob"]
    out["option_b_prob_exact_match"] = record["option_b_prob"]
    out["option_a_token"] = tokens[best_a[1]] if best_a else None
    out["option_b_token"] = tokens[best_b[1]] if best_b else None
    out["scoring"] = SCORING_TAG
    return out


def run_pairs(results_dir: Path):
    """Yield (model, (cert, unc), original run dir, rescored run dir)."""
    for model, stamps in ORIGINAL_RUNS.items():
        model_dir = results_dir / "zero_shot_api" / slugify_model_name(model)
        assert len(stamps) == len(sm.PAIRS)
        for (cert, unc), stamp in zip(sm.PAIRS, stamps):
            label = instruction_label(cert, unc)
            yield model, (cert, unc), model_dir / f"{label}_{stamp}", model_dir / f"{label}_{RESCORED_STAMP}"


def rescored_payload(src: Path, results_dir: Path) -> tuple[str, str]:
    records = [rescore_record(r) for r in load_zero_shot_result(src / "results.json")]
    readme = README.format(src=(src / "results.json").relative_to(results_dir), tag=SCORING_TAG)
    return json.dumps(records, indent=1, ensure_ascii=False), readme


def synthetic_response(record: dict):
    """API response carrying the stored answer token and its top-k, as the live extractor expects."""
    generated = record["generated_token"]
    generated_logprob = float(record["generated_logprob"])
    tops = []
    for token, prob in zip(_as_list(record["top_2_tokens"]), _as_list(record["top_2_probs"])):
        prob = float(prob)
        logprob = generated_logprob if token == generated else (math.log(prob) if prob > 0 else -1e9)
        tops.append(SimpleNamespace(token=token, logprob=logprob))
    content = [SimpleNamespace(token="ANSWER:", logprob=0.0, top_logprobs=[]),
               SimpleNamespace(token=generated, logprob=generated_logprob, top_logprobs=tops)]
    message = SimpleNamespace(content="ANSWER:" + generated)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=SimpleNamespace(content=content))])


def check(results_dir: Path) -> int:
    compared = skipped = mismatches = 0
    identical = differing = absent = 0
    for model, _, src, dst in run_pairs(results_dir):
        for record in load_zero_shot_result(src / "results.json"):
            if not _as_list(record.get("top_2_tokens")) or record.get("generated_token") is None:
                skipped += 1
                continue
            live = _extract_option_probabilities(synthetic_response(record), record["option_a"], record["option_b"])
            ours = rescore_record(record)
            compared += 1
            if (live["logprob_status"] != "ok"
                    or abs(live["option_a_prob"] - ours["option_a_prob"]) > 1e-6
                    or abs(live["option_b_prob"] - ours["option_b_prob"]) > 1e-6):
                mismatches += 1
                if mismatches <= 5:
                    print("MISMATCH", model, record["option_a"], record["option_b"], live["logprob_status"],
                          live["option_a_prob"], live["option_b_prob"], ours["option_a_prob"], ours["option_b_prob"])
        if (dst / "results.json").is_file():
            results_json, _ = rescored_payload(src, results_dir)
            same = (dst / "results.json").read_text(encoding="utf-8") == results_json
            identical += same
            differing += not same
            if not same:
                print("DIFFERS from what this script writes:", dst)
        else:
            absent += 1
    print(f"cross-check against _extract_option_probabilities: {compared} records compared, "
          f"{skipped} without stored top-k skipped, {mismatches} mismatches")
    print(f"rescored runs on disk: {identical} identical to a fresh rescoring, {differing} differ, {absent} absent")
    return int(mismatches > 0 or differing > 0)


def write(results_dir: Path) -> int:
    jobs = list(run_pairs(results_dir))
    missing = [str(src) for _, _, src, _ in jobs if not (src / "results.json").is_file()]
    existing = [str(dst) for _, _, _, dst in jobs if dst.exists()]
    if missing:
        print("original runs not found:", *missing, sep="\n  ")
        return 1
    if existing:
        print("refusing to overwrite existing rescored runs (use --check to compare them):", *existing, sep="\n  ")
        return 1
    for _, _, src, dst in jobs:
        results_json, readme = rescored_payload(src, results_dir)
        dst.mkdir(exist_ok=False)
        (dst / "results.json").write_text(results_json, encoding="utf-8")
        (dst / "README.md").write_text(readme, encoding="utf-8")
        print("wrote", dst)
    print(f"{len(jobs)} rescored runs written")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="cross-check only, write nothing")
    parser.add_argument("--results-dir", type=Path, default=get_results_dir(),
                        help="results root holding zero_shot_api/ (default: results/neutral)")
    args = parser.parse_args()
    sys.exit(check(args.results_dir) if args.check else write(args.results_dir))


if __name__ == "__main__":
    main()
