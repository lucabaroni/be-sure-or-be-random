"""Check that the filler prompt variants add exactly 64 or 256 tokens.

Paper: appendix "Filler tokens": "We verified with each model's tokenizer and chat template that this adds
exactly 64 or 256 tokens."

For each model, base prompt (standard, barebone) and filler length N (64, 256), every prompt of the sweep
(100 option pairs x 2 orders x 7 instruction pairs x 2 instructions) is built exactly as the local runner
builds it (experiments.resolve_prompt_variant, experiments._pair_messages, utils.build_chat_input,
utils.tokenize_chat_text), once with the '<base>_fill<N>' variant and once with '<base>', and the token
counts are compared. Only tokenizers are loaded (no model weights, no GPU). Set HF_HUB_OFFLINE=1 to use
the local Hugging Face cache only; a tokenizer that cannot be loaded is reported as missing.
Output: analysis_outputs/filler_tokens.json

Run from the repository root:
  AB_PAIRS_VARIANT=neutral HF_HUB_OFFLINE=1 python scripts/analysis/check_filler_tokens.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from _analysis_common import ANALYSIS_DIR

import sweep_models as sm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from be_sure_or_random.experiments import _pair_messages, resolve_prompt_variant  # noqa: E402
from be_sure_or_random.utils import build_chat_input, load_ab_pairs, tokenize_chat_text  # noqa: E402

# Hugging Face repository ids of the four models of the filler experiment.
MODELS = ["Qwen/Qwen3.5-27B", "Qwen/Qwen3-32B", "Qwen/Qwen3.5-35B-A3B", "meta-llama/Meta-Llama-3-70B-Instruct"]
BASES = ["standard", "barebone"]
FILLER_LENGTHS = [64, 256]


def n_tokens(tokenizer, model: str, template: str, cert: str, unc: str, options: tuple[str, str], instruction: str) -> int:
    """Token count of the full local input (chat template plus 'ANSWER:') for one prompt."""
    spec = resolve_prompt_variant(template)
    fmt = {"certainty_instruction": cert, "uncertainty_instruction": unc}
    system_prompt = None if spec.omit_system else spec.system_template.format(**fmt)
    preamble = spec.user_preamble_template.format(**fmt) if spec.user_preamble_template is not None else None
    _, messages = _pair_messages(system_prompt, options[0], options[1], instruction, prefill_assistant=False,
                                 user_padding_spaces=spec.user_padding_spaces, user_padding_unit=spec.user_padding_unit,
                                 user_preamble=preamble)
    return len(tokenize_chat_text(tokenizer, build_chat_input(tokenizer, model, messages))["input_ids"])


def check_model(model: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)
    ab_pairs = load_ab_pairs()
    out = {}
    for base in BASES:
        for n in FILLER_LENGTHS:
            added = Counter()
            for cert, unc in sm.PAIRS:
                for pair in ab_pairs:
                    for options in ((pair["token_a"], pair["token_b"]), (pair["token_b"], pair["token_a"])):
                        for instruction in (cert, unc):
                            added[n_tokens(tokenizer, model, f"{base}_fill{n}", cert, unc, options, instruction)
                                  - n_tokens(tokenizer, model, base, cert, unc, options, instruction)] += 1
            out[f"{base}_fill{n}"] = {"expected": n, "n_prompts": sum(added.values()),
                                      "added_tokens_counts": {str(k): v for k, v in sorted(added.items())},
                                      "ok": set(added) == {n}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", help="Hugging Face id to check (repeatable; default: the 4 models)")
    parser.add_argument("--out", type=Path, default=ANALYSIS_DIR / "filler_tokens.json")
    args = parser.parse_args()

    results, missing = {}, {}
    for model in args.model or MODELS:
        try:
            results[model] = check_model(model)
        except OSError as err:  # tokenizer not in the cache (offline) or not downloadable
            missing[model] = str(err).splitlines()[0]
            print(f"{model:40s} MISSING tokenizer: {missing[model]}")
            continue
        for variant, row in results[model].items():
            print(f"{model:40s} {variant:18s} added tokens {row['added_tokens_counts']} over {row['n_prompts']} prompts: "
                  f"{'OK' if row['ok'] else 'FAIL'}")
    all_ok = bool(results) and all(r["ok"] for v in results.values() for r in v.values())
    print(f"\nall checked variants add exactly N tokens: {all_ok}; missing tokenizers: {sorted(missing) or 'none'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "missing_tokenizers": missing, "all_ok": all_ok}, indent=1) + "\n")
    print(f"wrote {args.out}")
    sys.exit(0 if all_ok and not missing else 1)


if __name__ == "__main__":
    main()
