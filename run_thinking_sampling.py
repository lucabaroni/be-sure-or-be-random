"""
Local batched sampling with Qwen3-32B, thinking tokens enabled.

Unlike the logit-reading experiments, this script:
  - Does NOT append "ANSWER:" as a generation prefix — the model generates
    freely after the assistant turn marker.
  - Enables thinking mode (enable_thinking=True in the chat template) so the
    model can emit <think>...</think> before answering.
  - Draws N_SAMPLES completions per (pair, flip_ab, instruction) condition at
    temperature=1 and derives empirical entropy from the resulting distribution.

GPU batching strategy
─────────────────────
All conditions are submitted to vLLM in a single llm.generate() call.
vLLM uses continuous batching and (optionally) prefix-caching so the shared
system-prompt KV state is computed only once per unique prefix.  Requesting
n=N_SAMPLES in SamplingParams makes vLLM schedule all repeats of a prompt
together, reusing the prompt KV cache across all N samples.

Usage
─────
    # standard ab_pairs + standard (logprob-style) system prompt
    python run_thinking_sampling.py

    # neutral ab_pairs + CoT across-trajectory prompt
    AB_PAIRS_VARIANT=neutral python run_thinking_sampling.py \
        --system-prompt cot_across_trajectory_system.txt

    # CoT within+across prompt
    AB_PAIRS_VARIANT=neutral python run_thinking_sampling.py \
        --system-prompt cot_within_and_across_system.txt

    # quick smoke-test (5 pairs, 3 samples)
    python run_thinking_sampling.py --pairs 5 --samples 3

Requirements
────────────
    pip install vllm   # or: uv pip install vllm
    # (not in the default project deps; see pyproject.toml [thinking] extra)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from math import exp, log2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Config defaults (override via CLI flags or environment variables)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_N_SAMPLES = 30
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 2048  # generous budget: thinking block + short answer

# Instruction pairs to sweep — mirrors the main paper experiment set.
INSTRUCTION_PAIRS: List[Tuple[str, str]] = [
    ("Be sure!", "Be random!"),
    ("Maximize certainty", "Minimize certainty"),
    ("Maximize certainty", "Maximize uncertainty"),
    ("Commit to one option", "Keep both options equally likely"),
    ("Reply with confidence", "Reply without confidence"),
    ("Aim for low entropy", "Aim for high entropy"),
    ("Be deterministic", "Assign equal probability to both options"),
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ab_pairs_path() -> Path:
    variant = os.environ.get("AB_PAIRS_VARIANT", "original")
    return DATA_DIR / ("ab_pairs_neutral.json" if variant == "neutral" else "ab_pairs.json")


def _results_root() -> Path:
    variant = os.environ.get("AB_PAIRS_VARIANT", "original")
    base = RESULTS_DIR / "neutral" if variant == "neutral" else RESULTS_DIR
    return base / "thinking_sampling"


def resolve_system_prompt_path(system_prompt: str) -> Path:
    """Accept a bare name, a filename, or a path under prompts/."""
    candidate = Path(system_prompt)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".txt":
        candidate = Path(f"{system_prompt}.txt")
    path = PROMPTS_DIR / candidate.name
    if not path.is_file():
        raise FileNotFoundError(
            f"System prompt not found: {system_prompt!r} (looked for {path})"
        )
    return path


def build_system_prompt(
    certainty_instruction: str,
    uncertainty_instruction: str,
    system_prompt_file: str | Path = "standard_system.txt",
) -> str:
    template = resolve_system_prompt_path(str(system_prompt_file)).read_text().strip()
    return template.format(
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
    )


def _post_thinking_text(text: str) -> str:
    """Return the visible answer text after any generated thinking block."""
    think_end = text.rfind("</think>")
    if think_end >= 0:
        return text[think_end + len("</think>"):]
    return text


def parse_answer(text: str, token_a: str, token_b: str) -> Optional[str]:
    """Strictly accept only ``ANSWER: <token_a>`` or ``ANSWER: <token_b>``.

    The model may emit chain-of-thought before ``</think>``.  After that point,
    the visible answer must contain nothing except the exact instructed answer
    format, modulo surrounding whitespace.
    """
    visible = _post_thinking_text(text).strip()
    if visible == f"ANSWER: {token_a}":
        return token_a
    if visible == f"ANSWER: {token_b}":
        return token_b
    return None


def answer_logprob_prefix(completion_text: str) -> Optional[str]:
    """Return generated text through ``ANSWER:`` for post-hoc option scoring."""
    visible_start = 0
    think_end = completion_text.rfind("</think>")
    if think_end >= 0:
        visible_start = think_end + len("</think>")
    match = re.search(r"ANSWER:", completion_text[visible_start:])
    if match is None:
        return None
    return completion_text[: visible_start + match.end()]


def _token_logprob_value(logprob_entry) -> Optional[float]:
    if logprob_entry is None:
        return None
    if hasattr(logprob_entry, "logprob"):
        return float(logprob_entry.logprob)
    if isinstance(logprob_entry, dict) and "logprob" in logprob_entry:
        return float(logprob_entry["logprob"])
    return None


def _suffix_prompt_logprob(output, suffix_token_ids: List[int]) -> Optional[float]:
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
    if not prompt_logprobs or len(prompt_token_ids) < len(suffix_token_ids):
        return None

    start = len(prompt_token_ids) - len(suffix_token_ids)
    if prompt_token_ids[start:] != suffix_token_ids:
        return None

    total = 0.0
    for pos, token_id in enumerate(suffix_token_ids, start=start):
        token_logprobs = prompt_logprobs[pos]
        if token_logprobs is None:
            return None
        entry = token_logprobs.get(token_id)
        if entry is None:
            entry = token_logprobs.get(str(token_id))
        logprob = _token_logprob_value(entry)
        if logprob is None:
            return None
        total += logprob
    return total


def _option_suffix(tokenizer, option: str) -> Tuple[str, List[int], str]:
    suffix = " " + option
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    return suffix, token_ids, decoded


def add_answer_option_logprobs(
    llm,
    tokenizer,
    prompts_by_condition: List[str],
    results: List[Dict],
    batch_size: int,
) -> None:
    """Score both options at the generated ``ANSWER:`` prefix for each result."""
    from vllm import SamplingParams

    score_requests: List[Dict] = []
    for result_idx, result in enumerate(results):
        prefix = answer_logprob_prefix(result["response"])
        if prefix is None:
            continue
        base_prompt = prompts_by_condition[result["condition_idx"]] + prefix
        for option_key, option in (("token_a", result["token_a"]), ("token_b", result["token_b"])):
            suffix, suffix_token_ids, decoded_suffix = _option_suffix(tokenizer, option)
            score_requests.append({
                "result_idx": result_idx,
                "option_key": option_key,
                "prompt": base_prompt + suffix,
                "suffix_token_ids": suffix_token_ids,
                "decoded_suffix": decoded_suffix,
            })

    if not score_requests:
        return

    scoring_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    for start in range(0, len(score_requests), batch_size):
        batch = score_requests[start:start + batch_size]
        outputs = llm.generate([item["prompt"] for item in batch], scoring_params)
        for item, output in zip(batch, outputs):
            logprob = _suffix_prompt_logprob(output, item["suffix_token_ids"])
            result = results[item["result_idx"]]
            option_key = item["option_key"]
            result[f"answer_{option_key}_logprob"] = logprob
            result[f"answer_{option_key}_prob"] = None if logprob is None else exp(logprob)
            result[f"answer_{option_key}_token_ids"] = item["suffix_token_ids"]
            result[f"answer_{option_key}_decoded"] = item["decoded_suffix"]

    for result in results:
        prob_a = result.get("answer_token_a_prob")
        prob_b = result.get("answer_token_b_prob")
        total = (prob_a or 0.0) + (prob_b or 0.0)
        result["answer_token_a_normalized_prob"] = prob_a / total if total > 0 else None
        result["answer_token_b_normalized_prob"] = prob_b / total if total > 0 else None


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


# --------------------------------------------------------------------------- #
# Build prompt list for one instruction pair
# --------------------------------------------------------------------------- #

def build_prompts(
    ab_pairs: List[Dict],
    certainty_instruction: str,
    uncertainty_instruction: str,
    tokenizer,
    system_prompt_file: str | Path = "standard_system.txt",
) -> Tuple[List[str], List[Dict]]:
    system_prompt = build_system_prompt(
        certainty_instruction,
        uncertainty_instruction,
        system_prompt_file=system_prompt_file,
    )
    prompts: List[str] = []
    meta: List[Dict] = []

    for pair_idx, pair in enumerate(ab_pairs):
        token_a, token_b = pair["token_a"], pair["token_b"]
        for label, instruction in [
            ("certainty", certainty_instruction),
            ("uncertainty", uncertainty_instruction),
        ]:
            for flip_ab in (False, True):
                first, second = (token_b, token_a) if flip_ab else (token_a, token_b)
                user_prompt = f"{first} or {second}? {instruction}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                # enable_thinking=True → model may emit <think>…</think> before answering.
                # No answer_prefix ("ANSWER:") is appended — free generation.
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompts.append(text)
                meta.append({
                    "pair_idx": pair_idx,
                    "token_a": token_a,
                    "token_b": token_b,
                    "label": label,
                    "instruction": instruction,
                    "certainty_instruction": certainty_instruction,
                    "uncertainty_instruction": uncertainty_instruction,
                    "flip_ab": flip_ab,
                    "user_prompt": user_prompt,
                })

    return prompts, meta


# --------------------------------------------------------------------------- #
# Compute and print per-pair summaries
# --------------------------------------------------------------------------- #

def compute_summaries(
    all_results: List[Dict],
    ab_pairs: List[Dict],
    certainty_instruction: str,
    uncertainty_instruction: str,
) -> List[Dict]:
    groups: Dict = defaultdict(list)
    for r in all_results:
        groups[(r["pair_idx"], r["label"], r["flip_ab"])].append(r)

    flip_summaries = []
    for pair_idx, pair in enumerate(ab_pairs):
        token_a, token_b = pair["token_a"], pair["token_b"]
        print(f"\n[{pair_idx + 1:3d}/{len(ab_pairs)}]  {token_a} / {token_b}")
        for flip_ab in (False, True):
            h_by_label: Dict[str, float] = {}
            for label in ("certainty", "uncertainty"):
                results = groups[(pair_idx, label, flip_ab)]
                valid = [r for r in results if r["chosen"] is not None]
                n_valid = len(valid)
                count_a = sum(1 for r in valid if r["chosen"] == token_a)
                count_b = n_valid - count_a
                n_inv = len(results) - n_valid
                if n_valid > 0:
                    p_a = count_a / n_valid
                    h = binary_entropy(p_a)
                    h_by_label[label] = h
                    h_str = f"{h:.3f}"
                else:
                    h_str = "N/A"
                order = f"{token_b}|{token_a}" if flip_ab else f"{token_a}|{token_b}"
                print(
                    f"  {label:>12}  flip={int(flip_ab)}  [{order:20s}]  "
                    f"{token_a}={count_a:3d}  {token_b}={count_b:3d}  "
                    f"invalid={n_inv}  H={h_str}"
                )
            if "certainty" in h_by_label and "uncertainty" in h_by_label:
                shift = h_by_label["uncertainty"] - h_by_label["certainty"]
                flip_summaries.append({
                    "token_a": token_a,
                    "token_b": token_b,
                    "flip_ab": flip_ab,
                    "certainty_instruction": certainty_instruction,
                    "uncertainty_instruction": uncertainty_instruction,
                    "entropy_certainty": h_by_label["certainty"],
                    "entropy_uncertainty": h_by_label["uncertainty"],
                    "entropy_shift": shift,
                })
                print(f"           → shift={shift:+.3f}")
            else:
                print("           → shift=N/A")

    return flip_summaries


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local batched sampling with Qwen3-32B thinking enabled."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id.")
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N AB pairs (None = all). Useful for quick tests.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        metavar="N",
        help=f"Samples per condition (default {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max new tokens per completion (default {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use (default 0.90).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism. Defaults to all available.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Context length vLLM allocates for. Defaults to the model's own "
            "maximum, which is wastefully large for these short prompts."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Cap on tokens scheduled per engine step. The post-hoc logprob pass "
            "materialises a float32 [scheduled_tokens x vocab] log_softmax, so "
            "this bounds its peak memory."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Cap concurrent sequences in vLLM. Required for single-GPU Qwen3.5 "
            "runs where Mamba cache blocks are fewer than the default 1024."
        ),
    )
    parser.add_argument(
        "--instruction-pair",
        metavar="CERT:UNCERT",
        action="append",
        default=None,
        help=(
            'Run only this instruction pair, e.g. "Be sure!:Be random!". '
            "Repeatable. Defaults to all pairs in INSTRUCTION_PAIRS."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default="standard_system.txt",
        help=(
            "System prompt template under prompts/ (or a path). "
            "Use standard_system.txt for the logprob-style prompt, "
            "cot_across_trajectory_system.txt for across-trajectory control, "
            "or cot_within_and_across_system.txt for within+across control."
        ),
    )
    parser.add_argument(
        "--max-resample-rounds",
        type=int,
        default=50,
        help="Maximum retry rounds used to replace malformed completions (default 50).",
    )
    parser.add_argument(
        "--logprob-batch-size",
        type=int,
        default=4096,
        help="Batch size for the post-hoc ANSWER: option logprob scoring pass (default 4096).",
    )
    parser.add_argument(
        "--skip-answer-logprobs",
        action="store_true",
        help="Skip the post-hoc option logprob scoring pass.",
    )
    args = parser.parse_args()
    if args.max_resample_rounds < 0:
        parser.error("--max-resample-rounds must be non-negative")
    if args.logprob_batch_size <= 0:
        parser.error("--logprob-batch-size must be positive")

    try:
        system_prompt_path = resolve_system_prompt_path(args.system_prompt)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    system_prompt_slug = system_prompt_path.stem

    # Resolve instruction pairs
    if args.instruction_pair:
        instruction_pairs = []
        for raw in args.instruction_pair:
            parts = raw.split(":", 1)
            if len(parts) != 2:
                parser.error(f"--instruction-pair must be CERT:UNCERT, got: {raw!r}")
            instruction_pairs.append((parts[0], parts[1]))
    else:
        instruction_pairs = INSTRUCTION_PAIRS

    # Tensor parallel size
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        try:
            import torch
            tensor_parallel_size = max(1, torch.cuda.device_count())
        except Exception:
            tensor_parallel_size = 1

    # ------------------------------------------------------------------ #
    # Load tokenizer
    # ------------------------------------------------------------------ #
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Install with: pip install transformers") from exc

    print(f"Loading tokenizer for {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ------------------------------------------------------------------ #
    # Load AB pairs
    # ------------------------------------------------------------------ #
    ab_pairs_path = _ab_pairs_path()
    ab_pairs: List[Dict] = json.loads(ab_pairs_path.read_text())
    if args.pairs is not None:
        ab_pairs = ab_pairs[: args.pairs]

    variant_label = os.environ.get("AB_PAIRS_VARIANT", "original")
    print(f"AB pairs:           {ab_pairs_path.name}  ({len(ab_pairs)} pairs, variant={variant_label!r})")
    print(f"System prompt:      {system_prompt_path.name}")
    print(f"Instruction pairs:  {len(instruction_pairs)}")
    print(f"Model:              {args.model}")
    print(f"Samples/condition:  {args.samples}")
    print(f"Temperature:        {args.temperature}")
    print(f"Max tokens:         {args.max_tokens}")
    print(f"Tensor parallel:    {tensor_parallel_size}")
    print(f"Max resample rounds:{args.max_resample_rounds}")
    print(f"Answer logprobs:    {not args.skip_answer_logprobs}")
    print()

    # ------------------------------------------------------------------ #
    # Build all prompts across all instruction pairs
    # ------------------------------------------------------------------ #
    all_prompts: List[str] = []
    all_meta: List[Dict] = []
    for certainty_instruction, uncertainty_instruction in instruction_pairs:
        prompts, meta = build_prompts(
            ab_pairs,
            certainty_instruction,
            uncertainty_instruction,
            tokenizer,
            system_prompt_file=system_prompt_path,
        )
        all_prompts.extend(prompts)
        all_meta.extend(meta)

    n_conditions = len(all_prompts)
    n_total = n_conditions * args.samples
    print(
        f"Conditions:         {n_conditions}  "
        f"({len(ab_pairs)} pairs × 2 orderings × 2 instructions × {len(instruction_pairs)} instr-pairs)"
    )
    print(f"Total completions:  {n_total}\n")

    # ------------------------------------------------------------------ #
    # Load vLLM and run inference
    # ------------------------------------------------------------------ #
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is required for this script.\n"
            "Install with:  pip install vllm\n"
            "or:            uv pip install 'be-sure-or-random[thinking]'"
        ) from exc

    print("Loading model with vLLM …")
    engine_kwargs = {}
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,  # share prompt KV cache across N_SAMPLES repeats
        **engine_kwargs,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.samples,
    )

    print(
        f"Generating {args.samples} samples × {n_conditions} conditions "
        f"= {n_total} total completions …\n"
    )
    outputs = llm.generate(all_prompts, sampling_params)

    # ------------------------------------------------------------------ #
    # Collect valid results, resampling malformed final answers
    # ------------------------------------------------------------------ #
    condition_states = [
        {
            "meta": meta,
            "valid": [],
            "rejected": [],
            "attempts": 0,
        }
        for meta in all_meta
    ]

    def record_completion(condition_idx: int, completion, resample_round: int) -> None:
        state = condition_states[condition_idx]
        m = state["meta"]
        text = completion.text
        chosen = parse_answer(text, m["token_a"], m["token_b"])
        attempt_idx = state["attempts"]
        state["attempts"] += 1
        record = {
            **m,
            "condition_idx": condition_idx,
            "attempt_idx": attempt_idx,
            "resample_round": resample_round,
            "response": text,
            "chosen": chosen,
        }
        if chosen is None:
            state["rejected"].append(record)
            return
        record["sample_idx"] = len(state["valid"])
        state["valid"].append(record)

    for condition_idx, output in enumerate(outputs):
        for completion in output.outputs:
            record_completion(condition_idx, completion, resample_round=0)

    retry_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
    )
    for resample_round in range(1, args.max_resample_rounds + 1):
        retry_condition_indices = [
            condition_idx
            for condition_idx, state in enumerate(condition_states)
            for _ in range(args.samples - len(state["valid"]))
        ]
        if not retry_condition_indices:
            break
        print(
            f"Resample round {resample_round}: replacing "
            f"{len(retry_condition_indices)} malformed completions ..."
        )
        retry_outputs = llm.generate(
            [all_prompts[condition_idx] for condition_idx in retry_condition_indices],
            retry_params,
        )
        for condition_idx, output in zip(retry_condition_indices, retry_outputs):
            record_completion(condition_idx, output.outputs[0], resample_round=resample_round)

    incomplete = [
        (condition_idx, args.samples - len(state["valid"]))
        for condition_idx, state in enumerate(condition_states)
        if len(state["valid"]) < args.samples
    ]
    if incomplete:
        missing_total = sum(missing for _, missing in incomplete)
        raise RuntimeError(
            f"Failed to collect {missing_total} valid completions after "
            f"{args.max_resample_rounds} resample rounds. "
            "Increase --max-resample-rounds or inspect rejected completions."
        )

    all_results: List[Dict] = [
        result
        for state in condition_states
        for result in state["valid"]
    ]
    rejected_results: List[Dict] = [
        result
        for state in condition_states
        for result in state["rejected"]
    ]

    if not args.skip_answer_logprobs:
        print(
            f"\nScoring option logprobs after ANSWER: for {len(all_results)} valid completions ..."
        )
        add_answer_option_logprobs(
            llm=llm,
            tokenizer=tokenizer,
            prompts_by_condition=all_prompts,
            results=all_results,
            batch_size=args.logprob_batch_size,
        )

    # ------------------------------------------------------------------ #
    # Per instruction-pair summaries
    # ------------------------------------------------------------------ #
    all_flip_summaries: List[Dict] = []
    for certainty_instruction, uncertainty_instruction in instruction_pairs:
        label_header = f"{certainty_instruction}  vs  {uncertainty_instruction}"
        print(f"\n{'═' * 72}")
        print(f"  {label_header}")
        print(f"{'═' * 72}")
        pair_results = [
            r for r in all_results
            if r["certainty_instruction"] == certainty_instruction
            and r["uncertainty_instruction"] == uncertainty_instruction
        ]
        summaries = compute_summaries(
            pair_results, ab_pairs, certainty_instruction, uncertainty_instruction
        )
        all_flip_summaries.extend(summaries)
        shifts = [s["entropy_shift"] for s in summaries]
        if shifts:
            mean_shift = sum(shifts) / len(shifts)
            print(f"\n  Mean entropy shift: {mean_shift:+.3f} bits")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")
    out_dir = _results_root() / model_slug / system_prompt_slug / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "n_samples": args.samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_resample_rounds": args.max_resample_rounds,
        "strict_answer_format": True,
        "answer_logprobs_collected": not args.skip_answer_logprobs,
        "logprob_batch_size": None if args.skip_answer_logprobs else args.logprob_batch_size,
        "ab_pairs_file": str(ab_pairs_path),
        "ab_pairs_variant": variant_label,
        "n_pairs": len(ab_pairs),
        "system_prompt_file": str(system_prompt_path),
        "system_prompt_slug": system_prompt_slug,
        "instruction_pairs": instruction_pairs,
        "thinking_enabled": True,
        "answer_prefix": None,
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    (out_dir / "rejected_results.json").write_text(json.dumps(rejected_results, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(all_flip_summaries, indent=2))

    all_shifts = [s["entropy_shift"] for s in all_flip_summaries]
    if all_shifts:
        overall_mean = sum(all_shifts) / len(all_shifts)
        print(f"\n{'═' * 72}")
        print(
            f"  Overall mean entropy shift across {len(all_shifts)} (pair, flip, instr-pair) records: "
            f"{overall_mean:+.3f} bits"
        )
        print(f"  (positive = model follows instructions as expected)")
        print(f"{'═' * 72}")

    print(
        f"\nSaved {len(all_results)} valid records and "
        f"{len(rejected_results)} rejected attempts → {out_dir}"
    )


if __name__ == "__main__":
    main()
