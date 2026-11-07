from __future__ import annotations

import gc
import math
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Prefer complete model snapshots in workspace_copied when workspace HF cache is incomplete.
_WORKSPACE_COPIED_HF_HUB = Path(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")))
if _WORKSPACE_COPIED_HF_HUB.exists() and "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = str(_WORKSPACE_COPIED_HF_HUB)
if "HF_DATASETS_CACHE" not in os.environ:
    _ws_datasets = Path(os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")))
    if _ws_datasets.exists():
        os.environ["HF_DATASETS_CACHE"] = str(_ws_datasets)

import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm

from be_sure_or_random.api_backends import (
    ANSWER_PREFIX,
    backend_uses_assistant_prefill,
    build_chat_completion_kwargs,
    create_chat_completion_with_provenance,
    fetch_openrouter_endpoints_snapshot,
    get_api_client,
    resolve_api_backend,
    resolve_api_concurrency,
    resolve_openrouter_endpoint,
    resolve_openrouter_model,
)
from be_sure_or_random.utils import (
    ARTIFACTS_DIR,
    PROMPTS_DIR,
    get_results_dir,
    get_steering_artifacts_dir,
    MultiLayerActivationExtractor,
    SteeringInjector,
    build_chat_input,
    tokenize_chat_text,
    choose_min_cosine_layer_from_similarities,
    categorical_entropy,
    get_layer_indices_every_n,
    get_num_layers,
    instruction_label,
    instruction_pair_label,
    latest_matching_dir,
    assert_ab_pairs_distinct_first_tokens,
    load_ab_pairs,
    load_hf_model_and_tokenizer,
    option_label_probabilities,
    option_token_probability,
    read_txt,
    save_json,
    slugify_model_name,
    slugify_instruction,
    timestamp,
)


_PAD_TEMPLATE_RE = re.compile(r"^(?P<base>.+)_pad(?P<n>\d+)$")
# Filler tokens: " ." is a single token for the Qwen3, Qwen3.5 and Llama-3 tokenizers
# and, unlike whitespace, survives chat templates that trim message content.
_FILL_TEMPLATE_RE = re.compile(r"^(?P<base>.+)_fill(?P<n>\d+)$")
FILLER_UNIT = " ."


@dataclass(frozen=True)
class PromptVariantSpec:
    """Resolved prompt-variant configuration for AB experiments."""

    template_name: str
    system_template: Optional[str]
    user_preamble_template: Optional[str] = None
    user_padding_spaces: int = 0
    user_padding_unit: str = " "

    @property
    def omit_system(self) -> bool:
        return self.system_template is None


def resolve_prompt_variant(template_name: str) -> PromptVariantSpec:
    """Resolve a prompt_variants template name into system/user construction rules.

    Supported names:
    - Existing files under ``prompts/variants/{name}.txt`` (and ``standard`` via
      ``standard_system.txt``)
    - ``format_only``: answer-format system prompt only (no logit-control text)
    - ``user_rules``: omit system; put reframed standard rules in the user message
    - ``{base}_pad{N}``: use base system template and append N spaces after the
      user question (e.g. ``standard_pad64``, ``barebone_pad256``). Note that
      Llama-3 and Qwen3.5 chat templates trim this whitespace, and tokenizers
      merge runs of spaces into one or two tokens.
    - ``{base}_fill{N}``: as ``_pad`` but append N filler tokens (``FILLER_UNIT``)
      after the user question (e.g. ``standard_fill64``, ``barebone_fill256``)
    """
    if template_name == "user_rules":
        return PromptVariantSpec(
            template_name=template_name,
            system_template=None,
            user_preamble_template=read_txt(PROMPTS_DIR / "variants" / "user_rules.txt"),
        )

    fill_match = _FILL_TEMPLATE_RE.fullmatch(template_name)
    if fill_match is not None:
        return PromptVariantSpec(
            template_name=template_name,
            system_template=_system_prompt_template(fill_match.group("base")),
            user_padding_spaces=int(fill_match.group("n")),
            user_padding_unit=FILLER_UNIT,
        )

    pad_match = _PAD_TEMPLATE_RE.fullmatch(template_name)
    if pad_match is not None:
        base = pad_match.group("base")
        n_spaces = int(pad_match.group("n"))
        if n_spaces < 0:
            raise ValueError(f"Invalid padding count in template name: {template_name}")
        return PromptVariantSpec(
            template_name=template_name,
            system_template=_system_prompt_template(base),
            user_padding_spaces=n_spaces,
        )

    return PromptVariantSpec(
        template_name=template_name,
        system_template=_system_prompt_template(template_name),
    )


def _pair_messages(
    system_prompt: Optional[str],
    option_a: str,
    option_b: str,
    instruction: str,
    *,
    prefill_assistant: bool = True,
    user_padding_spaces: int = 0,
    user_padding_unit: str = " ",
    user_preamble: Optional[str] = None,
):
    """Build chat messages for an AB prompt.

    Fireworks callers keep ``prefill_assistant=True`` (assistant content ``ANSWER:``).
    OpenRouter callers must pass ``prefill_assistant=False``: no provider honours the
    prefill, and several derail completely when given one — see
    ``api_backends.backend_uses_assistant_prefill``.
    Local logit callers must pass ``prefill_assistant=False`` and let
    ``build_chat_input`` open a single assistant turn via
    ``add_generation_prompt=True`` + appended ``ANSWER:``. Combining both
    produces a closed assistant turn followed by a second open one.

    If ``system_prompt`` is None/empty, the system role is omitted.
    If ``user_preamble`` is set, it is placed before the question in the user turn.
    If ``user_padding_spaces`` > 0, that many copies of ``user_padding_unit``
    (an ASCII space by default) are appended after the
    question (after any preamble+question assembly, still inside the user turn).
    """
    question = f"{option_a} or {option_b}? {instruction}"
    if user_padding_spaces:
        question = question + (user_padding_unit * user_padding_spaces)
    if user_preamble:
        user_prompt = f"{user_preamble.rstrip()}\n\n{question}"
    else:
        user_prompt = question

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if prefill_assistant:
        messages.append({"role": "assistant", "content": ANSWER_PREFIX})
    return user_prompt, messages


def _no_instruction_messages(system_prompt: str, option_a: str, option_b: str):
    user_prompt = f"{option_a} or {option_b}?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return user_prompt, messages


def _clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@contextmanager
def _loaded_model_and_tokenizer(model_name: str, dtype):
    model, tokenizer = load_hf_model_and_tokenizer(model_name, dtype)
    try:
        yield model, tokenizer
    finally:
        del model
        del tokenizer
        _clear_torch_memory()


def _run_local_pair_experiment(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    system_prompt_template: str,
    results_dir: Path,
    extra_metadata: Optional[Dict] = None,
    *,
    user_preamble_template: Optional[str] = None,
    user_padding_spaces: int = 0,
    user_padding_unit: str = " ",
    omit_system: bool = False,
) -> Path:
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        return _run_local_pair_experiment_with_loaded_model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
            system_prompt_template=system_prompt_template,
            results_dir=results_dir,
            extra_metadata=extra_metadata,
            user_preamble_template=user_preamble_template,
            user_padding_spaces=user_padding_spaces,
            user_padding_unit=user_padding_unit,
            omit_system=omit_system,
        )


def _run_local_pair_experiment_with_loaded_model(
    model,
    tokenizer,
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    system_prompt_template: str,
    results_dir: Path,
    extra_metadata: Optional[Dict] = None,
    *,
    user_preamble_template: Optional[str] = None,
    user_padding_spaces: int = 0,
    user_padding_unit: str = " ",
    omit_system: bool = False,
) -> Path:
    format_kwargs = {
        "certainty_instruction": certainty_instruction,
        "uncertainty_instruction": uncertainty_instruction,
    }
    system_prompt = None if omit_system else system_prompt_template.format(**format_kwargs)
    user_preamble = (
        user_preamble_template.format(**format_kwargs) if user_preamble_template is not None else None
    )
    ab_pairs = load_ab_pairs()
    assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
    results = []
    for pair in tqdm(ab_pairs):
        for flip_ab in [True, False]:
            options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
            for instruction in [certainty_instruction, uncertainty_instruction]:
                user_prompt, messages = _pair_messages(
                    system_prompt,
                    options[0],
                    options[1],
                    instruction,
                    prefill_assistant=False,
                    user_padding_spaces=user_padding_spaces,
                    user_padding_unit=user_padding_unit,
                    user_preamble=user_preamble,
                )
                text = build_chat_input(tokenizer, model_name, messages)
                first_device = next(model.parameters()).device
                inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                token_a_id, prob_a, decoded_a = option_token_probability(tokenizer, probs, options[0])
                token_b_id, prob_b, decoded_b = option_token_probability(tokenizer, probs, options[1])
                results.append(
                    {
                        "option_a": options[0],
                        "option_b": options[1],
                        "decoded_option_a": decoded_a,
                        "decoded_option_b": decoded_b,
                        "instruction": instruction,
                        "option_a_prob": prob_a,
                        "option_b_prob": prob_b,
                        "user_prompt": user_prompt,
                        "system_prompt": system_prompt or "",
                        "flip_ab": flip_ab,
                        "option_a_token_id": token_a_id,
                        "option_b_token_id": token_b_id,
                        **(extra_metadata or {}),
                    }
                )

    model_dir = slugify_model_name(model_name)
    pair_dir = results_dir / model_dir / f"{instruction_label(certainty_instruction, uncertainty_instruction)}_{timestamp()}"
    output_path = pair_dir / "results.json"
    save_json(output_path, results)
    return output_path


def run_zero_shot_local(model_name: str, certainty_instruction: str, uncertainty_instruction: str) -> Path:
        return _run_local_pair_experiment(
        model_name=model_name,
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
        system_prompt_template=read_txt(PROMPTS_DIR / "standard_system.txt"),
        results_dir=get_results_dir() / "zero_shot_local",
    )


def run_zero_shot_local_batch(
    model_name: str,
    instruction_pairs: List[Tuple[str, str]],
    force: bool = False,
) -> List[Path]:
    system_prompt_template = read_txt(PROMPTS_DIR / "standard_system.txt")
    results_dir = get_results_dir() / "zero_shot_local"
    output_paths: List[Path] = []
    pending_pairs: List[Tuple[str, str]] = []
    for certainty_instruction, uncertainty_instruction in instruction_pairs:
        existing_result = latest_zero_shot_local_result(model_name, certainty_instruction, uncertainty_instruction)
        if existing_result is not None and not force:
            print(
                f"  -> skipping existing result for {certainty_instruction} vs {uncertainty_instruction}: "
                f"{existing_result}"
            )
            output_paths.append(existing_result)
            continue
        pending_pairs.append((certainty_instruction, uncertainty_instruction))

    if not pending_pairs:
        return output_paths

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        for certainty_instruction, uncertainty_instruction in pending_pairs:
            output_paths.append(
                _run_local_pair_experiment_with_loaded_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    certainty_instruction=certainty_instruction,
                    uncertainty_instruction=uncertainty_instruction,
                    system_prompt_template=system_prompt_template,
                    results_dir=results_dir,
                )
            )
    return output_paths


def latest_zero_shot_local_result(model_name: str, certainty_instruction: str, uncertainty_instruction: str) -> Optional[Path]:
    parent = get_results_dir() / "zero_shot_local" / slugify_model_name(model_name)
    match = latest_matching_dir(parent, instruction_label(certainty_instruction, uncertainty_instruction) + "_")
    return None if match is None else match / "results.json"


def _get_fireworks_client() -> OpenAI:
    """Backward-compatible alias; prefer get_api_client('fireworks')."""
    return get_api_client("fireworks")


def _make_api_request_with_retry(
    client: OpenAI,
    model_name: str,
    messages: List[Dict],
    max_retries: int = 5,
    api_backend: Optional[str] = None,
    api_provider: Optional[str] = None,
):
    backend = resolve_api_backend(api_backend)
    create_kwargs = build_chat_completion_kwargs(
        model_name=model_name,
        messages=messages,
        backend=backend,
        provider=api_provider,
    )
    for attempt in range(max_retries):
        try:
            response, provenance = create_chat_completion_with_provenance(
                client, create_kwargs, backend=backend
            )
        except Exception as exc:
            if "RATE_LIMIT_EXCEEDED" in str(exc) or "rate_limit" in str(exc).lower() or "429" in str(exc):
                wait_time = 30 * (2 ** attempt) + random.uniform(0, 5)
                print(f"Rate limited. Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise
        if _response_has_logprobs(response) or attempt == max_retries - 1:
            return response, provenance
        # Some providers drop the logprobs block on an otherwise-200 response. That is
        # intermittent, so retry briefly rather than recording an unusable trial.
        time.sleep(2 ** attempt + random.uniform(0, 1))
    raise RuntimeError("Max retries exceeded for API request.")


def _response_has_logprobs(response) -> bool:
    try:
        logprobs = getattr(response.choices[0], "logprobs", None)
    except (AttributeError, IndexError):
        return False
    return bool(logprobs and getattr(logprobs, "content", None))


def _option_prefix_match_length(token: str, option: str) -> Optional[int]:
    """Length of the match if ``token`` could be ``option``'s first token, else None.

    Mirrors the local path, which scores the first token of ``" " + option``
    (``utils.option_first_token_id``). Over the API there is no tokenizer, so the
    option's first token has to be identified from the returned strings: BPE takes the
    longest vocabulary entry matching a prefix, so among the tokens actually present
    the longest prefix is the best estimate. Leading whitespace is ignored because
    providers differ on whether the answer token carries it.
    """
    stripped = token.strip()
    if not stripped or not option.startswith(stripped):
        return None
    return len(stripped)


# Trailing end-of-turn markers that some providers append to the logprob sequence but
# not to the message content (<|eot_id|>, <end_of_turn>, <｜end▁of▁sentence｜>, <turn|>).
# Stripped before checking that the tokens reconstruct the content.
_EOS_TOKEN_RE = re.compile(r"^\s*<[^>]*>\s*$")

# Leading "ANSWER:" the model emits itself when the assistant prefill is not honoured.
_ANSWER_PREFIX_RE = re.compile(r"^\s*" + re.escape(ANSWER_PREFIX) + r"\s*", re.IGNORECASE)


def _answer_token_index(token_infos, joined: str) -> Tuple[Optional[int], Optional[str]]:
    """Index of the logprob entry holding the answer token.

    The answer is the first non-whitespace character after any echoed ``ANSWER:``
    prefix. Providers tokenize that prefix differently (``ANSWER``/``:`` vs
    ``ANS``/``WER``/``:``), so locate it by character offset rather than by counting
    tokens. Returns ``(index, None)`` or ``(None, reason)``.
    """
    match = _ANSWER_PREFIX_RE.match(joined)
    offset = match.end() if match else 0
    while offset < len(joined) and joined[offset].isspace():
        offset += 1
    if offset >= len(joined):
        return None, "no_answer_after_prefix"

    pos = 0
    for idx, info in enumerate(token_infos):
        end = pos + len(info.token)
        if pos <= offset < end:
            if info.token[: offset - pos].strip():
                # e.g. a single ": Y" token: its top_logprobs range over the merged
                # unit, so option probabilities read off it are not comparable.
                return None, "answer_merged_with_prefix_token"
            # A leading space belongs to the answer token itself (" Y"), which is why
            # the token is located by containment rather than by exact offset match.
            return idx, None
        pos = end
    return None, "answer_token_not_found"


def _extract_option_probabilities(response, option_a: str, option_b: str) -> Dict:
    """Read P(option_a) / P(option_b) off the answer token's top_logprobs.

    Guards against every way providers were observed to silently corrupt this:
      * no logprobs block at all (StreamLake drops it on an otherwise-200 response),
      * logprobs for the final token only, or a dropped token, so the sequence cannot
        reconstruct the message content (CoreWeave, Novita),
      * top_logprobs copied from the preceding position, leaving the answer token
        absent from its own list (Google Vertex),
      * the answer sitting at an index other than 0 because the model re-emitted the
        ``ANSWER:`` prefix itself (every OpenRouter provider — see
        ``api_backends.backend_uses_assistant_prefill``).
    ``logprob_status`` is ``"ok"`` only when the answer token was located and read;
    callers must treat anything else as a failed trial, not as probability 0.
    """
    choice = response.choices[0]
    generated_content = choice.message.content or ""
    result: Dict = {
        "option_a_prob": 0.0,
        "option_b_prob": 0.0,
        "generated_token": None,
        "generated_logprob": None,
        "generated_content": generated_content,
        "top_2_tokens": [],
        "top_2_probs": [],
        "logprob_status": "ok",
    }

    logprobs = getattr(choice, "logprobs", None)
    token_infos = list(getattr(logprobs, "content", None) or []) if logprobs else []
    if not token_infos:
        result["logprob_status"] = "no_logprobs"
        return result

    # Providers that return only the final token cannot reconstruct the content, so
    # the answer token is simply not in the response.
    while token_infos and _EOS_TOKEN_RE.match(token_infos[-1].token):
        token_infos = token_infos[:-1]
    joined = "".join(info.token for info in token_infos)
    if joined.strip() != generated_content.strip():
        result["logprob_status"] = "tokens_do_not_match_content"
        return result

    idx, reason = _answer_token_index(token_infos, joined)
    if idx is None:
        result["logprob_status"] = reason
        return result

    answer_info = token_infos[idx]
    result["generated_token"] = answer_info.token
    result["generated_logprob"] = answer_info.logprob
    tops = list(answer_info.top_logprobs or [])
    # The sampled token must appear in its own top_logprobs at its own logprob. Google
    # Vertex violates this: it repeats the previous position's list, so the answer
    # token's distribution is silently that of the ":" before it.
    if tops and not any(
        t.token == answer_info.token and abs(t.logprob - answer_info.logprob) < 1e-6 for t in tops
    ):
        result["logprob_status"] = "top_logprobs_inconsistent_with_sampled_token"
        return result
    # Best (longest-prefix) candidate for each option's first token: (match_len, idx, prob).
    best_a = None
    best_b = None
    for idx_top, top_logprob in enumerate(tops):
        prob = math.exp(top_logprob.logprob)
        result["top_2_tokens"].append(top_logprob.token)
        result["top_2_probs"].append(prob)
        len_a = _option_prefix_match_length(top_logprob.token, option_a)
        len_b = _option_prefix_match_length(top_logprob.token, option_b)
        # Strict > keeps the first of equally-long matches, and top_logprobs is ordered
        # by descending probability, so ties resolve to the more probable spelling
        # (" X" over "X").
        if len_a is not None and (best_a is None or len_a > best_a[0]):
            best_a = (len_a, idx_top, prob)
        if len_b is not None and (best_b is None or len_b > best_b[0]):
            best_b = (len_b, idx_top, prob)

    if not result["top_2_tokens"]:
        result["logprob_status"] = "no_top_logprobs"
        return result
    if best_a is None and best_b is None:
        # The model answered off-format; neither option is a candidate at this position.
        result["logprob_status"] = "options_absent_from_top_logprobs"
        return result
    if best_a is not None and best_b is not None and best_a[1] == best_b[1]:
        # One token is the best first-token candidate for both options, so its mass
        # cannot be attributed. This is the API-side counterpart of the local
        # ``assert_ab_pairs_distinct_first_tokens`` guard.
        result["logprob_status"] = "ambiguous_option_prefix"
        return result

    result["option_a_prob"] = best_a[2] if best_a else 0.0
    result["option_b_prob"] = best_b[2] if best_b else 0.0
    result["option_a_token"] = tops[best_a[1]].token if best_a else None
    result["option_b_token"] = tops[best_b[1]].token if best_b else None
    # One option outside top-k is legitimate for a sharply peaked distribution, but it
    # floors that option at 0; record it so analyses can filter or widen top_logprobs.
    result["both_options_in_top_logprobs"] = best_a is not None and best_b is not None
    return result


def _resolve_api_run_config(
    model_name: str,
    api_backend: Optional[str] = None,
    api_provider: Optional[str] = None,
) -> Tuple[str, str, Optional[str], OpenAI, Optional[Dict]]:
    """Return (backend, request_model, endpoint_tag, client, endpoints_snapshot)."""
    backend = resolve_api_backend(api_backend)
    endpoint_tag = None
    endpoints_snapshot = None
    request_model = model_name
    if backend == "openrouter":
        request_model = resolve_openrouter_model(model_name)
        endpoint_tag = resolve_openrouter_endpoint(model_name, api_provider)
        endpoints_snapshot = fetch_openrouter_endpoints_snapshot(request_model)
        print(f"Connecting to OpenRouter for model: {request_model} (endpoint={endpoint_tag})")
    else:
        print(f"Connecting to Fireworks API for model: {model_name}")
    return backend, request_model, endpoint_tag, get_api_client(backend), endpoints_snapshot


def _api_routing_fields(backend: str, endpoint_tag: Optional[str], provenance: Optional[Dict] = None) -> Dict:
    fields: Dict = {}
    if backend == "openrouter":
        fields["api_backend"] = backend
        fields["api_endpoint"] = endpoint_tag
        fields["api_provider"] = endpoint_tag
        if provenance:
            fields["generation_id"] = provenance.get("generation_id")
            fields["served_provider"] = provenance.get("served_provider")
            fields["response_model"] = provenance.get("response_model")
            fields["system_fingerprint"] = provenance.get("system_fingerprint")
            fields["openrouter_metadata"] = provenance.get("openrouter_metadata")
    return fields


def _api_run_config_payload(
    *,
    experiment: str,
    model_name: str,
    request_model: str,
    backend: str,
    endpoint_tag: Optional[str],
    endpoints_snapshot: Optional[Dict],
    certainty_instruction: str,
    uncertainty_instruction: str,
    api_concurrency: int,
    extra: Optional[Dict] = None,
) -> Dict:
    payload: Dict = {
        "experiment": experiment,
        "model": request_model,
        "model_input": model_name,
        "api_backend": backend,
        "api_provider": endpoint_tag,
        "api_endpoint": endpoint_tag,
        "api_concurrency": api_concurrency,
        "certainty_instruction": certainty_instruction,
        "uncertainty_instruction": uncertainty_instruction,
    }
    if endpoints_snapshot is not None:
        payload["openrouter_endpoints_snapshot"] = endpoints_snapshot
    if extra:
        payload.update(extra)
    return payload


def _build_api_ab_jobs(
    ab_pairs: List[Dict],
    *,
    certainty_instruction: str,
    uncertainty_instruction: str,
    system_prompt: Optional[str],
    backend: str,
    user_padding_spaces: int = 0,
    user_padding_unit: str = " ",
    user_preamble: Optional[str] = None,
    extra_row_fields: Optional[Dict] = None,
) -> List[Dict]:
    """Build ordered API trial jobs matching the historical nested-loop order."""
    prefill_assistant = backend_uses_assistant_prefill(backend)
    jobs: List[Dict] = []
    for pair in ab_pairs:
        for flip_ab in [True, False]:
            options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
            for instruction in [certainty_instruction, uncertainty_instruction]:
                user_prompt, messages = _pair_messages(
                    system_prompt,
                    options[0],
                    options[1],
                    instruction,
                    prefill_assistant=prefill_assistant,
                    user_padding_spaces=user_padding_spaces,
                    user_padding_unit=user_padding_unit,
                    user_preamble=user_preamble,
                )
                jobs.append(
                    {
                        "option_a": options[0],
                        "option_b": options[1],
                        "instruction": instruction,
                        "flip_ab": flip_ab,
                        "user_prompt": user_prompt,
                        "system_prompt": system_prompt or "",
                        "messages": messages,
                        "extra_row_fields": dict(extra_row_fields or {}),
                    }
                )
    return jobs


def _run_one_api_ab_job(
    job: Dict,
    *,
    client: OpenAI,
    model_name: str,
    backend: str,
    endpoint_tag: Optional[str],
) -> Dict:
    option_a = job["option_a"]
    option_b = job["option_b"]
    try:
        response, provenance = _make_api_request_with_retry(
            client,
            model_name,
            job["messages"],
            api_backend=backend,
            api_provider=endpoint_tag,
        )
        extracted = _extract_option_probabilities(response, option_a, option_b)
        row = {
            "option_a": option_a,
            "option_b": option_b,
            "decoded_option_a": option_a,
            "decoded_option_b": option_b,
            "instruction": job["instruction"],
            "user_prompt": job["user_prompt"],
            "system_prompt": job["system_prompt"],
            "flip_ab": job["flip_ab"],
            **extracted,
        }
        if extracted["logprob_status"] != "ok":
            # Never let an unreadable response masquerade as P(a)=P(b)=0.
            row["error"] = f"unusable logprobs: {extracted['logprob_status']}"
        row.update(job["extra_row_fields"])
        row.update(_api_routing_fields(backend, endpoint_tag, provenance))
        return row
    except Exception as exc:
        row = {
            "option_a": option_a,
            "option_b": option_b,
            "decoded_option_a": option_a,
            "decoded_option_b": option_b,
            "instruction": job["instruction"],
            "option_a_prob": 0.0,
            "option_b_prob": 0.0,
            "user_prompt": job["user_prompt"],
            "system_prompt": job["system_prompt"],
            "flip_ab": job["flip_ab"],
            "logprob_status": "request_failed",
            "error": str(exc),
        }
        row.update(job["extra_row_fields"])
        row.update(_api_routing_fields(backend, endpoint_tag))
        return row


def _run_api_ab_jobs(
    jobs: List[Dict],
    *,
    client: OpenAI,
    model_name: str,
    backend: str,
    endpoint_tag: Optional[str],
    api_concurrency: int,
) -> List[Dict]:
    """Execute API jobs with bounded concurrency; preserve input order in results."""
    n = len(jobs)
    if n == 0:
        return []
    print(f"Running {n} API trials with concurrency={api_concurrency}")
    results: List[Optional[Dict]] = [None] * n

    if api_concurrency <= 1:
        for idx, job in enumerate(tqdm(jobs, desc="Processing")):
            results[idx] = _run_one_api_ab_job(
                job,
                client=client,
                model_name=model_name,
                backend=backend,
                endpoint_tag=endpoint_tag,
            )
    else:
        with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
            future_to_idx = {
                executor.submit(
                    _run_one_api_ab_job,
                    job,
                    client=client,
                    model_name=model_name,
                    backend=backend,
                    endpoint_tag=endpoint_tag,
                ): idx
                for idx, job in enumerate(jobs)
            }
            for future in tqdm(as_completed(future_to_idx), total=n, desc="Processing"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    return [row for row in results if row is not None]


def run_zero_shot_api(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    api_backend: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_concurrency: Optional[int] = None,
) -> Path:
    backend, request_model, endpoint_tag, client, endpoints_snapshot = _resolve_api_run_config(
        model_name, api_backend, api_provider
    )
    concurrency = resolve_api_concurrency(api_concurrency)
    system_prompt = read_txt(PROMPTS_DIR / "standard_system.txt").format(
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
    )
    ab_pairs = load_ab_pairs()
    jobs = _build_api_ab_jobs(
        ab_pairs,
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
        system_prompt=system_prompt,
        backend=backend,
    )
    results = _run_api_ab_jobs(
        jobs,
        client=client,
        model_name=model_name,
        backend=backend,
        endpoint_tag=endpoint_tag,
        api_concurrency=concurrency,
    )

    model_dir = slugify_model_name(request_model)
    pair_dir = get_results_dir() / "zero_shot_api" / model_dir / f"{instruction_label(certainty_instruction, uncertainty_instruction)}_{timestamp()}"
    output_path = pair_dir / "results.json"
    save_json(output_path, results)
    save_json(
        pair_dir / "config.json",
        _api_run_config_payload(
            experiment="zero_shot_api",
            model_name=model_name,
            request_model=request_model,
            backend=backend,
            endpoint_tag=endpoint_tag,
            endpoints_snapshot=endpoints_snapshot,
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
            api_concurrency=concurrency,
        ),
    )
    return output_path


def latest_zero_shot_api_result(model_name: str, certainty_instruction: str, uncertainty_instruction: str) -> Optional[Path]:
    # Check both the requested name and OpenRouter-resolved alias so status works
    # for Fireworks IDs when results were written under OpenRouter slugs.
    candidates = [model_name]
    try:
        resolved = resolve_openrouter_model(model_name)
        if resolved not in candidates:
            candidates.append(resolved)
    except Exception:
        pass
    for candidate in candidates:
        parent = get_results_dir() / "zero_shot_api" / slugify_model_name(candidate)
        match = latest_matching_dir(parent, instruction_label(certainty_instruction, uncertainty_instruction) + "_")
        if match is not None:
            return match / "results.json"
    return None


def run_zero_shot_api_variant(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str,
    api_backend: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_concurrency: Optional[int] = None,
) -> Path:
    backend, request_model, endpoint_tag, client, endpoints_snapshot = _resolve_api_run_config(
        model_name, api_backend, api_provider
    )
    concurrency = resolve_api_concurrency(api_concurrency)
    spec = resolve_prompt_variant(template_name)
    format_kwargs = {
        "certainty_instruction": certainty_instruction,
        "uncertainty_instruction": uncertainty_instruction,
    }
    system_prompt = None if spec.omit_system else (spec.system_template or "").format(**format_kwargs)
    user_preamble = (
        spec.user_preamble_template.format(**format_kwargs) if spec.user_preamble_template is not None else None
    )
    ab_pairs = load_ab_pairs()
    jobs = _build_api_ab_jobs(
        ab_pairs,
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
        system_prompt=system_prompt,
        backend=backend,
        user_padding_spaces=spec.user_padding_spaces,
        user_padding_unit=spec.user_padding_unit,
        user_preamble=user_preamble,
        extra_row_fields={
            "system_prompt_template": template_name,
            "user_padding_spaces": spec.user_padding_spaces,
            "user_padding_unit": spec.user_padding_unit,
        },
    )
    results = _run_api_ab_jobs(
        jobs,
        client=client,
        model_name=model_name,
        backend=backend,
        endpoint_tag=endpoint_tag,
        api_concurrency=concurrency,
    )

    model_dir = slugify_model_name(request_model)
    pair_dir = get_results_dir() / "prompt_variants_api" / template_name / model_dir / f"{instruction_label(certainty_instruction, uncertainty_instruction)}_{timestamp()}"
    output_path = pair_dir / "results.json"
    save_json(output_path, results)
    save_json(
        pair_dir / "config.json",
        _api_run_config_payload(
            experiment="zero_shot_api_variant",
            model_name=model_name,
            request_model=request_model,
            backend=backend,
            endpoint_tag=endpoint_tag,
            endpoints_snapshot=endpoints_snapshot,
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
            api_concurrency=concurrency,
            extra={"template_name": template_name},
        ),
    )
    return output_path


def latest_zero_shot_api_variant_result(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str,
) -> Optional[Path]:
    candidates = [model_name]
    try:
        resolved = resolve_openrouter_model(model_name)
        if resolved not in candidates:
            candidates.append(resolved)
    except Exception:
        pass
    for candidate in candidates:
        parent = get_results_dir() / "prompt_variants_api" / template_name / slugify_model_name(candidate)
        match = latest_matching_dir(parent, instruction_label(certainty_instruction, uncertainty_instruction) + "_")
        if match is not None:
            return match / "results.json"
    return None


def run_prompt_variant(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str,
) -> Path:
    spec = resolve_prompt_variant(template_name)
    return _run_local_pair_experiment(
        model_name=model_name,
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
        system_prompt_template=spec.system_template or "",
        results_dir=get_results_dir() / "prompt_variants" / template_name,
        extra_metadata={
            "system_prompt_template": template_name,
            "user_padding_spaces": spec.user_padding_spaces,
            "user_padding_unit": spec.user_padding_unit,
        },
        user_preamble_template=spec.user_preamble_template,
        user_padding_spaces=spec.user_padding_spaces,
        user_padding_unit=spec.user_padding_unit,
        omit_system=spec.omit_system,
    )


def run_prompt_variant_batch(
    model_name: str,
    template_name: str,
    instruction_pairs: List[Tuple[str, str]],
    force: bool = False,
) -> List[Path]:
    """Load the model once and evaluate all pending instruction pairs.

    Re-loading a ~70B ``device_map=auto`` model in the same process after
    ``del`` often OOMs; batching avoids that and is much faster.
    """
    spec = resolve_prompt_variant(template_name)
    results_dir = get_results_dir() / "prompt_variants" / template_name
    output_paths: List[Path] = []
    pending_pairs: List[Tuple[str, str]] = []
    for certainty_instruction, uncertainty_instruction in instruction_pairs:
        existing_result = latest_prompt_variant_result(
            model_name, certainty_instruction, uncertainty_instruction, template_name
        )
        if existing_result is not None and not force:
            print(
                f"  -> skipping existing result for {certainty_instruction} vs {uncertainty_instruction}: "
                f"{existing_result}"
            )
            output_paths.append(existing_result)
            continue
        pending_pairs.append((certainty_instruction, uncertainty_instruction))

    if not pending_pairs:
        return output_paths

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        for certainty_instruction, uncertainty_instruction in pending_pairs:
            print(f"  running {certainty_instruction} vs {uncertainty_instruction}")
            output_paths.append(
                _run_local_pair_experiment_with_loaded_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    certainty_instruction=certainty_instruction,
                    uncertainty_instruction=uncertainty_instruction,
                    system_prompt_template=spec.system_template or "",
                    results_dir=results_dir,
                    extra_metadata={
                        "system_prompt_template": template_name,
                        "user_padding_spaces": spec.user_padding_spaces,
                        "user_padding_unit": spec.user_padding_unit,
                    },
                    user_preamble_template=spec.user_preamble_template,
                    user_padding_spaces=spec.user_padding_spaces,
                    user_padding_unit=spec.user_padding_unit,
                    omit_system=spec.omit_system,
                )
            )
    return output_paths


def latest_prompt_variant_result(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str,
) -> Optional[Path]:
    parent = get_results_dir() / "prompt_variants" / template_name / slugify_model_name(model_name)
    match = latest_matching_dir(parent, instruction_label(certainty_instruction, uncertainty_instruction) + "_")
    return None if match is None else match / "results.json"


def _system_prompt_template(template_name: str) -> str:
    if template_name == "standard":
        return read_txt(PROMPTS_DIR / "standard_system.txt")
    return read_txt(PROMPTS_DIR / "variants" / f"{template_name}.txt")


def _activation_artifacts_dir(template_name: str = "standard") -> Path:
    base = get_steering_artifacts_dir() / "activations"
    return base if template_name == "standard" else base / template_name


def _steering_vectors_dir(template_name: str = "standard") -> Path:
    base = get_steering_artifacts_dir() / "steering_vectors"
    return base if template_name == "standard" else base / template_name


def extract_multi_layer_activations(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    layer_step: int = 4,
    layer_indices: Optional[List[int]] = None,
    template_name: str = "standard",
) -> Path:
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        num_layers = get_num_layers(model)
        if layer_indices is None:
            layer_indices = get_layer_indices_every_n(num_layers, layer_step)
        system_prompt = _system_prompt_template(template_name).format(
            certainty_instruction=certainty_instruction,
            uncertainty_instruction=uncertainty_instruction,
        )
        extractor = MultiLayerActivationExtractor(model, layer_indices)
        activations_dict = {}
        ab_pairs = load_ab_pairs()
        assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
        with extractor:
            for pair in tqdm(ab_pairs):
                for flip_ab in [False, True]:
                    options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
                    for instruction in [certainty_instruction, uncertainty_instruction]:
                        user_prompt, messages = _pair_messages(system_prompt, options[0], options[1], instruction, prefill_assistant=False)
                        text = build_chat_input(tokenizer, model_name, messages)
                        first_device = next(model.parameters()).device
                        inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                        with torch.no_grad():
                            _ = model(**inputs)
                        activations_dict[user_prompt] = {
                            layer_name: activation.squeeze(0)
                            for layer_name, activation in extractor.get_activations_as_dict().items()
                        }
                        extractor.clear()

        output_path = (
            _activation_artifacts_dir(template_name)
            / f"{slugify_model_name(model_name)}_{instruction_pair_label(certainty_instruction, uncertainty_instruction)}_activations.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations": activations_dict,
                "metadata": {
                    "model_name": model_name,
                    "certainty_instruction": certainty_instruction,
                    "uncertainty_instruction": uncertainty_instruction,
                    "num_layers": num_layers,
                    "layer_indices": layer_indices,
                    "layer_step": layer_step,
                    "template_name": template_name,
                },
            },
            output_path,
        )
        return output_path


def latest_activation_artifact(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str = "standard",
) -> Optional[Path]:
    path = (
        _activation_artifacts_dir(template_name)
        / f"{slugify_model_name(model_name)}_{instruction_pair_label(certainty_instruction, uncertainty_instruction)}_activations.pt"
    )
    return path if path.exists() else None


def compute_condition_cosine_similarities_from_artifact(
    artifact_path: Path,
    certainty_instruction: str,
    uncertainty_instruction: str,
) -> Dict[int, float]:
    """Per-layer cosine similarity between mean certainty and mean uncertainty activations."""
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    grouped_by_layer: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    for user_prompt, layer_activations in payload["activations"].items():
        if "?" not in user_prompt:
            continue
        instruction = user_prompt.rsplit("?", maxsplit=1)[-1].strip()
        if instruction not in {certainty_instruction, uncertainty_instruction}:
            continue
        for layer_name, activation in layer_activations.items():
            layer_idx = int(layer_name.replace("layer_", ""))
            layer_group = grouped_by_layer.setdefault(
                layer_idx,
                {certainty_instruction: [], uncertainty_instruction: []},
            )
            layer_group[instruction].append(activation.float().reshape(-1))

    similarities: Dict[int, float] = {}
    for layer_idx in payload["metadata"]["layer_indices"]:
        layer_group = grouped_by_layer.get(layer_idx)
        if layer_group is None:
            continue
        certainty_activations = layer_group[certainty_instruction]
        uncertainty_activations = layer_group[uncertainty_instruction]
        if not certainty_activations or not uncertainty_activations:
            continue
        certainty_mean = torch.stack(certainty_activations).mean(dim=0)
        uncertainty_mean = torch.stack(uncertainty_activations).mean(dim=0)
        similarities[layer_idx] = float(
            F.cosine_similarity(certainty_mean.unsqueeze(0), uncertainty_mean.unsqueeze(0)).item()
        )
    return similarities


def _parse_user_prompt(user_prompt: str) -> Tuple[str, str, str]:
    options_part, instruction = user_prompt.split("?")
    option_a, option_b = options_part.strip().split(" or ")
    return option_a.strip(), option_b.strip(), instruction.strip()


def compute_all_steering_vectors(template_name: str = "standard") -> List[Path]:
    activations_dir = _activation_artifacts_dir(template_name)
    if not activations_dir.exists():
        return []
    output_paths = []
    for activation_path in sorted(activations_dir.glob("*_activations.pt")):
        payload = torch.load(activation_path, map_location="cpu", weights_only=False)
        activations_dict = payload["activations"]
        metadata = payload["metadata"]
        grouped = defaultdict(lambda: {"certainty": None, "uncertainty": None})
        for user_prompt, layer_activations in activations_dict.items():
            option_a, option_b, instruction = _parse_user_prompt(user_prompt)
            pair_key = f"{option_a}_or_{option_b}"
            if instruction == metadata["certainty_instruction"]:
                grouped[pair_key]["certainty"] = layer_activations
            elif instruction == metadata["uncertainty_instruction"]:
                grouped[pair_key]["uncertainty"] = layer_activations

        model_dir = slugify_model_name(metadata["model_name"])
        pair_dir = instruction_pair_label(metadata["certainty_instruction"], metadata["uncertainty_instruction"])
        artifact_template_name = metadata.get("template_name", template_name)
        steering_dir = _steering_vectors_dir(artifact_template_name) / model_dir / pair_dir
        steering_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in metadata["layer_indices"]:
            layer_name = f"layer_{layer_idx}"
            differences = []
            for pair_data in grouped.values():
                certainty = pair_data["certainty"]
                uncertainty = pair_data["uncertainty"]
                if certainty is None or uncertainty is None:
                    continue
                if layer_name not in certainty or layer_name not in uncertainty:
                    continue
                differences.append(certainty[layer_name] - uncertainty[layer_name])
            if not differences:
                continue
            steering_vector = torch.stack(differences).mean(dim=0)
            output_path = steering_dir / f"{layer_name}.pt"
            torch.save(
                {
                    "steering_vector": steering_vector,
                    "metadata": {
                        "model_name": metadata["model_name"],
                        "layer_idx": layer_idx,
                        "certainty_instruction": metadata["certainty_instruction"],
                        "uncertainty_instruction": metadata["uncertainty_instruction"],
                        "num_pairs": len(differences),
                        "template_name": artifact_template_name,
                    },
                },
                output_path,
            )
            output_paths.append(output_path)
    return output_paths


def steering_vectors_exist(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str = "standard",
) -> bool:
    base = (
        _steering_vectors_dir(template_name)
        / slugify_model_name(model_name)
        / instruction_pair_label(certainty_instruction, uncertainty_instruction)
    )
    return base.exists() and any(base.glob("layer_*.pt"))


def _load_layer_vector(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    layer_idx: int,
    device,
    template_name: str = "standard",
):
    path = (
        _steering_vectors_dir(template_name)
        / slugify_model_name(model_name)
        / instruction_pair_label(certainty_instruction, uncertainty_instruction)
        / f"layer_{layer_idx}.pt"
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload["steering_vector"]


def _available_vector_layers(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
    template_name: str = "standard",
) -> List[int]:
    base = (
        _steering_vectors_dir(template_name)
        / slugify_model_name(model_name)
        / instruction_pair_label(certainty_instruction, uncertainty_instruction)
    )
    if not base.exists():
        return []
    layers = []
    for path in base.glob("layer_*.pt"):
        layers.append(int(path.stem.replace("layer_", "")))
    return sorted(layers)


def run_all_layers_steering(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    test_certainty_instruction: str,
    test_uncertainty_instruction: str,
    steering_strength: float,
) -> Path:
    layers = _available_vector_layers(model_name, vector_certainty_instruction, vector_uncertainty_instruction)
    if not layers:
        raise FileNotFoundError("No steering vectors found for the requested model/pair.")
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        system_prompt = read_txt(PROMPTS_DIR / "standard_system.txt").format(
            certainty_instruction=test_certainty_instruction,
            uncertainty_instruction=test_uncertainty_instruction,
        )
        ab_pairs = load_ab_pairs()
        assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
        all_results = {}
        for layer_idx in tqdm(layers, desc="Processing layers"):
            steering_vector = _load_layer_vector(
                model_name, vector_certainty_instruction, vector_uncertainty_instruction, layer_idx, device
            )
            injector = SteeringInjector(
                model,
                layer_idx,
                steering_vector,
                strength=steering_strength,
                certainty_instruction=test_certainty_instruction,
                uncertainty_instruction=test_uncertainty_instruction,
            )
            layer_results = []
            with injector:
                for pair in tqdm(ab_pairs, leave=False):
                    for flip_ab in [True, False]:
                        options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
                        for instruction in [test_certainty_instruction, test_uncertainty_instruction]:
                            injector.set_current_instruction(instruction)
                            user_prompt, messages = _pair_messages(system_prompt, options[0], options[1], instruction, prefill_assistant=False)
                            text = build_chat_input(tokenizer, model_name, messages)
                            first_device = next(model.parameters()).device
                            inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                            with torch.no_grad():
                                logits = model(**inputs).logits[0, -1, :]
                                probs = torch.softmax(logits, dim=-1)
                            _, prob_a, decoded_a = option_token_probability(tokenizer, probs, options[0])
                            _, prob_b, decoded_b = option_token_probability(tokenizer, probs, options[1])
                            layer_results.append(
                                {
                                    "option_a": options[0],
                                    "option_b": options[1],
                                    "decoded_option_a": decoded_a,
                                    "decoded_option_b": decoded_b,
                                    "instruction": instruction,
                                    "option_a_prob": prob_a,
                                    "option_b_prob": prob_b,
                                    "user_prompt": user_prompt,
                                    "flip_ab": flip_ab,
                                    "layer_idx": layer_idx,
                                    "steering_strength": steering_strength,
                                }
                            )
            all_results[f"layer_{layer_idx}"] = layer_results

        model_dir = slugify_model_name(model_name)
        vector_label = instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        test_label = instruction_pair_label(test_certainty_instruction, test_uncertainty_instruction)
        combined_dir = (
            get_results_dir()
            / "steering_all_layers"
            / model_dir
            / f"vector_{vector_label}_test_{test_label}_all_layers_strength{steering_strength}_{timestamp()}"
        )
        output_path = combined_dir / "all_layers_results.json"
        save_json(
            output_path,
            {
                "metadata": {
                    "model_name": model_name,
                    "vector_certainty_instruction": vector_certainty_instruction,
                    "vector_uncertainty_instruction": vector_uncertainty_instruction,
                    "test_certainty_instruction": test_certainty_instruction,
                    "test_uncertainty_instruction": test_uncertainty_instruction,
                    "steering_strength": steering_strength,
                    "available_layers": layers,
                },
                "results": all_results,
            },
        )
        return output_path


def latest_all_layers_result(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    test_certainty_instruction: str,
    test_uncertainty_instruction: str,
    steering_strength: float,
) -> Optional[Path]:
    parent = get_results_dir() / "steering_all_layers" / slugify_model_name(model_name)
    prefix = (
        f"vector_{instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)}"
        f"_test_{instruction_pair_label(test_certainty_instruction, test_uncertainty_instruction)}"
        f"_all_layers_strength{steering_strength}_"
    )
    match = latest_matching_dir(parent, prefix)
    return None if match is None else match / "all_layers_results.json"


def run_no_instruction_control(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    steering_strength: float,
    steering_conditions: List[str],
    fixed_layer: Optional[int] = None,
    normalize_steering_vector: bool = False,
    preserve_norm: bool = False,
) -> Path:
    # Ensure activation artifact exists — extract if missing.
    activation_path = latest_activation_artifact(
        model_name,
        vector_certainty_instruction,
        vector_uncertainty_instruction,
    )
    if activation_path is None:
        print(
            f"[auto-prepare] No activation artifact found for {model_name!r}; "
            "extracting activations now..."
        )
        activation_path = extract_multi_layer_activations(
            model_name,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
        )

    # Determine fixed layer from cosine similarities if not provided.
    if fixed_layer is None:
        similarities = compute_condition_cosine_similarities_from_artifact(
            activation_path,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
        )
        if not similarities:
            raise FileNotFoundError(
                f"Could not compute per-layer cosine similarities from {activation_path}."
            )
        fixed_layer = choose_min_cosine_layer_from_similarities(similarities)

    # Ensure the steering vector for the chosen layer exists — compute if missing.
    vector_path = (
        get_steering_artifacts_dir()
        / "steering_vectors"
        / slugify_model_name(model_name)
        / instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        / f"layer_{fixed_layer}.pt"
    )
    if not vector_path.exists():
        print(
            f"[auto-prepare] Steering vector not found at {vector_path}; "
            "computing steering vectors now..."
        )
        compute_all_steering_vectors()
        if not vector_path.exists():
            raise FileNotFoundError(
                f"Steering vector still missing after compute_all_steering_vectors(): {vector_path}. "
                "The activation artifact may not contain layer {fixed_layer}."
            )

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        steering_vector = _load_layer_vector(
            model_name,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
            fixed_layer,
            device,
        )
        if normalize_steering_vector:
            norm = torch.norm(steering_vector)
            if norm > 0:
                steering_vector = steering_vector / norm
        system_prompt = read_txt(PROMPTS_DIR / "no_instruction_system.txt")
        ab_pairs = load_ab_pairs()
        assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
        grouped_results = {}

        for condition in steering_conditions:
            if condition == "positive":
                effective_strength = steering_strength
            elif condition == "negative":
                effective_strength = -steering_strength
            else:
                effective_strength = 0.0
            context = (
                SteeringInjector(model, fixed_layer, steering_vector, strength=effective_strength, preserve_norm=preserve_norm)
                if effective_strength != 0.0
                else nullcontext()
            )
            condition_results = []
            with context:
                for pair in tqdm(ab_pairs, leave=False):
                    for flip_ab in [True, False]:
                        options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
                        user_prompt, messages = _no_instruction_messages(system_prompt, options[0], options[1])
                        text = build_chat_input(tokenizer, model_name, messages)
                        first_device = next(model.parameters()).device
                        inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                        with torch.no_grad():
                            logits = model(**inputs).logits[0, -1, :]
                            probs = torch.softmax(logits, dim=-1)
                        _, prob_a, decoded_a = option_token_probability(tokenizer, probs, options[0])
                        _, prob_b, decoded_b = option_token_probability(tokenizer, probs, options[1])
                        condition_results.append(
                            {
                                "option_a": options[0],
                                "option_b": options[1],
                                "decoded_option_a": decoded_a,
                                "decoded_option_b": decoded_b,
                                "steering_condition": condition,
                                "effective_strength": effective_strength,
                                "option_a_prob": prob_a,
                                "option_b_prob": prob_b,
                                "user_prompt": user_prompt,
                                "flip_ab": flip_ab,
                                "layer_idx": fixed_layer,
                            }
                        )
            grouped_results[condition] = condition_results

        output_dir = (
            get_results_dir()
            / "no_instruction_control"
            / slugify_model_name(model_name)
            / f"no_instruction_layer{fixed_layer}_strength{steering_strength}_{timestamp()}"
        )
        output_path = output_dir / "results.json"
        save_json(
            output_path,
            {
                "metadata": {
                    "model_name": model_name,
                    "vector_certainty_instruction": vector_certainty_instruction,
                    "vector_uncertainty_instruction": vector_uncertainty_instruction,
                    "fixed_layer": fixed_layer,
                    "steering_conditions": steering_conditions,
                    "steering_strength": steering_strength,
                    "normalize_steering_vector": normalize_steering_vector,
                    "preserve_norm": preserve_norm,
                },
                "results": grouped_results,
            },
        )
        return output_path


def latest_no_instruction_result(model_name: str, steering_strength: float) -> Optional[Path]:
    parent = get_results_dir() / "no_instruction_control" / slugify_model_name(model_name)
    if not parent.exists():
        return None
    matches = [
        path
        for path in parent.iterdir()
        if path.is_dir() and f"_strength{steering_strength}_" in path.name and path.name.startswith("no_instruction_layer")
    ]
    if not matches:
        return None
    match = max(matches, key=lambda path: path.stat().st_mtime)
    results_path = match / "results.json"
    if not results_path.exists():
        return None
    return results_path


MMLU_LABELS = ["A", "B", "C", "D"]
MMLU_SYSTEM_PROMPT = (
    "Answer the multiple-choice question with exactly one letter: A, B, C, or D. "
    "Do not include any explanation."
)
MMLU_SYSTEM_PROMPT_FORMAT_INSTRUCTED = (
    "Answer the multiple-choice question with exactly one letter: A, B, C, or D. "
    "Format your reply as: ANSWER: <letter>. "
    "Do not include any explanation."
)
# Prompt-only MMLU variants: system text lives in prompts/mmlu_*.txt (ANSWER: formatting).
MMLU_FILE_PROMPT_VARIANTS = {
    "maximize_certainty": "mmlu_maximize_certainty.txt",
    "maximize_entropy": "mmlu_maximize_entropy.txt",
    "sandbag_via_uncertainty": "mmlu_sandbag_via_uncertainty.txt",
}
DEFAULT_MMLU_PROMPT_CONDITIONS = list(MMLU_FILE_PROMPT_VARIANTS.keys())


def _batch_items(items: Sequence[Dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _normalize_mmlu_answer(answer) -> tuple[int, str]:
    if isinstance(answer, str):
        stripped = answer.strip().upper()
        if stripped in MMLU_LABELS:
            return MMLU_LABELS.index(stripped), stripped
        answer = int(stripped)
    answer_idx = int(answer)
    if answer_idx < 0 or answer_idx >= len(MMLU_LABELS):
        raise ValueError(f"Unexpected MMLU answer index: {answer!r}")
    return answer_idx, MMLU_LABELS[answer_idx]


def _load_mmlu_examples(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for Figure 7 MMLU runs. "
            "Install project dependencies with `uv sync`."
        ) from exc

    dataset_args = [dataset_name]
    if dataset_config:
        dataset_args.append(dataset_config)
    dataset = load_dataset(*dataset_args, split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    examples = []
    for idx, item in enumerate(dataset):
        choices = list(item["choices"])
        if len(choices) != len(MMLU_LABELS):
            raise ValueError(f"Expected 4 choices for MMLU example {idx}, got {len(choices)}")
        answer_idx, answer_label = _normalize_mmlu_answer(item["answer"])
        examples.append(
            {
                "example_idx": idx,
                "subject": item.get("subject"),
                "question": item["question"],
                "choices": choices,
                "answer_index": answer_idx,
                "answer_label": answer_label,
            }
        )
    return examples


def _mmlu_system_prompt(prompt_variant: str) -> str:
    if prompt_variant == "standard":
        return MMLU_SYSTEM_PROMPT
    if prompt_variant == "format_instructed":
        return MMLU_SYSTEM_PROMPT_FORMAT_INSTRUCTED
    if prompt_variant in MMLU_FILE_PROMPT_VARIANTS:
        return read_txt(PROMPTS_DIR / MMLU_FILE_PROMPT_VARIANTS[prompt_variant])
    raise ValueError(
        f"Unknown MMLU prompt_variant {prompt_variant!r}. "
        f"Expected one of: standard, format_instructed, {', '.join(MMLU_FILE_PROMPT_VARIANTS)}"
    )


def _mmlu_messages(example: Dict, prompt_variant: str = "standard") -> tuple[str, List[Dict[str, str]]]:
    rendered_options = "\n".join(
        f"{label}. {choice}" for label, choice in zip(MMLU_LABELS, example["choices"])
    )
    system_prompt = _mmlu_system_prompt(prompt_variant)
    # File-backed variants already include ANSWER: formatting in the system prompt;
    # keep the format_instructed user cue so the chat is consistent.
    if prompt_variant == "standard":
        user_prompt = (
            f"Question:\n{example['question']}\n\n"
            f"Choices:\n{rendered_options}\n\n"
            "Answer with A, B, C, or D."
        )
    else:
        user_prompt = (
            f"Question:\n{example['question']}\n\n"
            f"Choices:\n{rendered_options}\n\n"
            "Reply in the format: ANSWER: followed by the letter (A, B, C, or D)."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return user_prompt, messages


def _mmlu_condition_strength(condition: str, steering_strength: float) -> float:
    if condition == "decrease_entropy":
        return steering_strength
    if condition == "increase_entropy":
        return -steering_strength
    if condition == "none":
        return 0.0
    raise ValueError(f"Unknown MMLU steering condition: {condition}")


def _mmlu_run_prefix(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
    vector_certainty_instruction: Optional[str] = None,
    vector_uncertainty_instruction: Optional[str] = None,
) -> str:
    dataset_slug = dataset_name.replace("/", "_")
    config_slug = dataset_config or "default"
    max_slug = "all" if max_examples is None else str(max_examples)
    prefill_slug = "" if answer_prefix else "_noprefill"
    variant_slug = "" if prompt_variant == "standard" else f"_{prompt_variant}"
    if vector_certainty_instruction and vector_uncertainty_instruction:
        vector_slug = f"_vec{slugify_instruction(vector_certainty_instruction)[:20]}"
    else:
        vector_slug = ""
    return f"{dataset_slug}_{config_slug}_{split}_max{max_slug}_layer{fixed_layer}_strength{steering_strength}{vector_slug}{prefill_slug}{variant_slug}_"


def _mmlu_random_control_prefix(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    random_seed: int,
    n_random_vectors: int,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
    vector_certainty_instruction: Optional[str] = None,
    vector_uncertainty_instruction: Optional[str] = None,
) -> str:
    dataset_slug = dataset_name.replace("/", "_")
    config_slug = dataset_config or "default"
    max_slug = "all" if max_examples is None else str(max_examples)
    prefill_slug = "" if answer_prefix else "_noprefill"
    variant_slug = "" if prompt_variant == "standard" else f"_{prompt_variant}"
    if vector_certainty_instruction and vector_uncertainty_instruction:
        vector_slug = f"_vec{slugify_instruction(vector_certainty_instruction)[:20]}"
    else:
        vector_slug = ""
    return (
        f"{dataset_slug}_{config_slug}_{split}_max{max_slug}_layer{fixed_layer}_"
        f"strength{steering_strength}_seed{random_seed}_n{n_random_vectors}{vector_slug}{prefill_slug}{variant_slug}_"
    )


def run_mmlu_entropy_steering(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    steering_strength: float,
    steering_conditions: List[str],
    *,
    fixed_layer: int,
    dataset_name: str = "cais/mmlu",
    dataset_config: Optional[str] = "all",
    split: str = "test",
    max_examples: Optional[int] = 100,
    batch_size: int = 4,
    normalize_steering_vector: bool = False,
    preserve_norm: bool = False,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vector_path = (
        get_steering_artifacts_dir()
        / "steering_vectors"
        / slugify_model_name(model_name)
        / instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        / f"layer_{fixed_layer}.pt"
    )
    if not vector_path.exists():
        activation_path = latest_activation_artifact(
            model_name, vector_certainty_instruction, vector_uncertainty_instruction
        )
        if activation_path is None:
            print(
                f"[auto-prepare] No activation artifact for {model_name!r} "
                f"{vector_certainty_instruction!r} vs {vector_uncertainty_instruction!r}; extracting at layer {fixed_layer}..."
            )
            activation_path = extract_multi_layer_activations(
                model_name,
                vector_certainty_instruction,
                vector_uncertainty_instruction,
                layer_indices=[fixed_layer],
            )
        print(f"[auto-prepare] Computing steering vector for layer {fixed_layer}...")
        compute_all_steering_vectors()
        if not vector_path.exists():
            raise FileNotFoundError(
                f"Steering vector still missing after auto-prepare: {vector_path}."
            )

    examples = _load_mmlu_examples(dataset_name, dataset_config, split, max_examples)
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched MMLU inference.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        steering_vector = _load_layer_vector(
            model_name,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
            fixed_layer,
            device,
        )
        if normalize_steering_vector:
            norm = torch.norm(steering_vector)
            if norm > 0:
                steering_vector = steering_vector / norm

        label_token_prefix = " " if answer_prefix else ""
        prompts = []
        for example in examples:
            user_prompt, messages = _mmlu_messages(example, prompt_variant=prompt_variant)
            prompts.append(
                {
                    **example,
                    "user_prompt": user_prompt,
                    "text": build_chat_input(tokenizer, model_name, messages, answer_prefix=answer_prefix),
                }
            )

        grouped_results = {}
        for condition in steering_conditions:
            effective_strength = _mmlu_condition_strength(condition, steering_strength)
            context = (
                SteeringInjector(
                    model,
                    fixed_layer,
                    steering_vector,
                    strength=effective_strength,
                    preserve_norm=preserve_norm,
                )
                if effective_strength != 0.0
                else nullcontext()
            )
            condition_results = []
            with context:
                for batch in tqdm(
                    _batch_items(prompts, batch_size),
                    desc=f"MMLU {condition}",
                    total=math.ceil(len(prompts) / batch_size),
                    leave=False,
                ):
                    texts = [item["text"] for item in batch]
                    inputs = tokenize_chat_text(tokenizer, texts, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        logits = model(**inputs).logits[:, -1, :]
                        probs_batch = torch.softmax(logits, dim=-1)
                    for item, probs in zip(batch, probs_batch):
                        label_records = option_label_probabilities(tokenizer, probs, MMLU_LABELS, token_prefix=label_token_prefix)
                        raw_probs = {label: float(label_records[label]["prob"]) for label in MMLU_LABELS}
                        normalized_probs = {
                            label: float(label_records[label]["normalized_prob"])
                            for label in MMLU_LABELS
                        }
                        predicted_label = max(MMLU_LABELS, key=lambda label: raw_probs[label])
                        condition_results.append(
                            {
                                "example_idx": item["example_idx"],
                                "subject": item["subject"],
                                "question": item["question"],
                                "choices": item["choices"],
                                "answer_index": item["answer_index"],
                                "answer_label": item["answer_label"],
                                "predicted_label": predicted_label,
                                "is_correct": predicted_label == item["answer_label"],
                                "steering_condition": condition,
                                "effective_strength": effective_strength,
                                "label_probability_mass": float(sum(raw_probs.values())),
                                "label_entropy": categorical_entropy(list(raw_probs.values())),
                                "prob_A": raw_probs["A"],
                                "prob_B": raw_probs["B"],
                                "prob_C": raw_probs["C"],
                                "prob_D": raw_probs["D"],
                                "normalized_prob_A": normalized_probs["A"],
                                "normalized_prob_B": normalized_probs["B"],
                                "normalized_prob_C": normalized_probs["C"],
                                "normalized_prob_D": normalized_probs["D"],
                                "label_token_ids": {
                                    label: int(label_records[label]["token_id"]) for label in MMLU_LABELS
                                },
                                "decoded_label_tokens": {
                                    label: str(label_records[label]["decoded"]) for label in MMLU_LABELS
                                },
                                "user_prompt": item["user_prompt"],
                                "layer_idx": fixed_layer,
                            }
                        )
            grouped_results[condition] = condition_results

    output_dir = (
        get_results_dir()
        / "mmlu_entropy_steering"
        / slugify_model_name(model_name)
        / f"{_mmlu_run_prefix(dataset_name, dataset_config, split, max_examples, fixed_layer, steering_strength, answer_prefix, prompt_variant, vector_certainty_instruction, vector_uncertainty_instruction)}{timestamp()}"
    )
    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "max_examples": max_examples,
                "batch_size": batch_size,
                "labels": MMLU_LABELS,
                "answer_prefix": answer_prefix,
                "label_token_prefix": label_token_prefix,
                "prompt_variant": prompt_variant,
                "vector_certainty_instruction": vector_certainty_instruction,
                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                "fixed_layer": fixed_layer,
                "steering_strength": steering_strength,
                "steering_conditions": steering_conditions,
                "condition_strengths": {
                    condition: _mmlu_condition_strength(condition, steering_strength)
                    for condition in steering_conditions
                },
                "normalize_steering_vector": normalize_steering_vector,
                "preserve_norm": preserve_norm,
            },
            "results": grouped_results,
        },
    )
    return output_path


def run_mmlu_random_vector_control(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    steering_strength: float,
    *,
    fixed_layer: int,
    n_random_vectors: int = 8,
    random_seed: int = 0,
    dataset_name: str = "cais/mmlu",
    dataset_config: Optional[str] = "all",
    split: str = "test",
    max_examples: Optional[int] = None,
    batch_size: int = 32,
    preserve_norm: bool = False,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_random_vectors <= 0:
        raise ValueError("n_random_vectors must be positive")

    vector_path = (
        get_steering_artifacts_dir()
        / "steering_vectors"
        / slugify_model_name(model_name)
        / instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        / f"layer_{fixed_layer}.pt"
    )
    if not vector_path.exists():
        activation_path = latest_activation_artifact(
            model_name, vector_certainty_instruction, vector_uncertainty_instruction
        )
        if activation_path is None:
            print(
                f"[auto-prepare] No activation artifact for {model_name!r} "
                f"{vector_certainty_instruction!r} vs {vector_uncertainty_instruction!r}; extracting at layer {fixed_layer}..."
            )
            activation_path = extract_multi_layer_activations(
                model_name,
                vector_certainty_instruction,
                vector_uncertainty_instruction,
                layer_indices=[fixed_layer],
            )
        print(f"[auto-prepare] Computing steering vector for layer {fixed_layer}...")
        compute_all_steering_vectors()
        if not vector_path.exists():
            raise FileNotFoundError(
                f"Steering vector still missing after auto-prepare: {vector_path}."
            )

    examples = _load_mmlu_examples(dataset_name, dataset_config, split, max_examples)
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched MMLU inference.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        reference_vector = _load_layer_vector(
            model_name,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
            fixed_layer,
            device,
        )
        reference_norm = torch.norm(reference_vector.float()).item()
        if reference_norm <= 0:
            raise ValueError("Reference steering vector norm must be positive.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        random_vectors = []
        random_vector_metadata = []
        for random_idx in range(n_random_vectors):
            random_vector = torch.randn(reference_vector.shape, generator=generator, dtype=torch.float32)
            random_norm = torch.norm(random_vector).item()
            if random_norm <= 0:
                raise ValueError(f"Random vector {random_idx} has zero norm.")
            random_vector = random_vector / random_norm * reference_norm
            random_vectors.append(random_vector.to(device=device, dtype=reference_vector.dtype))
            random_vector_metadata.append(
                {
                    "condition": f"random_{random_idx:02d}",
                    "random_vector_index": random_idx,
                    "pre_normalization_norm": random_norm,
                    "post_normalization_norm": float(torch.norm(random_vector).item()),
                }
            )

        label_token_prefix = " " if answer_prefix else ""
        prompts = []
        for example in examples:
            user_prompt, messages = _mmlu_messages(example, prompt_variant=prompt_variant)
            prompts.append(
                {
                    **example,
                    "user_prompt": user_prompt,
                    "text": build_chat_input(tokenizer, model_name, messages, answer_prefix=answer_prefix),
                }
            )

        def append_condition_result(
            condition_results: list,
            item: Dict,
            probs,
            condition: str,
            random_idx: Optional[int],
            effective_strength: float,
        ) -> None:
            label_records = option_label_probabilities(tokenizer, probs, MMLU_LABELS, token_prefix=label_token_prefix)
            raw_probs = {label: float(label_records[label]["prob"]) for label in MMLU_LABELS}
            normalized_probs = {
                label: float(label_records[label]["normalized_prob"])
                for label in MMLU_LABELS
            }
            predicted_label = max(MMLU_LABELS, key=lambda label: raw_probs[label])
            condition_results.append(
                {
                    "example_idx": item["example_idx"],
                    "subject": item["subject"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer_index": item["answer_index"],
                    "answer_label": item["answer_label"],
                    "predicted_label": predicted_label,
                    "is_correct": predicted_label == item["answer_label"],
                    "steering_condition": condition,
                    "random_vector_index": random_idx,
                    "effective_strength": effective_strength,
                    "label_probability_mass": float(sum(raw_probs.values())),
                    "label_entropy": categorical_entropy(list(raw_probs.values())),
                    "prob_A": raw_probs["A"],
                    "prob_B": raw_probs["B"],
                    "prob_C": raw_probs["C"],
                    "prob_D": raw_probs["D"],
                    "normalized_prob_A": normalized_probs["A"],
                    "normalized_prob_B": normalized_probs["B"],
                    "normalized_prob_C": normalized_probs["C"],
                    "normalized_prob_D": normalized_probs["D"],
                    "label_token_ids": {
                        label: int(label_records[label]["token_id"]) for label in MMLU_LABELS
                    },
                    "decoded_label_tokens": {
                        label: str(label_records[label]["decoded"]) for label in MMLU_LABELS
                    },
                    "user_prompt": item["user_prompt"],
                    "layer_idx": fixed_layer,
                }
            )

        grouped_results = {}
        condition_specs = [("none", None, None)] + [
            (f"random_{random_idx:02d}", random_idx, random_vectors[random_idx])
            for random_idx in range(n_random_vectors)
        ]
        for condition, random_idx, steering_vector in condition_specs:
            context = (
                SteeringInjector(
                    model,
                    fixed_layer,
                    steering_vector,
                    strength=steering_strength,
                    preserve_norm=preserve_norm,
                )
                if steering_vector is not None
                else nullcontext()
            )
            condition_results = []
            with context:
                for batch in tqdm(
                    _batch_items(prompts, batch_size),
                    desc=f"MMLU {condition}",
                    total=math.ceil(len(prompts) / batch_size),
                    leave=False,
                ):
                    texts = [item["text"] for item in batch]
                    inputs = tokenize_chat_text(tokenizer, texts, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        logits = model(**inputs).logits[:, -1, :]
                        probs_batch = torch.softmax(logits, dim=-1)
                    for item, probs in zip(batch, probs_batch):
                        append_condition_result(
                            condition_results,
                            item,
                            probs,
                            condition,
                            random_idx,
                            steering_strength if steering_vector is not None else 0.0,
                        )
            grouped_results[condition] = condition_results

    output_dir = (
        get_results_dir()
        / "mmlu_random_vector_control"
        / slugify_model_name(model_name)
        / f"{_mmlu_random_control_prefix(dataset_name, dataset_config, split, max_examples, fixed_layer, steering_strength, random_seed, n_random_vectors, answer_prefix, prompt_variant, vector_certainty_instruction, vector_uncertainty_instruction)}{timestamp()}"
    )
    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "max_examples": max_examples,
                "batch_size": batch_size,
                "labels": MMLU_LABELS,
                "answer_prefix": answer_prefix,
                "label_token_prefix": label_token_prefix,
                "prompt_variant": prompt_variant,
                "reference_vector_certainty_instruction": vector_certainty_instruction,
                "reference_vector_uncertainty_instruction": vector_uncertainty_instruction,
                "fixed_layer": fixed_layer,
                "steering_strength": steering_strength,
                "n_random_vectors": n_random_vectors,
                "random_seed": random_seed,
                "random_distribution": "iid_gaussian_unit_direction",
                "control_execution": "sequential_single_vector_batched_questions",
                "reference_vector_norm": reference_norm,
                "random_vectors": random_vector_metadata,
                "preserve_norm": preserve_norm,
            },
            "results": grouped_results,
        },
    )
    return output_path


def _mmlu_prompt_conditions_prefix(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    answer_prefix: str,
    prompt_variants: Sequence[str],
) -> str:
    dataset_slug = dataset_name.replace("/", "_")
    config_slug = dataset_config or "default"
    max_slug = "all" if max_examples is None else str(max_examples)
    prefill_slug = "" if answer_prefix else "_noprefill"
    variants_slug = "+".join(prompt_variants)
    return f"{dataset_slug}_{config_slug}_{split}_max{max_slug}{prefill_slug}_prompts_{variants_slug}_"


def run_mmlu_prompt_conditions(
    model_name: str,
    prompt_variants: Optional[Sequence[str]] = None,
    *,
    dataset_name: str = "cais/mmlu",
    dataset_config: Optional[str] = "all",
    split: str = "test",
    max_examples: Optional[int] = None,
    batch_size: int = 16,
    answer_prefix: str = "ANSWER:",
) -> Path:
    """Prompt-only MMLU: no activation steering; one condition per system prompt variant.

    Stores per-label probs and logprobs under ``mmlu_prompt_conditions/``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    variants = list(prompt_variants) if prompt_variants is not None else list(DEFAULT_MMLU_PROMPT_CONDITIONS)
    if not variants:
        raise ValueError("prompt_variants must be non-empty")
    for variant in variants:
        _mmlu_system_prompt(variant)  # validate early

    examples = _load_mmlu_examples(dataset_name, dataset_config, split, max_examples)
    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched MMLU inference.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        label_token_prefix = " " if answer_prefix else ""
        grouped_results: Dict[str, List[Dict]] = {}
        system_prompts: Dict[str, str] = {}

        for variant in variants:
            system_prompts[variant] = _mmlu_system_prompt(variant)
            prompts = []
            for example in examples:
                user_prompt, messages = _mmlu_messages(example, prompt_variant=variant)
                prompts.append(
                    {
                        **example,
                        "user_prompt": user_prompt,
                        "system_prompt": system_prompts[variant],
                        "text": build_chat_input(tokenizer, model_name, messages, answer_prefix=answer_prefix),
                    }
                )

            condition_results = []
            for batch in tqdm(
                _batch_items(prompts, batch_size),
                desc=f"MMLU prompt {variant}",
                total=math.ceil(len(prompts) / batch_size),
                leave=False,
            ):
                texts = [item["text"] for item in batch]
                inputs = tokenize_chat_text(tokenizer, texts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs_batch = torch.exp(log_probs)
                for item, probs, log_probs_row in zip(batch, probs_batch, log_probs):
                    label_records = option_label_probabilities(
                        tokenizer, probs, MMLU_LABELS, token_prefix=label_token_prefix
                    )
                    raw_probs = {label: float(label_records[label]["prob"]) for label in MMLU_LABELS}
                    raw_logprobs = {
                        label: float(log_probs_row[int(label_records[label]["token_id"])].item())
                        for label in MMLU_LABELS
                    }
                    normalized_probs = {
                        label: float(label_records[label]["normalized_prob"])
                        for label in MMLU_LABELS
                    }
                    predicted_label = max(MMLU_LABELS, key=lambda label: raw_probs[label])
                    condition_results.append(
                        {
                            "example_idx": item["example_idx"],
                            "subject": item["subject"],
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer_index": item["answer_index"],
                            "answer_label": item["answer_label"],
                            "predicted_label": predicted_label,
                            "is_correct": predicted_label == item["answer_label"],
                            "prompt_condition": variant,
                            "label_probability_mass": float(sum(raw_probs.values())),
                            "label_entropy": categorical_entropy(list(raw_probs.values())),
                            "prob_A": raw_probs["A"],
                            "prob_B": raw_probs["B"],
                            "prob_C": raw_probs["C"],
                            "prob_D": raw_probs["D"],
                            "logprob_A": raw_logprobs["A"],
                            "logprob_B": raw_logprobs["B"],
                            "logprob_C": raw_logprobs["C"],
                            "logprob_D": raw_logprobs["D"],
                            "normalized_prob_A": normalized_probs["A"],
                            "normalized_prob_B": normalized_probs["B"],
                            "normalized_prob_C": normalized_probs["C"],
                            "normalized_prob_D": normalized_probs["D"],
                            "label_token_ids": {
                                label: int(label_records[label]["token_id"]) for label in MMLU_LABELS
                            },
                            "decoded_label_tokens": {
                                label: str(label_records[label]["decoded"]) for label in MMLU_LABELS
                            },
                            "user_prompt": item["user_prompt"],
                            "system_prompt": item["system_prompt"],
                        }
                    )
            grouped_results[variant] = condition_results

    output_dir = (
        get_results_dir()
        / "mmlu_prompt_conditions"
        / slugify_model_name(model_name)
        / f"{_mmlu_prompt_conditions_prefix(dataset_name, dataset_config, split, max_examples, answer_prefix, variants)}{timestamp()}"
    )
    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "max_examples": max_examples,
                "batch_size": batch_size,
                "labels": MMLU_LABELS,
                "answer_prefix": answer_prefix,
                "label_token_prefix": label_token_prefix,
                "prompt_variants": variants,
                "system_prompts": system_prompts,
                "stores_logprobs": True,
                "steering": None,
            },
            "results": grouped_results,
        },
    )
    return output_path


def latest_mmlu_prompt_conditions_result(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    answer_prefix: str = "ANSWER:",
    prompt_variants: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    variants = list(prompt_variants) if prompt_variants is not None else list(DEFAULT_MMLU_PROMPT_CONDITIONS)
    parent = get_results_dir() / "mmlu_prompt_conditions" / slugify_model_name(model_name)
    match = latest_matching_dir(
        parent,
        _mmlu_prompt_conditions_prefix(
            dataset_name, dataset_config, split, max_examples, answer_prefix, variants
        ),
    )
    return None if match is None else match / "results.json"


def latest_mmlu_entropy_steering_result(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
    vector_certainty_instruction: Optional[str] = None,
    vector_uncertainty_instruction: Optional[str] = None,
) -> Optional[Path]:
    parent = get_results_dir() / "mmlu_entropy_steering" / slugify_model_name(model_name)
    match = latest_matching_dir(
        parent,
        _mmlu_run_prefix(dataset_name, dataset_config, split, max_examples, fixed_layer, steering_strength, answer_prefix, prompt_variant, vector_certainty_instruction, vector_uncertainty_instruction),
    )
    return None if match is None else match / "results.json"


def latest_mmlu_random_vector_control_result(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    random_seed: int,
    n_random_vectors: int,
    answer_prefix: str = "ANSWER:",
    prompt_variant: str = "standard",
    vector_certainty_instruction: Optional[str] = None,
    vector_uncertainty_instruction: Optional[str] = None,
) -> Optional[Path]:
    parent = get_results_dir() / "mmlu_random_vector_control" / slugify_model_name(model_name)
    match = latest_matching_dir(
        parent,
        _mmlu_random_control_prefix(
            dataset_name,
            dataset_config,
            split,
            max_examples,
            fixed_layer,
            steering_strength,
            random_seed,
            n_random_vectors,
            answer_prefix,
            prompt_variant,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
        ),
    )
    return None if match is None else match / "results.json"


def run_fixed_layer_cross_instruction_steering(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    test_certainty_instruction: str,
    test_uncertainty_instruction: str,
    fixed_layer: int,
    steering_strength: float,
) -> Path:
    vector_path = (
        get_steering_artifacts_dir()
        / "steering_vectors"
        / slugify_model_name(model_name)
        / instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        / f"layer_{fixed_layer}.pt"
    )
    if not vector_path.exists():
        activation_path = latest_activation_artifact(
            model_name, vector_certainty_instruction, vector_uncertainty_instruction
        )
        if activation_path is None:
            print(
                f"[auto-prepare] No activation artifact found for {model_name!r}; "
                f"extracting activations at layer {fixed_layer}..."
            )
            activation_path = extract_multi_layer_activations(
                model_name,
                vector_certainty_instruction,
                vector_uncertainty_instruction,
                layer_indices=[fixed_layer],
            )
        print(f"[auto-prepare] Computing steering vector for layer {fixed_layer}...")
        compute_all_steering_vectors()
        if not vector_path.exists():
            raise FileNotFoundError(
                f"Steering vector still missing after compute_all_steering_vectors(): {vector_path}"
            )

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        steering_vector = _load_layer_vector(
            model_name, vector_certainty_instruction, vector_uncertainty_instruction, fixed_layer, device
        )
        system_prompt = read_txt(PROMPTS_DIR / "standard_system.txt").format(
            certainty_instruction=test_certainty_instruction,
            uncertainty_instruction=test_uncertainty_instruction,
        )
        ab_pairs = load_ab_pairs()
        assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
        injector = SteeringInjector(
            model,
            fixed_layer,
            steering_vector,
            strength=steering_strength,
            certainty_instruction=test_certainty_instruction,
            uncertainty_instruction=test_uncertainty_instruction,
        )
        layer_results = []
        with injector:
            for pair in tqdm(ab_pairs):
                for flip_ab in [True, False]:
                    options = [pair["token_b"], pair["token_a"]] if flip_ab else [pair["token_a"], pair["token_b"]]
                    for instruction in [test_certainty_instruction, test_uncertainty_instruction]:
                        injector.set_current_instruction(instruction)
                        user_prompt, messages = _pair_messages(system_prompt, options[0], options[1], instruction, prefill_assistant=False)
                        text = build_chat_input(tokenizer, model_name, messages)
                        first_device = next(model.parameters()).device
                        inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                        with torch.no_grad():
                            logits = model(**inputs).logits[0, -1, :]
                            probs = torch.softmax(logits, dim=-1)
                        _, prob_a, decoded_a = option_token_probability(tokenizer, probs, options[0])
                        _, prob_b, decoded_b = option_token_probability(tokenizer, probs, options[1])
                        layer_results.append(
                            {
                                "option_a": options[0],
                                "option_b": options[1],
                                "decoded_option_a": decoded_a,
                                "decoded_option_b": decoded_b,
                                "instruction": instruction,
                                "option_a_prob": prob_a,
                                "option_b_prob": prob_b,
                                "user_prompt": user_prompt,
                                "flip_ab": flip_ab,
                                "layer_idx": fixed_layer,
                                "steering_strength": steering_strength,
                            }
                        )

    model_dir = slugify_model_name(model_name)
    vector_label = instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
    test_label = instruction_pair_label(test_certainty_instruction, test_uncertainty_instruction)
    output_dir = (
        get_results_dir()
        / "steering_fixed_layer"
        / model_dir
        / f"vector_{vector_label}_test_{test_label}_layer{fixed_layer}_strength{steering_strength}_{timestamp()}"
    )
    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "vector_certainty_instruction": vector_certainty_instruction,
                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                "test_certainty_instruction": test_certainty_instruction,
                "test_uncertainty_instruction": test_uncertainty_instruction,
                "fixed_layer": fixed_layer,
                "steering_strength": steering_strength,
            },
            "results": layer_results,
        },
    )
    return output_path


def latest_fixed_layer_cross_instruction_result(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    test_certainty_instruction: str,
    test_uncertainty_instruction: str,
    fixed_layer: int,
    steering_strength: float,
) -> Optional[Path]:
    parent = get_results_dir() / "steering_fixed_layer" / slugify_model_name(model_name)
    if not parent.exists():
        return None
    prefix = (
        f"vector_{instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)}"
        f"_test_{instruction_pair_label(test_certainty_instruction, test_uncertainty_instruction)}"
        f"_layer{fixed_layer}_strength{steering_strength}_"
    )
    match = latest_matching_dir(parent, prefix)
    return None if match is None else match / "results.json"


# ---------------------------------------------------------------------------
# Figure 8 — WildChat steering: distribution of next-token probabilities
# ---------------------------------------------------------------------------


def _load_wildchat_prompts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    user_field: str,
    conversation_field: str,
    seed: int,
    streaming: bool = True,
    shuffle_buffer_size: int = 1000,
) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for Figure 8 WildChat runs. "
            "Install project dependencies with `uv sync`."
        ) from exc

    dataset_args = [dataset_name]
    if dataset_config:
        dataset_args.append(dataset_config)
    dataset = load_dataset(*dataset_args, split=split, streaming=streaming)
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)

    examples: List[Dict] = []
    for raw_idx, item in enumerate(dataset):
        if max_examples is not None and len(examples) >= max_examples:
            break
        user_text: Optional[str] = None
        if user_field and user_field in item and isinstance(item[user_field], str):
            user_text = item[user_field]
        elif conversation_field and conversation_field in item:
            conversation = item[conversation_field]
            if isinstance(conversation, list):
                for turn in conversation:
                    if isinstance(turn, dict) and turn.get("role") == "user":
                        content = turn.get("content")
                        if isinstance(content, str) and content.strip():
                            user_text = content
                            break
        if not user_text or not user_text.strip():
            continue
        examples.append(
            {
                "example_idx": len(examples),
                "source_idx": raw_idx,
                "user_prompt": user_text.strip(),
            }
        )
    return examples


def _figure8_condition_strength(condition: str, steering_strength: float) -> float:
    if condition == "none":
        return 0.0
    if condition == "decrease_entropy":
        return steering_strength
    if condition == "increase_entropy":
        return -steering_strength
    raise ValueError(f"Unknown Figure 8 steering condition: {condition}")


def _figure8_run_prefix(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    max_new_tokens: int,
) -> str:
    dataset_slug = dataset_name.replace("/", "_")
    config_slug = dataset_config or "default"
    max_slug = "all" if max_examples is None else str(max_examples)
    return (
        f"{dataset_slug}_{config_slug}_{split}_max{max_slug}_layer{fixed_layer}_"
        f"strength{steering_strength}_newtok{max_new_tokens}_"
    )


def run_wildchat_steering_token_distributions(
    model_name: str,
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    steering_strength: float,
    steering_conditions: List[str],
    *,
    fixed_layer: int,
    dataset_name: str = "allenai/WildChat-1M",
    dataset_config: Optional[str] = None,
    split: str = "train",
    max_examples: Optional[int] = 50,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    seed: int = 0,
    user_field: str = "user_prompt",
    conversation_field: str = "conversation",
    do_sample: bool = False,
    temperature: float = 1.0,
    preserve_norm: bool = False,
    streaming: bool = True,
    shuffle_buffer_size: int = 1000,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    vector_path = (
        get_steering_artifacts_dir()
        / "steering_vectors"
        / slugify_model_name(model_name)
        / instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
        / f"layer_{fixed_layer}.pt"
    )
    if not vector_path.exists():
        raise FileNotFoundError(
            f"Missing steering vector for Figure 8: {vector_path}. "
            "Run the Figure 5 steering prepare stage for this model/vector source first."
        )

    prompts_meta = _load_wildchat_prompts(
        dataset_name,
        dataset_config,
        split,
        max_examples,
        user_field,
        conversation_field,
        seed,
        streaming=streaming,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    if not prompts_meta:
        raise RuntimeError("Loaded zero WildChat prompts; check dataset_name / split / fields.")

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched generation.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        steering_vector = _load_layer_vector(
            model_name,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
            fixed_layer,
            device,
        )

        for item in prompts_meta:
            messages = [{"role": "user", "content": item["user_prompt"]}]
            item["text"] = build_chat_input(tokenizer, model_name, messages, answer_prefix="")

        grouped_results: Dict[str, List[Dict]] = {}
        for condition in steering_conditions:
            effective_strength = _figure8_condition_strength(condition, steering_strength)
            context = (
                SteeringInjector(
                    model,
                    fixed_layer,
                    steering_vector,
                    strength=effective_strength,
                    preserve_norm=preserve_norm,
                )
                if effective_strength != 0.0
                else nullcontext()
            )
            condition_results: List[Dict] = []
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
            with context:
                for batch in tqdm(
                    list(_batch_items(prompts_meta, batch_size)),
                    desc=f"WildChat {condition}",
                    leave=False,
                ):
                    texts = [item["text"] for item in batch]
                    inputs = tokenize_chat_text(tokenizer, texts, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)
                    sequences = outputs.sequences
                    prompt_len = inputs["input_ids"].shape[1]
                    generated_ids = sequences[:, prompt_len:]
                    scores = outputs.scores  # tuple of [B, V] tensors, length max_new_tokens
                    for batch_idx, item in enumerate(batch):
                        token_records: List[Dict] = []
                        for step, step_logits in enumerate(scores):
                            token_id = int(generated_ids[batch_idx, step].item())
                            if token_id == tokenizer.pad_token_id:
                                continue
                            log_probs = F.log_softmax(step_logits[batch_idx].float(), dim=-1)
                            probs = log_probs.exp()
                            chosen_logprob = float(log_probs[token_id].item())
                            chosen_prob = float(math.exp(chosen_logprob))
                            top_prob = float(probs.max().item())
                            entropy_nats = float(-(probs * log_probs).sum().item())
                            entropy_bits = entropy_nats / math.log(2)
                            token_records.append(
                                {
                                    "step": step,
                                    "token_id": token_id,
                                    "token_str": tokenizer.decode([token_id]),
                                    "logprob": chosen_logprob,
                                    "prob": chosen_prob,
                                    "top_prob": top_prob,
                                    "entropy_bits": entropy_bits,
                                }
                            )
                            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                                break
                        generated_text = tokenizer.decode(
                            generated_ids[batch_idx], skip_special_tokens=True
                        )
                        condition_results.append(
                            {
                                "example_idx": item["example_idx"],
                                "source_idx": item["source_idx"],
                                "user_prompt": item["user_prompt"],
                                "steering_condition": condition,
                                "effective_strength": effective_strength,
                                "generated_text": generated_text,
                                "n_generated_tokens": len(token_records),
                                "tokens": token_records,
                            }
                        )
            grouped_results[condition] = condition_results

    output_dir = (
        get_results_dir()
        / "wildchat_steering"
        / slugify_model_name(model_name)
        / f"{_figure8_run_prefix(dataset_name, dataset_config, split, max_examples, fixed_layer, steering_strength, max_new_tokens)}{timestamp()}"
    )
    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "max_examples": max_examples,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                "seed": seed,
                "user_field": user_field,
                "conversation_field": conversation_field,
                "vector_certainty_instruction": vector_certainty_instruction,
                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                "fixed_layer": fixed_layer,
                "steering_strength": steering_strength,
                "steering_conditions": steering_conditions,
                "condition_strengths": {
                    condition: _figure8_condition_strength(condition, steering_strength)
                    for condition in steering_conditions
                },
                "do_sample": do_sample,
                "temperature": temperature,
                "preserve_norm": preserve_norm,
                "streaming": streaming,
                "shuffle_buffer_size": shuffle_buffer_size,
            },
            "results": grouped_results,
        },
    )
    return output_path


def latest_wildchat_steering_token_distributions_result(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    fixed_layer: int,
    steering_strength: float,
    max_new_tokens: int,
) -> Optional[Path]:
    parent = get_results_dir() / "wildchat_steering" / slugify_model_name(model_name)
    match = latest_matching_dir(
        parent,
        _figure8_run_prefix(
            dataset_name,
            dataset_config,
            split,
            max_examples,
            fixed_layer,
            steering_strength,
            max_new_tokens,
        ),
    )
    return None if match is None else match / "results.json"


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for binary labels in {0, 1} via Mann–Whitney U (average ranks for ties)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # 1-indexed average ranks over the tied block
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[labels == 1].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _confusion_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float | int | List[List[int]]]:
    """Predict label=1 when score >= threshold."""
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else float("nan"),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _best_threshold_on_val(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float | int | List[List[int]]]:
    """Choose threshold maximizing accuracy on val; ties → highest threshold (more conservative)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(scores)
    if len(unique) == 1:
        return _confusion_at_threshold(scores, labels, float(unique[0]))
    # Candidate thresholds: midpoints between consecutive unique scores, plus ends
    candidates = [float(unique[0] - 1e-12)]
    for left, right in zip(unique[:-1], unique[1:]):
        candidates.append(float(0.5 * (left + right)))
    candidates.append(float(unique[-1] + 1e-12))

    best: Optional[Dict[str, float | int | List[List[int]]]] = None
    for thr in candidates:
        metrics = _confusion_at_threshold(scores, labels, thr)
        if best is None:
            best = metrics
            continue
        if metrics["accuracy"] > best["accuracy"] or (
            metrics["accuracy"] == best["accuracy"] and metrics["threshold"] > best["threshold"]
        ):
            best = metrics
    assert best is not None
    return best


def _stratified_val_test_split(
    n_examples: int,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same example indices for both conditions (paired). Returns (val_idx, test_idx)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_examples)
    n_val = max(1, int(round(n_examples * val_fraction)))
    n_val = min(n_val, n_examples - 1)
    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val:])
    return val_idx, test_idx


def _mmlu_steering_vector_probe_prefix(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    answer_prefix: str,
    prompt_variants: Sequence[str],
    vector_certainty_instruction: str,
    vector_uncertainty_instruction: str,
    layer_step: int,
) -> str:
    dataset_slug = dataset_name.replace("/", "_")
    config_slug = dataset_config or "default"
    max_slug = "all" if max_examples is None else str(max_examples)
    prefill_slug = "" if answer_prefix else "_noprefill"
    variants_slug = "+".join(prompt_variants)
    vector_slug = instruction_pair_label(vector_certainty_instruction, vector_uncertainty_instruction)
    return (
        f"{dataset_slug}_{config_slug}_{split}_max{max_slug}{prefill_slug}_"
        f"probe_{variants_slug}_vec{vector_slug}_step{layer_step}_"
    )


def run_mmlu_steering_vector_probe(
    model_name: str,
    vector_certainty_instruction: str = "Be sure!",
    vector_uncertainty_instruction: str = "Be random!",
    *,
    prompt_variants: Optional[Sequence[str]] = None,
    positive_variant: str = "format_instructed",
    negative_variant: str = "maximize_entropy",
    dataset_name: str = "cais/mmlu",
    dataset_config: Optional[str] = "all",
    split: str = "test",
    max_examples: Optional[int] = None,
    batch_size: int = 16,
    answer_prefix: str = "ANSWER:",
    layer_step: int = 4,
    layer_indices: Optional[List[int]] = None,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Path:
    """Detect prompt-condition via projection onto the certainty−uncertainty steering vector.

    For each MMLU example under ``positive_variant`` (default format_instructed) and
    ``negative_variant`` (default maximize_entropy), extract last-token residual activations
    at every ``layer_step`` layer, project onto the unit steering vector at that layer, and
    report AUROC (full set) plus accuracy / confusion matrix with threshold chosen on a
    20% validation split and evaluated on the held-out 80%.

    Higher projection = more certainty-like. Positive class label is ``positive_variant``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    variants = list(prompt_variants) if prompt_variants is not None else [positive_variant, negative_variant]
    if positive_variant not in variants or negative_variant not in variants:
        raise ValueError(
            f"positive_variant={positive_variant!r} and negative_variant={negative_variant!r} "
            f"must both appear in prompt_variants={variants}"
        )
    if positive_variant == negative_variant:
        raise ValueError("positive_variant and negative_variant must differ")
    for variant in variants:
        _mmlu_system_prompt(variant)

    examples = _load_mmlu_examples(dataset_name, dataset_config, split, max_examples)
    n_examples = len(examples)
    val_idx, test_idx = _stratified_val_test_split(n_examples, val_fraction, seed)
    label_token_prefix = " " if answer_prefix else ""
    system_prompts: Dict[str, str] = {variant: _mmlu_system_prompt(variant) for variant in variants}
    behavior: Dict[str, List[Dict]] = {variant: [] for variant in variants}
    scores: Dict[str, Dict[int, List[float]]] = {}
    layer_indices_resolved: List[int] = []
    num_layers = 0

    with _loaded_model_and_tokenizer(model_name, torch.float16) as (model, tokenizer):
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched MMLU inference.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        num_layers = get_num_layers(model)
        if layer_indices is None:
            layer_indices_resolved = get_layer_indices_every_n(num_layers, layer_step)
        else:
            layer_indices_resolved = sorted(layer_indices)

        unit_vectors: Dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices_resolved:
            raw = _load_layer_vector(
                model_name,
                vector_certainty_instruction,
                vector_uncertainty_instruction,
                layer_idx,
                device=torch.device("cpu"),
            ).float()
            norm = torch.norm(raw)
            if float(norm) < 1e-12:
                raise ValueError(f"Steering vector at layer {layer_idx} has near-zero norm")
            unit_vectors[layer_idx] = (raw / norm).cpu()

        scores = {
            variant: {
                layer_idx: [float("nan")] * n_examples for layer_idx in layer_indices_resolved
            }
            for variant in variants
        }

        extractor = MultiLayerActivationExtractor(model, layer_indices_resolved)
        with extractor:
            for variant in variants:
                prompts = []
                for example in examples:
                    user_prompt, messages = _mmlu_messages(example, prompt_variant=variant)
                    prompts.append(
                        {
                            **example,
                            "user_prompt": user_prompt,
                            "system_prompt": system_prompts[variant],
                            "text": build_chat_input(
                                tokenizer, model_name, messages, answer_prefix=answer_prefix
                            ),
                        }
                    )

                for batch in tqdm(
                    _batch_items(prompts, batch_size),
                    desc=f"MMLU probe {variant}",
                    total=math.ceil(len(prompts) / batch_size),
                    leave=False,
                ):
                    texts = [item["text"] for item in batch]
                    inputs = tokenize_chat_text(
                        tokenizer, texts, return_tensors="pt", padding=True
                    ).to(device)
                    extractor.clear()
                    with torch.no_grad():
                        logits = model(**inputs).logits[:, -1, :]
                        probs_batch = torch.softmax(logits, dim=-1)
                    layer_acts = extractor.get_activations_as_dict()

                    for row_i, item in enumerate(batch):
                        ex_idx = int(item["example_idx"])
                        label_records = option_label_probabilities(
                            tokenizer,
                            probs_batch[row_i],
                            MMLU_LABELS,
                            token_prefix=label_token_prefix,
                        )
                        raw_probs = {
                            label: float(label_records[label]["prob"]) for label in MMLU_LABELS
                        }
                        predicted_label = max(MMLU_LABELS, key=lambda label: raw_probs[label])
                        behavior[variant].append(
                            {
                                "example_idx": ex_idx,
                                "subject": item["subject"],
                                "answer_label": item["answer_label"],
                                "predicted_label": predicted_label,
                                "is_correct": predicted_label == item["answer_label"],
                                "label_entropy": categorical_entropy(list(raw_probs.values())),
                                "prob_A": raw_probs["A"],
                                "prob_B": raw_probs["B"],
                                "prob_C": raw_probs["C"],
                                "prob_D": raw_probs["D"],
                            }
                        )
                        for layer_idx in layer_indices_resolved:
                            act = layer_acts[f"layer_{layer_idx}"][row_i].float().cpu()
                            score = float(torch.dot(act, unit_vectors[layer_idx]).item())
                            scores[variant][layer_idx][ex_idx] = score

        for variant in variants:
            behavior[variant].sort(key=lambda row: row["example_idx"])

    if not scores:
        raise RuntimeError("No probe scores collected; model forward pass did not run.")

    # --- metrics ---
    per_layer_metrics: Dict[str, Dict] = {}
    for layer_idx in layer_indices_resolved:
        pos_scores = np.asarray(scores[positive_variant][layer_idx], dtype=np.float64)
        neg_scores = np.asarray(scores[negative_variant][layer_idx], dtype=np.float64)
        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate(
            [np.ones(n_examples, dtype=np.int64), np.zeros(n_examples, dtype=np.int64)]
        )
        auroc_full = _binary_auroc(all_scores, all_labels)

        val_scores = np.concatenate([pos_scores[val_idx], neg_scores[val_idx]])
        val_labels = np.concatenate(
            [np.ones(len(val_idx), dtype=np.int64), np.zeros(len(val_idx), dtype=np.int64)]
        )
        test_scores = np.concatenate([pos_scores[test_idx], neg_scores[test_idx]])
        test_labels = np.concatenate(
            [np.ones(len(test_idx), dtype=np.int64), np.zeros(len(test_idx), dtype=np.int64)]
        )
        val_best = _best_threshold_on_val(val_scores, val_labels)
        test_at_val_thr = _confusion_at_threshold(
            test_scores, test_labels, float(val_best["threshold"])
        )
        per_layer_metrics[f"layer_{layer_idx}"] = {
            "auroc_full": auroc_full,
            "mean_score": {
                positive_variant: float(pos_scores.mean()),
                negative_variant: float(neg_scores.mean()),
            },
            "std_score": {
                positive_variant: float(pos_scores.std()),
                negative_variant: float(neg_scores.std()),
            },
            "val": {
                "n": int(len(val_idx) * 2),
                "auroc": _binary_auroc(val_scores, val_labels),
                **val_best,
            },
            "test": {
                "n": int(len(test_idx) * 2),
                "auroc": _binary_auroc(test_scores, test_labels),
                **test_at_val_thr,
                "confusion_matrix_row_true": [negative_variant, positive_variant],
                "confusion_matrix_col_pred": [negative_variant, positive_variant],
                "confusion_matrix_note": (
                    "matrix[[tn, fp], [fn, tp]] with positive class = "
                    f"{positive_variant} (score >= threshold)"
                ),
            },
        }

    # Rank layers by full-set AUROC
    ranked = sorted(
        (
            (int(name.split("_")[1]), metrics["auroc_full"])
            for name, metrics in per_layer_metrics.items()
        ),
        key=lambda item: (-(item[1] if item[1] == item[1] else -1.0), item[0]),
    )

    output_dir = (
        get_results_dir()
        / "mmlu_steering_vector_probe"
        / slugify_model_name(model_name)
        / (
            f"{_mmlu_steering_vector_probe_prefix(dataset_name, dataset_config, split, max_examples, answer_prefix, variants, vector_certainty_instruction, vector_uncertainty_instruction, layer_step)}"
            f"{timestamp()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.pt"
    torch.save(
        {
            "scores": {
                variant: {f"layer_{idx}": torch.tensor(layer_scores, dtype=torch.float32)
                          for idx, layer_scores in layer_map.items()}
                for variant, layer_map in scores.items()
            },
            "example_idx": torch.arange(n_examples, dtype=torch.int64),
            "val_idx": torch.tensor(val_idx, dtype=torch.int64),
            "test_idx": torch.tensor(test_idx, dtype=torch.int64),
        },
        scores_path,
    )

    output_path = output_dir / "results.json"
    save_json(
        output_path,
        {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "max_examples": max_examples,
                "n_examples": n_examples,
                "batch_size": batch_size,
                "answer_prefix": answer_prefix,
                "label_token_prefix": label_token_prefix,
                "prompt_variants": variants,
                "positive_variant": positive_variant,
                "negative_variant": negative_variant,
                "positive_class_meaning": (
                    f"label=1 is {positive_variant}; higher projection onto "
                    f"{vector_certainty_instruction!r} - {vector_uncertainty_instruction!r} "
                    "predicts the positive class"
                ),
                "system_prompts": system_prompts,
                "vector_certainty_instruction": vector_certainty_instruction,
                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                "layer_step": layer_step,
                "layer_indices": layer_indices_resolved,
                "num_layers": num_layers,
                "val_fraction": val_fraction,
                "seed": seed,
                "n_val_examples": int(len(val_idx)),
                "n_test_examples": int(len(test_idx)),
                "score_definition": "dot(activation, steering_vector / ||steering_vector||) at last prompt token",
                "scores_path": str(scores_path),
            },
            "behavior_summary": {
                variant: {
                    "accuracy": float(np.mean([row["is_correct"] for row in rows])),
                    "mean_label_entropy": float(np.mean([row["label_entropy"] for row in rows])),
                }
                for variant, rows in behavior.items()
            },
            "layer_ranking_by_auroc_full": [
                {"layer": layer, "auroc_full": auroc} for layer, auroc in ranked
            ],
            "per_layer": per_layer_metrics,
            "behavior": behavior,
        },
    )
    return output_path


def latest_mmlu_steering_vector_probe_result(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_examples: Optional[int],
    answer_prefix: str = "ANSWER:",
    prompt_variants: Optional[Sequence[str]] = None,
    vector_certainty_instruction: str = "Be sure!",
    vector_uncertainty_instruction: str = "Be random!",
    layer_step: int = 4,
) -> Optional[Path]:
    variants = (
        list(prompt_variants)
        if prompt_variants is not None
        else ["format_instructed", "maximize_entropy"]
    )
    parent = get_results_dir() / "mmlu_steering_vector_probe" / slugify_model_name(model_name)
    match = latest_matching_dir(
        parent,
        _mmlu_steering_vector_probe_prefix(
            dataset_name,
            dataset_config,
            split,
            max_examples,
            answer_prefix,
            variants,
            vector_certainty_instruction,
            vector_uncertainty_instruction,
            layer_step,
        ),
    )
    return None if match is None else match / "results.json"
