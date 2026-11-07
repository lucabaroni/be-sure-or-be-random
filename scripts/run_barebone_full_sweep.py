#!/usr/bin/env python3
"""Barebone multi-layer (every-4th) activation extract — one model per process.

Writes the same activation artifact format as extract_multi_layer_activations
under artifacts/neutral/steering_layers/activations/barebone/.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from be_sure_or_random.experiments import (  # noqa: E402
    _activation_artifacts_dir,
    _pair_messages,
    _system_prompt_template,
)
from be_sure_or_random.utils import (  # noqa: E402
    MultiLayerActivationExtractor,
    assert_ab_pairs_distinct_first_tokens,
    build_chat_input,
    get_layer_indices_every_n,
    get_num_layers,
    instruction_pair_label,
    load_ab_pairs,
    load_hf_model_and_tokenizer,
    slugify_model_name,
    tokenize_chat_text,
)

MANIFEST_PATH = ROOT / "paper_manifest.yaml"
TEMPLATE = "barebone"
LAYER_STEP = 4
MODELS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-27B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]


def _clear_gpus() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _assert_no_cpu_offload(model, model_name: str) -> None:
    cpu_params = sum(p.numel() for p in model.parameters() if p.device.type != "cuda")
    total = sum(p.numel() for p in model.parameters())
    if cpu_params:
        raise RuntimeError(
            f"{model_name}: {cpu_params}/{total} params not on CUDA (CPU/meta offload)."
        )
    print(f"[ok] {model_name}: all parameters on CUDA")


def _load_model(model_name: str):
    _clear_gpus()
    max_memory = None
    if "70B" in model_name or "70b" in model_name:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            raise RuntimeError(f"{model_name} needs 2 GPUs; only {n_gpus} visible")
        max_memory = {i: "75GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "0GiB"
        print(f"[load-config] {model_name}: max_memory={max_memory}")
    return load_hf_model_and_tokenizer(model_name, dtype=torch.float16, max_memory=max_memory)


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[str, str]]:
    manifest = yaml.safe_load(manifest_path.read_text())
    for group in manifest["figures"]["figure5"]["groups"]:
        if group["id"] == "steering-extract":
            return [tuple(p) for p in group["matrix"]["instruction_pairs"]]
    raise ValueError("steering-extract group not found in figure5")


def _activation_path(model_name: str, pair: Tuple[str, str]) -> Path:
    return (
        _activation_artifacts_dir(TEMPLATE)
        / f"{slugify_model_name(model_name)}_{instruction_pair_label(*pair)}_activations.pt"
    )


def _extract_pair(
    *,
    model,
    tokenizer,
    model_name: str,
    pair: Tuple[str, str],
    layer_indices: List[int],
    layer_step: int,
    ab_pairs,
) -> Path:
    certainty_instruction, uncertainty_instruction = pair
    system_prompt = _system_prompt_template(TEMPLATE).format(
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
    )
    extractor = MultiLayerActivationExtractor(model, layer_indices)
    activations_dict = {}
    first_device = next(model.parameters()).device
    pair_label = instruction_pair_label(certainty_instruction, uncertainty_instruction)
    with extractor:
        for ab_pair in tqdm(ab_pairs, desc=f"  {pair_label}", leave=False):
            for flip_ab in (False, True):
                options = (
                    [ab_pair["token_b"], ab_pair["token_a"]]
                    if flip_ab
                    else [ab_pair["token_a"], ab_pair["token_b"]]
                )
                for instruction in (certainty_instruction, uncertainty_instruction):
                    user_prompt, messages = _pair_messages(
                        system_prompt,
                        options[0],
                        options[1],
                        instruction,
                        prefill_assistant=False,
                    )
                    text = build_chat_input(tokenizer, model_name, messages)
                    inputs = tokenize_chat_text(tokenizer, text, return_tensors="pt").to(first_device)
                    with torch.no_grad():
                        model(**inputs)
                    activations_dict[user_prompt] = {
                        layer_name: activation.squeeze(0)
                        for layer_name, activation in extractor.get_activations_as_dict().items()
                    }
                    extractor.clear()

    output_path = _activation_path(model_name, pair)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "activations": activations_dict,
            "metadata": {
                "model_name": model_name,
                "certainty_instruction": certainty_instruction,
                "uncertainty_instruction": uncertainty_instruction,
                "num_layers": get_num_layers(model),
                "layer_indices": layer_indices,
                "layer_step": layer_step,
                "template_name": TEMPLATE,
            },
        },
        output_path,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--include-model", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _load_manifest_pairs(args.manifest)
    models = list(MODELS)
    if args.include_model:
        keep = set(args.include_model)
        models = [m for m in models if m in keep]
        if not models:
            raise ValueError("No models selected")

    ab_pairs = load_ab_pairs()
    print(
        f"[plan] template={TEMPLATE} layer_step={LAYER_STEP} models={models} "
        f"pairs={len(pairs)} ab_pairs={len(ab_pairs)}"
    )

    for model_name in models:
        missing = [
            pair
            for pair in pairs
            if args.overwrite or not _activation_path(model_name, pair).exists()
        ]
        if not missing:
            print(f"[skip] {model_name}: all {len(pairs)} activation files exist")
            continue

        print(f"[load] {model_name} | {len(missing)}/{len(pairs)} pair(s) missing")
        model, tokenizer = _load_model(model_name)
        try:
            _assert_no_cpu_offload(model, model_name)
            assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
            layer_indices = get_layer_indices_every_n(get_num_layers(model), LAYER_STEP)
            print(f"[layers] {model_name}: {layer_indices}")
            for pair in missing:
                path = _extract_pair(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    pair=pair,
                    layer_indices=layer_indices,
                    layer_step=LAYER_STEP,
                    ab_pairs=ab_pairs,
                )
                print(f"  -> {path}")
        finally:
            del model
            del tokenizer
            _clear_gpus()
            print(f"[unload] {model_name}")

    print("[done] barebone full-sweep extract for selected models complete")


if __name__ == "__main__":
    main()
