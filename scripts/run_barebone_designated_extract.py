#!/usr/bin/env python3
"""Extract designated-layer steering vectors under the barebone (paper: minimal) prompt.

One model at a time, one load for all instruction pairs — avoids the CPU-offload
path that hit when 32B+27B shared GPUs. Writes under:

  artifacts/neutral/steering_layers/steering_vectors/barebone/<model>/<pair>/layer_<idx>.pt

``--layer MODEL=IDX`` replaces a model's designated layer (FIXED_LAYERS) and
``--output-dir`` writes the vectors under another root instead of the barebone
directory, e.g. for the Llama-3-70B layer-44 comparison:

  --include-model meta-llama/Meta-Llama-3-70B-Instruct \
  --layer meta-llama/Meta-Llama-3-70B-Instruct=44 \
  --output-dir artifacts/neutral/steering_layers/steering_vectors/barebone_designated_layer44
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from be_sure_or_random.experiments import (  # noqa: E402
    _pair_messages,
    _steering_vectors_dir,
    _system_prompt_template,
)
from be_sure_or_random.utils import (  # noqa: E402
    assert_ab_pairs_distinct_first_tokens,
    build_chat_input,
    get_layer_module,
    get_num_layers,
    instruction_pair_label,
    load_ab_pairs,
    load_hf_model_and_tokenizer,
    slugify_model_name,
    tokenize_chat_text,
)

MANIFEST_PATH = ROOT / "paper_manifest.yaml"
TEMPLATE = "barebone"
FIXED_LAYERS = {
    "Qwen/Qwen3-32B": 56,
    "Qwen/Qwen3.5-27B": 56,
    "meta-llama/Meta-Llama-3-70B-Instruct": 64,
}


class _EarlyExit(Exception):
    pass


class _DesignatedLayerExtractor:
    def __init__(self, model, layer_idx: int):
        self._module = get_layer_module(model, layer_idx)
        self._handle = None
        self.activation: Optional[torch.Tensor] = None

    def _hook(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.dim() == 3:
            self.activation = hidden_states[:, -1, :].detach().to("cpu", dtype=torch.float32).clone()
        else:
            self.activation = hidden_states.detach().to("cpu", dtype=torch.float32).clone()
        raise _EarlyExit

    def __enter__(self):
        self.activation = None
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[str, str]]:
    with open(manifest_path, "r") as handle:
        manifest = yaml.safe_load(handle)
    for group in manifest["figures"]["figure5"]["groups"]:
        if group["id"] == "steering-extract":
            return [tuple(pair) for pair in group["matrix"]["instruction_pairs"]]
    raise ValueError("steering-extract group not found in figure5")


def _vector_path(
    model_name: str, pair: Tuple[str, str], layer_idx: int, root: Optional[Path] = None
) -> Path:
    return (
        (root if root is not None else _steering_vectors_dir(TEMPLATE))
        / slugify_model_name(model_name)
        / instruction_pair_label(pair[0], pair[1])
        / f"layer_{layer_idx}.pt"
    )


def _assert_no_cpu_offload(model, model_name: str) -> None:
    cpu_params = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.device.type != "cuda":
            cpu_params += param.numel()
    if cpu_params:
        raise RuntimeError(
            f"{model_name}: {cpu_params}/{total} params not on CUDA "
            f"(CPU/meta offload). Aborting — free GPUs and retry one model at a time."
        )
    print(f"[ok] {model_name}: all parameters on CUDA")


def _clear_gpus() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _load_model(model_name: str):
    """Same loader as the rest of the repo; Llama-70B adds max_memory to block CPU offload."""
    _clear_gpus()
    max_memory = None
    if "70B" in model_name or "70b" in model_name:
        # Placement-only override: keep fp16 + tokenizer identical to load_hf_model_and_tokenizer.
        # Forbids CPU fallback after prior-model unload fragmented the allocator.
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            max_memory = {i: "75GiB" for i in range(n_gpus)}
        else:
            # A single large GPU (e.g. B200, 180 GB) also holds the fp16 weights (about 132 GiB).
            total_gib = int(torch.cuda.get_device_properties(0).total_memory / 2**30) if n_gpus else 0
            if total_gib < 140:
                raise RuntimeError(
                    f"{model_name} needs 2 GPUs or one GPU with >= 140 GiB; "
                    f"{n_gpus} visible ({total_gib} GiB)"
                )
            max_memory = {0: f"{total_gib - 5}GiB"}
        max_memory["cpu"] = "0GiB"
        print(f"[load-config] {model_name}: max_memory={max_memory}")
    return load_hf_model_and_tokenizer(
        model_name, dtype=torch.float16, max_memory=max_memory
    )


def _forward_capture(model, inputs, extractor: _DesignatedLayerExtractor) -> torch.Tensor:
    extractor.activation = None
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyExit:
        pass
    if extractor.activation is None:
        raise RuntimeError("Designated-layer hook did not fire")
    return extractor.activation.reshape(-1)


def _extract_pair(
    *,
    model,
    tokenizer,
    model_name: str,
    pair: Tuple[str, str],
    layer_idx: int,
    ab_pairs: List[Dict[str, str]],
) -> Tuple[torch.Tensor, int]:
    certainty_instruction, uncertainty_instruction = pair
    system_prompt = _system_prompt_template(TEMPLATE).format(
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
    )
    grouped: Dict[str, Dict[str, Optional[torch.Tensor]]] = defaultdict(
        lambda: {"certainty": None, "uncertainty": None}
    )
    extractor = _DesignatedLayerExtractor(model, layer_idx)
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
                pair_key = f"{options[0]}_or_{options[1]}"
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
                    activation = _forward_capture(model, inputs, extractor)
                    role = "certainty" if instruction == certainty_instruction else "uncertainty"
                    grouped[pair_key][role] = activation

    differences = []
    for entry in grouped.values():
        if entry["certainty"] is None or entry["uncertainty"] is None:
            continue
        differences.append(entry["certainty"] - entry["uncertainty"])
    if not differences:
        raise RuntimeError(f"No paired activations for {model_name} | {pair_label}")
    return torch.stack(differences).mean(dim=0), len(differences)


def _save_vector(
    *,
    model_name: str,
    pair: Tuple[str, str],
    layer_idx: int,
    steering_vector: torch.Tensor,
    num_pairs: int,
    root: Optional[Path] = None,
) -> Path:
    output_path = _vector_path(model_name, pair, layer_idx, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "steering_vector": steering_vector,
            "metadata": {
                "model_name": model_name,
                "layer_idx": layer_idx,
                "certainty_instruction": pair[0],
                "uncertainty_instruction": pair[1],
                "num_pairs": num_pairs,
                "template_name": TEMPLATE,
            },
        },
        output_path,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--include-model",
        action="append",
        default=None,
        help="Restrict to these models (repeatable). Default: all three.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        default=None,
        metavar="MODEL=IDX",
        help="Override a model's designated layer (repeatable). Default: FIXED_LAYERS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for the vectors (default: the barebone steering_vectors directory).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _load_manifest_pairs(args.manifest)
    fixed_layers = dict(FIXED_LAYERS)
    for spec in args.layer or []:
        model_name, _, idx = spec.rpartition("=")
        if model_name not in fixed_layers or not idx.isdigit():
            raise ValueError(f"Bad --layer {spec!r}; expected MODEL=IDX with MODEL in {list(fixed_layers)}")
        fixed_layers[model_name] = int(idx)
    root = args.output_dir
    if root is not None and not root.is_absolute():
        root = ROOT / root
    models = list(fixed_layers)
    if args.include_model:
        keep = set(args.include_model)
        models = [m for m in models if m in keep]
        if not models:
            raise ValueError("No models selected")

    ab_pairs = load_ab_pairs()
    print(
        f"[plan] template={TEMPLATE} models={models} pairs={len(pairs)} "
        f"ab_pairs={len(ab_pairs)} AB_PAIRS_VARIANT={os.environ.get('AB_PAIRS_VARIANT')} "
        f"layers={ {m: fixed_layers[m] for m in models} } root={root or _steering_vectors_dir(TEMPLATE)}"
    )

    for model_name in models:
        layer_idx = fixed_layers[model_name]
        missing = [
            pair
            for pair in pairs
            if args.overwrite or not _vector_path(model_name, pair, layer_idx, root).exists()
        ]
        if not missing:
            print(f"[skip] {model_name} layer {layer_idx}: already complete")
            continue

        print(f"[load] {model_name} | layer {layer_idx} | {len(missing)} pair(s)")
        model, tokenizer = _load_model(model_name)
        try:
            _assert_no_cpu_offload(model, model_name)
            assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs)
            num_layers = get_num_layers(model)
            if not 0 <= layer_idx < num_layers:
                raise ValueError(f"layer {layer_idx} out of range (num_layers={num_layers})")
            for pair in missing:
                vector, n_diffs = _extract_pair(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    pair=pair,
                    layer_idx=layer_idx,
                    ab_pairs=ab_pairs,
                )
                path = _save_vector(
                    model_name=model_name,
                    pair=pair,
                    layer_idx=layer_idx,
                    steering_vector=vector,
                    num_pairs=n_diffs,
                    root=root,
                )
                print(f"  -> {path} (n_diffs={n_diffs})")
        finally:
            del model
            del tokenizer
            _clear_gpus()
            print(f"[unload] {model_name}")

    print("[done] barebone designated-layer extract complete")


if __name__ == "__main__":
    main()
