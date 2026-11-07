#!/usr/bin/env python3
"""No-instruction steering dose response at a layer given on the command line.

Runs the work items of the figure-5 ``no-instruction-control`` group for one model
(vector source, strengths and steering conditions read from ``paper_manifest.yaml``)
with ``--layer`` in place of the group's ``fixed_layer_overrides``, so the manifest
is not edited. Everything else is ``run_no_instruction_control`` as called by
``run_paper.py`` (fp16, no vector normalisation, no norm preservation).

Safeguards:
  * the model loader is wrapped to abort if any parameter lands on CPU or disk;
  * the activation artifact and the steering vector must already exist (the
    auto-prepare paths of ``run_no_instruction_control`` would otherwise extract
    or recompute and overwrite artifacts);
  * a strength whose ``no_instruction_layer<L>_strength<s>_*/results.json`` exists
    is skipped (no overwrite; outputs are timestamped).

Example (camera-ready Llama-3-70B layer-44 check):

  AB_PAIRS_VARIANT=neutral python scripts/camera_ready_no_instruction_layer.py \\
      --model meta-llama/Meta-Llama-3-70B-Instruct --layer 44
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

import yaml  # noqa: E402

from be_sure_or_random import experiments  # noqa: E402
from be_sure_or_random.utils import (  # noqa: E402
    get_results_dir,
    get_steering_artifacts_dir,
    instruction_pair_label,
    slugify_model_name,
)

MANIFEST_PATH = ROOT / "paper_manifest.yaml"
GROUP_ID = "no-instruction-control"


def _group(manifest_path: Path) -> dict:
    with open(manifest_path) as handle:
        manifest = yaml.safe_load(handle)
    for group in manifest["figures"]["figure5"]["groups"]:
        if group["id"] == GROUP_ID:
            return group
    raise ValueError(f"{GROUP_ID} not found in figure5 of {manifest_path}")


def _checked_loader(original):
    """Wrap load_hf_model_and_tokenizer: abort if any parameter is not on CUDA."""

    def load(model_name, dtype=None, device=None, max_memory=None):
        model, tokenizer = original(model_name, dtype=dtype, device=device, max_memory=max_memory)
        device_map = getattr(model, "hf_device_map", None) or {}
        offloaded = sorted({str(v) for v in device_map.values() if str(v) in {"cpu", "disk"}})
        not_cuda = [name for name, p in model.named_parameters() if p.device.type != "cuda"]
        if offloaded or not_cuda:
            raise RuntimeError(
                f"{model_name}: offload detected (device_map {offloaded}, "
                f"{len(not_cuda)} parameters off CUDA, e.g. {not_cuda[:3]})"
            )
        devices = sorted({str(v) for v in device_map.values()}) or [str(next(model.parameters()).device)]
        print(f"[ok] {model_name}: all parameters on CUDA (devices {devices}, dtype {next(model.parameters()).dtype})")
        return model, tokenizer

    return load


def _existing(model_name: str, layer: int, strength: float) -> list[Path]:
    parent = get_results_dir() / "no_instruction_control" / slugify_model_name(model_name)
    if not parent.exists():
        return []
    return sorted(parent.glob(f"no_instruction_layer{layer}_strength{strength}_*/results.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--strength", type=float, action="append", default=None, help="Subset of the group's strengths.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("AB_PAIRS_VARIANT") != "neutral":
        raise RuntimeError("Run with AB_PAIRS_VARIANT=neutral")
    group = _group(args.manifest)
    matrix = group["matrix"]
    if args.model not in matrix["models"]:
        raise ValueError(f"{args.model} is not in {GROUP_ID}")
    strengths = [float(s) for s in matrix["strengths"]]
    if args.strength:
        unknown = set(args.strength) - set(strengths)
        if unknown:
            raise ValueError(f"Strengths {sorted(unknown)} are not in {GROUP_ID} ({strengths})")
        strengths = [s for s in strengths if s in set(args.strength)]
    conditions = list(matrix["steering_conditions"])
    print(
        f"[plan] {args.model} layer {args.layer} (manifest layer "
        f"{matrix.get('fixed_layer_overrides', {}).get(args.model)}) strengths {strengths} "
        f"conditions {conditions} vectors {matrix['vector_source']}"
    )

    experiments.load_hf_model_and_tokenizer = _checked_loader(experiments.load_hf_model_and_tokenizer)

    for vector_certainty, vector_uncertainty in matrix["vector_source"]:
        if experiments.latest_activation_artifact(args.model, vector_certainty, vector_uncertainty) is None:
            raise FileNotFoundError(f"No activation artifact for {args.model} | {vector_certainty} vs {vector_uncertainty}")
        vector_path = (
            get_steering_artifacts_dir()
            / "steering_vectors"
            / slugify_model_name(args.model)
            / instruction_pair_label(vector_certainty, vector_uncertainty)
            / f"layer_{args.layer}.pt"
        )
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing steering vector {vector_path}")
        for strength in strengths:
            done = _existing(args.model, args.layer, strength)
            if done:
                print(f"[skip] strength {strength}: {done[-1]}")
                continue
            output = experiments.run_no_instruction_control(
                args.model,
                vector_certainty,
                vector_uncertainty,
                strength,
                conditions,
                fixed_layer=args.layer,
                normalize_steering_vector=False,
                preserve_norm=False,
            )
            print(f"  -> {output}")


if __name__ == "__main__":
    main()
