#!/usr/bin/env python3
"""Compute designated-layer steering-vector cosine-similarity matrices."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

from be_sure_or_random.experiments import _steering_vectors_dir
from be_sure_or_random.utils import (
    RESULTS_DIR,
    get_results_dir,
    instruction_pair_label,
    slugify_model_name,
    timestamp,
)

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "paper_manifest.yaml"
FIXED_LAYERS = {
    "Qwen/Qwen3-32B": 56,
    "Qwen/Qwen3.5-27B": 56,
    "meta-llama/Meta-Llama-3-70B-Instruct": 64,
}


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def find_manifest_group(manifest: dict[str, Any], group_id: str) -> dict[str, Any]:
    for figure in manifest["figures"].values():
        for group in figure["groups"]:
            if group["id"] == group_id:
                return group
    raise ValueError(f"No manifest group found with id {group_id!r}")


def matrix_from_group(group: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    matrix = group["matrix"]
    instruction_pairs = matrix.get("instruction_pairs") or matrix.get("vector_source")
    if not instruction_pairs:
        raise ValueError(f"Group {group.get('id')!r} has no instruction_pairs/vector_source")
    return list(matrix["models"]), [tuple(pair) for pair in instruction_pairs]


def vector_path(
    model_name: str,
    instruction_pair: tuple[str, str],
    layer_idx: int,
    template_name: str,
) -> Path:
    certainty, uncertainty = instruction_pair
    return (
        _steering_vectors_dir(template_name)
        / slugify_model_name(model_name)
        / instruction_pair_label(certainty, uncertainty)
        / f"layer_{layer_idx}.pt"
    )


def load_vector(
    model_name: str,
    instruction_pair: tuple[str, str],
    layer_idx: int,
    template_name: str,
) -> torch.Tensor:
    payload = torch.load(
        vector_path(model_name, instruction_pair, layer_idx, template_name),
        map_location="cpu",
        weights_only=False,
    )
    return payload["steering_vector"].float().reshape(-1)


def pair_label(instruction_pair: tuple[str, str]) -> str:
    return instruction_pair_label(instruction_pair[0], instruction_pair[1])


def compute_model_matrix(
    model_name: str,
    instruction_pairs: list[tuple[str, str]],
    *,
    fixed_layer: int,
    template_name: str,
) -> dict[str, Any]:
    labels = [pair_label(pair) for pair in instruction_pairs]
    matrix: list[list[float | None]] = []
    per_layer: dict[str, dict[str, list[dict[str, float | int]]]] = {}

    for left in instruction_pairs:
        row = []
        left_label = pair_label(left)
        per_layer[left_label] = {}
        for right in instruction_pairs:
            right_label = pair_label(right)
            left_path = vector_path(model_name, left, fixed_layer, template_name)
            right_path = vector_path(model_name, right, fixed_layer, template_name)
            if not left_path.exists() or not right_path.exists():
                row.append(None)
                per_layer[left_label][right_label] = []
                continue
            left_vector = load_vector(model_name, left, fixed_layer, template_name)
            right_vector = load_vector(model_name, right, fixed_layer, template_name)
            cosine = float(
                F.cosine_similarity(left_vector.unsqueeze(0), right_vector.unsqueeze(0)).item()
            )
            row.append(cosine)
            per_layer[left_label][right_label] = [
                {"layer_idx": fixed_layer, "cosine_similarity": cosine}
            ]
        matrix.append(row)

    return {"labels": labels, "matrix": matrix, "per_layer": per_layer}


def write_csv(path: Path, labels: list[str], matrix: list[list[float | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instruction_pair", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *["" if value is None else f"{value:.6f}" for value in row]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--group", default="steering-extract")
    parser.add_argument("--template", default="standard")
    parser.add_argument("--include-model", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    group = find_manifest_group(manifest, args.group)
    models, instruction_pairs = matrix_from_group(group)
    if args.include_model:
        keep = set(args.include_model)
        models = [m for m in models if m in keep]
    if not models:
        raise ValueError("No models selected")

    for model_name in models:
        if model_name not in FIXED_LAYERS:
            raise ValueError(f"No fixed layer for {model_name}")

    output_dir = args.output_dir or (
        get_results_dir() / "steering_vector_similarity" / f"designated_{args.template}_{timestamp()}"
    )
    # Also keep a stable alias path for the plotter
    stable_dir = get_results_dir() / "steering_vector_similarity" / f"designated_{args.template}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "manifest": str(args.manifest),
            "group": args.group,
            "models": models,
            "instruction_pairs": instruction_pairs,
            "use_fixed_layer_overrides": True,
            "fixed_layers_by_model": {m: FIXED_LAYERS[m] for m in models},
            "template": args.template,
            "ab_pairs_variant": os.environ.get("AB_PAIRS_VARIANT", "original"),
        },
        "results": {},
    }

    for model_name in models:
        model_result = compute_model_matrix(
            model_name,
            instruction_pairs,
            fixed_layer=FIXED_LAYERS[model_name],
            template_name=args.template,
        )
        results["results"][model_name] = model_result
        write_csv(
            output_dir / f"{slugify_model_name(model_name)}.csv",
            model_result["labels"],
            model_result["matrix"],
        )
        write_csv(
            stable_dir / f"{slugify_model_name(model_name)}.csv",
            model_result["labels"],
            model_result["matrix"],
        )
        print(f"\n{model_name} @ layer {FIXED_LAYERS[model_name]} ({args.template})")
        none_count = sum(1 for row in model_result["matrix"] for v in row if v is None)
        if none_count:
            print(f"  WARNING: {none_count} missing cells")

    for dest in (output_dir, stable_dir):
        json_path = dest / "similarity_matrices.json"
        with open(json_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
