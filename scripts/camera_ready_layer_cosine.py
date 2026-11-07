#!/usr/bin/env python3
"""Recompute the cosine half of the layer-choice figure on the fixed prompt path.

For each model, extract residual-stream activations every 4 layers under the
standard prompt for the Be sure!/Be random! pair (neutral AB pairs), then save
per-layer cos(mean_certainty, mean_uncertainty). Also compares the recomputed
mean difference against the committed steering vector at each layer.

Run with AB_PAIRS_VARIANT=neutral.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from be_sure_or_random.experiments import extract_multi_layer_activations  # noqa: E402
from be_sure_or_random.utils import slugify_model_name  # noqa: E402

CERT, UNC = "Be sure!", "Be random!"
OUT_DIR = ROOT / "results" / "neutral" / "layer_condition_cosine"
VEC_DIR = ROOT / "artifacts" / "neutral" / "steering_layers" / "steering_vectors"


def main(models: list[str]) -> None:
    assert os.environ.get("AB_PAIRS_VARIANT") == "neutral", "set AB_PAIRS_VARIANT=neutral"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        path = extract_multi_layer_activations(model_name, CERT, UNC, layer_step=4)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        acts = blob["activations"]
        layer_names = list(next(iter(acts.values())).keys())
        rows = []
        for layer_name in layer_names:
            cert = torch.stack([v[layer_name].float() for k, v in acts.items() if k.endswith(CERT)])
            unc = torch.stack([v[layer_name].float() for k, v in acts.items() if k.endswith(UNC)])
            mc, mu = cert.mean(0), unc.mean(0)
            layer_idx = int("".join(ch for ch in layer_name if ch.isdigit()))
            row = {
                "layer": layer_idx,
                "n_certainty": len(cert),
                "n_uncertainty": len(unc),
                "cos_condition_means": torch.nn.functional.cosine_similarity(mc, mu, dim=0).item(),
            }
            vec_path = VEC_DIR / slugify_model_name(model_name) / "Be_sure_vs_Be_random" / f"layer_{layer_idx}.pt"
            if vec_path.exists():
                ref = torch.load(vec_path, map_location="cpu", weights_only=False)["steering_vector"].float()
                diff = mc - mu
                row["cos_vs_committed_vector"] = torch.nn.functional.cosine_similarity(diff, ref, dim=0).item()
                row["norm_ratio_vs_committed_vector"] = (diff.norm() / ref.norm()).item()
            rows.append(row)
        out = OUT_DIR / f"{slugify_model_name(model_name)}.json"
        out.write_text(json.dumps({"model_name": model_name, "pair": [CERT, UNC], "activations_path": str(path), "layers": rows}, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
