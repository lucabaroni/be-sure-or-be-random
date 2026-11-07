#!/usr/bin/env python3
"""Random-direction baseline for the MMLU steering-vector probe (rebuttal R2).

Re-extracts last-prompt-token residual activations of Qwen3.5-27B on all MMLU test
questions under three system prompts (format_instructed = baseline, maximize_entropy,
sandbag_via_uncertainty) at every 4th layer, and projects them onto
  * the unit Be sure! - Be random! steering vector of that layer (as in the probe run), and
  * K isotropic Gaussian random unit directions per layer (fixed seed).
Saves the projections plus full layer-56 activations, so AUROCs for any direction can be
recomputed offline. AUROC convention as in the stored probe: P(score(baseline) > score(prompt))
for the code-sign vector, i.e. P(prompt projection > baseline projection) in paper sign.

  # from the repository root
  AB_PAIRS_VARIANT=neutral python \
      scripts/camera_ready_probe_random_directions.py [--max-examples N] [--k 200]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
import be_sure_or_random.experiments as E  # noqa: E402

MODEL = "Qwen/Qwen3.5-27B"
VARIANTS = ["format_instructed", "maximize_entropy", "sandbag_via_uncertainty"]
SAVE_FULL_LAYERS = [56]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = E._load_mmlu_examples("cais/mmlu", "all", "test", args.max_examples)
    n = len(examples)
    out = E.get_results_dir() / "camera_ready_checks" / "probe_random_directions" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print("examples", n, "->", out, flush=True)

    with E._loaded_model_and_tokenizer(MODEL, torch.float16) as (model, tok):
        device = next(model.parameters()).device
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        layers = E.get_layer_indices_every_n(E.get_num_layers(model), 4)
        g = torch.Generator().manual_seed(args.seed)
        dirs = {}
        for L in layers:
            v = E._load_layer_vector(MODEL, "Be sure!", "Be random!", L, device=torch.device("cpu")).float()
            v = v / v.norm()
            r = torch.randn(args.k, v.numel(), generator=g)
            r = r / r.norm(dim=1, keepdim=True)
            dirs[L] = torch.cat([v[None], r]).T.contiguous().to(device)  # d x (1+K)
        proj = {var: {L: torch.full((n, 1 + args.k), float("nan")) for L in layers} for var in VARIANTS}
        full = {var: {L: torch.zeros(n, dirs[L].shape[0], dtype=torch.float16) for L in SAVE_FULL_LAYERS} for var in VARIANTS}
        ext = E.MultiLayerActivationExtractor(model, layers)
        with ext:
            for var in VARIANTS:
                prompts = []
                for ex in examples:
                    _, msgs = E._mmlu_messages(ex, prompt_variant=var)
                    prompts.append((int(ex["example_idx"]), E.build_chat_input(tok, MODEL, msgs, answer_prefix="ANSWER:")))
                for b in tqdm(range(0, n, args.batch_size), desc=var, mininterval=30):
                    batch = prompts[b:b + args.batch_size]
                    inputs = E.tokenize_chat_text(tok, [t for _, t in batch], return_tensors="pt", padding=True).to(device)
                    ext.clear()
                    with torch.no_grad():
                        model(**inputs)
                    acts = ext.get_activations_as_dict()
                    idx = torch.tensor([i for i, _ in batch])
                    for L in layers:
                        a = acts[f"layer_{L}"].float().to(device)
                        proj[var][L][idx] = (a @ dirs[L]).cpu()
                        if L in SAVE_FULL_LAYERS:
                            full[var][L][idx] = a.half().cpu()
    torch.save({"layers": layers, "variants": VARIANTS, "k": args.k, "seed": args.seed,
                "col0": "unit Be sure! - Be random! vector (code sign)", "proj": proj}, out / "projections.pt")
    torch.save({"variants": VARIANTS, "acts": full}, out / "acts_layer56_fp16.pt")

    def auroc(pos, neg):  # P(pos > neg)
        s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return float(E._binary_auroc(s, y))

    summary = {"model": MODEL, "n": n, "k": args.k, "per_layer": {}}
    for var in VARIANTS[1:]:
        summary["per_layer"][var] = {}
        for L in layers:
            P0 = proj["format_instructed"][L].numpy(); P1 = proj[var][L].numpy()
            au = np.array([auroc(P0[:, j], P1[:, j]) for j in range(P0.shape[1])])
            folded = np.maximum(au[1:], 1 - au[1:])
            summary["per_layer"][var][str(L)] = {
                "steering_auroc": au[0],
                "random_auroc_mean": float(au[1:].mean()),
                "random_folded_median": float(np.median(folded)),
                "random_folded_q95": float(np.quantile(folded, 0.95)),
                "random_folded_max": float(folded.max()),
                "steering_rank_vs_random_folded": int((folded >= max(au[0], 1 - au[0])).sum()),
            }
            print(var, L, json.dumps(summary["per_layer"][var][str(L)]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
