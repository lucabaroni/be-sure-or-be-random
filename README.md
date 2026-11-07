# Be sure! or Be random!

Code and results for **Language Models Can Coarsely Modulate Entropy Under Instruction** (NeurIPS 2026).

We instruct language models to raise or lower the entropy of their answer in a two-option choice with no correct answer, measure the shift in output entropy, extract a steering direction for it from the residual stream and test that direction on MMLU.

## Setup

```bash
uv sync                      # Python 3.12 or 3.13
uv sync --extra qwen3-fast   # optional, faster Qwen3.5 layers on CUDA
cp .env.example .env         # FIREWORKS_API_KEY for the API models, HF_TOKEN for gated models
```

## Reproduce the figures

```bash
bash scripts/download_results.sh   # paper results (about 0.7 GB) from the GitHub release
bash make_figures.sh               # CPU only
```

## Rerun the experiments

All runs are defined in `paper_manifest.yaml` and launched with `run_paper.py`:

```bash
python run_paper.py status
python run_paper.py run --group <group id>
```

| Paper | Manifest groups or scripts |
|---|---|
| Sec. 4.1, App. C (21-model sweep) | `local-model-sweep`, `api-model-sweep`, `local-model-sweep-barebone` |
| App. D (prompt variations, control pairs) | `prompt-variation-sweep`, `prompt-structure-ablations`, `prompt-structure-filler` |
| Sec. 4.2, App. E, F (steering) | `steering-extract`, `steering-compute`, `steering-all-layers`, `steering-cross-instruction`, `no-instruction-control`, `run_steering_vector_similarity.py` |
| Sec. 4.3, 4.4, App. G to I (MMLU) | `mmlu-entropy-steering`, `mmlu-random-vector-control`, `mmlu-prompt-conditions`, `mmlu-steering-vector-probe` |
| App. L (reasoning) | `scripts/launch_thinking_sampling_3prompt_sweep.sh` |
| App. M (calibration fine-tuning) | `calibration-ft-qwen3-14b` |

Local models run with Hugging Face Transformers on one or two 80 GB GPUs; `Llama-3-70B` needs two.

## Notes

- The five API models were queried through Fireworks without prefilling `ANSWER:`, and their answers were scored by first-token prefix match (`scripts/analysis/rescore_fireworks_first_token.py`).
- Models with more than 20% of items below the 0.9 probability-mass threshold are excluded from the sweep (`scripts/analysis/check_exclusion_rule.py`).
- The calibration fine-tuned model is not included. Train it with [chandar-lab/calibration-finetuning](https://github.com/chandar-lab/calibration-finetuning) (soft-target variant, default settings, LoRA merged into `Qwen/Qwen3-14B`) and place it at `models/Qwen3-14B-calib-soft`.

## Citation

```bibtex
@inproceedings{baroni2026entropy,
  title     = {Language Models Can Coarsely Modulate Entropy Under Instruction},
  author    = {Baroni, Luca and Ayonrinde, Kola and Feng, Shi and Radmard, Puria},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT
