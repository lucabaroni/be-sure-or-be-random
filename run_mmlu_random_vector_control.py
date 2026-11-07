from __future__ import annotations

import argparse

from be_sure_or_random.experiments import run_mmlu_random_vector_control


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU random-vector steering controls.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-32B")
    parser.add_argument("--vector-certainty-instruction", default="Aim for low entropy")
    parser.add_argument("--vector-uncertainty-instruction", default="Aim for high entropy")
    parser.add_argument("--steering-strength", type=float, default=2.0)
    parser.add_argument("--fixed-layer", type=int, default=56)
    parser.add_argument("--n-random-vectors", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dataset-name", default="cais/mmlu")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--answer-prefix", default="ANSWER:", help="Prefill text appended after the assistant turn header. Pass empty string for no prefill.")
    args = parser.parse_args()

    output_path = run_mmlu_random_vector_control(
        model_name=args.model_name,
        vector_certainty_instruction=args.vector_certainty_instruction,
        vector_uncertainty_instruction=args.vector_uncertainty_instruction,
        steering_strength=args.steering_strength,
        fixed_layer=args.fixed_layer,
        n_random_vectors=args.n_random_vectors,
        random_seed=args.random_seed,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        answer_prefix=args.answer_prefix,
    )
    print(output_path)


if __name__ == "__main__":
    main()
