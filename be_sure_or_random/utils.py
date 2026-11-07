from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"


def get_results_dir() -> Path:
    """Return the results root directory, namespaced by AB_PAIRS_VARIANT env var.

    When AB_PAIRS_VARIANT=neutral the results land in results/neutral/ so they
    never collide with the original-pairs runs that live directly in results/.
    """
    variant = os.environ.get("AB_PAIRS_VARIANT", "original")
    if variant == "neutral":
        return RESULTS_DIR / "neutral"
    return RESULTS_DIR


def get_steering_artifacts_dir() -> Path:
    """Return the steering artifacts base directory, namespaced by AB_PAIRS_VARIANT.

    Steering vectors extracted from neutral AB pairs are stored under
    artifacts/neutral/steering_layers/ so they never overwrite the standard ones.
    """
    variant = os.environ.get("AB_PAIRS_VARIANT", "original")
    if variant == "neutral":
        return ARTIFACTS_DIR / "neutral" / "steering_layers"
    return ARTIFACTS_DIR / "steering_layers"


def read_txt(path: Path | str) -> str:
    with open(path, "r") as handle:
        return handle.read().strip()


def load_json(path: Path | str):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(path: Path | str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def load_ab_pairs() -> List[Dict[str, str]]:
    """Load AB pairs, selecting the file based on AB_PAIRS_VARIANT env var.

    AB_PAIRS_VARIANT=original (default): data/ab_pairs.json
    AB_PAIRS_VARIANT=neutral:            data/ab_pairs_neutral.json
    """
    variant = os.environ.get("AB_PAIRS_VARIANT", "original")
    fname = "ab_pairs_neutral.json" if variant == "neutral" else "ab_pairs.json"
    return load_json(DATA_DIR / fname)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def slugify_instruction(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("!", "")
        .replace(",", "")
        .replace("?", "")
        .replace("/", "_")
    )


def instruction_label(certainty_instruction: str, uncertainty_instruction: str) -> str:
    return f"{slugify_instruction(certainty_instruction)}_{slugify_instruction(uncertainty_instruction)}"


def instruction_pair_label(certainty_instruction: str, uncertainty_instruction: str) -> str:
    return f"{slugify_instruction(certainty_instruction)}_vs_{slugify_instruction(uncertainty_instruction)}"


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_hf_model_and_tokenizer(model_name: str, dtype=None, device=None, max_memory=None):
    if device is None:
        device_map = "auto"
    else:
        device_map = device
    if dtype is None:
        dtype = torch.float16 if device != "mps" else torch.float32
    kwargs = {"device_map": device_map, "dtype": dtype}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def build_chat_input(tokenizer, model_name: str, messages: List[Dict[str, str]], answer_prefix: str = "ANSWER:"):
    is_qwen = "qwen" in model_name.lower()
    is_gemma = "gemma" in model_name.lower()
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if is_qwen or is_gemma:
        kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **kwargs)
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")]
    return text + answer_prefix


def tokenize_chat_text(tokenizer, texts, **kwargs):
    """Tokenize strings produced by ``build_chat_input`` / chat templates.

    Llama templates already include ``<|begin_of_text|>``. The default
    ``tokenizer(text)`` path would prepend a second BOS. Force
    ``add_special_tokens=False`` so Qwen/Gemma/Olmo stay unchanged and Llama
    keeps a single BOS.
    """
    kwargs["add_special_tokens"] = False
    return tokenizer(texts, **kwargs)


def option_first_token_id(tokenizer, option: str) -> int:
    """First token id of the space-prefixed option string used for scoring."""
    token_ids = tokenizer.encode(" " + option, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for option {option!r}.")
    return token_ids[0]


def assert_ab_pairs_distinct_first_tokens(tokenizer, ab_pairs: List[Dict[str, str]]) -> None:
    """Fail fast if any pair collapses to the same first answer token under this tokenizer."""
    collisions: List[str] = []
    for pair in ab_pairs:
        option_a = pair["token_a"]
        option_b = pair["token_b"]
        token_a = option_first_token_id(tokenizer, option_a)
        token_b = option_first_token_id(tokenizer, option_b)
        if token_a == token_b:
            decoded = tokenizer.decode(token_a)
            collisions.append(
                f"{option_a!r}/{option_b!r} → id={token_a} decoded={decoded!r}"
            )
    if collisions:
        joined = "; ".join(collisions)
        raise AssertionError(
            "Option pairs must have distinct first tokens for binary entropy scoring; "
            f"collisions: {joined}"
        )


def option_token_probability(tokenizer, probs: torch.Tensor, option: str) -> tuple[int, float, str]:
    token_id = option_first_token_id(tokenizer, option)
    return token_id, probs[token_id].item(), tokenizer.decode(token_id)


def option_label_probabilities(tokenizer, probs: torch.Tensor, labels: List[str], token_prefix: str = " ") -> Dict[str, Dict[str, float | int | str]]:
    records: Dict[str, Dict[str, float | int | str]] = {}
    raw_probs = []
    for label in labels:
        token_id = tokenizer.encode(token_prefix + label, add_special_tokens=False)[0]
        probability = float(probs[token_id].item())
        raw_probs.append(probability)
        records[label] = {
            "token_id": token_id,
            "prob": probability,
            "decoded": tokenizer.decode(token_id),
        }

    total = float(sum(raw_probs))
    for label in labels:
        records[label]["normalized_prob"] = float(records[label]["prob"]) / total if total > 0 else 0.0
    return records


def categorical_entropy(probabilities: List[float]) -> float:
    total = float(sum(probabilities))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for probability in probabilities:
        normalized = probability / total
        if normalized > 0:
            entropy -= normalized * math.log2(normalized)
    return float(entropy)


def latest_matching_dir(parent: Path, prefix: str) -> Optional[Path]:
    if not parent.exists():
        return None
    matches = [path for path in parent.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model.config, "num_layers"):
        return model.config.num_layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError(f"Could not determine number of layers for model type: {type(model)}")


def get_layer_module(model, layer_idx: int) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Could not access layers for model type: {type(model)}")


def get_layer_indices_every_n(num_layers: int, n: int = 4) -> List[int]:
    indices = list(range(0, num_layers, n))
    if (num_layers - 1) not in indices:
        indices.append(num_layers - 1)
    return sorted(indices)


class ActivationExtractor:
    def __init__(self, model, layer_idx: int):
        self.layer_module = get_layer_module(model, layer_idx)
        self.activations: List[torch.Tensor] = []
        self.hook_handle = None

    def _hook_fn(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if len(hidden_states.shape) == 3:
            activation = hidden_states[:, -1, :].clone().detach()
        else:
            activation = hidden_states.clone().detach()
        self.activations.append(activation)

    def __enter__(self):
        self.activations = []
        self.hook_handle = self.layer_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear(self):
        self.activations = []

    def get_activations(self) -> List[torch.Tensor]:
        return self.activations


class MultiLayerActivationExtractor:
    def __init__(self, model, layer_indices: List[int]):
        self.layer_indices = sorted(layer_indices)
        self.activations: Dict[int, torch.Tensor] = {}
        self.hook_handles = []
        self.layer_modules = {idx: get_layer_module(model, idx) for idx in layer_indices}

    def _create_hook_fn(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if len(hidden_states.shape) == 3:
                activation = hidden_states[:, -1, :].clone().detach().cpu()
            else:
                activation = hidden_states.clone().detach().cpu()
            self.activations[layer_idx] = activation

        return hook_fn

    def __enter__(self):
        self.activations = {}
        self.hook_handles = []
        for layer_idx, module in self.layer_modules.items():
            handle = module.register_forward_hook(self._create_hook_fn(layer_idx))
            self.hook_handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def clear(self):
        self.activations = {}

    def get_activations_as_dict(self) -> Dict[str, torch.Tensor]:
        return {f"layer_{idx}": activation for idx, activation in self.activations.items()}


class SteeringInjector:
    def __init__(
        self,
        model,
        layer_idx: int,
        steering_vector: torch.Tensor,
        strength: float = 1.0,
        certainty_instruction: str | None = None,
        uncertainty_instruction: str | None = None,
        preserve_norm: bool = False,
    ):
        self.layer_module = get_layer_module(model, layer_idx)
        self.steering_vector = steering_vector.to(next(model.parameters()).device)
        self.strength = strength
        self.certainty_instruction = certainty_instruction
        self.uncertainty_instruction = uncertainty_instruction
        self.current_instruction = None
        self.hook_handle = None
        self.preserve_norm = preserve_norm

    def set_current_instruction(self, instruction: Optional[str]):
        self.current_instruction = instruction

    def _signed_strength(self) -> float:
        if self.certainty_instruction is None and self.uncertainty_instruction is None:
            return self.strength
        if self.current_instruction == self.certainty_instruction:
            return self.strength
        if self.current_instruction == self.uncertainty_instruction:
            return -self.strength
        return 0.0

    def _hook_fn(self, module, inputs, output):
        signed_strength = self._signed_strength()
        hidden_states = output[0] if isinstance(output, tuple) else output
        modified = hidden_states.clone()
        steering_vector = self.steering_vector.to(device=modified.device, dtype=modified.dtype)
        if len(modified.shape) == 3:
            target = modified[:, -1, :]
            if self.preserve_norm:
                original_norm = torch.norm(target, dim=-1, keepdim=True)
            target = target + signed_strength * steering_vector
            if self.preserve_norm:
                new_norm = torch.norm(target, dim=-1, keepdim=True)
                target = target / torch.clamp(new_norm, min=1e-12) * original_norm
            modified[:, -1, :] = target
        else:
            modified = modified + signed_strength * steering_vector.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def __enter__(self):
        self.hook_handle = self.layer_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def compute_steering_vector(
    activations_certainty: List[torch.Tensor],
    activations_uncertainty: List[torch.Tensor],
) -> torch.Tensor:
    if len(activations_certainty) != len(activations_uncertainty):
        raise ValueError("Activation list sizes do not match.")
    if not activations_certainty:
        raise ValueError("No activations provided.")
    differences = []
    for certainty, uncertainty in zip(activations_certainty, activations_uncertainty):
        certainty = certainty.to(uncertainty.device)
        diff = certainty - uncertainty
        if len(diff.shape) == 2:
            diff = diff.mean(dim=0)
        differences.append(diff)
    return torch.stack(differences).mean(dim=0)


def choose_best_layer_from_all_layers(results_path: Path) -> int:
    payload = load_json(results_path)
    results = payload["results"]
    best_layer = None
    best_score = None
    certainty_instruction = payload["metadata"]["test_certainty_instruction"]
    uncertainty_instruction = payload["metadata"]["test_uncertainty_instruction"]
    for layer_key, layer_results in results.items():
        pairs: Dict[tuple[str, str, bool], Dict[str, float]] = {}
        for item in layer_results:
            prob_sum = item["option_a_prob"] + item["option_b_prob"]
            if prob_sum <= 0.9:
                continue
            p = item["option_a_prob"] / prob_sum
            entropy = 0.0 if p in (0, 1) else float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))
            key = (item["option_a"], item["option_b"], item["flip_ab"])
            pairs.setdefault(key, {})[item["instruction"]] = entropy
        diffs = []
        for grouped in pairs.values():
            if certainty_instruction in grouped and uncertainty_instruction in grouped:
                diffs.append(grouped[uncertainty_instruction] - grouped[certainty_instruction])
        score = abs(sum(diffs) / len(diffs)) if diffs else float("-inf")
        if best_score is None or score > best_score:
            best_score = score
            best_layer = int(layer_key.replace("layer_", ""))
    if best_layer is None:
        raise FileNotFoundError(f"Could not determine best layer from {results_path}")
    return best_layer


def choose_min_cosine_layer_from_similarities(
    similarities: Dict[int, float],
    exclude_last_n_layers: int = 5,
) -> int:
    """Layer index with lowest certainty/uncertainty activation cosine similarity.

    Excludes the ``exclude_last_n_layers`` highest indices (typically the final
    transformer blocks), matching the no-instruction control and supplementary figure.
    """
    if not similarities:
        raise ValueError("similarities must be non-empty")
    candidate_layers = sorted(similarities)
    if len(candidate_layers) > exclude_last_n_layers:
        excluded = set(candidate_layers[-exclude_last_n_layers:])
        filtered = {k: v for k, v in similarities.items() if k not in excluded}
    else:
        filtered = dict(similarities)
    if not filtered:
        raise ValueError("No layers left after excluding the last N layers")
    return min(filtered, key=filtered.get)


@dataclass
class RunRecord:
    label: str
    path: Optional[Path]
    exists: bool
