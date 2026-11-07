from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import gaussian_kde

# numpy 2.x removed np.trapz; common.py still uses it for KDE degenerate-case normalization.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from be_sure_or_random.experiments import (
    latest_all_layers_result,
    latest_no_instruction_result,
    latest_prompt_variant_result,
    latest_zero_shot_api_result,
    latest_zero_shot_api_variant_result,
    latest_zero_shot_local_result,
)
from be_sure_or_random.utils import ROOT as REPO_ROOT
from be_sure_or_random.utils import load_json


FIGURES_DIR = REPO_ROOT / "figures"
# Allow override via FIGURES_OUTPUT_DIR env var (absolute or repo-relative).
_override = os.environ.get("FIGURES_OUTPUT_DIR")
if _override:
    GENERATED_DIR = Path(_override) if Path(_override).is_absolute() else REPO_ROOT / _override
else:
    GENERATED_DIR = FIGURES_DIR / "generated"
MANIFEST_PATH = REPO_ROOT / "paper_manifest.yaml"

ORANGE = "#D85A30"
GREEN = "#0F6E56"
PURPLE = "#534AB7"
# Pairwise entropy-difference lane: same hex/stroke as figures/schematic_red_blue.html (diffKde fill #534AB7, α=0.18).
SCHEMATIC_ENTROPY_DIFF_PURPLE = PURPLE
SLATE = "#64748B"
LIGHT_SLATE = "#94A3B8"
GRAY = "#6B7280"

TEMPLATE_COLORS = {
    "minimal": SLATE,
    "barebone": LIGHT_SLATE,
    "barebone2": LIGHT_SLATE,
    "standard": PURPLE,
    "extended": ORANGE,
}


@dataclass(frozen=True)
class ModelMeta:
    display_name: str
    params_b: float
    source: str
    family: str


# params_b is total parameters in billions; NaN = undisclosed. GLM-4.7 and GLM-5 corrected on 2026-11-02
MODEL_META: dict[str, ModelMeta] = {
    "qwen/Qwen3-0.6B": ModelMeta("Qwen3-0.6B", 0.6, "local", "Qwen"),
    "qwen/Qwen3-1.7B": ModelMeta("Qwen3-1.7B", 1.7, "local", "Qwen"),
    "qwen/Qwen3-4B": ModelMeta("Qwen3-4B", 4, "local", "Qwen"),
    "qwen/Qwen3-8B": ModelMeta("Qwen3-8B", 8, "local", "Qwen"),
    "qwen/Qwen3-14B": ModelMeta("Qwen3-14B", 14, "local", "Qwen"),
    "qwen/Qwen3-32B": ModelMeta("Qwen3-32B", 32, "local", "Qwen"),
    "qwen/Qwen3.5-0.8B": ModelMeta("Qwen3.5-0.8B", 0.8, "local", "Qwen"),
    "qwen/Qwen3.5-2B": ModelMeta("Qwen3.5-2B", 2, "local", "Qwen"),
    "qwen/Qwen3.5-4B": ModelMeta("Qwen3.5-4B", 4, "local", "Qwen"),
    "qwen/Qwen3.5-9B": ModelMeta("Qwen3.5-9B", 9, "local", "Qwen"),
    "qwen/Qwen3.5-27B": ModelMeta("Qwen3.5-27B", 27, "local", "Qwen"),
    "qwen/Qwen3.5-35B-A3B": ModelMeta("Qwen3.5-35B-A3B", 35, "local", "Qwen"),
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelMeta("Llama-3-8B", 8, "local", "Llama"),
    "meta-llama/Meta-Llama-3-70B-Instruct": ModelMeta("Llama-3-70B", 70, "local", "Llama"),
    "allenai/Olmo-3.1-32B-Instruct-SFT": ModelMeta("OLMo-3.1-32B-SFT", 32, "local", "OLMo"),
    "allenai/Olmo-3.1-32B-Instruct-DPO": ModelMeta("OLMo-3.1-32B-DPO", 32, "local", "OLMo"),
    "allenai/Olmo-3.1-32B-Instruct": ModelMeta("OLMo-3.1-32B", 32, "local", "OLMo"),
    "accounts/fireworks/models/deepseek-v3p1": ModelMeta("DeepSeek-V3.1", 671, "api", "DeepSeek"),
    "accounts/fireworks/models/deepseek-v3p2": ModelMeta("DeepSeek-V3.2", 671, "api", "DeepSeek"),
    "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelMeta("Llama-3.3-70B", 70, "api", "Llama"),
    "accounts/fireworks/models/glm-4p7": ModelMeta("GLM-4.7", 355, "api", "GLM"),
    "accounts/fireworks/models/glm-5": ModelMeta("GLM-5", 744, "api", "GLM"),
    "google/gemma-4-26b-a4b-it": ModelMeta("Gemma-4-26B-A4B", 26, "api", "Gemma"),
    "google/gemma-4-31b-it": ModelMeta("Gemma-4-31B", 31, "api", "Gemma"),
    "google/gemma-4-26B-A4B-it": ModelMeta("Gemma-4-26B-A4B", 26, "local", "Gemma"),
    "google/gemma-4-31B-it": ModelMeta("Gemma-4-31B", 31, "local", "Gemma"),
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, "r") as handle:
        return yaml.safe_load(handle)


def figure_group(figure_key: str, group_id: str) -> dict[str, Any]:
    manifest = load_manifest()
    for group in manifest["figures"][figure_key]["groups"]:
        if group["id"] == group_id:
            return group
    raise KeyError(f"Unknown group {group_id} for {figure_key}")


def ensure_generated_dir() -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_DIR


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    output_dir = ensure_generated_dir()
    svg_path = output_dir / f"{stem}.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    return [svg_path, pdf_path, png_path]


def print_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        print(path)


def probability_sum(record: dict[str, Any]) -> float:
    return float(record.get("option_a_prob", 0.0)) + float(record.get("option_b_prob", 0.0))


def is_valid_probability_record(record: dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    prob_a = record.get("option_a_prob")
    prob_b = record.get("option_b_prob")
    if prob_a is None or prob_b is None:
        return False
    total = float(prob_a) + float(prob_b)
    if not math.isfinite(total):
        return False
    return total > 0.9


def clean_probability_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if is_valid_probability_record(record)]


def normalized_option_a_probability(record: dict[str, Any]) -> float:
    total = probability_sum(record)
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    return float(record["option_a_prob"]) / total


def binary_entropy(prob_a: float, prob_b: float) -> float:
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    p = prob_a / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def record_entropy(record: dict[str, Any]) -> float:
    return binary_entropy(float(record["option_a_prob"]), float(record["option_b_prob"]))


def paired_entropy_differences(
    records: Iterable[dict[str, Any]],
    certainty_instruction: str,
    uncertainty_instruction: str,
) -> list[float]:
    paired: dict[tuple[str, str, bool], dict[str, float]] = {}
    for record in clean_probability_records(records):
        instruction = record.get("instruction")
        if instruction not in {certainty_instruction, uncertainty_instruction}:
            continue
        key = (record["option_a"], record["option_b"], bool(record.get("flip_ab", False)))
        paired.setdefault(key, {})[instruction] = record_entropy(record)
    diffs = []
    for grouped in paired.values():
        if certainty_instruction in grouped and uncertainty_instruction in grouped:
            diffs.append(grouped[uncertainty_instruction] - grouped[certainty_instruction])
    return diffs


def entropy_values_for_instruction(records: Iterable[dict[str, Any]], instruction: str) -> list[float]:
    return [record_entropy(record) for record in clean_probability_records(records) if record.get("instruction") == instruction]


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return float("nan")
    return float(np.mean(collected))


def mean_entropy_shift(records: Iterable[dict[str, Any]], certainty_instruction: str, uncertainty_instruction: str) -> float:
    return mean(paired_entropy_differences(records, certainty_instruction, uncertainty_instruction))


def load_zero_shot_result(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}")
    return payload


def load_all_layers_result(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or "results" not in payload:
        raise TypeError(f"Expected steering payload in {path}")
    return payload


def load_no_instruction_result(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or "results" not in payload:
        raise TypeError(f"Expected no-instruction payload in {path}")
    return payload


def layer_mean_shifts(payload: dict[str, Any]) -> dict[int, float]:
    metadata = payload["metadata"]
    certainty_instruction = metadata["test_certainty_instruction"]
    uncertainty_instruction = metadata["test_uncertainty_instruction"]
    summary: dict[int, float] = {}
    for layer_name, records in payload["results"].items():
        layer_idx = int(layer_name.replace("layer_", ""))
        summary[layer_idx] = mean_entropy_shift(records, certainty_instruction, uncertainty_instruction)
    return summary


def condition_entropies(payload: dict[str, Any]) -> dict[str, list[float]]:
    return {
        condition: [record_entropy(record) for record in clean_probability_records(records)]
        for condition, records in payload["results"].items()
    }


def kde_curve(
    values: Iterable[float],
    x_min: float,
    x_max: float,
    points: int = 256,
    *,
    kde_bandwidth_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    grid = np.linspace(x_min, x_max, points)
    if data.size == 0:
        return grid, np.zeros_like(grid)
    if data.size == 1 or np.allclose(data, data[0]):
        base_sigma = max((x_max - x_min) / 60.0, 1e-3)
        sigma = base_sigma * kde_bandwidth_factor if kde_bandwidth_factor is not None else base_sigma
        sigma = max(sigma, 1e-4)
        density = np.exp(-0.5 * ((grid - data[0]) / sigma) ** 2)
        density /= np.trapz(density, grid)
        return grid, density
    bw = kde_bandwidth_factor if kde_bandwidth_factor is not None else "scott"
    kde = gaussian_kde(data, bw_method=bw)
    return grid, kde(grid)


def plot_kde(
    ax: plt.Axes,
    values: Iterable[float],
    color: str,
    label: str,
    x_min: float,
    x_max: float,
    alpha: float = 0.18,
    *,
    linewidth: float = 2.0,
    kde_bandwidth_factor: float | None = None,
    zorder: int | None = None,
) -> None:
    grid, density = kde_curve(values, x_min, x_max, kde_bandwidth_factor=kde_bandwidth_factor)
    kw = {} if zorder is None else {"zorder": zorder}
    ax.plot(grid, density, color=color, linewidth=linewidth, label=label, **kw)
    ax.fill_between(grid, 0, density, color=color, alpha=alpha, **kw)


def style_distribution_axis(ax: plt.Axes, xlabel: str, title: str, x_limits: tuple[float, float]) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(*x_limits)


def display_model_name(model_name: str) -> str:
    meta = MODEL_META.get(model_name)
    if meta:
        return meta.display_name
    return model_name.split("/")[-1]


def model_params_b(model_name: str) -> float | None:
    meta = MODEL_META.get(model_name)
    if meta:
        return meta.params_b
    match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if match:
        return float(match.group(1))
    return None


def source_color(source: str) -> str:
    return ORANGE if source == "local" else GREEN


def model_source(model_name: str) -> str:
    meta = MODEL_META.get(model_name)
    if meta:
        return meta.source
    if model_name.startswith("accounts/fireworks/models/"):
        return "api"
    # OpenRouter-style org/model IDs used for hosted API sweeps.
    openrouter_prefixes = (
        "deepseek/",
        "meta-llama/llama-3.3",
    )
    if model_name.startswith(openrouter_prefixes):
        return "api"
    return "local"


def latest_zero_shot_result_path(
    model_name: str,
    certainty_instruction: str,
    uncertainty_instruction: str,
) -> Path | None:
    """Resolve a zero-shot entropy result for either a local or API model."""
    kind = "zero_shot_local" if model_source(model_name) == "local" else "zero_shot_api"
    return latest_result_path(
        kind,
        model_name=model_name,
        certainty_instruction=certainty_instruction,
        uncertainty_instruction=uncertainty_instruction,
    )


def latest_result_path(kind: str, **kwargs: Any) -> Path | None:
    if kind == "zero_shot_local":
        return latest_zero_shot_local_result(
            kwargs["model_name"],
            kwargs["certainty_instruction"],
            kwargs["uncertainty_instruction"],
        )
    if kind == "zero_shot_api":
        return latest_zero_shot_api_result(
            kwargs["model_name"],
            kwargs["certainty_instruction"],
            kwargs["uncertainty_instruction"],
        )
    if kind == "prompt_variants":
        return latest_prompt_variant_result(
            kwargs["model_name"],
            kwargs["certainty_instruction"],
            kwargs["uncertainty_instruction"],
            kwargs["template_name"],
        )
    if kind == "steering_all_layers":
        return latest_all_layers_result(
            kwargs["model_name"],
            kwargs["vector_certainty_instruction"],
            kwargs["vector_uncertainty_instruction"],
            kwargs["test_certainty_instruction"],
            kwargs["test_uncertainty_instruction"],
            kwargs["steering_strength"],
        )
    if kind == "no_instruction":
        return latest_no_instruction_result(
            kwargs["model_name"],
            kwargs["steering_strength"],
        )
    if kind == "prompt_variants_api":
        return latest_zero_shot_api_variant_result(
            kwargs["model_name"],
            kwargs["certainty_instruction"],
            kwargs["uncertainty_instruction"],
            kwargs["template_name"],
        )
    raise ValueError(f"Unsupported kind: {kind}")

