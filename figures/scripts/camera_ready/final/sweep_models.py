"""Model set and data access for the cross-model sweep figures (Appendix "Results across models").

Sweep: 16 local models plus 5 Fireworks API models, standard system prompt, 0.9 option-mass validity rule
(common.is_valid_probability_record). Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-2B are excluded by the validity rule
(see scripts/analysis/check_exclusion_rule.py).
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]  # figures/scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    FIGURES_DIR,
    MODEL_META,
    clean_probability_records,
    latest_zero_shot_result_path,
    load_zero_shot_result,
    model_source,
    paired_entropy_differences,
)
from generate_all_models_standard_deltaH_kde_grid import SHORT_PAIR_LABEL  # noqa: E402

PAIRS: list[tuple[str, str]] = list(SHORT_PAIR_LABEL)  # (certainty, uncertainty) instruction, paper order

LOCAL_MODELS = [
    "qwen/Qwen3-1.7B", "qwen/Qwen3-4B", "qwen/Qwen3-8B", "qwen/Qwen3-14B", "qwen/Qwen3-32B",
    "qwen/Qwen3.5-4B", "qwen/Qwen3.5-9B", "qwen/Qwen3.5-27B", "qwen/Qwen3.5-35B-A3B",
    "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct",
    "allenai/Olmo-3.1-32B-Instruct", "allenai/Olmo-3.1-32B-Instruct-SFT", "allenai/Olmo-3.1-32B-Instruct-DPO",
    "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it",
]
API_MODELS = [
    "accounts/fireworks/models/deepseek-v3p1", "accounts/fireworks/models/deepseek-v3p2",
    "accounts/fireworks/models/glm-4p7", "accounts/fireworks/models/glm-5",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
]

VERSIONS: dict[str, list[str]] = {"submitted": LOCAL_MODELS + API_MODELS}
OUTPUT_DIRS = {version: FIGURES_DIR / "camera_ready" / f"sweep_{version}" for version in VERSIONS}
POST_FIX_LOCAL = 20260720


def family(model: str) -> str:
    return MODEL_META[model].family


def size_b(model: str) -> float:
    """Total parameters in billions."""
    return float(MODEL_META[model].params_b)


@lru_cache(maxsize=None)
def result_path(model: str, cert: str, unc: str) -> Path | None:
    path = latest_zero_shot_result_path(model, cert, unc)
    if path is not None and model_source(model) == "local":
        date = re.search(r"_(\d{8})_\d{6}", str(path))
        assert date and int(date.group(1)) >= POST_FIX_LOCAL, f"pre-fix local run: {path}"
    return path


@lru_cache(maxsize=None)
def valid_records(model: str, cert: str, unc: str) -> tuple[dict, ...]:
    path = result_path(model, cert, unc)
    return tuple(clean_probability_records(load_zero_shot_result(path))) if path else ()


def delta_h(model: str, cert: str, unc: str) -> list[float]:
    """Per-question ΔH = H(uncertainty) - H(certainty) in bits, over items valid under both instructions."""
    return paired_entropy_differences(list(valid_records(model, cert, unc)), cert, unc)


def overall_median_delta_h(model: str) -> float:
    """Mean over the 7 pairs of the per-pair median ΔH (used for ordering models)."""
    meds = [np.median(d) for d in (delta_h(model, c, u) for c, u in PAIRS) if d]
    return float(np.mean(meds)) if meds else float("nan")
