from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import yaml

from be_sure_or_random.experiments import (
    DEFAULT_MMLU_PROMPT_CONDITIONS,
    compute_all_steering_vectors,
    extract_multi_layer_activations,
    latest_activation_artifact,
    latest_all_layers_result,
    latest_fixed_layer_cross_instruction_result,
    latest_mmlu_entropy_steering_result,
    latest_mmlu_prompt_conditions_result,
    latest_mmlu_random_vector_control_result,
    latest_mmlu_steering_vector_probe_result,
    latest_no_instruction_result,
    latest_prompt_variant_result,
    latest_wildchat_steering_token_distributions_result,
    latest_zero_shot_api_result,
    latest_zero_shot_api_variant_result,
    latest_zero_shot_local_result,
    run_all_layers_steering,
    run_fixed_layer_cross_instruction_steering,
    run_mmlu_entropy_steering,
    run_mmlu_prompt_conditions,
    run_mmlu_random_vector_control,
    run_mmlu_steering_vector_probe,
    run_no_instruction_control,
    run_prompt_variant,
    run_prompt_variant_batch,
    run_wildchat_steering_token_distributions,
    run_zero_shot_api,
    run_zero_shot_api_variant,
    run_zero_shot_local_batch,
    run_zero_shot_local,
    steering_vectors_exist,
)


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "paper_manifest.yaml"


@dataclass
class WorkItem:
    figure: str
    group_id: str
    kind: str
    stage: str
    label: str
    model_name: Optional[str]
    runner: Callable[[], object]
    status_path: Callable[[], Optional[Path] | bool]
    child_args: Optional[Dict[str, object]] = None


@dataclass
class LocalModelBatch:
    figure: str
    group_id: str
    kind: str
    stage: str
    model_name: str
    items: List[WorkItem]
    template_name: Optional[str] = None


def load_manifest() -> Dict:
    with open(MANIFEST_PATH, "r") as handle:
        return yaml.safe_load(handle)


@contextlib.contextmanager
def _ab_variant_env(variant: str):
    old = os.environ.get("AB_PAIRS_VARIANT")
    os.environ["AB_PAIRS_VARIANT"] = variant
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("AB_PAIRS_VARIANT", None)
        else:
            os.environ["AB_PAIRS_VARIANT"] = old


def _wrap_with_variant(variant: str, fn: Callable) -> Callable:
    """Return a zero-arg callable that runs fn() with AB_PAIRS_VARIANT=variant."""
    def wrapper():
        with _ab_variant_env(variant):
            return fn()
    return wrapper


def _figure_key(user_value: str) -> str:
    return user_value if user_value.startswith("figure") else f"figure{user_value}"


def _selected_groups(group_filter: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not group_filter:
        return None
    return set(group_filter)


def _selected_models(include_models: Optional[Sequence[str]], exclude_models: Optional[Sequence[str]]) -> tuple[Optional[set[str]], set[str]]:
    include_set = set(include_models) if include_models else None
    exclude_set = set(exclude_models) if exclude_models else set()
    return include_set, exclude_set


def _model_is_selected(model_name: Optional[str], include_models: Optional[set[str]], exclude_models: set[str]) -> bool:
    if model_name is None:
        return True
    if include_models is not None and model_name not in include_models:
        return False
    if model_name in exclude_models:
        return False
    return True


def build_items(
    manifest: Dict,
    figure_filter: Optional[str],
    stage_filter: Optional[str],
    group_filter: Optional[Sequence[str]],
    include_models: Optional[Sequence[str]],
    exclude_models: Optional[Sequence[str]],
) -> List[WorkItem]:
    figures = manifest["figures"]
    items: List[WorkItem] = []
    selected_groups = _selected_groups(group_filter)
    selected_include_models, selected_exclude_models = _selected_models(include_models, exclude_models)
    for figure_name, figure_data in figures.items():
        if figure_filter and figure_name != _figure_key(figure_filter):
            continue
        for group in figure_data["groups"]:
            if selected_groups is not None and group["id"] not in selected_groups:
                continue
            items.extend(
                build_group_items(
                    figure_name,
                    group,
                    figures["figure5"]["groups"],
                    stage_filter,
                    selected_include_models,
                    selected_exclude_models,
                )
            )
    return items


def build_group_items(
    figure_name: str,
    group: Dict,
    figure5_groups: List[Dict],
    stage_filter: Optional[str],
    include_models: Optional[set[str]],
    exclude_models: set[str],
) -> List[WorkItem]:
    kind = group["kind"]
    items: List[WorkItem] = []

    def include(stage: str) -> bool:
        return stage_filter is None or stage_filter == stage

    if kind == "zero_shot_local" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for certainty_instruction, uncertainty_instruction in group["matrix"]["instruction_pairs"]:
                label = f"{model_name} | {certainty_instruction} vs {uncertainty_instruction}"
                items.append(
                    WorkItem(
                        figure=figure_name,
                        group_id=group["id"],
                        kind=kind,
                        stage="run",
                        label=label,
                        model_name=model_name,
                        runner=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction: run_zero_shot_local(m, c, u),
                        status_path=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction: latest_zero_shot_local_result(m, c, u),
                    )
                )
    elif kind == "zero_shot_api" and include("run"):
        # Per-group api_backend wins; else CLI/env API_BACKEND; else fireworks default.
        api_backend = group.get("api_backend") or os.environ.get("API_BACKEND")
        api_provider = group.get("api_provider") or os.environ.get("OPENROUTER_PROVIDER")
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for certainty_instruction, uncertainty_instruction in group["matrix"]["instruction_pairs"]:
                label = f"{model_name} | {certainty_instruction} vs {uncertainty_instruction}"
                items.append(
                    WorkItem(
                        figure=figure_name,
                        group_id=group["id"],
                        kind=kind,
                        stage="run",
                        label=label,
                        model_name=model_name,
                        runner=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, b=api_backend, p=api_provider: run_zero_shot_api(
                            m, c, u, api_backend=b, api_provider=p
                        ),
                        status_path=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction: latest_zero_shot_api_result(m, c, u),
                    )
                )
    elif kind == "prompt_variants_api" and include("run"):
        api_backend = group.get("api_backend") or os.environ.get("API_BACKEND")
        api_provider = group.get("api_provider") or os.environ.get("OPENROUTER_PROVIDER")
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for template_name in group["matrix"]["templates"]:
                for certainty_instruction, uncertainty_instruction in group["matrix"]["instruction_pairs"]:
                    label = f"{model_name} | {template_name} | {certainty_instruction} vs {uncertainty_instruction}"
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, t=template_name, b=api_backend, p=api_provider: run_zero_shot_api_variant(
                                m, c, u, t, api_backend=b, api_provider=p
                            ),
                            status_path=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, t=template_name: latest_zero_shot_api_variant_result(m, c, u, t),
                        )
                    )
    elif kind == "prompt_variants" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for template_name in group["matrix"]["templates"]:
                for certainty_instruction, uncertainty_instruction in group["matrix"]["instruction_pairs"]:
                    label = f"{model_name} | {template_name} | {certainty_instruction} vs {uncertainty_instruction}"
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, t=template_name: run_prompt_variant(m, c, u, t),
                            status_path=lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, t=template_name: latest_prompt_variant_result(m, c, u, t),
                        )
                    )
    elif kind == "steering_extract" and include("prepare"):
        layer_step = group["matrix"].get("layer_step", 4)
        ab_variant = group.get("ab_pairs_variant", "original")
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for certainty_instruction, uncertainty_instruction in group["matrix"]["instruction_pairs"]:
                variant_tag = f" | {ab_variant}" if ab_variant != "original" else ""
                label = f"{model_name} | activations | {certainty_instruction} vs {uncertainty_instruction}{variant_tag}"
                _raw_runner = lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction, s=layer_step: extract_multi_layer_activations(m, c, u, s)
                _raw_status = lambda m=model_name, c=certainty_instruction, u=uncertainty_instruction: latest_activation_artifact(m, c, u)
                items.append(
                    WorkItem(
                        figure=figure_name,
                        group_id=group["id"],
                        kind=kind,
                        stage="prepare",
                        label=label,
                        model_name=model_name,
                        runner=_wrap_with_variant(ab_variant, _raw_runner),
                        status_path=_wrap_with_variant(ab_variant, _raw_status),
                        # Isolate each extract: reloading 70B in-process after del OOMs.
                        child_args={
                            "model_name": model_name,
                            "certainty_instruction": certainty_instruction,
                            "uncertainty_instruction": uncertainty_instruction,
                            "layer_step": layer_step,
                            "ab_pairs_variant": ab_variant,
                        },
                    )
                )
    elif kind == "steering_compute" and include("prepare"):
        ab_variant = group.get("ab_pairs_variant", "original")
        extract_group = next(
            (g for g in figure5_groups
             if g["kind"] == "steering_extract"
             and g.get("ab_pairs_variant", "original") == ab_variant),
            None,
        )
        if extract_group is None:
            raise ValueError(f"No steering_extract group with ab_pairs_variant={ab_variant!r} found in figure5.")
        expected_pairs = [
            (model_name, certainty_instruction, uncertainty_instruction)
            for model_name in extract_group["matrix"]["models"]
            for certainty_instruction, uncertainty_instruction in extract_group["matrix"]["instruction_pairs"]
            if _model_is_selected(model_name, include_models, exclude_models)
        ]
        if expected_pairs:
            variant_tag = f" ({ab_variant})" if ab_variant != "original" else ""
            items.append(
                WorkItem(
                    figure=figure_name,
                    group_id=group["id"],
                    kind=kind,
                    stage="prepare",
                    label=f"compute steering vectors from extracted activations{variant_tag}",
                    model_name=None,
                    runner=_wrap_with_variant(ab_variant, lambda: compute_all_steering_vectors()),
                    status_path=_wrap_with_variant(ab_variant, lambda pairs=expected_pairs: all(steering_vectors_exist(m, c, u) for m, c, u in pairs)),
                )
            )
    elif kind == "steering_all_layers" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for vector_certainty_instruction, vector_uncertainty_instruction in group["matrix"]["vector_source"]:
                for test_certainty_instruction, test_uncertainty_instruction in group["matrix"]["test_pairs"]:
                    for strength in group["matrix"]["strengths"]:
                        label = (
                            f"{model_name} | all-layers | vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                            f"test {test_certainty_instruction} vs {test_uncertainty_instruction} | strength {strength}"
                        )
                        items.append(
                            WorkItem(
                                figure=figure_name,
                                group_id=group["id"],
                                kind=kind,
                                stage="run",
                                label=label,
                                model_name=model_name,
                                runner=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, tc=test_certainty_instruction, tu=test_uncertainty_instruction, s=strength: run_all_layers_steering(m, vc, vu, tc, tu, s),
                                status_path=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, tc=test_certainty_instruction, tu=test_uncertainty_instruction, s=strength: latest_all_layers_result(m, vc, vu, tc, tu, s),
                                # Isolate each load: reloading 70B in-process after del OOMs.
                                child_args={
                                    "model_name": model_name,
                                    "vector_certainty_instruction": vector_certainty_instruction,
                                    "vector_uncertainty_instruction": vector_uncertainty_instruction,
                                    "test_certainty_instruction": test_certainty_instruction,
                                    "test_uncertainty_instruction": test_uncertainty_instruction,
                                    "steering_strength": strength,
                                },
                            )
                        )
    elif kind == "no_instruction" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            fixed_layer = group["matrix"].get("fixed_layer_overrides", {}).get(model_name)
            for vector_certainty_instruction, vector_uncertainty_instruction in group["matrix"]["vector_source"]:
                for strength in group["matrix"]["strengths"]:
                    label = f"{model_name} | no-instruction | vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | strength {strength}"
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, s=strength, fl=fixed_layer, conds=group["matrix"]["steering_conditions"]: run_no_instruction_control(
                                m,
                                vc,
                                vu,
                                s,
                                conds,
                                fixed_layer=fl,
                                normalize_steering_vector=False,
                                preserve_norm=False,
                            ),
                            status_path=lambda m=model_name, s=strength: latest_no_instruction_result(m, s),
                            child_args={
                                "model_name": model_name,
                                "vector_certainty_instruction": vector_certainty_instruction,
                                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                                "steering_strength": strength,
                                "steering_conditions": group["matrix"]["steering_conditions"],
                                "fixed_layer": fixed_layer,
                                "normalize_steering_vector": False,
                                "preserve_norm": False,
                            },
                        )
                    )
    elif kind == "thinking_sampling" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            label = f"{model_name} | thinking sampling"
            items.append(
                WorkItem(
                    figure=figure_name,
                    group_id=group["id"],
                    kind=kind,
                    stage="run",
                    label=label,
                    model_name=model_name,
                    runner=lambda m=model_name: _run_thinking_sampling_subprocess(m),
                    status_path=lambda m=model_name: _latest_thinking_sampling_result(m),
                )
            )
    elif kind == "mmlu_entropy_steering" and include("run"):
        matrix = group["matrix"]
        dataset_name = matrix.get("dataset_name", "cais/mmlu")
        dataset_config = matrix.get("dataset_config", "all")
        split = matrix.get("split", "test")
        max_examples = matrix.get("max_examples", None)
        batch_size = matrix.get("batch_size", 32)
        steering_conditions = matrix.get("steering_conditions", ["none", "increase_entropy", "decrease_entropy"])
        answer_prefix = matrix.get("answer_prefix", "ANSWER:")
        prompt_variant = matrix.get("prompt_variant", "standard")
        ab_variant = group.get("ab_pairs_variant", "original")
        for model_name in matrix["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            fixed_layer = matrix["fixed_layer_overrides"][model_name]
            for vector_certainty_instruction, vector_uncertainty_instruction in matrix["vector_source"]:
                for strength in matrix["strengths"]:
                    variant_tag = "" if prompt_variant == "standard" else f" | {prompt_variant}"
                    ab_tag = "" if ab_variant == "original" else f" | ab={ab_variant}"
                    label = (
                        f"{model_name} | MMLU {dataset_config}/{split} max {max_examples} | "
                        f"vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                        f"layer {fixed_layer} | strength {strength}"
                        + ("" if answer_prefix else " | no-prefill")
                        + variant_tag
                        + ab_tag
                    )
                    _raw_runner = lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, s=strength, conds=steering_conditions, fl=fixed_layer, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, bs=batch_size, ap=answer_prefix, pv=prompt_variant: run_mmlu_entropy_steering(
                        m,
                        vc,
                        vu,
                        s,
                        conds,
                        fixed_layer=fl,
                        dataset_name=dn,
                        dataset_config=dc,
                        split=sp,
                        max_examples=mx,
                        batch_size=bs,
                        answer_prefix=ap,
                        prompt_variant=pv,
                    )
                    _raw_status = lambda m=model_name, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, fl=fixed_layer, s=strength, ap=answer_prefix, pv=prompt_variant, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction: latest_mmlu_entropy_steering_result(
                        m,
                        dn,
                        dc,
                        sp,
                        mx,
                        fl,
                        s,
                        ap,
                        pv,
                        vc,
                        vu,
                    )
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=_wrap_with_variant(ab_variant, _raw_runner),
                            status_path=_wrap_with_variant(ab_variant, _raw_status),
                            child_args={
                                "model_name": model_name,
                                "vector_certainty_instruction": vector_certainty_instruction,
                                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                                "steering_strength": strength,
                                "steering_conditions": steering_conditions,
                                "fixed_layer": fixed_layer,
                                "dataset_name": dataset_name,
                                "dataset_config": dataset_config,
                                "split": split,
                                "max_examples": max_examples,
                                "batch_size": batch_size,
                                "answer_prefix": answer_prefix,
                                "prompt_variant": prompt_variant,
                                "ab_pairs_variant": ab_variant,
                            },
                        )
                    )
    elif kind == "mmlu_prompt_conditions" and include("run"):
        matrix = group["matrix"]
        dataset_name = matrix.get("dataset_name", "cais/mmlu")
        dataset_config = matrix.get("dataset_config", "all")
        split = matrix.get("split", "test")
        max_examples = matrix.get("max_examples", None)
        batch_size = matrix.get("batch_size", 16)
        answer_prefix = matrix.get("answer_prefix", "ANSWER:")
        prompt_variants = matrix.get("prompt_variants", list(DEFAULT_MMLU_PROMPT_CONDITIONS))
        ab_variant = group.get("ab_pairs_variant", "original")
        for model_name in matrix["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            variants_tag = ", ".join(prompt_variants)
            label = (
                f"{model_name} | MMLU prompt-only {dataset_config}/{split} max {max_examples} | "
                f"{variants_tag}"
                + ("" if answer_prefix else " | no-prefill")
                + ("" if ab_variant == "original" else f" | ab={ab_variant}")
            )
            _raw_runner = lambda m=model_name, pvs=prompt_variants, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, bs=batch_size, ap=answer_prefix: run_mmlu_prompt_conditions(
                m,
                pvs,
                dataset_name=dn,
                dataset_config=dc,
                split=sp,
                max_examples=mx,
                batch_size=bs,
                answer_prefix=ap,
            )
            _raw_status = lambda m=model_name, pvs=prompt_variants, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, ap=answer_prefix: latest_mmlu_prompt_conditions_result(
                m,
                dn,
                dc,
                sp,
                mx,
                ap,
                pvs,
            )
            items.append(
                WorkItem(
                    figure=figure_name,
                    group_id=group["id"],
                    kind=kind,
                    stage="run",
                    label=label,
                    model_name=model_name,
                    runner=_wrap_with_variant(ab_variant, _raw_runner),
                    status_path=_wrap_with_variant(ab_variant, _raw_status),
                    child_args={
                        "model_name": model_name,
                        "prompt_variants": prompt_variants,
                        "dataset_name": dataset_name,
                        "dataset_config": dataset_config,
                        "split": split,
                        "max_examples": max_examples,
                        "batch_size": batch_size,
                        "answer_prefix": answer_prefix,
                        "ab_pairs_variant": ab_variant,
                    },
                )
            )
    elif kind == "mmlu_steering_vector_probe" and include("run"):
        matrix = group["matrix"]
        dataset_name = matrix.get("dataset_name", "cais/mmlu")
        dataset_config = matrix.get("dataset_config", "all")
        split = matrix.get("split", "test")
        max_examples = matrix.get("max_examples", None)
        batch_size = matrix.get("batch_size", 16)
        answer_prefix = matrix.get("answer_prefix", "ANSWER:")
        prompt_variants = matrix.get(
            "prompt_variants", ["format_instructed", "maximize_entropy"]
        )
        positive_variant = matrix.get("positive_variant", "format_instructed")
        negative_variant = matrix.get("negative_variant", "maximize_entropy")
        layer_step = matrix.get("layer_step", 4)
        val_fraction = matrix.get("val_fraction", 0.2)
        seed = matrix.get("seed", 0)
        ab_variant = group.get("ab_pairs_variant", "original")
        for model_name in matrix["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            for vector_certainty_instruction, vector_uncertainty_instruction in matrix["vector_source"]:
                variants_tag = ", ".join(prompt_variants)
                ab_tag = "" if ab_variant == "original" else f" | ab={ab_variant}"
                label = (
                    f"{model_name} | MMLU steering-vector probe {dataset_config}/{split} "
                    f"max {max_examples} | {variants_tag} | "
                    f"vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                    f"step {layer_step}"
                    + ("" if answer_prefix else " | no-prefill")
                    + ab_tag
                )
                _raw_runner = (
                    lambda m=model_name,
                    vc=vector_certainty_instruction,
                    vu=vector_uncertainty_instruction,
                    pvs=prompt_variants,
                    pos=positive_variant,
                    neg=negative_variant,
                    dn=dataset_name,
                    dc=dataset_config,
                    sp=split,
                    mx=max_examples,
                    bs=batch_size,
                    ap=answer_prefix,
                    ls=layer_step,
                    vf=val_fraction,
                    sd=seed: run_mmlu_steering_vector_probe(
                        m,
                        vc,
                        vu,
                        prompt_variants=pvs,
                        positive_variant=pos,
                        negative_variant=neg,
                        dataset_name=dn,
                        dataset_config=dc,
                        split=sp,
                        max_examples=mx,
                        batch_size=bs,
                        answer_prefix=ap,
                        layer_step=ls,
                        val_fraction=vf,
                        seed=sd,
                    )
                )
                _raw_status = (
                    lambda m=model_name,
                    vc=vector_certainty_instruction,
                    vu=vector_uncertainty_instruction,
                    pvs=prompt_variants,
                    dn=dataset_name,
                    dc=dataset_config,
                    sp=split,
                    mx=max_examples,
                    ap=answer_prefix,
                    ls=layer_step: latest_mmlu_steering_vector_probe_result(
                        m,
                        dn,
                        dc,
                        sp,
                        mx,
                        ap,
                        pvs,
                        vc,
                        vu,
                        ls,
                    )
                )
                items.append(
                    WorkItem(
                        figure=figure_name,
                        group_id=group["id"],
                        kind=kind,
                        stage="run",
                        label=label,
                        model_name=model_name,
                        runner=_wrap_with_variant(ab_variant, _raw_runner),
                        status_path=_wrap_with_variant(ab_variant, _raw_status),
                        child_args={
                            "model_name": model_name,
                            "vector_certainty_instruction": vector_certainty_instruction,
                            "vector_uncertainty_instruction": vector_uncertainty_instruction,
                            "prompt_variants": prompt_variants,
                            "positive_variant": positive_variant,
                            "negative_variant": negative_variant,
                            "dataset_name": dataset_name,
                            "dataset_config": dataset_config,
                            "split": split,
                            "max_examples": max_examples,
                            "batch_size": batch_size,
                            "answer_prefix": answer_prefix,
                            "layer_step": layer_step,
                            "val_fraction": val_fraction,
                            "seed": seed,
                            "ab_pairs_variant": ab_variant,
                        },
                    )
                )
    elif kind == "mmlu_random_vector_control" and include("run"):
        matrix = group["matrix"]
        dataset_name = matrix.get("dataset_name", "cais/mmlu")
        dataset_config = matrix.get("dataset_config", "all")
        split = matrix.get("split", "test")
        max_examples = matrix.get("max_examples", None)
        batch_size = matrix.get("batch_size", 32)
        n_random_vectors = matrix.get("n_random_vectors", 8)
        random_seed = matrix.get("random_seed", 0)
        answer_prefix = matrix.get("answer_prefix", "ANSWER:")
        prompt_variant = matrix.get("prompt_variant", "standard")
        ab_variant = group.get("ab_pairs_variant", "original")
        for model_name in matrix["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            fixed_layer = matrix["fixed_layer_overrides"][model_name]
            for vector_certainty_instruction, vector_uncertainty_instruction in matrix["vector_source"]:
                for strength in matrix["strengths"]:
                    variant_tag = "" if prompt_variant == "standard" else f" | {prompt_variant}"
                    ab_tag = "" if ab_variant == "original" else f" | ab={ab_variant}"
                    label = (
                        f"{model_name} | MMLU random-vector control {dataset_config}/{split} max {max_examples} | "
                        f"reference vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                        f"layer {fixed_layer} | strength {strength} | seed {random_seed} | n {n_random_vectors}"
                        + ("" if answer_prefix else " | no-prefill")
                        + variant_tag
                        + ab_tag
                    )
                    _raw_runner = lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, s=strength, fl=fixed_layer, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, bs=batch_size, nr=n_random_vectors, rs=random_seed, ap=answer_prefix, pv=prompt_variant: run_mmlu_random_vector_control(
                        m,
                        vc,
                        vu,
                        s,
                        fixed_layer=fl,
                        n_random_vectors=nr,
                        random_seed=rs,
                        dataset_name=dn,
                        dataset_config=dc,
                        split=sp,
                        max_examples=mx,
                        batch_size=bs,
                        answer_prefix=ap,
                        prompt_variant=pv,
                    )
                    _raw_status = lambda m=model_name, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, fl=fixed_layer, s=strength, rs=random_seed, nr=n_random_vectors, ap=answer_prefix, pv=prompt_variant, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction: latest_mmlu_random_vector_control_result(
                        m,
                        dn,
                        dc,
                        sp,
                        mx,
                        fl,
                        s,
                        rs,
                        nr,
                        ap,
                        pv,
                        vc,
                        vu,
                    )
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=_wrap_with_variant(ab_variant, _raw_runner),
                            status_path=_wrap_with_variant(ab_variant, _raw_status),
                            child_args={
                                "model_name": model_name,
                                "vector_certainty_instruction": vector_certainty_instruction,
                                "vector_uncertainty_instruction": vector_uncertainty_instruction,
                                "steering_strength": strength,
                                "fixed_layer": fixed_layer,
                                "n_random_vectors": n_random_vectors,
                                "random_seed": random_seed,
                                "dataset_name": dataset_name,
                                "dataset_config": dataset_config,
                                "split": split,
                                "max_examples": max_examples,
                                "batch_size": batch_size,
                                "answer_prefix": answer_prefix,
                                "prompt_variant": prompt_variant,
                                "ab_pairs_variant": ab_variant,
                            },
                        )
                    )
    elif kind == "wildchat_steering_token_distributions" and include("run"):
        matrix = group["matrix"]
        dataset_name = matrix.get("dataset_name", "allenai/WildChat-1M")
        dataset_config = matrix.get("dataset_config", None)
        split = matrix.get("split", "train")
        max_examples = matrix.get("max_examples", 50)
        max_new_tokens = matrix.get("max_new_tokens", 64)
        batch_size = matrix.get("batch_size", 4)
        seed = matrix.get("seed", 0)
        user_field = matrix.get("user_field", "user_prompt")
        conversation_field = matrix.get("conversation_field", "conversation")
        do_sample = matrix.get("do_sample", False)
        temperature = matrix.get("temperature", 1.0)
        streaming = matrix.get("streaming", True)
        shuffle_buffer_size = matrix.get("shuffle_buffer_size", 1000)
        steering_conditions = matrix.get(
            "steering_conditions", ["none", "decrease_entropy", "increase_entropy"]
        )
        for model_name in matrix["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            fixed_layer = matrix["fixed_layer_overrides"][model_name]
            for vector_certainty_instruction, vector_uncertainty_instruction in matrix["vector_source"]:
                for strength in matrix["strengths"]:
                    label = (
                        f"{model_name} | WildChat {dataset_name}/{split} max {max_examples} | "
                        f"vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                        f"layer {fixed_layer} | strength {strength}"
                    )
                    items.append(
                        WorkItem(
                            figure=figure_name,
                            group_id=group["id"],
                            kind=kind,
                            stage="run",
                            label=label,
                            model_name=model_name,
                            runner=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, s=strength, conds=steering_conditions, fl=fixed_layer, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, mt=max_new_tokens, bs=batch_size, sd=seed, uf=user_field, cf=conversation_field, ds=do_sample, tp=temperature, st=streaming, sbs=shuffle_buffer_size: run_wildchat_steering_token_distributions(
                                m,
                                vc,
                                vu,
                                s,
                                conds,
                                fixed_layer=fl,
                                dataset_name=dn,
                                dataset_config=dc,
                                split=sp,
                                max_examples=mx,
                                max_new_tokens=mt,
                                batch_size=bs,
                                seed=sd,
                                user_field=uf,
                                conversation_field=cf,
                                do_sample=ds,
                                temperature=tp,
                                streaming=st,
                                shuffle_buffer_size=sbs,
                            ),
                            status_path=lambda m=model_name, dn=dataset_name, dc=dataset_config, sp=split, mx=max_examples, fl=fixed_layer, s=strength, mt=max_new_tokens: latest_wildchat_steering_token_distributions_result(
                                m,
                                dn,
                                dc,
                                sp,
                                mx,
                                fl,
                                s,
                                mt,
                            ),
                        )
                    )
    elif kind == "steering_fixed_layer_cross_instruction" and include("run"):
        for model_name in group["matrix"]["models"]:
            if not _model_is_selected(model_name, include_models, exclude_models):
                continue
            fixed_layer = group["matrix"]["fixed_layer_overrides"][model_name]
            for vector_certainty_instruction, vector_uncertainty_instruction in group["matrix"]["vector_source"]:
                for test_certainty_instruction, test_uncertainty_instruction in group["matrix"]["test_pairs"]:
                    for strength in group["matrix"]["strengths"]:
                        label = (
                            f"{model_name} | fixed-layer {fixed_layer} | vector {vector_certainty_instruction} vs {vector_uncertainty_instruction} | "
                            f"test {test_certainty_instruction} vs {test_uncertainty_instruction} | strength {strength}"
                        )
                        items.append(
                            WorkItem(
                                figure=figure_name,
                                group_id=group["id"],
                                kind=kind,
                                stage="run",
                                label=label,
                                model_name=model_name,
                                runner=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, tc=test_certainty_instruction, tu=test_uncertainty_instruction, fl=fixed_layer, s=strength: run_fixed_layer_cross_instruction_steering(m, vc, vu, tc, tu, fl, s),
                                status_path=lambda m=model_name, vc=vector_certainty_instruction, vu=vector_uncertainty_instruction, tc=test_certainty_instruction, tu=test_uncertainty_instruction, fl=fixed_layer, s=strength: latest_fixed_layer_cross_instruction_result(m, vc, vu, tc, tu, fl, s),
                                child_args={
                                    "model_name": model_name,
                                    "vector_certainty_instruction": vector_certainty_instruction,
                                    "vector_uncertainty_instruction": vector_uncertainty_instruction,
                                    "test_certainty_instruction": test_certainty_instruction,
                                    "test_uncertainty_instruction": test_uncertainty_instruction,
                                    "fixed_layer": fixed_layer,
                                    "steering_strength": strength,
                                },
                            )
                        )
    return items


def print_status(items: List[WorkItem]) -> None:
    print(f"Found {len(items)} work items.\n")
    for item in items:
        status = item.status_path()
        exists = bool(status)
        rendered = str(status) if isinstance(status, Path) else ("ready" if exists else "missing")
        print(f"[{'x' if exists else ' '}] {item.figure} / {item.group_id} / {item.stage}")
        print(f"    {item.label}")
        print(f"    {rendered}")


def _group_local_model_batches(items: List[WorkItem]) -> List[LocalModelBatch]:
    """Group zero_shot_local and prompt_variants so each model loads once in a child process."""
    batches: List[LocalModelBatch] = []
    zero_shot_grouped: Dict[tuple[str, str, str, str, str], List[WorkItem]] = {}
    prompt_grouped: Dict[tuple[str, str, str, str, str, str], List[WorkItem]] = {}
    for item in items:
        if item.model_name is None:
            continue
        if item.kind == "zero_shot_local":
            key = (item.figure, item.group_id, item.kind, item.stage, item.model_name)
            zero_shot_grouped.setdefault(key, []).append(item)
        elif item.kind == "prompt_variants":
            # label: "{model} | {template} | {cert} vs {uncert}"
            parts = item.label.split(" | ")
            if len(parts) < 3:
                continue
            template_name = parts[1]
            key = (item.figure, item.group_id, item.kind, item.stage, item.model_name, template_name)
            prompt_grouped.setdefault(key, []).append(item)
    for (figure, group_id, kind, stage, model_name), grouped_items in zero_shot_grouped.items():
        batches.append(
            LocalModelBatch(
                figure=figure,
                group_id=group_id,
                kind=kind,
                stage=stage,
                model_name=model_name,
                items=grouped_items,
            )
        )
    for (figure, group_id, kind, stage, model_name, template_name), grouped_items in prompt_grouped.items():
        batches.append(
            LocalModelBatch(
                figure=figure,
                group_id=group_id,
                kind=kind,
                stage=stage,
                model_name=model_name,
                items=grouped_items,
                template_name=template_name,
            )
        )
    return batches


def _instruction_pairs_from_batch_labels(batch: LocalModelBatch) -> List[List[str]]:
    instruction_pairs = []
    for item in batch.items:
        if batch.kind == "prompt_variants":
            _, _, instructions = item.label.split(" | ", 2)
        else:
            _, instructions = item.label.split(" | ", 1)
        certainty_instruction, uncertainty_instruction = instructions.split(" vs ", 1)
        instruction_pairs.append([certainty_instruction, uncertainty_instruction])
    return instruction_pairs


def _run_local_zero_shot_batch_subprocess(batch: LocalModelBatch, force: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "run_paper.py"),
        "run",
        "--execution-mode",
        "child-local-zero-shot-batch",
        "--child-model",
        batch.model_name,
        "--child-instruction-pairs-json",
        json.dumps(_instruction_pairs_from_batch_labels(batch)),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True, cwd=ROOT)


def _run_prompt_variant_batch_subprocess(batch: LocalModelBatch, force: bool) -> None:
    if not batch.template_name:
        raise ValueError("Prompt-variant batch is missing template_name.")
    command = [
        sys.executable,
        str(ROOT / "run_paper.py"),
        "run",
        "--execution-mode",
        "child-prompt-variant-batch",
        "--child-model",
        batch.model_name,
        "--child-payload-json",
        json.dumps({"template_name": batch.template_name}),
        "--child-instruction-pairs-json",
        json.dumps(_instruction_pairs_from_batch_labels(batch)),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True, cwd=ROOT)


def _run_thinking_sampling_subprocess(model_name: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "run_thinking_sampling.py"),
        "--model",
        model_name,
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def _latest_thinking_sampling_result(model_name: str) -> Optional[Path]:
    from be_sure_or_random.utils import get_results_dir, slugify_model_name
    parent = get_results_dir() / "thinking_sampling" / slugify_model_name(model_name)
    if not parent.exists():
        return None
    matches = [
        p / "results.json"
        for p in parent.iterdir()
        if p.is_dir() and (p / "results.json").exists()
    ]
    if not matches:
        return None
    return max(matches, key=lambda p: p.parent.stat().st_mtime)


def _run_fixed_layer_cross_instruction_subprocess(item: WorkItem) -> None:
    if item.child_args is None:
        raise ValueError("Fixed-layer cross-instruction item is missing child process arguments.")
    command = [
        sys.executable,
        str(ROOT / "run_paper.py"),
        "run",
        "--execution-mode",
        "child-fixed-layer-cross-instruction",
        "--child-payload-json",
        json.dumps(item.child_args),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def _run_steering_extract_subprocess(item: WorkItem) -> None:
    if item.child_args is None:
        raise ValueError("Steering-extract item is missing child process arguments.")
    command = [
        sys.executable,
        str(ROOT / "run_paper.py"),
        "run",
        "--execution-mode",
        "child-steering-extract",
        "--child-payload-json",
        json.dumps(item.child_args),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def _run_child_payload_subprocess(execution_mode: str, item: WorkItem) -> None:
    if item.child_args is None:
        raise ValueError(f"{item.kind} item is missing child process arguments.")
    command = [
        sys.executable,
        str(ROOT / "run_paper.py"),
        "run",
        "--execution-mode",
        execution_mode,
        "--child-payload-json",
        json.dumps(item.child_args),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


_CHILD_PAYLOAD_KINDS = {
    "steering_all_layers": "child-steering-all-layers",
    "no_instruction": "child-no-instruction",
    "mmlu_entropy_steering": "child-mmlu-entropy-steering",
    "mmlu_random_vector_control": "child-mmlu-random-vector-control",
    "mmlu_prompt_conditions": "child-mmlu-prompt-conditions",
    "mmlu_steering_vector_probe": "child-mmlu-steering-vector-probe",
}


def run_items(items: List[WorkItem], dry_run: bool, force: bool) -> None:
    print(f"Planned {len(items)} work items.\n")
    local_model_batches = _group_local_model_batches(items)
    batched_item_ids = {id(item) for batch in local_model_batches for item in batch.items}
    planned_units = list(local_model_batches) + [item for item in items if id(item) not in batched_item_ids]

    for index, unit in enumerate(planned_units, start=1):
        if isinstance(unit, LocalModelBatch):
            template_tag = f" | {unit.template_name}" if unit.template_name else ""
            print(f"[{index}/{len(planned_units)}] {unit.figure} / {unit.group_id} / {unit.stage}")
            print(f"  {unit.model_name}{template_tag} | {len(unit.items)} instruction pairs (one load)")
            completed_items = [item for item in unit.items if item.status_path()]
            if len(completed_items) == len(unit.items) and not force:
                print("  -> skipping existing results for all instruction pairs")
                continue
            if dry_run:
                continue
            if unit.kind == "prompt_variants":
                _run_prompt_variant_batch_subprocess(unit, force)
            else:
                _run_local_zero_shot_batch_subprocess(unit, force)
            print("  -> completed in isolated child process")
            continue

        item = unit
        print(f"[{index}/{len(planned_units)}] {item.figure} / {item.group_id} / {item.stage}")
        print(f"  {item.label}")
        status = item.status_path()
        if status and not force:
            rendered = str(status) if isinstance(status, Path) else "ready"
            print(f"  -> skipping existing result: {rendered}")
            continue
        if dry_run:
            continue
        if item.kind == "steering_fixed_layer_cross_instruction":
            _run_fixed_layer_cross_instruction_subprocess(item)
            print("  -> completed in isolated child process")
            continue
        if item.kind == "steering_extract":
            _run_steering_extract_subprocess(item)
            print("  -> completed in isolated child process")
            continue
        if item.kind in _CHILD_PAYLOAD_KINDS:
            _run_child_payload_subprocess(_CHILD_PAYLOAD_KINDS[item.kind], item)
            print("  -> completed in isolated child process")
            continue
        output = item.runner()
        if output is not None:
            print(f"  -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Run or inspect the paper experiment matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ["status", "run"]:
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument(
            "--execution-mode",
            choices=[
                "parent",
                "child-local-zero-shot-batch",
                "child-prompt-variant-batch",
                "child-fixed-layer-cross-instruction",
                "child-steering-extract",
                "child-steering-all-layers",
                "child-no-instruction",
                "child-mmlu-entropy-steering",
                "child-mmlu-random-vector-control",
                "child-mmlu-prompt-conditions",
                "child-mmlu-steering-vector-probe",
            ],
            default="parent",
            help=argparse.SUPPRESS,
        )
        subparser.add_argument("--figure", type=str, default=None, help="Filter to a figure number like 2, 3, 4, or 5.")
        subparser.add_argument("--stage", choices=["prepare", "run"], default=None)
        subparser.add_argument("--group", action="append", default=None, help="Filter to a manifest group id. Repeatable.")
        subparser.add_argument("--include-model", action="append", default=None, help="Only include this exact model name. Repeatable.")
        subparser.add_argument("--exclude-model", action="append", default=None, help="Exclude this exact model name. Repeatable.")
        subparser.add_argument(
            "--ab-pairs",
            choices=["original", "neutral"],
            default=None,
            help=(
                "Which AB pairs file to use. 'original' = data/ab_pairs.json (default), "
                "'neutral' = data/ab_pairs_neutral.json. Results are stored under "
                "results/neutral/ when 'neutral' is selected."
            ),
        )
        subparser.add_argument(
            "--api-backend",
            choices=["fireworks", "openrouter"],
            default=None,
            help=(
                "Backend for zero_shot_api / prompt_variants_api runs. "
                "Default is fireworks (or API_BACKEND env / per-group api_backend). "
                "OpenRouter pins a logprobs-capable provider per model."
            ),
        )
        subparser.add_argument(
            "--api-provider",
            type=str,
            default=None,
            help=(
                "Optional OpenRouter endpoint override (sets OPENROUTER_PROVIDER). "
                "Prefer provider/quant tags like 'parasail/fp8'. "
                "When unset, each model uses its pinned logprobs endpoint."
            ),
        )
        subparser.add_argument(
            "--api-concurrency",
            type=int,
            default=None,
            help=(
                "Max parallel API requests for zero_shot_api / prompt_variants_api "
                "(sets API_CONCURRENCY; default 32). Use 1 for sequential."
            ),
        )
        subparser.add_argument("--child-model", type=str, default=None, help=argparse.SUPPRESS)
        subparser.add_argument("--child-instruction-pairs-json", type=str, default=None, help=argparse.SUPPRESS)
        subparser.add_argument("--child-payload-json", type=str, default=None, help=argparse.SUPPRESS)
        if command_name == "run":
            subparser.add_argument("--dry-run", action="store_true")
            subparser.add_argument(
                "--force",
                action="store_true",
                help="Rerun work items even when matching output already exists.",
            )

    args = parser.parse_args()

    # Propagate AB_PAIRS_VARIANT via the environment so that child subprocesses
    # (which re-invoke run_paper.py) inherit the setting automatically.
    if getattr(args, "ab_pairs", None):
        os.environ["AB_PAIRS_VARIANT"] = args.ab_pairs
    elif "AB_PAIRS_VARIANT" not in os.environ:
        os.environ["AB_PAIRS_VARIANT"] = "original"

    if getattr(args, "api_backend", None):
        os.environ["API_BACKEND"] = args.api_backend
    if getattr(args, "api_provider", None):
        os.environ["OPENROUTER_PROVIDER"] = args.api_provider
    if getattr(args, "api_concurrency", None) is not None:
        os.environ["API_CONCURRENCY"] = str(args.api_concurrency)

    if args.execution_mode == "child-local-zero-shot-batch":
        if not args.child_model or not args.child_instruction_pairs_json:
            raise ValueError("Child local zero-shot batch execution requires child model and instruction pairs.")
        instruction_pairs = [tuple(pair) for pair in json.loads(args.child_instruction_pairs_json)]
        outputs = run_zero_shot_local_batch(args.child_model, instruction_pairs, force=args.force)
        for output in outputs:
            print(f"  -> {output}")
    elif args.execution_mode == "child-prompt-variant-batch":
        if not args.child_model or not args.child_instruction_pairs_json or not args.child_payload_json:
            raise ValueError(
                "Child prompt-variant batch execution requires child model, instruction pairs, and payload."
            )
        payload = json.loads(args.child_payload_json)
        instruction_pairs = [tuple(pair) for pair in json.loads(args.child_instruction_pairs_json)]
        outputs = run_prompt_variant_batch(
            args.child_model,
            payload["template_name"],
            instruction_pairs,
            force=args.force,
        )
        for output in outputs:
            print(f"  -> {output}")
    elif args.execution_mode == "child-fixed-layer-cross-instruction":
        if not args.child_payload_json:
            raise ValueError("Child fixed-layer cross-instruction execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        output = run_fixed_layer_cross_instruction_steering(**payload)
        print(f"  -> {output}")
    elif args.execution_mode == "child-steering-extract":
        if not args.child_payload_json:
            raise ValueError("Child steering-extract execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        ab_variant = payload.pop("ab_pairs_variant", os.environ.get("AB_PAIRS_VARIANT", "original"))
        os.environ["AB_PAIRS_VARIANT"] = ab_variant
        output = extract_multi_layer_activations(**payload)
        print(f"  -> {output}")
    elif args.execution_mode == "child-steering-all-layers":
        if not args.child_payload_json:
            raise ValueError("Child steering-all-layers execution requires a payload.")
        output = run_all_layers_steering(**json.loads(args.child_payload_json))
        print(f"  -> {output}")
    elif args.execution_mode == "child-no-instruction":
        if not args.child_payload_json:
            raise ValueError("Child no-instruction execution requires a payload.")
        output = run_no_instruction_control(**json.loads(args.child_payload_json))
        print(f"  -> {output}")
    elif args.execution_mode == "child-mmlu-entropy-steering":
        if not args.child_payload_json:
            raise ValueError("Child mmlu-entropy-steering execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        ab_variant = payload.pop("ab_pairs_variant", os.environ.get("AB_PAIRS_VARIANT", "original"))
        os.environ["AB_PAIRS_VARIANT"] = ab_variant
        output = run_mmlu_entropy_steering(**payload)
        print(f"  -> {output}")
    elif args.execution_mode == "child-mmlu-random-vector-control":
        if not args.child_payload_json:
            raise ValueError("Child mmlu-random-vector-control execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        ab_variant = payload.pop("ab_pairs_variant", os.environ.get("AB_PAIRS_VARIANT", "original"))
        os.environ["AB_PAIRS_VARIANT"] = ab_variant
        output = run_mmlu_random_vector_control(**payload)
        print(f"  -> {output}")
    elif args.execution_mode == "child-mmlu-prompt-conditions":
        if not args.child_payload_json:
            raise ValueError("Child mmlu-prompt-conditions execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        ab_variant = payload.pop("ab_pairs_variant", os.environ.get("AB_PAIRS_VARIANT", "original"))
        os.environ["AB_PAIRS_VARIANT"] = ab_variant
        output = run_mmlu_prompt_conditions(**payload)
        print(f"  -> {output}")
    elif args.execution_mode == "child-mmlu-steering-vector-probe":
        if not args.child_payload_json:
            raise ValueError("Child mmlu-steering-vector-probe execution requires a payload.")
        payload = json.loads(args.child_payload_json)
        ab_variant = payload.pop("ab_pairs_variant", os.environ.get("AB_PAIRS_VARIANT", "original"))
        os.environ["AB_PAIRS_VARIANT"] = ab_variant
        output = run_mmlu_steering_vector_probe(**payload)
        print(f"  -> {output}")
    else:
        manifest = load_manifest()
        items = build_items(
            manifest,
            args.figure,
            args.stage,
            args.group,
            args.include_model,
            args.exclude_model,
        )
        if args.command == "status":
            print_status(items)
            return
        run_items(items, args.dry_run, args.force)


if __name__ == "__main__":
    main()
