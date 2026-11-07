from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common import GENERATED_DIR, apply_paper_style, print_outputs
from be_sure_or_random.experiments import (
    latest_mmlu_entropy_steering_result,
    latest_mmlu_random_vector_control_result,
)
from be_sure_or_random.utils import load_json, slugify_model_name


LABELS = ("A", "B", "C", "D")
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MODELS = (
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3-32B",
)
MMLU_LOOKUP = dict(
    dataset_name="cais/mmlu",
    dataset_config="all",
    split="test",
    max_examples=None,
    fixed_layer=56,
    steering_strength=2.0,
    vector_certainty_instruction="Be sure!",
    vector_uncertainty_instruction="Be random!",
    prompt_variant="format_instructed",
)
CONDITION_LABELS = {
    "none": "No steering",
    "increase_entropy": "Entropy increase",
    "decrease_entropy": "Entropy decrease",
}
STEERING_LABELS = {
    "increase_entropy": "Positive steering",
    "decrease_entropy": "Negative steering",
}
CONDITION_COLORS = {
    "none": "#64748B",
    "increase_entropy": "#DC2626",
    "decrease_entropy": "#2563EB",
}
RANDOM_COLOR = "#64748B"
RANDOM_CALIBRATION_COLOR = "#D1D5DB"


@dataclass(frozen=True)
class CalibrationCurve:
    mean_confidence: list[float]
    empirical_accuracy: list[float]
    counts: list[int]
    ece: float


def _prob(record: dict, label: str) -> float:
    return float(record[f"normalized_prob_{label}"])


def _top_label(record: dict) -> str:
    return max(LABELS, key=lambda label: _prob(record, label))


def _by_example(records: list[dict]) -> dict[int, dict]:
    return {int(record["example_idx"]): record for record in records}


def _delta_entropy_by_baseline_correctness(
    baseline_records: list[dict],
    condition_records: list[dict],
) -> dict[str, np.ndarray]:
    baseline_by_idx = _by_example(baseline_records)
    condition_by_idx = _by_example(condition_records)
    deltas = {"baseline_top_correct": [], "baseline_top_wrong": []}
    for idx in sorted(set(baseline_by_idx) & set(condition_by_idx)):
        baseline = baseline_by_idx[idx]
        baseline_top_label = _top_label(baseline)
        key = (
            "baseline_top_correct"
            if baseline_top_label == baseline["answer_label"]
            else "baseline_top_wrong"
        )
        deltas[key].append(
            float(condition_by_idx[idx]["label_entropy"])
            - float(baseline["label_entropy"])
        )
    return {key: np.asarray(values, dtype=float) for key, values in deltas.items()}


def _paired_delta_entropy(
    baseline_records: list[dict],
    condition_records: list[dict],
) -> np.ndarray:
    baseline_by_idx = _by_example(baseline_records)
    condition_by_idx = _by_example(condition_records)
    common = sorted(set(baseline_by_idx) & set(condition_by_idx))
    return np.asarray(
        [
            float(condition_by_idx[idx]["label_entropy"])
            - float(baseline_by_idx[idx]["label_entropy"])
            for idx in common
        ],
        dtype=float,
    )


def _top_label_points(records: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    confidences = []
    outcomes = []
    for record in records:
        top_label = _top_label(record)
        confidences.append(_prob(record, top_label))
        outcomes.append(float(top_label == record["answer_label"]))
    return np.asarray(confidences, dtype=float), np.asarray(outcomes, dtype=float)


def _adaptive_calibration_curve(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationCurve:
    order = np.argsort(probabilities)
    probability_chunks = np.array_split(probabilities[order], min(n_bins, len(probabilities)))
    outcome_chunks = np.array_split(outcomes[order], min(n_bins, len(outcomes)))
    mean_confidence = []
    empirical_accuracy = []
    counts = []
    ece = 0.0
    total = float(len(probabilities))
    for prob_chunk, outcome_chunk in zip(probability_chunks, outcome_chunks):
        conf = float(np.mean(prob_chunk))
        acc = float(np.mean(outcome_chunk))
        count = int(len(prob_chunk))
        mean_confidence.append(conf)
        empirical_accuracy.append(acc)
        counts.append(count)
        ece += (count / total) * abs(acc - conf)
    return CalibrationCurve(mean_confidence, empirical_accuracy, counts, float(ece))


def _calibration_metrics(records: list[dict], n_bins: int = 15) -> dict:
    confidences, outcomes = _top_label_points(records)
    curve = _adaptive_calibration_curve(confidences, outcomes, n_bins=n_bins)
    return {
        "n": len(records),
        "accuracy": float(np.mean(outcomes)),
        "mean_confidence": float(np.mean(confidences)),
        "ece": curve.ece,
        "curve": {
            "mean_confidence": curve.mean_confidence,
            "empirical_accuracy": curve.empirical_accuracy,
            "counts": curve.counts,
        },
    }


def _summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "fraction_positive": float(np.mean(values > 0)),
        "fraction_negative": float(np.mean(values < 0)),
    }


def _save_figure(fig: plt.Figure, stem: str, *, model_name: str | None = None) -> list[Path]:
    output_dir = GENERATED_DIR / "figure7_analysis"
    if model_name is not None:
        output_dir = output_dir / slugify_model_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.svg",
    ]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    fig.savefig(paths[2])
    return paths


def _plot_delta_entropy(ax: plt.Axes, delta_groups: dict[str, dict[str, np.ndarray]]) -> None:
    entries = [
        ("increase_entropy", "baseline_top_correct", -0.18),
        ("decrease_entropy", "baseline_top_correct", 0.18),
        ("increase_entropy", "baseline_top_wrong", 1.05),
        ("decrease_entropy", "baseline_top_wrong", 1.41),
    ]
    positions = [entry[2] for entry in entries]
    data = [delta_groups[condition][correctness] for condition, correctness, _ in entries]
    colors = [CONDITION_COLORS[condition] for condition, _, _ in entries]
    parts = ax.violinplot(data, positions=positions, widths=0.28, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)
    for position, values, color in zip(positions, data, colors):
        q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        ax.vlines(position, q10, q90, color=color, linewidth=1.8)
        ax.scatter([position], [q50], color=color, s=26, zorder=4)
        ax.scatter([position], [float(np.mean(values))], color="white", edgecolor=color, s=30, zorder=5)
    ax.axhline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_title("Change in answer entropy")
    ax.set_ylabel("Entropy difference wrt no steering")
    ax.set_xticks([0.0, 1.23])
    ax.set_xticklabels(["Top answer\ncorrect", "Top answer\nwrong"])
    ax.set_xlim(-0.5, 1.75)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=CONDITION_COLORS["increase_entropy"], marker="o", linestyle="", label=STEERING_LABELS["increase_entropy"]),
            plt.Line2D([0], [0], color=CONDITION_COLORS["decrease_entropy"], marker="o", linestyle="", label=STEERING_LABELS["decrease_entropy"]),
            plt.Line2D([0], [0], color="white", marker="o", markeredgecolor="#111827", linestyle="", label="Mean"),
        ],
        frameon=False,
        loc="upper left",
    )


def _plot_delta_entropy_histogram_panel(
    ax: plt.Axes,
    delta_groups: dict[str, dict[str, np.ndarray]],
    correctness: str,
    title: str,
    bins: np.ndarray,
    marker: str = "median",
) -> None:
    for condition in ("increase_entropy", "decrease_entropy"):
        values = delta_groups[condition][correctness]
        color = CONDITION_COLORS[condition]
        summary_value = (
            float(np.mean(values))
            if marker == "mean"
            else float(np.median(values))
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.16,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            label=STEERING_LABELS[condition],
        )
        ax.axvline(
            summary_value,
            color=color,
            linestyle="--",
            linewidth=2.2,
            label=f"{STEERING_LABELS[condition]} {marker}",
        )
    ax.axvline(0.0, color="#9CA3AF", linewidth=1.2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Entropy change after steering")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")


def _plot_delta_entropy_histograms(
    axes: np.ndarray,
    delta_groups: dict[str, dict[str, np.ndarray]],
    marker: str = "median",
) -> None:
    all_values = np.concatenate(
        [
            delta_groups[condition][correctness]
            for condition in ("increase_entropy", "decrease_entropy")
            for correctness in ("baseline_top_correct", "baseline_top_wrong")
        ]
    )
    lo, hi = np.quantile(all_values, [0.005, 0.995])
    bins = np.linspace(float(lo), float(hi), 44)
    _plot_delta_entropy_histogram_panel(
        axes[0],
        delta_groups,
        "baseline_top_correct",
        "Top answer correct",
        bins,
        marker,
    )
    _plot_delta_entropy_histogram_panel(
        axes[1],
        delta_groups,
        "baseline_top_wrong",
        "Top answer wrong",
        bins,
        marker,
    )


def _plot_calibration(
    ax: plt.Axes,
    calibration: dict[str, dict],
    random_calibration: dict[str, dict] | None = None,
) -> None:
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.2, label="Perfect calibration")
    if random_calibration:
        for idx, condition in enumerate(sorted(random_calibration)):
            curve = random_calibration[condition]["curve"]
            ax.plot(
                curve["mean_confidence"],
                curve["empirical_accuracy"],
                color=RANDOM_CALIBRATION_COLOR,
                alpha=0.75,
                linewidth=1.2,
                label="Random vectors" if idx == 0 else None,
            )
    for condition in ("none", "increase_entropy", "decrease_entropy"):
        metrics = calibration[condition]
        curve = metrics["curve"]
        label = CONDITION_LABELS[condition]
        if condition in STEERING_LABELS:
            label = STEERING_LABELS[condition]
        ax.plot(
            curve["mean_confidence"],
            curve["empirical_accuracy"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=CONDITION_COLORS[condition],
            label=f"{label} (ECE={metrics['ece']:.3f})",
        )
    ax.set_title("Top-answer confidence calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = [
        "Perfect calibration",
        next(label for label in labels if label.startswith("No steering")),
        next(label for label in labels if label.startswith("Positive steering")),
        next(label for label in labels if label.startswith("Negative steering")),
    ]
    if random_calibration:
        ordered_labels.append("Random vectors")
    ordered_handles = [handles[labels.index(label)] for label in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, frameon=False, loc="upper left")


def _random_delta_entropies(payload: dict) -> dict[str, np.ndarray]:
    baseline = payload["results"]["none"]
    return {
        condition: _paired_delta_entropy(baseline, records)
        for condition, records in payload["results"].items()
        if condition.startswith("random_")
    }


def _pool_correctness_groups(
    delta_groups: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        condition: np.concatenate(
            [
                delta_groups[condition]["baseline_top_correct"],
                delta_groups[condition]["baseline_top_wrong"],
            ]
        )
        for condition in ("increase_entropy", "decrease_entropy")
    }


def _plot_random_mean_delta_panel(
    ax: plt.Axes,
    random_deltas: dict[str, np.ndarray],
    entropy_deltas: dict[str, np.ndarray],
    statistic: str = "mean",
) -> None:
    reducer = np.median if statistic == "median" else np.mean
    label_statistic = "Median" if statistic == "median" else "Mean"
    rng = np.random.default_rng(0)
    random_values = np.asarray(
        [reducer(values) for _, values in sorted(random_deltas.items())],
        dtype=float,
    )
    random_x = 0.48
    jitter = random_x + rng.uniform(-0.04, 0.04, size=random_values.size)
    ax.scatter(
        jitter,
        random_values,
        color=RANDOM_COLOR,
        s=40,
        alpha=0.85,
        label="Random vectors",
        zorder=3,
    )
    ax.errorbar(
        [random_x],
        [float(np.mean(random_values))],
        yerr=[float(np.std(random_values, ddof=1))],
        fmt="o",
        color="#111827",
        markersize=4,
        capsize=3,
        label=f"Random vectors {statistic} +/- SD",
        zorder=4,
    )
    for x, condition in [(0.0, "increase_entropy"), (0.2, "decrease_entropy")]:
        ax.scatter(
            [x],
            [float(reducer(entropy_deltas[condition]))],
            color=CONDITION_COLORS[condition],
            marker="o",
            s=58,
            label=STEERING_LABELS[condition],
            zorder=5,
        )
    ax.axhline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_title("Control: comparison with random vectors")
    ax.set_ylabel(f"{label_statistic} entropy change")
    ax.set_xticks([0.0, 0.2, random_x])
    ax.set_xticklabels(["Positive\nsteering", "Negative\nsteering", "Random"])
    ax.set_xlim(-0.16, 0.64)
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = [
        "Positive steering",
        "Negative steering",
        "Random vectors",
        f"Random vectors {statistic} +/- SD",
    ]
    ordered_handles = [handles[labels.index(label)] for label in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, frameon=False, loc="best")


def build_figure(
    result_path: Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[plt.Figure, dict]:
    if result_path is None:
        result_path = latest_mmlu_entropy_steering_result(
            model_name=model_name,
            **MMLU_LOOKUP,
        )
    if result_path is None:
        raise FileNotFoundError(f"No full MMLU entropy-steering result found for {model_name}.")

    payload = load_json(result_path)
    records_by_condition = payload["results"]
    baseline = records_by_condition["none"]
    delta_groups = {
        condition: _delta_entropy_by_baseline_correctness(baseline, records_by_condition[condition])
        for condition in ("increase_entropy", "decrease_entropy")
    }
    calibration = {
        condition: _calibration_metrics(records_by_condition[condition])
        for condition in ("none", "increase_entropy", "decrease_entropy")
    }

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.2),
        constrained_layout=True,
    )
    _plot_delta_entropy(axes[0], delta_groups)
    _plot_calibration(axes[1], calibration)

    root = Path(__file__).resolve().parents[2]
    summary = {
        "result_path": str(result_path.relative_to(root)),
        "delta_entropy_definition": "label_entropy(condition) - label_entropy(no steering), paired by example_idx",
        "correct_wrong_split": "Whether the no-steering top label equals the answer label",
        "delta_entropy": {
            condition: {key: _summarize(values) for key, values in groups.items()}
            for condition, groups in delta_groups.items()
        },
        "calibration": calibration,
    }
    return fig, summary


def build_figure_with_random_mean(
    result_path: Path | None = None,
    random_result_path: Path | None = None,
    statistic: str = "mean",
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[plt.Figure, dict]:
    if result_path is None:
        result_path = latest_mmlu_entropy_steering_result(
            model_name=model_name,
            **MMLU_LOOKUP,
        )
    if random_result_path is None:
        random_result_path = latest_mmlu_random_vector_control_result(
            model_name=model_name,
            random_seed=0,
            n_random_vectors=8,
            **MMLU_LOOKUP,
        )
    if result_path is None:
        raise FileNotFoundError(f"No full MMLU entropy-steering result found for {model_name}.")
    if random_result_path is None:
        raise FileNotFoundError(f"No full MMLU random-vector control result found for {model_name}.")

    payload = load_json(result_path)
    random_payload = load_json(random_result_path)
    records_by_condition = payload["results"]
    baseline = records_by_condition["none"]
    delta_groups = {
        condition: _delta_entropy_by_baseline_correctness(baseline, records_by_condition[condition])
        for condition in ("increase_entropy", "decrease_entropy")
    }
    entropy_deltas = _pool_correctness_groups(delta_groups)
    calibration = {
        condition: _calibration_metrics(records_by_condition[condition])
        for condition in ("none", "increase_entropy", "decrease_entropy")
    }
    random_deltas = _random_delta_entropies(random_payload)
    random_calibration = {
        condition: _calibration_metrics(records)
        for condition, records in sorted(random_payload["results"].items())
        if condition.startswith("random_")
    }

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 4.2),
        constrained_layout=True,
    )
    _plot_delta_entropy(axes[0], delta_groups)
    _plot_random_mean_delta_panel(axes[1], random_deltas, entropy_deltas, statistic)
    _plot_calibration(axes[2], calibration, random_calibration)

    root = Path(__file__).resolve().parents[2]
    reducer = np.median if statistic == "median" else np.mean
    random_statistics = [float(reducer(values)) for _, values in sorted(random_deltas.items())]
    summary = {
        "result_path": str(result_path.relative_to(root)),
        "random_result_path": str(random_result_path.relative_to(root)),
        "plot_variant": f"violins_with_random_{statistic}_delta_panel",
        "delta_entropy_definition": "label_entropy(condition) - label_entropy(no steering), paired by example_idx",
        "correct_wrong_split": "Whether the no-steering top label equals the answer label",
        "delta_entropy": {
            condition: {key: _summarize(values) for key, values in groups.items()}
            for condition, groups in delta_groups.items()
        },
        "calibration": calibration,
        "random_calibration": random_calibration,
        f"random_{statistic}_delta_entropy": {
            "per_vector": random_statistics,
            "mean_across_vectors": float(np.mean(random_statistics)),
            "sd_across_vectors": float(np.std(random_statistics, ddof=1)),
        },
    }
    return fig, summary


def build_histogram_figure(
    result_path: Path | None = None,
    marker: str = "median",
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[plt.Figure, dict]:
    if result_path is None:
        result_path = latest_mmlu_entropy_steering_result(
            model_name=model_name,
            **MMLU_LOOKUP,
        )
    if result_path is None:
        raise FileNotFoundError(f"No full MMLU entropy-steering result found for {model_name}.")

    payload = load_json(result_path)
    records_by_condition = payload["results"]
    baseline = records_by_condition["none"]
    delta_groups = {
        condition: _delta_entropy_by_baseline_correctness(baseline, records_by_condition[condition])
        for condition in ("increase_entropy", "decrease_entropy")
    }
    calibration = {
        condition: _calibration_metrics(records_by_condition[condition])
        for condition in ("none", "increase_entropy", "decrease_entropy")
    }

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.2),
        constrained_layout=True,
    )
    _plot_delta_entropy_histograms(axes[:2], delta_groups, marker)
    _plot_calibration(axes[2], calibration)

    root = Path(__file__).resolve().parents[2]
    summary = {
        "result_path": str(result_path.relative_to(root)),
        "plot_variant": f"histograms_with_{marker}_lines",
        "delta_entropy_definition": "label_entropy(condition) - label_entropy(no steering), paired by example_idx",
        "correct_wrong_split": "Whether the no-steering top label equals the answer label",
        "delta_entropy": {
            condition: {key: _summarize(values) for key, values in groups.items()}
            for condition, groups in delta_groups.items()
        },
        "calibration": calibration,
    }
    return fig, summary


def _write_summary(summary: dict, stem: str, output_paths: list[Path], *, model_name: str) -> Path:
    summary_dir = GENERATED_DIR / "figure7_analysis" / slugify_model_name(model_name)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{stem}_summary.json"
    payload = {
        **summary,
        "model_name": model_name,
        "output_paths": [str(path) for path in output_paths],
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return summary_path


def generate_for_model(model_name: str) -> None:
    print(f"Generating figure7 MMLU panels for {model_name}", flush=True)

    fig, summary = build_figure(model_name=model_name)
    try:
        output_paths = _save_figure(fig, "mmlu_entropy_correctness_calibration", model_name=model_name)
        summary_path = _write_summary(summary, "mmlu_entropy_correctness_calibration", output_paths, model_name=model_name)
        print_outputs([*output_paths, summary_path])
    finally:
        plt.close(fig)

    fig, summary = build_figure_with_random_mean(model_name=model_name)
    try:
        output_paths = _save_figure(
            fig, "mmlu_entropy_correctness_calibration_with_random_mean", model_name=model_name
        )
        summary_path = _write_summary(
            summary, "mmlu_entropy_correctness_calibration_with_random_mean", output_paths, model_name=model_name
        )
        print_outputs([*output_paths, summary_path])
    finally:
        plt.close(fig)

    fig, summary = build_figure_with_random_mean(statistic="median", model_name=model_name)
    try:
        output_paths = _save_figure(
            fig, "mmlu_entropy_correctness_calibration_with_random_median", model_name=model_name
        )
        summary_path = _write_summary(
            summary, "mmlu_entropy_correctness_calibration_with_random_median", output_paths, model_name=model_name
        )
        print_outputs([*output_paths, summary_path])
    finally:
        plt.close(fig)

    fig, summary = build_histogram_figure(marker="mean", model_name=model_name)
    try:
        output_paths = _save_figure(
            fig, "mmlu_entropy_correctness_calibration_histograms_mean", model_name=model_name
        )
        summary_path = _write_summary(
            summary, "mmlu_entropy_correctness_calibration_histograms_mean", output_paths, model_name=model_name
        )
        print_outputs([*output_paths, summary_path])
    finally:
        plt.close(fig)

    fig, summary = build_histogram_figure(model_name=model_name)
    try:
        output_paths = _save_figure(
            fig, "mmlu_entropy_correctness_calibration_histograms", model_name=model_name
        )
        summary_path = _write_summary(
            summary, "mmlu_entropy_correctness_calibration_histograms", output_paths, model_name=model_name
        )
        print_outputs([*output_paths, summary_path])
    finally:
        plt.close(fig)


def main() -> None:
    models = sys.argv[1:] if len(sys.argv) > 1 else list(DEFAULT_MODELS)
    for model_name in models:
        generate_for_model(model_name)


if __name__ == "__main__":
    main()
