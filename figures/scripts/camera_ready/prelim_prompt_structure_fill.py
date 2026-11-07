"""Preliminary appendix figure (rebuttal R5): prompt structure, padding and user rules.

Per model, median DeltaH for each of the 7 meaningful instruction pairs (small dots)
and the median over pairs (bar) under eight prompt arms:

  Standard            results/neutral/zero_shot_local/<model>/          (paper "standard")
  Standard + 64/256   results/neutral/prompt_variants/standard_pad{64,256}/<model>/
  Minimal             results/neutral/prompt_variants/barebone/<model>/  (paper "minimal")
  Minimal + 64/256    results/neutral/prompt_variants/barebone_pad{64,256}/<model>/
  Format only         results/neutral/prompt_variants/format_only/<model>/
  User rules          results/neutral/prompt_variants/user_rules/<model>/

Padding = N ASCII spaces appended after the instruction in the user turn. Only runs
dated 20260720 or later are used. DeltaH = H(uncertainty) - H(certainty), binary
entropy in bits over the two option tokens, paired per (option pair, ordering) item,
n = 200 items per instruction pair; records with < 0.9 mass on the two options are
dropped (common.paired_entropy_differences).

Padding validity check: several chat templates trim the user content (Llama-3:
``message['content'] | trim``; Qwen3.5: ``render_content(...)|trim``), which deletes
the trailing spaces. If the +64 and +256 runs of a model are bitwise identical for
every record, the padding never reached the model and those cells are not plotted.

  cd ~/Luca/be-sure-or-random-cr
  AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
      figures/scripts/camera_ready/prelim_prompt_structure.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (  # noqa: E402
    REPO_ROOT,
    clean_probability_records,
    display_model_name,
    load_zero_shot_result,
    paired_entropy_differences,
)
# Shared font scale / output helpers for all camera-ready preliminary figures.
from _cr_prelim_style import FULL_WIDTH_IN, OUT_DIR, apply_prelim_style, save  # noqa: E402

STEM = "prelim_prompt_structure_fill"
RESULTS = REPO_ROOT / "results" / "neutral"
MIN_RUN_DATE = "20260720"

MODELS = [
    ("qwen/Qwen3.5-27B", "qwen_Qwen3.5-27B"),
    ("qwen/Qwen3-32B", "qwen_Qwen3-32B"),
    ("qwen/Qwen3.5-35B-A3B", "qwen_Qwen3.5-35B-A3B"),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama_Meta-Llama-3-70B-Instruct"),
]

PAIRS = [
    ("Be sure!", "Be random!"),
    ("Maximize certainty", "Minimize certainty"),
    ("Maximize certainty", "Maximize uncertainty"),
    ("Commit to one option", "Keep both options equally likely"),
    ("Reply with confidence", "Reply without confidence"),
    ("Aim for low entropy", "Aim for high entropy"),
    ("Be deterministic", "Assign equal probability to both options"),
]

# Paper colours: standard = purple, minimal = amber (prompt_ablation.pdf). The two new
# arms reuse darker / lighter tints of the family they derive from.
STANDARD_COLOR = "#9333EA"
MINIMAL_COLOR = "#F59E0B"
FORMAT_ONLY_COLOR = "#B45309"
USER_RULES_COLOR = "#C084FC"
ZERO_GRAY = "#9CA3AF"
TEXT_GRAY = "#374151"
NA_FILL = "#F3F4F6"

# (key, results dir template, tick label, colour, x position, padding base or None)
ARMS = [
    ("standard", "zero_shot_local", "Standard", STANDARD_COLOR, 0.0, None),
    ("standard_fill64", "prompt_variants/standard_fill64", "+ 64 filler tokens", STANDARD_COLOR, 1.0, "standard"),
    ("standard_fill256", "prompt_variants/standard_fill256", "+ 256 filler tokens", STANDARD_COLOR, 2.0, "standard"),
    ("barebone", "prompt_variants/barebone", "Minimal", MINIMAL_COLOR, 3.5, None),
    ("barebone_fill64", "prompt_variants/barebone_fill64", "+ 64 filler tokens", MINIMAL_COLOR, 4.5, "barebone"),
    ("barebone_fill256", "prompt_variants/barebone_fill256", "+ 256 filler tokens", MINIMAL_COLOR, 5.5, "barebone"),
    ("format_only", "prompt_variants/format_only", "Format only", FORMAT_ONLY_COLOR, 7.0, None),
    ("user_rules", "prompt_variants/user_rules", "User rules", USER_RULES_COLOR, 8.2, None),
]

FIG_WIDTH = FULL_WIDTH_IN


def slug(text: str) -> str:
    return text.replace(" ", "_").replace("!", "").replace("?", "").replace(",", "").replace(".", "")


def latest_run(arm_dir: str, model_slug: str, cert: str, unc: str) -> Path | None:
    parent = RESULTS / arm_dir / model_slug
    if not parent.is_dir():
        return None
    prefix = f"{slug(cert)}_{slug(unc)}_"
    runs = []
    for d in parent.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        stamp = d.name[len(prefix):]
        # Guard against prefix collisions (e.g. "Be_sure_Be_random_" vs longer slugs).
        if len(stamp) != 15 or not stamp[:8].isdigit() or stamp[:8] < MIN_RUN_DATE:
            continue
        if (d / "results.json").exists():
            runs.append((stamp, d / "results.json"))
    return max(runs)[1] if runs else None


def probs_signature(path: Path) -> dict:
    return {
        (r["option_a"], r["option_b"], bool(r.get("flip_ab")), r["instruction"]): (r["option_a_prob"], r["option_b_prob"])
        for r in load_zero_shot_result(path)
    }


def padding_reached_model(model_slug: str, base: str) -> bool:
    """False if the +64 and +256 runs are bitwise identical for every record of every pair."""
    for cert, unc in PAIRS:
        p64 = latest_run(f"prompt_variants/{base}_fill64", model_slug, cert, unc)
        p256 = latest_run(f"prompt_variants/{base}_fill256", model_slug, cert, unc)
        if p64 is None or p256 is None:
            continue
        if probs_signature(p64) != probs_signature(p256):
            return True
    return False


def main() -> None:
    apply_prelim_style()

    rows = []
    data: dict[tuple[str, str], list[float]] = {}
    validity: dict[tuple[str, str], bool] = {}
    for model_name, model_slug in MODELS:
        for base in ("standard", "barebone"):
            validity[(model_slug, base)] = padding_reached_model(model_slug, base)
        for key, arm_dir, *_ in ARMS:
            meds = []
            for cert, unc in PAIRS:
                path = latest_run(arm_dir, model_slug, cert, unc)
                if path is None:
                    rows.append({"model": model_name, "arm": key, "certainty": cert, "uncertainty": unc, "run": "", "n_records": 0, "n_clean": 0, "n_paired": 0, "median_dH": float("nan"), "mean_dH": float("nan"), "padding_reached_model": ""})
                    meds.append(float("nan"))
                    continue
                records = load_zero_shot_result(path)
                diffs = paired_entropy_differences(records, cert, unc)
                med = float(np.median(diffs)) if diffs else float("nan")
                meds.append(med)
                base = "standard" if key.startswith("standard_fill") else "barebone" if key.startswith("barebone_fill") else None
                rows.append(
                    {
                        "model": model_name,
                        "arm": key,
                        "certainty": cert,
                        "uncertainty": unc,
                        "run": str(path.parent.relative_to(REPO_ROOT)),
                        "n_records": len(records),
                        "n_clean": len(clean_probability_records(records)),
                        "n_paired": len(diffs),
                        "median_dH": med,
                        "mean_dH": float(np.mean(diffs)) if diffs else float("nan"),
                        "padding_reached_model": "" if base is None else validity[(model_slug, base)],
                    }
                )
            data[(model_slug, key)] = meds

    fig, axes = plt.subplots(1, len(MODELS), figsize=(FIG_WIDTH, 4.6), sharey=True)
    offsets = np.linspace(-0.16, 0.16, len(PAIRS))
    y_lo, y_hi = -0.3, 0.85

    for ax, (model_name, model_slug) in zip(axes, MODELS):
        ax.axhline(0.0, color=ZERO_GRAY, ls="--", lw=1.0, zorder=1)
        # Thin lines link each instruction pair across the padded trio of a family.
        for family, keys in (("standard", ["standard", "standard_fill64", "standard_fill256"]), ("barebone", ["barebone", "barebone_fill64", "barebone_fill256"])):
            if not validity[(model_slug, family)]:
                continue
            arm_meta = {a[0]: a for a in ARMS}
            color = arm_meta[keys[0]][3]
            for i in range(len(PAIRS)):
                xs = [arm_meta[k][4] + offsets[i] for k in keys]
                ys = [data[(model_slug, k)][i] for k in keys]
                ax.plot(xs, ys, color=color, lw=0.9, alpha=0.35, zorder=2)

        # Padded arms whose padding was stripped by the chat template: one shaded band
        # per family, no data plotted.
        for family in ("standard", "barebone"):
            if validity[(model_slug, family)]:
                continue
            xs = [a[4] for a in ARMS if a[5] == family]
            ax.axvspan(min(xs) - 0.45, max(xs) + 0.45, color=NA_FILL, lw=0, zorder=0)
            ax.text(
                0.5 * (min(xs) + max(xs)),
                0.5 * (y_lo + y_hi),
                "padding stripped\nby chat template",
                rotation=90,
                ha="center",
                va="center",
                fontsize=plt.rcParams["legend.fontsize"],
                color="#6B7280",
                linespacing=1.15,
            )

        for key, _, _, color, x, base in ARMS:
            if base is not None and not validity[(model_slug, base)]:
                continue
            meds = np.asarray(data[(model_slug, key)], dtype=float)
            ax.scatter(x + offsets, meds, s=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
            centre = float(np.nanmedian(meds))
            ax.plot([x - 0.32, x + 0.32], [centre, centre], color=color, lw=3.2, solid_capstyle="butt", zorder=4)
            ax.plot([x - 0.32, x + 0.32], [centre, centre], color="#111827", lw=0.8, solid_capstyle="butt", zorder=5)

        ax.set_xticks([a[4] for a in ARMS])
        ax.set_xticklabels([a[2] for a in ARMS], rotation=50, ha="right", rotation_mode="anchor")
        for tick, arm in zip(ax.get_xticklabels(), ARMS):
            tick.set_color(arm[3] if arm[3] not in (MINIMAL_COLOR, USER_RULES_COLOR) else {MINIMAL_COLOR: "#B7791F", USER_RULES_COLOR: "#9F67E0"}[arm[3]])
        ax.set_xlim(-0.6, 8.8)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(display_model_name(model_name), color=TEXT_GRAY, pad=6)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel(r"Median $\Delta$Entropy" "\n" r"H(uncertainty) $-$ H(certainty)")

    handles = [
        Line2D([0], [0], marker="o", ls="none", markersize=6.5, markerfacecolor="#6B7280", markeredgecolor="white", label="Instruction pair (median over 200 items)"),
        Line2D([0], [0], color="#6B7280", lw=3.2, label="Median over the 7 pairs"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=True)
    fig.tight_layout(w_pad=1.0)

    pdf, png = save(fig, STEM)
    plt.close(fig)

    csv_path = OUT_DIR / f"{STEM}_medians.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "padding_reached_model": {f"{m}|{b}": v for (m, b), v in validity.items()},
        "median_over_pairs": {
            model_slug: {key: float(np.nanmedian(data[(model_slug, key)])) for key, *_ in ARMS} for _, model_slug in MODELS
        },
        "per_pair_medians": {
            model_slug: {key: data[(model_slug, key)] for key, *_ in ARMS} for _, model_slug in MODELS
        },
        "pairs": PAIRS,
    }
    (OUT_DIR / f"{STEM}_summary.json").write_text(json.dumps(summary, indent=2))

    print("padding reached model:", {f"{m}|{b}": v for (m, b), v in validity.items()})
    header = f"{'model':38s}" + "".join(f"{a[0][:15]:>16s}" for a in ARMS)
    print(header)
    for model_name, model_slug in MODELS:
        print(f"{model_name:38s}" + "".join(f"{np.nanmedian(data[(model_slug, a[0])]):>+16.3f}" for a in ARMS))
    print("range of per-pair medians:")
    for model_name, model_slug in MODELS:
        print(f"{model_name:38s}" + "".join(
            f"{np.nanmin(data[(model_slug, a[0])]):>+8.2f}{np.nanmax(data[(model_slug, a[0])]):>+8.2f}" for a in ARMS))
    bad = [r for r in rows if r["n_paired"] != 200]
    print(f"cells with n_paired != 200: {len(bad)}")
    for r in bad:
        print("  ", r["model"], r["arm"], r["certainty"], r["n_records"], r["n_clean"], r["n_paired"])
    for p in (pdf, png, csv_path, OUT_DIR / f"{STEM}_summary.json"):
        print(p)


if __name__ == "__main__":
    main()
