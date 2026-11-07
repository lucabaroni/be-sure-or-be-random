#!/usr/bin/env python3
"""Fig. 3 (multimodel_distributions.pdf), camera-ready, two variants.

  mixed   -> multimodel_distributions_mixed.pdf
             Qwen3.5-27B (standard, "Be sure!"/"Be random!"), Llama-3-8B (standard, same pair),
             Qwen3.5-35B-A3B (minimal prompt = template `barebone`, "Aim for low/high entropy").
  all_aim -> multimodel_distributions_all_aim.pdf
             same rows, all with "Aim for low entropy"/"Aim for high entropy"
             (rows 1-2 standard prompt, row 3 minimal).

Style = SUBMITTED figures/multimodel_distributions.pdf (generator:
generate_supplementary_multimodel_figure2_rows_with_prob_five_models.py at larger fonts):
page 1376.39 x 418.67 pt and axes rectangles copied from the submitted PDF; legend 16 pt, ticks
18 pt, axis labels 19 pt, bold black row labels 17 pt; certainty #2563EB, uncertainty #DC2626,
ΔH #9333EA; histograms 15 bins, alpha 0.45; KDE lw 2 (ΔH 1.6), fill alpha 0.18; zero line
#9CA3AF dashed; ΔH median dashed lw 1.6. `--validate-old DIR` rebuilds the submitted figure from
the pre-fix runs it was drawn from (rows 27B / Llama-3-8B / 9B), to prove the style match.

Deliberate deviations from the submitted figure (camera-ready decisions):
  1. per-row y-scaling of the KDE panels: within each row the two entropy panels share one
     y-limit and the ΔH panel has its own, each = 1.155 x the tallest KDE peak (1.155 = the
     submitted 1.05 autoscale margin x 1.1 headroom). The submitted figure shared one scale
     across all rows (ΔH limit = half the entropy limit); after the fix Llama-3-8B's entropies
     and ΔH are almost all 0, so its KDEs are narrow spikes that flatten every other row under a
     shared scale (and, under a per-row 2:1 coupling, its own entropy panels). The count axis
     stays shared across rows, as submitted.
  2. row 3 label states "minimal prompt" (two lines; 13 pt, the largest size that fits between
     the row-2 label and the "Count" label on the submitted page).
  3. mixed variant only: rows use different pairs, so the column x-labels are generic
     ("Entropy (uncertainty instruction)" / "Entropy (certainty instruction)"); the instruction
     names are in each row's legend (as in the submitted figure).
  4. validity rule P(A)+P(B) > 0.9 also applied to the P(option A) histograms (the submitted
     generator applied it only to the entropy panels); TrueType fonts (pdf.fonttype 42).
  5. the bottom-row "0" count tick label is lifted by 4.5 pt: in the submitted layout its text
     extent overlaps the "0.0" x tick label below it (3.5 pt), failing the no-overlap check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig123_common as fc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import ScaledTranslation, offset_copy  # noqa: E402

from common import (  # noqa: E402
    apply_paper_style,
    clean_probability_records,
    display_model_name,
    entropy_values_for_instruction,
    kde_curve,
    load_zero_shot_result,
    normalized_option_a_probability,
    paired_entropy_differences,
    plot_kde,
)

CERT_COLOR, UNC_COLOR, DIFF_COLOR = "#2563EB", "#DC2626", "#9333EA"
KDE_ALPHA, DIFF_LW = 0.18, 1.6
SURE = ("Be sure!", "Be random!")
AIM = ("Aim for low entropy", "Aim for high entropy")
# row = (model, template, (certainty, uncertainty), note)
VARIANTS = {
    "mixed": [("qwen/Qwen3.5-27B", "standard", SURE, None),
              ("meta-llama/Meta-Llama-3-8B-Instruct", "standard", SURE, None),
              ("qwen/Qwen3.5-35B-A3B", "barebone", AIM, "minimal prompt")],
    "all_aim": [("qwen/Qwen3.5-27B", "standard", AIM, None),
                ("meta-llama/Meta-Llama-3-8B-Instruct", "standard", AIM, None),
                ("qwen/Qwen3.5-35B-A3B", "barebone", AIM, "minimal prompt")],
}
SUBMITTED_ROWS = [("qwen/Qwen3.5-27B", "standard", SURE, None),
                  ("meta-llama/Meta-Llama-3-8B-Instruct", "standard", SURE, None),
                  ("qwen/Qwen3.5-9B", "standard", SURE, None)]

# Geometry of the submitted PDF (points, origin bottom-left).
PAGE_W, PAGE_H = 1376.39376, 418.673037
COL_X = [95.6008094432, 417.402405669, 739.2040018949, 1068.5446606207]
ROW_Y = [300.175160492, 176.109214508, 54.0125]
AX_W, AX_H = 286.3365962258, 95.017628032
ROW_LABEL_X = -0.28  # axes fraction of column 0
ROW_LABEL_FS, NOTE_LABEL_FS = 17, 13
NOTE_LINE_HALF_GAP_PT = 7.5  # half the distance between the two row-3 label lines (pt)
ZERO_LABEL_NUDGE_PT = 4.5  # bottom-row "0" count label vs the "0.0" x tick label (see deviation 5)


def load_row(model, template, pair, old):
    cert, unc = pair
    path = fc.resolve_submitted_run(model, cert, unc) if old else fc.resolve_run(model, cert, unc, template)
    recs = load_zero_shot_result(path)
    valid = clean_probability_records(recs)
    probs = {i: [normalized_option_a_probability(r) for r in valid if r.get("instruction") == i] for i in pair}
    return dict(path=path, n_records={i: sum(r.get("instruction") == i for r in recs) for i in pair},
                probs=probs, ent={i: entropy_values_for_instruction(recs, i) for i in pair},
                dh=np.asarray(paired_entropy_differences(recs, cert, unc), dtype=float))


def peak(values, lo, hi):
    return float(kde_curve(values, lo, hi)[1].max())


def build_figure(rows, data, *, per_row_scale: bool, generic_xlabels: bool, nudge_zero: bool) -> plt.Figure:
    apply_paper_style()
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 19, "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 16, "pdf.fonttype": 42, "savefig.bbox": None})
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    axes = np.array([[fig.add_axes([x / PAGE_W, y / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H]) for x in COL_X]
                     for y in ROW_Y])
    for r in range(1, len(rows)):
        axes[r, 0].sharey(axes[0, 0])
        for c in range(4):
            axes[r, c].sharex(axes[0, c])
    bins = np.linspace(0.0, 1.0, 16)
    # KDE y-limits. Submitted rule (generate_supplementary_multimodel_figure2_rows_with_prob_five_models.py):
    # one scale for all rows, ΔH limit = max(ΔH autoscale top, entropy autoscale top / 2) * 1.1,
    # entropy limit = 2 x ΔH limit. Camera-ready (per_row_scale): see deviation 1 in the docstring.
    ent_peak = [max(peak(d["ent"][p[1]], 0, 1), peak(d["ent"][p[0]], 0, 1)) for (_, _, p, _), d in zip(rows, data)]
    diff_peak = [peak(d["dh"], -1, 1) for d in data]
    if per_row_scale:
        ent_lim = [1.155 * v for v in ent_peak]
        diff_lim = [1.155 * v for v in diff_peak]
    else:
        dmax = max(1.05 * max(diff_peak), 1.05 * max(ent_peak) / 2.0) * 1.1
        ent_lim, diff_lim = [2 * dmax] * len(rows), [dmax] * len(rows)
    for r, ((model, template, (cert, unc), note), d) in enumerate(zip(rows, data)):
        ax = axes[r]
        lab_kw = dict(transform=ax[0].transAxes, va="center", ha="center", rotation=90,
                      fontweight="bold", color="black", clip_on=False)
        if note is None:
            ax[0].text(ROW_LABEL_X, 0.5, display_model_name(model), fontsize=ROW_LABEL_FS, **lab_kw)
        else:  # two lines: bold model name (left), regular-weight note (right)
            dx = NOTE_LINE_HALF_GAP_PT
            ax[0].text(ROW_LABEL_X, 0.5, display_model_name(model), fontsize=NOTE_LABEL_FS,
                       **{**lab_kw, "transform": offset_copy(ax[0].transAxes, fig=fig, x=-dx, y=0, units="points")})
            ax[0].text(ROW_LABEL_X, 0.5, f"({note})", fontsize=NOTE_LABEL_FS,
                       **{**lab_kw, "fontweight": "normal",
                          "transform": offset_copy(ax[0].transAxes, fig=fig, x=dx, y=0, units="points")})
        ax[0].hist(d["probs"][cert], bins=bins, alpha=0.45, color=CERT_COLOR, label=f'"{cert}"')
        ax[0].hist(d["probs"][unc], bins=bins, alpha=0.45, color=UNC_COLOR, label=f'"{unc}"')
        ax[0].set_xlim(0.0, 1.0)
        ax[0].legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#D1D5DB")
        ax[0].set_ylabel("Count")
        plot_kde(ax[1], d["ent"][unc], UNC_COLOR, "", 0.0, 1.0, alpha=KDE_ALPHA)
        plot_kde(ax[2], d["ent"][cert], CERT_COLOR, "", 0.0, 1.0, alpha=KDE_ALPHA)
        plot_kde(ax[3], d["dh"], DIFF_COLOR, "", -1.0, 1.0, alpha=KDE_ALPHA, linewidth=DIFF_LW)
        ax[3].axvline(0.0, color="#9CA3AF", linewidth=1, linestyle="--")
        if r == len(rows) - 1:
            ax[0].set_xlabel("P(option A)")
            ax[1].set_xlabel("Entropy (uncertainty instruction)" if generic_xlabels else f'Entropy ("{unc}")')
            ax[2].set_xlabel("Entropy (certainty instruction)" if generic_xlabels else f'Entropy ("{cert}")')
            ax[3].set_xlabel("ΔEntropy")
        ax[1].set_xlim(0.0, 1.0)
        ax[2].set_xlim(0.0, 1.0)
        ax[3].set_xlim(-1.0, 1.0)
        ax[1].set_ylim(0, ent_lim[r])
        ax[2].set_ylim(0, ent_lim[r])
        ax[3].set_ylim(0, diff_lim[r])
        ax[3].axvline(float(np.median(d["dh"])), color=DIFF_COLOR, linestyle="--", linewidth=DIFF_LW,
                      zorder=6, clip_on=False)
        for c in range(4):
            a = ax[c]
            a.grid(False)
            a.set_xticks([-1, -0.5, 0, 0.5, 1] if c == 3 else [0, 0.5, 1])
            if r != len(rows) - 1:
                a.tick_params(labelbottom=False)
            if c:
                a.tick_params(axis="y", which="both", left=False, labelleft=False)
                a.spines["left"].set_visible(False)
    axes[0, 0].set_ylim(0, axes[0, 0].get_ylim()[1])
    if nudge_zero:
        # In the submitted layout the bottom-left "0" count label and the "0.0" x tick label
        # overlap by 3.5 pt (text extents); lift that single label by 4.5 pt.
        fig.canvas.draw()
        ax = axes[-1, 0]
        tick = next(t for t in ax.yaxis.get_major_ticks() if abs(t.get_loc()) < 1e-9)
        tick.label1.set_transform(tick.label1.get_transform()
                                  + ScaledTranslation(0, ZERO_LABEL_NUDGE_PT / 72, fig.dpi_scale_trans))
    return fig


def summary(rows, data) -> dict:
    out = {}
    for (model, template, (cert, unc), note), d in zip(rows, data):
        row = dict(source=str(d["path"].relative_to(fc.REPO_ROOT)), template=template, n_records=d["n_records"])
        for i in (cert, unc):
            pa, h = np.asarray(d["probs"][i]), np.asarray(d["ent"][i])
            row[i] = dict(n_valid=int(h.size), median_H=round(float(np.median(h)), 3), mean_H=round(float(h.mean()), 3),
                          frac_H_below_0p1=round(float((h < 0.1).mean()), 3), frac_H_above_0p9=round(float((h > 0.9).mean()), 3),
                          mean_PA=round(float(pa.mean()), 3), frac_PA_gt_0p5=round(float((pa > 0.5).mean()), 3),
                          frac_PA_gt_0p9=round(float((pa > 0.9).mean()), 3))
        dh = d["dh"]
        row["dH"] = dict(n=int(dh.size), median=round(float(np.median(dh)), 3), mean=round(float(dh.mean()), 3),
                         frac_pos=round(float((dh > 0).mean()), 3), frac_abs_lt_0p05=round(float((np.abs(dh) < 0.05).mean()), 3))
        out[f"{display_model_name(model)} [{template}] {unc!r} vs {cert!r}"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="mixed")
    ap.add_argument("--validate-old", metavar="OUTDIR")
    args = ap.parse_args()
    old = args.validate_old is not None
    rows = SUBMITTED_ROWS if old else VARIANTS[args.variant]
    data = [load_row(m, t, p, old) for m, t, p, _ in rows]
    same_pair = len({p for _, _, p, _ in rows}) == 1
    fig = build_figure(rows, data, per_row_scale=not old, generic_xlabels=not same_pair, nudge_zero=not old)
    problems = fc.text_overlap_report(fig)
    print("text overlap/clipping check:", "PASS" if not problems else "FAIL", *problems, sep="\n  ")
    out_dir = Path(args.validate_old) if old else fc.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / ("multimodel_distributions" if old else f"multimodel_distributions_{args.variant}")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary(rows, data), indent=1))
    print("wrote", f"{stem}.pdf", f"{stem}.png")


if __name__ == "__main__":
    main()
