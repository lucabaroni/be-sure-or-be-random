"""Shared helpers for the camera-ready replicas of Figs. 4 (now appendix), 5, 6 and 11.

Used by:
  fig5_steering_dose_response.py
  fig6_mmlu_entropy_correctness_calibration.py
  fig11_steering_vector_similarity.py
  figA_prompt_ablation.py

Conventions
-----------
* Only post-fix results under results/neutral/ dated 2026-07-20 or later are used.
* Binary-choice records count only if P(A) + P(B) > 0.9 (common.is_valid_probability_record).
* PDFs are written with TrueType fonts (pdf.fonttype 42) plus a 300-dpi PNG preview.
"""
from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]  # figures/scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

REPO = SCRIPTS.parents[1]
RESULTS = REPO / "results" / "neutral"
OUT_DIR = REPO / "figures" / "camera_ready"
COMPARE_DIR = OUT_DIR / "compare"
SUBMITTED_DIR = Path.home() / "Luca" / "Can_model_control_entropy" / "figures"
MIN_DATE = 20260720

_DATE_RE = re.compile(r"_(\d{8})_(\d{6})$")


def run_date(path: Path) -> int | None:
    """Date stamp (YYYYMMDD) of a run directory named ..._YYYYMMDD_HHMMSS."""
    m = _DATE_RE.search(path.name)
    return int(m.group(1)) if m else None


def latest_post_fix(parent: Path, prefix: str, filename: str = "results.json") -> Path:
    """Latest run dir under parent matching prefix_YYYYMMDD_HHMMSS with date >= MIN_DATE."""
    cands = []
    for d in parent.glob(f"{prefix}_*"):
        m = _DATE_RE.search(d.name)
        if not m or not d.name[: m.start()] == prefix:
            continue
        if int(m.group(1)) < MIN_DATE:
            continue
        if (d / filename).exists():
            cands.append((m.group(1) + m.group(2), d / filename))
    if not cands:
        raise FileNotFoundError(f"No post-fix run {prefix}_* under {parent}")
    return sorted(cands)[-1][1]


def set_truetype() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save(fig: plt.Figure, stem: str, **savefig_kw) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, **savefig_kw)
    fig.savefig(png, dpi=300, **savefig_kw)
    return [pdf, png]


def check_text_overlap(fig: plt.Figure, tight: bool = True, pad_px: float = 0.0) -> list[str]:
    """Return a list of problems: overlapping visible texts or texts outside the saved area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if tight:
        area = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    else:
        area = fig.bbox
    # Tick labels: only those matplotlib actually draws (ticks inside the view interval).
    all_tick_texts, drawn_tick_texts = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_tick_texts.update({id(tick.label1), id(tick.label2)})
            for tick in axis._update_ticks():
                drawn_tick_texts.update({id(tick.label1), id(tick.label2)})
    items = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_tick_texts and id(t) not in drawn_tick_texts:
            continue
        # skip texts whose parent axes/legend is invisible
        if t.axes is not None and not t.axes.get_visible():
            continue
        bb = t.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((t.get_text().replace("\n", " | "), bb))
    problems = []
    for (s1, b1), (s2, b2) in itertools.combinations(items, 2):
        if (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > pad_px) and (min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > pad_px):
            problems.append(f"overlap: {s1!r} <-> {s2!r}")
    for s, b in items:
        if b.x0 < area.x0 - 0.5 or b.x1 > area.x1 + 0.5 or b.y0 < area.y0 - 0.5 or b.y1 > area.y1 + 0.5:
            problems.append(f"outside: {s!r}")
    print(f"[overlap check] {len(items)} visible texts, {len(problems)} problems")
    for p in problems:
        print("   ", p)
    return problems


def compare(stem: str, new_pdf: Path | None = None, submitted_stem: str | None = None) -> Path:
    """Render submitted and new PDFs at 100 dpi and stack them (old on top) into compare/<stem>__old_vs_new.png."""
    from PIL import Image, ImageDraw

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    new_pdf = new_pdf or OUT_DIR / f"{stem}.pdf"
    old_pdf = SUBMITTED_DIR / f"{submitted_stem or stem}.pdf"
    tmp_old = COMPARE_DIR / f".{stem}__old"
    tmp_new = COMPARE_DIR / f".{stem}__new"
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", str(old_pdf), str(tmp_old)], check=True)
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", str(new_pdf), str(tmp_new)], check=True)
    a = Image.open(f"{tmp_old}.png").convert("RGB")
    b = Image.open(f"{tmp_new}.png").convert("RGB")
    W = max(a.width, b.width)
    lab = 22
    out = Image.new("RGB", (W, a.height + b.height + 2 * lab + 6), "white")
    d = ImageDraw.Draw(out)
    d.text((4, 4), f"SUBMITTED  {old_pdf.name}  ({a.width}x{a.height} px @100dpi)", fill=(180, 0, 0))
    out.paste(a, (0, lab))
    y = lab + a.height + 6
    d.line([(0, y - 3), (W, y - 3)], fill=(0, 0, 0), width=2)
    d.text((4, y + 4), f"NEW  {new_pdf.name}  ({b.width}x{b.height} px @100dpi)", fill=(0, 0, 180))
    out.paste(b, (0, y + lab))
    dest = COMPARE_DIR / f"{stem}__old_vs_new.png"
    out.save(dest)
    Path(f"{tmp_old}.png").unlink()
    Path(f"{tmp_new}.png").unlink()
    print(dest)
    return dest
