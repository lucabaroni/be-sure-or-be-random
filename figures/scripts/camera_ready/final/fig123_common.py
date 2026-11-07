"""Shared helpers for the camera-ready Fig. 1-3 scripts (fig1_*, fig2_*, fig3_* in this folder).

* Run resolution restricted to post-fix results (results/neutral/, run date >= 2026-07-20).
* Validity rule: a record counts only if P(A) + P(B) > 0.9 (common.is_valid_probability_record).
* Programmatic text-overlap / clipping check on a drawn matplotlib figure.
"""
from __future__ import annotations

import itertools
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[2]  # figures/scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import REPO_ROOT  # noqa: E402
from be_sure_or_random.utils import instruction_label, slugify_model_name  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "figures" / "camera_ready"
SUBMITTED_DIR = HERE.parents[3]  # figures/
MIN_DATE = "20260720"
_STAMP = re.compile(r"_(\d{8})_(\d{6})$")


def _pick(parent: Path, label: str, min_date: str | None, max_date: str | None = None) -> Path:
    cands = []
    for d in parent.iterdir() if parent.exists() else []:
        m = _STAMP.search(d.name)
        if not (d.is_dir() and m and d.name[: m.start()] == label):
            continue
        if min_date and m.group(1) < min_date:
            continue
        if max_date and m.group(1) > max_date:
            continue
        if (d / "results.json").exists():
            cands.append((m.group(1) + m.group(2), d))
    if not cands:
        raise FileNotFoundError(f"no run {label} in {parent} (dates {min_date}..{max_date})")
    return max(cands)[1] / "results.json"


def resolve_run(model: str, cert: str, unc: str, template: str = "standard") -> Path:
    """Latest post-fix run (results/neutral, date >= MIN_DATE). template: 'standard' or a prompt variant."""
    assert os.environ.get("AB_PAIRS_VARIANT") == "neutral"
    base = RESULTS / "neutral"
    parent = (base / "zero_shot_local" / slugify_model_name(model) if template == "standard"
              else base / "prompt_variants" / template / slugify_model_name(model))
    return _pick(parent, instruction_label(cert, unc), MIN_DATE)


def resolve_submitted_run(model: str, cert: str, unc: str) -> Path:
    """Pre-fix run used for the SUBMITTED figures (original pairs, 2026-04-02); validation only."""
    parent = RESULTS / "zero_shot_local" / slugify_model_name(model)
    return _pick(parent, instruction_label(cert, unc), "20260401", "20260410")


def text_overlap_report(fig, tol: float = 0.0) -> list[str]:
    """Draw the figure and record every text artist that is actually rendered (this excludes
    tick labels of ticks outside the view limits, which matplotlib keeps but never draws). Then
    check that no two non-empty texts overlap (window extents) and that none extends beyond the
    figure edge. Returns a list of problems (empty == pass)."""
    from matplotlib.text import Text

    drawn: list[Text] = []
    orig_draw = Text.draw

    def recording_draw(self, renderer):
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return orig_draw(self, renderer)

    Text.draw = recording_draw
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig_draw
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    items = [(t, t.get_window_extent(renderer)) for t in dict.fromkeys(drawn)]
    problems = []
    for t, bb in items:
        if bb.x0 < fb.x0 - tol or bb.y0 < fb.y0 - tol or bb.x1 > fb.x1 + tol or bb.y1 > fb.y1 + tol:
            problems.append(f"clipped by figure edge: {t.get_text()!r} {tuple(round(v, 1) for v in bb.bounds)}")
    for (t1, b1), (t2, b2) in itertools.combinations(items, 2):
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox > tol and oy > tol:
            problems.append(f"overlap ({ox:.1f}x{oy:.1f}px): {t1.get_text()!r} <-> {t2.get_text()!r}")
    print(f"[overlap check] {len(items)} rendered text artists checked")
    return problems
