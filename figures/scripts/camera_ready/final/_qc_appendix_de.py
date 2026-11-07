"""QC helpers for the Appendix D/E camera-ready figures (layer_choice, cross_instruction_steering).

check_text_overlap(fig): draws the figure while recording every text artist that is actually
rendered (visible, non-empty; out-of-range tick labels are skipped by matplotlib and therefore
not recorded), then checks that no two rendered texts overlap and that each lies inside the
saved figure area (the tight bbox used by savefig.bbox="tight", which is how the paper figures
are saved).
"""
from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.text import Text


def _drawn_texts(fig: plt.Figure) -> list[Text]:
    drawn: list[Text] = []
    original_draw = Text.draw

    def recording_draw(self, renderer):
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return original_draw(self, renderer)

    Text.draw = recording_draw
    try:
        fig.canvas.draw()
    finally:
        Text.draw = original_draw
    unique: list[Text] = []
    seen: set[int] = set()
    for t in drawn:
        if id(t) not in seen:
            seen.add(id(t))
            unique.append(t)
    return unique


def check_text_overlap(fig: plt.Figure, tol_px: float = 0.0) -> list[str]:
    texts = _drawn_texts(fig)
    renderer = fig.canvas.get_renderer()
    saved_area = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    canvas = fig.bbox
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    problems: list[str] = []
    outside_canvas = 0
    for t, bb in boxes:
        if bb.x0 < saved_area.x0 - 1 or bb.y0 < saved_area.y0 - 1 or bb.x1 > saved_area.x1 + 1 or bb.y1 > saved_area.y1 + 1:
            problems.append(f"outside saved area: {t.get_text()!r}")
        if bb.x0 < canvas.x0 or bb.y0 < canvas.y0 or bb.x1 > canvas.x1 or bb.y1 > canvas.y1:
            outside_canvas += 1
    for (t1, b1), (t2, b2) in combinations(boxes, 2):
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > tol_px and iy > tol_px:
            problems.append(f"overlap ({ix:.1f}x{iy:.1f}px): {t1.get_text()!r} <-> {t2.get_text()!r}")
    print(
        f"[overlap-check] {len(boxes)} rendered text artists, {len(problems)} problems "
        f"({outside_canvas} extend beyond the nominal canvas but lie inside the tight saved area)"
    )
    for p in problems:
        print("   ", p)
    return problems
