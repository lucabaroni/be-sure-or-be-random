#!/usr/bin/env python3
"""Render the Fig. 1 schematic HTML to a cropped vector PDF (camera-ready).

Run with the separate tools venv (NOT the project venv):
    ~/Luca/tools/render-venv/bin/python fig1_render_schematic_pdf.py IN.html OUT.pdf [--size 554x171]

Pipeline (mirrors how the submitted schematic_red_blue.pdf was made: browser print-to-PDF
at 96 px/in, then a pdfcrop-style tight crop):
  1. Load the HTML in headless Chromium (Playwright), wait for the Google-Fonts webfont
     (IBM Plex Sans) and for the page scripts that draw the KDEs and connector lines.
  2. Verify through the DevTools protocol that every text node is rendered with the
     IBM Plex Sans webfont (CSS.getPlatformFontsForNode); abort otherwise.
  3. print-to-PDF on a large page with backgrounds, scale 1 (1 CSS px = 0.75 pt).
  4. Find the ink bounding box (non-white pixels of a 600-dpi raster, like pdfcrop/gs bbox)
     and set Media/CropBox to a box of exactly --size points centred on it.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

PAGE_W_PX, PAGE_H_PX = 1200, 520


def check_fonts(page) -> dict:
    cdp = page.context.new_cdp_session(page)
    cdp.send("DOM.enable")
    cdp.send("CSS.enable")
    doc = cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})
    usage: dict[str, int] = {}
    bad: list[str] = []

    def walk(node):
        if node.get("nodeType") == 1 and node.get("nodeName") not in ("SCRIPT", "STYLE", "HEAD"):
            has_text = any(c.get("nodeType") == 3 and c.get("nodeValue", "").strip() for c in node.get("children", []))
            if has_text:
                fonts = cdp.send("CSS.getPlatformFontsForNode", {"nodeId": node["nodeId"]})["fonts"]
                for f in fonts:
                    key = f"{f['familyName']} ({f.get('postScriptName', '')}, custom={f['isCustomFont']})"
                    usage[key] = usage.get(key, 0) + f["glyphCount"]
                    if "IBM Plex Sans" not in f["familyName"] or not f["isCustomFont"]:
                        bad.append(f"{node['nodeName']}: {f}")
        for c in node.get("children", []):
            walk(c)

    walk(doc["root"])
    return {"usage": usage, "bad": bad}


TEXT_RECTS_JS = """() => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n, id = 0;
  while ((n = walker.nextNode())) {
    if (!n.nodeValue.trim()) continue;
    const parent = n.parentElement;
    if (!parent || getComputedStyle(parent).visibility === "hidden") continue;
    const r = document.createRange();
    r.selectNodeContents(n);
    for (const b of r.getClientRects())
      if (b.width > 0 && b.height > 0)
        out.push({id, text: n.nodeValue.trim(), parent: parent.className || parent.tagName,
                  fs: parseFloat(getComputedStyle(parent).fontSize), x0: b.left, y0: b.top, x1: b.right, y1: b.bottom});
    id++;
  }
  return out;
}"""


def text_overlap_report(rects: list[dict]) -> list[str]:
    """Pairwise overlap of the glyph em-boxes of all rendered text nodes. Each Range rect is the
    inline content area (taller than the em box); shrink it vertically to the font-size so that the
    tight line-height of stacked labels (e.g. 'N binary' / 'questions') is not a false positive."""
    boxes = []
    for r in rects:
        cy = (r["y0"] + r["y1"]) / 2
        boxes.append((r, r["x0"], cy - r["fs"] / 2, r["x1"], cy + r["fs"] / 2))
    problems = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (a, ax0, ay0, ax1, ay1), (b, bx0, by0, bx1, by1) = boxes[i], boxes[j]
            if a["id"] == b["id"]:
                continue
            ox, oy = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if ox > 0 and oy > 0:
                problems.append(f"overlap {ox:.1f}x{oy:.1f}px: {a['text']!r} <-> {b['text']!r}")
    return problems


def ink_bbox_pt(pdf_bytes: bytes, dpi: int = 600) -> tuple[float, float, float, float, float]:
    """Return (llx, lly, urx, ury, page_h) of non-white ink in PDF points (origin bottom-left)."""
    import numpy as np  # noqa: F401  (available in render venv? fall back to pure python)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "page.pdf"
        src.write_bytes(pdf_bytes)
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-singlefile", str(src), str(Path(td) / "r")], check=True)
        pgm = (Path(td) / "r.pgm").read_bytes()
    # parse binary PGM (P5)
    parts = pgm.split(maxsplit=4)
    assert parts[0] == b"P5"
    w, h, maxv = int(parts[1]), int(parts[2]), int(parts[3])
    data = parts[4]
    import numpy as np
    img = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
    ys, xs = np.nonzero(img < maxv)
    page_h = PdfReader(io.BytesIO(pdf_bytes)).pages[0].mediabox.height
    s = 72.0 / dpi
    llx, urx = xs.min() * s, (xs.max() + 1) * s
    top, bottom = ys.min() * s, (ys.max() + 1) * s
    return llx, float(page_h) - bottom, urx, float(page_h) - top, float(page_h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("out_pdf")
    ap.add_argument("--size", default="554x171", help="final page size in pt (submitted: 554x171)")
    ap.add_argument("--seed", type=int, default=None,
                    help="if given, replace Math.random by a seeded PRNG (deterministic KDE sampling)")
    args = ap.parse_args()
    tw, th = (float(v) for v in args.size.split("x"))

    html = Path(args.html).resolve()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--font-render-hinting=none"])  # unhinted advances, as in the submitted (macOS) render
        page = browser.new_page(viewport={"width": PAGE_W_PX, "height": PAGE_H_PX})
        if args.seed is not None:
            page.add_init_script(
                "(() => { let s = %d >>> 0; Math.random = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s;"
                " t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);"
                " return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();" % args.seed)
        page.goto(html.as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        loaded = page.evaluate("""() => [...document.fonts].filter(f => f.status === "loaded")
                                       .map(f => `${f.family} ${f.weight} ${f.style}`)""")
        page.wait_for_timeout(1500)  # page draws connector lines 80 ms after load
        page.emulate_media(media="print")
        page.wait_for_timeout(300)
        fonts = check_fonts(page)
        rects = page.evaluate(TEXT_RECTS_JS)
        pdf_bytes = page.pdf(width=f"{PAGE_W_PX}px", height=f"{PAGE_H_PX}px", print_background=True,
                             margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                             scale=1, prefer_css_page_size=False)
        browser.close()

    print("webfonts loaded:", sorted(set(loaded)))
    print("platform fonts used (glyph counts):", json.dumps(fonts["usage"], indent=1))
    if fonts["bad"]:
        print("ERROR: text rendered with a non-IBM-Plex font:", *fonts["bad"], sep="\n  ")
        sys.exit(1)

    problems = text_overlap_report(rects)
    print(f"text overlap check ({len(rects)} text boxes):", "PASS" if not problems else "FAIL", *problems, sep="\n  ")
    llx, lly, urx, ury, _ = ink_bbox_pt(pdf_bytes)
    iw, ih = urx - llx, ury - lly
    print(f"ink bbox: {iw:.2f} x {ih:.2f} pt at ({llx:.2f}, {lly:.2f})")
    if iw > tw + 0.5 or ih > th + 0.5:
        print(f"WARNING: ink ({iw:.1f}x{ih:.1f}) larger than target {tw}x{th}")
    cx, cy = (llx + urx) / 2, (lly + ury) / 2
    box = RectangleObject([cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2])
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    pg = reader.pages[0]
    pg.mediabox = box
    pg.cropbox = box
    pg.trimbox = box
    pg.artbox = box
    pg.bleedbox = box
    writer.add_page(pg)
    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        writer.write(fh)
    print("wrote", out)


if __name__ == "__main__":
    main()
