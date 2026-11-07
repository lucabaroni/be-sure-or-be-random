#!/usr/bin/env python3
"""Fig. 1 (schematic_red_blue): regenerate the embedded entropy histograms from the post-fix run.

The submitted figures/schematic_red_blue.html embeds two 10-bin histograms in the
`data-dist` attributes of #hist1 ("Be random!") and #hist2 ("Be sure!"). They are
np.histogram(H, bins=10, range=(0, 1)) of per-record binary entropy (bits, renormalised
over the two option tokens) of Qwen3.5-27B, standard prompt; the submitted values
({0,0,0,0,0,1,9,13,40,135} / {54,38,20,15,10,8,11,6,11,25}) are reproduced exactly by the
pre-fix run results/zero_shot_local/qwen_Qwen3.5-27B/Be_sure_Be_random_20260402_212746.
The page's JS turns them into KDEs and into the ΔEntropy panel (independent sampling from
the two histograms, 5000 draws, KDE bandwidth 0.12; dashed median). Only the two
attributes are changed here; everything else in the HTML is copied verbatim.

Run (project venv, from the repo root):
    AB_PAIRS_VARIANT=neutral ~/Luca/be-sure-or-random/.venv/bin/python \
        figures/scripts/camera_ready/final/fig1_schematic_data.py
then render with fig1_render_schematic_pdf.py (tools venv).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")
assert os.environ["AB_PAIRS_VARIANT"] == "neutral"

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # figures/scripts
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    REPO_ROOT,
    clean_probability_records,
    latest_result_path,
    load_zero_shot_result,
    paired_entropy_differences,
    record_entropy,
)

MODEL = "qwen/Qwen3.5-27B"
CERT, UNC = "Be sure!", "Be random!"
SUBMITTED_HTML = HERE.parents[3] / "schematic_red_blue.html"  # figures/schematic_red_blue.html
OUT_DIR = REPO_ROOT / "figures" / "camera_ready"
OLD = {"hist1": "{0,0,0,0,0,1,9,13,40,135}", "hist2": "{54,38,20,15,10,8,11,6,11,25}"}
MIN_DATE = "20260720"


def run_date(path: Path) -> str:
    return re.search(r"_(\d{8})_\d{6}$", path.parent.name).group(1)


def main() -> None:
    path = latest_result_path("zero_shot_local", model_name=MODEL, certainty_instruction=CERT,
                              uncertainty_instruction=UNC)
    assert "results/neutral/" in str(path) and run_date(path) >= MIN_DATE, path
    records = load_zero_shot_result(path)
    valid = clean_probability_records(records)  # P(A)+P(B) > 0.9
    hists, stats = {}, {}
    for key, instr in (("hist1", UNC), ("hist2", CERT)):
        n_all = sum(r.get("instruction") == instr for r in records)
        ent = np.array([record_entropy(r) for r in valid if r.get("instruction") == instr])
        counts, _ = np.histogram(ent, bins=10, range=(0.0, 1.0))
        hists[key] = "{" + ",".join(str(int(c)) for c in counts) + "}"
        stats[instr] = dict(n_records=n_all, n_valid=len(ent), median_H=float(np.median(ent)),
                            mean_H=float(np.mean(ent)), hist=counts.tolist())
    dh = np.asarray(paired_entropy_differences(records, CERT, UNC))
    stats["paired_dH"] = dict(n=len(dh), median=float(np.median(dh)), mean=float(dh.mean()),
                              frac_pos=float((dh > 0).mean()), frac_zero=float((dh == 0).mean()),
                              frac_neg=float((dh < 0).mean()))
    html = SUBMITTED_HTML.read_text()
    for key, new in hists.items():
        old_attr = f'id="{key}" data-dist="{OLD[key]}"'
        assert html.count(old_attr) == 1, key
        html = html.replace(old_attr, f'id="{key}" data-dist="{new}"')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "schematic_red_blue.html").write_text(html)
    print("source:", path.relative_to(REPO_ROOT))
    print(json.dumps({"hist1 (Be random!)": hists["hist1"], "hist2 (Be sure!)": hists["hist2"], **stats}, indent=1))
    print("wrote", OUT_DIR / "schematic_red_blue.html")


if __name__ == "__main__":
    main()
