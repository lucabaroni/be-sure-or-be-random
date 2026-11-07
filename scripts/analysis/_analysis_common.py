"""Shared setup for scripts/analysis: repository paths and run selection by name.

Importing this module puts the repository root, figures/scripts (common.py) and
figures/scripts/camera_ready/final (sweep_models.py) at the front of sys.path, so the
in-repo be_sure_or_random package is used even when another copy is installed.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("AB_PAIRS_VARIANT", "neutral")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "figures" / "scripts" / "camera_ready" / "final", ROOT / "figures" / "scripts", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from be_sure_or_random.utils import get_results_dir, instruction_label, slugify_model_name  # noqa: E402

ANALYSIS_DIR = ROOT / "analysis_outputs"
POST_FIX_LOCAL = 20260720  # local runs before this date predate the chat-template fixes


def run_stamp(path: Path) -> str:
    """'YYYYMMDD_HHMMSS' of a run directory (or of the results.json inside it)."""
    name = path.parent.name if path.name == "results.json" else path.name
    return name[-15:]


def latest_run(model_dir: Path, cert: str, unc: str, *, exclude_stamps: tuple[str, ...] = ()) -> Path | None:
    """results.json of the run with the latest <YYYYMMDD_HHMMSS> stamp for this pair.

    Selection is by directory name, so it does not depend on file modification times
    (which a fresh clone or copy does not preserve).
    """
    if not model_dir.is_dir():
        return None
    pattern = re.compile(re.escape(instruction_label(cert, unc)) + r"_\d{8}_\d{6}")
    runs = [
        p for p in model_dir.iterdir()
        if pattern.fullmatch(p.name) and (p / "results.json").is_file() and run_stamp(p) not in exclude_stamps
    ]
    return max(runs, key=lambda p: p.name) / "results.json" if runs else None


__all__ = [
    "ANALYSIS_DIR", "POST_FIX_LOCAL", "ROOT", "get_results_dir",
    "instruction_label", "latest_run", "run_stamp", "slugify_model_name",
]
