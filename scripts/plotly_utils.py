"""Shared helper so every generated dashboard/comparison HTML page points at
the same plotly.min.js instead of each script (plot_metrics.py,
plot_comparison.py) copying its own duplicate into every output directory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import plotly

# Used only if the pip-installed plotly package doesn't ship its bundled
# plotly.min.js for some reason (e.g. a minimal/custom install).
CDN_FALLBACK = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def plotly_src(output_dir: str | Path, runs_dir: str | Path) -> str:
    """The value to pass as ``include_plotlyjs`` (plot_metrics.py) or
    substitute into a <script src=...> tag (plot_comparison.py) so an HTML
    page written to output_dir loads Plotly via a relative path to a single
    plotly.min.js copied into runs_dir -- shared by every dashboard/
    comparison page under runs_dir, rather than each script copying its own
    duplicate into every output directory, and independent of this
    checkout's venv location or Python version (unlike pointing straight at
    the venv's own bundled copy, which breaks the moment either changes).
    """
    runs_dir = Path(runs_dir).resolve()
    dest = runs_dir / "plotly.min.js"
    if not dest.exists():
        bundled = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if not bundled.exists():
            return CDN_FALLBACK
        runs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, dest)
    return os.path.relpath(dest, Path(output_dir).resolve())
