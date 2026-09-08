"""Shared helper so every generated dashboard/comparison HTML page points at
the same plotly.min.js instead of each script (plot_metrics.py,
plot_comparison.py) copying its own duplicate into every output directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly

# Used only if the pip-installed plotly package doesn't ship its bundled
# plotly.min.js for some reason (e.g. a minimal/custom install).
CDN_FALLBACK = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def plotly_src(output_dir: str | Path) -> str:
    """The value to pass as ``include_plotlyjs`` (plot_metrics.py) or
    substitute into a <script src=...> tag (plot_comparison.py) so an HTML
    page written to output_dir loads Plotly from the plotly package's own
    bundled plotly.min.js -- the exact same file every script here already
    has installed locally -- via a relative path, rather than each script
    copying its own duplicate into output_dir.

    Note: this ties the generated HTML to this checkout + venv. Moving or
    sharing a dashboard/comparison HTML file on its own, without this
    repo and venv at the same relative location, will leave it unable to
    load Plotly. That's an acceptable trade-off for viewing runs/ locally,
    but not for handing a standalone HTML file to someone else.
    """
    bundled = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not bundled.exists():
        return CDN_FALLBACK
    return os.path.relpath(bundled, Path(output_dir).resolve())
