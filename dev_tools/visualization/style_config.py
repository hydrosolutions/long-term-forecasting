"""Global plotting style configuration for matplotlib and seaborn.

Provides a single place to standardize figure aesthetics across the
monthly forecasting project. Call ``set_global_plot_style()`` once early
in your program (e.g. in a script entrypoint or inside a dashboard/app
initialization) to apply consistent styling.

Features
--------
- Central rcParams (figure size, DPI, font sizes, line widths)
- Consistent seaborn theme + color palette
- Convenience context manager ``temporary_style`` to temporarily
  override global style inside a ``with`` block
- Idempotent application (safe to call multiple times)

The chosen defaults aim for publication-quality figures while remaining
legible in dashboards:
- Base font size 12 (adjust axes/title accordingly)
- Default figure size (width=10, height=6) -> good landscape aspect
- High DPI (150) good compromise between clarity & file size
- Tight layout improvements (``constrained_layout=True`` recommended when creating figures)

Usage
-----
>>> from dev_tools.visualization.style_config import set_global_plot_style
>>> set_global_plot_style()

Override selectively:
>>> set_global_plot_style(base_font_size=14, dpi=200)

Temporary overrides:
>>> from dev_tools.visualization.style_config import temporary_style
>>> with temporary_style({"font.size": 16, "figure.dpi": 220}):
...     fig, ax = plt.subplots()
...     ax.plot(x, y)

"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Default color palette (can be adjusted to project branding later)
DEFAULT_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
]

# Track whether we already applied style to avoid redundant logging or resets
_ALREADY_SET = False


def set_global_plot_style(
    *,
    width: float = 10.0,
    height: float = 6.0,
    dpi: int = 150,
    base_font_size: int = 14,
    font_family: str = "DejaVu Sans",
    palette: list[str] | None = None,
    grid: bool = True,
    tight_layout: bool = True,
    override_existing: bool = False,
) -> None:
    """Apply global matplotlib & seaborn styling.

    Args:
        width: Default figure width (inches) used when not explicitly provided
        height: Default figure height (inches)
        dpi: Figure resolution (dots per inch)
        base_font_size: Base font size; other sizes derive from it
        font_family: Primary font family
        palette: Optional custom categorical palette; falls back to DEFAULT_PALETTE
        grid: Whether to enable a subtle grid on major y-axis by default
        tight_layout: If True, set rcParams to improve layout spacing
        override_existing: If True, force reapplication even if already set
    """
    global _ALREADY_SET
    if _ALREADY_SET and not override_existing:
        return

    # Derive related font sizes (heuristic scaling)
    small = base_font_size - 2
    medium = base_font_size
    large = base_font_size + 2
    xlarge = base_font_size + 4

    rc_updates: Dict[str, Any] = {
        # Figure / save
        "figure.figsize": (width, height),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.autolayout": False,  # We'll manage spacing or rely on constrained_layout
        # Fonts
        "font.size": medium,
        "font.family": font_family,
        "axes.titlesize": xlarge,
        "axes.labelsize": large,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": medium,
        # Axes appearance
        "axes.grid": grid,
        "axes.grid.which": "major",
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        # Layout hints
        "figure.constrained_layout.use": tight_layout,
        # Misc
        "axes.formatter.use_mathtext": False,
        "axes.formatter.limits": (-4, 4),  # scientific notation thresholds
    }

    # draw boarders around the the whole figures
    rc_updates["axes.spines.left"] = True
    rc_updates["axes.spines.bottom"] = True
    rc_updates["axes.spines.right"] = True
    rc_updates["axes.spines.top"] = True

    mpl.rcParams.update(rc_updates)

    # Seaborn theme (context handles scaling for talk/paper if needed later)
    sns.set_theme(
        context="notebook",
        style="whitegrid" if grid else "white",
        font=font_family,
        rc={},  # Already applied above
    )

    sns.set_palette(palette or DEFAULT_PALETTE)

    _ALREADY_SET = True


def get_current_rc() -> Dict[str, Any]:
    """Return a shallow copy of current matplotlib rcParams of interest."""
    keys_of_interest = [
        "figure.figsize",
        "figure.dpi",
        "font.size",
        "font.family",
        "axes.titlesize",
        "axes.labelsize",
        "lines.linewidth",
        "legend.fontsize",
    ]
    return {k: mpl.rcParams[k] for k in keys_of_interest}


@contextmanager
def temporary_style(overrides: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily apply rcParam overrides inside a ``with`` block.

    Only the supplied keys are changed and then restored.

    Example:
        >>> with temporary_style({"figure.dpi": 300, "font.size": 16}):
        ...     fig, ax = plt.subplots()
        ...     ax.plot(range(10))

    Args:
        overrides: Mapping of rcParam keys to temporary values
    """
    # Store original values
    original = {k: mpl.rcParams.get(k) for k in overrides.keys()}
    try:
        mpl.rcParams.update(overrides)
        yield
    finally:
        mpl.rcParams.update(original)


__all__ = [
    "set_global_plot_style",
    "temporary_style",
    "get_current_rc",
    "DEFAULT_PALETTE",
]
