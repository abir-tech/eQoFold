"""Figure styling shared by every plotting script in this repository.

This is a self-contained copy of the style module the circuit-search track
originally imported from its parent repository (`figures/src/style.py` there),
kept here so that `make figures` works from a clean clone of this folder alone.
`style_wiser` prefers the parent module when it is on the path and falls back to
this one otherwise, so figures rebuilt either way look the same.

Conventions: serif type at print sizes, no top or right spine, bold lowercase
panel labels outside the axes, and one PDF plus one PNG preview per figure.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

#: Semantic colors. Reused by the paper figures so that a hue means the same
#: thing in every panel of the manuscript.
PALETTE = {
    "reference": "#0F766E",   # the classical answer key, and the pair encoding
    "generated": "#4338CA",   # anything a trained model produced
    "decoder": "#B45309",     # encode / decode / repair stages
    "hardware": "#6B7280",    # device and resource annotations
}


def setup() -> None:
    """Apply the shared rcParams. Call once, before building any figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def panel_label(ax, text: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Bold "(a)" style label placed outside the top-left corner of `ax`."""
    ax.text(dx, dy, f"({text})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")


def savefig(fig, path) -> None:
    """Write `fig` to `path` (a .pdf) plus a PNG preview beside it.

    The preview lives in a `_previews/` subdirectory so that the PDFs stay the
    only thing the manuscript references, while the PNGs remain available for
    the slide deck and for reading the figures without a PDF viewer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = path.parent / "_previews"
    preview_dir.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(preview_dir / f"{path.stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Preferred over the normal approximation because the rates reported here come
    from small samples and sit near 0 or 1, where the normal interval leaves the
    unit interval entirely.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
