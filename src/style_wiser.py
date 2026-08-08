import sys
from pathlib import Path

# The parent repository this track came from ships its own style module; prefer
# it when present so figures rebuilt there stay byte-identical. Falling back to
# src/style.py is what makes `make figures` work from a clean clone of this
# folder alone, and the two produce the same look.
REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_STYLE = REPO_ROOT / "figures" / "src"
if PARENT_STYLE.is_dir():
    sys.path.insert(0, str(PARENT_STYLE))

from style import panel_label, savefig, setup, wilson_ci  # noqa: F401,E402

METHOD_PALETTE = {
    "GQE": "#0F766E",          # teal
    "PCE-direct": "#4338CA",   # indigo
    "SimAnneal": "#B45309",    # amber
    "Tabu": "#B91C1C",         # red
    "Blind": "#6B7280",        # gray
}
ENCODING_PALETTE = {
    "pair": "#0F766E",
    "stem": "#4338CA",
}
WEIGHT_PALETTE = {
    "canonical": "#B45309",
    "vienna_calibrated": "#0F766E",
}
