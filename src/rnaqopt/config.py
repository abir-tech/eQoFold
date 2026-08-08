"""Frozen ViennaRNA model details and global project settings.

Project plan section 4.1: *energy comparisons are meaningless unless the model
details match across the whole project.*  Every setting that can change a free
energy lives here, is frozen at import time, and is stamped into every results
file through :func:`ViennaConfig.fingerprint`.

Nothing in this project may build a ``fold_compound`` by hand.  Use
:func:`ViennaConfig.model_details` (or ``rnaqopt.reference.fold_compound``) so
that the reference MFE, the loop energies extracted for the model ladder, and
the final ``eval_structure`` scoring all agree by construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Repository paths
# --------------------------------------------------------------------------

#: Repository root (``src/rnaqopt/config.py`` -> ``src/rnaqopt`` -> ``src`` -> root).
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
SEQUENCE_DIR = DATA_DIR / "sequences"
REFERENCE_DIR = DATA_DIR / "references"
RESULTS_DIR = REPO_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLE_DIR = RESULTS_DIR / "tables"

#: All generated figures land in one directory, shared with the paper figures
#: built by ``experiments/make_figures.py``.
FIGURE_DIR = REPO_ROOT / "figures"

#: Canonical reference table produced by ``make reference``.
VIENNA_REFERENCE_CSV = REFERENCE_DIR / "vienna_reference.csv"


def ensure_dirs() -> None:
    """Create the generated-output directories if they do not exist."""
    for d in (REFERENCE_DIR, RAW_RESULTS_DIR, TABLE_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# ViennaRNA model details
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ViennaConfig:
    """Immutable ViennaRNA model-detail settings.

    The defaults are the project-wide frozen configuration.  See
    ``docs/ASSUMPTIONS.md`` for the justification of each choice, in particular
    ``dangles=0``.
    """

    #: Folding temperature in degrees Celsius.
    temperature_c: float = 37.0

    #: Dangling-end treatment.  ``0`` = no dangling-end contributions.
    #:
    #: Plan section 4.1 permits ``2`` (ViennaRNA default) or ``0``.  We take
    #: ``0``: our stem-variable model has no representation of a dangling
    #: nucleotide, so ``dangles=2`` would bake a constant, irreducible encoding
    #: gap into every rung of the fidelity ladder -- noise that has nothing to
    #: do with the hypothesis under test (section 2.2).  The ``dangles=2``
    #: reference is still computed and reported as a secondary column so that
    #: our numbers can be tied back to stock ViennaRNA output.
    dangles: int = 0

    #: Forbid lonely (isolated, unstacked) base pairs.
    #:
    #: With ``L_min = 3`` a stem-based encoding cannot express a lonely pair at
    #: all, so allowing them in the reference would again be a pure encoding
    #: penalty unrelated to the model ladder.
    no_lonely_pairs: bool = True

    #: Allow G-U wobble pairs.  Disabling is offered as a documented ablation.
    allow_gu: bool = True

    #: Allow G-U pairs at the end of a helix.
    allow_gu_closure: bool = True

    #: Use tabulated energies for special (tetra/tri/hexa-loop) hairpins.
    special_hairpins: bool = True

    #: Minimum number of unpaired nucleotides in a hairpin loop.
    min_hairpin_loop: int = 3

    #: Maximum base-pair span; ``-1`` means unlimited.
    max_bp_span: int = -1

    #: Nearest-neighbour parameter set.  ViennaRNA 2.x ships Turner 2004 as its
    #: compiled-in default; we additionally load it explicitly when the API is
    #: available so the choice survives a change of upstream default.
    param_set: str = "Turner2004"

    # -- derived -----------------------------------------------------------

    def model_details(self) -> Any:
        """Return a configured ``RNA.md`` instance.

        Import of ``RNA`` is deferred so that this module stays importable (for
        path constants and for documentation builds) without ViennaRNA present.
        """
        import RNA

        if self.param_set == "Turner2004":
            loader = getattr(RNA, "params_load_RNA_Turner2004", None)
            if loader is not None:  # pragma: no branch - present in 2.6+
                loader()
        else:  # pragma: no cover - single supported parameter set today
            raise ValueError(f"unsupported parameter set: {self.param_set!r}")

        md = RNA.md()
        md.temperature = float(self.temperature_c)
        md.dangles = int(self.dangles)
        md.noLP = 1 if self.no_lonely_pairs else 0
        md.noGU = 0 if self.allow_gu else 1
        md.noGUclosure = 0 if self.allow_gu_closure else 1
        md.special_hp = 1 if self.special_hairpins else 0
        md.max_bp_span = int(self.max_bp_span)

        # Attribute name for the minimum hairpin size moved across releases.
        for attr in ("min_loop_size", "minLoopSize"):
            if hasattr(md, attr):
                setattr(md, attr, int(self.min_hairpin_loop))
                break

        return md

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, used for CSV stamping and hashing."""
        return asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of the settings.

        Stamped into every generated CSV.  If two result files carry different
        fingerprints their energies are not comparable, and any table mixing
        them is invalid.
        """
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def header_line(self) -> str:
        """One-line human-readable summary for table headers and figure captions."""
        gu = "GU" if self.allow_gu else "noGU"
        return (
            f"ViennaRNA {self.param_set} | T={self.temperature_c:g}C | "
            f"dangles={self.dangles} | noLP={int(self.no_lonely_pairs)} | "
            f"{gu} | minHP={self.min_hairpin_loop} | fp={self.fingerprint()}"
        )


@dataclass(frozen=True)
class StemConfig:
    """Candidate-stem enumeration settings (plan section 4.2).

    Consumed from Phase 2 onward; kept here so that every knob that can change a
    reported number lives in one file.
    """

    #: Minimum stem length.
    #:
    #: Kept at 3.  The measured ablation
    #: (``results/tables/enumeration_ablation.csv``) shows ``L_min = 2`` is
    #: actively harmful: on set M it raises the mean encoding gap from 1.58 to
    #: 3.57 kcal/mol *while* tripling the variable count.  Extra weak
    #: candidates give the model's own energy errors more ways to express
    #: themselves, so representability and accuracy pull in opposite
    #: directions.
    min_stem_length: int = 3

    #: Emit truncated sub-stems of length >= ``min_stem_length`` as separate
    #: variables.
    #:
    #: **On**, following the plan's own decision rule (section 4.2: "start
    #: without sub-stems; add them only if the encoding gap analysis shows they
    #: matter").  The analysis shows they matter: on set M sub-stems take
    #: reference representability from 0.80 to 1.00 and cut the mean encoding
    #: gap from 1.58 to 0.57 kcal/mol, for 2.2x the variables.  Without them
    #: part of the measured "encoding gap" is really a *representability*
    #: failure -- the reference structure using 3 pairs of a maximal 4-pair
    #: stem -- which is not what the fidelity ladder is meant to be measuring.
    include_substems: bool = True

    #: Drop the crossing penalty -> pseudoknot-capable folding (plan section 2.6).
    pseudoknot_mode: bool = False


# --------------------------------------------------------------------------
# The frozen project-wide instances
# --------------------------------------------------------------------------

#: The one configuration used for every energy in this project.
VIENNA = ViennaConfig()

#: The secondary reference configuration, reported alongside for comparability
#: with stock ViennaRNA output.  Never used for model fitting.
VIENNA_STOCK = ViennaConfig(dangles=2, no_lonely_pairs=False)

#: Default stem-enumeration settings.
STEMS = StemConfig()

#: Global RNG seed.  Every stochastic step derives its seed from this.
GLOBAL_SEED = 20260807

#: Alphabet accepted by every data-loading path.
RNA_ALPHABET = frozenset("ACGU")
