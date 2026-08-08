"""ViennaRNA reference layer -- the answer key.

Plan section 2.1: ViennaRNA solves non-pseudoknotted MFE folding *exactly* in
O(n^3).  It is not a competitor, it is the ground truth.  Plan section 2.2
corollary: **every** predicted structure is finally scored by
``eval_structure`` on its dot-bracket string, whatever our internal model
believes its energy to be.

This module is the only place in the project allowed to construct a
``RNA.fold_compound``.  Everything routes through :func:`fold_compound` so the
frozen configuration of :mod:`rnaqopt.config` cannot be bypassed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import VIENNA, VIENNA_REFERENCE_CSV, VIENNA_STOCK, ViennaConfig, ensure_dirs
from .metrics import pairs_from_dotbracket
from .sequences import RNASequence

#: Gas constant in kcal/(mol*K).
_R_KCAL = 1.98717e-3

#: ViennaRNA reports energies quantised to 0.01 kcal/mol; its Python bindings
#: return single-precision artefacts (e.g. -1.2000000476837158).  Round on the
#: way out so results files are stable and diffable.
_ENERGY_DP = 2


def _round(x: float) -> float:
    return round(float(x), _ENERGY_DP)


def vienna_version() -> str:
    """Installed ViennaRNA version string."""
    import RNA

    return str(RNA.__version__)


def fold_compound(sequence: str, cfg: ViennaConfig = VIENNA):
    """Build a ``RNA.fold_compound`` under the frozen model details.

    A fresh compound is returned on every call.  ViennaRNA mutates internal
    state during ``pf()`` (energy rescaling), so reusing one compound across
    MFE and partition-function calls is a known source of subtle drift.
    """
    import RNA

    return RNA.fold_compound(sequence, cfg.model_details())


def mfe(sequence: str, cfg: ViennaConfig = VIENNA) -> tuple[str, float]:
    """Minimum free energy structure and its energy in kcal/mol."""
    structure, energy = fold_compound(sequence, cfg).mfe()
    return structure, _round(energy)


def eval_structure(sequence: str, structure: str, cfg: ViennaConfig = VIENNA) -> float:
    """Turner free energy of ``structure`` on ``sequence``, in kcal/mol.

    This is the scoring function for every predicted structure in the project,
    regardless of which solver or model level produced it.
    """
    if len(structure) != len(sequence):
        raise ValueError(
            f"structure length {len(structure)} != sequence length {len(sequence)}"
        )
    return _round(fold_compound(sequence, cfg).eval_structure(structure))


def partition_function(sequence: str, cfg: ViennaConfig = VIENNA) -> tuple[str, float]:
    """Ensemble representation and ensemble free energy in kcal/mol."""
    fc = fold_compound(sequence, cfg)
    fc.mfe()  # establishes the scaling factor used by pf()
    ens_structure, ens_energy = fc.pf()
    return ens_structure, _round(ens_energy)


def subopt(
    sequence: str,
    delta_kcal: float = 1.0,
    cfg: ViennaConfig = VIENNA,
) -> list[tuple[str, float]]:
    """All structures within ``delta_kcal`` of the MFE, sorted by energy.

    Needed for the degeneracy problem in plan section 6: several distinct
    structures often share the MFE, and scoring against a single returned
    structure understates accuracy.
    """
    delta_dacal = int(round(delta_kcal * 100))
    fc = fold_compound(sequence, cfg)
    sols = fc.subopt(delta_dacal)
    out = [(s.structure, _round(s.energy)) for s in sols if s.structure is not None]
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def rt_kcal(cfg: ViennaConfig = VIENNA) -> float:
    """Thermal energy RT in kcal/mol at the configured temperature."""
    return _R_KCAL * (cfg.temperature_c + 273.15)


# --------------------------------------------------------------------------
# Reference table
# --------------------------------------------------------------------------

#: Column order of ``vienna_reference.csv``.
REFERENCE_COLUMNS: tuple[str, ...] = (
    "seq_id",
    "tier",
    "source",
    "notes",
    "length",
    "gc_content",
    "sequence",
    "mfe_structure",
    "mfe_energy",
    "n_pairs",
    "n_helices",
    "max_helix_len",
    "has_multiloop",
    "ensemble_free_energy",
    "mfe_probability",
    "n_mfe_degenerate",
    "n_subopt_0.5",
    "n_subopt_1.0",
    "stock_mfe_structure",
    "stock_mfe_energy",
    "stock_bp_distance",
    "vienna_version",
    "config_fingerprint",
    "config",
)


@dataclass(frozen=True)
class ReferenceRecord:
    """One row of the reference table."""

    seq_id: str
    tier: str
    source: str
    notes: str
    length: int
    gc_content: float
    sequence: str
    mfe_structure: str
    mfe_energy: float
    n_pairs: int
    n_helices: int
    max_helix_len: int
    has_multiloop: bool
    ensemble_free_energy: float
    mfe_probability: float
    n_mfe_degenerate: int
    n_subopt_05: int
    n_subopt_10: int
    stock_mfe_structure: str
    stock_mfe_energy: float
    stock_bp_distance: int
    vienna_version: str
    config_fingerprint: str
    config: str

    def as_row(self) -> dict[str, object]:
        d = {
            "seq_id": self.seq_id,
            "tier": self.tier,
            "source": self.source,
            "notes": self.notes,
            "length": self.length,
            "gc_content": round(self.gc_content, 4),
            "sequence": self.sequence,
            "mfe_structure": self.mfe_structure,
            "mfe_energy": self.mfe_energy,
            "n_pairs": self.n_pairs,
            "n_helices": self.n_helices,
            "max_helix_len": self.max_helix_len,
            "has_multiloop": self.has_multiloop,
            "ensemble_free_energy": self.ensemble_free_energy,
            "mfe_probability": self.mfe_probability,
            "n_mfe_degenerate": self.n_mfe_degenerate,
            "n_subopt_0.5": self.n_subopt_05,
            "n_subopt_1.0": self.n_subopt_10,
            "stock_mfe_structure": self.stock_mfe_structure,
            "stock_mfe_energy": self.stock_mfe_energy,
            "stock_bp_distance": self.stock_bp_distance,
            "vienna_version": self.vienna_version,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
        }
        return {k: d[k] for k in REFERENCE_COLUMNS}


def helix_stats(structure: str) -> tuple[int, int]:
    """Return ``(n_helices, max_helix_len)`` for a dot-bracket structure.

    A helix is a maximal run of stacked pairs ``(i,j), (i+1,j-1), ...``.  This
    is a preview of the problem size that Phase 2's stem enumeration will face:
    the MFE structure's helices are a lower bound on the stems that must be
    representable.
    """
    pairs = pairs_from_dotbracket(structure)
    if not pairs:
        return 0, 0
    remaining = set(pairs)
    n_helices = 0
    longest = 0
    for i, j in sorted(pairs):
        if (i - 1, j + 1) in remaining:
            continue  # not the outermost pair of its helix
        n_helices += 1
        length = 0
        a, b = i, j
        while (a, b) in remaining:
            length += 1
            a, b = a + 1, b - 1
        longest = max(longest, length)
    return n_helices, longest


def has_multiloop(structure: str) -> bool:
    """True if the structure contains a multiloop (a loop with >=3 branches).

    Detected structurally: a pair that directly encloses two or more helices.
    Relevant because multiloops are exactly what the Level 2 cubic terms exist
    to model (plan section 4.3).
    """
    pairs = sorted(pairs_from_dotbracket(structure))
    if not pairs:
        return False
    pairset = set(pairs)

    def children(i: int, j: int) -> int:
        """Count helices directly nested inside the pair (i, j)."""
        count = 0
        k = i + 1
        while k < j:
            partner = next((b for a, b in pairset if a == k), None)
            if partner is not None and partner < j:
                count += 1
                k = partner + 1
            else:
                k += 1
        return count

    for i, j in pairs:
        if (i + 1, j - 1) in pairset:
            continue  # interior of a helix, not a loop-closing pair
        if children(i, j) >= 2:
            return True
    return False


def analyze(
    record: RNASequence,
    cfg: ViennaConfig = VIENNA,
    stock_cfg: ViennaConfig = VIENNA_STOCK,
) -> ReferenceRecord:
    """Compute the full reference row for one sequence."""
    seq = record.sequence

    structure, energy = mfe(seq, cfg)
    _, ens_energy = partition_function(seq, cfg)

    # Boltzmann probability of the MFE structure within the ensemble.
    prob = math.exp(-(energy - ens_energy) / rt_kcal(cfg))
    prob = min(1.0, max(0.0, prob))

    sub05 = subopt(seq, 0.5, cfg)
    sub10 = subopt(seq, 1.0, cfg)
    n_degenerate = sum(1 for _, e in sub05 if abs(e - energy) < 1e-9)

    stock_structure, stock_energy = mfe(seq, stock_cfg)
    stock_dist = len(
        pairs_from_dotbracket(structure) ^ pairs_from_dotbracket(stock_structure)
    )

    n_helices, max_helix = helix_stats(structure)

    return ReferenceRecord(
        seq_id=record.seq_id,
        tier=record.tier,
        source=record.source,
        notes=record.notes,
        length=record.length,
        gc_content=record.gc_content,
        sequence=seq,
        mfe_structure=structure,
        mfe_energy=energy,
        n_pairs=len(pairs_from_dotbracket(structure)),
        n_helices=n_helices,
        max_helix_len=max_helix,
        has_multiloop=has_multiloop(structure),
        ensemble_free_energy=ens_energy,
        mfe_probability=round(prob, 6),
        n_mfe_degenerate=n_degenerate,
        n_subopt_05=len(sub05),
        n_subopt_10=len(sub10),
        stock_mfe_structure=stock_structure,
        stock_mfe_energy=stock_energy,
        stock_bp_distance=stock_dist,
        vienna_version=vienna_version(),
        config_fingerprint=cfg.fingerprint(),
        config=cfg.header_line(),
    )


def build_reference_table(
    records: Iterable[RNASequence],
    cfg: ViennaConfig = VIENNA,
) -> list[dict[str, object]]:
    """Compute reference rows for a set of sequences."""
    return [analyze(r, cfg).as_row() for r in records]


def write_reference_csv(
    rows: Sequence[dict[str, object]],
    path: Path | str = VIENNA_REFERENCE_CSV,
) -> Path:
    """Write the reference table to CSV with a stable column order.

    Uses ``\\n`` line endings and no index so the file is byte-identical across
    platforms -- the exit criterion for Phase 1 is that ``make reference``
    reproduces this file from a clean clone.
    """
    import pandas as pd

    ensure_dirs()
    path = Path(path)
    df = pd.DataFrame(rows, columns=list(REFERENCE_COLUMNS))
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def read_reference_csv(path: Path | str = VIENNA_REFERENCE_CSV):
    """Load the reference table as a DataFrame."""
    import pandas as pd

    return pd.read_csv(path, keep_default_na=False, na_values=[""])
