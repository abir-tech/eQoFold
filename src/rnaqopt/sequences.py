"""Sequence generation, loading and validation.

Data policy (plan section 1.9): only public, synthetic, or randomly generated
sequences.  Every loading path in this module validates the alphabet and every
record carries an explicit ``source`` field, so a sequence with unknown
provenance cannot enter the pipeline silently.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from collections.abc import Sequence as TSequence
from dataclasses import dataclass
from pathlib import Path

from .config import RNA_ALPHABET, SEQUENCE_DIR

#: Recognised tier names.  ``A``/``B``/``C`` are the size tiers of plan section
#: 4.8; ``M`` is a special-purpose set added during Phase 1 (see below), in the
#: same spirit as the plan's separate pseudoknot set.
TIERS = ("A", "B", "C", "M")

#: The three size tiers, for scaling plots and size-stratified reporting.
SIZE_TIERS = ("A", "B", "C")

TIER_FILES = {
    "A": SEQUENCE_DIR / "tierA_small.fasta",
    "B": SEQUENCE_DIR / "tierB_medium.fasta",
    "C": SEQUENCE_DIR / "tierC_scaling.fasta",
    "M": SEQUENCE_DIR / "tierM_multiloop.fasta",
}

TIER_LENGTH_RANGE = {
    "A": (15, 25),
    "B": (30, 60),
    "C": (60, 120),
    "M": (30, 60),
}

#: Field separator inside a FASTA header. Values may contain ``=`` but not this.
_HEADER_SEP = "|"


class SequenceValidationError(ValueError):
    """Raised when a sequence violates the project data policy or alphabet."""


@dataclass(frozen=True)
class RNASequence:
    """A single validated RNA sequence with provenance."""

    seq_id: str
    sequence: str
    tier: str
    source: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.sequence:
            raise SequenceValidationError(f"{self.seq_id}: empty sequence")
        bad = sorted(set(self.sequence) - RNA_ALPHABET)
        if bad:
            raise SequenceValidationError(
                f"{self.seq_id}: illegal symbols {bad!r}; "
                f"only {''.join(sorted(RNA_ALPHABET))} permitted "
                "(convert DNA T->U before loading)"
            )
        if not self.source:
            raise SequenceValidationError(
                f"{self.seq_id}: missing provenance; plan section 1.9 requires an "
                "explicit source for every sequence"
            )
        for name, value in (
            ("seq_id", self.seq_id),
            ("tier", self.tier),
            ("source", self.source),
            ("notes", self.notes),
        ):
            if _HEADER_SEP in value:
                raise SequenceValidationError(
                    f"{self.seq_id}: {name} may not contain {_HEADER_SEP!r} "
                    "(it is the FASTA header field separator)"
                )
            if ">" in value:
                raise SequenceValidationError(
                    f"{self.seq_id}: {name} may not contain '>' "
                    "(it would break naive FASTA parsers)"
                )

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        """Fraction of G or C nucleotides."""
        return (self.sequence.count("G") + self.sequence.count("C")) / len(self.sequence)

    def header(self) -> str:
        """FASTA header encoding tier, source and notes as key=value fields."""
        parts = [self.seq_id, f"tier={self.tier}", f"source={self.source}"]
        if self.notes:
            parts.append(f"notes={self.notes}")
        return " | ".join(parts)


# --------------------------------------------------------------------------
# FASTA I/O
# --------------------------------------------------------------------------


def _parse_header(header: str) -> tuple[str, dict[str, str]]:
    """Split ``id | key=value | key=value`` into an id and a field dict.

    Fields are separated by ``|`` and each is split on its *first* ``=`` only,
    so a value may itself contain ``=`` (``notes=len=8 gc_target=0.50``).
    """
    parts = [p.strip() for p in header.split("|")]
    seq_id = parts[0]
    fields: dict[str, str] = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return seq_id, fields


def read_fasta(path: Path | str, default_tier: str = "") -> list[RNASequence]:
    """Read a FASTA file into validated :class:`RNASequence` records."""
    path = Path(path)
    records: list[RNASequence] = []
    seq_id: str | None = None
    fields: dict[str, str] = {}
    chunks: list[str] = []

    def flush() -> None:
        if seq_id is None:
            return
        records.append(
            RNASequence(
                seq_id=seq_id,
                sequence="".join(chunks).upper().replace("T", "U"),
                tier=fields.get("tier", default_tier),
                source=fields.get("source", ""),
                notes=fields.get("notes", ""),
            )
        )

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith((";", "#")):
                continue
            if line.startswith(">"):
                flush()
                seq_id, fields = _parse_header(line[1:])
                chunks = []
            else:
                chunks.append(line)
    flush()

    ids = [r.seq_id for r in records]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SequenceValidationError(f"{path}: duplicate ids {sorted(dupes)}")
    return records


def write_fasta(records: Iterable[RNASequence], path: Path | str, width: int = 60) -> None:
    """Write records to FASTA, wrapping sequence lines at ``width``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(f">{rec.header()}\n")
            for i in range(0, len(rec.sequence), width):
                fh.write(rec.sequence[i : i + width] + "\n")


def load_tier(tier: str) -> list[RNASequence]:
    """Load one tier by name (``"A"``, ``"B"`` or ``"C"``)."""
    tier = tier.upper()
    if tier not in TIER_FILES:
        raise KeyError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return read_fasta(TIER_FILES[tier], default_tier=tier)


def load_all(tiers: TSequence[str] = TIERS) -> list[RNASequence]:
    """Load several tiers in order."""
    out: list[RNASequence] = []
    for t in tiers:
        out.extend(load_tier(t))
    return out


# --------------------------------------------------------------------------
# Deterministic synthetic generation
# --------------------------------------------------------------------------

_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def random_sequence(rng: random.Random, length: int, gc_target: float = 0.5) -> str:
    """Uniform random sequence with an expected GC fraction of ``gc_target``."""
    gc = ("G", "C")
    au = ("A", "U")
    return "".join(
        rng.choice(gc) if rng.random() < gc_target else rng.choice(au)
        for _ in range(length)
    )


def designed_hairpin(
    rng: random.Random,
    stem_length: int,
    loop_length: int,
    flank5: int = 0,
    flank3: int = 0,
) -> str:
    """Build a sequence whose intended fold is a single hairpin.

    Used for Tier A so that at least some instances have a known, hand-checkable
    ground-truth structure rather than whatever a random sequence happens to fold
    into.
    """
    if loop_length < 3:
        raise ValueError("hairpin loop must be at least 3 nt")
    stem5 = "".join(rng.choice("ACGU") for _ in range(stem_length))
    stem3 = "".join(_COMPLEMENT[c] for c in reversed(stem5))
    # Loops rich in A/U discourage the loop from pairing with itself.
    loop = "".join(rng.choice("AAAUUC") for _ in range(loop_length))
    left = "".join(rng.choice("ACGU") for _ in range(flank5))
    right = "".join(rng.choice("ACGU") for _ in range(flank3))
    return left + stem5 + loop + stem3 + right


def designed_multiloop(
    rng: random.Random,
    closing_len: int,
    n_branches: int,
    branch_len: int,
    loop_len: int,
    spacer: int,
    gc_bias: float = 0.9,
) -> str:
    """Build a sequence intended to fold into a multi-branch junction.

    Layout::

        [closing stem][spacer]([branch hairpin][spacer])*n[closing stem']

    Stems are GC-biased and loops/spacers are A-rich, because a multiloop pays
    a large Turner closing penalty (``a`` ~ +9.3 kcal/mol) that weak stems
    cannot overcome.  A designed sequence is only a *candidate*: whether the
    junction is actually the MFE is decided by ViennaRNA in
    :func:`mfe_has_multiloop`, and only accepted candidates enter the tier.
    """

    def stem(n: int) -> str:
        return "".join(
            rng.choice("GC") if rng.random() < gc_bias else rng.choice("AU")
            for _ in range(n)
        )

    closing5 = stem(closing_len)
    parts = ["A" * spacer]
    for _ in range(n_branches):
        b5 = stem(branch_len)
        loop = "".join(rng.choice("AAAUC") for _ in range(loop_len))
        parts.append(b5 + loop + "".join(_COMPLEMENT[c] for c in reversed(b5)))
        parts.append("A" * spacer)
    closing3 = "".join(_COMPLEMENT[c] for c in reversed(closing5))
    return closing5 + "".join(parts) + closing3


# --------------------------------------------------------------------------
# Acceptance screens
# --------------------------------------------------------------------------
#
# Random sequences at these lengths frequently fold into nothing at all: in the
# first unscreened draw, 7 of 50 sequences had an empty MFE structure.  An
# empty reference is a degenerate benchmark -- every model "solves" it by
# selecting no stems -- so it consumes a slot without contributing to the
# encoding-gap measurement.
#
# The screens below select on *non-degeneracy of the instance*, never on
# whether our model or solver does well on it.  That distinction is what keeps
# the benchmark honest, and it is restated in docs/ASSUMPTIONS.md.


def mfe_helix_count(sequence: str) -> int:
    """Number of helices in the ViennaRNA MFE structure of ``sequence``."""
    from .reference import helix_stats, mfe  # local: avoids an import cycle

    structure, _ = mfe(sequence)
    return helix_stats(structure)[0]


def mfe_has_multiloop(sequence: str) -> bool:
    """True if the ViennaRNA MFE structure of ``sequence`` contains a multiloop."""
    from .reference import has_multiloop, mfe  # local: avoids an import cycle

    structure, _ = mfe(sequence)
    return has_multiloop(structure)


def _accept(
    make: Callable[[], str],
    predicate: Callable[[str], bool],
    label: str,
    max_attempts: int = 4000,
) -> str:
    """Rejection-sample a candidate until ``predicate`` accepts it.

    Deterministic given the caller's RNG state.  Raises rather than silently
    returning a degenerate sequence, so a screen that becomes unsatisfiable
    fails loudly instead of quietly weakening the benchmark.
    """
    for _ in range(max_attempts):
        candidate = make()
        if predicate(candidate):
            return candidate
    raise SequenceValidationError(
        f"{label}: no candidate satisfied the acceptance screen in "
        f"{max_attempts} attempts"
    )


#: Designed junction geometries for tier M, spanning 34-58 nt.
#: ``(closing_len, n_branches, branch_len, loop_len, spacer)``.
#: Chosen from a measured sweep: nothing below 31 nt ever folds into a
#: multiloop, and these configurations had the highest acceptance rates.
_MULTILOOP_DESIGNS: tuple[tuple[int, int, int, int, int], ...] = (
    (5, 2, 4, 4, 0),  # 34 nt, 3-way junction
    (4, 2, 4, 4, 1),  # 35 nt
    (4, 2, 5, 4, 0),  # 36 nt
    (5, 2, 4, 4, 1),  # 37 nt
    (4, 2, 5, 4, 1),  # 39 nt
    (5, 2, 5, 4, 1),  # 41 nt
    (5, 2, 5, 4, 2),  # 44 nt
    (4, 3, 4, 4, 1),  # 48 nt, 4-way junction
    (5, 3, 4, 4, 2),  # 54 nt
    (4, 3, 5, 4, 2),  # 58 nt
)


def generate_tier(tier: str, seed: int) -> list[RNASequence]:
    """Deterministically generate one tier.

    Given the same ``seed`` this returns byte-identical sequences on every
    platform: it uses only :class:`random.Random`, whose Mersenne Twister stream
    is stable across CPython versions.
    """
    tier = tier.upper()
    rng = random.Random(seed)
    lo, hi = TIER_LENGTH_RANGE[tier]
    out: list[RNASequence] = []

    if tier == "A":
        # 8 designed hairpins with hand-predictable folds.  Stem length is
        # derived from a target total length so every instance lands inside the
        # tier range; flanks are kept to 0-2 nt so they cannot easily nucleate a
        # competing helix and obscure the intended fold.
        for k in range(8):
            target = lo + k  # 15..22
            flank5 = k % 2
            flank3 = (k + 1) % 2
            loop_len = 4 + (k % 3)
            stem_len = (target - flank5 - flank3 - loop_len) // 2
            # Any odd nucleotide left over goes to the 3' flank.
            flank3 += target - (2 * stem_len + loop_len + flank5 + flank3)
            seq = designed_hairpin(rng, stem_len, loop_len, flank5, flank3)
            out.append(
                RNASequence(
                    seq_id=f"A_hp{k + 1:02d}",
                    sequence=seq,
                    tier="A",
                    source="synthetic:designed-hairpin",
                    notes=f"stem={stem_len} loop={loop_len} flank={flank5}/{flank3}",
                )
            )
        # ... and 12 randoms spanning the GC range, for unbiased statistics.
        # Screened to fold into at least one helix.
        for k in range(12):
            length = lo + (k * (hi - lo)) // 11
            gc = 0.30 + 0.40 * (k % 4) / 3.0
            seq = _accept(
                lambda ln=length, g=gc: random_sequence(rng, ln, g),
                lambda s: mfe_helix_count(s) >= 1,
                label=f"A_rand{k + 1:02d}",
            )
            out.append(
                RNASequence(
                    seq_id=f"A_rand{k + 1:02d}",
                    sequence=seq,
                    tier="A",
                    source="synthetic:random",
                    notes=f"len={length} gc_target={gc:.2f} screen=helices_ge_1",
                )
            )

    elif tier == "B":
        for k in range(20):
            length = lo + (k * (hi - lo)) // 19
            gc = 0.30 + 0.40 * (k % 5) / 4.0
            seq = _accept(
                lambda ln=length, g=gc: random_sequence(rng, ln, g),
                lambda s: mfe_helix_count(s) >= 1,
                label=f"B_rand{k + 1:02d}",
            )
            out.append(
                RNASequence(
                    seq_id=f"B_rand{k + 1:02d}",
                    sequence=seq,
                    tier="B",
                    source="synthetic:random",
                    notes=f"len={length} gc_target={gc:.2f} screen=helices_ge_1",
                )
            )

    elif tier == "C":
        for k in range(10):
            length = lo + (k * (hi - lo)) // 9
            gc = 0.35 + 0.30 * (k % 4) / 3.0
            seq = _accept(
                lambda ln=length, g=gc: random_sequence(rng, ln, g),
                lambda s: mfe_helix_count(s) >= 2,
                label=f"C_rand{k + 1:02d}",
            )
            out.append(
                RNASequence(
                    seq_id=f"C_rand{k + 1:02d}",
                    sequence=seq,
                    tier="C",
                    source="synthetic:random",
                    notes=f"len={length} gc_target={gc:.2f} screen=helices_ge_2",
                )
            )

    elif tier == "M":
        # Designed multi-branch junctions, accepted only when ViennaRNA agrees
        # the MFE actually contains a multiloop.  Exists because Tier A cannot
        # host one: a measured sweep found no multiloop MFE below 31 nt at any
        # stem design or GC content, so the Level 2 cubic terms (which model
        # exactly the multiloop branch penalty) have nothing to act on in the
        # brute-forceable size tier.  See docs/ASSUMPTIONS.md.
        for k, (cl, nb, bl, ll, sp) in enumerate(_MULTILOOP_DESIGNS):
            seq = _accept(
                lambda c=cl, n=nb, b=bl, lp=ll, s=sp: designed_multiloop(
                    rng, c, n, b, lp, s
                ),
                mfe_has_multiloop,
                label=f"M_ml{k + 1:02d}",
            )
            out.append(
                RNASequence(
                    seq_id=f"M_ml{k + 1:02d}",
                    sequence=seq,
                    tier="M",
                    source="synthetic:designed-multiloop",
                    notes=(
                        f"branches={nb + 1} closing={cl} branch_stem={bl} "
                        f"loop={ll} spacer={sp} screen=multiloop"
                    ),
                )
            )
    else:
        raise KeyError(f"unknown tier {tier!r}")

    for rec in out:
        if not lo <= rec.length <= hi:
            raise SequenceValidationError(
                f"{rec.seq_id}: length {rec.length} outside tier {tier} range {lo}-{hi}"
            )
    return out


def iter_tier_files() -> Iterator[tuple[str, Path]]:
    """Yield ``(tier, path)`` for each tier FASTA."""
    for tier in TIERS:
        yield tier, TIER_FILES[tier]
