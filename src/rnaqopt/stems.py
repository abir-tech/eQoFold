"""Candidate stem enumeration and the structural graphs over stems.

Plan section 4.2.  A **stem** is a run of consecutive complementary base pairs
between positions ``i..i+L-1`` and ``j-L+1..j``.  Stems are the binary decision
variables of every model in this project: ``x_s = 1`` iff stem *s* is selected.

Three graphs are built over the candidate set:

``conflict``   two stems share a nucleotide -> mutually exclusive
``crossing``   two stems' pairs cross -> pseudoknot
``nesting``    stem *t* lies inside the loop closed by stem *s*

``|stems|`` is the problem size *n* and the x-axis of every scaling plot -- not
sequence length.  Both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .config import STEMS, StemConfig

#: Watson-Crick and wobble pairs, as ordered nucleotide tuples.
CANONICAL_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
)
WOBBLE_PAIRS: frozenset[tuple[str, str]] = frozenset({("G", "U"), ("U", "G")})


@dataclass(frozen=True, order=True)
class Stem:
    """A helix of ``length`` consecutive pairs, outermost pair ``(i, j)``.

    All indices are 0-based.  The pairs are ``(i+k, j-k)`` for
    ``k = 0 .. length-1``, so the innermost pair is
    ``(i+length-1, j-length+1)``.
    """

    i: int
    j: int
    length: int

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"stem length must be >= 1, got {self.length}")
        if self.i >= self.j:
            raise ValueError(f"stem must satisfy i < j, got ({self.i}, {self.j})")

    # -- geometry ----------------------------------------------------------

    @property
    def outer(self) -> tuple[int, int]:
        """Outermost (closing) base pair."""
        return (self.i, self.j)

    @property
    def inner(self) -> tuple[int, int]:
        """Innermost base pair -- the one that encloses the loop."""
        return (self.i + self.length - 1, self.j - self.length + 1)

    @cached_property
    def pairs(self) -> frozenset[tuple[int, int]]:
        """All base pairs belonging to this stem."""
        return frozenset(
            (self.i + k, self.j - k) for k in range(self.length)
        )

    @cached_property
    def positions(self) -> frozenset[int]:
        """Every nucleotide index occupied by this stem."""
        out: set[int] = set()
        for a, b in self.pairs:
            out.add(a)
            out.add(b)
        return frozenset(out)

    @property
    def loop_size(self) -> int:
        """Unpaired nucleotides enclosed by the innermost pair."""
        a, b = self.inner
        return b - a - 1

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"Stem({self.i}-{self.j}, L={self.length})"


# --------------------------------------------------------------------------
# Pair-level relations
# --------------------------------------------------------------------------


def pairs_cross(p: tuple[int, int], q: tuple[int, int]) -> bool:
    """True if base pairs ``p`` and ``q`` cross (form a pseudoknot)."""
    (a, b), (c, d) = p, q
    return a < c < b < d or c < a < d < b


def can_pair(x: str, y: str, allow_gu: bool = True) -> bool:
    """True if nucleotides ``x`` and ``y`` can form a canonical pair."""
    if not allow_gu and (x, y) in WOBBLE_PAIRS:
        return False
    return (x, y) in CANONICAL_PAIRS


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def enumerate_stems(
    sequence: str,
    cfg: StemConfig = STEMS,
    allow_gu: bool = True,
    min_hairpin: int = 3,
) -> list[Stem]:
    """Enumerate candidate stems, sorted and deduplicated.

    Maximal stems only by default (``cfg.include_substems`` adds truncations).
    A stem is *maximal* when it cannot be extended outward: ``(i-1, j+1)``
    cannot pair.  Extension inward stops when the enclosed loop would fall
    below ``min_hairpin`` unpaired nucleotides.

    The returned order is deterministic (sorted by ``(i, j, length)``), because
    stem index is the variable index of every model and must not depend on
    dictionary or set iteration order.
    """
    n = len(sequence)
    seen: set[Stem] = set()

    def pairable(a: int, b: int) -> bool:
        return (
            0 <= a < b < n
            and (b - a - 1) >= min_hairpin
            and can_pair(sequence[a], sequence[b], allow_gu)
        )

    for i in range(n):
        for j in range(i + min_hairpin + 1, n):
            if not pairable(i, j):
                continue
            # Keep only maximal stems: skip if this pair extends an outer pair.
            if pairable(i - 1, j + 1):
                continue
            # Extend inward as far as the geometry allows.
            length = 0
            while pairable(i + length, j - length):
                length += 1
            if length < cfg.min_stem_length:
                continue
            seen.add(Stem(i, j, length))
            if cfg.include_substems:
                # Every contiguous sub-run of length >= min_stem_length.
                for start in range(length):
                    for sub_len in range(cfg.min_stem_length, length - start + 1):
                        if (j - start) - (i + start) - 1 >= min_hairpin:
                            seen.add(Stem(i + start, j - start, sub_len))
    return sorted(seen)


# --------------------------------------------------------------------------
# Structural relations between stems
# --------------------------------------------------------------------------


def stems_conflict(s: Stem, t: Stem) -> bool:
    """True if two stems share at least one nucleotide."""
    return bool(s.positions & t.positions)


def stems_cross(s: Stem, t: Stem) -> bool:
    """True if any pair of *s* crosses any pair of *t* (a pseudoknot)."""
    return any(pairs_cross(p, q) for p in s.pairs for q in t.pairs)


def is_nested(outer: Stem, inner: Stem) -> bool:
    """True if ``inner`` lies strictly inside the loop closed by ``outer``.

    "Inside the loop" means inside the *innermost* pair of ``outer``, which is
    the pair that actually encloses the loop.
    """
    a, b = outer.inner
    return a < inner.i and inner.j < b


@dataclass(frozen=True)
class StemGraphs:
    """The structural relations over a candidate stem set."""

    stems: tuple[Stem, ...]
    conflict: frozenset[tuple[int, int]]
    crossing: frozenset[tuple[int, int]]
    nesting: frozenset[tuple[int, int]]

    @property
    def n(self) -> int:
        """Problem size: number of binary variables."""
        return len(self.stems)

    def compatible(self, a: int, b: int, pseudoknot_mode: bool = False) -> bool:
        """True if stems ``a`` and ``b`` may be selected together."""
        key = (a, b) if a < b else (b, a)
        if key in self.conflict:
            return False
        if not pseudoknot_mode and key in self.crossing:
            return False
        return True

    def nested_children(self, s: int) -> list[int]:
        """Candidate stems that could sit directly inside stem ``s``."""
        return sorted(t for (u, t) in self.nesting if u == s)

    def summary(self) -> dict[str, int]:
        return {
            "n_stems": self.n,
            "n_conflict": len(self.conflict),
            "n_crossing": len(self.crossing),
            "n_nesting": len(self.nesting),
        }


def build_graphs(stems: list[Stem]) -> StemGraphs:
    """Build the conflict, crossing and nesting relations.

    ``conflict`` and ``crossing`` are stored as unordered pairs ``(a, b)`` with
    ``a < b``; ``nesting`` is *directed*, ``(outer, inner)``.

    Note that conflict and crossing are disjoint by construction: two stems
    that share a nucleotide are already mutually exclusive, so classifying them
    as crossing as well would double-count the penalty.
    """
    conflict: set[tuple[int, int]] = set()
    crossing: set[tuple[int, int]] = set()
    nesting: set[tuple[int, int]] = set()

    for a in range(len(stems)):
        for b in range(a + 1, len(stems)):
            s, t = stems[a], stems[b]
            if stems_conflict(s, t):
                conflict.add((a, b))
                continue  # already mutually exclusive; do not also penalise crossing
            if stems_cross(s, t):
                crossing.add((a, b))
                continue  # crossing stems cannot be nested
            if is_nested(s, t):
                nesting.add((a, b))
            elif is_nested(t, s):
                nesting.add((b, a))

    return StemGraphs(
        stems=tuple(stems),
        conflict=frozenset(conflict),
        crossing=frozenset(crossing),
        nesting=frozenset(nesting),
    )


def enumerate_with_graphs(
    sequence: str,
    cfg: StemConfig = STEMS,
    allow_gu: bool = True,
    min_hairpin: int = 3,
) -> StemGraphs:
    """Convenience: enumerate stems and build all three graphs."""
    return build_graphs(enumerate_stems(sequence, cfg, allow_gu, min_hairpin))


def selected_stems_are_valid(
    graphs: StemGraphs,
    selection: list[int],
    pseudoknot_mode: bool = False,
) -> bool:
    """True if every pair in ``selection`` is mutually compatible."""
    return all(
        graphs.compatible(a, b, pseudoknot_mode)
        for idx, a in enumerate(selection)
        for b in selection[idx + 1 :]
    )
