"""Structural and energetic metrics, and the two-gap error decomposition.

Plan section 4.7 asks for base-pair sensitivity, PPV and F1 *reported
separately* -- F1 alone hides the sensitivity/precision trade-off -- plus
base-pair distance, exact-match, and the section 2.2 decomposition of total
error into an **encoding gap** and an **optimizer gap**.

All positions are 0-based half-open indices into the sequence.  Base pairs are
stored as ``(i, j)`` tuples with ``i < j``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

#: Bracket pairs understood by the dot-bracket parser.  The extended alphabet is
#: needed for pseudoknot mode (plan section 4.4).
BRACKETS: dict[str, str] = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
}
_OPENERS = frozenset(BRACKETS.values())
_CLOSERS = frozenset(BRACKETS)
UNPAIRED = frozenset(".:-_,~")


class DotBracketError(ValueError):
    """Raised for malformed dot-bracket strings."""


# --------------------------------------------------------------------------
# Dot-bracket <-> base-pair set
# --------------------------------------------------------------------------


def pairs_from_dotbracket(db: str) -> set[tuple[int, int]]:
    """Parse a dot-bracket string into a set of 0-based ``(i, j)`` pairs.

    Supports the extended alphabet ``()[]{}<>`` so pseudoknotted structures
    round-trip.  Raises :class:`DotBracketError` on unbalanced input.
    """
    stacks: dict[str, list[int]] = {o: [] for o in _OPENERS}
    pairs: set[tuple[int, int]] = set()
    for pos, ch in enumerate(db):
        if ch in _OPENERS:
            stacks[ch].append(pos)
        elif ch in _CLOSERS:
            opener = BRACKETS[ch]
            if not stacks[opener]:
                raise DotBracketError(
                    f"unbalanced {ch!r} at position {pos} in {db!r}"
                )
            pairs.add((stacks[opener].pop(), pos))
        elif ch not in UNPAIRED:
            raise DotBracketError(f"illegal character {ch!r} at position {pos}")
    leftover = {o: s for o, s in stacks.items() if s}
    if leftover:
        raise DotBracketError(f"unclosed brackets {leftover} in {db!r}")
    return pairs


def dotbracket_from_pairs(pairs: Iterable[tuple[int, int]], length: int) -> str:
    """Render a base-pair set as dot-bracket.

    Crossing pairs are assigned to successive bracket classes, so pseudoknotted
    structures render correctly.  Raises :class:`DotBracketError` if the pairs
    need more classes than the alphabet provides, or if a position is used
    twice.
    """
    ordered = sorted(pairs)
    seen: set[int] = set()
    for i, j in ordered:
        if not 0 <= i < j < length:
            raise DotBracketError(f"pair ({i},{j}) outside sequence of length {length}")
        if i in seen or j in seen:
            raise DotBracketError(f"position reused in pair ({i},{j})")
        seen.update((i, j))

    classes: list[list[tuple[int, int]]] = []
    for pair in ordered:
        for group in classes:
            if not any(_crosses(pair, other) for other in group):
                group.append(pair)
                break
        else:
            classes.append([pair])

    alphabet = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    if len(classes) > len(alphabet):
        raise DotBracketError(
            f"structure needs {len(classes)} bracket classes, "
            f"only {len(alphabet)} available"
        )

    out = ["."] * length
    for group, (op, cl) in zip(classes, alphabet, strict=False):
        for i, j in group:
            out[i] = op
            out[j] = cl
    return "".join(out)


def _crosses(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if pairs ``a`` and ``b`` cross (i.e. form a pseudoknot)."""
    (i, j), (k, m) = a, b
    return i < k < j < m or k < i < m < j


def has_crossings(pairs: Iterable[tuple[int, int]]) -> bool:
    """True if any two pairs cross."""
    plist = list(pairs)
    return any(
        _crosses(plist[a], plist[b])
        for a in range(len(plist))
        for b in range(a + 1, len(plist))
    )


def is_valid_dotbracket(db: str, length: int | None = None) -> bool:
    """True if ``db`` parses and (optionally) has the expected length."""
    try:
        pairs_from_dotbracket(db)
    except DotBracketError:
        return False
    return length is None or len(db) == length


# --------------------------------------------------------------------------
# Structural accuracy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureMetrics:
    """Base-pair accuracy of a predicted structure against a reference."""

    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    ppv: float
    f1: float
    bp_distance: int
    exact_match: bool
    n_pairs_pred: int
    n_pairs_ref: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def compare_structures(predicted: str, reference: str) -> StructureMetrics:
    """Compare two dot-bracket strings.

    ``sensitivity`` and ``ppv`` are reported separately by design: a prediction
    that emits very few pairs scores high PPV and low sensitivity, and F1 alone
    would hide that.

    Edge-case convention: when the reference is unfolded (no pairs) sensitivity
    is defined as 1.0; when the prediction is unfolded PPV is defined as 1.0.
    This makes an unfolded-vs-unfolded comparison score a perfect 1.0 rather
    than 0/0.
    """
    if len(predicted) != len(reference):
        raise DotBracketError(
            f"length mismatch: predicted {len(predicted)} vs reference {len(reference)}"
        )
    pred = pairs_from_dotbracket(predicted)
    ref = pairs_from_dotbracket(reference)

    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)

    sensitivity = tp / len(ref) if ref else 1.0
    ppv = tp / len(pred) if pred else 1.0
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if (sensitivity + ppv) > 0
        else 0.0
    )

    return StructureMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        sensitivity=sensitivity,
        ppv=ppv,
        f1=f1,
        bp_distance=fp + fn,
        exact_match=pred == ref,
        n_pairs_pred=len(pred),
        n_pairs_ref=len(ref),
    )


def matches_any(
    predicted: str,
    candidates: Sequence[str],
) -> bool:
    """True if ``predicted`` equals any structure in ``candidates`` pairwise.

    Used against ViennaRNA's suboptimal set to handle degenerate optima (plan
    section 6): several distinct structures can share the MFE, and scoring only
    against the single returned MFE structure understates accuracy.
    """
    pred = pairs_from_dotbracket(predicted)
    return any(pred == pairs_from_dotbracket(c) for c in candidates)


# --------------------------------------------------------------------------
# The two-gap decomposition (plan section 2.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GapDecomposition:
    """Total energy error split into encoding error and optimizer error.

    All energies are ViennaRNA-evaluated (kcal/mol) except
    :attr:`optimizer_gap_model`, which is in the internal model's own units.
    Per the plan's corollary rule (section 2.2), the *reported* numbers are
    always the ViennaRNA-evaluated ones; internal model energy is a diagnostic.

    The identity ``total_gap = encoding_gap + optimizer_gap`` holds by
    construction and is checked in :meth:`validate`.
    """

    #: ViennaRNA MFE for the sequence.
    e_vienna_mfe: float
    #: ViennaRNA-evaluated energy of the structure decoded from the model's
    #: *exact* optimum (brute force / Gurobi).
    e_model_optimum: float
    #: ViennaRNA-evaluated energy of the structure the solver actually returned.
    e_solver: float
    #: Solver's model energy minus the model's optimal energy, in model units.
    optimizer_gap_model: float | None = None

    @property
    def encoding_gap(self) -> float:
        """How much fidelity the *model* loses relative to Turner physics."""
        return self.e_model_optimum - self.e_vienna_mfe

    @property
    def optimizer_gap(self) -> float:
        """How much the *solver* loses relative to its own model's optimum."""
        return self.e_solver - self.e_model_optimum

    @property
    def total_gap(self) -> float:
        """End-to-end error against the answer key."""
        return self.e_solver - self.e_vienna_mfe

    def validate(self, tol: float = 1e-6) -> None:
        """Assert the additive identity holds."""
        residual = self.total_gap - (self.encoding_gap + self.optimizer_gap)
        if abs(residual) > tol:  # pragma: no cover - arithmetic identity
            raise AssertionError(f"gap decomposition inconsistent: residual={residual}")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "e_vienna_mfe": self.e_vienna_mfe,
            "e_model_optimum": self.e_model_optimum,
            "e_solver": self.e_solver,
            "encoding_gap": self.encoding_gap,
            "optimizer_gap": self.optimizer_gap,
            "total_gap": self.total_gap,
            "optimizer_gap_model": self.optimizer_gap_model,
        }


def summarize(rows: Iterable[Mapping[str, float]], keys: Sequence[str]) -> dict[str, float]:
    """Mean of each key over ``rows``, skipping missing/None values.

    Deliberately minimal: aggregation policy belongs to the experiment layer,
    not to the metric definitions.
    """
    rows = list(rows)
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out
