"""Bitstring -> stem set -> dot-bracket, with deterministic repair.

Plan section 4.4.  A solver returns a bitstring over stem variables; that
bitstring may select mutually incompatible stems (penalties are soft).  This
module maps it to a dot-bracket string, repairing if necessary.

**Repair is always reported separately.**  It is a classical post-processing
step, and folding it silently into the result would inflate apparent solver
quality -- explicitly called out as a risk in plan section 6.  Every
:class:`DecodeResult` therefore carries both the raw feasibility verdict and
what repair had to remove.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import dotbracket_from_pairs
from .stems import Stem, StemGraphs


@dataclass(frozen=True)
class DecodeResult:
    """The outcome of turning one bitstring into a structure."""

    #: Stem indices the solver actually selected.
    selected_raw: tuple[int, ...]
    #: Stem indices surviving repair (equal to ``selected_raw`` when feasible).
    selected: tuple[int, ...]
    #: Dot-bracket string of the repaired selection.
    structure: str
    #: True if ``selected_raw`` was already pairwise compatible.
    feasible_raw: bool
    #: Stems dropped by repair.
    removed: tuple[int, ...]

    @property
    def was_repaired(self) -> bool:
        return bool(self.removed)

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_selected_raw": len(self.selected_raw),
            "n_selected": len(self.selected),
            "feasible_raw": self.feasible_raw,
            "was_repaired": self.was_repaired,
            "n_removed": self.n_removed,
            "structure": self.structure,
        }


def bits_to_selection(bits: Sequence[int]) -> tuple[int, ...]:
    """Indices where ``bits`` is truthy."""
    return tuple(i for i, b in enumerate(bits) if b)


def selection_to_pairs(
    graphs: StemGraphs, selection: Sequence[int]
) -> set[tuple[int, int]]:
    """Union of the base pairs of the selected stems."""
    pairs: set[tuple[int, int]] = set()
    for idx in selection:
        pairs |= graphs.stems[idx].pairs
    return pairs


def greedy_repair(
    graphs: StemGraphs,
    selection: Sequence[int],
    priority: Sequence[float] | None = None,
    pseudoknot_mode: bool = False,
) -> tuple[int, ...]:
    """Deterministic greedy repair: keep the best stems that stay compatible.

    Stems are considered in order of increasing ``priority`` (most stabilising
    first) and accepted when compatible with everything accepted so far. Ties
    break on stem index, so the result does not depend on input order or on any
    hash iteration order.

    ``priority`` defaults to stem length (longer first), which is a reasonable
    energy proxy when no model is supplied.
    """
    if priority is None:
        order = sorted(selection, key=lambda s: (-graphs.stems[s].length, s))
    else:
        order = sorted(selection, key=lambda s: (priority[s], s))

    kept: list[int] = []
    for candidate in order:
        if all(
            graphs.compatible(candidate, other, pseudoknot_mode) for other in kept
        ):
            kept.append(candidate)
    return tuple(sorted(kept))


def decode(
    bits: Sequence[int],
    graphs: StemGraphs,
    sequence_length: int,
    priority: Sequence[float] | None = None,
    pseudoknot_mode: bool = False,
) -> DecodeResult:
    """Full decode pipeline for one bitstring.

    1. read selected stems,
    2. check pairwise validity,
    3. greedy-repair if invalid,
    4. emit dot-bracket (extended alphabet in pseudoknot mode).
    """
    raw = bits_to_selection(bits)

    feasible = all(
        graphs.compatible(a, b, pseudoknot_mode)
        for k, a in enumerate(raw)
        for b in raw[k + 1 :]
    )
    kept = raw if feasible else greedy_repair(graphs, raw, priority, pseudoknot_mode)

    structure = dotbracket_from_pairs(
        selection_to_pairs(graphs, kept), sequence_length
    )

    return DecodeResult(
        selected_raw=raw,
        selected=tuple(kept),
        structure=structure,
        feasible_raw=feasible,
        removed=tuple(sorted(set(raw) - set(kept))),
    )


def selection_to_structure(
    graphs: StemGraphs, selection: Sequence[int], sequence_length: int
) -> str:
    """Dot-bracket for an already-valid selection."""
    return dotbracket_from_pairs(selection_to_pairs(graphs, selection), sequence_length)


def structure_to_selection(
    graphs: StemGraphs, structure: str
) -> tuple[int, ...] | None:
    """Inverse map: the stem set whose pairs are exactly ``structure``.

    Returns ``None`` when the structure is not representable by the candidate
    stem set -- which is itself an encoding-gap diagnostic, since it means the
    reference structure lies outside the model's expressive range entirely.
    """
    from .metrics import pairs_from_dotbracket

    target = pairs_from_dotbracket(structure)
    chosen: list[int] = []
    covered: set[tuple[int, int]] = set()
    for idx, stem in enumerate(graphs.stems):
        if stem.pairs <= target:
            chosen.append(idx)
            covered |= stem.pairs
    if covered != target:
        return None
    # Drop stems fully contained in another selected stem's pair set.
    maximal = [
        idx
        for idx in chosen
        if not any(
            other != idx and graphs.stems[idx].pairs < graphs.stems[other].pairs
            for other in chosen
        )
    ]
    return tuple(sorted(maximal))


def stem_priority(model_linear: dict[tuple[int, ...], float], n_vars: int) -> list[float]:
    """Repair priority from a model's linear coefficients (most negative first)."""
    return [model_linear.get((i,), 0.0) for i in range(n_vars)]


def describe_selection(graphs: StemGraphs, selection: Sequence[int]) -> list[Stem]:
    """The selected stems themselves, for inspection and notebooks."""
    return [graphs.stems[i] for i in selection]
