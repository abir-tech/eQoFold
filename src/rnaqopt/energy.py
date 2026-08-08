"""Turner nearest-neighbour energy terms, extracted from ViennaRNA.

Plan section 4.3: *extract these from ViennaRNA rather than hard-coding a
table.*  Every coefficient in every model on the fidelity ladder comes from
ViennaRNA's own loop-energy primitives under the frozen configuration, so a
model term and the reference energy cannot drift apart.

The decomposition this module implements is exact.  For any pseudoknot-free
structure,

    eval_structure(db) == sum of (stack + hairpin + interior + multiloop
                                  + exterior) terms over its loop tree

and :func:`decompose_structure` computes exactly that sum.
``tests/test_energy.py`` asserts the identity on every reference structure --
which is what licenses using these terms as model coefficients.

Units: ViennaRNA's loop primitives return **dacal/mol** (integer, x100).
Everything this module exposes is **kcal/mol** (float).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .config import VIENNA, ViennaConfig
from .reference import fold_compound
from .stems import Stem

#: ViennaRNA loop primitives return energies in units of 0.01 kcal/mol.
DACAL_PER_KCAL = 100.0

#: Pairs that incur the terminal-AU/GU penalty when they close an exterior loop
#: or sit at a multiloop branch.
_TERMINAL_PENALTY_PAIRS = frozenset({("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")})


@dataclass
class LoopEnergies:
    """Turner loop-energy accessor for one sequence.

    Wraps a single ``fold_compound``.  All positions taken and returned are
    **0-based**; ViennaRNA's 1-based convention is handled internally and never
    leaks out.
    """

    sequence: str
    cfg: ViennaConfig = VIENNA

    def __post_init__(self) -> None:
        self._fc = fold_compound(self.sequence, self.cfg)
        self._params = self._fc.params

    # -- multiloop constants ----------------------------------------------

    @cached_property
    def ml_closing(self) -> float:
        """Multiloop closing penalty ``a`` (kcal/mol)."""
        return self._params.MLclosing / DACAL_PER_KCAL

    @cached_property
    def ml_intern(self) -> float:
        """Per-branch multiloop term ``c`` (kcal/mol), applied to every stem in
        the loop *including* the closing one."""
        return self._params.MLintern[0] / DACAL_PER_KCAL

    @cached_property
    def ml_base(self) -> float:
        """Per-unpaired-nucleotide multiloop term ``b`` (kcal/mol).

        Zero in Turner 2004, which removes the unpaired-count dependence from
        the multiloop energy entirely and makes the Level 2 term depend only on
        branch count.
        """
        return self._params.MLbase / DACAL_PER_KCAL

    @cached_property
    def terminal_au(self) -> float:
        """Terminal AU/GU penalty (kcal/mol)."""
        return self._params.TerminalAU / DACAL_PER_KCAL

    def terminal_penalty(self, i: int, j: int) -> float:
        """Terminal AU/GU penalty for the pair ``(i, j)``, 0-based."""
        pair = (self.sequence[i], self.sequence[j])
        return self.terminal_au if pair in _TERMINAL_PENALTY_PAIRS else 0.0

    # -- loop primitives ---------------------------------------------------

    def stack_pair(self, i: int, j: int, k: int, ll: int) -> float:
        """Interior-loop energy between closing pair ``(i,j)`` and enclosed
        pair ``(k,l)``.  With ``k=i+1, l=j-1`` this is a plain stack."""
        return self._fc.eval_int_loop(i + 1, j + 1, k + 1, ll + 1) / DACAL_PER_KCAL

    def hairpin_pair(self, i: int, j: int) -> float:
        """Hairpin-loop energy for the closing pair ``(i, j)``."""
        return self._fc.eval_hp_loop(i + 1, j + 1) / DACAL_PER_KCAL

    def exterior_pair(self, i: int, j: int) -> float:
        """Exterior-loop contribution of a helix closed by ``(i, j)``.

        Under ``dangles=0`` this is exactly the terminal AU/GU penalty.
        """
        return self._fc.eval_ext_stem(i + 1, j + 1) / DACAL_PER_KCAL

    # -- stem-level terms --------------------------------------------------

    def stack(self, stem: Stem) -> float:
        """Total stacking energy *within* a stem (negative, stabilising).

        This is ``E_stack(s)`` of plan section 4.3 and the sole linear
        coefficient of the Level 0 model.  A stem of length 1 has no stack and
        therefore contributes zero here.
        """
        total = 0.0
        for k in range(stem.length - 1):
            total += self.stack_pair(
                stem.i + k, stem.j - k, stem.i + k + 1, stem.j - k - 1
            )
        return total

    def hairpin(self, stem: Stem) -> float:
        """Hairpin energy assuming nothing is nested inside ``stem``."""
        a, b = stem.inner
        return self.hairpin_pair(a, b)

    def exterior(self, stem: Stem) -> float:
        """Exterior-loop contribution assuming ``stem`` sits on the exterior loop."""
        return self.exterior_pair(stem.i, stem.j)

    def interior(self, outer: Stem, inner: Stem) -> float:
        """Interior/bulge/stack energy between ``outer`` and a directly nested
        ``inner`` stem."""
        a, b = outer.inner
        return self.stack_pair(a, b, inner.i, inner.j)

    def multiloop(
        self,
        closing: Stem,
        branches: list[Stem] | tuple[Stem, ...],
        n_unpaired: int = 0,
    ) -> float:
        """Energy of a multiloop closed by ``closing`` with ``branches`` inside.

        Turner form, verified exact against ViennaRNA's own loop decomposition
        on every multiloop in the reference corpus::

            E_ML = a + c*(branches + 1) + b*unpaired
                   + terminal(closing) + sum terminal(branch)

        The ``+1`` counts the closing stem itself.  The affine dependence on
        branch count is what becomes a three-body term in stem variables, and
        is the reason the Level 2 model needs degree 3.
        """
        n = len(branches)
        total = (
            self.ml_closing
            + self.ml_intern * (n + 1)
            + self.ml_base * n_unpaired
            + self.terminal_penalty(*closing.inner)
        )
        for b in branches:
            total += self.terminal_penalty(b.i, b.j)
        return total


# --------------------------------------------------------------------------
# Loop-tree decomposition -- the correctness anchor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopTerm:
    """One labelled contribution to a structure's total free energy."""

    kind: str  # "stack" | "hairpin" | "interior" | "multiloop" | "exterior"
    closing: tuple[int, int] | None
    energy: float


def _pair_table(structure: str) -> list[int]:
    """0-based pair table; ``-1`` where unpaired."""
    pt = [-1] * len(structure)
    stack: list[int] = []
    for k, ch in enumerate(structure):
        if ch == "(":
            stack.append(k)
        elif ch == ")":
            a = stack.pop()
            pt[a] = k
            pt[k] = a
    if stack:
        raise ValueError(f"unbalanced structure: {structure!r}")
    return pt


def _children(pt: list[int], i: int, j: int) -> tuple[list[tuple[int, int]], int]:
    """Pairs directly enclosed by ``(i, j)``, plus the unpaired count between them."""
    kids: list[tuple[int, int]] = []
    unpaired = 0
    k = i + 1
    while k < j:
        if pt[k] > k:
            kids.append((k, pt[k]))
            k = pt[k] + 1
        else:
            unpaired += 1
            k += 1
    return kids, unpaired


def decompose_structure(
    sequence: str,
    structure: str,
    cfg: ViennaConfig = VIENNA,
) -> list[LoopTerm]:
    """Decompose a pseudoknot-free structure into labelled loop contributions.

    The sum of the returned energies equals ``eval_structure(sequence,
    structure)`` exactly.  This function is the ground truth against which the
    model ladder's coefficients are validated: a model term is only trustworthy
    if the same primitives reproduce ViennaRNA's own total.
    """
    le = LoopEnergies(sequence, cfg)
    pt = _pair_table(structure)
    terms: list[LoopTerm] = []

    # Exterior loop: every helix whose closing pair is not enclosed by another.
    top, _ = _children(pt, -1, len(structure))
    for i, j in top:
        terms.append(LoopTerm("exterior", (i, j), le.exterior_pair(i, j)))

    stack: list[tuple[int, int]] = list(top)
    while stack:
        i, j = stack.pop()
        kids, unpaired = _children(pt, i, j)
        if not kids:
            terms.append(LoopTerm("hairpin", (i, j), le.hairpin_pair(i, j)))
        elif len(kids) == 1:
            k, m = kids[0]
            kind = "stack" if (k == i + 1 and m == j - 1) else "interior"
            terms.append(LoopTerm(kind, (i, j), le.stack_pair(i, j, k, m)))
        else:
            energy = (
                le.ml_closing
                + le.ml_intern * (len(kids) + 1)
                + le.ml_base * unpaired
                + le.terminal_penalty(i, j)
                + sum(le.terminal_penalty(a, b) for a, b in kids)
            )
            terms.append(LoopTerm("multiloop", (i, j), energy))
        stack.extend(kids)

    return terms


def decomposed_energy(
    sequence: str,
    structure: str,
    cfg: ViennaConfig = VIENNA,
) -> float:
    """Total free energy from the loop-tree decomposition, in kcal/mol."""
    return round(sum(t.energy for t in decompose_structure(sequence, structure, cfg)), 2)
