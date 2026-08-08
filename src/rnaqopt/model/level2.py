"""Level 2 -- Level 1 plus cubic terms: multiloops and deep nesting chains.

Plan section 4.3.  Turner multiloop energy is affine in branch count,
``E_ML = a + b*unpaired + c*branches``, and *the branch-count dependence is what
generates the three-body structure once expressed in stem variables.*  Degree 3
is the ceiling of the available Dirac-3 hardware, so this rung is where the
hardware/model co-design argument is cashed out.

Two distinct three-body effects are corrected here.

**1. Deep nesting chains -- exactly representable.**

Level 1 adds a nesting correction ``Q(s,t)`` for *every* nested pair, but the
correction is only physical when ``t`` sits directly inside ``s``.  In a chain
``s > u > t`` the term ``Q(s,t)`` also fires and is spurious.  Subtracting it on
the triple removes the error exactly::

    C_chain(s,u,t) = -Q(s,t)

**2. Multiloops -- exactly representable for 3-way junctions only.**

Write the total loop correction that stem ``s`` needs when it encloses ``k``
directly-nested branches.  Level 1 supplies ``(1-k)*hp(s) + sum_i int(s,t_i)``;
Turner requires ``E_ML(k) = a + c*(k+1) + terminal(s) + sum_i terminal(t_i)``.
The difference separates cleanly::

    Delta(k) = K(s) + sum_i P(s,t_i)

    K(s)    = a + c + terminal(s_inner) - hp(s)            (fires once, if k>=2)
    P(s,t)  = c + terminal(t_outer) - int(s,t) + hp(s)     (fires per branch)

``P`` is pairwise and ``K`` is a constant -- but ``K`` must fire **exactly once
when k >= 2 and never when k <= 1**, and that indicator is the obstruction.

*This is a provable limitation, not an implementation shortcut.*  Suppose the
branch-count dependence were carried by a linear term ``alpha*k`` plus a cubic
term ``beta*C(k,2)``.  Exactness would demand ``f(1)=0``, ``f(2)=K``,
``f(3)=K``, giving ``alpha=0``, ``beta=K`` from the first two, and then
``f(3)=3K != K``.  **No degree-3 polynomial in stem variables reproduces the
multiloop closing penalty for arbitrary branch count.**  Degree 3 buys exactly
the 3-way junction; a 4-way junction needs degree 4, a 5-way junction degree 5.

That is a genuine and quantitative answer to "what does it cost to represent
RNA folding physics faithfully on optimization hardware", and it is reported as
such rather than hidden.  ``branch_damping`` offers the practical mitigation:
scaling the cubic coefficient trades exactness at ``k=2`` for smaller error at
``k>=3``, and the trade is measured rather than asserted.

**No coaxial-stacking term.**  Plan section 4.3 lists coaxial stacking as a
candidate cubic term, but under the project's frozen ``dangles=0`` setting
ViennaRNA does not apply coaxial stacking at all -- verified by the loop-tree
decomposition reproducing ``eval_structure`` exactly on all 60 reference
structures without one.  Adding the term would move the model *away* from the
reference it is scored against.
"""

from __future__ import annotations

from ..config import VIENNA, ViennaConfig
from ..energy import LoopEnergies
from ..stems import StemGraphs
from .base import PolynomialModel
from .bundle import StemModel
from .level1 import build_level1
from .penalties import default_penalty


def build_level2(
    sequence: str,
    graphs: StemGraphs,
    cfg: ViennaConfig = VIENNA,
    lambda_conflict: float | None = None,
    lambda_cross: float | None = None,
    pseudoknot_mode: bool = False,
    energies: LoopEnergies | None = None,
    branch_damping: float = 1.0,
    include_chain_correction: bool = True,
    include_multiloop: bool = True,
) -> StemModel:
    """Build the Level 2 (degree-3) model.

    ``branch_damping`` scales the multiloop cubic coefficient. ``1.0`` is exact
    for 3-way junctions; smaller values reduce the over-count on junctions with
    more branches at the cost of that exactness.
    """
    le = energies or LoopEnergies(sequence, cfg)
    base = build_level1(
        sequence, graphs, cfg, lambda_conflict, lambda_cross, pseudoknot_mode, le
    )

    objective = PolynomialModel(
        n_vars=graphs.n,
        terms=dict(base.objective.terms),
        metadata={"level": 2, "sequence_length": len(sequence)},
    )

    nesting = set(graphs.nesting)
    children: dict[int, list[int]] = {}
    for outer, inner in nesting:
        children.setdefault(outer, []).append(inner)
    for v in children.values():
        v.sort()

    n_chain = 0
    n_multiloop = 0

    # -- 1. deep-chain corrections (exact) ---------------------------------
    if include_chain_correction:
        for s, kids in children.items():
            for u in kids:
                for t in children.get(u, ()):
                    if (s, t) not in nesting:
                        continue
                    # Undo the spurious Level 1 term for the indirect pair.
                    spurious = (
                        le.interior(graphs.stems[s], graphs.stems[t])
                        - le.hairpin(graphs.stems[s])
                        - le.exterior(graphs.stems[t])
                    )
                    objective.add((s, u, t), -spurious)
                    n_chain += 1

    # -- 2. multiloop corrections (exact for k = 2) ------------------------
    if include_multiloop:
        for s, kids in children.items():
            stem_s = graphs.stems[s]
            k_const = (
                le.ml_closing
                + le.ml_intern
                + le.terminal_penalty(*stem_s.inner)
                - le.hairpin(stem_s)
            )
            for a_idx in range(len(kids)):
                for b_idx in range(a_idx + 1, len(kids)):
                    t, u = kids[a_idx], kids[b_idx]
                    # Siblings only: mutually compatible and not nested in each
                    # other (that case is the chain correction above).
                    if not graphs.compatible(t, u, pseudoknot_mode):
                        continue
                    if (t, u) in nesting or (u, t) in nesting:
                        continue
                    coeff = k_const + _branch_term(le, graphs, s, t) + _branch_term(
                        le, graphs, s, u
                    )
                    objective.add((s, t, u), branch_damping * coeff)
                    n_multiloop += 1

    model = StemModel.assemble(
        objective=objective,
        graphs=graphs,
        level=2,
        lambda_conflict=lambda_conflict,
        lambda_cross=lambda_cross,
        pseudoknot_mode=pseudoknot_mode,
        default_lambda=default_penalty,
        extra_metadata={
            "n_chain_corrections": n_chain,
            "n_multiloop_corrections": n_multiloop,
            "branch_damping": branch_damping,
        },
    )
    return model


def _branch_term(
    le: LoopEnergies, graphs: StemGraphs, s: int, t: int
) -> float:
    """``P(s,t) = c + terminal(t_outer) - int(s,t) + hp(s)``."""
    stem_s, stem_t = graphs.stems[s], graphs.stems[t]
    return (
        le.ml_intern
        + le.terminal_penalty(stem_t.i, stem_t.j)
        - le.interior(stem_s, stem_t)
        + le.hairpin(stem_s)
    )


def max_exact_branch_count() -> int:
    """The largest multiloop branch count a degree-3 model represents exactly.

    Two branches (a 3-way junction). See the module docstring for the proof
    that three is unreachable at degree 3.
    """
    return 2


def structure_is_level2_exact(structure: str) -> bool:
    """True if Level 2 represents ``structure`` exactly.

    Exact unless the structure contains a multiloop with three or more enclosed
    branches, which no degree-3 model can represent.
    """
    from ..energy import _children, _pair_table

    pt = _pair_table(structure)

    def walk(pairs: list[tuple[int, int]]) -> bool:
        for i, j in pairs:
            kids, _ = _children(pt, i, j)
            while len(kids) == 1 and kids[0] == (i + 1, j - 1):
                i, j = kids[0]
                kids, _ = _children(pt, i, j)
            if len(kids) > max_exact_branch_count():
                return False
            if not walk(kids):
                return False
        return True

    top, _ = _children(pt, -1, len(structure))
    return walk(top)
