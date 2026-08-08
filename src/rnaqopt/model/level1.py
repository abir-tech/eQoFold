"""Level 1 -- Level 0 plus quadratic loop energies.

Plan section 4.3.  Level 0 charges nothing for loops.  Level 1 adds them, and
the plan flags the subtlety that makes it interesting:

    *Also add the hairpin term for a stem with nothing nested inside it -- note
    this is conditional, so model it as a linear hairpin term corrected by
    nesting terms, and verify the correction is exact by brute-force comparison
    on Tier A.*

That is exactly what this module does.

**Linear coefficient** of stem *s*, i.e. its cost when it stands alone on the
exterior loop closing a hairpin::

    L(s) = E_stack(s) + E_hairpin(s) + E_exterior(s)

**Quadratic correction** when *t* is nested directly inside *s*: the hairpin of
*s* is replaced by the interior loop between them, and *t* is no longer on the
exterior loop::

    Q(s,t) = E_interior(s,t) - E_hairpin(s) - E_exterior(t)

Summing ``L`` over selected stems and ``Q`` over nested pairs then telescopes to
the true Turner energy::

    s alone            L(s)                     = stack + hairpin + ext        exact
    s > t              L(s)+L(t)+Q(s,t)         = stack(s)+ext(s)+int(s,t)
                                                  +stack(t)+hairpin(t)         exact
    s > t > u          + Q(t,u)                                                exact
                       ... except Q(s,u) also fires                            ERROR

**Where Level 1 is inexact, by construction:**

1. *Nesting chains of depth >= 3.* ``Q(s,u)`` fires for the indirectly-nested
   pair, adding a spurious interior-loop correction.
2. *Multiloops.* When two stems *t*, *u* are both nested in *s*, the hairpin of
   *s* is subtracted twice and the multiloop closing penalty is never charged.

Both are three-body effects, and both are repaired by the Level 2 cubic terms.
This is not a defect being hidden -- it is the measured content of the
encoding-gap ladder, and :func:`level1_exactness` reports it per structure.
"""

from __future__ import annotations

from ..config import VIENNA, ViennaConfig
from ..energy import LoopEnergies
from ..stems import StemGraphs
from .base import PolynomialModel
from .bundle import StemModel
from .penalties import default_penalty


def build_level1(
    sequence: str,
    graphs: StemGraphs,
    cfg: ViennaConfig = VIENNA,
    lambda_conflict: float | None = None,
    lambda_cross: float | None = None,
    pseudoknot_mode: bool = False,
    energies: LoopEnergies | None = None,
) -> StemModel:
    """Build the Level 1 model for one sequence."""
    le = energies or LoopEnergies(sequence, cfg)

    objective = PolynomialModel(
        n_vars=graphs.n,
        metadata={"level": 1, "sequence_length": len(sequence)},
    )

    for idx, stem in enumerate(graphs.stems):
        objective.add((idx,), le.stack(stem) + le.hairpin(stem) + le.exterior(stem))

    for outer_idx, inner_idx in sorted(graphs.nesting):
        outer = graphs.stems[outer_idx]
        inner = graphs.stems[inner_idx]
        correction = (
            le.interior(outer, inner) - le.hairpin(outer) - le.exterior(inner)
        )
        objective.add((outer_idx, inner_idx), correction)

    return StemModel.assemble(
        objective=objective,
        graphs=graphs,
        level=1,
        lambda_conflict=lambda_conflict,
        lambda_cross=lambda_cross,
        pseudoknot_mode=pseudoknot_mode,
        default_lambda=default_penalty,
    )


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def structure_is_level1_exact(structure: str) -> bool:
    """True if Level 1 represents ``structure`` exactly.

    Exact when every loop has at most one enclosed helix (no multiloops) *and*
    no helix is nested at depth >= 2 below another, since both cases introduce
    three-body corrections Level 1 cannot express.
    """
    from ..energy import _children, _pair_table

    pt = _pair_table(structure)
    top, _ = _children(pt, -1, len(structure))

    # depth of each helix in the nesting forest, counting helices not pairs
    def walk(pairs: list[tuple[int, int]], depth: int) -> bool:
        for i, j in pairs:
            kids, _ = _children(pt, i, j)
            # collapse stacked continuations: they are the same helix
            while len(kids) == 1 and kids[0] == (i + 1, j - 1):
                i, j = kids[0]
                kids, _ = _children(pt, i, j)
            if len(kids) >= 2:
                return False  # multiloop
            if kids and depth >= 1:
                return False  # helix nested two deep
            if kids and not walk(kids, depth + 1):
                return False
        return True

    return walk(top, 0)
