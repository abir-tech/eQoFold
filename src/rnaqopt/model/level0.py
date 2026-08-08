"""Level 0 -- linear stacking energies plus hard-constraint penalties.

Plan section 4.3::

    H0 = sum_s E_stack(s)*x_s
       + lambda_conflict * sum_{(s,t) in conflict} x_s*x_t
       + lambda_cross    * sum_{(s,t) in crossing} x_s*x_t

This is the rung most published QUBO formulations of RNA folding sit on, and it
is deliberately the *worst* model in the ladder: it accounts for the energy a
helix gains by stacking, and for nothing else.  Every loop in the structure --
hairpin, bulge, interior, multiloop -- is free.  Because loop penalties are
large and positive, Level 0 systematically over-predicts pairing.

Its role is to be the baseline the other rungs are measured against.
"""

from __future__ import annotations

from ..config import VIENNA, ViennaConfig
from ..energy import LoopEnergies
from ..stems import StemGraphs
from .base import PolynomialModel
from .bundle import StemModel
from .penalties import default_penalty


def build_level0(
    sequence: str,
    graphs: StemGraphs,
    cfg: ViennaConfig = VIENNA,
    lambda_conflict: float | None = None,
    lambda_cross: float | None = None,
    pseudoknot_mode: bool = False,
    energies: LoopEnergies | None = None,
) -> StemModel:
    """Build the Level 0 model for one sequence."""
    le = energies or LoopEnergies(sequence, cfg)

    objective = PolynomialModel(
        n_vars=graphs.n,
        metadata={"level": 0, "sequence_length": len(sequence)},
    )
    for idx, stem in enumerate(graphs.stems):
        objective.add((idx,), le.stack(stem))

    return StemModel.assemble(
        objective=objective,
        graphs=graphs,
        level=0,
        lambda_conflict=lambda_conflict,
        lambda_cross=lambda_cross,
        pseudoknot_mode=pseudoknot_mode,
        default_lambda=default_penalty,
    )
