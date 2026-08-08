"""``StemModel`` -- an objective, its constraint penalties, and the graphs.

Keeping the energy terms and the constraint penalties in *separate*
:class:`PolynomialModel` instances is what makes the plan's constraint study
(section 4.5, and optional advanced task "trade-offs between qubit count and
constraint enforcement") possible: a penalty-based solver optimises
``model.full``, while a constraint-aware solver optimises ``model.objective``
subject to ``model.hard_constraints()``.  Both are exposed from one object, so
the two enforcement strategies are compared on an identical objective.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..stems import StemGraphs
from .base import PolynomialModel


@dataclass
class StemModel:
    """A fidelity-ladder model over stem variables."""

    objective: PolynomialModel
    penalties: PolynomialModel
    graphs: StemGraphs
    level: int
    lambda_conflict: float
    lambda_cross: float
    pseudoknot_mode: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def assemble(
        cls,
        objective: PolynomialModel,
        graphs: StemGraphs,
        level: int,
        lambda_conflict: float | None,
        lambda_cross: float | None,
        pseudoknot_mode: bool,
        default_lambda: Callable[[PolynomialModel], float],
        extra_metadata: dict[str, Any] | None = None,
    ) -> StemModel:
        """Attach constraint penalties to an objective.

        When ``pseudoknot_mode`` is on, the crossing penalty is dropped
        entirely -- the "free win" of plan section 2.6. Conflict penalties are
        never dropped: two stems sharing a nucleotide is a physical
        impossibility, not a modelling choice.
        """
        lam_c = (
            default_lambda(objective) if lambda_conflict is None else lambda_conflict
        )
        lam_x = default_lambda(objective) if lambda_cross is None else lambda_cross
        if pseudoknot_mode:
            lam_x = 0.0

        penalties = PolynomialModel(
            n_vars=graphs.n, metadata={"level": level, "kind": "penalties"}
        )
        for a, b in sorted(graphs.conflict):
            penalties.add((a, b), lam_c)
        if not pseudoknot_mode:
            for a, b in sorted(graphs.crossing):
                penalties.add((a, b), lam_x)

        return cls(
            objective=objective,
            penalties=penalties,
            graphs=graphs,
            level=level,
            lambda_conflict=lam_c,
            lambda_cross=lam_x,
            pseudoknot_mode=pseudoknot_mode,
            metadata=dict(extra_metadata or {}),
        )

    # -- views -------------------------------------------------------------

    @property
    def full(self) -> PolynomialModel:
        """Objective plus penalties -- what a penalty-based solver minimises."""
        return self.objective.merged(self.penalties)

    @property
    def n_vars(self) -> int:
        return self.graphs.n

    @property
    def degree(self) -> int:
        return max(self.objective.degree, self.penalties.degree)

    def hard_constraints(self) -> list[tuple[int, int]]:
        """Pairs of variables that may not both be 1.

        The constraint-based alternative to penalties. Includes crossings only
        when not in pseudoknot mode.
        """
        pairs = set(self.graphs.conflict)
        if not self.pseudoknot_mode:
            pairs |= set(self.graphs.crossing)
        return sorted(pairs)

    def is_feasible(self, selection: list[int] | set[int]) -> bool:
        """True if no hard constraint is violated by ``selection``."""
        chosen = set(selection)
        return not any(a in chosen and b in chosen for a, b in self.hard_constraints())

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        obj = self.objective.term_counts()
        return {
            "level": self.level,
            "n_vars": self.n_vars,
            "degree": self.degree,
            "n_obj_linear": obj.get(1, 0),
            "n_obj_quadratic": obj.get(2, 0),
            "n_obj_cubic": obj.get(3, 0),
            "n_penalty_terms": self.penalties.n_terms,
            "lambda_conflict": self.lambda_conflict,
            "lambda_cross": self.lambda_cross,
            "pseudoknot_mode": self.pseudoknot_mode,
            **self.graphs.summary(),
            **self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"StemModel(level={self.level}, n_vars={self.n_vars}, "
            f"degree={self.degree}, pk={self.pseudoknot_mode})"
        )
