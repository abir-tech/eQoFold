"""Penalty-weight calibration (plan section 4.5).

*Do not guess lambda.*  A penalty that is too small lets an infeasible
assignment win on energy; one that is too large flattens the objective and
makes the landscape harder for every heuristic solver.  This module provides
the principled lower bound and the sweep that locates the knee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .base import PolynomialModel


def max_single_variable_gain(objective: PolynomialModel) -> float:
    """Largest energy *decrease* any single variable can produce.

    Plan section 4.5 states the bound as ``lambda > max_s |E_stack(s)|``.  That
    is the right idea but slightly too weak once the model has loop terms: a
    stem can also unlock stabilising *quadratic and cubic* terms.  The correct
    bound sums, for each variable, every negative coefficient of every term
    containing it -- the most energy that selecting it can possibly buy.

    Returns a non-negative number.
    """
    if objective.n_vars == 0:
        return 0.0
    gains = [0.0] * objective.n_vars
    for key, coeff in objective.terms.items():
        if coeff < 0.0:
            for v in key:
                gains[v] += -coeff
    return max(gains, default=0.0)


def default_penalty(objective: PolynomialModel, safety: float = 1.5) -> float:
    """A penalty weight guaranteed to make every violation unprofitable.

    ``safety`` > 1 keeps the constraint strictly binding rather than merely
    tied; 1.5 is the project default and is swept in the calibration figure.

    The result is rounded **up** to a whole 0.01 kcal/mol.  Every Turner energy
    is a multiple of 0.01, and keeping the penalty on the same grid keeps the
    whole model exactly integral, which is what lets CP-SAT prove optimality
    (see ``PolynomialModel.to_integer_terms``).  Rounding up rather than to
    nearest preserves the strict inequality against the bound.
    """
    bound = max_single_variable_gain(objective)
    return max(_round_up_cent(safety * bound), 1.0)


def _round_up_cent(value: float) -> float:
    """Smallest multiple of 0.01 that is >= ``value``."""
    import math

    return math.ceil(round(value * 100, 6)) / 100.0


@dataclass(frozen=True)
class PenaltySweepPoint:
    """One point of the lambda sweep."""

    lam: float
    feasible_rate: float
    mean_optimality_gap: float
    n_instances: int

    def as_row(self) -> dict[str, float]:
        return {
            "lambda": self.lam,
            "feasible_rate": self.feasible_rate,
            "mean_optimality_gap": self.mean_optimality_gap,
            "n_instances": self.n_instances,
        }


def sweep_range(objective: PolynomialModel, n_points: int = 12) -> list[float]:
    """Geometric lambda grid spanning well below to well above the bound.

    Deliberately starts below the guaranteed-feasible bound so the sweep figure
    shows the infeasible regime rather than only the safe one -- the shape of
    the transition is the result, not just the chosen value.
    """
    bound = max(max_single_variable_gain(objective), 1e-6)
    lo, hi = 0.1 * bound, 4.0 * bound
    if n_points < 2:
        return [bound]
    ratio = (hi / lo) ** (1.0 / (n_points - 1))
    return [lo * ratio**k for k in range(n_points)]


def knee(points: Sequence[PenaltySweepPoint], target_rate: float = 1.0) -> float:
    """Smallest lambda achieving ``target_rate`` feasibility.

    "Smallest that works" rather than "largest tried": beyond the knee the
    penalty only degrades the landscape.
    """
    ok = [p for p in points if p.feasible_rate >= target_rate]
    if ok:
        return min(p.lam for p in ok)
    return max(points, key=lambda p: p.feasible_rate).lam if points else 1.0
