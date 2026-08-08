"""Exact solvers -- the ground truth for the optimizer gap.

Plan section 4.6: brute force for small instances, an exact MIP/CP solver for
larger ones.  Without a proven optimum for *our own model* there is no way to
separate encoding error from optimizer error, so this module is what makes the
section 2.2 decomposition possible at all.

Two independent implementations are provided and cross-checked against each
other in the tests.  That redundancy is deliberate: the exact optimum is the
denominator of every optimizer-gap number in the report, and a silent bug here
would corrupt the whole results section without ever failing loudly.

Every Turner energy is a multiple of 0.01 kcal/mol, so scaling by 100 makes the
model exactly integral and CP-SAT returns a *proven* optimum rather than a
floating-point approximation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model.base import PolynomialModel
from ..model.bundle import StemModel
from .base import (
    InfeasibleModelError,
    SolverResult,
    bitstring_from_selection,
    timed,
)

#: Above this many variables, brute force is refused rather than left to run.
DEFAULT_BRUTE_FORCE_LIMIT = 22

#: Coefficient scaling for the integer encoding used by CP-SAT.
ENERGY_SCALE = 100


# --------------------------------------------------------------------------
# Brute force
# --------------------------------------------------------------------------


class BruteForceSolver:
    """Exhaustive enumeration of all 2^n assignments, vectorised with numpy.

    Exact by construction and free of modelling subtleties, which makes it the
    reference implementation the CP-SAT path is validated against.
    """

    name = "brute_force"

    def __init__(self, limit: int = DEFAULT_BRUTE_FORCE_LIMIT) -> None:
        self.limit = limit

    def solve(
        self,
        model: StemModel,
        use_penalties: bool = True,
        **kwargs: Any,
    ) -> SolverResult:
        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        if n > self.limit:
            raise ValueError(
                f"brute force refused: {n} variables exceeds the limit of "
                f"{self.limit} (2^{n} assignments). Use CPSATSolver instead."
            )

        # With penalties off, feasibility must be enforced *structurally* --
        # otherwise this minimises the bare objective over all assignments,
        # including ones that select conflicting stems, and the resulting
        # "hard-constrained optimum" is not an optimum of the constrained
        # problem at all.
        constraints = None if use_penalties else model.hard_constraints()

        with timed() as elapsed:
            energies, best_state = _brute_force_scan(poly, n, constraints)

        bits = tuple((best_state >> k) & 1 for k in range(n))
        return SolverResult(
            bitstring=bits,
            model_energy=float(energies),
            wall_time=elapsed[0],
            resource_dict={
                "n_vars": n,
                "states_enumerated": 2**n,
                "function_evaluations": 2**n,
            },
            solver_metadata={
                "proven_optimal": True,
                "use_penalties": use_penalties,
            },
            seed=None,
            solver_name=self.name,
        )


def _brute_force_scan(
    poly: PolynomialModel,
    n: int,
    constraints: list[tuple[int, int]] | None = None,
) -> tuple[float, int]:
    """``(best_energy, best_state)`` over all 2^n assignments.

    When ``constraints`` is given, assignments violating any at-most-one pair
    are excluded from the search rather than merely penalised.
    """
    if n == 0:
        return poly.constant, 0
    states = np.arange(2**n, dtype=np.int64)
    energies = np.full(2**n, poly.constant, dtype=np.float64)
    for key, coeff in poly.terms.items():
        if not key:
            continue
        mask = 0
        for v in key:
            mask |= 1 << v
        energies += coeff * ((states & mask) == mask)

    if constraints:
        feasible = np.ones(2**n, dtype=bool)
        for a, b in constraints:
            pair_mask = (1 << a) | (1 << b)
            feasible &= (states & pair_mask) != pair_mask
        if not feasible.any():  # pragma: no cover - all-zero is always feasible
            raise InfeasibleModelError("no assignment satisfies every constraint")
        energies = np.where(feasible, energies, np.inf)

    best = int(np.argmin(energies))
    return float(energies[best]), best


def enumerate_optima(
    model: StemModel,
    use_penalties: bool = True,
    tolerance: float = 1e-9,
    limit: int = DEFAULT_BRUTE_FORCE_LIMIT,
) -> list[tuple[int, ...]]:
    """All assignments achieving the optimum, for degeneracy analysis.

    Plan section 6 lists degenerate optima as a known pitfall: several distinct
    structures can share the same model energy, and reporting only one
    understates what the model actually predicts.
    """
    poly = model.full if use_penalties else model.objective
    n = poly.n_vars
    if n > limit:
        raise ValueError(f"enumerate_optima refused: {n} variables exceeds {limit}")
    if n == 0:
        return [()]
    states = np.arange(2**n, dtype=np.int64)
    energies = np.full(2**n, poly.constant, dtype=np.float64)
    for key, coeff in poly.terms.items():
        if not key:
            continue
        mask = 0
        for v in key:
            mask |= 1 << v
        energies += coeff * ((states & mask) == mask)
    best = energies.min()
    winners = np.flatnonzero(energies <= best + tolerance)
    return [tuple((int(s) >> k) & 1 for k in range(n)) for s in winners]


# --------------------------------------------------------------------------
# CP-SAT
# --------------------------------------------------------------------------


class CPSATSolver:
    """Exact solver via OR-Tools CP-SAT.

    Handles arbitrary polynomial degree by reifying each product term with a
    fresh boolean and two linear constraints, which is exact (not a relaxation).

    Supports both constraint-enforcement strategies, so the plan's
    penalty-vs-hard-constraint comparison (sections 4.5 and 1.6) runs on an
    identical objective:

    ``use_penalties=True``   minimise objective + penalties, constraints soft
    ``use_penalties=False``  minimise objective subject to hard pairwise bans
    """

    name = "cpsat"

    def __init__(self, max_seconds: float = 60.0, workers: int = 8) -> None:
        self.max_seconds = max_seconds
        self.workers = workers

    def solve(
        self,
        model: StemModel,
        use_penalties: bool = True,
        max_seconds: float | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        from ortools.sat.python import cp_model

        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        budget = self.max_seconds if max_seconds is None else max_seconds

        cp = cp_model.CpModel()
        x = [cp.NewBoolVar(f"x{i}") for i in range(n)]

        integer_terms = poly.to_integer_terms(ENERGY_SCALE)
        objective_terms = []
        constant = 0
        n_aux = 0

        for key, coeff in integer_terms.items():
            if not key:
                constant += coeff
            elif len(key) == 1:
                objective_terms.append(coeff * x[key[0]])
            else:
                y = cp.NewBoolVar(f"y{'_'.join(map(str, key))}")
                n_aux += 1
                # Full reification: y == AND(x_k for k in key)
                for v in key:
                    cp.AddImplication(y, x[v])
                cp.Add(sum(x[v] for v in key) - y <= len(key) - 1)
                objective_terms.append(coeff * y)

        if not use_penalties:
            for a, b in model.hard_constraints():
                cp.AddAtMostOne([x[a], x[b]])

        cp.Minimize(sum(objective_terms) + constant)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = budget
        solver.parameters.num_search_workers = self.workers

        with timed() as elapsed:
            status = solver.Solve(cp)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"CP-SAT returned status {solver.StatusName(status)} for a model "
                "that should always admit the all-zero assignment"
            )

        selection = [i for i in range(n) if solver.Value(x[i])]
        bits = bitstring_from_selection(selection, n)

        return SolverResult(
            bitstring=bits,
            model_energy=poly.energy(bits),
            wall_time=elapsed[0],
            resource_dict={
                "n_vars": n,
                "n_auxiliary_vars": n_aux,
                "branches": solver.NumBranches(),
                "conflicts": solver.NumConflicts(),
            },
            solver_metadata={
                "proven_optimal": status == cp_model.OPTIMAL,
                "status": solver.StatusName(status),
                "objective_bound": solver.BestObjectiveBound() / ENERGY_SCALE,
                "use_penalties": use_penalties,
            },
            seed=None,
            solver_name=self.name,
        )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def solve_exact(
    model: StemModel,
    use_penalties: bool = True,
    brute_force_limit: int = DEFAULT_BRUTE_FORCE_LIMIT,
    max_seconds: float = 60.0,
) -> SolverResult:
    """Exact optimum of ``model``, by whichever exact method fits.

    Brute force below ``brute_force_limit`` variables, CP-SAT above. Both are
    exact, so the optimizer gap computed against either is a true gap.
    """
    if model.n_vars <= brute_force_limit:
        return BruteForceSolver(limit=brute_force_limit).solve(
            model, use_penalties=use_penalties
        )
    return CPSATSolver(max_seconds=max_seconds).solve(
        model, use_penalties=use_penalties
    )
