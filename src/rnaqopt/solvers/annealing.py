"""Simulated annealing over the polynomial model.

Plan section 4.6 requires a heuristic classical baseline run under a **matched
wall-clock budget**, because unmatched comparisons will be discounted.  This
implementation therefore accepts either a sweep count or a time budget, and
reports whichever bound actually stopped it.

Works at any polynomial degree: the incremental energy delta of a single spin
flip is computed from the terms containing that variable, with no assumption
that the model is quadratic.  That matters because the same solver has to serve
as the baseline for the Level 2 cubic model, where a QUBO-only annealer (such
as ``neal``) would silently require quadratization and stop being a like-for-
like comparison.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Any

from ..model.base import PolynomialModel
from ..model.bundle import StemModel
from .base import SolverResult, timed


def _index_terms(poly: PolynomialModel) -> dict[int, list[tuple[tuple[int, ...], float]]]:
    """Map each variable to the terms it appears in."""
    index: dict[int, list[tuple[tuple[int, ...], float]]] = defaultdict(list)
    for key, coeff in poly.terms.items():
        for v in key:
            index[v].append((key, coeff))
    return index


def _flip_delta(
    x: list[int],
    v: int,
    terms_with_v: list[tuple[tuple[int, ...], float]],
) -> float:
    """Energy change from flipping variable ``v``.

    A term fires iff every variable in it is 1.  Flipping ``v`` can only change
    terms containing ``v``, and only when all *other* variables in that term are
    already 1.  Sign follows the direction of the flip.
    """
    sign = -1.0 if x[v] else 1.0
    delta = 0.0
    for key, coeff in terms_with_v:
        if all(x[u] for u in key if u != v):
            delta += sign * coeff
    return delta


class SimulatedAnnealingSolver:
    """Geometric-schedule simulated annealing with restarts."""

    name = "simulated_annealing"

    def __init__(
        self,
        n_sweeps: int = 2000,
        n_restarts: int = 8,
        seed: int = 0,
        time_budget: float | None = None,
        beta_range: tuple[float, float] | None = None,
    ) -> None:
        self.n_sweeps = n_sweeps
        self.n_restarts = n_restarts
        self.seed = seed
        self.time_budget = time_budget
        self.beta_range = beta_range

    # -- schedule ----------------------------------------------------------

    def _auto_beta_range(self, poly: PolynomialModel) -> tuple[float, float]:
        """Choose the inverse-temperature range from the coefficient scale.

        Hot enough that the largest single-term move is accepted with high
        probability at the start, cold enough that it is essentially never
        accepted at the end. Deriving this from the model rather than hard-coding
        it keeps the annealer sane across the three fidelity levels, whose
        coefficient magnitudes differ by an order of magnitude.
        """
        magnitudes = [abs(c) for k, c in poly.terms.items() if k]
        if not magnitudes:
            return 0.1, 10.0
        scale = max(magnitudes)
        return 0.1 / scale, 10.0 / scale

    # -- solve -------------------------------------------------------------

    def solve(
        self,
        model: StemModel,
        use_penalties: bool = True,
        **kwargs: Any,
    ) -> SolverResult:
        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        index = _index_terms(poly)
        beta0, beta1 = self.beta_range or self._auto_beta_range(poly)

        best_energy = math.inf
        best_x: list[int] = [0] * n
        evaluations = 0
        restarts_done = 0
        stopped_by = "restarts"
        deadline = (
            time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else math.inf
        )

        with timed() as elapsed:
            if n == 0:
                best_energy = poly.constant
            for restart in range(self.n_restarts):
                rng = random.Random(self.seed * 100003 + restart)
                x = [rng.randint(0, 1) for _ in range(n)]
                energy = poly.energy(x)

                for sweep in range(self.n_sweeps):
                    frac = sweep / max(self.n_sweeps - 1, 1)
                    # Geometric interpolation of inverse temperature.
                    beta = beta0 * (beta1 / beta0) ** frac
                    for v in range(n):
                        delta = _flip_delta(x, v, index[v])
                        evaluations += 1
                        if delta <= 0.0 or rng.random() < math.exp(-beta * delta):
                            x[v] ^= 1
                            energy += delta
                    if energy < best_energy:
                        best_energy = energy
                        best_x = list(x)
                    if time.perf_counter() > deadline:
                        stopped_by = "time_budget"
                        break

                restarts_done = restart + 1
                if time.perf_counter() > deadline:
                    stopped_by = "time_budget"
                    break

        # Recompute from scratch: guards against incremental-delta drift.
        final_energy = poly.energy(best_x) if n else poly.constant

        return SolverResult(
            bitstring=tuple(best_x),
            model_energy=final_energy,
            wall_time=elapsed[0],
            resource_dict={
                "n_vars": n,
                "function_evaluations": evaluations,
                "restarts": restarts_done,
                "sweeps_per_restart": self.n_sweeps,
            },
            solver_metadata={
                "proven_optimal": False,
                "stopped_by": stopped_by,
                "beta_range": (beta0, beta1),
                "use_penalties": use_penalties,
                "incremental_energy": best_energy,
            },
            seed=self.seed,
            solver_name=self.name,
        )


class RandomSearchSolver:
    """Uniform random sampling -- the floor every other solver must clear.

    Included because "our heuristic beat brute-force-free guessing" is the
    weakest claim a solver can make, and a report that cannot demonstrate even
    that is not reporting a result. Runs under the same budget interface.
    """

    name = "random_search"

    def __init__(
        self, n_samples: int = 10000, seed: int = 0, time_budget: float | None = None
    ) -> None:
        self.n_samples = n_samples
        self.seed = seed
        self.time_budget = time_budget

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        rng = random.Random(self.seed)
        best_x = [0] * n
        best_energy = poly.energy(best_x) if n else poly.constant
        deadline = (
            time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else math.inf
        )
        drawn = 0

        with timed() as elapsed:
            for _ in range(self.n_samples):
                x = [rng.randint(0, 1) for _ in range(n)]
                e = poly.energy(x)
                drawn += 1
                if e < best_energy:
                    best_energy, best_x = e, x
                if time.perf_counter() > deadline:
                    break

        return SolverResult(
            bitstring=tuple(best_x),
            model_energy=best_energy,
            wall_time=elapsed[0],
            resource_dict={"n_vars": n, "function_evaluations": drawn},
            solver_metadata={"proven_optimal": False, "use_penalties": use_penalties},
            seed=self.seed,
            solver_name=self.name,
        )
