"""Classical low-rank (Burer-Monteiro) relaxation baseline.

Plan section 2.4, mandatory honesty clause: *PCE's relaxation is closely related
to classical low-rank / Burer-Monteiro SDP relaxations, and there is an active
argument in the literature that it is dequantizable.  We raise this ourselves,
and we benchmark PCE against a classical low-rank relaxation baseline.*

This module is that baseline.  Each binary variable becomes a unit vector in
R^k; the Ising energy is relaxed to inner products between those vectors;
gradient descent on the product of spheres minimises it; and a random-hyperplane
rounding recovers a bitstring.  At ``k = 1`` this is plain local search over
spins; at ``k = n`` it is the full SDP relaxation.

If PCE cannot beat this at comparable ``k``, the honest conclusion is that the
quantum encoding is buying nothing on this problem -- and saying so is worth
more than a result that quietly omits the comparison.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model.base import PolynomialModel
from ..model.bundle import StemModel
from .base import SolverResult, timed


def to_ising(poly: PolynomialModel) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a degree-<=2 pseudo-Boolean polynomial to Ising form.

    With ``x_i = (1 + s_i) / 2`` and ``s_i in {-1, +1}``, returns ``(J, h,
    offset)`` such that ``energy(x) == s @ J @ s + h @ s + offset``.  ``J`` is
    symmetric with zero diagonal.
    """
    if poly.degree > 2:
        raise ValueError(
            f"low-rank relaxation needs degree <= 2, got {poly.degree}; "
            "quadratize the model first (see rnaqopt.model.quadratize)"
        )
    n = poly.n_vars
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = poly.constant

    for key, coeff in poly.terms.items():
        if len(key) == 1:
            # c*x_i = c/2 + c/2 * s_i
            (i,) = key
            offset += coeff / 2.0
            h[i] += coeff / 2.0
        elif len(key) == 2:
            # c*x_i*x_j = c/4 * (1 + s_i + s_j + s_i s_j)
            i, j = key
            offset += coeff / 4.0
            h[i] += coeff / 4.0
            h[j] += coeff / 4.0
            J[i, j] += coeff / 8.0
            J[j, i] += coeff / 8.0
    return J, h, offset


def ising_energy(J: np.ndarray, h: np.ndarray, offset: float, s: np.ndarray) -> float:
    return float(s @ J @ s + h @ s + offset)


def spins_to_bits(s: np.ndarray) -> tuple[int, ...]:
    return tuple(int((v + 1) // 2) for v in s)


class LowRankSolver:
    """Burer-Monteiro relaxation with random-hyperplane rounding."""

    name = "lowrank"

    def __init__(
        self,
        rank: int | None = None,
        n_restarts: int = 8,
        n_steps: int = 400,
        learning_rate: float = 0.1,
        n_roundings: int = 64,
        seed: int = 0,
        time_budget: float | None = None,
    ) -> None:
        self.rank = rank
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.n_roundings = n_roundings
        self.seed = seed
        self.time_budget = time_budget

    def _default_rank(self, n: int) -> int:
        """The Barvinok-Pataki bound: rank ~ sqrt(2n) suffices for the SDP
        optimum, and is the natural point of comparison with PCE's O(sqrt(n))
        qubit count."""
        return max(1, min(n, int(math.ceil(math.sqrt(2 * n)))))

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        import time as _time

        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        if n == 0:
            return SolverResult(
                bitstring=(),
                model_energy=poly.constant,
                wall_time=0.0,
                resource_dict={"n_vars": 0, "rank": 0},
                solver_metadata={"proven_optimal": True},
                seed=self.seed,
                solver_name=self.name,
            )

        J, h, offset = to_ising(poly)
        k = self.rank or self._default_rank(n)
        rng = np.random.default_rng(self.seed)

        best_energy = math.inf
        best_bits: tuple[int, ...] = tuple([0] * n)
        deadline = (
            _time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else math.inf
        )
        steps_done = 0

        with timed() as elapsed:
            for _ in range(self.n_restarts):
                # Unit vectors on the sphere in R^k, plus a fixed reference
                # vector carrying the linear term.
                V = rng.normal(size=(n, k))
                V /= np.linalg.norm(V, axis=1, keepdims=True)
                ref = np.zeros(k)
                ref[0] = 1.0

                for _step in range(self.n_steps):
                    # d/dV of  sum_ij J_ij <v_i,v_j> + sum_i h_i <v_i,ref>
                    grad = 2.0 * (J @ V) + h[:, None] * ref[None, :]
                    # Project onto the tangent space of the sphere, then retract.
                    radial = np.sum(grad * V, axis=1, keepdims=True) * V
                    V -= self.learning_rate * (grad - radial)
                    V /= np.linalg.norm(V, axis=1, keepdims=True)
                    steps_done += 1
                    if _time.perf_counter() > deadline:
                        break

                # Random-hyperplane rounding, plus the reference direction.
                for r in range(self.n_roundings):
                    plane = ref if r == 0 else rng.normal(size=k)
                    s = np.sign(V @ plane)
                    s[s == 0] = 1.0
                    e = ising_energy(J, h, offset, s)
                    if e < best_energy:
                        best_energy, best_bits = e, spins_to_bits(s)

                if _time.perf_counter() > deadline:
                    break

        # Score through the model itself, not the Ising surrogate.
        final = poly.energy(best_bits)

        return SolverResult(
            bitstring=best_bits,
            model_energy=final,
            wall_time=elapsed[0],
            resource_dict={
                "n_vars": n,
                "rank": k,
                "compression_ratio": n / k,
                "gradient_steps": steps_done,
                "function_evaluations": steps_done + self.n_restarts * self.n_roundings,
            },
            solver_metadata={
                "proven_optimal": False,
                "ising_energy": best_energy,
                "use_penalties": use_penalties,
            },
            seed=self.seed,
            solver_name=self.name,
        )
