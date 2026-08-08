"""Pauli Correlation Encoding (Sciorilli et al., Nature Communications, 2025).

Plan section 2.4.  PCE encodes ``n`` binary variables in the **signs of k-body
Pauli correlators** on ``m`` qubits rather than one qubit per variable.  With
``C(m,k) * 3^k >= n`` the qubit count scales as ``O(n^(1/k))`` -- roughly
``sqrt(n)`` at ``k = 2``.  Each variable ``a`` owns a distinct Pauli string
``P_a``, and is read out as ``x_a = 1 if <P_a> > 0 else 0``.

Training uses the relaxed surrogate of the paper: replace each binary variable
by ``(1 + tanh(alpha * <P_a>)) / 2``, which is differentiable and saturates
toward 0/1, evaluate the *actual polynomial objective* on those relaxed values,
and minimise over the ansatz parameters.  The state is prepared by a
hardware-efficient brick-wall ansatz of RY rotations and CZ entanglers.

**Mandatory honesty clause (plan section 2.4).**  PCE's relaxation is closely
related to classical low-rank / Burer-Monteiro SDP relaxations, and there is an
active argument in the literature that it is dequantizable.  We raise this
ourselves and benchmark against :mod:`rnaqopt.solvers.lowrank` at comparable
rank.  If PCE does not beat that baseline, the honest reading is that the
quantum encoding buys nothing here -- and reporting that is worth more than a
comparison quietly omitted.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ..model.bundle import StemModel
from ..simulator import apply_1q, apply_2q, pauli_rotation
from .base import SolverResult, timed

#: Non-identity single-qubit Paulis available at each position of a string.
_AXES = ("X", "Y", "Z")


def min_qubits(n_vars: int, k: int = 2) -> int:
    """Smallest ``m`` with ``C(m,k) * 3^k >= n_vars``."""
    if n_vars <= 0:
        return 0
    m = k
    while math.comb(m, k) * (3**k) < n_vars:
        m += 1
    return m


def assign_paulis(n_vars: int, m: int, k: int = 2) -> list[tuple[str, tuple[int, ...]]]:
    """Assign a distinct ``k``-body Pauli string to each variable.

    Deterministic: qubit subsets in lexicographic order, axis combinations in
    fixed order, so a rerun reproduces the same encoding exactly.
    """
    assignments: list[tuple[str, tuple[int, ...]]] = []
    for qubits in itertools.combinations(range(m), k):
        for axes in itertools.product(_AXES, repeat=k):
            assignments.append(("".join(axes), qubits))
            if len(assignments) == n_vars:
                return assignments
    raise ValueError(
        f"cannot encode {n_vars} variables in {m} qubits at k={k}: "
        f"capacity is {math.comb(m, k) * 3**k}"
    )


def _apply_pauli_operator(
    psi: np.ndarray, pauli: str, qubits: tuple[int, ...], m: int
) -> np.ndarray:
    """Apply the Pauli operator (not its rotation): ``P = i * exp(-i*(pi/2)*P)``."""
    u = pauli_rotation(pauli, -math.pi / 2) * (-1j)
    if len(qubits) == 1:
        return apply_1q(psi, u, qubits[0], m)
    return apply_2q(psi, u, qubits[0], qubits[1], m)


def brickwall_state(params: np.ndarray, m: int, depth: int) -> np.ndarray:
    """Hardware-efficient brick-wall ansatz: RY layers with CZ entanglers."""
    psi = np.zeros(2**m, dtype=complex)
    psi[0] = 1.0
    idx = 0
    for layer in range(depth):
        for q in range(m):
            u = pauli_rotation("Y", params[idx] / 2.0)
            psi = apply_1q(psi, u, q, m)
            idx += 1
        # Brick-wall CZ pattern: even pairs on even layers, odd on odd.
        for q in range(layer % 2, m - 1, 2):
            psi = _apply_cz(psi, q, q + 1, m)
    for q in range(m):
        u = pauli_rotation("Y", params[idx] / 2.0)
        psi = apply_1q(psi, u, q, m)
        idx += 1
    return psi


def _apply_cz(psi: np.ndarray, q0: int, q1: int, m: int) -> np.ndarray:
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    return apply_2q(psi, cz, q0, q1, m)


def n_brickwall_params(m: int, depth: int) -> int:
    return m * (depth + 1)


class PCESolver:
    """Pauli Correlation Encoding with a relaxed tanh objective."""

    name = "pce"

    def __init__(
        self,
        k: int = 2,
        depth: int = 3,
        alpha: float = 3.0,
        n_restarts: int = 4,
        maxiter: int = 600,
        seed: int = 0,
        max_qubits: int = 14,
        time_budget: float | None = None,
    ) -> None:
        self.time_budget = time_budget
        self.k = k
        self.depth = depth
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.seed = seed
        self.max_qubits = max_qubits

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        from scipy.optimize import minimize

        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        if n == 0:
            return SolverResult(
                bitstring=(),
                model_energy=poly.constant,
                wall_time=0.0,
                resource_dict={"n_vars": 0, "n_qubits": 0},
                solver_metadata={},
                seed=self.seed,
                solver_name=self.name,
            )

        m = min_qubits(n, self.k)
        if m > self.max_qubits:
            raise ValueError(f"PCE needs {m} qubits, above max_qubits={self.max_qubits}")
        paulis = assign_paulis(n, m, self.k)
        n_params = n_brickwall_params(m, self.depth)
        rng = np.random.default_rng(self.seed)

        def correlators(params: np.ndarray) -> np.ndarray:
            psi = brickwall_state(params, m, self.depth)
            return np.array(
                [
                    float(
                        np.real(
                            np.vdot(psi, _apply_pauli_operator(psi, p, q, m))
                        )
                    )
                    for p, q in paulis
                ]
            )

        def relaxed_loss(params: np.ndarray) -> float:
            c = correlators(params)
            x = 0.5 * (1.0 + np.tanh(self.alpha * c))
            total = 0.0
            for key, coeff in poly.terms.items():
                if not key:
                    total += coeff
                    continue
                prod = 1.0
                for v in key:
                    prod *= x[v]
                total += coeff * prod
            return float(total)

        best_bits = tuple([0] * n)
        best_energy = math.inf
        evaluations = 0

        # Matched-budget protocol (plan section 4.6).
        import time as _time

        deadline = (
            _time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else math.inf
        )

        with timed() as elapsed:
            for _ in range(self.n_restarts):
                if _time.perf_counter() > deadline:
                    break
                p0 = rng.uniform(-math.pi, math.pi, size=n_params)
                res = minimize(
                    relaxed_loss,
                    p0,
                    method="COBYLA",
                    options={"maxiter": self.maxiter, "rhobeg": 0.5},
                )
                evaluations += int(res.nfev) if hasattr(res, "nfev") else self.maxiter
                c = correlators(np.asarray(res.x, dtype=float))
                bits = tuple(1 if v > 0 else 0 for v in c)
                e = poly.energy(bits)
                if e < best_energy:
                    best_energy, best_bits = e, bits

        capacity = math.comb(m, self.k) * (3**self.k)
        return SolverResult(
            bitstring=best_bits,
            model_energy=best_energy,
            wall_time=elapsed[0],
            resource_dict={
                "n_vars": n,
                "n_qubits": m,
                "compression_ratio": round(n / m, 4),
                "k_body": self.k,
                "encoding_capacity": capacity,
                "n_parameters": n_params,
                "ansatz_depth": self.depth,
                "function_evaluations": evaluations,
                # A direct-encoding QAOA would need one qubit per variable.
                "qubits_saved_vs_direct": n - m,
            },
            solver_metadata={
                "proven_optimal": False,
                "alpha": self.alpha,
                "restarts": self.n_restarts,
                "use_penalties": use_penalties,
            },
            seed=self.seed,
            solver_name=self.name,
        )
