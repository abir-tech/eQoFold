"""ADAPT-QAOA -- adaptive mixer selection with a CVaR objective.

Plan section 2.5 selects ADAPT-QAOA as the gate-based baseline, and explicitly
rejects vanilla ADAPT-VQE: its fermionic/UCCSD operator pool targets molecular
electronic structure, whereas our Hamiltonian is diagonal in the computational
basis.  What adapts here is the **mixer**, not an excitation ansatz.

Plan section 4.6 specifies the algorithm:

* mixer pool ``{X_i} u {Y_i} u {X_iX_j, Y_iY_j, Y_iZ_j for (i,j) in conflict}``
* at each layer, select the mixer with the largest energy gradient
  ``dE/dbeta|_0 = <psi| i[H_C, A] |psi>``, append it, and re-optimise *all*
  parameters
* terminate on a gradient threshold or a maximum layer count
* record per layer: depth, two-qubit gate count, parameter count, energy
* use CVaR (alpha ~ 0.1-0.25) as the objective

Restricting the two-body mixers to conflict-graph edges is the point of using
an adaptive pool on this problem: the conflict graph is exactly where the
constraint penalties live, so those are the correlations the mixer needs to
move population across.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model.bundle import StemModel
from ..resources import adapt_qaoa_resources
from ..simulator import (
    apply_diagonal_phase,
    apply_pauli_rotation,
    best_bitstring,
    cvar,
    diagonal_from_model,
    expectation,
    plus_state,
)
from .base import SolverResult, timed

#: A mixer is a Pauli string plus the qubits it acts on.
Mixer = tuple[str, tuple[int, ...]]


def build_mixer_pool(model: StemModel, max_two_body: int | None = None) -> list[Mixer]:
    """The pool of plan section 4.6.

    ``max_two_body`` caps the number of conflict edges contributing two-body
    mixers, which keeps the per-layer gradient scan affordable on the larger
    instances. Edges are taken in sorted order so the choice is deterministic.
    """
    n = model.n_vars
    pool: list[Mixer] = [("X", (i,)) for i in range(n)]
    pool += [("Y", (i,)) for i in range(n)]

    edges = sorted(model.graphs.conflict)
    if max_two_body is not None:
        edges = edges[:max_two_body]
    for i, j in edges:
        pool.append(("XX", (i, j)))
        pool.append(("YY", (i, j)))
        pool.append(("YZ", (i, j)))
        pool.append(("YZ", (j, i)))
    return pool


def _apply_pauli(psi: np.ndarray, pauli: str, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply the Pauli operator itself (not its rotation)."""
    from ..simulator import apply_1q, apply_2q, pauli_rotation

    # exp(-i * (-pi/2) * P) = cos(pi/2) I + i sin(pi/2) P = i P, so P = -i * U.
    u = pauli_rotation(pauli, -math.pi / 2) * (-1j)
    if len(qubits) == 1:
        return apply_1q(psi, u, qubits[0], n)
    return apply_2q(psi, u, qubits[0], qubits[1], n)


def mixer_gradient(
    psi: np.ndarray, diag: np.ndarray, pauli: str, qubits: tuple[int, ...], n: int
) -> float:
    """``dE/dbeta`` at ``beta = 0`` for appending ``exp(-i beta A)``.

    ``E(beta) = <psi| e^{i beta A} H e^{-i beta A} |psi>`` gives
    ``dE/dbeta|_0 = i<psi|[A, H]|psi> = -2 Im<A psi | H psi>``.
    """
    a_psi = _apply_pauli(psi, pauli, qubits, n)
    h_psi = diag * psi
    return float(-2.0 * np.imag(np.vdot(a_psi, h_psi)))


class AdaptQAOASolver:
    """ADAPT-QAOA with CVaR, exact statevector simulation."""

    name = "adapt_qaoa"

    def __init__(
        self,
        max_layers: int = 8,
        alpha: float = 0.15,
        gradient_tol: float = 1e-4,
        gamma_init: float = 0.01,
        maxiter: int = 300,
        seed: int = 0,
        max_qubits: int = 20,
        max_two_body: int | None = 200,
        shots: int = 1024,
        time_budget: float | None = None,
    ) -> None:
        self.shots = shots
        self.max_layers = max_layers
        self.alpha = alpha
        self.gradient_tol = gradient_tol
        self.gamma_init = gamma_init
        self.maxiter = maxiter
        self.seed = seed
        self.max_qubits = max_qubits
        self.max_two_body = max_two_body
        self.time_budget = time_budget

    # -- ansatz ------------------------------------------------------------

    def _state(
        self, params: np.ndarray, mixers: list[Mixer], diag: np.ndarray, n: int
    ) -> np.ndarray:
        psi = plus_state(n)
        for layer, (pauli, qubits) in enumerate(mixers):
            psi = apply_diagonal_phase(psi, diag, params[2 * layer])
            psi = apply_pauli_rotation(psi, pauli, qubits, params[2 * layer + 1], n)
        return psi

    # -- solve -------------------------------------------------------------

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        from scipy.optimize import minimize

        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        if n > self.max_qubits:
            raise ValueError(
                f"ADAPT-QAOA refused: {n} qubits exceeds max_qubits="
                f"{self.max_qubits}. Exact statevector simulation needs "
                f"2^{n} amplitudes."
            )
        if n == 0:
            return SolverResult(
                bitstring=(),
                model_energy=poly.constant,
                wall_time=0.0,
                resource_dict={"n_qubits": 0},
                solver_metadata={"layers": 0},
                seed=self.seed,
                solver_name=self.name,
            )

        diag = diagonal_from_model(poly.terms, n, 0.0)
        pool = build_mixer_pool(model, self.max_two_body)

        mixers: list[Mixer] = []
        params = np.zeros(0)
        history: list[dict[str, Any]] = []
        n_evaluations = 0
        stop_reason = "max_layers"

        def objective(p: np.ndarray) -> float:
            nonlocal n_evaluations
            n_evaluations += 1
            return cvar(self._state(p, mixers, diag, n), diag, self.alpha)

        # Matched-budget protocol (plan section 4.6): stop adding layers once
        # the shared wall-clock budget is spent, so this solver is compared on
        # the same footing as the classical baselines rather than being allowed
        # to run longer.
        import time as _time

        deadline = (
            _time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else math.inf
        )

        with timed() as elapsed:
            for layer in range(self.max_layers):
                if _time.perf_counter() > deadline:
                    stop_reason = "time_budget"
                    break
                psi = self._state(params, mixers, diag, n)
                # A trial cost layer first: the gradient of a mixer is zero on a
                # state with no phase structure, so the standard formulation
                # evaluates it after applying exp(-i gamma H_C).
                psi_trial = apply_diagonal_phase(psi, diag, self.gamma_init)

                grads = [
                    abs(mixer_gradient(psi_trial, diag, pauli, qubits, n))
                    for pauli, qubits in pool
                ]
                best = int(np.argmax(grads))
                if grads[best] < self.gradient_tol:
                    stop_reason = "gradient_below_tolerance"
                    break

                mixers.append(pool[best])
                params = np.concatenate([params, [self.gamma_init, 0.0]])

                result = minimize(
                    objective,
                    params,
                    method="COBYLA",
                    options={"maxiter": self.maxiter, "rhobeg": 0.3},
                )
                params = np.asarray(result.x, dtype=float)

                psi_now = self._state(params, mixers, diag, n)
                res = adapt_qaoa_resources(poly.terms, mixers, n)
                history.append(
                    {
                        "layer": layer + 1,
                        "mixer": f"{pool[best][0]}{list(pool[best][1])}",
                        "gradient": grads[best],
                        "cvar": float(result.fun),
                        "expectation": expectation(psi_now, diag),
                        "best_energy": float(diag[np.argmin(diag)]),
                        "depth": res.depth,
                        "two_qubit_gates": res.n_two_qubit_gates,
                        "n_parameters": res.n_parameters,
                    }
                )

        psi_final = self._state(params, mixers, diag, n) if mixers else plus_state(n)
        resources = adapt_qaoa_resources(poly.terms, mixers, n)

        probs = np.abs(psi_final) ** 2
        optimum = float(diag.min())
        success_probability = float(probs[diag <= optimum + 1e-9].sum())

        # Readout is by *finite sampling*, as on hardware.  Scanning the whole
        # statevector for its lowest-energy amplitude would be reading the
        # answer out of the diagonal rather than out of the circuit, and would
        # make the solver look exact on every instance regardless of how badly
        # the variational state was prepared.  The statevector argmin is kept
        # only as a clearly-labelled upper bound.
        rng = np.random.default_rng(self.seed)
        samples = rng.choice(len(probs), size=self.shots, p=probs / probs.sum())
        best_sample = int(samples[np.argmin(diag[samples])])
        bits = tuple((best_sample >> q) & 1 for q in range(n))
        unique_sampled = int(np.unique(samples).size)
        ideal_bits = best_bitstring(psi_final, diag, n)

        return SolverResult(
            bitstring=bits,
            model_energy=poly.energy(bits),
            wall_time=elapsed[0],
            resource_dict={
                **resources.as_dict(),
                "function_evaluations": n_evaluations,
                "pool_size": len(pool),
                "shots": self.shots,
            },
            solver_metadata={
                "proven_optimal": False,
                "layers": len(mixers),
                "stop_reason": stop_reason,
                "alpha": self.alpha,
                "final_cvar": cvar(psi_final, diag, self.alpha),
                "expectation": expectation(psi_final, diag),
                "success_probability": success_probability,
                "unique_bitstrings_sampled": unique_sampled,
                # Upper bound only: the best state in the *entire* statevector,
                # which finite sampling generally will not reach.
                "statevector_argmin_energy": poly.energy(ideal_bits),
                "history": history,
                "use_penalties": use_penalties,
                # The prepared state and its diagonal, so downstream studies
                # (the Phase 7 noise sweep) can resample the *same* circuit
                # output under different noise models without re-optimising.
                "final_state": psi_final,
                "diagonal": diag,
            },
            seed=self.seed,
            solver_name=self.name,
        )
