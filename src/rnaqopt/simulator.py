"""A small exact statevector simulator, specialised to diagonal cost Hamiltonians.

Our cost Hamiltonian is diagonal in the computational basis, which makes the
cost layer a pure phase multiply and removes the need for a general circuit
simulator.  Keeping this in-house rather than reaching for a full framework buys
two things that matter here:

* speed -- ADAPT-QAOA evaluates the energy tens of thousands of times per
  instance, and a vectorised phase multiply is orders of magnitude faster than
  rebuilding and transpiling a circuit each call;
* determinism -- no transpiler heuristics between the mathematics and the
  reported number.

Circuit *resources* (depth, two-qubit gate count) are counted analytically in
:mod:`rnaqopt.resources` from the same gate sequence, so the resource claims
describe the circuit that these operations represent.

Correctness is pinned in ``tests/test_simulator.py`` against dense matrix
exponentials computed with :func:`scipy.linalg.expm`.

Convention: qubit ``q`` is bit ``q`` of the basis-state index, i.e.
``bit_q(i) = (i >> q) & 1``.
"""

from __future__ import annotations

import numpy as np

#: Single-qubit Pauli matrices.
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def plus_state(n: int) -> np.ndarray:
    """Uniform superposition |+>^{otimes n}, the standard QAOA initial state."""
    return np.full(2**n, 2 ** (-n / 2), dtype=complex)


def apply_1q(psi: np.ndarray, u: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a 2x2 unitary to qubit ``q``."""
    psi = psi.reshape(2 ** (n - 1 - q), 2, 2**q)
    out = np.einsum("ab,ibj->iaj", u, psi, optimize=True)
    return out.reshape(-1)


def apply_2q(psi: np.ndarray, u: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    """Apply a 4x4 unitary to qubits ``q0`` and ``q1``.

    Convention, matching :func:`pauli_rotation`: the **least significant**
    tensor factor of ``u`` is qubit ``q0`` and the most significant is ``q1``.
    So for ``pauli_rotation("XY", t)``, ``X`` acts on ``q0`` and ``Y`` on ``q1``.
    ``q0`` and ``q1`` need not be ordered.
    """
    if q0 == q1:
        raise ValueError("apply_2q needs two distinct qubits")

    # t[o_q1, o_q0, i_q1, i_q0]
    t = u.reshape(2, 2, 2, 2)

    lo, hi = min(q0, q1), max(q0, q1)
    psi = psi.reshape(2 ** (n - 1 - hi), 2, 2 ** (hi - lo - 1), 2, 2**lo)
    # psi axes: (top, bit at qubit `hi`, mid, bit at qubit `lo`, bottom)

    if q0 < q1:
        # q0 is the low-numbered qubit -> psi axis 3; q1 -> psi axis 1
        out = np.einsum("PQRS,iRjSk->iPjQk", t, psi, optimize=True)
    else:
        # q0 is the high-numbered qubit -> psi axis 1; q1 -> psi axis 3
        out = np.einsum("PQRS,iSjRk->iQjPk", t, psi, optimize=True)
    return out.reshape(-1)


def pauli_rotation(pauli: str, theta: float) -> np.ndarray:
    """``exp(-i*theta*P)`` for a Pauli string ``P`` with ``P^2 = I``.

    Uses ``exp(-i t P) = cos(t) I - i sin(t) P``, exact for any involutory P.
    """
    mat = np.array([[1.0]], dtype=complex)
    for ch in pauli:
        mat = np.kron(PAULI[ch], mat)  # first char = lowest-numbered qubit
    dim = mat.shape[0]
    return np.cos(theta) * np.eye(dim, dtype=complex) - 1j * np.sin(theta) * mat


def apply_pauli_rotation(
    psi: np.ndarray, pauli: str, qubits: tuple[int, ...], theta: float, n: int
) -> np.ndarray:
    """Apply ``exp(-i*theta*P)`` where ``P`` acts on ``qubits``."""
    u = pauli_rotation(pauli, theta)
    if len(qubits) == 1:
        return apply_1q(psi, u, qubits[0], n)
    if len(qubits) == 2:
        return apply_2q(psi, u, qubits[0], qubits[1], n)
    raise ValueError(f"only 1- and 2-qubit rotations supported, got {len(qubits)}")


def apply_diagonal_phase(psi: np.ndarray, diag: np.ndarray, gamma: float) -> np.ndarray:
    """Apply ``exp(-i*gamma*H_C)`` for a diagonal ``H_C``."""
    return psi * np.exp(-1j * gamma * diag)


def diagonal_from_model(terms: dict, n: int, constant: float = 0.0) -> np.ndarray:
    """Dense diagonal of a pseudo-Boolean polynomial over ``n`` binary variables.

    Basis state ``i`` encodes ``x_v = (i >> v) & 1``, matching the qubit
    convention above.
    """
    states = np.arange(2**n, dtype=np.int64)
    diag = np.full(2**n, constant, dtype=np.float64)
    for key, coeff in terms.items():
        if not key:
            diag += coeff
            continue
        mask = 0
        for v in key:
            mask |= 1 << v
        diag += coeff * ((states & mask) == mask)
    return diag


def expectation(psi: np.ndarray, diag: np.ndarray) -> float:
    """<psi|H_C|psi> for diagonal ``H_C``."""
    return float(np.sum(np.abs(psi) ** 2 * diag).real)


def cvar(psi: np.ndarray, diag: np.ndarray, alpha: float) -> float:
    """Conditional Value at Risk of the energy distribution.

    Plan section 4.6: *use CVaR (alpha ~ 0.1-0.25) as the objective -- standard
    practice for combinatorial problems and it materially improves results.*
    CVaR averages only the best ``alpha`` fraction of the measured energies, so
    the optimiser is rewarded for putting weight on good bitstrings rather than
    for lowering the mean of the whole distribution.

    ``alpha = 1`` recovers the plain expectation value.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        return expectation(psi, diag)

    probs = np.abs(psi) ** 2
    order = np.argsort(diag, kind="stable")
    sorted_p = probs[order]
    sorted_e = diag[order]
    cumulative = np.cumsum(sorted_p)
    cut = int(np.searchsorted(cumulative, alpha)) + 1
    cut = min(cut, len(sorted_e))
    weights = sorted_p[:cut].copy()
    # Trim the final bucket so the weights sum to exactly alpha.
    excess = weights.sum() - alpha
    if excess > 0:
        weights[-1] -= excess
    total = weights.sum()
    if total <= 0:
        return float(sorted_e[0])
    return float(np.dot(weights, sorted_e[:cut]) / total)


def best_bitstring(psi: np.ndarray, diag: np.ndarray, n: int) -> tuple[int, ...]:
    """Lowest-energy basis state carrying non-negligible amplitude.

    Reading out the best *sampled* state rather than the argmax of the diagonal
    keeps this honest: it must be a state the circuit actually produces.
    """
    probs = np.abs(psi) ** 2
    support = np.flatnonzero(probs > 1e-12)
    if support.size == 0:
        support = np.arange(len(diag))
    best = support[np.argmin(diag[support])]
    return tuple((int(best) >> q) & 1 for q in range(n))


def sample_bitstrings(
    psi: np.ndarray, n: int, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``shots`` measurement outcomes as basis-state indices.

    Used by the finite-sampling noise study in Phase 7.
    """
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    return rng.choice(len(probs), size=shots, p=probs)
