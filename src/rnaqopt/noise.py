"""Sampling and hardware-inspired noise models.

Challenge optional advanced task: *evaluate the approach under sampling or
hardware-inspired noise.*  Plan section 7 orders this deliberately -- finite
sampling first, then depolarizing and readout error -- because finite sampling
is not really "noise" at all: it is present on every real device even a perfect
one, and it is usually the dominant effect at the circuit sizes reachable here.

Three effects, applied at the readout stage of an ideal statevector:

``finite sampling``   draw a finite number of shots from |psi|^2
``depolarizing``      global depolarizing channel: with probability ``lam`` the
                      outcome is drawn uniformly instead of from the circuit
``readout error``     each measured bit flips independently with probability p

The global depolarizing channel is the right level of detail here: our cost
Hamiltonian is diagonal, so only the *measurement distribution* matters, and a
global channel captures the leading effect of accumulated gate error on that
distribution without pretending to a device-specific error model we have not
calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Readout-stage noise parameters."""

    #: Measurement shots. ``None`` means the exact distribution (no sampling).
    shots: int | None = 1024
    #: Global depolarizing probability in [0, 1].
    depolarizing: float = 0.0
    #: Independent per-bit readout flip probability in [0, 1].
    readout_error: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("depolarizing", self.depolarizing),
            ("readout_error", self.readout_error),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"shots must be >= 1, got {self.shots}")

    @property
    def is_ideal(self) -> bool:
        return (
            self.shots is None
            and self.depolarizing == 0.0
            and self.readout_error == 0.0
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "shots": self.shots,
            "depolarizing": self.depolarizing,
            "readout_error": self.readout_error,
        }


def noisy_distribution(psi: np.ndarray, model: NoiseModel) -> np.ndarray:
    """Measurement distribution after the global depolarizing channel."""
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    if model.depolarizing > 0.0:
        uniform = np.full_like(probs, 1.0 / len(probs))
        probs = (1.0 - model.depolarizing) * probs + model.depolarizing * uniform
    return probs


def sample_outcomes(
    psi: np.ndarray, n_qubits: int, model: NoiseModel, rng: np.random.Generator
) -> np.ndarray:
    """Draw measurement outcomes as basis-state indices, with noise applied."""
    probs = noisy_distribution(psi, model)
    shots = model.shots if model.shots is not None else 1
    outcomes = rng.choice(len(probs), size=shots, p=probs)

    if model.readout_error > 0.0:
        flips = rng.random((shots, n_qubits)) < model.readout_error
        for q in range(n_qubits):
            mask = flips[:, q].astype(np.int64) << q
            outcomes = outcomes ^ mask
    return outcomes


def best_under_noise(
    psi: np.ndarray,
    diag: np.ndarray,
    n_qubits: int,
    model: NoiseModel,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], float, dict[str, float]]:
    """Best bitstring obtainable from noisy finite sampling.

    Returns the bitstring, its energy, and diagnostics. If ``shots`` is ``None``
    the exact distribution is used and the lowest-energy supported state is
    returned, which is the noiseless upper bound.
    """
    if model.shots is None and model.depolarizing == 0.0 and model.readout_error == 0.0:
        best = int(np.argmin(diag))
        bits = tuple((best >> q) & 1 for q in range(n_qubits))
        return bits, float(diag[best]), {"unique_outcomes": 1.0}

    outcomes = sample_outcomes(psi, n_qubits, model, rng)
    energies = diag[outcomes]
    best = int(outcomes[int(np.argmin(energies))])
    bits = tuple((best >> q) & 1 for q in range(n_qubits))
    return (
        bits,
        float(diag[best]),
        {
            "unique_outcomes": float(np.unique(outcomes).size),
            "mean_sampled_energy": float(energies.mean()),
        },
    )


def success_probability(
    psi: np.ndarray, diag: np.ndarray, model: NoiseModel, tol: float = 1e-9
) -> float:
    """Probability of measuring an optimal bitstring under the noise model."""
    probs = noisy_distribution(psi, model)
    return float(probs[diag <= diag.min() + tol].sum())


def shots_for_target_success(
    p_single: float, target: float = 0.99, max_shots: int = 10**9
) -> int:
    """Shots needed so at least one of them is optimal, with probability ``target``.

    ``1 - (1 - p)^N >= target``. The headline sampling-cost number: it converts
    a per-shot success probability into the repetition count a device would
    actually have to pay.
    """
    import math

    if p_single <= 0.0:
        return max_shots
    if p_single >= 1.0:
        return 1
    n = math.ceil(math.log(1.0 - target) / math.log(1.0 - p_single))
    return int(min(max(n, 1), max_shots))
