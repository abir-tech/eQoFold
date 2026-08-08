"""Shallow classical local search over returned samples.

Adopted from the published state of the art: Kumar, Alevras, Metkar et al.
(arXiv:2505.05782) pass hardware samples through "a shallow local search on
classical nodes" before reporting. Not doing so makes our numbers
systematically pessimistic relative to theirs, so the comparison would be
unfair in the other direction.

**Reported separately, always.** Like the repair step of :mod:`rnaqopt.decode`,
this is classical post-processing bolted onto a quantum result. Folding it
silently into a solver's score would inflate apparent solver quality -- the
exact failure mode plan section 6 warns about. Every
:class:`LocalSearchResult` therefore carries the energy *before* as well as
after, and the experiment layer reports both columns.

The search is deterministic 1-opt hill climbing: sweep variables in index
order, flip any single variable that strictly lowers the objective, repeat
until a full sweep finds no improvement. No randomness, no seed, no tuning
knobs that could be quietly optimised against the benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .model.base import PolynomialModel


@dataclass(frozen=True)
class LocalSearchResult:
    """Outcome of local search on one bitstring."""

    bitstring: tuple[int, ...]
    energy_before: float
    energy_after: float
    n_flips: int
    n_passes: int
    n_delta_evaluations: int

    @property
    def improved(self) -> bool:
        return self.n_flips > 0

    @property
    def improvement(self) -> float:
        """How much the objective fell. Non-negative by construction."""
        return self.energy_before - self.energy_after

    def as_dict(self) -> dict[str, object]:
        return {
            "ls_energy_before": self.energy_before,
            "ls_energy_after": self.energy_after,
            "ls_improvement": self.improvement,
            "ls_n_flips": self.n_flips,
            "ls_n_passes": self.n_passes,
            "ls_improved": self.improved,
        }


def _terms_by_variable(
    model: PolynomialModel,
) -> list[list[tuple[tuple[int, ...], float]]]:
    """For each variable, the terms containing it. Built once per search."""
    index: list[list[tuple[tuple[int, ...], float]]] = [
        [] for _ in range(model.n_vars)
    ]
    for key, coeff in model.terms.items():
        for v in key:
            index[v].append((key, coeff))
    return index


def flip_delta(
    bits: Sequence[int],
    var: int,
    terms: Sequence[tuple[tuple[int, ...], float]],
) -> float:
    """Change in objective from flipping ``var``, without re-evaluating the whole
    polynomial.

    A term containing ``var`` fires exactly when every *other* variable in it is
    already 1. Turning ``var`` on adds those coefficients; turning it off
    removes them.
    """
    sign = -1.0 if bits[var] else 1.0
    total = 0.0
    for key, coeff in terms:
        if all(bits[u] for u in key if u != var):
            total += coeff
    return sign * total


def local_search(
    model: PolynomialModel,
    bits: Sequence[int],
    max_passes: int = 64,
    tolerance: float = 1e-12,
) -> LocalSearchResult:
    """Deterministic 1-opt hill climbing on ``model``.

    Returns the improved bitstring together with the energy it started from, so
    the caller can report the raw and post-processed numbers side by side.
    """
    current = list(bits)
    if len(current) != model.n_vars:
        raise ValueError(
            f"bitstring length {len(current)} != model.n_vars {model.n_vars}"
        )

    index = _terms_by_variable(model)
    energy_before = model.energy(current)
    energy = energy_before
    flips = 0
    evaluations = 0
    passes = 0

    for _ in range(max_passes):
        passes += 1
        moved = False
        for v in range(model.n_vars):
            delta = flip_delta(current, v, index[v])
            evaluations += 1
            if delta < -tolerance:
                current[v] ^= 1
                energy += delta
                flips += 1
                moved = True
        if not moved:
            break

    return LocalSearchResult(
        bitstring=tuple(current),
        energy_before=energy_before,
        energy_after=model.energy(current),
        n_flips=flips,
        n_passes=passes,
        n_delta_evaluations=evaluations,
    )


def improve_samples(
    model: PolynomialModel,
    samples: Sequence[Sequence[int]],
) -> LocalSearchResult:
    """Run local search on several samples and keep the best outcome.

    ``energy_before`` is the best energy among the *raw* samples, so the
    reported improvement is measured against what the solver actually returned
    rather than against whichever sample happened to polish up best.
    """
    if not samples:
        raise ValueError("no samples to improve")

    raw_best = min(model.energy(s) for s in samples)
    best: LocalSearchResult | None = None
    total_flips = 0
    total_evals = 0
    total_passes = 0
    for sample in samples:
        result = local_search(model, sample)
        total_flips += result.n_flips
        total_evals += result.n_delta_evaluations
        total_passes += result.n_passes
        if best is None or result.energy_after < best.energy_after:
            best = result
    assert best is not None

    return LocalSearchResult(
        bitstring=best.bitstring,
        energy_before=raw_best,
        energy_after=best.energy_after,
        n_flips=total_flips,
        n_passes=total_passes,
        n_delta_evaluations=total_evals,
    )
