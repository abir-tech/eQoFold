"""The solver protocol and its uniform result type.

Plan section 3, non-negotiable engineering rule: *every solver returns a
``SolverResult`` dataclass:* ``bitstring``, ``model_energy``, ``wall_time``,
``resource_dict``, ``solver_metadata``, ``seed``.

Every solver in this project -- brute force, CP-SAT, simulated annealing,
low-rank, ADAPT-QAOA, PCE, Dirac-3 -- consumes the same :class:`StemModel` and
returns the same result type, so the optimizer gap is computed identically for
all of them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..model.bundle import StemModel


@dataclass
class SolverResult:
    """What every solver returns."""

    #: Assignment over stem variables, in variable order.
    bitstring: tuple[int, ...]
    #: Objective value of ``bitstring`` under the model the solver was given.
    model_energy: float
    #: Wall-clock seconds spent solving (excludes model construction).
    wall_time: float
    #: Countable resources: qubits, ancillas, depth, gates, evaluations, shots.
    resource_dict: dict[str, Any] = field(default_factory=dict)
    #: Free-form solver-specific information, including convergence status.
    solver_metadata: dict[str, Any] = field(default_factory=dict)
    #: RNG seed, or ``None`` for deterministic solvers.
    seed: int | None = None
    #: Name of the solver that produced this result.
    solver_name: str = ""

    @property
    def selection(self) -> tuple[int, ...]:
        """Indices of the selected stems."""
        return tuple(i for i, b in enumerate(self.bitstring) if b)

    @property
    def n_selected(self) -> int:
        return sum(1 for b in self.bitstring if b)

    @property
    def is_proven_optimal(self) -> bool:
        """True only when the solver certifies global optimality of the model."""
        return bool(self.solver_metadata.get("proven_optimal", False))

    def as_row(self) -> dict[str, Any]:
        return {
            "solver": self.solver_name,
            "model_energy": self.model_energy,
            "wall_time": self.wall_time,
            "n_selected": self.n_selected,
            "proven_optimal": self.is_proven_optimal,
            "seed": self.seed,
            **{f"res_{k}": v for k, v in self.resource_dict.items()},
        }


@runtime_checkable
class Solver(Protocol):
    """Anything that can minimise a :class:`StemModel`."""

    name: str

    def solve(self, model: StemModel, **kwargs: Any) -> SolverResult:
        """Minimise ``model`` and return the best assignment found."""
        ...


@contextmanager
def timed():
    """Context manager yielding a one-element list that receives the elapsed time.

    ``time.perf_counter`` rather than ``time.time``: the matched-budget protocol
    of plan section 4.6 compares solvers on wall clock, so the clock must be
    monotonic.
    """
    holder: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


def bitstring_from_selection(selection: Sequence[int], n_vars: int) -> tuple[int, ...]:
    """Dense 0/1 tuple from a list of selected variable indices."""
    bits = [0] * n_vars
    for idx in selection:
        bits[idx] = 1
    return tuple(bits)


class InfeasibleModelError(RuntimeError):
    """Raised when a solver proves no feasible assignment exists."""
