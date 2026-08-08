"""Quantum resource accounting: qubits, depth, two-qubit gates, parameters.

Challenge task 6 and plan section 4.7 both require this, and it is one of the
eight judging criteria.  Resources are counted **analytically** from the gate
sequence an ansatz represents, rather than by transpiling a circuit, so the
numbers are deterministic and independent of any transpiler's heuristics.

Gate-count conventions (standard textbook decompositions):

* a ``k``-body Pauli-Z rotation ``exp(-i g Z_{q1}...Z_{qk})`` costs
  ``2(k-1)`` CNOTs and one RZ -- a CNOT ladder in, a rotation, a ladder out;
* a ``k``-body rotation over general Paulis adds basis-change single-qubit
  gates (H for X, S-dagger+H for Y) before and after the ladder;
* depth is computed by greedy as-soon-as-possible scheduling on **all-to-all**
  connectivity, which is the optimistic bound; the hardware-connectivity
  penalty is a separate, larger number and is not claimed here.

Because our cost Hamiltonian is diagonal, a Level-2 cubic term is a genuine
three-body Z rotation costing 4 CNOTs.  That is the number the Dirac-3
comparison in Phase 5 is set against: on a gate-based device, degree-3 fidelity
is paid for in CNOTs (or in ancillas after quadratization), whereas Dirac-3
supports it natively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Single-qubit gates needed to rotate a Pauli into the Z basis, per axis.
_BASIS_CHANGE = {"X": 1, "Y": 2, "Z": 0, "I": 0}


@dataclass(frozen=True)
class Gate:
    """One scheduled gate: a name and the qubits it touches."""

    name: str
    qubits: tuple[int, ...]

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


@dataclass
class CircuitResources:
    """Countable resources of a circuit."""

    n_qubits: int = 0
    n_layers: int = 0
    depth: int = 0
    n_two_qubit_gates: int = 0
    n_single_qubit_gates: int = 0
    n_parameters: int = 0
    n_ancillas: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_gates(self) -> int:
        return self.n_two_qubit_gates + self.n_single_qubit_gates

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        extra = out.pop("extra")
        out["n_gates"] = self.n_gates
        out.update(extra)
        return out


def pauli_rotation_gates(pauli: str, qubits: tuple[int, ...]) -> list[Gate]:
    """Gate list for ``exp(-i theta P)`` with ``P`` acting on ``qubits``.

    Basis changes, a CNOT ladder onto the last qubit, the parametrised RZ, then
    the ladder and basis changes undone.
    """
    if len(pauli) != len(qubits):
        raise ValueError("pauli string and qubit list must have equal length")
    gates: list[Gate] = []

    for axis, q in zip(pauli, qubits, strict=True):
        gates.extend(Gate(f"basis_{axis}", (q,)) for _ in range(_BASIS_CHANGE[axis]))
    for a, b in zip(qubits, qubits[1:], strict=False):
        gates.append(Gate("cx", (a, b)))
    gates.append(Gate("rz", (qubits[-1],)))
    for a, b in reversed(list(zip(qubits, qubits[1:], strict=False))):
        gates.append(Gate("cx", (a, b)))
    for axis, q in zip(pauli, qubits, strict=True):
        gates.extend(Gate(f"basis_{axis}", (q,)) for _ in range(_BASIS_CHANGE[axis]))
    return gates


def cost_layer_gates(terms: dict[tuple[int, ...], float]) -> list[Gate]:
    """Gate list for ``exp(-i gamma H_C)`` with diagonal ``H_C``.

    Every non-constant term becomes a Pauli-Z rotation over its variables. Term
    order maps directly to gate cost: linear terms are free of CNOTs, quadratic
    terms cost 2, cubic terms cost 4.
    """
    gates: list[Gate] = []
    for key in sorted(terms):
        if not key:
            continue  # a constant is a global phase
        gates.extend(pauli_rotation_gates("Z" * len(key), key))
    return gates


def schedule_depth(gates: list[Gate], n_qubits: int) -> int:
    """As-soon-as-possible depth on all-to-all connectivity."""
    if not gates:
        return 0
    ready = [0] * max(n_qubits, 1)
    depth = 0
    for gate in gates:
        start = max(ready[q] for q in gate.qubits)
        finish = start + 1
        for q in gate.qubits:
            ready[q] = finish
        depth = max(depth, finish)
    return depth


def count(gates: list[Gate], n_qubits: int, n_parameters: int = 0) -> CircuitResources:
    """Summarise a gate list."""
    return CircuitResources(
        n_qubits=n_qubits,
        depth=schedule_depth(gates, n_qubits),
        n_two_qubit_gates=sum(1 for g in gates if g.is_two_qubit),
        n_single_qubit_gates=sum(1 for g in gates if not g.is_two_qubit),
        n_parameters=n_parameters,
    )


def adapt_qaoa_resources(
    terms: dict[tuple[int, ...], float],
    mixers: list[tuple[str, tuple[int, ...]]],
    n_qubits: int,
) -> CircuitResources:
    """Resources of an ADAPT-QAOA ansatz with the given selected mixers.

    One cost layer and one mixer layer per entry in ``mixers``; two variational
    parameters per layer.
    """
    gates: list[Gate] = []
    for pauli, qubits in mixers:
        gates.extend(cost_layer_gates(terms))
        gates.extend(pauli_rotation_gates(pauli, qubits))

    res = count(gates, n_qubits, n_parameters=2 * len(mixers))
    res.n_layers = len(mixers)
    res.extra["cost_layer_two_qubit_gates"] = sum(
        1 for g in cost_layer_gates(terms) if g.is_two_qubit
    )
    res.extra["cost_layer_depth"] = schedule_depth(cost_layer_gates(terms), n_qubits)
    return res


def term_order_histogram(terms: dict[tuple[int, ...], float]) -> dict[int, int]:
    """How many terms of each order, i.e. where the gate cost comes from."""
    hist: dict[int, int] = {}
    for key in terms:
        hist[len(key)] = hist.get(len(key), 0) + 1
    return dict(sorted(hist.items()))


def two_qubit_cost_of_degree(terms: dict[tuple[int, ...], float]) -> dict[int, int]:
    """CNOT count contributed by each term order in one cost layer.

    The headline number for the Phase 5 comparison: how many CNOTs a gate-based
    device spends specifically on the degree-3 terms that Dirac-3 executes
    natively.
    """
    out: dict[int, int] = {}
    for key in terms:
        k = len(key)
        if k < 2:
            continue
        out[k] = out.get(k, 0) + 2 * (k - 1)
    return dict(sorted(out.items()))
