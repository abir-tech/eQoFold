"""Adapter: rna_encoding.QUBOProblem -> the repo's qms.instance_factory.QUBOInstance.

The repo's GQE (qms/gqe) and PCE (qms/pce) machinery both consume a
QUBOInstance duck-typed around a domain-wall/budget siting problem, but every
function they actually call only touches `.m .B .L .c .Q .cost .is_feasible
.budget_used .bus_bit_slices .budget_weights .budget_cap .levels_from_x`. The
project's own MaxCut arm (experiments/sprintX2_maxcut_sciorilli.py) reuses
this machinery unmodified for an unconstrained QUBO by building a
"degenerate" instance: B = m independent length-1 domain-wall chains (so
domain-wall repair is a no-op) and a zero budget vector with zero cap (so
budget repair never triggers). That leaves `joint_neighborhood_search`
falling through to plain single-bit-flip (or paired-bit) local search over
`inst.cost`, which is exactly the generic QUBO local search RNA folding
needs. We follow the same pattern here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qms.instance_factory import QUBOInstance  # noqa: E402

from rna_encoding import QUBOProblem  # noqa: E402


def to_qubo_instance(qp: QUBOProblem, seed: int = 0) -> QUBOInstance:
    m = qp.num_vars
    c = np.zeros(m)
    Q = np.zeros((m, m))
    for (a, b), coeff in qp.Q.items():
        if a == b:
            c[a] += coeff
        else:
            Q[a, b] += coeff / 2.0
            Q[b, a] += coeff / 2.0
    return QUBOInstance(
        m=m, B=m, L=1, candidate_buses=list(range(m)), c=c, Q=Q,
        budget_weights=np.zeros(m), budget_cap=0.0,
        bus_bit_slices=[(i, i + 1) for i in range(m)], seed=seed,
        load_scale=np.ones(1), budget_fraction=1.0,
        meta=dict(kind=qp.kind, n_seq=qp.n_seq, penalty=qp.penalty),
    )
