"""Cubic -> quadratic reduction with ancillas, purely to measure the overhead.

Plan section 4.3: *provide ``quadratize.py`` to reduce Level 2 to a QUBO with
ancilla variables (standard Rosenberg reduction), purely to measure the ancilla
overhead that a gate-based device would pay.  That measured overhead is a
headline number.*

**Rosenberg reduction.**  Replace a product ``x_i x_j`` by a fresh binary
ancilla ``y`` and add the penalty

    M * (x_i x_j - 2 x_i y - 2 x_j y + 3 y)

which is ``0`` when ``y = x_i x_j`` and ``>= M`` otherwise, for ``M > 0``.  A
cubic term ``c * x_i x_j x_k`` then becomes ``c * y x_k`` plus that penalty:
degree 2, at the cost of one ancilla per distinct pair reduced.

Ancillas are shared across every cubic term that contains the same pair, so the
overhead is the number of *distinct* pairs appearing in cubic terms, not the
number of cubic terms.  The pair chosen for each triple is the one that already
has an ancilla where possible, then the globally most frequent pair -- a greedy
cover that measurably beats reducing each triple independently.

This module never feeds Dirac-3.  Dirac-3 takes the degree-3 model natively;
the whole point of the number produced here is what a *gate-based* device would
have to spend to reach the same fidelity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .base import PolynomialModel


@dataclass
class QuadratizationResult:
    """A quadratic model plus the accounting of what it cost."""

    model: PolynomialModel
    n_ancillas: int
    n_original_vars: int
    penalty_weight: float
    ancilla_pairs: dict[tuple[int, int], int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overhead_ratio(self) -> float:
        """Ancillas per original variable."""
        return self.n_ancillas / self.n_original_vars if self.n_original_vars else 0.0

    @property
    def total_vars(self) -> int:
        return self.n_original_vars + self.n_ancillas

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_original_vars": self.n_original_vars,
            "n_ancillas": self.n_ancillas,
            "total_vars": self.total_vars,
            "overhead_ratio": round(self.overhead_ratio, 4),
            "penalty_weight": self.penalty_weight,
            "degree_after": self.model.degree,
            **self.metadata,
        }


def rosenberg_penalty_weight(model: PolynomialModel, safety: float = 1.5) -> float:
    """A penalty large enough that violating ``y = x_i x_j`` is never profitable.

    The bound is the total magnitude of all coefficients: no assignment can gain
    more than that, so a penalty above it makes any violation strictly worse.
    Conservative, and conservatism is right here -- the reduction must be exact
    or the ancilla-cost comparison is meaningless.
    """
    total = sum(abs(c) for k, c in model.terms.items() if k)
    return max(safety * total, 1.0)


def quadratize(
    model: PolynomialModel,
    penalty_weight: float | None = None,
) -> QuadratizationResult:
    """Reduce a model of any degree to degree <= 2 using Rosenberg ancillas.

    A model already of degree <= 2 is returned unchanged with zero ancillas, so
    the Level 0 and Level 1 rows of the resource table are genuine zeros rather
    than an artefact of running the reduction anyway.
    """
    if model.degree <= 2:
        return QuadratizationResult(
            model=PolynomialModel(
                model.n_vars, dict(model.terms), dict(model.metadata)
            ),
            n_ancillas=0,
            n_original_vars=model.n_vars,
            penalty_weight=0.0,
            metadata={"reduced": False},
        )

    weight = (
        rosenberg_penalty_weight(model) if penalty_weight is None else penalty_weight
    )

    high_order = {k: c for k, c in model.terms.items() if len(k) >= 3}
    # Greedy: prefer pairs that appear in many high-order terms, so one ancilla
    # serves as many reductions as possible.
    pair_counts: Counter[tuple[int, int]] = Counter()
    for key in high_order:
        for a in range(len(key)):
            for b in range(a + 1, len(key)):
                pair_counts[(key[a], key[b])] += 1

    out = PolynomialModel(model.n_vars, metadata=dict(model.metadata))
    for key, coeff in model.terms.items():
        if len(key) < 3:
            out.add(key, coeff)

    ancilla_of: dict[tuple[int, int], int] = {}

    def ancilla_for(pair: tuple[int, int]) -> int:
        if pair in ancilla_of:
            return ancilla_of[pair]
        idx = out.n_vars
        out.n_vars += 1
        ancilla_of[pair] = idx
        # Rosenberg penalty: M*(x_i x_j - 2 x_i y - 2 x_j y + 3 y)
        i, j = pair
        out.add((i, j), weight)
        out.add((i, idx), -2.0 * weight)
        out.add((j, idx), -2.0 * weight)
        out.add((idx,), 3.0 * weight)
        return idx

    for key, coeff in sorted(high_order.items()):
        remaining = list(key)
        while len(remaining) > 2:
            # Pick the pair already reduced if possible, else the most common.
            best_pair = None
            best_score = -1
            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    pair = (remaining[a], remaining[b])
                    score = (10**6 if pair in ancilla_of else 0) + pair_counts[pair]
                    if score > best_score:
                        best_score, best_pair = score, pair
            assert best_pair is not None
            y = ancilla_for(best_pair)
            remaining = [v for v in remaining if v not in best_pair] + [y]
            remaining.sort()
        out.add(tuple(remaining), coeff)

    return QuadratizationResult(
        model=out,
        n_ancillas=len(ancilla_of),
        n_original_vars=model.n_vars,
        penalty_weight=weight,
        ancilla_pairs=dict(ancilla_of),
        metadata={"reduced": True, "n_high_order_terms": len(high_order)},
    )


def verify_quadratization(
    original: PolynomialModel,
    result: QuadratizationResult,
    assignments: list[tuple[int, ...]] | None = None,
) -> bool:
    """Check that minimising over ancillas reproduces the original energy.

    For each assignment of the original variables, the quadratized model's
    energy minimised over the ancillas must equal the original energy -- that is
    exactly what makes the reduction faithful. Exhaustive over ancillas, so use
    on small instances.
    """
    import itertools

    n = original.n_vars
    n_anc = result.n_ancillas
    if assignments is None:
        if n > 12:
            raise ValueError("supply explicit assignments for n > 12")
        assignments = list(itertools.product((0, 1), repeat=n))

    for x in assignments:
        target = original.energy(x)
        best = min(
            result.model.energy(tuple(x) + anc)
            for anc in itertools.product((0, 1), repeat=n_anc)
        )
        if abs(best - target) > 1e-6:
            return False
    return True
