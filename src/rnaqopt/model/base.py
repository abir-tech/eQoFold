"""The one polynomial model type used by every fidelity level and every solver.

Plan section 3, non-negotiable engineering rule: *one ``PolynomialModel`` type
carries all fidelity levels: ``dict[tuple[int, ...], float]`` mapping variable
index tuples to coefficients, plus a constant offset.  Degree is a property,
not a separate class.*

Binary variables, so ``x_i^2 = x_i``: a key is a **sorted tuple of distinct**
indices.  The empty tuple is the constant offset.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolynomialModel:
    """A pseudo-Boolean polynomial over ``n_vars`` binary variables.

    ``terms[()]`` is the constant offset; ``terms[(i,)]`` is linear;
    ``terms[(i, j)]`` quadratic; ``terms[(i, j, k)]`` cubic.
    """

    n_vars: int
    terms: dict[tuple[int, ...], float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def add(self, variables: Iterable[int], coefficient: float) -> None:
        """Add ``coefficient`` to the term over ``variables``.

        Duplicate indices are collapsed (``x*x = x``), the key is sorted, and a
        coefficient that cancels to zero is dropped so that :meth:`degree` and
        term counts stay honest.
        """
        if coefficient == 0.0:
            return
        key = tuple(sorted(set(variables)))
        for v in key:
            if not 0 <= v < self.n_vars:
                raise IndexError(f"variable {v} out of range for n_vars={self.n_vars}")
        new = self.terms.get(key, 0.0) + coefficient
        if new == 0.0:
            self.terms.pop(key, None)
        else:
            self.terms[key] = new

    def add_constant(self, value: float) -> None:
        self.add((), value)

    # -- properties --------------------------------------------------------

    @property
    def constant(self) -> float:
        return self.terms.get((), 0.0)

    @property
    def degree(self) -> int:
        """Highest term order present. A constant-only model has degree 0."""
        return max((len(k) for k in self.terms), default=0)

    def terms_of_degree(self, d: int) -> dict[tuple[int, ...], float]:
        return {k: v for k, v in self.terms.items() if len(k) == d}

    def term_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for k in self.terms:
            counts[len(k)] += 1
        return dict(sorted(counts.items()))

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def active_variables(self) -> set[int]:
        """Variables that appear in at least one non-constant term."""
        return {v for k in self.terms for v in k}

    # -- evaluation --------------------------------------------------------

    def energy(self, x: Sequence[int] | Mapping[int, int]) -> float:
        """Objective value at assignment ``x``.

        ``x`` may be a sequence indexed by variable, or a mapping. Values are
        treated as truthy/falsy, so bitstrings of ``0``/``1`` ints, bools or
        numpy integers all work.
        """
        total = 0.0
        for key, coeff in self.terms.items():
            if all(x[v] for v in key):
                total += coeff
        return total

    def energy_of_selection(self, selected: Iterable[int]) -> float:
        """Objective value where exactly ``selected`` variables are 1."""
        chosen = set(selected)
        return sum(
            coeff for key, coeff in self.terms.items() if chosen.issuperset(key)
        )

    # -- transformation ----------------------------------------------------

    def scaled(self, factor: float) -> PolynomialModel:
        """A copy with every coefficient multiplied by ``factor``."""
        out = PolynomialModel(self.n_vars, metadata=dict(self.metadata))
        for key, coeff in self.terms.items():
            out.terms[key] = coeff * factor
        return out

    def to_integer_terms(self, scale: int = 100) -> dict[tuple[int, ...], int]:
        """Coefficients as exact integers after multiplying by ``scale``.

        Every Turner energy is a multiple of 0.01 kcal/mol, so ``scale=100``
        makes the model exactly integral. This is what lets CP-SAT solve the
        model to proven optimality rather than to a floating-point tolerance.
        Raises if a coefficient is not representable, rather than rounding
        silently.
        """
        out: dict[tuple[int, ...], int] = {}
        for key, coeff in self.terms.items():
            scaled = coeff * scale
            nearest = round(scaled)
            if abs(scaled - nearest) > 1e-6:
                raise ValueError(
                    f"coefficient {coeff!r} for {key} is not a multiple of 1/{scale}; "
                    "integer conversion would lose information"
                )
            out[key] = int(nearest)
        return out

    def merged(self, other: PolynomialModel) -> PolynomialModel:
        """Sum of two models over the same variable set."""
        if self.n_vars != other.n_vars:
            raise ValueError("cannot merge models with different variable counts")
        out = PolynomialModel(self.n_vars, dict(self.terms), dict(self.metadata))
        for key, coeff in other.terms.items():
            out.add(key, coeff)
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        counts = self.term_counts()
        return {
            "n_vars": self.n_vars,
            "degree": self.degree,
            "n_terms": self.n_terms,
            "n_linear": counts.get(1, 0),
            "n_quadratic": counts.get(2, 0),
            "n_cubic": counts.get(3, 0),
            "constant": self.constant,
            **{
                k: v
                for k, v in self.metadata.items()
                if isinstance(v, int | float | str)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        c = self.term_counts()
        return (
            f"PolynomialModel(n_vars={self.n_vars}, degree={self.degree}, "
            f"terms={dict(c)}, const={self.constant:.3f})"
        )
