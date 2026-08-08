"""The fidelity ladder: one polynomial model type, three levels of physics."""

from __future__ import annotations

from .base import PolynomialModel
from .bundle import StemModel
from .level0 import build_level0
from .level1 import build_level1, structure_is_level1_exact
from .level2 import build_level2, max_exact_branch_count, structure_is_level2_exact
from .penalties import default_penalty, max_single_variable_gain

#: Dispatch table so experiments can select a rung by number from a YAML config.
BUILDERS = {
    0: build_level0,
    1: build_level1,
    2: build_level2,
}


def build_model(level: int, *args, **kwargs) -> StemModel:
    """Build the model at the requested fidelity level."""
    try:
        builder = BUILDERS[level]
    except KeyError:
        raise ValueError(
            f"unknown model level {level!r}; available: {sorted(BUILDERS)}"
        ) from None
    return builder(*args, **kwargs)


__all__ = [
    "PolynomialModel",
    "StemModel",
    "build_level0",
    "build_level1",
    "build_level2",
    "build_model",
    "BUILDERS",
    "structure_is_level1_exact",
    "structure_is_level2_exact",
    "max_exact_branch_count",
    "default_penalty",
    "max_single_variable_gain",
]
