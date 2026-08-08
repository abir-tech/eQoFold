"""rnaqopt -- quantum and quantum-inspired optimization of RNA secondary structure.

WISER Summer Program 2026, Moderna Challenge.  Team eQoSystem.

The package is layered so that the frozen ViennaRNA configuration cannot be
bypassed:

``config``      frozen model details + repository paths (single source of truth)
``sequences``   generation, loading, validation, provenance
``reference``   ViennaRNA answer key: MFE, partition function, eval_structure
``metrics``     structural accuracy + the two-gap error decomposition

Later phases add ``stems``, ``energy``, ``model/``, ``solvers/``, ``decode``,
``resources`` and ``noise``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import STEMS, VIENNA, VIENNA_STOCK, StemConfig, ViennaConfig

__all__ = [
    "__version__",
    "VIENNA",
    "VIENNA_STOCK",
    "STEMS",
    "ViennaConfig",
    "StemConfig",
]
