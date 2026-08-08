#!/usr/bin/env python
"""Build the ViennaRNA reference table. Thin wrapper -- all logic lives in src/."""

from __future__ import annotations

import sys

from rnaqopt.cli import build_reference_main

if __name__ == "__main__":
    sys.exit(build_reference_main())
