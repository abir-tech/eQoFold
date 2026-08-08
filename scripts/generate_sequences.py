#!/usr/bin/env python
"""Regenerate the tier FASTA files. Thin wrapper -- all logic lives in src/."""

from __future__ import annotations

import sys

from rnaqopt.cli import generate_sequences_main

if __name__ == "__main__":
    sys.exit(generate_sequences_main())
