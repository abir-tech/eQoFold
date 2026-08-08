"""Command-line entry points.

Two commands, both deterministic:

``rnaqopt-sequences``  regenerate the tier FASTA files from ``GLOBAL_SEED``
``rnaqopt-reference``  fold every sequence and write ``vienna_reference.csv``

Both are wired into the ``Makefile``; ``make reference`` is the Phase 1 exit
criterion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    GLOBAL_SEED,
    SEQUENCE_DIR,
    VIENNA,
    VIENNA_REFERENCE_CSV,
    ensure_dirs,
)
from .reference import build_reference_table, vienna_version, write_reference_csv
from .sequences import TIER_FILES, TIERS, generate_tier, load_all, write_fasta


def generate_sequences_main(argv: list[str] | None = None) -> int:
    """Regenerate the three tier FASTA files deterministically."""
    ap = argparse.ArgumentParser(
        prog="rnaqopt-sequences",
        description="Deterministically regenerate the tier FASTA files.",
    )
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--outdir", type=Path, default=SEQUENCE_DIR)
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for offset, tier in enumerate(TIERS):
        # Distinct per-tier stream, derived from the one global seed.
        records = generate_tier(tier, args.seed + offset)
        path = args.outdir / TIER_FILES[tier].name
        write_fasta(records, path)
        lengths = [r.length for r in records]
        print(
            f"tier {tier}: {len(records):2d} sequences, "
            f"len {min(lengths)}-{max(lengths)} -> {path.name}"
        )
        total += len(records)
    print(f"seed={args.seed}  total={total} sequences")
    return 0


def build_reference_main(argv: list[str] | None = None) -> int:
    """Fold every sequence with ViennaRNA and write the reference table."""
    ap = argparse.ArgumentParser(
        prog="rnaqopt-reference",
        description="Generate the ViennaRNA reference table (the answer key).",
    )
    ap.add_argument("--out", type=Path, default=VIENNA_REFERENCE_CSV)
    ap.add_argument(
        "--tiers",
        default=",".join(TIERS),
        help="comma-separated tiers to include (default: all)",
    )
    args = ap.parse_args(argv)

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    ensure_dirs()

    records = load_all(tiers)
    print(f"ViennaRNA {vienna_version()}")
    print(VIENNA.header_line())
    print(f"folding {len(records)} sequences from tiers {','.join(tiers)} ...")

    rows = build_reference_table(records)
    path = write_reference_csv(rows, args.out)

    by_tier: dict[str, list[dict]] = {}
    for row in rows:
        by_tier.setdefault(str(row["tier"]), []).append(row)
    for tier in sorted(by_tier):
        group = by_tier[tier]
        mean_e = sum(float(r["mfe_energy"]) for r in group) / len(group)
        mean_p = sum(int(r["n_pairs"]) for r in group) / len(group)
        n_ml = sum(1 for r in group if r["has_multiloop"])
        print(
            f"  tier {tier}: n={len(group):2d}  "
            f"mean MFE={mean_e:7.2f} kcal/mol  "
            f"mean pairs={mean_p:5.1f}  multiloops={n_ml}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(build_reference_main())
