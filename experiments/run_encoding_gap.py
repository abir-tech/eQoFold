#!/usr/bin/env python
"""The encoding-gap table: how faithfully does each model rung represent Turner?

Plan section 2.2.  Solving each model to its **exact** optimum makes the
optimizer gap zero by construction, so every kcal/mol of remaining error is
attributable to the model.  That is the encoding gap.

Two columns are reported side by side, and both are needed:

``encoding_gap``        ViennaRNA energy of the decoded model optimum, minus
                        the ViennaRNA MFE.  The decision-relevant number, but
                        insensitive to energy errors that do not change which
                        structure wins.
``model_energy_error``  |H(x_ref) - E_vienna(reference)|.  How wrong the model's
                        energy *function* is on a structure it can represent.
                        Moves whenever the loop-energy extraction improves, even
                        when the argmin does not change.

Writes ``results/tables/encoding_gap.csv`` and the per-tier summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rnaqopt.config import STEMS, TABLE_DIR, VIENNA, ensure_dirs
from rnaqopt.pipeline import encoding_gap_table
from rnaqopt.sequences import load_tier

SUMMARY_COLUMNS = [
    "n_stems",
    "encoding_gap",
    "model_energy_error",
    "f1",
    "sensitivity",
    "ppv",
    "exact_match",
    "reference_representable",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="A,M")
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--max-stems", type=int, default=64,
                    help="skip instances larger than this (exact solve gets slow)")
    ap.add_argument("--brute-force-limit", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=TABLE_DIR / "encoding_gap.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ensure_dirs()
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())

    from rnaqopt.stems import enumerate_stems

    records = []
    skipped = []
    for tier in tiers:
        for rec in load_tier(tier):
            n = len(enumerate_stems(rec.sequence, cfg=STEMS))
            if n > args.max_stems:
                skipped.append((rec.seq_id, n))
            else:
                records.append(rec)

    print(VIENNA.header_line())
    print(f"stem enumeration: L_min={STEMS.min_stem_length} "
          f"substems={STEMS.include_substems}")
    print(f"instances: {len(records)} (skipped {len(skipped)} above "
          f"{args.max_stems} stems)")
    print()

    rows = encoding_gap_table(
        records,
        levels=levels,
        brute_force_limit=args.brute_force_limit,
        max_seconds=args.max_seconds,
        progress=not args.quiet,
    )
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    print()
    print("=" * 78)
    print("ENCODING GAP BY TIER AND MODEL LEVEL")
    print("=" * 78)
    summary = (
        df.groupby(["tier", "level"])[SUMMARY_COLUMNS].mean().round(3)
    )
    print(summary.to_string())

    print()
    print("gap closure from Level 0 to Level 1 (kcal/mol, positive = improvement):")
    for tier in sorted(df.tier.unique()):
        sub = df[df.tier == tier]
        by_level = sub.groupby("level")[["encoding_gap", "model_energy_error"]].mean()
        if 0 in by_level.index and 1 in by_level.index:
            d_enc = by_level.loc[0, "encoding_gap"] - by_level.loc[1, "encoding_gap"]
            d_err = (
                by_level.loc[0, "model_energy_error"]
                - by_level.loc[1, "model_energy_error"]
            )
            print(
                f"  tier {tier}: encoding_gap {by_level.loc[0, 'encoding_gap']:.3f}"
                f" -> {by_level.loc[1, 'encoding_gap']:.3f}  (closed {d_enc:+.3f})"
                f"   |  energy_error {by_level.loc[0, 'model_energy_error']:.3f}"
                f" -> {by_level.loc[1, 'model_energy_error']:.3f}  (closed {d_err:+.3f})"
            )

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
