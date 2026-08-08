#!/usr/bin/env python
"""Enumeration ablation: does relaxing L_min or adding sub-stems pay for itself?

Plan section 4.2: *start without sub-stems; add them only if the encoding gap
analysis shows they matter.  Document the choice.*  Plan section 6 lists
"variable count explodes with L_min = 2 or sub-stems" as a known risk.

This script measures both sides of that trade at once:

  benefit  fraction of reference structures the candidate set can represent,
           and the resulting encoding gap
  cost     |stems|, i.e. the number of binary variables, which is the problem
           size every resource and scaling claim is expressed in

Writes ``results/tables/enumeration_ablation.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rnaqopt.config import TABLE_DIR, VIENNA, StemConfig, ensure_dirs
from rnaqopt.decode import decode, structure_to_selection
from rnaqopt.model import build_level1
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.stems import enumerate_with_graphs

#: The enumeration variants under test.
VARIANTS: dict[str, StemConfig] = {
    "Lmin3_maximal": StemConfig(min_stem_length=3, include_substems=False),
    "Lmin3_substems": StemConfig(min_stem_length=3, include_substems=True),
    "Lmin2_maximal": StemConfig(min_stem_length=2, include_substems=False),
    "Lmin2_substems": StemConfig(min_stem_length=2, include_substems=True),
}


def run(tiers: list[str], brute_force_limit: int, max_seconds: float) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        for rec in load_tier(tier):
            ref_db, ref_e = mfe(rec.sequence, VIENNA)
            for name, cfg in VARIANTS.items():
                graphs = enumerate_with_graphs(
                    rec.sequence, cfg=cfg, min_hairpin=VIENNA.min_hairpin_loop
                )
                sel = structure_to_selection(graphs, ref_db)
                representable = sel is not None

                encoding_gap = None
                if graphs.n <= brute_force_limit or graphs.n <= 200:
                    model = build_level1(rec.sequence, graphs, cfg=VIENNA)
                    exact = solve_exact(
                        model,
                        brute_force_limit=brute_force_limit,
                        max_seconds=max_seconds,
                    )
                    priority = [
                        model.full.terms.get((i,), 0.0) for i in range(graphs.n)
                    ]
                    decoded = decode(
                        exact.bitstring, graphs, len(rec.sequence), priority
                    )
                    encoding_gap = round(
                        eval_structure(rec.sequence, decoded.structure, VIENNA) - ref_e,
                        4,
                    )

                rows.append(
                    {
                        "tier": tier,
                        "seq_id": rec.seq_id,
                        "length": rec.length,
                        "variant": name,
                        "min_stem_length": cfg.min_stem_length,
                        "include_substems": cfg.include_substems,
                        "n_stems": graphs.n,
                        "n_conflict": len(graphs.conflict),
                        "n_nesting": len(graphs.nesting),
                        "representable": representable,
                        "encoding_gap": encoding_gap,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="A,M")
    ap.add_argument("--brute-force-limit", type=int, default=22)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=TABLE_DIR / "enumeration_ablation.csv")
    args = ap.parse_args(argv)

    ensure_dirs()
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    df = run(tiers, args.brute_force_limit, args.max_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    print(VIENNA.header_line())
    print()
    summary = (
        df.groupby(["tier", "variant"])
        .agg(
            n=("seq_id", "size"),
            mean_stems=("n_stems", "mean"),
            max_stems=("n_stems", "max"),
            representable=("representable", "mean"),
            mean_encoding_gap=("encoding_gap", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
