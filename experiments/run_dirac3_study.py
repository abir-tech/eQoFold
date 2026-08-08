#!/usr/bin/env python
"""Dirac-3 sum-constraint encoding study and R sweep.

Plan section 4.6:

* *Binary decisions require either a complementary slack variable per stem, or a
  single global slack ... Test both encodings and report which performs better
  -- that comparison is itself a result.*
* *Sweep R and report sensitivity; R is effectively a prior on structure
  density.*
* *Run each configuration multiple times (the device is stochastic), report
  distributions not single shots.*
* *Apply the same greedy repair + ViennaRNA scoring as all other solvers.*

Runs against the classical simplex simulator, which solves the identical
continuous programme, so the whole study reproduces without device credentials.

Writes ``results/tables/dirac3_encodings.csv`` and
``results/tables/dirac3_r_sweep.csv``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from rnaqopt.config import STEMS, TABLE_DIR, VIENNA, ensure_dirs
from rnaqopt.decode import decode
from rnaqopt.model import build_level2
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.dirac3 import (
    FREE_TIER_MAX_DEGREE,
    MEASURED_DEGREE4_VAR_LIMIT,
    DiracSimplexSimulator,
    encode,
)
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.stems import enumerate_with_graphs

SCHEMES = ("per_stem", "global_slack")


def evaluate(model, graphs, rec, result, exact, ref_e):
    priority = [model.full.terms.get((i, ), 0.0) for i in range(graphs.n)]
    decoded = decode(result.bitstring, graphs, len(rec.sequence), priority)
    e_vienna = eval_structure(rec.sequence, decoded.structure, VIENNA)
    return {
        "model_energy": result.model_energy,
        "optimizer_gap_model": round(result.model_energy - exact.model_energy, 4),
        "e_vienna": e_vienna,
        "total_gap": round(e_vienna - ref_e, 4),
        "feasible_raw": decoded.feasible_raw,
        "was_repaired": decoded.was_repaired,
        "n_removed": decoded.n_removed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="A,M")
    ap.add_argument("--max-stems", type=int, default=45)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args(argv)

    ensure_dirs()
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]

    print(VIENNA.header_line())
    print(
        f"free-tier degree ceiling: {FREE_TIER_MAX_DEGREE} | "
        f"measured device var limit at degree 4: {MEASURED_DEGREE4_VAR_LIMIT}"
    )
    print()

    enc_rows, sweep_rows = [], []
    for tier in tiers:
        for rec in load_tier(tier):
            graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
            if graphs.n > args.max_stems or graphs.n == 0:
                continue
            model = build_level2(rec.sequence, graphs, cfg=VIENNA)
            exact = solve_exact(model, max_seconds=60)
            ref_db, ref_e = mfe(rec.sequence, VIENNA)

            for scheme in SCHEMES:
                enc = encode(model, scheme=scheme)
                for seed in range(args.seeds):
                    sim = DiracSimplexSimulator(
                        scheme=scheme,
                        n_restarts=args.restarts,
                        n_steps=args.steps,
                        seed=seed,
                    )
                    r = sim.solve(model)
                    md = r.solver_metadata
                    enc_rows.append(
                        {
                            "tier": tier,
                            "seq_id": rec.seq_id,
                            "n_stems": graphs.n,
                            "scheme": scheme,
                            "encoded_vars": enc.n_vars,
                            "degree": enc.degree,
                            "fits_free_tier": enc.fits_free_tier(),
                            "R": enc.R,
                            "seed": seed,
                            "collapse_rate": md["collapse_rate"],
                            "max_stem_share": round(md["mean_max_stem_share"], 4),
                            "wall_time": r.wall_time,
                            **evaluate(model, graphs, rec, r, exact, ref_e),
                        }
                    )

            # R sweep on the encoding that has a meaningful density prior.
            base_R = graphs.n * 1.0
            for frac in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
                R = max(frac * base_R, 1.0)
                sim = DiracSimplexSimulator(
                    scheme="global_slack",
                    R=R,
                    n_restarts=args.restarts,
                    n_steps=args.steps,
                    seed=0,
                )
                r = sim.solve(model)
                sweep_rows.append(
                    {
                        "tier": tier,
                        "seq_id": rec.seq_id,
                        "n_stems": graphs.n,
                        "R_fraction": frac,
                        "R": R,
                        "n_selected": sum(r.bitstring),
                        "collapse_rate": r.solver_metadata["collapse_rate"],
                        **evaluate(model, graphs, rec, r, exact, ref_e),
                    }
                )

    enc_df = pd.DataFrame(enc_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    enc_df.to_csv(TABLE_DIR / "dirac3_encodings.csv", index=False, lineterminator="\n")
    sweep_df.to_csv(TABLE_DIR / "dirac3_r_sweep.csv", index=False, lineterminator="\n")

    print("=" * 92)
    print("ENCODING COMPARISON (Level 2, distributions over seeds)")
    print("=" * 92)
    print(
        enc_df.groupby(["tier", "scheme"])
        .agg(
            n=("seq_id", "size"),
            enc_vars=("encoded_vars", "mean"),
            opt_gap_mean=("optimizer_gap_model", "mean"),
            opt_gap_std=("optimizer_gap_model", "std"),
            total_gap=("total_gap", "mean"),
            feasible=("feasible_raw", "mean"),
            repaired=("was_repaired", "mean"),
            collapse=("collapse_rate", "mean"),
            secs=("wall_time", "mean"),
        )
        .round(3)
        .to_string()
    )

    print()
    print("=" * 92)
    print("R SWEEP (global_slack): R is a prior on structure density")
    print("=" * 92)
    if not sweep_df.empty:
        print(
            sweep_df.groupby("R_fraction")
            .agg(
                n_selected=("n_selected", "mean"),
                opt_gap=("optimizer_gap_model", "mean"),
                total_gap=("total_gap", "mean"),
                feasible=("feasible_raw", "mean"),
                collapse=("collapse_rate", "mean"),
            )
            .round(3)
            .to_string()
        )

    print(f"\nwrote {TABLE_DIR / 'dirac3_encodings.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
