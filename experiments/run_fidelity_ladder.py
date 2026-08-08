#!/usr/bin/env python
"""The fidelity-ladder figure: encoding gap at Levels 0, 1 and 2.

Plan section 5, Phase 5 exit criterion: *produce the fidelity-ladder figure --
encoding gap at Levels 0/1/2, with the ancilla cost that gate-based hardware
would pay for Level 2.*

Reports three things per rung:

``model_energy_error``   |H(x_ref) - E_vienna(reference)|, the fidelity of the
                         energy *function* on a representable structure
``encoding_gap``         E_vienna(decoded model optimum) - MFE, the
                         decision-relevant error
``resources``            variables, cubic terms, CNOTs per cost layer, and the
                         ancillas a gate-based device needs to reach Level 2

Also runs the ``branch_damping`` sweep, which trades exactness on 3-way
junctions against over-correction on junctions with more branches -- the
practical face of the degree-3 impossibility proved in ``model/level2.py``.

Writes ``results/tables/fidelity_ladder.csv`` and
``results/tables/branch_damping_sweep.csv``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from rnaqopt.config import STEMS, TABLE_DIR, VIENNA, ensure_dirs
from rnaqopt.decode import decode, structure_to_selection
from rnaqopt.energy import _children, _pair_table
from rnaqopt.model import build_level0, build_level1, build_level2
from rnaqopt.model.quadratize import quadratize
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.resources import cost_layer_gates, two_qubit_cost_of_degree
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.stems import enumerate_with_graphs

BUILDERS = {0: build_level0, 1: build_level1, 2: build_level2}


def max_branch_count(structure: str) -> int:
    pt = _pair_table(structure)
    best = 0
    for i in range(len(structure)):
        if pt[i] > i:
            kids, _ = _children(pt, i, pt[i])
            best = max(best, len(kids))
    return best


def run(tiers: list[str], max_stems: int, max_seconds: float) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        for rec in load_tier(tier):
            graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
            if graphs.n > max_stems:
                continue
            ref_db, ref_e = mfe(rec.sequence, VIENNA)
            sel = structure_to_selection(graphs, ref_db)
            branches = max_branch_count(ref_db)

            for level, build in BUILDERS.items():
                model = build(rec.sequence, graphs, cfg=VIENNA)
                exact = solve_exact(model, max_seconds=max_seconds)
                priority = [model.full.terms.get((i,), 0.0) for i in range(graphs.n)]
                decoded = decode(exact.bitstring, graphs, len(rec.sequence), priority)
                e_model = eval_structure(rec.sequence, decoded.structure, VIENNA)

                err = None
                if sel is not None:
                    err = abs(model.objective.energy_of_selection(sel) - ref_e)

                counts = model.objective.term_counts()
                cnots = sum(
                    1 for g in cost_layer_gates(model.full.terms) if g.is_two_qubit
                )
                quad = quadratize(model.full)

                rows.append(
                    {
                        "tier": tier,
                        "seq_id": rec.seq_id,
                        "length": rec.length,
                        "n_stems": graphs.n,
                        "max_branches": branches,
                        "level": level,
                        "degree": model.degree,
                        "encoding_gap": round(e_model - ref_e, 4),
                        "model_energy_error": None if err is None else round(err, 4),
                        "representable": sel is not None,
                        "n_linear": counts.get(1, 0),
                        "n_quadratic": counts.get(2, 0),
                        "n_cubic": counts.get(3, 0),
                        "cost_layer_cnots": cnots,
                        "cubic_cnots": two_qubit_cost_of_degree(model.full.terms).get(
                            3, 0
                        ),
                        "n_ancillas": quad.n_ancillas,
                        "quadratized_vars": quad.model.n_vars,
                        "wall_time": exact.wall_time,
                    }
                )
    return pd.DataFrame(rows)


def damping_sweep(tiers: list[str], max_stems: int, values: list[float]) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        for rec in load_tier(tier):
            graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
            if graphs.n > max_stems:
                continue
            ref_db, ref_e = mfe(rec.sequence, VIENNA)
            sel = structure_to_selection(graphs, ref_db)
            if sel is None:
                continue
            branches = max_branch_count(ref_db)
            for d in values:
                model = build_level2(rec.sequence, graphs, cfg=VIENNA, branch_damping=d)
                rows.append(
                    {
                        "tier": tier,
                        "seq_id": rec.seq_id,
                        "max_branches": branches,
                        "branch_damping": round(d, 4),
                        "model_energy_error": round(
                            abs(model.objective.energy_of_selection(sel) - ref_e), 4
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="A,M")
    ap.add_argument("--max-stems", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)

    ensure_dirs()
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]

    print(VIENNA.header_line())
    print()
    df = run(tiers, args.max_stems, args.max_seconds)
    df.to_csv(TABLE_DIR / "fidelity_ladder.csv", index=False, lineterminator="\n")

    print("=" * 92)
    print("FIDELITY LADDER")
    print("=" * 92)
    summary = (
        df.groupby(["tier", "level"])
        .agg(
            n=("seq_id", "size"),
            n_stems=("n_stems", "mean"),
            degree=("degree", "max"),
            energy_error=("model_energy_error", "mean"),
            encoding_gap=("encoding_gap", "mean"),
            cubic_terms=("n_cubic", "mean"),
            cost_cnots=("cost_layer_cnots", "mean"),
            ancillas=("n_ancillas", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())

    print()
    print("by multiloop branch count (set M):")
    m = df[df.tier == "M"]
    if not m.empty:
        print(
            m.groupby(["max_branches", "level"])[["model_energy_error", "encoding_gap"]]
            .mean()
            .round(3)
            .to_string()
        )

    print()
    print("=" * 92)
    print("BRANCH-DAMPING SWEEP (Level 2)")
    print("=" * 92)
    sweep = damping_sweep(tiers, args.max_stems, [0.25, 1 / 3, 0.5, 0.75, 1.0, 1.25])
    sweep.to_csv(
        TABLE_DIR / "branch_damping_sweep.csv", index=False, lineterminator="\n"
    )
    if not sweep.empty:
        piv = sweep.pivot_table(
            index="branch_damping",
            columns="max_branches",
            values="model_energy_error",
            aggfunc="mean",
        ).round(3)
        print("mean |model energy error| by damping (columns = branch count)")
        print(piv.to_string())

    print(f"\nwrote {TABLE_DIR / 'fidelity_ladder.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
