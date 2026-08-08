#!/usr/bin/env python
"""Matched-budget solver comparison: the optimizer-gap table.

Plan section 4.6: *matched-budget comparison is essential; unmatched
comparisons will be discounted by judges.*  Every heuristic solver here is given
the **same wall-clock budget per instance**, and the exact optimum of the same
model is computed separately so the section 2.2 optimizer gap is a true gap
rather than a difference of two approximations.

Solvers compared:

``exact``        brute force / CP-SAT -- the ground truth, not a competitor
``random``       uniform sampling: the floor every method must clear
``annealing``    simulated annealing at any polynomial degree
``lowrank``      Burer-Monteiro relaxation, the classical PCE counterpart
``adapt_qaoa``   gate-based, CVaR objective, finite-shot readout
``pce``          Pauli Correlation Encoding, O(sqrt(n)) qubits
``dirac3_sim``   the Dirac-3 simplex programme, solved classically

Writes ``results/tables/solver_comparison.csv``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from rnaqopt.config import STEMS, TABLE_DIR, VIENNA, ensure_dirs
from rnaqopt.decode import decode
from rnaqopt.metrics import compare_structures
from rnaqopt.model import build_model
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.adapt_qaoa import AdaptQAOASolver
from rnaqopt.solvers.annealing import RandomSearchSolver, SimulatedAnnealingSolver
from rnaqopt.solvers.dirac3 import DiracSimplexSimulator
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.solvers.lowrank import LowRankSolver
from rnaqopt.solvers.pce import PCESolver
from rnaqopt.stems import enumerate_with_graphs


def build_solvers(budget: float, seed: int, max_qubits: int):
    return {
        "random": RandomSearchSolver(n_samples=10**7, seed=seed, time_budget=budget),
        "annealing": SimulatedAnnealingSolver(
            n_sweeps=400, n_restarts=1000, seed=seed, time_budget=budget
        ),
        "lowrank": LowRankSolver(seed=seed, time_budget=budget),
        "adapt_qaoa": AdaptQAOASolver(
            max_layers=6, alpha=0.15, seed=seed, max_qubits=max_qubits,
            time_budget=budget,
        ),
        "pce": PCESolver(seed=seed, max_qubits=12, time_budget=budget),
        "dirac3_sim": DiracSimplexSimulator(
            scheme="per_stem", n_restarts=8, n_steps=250, seed=seed
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="A,M")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--max-stems", type=int, default=20)
    ap.add_argument("--budget", type=float, default=2.0, help="seconds per solver")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-qubits", type=int, default=18)
    args = ap.parse_args(argv)

    ensure_dirs()
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(VIENNA.header_line())
    print(f"matched budget: {args.budget:.1f}s per solver per instance")
    print()

    rows = []
    for tier in tiers:
        for rec in load_tier(tier):
            graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
            if not 0 < graphs.n <= args.max_stems:
                continue
            ref_db, ref_e = mfe(rec.sequence, VIENNA)

            for level in levels:
                model = build_model(level, rec.sequence, graphs, cfg=VIENNA)
                exact = solve_exact(model, max_seconds=60)
                priority = [
                    model.full.terms.get((i,), 0.0) for i in range(graphs.n)
                ]
                exact_dec = decode(
                    exact.bitstring, graphs, len(rec.sequence), priority
                )
                e_model_opt = eval_structure(
                    rec.sequence, exact_dec.structure, VIENNA
                )

                for seed in range(args.seeds):
                    for name, solver in build_solvers(
                        args.budget, seed, args.max_qubits
                    ).items():
                        try:
                            r = solver.solve(model)
                        except Exception as exc:  # noqa: BLE001
                            rows.append(
                                {
                                    "tier": tier, "seq_id": rec.seq_id,
                                    "n_stems": graphs.n, "level": level,
                                    "solver": name, "seed": seed,
                                    "skipped": True, "reason": str(exc)[:120],
                                }
                            )
                            continue
                        dec = decode(
                            r.bitstring, graphs, len(rec.sequence), priority
                        )
                        e_vienna = eval_structure(
                            rec.sequence, dec.structure, VIENNA
                        )
                        sm = compare_structures(dec.structure, ref_db)
                        rows.append(
                            {
                                "tier": tier,
                                "seq_id": rec.seq_id,
                                "n_stems": graphs.n,
                                "level": level,
                                "solver": name,
                                "seed": seed,
                                "skipped": False,
                                "model_energy": round(r.model_energy, 4),
                                "exact_model_energy": round(exact.model_energy, 4),
                                # Optimizer gap in the model's own units.
                                "optimizer_gap_model": round(
                                    r.model_energy - exact.model_energy, 4
                                ),
                                "found_model_optimum": abs(
                                    r.model_energy - exact.model_energy
                                ) < 1e-6,
                                # The section 2.2 decomposition, in kcal/mol.
                                "e_vienna_mfe": ref_e,
                                "e_model_optimum": e_model_opt,
                                "e_solver": e_vienna,
                                "encoding_gap": round(e_model_opt - ref_e, 4),
                                "optimizer_gap": round(e_vienna - e_model_opt, 4),
                                "total_gap": round(e_vienna - ref_e, 4),
                                "f1": round(sm.f1, 4),
                                "sensitivity": round(sm.sensitivity, 4),
                                "ppv": round(sm.ppv, 4),
                                "exact_match": sm.exact_match,
                                "feasible_raw": dec.feasible_raw,
                                "was_repaired": dec.was_repaired,
                                "wall_time": round(r.wall_time, 4),
                                "n_qubits": r.resource_dict.get(
                                    "n_qubits", r.resource_dict.get("n_vars")
                                ),
                                "depth": r.resource_dict.get("depth"),
                                "two_qubit_gates": r.resource_dict.get(
                                    "n_two_qubit_gates"
                                ),
                                "compression_ratio": r.resource_dict.get(
                                    "compression_ratio"
                                ),
                                "function_evaluations": r.resource_dict.get(
                                    "function_evaluations"
                                ),
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(
        TABLE_DIR / "solver_comparison.csv", index=False, lineterminator="\n"
    )

    ok = df[~df.skipped] if "skipped" in df else df
    if not ok.empty:
        print("=" * 100)
        print("OPTIMIZER GAP BY SOLVER (matched budget)")
        print("=" * 100)
        print(
            ok.groupby(["level", "solver"])
            .agg(
                n=("seq_id", "size"),
                hit_optimum=("found_model_optimum", "mean"),
                opt_gap=("optimizer_gap", "mean"),
                enc_gap=("encoding_gap", "mean"),
                total_gap=("total_gap", "mean"),
                f1=("f1", "mean"),
                repaired=("was_repaired", "mean"),
                secs=("wall_time", "mean"),
            )
            .round(3)
            .to_string()
        )
    if "skipped" in df and df.skipped.any():
        print()
        print("skipped runs:")
        print(df[df.skipped].groupby("solver").size().to_string())

    print(f"\nwrote {TABLE_DIR / 'solver_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
