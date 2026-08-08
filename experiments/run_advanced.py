#!/usr/bin/env python
"""Phase 7 advanced tasks: pseudoknots, noise, and constraint enforcement.

Three of the challenge's optional advanced tasks (section 1.6), each producing
a table:

**Pseudoknots.**  Plan section 2.6 calls this the free win: dropping the
crossing penalty is a one-line configuration change and yields pseudoknot-capable
folding, a regime ViennaRNA cannot enter.  Benchmarked on designed H-type
pseudoknots whose intended structure is known by construction.

*Honest scope note.*  We do **not** report a Turner energy for a pseudoknotted
structure.  ``eval_structure`` cannot score crossing pairs, and no
Turner-parameterised pseudoknot model is implemented here, so energies are not
comparable across the two modes.  Pseudoknot results are reported as base-pair
recovery against the designed structure only.

**Noise.**  Finite sampling first, then depolarizing and readout error, applied
to the ADAPT-QAOA output state.  Reports the shots needed to reach 99% success.

**Constraint enforcement.**  Penalty weights versus hard constraints, and the
lambda sweep of plan section 4.5.

Writes ``results/tables/{pseudoknot,noise,penalty_sweep}.csv``.
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import pandas as pd

from rnaqopt.config import STEMS, TABLE_DIR, VIENNA, ensure_dirs
from rnaqopt.decode import decode
from rnaqopt.metrics import has_crossings, pairs_from_dotbracket
from rnaqopt.model import build_level1
from rnaqopt.model.penalties import (
    PenaltySweepPoint,
    knee,
    max_single_variable_gain,
)
from rnaqopt.noise import NoiseModel, best_under_noise, shots_for_target_success
from rnaqopt.sequences import _COMPLEMENT, RNASequence, load_tier
from rnaqopt.solvers.adapt_qaoa import AdaptQAOASolver
from rnaqopt.solvers.exact import CPSATSolver, solve_exact
from rnaqopt.stems import enumerate_with_graphs

# --------------------------------------------------------------------------
# Pseudoknots
# --------------------------------------------------------------------------

def designed_pseudoknot(rng: random.Random, stem_a: int, stem_b: int, loop: int):
    """Build an H-type pseudoknot: stem A crosses stem B.

    Layout ``A1 L1 B1 L2 A2 L3 B2`` so that A's pairs and B's pairs cross.
    """
    def gc(k: int) -> str:
        return "".join(rng.choice("GC") for _ in range(k))

    a1 = gc(stem_a)
    b1 = gc(stem_b)
    a2 = "".join(_COMPLEMENT[c] for c in reversed(a1))
    b2 = "".join(_COMPLEMENT[c] for c in reversed(b1))
    l1 = "A" * loop
    l2 = "A" * loop
    l3 = "A" * loop
    seq = a1 + l1 + b1 + l2 + a2 + l3 + b2

    pairs = set()
    off_a1, off_b1 = 0, stem_a + loop
    off_a2 = off_b1 + stem_b + loop
    off_b2 = off_a2 + stem_a + loop
    for k in range(stem_a):
        pairs.add((off_a1 + k, off_a2 + stem_a - 1 - k))
    for k in range(stem_b):
        pairs.add((off_b1 + k, off_b2 + stem_b - 1 - k))
    return seq, pairs


def pseudoknot_study(n_designs: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    designs = [(4, 4, 4), (5, 4, 4), (4, 5, 5), (5, 5, 4), (6, 5, 4),
               (5, 6, 5), (6, 6, 4), (4, 4, 6), (5, 5, 6), (6, 6, 6)]
    for idx, (sa, sb, lp) in enumerate(designs[:n_designs]):
        seq, target = designed_pseudoknot(rng, sa, sb, lp)
        rec = RNASequence(
            seq_id=f"PK_{idx + 1:02d}",
            sequence=seq,
            tier="PK",
            source="synthetic:designed-pseudoknot",
            notes=f"stemA={sa} stemB={sb} loop={lp}",
        )
        for pk_mode in (False, True):
            graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
            if graphs.n == 0:
                continue
            model = build_level1(
                rec.sequence, graphs, cfg=VIENNA, pseudoknot_mode=pk_mode
            )
            result = CPSATSolver(max_seconds=30).solve(model)
            priority = [
                model.full.terms.get((i,), 0.0) for i in range(graphs.n)
            ]
            dec = decode(
                result.bitstring, graphs, len(rec.sequence), priority,
                pseudoknot_mode=pk_mode,
            )
            found = pairs_from_dotbracket(dec.structure)
            tp = len(found & target)
            rows.append(
                {
                    "seq_id": rec.seq_id,
                    "length": rec.length,
                    "n_stems": graphs.n,
                    "pseudoknot_mode": pk_mode,
                    "lambda_cross": model.lambda_cross,
                    "n_crossing_pairs_in_graph": len(graphs.crossing),
                    "target_pairs": len(target),
                    "found_pairs": len(found),
                    "true_positives": tp,
                    "sensitivity": round(tp / len(target), 4) if target else 1.0,
                    "ppv": round(tp / len(found), 4) if found else 1.0,
                    "structure_has_crossings": has_crossings(found),
                    "structure": dec.structure,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

def noise_study(max_stems: int, seed: int) -> pd.DataFrame:
    rows = []
    for rec in load_tier("A"):
        graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
        if not 0 < graphs.n <= max_stems:
            continue
        model = build_level1(rec.sequence, graphs, cfg=VIENNA)
        exact = solve_exact(model)
        poly = model.full
        solver = AdaptQAOASolver(max_layers=5, alpha=0.15, seed=seed,
                                 max_qubits=max_stems)
        r = solver.solve(model)
        p_single = r.solver_metadata["success_probability"]

        # The solver hands back the state it actually prepared, so the noise
        # sweep resamples that exact circuit output rather than an approximation.
        psi = r.solver_metadata["final_state"]
        diag = r.solver_metadata["diagonal"]
        rng = np.random.default_rng(seed)

        for shots in (128, 512, 2048, 8192):
            for depol in (0.0, 0.05, 0.2):
                for ro in (0.0, 0.02):
                    nm = NoiseModel(
                        shots=shots, depolarizing=depol, readout_error=ro
                    )
                    bits, energy, diag_info = best_under_noise(
                        psi, diag, graphs.n, nm, rng
                    )
                    rows.append(
                        {
                            "seq_id": rec.seq_id,
                            "n_stems": graphs.n,
                            "shots": shots,
                            "depolarizing": depol,
                            "readout_error": ro,
                            "model_energy": round(poly.energy(bits), 4),
                            "exact_model_energy": round(exact.model_energy, 4),
                            "optimizer_gap_model": round(
                                poly.energy(bits) - exact.model_energy, 4
                            ),
                            "found_optimum": abs(
                                poly.energy(bits) - exact.model_energy
                            ) < 1e-6,
                            "ideal_success_probability": round(p_single, 6),
                            "shots_for_99pct": shots_for_target_success(p_single),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Constraint enforcement
# --------------------------------------------------------------------------

#: Penalty weights are swept as *multiples of each instance's own lower bound*,
#: not as absolute values. The bound scales with the instance's coefficients, so
#: an absolute grid would put every sequence at a different point of the curve
#: and the aggregate would be meaningless.
LAMBDA_RATIOS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0)


def penalty_study(max_stems: int) -> pd.DataFrame:
    rows = []
    for rec in load_tier("A"):
        graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
        if not 0 < graphs.n <= max_stems:
            continue
        base = build_level1(rec.sequence, graphs, cfg=VIENNA)
        bound = max_single_variable_gain(base.objective)
        for ratio in LAMBDA_RATIOS:
            lam = max(round(ratio * bound, 2), 0.01)
            model = build_level1(
                rec.sequence, graphs, cfg=VIENNA,
                lambda_conflict=round(lam, 2), lambda_cross=round(lam, 2),
            )
            soft = solve_exact(model, use_penalties=True)
            hard = solve_exact(model, use_penalties=False)
            feasible = model.is_feasible(soft.selection)
            rows.append(
                {
                    "seq_id": rec.seq_id,
                    "n_stems": graphs.n,
                    "lambda_ratio": ratio,
                    "lambda": round(lam, 4),
                    "lambda_bound": round(bound, 4),
                    "feasible": feasible,
                    "penalty_objective": round(
                        model.objective.energy_of_selection(soft.selection), 4
                    ),
                    "hard_objective": round(
                        model.objective.energy_of_selection(hard.selection), 4
                    ),
                    "optimality_gap": round(
                        model.objective.energy_of_selection(soft.selection)
                        - model.objective.energy_of_selection(hard.selection),
                        4,
                    ),
                    # Ancillas a hard-constraint encoding would need instead.
                    "n_hard_constraints": len(model.hard_constraints()),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-stems", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--designs", type=int, default=10)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args(argv)

    ensure_dirs()
    print(VIENNA.header_line())

    print()
    print("=" * 88)
    print("PSEUDOKNOTS (crossing penalty off = one config change)")
    print("=" * 88)
    pk = pseudoknot_study(args.designs, args.seed)
    pk.to_csv(TABLE_DIR / "pseudoknot.csv", index=False, lineterminator="\n")
    if not pk.empty:
        print(
            pk.groupby("pseudoknot_mode")
            .agg(
                n=("seq_id", "size"),
                sensitivity=("sensitivity", "mean"),
                ppv=("ppv", "mean"),
                found_crossings=("structure_has_crossings", "mean"),
                lambda_cross=("lambda_cross", "mean"),
            )
            .round(3)
            .to_string()
        )

    print()
    print("=" * 88)
    print("CONSTRAINT ENFORCEMENT: penalty weight sweep")
    print("=" * 88)
    ps = penalty_study(args.max_stems)
    ps.to_csv(TABLE_DIR / "penalty_sweep.csv", index=False, lineterminator="\n")
    if not ps.empty:
        agg = ps.groupby("lambda_ratio").agg(
            feasible_rate=("feasible", "mean"),
            optimality_gap=("optimality_gap", "mean"),
            n=("seq_id", "size"),
        ).round(4)
        print("lambda as a multiple of the per-instance lower bound:")
        print(agg.to_string())
        pts = [
            PenaltySweepPoint(r, row.feasible_rate, row.optimality_gap, int(row.n))
            for r, row in agg.iterrows()
        ]
        print(f"\nknee (smallest lambda/bound reaching full feasibility): {knee(pts):.2f}")

    if not args.skip_noise:
        print()
        print("=" * 88)
        print("NOISE (finite sampling, depolarizing, readout)")
        print("=" * 88)
        ns = noise_study(args.max_stems, args.seed)
        ns.to_csv(TABLE_DIR / "noise.csv", index=False, lineterminator="\n")
        if not ns.empty:
            print(
                ns.groupby(["shots", "depolarizing", "readout_error"])
                .agg(
                    found_optimum=("found_optimum", "mean"),
                    opt_gap=("optimizer_gap_model", "mean"),
                )
                .round(3)
                .to_string()
            )

    print(f"\nwrote tables to {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
