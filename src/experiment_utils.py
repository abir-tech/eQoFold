"""Shared benchmarking harness: given an RNA sequence and a built QUBOProblem,
run every solver at a matched evaluation budget, decode each result to a
dot-bracket structure, and score it against the ViennaRNA MFE reference. One
row per (sequence, encoding, method, seed).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qubo_adapter import to_qubo_instance  # noqa: E402
from rna_encoding import QUBOProblem, decode_bits_to_dotbracket, nussinov_exact, pairs_to_dotbracket  # noqa: E402
from solvers import (PCECapacityError, choose_n_k, run_exact, run_gqe,  # noqa: E402
                      run_pce_direct, run_random_blind, run_simulated_annealing, run_tabu)
from vienna_utils import base_pair_distance, base_pair_f1, dotbracket_to_pairs, eval_structure_energy, mfe_structure  # noqa: E402


def benchmark_instance(seq: str, qp: QUBOProblem, seq_id: str, weight_model: str = "canonical",
                        gqe_budget: int = 1200, pce_steps: int = 150, sa_tabu_budget: int = 1200,
                        seeds=(1,), n_max: int = 8, run_exact_flag: bool = True,
                        verbose: bool = True) -> list[dict]:
    rows = []
    m = qp.num_vars
    if m == 0:
        return [dict(seq_id=seq_id, length=len(seq), encoding=qp.kind, weight_model=weight_model,
                      method="N/A", status="no_candidate_pairs", m=0)]

    try:
        n, k, cap = choose_n_k(m, n_max=n_max)
    except PCECapacityError as e:
        return [dict(seq_id=seq_id, length=len(seq), encoding=qp.kind, weight_model=weight_model,
                      method="N/A", status=f"exceeds_qubit_budget: {e}", m=m)]

    inst = to_qubo_instance(qp, seed=0)
    mfe_db, mfe_e = mfe_structure(seq)
    mfe_pairs = dotbracket_to_pairs(mfe_db)

    exact_cost = None
    exact_method = None
    if run_exact_flag:
        if qp.kind == "pair" and weight_model == "canonical":
            score, _ = nussinov_exact(seq)
            exact_cost, exact_method = -score, "Nussinov-DP"
        else:
            er = run_exact(inst, m_bruteforce_cap=20, mip_time_limit_s=20.0)
            if er.best_x is not None:
                exact_cost, exact_method = er.best_cost, er.method

    def record(method, r, wall_extra=None):
        db = decode_bits_to_dotbracket(qp, r.best_x, seq) if r.best_x is not None else "." * len(seq)
        e = eval_structure_energy(seq, db)
        pred_pairs = dotbracket_to_pairs(db)
        f1_stats = base_pair_f1(pred_pairs, mfe_pairs)
        bpd = base_pair_distance(db, mfe_db)
        row = dict(seq_id=seq_id, length=len(seq), seq=seq, encoding=qp.kind, weight_model=weight_model,
                   m=m, n_qubits=n, k=k, pce_capacity=cap, method=method, status="ok",
                   qubo_cost=r.best_cost, n_evals=r.n_evals, wall_s=r.wall_s,
                   exact_cost=exact_cost, exact_method=exact_method,
                   gap_to_exact=(r.best_cost - exact_cost) if exact_cost is not None else None,
                   n_gates=r.n_gates, structure=db, mfe_structure=mfe_db,
                   energy=e, mfe_energy=mfe_e, energy_gap=e - mfe_e,
                   bp_distance=bpd, bp_f1=f1_stats["f1"], bp_precision=f1_stats["precision"],
                   bp_recall=f1_stats["recall"], n_pred_pairs=f1_stats["n_pred"], n_mfe_pairs=f1_stats["n_ref"])
        rows.append(row)
        if verbose:
            print(f"  [{seq_id}/{qp.kind}/{weight_model}] {method:10s} cost={r.best_cost:9.2f} "
                  f"E={e:8.2f} bp_dist={bpd:3d} F1={f1_stats['f1']:.2f} wall={r.wall_s:6.1f}s")

    for seed in seeds:
        try:
            record("GQE", run_gqe(inst, n, k, seed=seed, max_evals=gqe_budget))
        except Exception:
            if verbose:
                traceback.print_exc()
        try:
            record("PCE-direct", run_pce_direct(inst, n, k, seed=seed, steps=pce_steps))
        except Exception:
            if verbose:
                traceback.print_exc()
        record("SimAnneal", run_simulated_annealing(inst, budget_evals=sa_tabu_budget, seed=seed))
        record("Tabu", run_tabu(inst, budget_evals=sa_tabu_budget, seed=seed))
        record("Blind", run_random_blind(inst, n_trials=gqe_budget, seed=seed))

    return rows
