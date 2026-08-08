import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rna_encoding import build_pair_qubo, nussinov_exact, pairs_to_dotbracket, decode_bits_to_dotbracket  # noqa: E402
from qubo_adapter import to_qubo_instance  # noqa: E402
from solvers import choose_n_k, run_gqe, run_pce_direct, run_simulated_annealing, run_tabu, run_random_blind, run_exact  # noqa: E402
from vienna_utils import mfe_structure, eval_structure_energy, base_pair_distance  # noqa: E402

SEQ = "GGCGCAGUAGUUCAGUCGGUUAGAAUACC"  # 29 nt, small enough for a quick smoke test


def main():
    t0 = time.time()
    print(f"sequence ({len(SEQ)} nt): {SEQ}")
    mfe_db, mfe_e = mfe_structure(SEQ)
    print(f"ViennaRNA MFE: {mfe_db}  E={mfe_e:.2f}")

    qp = build_pair_qubo(SEQ)
    print(f"pair-level QUBO: m={qp.num_vars} nnz={len(qp.Q)}")
    score, pairs = nussinov_exact(SEQ)
    nuss_db = pairs_to_dotbracket(len(SEQ), pairs)
    print(f"Nussinov-weighted exact optimum: score={score:.2f}  db={nuss_db}")

    inst = to_qubo_instance(qp, seed=0)
    n, k, cap = choose_n_k(inst.m)
    print(f"chosen encoding: n_qubits={n} k={k} capacity={cap} (m={inst.m})")

    exact = run_exact(inst, m_bruteforce_cap=20)
    print(f"exact QUBO optimum ({exact.method}): cost={exact.best_cost:.3f} ({exact.wall_s:.2f}s)")
    print(f"  matches -Nussinov score? {abs(exact.best_cost - (-score)) < 1e-6}")

    budget = 2000
    results = []
    results.append(run_gqe(inst, n, k, seed=1, max_evals=budget))
    results.append(run_pce_direct(inst, n, k, seed=1, steps=300))
    results.append(run_simulated_annealing(inst, budget_evals=budget, seed=1))
    results.append(run_tabu(inst, budget_evals=budget, seed=1))
    results.append(run_random_blind(inst, n_trials=budget, seed=1))

    print(f"\n{'method':12s} {'cost':>10s} {'gap_to_opt':>12s} {'wall_s':>8s} {'bp_dist_to_MFE':>15s} {'E(kcal/mol)':>12s}")
    for r in results:
        db = decode_bits_to_dotbracket(qp, r.best_x, SEQ)
        e = eval_structure_energy(SEQ, db)
        bpd = base_pair_distance(db, mfe_db)
        gap = r.best_cost - exact.best_cost
        print(f"{r.method:12s} {r.best_cost:10.3f} {gap:12.3f} {r.wall_s:8.2f} {bpd:15d} {e:12.2f}   {db}")

    print(f"\ntotal wall clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
