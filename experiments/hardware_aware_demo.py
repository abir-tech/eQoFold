"""E2: a simulated (not real-hardware) design study testing whether GQE
still finds good, efficient circuits when its two-qubit gate vocabulary is
restricted to a physically realistic nearest-neighbor chain instead of the
default all-to-all connectivity. Uses the same instance, budget, and seeds
as the existing all-to-all flagship GQE run (flagship_deep_dive.csv,
challenge_example_44nt, canonical weights, budget=4000, seeds 1-3), so the
two are directly comparable without rerunning the all-to-all side.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dataset import CHALLENGE_EXAMPLE_SEQ  # noqa: E402
from hardware_aware import build_linear_chain_vocab, linear_chain_edges  # noqa: E402
from qubo_adapter import to_qubo_instance  # noqa: E402
from rna_encoding import build_stem_qubo, decode_bits_to_dotbracket  # noqa: E402
from solvers import choose_n_k, run_gqe  # noqa: E402
from vienna_utils import base_pair_distance, eval_structure_energy, mfe_structure  # noqa: E402

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "hardware_aware_demo.csv"

GQE_BUDGET = 4000
SEEDS = (1, 2, 3)


def main():
    t0 = time.time()
    seq = CHALLENGE_EXAMPLE_SEQ
    mfe_db, mfe_e = mfe_structure(seq)
    qp = build_stem_qubo(seq)
    inst = to_qubo_instance(qp, seed=0)
    n, k, cap = choose_n_k(qp.num_vars)
    edges = linear_chain_edges(n)
    print(f"seq={seq}  m={qp.num_vars}  n_qubits={n}  k={k}")
    print(f"linear-chain edges (n={n}): {edges}")

    rows = []
    for seed in SEEDS:
        vocab = build_linear_chain_vocab(n)
        r = run_gqe(inst, n, k, seed=seed, max_evals=GQE_BUDGET, vocab_override=vocab)
        db = decode_bits_to_dotbracket(qp, r.best_x, seq)
        e = eval_structure_energy(seq, db)
        bpd = base_pair_distance(db, mfe_db)
        row = dict(seed=seed, topology="linear_chain", n_qubits=n, k=k,
                   n_2q_pairs_available=len(edges), qubo_cost=r.best_cost,
                   n_evals=r.n_evals, wall_s=r.wall_s, n_gates=r.n_gates,
                   structure=db, energy=e, mfe_energy=mfe_e, energy_gap=e - mfe_e,
                   bp_distance=bpd)
        rows.append(row)
        print(f"  seed={seed}  cost={r.best_cost:.2f}  gates={r.n_gates}  "
              f"E={e:.2f}  bp_dist={bpd}  wall={r.wall_s:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
