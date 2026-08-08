"""Main scaling experiment: for RNA sequences of increasing length, build
both the pair-level and stem-level QUBO encodings, run GQE / PCE-direct /
simulated annealing / tabu / blind-control at matched evaluation budgets, and
record qubit count, gate/circuit resources, wall-clock, and structure quality
against the ViennaRNA MFE reference. This is the primary data source for the
challenge's "scaling and quantum resource analysis" deliverable.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dataset import scaling_dataset  # noqa: E402
from experiment_utils import benchmark_instance  # noqa: E402
from rna_encoding import build_pair_qubo, build_stem_qubo  # noqa: E402

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "scaling_sweep.csv"

GQE_BUDGET = 1200
PCE_STEPS = 150
SA_TABU_BUDGET = 1200
LENGTHS = (10, 14, 18, 22, 26, 30, 36, 44)


def main():
    t0 = time.time()
    seqs = scaling_dataset(lengths=LENGTHS, seeds_per_length=1, gc_bias=0.55)
    all_rows = []
    for row in seqs:
        seq, L, seed = row["seq"], row["length"], row["seed"]
        seq_id = f"L{L}_s{seed}"
        print(f"\n=== {seq_id}  ({seq}) ===")

        qp_pair = build_pair_qubo(seq)
        rows = benchmark_instance(seq, qp_pair, seq_id, weight_model="canonical",
                                   gqe_budget=GQE_BUDGET, pce_steps=PCE_STEPS,
                                   sa_tabu_budget=SA_TABU_BUDGET, seeds=(1,))
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        qp_stem = build_stem_qubo(seq)
        rows = benchmark_instance(seq, qp_stem, seq_id, weight_model="canonical",
                                   gqe_budget=GQE_BUDGET, pce_steps=PCE_STEPS,
                                   sa_tabu_budget=SA_TABU_BUDGET, seeds=(1,))
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        print(f"  [{seq_id}] cumulative wall clock: {time.time()-t0:.1f}s")

    print(f"\nwrote {OUT_CSV}  ({len(all_rows)} rows, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
