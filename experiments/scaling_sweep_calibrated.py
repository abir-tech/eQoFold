"""E1: the same scaling-sweep benchmark as scaling_sweep.py, but built with
ViennaRNA-calibrated stem weights (Sec. 7.3 / flagship_deep_dive.py) instead
of the flat canonical score, across every sequence length rather than just
the single flagship sequence. This tests whether GQE's advantage over the
classical heuristics and PCE-direct (scaling_sweep.py, canonical weights)
holds up under the more thermodynamically realistic objective. Stem-level
encoding only: it already covers the full length range without hitting the
n=8 qubit budget wall that stops the pair-level encoding past 30nt.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dataset import scaling_dataset  # noqa: E402
from experiment_utils import benchmark_instance  # noqa: E402
from rna_encoding import build_stem_qubo, enumerate_candidate_stems  # noqa: E402
from vienna_utils import vienna_stem_weights  # noqa: E402

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "scaling_sweep_calibrated.csv"

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

        stems = enumerate_candidate_stems(seq)
        vweights = vienna_stem_weights(seq, stems)
        qp_stem = build_stem_qubo(seq, weight_override=vweights)
        print(f"  m={qp_stem.num_vars}")
        rows = benchmark_instance(seq, qp_stem, seq_id, weight_model="vienna_calibrated",
                                   gqe_budget=GQE_BUDGET, pce_steps=PCE_STEPS,
                                   sa_tabu_budget=SA_TABU_BUDGET, seeds=(1,),
                                   run_exact_flag=True)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        print(f"  [{seq_id}] cumulative wall clock: {time.time()-t0:.1f}s")

    print(f"\nwrote {OUT_CSV}  ({len(all_rows)} rows, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
