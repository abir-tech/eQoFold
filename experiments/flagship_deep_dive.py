"""Deep-dive benchmark on the challenge's own worked example sequence
(GGAGCAAAACUUGUCGAUUGAGAACAAAAUACAGAAUUUGCUUG, 44 nt), plus a designed
positive-control hairpin, with a larger evaluation budget, multiple training
seeds (for statistical comparison, matched-budget across every arm), and both
the canonical GC/AU/GU-weighted QUBO objective and a ViennaRNA-calibrated
per-stem weight objective (empirical Turner-model stem stability in place of
the flat surrogate).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dataset import CHALLENGE_EXAMPLE_SEQ, perfect_hairpin  # noqa: E402
from experiment_utils import benchmark_instance  # noqa: E402
from rna_encoding import build_stem_qubo, enumerate_candidate_stems  # noqa: E402
from vienna_utils import vienna_stem_weights  # noqa: E402

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "flagship_deep_dive.csv"

sequences = {
    "challenge_example_44nt": dict(seq=CHALLENGE_EXAMPLE_SEQ, gqe_budget=4000, pce_steps=300,
                                    sa_tabu_budget=4000, seeds=(1, 2, 3)),
    "positive_control_hairpin": dict(seq=perfect_hairpin(stem_len=6, loop_len=5), gqe_budget=1000,
                                      pce_steps=150, sa_tabu_budget=1000, seeds=(1,)),
}


def main():
    t0 = time.time()
    all_rows = []

    for seq_id, cfg in sequences.items():
        seq = cfg["seq"]
        print(f"\n=== {seq_id} ({len(seq)} nt): {seq} ===")

        qp_canon = build_stem_qubo(seq)
        print(f"stem-level QUBO (canonical weights): m={qp_canon.num_vars}")
        rows = benchmark_instance(seq, qp_canon, seq_id, weight_model="canonical",
                                   gqe_budget=cfg["gqe_budget"], pce_steps=cfg["pce_steps"],
                                   sa_tabu_budget=cfg["sa_tabu_budget"], seeds=cfg["seeds"])
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        stems = enumerate_candidate_stems(seq)
        vweights = vienna_stem_weights(seq, stems)
        qp_vienna = build_stem_qubo(seq, weight_override=vweights)
        print(f"stem-level QUBO (ViennaRNA-calibrated weights): m={qp_vienna.num_vars}")
        rows = benchmark_instance(seq, qp_vienna, seq_id, weight_model="vienna_calibrated",
                                   gqe_budget=cfg["gqe_budget"], pce_steps=cfg["pce_steps"],
                                   sa_tabu_budget=cfg["sa_tabu_budget"], seeds=cfg["seeds"],
                                   run_exact_flag=False)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        print(f"  [{seq_id}] cumulative wall clock: {time.time()-t0:.1f}s")

    print(f"\nwrote {OUT_CSV}  ({len(all_rows)} rows, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
