"""Turn results/scaling_sweep.csv into the statistical summary tables the
report draws on: matched-budget solver-quality comparison (bootstrap CI on
mean QUBO-cost gap-to-exact, Wilson CI on exact-match rate, exact McNemar
test GQE vs. each baseline), and structure-quality-vs-length aggregates.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[0]))  # repo root, for qms.stats

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qms.stats import mcnemar_exact, paired_bootstrap_diff, wilson_ci  # noqa: E402

RESULTS = ROOT / "results"
METHOD_ORDER = ["GQE", "PCE-direct", "SimAnneal", "Tabu", "Blind"]


def main():
    df = pd.read_csv(RESULTS / "scaling_sweep.csv")
    df = df[df.status == "ok"].copy()
    df["exact_match"] = df["gap_to_exact"].abs() < 1e-6

    print("=== overall solver quality (all instances, all encodings, matched eval budget) ===")
    for m in METHOD_ORDER:
        sub = df[df.method == m]
        if len(sub) == 0:
            continue
        gap = sub["gap_to_exact"].values
        n_ok = int((sub["exact_match"]).sum())
        wci = wilson_ci(n_ok, len(sub))
        print(f"{m:12s}  n={len(sub):3d}  exact_match_rate={wci}  mean_gap={np.nanmean(gap):8.3f}  "
              f"median_gap={np.nanmedian(gap):8.3f}")

    print("\n=== paired comparison vs GQE (same instances only) ===")
    piv = df.pivot_table(index=["seq_id", "encoding"], columns="method", values="exact_match", aggfunc="first")
    piv_gap = df.pivot_table(index=["seq_id", "encoding"], columns="method", values="gap_to_exact", aggfunc="first")
    if "GQE" in piv.columns:
        for m in METHOD_ORDER:
            if m == "GQE" or m not in piv.columns:
                continue
            paired = piv[["GQE", m]].dropna()
            if len(paired) < 2:
                continue
            n01, n10, p = mcnemar_exact(paired["GQE"].values, paired[m].values)
            gap_paired = piv_gap[["GQE", m]].dropna()
            diff, lo, hi = paired_bootstrap_diff(gap_paired["GQE"].values, gap_paired[m].values, seed=0)
            print(f"GQE vs {m:12s}  n={len(paired):3d}  McNemar p={p:.3f} (n01={n01}, n10={n10})  "
                  f"mean(gap_GQE - gap_{m})={diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== structure quality vs MFE, instances where MFE has >=1 pair only ===")
    d2 = df[df.n_mfe_pairs > 0]
    print(f"({d2['seq_id'].nunique()} sequences with a non-trivial MFE fold, out of {df['seq_id'].nunique()} total)")
    for m in METHOD_ORDER:
        sub = d2[d2.method == m]
        if len(sub) == 0:
            continue
        print(f"{m:12s}  n={len(sub):3d}  mean_bp_F1={sub.bp_f1.mean():.3f}  "
              f"mean_energy_gap={sub.energy_gap.mean():7.2f} kcal/mol  mean_bp_dist={sub.bp_distance.mean():.2f}")

    print("\n=== QUBO size m and chosen qubit count n by length/encoding ===")
    sizes = df.drop_duplicates(["seq_id", "encoding"])[["seq_id", "length", "encoding", "m", "n_qubits", "k", "pce_capacity"]]
    print(sizes.sort_values(["encoding", "length"]).to_string(index=False))

    print("\n=== circuit resources: GQE vs PCE-direct gate counts ===")
    import ast
    for m in ["GQE", "PCE-direct"]:
        sub = df[(df.method == m) & df.n_gates.notna()]
        totals = sub.n_gates.apply(lambda s: ast.literal_eval(s).get("n_total", np.nan) if isinstance(s, str) else np.nan)
        if len(totals.dropna()):
            print(f"{m:12s}  mean_total_gates={totals.mean():.1f}  min={totals.min():.0f}  max={totals.max():.0f}")


if __name__ == "__main__":
    main()
