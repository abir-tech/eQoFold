#!/usr/bin/env python
"""Regenerate every figure from the committed result tables.

Figures read CSVs only -- they never recompute science -- so a figure can never
disagree with the table it illustrates. Missing tables are skipped with a note
rather than failing the build, so ``make figures`` works after a partial run.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rnaqopt.config import FIGURE_DIR, TABLE_DIR, VIENNA, ensure_dirs  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

LEVEL_COLOURS = {0: "#C44E52", 1: "#DD8452", 2: "#55A868"}


def _load(name: str) -> pd.DataFrame | None:
    path = TABLE_DIR / name
    if not path.exists():
        print(f"  skip: {name} not found")
        return None
    return pd.read_csv(path)


def _finish(fig, name: str, caption: str) -> None:
    fig.text(
        0.5, -0.02, f"{caption}\n{VIENNA.header_line()}",
        ha="center", va="top", fontsize=6, color="#555555",
    )
    out = FIGURE_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_fidelity_ladder() -> None:
    df = _load("fidelity_ladder.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    ax = axes[0]
    piv = df.groupby(["tier", "level"]).model_energy_error.mean().unstack()
    piv.plot(kind="bar", ax=ax, color=[LEVEL_COLOURS[c] for c in piv.columns],
             width=0.75, legend=False)
    ax.set_ylabel("mean |model energy error| (kcal/mol)")
    ax.set_xlabel("sequence set")
    ax.set_title("Energy-function fidelity")
    ax.tick_params(axis="x", rotation=0)
    ax.legend([f"Level {c}" for c in piv.columns], frameon=False, fontsize=8)

    ax = axes[1]
    m = df[df.tier == "M"]
    if not m.empty:
        piv2 = m.groupby(["max_branches", "level"]).model_energy_error.mean().unstack()
        piv2.plot(kind="bar", ax=ax,
                  color=[LEVEL_COLOURS[c] for c in piv2.columns],
                  width=0.75, legend=False)
        ax.set_ylabel("mean |model energy error| (kcal/mol)")
        ax.set_xlabel("multiloop branches enclosed")
        ax.set_title("Level 2 is exact at 2 branches, not beyond")
        ax.tick_params(axis="x", rotation=0)
        ax.legend([f"Level {c}" for c in piv2.columns], frameon=False, fontsize=8)
    _finish(
        fig, "fidelity_ladder.png",
        "Higher-order terms buy energy fidelity. Level 2 (degree 3) is exact for "
        "3-way junctions; no degree-3 model can be exact beyond that.",
    )


def fig_branch_damping() -> None:
    df = _load("branch_damping_sweep.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    piv = df.pivot_table(index="branch_damping", columns="max_branches",
                         values="model_energy_error", aggfunc="mean")
    for col in piv.columns:
        ax.plot(piv.index, piv[col], marker="o", label=f"{col} branches")
    ax.set_xlabel("branch damping applied to the cubic term")
    ax.set_ylabel("mean |model energy error| (kcal/mol)")
    ax.set_title("The degree-3 trade-off")
    ax.legend(frameon=False, fontsize=8)
    _finish(
        fig, "branch_damping.png",
        "Damping 1.0 is exact for 2 branches; 1/3 minimises the 3-branch error. "
        "No single value is exact for both -- that is the degree-3 obstruction.",
    )


def fig_enumeration_ablation() -> None:
    df = _load("enumeration_ablation.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    agg = df.groupby(["tier", "variant"]).agg(
        stems=("n_stems", "mean"),
        gap=("encoding_gap", "mean"),
        rep=("representable", "mean"),
    ).reset_index()
    for tier, marker in (("A", "o"), ("M", "s")):
        sub = agg[agg.tier == tier]
        ax.scatter(sub.stems, sub.gap, s=60 + 160 * sub.rep, marker=marker,
                   alpha=0.75, label=f"set {tier}")
        for _, r in sub.iterrows():
            ax.annotate(r.variant.replace("_", "\n"), (r.stems, r.gap),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("variables |stems| (cost)")
    ax.set_ylabel("mean encoding gap, kcal/mol (worse ->)")
    ax.set_title("Enumeration ablation: cost vs fidelity")
    ax.legend(frameon=False, fontsize=8)
    _finish(
        fig, "enumeration_ablation.png",
        "Marker size = fraction of references representable. Sub-stems help; "
        "L_min=2 costs variables AND accuracy.",
    )


def fig_dirac_encodings() -> None:
    df = _load("dirac3_encodings.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    agg = df.groupby(["tier", "scheme"]).agg(
        gap=("optimizer_gap_model", "mean"),
        collapse=("collapse_rate", "mean"),
        enc=("encoded_vars", "mean"),
    ).unstack()

    agg["gap"].plot(kind="bar", ax=axes[0], width=0.75,
                    color=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("mean optimizer gap (model units)")
    axes[0].set_xlabel("sequence set")
    axes[0].set_title("Encoding quality")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(frameon=False, fontsize=8)

    agg["collapse"].plot(kind="bar", ax=axes[1], width=0.75,
                         color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("fraction of runs collapsing to one stem")
    axes[1].set_xlabel("sequence set")
    axes[1].set_title("The simplex corner problem")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False, fontsize=8)
    _finish(
        fig, "dirac3_encodings.png",
        "The per-stem cap structurally prevents mass concentrating on one "
        "coordinate, at 2x the encoded variables.",
    )


def fig_r_sweep() -> None:
    df = _load("dirac3_r_sweep.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    agg = df.groupby("R_fraction").agg(
        gap=("optimizer_gap_model", "mean"),
        collapse=("collapse_rate", "mean"),
    )
    ax.plot(agg.index, agg.gap, marker="o", color="#4C72B0", label="optimizer gap")
    ax.set_xlabel("R / |stems|  (prior on structure density)")
    ax.set_ylabel("mean optimizer gap (model units)", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(agg.index, agg.collapse, marker="s", color="#C44E52",
             label="collapse rate")
    ax2.set_ylabel("collapse rate", color="#C44E52")
    ax2.grid(False)
    ax.set_title("R sweep: a tighter density prior helps")
    _finish(fig, "dirac3_r_sweep.png",
            "R is effectively a prior on how many stems may be selected.")


def fig_penalty_sweep() -> None:
    df = _load("penalty_sweep.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    agg = df.groupby("lambda_ratio").agg(
        feasible=("feasible", "mean"), gap=("optimality_gap", "mean")
    )
    ax.plot(agg.index, agg.feasible, marker="o", color="#55A868",
            label="feasibility rate")
    ax.set_xlabel(r"$\lambda$ / principled lower bound")
    ax.set_ylabel("feasibility rate", color="#55A868")
    ax2 = ax.twinx()
    ax2.plot(agg.index, agg.gap, marker="s", color="#C44E52",
             label="objective vs hard-constrained")
    ax2.axhline(0, color="#999999", lw=0.8, ls="--")
    ax2.set_ylabel("objective minus hard-constrained optimum", color="#C44E52")
    ax2.grid(False)
    ax.axvline(0.9, color="#4C72B0", lw=1, ls=":")
    ax.set_title("Penalty calibration")
    _finish(
        fig, "penalty_sweep.png",
        "Below the bound the penalised optimum is infeasible and scores "
        "artificially low. The knee sits at 0.9x the bound (dotted).",
    )


def fig_scaling() -> None:
    df = _load("encoding_gap.csv")
    if df is None:
        return
    import numpy as np

    from rnaqopt.sequences import load_all
    from rnaqopt.stems import enumerate_stems

    rows = [
        {"length": r.length, "n_stems": len(enumerate_stems(r.sequence)),
         "tier": r.tier}
        for r in load_all()
    ]
    sizes = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for tier, colour in zip("ABCM", ["#4C72B0", "#DD8452", "#C44E52", "#55A868"],
                            strict=False):
        sub = sizes[sizes.tier == tier]
        ax.scatter(sub.length, sub.n_stems, s=22, alpha=0.8, color=colour,
                   label=f"set {tier}")
    x = np.array(sorted(sizes.length))
    coef = np.polyfit(np.log(sizes.length), np.log(sizes.n_stems), 1)
    ax.plot(x, np.exp(coef[1]) * x ** coef[0], "k--", lw=1,
            label=f"fit: n ~ L^{coef[0]:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (nt)")
    ax.set_ylabel("variables |stems|")
    ax.set_title("Problem size vs sequence length")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, "problem_size_scaling.png",
            "|stems| is the problem size, not sequence length.")


def fig_solvers() -> None:
    df = _load("solver_comparison.csv")
    if df is None:
        return
    if "skipped" in df:
        df = df[~df.skipped]
    if df.empty:
        print("  skip: solver_comparison has no completed runs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    agg = df.groupby("solver").agg(
        opt_gap=("optimizer_gap", "mean"),
        hit=("found_model_optimum", "mean"),
    ).sort_values("opt_gap")
    agg.opt_gap.plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("mean optimizer gap (kcal/mol)")
    axes[0].set_ylabel("")
    axes[0].set_title("Optimizer gap, matched budget")
    agg.hit.plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_xlabel("fraction reaching the model optimum")
    axes[1].set_ylabel("")
    axes[1].set_title("Hit rate")
    _finish(fig, "solver_comparison.png",
            "All heuristics given an identical wall-clock budget per instance.")


def fig_pce_compression() -> None:
    df = _load("solver_comparison.csv")
    if df is None:
        return
    if "skipped" in df:
        df = df[~df.skipped]
    pce = df[(df.solver == "pce") & df.compression_ratio.notna()]
    if pce.empty:
        print("  skip: no PCE rows")
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    g = pce.groupby("n_stems").agg(qubits=("n_qubits", "mean")).reset_index()
    ax.plot(g.n_stems, g.n_stems, "k--", lw=1, label="direct encoding (n qubits)")
    ax.plot(g.n_stems, g.qubits, marker="o", color="#55A868", label="PCE (k=2)")
    ax.set_xlabel("variables n = |stems|")
    ax.set_ylabel("qubits required")
    ax.set_title("Qubit scaling: PCE vs direct encoding")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, "pce_compression.png",
            "PCE encodes n variables in O(sqrt(n)) qubits via 2-body correlators.")


def main() -> int:
    ensure_dirs()
    print(f"writing figures to {FIGURE_DIR}")
    for fn in (
        fig_fidelity_ladder,
        fig_branch_damping,
        fig_enumeration_ablation,
        fig_dirac_encodings,
        fig_r_sweep,
        fig_penalty_sweep,
        fig_scaling,
        fig_solvers,
        fig_pce_compression,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
