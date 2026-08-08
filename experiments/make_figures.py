"""Generate the figure set for the WISER<>Moderna submission from
results/scaling_sweep.csv, results/flagship_deep_dive.csv, and
results/noise_robustness.csv. Robust to any of these being partially written
or absent (background experiments may still be running).
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from style_wiser import METHOD_PALETTE, WEIGHT_PALETTE, panel_label, savefig, setup, wilson_ci  # noqa: E402

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
setup()

METHOD_ORDER = ["GQE", "PCE-direct", "SimAnneal", "Tabu", "Blind"]


def _parse_gates(s):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def load_sweep():
    p = RESULTS / "scaling_sweep.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["status"] == "ok"].copy()
    df["n_gates_parsed"] = df["n_gates"].apply(_parse_gates)
    df["n_gates_total"] = df["n_gates_parsed"].apply(lambda d: d.get("n_total", np.nan))
    return df


def fig_scaling_qubits(df):
    sizes = (df.drop_duplicates(["seq_id", "encoding"])
               .groupby(["encoding", "length"])
               .agg(m=("m", "mean"), n_qubits=("n_qubits", "mean")).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for enc, color in [("pair", "#0F766E"), ("stem", "#4338CA")]:
        sub = sizes[sizes.encoding == enc].sort_values("length")
        if len(sub) == 0:
            continue
        axes[0].plot(sub.length, sub.m, "o-", color=color, label=enc, markersize=4)
        axes[1].plot(sub.length, sub.n_qubits, "o-", color=color, label=enc, markersize=4)
    axes[0].set_xlabel("sequence length (nt)")
    axes[0].set_ylabel("QUBO variables $m$\n(candidate pairs / stems)")
    axes[0].legend(frameon=False, title="encoding")
    panel_label(axes[0], "a")
    axes[1].set_xlabel("sequence length (nt)")
    axes[1].set_ylabel("qubits $n$ (PCE-compressed)")
    axes[1].legend(frameon=False, title="encoding")
    panel_label(axes[1], "b")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig01_scaling_qubits.pdf")
    plt.close(fig)


def fig_solver_quality(df):
    d = df.dropna(subset=["gap_to_exact"]).copy()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    positions = range(len(METHOD_ORDER))
    data = [d[d.method == m]["gap_to_exact"].values for m in METHOD_ORDER]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_PALETTE[m])
        patch.set_alpha(0.55)
    y_cap = 25  # a handful of large-instance outliers (esp. Tabu) are annotated instead of shown in-frame
    for m, x in zip(METHOD_ORDER, positions):
        y = d[d.method == m]["gap_to_exact"].values
        if len(y):
            y_plot = np.minimum(y, y_cap)
            jitter = np.random.default_rng(0).normal(x, 0.06, size=len(y))
            ax.scatter(jitter, y_plot, s=8, color=METHOD_PALETTE[m], alpha=0.5, zorder=3)
            for xi, yi, yorig in zip(jitter, y_plot, y):
                if yorig > y_cap:
                    ax.annotate(f"{yorig:.0f}", (xi, y_cap), textcoords="offset points",
                                xytext=(0, 3), fontsize=6, ha="center", color=METHOD_PALETTE[m])
    ax.set_xticks(list(positions))
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
    ax.set_ylabel("QUBO cost gap to exact optimum\n(lower is better; 0 = optimal)")
    ax.set_ylim(-1, y_cap + 3)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig02_solver_quality.pdf")
    plt.close(fig)


def fig_structure_quality(df):
    d = df[df["n_mfe_pairs"] > 0].copy()
    if len(d) == 0:
        print("no instances with a non-trivial MFE fold yet -- skipping fig_structure_quality")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for m in METHOD_ORDER:
        sub = d[d.method == m].groupby("length")["bp_f1"].mean().reset_index()
        if len(sub) == 0:
            continue
        axes[0].plot(sub.length, sub.bp_f1, "o-", color=METHOD_PALETTE[m], label=m, markersize=4)
        sub2 = d[d.method == m].groupby("length")["energy_gap"].mean().reset_index()
        axes[1].plot(sub2.length, sub2.energy_gap, "o-", color=METHOD_PALETTE[m], label=m, markersize=4)
    axes[0].set_xlabel("sequence length (nt)")
    axes[0].set_ylabel("base-pair F1 vs ViennaRNA MFE")
    axes[0].set_ylim(-0.05, 1.05)
    panel_label(axes[0], "a")
    axes[1].set_xlabel("sequence length (nt)")
    axes[1].set_ylabel("energy gap to MFE (kcal/mol)\nlower is better")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=1)
    panel_label(axes[1], "b")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig03_structure_quality.pdf")
    plt.close(fig)


def fig_runtime(df):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in METHOD_ORDER:
        sub = df[df.method == m].sort_values("m")
        if len(sub) == 0:
            continue
        ax.scatter(sub["m"], sub["wall_s"], s=10, color=METHOD_PALETTE[m], label=m, alpha=0.7)
    ax.set_xlabel("QUBO size $m$")
    ax.set_ylabel("wall-clock (s)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig04_runtime.pdf")
    plt.close(fig)


def fig_gate_resources(df):
    d = df.dropna(subset=["n_gates_total"])
    d = d[d.method.isin(["GQE", "PCE-direct"])]
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in ["GQE", "PCE-direct"]:
        sub = d[d.method == m].groupby("n_qubits")["n_gates_total"].mean().reset_index()
        ax.plot(sub.n_qubits, sub.n_gates_total, "o-", color=METHOD_PALETTE[m], label=m)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("total gate count")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig05_gate_resources.pdf")
    plt.close(fig)


def fig_weight_model(flagship_path):
    if not flagship_path.exists():
        return
    df = pd.read_csv(flagship_path)
    df = df[(df.status == "ok") & (df.seq_id == "challenge_example_44nt")]
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(METHOD_ORDER))
    width = 0.35
    for i, wm in enumerate(["canonical", "vienna_calibrated"]):
        means = [df[(df.method == m) & (df.weight_model == wm)]["energy_gap"].mean() for m in METHOD_ORDER]
        ax.bar(x + (i - 0.5) * width, means, width, label=wm, color=WEIGHT_PALETTE[wm], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
    ax.set_ylabel("energy gap to MFE (kcal/mol)")
    ax.legend(frameon=False, title="QUBO weight model")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig06_weight_model_comparison.pdf")
    plt.close(fig)


def fig_noise(noise_path):
    if not noise_path.exists():
        return
    df = pd.read_csv(noise_path)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))

    shot = df[df.condition == "shot_noise"].copy()
    if len(shot):
        shot["param"] = shot["param"].astype(float)
        agg = shot.groupby("param")["bp_dist_to_noiseless"].agg(["mean", "std", "count"]).reset_index()
        axes[0].errorbar(agg["param"], agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                          marker="o", color="#0F766E")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("shots $N$")
        axes[0].set_ylabel("bp-distance to noiseless decode")
        panel_label(axes[0], "a")

    depol = df[df.condition == "depolarizing"]
    if len(depol):
        depol = depol.copy()
        depol["param"] = depol["param"].astype(float)
        agg = depol.groupby("param")["bp_dist_to_noiseless"].agg(["mean", "std", "count"]).reset_index()
        axes[1].errorbar(agg["param"], agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                          marker="o", color="#B45309")
        axes[1].set_xlabel("depolarizing probability $p$")
        axes[1].set_ylabel("bp-distance to noiseless decode")
        panel_label(axes[1], "b")

    dev = df[df.condition.str.startswith("device_")]
    if len(dev):
        order = ["device_ideal", "device_low_noise", "device_moderate_noise", "device_high_noise"]
        agg = dev.groupby("condition")["bp_dist_to_noiseless"].agg(["mean", "std", "count"]).reindex(order).reset_index()
        axes[2].bar(range(len(agg)), agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]), color="#4338CA", alpha=0.8)
        axes[2].set_xticks(range(len(agg)))
        axes[2].set_xticklabels([c.replace("device_", "") for c in agg["condition"]], rotation=20, ha="right")
        axes[2].set_ylabel("bp-distance to noiseless decode")
        panel_label(axes[2], "c")

    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig07_noise_robustness.pdf")
    plt.close(fig)


def main():
    df = load_sweep()
    if df is not None and len(df):
        fig_scaling_qubits(df)
        fig_solver_quality(df)
        fig_structure_quality(df)
        fig_runtime(df)
        fig_gate_resources(df)
    else:
        print("results/scaling_sweep.csv not ready yet")

    fig_weight_model(RESULTS / "flagship_deep_dive.csv")
    fig_noise(RESULTS / "noise_robustness.csv")
    print("done")


if __name__ == "__main__":
    main()
