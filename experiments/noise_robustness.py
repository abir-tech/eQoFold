"""Optional advanced task: how much does a trained GQE circuit's decoded RNA
structure degrade under (a) finite measurement shots, (b) global depolarizing
noise, and (c) a device-calibrated gate-infidelity + readout(SPAM) noise
model? Trains one GQE circuit at the exact-statevector (noiseless) objective
(matching this repo's own established practice, e.g. Sprint 2 arm N4's
documented rationale in qms/gqe/reward.py), then re-evaluates its readout
under each noise condition with repeated random shot draws, reporting how
often the decoded structure still matches the noiseless decode and how the
ViennaRNA base-pair distance / free-energy gap degrade.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dataset import CHALLENGE_EXAMPLE_SEQ  # noqa: E402
from qubo_adapter import to_qubo_instance  # noqa: E402
from rna_encoding import build_stem_qubo, decode_bits_to_dotbracket  # noqa: E402
from solvers import choose_n_k, get_assignment  # noqa: E402
from vienna_utils import base_pair_distance, eval_structure_energy, mfe_structure  # noqa: E402

from qms.gqe.executor import build_unitary_table, execute_batch  # noqa: E402
from qms.gqe.train import RegimeAConfig, train_regime_a  # noqa: E402
from qms.gqe.vocab import GQEVocab  # noqa: E402
from qms.pce.decode import joint_neighborhood_search, repair_budget, repair_domain_wall, sign_readout  # noqa: E402
from qms.pce.loss import correlators  # noqa: E402
from qms.pce.shot_noise import (all_string_shot_estimates, all_string_shot_estimates_depolarized,  # noqa: E402
                                 all_string_shot_estimates_device_noise, gather_assignment_estimates)

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "noise_robustness.csv"

GQE_BUDGET = 3000
N_SHOT_SEEDS = 10
SHOT_COUNTS = (64, 256, 1024, 4096)
DEPOL_PROBS = (0.0, 0.01, 0.02, 0.05, 0.10)
DEVICE_CONDITIONS = [  # illustrative near-term-hardware-scale rates, NOT fit to any specific device
    dict(name="ideal", p_gate=0.0, p_spam=0.0),
    dict(name="low_noise", p_gate=0.005, p_spam=0.01),
    dict(name="moderate_noise", p_gate=0.02, p_spam=0.03),
    dict(name="high_noise", p_gate=0.05, p_spam=0.05),
]


def decode_from_mu(mu: np.ndarray, inst):
    x0 = repair_budget(repair_domain_wall(sign_readout(mu.copy()), inst), inst)
    x1 = joint_neighborhood_search(x0, inst, large_b_boost=True)
    c0, c1 = inst.cost(x0), inst.cost(x1)
    return (x1, c1) if c1 <= c0 else (x0, c0)


def main():
    t0 = time.time()
    seq = CHALLENGE_EXAMPLE_SEQ
    mfe_db, mfe_e = mfe_structure(seq)
    qp = build_stem_qubo(seq)
    inst = to_qubo_instance(qp, seed=0)
    n, k, cap = choose_n_k(qp.num_vars)
    print(f"seq={seq}  m={qp.num_vars}  n_qubits={n}  k={k}")

    assignment = get_assignment(inst, n, k, seed=1)
    vocab = GQEVocab(n)
    unitary_table = build_unitary_table(vocab)
    cfg = RegimeAConfig(max_evals=GQE_BUDGET, group_size=16, n_iter=4, replay_batch_size=64,
                         buffer_capacity=2000, lr=3e-4, checkpoints=(GQE_BUDGET,),
                         objective="dpo", seed=1, large_b_boost=True)
    result = train_regime_a(inst, vocab, unitary_table, n, assignment, None, cfg)
    tokens = torch.tensor(result.best_tokens[None], dtype=torch.long)
    with torch.no_grad():
        state = execute_batch(tokens, unitary_table, n)
        exact_pi = correlators(state, assignment, n).numpy()[0]

    x_exact, cost_exact = decode_from_mu(exact_pi, inst)
    db_exact = decode_bits_to_dotbracket(qp, x_exact, seq)
    e_exact = eval_structure_energy(seq, db_exact)
    bpd_exact = base_pair_distance(db_exact, mfe_db)
    print(f"noiseless decode: cost={cost_exact:.2f} bp_dist_to_MFE={bpd_exact} E={e_exact:.2f}")
    print(f"structure: {db_exact}")

    rows = []
    rows.append(dict(condition="noiseless", param=None, seed=None, cost=cost_exact,
                      exact_match=True, bp_dist=bpd_exact, energy=e_exact,
                      bp_dist_to_noiseless=0))

    # (a) finite shot noise
    for N in SHOT_COUNTS:
        for seed in range(N_SHOT_SEEDS):
            mu_all, sigma_all, strings = all_string_shot_estimates(state, n, k, N, seed=seed * 7919 + N)
            mu, _ = gather_assignment_estimates(mu_all, sigma_all, strings, assignment)
            x, c = decode_from_mu(mu.numpy()[0], inst)
            db = decode_bits_to_dotbracket(qp, x, seq)
            e = eval_structure_energy(seq, db)
            rows.append(dict(condition="shot_noise", param=N, seed=seed, cost=c,
                              exact_match=bool(np.array_equal(x, x_exact)),
                              bp_dist=base_pair_distance(db, mfe_db), energy=e,
                              bp_dist_to_noiseless=base_pair_distance(db, db_exact)))

    # (b) global depolarizing noise at a generous fixed shot count (isolates gate/state noise from shot noise)
    for p in DEPOL_PROBS:
        for seed in range(N_SHOT_SEEDS):
            mu_all, strings = all_string_shot_estimates_depolarized(state, n, k, 4096, seed=seed * 104729 + int(p * 1000), p=p)
            mu, _ = gather_assignment_estimates(mu_all, torch.zeros_like(mu_all), strings, assignment)
            x, c = decode_from_mu(mu.numpy()[0], inst)
            db = decode_bits_to_dotbracket(qp, x, seq)
            e = eval_structure_energy(seq, db)
            rows.append(dict(condition="depolarizing", param=p, seed=seed, cost=c,
                              exact_match=bool(np.array_equal(x, x_exact)),
                              bp_dist=base_pair_distance(db, mfe_db), energy=e,
                              bp_dist_to_noiseless=base_pair_distance(db, db_exact)))

    # (c) device-calibrated gate + SPAM noise
    for dc in DEVICE_CONDITIONS:
        for seed in range(N_SHOT_SEEDS):
            mu_all, strings = all_string_shot_estimates_device_noise(
                state, n, k, 4096, seed=seed * 15485863 + hash(dc["name"]) % 10000,
                p_gate_effective=dc["p_gate"], p_spam=dc["p_spam"])
            mu, _ = gather_assignment_estimates(mu_all, torch.zeros_like(mu_all), strings, assignment)
            x, c = decode_from_mu(mu.numpy()[0], inst)
            db = decode_bits_to_dotbracket(qp, x, seq)
            e = eval_structure_energy(seq, db)
            rows.append(dict(condition=f"device_{dc['name']}", param=str(dc), seed=seed, cost=c,
                              exact_match=bool(np.array_equal(x, x_exact)),
                              bp_dist=base_pair_distance(db, mfe_db), energy=e,
                              bp_dist_to_noiseless=base_pair_distance(db, db_exact)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows, {time.time()-t0:.1f}s total)")
    print(df.groupby("condition")[["exact_match", "bp_dist_to_noiseless"]].mean())


if __name__ == "__main__":
    main()
