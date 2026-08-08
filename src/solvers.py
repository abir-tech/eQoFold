"""Solver wrappers around the repo's qms.gqe / qms.pce machinery plus
matched-budget classical baselines, all operating on a degenerate
QUBOInstance (see qubo_adapter.py) built from an RNA rna_encoding.QUBOProblem.

Every solver reports (best_x, best_cost, n_evals, wall_s, ...) so results are
directly comparable at matched evaluation budgets, following this repo's own
established practice (never compare methods run at different evaluation
budgets -- see qms/stats.py and experiments/paper_s2_w3a_sa_tabu_certification.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np
import torch

from qubo_adapter import REPO_ROOT, to_qubo_instance  # noqa: E402 (sets sys.path)

from qms.gqe.executor import build_unitary_table, execute_batch, sequence_gate_count  # noqa: E402
from qms.gqe.train import RegimeAConfig, train_regime_a  # noqa: E402
from qms.gqe.vocab import GQEVocab  # noqa: E402
from qms.instance_factory import QUBOInstance  # noqa: E402
from qms.mip_baseline import solve_mip  # noqa: E402
from qms.oracle import vectorized_oracle  # noqa: E402
from qms.pce.decode import (joint_neighborhood_search, repair_budget,  # noqa: E402
                             repair_domain_wall, sign_readout)
from qms.pce.loss import correlators  # noqa: E402
from qms.pce.pauli_families import graph_aware_assignment, random_assignment  # noqa: E402
from qms.pce.pipeline import solve_pce_batch  # noqa: E402


def choose_n_k(m: int, n_max: int = 8, k_max: int = 4) -> tuple[int, int, int]:
    """Smallest qubit count n (then smallest locality k) whose PCE capacity
    3*C(n,k) covers m variables. Preferring low k at fixed n favors
    lower-weight (more hardware-measurable) Pauli strings when there is slack.
    n_max=8 by default: GQE's dense unitary-table executor costs O(4^n) memory
    (~900MB already at n=8; ~4.5GB at n=9 per qms/gqe/executor.py's own
    docstring), so we cap qubit count there rather than risk exhausting this
    machine's memory. Instances whose m exceeds capacity at n_max=8 raise
    PCECapacityError -- callers should treat that as a real scaling-limit
    data point (this problem size needs the scalable executor / more qubits),
    not silently retry at a larger n."""
    for n in range(4, n_max + 1):
        for k in range(2, min(k_max, n - 1) + 1):
            cap = 3 * comb(n, k)
            if cap >= m:
                return n, k, cap
    raise PCECapacityError(f"m={m} exceeds PCE capacity up to n={n_max}, k={k_max} "
                            f"(max capacity {3 * comb(n_max, min(k_max, n_max - 1))})")


class PCECapacityError(ValueError):
    pass


@dataclass
class SolveResult:
    method: str
    best_x: np.ndarray
    best_cost: float
    n_evals: int
    wall_s: float
    n_qubits: Optional[int] = None
    n_gates: Optional[dict] = None
    extra: Optional[dict] = None


def sequence_gate_depth(tokens: np.ndarray, vocab: GQEVocab) -> int:
    """Circuit depth of one GQE token sequence: greedy (ASAP) layering over
    each gate's target qubits, same convention as qms.pce.ansatz's layered
    depth for the fixed PCE-direct ansatz, so the two are directly
    comparable. Gates on disjoint qubits are treated as parallel; a gate
    touching a qubit already scheduled at layer L is placed at layer L+1."""
    layer = {}
    depth = 0
    for tid in tokens[1:]:
        if tid == vocab.EOS_ID:
            break
        tok = vocab.tokens[tid]
        if tok.kind == "special":
            continue
        cur = max((layer.get(q, 0) for q in tok.qubits), default=0)
        new_layer = cur + 1
        for q in tok.qubits:
            layer[q] = new_layer
        depth = max(depth, new_layer)
    return depth


def get_assignment(inst: QUBOInstance, n: int, k: int, seed: int, mode: str = "random"):
    if mode == "graph_aware":
        return graph_aware_assignment(n, k, inst.m, inst.Q, seed=seed)
    return random_assignment(n, k, inst.m, seed=seed)


def _decode_stages(pi: np.ndarray, inst: QUBOInstance, radius: int = 1, large_b_boost: bool = False):
    x0 = repair_budget(repair_domain_wall(sign_readout(pi.copy()), inst), inst)
    x1 = joint_neighborhood_search(x0, inst, radius=radius, large_b_boost=large_b_boost)
    return x0, x1


def run_gqe(inst: QUBOInstance, n: int, k: int, seed: int = 0, max_evals: int = 3000,
            group_size: int = 16, assignment=None, assignment_mode: str = "random",
            objective: str = "dpo", large_b_boost: bool = True, vocab_override=None) -> SolveResult:
    """vocab_override: a pre-built GQEVocab to train against instead of the
    default all-to-all one (e.g. wiser.src.hardware_aware's connectivity-
    constrained vocab), for the hardware-aware circuit-generation study."""
    t0 = time.time()
    vocab = vocab_override if vocab_override is not None else GQEVocab(n)
    unitary_table = build_unitary_table(vocab)
    if assignment is None:
        assignment = get_assignment(inst, n, k, seed, mode=assignment_mode)
    cfg = RegimeAConfig(max_evals=max_evals, group_size=group_size, n_iter=4,
                         replay_batch_size=64, buffer_capacity=2000, lr=3e-4,
                         checkpoints=(max_evals,), objective=objective, seed=seed,
                         large_b_boost=large_b_boost)
    result = train_regime_a(inst, vocab, unitary_table, n, assignment, None, cfg)
    with torch.no_grad():
        state = execute_batch(torch.tensor(result.best_tokens[None], dtype=torch.long), unitary_table, n)
        pi = correlators(state, assignment, n).numpy()[0]
    x0, x1 = _decode_stages(pi, inst, large_b_boost=large_b_boost)
    cost0, cost1 = inst.cost(x0), inst.cost(x1)
    best_x, best_cost = (x1, cost1) if cost1 <= cost0 else (x0, cost0)
    gates = sequence_gate_count(result.best_tokens, vocab)
    gates["depth"] = sequence_gate_depth(result.best_tokens, vocab)
    return SolveResult(method="GQE", best_x=best_x, best_cost=float(best_cost), n_evals=max_evals,
                        wall_s=time.time() - t0, n_qubits=n, n_gates=gates,
                        extra=dict(k=k, assignment_mode=assignment_mode, objective=objective,
                                   cost_no_ls=float(cost0), cost_with_ls=float(cost1),
                                   telemetry=result.telemetry))


def run_pce_direct(inst: QUBOInstance, n: int, k: int, seed: int = 0, steps: int = 300,
                    assignment=None, assignment_mode: str = "random",
                    large_b_boost: bool = True) -> SolveResult:
    t0 = time.time()
    if assignment is None:
        assignment = get_assignment(inst, n, k, seed, mode=assignment_mode)
    result, _ = solve_pce_batch(inst.c[None, :], inst.Q[None, :, :], n=n, k=k, beta=0.5,
                                 steps=steps, seed=seed, assignment=[assignment])
    pi = result.pi_vals[0].numpy()
    x0, x1 = _decode_stages(pi, inst, large_b_boost=large_b_boost)
    cost0, cost1 = inst.cost(x0), inst.cost(x1)
    best_x, best_cost = (x1, cost1) if cost1 <= cost0 else (x0, cost0)
    # circuit resource accounting for the fixed brickwork ansatz (see ansatz.recommended_depth)
    from qms.pce.ansatz import build_spec, recommended_depth
    depth = recommended_depth(n)
    spec = build_spec(n, depth)
    n_1q = sum(layer["count"] for layer in spec.layers if layer["kind"] == "rot")
    n_2q = sum(3 for layer in spec.layers if layer["kind"] == "ent")  # RXX+RYY+RZZ per entangling pair
    n_gates = dict(n_1q=n_1q, n_2q=n_2q, n_total=n_1q + n_2q, depth=depth, n_params=spec.n_params)
    return SolveResult(method="PCE-direct", best_x=best_x, best_cost=float(best_cost), n_evals=steps,
                        wall_s=time.time() - t0, n_qubits=n, n_gates=n_gates,
                        extra=dict(k=k, assignment_mode=assignment_mode,
                                   cost_no_ls=float(cost0), cost_with_ls=float(cost1)))


def _levels_to_x(levels, L, B, m):
    x = np.zeros(m, dtype=int)
    ptr = 0
    for b in range(B):
        x[ptr:ptr + int(levels[b])] = 1
        ptr += L
    return x


def run_simulated_annealing(inst: QUBOInstance, budget_evals: int = 3000, seed: int = 0,
                             T0: float = 2.0, Tmin: float = 0.01) -> SolveResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    L, B, m = inst.L, inst.B, inst.m
    levels = rng.integers(0, L + 1, size=B)
    x = _levels_to_x(levels, L, B, m)
    cur_cost = inst.cost(x)
    best_cost, best_levels = cur_cost, levels.copy()
    evals, step = 0, 0
    while evals < budget_evals:
        step += 1
        T = T0 * (Tmin / T0) ** min(step / budget_evals, 1.0)
        b = rng.integers(0, B)
        delta = rng.choice([-1, 1])
        nl = levels[b] + delta
        if nl < 0 or nl > L:
            continue
        new_levels = levels.copy()
        new_levels[b] = nl
        x_new = _levels_to_x(new_levels, L, B, m)
        new_cost = inst.cost(x_new)
        evals += 1
        d = new_cost - cur_cost
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            levels, cur_cost = new_levels, new_cost
            if cur_cost < best_cost:
                best_cost, best_levels = cur_cost, levels.copy()
    best_x = _levels_to_x(best_levels, L, B, m)
    return SolveResult(method="SimAnneal", best_x=best_x, best_cost=float(best_cost),
                        n_evals=evals, wall_s=time.time() - t0)


def run_tabu(inst: QUBOInstance, budget_evals: int = 3000, seed: int = 0, tenure: int = 10,
             max_candidates_per_step: int = 20) -> SolveResult:
    """max_candidates_per_step: classic tabu evaluates the FULL neighborhood
    (2*B candidates) every step, which is fine at small B but means the
    number of actual search steps affordable under a fixed evaluation budget
    collapses as B grows (observed directly on RNA instances: at B=127,
    budget=3000 allows only ~12 steps total, far too few to escape a
    penalty-dominated random start). Bounding the per-step neighborhood to a
    random sample of bits, the same fix this repo's own decode.py applies for
    B>10 joint local search (_pairwise_boost_search's docstring), keeps the
    number of steps roughly budget/max_candidates_per_step regardless of m,
    so tabu and SA get a comparable number of update steps at matched total
    evaluation budgets."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    L, B, m = inst.L, inst.B, inst.m
    levels = rng.integers(0, L + 1, size=B)
    x = _levels_to_x(levels, L, B, m)
    cur_cost = inst.cost(x)
    best_cost, best_levels = cur_cost, levels.copy()
    tabu = {}
    evals, step = 0, 0
    n_sample = min(B, max_candidates_per_step)
    while evals < budget_evals:
        step += 1
        candidates = []
        sampled_bits = rng.choice(B, size=n_sample, replace=False)
        for b in sampled_bits:
            for delta in (-1, 1):
                nl = levels[b] + delta
                if nl < 0 or nl > L:
                    continue
                new_levels = levels.copy()
                new_levels[b] = nl
                x_new = _levels_to_x(new_levels, L, B, m)
                c = inst.cost(x_new)
                evals += 1
                is_tabu = tabu.get((b, nl), -1) > step
                candidates.append((c, b, nl, is_tabu))
                if evals >= budget_evals:
                    break
            if evals >= budget_evals:
                break
        if not candidates:
            break
        non_tabu = [cand for cand in candidates if not cand[3]]
        pool = non_tabu if non_tabu else candidates
        c, b, nl, _ = min(pool, key=lambda t: t[0])
        tabu[(b, int(levels[b]))] = step + tenure
        levels = levels.copy()
        levels[b] = nl
        cur_cost = c
        if cur_cost < best_cost:
            best_cost, best_levels = cur_cost, levels.copy()
    best_x = _levels_to_x(best_levels, L, B, m)
    return SolveResult(method="Tabu", best_x=best_x, best_cost=float(best_cost),
                        n_evals=evals, wall_s=time.time() - t0)


def run_random_blind(inst: QUBOInstance, n_trials: int = 20, seed: int = 0,
                      large_b_boost: bool = True) -> SolveResult:
    """No circuit at all: a uniformly random correlator vector, decoded and
    locally searched exactly like a real quantum arm. The confound control
    this project's own experiments always report alongside a quantum arm."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    m = inst.m
    best_x, best_cost = None, np.inf
    for _ in range(n_trials):
        pi_blind = rng.choice([-1.0, 1.0], size=m) * rng.uniform(0.3, 0.95, size=m)
        _, x1 = _decode_stages(pi_blind, inst, large_b_boost=large_b_boost)
        c1 = inst.cost(x1)
        if c1 < best_cost:
            best_cost, best_x = c1, x1
    return SolveResult(method="Blind", best_x=best_x, best_cost=float(best_cost),
                        n_evals=n_trials, wall_s=time.time() - t0)


def run_exact(inst: QUBOInstance, m_bruteforce_cap: int = 20, mip_time_limit_s: float = 30.0) -> SolveResult:
    t0 = time.time()
    if inst.m <= m_bruteforce_cap:
        top_x, top_cost, n_feasible, n_enum = vectorized_oracle(inst, top_k=1)
        best_x, best_cost = top_x[0], float(top_cost[0])
        method = "BruteForce"
    else:
        best_x, best_cost, dt, status = solve_mip(inst, time_limit_s=mip_time_limit_s)
        method = f"MIP({status})"
        if best_x is None:
            return SolveResult(method=method, best_x=None, best_cost=np.inf, n_evals=0,
                                wall_s=time.time() - t0)
    return SolveResult(method=method, best_x=best_x, best_cost=float(best_cost),
                        n_evals=2 ** inst.m if inst.m <= m_bruteforce_cap else -1,
                        wall_s=time.time() - t0)
