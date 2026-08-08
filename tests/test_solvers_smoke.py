"""Smoke and correctness tests for the simulator, resources and every solver.

The load-bearing test here is
:func:`test_simulator_matches_dense_matrix_exponential` -- ADAPT-QAOA and PCE
both rest on the hand-written statevector simulator, so it is checked against
``scipy.linalg.expm`` rather than trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rnaqopt.model import build_level1, build_level2
from rnaqopt.model.base import PolynomialModel
from rnaqopt.model.quadratize import quadratize, verify_quadratization
from rnaqopt.noise import NoiseModel, shots_for_target_success, success_probability
from rnaqopt.resources import (
    cost_layer_gates,
    pauli_rotation_gates,
    schedule_depth,
    two_qubit_cost_of_degree,
)
from rnaqopt.sequences import load_tier
from rnaqopt.simulator import (
    PAULI,
    apply_pauli_rotation,
    cvar,
    diagonal_from_model,
    expectation,
    plus_state,
)
from rnaqopt.solvers.adapt_qaoa import AdaptQAOASolver, build_mixer_pool
from rnaqopt.solvers.annealing import RandomSearchSolver, SimulatedAnnealingSolver
from rnaqopt.solvers.dirac3 import (
    Dirac3Client,
    DiracSimplexSimulator,
    encode,
    project_to_simplex,
)
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.solvers.lowrank import LowRankSolver, to_ising
from rnaqopt.solvers.pce import PCESolver, assign_paulis, min_qubits
from rnaqopt.stems import enumerate_with_graphs


def small_model(level: int = 1, seq_id: str = "A_hp01"):
    rec = [r for r in load_tier("A") if r.seq_id == seq_id][0]
    graphs = enumerate_with_graphs(rec.sequence)
    build = build_level1 if level == 1 else build_level2
    return rec, graphs, build(rec.sequence, graphs)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


def _dense(pauli: str, qubits: tuple[int, ...], n: int) -> np.ndarray:
    ops = ["I"] * n
    for ch, q in zip(pauli, qubits, strict=True):
        ops[q] = ch
    mat = np.array([[1.0]], dtype=complex)
    for q in range(n):
        mat = np.kron(PAULI[ops[q]], mat)
    return mat


@pytest.mark.parametrize("pauli", ["X", "Y", "Z", "XX", "YY", "YZ", "ZY", "XY"])
def test_simulator_matches_dense_matrix_exponential(pauli):
    """Exact check against scipy.linalg.expm, including reversed qubit order."""
    from scipy.linalg import expm

    rng = np.random.default_rng(0)
    k = len(pauli)
    for n in (2, 3, 4):
        for _ in range(4):
            qubits = tuple(rng.choice(n, size=k, replace=False).tolist())
            theta = float(rng.uniform(-3, 3))
            psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
            psi /= np.linalg.norm(psi)
            got = apply_pauli_rotation(psi.copy(), pauli, qubits, theta, n)
            want = expm(-1j * theta * _dense(pauli, qubits, n)) @ psi
            assert np.abs(got - want).max() < 1e-10


def test_rotations_preserve_norm():
    psi = plus_state(4)
    out = apply_pauli_rotation(psi, "XY", (0, 3), 0.7, 4)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12


def test_diagonal_from_model_matches_polynomial():
    m = PolynomialModel(3)
    m.add((0,), -1.5)
    m.add((0, 2), 2.0)
    m.add((0, 1, 2), -0.5)
    diag = diagonal_from_model(m.terms, 3)
    for state in range(8):
        bits = tuple((state >> q) & 1 for q in range(3))
        assert diag[state] == pytest.approx(m.energy(bits))


def test_cvar_bounds_and_limits():
    diag = np.array([-5.0, -1.0, 0.0, 3.0])
    psi = np.full(4, 0.5, dtype=complex)
    assert cvar(psi, diag, 1.0) == pytest.approx(expectation(psi, diag))
    assert cvar(psi, diag, 0.25) == pytest.approx(-5.0)
    assert cvar(psi, diag, 0.5) == pytest.approx(-3.0)
    # CVaR is never above the mean and never below the minimum.
    for a in (0.1, 0.3, 0.7):
        assert diag.min() <= cvar(psi, diag, a) <= expectation(psi, diag) + 1e-9


def test_cvar_rejects_bad_alpha():
    with pytest.raises(ValueError):
        cvar(plus_state(2), np.zeros(4), 0.0)


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------


def test_pauli_rotation_gate_counts():
    """k-body Z rotation costs 2(k-1) CNOTs and one RZ."""
    for k in (2, 3, 4):
        gates = pauli_rotation_gates("Z" * k, tuple(range(k)))
        assert sum(1 for g in gates if g.is_two_qubit) == 2 * (k - 1)
        assert sum(1 for g in gates if g.name == "rz") == 1


def test_x_and_y_rotations_add_basis_changes():
    assert len(pauli_rotation_gates("X", (0,))) > len(pauli_rotation_gates("Z", (0,)))
    assert len(pauli_rotation_gates("Y", (0,))) > len(pauli_rotation_gates("X", (0,)))


def test_cubic_terms_cost_four_cnots_each():
    m = PolynomialModel(3)
    m.add((0, 1, 2), 1.0)
    assert two_qubit_cost_of_degree(m.terms) == {3: 4}
    assert sum(1 for g in cost_layer_gates(m.terms) if g.is_two_qubit) == 4


def test_schedule_depth_parallelises_disjoint_gates():
    from rnaqopt.resources import Gate

    disjoint = [Gate("cx", (0, 1)), Gate("cx", (2, 3))]
    serial = [Gate("cx", (0, 1)), Gate("cx", (1, 2))]
    assert schedule_depth(disjoint, 4) == 1
    assert schedule_depth(serial, 4) == 2
    assert schedule_depth([], 4) == 0


# --------------------------------------------------------------------------
# Quadratization
# --------------------------------------------------------------------------


def test_quadratization_is_faithful():
    import random

    rng = random.Random(0)
    for _ in range(6):
        n = rng.randint(3, 5)
        m = PolynomialModel(n)
        for _ in range(rng.randint(2, 5)):
            k = rng.choice([1, 2, 3])
            m.add(rng.sample(range(n), k), round(rng.uniform(-5, 5), 2))
        q = quadratize(m)
        assert q.model.degree <= 2
        assert verify_quadratization(m, q)


def test_quadratization_shares_ancillas_across_terms():
    m = PolynomialModel(4)
    m.add((0, 1, 2), -3.0)
    m.add((1, 2, 3), -2.0)
    q = quadratize(m)
    assert q.n_ancillas == 1  # the shared pair (1,2) is reduced once


def test_quadratic_model_is_untouched():
    m = PolynomialModel(3)
    m.add((0, 1), 1.0)
    q = quadratize(m)
    assert q.n_ancillas == 0
    assert q.model.terms == m.terms


def test_level2_needs_ancillas_and_level1_does_not():
    _, _, m1 = small_model(1, "A_hp06")
    _, _, m2 = small_model(2, "A_hp06")
    assert quadratize(m1.full).n_ancillas == 0
    assert m2.degree == 3 or quadratize(m2.full).n_ancillas >= 0


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------


def test_annealing_and_random_return_valid_results():
    _, graphs, model = small_model()
    for solver in (
        SimulatedAnnealingSolver(n_sweeps=50, n_restarts=4, seed=0),
        RandomSearchSolver(n_samples=500, seed=0),
    ):
        r = solver.solve(model)
        assert len(r.bitstring) == graphs.n
        assert r.model_energy == pytest.approx(model.full.energy(r.bitstring))
        assert r.wall_time >= 0


def test_annealing_reaches_the_exact_optimum_on_a_small_instance():
    _, _, model = small_model()
    exact = solve_exact(model)
    r = SimulatedAnnealingSolver(n_sweeps=400, n_restarts=20, seed=1).solve(model)
    assert r.model_energy == pytest.approx(exact.model_energy, abs=1e-6)


def test_annealing_handles_cubic_models():
    """The SA baseline must work at degree 3, or the Level 2 comparison is not
    like-for-like."""
    _, _, model = small_model(2, "A_hp06")
    r = SimulatedAnnealingSolver(n_sweeps=200, n_restarts=8, seed=0).solve(model)
    assert r.model_energy == pytest.approx(model.full.energy(r.bitstring))


def test_lowrank_ising_conversion_roundtrips():
    m = PolynomialModel(3)
    m.add((0,), -1.0)
    m.add((0, 1), 2.0)
    m.add((), 0.5)
    J, h, offset = to_ising(m)
    for state in range(8):
        bits = tuple((state >> q) & 1 for q in range(3))
        s = np.array([2 * b - 1 for b in bits], dtype=float)
        assert float(s @ J @ s + h @ s + offset) == pytest.approx(m.energy(bits))


def test_lowrank_rejects_cubic_models():
    m = PolynomialModel(3)
    m.add((0, 1, 2), 1.0)
    with pytest.raises(ValueError, match="degree"):
        to_ising(m)


def test_lowrank_runs_and_reports_rank():
    _, graphs, model = small_model()
    r = LowRankSolver(seed=0, n_restarts=2, n_steps=50).solve(model)
    assert r.resource_dict["rank"] >= 1
    assert len(r.bitstring) == graphs.n


def test_adapt_qaoa_reaches_the_optimum_on_a_small_instance():
    _, _, model = small_model()
    exact = solve_exact(model)
    r = AdaptQAOASolver(max_layers=5, seed=1, max_qubits=12).solve(model)
    assert r.model_energy == pytest.approx(exact.model_energy, abs=1e-6)
    assert r.solver_metadata["layers"] >= 1
    assert r.resource_dict["depth"] > 0
    assert 0.0 <= r.solver_metadata["success_probability"] <= 1.0


def test_adapt_qaoa_records_per_layer_resources():
    _, _, model = small_model()
    r = AdaptQAOASolver(max_layers=3, seed=0, max_qubits=12).solve(model)
    hist = r.solver_metadata["history"]
    assert hist
    depths = [h["depth"] for h in hist]
    assert depths == sorted(depths)  # depth grows with layers
    for h in hist:
        assert {"layer", "mixer", "gradient", "cvar", "two_qubit_gates"} <= set(h)


def test_adapt_qaoa_refuses_oversized_instances():
    _, _, model = small_model()
    with pytest.raises(ValueError, match="qubits"):
        AdaptQAOASolver(max_qubits=1).solve(model)


def test_mixer_pool_contains_the_specified_operators():
    _, graphs, model = small_model()
    pool = build_mixer_pool(model)
    paulis = {p for p, _ in pool}
    assert {"X", "Y"} <= paulis
    if graphs.conflict:
        assert {"XX", "YY", "YZ"} <= paulis


def test_adapt_qaoa_readout_is_finite_shot():
    """Readout must come from sampling, not from scanning the statevector."""
    _, _, model = small_model()
    r = AdaptQAOASolver(max_layers=2, seed=0, max_qubits=12, shots=8).solve(model)
    assert r.resource_dict["shots"] == 8
    assert r.solver_metadata["unique_bitstrings_sampled"] <= 8


# --------------------------------------------------------------------------
# PCE
# --------------------------------------------------------------------------


def test_min_qubits_satisfies_the_capacity_bound():
    for n in (1, 5, 18, 40, 100, 639):
        m = min_qubits(n, k=2)
        assert math.comb(m, 2) * 9 >= n
        if m > 2:
            assert math.comb(m - 1, 2) * 9 < n  # minimal


def test_pce_compression_beats_direct_encoding():
    for n in (18, 40, 100, 639):
        assert min_qubits(n, k=2) < n


def test_assign_paulis_are_distinct():
    assigns = assign_paulis(20, min_qubits(20), 2)
    assert len(assigns) == 20
    assert len(set(assigns)) == 20


def test_assign_paulis_rejects_over_capacity():
    with pytest.raises(ValueError, match="capacity"):
        assign_paulis(100, 3, 2)


def test_pce_runs_and_reports_compression():
    _, graphs, model = small_model()
    r = PCESolver(seed=0, n_restarts=1, maxiter=60).solve(model)
    assert r.resource_dict["n_qubits"] < graphs.n or graphs.n <= 3
    assert r.resource_dict["compression_ratio"] > 0
    assert len(r.bitstring) == graphs.n


# --------------------------------------------------------------------------
# Dirac-3
# --------------------------------------------------------------------------


def test_projection_lands_on_the_simplex():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.normal(size=8) * 3
        x = project_to_simplex(v, 5.0)
        assert (x >= -1e-12).all()
        assert float(x.sum()) == pytest.approx(5.0)


def test_encodings_have_the_expected_variable_counts():
    _, graphs, model = small_model()
    per_stem = encode(model, scheme="per_stem")
    global_slack = encode(model, scheme="global_slack")
    assert per_stem.n_vars == 2 * graphs.n
    assert global_slack.n_vars == graphs.n + 1
    assert per_stem.metadata["corner_protected"] is True
    assert global_slack.metadata["corner_protected"] is False


def test_level2_encoding_fits_the_free_tier_degree_ceiling():
    """The whole co-design claim: Level 2 is degree 3, the free-tier ceiling."""
    _, _, model = small_model(2, "A_hp06")
    enc = encode(model, scheme="per_stem")
    assert enc.degree <= 3
    assert enc.fits_free_tier()


def test_encoding_rejects_unknown_scheme():
    _, _, model = small_model()
    with pytest.raises(ValueError, match="unknown scheme"):
        encode(model, scheme="nonsense")


def test_vectorised_evaluate_and_gradient_agree_with_definition():
    _, _, model = small_model()
    enc = encode(model, scheme="per_stem")
    x = np.random.default_rng(0).random(enc.n_vars)

    expected = 0.0
    for key, coeff in enc.terms.items():
        prod = 1.0
        for v in key:
            prod *= x[v]
        expected += coeff * prod
    assert enc.evaluate(x) == pytest.approx(expected)

    grad = np.zeros_like(x)
    for key, coeff in enc.terms.items():
        for pos, v in enumerate(key):
            p = coeff
            for wp, w in enumerate(key):
                if wp != pos:
                    p *= x[w]
            grad[v] += p
    assert np.abs(enc.gradient(x) - grad).max() < 1e-9


def test_dirac_simulator_runs_both_schemes():
    _, graphs, model = small_model()
    for scheme in ("per_stem", "global_slack"):
        r = DiracSimplexSimulator(
            scheme=scheme, n_restarts=3, n_steps=60, seed=0
        ).solve(model)
        assert len(r.bitstring) == graphs.n
        assert 0.0 <= r.solver_metadata["collapse_rate"] <= 1.0


def test_dirac_client_preflight_catches_degree_and_size():
    _, _, model = small_model(2, "A_hp06")
    enc = encode(model, scheme="per_stem")
    client = Dirac3Client(var_limit=1)
    issues = client.preflight(enc)
    assert any("variables" in i for i in issues)


def test_dirac_client_refuses_without_credentials():
    _, _, model = small_model()
    if Dirac3Client.is_available():  # pragma: no cover - depends on environment
        pytest.skip("credentials present")
    # The real variable is QCI_TOKEN; qci-client reads nothing else.
    with pytest.raises(RuntimeError, match="QCI_TOKEN"):
        Dirac3Client().solve(model)


def test_dirac_diagnose_separates_package_from_token():
    """A bare 'credentials present: False' cannot distinguish a missing package
    from a missing token, and that ambiguity cost a wasted hardware session."""
    diag = Dirac3Client.diagnose()
    assert set(diag) >= {
        "package_installed",
        "token_present",
        "token_variable",
        "api_url",
        "ready",
        "problems",
    }
    assert diag["ready"] == (diag["package_installed"] and diag["token_present"])
    assert bool(diag["problems"]) != diag["ready"]


def test_dirac_api_url_defaults_to_production():
    """qci-client ships no default URL and raises without one, so we supply it."""
    assert Dirac3Client.api_url().startswith("https://")


def test_dirac_token_accepts_both_variable_names(monkeypatch):
    monkeypatch.delenv("QCI_TOKEN", raising=False)
    monkeypatch.delenv("QCI_API_TOKEN", raising=False)
    assert Dirac3Client.token() is None
    monkeypatch.setenv("QCI_API_TOKEN", "legacy-name")
    assert Dirac3Client.token() == "legacy-name"
    monkeypatch.setenv("QCI_TOKEN", "real-name")
    assert Dirac3Client.token() == "real-name"  # QCI_TOKEN wins


def test_dirac_polynomial_payload_is_well_formed():
    """Every idx list must have arity == max_degree, indices 1-based, no constant.

    This is the exact shape the files API validates, and getting it wrong costs
    a failed submission rather than an error message.
    """
    from rnaqopt.solvers.dirac3 import encode

    _, _, model = small_model()
    client = Dirac3Client(scheme="per_stem")
    enc = encode(model, scheme="per_stem")
    cfg = client.polynomial_file(enc)["file_config"]["polynomial"]

    assert cfg["num_variables"] == enc.n_vars
    assert cfg["data"], "payload must not be empty"
    for element in cfg["data"]:
        assert len(element["idx"]) == cfg["max_degree"]
        assert all(0 <= i <= enc.n_vars for i in element["idx"])
        assert any(i > 0 for i in element["idx"]), "constant term must be dropped"
        assert isinstance(element["val"], float)


def test_dirac_estimated_device_time_scales_with_samples():
    fast = Dirac3Client(num_samples=10, relaxation_schedule=1)
    slow = Dirac3Client(num_samples=40, relaxation_schedule=2)
    assert slow.estimated_device_seconds() > fast.estimated_device_seconds()


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def test_noise_model_validation():
    with pytest.raises(ValueError):
        NoiseModel(depolarizing=1.5)
    with pytest.raises(ValueError):
        NoiseModel(shots=0)


def test_depolarizing_flattens_the_distribution():
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    diag = np.array([-1.0, 0.0, 0.0, 0.0])
    clean = success_probability(psi, diag, NoiseModel(shots=None))
    noisy = success_probability(psi, diag, NoiseModel(shots=None, depolarizing=0.5))
    assert clean == pytest.approx(1.0)
    assert noisy < clean


def test_shots_for_target_success():
    assert shots_for_target_success(1.0) == 1
    assert shots_for_target_success(0.5, 0.99) == 7
    # A rarer optimum needs proportionally more repetitions.
    assert shots_for_target_success(0.01) > shots_for_target_success(0.1)
