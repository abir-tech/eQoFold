"""Tests for the local-search post-processing step."""

from __future__ import annotations

import itertools
import random

import pytest

from rnaqopt.model.base import PolynomialModel
from rnaqopt.postprocess import flip_delta, improve_samples, local_search


def brute_force_min(model: PolynomialModel) -> float:
    best = float("inf")
    for combo in itertools.product((0, 1), repeat=model.n_vars):
        best = min(best, model.energy(combo))
    return best


def make_model(n: int, seed: int) -> PolynomialModel:
    rng = random.Random(seed)
    m = PolynomialModel(n_vars=n)
    for i in range(n):
        m.add((i,), rng.uniform(-3, 3))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                m.add((i, j), rng.uniform(-2, 2))
    for _ in range(n):
        a, b, c = rng.sample(range(n), 3)
        m.add((a, b, c), rng.uniform(-1, 1))
    return m


# --------------------------------------------------------------------------
# Delta evaluation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_flip_delta_matches_full_reevaluation(seed):
    """The incremental delta must equal the true energy change exactly, or the
    search silently optimises the wrong function."""
    model = make_model(7, seed)
    index: list[list] = [[] for _ in range(model.n_vars)]
    for key, coeff in model.terms.items():
        for v in key:
            index[v].append((key, coeff))

    rng = random.Random(seed)
    for _ in range(20):
        bits = [rng.randint(0, 1) for _ in range(model.n_vars)]
        for v in range(model.n_vars):
            before = model.energy(bits)
            flipped = list(bits)
            flipped[v] ^= 1
            expected = model.energy(flipped) - before
            assert flip_delta(bits, v, index[v]) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_local_search_never_worsens(seed):
    model = make_model(8, seed)
    rng = random.Random(seed + 100)
    bits = [rng.randint(0, 1) for _ in range(model.n_vars)]
    result = local_search(model, bits)
    assert result.energy_after <= result.energy_before + 1e-12
    assert result.improvement >= -1e-12


@pytest.mark.parametrize("seed", range(8))
def test_result_is_a_1opt_local_minimum(seed):
    """No single flip may improve the returned assignment."""
    model = make_model(8, seed)
    rng = random.Random(seed + 200)
    bits = [rng.randint(0, 1) for _ in range(model.n_vars)]
    out = list(local_search(model, bits).bitstring)
    base = model.energy(out)
    for v in range(model.n_vars):
        flipped = list(out)
        flipped[v] ^= 1
        assert model.energy(flipped) >= base - 1e-9


def test_reported_energies_are_consistent_with_the_bitstring():
    model = make_model(9, 3)
    result = local_search(model, [0] * 9)
    assert model.energy(result.bitstring) == pytest.approx(result.energy_after)


def test_already_optimal_input_is_left_alone():
    model = PolynomialModel(n_vars=3)
    model.add((0,), -1.0)
    model.add((1,), -1.0)
    model.add((2,), -1.0)
    result = local_search(model, [1, 1, 1])
    assert result.bitstring == (1, 1, 1)
    assert result.n_flips == 0
    assert not result.improved
    assert result.improvement == pytest.approx(0.0)


def test_search_finds_the_obvious_improvement():
    model = PolynomialModel(n_vars=3)
    model.add((0,), -5.0)
    model.add((1,), 2.0)
    result = local_search(model, [0, 1, 0])
    assert result.bitstring[0] == 1  # turned on the stabilising variable
    assert result.bitstring[1] == 0  # turned off the destabilising one
    assert result.improved


def test_deterministic():
    model = make_model(8, 11)
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    a = local_search(model, bits)
    b = local_search(model, bits)
    assert a == b


def test_rejects_wrong_length_bitstring():
    model = make_model(5, 1)
    with pytest.raises(ValueError):
        local_search(model, [0, 1])


# --------------------------------------------------------------------------
# Multi-sample
# --------------------------------------------------------------------------


def test_improve_samples_measures_against_best_raw_sample():
    """The reported improvement must be relative to what the solver actually
    returned, not to whichever sample happened to polish up best."""
    model = make_model(8, 5)
    rng = random.Random(7)
    samples = [[rng.randint(0, 1) for _ in range(8)] for _ in range(6)]
    result = improve_samples(model, samples)

    assert result.energy_before == pytest.approx(
        min(model.energy(s) for s in samples)
    )
    assert result.energy_after <= result.energy_before + 1e-12


def test_improve_samples_rejects_empty():
    with pytest.raises(ValueError):
        improve_samples(make_model(4, 1), [])


@pytest.mark.parametrize("seed", range(5))
def test_local_search_is_a_heuristic_not_an_oracle(seed):
    """Documents the honest limit: 1-opt reaches a local, not global, optimum.

    If this ever started passing as equality for every seed, the search would be
    doing more than advertised and the 'reported separately' discipline would
    need revisiting.
    """
    model = make_model(9, seed)
    result = local_search(model, [0] * 9)
    assert result.energy_after >= brute_force_min(model) - 1e-9
