"""Tests for the polynomial model type and the fidelity ladder.

The central claim under test is that **Level 1 reproduces the true Turner
energy exactly** for the structure class it is designed to represent (helices
nested at most one deep, no multiloops), and that where it is inexact, it is
inexact for the two documented three-body reasons and no others.
"""

from __future__ import annotations

import pytest

from rnaqopt.decode import structure_to_selection
from rnaqopt.energy import LoopEnergies
from rnaqopt.model import build_level0, build_level1, build_model
from rnaqopt.model.base import PolynomialModel
from rnaqopt.model.level1 import structure_is_level1_exact
from rnaqopt.model.penalties import (
    PenaltySweepPoint,
    default_penalty,
    knee,
    max_single_variable_gain,
    sweep_range,
)
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.exact import BruteForceSolver, CPSATSolver, solve_exact
from rnaqopt.stems import enumerate_with_graphs

# --------------------------------------------------------------------------
# PolynomialModel
# --------------------------------------------------------------------------


def test_add_collapses_repeated_indices():
    """Binary variables: x*x = x, so (1,1) is the linear term (1,)."""
    m = PolynomialModel(3)
    m.add((1, 1), 2.0)
    assert m.terms == {(1,): 2.0}


def test_add_sorts_keys_and_accumulates():
    m = PolynomialModel(3)
    m.add((2, 0), 1.0)
    m.add((0, 2), 0.5)
    assert m.terms == {(0, 2): 1.5}


def test_cancelling_terms_are_removed():
    m = PolynomialModel(2)
    m.add((0,), 1.0)
    m.add((0,), -1.0)
    assert m.terms == {}
    assert m.degree == 0


def test_zero_coefficient_is_not_stored():
    m = PolynomialModel(2)
    m.add((0,), 0.0)
    assert m.n_terms == 0


def test_out_of_range_variable_rejected():
    m = PolynomialModel(2)
    with pytest.raises(IndexError):
        m.add((5,), 1.0)


def test_degree_and_counts():
    m = PolynomialModel(4)
    m.add((), 1.0)
    m.add((0,), 1.0)
    m.add((0, 1), 1.0)
    m.add((0, 1, 2), 1.0)
    assert m.degree == 3
    assert m.term_counts() == {0: 1, 1: 1, 2: 1, 3: 1}
    assert m.constant == 1.0


def test_energy_matches_energy_of_selection():
    m = PolynomialModel(3)
    m.add((0,), -2.0)
    m.add((1,), -1.0)
    m.add((0, 1), 5.0)
    m.add((0, 1, 2), -0.5)
    for bits, sel in [((1, 1, 0), [0, 1]), ((1, 0, 1), [0, 2]), ((1, 1, 1), [0, 1, 2])]:
        assert m.energy(bits) == pytest.approx(m.energy_of_selection(sel))


def test_integer_conversion_is_exact_for_turner_energies():
    m = PolynomialModel(2)
    m.add((0,), -3.30)
    m.add((0, 1), 4.50)
    assert m.to_integer_terms(100) == {(0,): -330, (0, 1): 450}


def test_integer_conversion_refuses_to_round_silently():
    m = PolynomialModel(1)
    m.add((0,), 0.123456)
    with pytest.raises(ValueError):
        m.to_integer_terms(100)


def test_merged_sums_coefficients():
    a, b = PolynomialModel(2), PolynomialModel(2)
    a.add((0,), 1.0)
    b.add((0,), 2.0)
    b.add((1,), 3.0)
    assert a.merged(b).terms == {(0,): 3.0, (1,): 3.0}


# --------------------------------------------------------------------------
# Level 0
# --------------------------------------------------------------------------


def test_level0_is_linear_plus_quadratic_penalties():
    rec = load_tier("A")[0]
    g = enumerate_with_graphs(rec.sequence)
    m = build_level0(rec.sequence, g)
    assert m.objective.degree <= 1
    assert m.level == 0
    assert m.degree <= 2  # penalties are quadratic


def test_level0_linear_coefficients_are_pure_stacking():
    rec = load_tier("A")[0]
    g = enumerate_with_graphs(rec.sequence)
    le = LoopEnergies(rec.sequence)
    m = build_level0(rec.sequence, g)
    for idx, stem in enumerate(g.stems):
        assert m.objective.terms.get((idx,), 0.0) == pytest.approx(le.stack(stem))


def test_level0_omits_loop_penalties_and_so_overbinds():
    """Level 0 charges nothing for the hairpin, so it reports a lower (more
    negative) energy than the truth for any real structure."""
    rec = load_tier("A")[0]
    g = enumerate_with_graphs(rec.sequence)
    m = build_level0(rec.sequence, g)
    ref, _ = mfe(rec.sequence)
    sel = structure_to_selection(g, ref)
    assert sel is not None
    assert m.objective.energy_of_selection(sel) < eval_structure(rec.sequence, ref)


# --------------------------------------------------------------------------
# Level 1 -- the exactness claim
# --------------------------------------------------------------------------


def test_level1_is_exact_for_a_lone_hairpin():
    seq = "GGGGAAAACCCC"
    g = enumerate_with_graphs(seq)
    m = build_level1(seq, g)
    db, energy = mfe(seq)
    sel = structure_to_selection(g, db)
    assert sel is not None
    assert m.objective.energy_of_selection(sel) == pytest.approx(energy, abs=1e-9)


def test_level1_is_exact_for_one_nested_helix():
    """s > t: the hairpin of s is replaced by the interior loop, and t leaves
    the exterior loop. Both corrections live in the single quadratic term."""
    seq = "GGGGAAAGGAAAACCAAACCCC"
    db = "((((...((....))...))))"
    g = enumerate_with_graphs(seq, min_hairpin=3)
    m = build_level1(seq, g)
    sel = structure_to_selection(g, db)
    if sel is None:
        pytest.skip("structure not representable with default enumeration")
    assert m.objective.energy_of_selection(sel) == pytest.approx(
        eval_structure(seq, db), abs=1e-9
    )


def test_level1_exact_on_every_representable_tier_a_reference():
    """The strong form: on Tier A, Level 1's energy function has zero error on
    every reference structure it can represent."""
    checked = 0
    for rec in load_tier("A"):
        g = enumerate_with_graphs(rec.sequence)
        db, energy = mfe(rec.sequence)
        sel = structure_to_selection(g, db)
        if sel is None or not structure_is_level1_exact(db):
            continue
        m = build_level1(rec.sequence, g)
        assert m.objective.energy_of_selection(sel) == pytest.approx(
            energy, abs=1e-9
        ), rec.seq_id
        checked += 1
    assert checked >= 10, "expected most of Tier A to be representable and exact"


def test_level1_beats_level0_on_energy_fidelity():
    """The ladder must actually climb: Level 1's energy error is strictly
    smaller than Level 0's, averaged over Tier A."""
    l0_err, l1_err, n = 0.0, 0.0, 0
    for rec in load_tier("A"):
        g = enumerate_with_graphs(rec.sequence)
        db, energy = mfe(rec.sequence)
        sel = structure_to_selection(g, db)
        if sel is None:
            continue
        m0 = build_level0(rec.sequence, g)
        m1 = build_level1(rec.sequence, g)
        l0_err += abs(m0.objective.energy_of_selection(sel) - energy)
        l1_err += abs(m1.objective.energy_of_selection(sel) - energy)
        n += 1
    assert n > 0
    assert l1_err < l0_err
    assert l1_err / n < 1e-9  # Level 1 is not merely better, it is exact


def test_level1_is_inexact_for_multiloops():
    """Documented limitation: a multiloop needs three-body terms, so Level 1
    cannot represent it. This test pins the limitation so that Level 2 has a
    measurable thing to fix."""
    rec = load_tier("M")[0]
    g = enumerate_with_graphs(rec.sequence)
    db, energy = mfe(rec.sequence)
    sel = structure_to_selection(g, db)
    if sel is None:
        pytest.skip("reference not representable")
    assert not structure_is_level1_exact(db)
    m = build_level1(rec.sequence, g)
    assert m.objective.energy_of_selection(sel) != pytest.approx(energy, abs=1e-6)


def test_structure_is_level1_exact_classifier():
    assert structure_is_level1_exact("(((...)))")
    assert structure_is_level1_exact("((..))..((..))")  # two exterior helices
    assert structure_is_level1_exact("(((..(((...)))..)))")  # one level of nesting
    assert not structure_is_level1_exact("((((..))..((..))))")  # multiloop


# --------------------------------------------------------------------------
# Penalties
# --------------------------------------------------------------------------


def test_max_single_variable_gain_sums_negative_terms():
    m = PolynomialModel(2)
    m.add((0,), -2.0)
    m.add((0, 1), -3.0)
    m.add((1,), 1.0)
    assert max_single_variable_gain(m) == pytest.approx(5.0)


def test_default_penalty_exceeds_the_bound():
    m = PolynomialModel(2)
    m.add((0,), -4.0)
    assert default_penalty(m, safety=1.5) == pytest.approx(6.0)


def test_penalty_makes_every_violation_unprofitable():
    """The point of the bound: the exact optimum of the penalised model must be
    feasible."""
    for rec in load_tier("A")[:8]:
        g = enumerate_with_graphs(rec.sequence)
        m = build_level1(rec.sequence, g)
        result = solve_exact(m)
        assert m.is_feasible(result.selection), rec.seq_id


def test_sweep_range_brackets_the_bound():
    m = PolynomialModel(2)
    m.add((0,), -4.0)
    pts = sweep_range(m, n_points=10)
    assert len(pts) == 10
    assert pts[0] < 4.0 < pts[-1]
    assert pts == sorted(pts)


def test_knee_picks_the_smallest_feasible_lambda():
    pts = [
        PenaltySweepPoint(1.0, 0.2, 3.0, 10),
        PenaltySweepPoint(5.0, 1.0, 0.1, 10),
        PenaltySweepPoint(9.0, 1.0, 0.1, 10),
    ]
    assert knee(pts) == 5.0


# --------------------------------------------------------------------------
# Exact solvers -- cross-validation
# --------------------------------------------------------------------------


def test_brute_force_and_cpsat_agree_on_tier_a():
    """Two independent exact implementations. The optimum is the denominator of
    every optimizer-gap number, so it is worth checking twice."""
    bf, cp = BruteForceSolver(limit=20), CPSATSolver(max_seconds=30)
    checked = 0
    for rec in load_tier("A"):
        g = enumerate_with_graphs(rec.sequence)
        if g.n > 20:
            continue
        m = build_level1(rec.sequence, g)
        a, b = bf.solve(m), cp.solve(m)
        assert a.model_energy == pytest.approx(b.model_energy, abs=1e-6), rec.seq_id
        assert b.is_proven_optimal
        checked += 1
    assert checked >= 8


def test_brute_force_refuses_oversized_instances():
    m = build_level1(
        load_tier("C")[-1].sequence,
        enumerate_with_graphs(load_tier("C")[-1].sequence),
    )
    with pytest.raises(ValueError, match="brute force refused"):
        BruteForceSolver(limit=10).solve(m)


def test_cpsat_hard_constraints_match_penalised_optimum():
    """Penalty enforcement and hard-constraint enforcement must reach the same
    optimum when the penalty is above its bound.

    Compared on decoded *structure* and objective value, not on the raw stem
    selection: with sub-stems enabled the encoding is redundant, since a long
    helix can be tiled either by one stem or by several stacked sub-stems. Both
    representations are legitimate and Level 1 assigns them the same energy --
    which is itself a useful consistency check on the nesting correction.
    """
    from rnaqopt.decode import selection_to_structure

    for rec in load_tier("A")[:8]:
        g = enumerate_with_graphs(rec.sequence)
        m = build_level1(rec.sequence, g)
        soft = CPSATSolver(max_seconds=30).solve(m, use_penalties=True)
        hard = CPSATSolver(max_seconds=30).solve(m, use_penalties=False)
        assert m.is_feasible(soft.selection), rec.seq_id
        assert m.objective.energy_of_selection(
            soft.selection
        ) == pytest.approx(
            m.objective.energy_of_selection(hard.selection), abs=1e-6
        ), rec.seq_id
        n = len(rec.sequence)
        assert selection_to_structure(g, soft.selection, n) == selection_to_structure(
            g, hard.selection, n
        ), rec.seq_id


def test_substem_tiling_is_energy_consistent():
    """A helix built from one stem and the same helix built from stacked
    sub-stems must receive identical Level 1 energy.

    This is the telescoping property of the nesting correction: for two stacked
    sub-stems the interior-loop term degenerates to the stack that joins them.
    """
    from rnaqopt.decode import selection_to_structure

    seq = "GGGGGGAAAACCCCCC"
    g = enumerate_with_graphs(seq)
    m = build_level1(seq, g)
    by_structure: dict[str, list[float]] = {}
    for idx in range(g.n):
        for jdx in range(idx, g.n):
            sel = (idx,) if idx == jdx else (idx, jdx)
            if not m.is_feasible(sel):
                continue
            db = selection_to_structure(g, sel, len(seq))
            by_structure.setdefault(db, []).append(
                m.objective.energy_of_selection(sel)
            )
    shared = [v for v in by_structure.values() if len(v) > 1]
    assert shared, "expected at least one structure with two stem representations"
    for energies in shared:
        assert max(energies) - min(energies) < 1e-9


def test_solver_result_reports_resources_and_optimality():
    rec = load_tier("A")[0]
    g = enumerate_with_graphs(rec.sequence)
    r = BruteForceSolver().solve(build_level1(rec.sequence, g))
    assert r.is_proven_optimal
    assert r.resource_dict["n_vars"] == g.n
    assert r.wall_time >= 0.0
    assert len(r.bitstring) == g.n


def test_build_model_dispatch():
    rec = load_tier("A")[0]
    g = enumerate_with_graphs(rec.sequence)
    assert build_model(0, rec.sequence, g).level == 0
    assert build_model(1, rec.sequence, g).level == 1
    with pytest.raises(ValueError):
        build_model(99, rec.sequence, g)
