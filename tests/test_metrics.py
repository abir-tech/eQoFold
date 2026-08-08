"""Tests for structural metrics and the two-gap decomposition."""

from __future__ import annotations

import pytest

from rnaqopt.metrics import (
    DotBracketError,
    GapDecomposition,
    compare_structures,
    dotbracket_from_pairs,
    has_crossings,
    is_valid_dotbracket,
    matches_any,
    pairs_from_dotbracket,
    summarize,
)

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_pairs_hand_computed():
    assert pairs_from_dotbracket("(((...)))") == {(0, 8), (1, 7), (2, 6)}


def test_pairs_empty_structure():
    assert pairs_from_dotbracket(".....") == set()


def test_pairs_two_hairpins():
    assert pairs_from_dotbracket("((..))((..))") == {(0, 5), (1, 4), (6, 11), (7, 10)}


def test_pairs_pseudoknot_extended_alphabet():
    # Classic H-type pseudoknot: the [ ] pairs cross the ( ) pairs.
    db = "((([[[)))]]]"
    pairs = pairs_from_dotbracket(db)
    assert pairs == {(0, 8), (1, 7), (2, 6), (3, 11), (4, 10), (5, 9)}
    assert has_crossings(pairs)


@pytest.mark.parametrize("bad", ["(((", ")))", "((.)", "(((...))))", "abc"])
def test_malformed_raises(bad):
    with pytest.raises(DotBracketError):
        pairs_from_dotbracket(bad)


def test_is_valid_dotbracket():
    assert is_valid_dotbracket("(((...)))", 9)
    assert not is_valid_dotbracket("(((...)))", 10)
    assert not is_valid_dotbracket("(((...))", 8)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "db",
    [
        ".........",
        "(((...)))",
        "((..))((..))",
        "(((..(((...)))..)))",
        "..((((....))))..",
    ],
)
def test_roundtrip_nested(db):
    pairs = pairs_from_dotbracket(db)
    assert dotbracket_from_pairs(pairs, len(db)) == db


def test_roundtrip_pseudoknot_preserves_pairs():
    db = "((([[[)))]]]"
    pairs = pairs_from_dotbracket(db)
    rendered = dotbracket_from_pairs(pairs, len(db))
    # The bracket classes may be assigned differently, but the pair set must match.
    assert pairs_from_dotbracket(rendered) == pairs


def test_dotbracket_rejects_reused_position():
    with pytest.raises(DotBracketError):
        dotbracket_from_pairs([(0, 5), (0, 6)], 10)


def test_dotbracket_rejects_out_of_range():
    with pytest.raises(DotBracketError):
        dotbracket_from_pairs([(0, 12)], 10)


def test_has_crossings_false_for_nested():
    assert not has_crossings(pairs_from_dotbracket("(((..(((...)))..)))"))


# --------------------------------------------------------------------------
# Accuracy metrics -- hand-computed
# --------------------------------------------------------------------------


def test_identical_structures_score_perfect():
    m = compare_structures("(((...)))", "(((...)))")
    assert (m.sensitivity, m.ppv, m.f1) == (1.0, 1.0, 1.0)
    assert m.bp_distance == 0
    assert m.exact_match


def test_missing_one_pair():
    # predicted has 3 pairs, reference has 2 of them -> 1 false positive.
    m = compare_structures("(((...)))", "((.....))")
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 1, 0)
    assert m.sensitivity == 1.0
    assert m.ppv == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(0.8)
    assert m.bp_distance == 1
    assert not m.exact_match


def test_sensitivity_and_ppv_are_not_symmetric():
    # Swapping prediction and reference swaps sensitivity and PPV. Reporting
    # only F1 would hide this, which is why they are separate fields.
    a = compare_structures("(((...)))", "((.....))")
    b = compare_structures("((.....))", "(((...)))")
    assert a.sensitivity == b.ppv
    assert a.ppv == b.sensitivity
    assert a.f1 == pytest.approx(b.f1)


def test_completely_wrong_prediction():
    m = compare_structures("((....))....", "....((....))")
    assert m.true_positives == 0
    assert m.sensitivity == 0.0
    assert m.ppv == 0.0
    assert m.f1 == 0.0
    assert m.bp_distance == 4


def test_unfolded_vs_unfolded_is_perfect():
    m = compare_structures("......", "......")
    assert (m.sensitivity, m.ppv, m.f1) == (1.0, 1.0, 1.0)
    assert m.exact_match


def test_unfolded_prediction_against_folded_reference():
    m = compare_structures("........", "((....))")
    assert m.sensitivity == 0.0
    assert m.ppv == 1.0  # vacuously precise: it predicted nothing wrong
    assert m.f1 == 0.0
    assert m.false_negatives == 2


def test_length_mismatch_raises():
    with pytest.raises(DotBracketError):
        compare_structures("(((...)))", "((...))")


def test_bp_distance_matches_viennarna():
    import RNA

    for a, b in [
        ("(((...)))", "((.....))"),
        ("((..))((..))", "............"),
        ("(((..(((...)))..)))", "(((...........))).."),
    ]:
        assert compare_structures(a, b).bp_distance == RNA.bp_distance(a, b)


def test_matches_any_handles_degeneracy():
    candidates = ["((....))..", ".((....)).", "..((....))"]
    assert matches_any(".((....)).", candidates)
    assert not matches_any("((......))", candidates)


# --------------------------------------------------------------------------
# Gap decomposition
# --------------------------------------------------------------------------


def test_gap_decomposition_is_additive():
    g = GapDecomposition(e_vienna_mfe=-10.0, e_model_optimum=-8.0, e_solver=-6.0)
    assert g.encoding_gap == pytest.approx(2.0)
    assert g.optimizer_gap == pytest.approx(2.0)
    assert g.total_gap == pytest.approx(4.0)
    g.validate()


def test_perfect_model_and_solver_has_zero_gaps():
    g = GapDecomposition(e_vienna_mfe=-7.5, e_model_optimum=-7.5, e_solver=-7.5)
    assert g.encoding_gap == 0.0
    assert g.optimizer_gap == 0.0
    assert g.total_gap == 0.0
    g.validate()


def test_encoding_gap_isolated_from_optimizer_gap():
    # Solver finds its model's optimum exactly; all remaining error is encoding.
    g = GapDecomposition(e_vienna_mfe=-12.0, e_model_optimum=-9.0, e_solver=-9.0)
    assert g.optimizer_gap == 0.0
    assert g.encoding_gap == pytest.approx(3.0)
    assert g.total_gap == g.encoding_gap


def test_gap_dict_roundtrip_has_all_reported_fields():
    g = GapDecomposition(-10.0, -8.0, -6.0, optimizer_gap_model=1.25)
    d = g.as_dict()
    assert set(d) == {
        "e_vienna_mfe",
        "e_model_optimum",
        "e_solver",
        "encoding_gap",
        "optimizer_gap",
        "total_gap",
        "optimizer_gap_model",
    }
    assert d["optimizer_gap_model"] == 1.25


def test_summarize_skips_missing():
    rows = [{"a": 1.0, "b": None}, {"a": 3.0, "b": 2.0}]
    out = summarize(rows, ["a", "b"])
    assert out["a"] == pytest.approx(2.0)
    assert out["b"] == pytest.approx(2.0)
