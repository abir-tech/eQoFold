"""Tests for stem enumeration and the structural graphs.

Enumeration correctness on hand-checked cases, plus invariants asserted over
the whole corpus -- because |stems| is the problem size *n* that every scaling
plot and every resource count is expressed in.
"""

from __future__ import annotations

import pytest

from rnaqopt.config import STEMS, StemConfig
from rnaqopt.sequences import load_all, load_tier
from rnaqopt.stems import (
    CANONICAL_PAIRS,
    Stem,
    build_graphs,
    can_pair,
    enumerate_stems,
    enumerate_with_graphs,
    is_nested,
    pairs_cross,
    stems_conflict,
    stems_cross,
)

# --------------------------------------------------------------------------
# Stem geometry
# --------------------------------------------------------------------------


def test_stem_pairs_and_inner():
    s = Stem(0, 8, 3)
    assert s.outer == (0, 8)
    assert s.inner == (2, 6)
    assert s.pairs == {(0, 8), (1, 7), (2, 6)}
    assert s.positions == {0, 1, 2, 6, 7, 8}
    assert s.loop_size == 3


def test_stem_of_length_one():
    s = Stem(0, 5, 1)
    assert s.inner == s.outer == (0, 5)
    assert s.pairs == {(0, 5)}
    assert s.loop_size == 4


@pytest.mark.parametrize("bad", [(5, 3, 2), (0, 0, 1)])
def test_stem_rejects_inverted_indices(bad):
    with pytest.raises(ValueError):
        Stem(*bad)


def test_stem_rejects_zero_length():
    with pytest.raises(ValueError):
        Stem(0, 8, 0)


def test_stems_are_ordered_and_hashable():
    a, b = Stem(0, 8, 3), Stem(1, 9, 3)
    assert a < b
    assert len({a, b, Stem(0, 8, 3)}) == 2


# --------------------------------------------------------------------------
# Pair predicates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x,y", sorted(CANONICAL_PAIRS))
def test_canonical_pairs_are_pairable(x, y):
    assert can_pair(x, y)


def test_gu_can_be_disabled():
    assert can_pair("G", "U", allow_gu=True)
    assert not can_pair("G", "U", allow_gu=False)
    assert can_pair("G", "C", allow_gu=False)


@pytest.mark.parametrize("x,y", [("A", "A"), ("A", "G"), ("C", "U"), ("C", "C")])
def test_non_canonical_pairs_rejected(x, y):
    assert not can_pair(x, y)


def test_pairs_cross():
    assert pairs_cross((0, 5), (3, 8))
    assert pairs_cross((3, 8), (0, 5))
    assert not pairs_cross((0, 9), (3, 6))  # nested
    assert not pairs_cross((0, 3), (5, 8))  # disjoint


# --------------------------------------------------------------------------
# Enumeration -- hand-checked
# --------------------------------------------------------------------------


def test_single_hairpin_yields_exactly_one_stem():
    """GGG AAA CCC: only (0,8) extends to length 3; every other register is
    either non-maximal or shorter than L_min."""
    stems = enumerate_stems("GGGAAACCC")
    assert stems == [Stem(0, 8, 3)]


def test_min_hairpin_loop_is_enforced():
    """GGGCCC has complementary arms but only 0 unpaired between them."""
    assert enumerate_stems("GGGCCC") == []


def test_stem_shorter_than_lmin_is_dropped():
    """GGAAACC would give a length-2 stem, below the default L_min = 3."""
    assert enumerate_stems("GGAAACC") == []
    longer = enumerate_stems("GGAAACC", StemConfig(min_stem_length=2))
    assert longer == [Stem(0, 6, 2)]


def test_no_pairs_possible():
    assert enumerate_stems("AAAAAAAAAA") == []


def test_gu_wobble_changes_the_candidate_set():
    seq = "GGGAAAUUU"
    with_gu = enumerate_stems(seq, allow_gu=True)
    without_gu = enumerate_stems(seq, allow_gu=False)
    assert with_gu == [Stem(0, 8, 3)]
    assert without_gu == []


def test_substems_expand_the_candidate_set_and_are_on_by_default():
    """Sub-stems are enabled by default (see config.StemConfig for the measured
    justification); the maximal-only set is a strict subset."""
    seq = "GGGGGAAAACCCCC"
    maximal = enumerate_stems(seq, StemConfig(include_substems=False))
    with_subs = enumerate_stems(seq, StemConfig(include_substems=True))
    # Several maximal stems exist because shifted *registers* -- (0,13), (0,12),
    # (1,13) ... -- are distinct helices, each maximal in its own alignment.
    assert len(maximal) > 1
    assert max(s.length for s in maximal) == 5
    assert len(with_subs) > len(maximal)
    assert all(s in with_subs for s in maximal)
    assert STEMS.include_substems is True
    assert enumerate_stems(seq) == with_subs


def test_substems_are_all_contained_in_a_maximal_stem():
    """Every emitted sub-stem must be a contiguous truncation of a maximal one,
    not an independent register."""
    seq = "GGGGGAAAACCCCC"
    maximal = enumerate_stems(seq, StemConfig(include_substems=False))
    for sub in enumerate_stems(seq, StemConfig(include_substems=True)):
        assert any(sub.pairs <= m.pairs for m in maximal), sub


def test_enumeration_is_deterministic_and_sorted():
    seq = load_tier("B")[0].sequence
    a = enumerate_stems(seq)
    b = enumerate_stems(seq)
    assert a == b
    assert a == sorted(a)


# --------------------------------------------------------------------------
# Corpus-wide invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["A", "M"])
def test_every_enumerated_stem_is_physically_valid(tier):
    for rec in load_tier(tier):
        seq = rec.sequence
        for stem in enumerate_stems(seq):
            assert stem.length >= STEMS.min_stem_length
            for a, b in stem.pairs:
                assert can_pair(seq[a], seq[b]), f"{rec.seq_id}: {stem} pair ({a},{b})"
            assert stem.loop_size >= 3, f"{rec.seq_id}: {stem} hairpin too small"


@pytest.mark.parametrize("tier", ["A", "M"])
def test_maximal_stems_cannot_be_extended_outward(tier):
    """Only meaningful for the maximal-only enumeration: a sub-stem is by
    definition a truncation and *can* be extended."""
    maximal_only = StemConfig(include_substems=False)
    for rec in load_tier(tier):
        seq = rec.sequence
        for stem in enumerate_stems(seq, maximal_only):
            a, b = stem.i - 1, stem.j + 1
            if 0 <= a and b < len(seq):
                assert not can_pair(seq[a], seq[b]), (
                    f"{rec.seq_id}: {stem} is not maximal"
                )


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------


def test_nesting_definition_uses_the_inner_pair():
    outer = Stem(0, 19, 3)  # inner pair (2, 17)
    inner = Stem(5, 12, 3)
    assert is_nested(outer, inner)
    assert not is_nested(inner, outer)
    # A stem starting inside the outer helix itself is not nested in its loop.
    assert not is_nested(outer, Stem(1, 18, 1))


def test_graph_classes_are_mutually_exclusive():
    """A pair of stems is conflicting, crossing, or nested -- never two of them.

    Double-classifying would double-penalise the same physical impossibility.
    """
    for rec in load_tier("M"):
        g = enumerate_with_graphs(rec.sequence)
        for a, b in g.conflict:
            assert (a, b) not in g.crossing
            assert (a, b) not in g.nesting and (b, a) not in g.nesting
        for a, b in g.crossing:
            assert (a, b) not in g.nesting and (b, a) not in g.nesting


def test_nesting_is_antisymmetric():
    for rec in load_tier("M"):
        g = enumerate_with_graphs(rec.sequence)
        for a, b in g.nesting:
            assert (b, a) not in g.nesting


def test_graphs_match_direct_recomputation():
    """The graph builder's short-circuiting must not change the classification."""
    rec = load_tier("M")[0]
    g = enumerate_with_graphs(rec.sequence)
    for a in range(g.n):
        for b in range(a + 1, g.n):
            s, t = g.stems[a], g.stems[b]
            if stems_conflict(s, t):
                assert (a, b) in g.conflict
            elif stems_cross(s, t):
                assert (a, b) in g.crossing
            elif is_nested(s, t):
                assert (a, b) in g.nesting
            elif is_nested(t, s):
                assert (b, a) in g.nesting


def test_a_stem_conflicts_with_itself_by_position():
    s = Stem(0, 8, 3)
    assert stems_conflict(s, Stem(0, 8, 3))
    assert not stems_conflict(s, Stem(10, 18, 3))


def test_compatible_respects_pseudoknot_mode():
    """Turning pseudoknot mode on is exactly dropping the crossing ban."""
    rec = next(r for r in load_tier("B") if enumerate_with_graphs(r.sequence).crossing)
    g = enumerate_with_graphs(rec.sequence)
    a, b = sorted(g.crossing)[0]
    assert not g.compatible(a, b, pseudoknot_mode=False)
    assert g.compatible(a, b, pseudoknot_mode=True)


def test_summary_counts_are_consistent():
    g = enumerate_with_graphs(load_tier("M")[0].sequence)
    s = g.summary()
    assert s["n_stems"] == g.n == len(g.stems)
    assert s["n_conflict"] == len(g.conflict)


def test_build_graphs_on_empty_stem_set():
    g = build_graphs([])
    assert g.n == 0
    assert not g.conflict and not g.crossing and not g.nesting


def test_problem_size_is_recorded_for_every_sequence():
    """|stems| is the x-axis of every scaling plot; it must exist for all."""
    for rec in load_all():
        assert len(enumerate_stems(rec.sequence)) >= 0


def test_tier_a_stays_within_exact_solving_reach():
    """Tier A carries the encoding-fidelity claims, which need an exact optimum.
    Brute force handles <= ~22 variables and CP-SAT the rest, so the ceiling
    that matters is CP-SAT's, not 2^n."""
    sizes = [len(enumerate_stems(r.sequence)) for r in load_tier("A")]
    assert max(sizes) <= 40, f"Tier A grew to {max(sizes)} variables"
    assert sum(1 for s in sizes if s <= 20) >= 12, (
        "at least 12 Tier A instances must fit a 20-qubit statevector "
        "simulation for the gate-based study"
    )
