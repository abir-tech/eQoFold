"""Tests for bitstring -> stem set -> dot-bracket decoding and repair.

Repair must be *deterministic* (same input, same output, independent of set
iteration order) and *reported* (never silently folded into a result), since
plan section 6 lists silent repair as a way to inflate apparent solver quality.
"""

from __future__ import annotations

import pytest

from rnaqopt.decode import (
    bits_to_selection,
    decode,
    greedy_repair,
    selection_to_pairs,
    selection_to_structure,
    structure_to_selection,
)
from rnaqopt.metrics import pairs_from_dotbracket
from rnaqopt.reference import mfe
from rnaqopt.sequences import load_tier
from rnaqopt.solvers.base import bitstring_from_selection
from rnaqopt.stems import enumerate_with_graphs

SEQ = "GGGGAAAACCCCAAAAGGGGAAAACCCC"


def _graphs(seq=SEQ):
    return enumerate_with_graphs(seq)


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_bits_to_selection():
    assert bits_to_selection([1, 0, 1, 0, 1]) == (0, 2, 4)
    assert bits_to_selection([0, 0]) == ()


def test_bitstring_roundtrip():
    sel = (0, 3, 4)
    bits = bitstring_from_selection(sel, 6)
    assert bits == (1, 0, 0, 1, 1, 0)
    assert bits_to_selection(bits) == sel


def test_empty_selection_decodes_to_all_dots():
    g = _graphs()
    r = decode([0] * g.n, g, len(SEQ))
    assert r.structure == "." * len(SEQ)
    assert r.feasible_raw
    assert not r.was_repaired


def test_selection_to_pairs_is_the_union():
    g = _graphs()
    pairs = selection_to_pairs(g, [0])
    assert pairs == set(g.stems[0].pairs)


# --------------------------------------------------------------------------
# Round trip against the reference structures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["A", "M"])
def test_reference_structures_roundtrip_when_representable(tier):
    """structure -> selection -> structure must be the identity whenever the
    reference lies inside the model's expressive range."""
    for rec in load_tier(tier):
        g = enumerate_with_graphs(rec.sequence)
        db, _ = mfe(rec.sequence)
        sel = structure_to_selection(g, db)
        if sel is None:
            continue
        assert selection_to_structure(g, sel, len(rec.sequence)) == db, rec.seq_id


def test_structure_to_selection_returns_none_when_not_representable():
    """A reference containing a 2-bp helix cannot be built from L_min=3 stems.
    Returning None rather than a partial answer is what makes the encoding-gap
    table able to distinguish representability failure from energy error."""
    g = enumerate_with_graphs("GGGAAACCC")
    # A structure whose pairs are not a union of candidate stems.
    assert structure_to_selection(g, ".((...)).") is None


def test_decoded_selection_reproduces_its_own_pairs():
    g = _graphs()
    for idx in range(min(g.n, 5)):
        r = decode(bitstring_from_selection([idx], g.n), g, len(SEQ))
        assert pairs_from_dotbracket(r.structure) == set(g.stems[idx].pairs)


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def test_conflicting_selection_is_detected_and_repaired():
    g = _graphs()
    conflict = sorted(g.conflict)
    if not conflict:
        pytest.skip("no conflicting stem pair in this instance")
    a, b = conflict[0]
    r = decode(bitstring_from_selection([a, b], g.n), g, len(SEQ))
    assert not r.feasible_raw
    assert r.was_repaired
    assert r.n_removed == 1
    assert len(r.selected) == 1


def test_repair_is_deterministic():
    g = _graphs()
    conflict = sorted(g.conflict)
    if not conflict:
        pytest.skip("no conflicting stem pair in this instance")
    a, b = conflict[0]
    bits = bitstring_from_selection([a, b], g.n)
    first = decode(bits, g, len(SEQ)).selected
    for _ in range(5):
        assert decode(bits, g, len(SEQ)).selected == first


def test_repair_respects_priority_order():
    """With an explicit priority, the more stabilising stem must survive."""
    g = _graphs()
    conflict = sorted(g.conflict)
    if not conflict:
        pytest.skip("no conflicting stem pair in this instance")
    a, b = conflict[0]
    prio = [0.0] * g.n
    prio[b] = -100.0  # b is far more stabilising
    r = decode(bitstring_from_selection([a, b], g.n), g, len(SEQ), priority=prio)
    assert r.selected == (b,)


def test_repair_output_is_always_feasible():
    g = _graphs()
    everything = list(range(g.n))
    kept = greedy_repair(g, everything)
    assert all(
        g.compatible(x, y) for i, x in enumerate(kept) for y in kept[i + 1 :]
    )


def test_repair_never_adds_stems():
    g = _graphs()
    subset = list(range(min(g.n, 6)))
    kept = greedy_repair(g, subset)
    assert set(kept) <= set(subset)


def test_feasible_selection_is_left_untouched():
    g = _graphs()
    kept = greedy_repair(g, list(range(g.n)))
    r = decode(bitstring_from_selection(kept, g.n), g, len(SEQ))
    assert r.feasible_raw
    assert not r.was_repaired
    assert r.selected == tuple(sorted(kept))


def test_pseudoknot_mode_keeps_crossing_stems():
    rec = next(
        (r for r in load_tier("B") if enumerate_with_graphs(r.sequence).crossing), None
    )
    if rec is None:
        pytest.skip("no crossing pair available")
    g = enumerate_with_graphs(rec.sequence)
    a, b = sorted(g.crossing)[0]
    bits = bitstring_from_selection([a, b], g.n)
    nested = decode(bits, g, len(rec.sequence), pseudoknot_mode=False)
    knotted = decode(bits, g, len(rec.sequence), pseudoknot_mode=True)
    assert nested.was_repaired
    assert not knotted.was_repaired
    assert len(knotted.selected) == 2


def test_pseudoknot_structure_uses_the_extended_alphabet():
    rec = next(
        (r for r in load_tier("B") if enumerate_with_graphs(r.sequence).crossing), None
    )
    if rec is None:
        pytest.skip("no crossing pair available")
    g = enumerate_with_graphs(rec.sequence)
    a, b = sorted(g.crossing)[0]
    r = decode(
        bitstring_from_selection([a, b], g.n),
        g,
        len(rec.sequence),
        pseudoknot_mode=True,
    )
    assert "[" in r.structure and "]" in r.structure


def test_decode_result_reports_repair_separately():
    g = _graphs()
    conflict = sorted(g.conflict)
    if not conflict:
        pytest.skip("no conflicting stem pair in this instance")
    a, b = conflict[0]
    d = decode(bitstring_from_selection([a, b], g.n), g, len(SEQ)).as_dict()
    assert d["feasible_raw"] is False
    assert d["was_repaired"] is True
    assert d["n_removed"] == 1
    assert d["n_selected_raw"] == 2
