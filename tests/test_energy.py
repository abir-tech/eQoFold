"""Tests for Turner term extraction.

The load-bearing test is
:func:`test_loop_tree_decomposition_is_exact_on_every_reference_structure`.
Every model coefficient in this project is built from these primitives, so if
the decomposition does not reproduce ``eval_structure`` exactly, every number
downstream is wrong.
"""

from __future__ import annotations

import pytest

from rnaqopt.energy import (
    LoopEnergies,
    _children,
    _pair_table,
    decompose_structure,
    decomposed_energy,
)
from rnaqopt.reference import eval_structure, read_reference_csv
from rnaqopt.stems import Stem

HAIRPIN_SEQ = "GGGGAAAACCCC"
HAIRPIN_DB = "((((....))))"


# --------------------------------------------------------------------------
# The correctness anchor
# --------------------------------------------------------------------------


def test_loop_tree_decomposition_is_exact_on_every_reference_structure():
    """Sum of extracted loop terms == ViennaRNA's own eval_structure.

    This is what licenses using these terms as model coefficients.
    """
    df = read_reference_csv()
    for _, row in df.iterrows():
        assert decomposed_energy(row.sequence, row.mfe_structure) == pytest.approx(
            row.mfe_energy, abs=1e-9
        ), row.seq_id


def test_decomposition_of_a_simple_hairpin():
    terms = decompose_structure(HAIRPIN_SEQ, HAIRPIN_DB)
    kinds = sorted(t.kind for t in terms)
    assert kinds == ["exterior", "hairpin", "stack", "stack", "stack"]
    assert sum(t.energy for t in terms) == pytest.approx(
        eval_structure(HAIRPIN_SEQ, HAIRPIN_DB), abs=1e-9
    )


def test_decomposition_labels_a_multiloop():
    seq = "CCCGGCGCGAUAACGCGCCCGUGAAAUCACGGCGGG"
    db = "(((((((((....)))))(((((....))))))))) "[:-1]
    kinds = [t.kind for t in decompose_structure(seq, db)]
    assert "multiloop" in kinds
    assert decomposed_energy(seq, db) == pytest.approx(
        eval_structure(seq, db), abs=1e-9
    )


def test_decomposition_of_the_unfolded_structure_is_zero():
    assert decomposed_energy(HAIRPIN_SEQ, "." * len(HAIRPIN_SEQ)) == 0.0


def test_decomposition_rejects_unbalanced_structure():
    with pytest.raises(ValueError):
        decompose_structure(HAIRPIN_SEQ, "((((....)))")  # one closer short


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_stacking_is_stabilising_and_hairpin_is_destabilising():
    le = LoopEnergies(HAIRPIN_SEQ)
    stem = Stem(0, 11, 4)
    assert le.stack(stem) < 0
    assert le.hairpin(stem) > 0


def test_stem_of_length_one_has_no_stack():
    le = LoopEnergies(HAIRPIN_SEQ)
    assert le.stack(Stem(0, 11, 1)) == 0.0


def test_stack_plus_hairpin_equals_eval_structure_for_a_lone_hairpin():
    """The Level 1 linear coefficient must be the true energy of a lone stem."""
    le = LoopEnergies(HAIRPIN_SEQ)
    stem = Stem(0, 11, 4)
    total = le.stack(stem) + le.hairpin(stem) + le.exterior(stem)
    assert total == pytest.approx(eval_structure(HAIRPIN_SEQ, HAIRPIN_DB), abs=1e-9)


def test_terminal_penalty_applies_to_au_and_gu_only():
    seq = "AGGGAAAACCCU"
    le = LoopEnergies(seq)
    assert le.terminal_penalty(0, 11) == pytest.approx(le.terminal_au)  # A-U
    assert le.terminal_penalty(1, 10) == 0.0  # G-C


def test_exterior_contribution_is_the_terminal_penalty_under_dangles0():
    seq = "AGGGAAAACCCU"
    le = LoopEnergies(seq)
    assert le.exterior(Stem(0, 11, 4)) == pytest.approx(le.terminal_au, abs=1e-9)


def test_multiloop_constants_match_turner_2004():
    le = LoopEnergies(HAIRPIN_SEQ)
    assert le.ml_closing == pytest.approx(9.3)
    assert le.ml_intern == pytest.approx(-0.9)
    # Zero in Turner 2004 -- this is why the Level 2 term depends only on
    # branch count and not on the number of unpaired nucleotides.
    assert le.ml_base == pytest.approx(0.0)
    assert le.terminal_au == pytest.approx(0.5)


def test_multiloop_energy_is_affine_in_branch_count():
    """The property the Level 2 cubic model exists to exploit."""
    le = LoopEnergies("G" * 40)
    closing = Stem(0, 39, 2)
    branches = [Stem(5 + 8 * k, 10 + 8 * k, 1) for k in range(4)]
    values = [le.multiloop(closing, branches[:k]) for k in range(1, 5)]
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert all(d == pytest.approx(le.ml_intern) for d in deltas)


def test_interior_energy_between_nested_stems():
    seq = "GGGGAAAGGAAAACCAAACCCC"
    db = "((((...((....))...))))"
    le = LoopEnergies(seq)
    outer, inner = Stem(0, 21, 4), Stem(7, 14, 2)
    assert le.interior(outer, inner) > 0  # an interior loop costs energy
    assert decomposed_energy(seq, db) == pytest.approx(
        eval_structure(seq, db), abs=1e-9
    )


# --------------------------------------------------------------------------
# Pair-table helpers
# --------------------------------------------------------------------------


def test_pair_table():
    assert _pair_table("((..))") == [5, 4, -1, -1, 1, 0]
    assert _pair_table("....") == [-1, -1, -1, -1]


def test_children_counts_branches_and_unpaired():
    pt = _pair_table("((..((..))..((..))..))")
    kids, unpaired = _children(pt, 1, 20)
    assert len(kids) == 2
    assert unpaired == 6


def test_children_of_a_hairpin_is_empty():
    pt = _pair_table("((....))")
    kids, unpaired = _children(pt, 1, 6)
    assert kids == []
    assert unpaired == 4
