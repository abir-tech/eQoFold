"""Tests for the ViennaRNA reference layer.

The critical property is the plan's section 2.2 corollary: the energy we report
for a structure is always ``eval_structure`` on its dot-bracket string.  If
``eval_structure(mfe_structure) != mfe_energy`` then the configuration is being
applied inconsistently somewhere, and every downstream number is suspect.
"""

from __future__ import annotations

import pytest

from rnaqopt.config import VIENNA, VIENNA_REFERENCE_CSV, VIENNA_STOCK
from rnaqopt.metrics import pairs_from_dotbracket
from rnaqopt.reference import (
    REFERENCE_COLUMNS,
    analyze,
    eval_structure,
    fold_compound,
    has_multiloop,
    helix_stats,
    mfe,
    partition_function,
    read_reference_csv,
    rt_kcal,
    subopt,
)
from rnaqopt.sequences import load_all, load_tier

HAIRPIN = "GGGGAAAACCCC"


def test_mfe_energy_equals_eval_structure():
    structure, energy = mfe(HAIRPIN)
    assert eval_structure(HAIRPIN, structure) == pytest.approx(energy, abs=1e-9)


def test_mfe_of_designed_hairpin_is_a_hairpin():
    structure, energy = mfe(HAIRPIN)
    assert structure == "((((....))))"
    assert energy < 0


def test_eval_structure_rejects_length_mismatch():
    with pytest.raises(ValueError):
        eval_structure(HAIRPIN, "((((...))))")


def test_unfolded_structure_has_zero_energy():
    assert eval_structure(HAIRPIN, "." * len(HAIRPIN)) == pytest.approx(0.0)


def test_config_actually_reaches_viennarna():
    """dangles and noLP must differ between the two configs, or they are ignored."""
    md0 = VIENNA.model_details()
    md2 = VIENNA_STOCK.model_details()
    assert md0.dangles == 0 and md0.noLP == 1
    assert md2.dangles == 2 and md2.noLP == 0
    assert md0.temperature == pytest.approx(37.0)


def test_dangles_setting_changes_energies():
    """Guards against the settings being silently dropped by the SWIG layer."""
    seq = "GGGAAAUCCCAGGGAAACCC"
    e0 = mfe(seq, VIENNA)[1]
    e2 = mfe(seq, VIENNA_STOCK)[1]
    assert e0 != e2


def test_fold_compound_returns_fresh_objects():
    a = fold_compound(HAIRPIN)
    b = fold_compound(HAIRPIN)
    assert a is not b


def test_ensemble_free_energy_is_at_most_mfe():
    """F <= E_mfe always: summing more Boltzmann weight can only lower F."""
    for seq in ("GGGGAAAACCCC", "GCGCUUAAGCGC", "AUAUAUAUAUAU"):
        _, e_mfe = mfe(seq)
        _, f_ens = partition_function(seq)
        assert f_ens <= e_mfe + 1e-6


def test_subopt_contains_the_mfe_structure():
    structure, energy = mfe(HAIRPIN)
    sols = subopt(HAIRPIN, 1.0)
    assert (structure, energy) in sols
    assert sols[0][1] == pytest.approx(energy)


def test_subopt_is_sorted_and_within_delta():
    structure, energy = mfe(HAIRPIN)
    sols = subopt(HAIRPIN, 1.5)
    assert sols == sorted(sols, key=lambda t: (t[1], t[0]))
    assert all(e <= energy + 1.5 + 1e-9 for _, e in sols)


def test_rt_is_physically_sensible():
    assert rt_kcal(VIENNA) == pytest.approx(0.6163, abs=1e-3)


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "db,n_helices,longest",
    [
        (".........", 0, 0),
        ("(((...)))", 1, 3),
        ("((..))((..))", 2, 2),
        ("(((..(((...)))..)))", 2, 3),  # helix broken by an internal loop
        ("((.((...)).))", 2, 2),
    ],
)
def test_helix_stats(db, n_helices, longest):
    assert helix_stats(db) == (n_helices, longest)


@pytest.mark.parametrize(
    "db,expected",
    [
        ("(((...)))", False),  # hairpin
        ("(((..(((...)))..)))", False),  # internal loop, one branch
        ("((..))((..))", False),  # two external hairpins: external loop, not ML
        ("((((..))..((..))))", True),  # two branches inside one closing pair
        ("(((((...))((...))))) ".strip(), True),
    ],
)
def test_has_multiloop(db, expected):
    assert has_multiloop(db) is expected


def test_has_multiloop_agrees_with_viennarna_on_the_whole_corpus():
    """Cross-validate our structural detector against ViennaRNA's own loop
    decomposition. This is the check that makes the Tier M design defensible."""
    import os
    import tempfile

    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    path = os.path.join(tempfile.gettempdir(), "rnaqopt_verbose_test.txt")
    for _, row in df.iterrows():
        fc = fold_compound(row.sequence, VIENNA)
        with open(path, "w") as fh:
            fc.eval_structure_verbose(row.mfe_structure, fh)
        vienna_says = "ulti" in open(path).read()  # "Multi loop" / "multiloop"
        assert has_multiloop(row.mfe_structure) is vienna_says, row.seq_id


# --------------------------------------------------------------------------
# The reference table
# --------------------------------------------------------------------------


def test_analyze_row_has_exactly_the_declared_columns():
    rec = load_tier("A")[0]
    row = analyze(rec).as_row()
    assert tuple(row) == REFERENCE_COLUMNS


def test_reference_csv_exists_and_covers_every_sequence():
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    ids = {r.seq_id for r in load_all()}
    assert set(df.seq_id) == ids
    assert len(df) == len(ids)


def test_reference_csv_energies_are_self_consistent():
    """Every committed row must survive re-evaluation. This is the check that
    catches a stale reference table after a configuration change."""
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    for _, row in df.iterrows():
        assert eval_structure(row.sequence, row.mfe_structure) == pytest.approx(
            row.mfe_energy, abs=1e-9
        ), row.seq_id


def test_reference_csv_has_one_configuration_only():
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    assert df.config_fingerprint.nunique() == 1
    assert df.config_fingerprint.iloc[0] == VIENNA.fingerprint()


def test_reference_pair_counts_match_structures():
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    for _, row in df.iterrows():
        assert len(pairs_from_dotbracket(row.mfe_structure)) == row.n_pairs, row.seq_id


def test_no_sequence_folds_to_nothing():
    """Non-degeneracy screen: an empty MFE is trivially 'solved' by every model
    and wastes a benchmark slot."""
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    assert (df.n_pairs > 0).all(), list(df.loc[df.n_pairs == 0, "seq_id"])


def test_tier_M_actually_contains_multiloops():
    """The reason tier M exists: Level 2's cubic terms model the multiloop
    branch penalty, and tiers A and B contain essentially no multiloops."""
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    m = df[df.tier == "M"]
    assert len(m) == 10
    assert m.has_multiloop.all()


def test_mfe_probability_within_unit_interval():
    df = read_reference_csv(VIENNA_REFERENCE_CSV)
    assert ((df.mfe_probability >= 0) & (df.mfe_probability <= 1)).all()
