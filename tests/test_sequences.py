"""Tests for sequence generation, validation and the data policy."""

from __future__ import annotations

import random

import pytest

from rnaqopt.config import GLOBAL_SEED
from rnaqopt.sequences import (
    TIER_FILES,
    TIER_LENGTH_RANGE,
    TIERS,
    RNASequence,
    SequenceValidationError,
    designed_hairpin,
    designed_multiloop,
    generate_tier,
    load_all,
    load_tier,
    random_sequence,
    read_fasta,
    write_fasta,
)

# --------------------------------------------------------------------------
# Validation / data policy
# --------------------------------------------------------------------------


def test_accepts_valid_rna():
    rec = RNASequence("s1", "ACGU", "A", "synthetic:test")
    assert rec.length == 4
    assert rec.gc_content == pytest.approx(0.5)


@pytest.mark.parametrize("bad", ["ACGT", "ACGN", "ACG U", "acgu!"])
def test_rejects_non_rna_alphabet(bad):
    with pytest.raises(SequenceValidationError):
        RNASequence("s1", bad, "A", "synthetic:test")


def test_rejects_empty_sequence():
    with pytest.raises(SequenceValidationError):
        RNASequence("s1", "", "A", "synthetic:test")


def test_rejects_missing_provenance():
    """Plan section 1.9: every sequence must declare where it came from."""
    with pytest.raises(SequenceValidationError):
        RNASequence("s1", "ACGU", "A", "")


def test_every_committed_sequence_is_public_or_synthetic():
    """No confidential, proprietary or patient-derived data, per plan 1.9."""
    for rec in load_all():
        assert rec.source.startswith(("synthetic:", "public:")), rec.seq_id


# --------------------------------------------------------------------------
# FASTA round trip
# --------------------------------------------------------------------------


def test_fasta_roundtrip_preserves_all_fields(tmp_path):
    records = [
        RNASequence("a1", "ACGUACGU", "A", "synthetic:random", "len=8 gc_target=0.50"),
        RNASequence("a2", "GGGGAAAACCCC", "A", "synthetic:designed-hairpin", "stem=4"),
    ]
    path = tmp_path / "t.fasta"
    write_fasta(records, path)
    back = read_fasta(path)
    assert back == records


def test_fasta_converts_dna_to_rna(tmp_path):
    path = tmp_path / "dna.fasta"
    path.write_text(">x | tier=A | source=synthetic:test\nACGT\n", encoding="utf-8")
    assert read_fasta(path)[0].sequence == "ACGU"


def test_fasta_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dup.fasta"
    path.write_text(
        ">x | tier=A | source=synthetic:test\nACGU\n"
        ">x | tier=A | source=synthetic:test\nACGU\n",
        encoding="utf-8",
    )
    with pytest.raises(SequenceValidationError):
        read_fasta(path)


def test_committed_fasta_headers_contain_no_gt_character():
    """A '>' anywhere but column 0 breaks naive FASTA parsers."""
    for path in TIER_FILES.values():
        for line in path.read_text(encoding="utf-8").splitlines():
            assert ">" not in line[1:], f"{path.name}: {line}"


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------


def test_designed_hairpin_is_self_complementary():
    rng = random.Random(0)
    seq = designed_hairpin(rng, stem_length=5, loop_length=4)
    assert len(seq) == 14
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    stem5, stem3 = seq[:5], seq[-5:]
    assert stem3 == "".join(comp[c] for c in reversed(stem5))


def test_designed_hairpin_rejects_short_loop():
    with pytest.raises(ValueError):
        designed_hairpin(random.Random(0), stem_length=4, loop_length=2)


def test_designed_multiloop_length_formula():
    rng = random.Random(0)
    cl, nb, bl, ll, sp = 4, 2, 4, 4, 1
    seq = designed_multiloop(rng, cl, nb, bl, ll, sp)
    assert len(seq) == 2 * cl + nb * (2 * bl + ll) + sp * (nb + 1)


def test_random_sequence_respects_gc_target():
    rng = random.Random(7)
    seq = random_sequence(rng, 2000, gc_target=0.75)
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    assert gc == pytest.approx(0.75, abs=0.05)


# --------------------------------------------------------------------------
# Determinism -- the reproducibility guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_generation_is_deterministic(tier):
    a = generate_tier(tier, GLOBAL_SEED)
    b = generate_tier(tier, GLOBAL_SEED)
    assert [r.sequence for r in a] == [r.sequence for r in b]


@pytest.mark.parametrize("tier", TIERS)
def test_generation_reproduces_the_committed_files(tier):
    """`make sequences` from a clean clone must give back exactly what is
    committed -- the Phase 1 reproducibility guarantee."""
    offset = TIERS.index(tier)
    generated = generate_tier(tier, GLOBAL_SEED + offset)
    committed = load_tier(tier)
    assert [(r.seq_id, r.sequence) for r in generated] == [
        (r.seq_id, r.sequence) for r in committed
    ]


def test_different_seeds_give_different_sequences():
    a = generate_tier("A", GLOBAL_SEED)
    b = generate_tier("A", GLOBAL_SEED + 1)
    assert [r.sequence for r in a] != [r.sequence for r in b]


# --------------------------------------------------------------------------
# Tier invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_tier_lengths_inside_declared_range(tier):
    lo, hi = TIER_LENGTH_RANGE[tier]
    for rec in load_tier(tier):
        assert lo <= rec.length <= hi, rec.seq_id


@pytest.mark.parametrize(
    "tier,expected", [("A", 20), ("B", 20), ("C", 10), ("M", 10)]
)
def test_tier_sizes(tier, expected):
    assert len(load_tier(tier)) == expected


def test_sequence_ids_are_globally_unique():
    ids = [r.seq_id for r in load_all()]
    assert len(ids) == len(set(ids))


def test_tier_field_matches_the_file_it_lives_in():
    for tier in TIERS:
        for rec in load_tier(tier):
            assert rec.tier == tier, rec.seq_id
