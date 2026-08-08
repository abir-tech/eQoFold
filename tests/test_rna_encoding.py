import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rna_encoding import (  # noqa: E402
    build_pair_qubo, build_stem_qubo, decode_bits_to_pairs,
    enumerate_candidate_pairs, is_valid_pair, nussinov_exact, pairs_cross,
    pairs_to_dotbracket, pair_weight,
)
from vienna_utils import dotbracket_to_pairs  # noqa: E402


def _brute_force_nested_matching(seq, min_loop=3, allow_wobble=True):
    """Reference brute-force over ALL subsets of candidate pairs -- exponential,
    only usable on tiny sequences, to independently check nussinov_exact."""
    candidates = enumerate_candidate_pairs(seq, min_loop=min_loop, allow_wobble=allow_wobble)
    best = 0.0
    seq_u = seq.upper().replace("T", "U")
    for r in range(len(candidates) + 1):
        for combo in _combinations(candidates, r):
            idx = set()
            ok = True
            for i, j in combo:
                if i in idx or j in idx:
                    ok = False
                    break
                idx.add(i)
                idx.add(j)
            if not ok:
                continue
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    if pairs_cross(combo[a], combo[b]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            score = sum(pair_weight(seq_u[i], seq_u[j], allow_wobble) for i, j in combo)
            best = max(best, score)
    return best


def _combinations(seq, r):
    from itertools import combinations
    return combinations(seq, r)


@pytest.mark.parametrize("seq", ["GGGGCCCC", "GCAUAUGC", "AAAAAAAA", "GGAUCC", "GCGCAAAAGCGC"])
def test_nussinov_matches_bruteforce(seq):
    exact_score, pairs = nussinov_exact(seq)
    bf_score = _brute_force_nested_matching(seq)
    assert abs(exact_score - bf_score) < 1e-9, f"{seq}: DP={exact_score} brute-force={bf_score}"
    # returned pairs must actually be non-crossing and non-conflicting
    idx = set()
    for i, j in pairs:
        assert i not in idx and j not in idx
        idx.add(i)
        idx.add(j)
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            assert not pairs_cross(pairs[a], pairs[b])


def test_valid_pair_rules():
    assert is_valid_pair("G", "C")
    assert is_valid_pair("A", "U")
    assert is_valid_pair("G", "U")  # wobble
    assert not is_valid_pair("G", "U", allow_wobble=False)
    assert not is_valid_pair("A", "G")
    assert not is_valid_pair("C", "U")


def test_pair_qubo_optimum_equals_nussinov():
    """The exact optimum of the pair-level QUBO (found by brute-force over all
    2^m bitstrings) must equal -1 * the exact weighted-Nussinov score, since
    the penalty is calibrated to always dominate any constraint violation."""
    seq = "GGGCAUGCCC"
    qp = build_pair_qubo(seq)
    m = qp.num_vars
    assert m <= 16, "keep this test's brute force cheap"
    best = np.inf
    for bits in product([0, 1], repeat=m):
        best = min(best, qp.energy(bits))
    score, _ = nussinov_exact(seq)
    assert abs(best - (-score)) < 1e-9


@pytest.mark.parametrize("seq,kind", [
    ("GGGCAUGCCCAUGCGCUA", "pair"),
    ("GGGCAUGCCCAUGCGCUA", "stem"),
])
def test_decode_always_feasible(seq, kind):
    """decode_bits_to_pairs must always return a valid nested, non-conflicting
    structure regardless of how garbled the input bitstring is (adversarial
    all-ones input is the hardest case: maximal raw conflict)."""
    qp = build_pair_qubo(seq) if kind == "pair" else build_stem_qubo(seq)
    rng = np.random.default_rng(0)
    for _ in range(20):
        bits = rng.integers(0, 2, size=qp.num_vars)
        pairs = decode_bits_to_pairs(qp, bits, seq)
        idx = set()
        for i, j in pairs:
            assert i not in idx and j not in idx
            idx.add(i)
            idx.add(j)
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                assert not pairs_cross(pairs[a], pairs[b])
    # all-ones is the maximal-conflict adversarial case
    pairs = decode_bits_to_pairs(qp, [1] * qp.num_vars, seq)
    idx = set()
    for i, j in pairs:
        assert i not in idx and j not in idx
        idx.add(i)
        idx.add(j)


def test_dotbracket_roundtrip():
    seq = "GGGCAUGCCC"
    _, pairs = nussinov_exact(seq)
    db = pairs_to_dotbracket(len(seq), pairs)
    recovered = dotbracket_to_pairs(db)
    assert sorted(recovered) == sorted(pairs)


def test_no_candidate_pairs_short_sequence():
    qp = build_pair_qubo("AAAA")  # too short for min_loop=3
    assert qp.num_vars == 0
