"""Synthetic/random RNA sequence dataset for benchmarking, per the challenge's
Data Privacy rule (publicly available, synthetic, or randomly generated
sequences only -- no proprietary or clinical data).
"""
from __future__ import annotations

import numpy as np

BASES = ["A", "U", "C", "G"]

# The sequence given verbatim as a worked example in the challenge PDF itself
# (WISER<>Moderna challenge doc, Task 2 code sample) -- publicly available via
# the challenge document, explicitly provided for participants to use.
CHALLENGE_EXAMPLE_SEQ = "GGAGCAAAACUUGUCGAUUGAGAACAAAAUACAGAAUUUGCUUG"


def random_sequence(length: int, seed: int, gc_bias: float = 0.5) -> str:
    """gc_bias: probability mass on {G,C} vs {A,U} (0.5 = uniform over 4 bases)."""
    rng = np.random.default_rng(seed)
    probs = np.array([
        (1 - gc_bias) / 2, (1 - gc_bias) / 2,  # A, U
        gc_bias / 2, gc_bias / 2,               # C, G
    ])
    return "".join(rng.choice(BASES, size=length, p=probs))


def perfect_hairpin(stem_len: int, loop_len: int = 4) -> str:
    """A designed positive-control sequence: a single unambiguous stem-loop
    with no alternative folds, for validating the pipeline end to end."""
    rng = np.random.default_rng(0)
    stem5 = "".join(rng.choice(["G", "C"], size=stem_len))
    complement = {"G": "C", "C": "G", "A": "U", "U": "A"}
    stem3 = "".join(complement[b] for b in reversed(stem5))
    loop = "".join(rng.choice(["A", "U"], size=loop_len))
    return stem5 + loop + stem3


def scaling_dataset(lengths=(10, 14, 18, 22, 26, 30, 36, 44), seeds_per_length: int = 2,
                     gc_bias: float = 0.55):
    """Returns list of dicts: {length, seed, seq}. gc_bias slightly above 0.5
    is a deliberate, documented choice -- pure-uniform random sequences at
    length<~20 frequently have NO valid canonical pairing at all (too few GC/AU/GU
    complements survive the min-loop filter), which produces a degenerate
    empty-QUBO edge case rather than a meaningful folding instance."""
    out = []
    for L in lengths:
        for s in range(seeds_per_length):
            seed = 1000 * L + s
            out.append(dict(length=L, seed=seed, seq=random_sequence(L, seed, gc_bias=gc_bias)))
    return out


if __name__ == "__main__":
    for row in scaling_dataset():
        print(row)
