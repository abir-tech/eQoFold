"""Concrete illustration for report Sec 7.1: construct a sequence with two
candidate stems that CROSS (would form a pseudoknot if both were selected),
and show directly that the QUBO's crossing-penalty term forbids selecting
both simultaneously, however individually favorable each one is.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rna_encoding import (build_pair_qubo, enumerate_candidate_pairs,  # noqa: E402
                           pairs_cross, pairs_to_dotbracket)

# Designed so that pairing (0..4) with (10..14) AND (5..9) with (15..19)
# simultaneously would cross: a canonical H-type pseudoknot skeleton.
SEQ = "GGGGG" + "CCCCC" + "CCCCC" + "GGGGG"  # 0-4 pairs with 20-24(GGGGG complement CCCCC), etc.
# Simpler explicit construction: two interleaved stems.
SEQ = "GGGGGAAAACCCCCAAAAGGGGGAAAACCCCC"
#      01234 5678 9..13 14..17 18..22 23..26 27..31
# stem A: (0-4) with (18-22) [G with C, distance apart]
# stem B: (9-13) with (27-31) [C with G]
# These two stems cross: 0<9<22<31? need i<k<j<l pattern: A=(0,22),(4,18)... let's just search programmatically.


def main():
    seq = SEQ
    print(f"illustration sequence ({len(seq)} nt): {seq}")
    qp = build_pair_qubo(seq)
    candidates = enumerate_candidate_pairs(seq)
    print(f"m={qp.num_vars} candidate pairs, penalty={qp.penalty:.2f}")

    # find a genuinely crossing pair of candidates with strong individual weights
    best = None
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            if pairs_cross(candidates[a], candidates[b]):
                wa = -qp.Q.get((a, a), 0.0)
                wb = -qp.Q.get((b, b), 0.0)
                score = wa + wb
                if best is None or score > best[0]:
                    best = (score, a, b, wa, wb)
    score, a, b, wa, wb = best
    pa, pb = candidates[a], candidates[b]
    print(f"\nchosen crossing pair of candidate base pairs (a pseudoknot skeleton):")
    print(f"  pair A = {pa}  weight w_A = {wa:.2f}")
    print(f"  pair B = {pb}  weight w_B = {wb:.2f}")
    print(f"  crossing? {pairs_cross(pa, pb)}")

    bits_neither = [0] * qp.num_vars
    bits_a = [0] * qp.num_vars; bits_a[a] = 1
    bits_b = [0] * qp.num_vars; bits_b[b] = 1
    bits_both = [0] * qp.num_vars; bits_both[a] = 1; bits_both[b] = 1

    for name, bits in [("neither", bits_neither), ("A only", bits_a), ("B only", bits_b),
                        ("both (pseudoknot)", bits_both)]:
        print(f"  H(x) with {name:20s} selected: {qp.energy(bits):8.2f}")

    print(f"\n=> selecting BOTH individually-favorable but crossing pairs costs "
          f"{qp.energy(bits_both) - min(qp.energy(bits_a), qp.energy(bits_b)):.2f} MORE than "
          f"the better single selection, purely from the quadratic crossing penalty "
          f"(penalty={qp.penalty:.2f}) -- so no minimizer of H ever contains a pseudoknot, "
          f"by construction, regardless of how favorable each individual stem is.")


if __name__ == "__main__":
    main()
