"""RNA secondary-structure -> QUBO encoding utilities.

Formulates MFE-style secondary structure prediction as a maximum-weight
nested-matching problem over candidate base pairs (i, j):

    maximize   sum_{(i,j) in C} w_ij * x_ij
    subject to each nucleotide index used by at most one selected pair
               (no two selected pairs may cross -> no pseudoknots)

C is the candidate set of chemically valid pairs (Watson-Crick + GU wobble)
respecting a minimum hairpin-loop length. This is the classic weighted
Nussinov-Jacobson objective, so it has a polynomial-time exact classical
solution (nussinov_exact) that we use as ground truth for "did the QUBO
solver actually solve the QUBO" -- independent of how close the QUBO
objective itself is to the true ViennaRNA thermodynamic MFE.

Two variable encodings are provided:
  - pair-level:  one binary variable per candidate base pair (i, j)
  - stem-level:  one binary variable per maximal candidate stem (a run of
                 stacked, consecutive nested pairs), which trades some
                 expressiveness for far fewer variables at longer sequence
                 lengths (the qubit-count/expressiveness tradeoff asked for
                 in the challenge's optional "compare encodings" task).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

Pair = Tuple[int, int]

# Simplified base-pair stability ordering (GC strongest, then AU, then GU
# wobble). This is the standard weighted-Nussinov surrogate used when a full
# nearest-neighbor thermodynamic model is not baked into the QUBO itself;
# ViennaRNA is used later as the real thermodynamic reference.
_CANONICAL = {frozenset({"G", "C"}): 3.0, frozenset({"A", "U"}): 2.0}
_WOBBLE = {frozenset({"G", "U"}): 1.0}


def is_valid_pair(b1: str, b2: str, allow_wobble: bool = True) -> bool:
    key = frozenset({b1, b2})
    if key in _CANONICAL:
        return True
    if allow_wobble and key in _WOBBLE:
        return True
    return False


def pair_weight(b1: str, b2: str, allow_wobble: bool = True) -> float:
    key = frozenset({b1, b2})
    if key in _CANONICAL:
        return _CANONICAL[key]
    if allow_wobble and key in _WOBBLE:
        return _WOBBLE[key]
    return 0.0


def enumerate_candidate_pairs(
    seq: str, min_loop: int = 3, allow_wobble: bool = True
) -> List[Pair]:
    """All (i, j), i < j, that are chemically valid and respect min hairpin loop."""
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    pairs = []
    for i in range(n):
        for j in range(i + min_loop + 1, n):
            if is_valid_pair(seq[i], seq[j], allow_wobble=allow_wobble):
                pairs.append((i, j))
    return pairs


def pairs_cross(a: Pair, b: Pair) -> bool:
    (i, j), (k, l) = a, b
    return (i < k < j < l) or (k < i < l < j)


def pairs_share_index(a: Pair, b: Pair) -> bool:
    return len(set(a) & set(b)) > 0


def pairs_to_dotbracket(n: int, pairs: Sequence[Pair]) -> str:
    dots = ["."] * n
    for i, j in pairs:
        dots[i] = "("
        dots[j] = ")"
    return "".join(dots)


# --------------------------------------------------------------------------
# Exact classical ground truth (weighted Nussinov DP) -- O(n^3) time, used to
# certify whether a QUBO solver found the true optimum of the QUBO objective
# itself (a solver-quality check, independent of how well the QUBO objective
# approximates real thermodynamics).
# --------------------------------------------------------------------------

def nussinov_exact(
    seq: str, min_loop: int = 3, allow_wobble: bool = True
) -> Tuple[float, List[Pair]]:
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    dp = [[0.0] * n for _ in range(n)]
    choice = [[None] * n for _ in range(n)]  # None, 'skip_i', 'skip_j', (k)

    for span in range(min_loop + 1, n):
        for i in range(0, n - span):
            j = i + span
            best = dp[i + 1][j]
            best_choice = "skip_i"
            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                best_choice = "skip_j"
            if is_valid_pair(seq[i], seq[j], allow_wobble=allow_wobble):
                inner = dp[i + 1][j - 1] if j - 1 >= i + 1 else 0.0
                val = inner + pair_weight(seq[i], seq[j], allow_wobble=allow_wobble)
                if val > best:
                    best = val
                    best_choice = "pair"
            for k in range(i, j):
                val = dp[i][k] + dp[k + 1][j]
                if val > best + 1e-12:
                    best = val
                    best_choice = ("bifurcate", k)
            dp[i][j] = best
            choice[i][j] = best_choice

    pairs: List[Pair] = []

    def traceback(i: int, j: int) -> None:
        if j - i <= min_loop or i >= j:
            return
        c = choice[i][j]
        if c == "skip_i":
            traceback(i + 1, j)
        elif c == "skip_j":
            traceback(i, j - 1)
        elif c == "pair":
            pairs.append((i, j))
            traceback(i + 1, j - 1)
        elif isinstance(c, tuple) and c[0] == "bifurcate":
            k = c[1]
            traceback(i, k)
            traceback(k + 1, j)

    traceback(0, n - 1)
    pairs.sort()
    return dp[0][n - 1], pairs


# --------------------------------------------------------------------------
# QUBO construction: pair-level encoding
# --------------------------------------------------------------------------

@dataclass
class QUBOProblem:
    """Q is upper-triangular: energy(x) = sum_a Q[a,a] x_a + sum_{a<b} Q[a,b] x_a x_b.

    Minimizing `energy` maximizes total pair weight subject to soft
    penalties for conflicting (index-sharing) and crossing (pseudoknot)
    pairs. var_items[k] gives the (i, j) pair or tuple-of-pairs a stem
    represented by variable k.
    """

    n_seq: int
    Q: Dict[Tuple[int, int], float]
    var_items: List[object]
    num_vars: int
    penalty: float
    kind: str  # "pair" or "stem"

    def energy(self, bits: Sequence[int]) -> float:
        e = 0.0
        for (a, b), c in self.Q.items():
            if a == b:
                e += c * bits[a]
            else:
                e += c * bits[a] * bits[b]
        return e


def build_pair_qubo(
    seq: str,
    min_loop: int = 3,
    allow_wobble: bool = True,
    penalty: Optional[float] = None,
    weight_override: Optional[List[float]] = None,
) -> QUBOProblem:
    """weight_override: precomputed per-candidate weights aligned with
    enumerate_candidate_pairs(seq, min_loop, allow_wobble)'s order (e.g. from
    vienna_utils's empirical Turner-model weights), replacing the default
    flat GC/AU/GU surrogate score."""
    candidates = enumerate_candidate_pairs(seq, min_loop=min_loop, allow_wobble=allow_wobble)
    seq_u = seq.upper().replace("T", "U")
    if weight_override is not None:
        assert len(weight_override) == len(candidates)
        weights = list(weight_override)
    else:
        weights = [pair_weight(seq_u[i], seq_u[j], allow_wobble=allow_wobble) for i, j in candidates]
    if penalty is None:
        penalty = 2.0 * max((abs(w) for w in weights), default=1.0) + 1.0

    Q: Dict[Tuple[int, int], float] = {}
    m = len(candidates)
    for a in range(m):
        Q[(a, a)] = Q.get((a, a), 0.0) - weights[a]  # maximize weight -> minimize -weight
    for a, b in combinations(range(m), 2):
        pa, pb = candidates[a], candidates[b]
        if pairs_share_index(pa, pb) or pairs_cross(pa, pb):
            Q[(a, b)] = Q.get((a, b), 0.0) + penalty

    return QUBOProblem(
        n_seq=len(seq_u), Q=Q, var_items=candidates, num_vars=m, penalty=penalty, kind="pair"
    )


# --------------------------------------------------------------------------
# QUBO construction: stem-level encoding (coarser, fewer variables)
# --------------------------------------------------------------------------

Stem = Tuple[Pair, ...]


def enumerate_candidate_stems(
    seq: str, min_loop: int = 3, allow_wobble: bool = True, min_stem_len: int = 2, max_stem_len: int = 8
) -> List[Stem]:
    """Maximal runs of stacked nested pairs (i,j),(i+1,j-1),...,(i+L-1,j-L+1)."""
    seq_u = seq.upper().replace("T", "U")
    n = len(seq_u)
    stems: List[Stem] = []
    for i in range(n):
        for j in range(i + min_loop + 1, n):
            if not is_valid_pair(seq_u[i], seq_u[j], allow_wobble=allow_wobble):
                continue
            run = [(i, j)]
            k = 1
            while k < max_stem_len:
                ii, jj = i + k, j - k
                if ii >= jj - min_loop or not is_valid_pair(seq_u[ii], seq_u[jj], allow_wobble=allow_wobble):
                    break
                run.append((ii, jj))
                k += 1
            for length in range(min_stem_len, len(run) + 1):
                stems.append(tuple(run[:length]))
    # dedupe
    return sorted(set(stems))


def stem_weight(seq: str, stem: Stem, allow_wobble: bool = True) -> float:
    seq_u = seq.upper().replace("T", "U")
    base = sum(pair_weight(seq_u[i], seq_u[j], allow_wobble=allow_wobble) for i, j in stem)
    # stacking bonus rewards longer contiguous helices, mirroring the fact
    # that stacked base pairs are thermodynamically more stabilizing than
    # isolated pairs in real nearest-neighbor models.
    stacking_bonus = 0.5 * (len(stem) - 1)
    return base + stacking_bonus


def stem_pairs(stem: Stem) -> set:
    idx = set()
    for i, j in stem:
        idx.add(i)
        idx.add(j)
    return idx


def stems_conflict(a: Stem, b: Stem) -> bool:
    if stem_pairs(a) & stem_pairs(b):
        return True
    for pa in a:
        for pb in b:
            if pairs_cross(pa, pb):
                return True
    return False


def build_stem_qubo(
    seq: str,
    min_loop: int = 3,
    allow_wobble: bool = True,
    min_stem_len: int = 2,
    max_stem_len: int = 8,
    penalty: Optional[float] = None,
    weight_override: Optional[List[float]] = None,
) -> QUBOProblem:
    """weight_override: precomputed per-candidate-stem weights aligned with
    enumerate_candidate_stems(seq, ...)'s order (e.g. vienna_utils's empirical
    per-stem Turner-model stability, in place of the flat surrogate score)."""
    stems = enumerate_candidate_stems(
        seq, min_loop=min_loop, allow_wobble=allow_wobble,
        min_stem_len=min_stem_len, max_stem_len=max_stem_len,
    )
    if weight_override is not None:
        assert len(weight_override) == len(stems)
        weights = list(weight_override)
    else:
        weights = [stem_weight(seq, s, allow_wobble=allow_wobble) for s in stems]
    if penalty is None:
        penalty = 2.0 * max((abs(w) for w in weights), default=1.0) + 1.0

    Q: Dict[Tuple[int, int], float] = {}
    m = len(stems)
    for a in range(m):
        Q[(a, a)] = Q.get((a, a), 0.0) - weights[a]
    for a, b in combinations(range(m), 2):
        if stems_conflict(stems[a], stems[b]):
            Q[(a, b)] = Q.get((a, b), 0.0) + penalty

    return QUBOProblem(
        n_seq=len(seq), Q=Q, var_items=stems, num_vars=m, penalty=penalty, kind="stem"
    )


# --------------------------------------------------------------------------
# Decode: project a (possibly QUBO-infeasible) bitstring to a valid,
# non-crossing, non-conflicting structure via greedy weight-first repair.
# This mirrors the role of a "decoder" layer in QUBO pipelines: raw sampled
# or generated bitstrings are not guaranteed feasible, so we always resolve
# them to the best feasible sub-selection rather than rejecting them.
# --------------------------------------------------------------------------

def decode_bits_to_pairs(problem: QUBOProblem, bits: Sequence[int], seq: str) -> List[Pair]:
    """Greedy weight-first repair. Weight is read back from the QUBO's own
    diagonal coefficient (-Q[k,k]), not recomputed from the canonical
    GC/AU/GU surrogate, so this stays correct when the problem was built with
    weight_override (e.g. ViennaRNA-calibrated stem weights)."""
    proposed_idx = [k for k, b in enumerate(bits) if b]
    proposed = [(problem.var_items[k], -problem.Q.get((k, k), 0.0)) for k in proposed_idx]

    proposed.sort(key=lambda t: -t[1])
    accepted: List = []
    used_idx = set()

    def as_pairs(item):
        return list(item) if problem.kind == "stem" else [item]

    for item, _w in proposed:
        cand_pairs = as_pairs(item)
        cand_idx = set()
        for i, j in cand_pairs:
            cand_idx.add(i)
            cand_idx.add(j)
        if cand_idx & used_idx:
            continue
        conflict = False
        for i, j in cand_pairs:
            for ai, aj in accepted:
                if pairs_cross((i, j), (ai, aj)):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        accepted.extend(cand_pairs)
        used_idx |= cand_idx

    accepted.sort()
    return accepted


def decode_bits_to_dotbracket(problem: QUBOProblem, bits: Sequence[int], seq: str) -> str:
    pairs = decode_bits_to_pairs(problem, bits, seq)
    return pairs_to_dotbracket(len(seq), pairs)
