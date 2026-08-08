"""Thin wrappers around the ViennaRNA Python package used as the classical
thermodynamic reference throughout this project (MFE structures, structure
energies, base-pair distance) and as an empirical source of per-stem energy
contributions for calibrating the QUBO objective (see rna_encoding.stem_weight
vs. `vienna_stem_weights` below).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import RNA

Pair = Tuple[int, int]


def mfe_structure(seq: str) -> Tuple[str, float]:
    seq = seq.upper().replace("T", "U")
    structure, mfe = RNA.fold(seq)
    return structure, float(mfe)


def eval_structure_energy(seq: str, dotbracket: str) -> float:
    seq = seq.upper().replace("T", "U")
    fc = RNA.fold_compound(seq)
    return float(fc.eval_structure(dotbracket))


def base_pair_distance(db_a: str, db_b: str) -> int:
    return int(RNA.bp_distance(db_a, db_b))


def dotbracket_to_pairs(db: str) -> List[Pair]:
    stack: List[int] = []
    pairs: List[Pair] = []
    for k, c in enumerate(db):
        if c == "(":
            stack.append(k)
        elif c == ")":
            i = stack.pop()
            pairs.append((i, k))
    pairs.sort()
    return pairs


def pairs_to_set(pairs: Sequence[Pair]) -> set:
    return {tuple(sorted(p)) for p in pairs}


def base_pair_f1(pred_pairs: Sequence[Pair], ref_pairs: Sequence[Pair]) -> Dict[str, float]:
    pred = pairs_to_set(pred_pairs)
    ref = pairs_to_set(ref_pairs)
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
            "n_pred": len(pred), "n_ref": len(ref)}


def vienna_stem_weight(seq: str, stem: Sequence[Pair]) -> float:
    """Empirical stabilizing free energy (kcal/mol, positive = stabilizing)
    of a single isolated stem, i.e. -eval_structure(structure containing
    only this stem's pairs, everything else unpaired). This folds the real
    Turner nearest-neighbor model (hairpin loop penalty + base-pair stacking)
    into the QUBO weight for that stem, in place of the flat GC/AU/GU
    surrogate score. It ignores multi-loop and coaxial-stacking terms that
    only appear once several stems combine, which is the same first-order
    per-loop decomposition classical/quantum RNA-folding QUBO formulations
    rely on.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    db = ["."] * n
    for i, j in stem:
        db[i] = "("
        db[j] = ")"
    e = eval_structure_energy(seq, "".join(db))
    return -e


def vienna_stem_weights(seq: str, stems: Sequence[Sequence[Pair]]) -> List[float]:
    return [vienna_stem_weight(seq, s) for s in stems]
