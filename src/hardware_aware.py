"""Hardware-aware GQE: a connectivity-constrained circuit vocabulary, used
only as a simulated design study (no real hardware in this project). Real
superconducting qubits are wired in a fixed, sparse topology, not
all-to-all, so a circuit generator that assumes it can place a two-qubit
gate between any pair of qubits is not directly deployable. This module
builds a vocabulary restricted to a nearest-neighbor chain, the same
linear-chain-inside-one-chiplet convention used for the Rigetti
Cepheus-1-108Q device in the sister power-grid study (eight of a
nine-qubit chiplet's qubits, connected in a line), so the resulting
circuits are, in principle, directly placeable on that real device's
connectivity graph, even though no job is ever submitted to it here.

This is a pure post-hoc filter of an ordinary qms.gqe.vocab.GQEVocab: it
does not modify that module at all. GQEVocab builds every single-qubit
token first, then every two-qubit token for every pair of qubits; this
module simply removes the two-qubit tokens whose qubit pair is not an
edge of the target connectivity graph, and rebuilds the lookup table the
same way GQEVocab's own constructor does, so every downstream consumer
(the executor, the training loop, the reward function) works unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qms.gqe.vocab import GQEVocab  # noqa: E402


def linear_chain_edges(n: int) -> list[tuple[int, int]]:
    """Nearest-neighbor chain 0-1-2-...-(n-1), the simplest realistic
    superconducting-qubit connectivity and the one already used for the
    n=8 chain inside one Rigetti Cepheus-1-108Q chiplet in the sister
    study (a 3x3 sub-lattice admits a Hamiltonian path through 8 of its 9
    qubits)."""
    return [(q, q + 1) for q in range(n - 1)]


def build_constrained_vocab(n: int, allowed_pairs, angle_grid=None, max_len=None) -> GQEVocab:
    """Returns a GQEVocab whose two-qubit gate tokens are restricted to
    `allowed_pairs` (each an (i, j) tuple). Single-qubit tokens, BOS/EOS,
    and every other attribute are untouched, so this vocab is a drop-in
    replacement anywhere a normal GQEVocab is used (execute_batch,
    build_unitary_table, train_regime_a, sequence_gate_count, ...)."""
    vocab = GQEVocab(n, angle_grid=angle_grid, max_len=max_len)
    allowed = {tuple(sorted(p)) for p in allowed_pairs}

    kept = [t for t in vocab.tokens if t.kind != "2q" or tuple(sorted(t.qubits)) in allowed]
    vocab.tokens = kept
    vocab.vocab_size = len(kept)

    reverse = {}
    for tid, t in enumerate(kept):
        if t.kind == "special":
            key = ("special", t.name)
        else:
            key = (t.kind, t.axis, t.qubits, round(t.angle, 10))
        reverse[key] = tid
    vocab._reverse = reverse

    assert vocab.tokens[vocab.BOS_ID].kind == "special" and vocab.tokens[vocab.BOS_ID].name == "BOS"
    assert vocab.tokens[vocab.EOS_ID].kind == "special" and vocab.tokens[vocab.EOS_ID].name == "EOS"
    return vocab


def build_linear_chain_vocab(n: int, angle_grid=None, max_len=None) -> GQEVocab:
    return build_constrained_vocab(n, linear_chain_edges(n), angle_grid=angle_grid, max_len=max_len)
