# Scaling and quantum-resource analysis

A required deliverable (challenge §1.5, task 6). All numbers are measured under
the frozen configuration `Turner2004 | T=37C | dangles=0 | noLP=1 | GU |
fp=73ff947c1696`, with `L_min = 3` and sub-stems enabled.

---

## 1. Problem size is |stems|, not sequence length

The x-axis of every scaling claim is the number of binary variables
`n = |stems|`. Measured across the 60-sequence corpus:

| set | length (nt) | variables n | median |
|---|---|---|---|
| A | 15–25 | 1–28 | 8.5 |
| M | 34–58 | 17–61 | 34 |
| B | 30–60 | 8–170 | 60.5 |
| C | 60–120 | 45–639 | 306.5 |

Empirical relationship: **n ~ L^1.9** (log–log fit over all 60 sequences). The
super-linear growth is why sequence length is a misleading axis — a 2× longer
sequence is a ~3.7× larger optimization problem.

Enumeration settings move this substantially (`enumeration_ablation.csv`):

| variant | mean n (set M) | representable | encoding gap |
|---|---|---|---|
| L≥3 maximal | 16.1 | 0.80 | 1.58 |
| **L≥3 + sub-stems** (default) | **35.2** | **1.00** | **0.57** |
| L≥2 maximal | 51.1 | 0.80 | 3.57 |
| L≥2 + sub-stems | 110.6 | 1.00 | 3.76 |

Sub-stems cost 2.2× the variables and buy full representability plus a 64 %
smaller encoding gap. `L_min = 2` costs 3× the variables and makes accuracy
*worse* — extra weak candidates give the model's own energy errors more ways to
express themselves.

## 2. Gate-based resources

Counted analytically from the gate sequence (`resources.py`), not by
transpilation. Conventions: a *k*-body Pauli-Z rotation costs `2(k−1)` CNOTs and
one RZ; depth is as-soon-as-possible scheduling on **all-to-all** connectivity,
which is the optimistic bound.

**Cost-layer CNOTs per model level** (mean, set M):

| level | degree | cost-layer CNOTs | cubic terms | ancillas to quadratize |
|---|---|---|---|---|
| 0 | 2 | 658 | 0 | 0 |
| 1 | 2 | 916 | 0 | 0 |
| **2** | **3** | **1805** | **222** | **36.9** |

Level 2 roughly **doubles** the two-qubit gate count of one cost layer relative
to Level 1. Every cubic term costs 4 CNOTs where a quadratic term costs 2.

### The headline co-design number

**A gate-based device needs ≈ 36.9 ancilla qubits on top of ≈ 32 problem
qubits — an overhead of ≈ 1.15 ancillas per variable — to reach Level 2
fidelity via Rosenberg quadratization. Dirac-3 supports degree 3 natively and
pays none of it.**

Ancillas are shared across cubic terms containing the same pair (a greedy cover),
so the overhead is the number of *distinct* pairs appearing in cubic terms, not
the number of cubic terms — 36.9 ancillas for 222 cubic terms. Reducing each
triple independently would cost ~6× more.

The reduction is verified faithful (minimising over ancillas reproduces the
original energy) on random models in `tests/`.

### ADAPT-QAOA depth growth

Per-layer, on Tier A instances: depth and CNOT count grow linearly in the number
of layers, with the per-layer cost set by the cost Hamiltonian's term count. A
typical 11-variable Level 1 instance reaches depth 286 and 670 CNOTs after 6
layers. The mixer contributes negligibly (≤ 2 CNOTs per layer); **essentially
all circuit cost is the cost layer**, i.e. it is set by the *model*, not the
algorithm.

This is the practical form of the co-design argument: on gate-based hardware,
model fidelity is paid for in circuit depth, and that payment recurs every
layer.

## 3. Qubit scaling: PCE vs direct encoding

Pauli Correlation Encoding maps *n* variables onto *k*-body Pauli correlators
over *m* qubits with `C(m,k)·3^k ≥ n`. At `k = 2`:

| n (variables) | direct encoding | PCE qubits | compression |
|---|---|---|---|
| 9 | 9 | 3 | 3.0× |
| 18 | 18 | 3 | 6.0× |
| 34 | 34 | 4 | 8.5× |
| 61 | 61 | 5 | 12.2× |
| 170 | 170 | 7 | 24.3× |
| 639 | 639 | 12 | 53.3× |

Compression grows with problem size: qubit count scales as O(√n) against O(n).
A Tier C instance that needs 639 qubits directly needs **12** under PCE.

**But see `docs/LIMITATIONS.md` §3**: PCE reached the model optimum in 32 % of
runs against 97 % for the classical low-rank relaxation at comparable rank. The
compression is real; the solution quality is not competitive here.

## 4. Dirac-3 resources

The device optimises degree ≤ 3 over `x ≥ 0, Σx = R`. Two encodings, measured:

| scheme | encoded vars | optimizer gap (M) | collapse rate (M) |
|---|---|---|---|
| `global_slack` | n + 1 (29.2) | 9.41 | 0.99 |
| `per_stem` | 2n (56.3) | **5.84** | **0.31** |

**Mechanism, not just outcome.** Our linear coefficients are stacking energies
and are negative; on a simplex a negative linear objective is minimised by
putting all mass on one coordinate. The minimiser runs to a vertex — "select one
stem with weight R" — which decodes to a nearly empty structure. Measured
`n_selected ≈ 1.0` for `global_slack` at every R. The per-stem scheme caps each
coordinate, so mass physically cannot concentrate, and collapse falls from 99 %
to 31 %.

**R is a prior on structure density.** Sweeping R/n from 0.1 to 1.0, the
optimizer gap rises monotonically (2.52 → 3.15) and collapse rises 0.81 → 0.99.
Tighter (smaller R) is better.

### Device limits

Measured against the QCi API on a previous project:

- **free tier: degree ≤ 3.** Our Level 2 model is degree 3 by construction, so
  it fits the free tier exactly. Fortunate rather than planned.
- **device variable cap is degree-dependent**: 39 variables at degree 4. **The
  degree-3 cap has not been measured.**
- allocation is time-based (~3 s per run at 20 samples); rejected submissions do
  not consume allocation.

Against a 39-variable reference point, set M under `per_stem` (56.3 encoded
variables) would not fit, while `global_slack` (29.2) would. **The better
encoding is the one more likely to be rejected.** Resolving this is the first
task with device access.

## 5. Classical solver costs, matched budget

Under an identical wall-clock budget per instance (n ≤ 16–18):

| solver | hit optimum | optimizer gap | F1 | wall time |
|---|---|---|---|---|
| simulated annealing | **1.00** | 0.000 | 0.875 | 1.78 s |
| **random sampling** | **1.00** | 0.000 | 0.875 | 2.00 s |
| low-rank (Burer–Monteiro) | 0.97 | 0.012 | 0.871 | 0.05 s |
| ADAPT-QAOA | 0.84 | 0.653 | 0.821 | 0.75 s |
| Dirac-3 simplex (simulated) | 0.66 | 0.603 | 0.746 | 0.10 s |
| PCE | 0.34 | 2.162 | 0.312 | 1.16 s |

**Uniform random sampling hits the model optimum 100 % of the time.** These
instances do not discriminate between optimizers at all; see
`docs/LIMITATIONS.md` §2. Note also that the encoding gap (0.275) exceeds the
optimizer gap of every method except PCE — the model, not the solver, is what
limits accuracy here.

## 6. Sampling cost under noise

Finite sampling dominates at these sizes. ADAPT-QAOA's per-shot success
probability ranged from 0.003 to 0.152 on Tier A; `shots_for_target_success`
converts that to the repetitions needed for 99 % confidence of seeing an
optimum at least once (≈ 1 500 shots at p = 0.003).

Measured degradation with 512+ shots is negligible even at 20 % global
depolarizing — again because the instances are easy, not because the method is
robust.

## 7. Where this breaks

| resource | ceiling reached at | why |
|---|---|---|
| exact statevector simulation | n ≈ 20–24 | 2ⁿ amplitudes |
| brute-force ground truth | n ≈ 22 | 2ⁿ enumeration |
| CP-SAT ground truth | n ≈ 60–100 | reified cubic terms |
| Dirac-3 (measured, degree 4) | 39 variables | device limit |
| PCE simulation | m ≈ 14 qubits → n ≈ 1600 | 2^m statevector |

The binding constraint for accuracy claims is **exact ground truth**, not
quantum simulation: without a proven model optimum the encoding and optimizer
gaps cannot be separated, and separating them is the point of the whole project.
