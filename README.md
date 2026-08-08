

# *Qubit-Efficient Generative Circuit Search for RNA Secondary-Structure Prediction*


**A quantum and quantum-inspired pipeline for minimum-free-energy mRNA folding: Pauli correlation encoding to fit the problem on a near-term qubit budget, a transformer that writes the circuit, a Turner-calibrated model ladder that says how much of the remaining error is the model's fault, and nine degree-three jobs on real photonic hardware.**

<img src="assets/banner.png" alt="eQFold" width="100%" />


| Achraf Boussahi | Abir Chekroun |
|:---:|:---:|
| Team Lead & Quantum Expert | Quantum Expert / Commercial Lead |
| [@AchrafBoussahi](https://github.com/AshrafBoussahi) | @Abeer |

<img width="450" height="430" alt="6030600942694108684" src="https://github.com/user-attachments/assets/07006c58-632a-4755-8562-e3c4558e7753" />



[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ViennaRNA 2.7.2](https://img.shields.io/badge/ViennaRNA-2.7.2-0F766E.svg)](https://www.tbi.univie.ac.at/RNA/)
[![tests](https://img.shields.io/badge/tests-312%20passing-brightgreen.svg)](tests/)
[![hardware](https://img.shields.io/badge/hardware-QCi%20Dirac--3-B45309.svg)](results/raw/dirac3_hardware.csv)

Submitted to the **WISER Global Quantum+AI Program 2026**, Moderna challenge:
*Optimization of mRNA Secondary Structure Prediction Using Quantum Computing*.

> **We wrote this up as a research paper, not a notebook dump.**
> The full manuscript (two-column, with every number traced to a committed
> results file) and the presentation deck live here:
> **[eQFold: paper + slides (Google Drive)](https://drive.google.com/drive/folders/17YslGheC_w08kT98M2XuiZFR-W9BXEgD?usp=sharing)**

---

## Why? and for what?


An mRNA molecule folds back on itself the moment it is transcribed. Which bases
pair with which decides how stable the molecule is, how efficiently it is
translated, and how easy it is to manufacture, so secondary structure is a
design variable for an mRNA therapeutic rather than an afterthought.

The trouble is that the number of ways a sequence can fold explodes with its
length. ViennaRNA solves the nested case exactly in `O(L^3)`, and it is our
answer key throughout, never our competitor. The question this challenge asks is
different: **can the problem be posed so that a quantum device could search it,
and what would that actually cost in qubits, gates, and device time?**

Answering that honestly requires two numbers, not one. When a prediction
disagrees with ViennaRNA, either the model was wrong about the energy or the
solver failed to minimise the model. This repository never reports a single
combined number. Every result is split into an **encoding gap** (the model's
fault) and an **optimizer gap** (the solver's fault), and they add up exactly.

That split is what produced our main finding: **the encoding is the binding
constraint, not the circuit.** One step up our model ladder is worth about seven
times the entire spread of solver quality we measured on the same instances.

## Why us in this challenge?


**We are familiar with implementing Generative Quantum Eigensolvers (GQE) in the field of quantum chemistry for molecular simulations.** We were selected as finalists in the global industry challenge for the Mitsubishi Chemical & The National Institute of Advanced Industrial Science and Technology (AIST) track, 'Harnessing the Generative Quantum Eigensolver for Next-Generation Materials Design.' You can check out our other work under the AiQC research group by clicking on our [website page](https://www.aiqcommunity.org/)

**We had already built the hard part, for a different problem.** The Pauli
correlation encoding and generative circuit search used here come from
[qGridX](https://github.com/AshrafBoussahi/qGridX), our submission to a
Department of Energy grid-optimization challenge. We plugged a new problem into
unchanged solver code. That the same machinery carries an RNA folding instance
without modification is itself a result about how general the approach is.

**We went after the part of the physics everyone skips.** Across every quantum
RNA-folding paper we could obtain, the literal string "multiloop" appears zero
times in the formulation, and no published model includes interior-loop
energies. The field names the obstruction (Zaborniak et al.: multiloops are
"difficult to penalize directly" inside a QUBO) without characterising it. We
characterise it: the multiloop closing penalty is an *indicator* on branch
count, not a polynomial, so it is unrepresentable at any fixed degree, and
degree *d* buys junctions with up to *d* branches. We prove it, measure it, and
then find it a third time in a damping sweep that lands on exactly the value the
proof predicts.

**We ran it on real hardware, and we report what it did not do.** Nine
degree-three jobs on a QCi Dirac-3 entropy quantum computer, all accepted, 107 s
of device time. The device comfortably *hosts* a multiloop-accurate model at 78
encoded variables. It did **not** solve those instances well: a two-flip
classical hill climb nearly doubles F1 afterwards. Both columns are in the CSV,
side by side, and they are never merged.

**We report against ourselves.** PCE loses to its own classical counterpart
here. Level 2 is worse than Level 1 at four-way junctions. GQE's edge over
simulated annealing does not survive a switch to a more realistic scoring
function. All of that is in the paper, in a section that says so.

## The problem this solves

Given an RNA sequence over `{A, U, C, G}`, choose which bases pair.

Each candidate pair, or each candidate stem, becomes one binary variable. The
objective rewards stable pairings and penalises two selections that share a
nucleotide or cross each other:

```
H(x) = -sum_k w_k x_k  +  P * sum_{k<l conflicting} x_k x_l
```

Two things then have to be decided, and they are independent of each other.

**Where do the `w_k` come from?** A flat heuristic (G-C > A-U > G-U) is what a
weighted Nussinov objective uses, and it over-predicts pairing because it has no
loop entropy. Scoring each stem with ViennaRNA's own energy function instead
cuts the free-energy gap to the true structure by 68%. Going further, extracting
the Turner loop primitives directly and building a three-rung ladder of models,
cuts the mean energy error on multiloop sequences from 16.31 to 0.85 kcal/mol.
The top rung is **cubic**, because multiloop energy depends on branch count and
branch count is a three-body property of stem variables.

**How do you fit `m` variables on a device with far fewer qubits?** Pauli
correlation encoding reads each variable off the sign of one `k`-body Pauli
correlator, so `n` qubits carry `3*C(n,k)` variables. On the challenge's own
44-nt example that is 179 variables on 8 qubits, a 22-fold compression, and it
is a property of the encoding alone, independent of whatever searches it.

The circuit that prepares the state is not hand-designed. A decoder-only
transformer emits it gate by gate and is trained by reinforcement learning
against the decoded objective. Its circuits use **six times fewer gates** than a
fixed brickwork ansatz at equal or better score.

## Install

Python 3.10 to 3.12. ViennaRNA ships native wheels for Linux, macOS and Windows,
so nothing here needs a compiler.

```bash
pip install -e ".[dev]"
```

That covers the entire model-fidelity track: the ladder, the two-gap
decomposition, the penalty sweep, the enumeration ablation, the noise study, the
pseudoknot benchmark, and the Dirac-3 simulator. No credentials required.

Two optional extras:

```bash
pip install -e ".[hardware]"   # qci-client, for real Dirac-3 submission
pip install -e ".[circuits]"   # torch, for the GQE / PCE statevector simulator
```

The GQE and PCE arms additionally import `qms`, the solver package from our
[qGridX](https://github.com/AshrafBoussahi/qGridX) project, which supplies the
transformer trainer and the correlator readout. Place it on the import path
alongside this folder to run `make circuits`. Everything else runs without it.

## Sixty seconds to a fold

```bash
python -c "
from rnaqopt.config import STEMS, VIENNA
from rnaqopt.stems import enumerate_with_graphs
from rnaqopt.model import build_level2
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.decode import decode
from rnaqopt.reference import mfe, eval_structure

seq = 'GGGAAACCCAUAGGGAAACCCUAUGGGAAACCCAAAA'
graphs = enumerate_with_graphs(seq, cfg=STEMS)
model  = build_level2(seq, graphs, cfg=VIENNA)
best   = solve_exact(model)
priority = [model.full.terms.get((i,), 0.0) for i in range(graphs.n)]
pred   = decode(best.bitstring, graphs, len(seq), priority).structure
ref, e = mfe(seq, VIENNA)

print('predicted', pred, eval_structure(seq, pred, VIENNA))
print('ViennaRNA', ref,  e)
"
```

## Reproducing every reported result

```bash
make all          # sequences, reference table, every experiment, figures, tests
```

Or one target at a time. On Windows without GNU make, `./make.ps1 <target>`
exposes the same targets with the same behaviour.

| Target | What it rebuilds | Needs |
|---|---|---|
| `make sequences` | the four tier FASTA files, from one seed | |
| `make reference` | `data/references/vienna_reference.csv`, the answer key | |
| `make experiments` | the ladder, ablation, encoding gap, solver comparison, Dirac-3 simulator, advanced tasks | ~15 min, CPU |
| `make circuits` | the GQE / PCE scaling sweeps, hardware-aware demo, shot noise | `qms`, `torch`, several hours |
| `make figures` | every figure from the committed tables, never recomputing science | |
| `make test` | 312 tests | |
| `make verify` | asserts the reference table regenerates byte-for-byte | |

The one thing `make all` deliberately does not do is submit to hardware. That
needs an allocation and a token:

```bash
QCI_TOKEN=... QCI_API_URL=... python experiments/run_dirac3_hardware.py --tiers M
```

Three details of that client each cost a failed submission if you get them
wrong, and none of them is in the documentation: the variable is `QCI_TOKEN`
(not `QCI_API_TOKEN`), a degree-3 polynomial must be uploaded as a file and
referenced by `polynomial_file_id` rather than inlined, and `QCI_API_URL` has no
package default. Payloads are validated offline before any allocation is spent.

## The repository, module by module

```
eQFold/
  src/rnaqopt/            the model-fidelity and hardware track
    config.py             every ViennaRNA setting that can move a number, frozen
                          and hashed into a fingerprint stamped on each CSV
    sequences.py          deterministic generation, provenance enforcement
    reference.py          the ViennaRNA answer key
    stems.py              candidate stems + conflict / crossing / nesting graphs
    energy.py             Turner loop extraction; reproduces eval_structure
                          exactly on 60/60 reference structures
    model/                Levels 0, 1, 2 as one polynomial type whose degree is a
                          property; penalties; Rosenberg quadratization
    solvers/              exact, annealing, low-rank, ADAPT-QAOA, PCE, Dirac-3
    decode.py             bitstring to dot-bracket, with the repair reported
    postprocess.py        1-opt hill climb, always reported in its own column
    metrics.py            accuracy plus the encoding / optimizer gap split
    resources.py          qubits, depth, gates, ancillas
    noise.py              finite sampling, then depolarizing and readout error

  src/                    the generative circuit-search track
    rna_encoding.py       sequence to QUBO: candidates, weights, exact Nussinov
                          answer key, and a repair that always returns a valid fold
    vienna_utils.py       ViennaRNA wrappers and calibrated per-stem weights
    qubo_adapter.py       plugs the RNA problem into the reused GQE / PCE code
    solvers.py            GQE, PCE-direct, simulated annealing, tabu, blind
                          control, all at matched evaluation budget
    hardware_aware.py     GQE's gate vocabulary restricted to a linear chain

  experiments/            one script per reported table (see the Makefile)
  results/tables/         the model-track CSVs
  results/raw/            raw Dirac-3 device records
  results/*.csv           the circuit-track CSVs
  figures/                every generated figure
  docs/                   FORMULATION, ASSUMPTIONS, LIMITATIONS, LITERATURE,
                          RESOURCE_ANALYSIS
  tests/                  312 tests, CI on 3 OSes x 2 Python versions
```

## Data

**Every sequence here is synthetic and computer-generated.** No confidential
Moderna data, no patient or clinical data, no proprietary sequences, no
personally identifiable information, matching the challenge's data requirement.

This is enforced in code rather than by policy: a sequence record without an
explicit `source` field is rejected at construction, and a test asserts that
every committed sequence declares a synthetic or public origin.

| Set | n | Length | Mean MFE | Multiloops | Purpose |
|---|---|---|---|---|---|
| A | 20 | 15 to 25 nt | -3.48 | 0 | brute-forceable; encoding fidelity |
| B | 20 | 30 to 60 nt | -6.91 | 0 | main accuracy benchmark |
| C | 10 | 60 to 120 nt | -21.18 | 1 | scaling only |
| M | 10 | 34 to 58 nt | -15.61 | 10 | where Level 2 is testable |

Set M exists because of a measurement, not a preference. We swept 44 designed
junction geometries at 20 draws each and found **no multiloop below 31 nt at any
GC content**: the Turner closing penalty of about +9.3 kcal/mol cannot be repaid
by the stacking energy a shorter sequence has available. Our small-sequence set
therefore cannot contain the phenomenon Level 2 models, so a separate set had to
be built. Sequences whose true structure is completely unpaired are screened
out, because an empty reference is trivially matched and measures nothing.

Plus the 44-nt sequence given as a worked example in the challenge brief, used
as the flagship benchmark for the circuit-search track.

## What the results say

**PCE compresses, and the compression is the encoding's, not the circuit's.**
179 decision variables on 8 qubits on the 44-nt benchmark, a 22-fold reduction.
Extrapolating the capacity formula, our largest instance at 247 variables would
need 8 qubits, a 30.9-fold reduction.

**GQE writes smaller circuits.** 17.6 gates on average (range 13 to 25) against
108.8 for a fixed ansatz (range 51 to 222), at equal or better score. A paired
bootstrap over 13 instances gives GQE a mean score advantage of -1.65, 95% CI
[-3.00, -0.35]. This is a **resource** result and we never report it as a
solver-quality result.

**Restricting to real device connectivity is free.** GQE constrained to a
nearest-neighbor linear chain reaches the identical score (-23.5) and the
identical decoded structure on 3/3 seeds, at comparable gate count (23.7 against
25.0).

**The model, not the solver, is the constraint.** On set M:

| Level | Degree | Energy error | Encoding gap | Cost-layer CNOTs | Ancillas |
|---|---|---|---|---|---|
| 0 (published baseline) | 2 | **16.311** | 0.000 | 658 | 0 |
| 1 (+ Turner loops) | 2 | **1.461** | 0.111 | 916 | 0 |
| 2 (+ cubic multiloops) | **3** | **0.849** | 0.489 | 1805 | **36.9** |

Level 0 to Level 1 closes **91%** of the energy error, Level 1 to Level 2 a
further **42%**, and Level 1 is *exactly* correct on every representable small
reference. Read the two columns together: Level 0 scores a **perfect** encoding
gap of 0.000 while being wrong by 16.31 kcal/mol, because its errors happen not
to move the argmin. A one-column report would rank these models backwards.

**Degree 3 buys 3-way junctions and stops.** Split set M by junction order:

| Junction | n | Level 0 | Level 1 | **Level 2** |
|---|---|---|---|---|
| 3-way | 7 | 15.314 | 1.199 | **0.000** |
| 4-way | 2 | 19.800 | **2.380** | 3.820 |

A cubic coefficient that reproduces the Turner penalty at 2 branches
over-counts by 3x at 3 branches, so the penalty is an indicator and not a
polynomial. Three independent confirmations: the algebra, the table above, and a
damping sweep whose 4-way minimum sits at **d = 0.3333**, found by search over a
grid, which is exactly the 1/3 the over-counting argument predicts.

| damping d | 0.25 | **1/3** | 0.50 | 0.75 | 1.00 | 1.25 |
|---|---|---|---|---|---|---|
| 4-way error | 0.830 | **0.313** | 0.720 | 2.270 | 3.820 | 5.370 |

**Nine degree-3 jobs on real Dirac-3 hardware, all accepted.** 107 s of a 422 s
allocation, zero rejections, largest instance 78 encoded variables.

| Instance | Scheme | Vars | Raw gap | Raw F1 | +1-opt gap | +1-opt F1 | Device s |
|---|---|---|---|---|---|---|---|
| M_ml02 | global_slack | 18 | 4.80 | 0.500 | 3.60 | 0.957 | 5 |
| M_ml08 | global_slack | 23 | 6.40 | 0.400 | **0.00** | 0.857 | 7 |
| M_ml06 | global_slack | 31 | 12.60 | 0.500 | 6.90 | 0.929 | 7 |
| M_ml09 | global_slack | 33 | 11.11 | 0.455 | 4.43 | 0.970 | 7 |
| M_ml02 | per_stem | 34 | **2.80** | 0.737 | 2.40 | 0.957 | 7 |
| M_ml04 | global_slack | 35 | 7.40 | 0.556 | 2.70 | 0.960 | 8 |
| M_ml08 | per_stem | 44 | **4.80** | 0.609 | 2.70 | 0.815 | 9 |
| M_ml10 | global_slack | 62 | 18.21 | 0.480 | 12.80 | 0.690 | 19 |
| M_ml01 | per_stem | 78 | 10.20 | 0.000 | 3.00 | 0.455 | 38 |
| **mean** | | | **8.70** | **0.471** | **4.28** | **0.843** | |

What the hardware established: the degree-3 variable cap is **above 78**, twice
the 39 we had measured at degree 4, so the cap is strongly degree-dependent.
Device time scales as **t = 0.095 * n^1.28** seconds in encoded variables
(R2 = 0.84), so cost grows with encoded dimension rather than shot count. The
`per_stem` encoding wins **both** paired comparisons on identical instances, so
the capacity-versus-accuracy trade-off we had anticipated does not exist at
these sizes. And the degree-3 result reproduced on the device: encoding gap
**exactly 0.000 on every 3-way junction** and nonzero on every 4-way one.

What it did not establish, stated plainly: the device returned **0/9** locally
optimal samples. A hill climb using a median of **2 single-bit flips** nearly
doubles F1. Dirac-3 *hosts* this model comfortably. Nothing here is a
performance claim for the device.

**Pseudoknots are one config entry away.** Dropping the crossing penalty yields
pseudoknot-capable folding, a regime ViennaRNA cannot enter at all. On 10
designed H-type pseudoknots, sensitivity **0.311 to 0.900** and PPV **0.600 to
0.900**; 9 of the 10 references are recovered exactly.

**The penalty bound is safe and slightly over-strong.** Feasibility reaches
**100% at 0.90x** the principled bound and stays there, so the knee is *below*
the conservative default, which is the direction a default should err in.

| lambda / bound | 0.10 | 0.25 | 0.50 | 0.75 | **0.90** | 1.00+ |
|---|---|---|---|---|---|---|
| feasible | 64% | 71% | 79% | 93% | **100%** | 100% |

**Sampling noise dominates, and it is cheap to remove.** Over 14 instances x 4
shot counts x 6 noise settings:

| shots | 128 | 512 | **2048** | 8192 |
|---|---|---|---|---|
| success | 91.7% | 97.6% | **100%** | 100% |

Depolarizing noise at 5% and 20% and readout error at 2% moved success by at
most 1.8 points, because the cost Hamiltonian is diagonal: errors that
redistribute amplitude without changing which basis state is most probable do
not change the decoded structure. The median instance needs about **32 shots**
for 99% confidence.

**Sub-stems are the best fidelity-per-variable trade on offer.** On set M:

| Variant | Mean stems | Reference representable | Encoding gap |
|---|---|---|---|
| `L_min=3`, maximal | 16.1 | 80% | 1.580 |
| `L_min=3`, + sub-stems (**default**) | 35.2 | **100%** | **0.570** |
| `L_min=2`, maximal | 51.1 | 80% | 3.570 |
| `L_min=2`, + sub-stems | 110.6 | **100%** | 3.760 |

Sub-stems at `L_min=3` cut the encoding gap by 64% for 2.2x the variables, and
they are the default here. Lowering `L_min` to 2 goes the other way: triple the
variables, double the gap. Without sub-stems, part of what the "encoding gap"
measures is really a *representability* failure, which is not what the ladder is
meant to be measuring.

## Provenance

Every number above comes from a committed results file. Where a number is
uncertain or reconstructed, the data says so.

- **Frozen model details.** `ViennaRNA Turner2004 | T=37C | dangles=0 | noLP=1 |
  GU | minHP=3 | fp=73ff947c1696`. That fingerprint is stamped into every CSV,
  and two files with different fingerprints are not comparable. `dangles=0` is a
  deliberate departure from the ViennaRNA default: a stem variable cannot
  represent a dangling nucleotide, so keeping them would add a roughly constant
  penalty to every rung of the ladder. Each reference row also carries the stock
  `dangles=2, noLP=0` result so the numbers tie back to unconfigured ViennaRNA.
- **Byte-identical regeneration**, verified by a clean-clone test: fresh clone,
  fresh venv, delete outputs, rebuild, empty `git status`.
- **312 tests**, CI on 3 operating systems and 2 Python versions.
- **One process failure, recorded rather than hidden.** The first hardware CSV
  was destroyed by a later `--dry-run` writing to the same path. The committed
  rows are reconstructed from the console log, are marked
  `record_provenance=reconstructed_from_console_log`, and all 9 satisfy
  `total_gap = encoding_gap + optimizer_gap` exactly. Per-sample device energies
  and wall times from that run are lost. Preflight and device outputs now use
  disjoint paths.

Known limits are in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), the full
derivation including the degree proof is in
[`docs/FORMULATION.md`](docs/FORMULATION.md), and the literature review with
per-paper evidence grades is in [`docs/LITERATURE.md`](docs/LITERATURE.md).

We do not beat ViennaRNA and never tried to. It solves this exactly in `O(L^3)`.
No quantum advantage is demonstrated here, and our instance sizes could not show
one.

## Citing

```bibtex
@misc{boussahi2026eqfold,
  author = {Boussahi, Achraf and Chekroun, Abir},
  title  = {Qubit-Efficient Generative Circuit Search for
            {RNA} Secondary-Structure Prediction},
  year   = {2026},
  note   = {WISER Global Quantum+AI Program 2026, Moderna challenge}
}
```

## License

MIT. See [LICENSE](LICENSE).
