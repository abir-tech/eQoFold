# Limitations

A required deliverable (challenge §1.5). Everything here is measured, not
speculated. Where a limitation is fundamental rather than an implementation
gap, that is stated.

---

## 1. The headline limitation

**ViennaRNA solves non-pseudoknotted MFE folding exactly, in O(n³), by dynamic
programming.** Nothing in this repository beats it on speed or accuracy for the
sequences in scope, and nothing here claims to. Our contribution is formulation
fidelity, encoding efficiency and honest resource characterization.

## 2. No quantum advantage is demonstrated, and the instances cannot show one

At the sizes reachable by exact statevector simulation (n ≤ ~16 stem variables),
**uniform random sampling finds the model optimum 100 % of the time** under a
2-second budget. Simulated annealing likewise.

An instance where random guessing always succeeds cannot discriminate between
optimizers. Every solver comparison in this report should be read as a
*correctness and resource* study, not as evidence about optimizer power. The
honest summary is that at the sizes we can simulate, the optimizer gap is
mostly zero for competent methods and the **encoding gap dominates the total
error**.

Reaching sizes where the optimizer gap matters means Tier B/C (60–639
variables), which exceeds both exact statevector simulation and the measured
Dirac-3 variable limits.

## 3. PCE loses to its classical counterpart here

Plan §2.4 requires we raise this ourselves rather than be caught by it.

Pauli Correlation Encoding achieves genuine qubit compression — *n* variables in
O(√n) qubits — but on this problem it reached the model optimum in **34 %** of
runs against **97 %** for the classical Burer–Monteiro low-rank relaxation at
comparable rank, under a matched budget. Its mean optimizer gap was 2.16
kcal/mol against 0.01, and its F1 was 0.31 against 0.87.

The honest reading: **on this problem class the quantum encoding buys qubit
count, not solution quality**, and the classical low-rank relaxation is the
better method. That is consistent with the active argument in the literature
that PCE-style relaxations are dequantizable. We report the compression as a
resource result and make no solution-quality claim for it.

## 4. Degree 3 cannot represent multiloops beyond 3-way junctions

Proved in `docs/FORMULATION.md` §5.2 and confirmed by measurement: the multiloop
closing penalty is an "at least 2 branches" indicator, and no degree-3
polynomial in stem variables reproduces it for arbitrary branch count.

Consequence: Level 2 is **exact for 3-way junctions (error 0.000)** and
**over-corrects for 4-way junctions (3.82 vs Level 1's 2.38)**. The
`branch_damping` parameter trades one against the other; no value is good for
both. Structures with 5-way or larger junctions are worse still.

This is a property of the hardware's degree ceiling, not of our implementation.

## 5. Multiloops are unreachable below 31 nt

Measured over 44 junction geometries × 20 draws: **no configuration below 31 nt
ever produced a multiloop MFE**, at any GC content or stem design. The Turner
closing penalty (a ≈ +9.3 kcal/mol) cannot be repaid by the stacking energy
available.

Consequence: the plan's assumption that "every claim about encoding fidelity is
established on Tier A (15–25 nt)" **does not hold for Level 2** — Tier A cannot
contain the phenomenon Level 2 models. Set M exists for exactly this reason,
and all Level 2 fidelity claims are made there.

## 6. Tier A is too small to exercise a solver

Under the default enumeration, Tier A yields 1–28 variables (median 8.5). That
is ample for encoding-gap work but means optimizer-gap and resource curves
measured on Tier A alone are degenerate. Those claims are made on sets M and B.

## 7. The encoding gap is not a sufficient statistic

The encoding gap — ViennaRNA energy of the decoded model optimum minus the MFE
— is **insensitive to energy errors that do not change the argmin**, and can
move the *wrong way*.

Concretely, on set M: Level 0 scores a perfect encoding gap of 0.000 while
carrying a 15–20 kcal/mol energy error, and Level 1 scores *worse* (0.111)
despite a 10× better energy function. Reading the encoding gap alone would rank
Level 0 above Level 1, which is backwards.

Every table therefore reports `model_energy_error` alongside. Readers of any
single-column version of these results will draw wrong conclusions.

## 8. Dirac-3 hosts the model but does not solve it well

**Superseded in part:** nine degree-3 jobs ran on real hardware on 2026-08-07
(`results/raw/dirac3_hardware.csv`). The device *capability* claim is confirmed.
The device *performance* is the limitation.

Raw device output, over 9 jobs at 18–78 encoded variables:

| | raw device | + classical 1-opt |
|---|---|---|
| mean optimizer gap | 8.70 kcal/mol | **4.28** |
| mean F1 | 0.471 | **0.843** |
| reached model optimum | **0 / 9** | 1 / 9 |

The local search is deterministic, uses a **median of 2 single-bit flips**, and
runs in microseconds. That a trivial classical post-process nearly doubles F1
means the device samples were far from locally optimal — it was not finding good
solutions, and cheap classical work recovered most of the loss.

Read plainly: **Dirac-3 comfortably hosts a degree-3 multiloop-accurate model,
which is the co-design claim and now has hardware backing. It did not solve
these instances well, and nothing here supports a performance claim for it.**

Note also that simulator and hardware numbers are not interchangeable. Results
labelled `dirac3_sim` come from the classical simplex simulator of the same
continuous programme; only `results/raw/dirac3_hardware*.csv` is device data.

**Still open:** the degree-3 variable cap is only bounded below (> 78) — every
rung was accepted, so the ladder exhausted our instances rather than the
device's capacity. Larger Tier M instances would be needed to find it.

**Process failure worth recording:** the first hardware run's full CSV was
destroyed by a subsequent `--dry-run` writing to the same path. The committed
row set was reconstructed from console output and carries
`record_provenance=reconstructed_from_console_log`; structures, per-sample
device energies and wall times from that run are permanently lost. It
reconstructs consistently — all 9 rows satisfy `total_gap = encoding_gap +
optimizer_gap` exactly — but it is not a full capture. Preflight and device
outputs now use disjoint paths and device runs archive under a timestamp.

## 9. ~~The better Dirac-3 encoding may not fit the device~~ — RESOLVED

Measured in simulation: the per-stem complementary-slack encoding halves the
optimizer gap (5.84 vs 9.41 on set M) and cuts corner-collapse from 99 % to
31 %, because capping each coordinate structurally prevents mass concentrating
on one stem. But it costs **2× the variables** (56.3 vs 29.2 on set M), and
against the degree-4 device limit of 39 the better encoding looked like the one
that might not fit.

**The hardware run settled it.** Both encodings were accepted at every size
tried, up to 78 encoded variables, and `per_stem` won both paired comparisons on
identical instances:

| instance | `global_slack` | `per_stem` |
|---|---|---|
| M_ml02 | 18 vars, gap 4.80 | 34 vars, gap **2.80** |
| M_ml08 | 23 vars, gap 6.40 | 44 vars, gap **4.80** |

The predicted trade-off does not exist at these sizes: `per_stem` is both better
and affordable. Use it. The caveat is that this only establishes the trade-off
is absent *below 78 variables* — the cap is still only bounded below (§8).

## 10. Sequences are entirely synthetic

Permitted by challenge §1.9, but it means results are not validated against
experimentally determined structures. Public database sequences (Rfam, RNA
STRAND) are not included: recording an accession requires actually fetching it,
and inventing accession numbers in a graded provenance document is not an
option. See `data/sequences/PROVENANCE.md`.

The designed sequences are also *easy* in a specific way — set M's junctions use
strong GC stems and A-rich loops, which is why Level 0's argmin is accidentally
correct there. Random natural sequences would likely be less forgiving.

## 11. Model-detail deviations from stock ViennaRNA

`dangles=0` and `noLP=1` are deliberate (see `docs/ASSUMPTIONS.md` §1.2–1.3),
but they mean our reference MFE is not the number stock ViennaRNA prints. The
stock values are carried in every reference row (`stock_mfe_energy`,
`stock_bp_distance`) for comparability, but no result here is validated against
the stock configuration.

## 12. Noise model is coarse

The noise study applies **finite sampling**, a **global depolarizing channel**
and **independent readout error** at the readout stage. It does not model gate-
level error propagation, crosstalk, coherent error, or any device-specific
calibration. Since our cost Hamiltonian is diagonal only the measurement
distribution matters, which makes a global channel defensible for the leading
effect — but it is not a hardware-faithful simulation.

At the sizes studied, finite sampling dominates: 512 shots suffice to reach the
optimum on nearly every instance even at 20 % depolarizing, because the
instances are easy (§2).

## 13. Scaling claims are extrapolations beyond ~64 variables

Exact ground truth is available only where brute force or CP-SAT closes. Above
that the optimizer gap cannot be computed at all, so Tier C is reported for
*variable counts, qubit counts and timing only* — never accuracy. Any statement
about behaviour at Tier C sizes is an extrapolation and is labelled as one.
