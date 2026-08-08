# Assumptions

A required deliverable (challenge section 1.5). Every choice here can change a
reported number. Each is stated with its justification and, where it matters,
its cost.

Status: written at the end of **Phase 1**. Assumptions introduced by later
phases (model ladder, solvers, Dirac-3 encoding) are appended as those phases
land.

---

## 1. ViennaRNA model details

All of these are frozen in `src/rnaqopt/config.py` as `VIENNA`, hashed into a
12-character fingerprint, and stamped into every generated results file. Two
result files with different fingerprints are not comparable, and
`tests/test_reference.py` fails if the committed reference table was produced
under a different fingerprint than the current configuration.

**Current fingerprint: `73ff947c1696`**
`ViennaRNA Turner2004 | T=37C | dangles=0 | noLP=1 | GU | minHP=3`

### 1.1 Temperature = 37 °C

Physiological temperature and the ViennaRNA default. Relevant to the mRNA
context of the challenge.

### 1.2 Dangling ends: `dangles=0` (**deviation from the ViennaRNA default**)

Plan section 4.1 permits either `dangles=2` (ViennaRNA default) or `dangles=0`,
and requires the choice to be justified here. **We use `dangles=0`.**

A dangling end is a stabilising contribution from an unpaired nucleotide
stacking onto an adjacent helix. Our optimization variables are *stems*: a
binary variable per candidate helix. The model has no variable, and no term,
that can represent a dangling nucleotide. Under `dangles=2` every reference
energy would therefore include a contribution that **no rung of the fidelity
ladder can ever reproduce** — a roughly constant offset added to the encoding
gap at Levels 0, 1 and 2 alike.

That offset would be pure noise with respect to the hypothesis under test. The
entire point of plan section 2.2 is to separate error sources; deliberately
importing a fourth, irreducible one into the measurement would work against it.
`dangles=0` removes it.

**Cost, stated plainly:** our reference MFE is not the number stock ViennaRNA
prints by default. To keep the results tied to something a reader can
reproduce with an unconfigured ViennaRNA, every row of `vienna_reference.csv`
also carries `stock_mfe_structure`, `stock_mfe_energy` and `stock_bp_distance`,
computed under `dangles=2, noLP=0`. The stock configuration is reported for
comparability only and is never used to fit or score a model.

### 1.3 No lonely pairs: `noLP=1`

An isolated, unstacked base pair cannot be expressed by a stem-based encoding
with `L_min = 3` at all. Allowing lonely pairs in the reference would add
another encoding penalty unrelated to the model ladder, for the same reason as
§1.2. `noLP=1` also matches common practice for structure prediction, since
lonely pairs are poorly determined experimentally.

### 1.4 Turner 2004 parameters

ViennaRNA 2.x compiles Turner 2004 in as its default. We additionally call
`RNA.params_load_RNA_Turner2004()` explicitly so the choice survives any change
of upstream default.

### 1.5 G–U wobble pairs included

G–U pairs are real and thermodynamically significant. `ViennaConfig(allow_gu=False)`
exists so the effect can be reported as an ablation.

### 1.6 Minimum hairpin loop = 3 nt

Standard steric constraint; also ViennaRNA's default.

### 1.7 Energies rounded to 2 decimal places

ViennaRNA quantises energies to 0.01 kcal/mol internally, but its Python
bindings return single-precision artifacts (`-1.2000000476837158` for what is
really `-1.20`). Rounding on the way out makes results files stable, diffable,
and byte-reproducible across platforms — which is what the Phase 1 exit
criterion requires.

---

## 2. Problem formulation

### 2.1 `L_min = 3`, no sub-stems

Plan section 4.2 default. `L_min = 2` explodes the variable count; `L_min = 4`
loses structures. Emitting truncated sub-stems as separate variables is off by
default. Both are documented ablations, not silent choices — `StemConfig`
exposes them.

**Consequence to keep in view:** with `noLP=1`, ViennaRNA's reference structures
may still contain 2-bp helices, which an `L_min = 3` model cannot express. That
is a genuine encoding gap and will be reported as one, not hidden.

### 2.2 Reported energy is always `eval_structure`

Plan section 2.2, corollary. Whatever a model or solver believes its objective
value to be, the number that appears in any results table is
`fc.eval_structure(dot_bracket)` under the frozen configuration. Internal model
energy is a diagnostic only, and appears in results as `optimizer_gap_model`,
clearly labelled as being in model units.

---

## 3. Benchmark corpus

### 3.1 All sequences are synthetic

Permitted explicitly by challenge section 1.9. See
`data/sequences/PROVENANCE.md` for the full data statement and for why public
database sequences (Rfam / RNA STRAND) are not yet included: recording an
accession requires actually fetching it, and inventing accession numbers in a
graded provenance document is not an option.

### 3.2 Acceptance screening selects for non-degeneracy only

Random sequences at these lengths often fold into nothing: 7 of the first 50
draws had a completely unpaired MFE. Those instances are trivially "solved" by
any model and measure nothing. Candidates are screened on a structural
predicate of the *instance* (helix count, multiloop presence) and never on
model or solver performance. The distinction is what keeps the benchmark
honest.

### 3.3 Tier M exists because multiloops are unreachable below 31 nt

**Measured, not assumed.** A sweep of 44 junction geometries × 20 draws found
no multiloop MFE at any length below 31 nt, at any GC content or stem design:
the Turner multiloop closing penalty (`a` ≈ +9.3 kcal/mol) cannot be repaid by
the stacking energy available in that little sequence.

This matters because Level 2's cubic terms exist precisely to model the
multiloop branch penalty. The plan states that "every claim about encoding
fidelity is established" on Tier A (15–25 nt). **For Level 2, that is not
possible** — Tier A cannot contain the phenomenon Level 2 models.

Tier M (10 designed junctions, 34–58 nt) is the answer. It is simultaneously:

- multiloop-bearing (10/10, verified against ViennaRNA's loop decomposition),
- brute-forceable (11–27 candidate stems, so exact ground truth for the
  optimizer gap remains computable), and
- small enough in variable count to plausibly fit Dirac-3 device limits.

Level 2's encoding-fidelity claim is therefore established on Tier M, and this
is stated wherever that claim is reported.

### 3.4 Tier A is too small to exercise a solver

Also measured. Under the default enumeration (`L_min = 3`, no sub-stems), Tier A
yields **1–5 candidate stems per sequence**. That is ample for measuring the
encoding gap, but it means:

- the optimizer gap on Tier A is identically zero for any competent solver, and
  carries no information;
- resource curves (qubit count, depth) measured on Tier A are degenerate;
- with so few competing structures, a model can rank them correctly even when
  its energies are badly wrong, so the *ladder* may look flat on Tier A even if
  the underlying energy model has genuinely improved.

Measured stem counts: Tier A 1–5, Tier M 11–27, Tier B 7–67, Tier C 23–247.
Optimizer-gap and resource claims are therefore made on Tiers M and B, not on
Tier A. (Enumeration itself is Phase 2; these counts come from a throwaway
Phase 1 probe used only to validate the corpus design.)

---

## 4. Hardware assumptions (Dirac-3)

Recorded now because they constrain the Phase 5 design. Sources: QCi error
responses measured by this team on a previous project (`Dirac-3 free-tier degree
limit`, July 2026).

- The device enforces **two independent limits, checked in sequence**: a device
  capability limit and a subscription entitlement limit.
- **Free tier: polynomial degree ≤ 3.** Our Level 2 model is degree 3 by
  construction, so it fits the free tier exactly. This is fortunate rather than
  planned, and it should be stated as such.
- The device variable cap is **degree-dependent**: measured at **39 variables
  for degree 4**, and **> 78 at degree 3** (measured 2026-08-07 — nine jobs
  submitted from 18 to 78 encoded variables, none rejected). The degree-3 cap is
  therefore at least twice the degree-4 cap and remains an upper bound only: the
  instance ladder ran out before the device did.
- The device optimizes over a **simplex**: `sum(x_i) = R`, `x_i ≥ 0`. A measured
  warning from the previous project: a cubic objective on this domain is
  generically unbounded below, so the minimiser can be driven to a simplex
  corner unrelated to the intended solution. The Level 2 objective must be
  checked for boundedness on the simplex before device time is spent.
- Allocation is **time-based**. Measured on our own run: **t = 0.095 · n^1.28
  seconds** in encoded variables (R² = 0.84 over 18–78 variables at 20 samples,
  schedule 1). The flat "~3 s per run" figure carried over from the previous
  project underpredicted our total by 3×, because cost scales with encoded
  dimension rather than shot count alone. Rejected submissions do not consume
  allocation.

### 4.1 What the hardware run changed

These were assumptions about a device we had not run *this* problem on. After
the 2026-08-07 run:

- **Confirmed:** degree ≤ 3 entitlement; time-based allocation; rejected jobs
  are free; the `per_stem` encoding outperforms `global_slack` (both paired
  comparisons, on identical instances).
- **Corrected:** the degree-3 variable cap is far larger than the degree-4 one,
  so the anticipated capacity-versus-accuracy trade-off does not exist at these
  sizes. Use `per_stem`.
- **Corrected:** the device-time model, as above.
- **New, and unwelcome:** raw device solution quality is poor (mean optimizer gap
  8.70 kcal/mol, mean F1 0.471, model optimum reached 0/9). A deterministic 1-opt
  classical hill climb using a median of 2 bit flips improves this to 4.28 and
  0.843. This is reported in `docs/LIMITATIONS.md` §8 rather than smoothed over.
