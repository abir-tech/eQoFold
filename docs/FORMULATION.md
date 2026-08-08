# Formulation

Full derivation of the model ladder. Every coefficient is extracted from
ViennaRNA's own loop-energy primitives, so a model term and the reference energy
cannot drift apart.

---

## 1. Variables

A **stem** is a run of consecutive complementary base pairs between positions
`i..i+L-1` and `j-L+1..j`. Its outermost pair is `(i, j)` and its innermost is
`(i+L-1, j-L+1)`; the innermost pair encloses the loop.

One binary variable per candidate stem: `x_s = 1` iff stem *s* is selected.

**Enumeration** (`stems.py`): maximal stems of length ≥ `L_min = 3`, plus
truncated sub-stems, with a minimum hairpin of 3 unpaired nucleotides and G–U
wobble allowed. Both knobs are measured ablations, not assumptions — see
`results/tables/enumeration_ablation.csv` and `docs/ASSUMPTIONS.md` §2.1.

`|stems|` is the problem size *n* and the x-axis of every scaling plot. It is
**not** sequence length: measured `n ~ L^1.9` across the corpus.

Three relations are precomputed over the candidate set, and they are mutually
exclusive by construction so no physical impossibility is penalised twice:

| relation | meaning |
|---|---|
| `conflict` | two stems share a nucleotide → mutually exclusive |
| `crossing` | their pairs cross → pseudoknot |
| `nesting` | *t* lies inside the loop closed by *s* |

## 2. Energy primitives

From ViennaRNA under the frozen configuration, all in kcal/mol:

| primitive | meaning |
|---|---|
| `stack(s)` | Σ stacking energies inside stem *s* (negative) |
| `hairpin(s)` | hairpin loop closed by *s*'s inner pair |
| `exterior(s)` | exterior-loop contribution; under `dangles=0` this is exactly the terminal AU/GU penalty |
| `interior(s,t)` | interior/bulge/stack loop between *s* and a nested *t* |

**The correctness anchor.** For any pseudoknot-free structure, summing these
over its loop tree reproduces `eval_structure` exactly. `decompose_structure`
does this, and the identity is asserted on all 60 reference structures in
`tests/test_energy.py`. That is what licenses using these terms as model
coefficients.

Turner multiloop energy, verified exact against ViennaRNA's own loop
decomposition on all 11 multiloops in the corpus:

```
E_ML = a + c*(branches + 1) + b*(unpaired)
       + terminal(closing) + Σ terminal(branch)
```

with `a = 9.30`, `c = -0.90`, **`b = 0.00`** (Turner 2004). Because `b` is zero,
the unpaired-nucleotide dependence vanishes entirely and the multiloop energy
depends only on **branch count** — which is precisely the structure that becomes
three-body in stem variables.

## 3. Level 0 — linear + hard-constraint penalties

```
H0 = Σ_s stack(s)·x_s
   + λ_conflict · Σ_(s,t)∈conflict x_s x_t
   + λ_cross    · Σ_(s,t)∈crossing x_s x_t     [omitted in pseudoknot mode]
```

Degree 2. This is the rung most published QUBO formulations of RNA folding sit
on, and it is deliberately the worst model here: every loop in the structure is
free. Since loop penalties are large and positive, Level 0 systematically
over-predicts pairing.

**Measured:** mean energy error **5.33 kcal/mol** on set A, **16.31 kcal/mol**
on set M.

> Level 0 scores a *perfect* 0.000 encoding gap on set M while carrying a
> 15–20 kcal/mol energy error. On designed sequences its argmin is right by
> accident. This is the clearest evidence that the encoding gap alone is not a
> sufficient statistic, and why both columns are reported everywhere.

## 4. Level 1 — quadratic loop energies

Linear coefficient — the cost of a stem standing alone on the exterior loop,
closing a hairpin:

```
L(s) = stack(s) + hairpin(s) + exterior(s)
```

Quadratic correction when *t* is nested directly inside *s* — the hairpin of *s*
is replaced by the interior loop, and *t* leaves the exterior loop:

```
Q(s,t) = interior(s,t) − hairpin(s) − exterior(t)
```

These telescope:

| structure | Level 1 total | exact? |
|---|---|---|
| *s* alone | `L(s)` | ✅ |
| *s* ⊃ *t* | `L(s)+L(t)+Q(s,t)` | ✅ |
| *s* ⊃ *t* ⊃ *u* | `… + Q(t,u)` **but `Q(s,u)` also fires** | ❌ |
| *s* ⊃ {*t*, *u*} | hairpin(s) subtracted twice, no `a` charged | ❌ |

Both failures are three-body. Degree 2.

**Measured:** energy error **0.000 kcal/mol** on every representable set-A
reference — Level 1 is not merely better, it is *exact* for the structure class
it is designed to represent. On set M it drops to 1.46, with the residual
concentrated on multiloops.

## 5. Level 2 — cubic terms, native on Dirac-3

### 5.1 Deep chains — exactly representable

For a chain *s* ⊃ *u* ⊃ *t*, the spurious Level 1 term is removed on the triple:

```
C_chain(s,u,t) = −Q(s,t)
```

Exact.

### 5.2 Multiloops — exact for 3-way junctions, and provably no further

Let *s* enclose *k* directly-nested branches. Level 1 supplies
`(1−k)·hairpin(s) + Σ interior(s,t_i)`; Turner requires `E_ML(k)`. The
difference separates cleanly:

```
Δ(k) = K(s) + Σ_i P(s, t_i)

K(s)   = a + c + terminal(s_inner) − hairpin(s)          fires once, iff k ≥ 2
P(s,t) = c + terminal(t_outer) − interior(s,t) + hairpin(s)   per branch
```

`P` is pairwise and `K` is a constant — but `K` must fire **exactly once when
k ≥ 2 and never when k ≤ 1**. That indicator is the obstruction.

> **Impossibility.** Suppose the branch-count dependence were carried by a
> linear term `α·k` plus a cubic term `β·C(k,2)`. Exactness demands `f(1)=0`,
> `f(2)=K`, `f(3)=K`. The first two give `α=0`, `β=K`; then `f(3)=3K ≠ K`.
> **No degree-3 polynomial in stem variables reproduces the multiloop closing
> penalty for arbitrary branch count.** Degree 3 buys exactly the 3-way
> junction; a 4-way junction needs degree 4, and so on.

This is a genuine, quantitative answer to *what does it cost to represent RNA
folding physics faithfully on optimization hardware* — and it is reported rather
than hidden.

**Measured, exactly as the algebra predicts:**

| branches | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| 2 (3-way junction) | 15.31 | 1.20 | **0.000** |
| 3 (4-way junction) | 19.80 | 2.38 | 3.82 |

The `branch_damping` sweep traces the obstruction directly: damping 1.0 is exact
at `k=2`; the `k=3` error is minimised at damping **1/3** (0.313 kcal/mol) —
precisely the value that makes three firing triples charge `K` once. No single
value is good for both.

### 5.3 No coaxial-stacking term

Plan §4.3 lists coaxial stacking as a candidate cubic term. Under the project's
frozen `dangles=0`, **ViennaRNA does not apply coaxial stacking at all** —
verified by the loop-tree decomposition reproducing `eval_structure` exactly on
all 60 references without one. Adding the term would move the model *away* from
the reference it is scored against.

## 6. Penalty calibration

`λ` must exceed the largest energy decrease any single variable can buy. The
plan states this as `λ > max_s |E_stack(s)|`; that is slightly too weak once
loop terms exist, since a stem also unlocks stabilising quadratic and cubic
terms. The bound used is therefore, per variable, the sum of **all** negative
coefficients of every term containing it.

**Measured:** feasibility reaches 1.00 at **0.90×** that bound, where the
penalised optimum coincides exactly with the hard-constrained optimum
(gap 0.0000). Below it the penalised optimum is infeasible and scores
artificially low — a *negative* apparent gap of up to −2.06 kcal/mol. The bound
is tight to within 10 %.

## 7. Decoding

1. read selected stems from the bitstring;
2. check pairwise validity (no shared nucleotides; no crossings unless in
   pseudoknot mode);
3. if invalid, apply deterministic greedy repair — sort by model linear
   coefficient, accept greedily while valid, ties broken on stem index;
4. emit dot-bracket, extended alphabet `.()[]{}` in pseudoknot mode;
5. score with `eval_structure`.

**Repair is always reported separately** (`feasible_raw`, `was_repaired`,
`n_removed`). Folding it silently into results would inflate apparent solver
quality.

Note that with sub-stems the encoding is **redundant**: a long helix can be
tiled by one stem or by several stacked sub-stems. Level 1 assigns both the same
energy — a useful consistency check on the nesting correction, pinned in
`tests/test_models.py::test_substem_tiling_is_energy_consistent`.

## 8. Pseudoknot mode

Setting `pseudoknot_mode=True` drops the `λ_cross` term. One configuration
change; nothing else in the pipeline differs.

**Measured on designed H-type pseudoknots:** base-pair sensitivity **0.311 →
0.900**, PPV **0.60 → 0.90**, and 90 % of recovered structures genuinely contain
crossing pairs (0 % without it).

*Scope note:* no Turner energy is reported for a pseudoknotted structure.
`eval_structure` cannot score crossing pairs and no Turner-parameterised
pseudoknot model is implemented here, so pseudoknot results are base-pair
recovery against the designed structure only, never energies.
