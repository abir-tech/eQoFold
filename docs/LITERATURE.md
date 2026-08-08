# Literature review: quantum and quantum-inspired RNA secondary structure prediction

Searched 2026-08-07. This document exists to answer one question honestly:
**which parts of this project are new, and which parts reproduce published
work?**

## How to read the evidence grades

Not every claim below rests on the same quality of evidence. Each paper is
tagged:

- **[full text]** — PDF downloaded and read; term counts and quotes are from the
  actual document.
- **[abstract]** — only the abstract or a search summary was accessible
  (paywall). Claims are correspondingly weaker and flagged inline.

Where a paper is graded *[abstract]*, do not cite its internals in the write-up
without obtaining the full text first.

---

## 1. The landscape

| Work | Team | Variables | Energy terms | Degree | Grade |
|---|---|---|---|---|---|
| Fox et al. 2021/22, *RNA folding using quantum computers* (PLOS Comput Biol) | NIH / Leidos | **stems** | stem **length** reward (not Turner energies) | 2 | [abstract] |
| Zaborniak et al. 2022, arXiv:2208.04367 | U. Victoria | **stems** | NN stacking + **hairpin** penalty + pseudoknot heuristic | 2 | [full text] |
| Alevras, Metkar et al. 2024, arXiv:2405.20328 (IEEE QCE) | **IBM + Moderna** | **quartets** | stacking only | 2 | [full text] |
| Kumar, Alevras, Metkar et al. 2025, arXiv:2505.05782 | **IBM + Moderna** | quartets | stacking only | 2 | [full text] |
| Friedhoff, Metkar, Davis et al. 2026, arXiv:2605.20163 | **IBM + Moderna** | stems | stacking | 2 | [full text] |
| *Exploring the Boundaries of Modern Quantum Annealers with RNA Structure Prediction*, JPCB 2025 | — | — | — | 2 | [abstract] |

**Measured term counts** in the full-text papers (literal string frequency
across the whole PDF):

| | multiloop | multi-loop | hairpin | interior loop | cubic | HUBO/HOBO |
|---|---|---|---|---|---|---|
| Alevras 2024 | 0 | 0 | 1 (figure caption only) | 0 | 0 | 0 |
| Kumar 2025 | 0 | 0 | 0 | 0 | 0 | 0 |
| Friedhoff 2026 (PCE) | 0 | 0 | 0 | 0 | 0 | 0 |
| Zaborniak 2022 | 0 | 5 | 10 | 0 (11× "internal loop") | 1 | 0 |

**Not one published quantum RNA-folding formulation includes interior-loop or
multiloop energies.** The most physically detailed of them (Zaborniak) reaches
stacking plus a hairpin penalty — which is between our Level 0 and Level 1.

---

## 2. What the field says about the obstruction

This is the most important finding for our positioning. The multiloop problem is
**named as an open difficulty** in the literature, without being characterised.

Zaborniak et al. 2022, figure 2 caption *[full text]*:

> "RNA structural motifs that are **difficult to penalize directly** within an
> RNA folding QUBO model. (a) bulges … (b) internal loops … and (c) multi-loops
> are internal loops with three or more participant stems."

And in the body, on why they use a heuristic instead:

> "bulges, internal loops, multi-loops, and other such structural motifs are
> known to be destabilizing, **features not feasibly pre-computable given that
> we must enumerate every possibility in advance prior to submission to a
> QUBO**."

Their workaround is not an energy model at all — it is a tunable heuristic
favouring long continuous helices.

A later statement, reported for JPCB 2025 *[abstract — verify before citing]*:

> "The QUBO formalism's restriction to one- and two-body terms prevents
> inclusion of essential constraints, and the framework in its current form is
> **structurally incapable** of accurately capturing RNA folding at biologically
> relevant scales."

So the field has established, qualitatively, that degree 2 is not enough. What
is **absent** from every source found:

- any statement of *which degree would suffice*,
- any derivation of *why* degree 2 fails,
- any HUBO/HOBO formulation of RNA folding actually being built.

A separate search for HUBO/HOBO applied to RNA returned general HUBO literature
(feature selection, railway rescheduling, MIMO detection) and **nothing on RNA**.

---

## 3. Where this project stands

### 3.1 Not novel — and we should say so plainly

**Our Level 0 is, to within details, the published state of the art.** Stem or
quartet binary variables, stacking energies, quadratic conflict and crossing
penalties. Fox et al. established it; IBM/Moderna have used it in three
consecutive papers on this exact problem.

This is good news for the write-up rather than bad: it means the fidelity ladder
is anchored at a recognisable baseline, and "Level 0 → Level 1 → Level 2" reads
as *"the published formulation → plus real loop energies → plus multiloops"*.
That framing should be made explicit in the README and slides.

**PCE on mRNA folding is scooped.** Friedhoff, Metkar, Davis et al.,
arXiv:2605.20163 (May 2026, IBM + Moderna) is *Pauli Correlation Encoding for
mRNA Secondary Structure Prediction* — same encoding, same application, three
months ahead of us. We must cite it and must not present PCE-for-RNA as our idea.

However: that paper contains **zero** occurrences of "Burer", "low-rank" or
"dequant". It reports PCE's compression without a classical low-rank control.
Our contribution on this axis is therefore not the encoding but **the missing
control** — and our result is that PCE loses to it. That is a sharper and more
defensible claim than "we implemented PCE".

### 3.2 Plausibly novel

1. **The degree/branch-count characterisation.** That the multiloop closing
   penalty is an "at least 2 branches" indicator, that this makes it
   unrepresentable at *any* fixed degree, and that degree *d* buys exactly
   junctions up to *d* branches. The field has the qualitative obstruction; the
   exact statement appears to be missing.
2. **A working degree-3 HOBO for RNA folding.** No HUBO/HOBO RNA formulation was
   found at all.
3. **Turner interior-loop and multiloop energies extracted into an optimization
   model**, validated to reproduce `eval_structure` exactly.
4. **Dirac-3 / photonic EQC applied to RNA.** No hits whatsoever. The simplex
   corner-collapse analysis appears unexplored for this problem.
5. **The encoding-gap / energy-error decomposition** as a reporting standard,
   including the demonstration that a model can score a perfect structural gap
   while being wrong by ~16 kcal/mol.

### 3.3 Honest caveats on the novelty claim

- Absence of evidence from ~8 searches is not proof. A dedicated HUBO-RNA paper
  in a bioinformatics venue would plausibly not surface on these query terms.
- The JPCB 2025 paper is paywalled and is the single most likely place for the
  degree argument to already exist. **Obtain and read it before claiming
  priority in any public write-up.**
- Fox et al. was read only via others' descriptions of it.

---

## 4. Techniques worth adopting

Found in the literature, applicable to us, and not already in the pipeline:

| Technique | Source | Status here |
|---|---|---|
| CVaR objective for the variational loop | Alevras 2024, Kumar 2025 | **already implemented** in ADAPT-QAOA |
| Shallow classical local search on returned samples | Kumar 2025 | **adopted** — see below |
| Gauge transformation of the Hamiltonian for noise mitigation | Kumar 2025 | not adopted; only meaningful on real gate hardware, which we do not have |
| Tensor-network scalability study | Kumar 2025 | out of scope, listed in Future Directions |
| Exact classical baseline (CPLEX) | Alevras 2024 | equivalent already present (CP-SAT, proven optimal) |

**Adopted: shallow local search post-processing.** Kumar et al. pass device
samples through "a shallow local search on classical nodes" before reporting.
This is standard practice and we were not doing it. It is now implemented in
`rnaqopt.postprocess` as 1-opt hill climbing over the model energy, and — like
the repair step — it is **reported separately**, so it can never silently
inflate a solver's apparent quality. Reporting a raw and a locally-improved
number is also what makes our results comparable to theirs.

---

## 5. What to change in the write-up

1. State that Level 0 reproduces the published formulation, and cite Fox and
   Alevras for it. It makes the ladder legible.
2. Cite arXiv:2605.20163 for PCE-on-mRNA and reframe our PCE section as *the
   dequantization control that paper omits*.
3. Quote Zaborniak's "difficult to penalize directly" as the open problem our
   degree analysis answers. Naming the gap we fill is stronger than asserting
   novelty.
4. Soften any unqualified "first" claim to "we are not aware of" until the JPCB
   paper has been read.
5. Report local-search-improved numbers alongside raw ones.

---

## Sources

- [Alevras, Metkar et al., *mRNA secondary structure prediction using utility-scale quantum computers*, arXiv:2405.20328](https://arxiv.org/abs/2405.20328)
- [Kumar, Alevras, Metkar et al., *Towards secondary structure prediction of longer mRNA sequences using a quantum-centric optimization scheme*, arXiv:2505.05782](https://arxiv.org/pdf/2505.05782)
- [Friedhoff, Metkar, Davis et al., *Pauli Correlation Encoding for mRNA Secondary Structure Prediction*, arXiv:2605.20163](https://arxiv.org/pdf/2605.20163)
- [Zaborniak et al., *A QUBO model of the RNA folding problem optimized by variational hybrid quantum annealing*, arXiv:2208.04367](https://arxiv.org/abs/2208.04367)
- [Fox et al., *RNA folding using quantum computers*, bioRxiv 2021.05.27.446060](https://www.biorxiv.org/content/10.1101/2021.05.27.446060v1.full.pdf)
- [*Exploring the Boundaries of Modern Quantum Annealers with RNA Structure Prediction*, J. Phys. Chem. B 2025](https://pubs.acs.org/doi/10.1021/acs.jpcb.5c07902)
- [Sciorilli et al., *Towards large-scale quantum optimization solvers with few qubits* (PCE), Nature Communications 2025](https://www.nature.com/articles/s41467-025-57580-5)
