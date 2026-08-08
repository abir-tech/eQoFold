"""Per-instance evaluation: model -> solve -> decode -> score -> gap decomposition.

This is where plan section 2.2 is enforced mechanically.  One call produces,
for a single (sequence, level, solver) triple, both halves of the error
decomposition plus the structural metrics, all scored through
``eval_structure`` per the section 2.2 corollary.

Two diagnostics beyond the plan's minimum are recorded, because the headline
"encoding gap" alone can be silently uninformative:

``reference_representable``
    Whether ViennaRNA's MFE structure can be expressed as a stem selection at
    all.  If not, the model cannot reach the right answer no matter how good
    the solver, and the gap is a *representability* failure rather than an
    energy-model failure.

``model_energy_error``
    ``|H(x_ref) - E_vienna(reference structure)|`` -- how wrong the model's
    energy *function* is on a structure it can represent.  This matters because
    the encoding gap as defined (energy of the decoded model optimum minus the
    MFE) is insensitive to energy errors that do not change the argmin.  On
    instances with few competing structures the encoding gap can read 0.00 at
    every rung while the underlying energy model is badly wrong; this column
    moves when the loop-energy extraction improves, and so is the diagnostic
    that actually validates the ladder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import STEMS, VIENNA, StemConfig, ViennaConfig
from .decode import decode, structure_to_selection
from .energy import LoopEnergies
from .metrics import compare_structures
from .model import build_model
from .model.bundle import StemModel
from .model.level1 import structure_is_level1_exact
from .reference import eval_structure, mfe, subopt
from .sequences import RNASequence
from .solvers.base import SolverResult
from .solvers.exact import solve_exact
from .stems import StemGraphs, enumerate_with_graphs


@dataclass
class InstanceEvaluation:
    """Everything measured for one (sequence, level, solver) triple."""

    seq_id: str
    tier: str
    length: int
    n_stems: int
    level: int
    solver: str

    # Energies, all ViennaRNA-evaluated (kcal/mol)
    e_vienna_mfe: float
    e_model_optimum: float
    e_solver: float

    # The section 2.2 decomposition
    encoding_gap: float
    optimizer_gap: float
    total_gap: float

    # Structural accuracy of the solver's structure vs the ViennaRNA MFE
    sensitivity: float
    ppv: float
    f1: float
    bp_distance: int
    exact_match: bool
    matches_subopt: bool

    # Repair accounting -- never folded into the headline numbers
    feasible_raw: bool
    was_repaired: bool
    n_removed: int

    # Diagnostics
    reference_representable: bool
    model_energy_error: float | None
    level_exact_for_reference: bool
    structure: str

    # Model / solver bookkeeping
    model_summary: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        model_summary = row.pop("model_summary")
        resources = row.pop("resources")
        row.update({f"model_{k}": v for k, v in model_summary.items()})
        row.update({f"res_{k}": v for k, v in resources.items()})
        return row


def build_instance(
    sequence: str,
    level: int,
    cfg: ViennaConfig = VIENNA,
    stem_cfg: StemConfig = STEMS,
    pseudoknot_mode: bool = False,
    allow_gu: bool = True,
) -> tuple[StemGraphs, StemModel]:
    """Enumerate stems and build the model at ``level`` for one sequence."""
    graphs = enumerate_with_graphs(
        sequence, cfg=stem_cfg, allow_gu=allow_gu, min_hairpin=cfg.min_hairpin_loop
    )
    energies = LoopEnergies(sequence, cfg)
    model = build_model(
        level,
        sequence,
        graphs,
        cfg=cfg,
        pseudoknot_mode=pseudoknot_mode,
        energies=energies,
    )
    return graphs, model


def evaluate_instance(
    record: RNASequence,
    level: int,
    solver_result: SolverResult | None = None,
    cfg: ViennaConfig = VIENNA,
    stem_cfg: StemConfig = STEMS,
    pseudoknot_mode: bool = False,
    allow_gu: bool = True,
    subopt_window: float = 0.5,
    brute_force_limit: int = 22,
    max_seconds: float = 60.0,
) -> InstanceEvaluation:
    """Evaluate one sequence at one fidelity level.

    ``solver_result`` is the assignment produced by the solver under test.  When
    it is ``None`` the model's own exact optimum is used, so the optimizer gap
    is zero by construction and the row measures *pure encoding gap* -- which is
    exactly what the Phase 2 table needs.
    """
    seq = record.sequence
    graphs, model = build_instance(
        seq, level, cfg, stem_cfg, pseudoknot_mode, allow_gu
    )

    ref_structure, e_vienna_mfe = mfe(seq, cfg)

    # -- the model's own exact optimum -> encoding gap ----------------------
    exact = solve_exact(
        model, brute_force_limit=brute_force_limit, max_seconds=max_seconds
    )
    priority = [model.full.terms.get((i,), 0.0) for i in range(graphs.n)]
    exact_decoded = decode(
        exact.bitstring, graphs, len(seq), priority, pseudoknot_mode
    )
    e_model_optimum = eval_structure(seq, exact_decoded.structure, cfg)

    # -- the solver under test -> optimizer gap ----------------------------
    used = solver_result if solver_result is not None else exact
    decoded = (
        exact_decoded
        if solver_result is None
        else decode(used.bitstring, graphs, len(seq), priority, pseudoknot_mode)
    )
    e_solver = (
        e_model_optimum
        if solver_result is None
        else eval_structure(seq, decoded.structure, cfg)
    )

    # -- structural accuracy ----------------------------------------------
    sm = compare_structures(decoded.structure, ref_structure)
    near_optimal = {s for s, _ in subopt(seq, subopt_window, cfg)}
    matches_subopt = decoded.structure in near_optimal

    # -- diagnostics -------------------------------------------------------
    ref_selection = structure_to_selection(graphs, ref_structure)
    representable = ref_selection is not None
    model_energy_error: float | None = None
    if representable:
        h_ref = model.objective.energy_of_selection(ref_selection)
        model_energy_error = abs(h_ref - e_vienna_mfe)

    return InstanceEvaluation(
        seq_id=record.seq_id,
        tier=record.tier,
        length=len(seq),
        n_stems=graphs.n,
        level=level,
        solver=used.solver_name or "exact",
        e_vienna_mfe=e_vienna_mfe,
        e_model_optimum=e_model_optimum,
        e_solver=e_solver,
        encoding_gap=round(e_model_optimum - e_vienna_mfe, 6),
        optimizer_gap=round(e_solver - e_model_optimum, 6),
        total_gap=round(e_solver - e_vienna_mfe, 6),
        sensitivity=sm.sensitivity,
        ppv=sm.ppv,
        f1=sm.f1,
        bp_distance=sm.bp_distance,
        exact_match=sm.exact_match,
        matches_subopt=matches_subopt,
        feasible_raw=decoded.feasible_raw,
        was_repaired=decoded.was_repaired,
        n_removed=decoded.n_removed,
        reference_representable=representable,
        model_energy_error=(
            None if model_energy_error is None else round(model_energy_error, 6)
        ),
        level_exact_for_reference=structure_is_level1_exact(ref_structure),
        structure=decoded.structure,
        model_summary=model.summary(),
        resources=dict(used.resource_dict),
        wall_time=used.wall_time,
    )


def encoding_gap_table(
    records: list[RNASequence],
    levels: tuple[int, ...] = (0, 1),
    cfg: ViennaConfig = VIENNA,
    stem_cfg: StemConfig = STEMS,
    brute_force_limit: int = 22,
    max_seconds: float = 60.0,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Exact optimum of each model level on each sequence -> pure encoding gap."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for level in levels:
            ev = evaluate_instance(
                record,
                level,
                cfg=cfg,
                stem_cfg=stem_cfg,
                brute_force_limit=brute_force_limit,
                max_seconds=max_seconds,
            )
            rows.append(ev.as_row())
            if progress:
                err = (
                    float("nan")
                    if ev.model_energy_error is None
                    else ev.model_energy_error
                )
                print(
                    f"  {record.seq_id:10s} L{level}  n={ev.n_stems:3d}  "
                    f"enc_gap={ev.encoding_gap:+6.2f}  "
                    f"E_err={err:6.2f}  F1={ev.f1:.2f}"
                )
    return rows
