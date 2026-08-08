#!/usr/bin/env python
"""Submit real jobs to QCi Dirac-3. **Run this yourself; it costs device time.**

Never invoked by ``make all``. Everything else in the repository reproduces
without credentials.

Setup (PowerShell)::

    pip install qci-client
    $env:QCI_TOKEN = "<your token>"
    # QCI_API_URL is optional; defaults to https://api.qci-prod.com

Then, in order::

    python experiments/run_dirac3_hardware.py --check       # free, no submission
    python experiments/run_dirac3_hardware.py --dry-run     # free, builds real payloads
    python experiments/run_dirac3_hardware.py               # SUBMITS, costs allocation

**Why the default instances are Tier M.** The headline claim is that a
multiloop-accurate model needs degree 3 and that Dirac-3 supports exactly that.
Tier A sequences contain no multiloops (measured: none exist below 31 nt), so a
Level 2 model on Tier A has **no cubic terms at all** and encodes at degree 2 --
it cannot test the claim. Only Tier M instances reach degree 3.

**What this run was for, and what it found.** The free tier's degree ceiling (3)
was known; the *variable* ceiling at degree 3 was not, only the degree-4 ceiling
(39). Instances are submitted **smallest first** so the first rejection would
localise the cap. Rejected submissions do not consume allocation, so probing
upward is free.

Run 2026-08-07: **all nine rungs were accepted, up to 78 encoded variables**, so
the degree-3 cap is > 78 and still unmeasured -- the ladder ran out of instances
before the device ran out of capacity. That is already 2x the degree-4 limit, so
the variable cap is strongly degree-dependent. To push further, add larger Tier
M instances or raise ``--samples``.

Device time was also measured: ``t = 0.095 * n_vars^1.28`` seconds
(R^2 = 0.84, see ``results/raw/dirac3_device_timing.csv``). Cost grows with
*encoded dimension*, not just sample count.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from rnaqopt.config import RAW_RESULTS_DIR, STEMS, VIENNA, ensure_dirs
from rnaqopt.decode import decode
from rnaqopt.metrics import compare_structures
from rnaqopt.model import build_model
from rnaqopt.postprocess import local_search
from rnaqopt.reference import eval_structure, mfe
from rnaqopt.sequences import load_all
from rnaqopt.solvers.dirac3 import (
    FREE_TIER_MAX_DEGREE,
    MEASURED_DEGREE3_VAR_ACCEPTED,
    MEASURED_DEGREE4_VAR_LIMIT,
    Dirac3Client,
    encode,
)
from rnaqopt.solvers.exact import solve_exact
from rnaqopt.stems import enumerate_with_graphs

#: (sequence, scheme) pairs ordered by **encoded variable count**, smallest
#: first. All are Tier M, so all encode at degree 3. Ascending size means the
#: first rejection tells you where the degree-3 variable cap sits, for free.
LADDER: tuple[tuple[str, str], ...] = (
    ("M_ml02", "global_slack"),  # 18 vars -- smallest degree-3 job that exists
    ("M_ml08", "global_slack"),  # 23 vars
    ("M_ml06", "global_slack"),  # 31 vars
    ("M_ml09", "global_slack"),  # 33 vars
    ("M_ml02", "per_stem"),      # 34 vars -- same instance, other scheme
    ("M_ml04", "global_slack"),  # 35 vars
    ("M_ml08", "per_stem"),      # 44 vars -- first probe above the degree-4 cap
    ("M_ml10", "global_slack"),  # 62 vars
    ("M_ml01", "per_stem"),      # 78 vars
)


def _fmt_alloc(alloc: dict) -> str:
    try:
        return json.dumps(alloc)[:400]
    except Exception:  # noqa: BLE001
        return str(alloc)[:400]


def check(client: Dirac3Client) -> int:
    """Report environment readiness and remaining allocation. Costs nothing."""
    diag = Dirac3Client.diagnose()
    print("Dirac-3 environment")
    print(f"  qci-client installed : {diag['package_installed']} "
          f"({diag['package_version']})")
    print(f"  token present        : {diag['token_present']} "
          f"(via {diag['token_variable']}, {diag['token_length']} chars)")
    print(f"  api url              : {diag['api_url']}"
          f"{'' if diag['api_url_from_env'] else '  [default]'}")
    print(f"  ready                : {diag['ready']}")
    for p in diag["problems"]:
        print(f"    - {p}")
    if not diag["ready"]:
        return 1
    try:
        print("\nallocation remaining:")
        print(" ", _fmt_alloc(client.allocations()))
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read allocations: {str(exc)[:200]}")
        return 1
    print(f"\nmeasured device-time scaling (2026-08-07 run, R^2 = 0.84): "
          f"t = {Dirac3Client.TIME_SCALE} * n_vars^{Dirac3Client.TIME_EXPONENT} s")
    for n in (20, 40, 80):
        print(f"    {n:3d} encoded vars -> {client.estimated_device_seconds(n):5.1f} s"
              f"  ({client.num_samples} samples, "
              f"schedule {client.relaxation_schedule})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default=None,
                    help="comma-separated seq_ids; default is the Tier M ladder")
    ap.add_argument("--scheme", default=None,
                    choices=["per_stem", "global_slack"],
                    help="force one scheme (default: as specified per ladder rung)")
    ap.add_argument("--level", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--relaxation-schedule", type=int, default=1)
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--var-limit", type=int, default=None,
                    help="refuse to submit above this many encoded variables")
    ap.add_argument("--max-seconds", type=float, default=120.0,
                    help="stop before exceeding this much estimated device time")
    ap.add_argument("--check", action="store_true",
                    help="report readiness and allocation, then exit; costs nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and preflight real payloads without submitting")
    ap.add_argument("--stop-on-reject", action="store_true", default=True,
                    help="stop climbing the ladder after the first rejection")
    args = ap.parse_args(argv)

    ensure_dirs()

    base = Dirac3Client(
        num_samples=args.samples,
        relaxation_schedule=args.relaxation_schedule,
        var_limit=args.var_limit,
        R=args.R,
    )
    if args.check:
        return check(base)

    if args.instances:
        rungs = [
            (s.strip(), args.scheme or "global_slack")
            for s in args.instances.split(",")
            if s.strip()
        ]
    else:
        rungs = [
            (sid, args.scheme or sch) for sid, sch in LADDER
        ]

    records = {r.seq_id: r for r in load_all()}
    diag = Dirac3Client.diagnose()

    print(VIENNA.header_line())
    print(f"free-tier degree ceiling: {FREE_TIER_MAX_DEGREE}   "
          f"measured var cap at degree 4: {MEASURED_DEGREE4_VAR_LIMIT}   "
          f"at degree 3: >{MEASURED_DEGREE3_VAR_ACCEPTED} (measured 2026-08-07, no rejection)")
    print(f"level={args.level}  samples={args.samples}  "
          f"schedule={args.relaxation_schedule}  "
          f"device time ~ {Dirac3Client.TIME_SCALE}*n^{Dirac3Client.TIME_EXPONENT} s")
    print(f"ready={diag['ready']}" + ("" if diag["ready"] else
          "   -> preflight only; " + "; ".join(diag["problems"])))
    print()

    rows: list[dict] = []
    spent = 0.0
    header = (f"{'instance':9s} {'scheme':13s} {'stems':>5} {'vars':>5} {'deg':>3} "
              f"{'R':>6}  {'status':10s} result")
    print(header)
    print("-" * len(header))

    for seq_id, scheme in rungs:
        rec = records.get(seq_id)
        if rec is None:
            print(f"{seq_id:9s} -- unknown sequence id")
            continue

        graphs = enumerate_with_graphs(rec.sequence, cfg=STEMS)
        model = build_model(args.level, rec.sequence, graphs, cfg=VIENNA)
        client = Dirac3Client(
            scheme=scheme,
            R=args.R,
            num_samples=args.samples,
            relaxation_schedule=args.relaxation_schedule,
            var_limit=args.var_limit,
        )
        enc = encode(model, scheme=scheme, R=args.R)
        issues = client.preflight(enc)

        row = {
            "seq_id": seq_id,
            "tier": rec.tier,
            "length": rec.length,
            "scheme": scheme,
            "level": args.level,
            "n_stems": graphs.n,
            "encoded_vars": enc.n_vars,
            "degree": enc.degree,
            "n_terms": len([k for k in enc.terms if k]),
            "R": enc.R,
            "preflight_ok": not issues,
            "preflight_issues": "; ".join(issues),
            "submitted": False,
            "est_device_seconds": round(
                client.estimated_device_seconds(enc.n_vars), 2),
        }
        prefix = (f"{seq_id:9s} {scheme:13s} {graphs.n:5d} {enc.n_vars:5d} "
                  f"{enc.degree:3d} {enc.R:6.1f}")

        if enc.degree < 3:
            row["note"] = "degree<3: cannot exercise the degree-3 claim"

        if issues:
            print(f"{prefix}  {'REJECTED':10s} {'; '.join(issues)[:60]}")
            rows.append(row)
            if args.stop_on_reject:
                print("  -> stopping: larger jobs would also be rejected")
                break
            continue

        if args.dry_run or not diag["ready"]:
            note = "dry-run" if args.dry_run else "no-creds"
            extra = "" if enc.degree >= 3 else "  [degree 2 -- not a degree-3 test]"
            print(f"{prefix}  {'OK':10s} {note}{extra}")
            rows.append(row)
            continue

        job_seconds = client.estimated_device_seconds(enc.n_vars)
        if spent + job_seconds > args.max_seconds:
            print(f"{prefix}  {'BUDGET':10s} would exceed --max-seconds "
                  f"({spent:.0f}s spent)")
            rows.append(row)
            break

        try:
            result = client.solve(model)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)[:400]
            rows.append(row)
            print(f"{prefix}  {'FAILED':10s} {str(exc)[:60]}")
            if args.stop_on_reject and "limit" in str(exc).lower():
                print("  -> stopping: looks like a device limit")
                break
            continue

        spent += job_seconds
        exact = solve_exact(model, max_seconds=60)
        ref_db, ref_e = mfe(rec.sequence, VIENNA)
        priority = [model.full.terms.get((i,), 0.0) for i in range(graphs.n)]
        dec = decode(result.bitstring, graphs, len(rec.sequence), priority)
        e_vienna = eval_structure(rec.sequence, dec.structure, VIENNA)
        sm = compare_structures(dec.structure, ref_db)

        # Shallow classical local search on the returned sample, as the
        # published state of the art does (arXiv:2505.05782). Reported as its
        # own columns so the raw device result is never overwritten by it.
        ls = local_search(model.full, result.bitstring)
        dec_ls = decode(ls.bitstring, graphs, len(rec.sequence), priority)
        e_vienna_ls = eval_structure(rec.sequence, dec_ls.structure, VIENNA)
        sm_ls = compare_structures(dec_ls.structure, ref_db)

        row.update({
            "submitted": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": result.solver_metadata.get("job_id"),
            "model_energy": result.model_energy,
            "exact_model_energy": exact.model_energy,
            "optimizer_gap_model": round(
                result.model_energy - exact.model_energy, 4),
            "e_vienna_mfe": ref_e,
            "e_solver": e_vienna,
            "total_gap": round(e_vienna - ref_e, 4),
            "f1": round(sm.f1, 4),
            "sensitivity": round(sm.sensitivity, 4),
            "ppv": round(sm.ppv, 4),
            "exact_match": sm.exact_match,
            "feasible_raw": dec.feasible_raw,
            "was_repaired": dec.was_repaired,
            "n_removed": dec.n_removed,
            "wall_time": round(result.wall_time, 3),
            "device_shots": args.samples,
            "n_solutions": result.solver_metadata.get("n_solutions"),
            "structure": dec.structure,
            "reference_structure": ref_db,
            "device_energies": json.dumps(
                result.solver_metadata.get("device_energies", [])),
            # post-processed, kept strictly separate
            **ls.as_dict(),
            "ls_optimizer_gap_model": round(
                ls.energy_after - exact.model_energy, 4),
            "ls_e_solver": e_vienna_ls,
            "ls_total_gap": round(e_vienna_ls - ref_e, 4),
            "ls_f1": round(sm_ls.f1, 4),
            "ls_exact_match": sm_ls.exact_match,
            "ls_structure": dec_ls.structure,
        })
        rows.append(row)
        print(f"{prefix}  {'SUBMITTED':10s} optgap {row['optimizer_gap_model']:+.3f}  "
              f"total {row['total_gap']:+.3f}  F1 {row['f1']:.3f}"
              + (f"  | +LS optgap {row['ls_optimizer_gap_model']:+.3f} "
                 f"F1 {row['ls_f1']:.3f}" if ls.improved else "  | LS: no change"))

    df = pd.DataFrame(rows)
    submitted_any = bool(df["submitted"].any()) if "submitted" in df else False

    # Preflight output must NEVER share a path with real device results. A later
    # --dry-run once silently overwrote an irreplaceable hardware run
    # (2026-08-07); device time is not refundable, so the paths are now disjoint
    # and archived runs are never rewritten.
    if not submitted_any:
        out = RAW_RESULTS_DIR / "dirac3_preflight.csv"
        df.to_csv(out, index=False, lineterminator="\n")
        print(f"\nwrote {out}  (preflight only; device results never land here)")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = RAW_RESULTS_DIR / f"dirac3_hardware_{stamp}.csv"
        df.to_csv(archive, index=False, lineterminator="\n")

        # The canonical file is the union of every archived device run, so a
        # later run extends the record instead of replacing it.
        frames = [
            pd.read_csv(p)
            for p in sorted(RAW_RESULTS_DIR.glob("dirac3_hardware_*.csv"))
        ]
        merged = pd.concat(frames, ignore_index=True) if frames else df
        canonical = RAW_RESULTS_DIR / "dirac3_hardware.csv"
        merged.to_csv(canonical, index=False, lineterminator="\n")
        print(f"\nwrote {archive}")
        print(f"merged {len(frames)} device run(s) -> {canonical}")

    submitted = df["submitted"].sum() if "submitted" in df else 0
    if submitted:
        ok = df[df.submitted]
        print(f"submitted {int(submitted)} job(s), "
              f"~{ok.est_device_seconds.sum():.0f} s estimated device time")
        deg3 = ok[ok.degree >= 3]
        if len(deg3):
            print(f"largest accepted degree-3 job: "
                  f"{int(deg3.encoded_vars.max())} encoded variables")
    rejected = df[~df.preflight_ok] if "preflight_ok" in df else df.iloc[:0]
    if len(rejected):
        print(f"first rejection at {int(rejected.encoded_vars.min())} variables "
              "-> that brackets the degree-3 cap")

    if args.dry_run:
        print("\nDRY RUN -- nothing was submitted and no allocation was used.")
        print("Re-run without --dry-run to submit the rungs marked OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
