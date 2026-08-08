"""QCi Dirac-3 (EQC) encoding, simplex simulator, and device client.

Plan section 4.6 flags this as *the main engineering hurdle; budget real time
for it.*  Dirac-3 minimises a polynomial of degree <= 3 over **non-negative
continuous** variables subject to a single **fixed sum constraint**

    sum_i x_i = R,    x_i >= 0

Our variables are binary, so the encoding is a research question, not a
translation exercise.  Two schemes are implemented and compared, as the plan
requires.

**Why the sum constraint bites: the corner problem.**

The domain is a compact simplex, so the objective cannot literally diverge.  But
our linear coefficients are stacking energies and are *negative*, and on a
simplex a negative linear objective is minimised by putting **all** the mass on
the single most negative coordinate.  The minimiser runs to a vertex --
"select one stem with weight R" -- which decodes to a nearly empty structure.
This matches the failure mode measured on this device on a previous project
(a cubic objective driving the minimiser "to a simplex corner unrelated to the
mode").  It is a property of the geometry, not a tuning problem.

**Encoding A -- complementary slack per stem (``per_stem``).**

Variables ``x_i`` and ``xbar_i`` for each stem, with ``R = n * cap`` and a
quadratic penalty pinning ``x_i + xbar_i = cap``.  The pin **caps every
coordinate at ``cap``**, so mass physically cannot concentrate on one stem, and
the corner problem is structurally removed.  Costs ``2n`` variables.

**Encoding B -- single global slack (``global_slack``).**

Variables ``x_i`` plus one slack ``s``, with ``sum_i x_i + s = R`` and ``R``
an upper bound on the number of simultaneously selected stems (times ``cap``).
Costs ``n + 1`` variables -- roughly half of Encoding A -- but nothing caps an
individual ``x_i``, so it is exposed to the corner problem.

The prediction is therefore that Encoding A is markedly more reliable and
Encoding B markedly cheaper in variables, and *that comparison is itself a
result* (plan section 4.6).  Both are measured rather than asserted.

**Binarisation.**  Both schemes add ``mu * sum_i x_i * (cap - x_i)``, which is
zero at ``x_i in {0, cap}`` and maximal at ``cap/2``, pushing the continuous
optimum toward the binary corners of each coordinate.

Device access is optional: :class:`DiracSimplexSimulator` solves the identical
continuous programme classically, so every encoding study, ``R`` sweep and
decode path runs and is testable without credentials.  :class:`Dirac3Client`
submits the same job to real hardware when credentials are present.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model.bundle import StemModel
from .base import SolverResult, timed

#: Free-tier polynomial degree ceiling, measured against the QCi API.
FREE_TIER_MAX_DEGREE = 3

#: Measured device variable ceiling *at degree 4*, from a previous project.
MEASURED_DEGREE4_VAR_LIMIT = 39

#: Largest degree-3 job **accepted** by the device, measured 2026-08-07: 78
#: encoded variables, submitted and completed without rejection. The true
#: degree-3 ceiling is therefore strictly greater than this and remains
#: unmeasured -- the ladder ran out of instances before the device ran out of
#: capacity. Note this is already 2x the degree-4 limit, so the variable cap is
#: strongly degree-dependent.
MEASURED_DEGREE3_VAR_ACCEPTED = 78


@dataclass
class DiracEncoding:
    """A polynomial in Dirac-3 form: coefficients plus a sum constraint."""

    #: Term coefficients over the *encoded* variables.
    terms: dict[tuple[int, ...], float]
    #: Number of encoded (continuous, non-negative) variables.
    n_vars: int
    #: The fixed sum constraint value.
    R: float
    #: Per-coordinate cap that represents "selected".
    cap: float
    #: Which scheme produced this.
    scheme: str
    #: Indices of the encoded variables that map back to stems.
    stem_vars: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def degree(self) -> int:
        return max((len(k) for k in self.terms), default=0)

    def _compiled(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Terms grouped by order as ``(index_matrix, coefficients)`` arrays.

        Compiled once and cached. The projected-gradient simulator evaluates the
        objective thousands of times per instance, and a per-term Python loop
        over a Level 2 model (which can carry over a thousand cubic terms)
        dominates the runtime of ``make all``.
        """
        cached = self.metadata.get("_compiled")
        if cached is not None:
            return cached
        groups: dict[int, list[tuple[tuple[int, ...], float]]] = {}
        for key, coeff in self.terms.items():
            groups.setdefault(len(key), []).append((key, coeff))
        compiled = {
            order: (
                np.array([k for k, _ in items], dtype=np.int64).reshape(
                    len(items), order
                ),
                np.array([c for _, c in items], dtype=np.float64),
            )
            for order, items in groups.items()
            if order > 0
        }
        self.metadata["_compiled"] = compiled
        return compiled

    @property
    def constant(self) -> float:
        return self.terms.get((), 0.0)

    def evaluate(self, x: np.ndarray) -> float:
        total = self.constant
        for _order, (idx, coeffs) in self._compiled().items():
            total += float(np.dot(coeffs, np.prod(x[idx], axis=1)))
        return float(total)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        for order, (idx, coeffs) in self._compiled().items():
            vals = x[idx]  # (n_terms, order)
            full = np.prod(vals, axis=1)
            for pos in range(order):
                # Product of the other factors; guard against division by zero
                # by recomputing rather than dividing out.
                others = np.delete(vals, pos, axis=1)
                partial = (
                    np.prod(others, axis=1)
                    if others.shape[1]
                    else np.ones_like(full)
                )
                np.add.at(grad, idx[:, pos], coeffs * partial)
        return grad

    def fits_free_tier(self) -> bool:
        return self.degree <= FREE_TIER_MAX_DEGREE

    def summary(self) -> dict[str, Any]:
        hist: dict[int, int] = {}
        for k in self.terms:
            hist[len(k)] = hist.get(len(k), 0) + 1
        return {
            "scheme": self.scheme,
            "encoded_vars": self.n_vars,
            "R": self.R,
            "cap": self.cap,
            "degree": self.degree,
            "n_terms": len(self.terms),
            "fits_free_tier": self.fits_free_tier(),
            **{f"n_order_{k}": v for k, v in sorted(hist.items())},
            **self.metadata,
        }


# --------------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------------


def encode(
    model: StemModel,
    scheme: str = "per_stem",
    cap: float = 1.0,
    R: float | None = None,
    mu: float | None = None,
    pin_weight: float | None = None,
    use_penalties: bool = True,
) -> DiracEncoding:
    """Encode a :class:`StemModel` for Dirac-3.

    ``R`` defaults to the value that makes the scheme self-consistent: ``n*cap``
    for ``per_stem`` (every pair contributes exactly ``cap``), and
    ``density * n * cap`` for ``global_slack``, where the default density is a
    prior on how many stems a structure selects.
    """
    poly = model.full if use_penalties else model.objective
    n = poly.n_vars
    scale = max(
        (abs(c) for k, c in poly.terms.items() if k), default=1.0
    )
    mu = 4.0 * scale if mu is None else mu
    pin_weight = 10.0 * scale if pin_weight is None else pin_weight

    if scheme == "per_stem":
        return _encode_per_stem(poly, n, cap, R, mu, pin_weight)
    if scheme == "global_slack":
        return _encode_global_slack(poly, n, cap, R, mu)
    raise ValueError(f"unknown scheme {scheme!r}; use 'per_stem' or 'global_slack'")


def _rescale_terms(
    terms: dict[tuple[int, ...], float], cap: float
) -> dict[tuple[int, ...], float]:
    """Rewrite a binary polynomial in terms of ``x_i in [0, cap]``.

    ``x_binary = x_continuous / cap``, so an order-``k`` term picks up
    ``cap^-k``.
    """
    return {k: c / (cap ** len(k)) if k else c for k, c in terms.items()}


def _encode_per_stem(
    poly, n: int, cap: float, R: float | None, mu: float, pin_weight: float
) -> DiracEncoding:
    """Encoding A: ``x_i`` at index ``i``, ``xbar_i`` at index ``n + i``."""
    terms: dict[tuple[int, ...], float] = {}

    def add(key: tuple[int, ...], value: float) -> None:
        key = tuple(sorted(key))
        terms[key] = terms.get(key, 0.0) + value

    for key, coeff in _rescale_terms(poly.terms, cap).items():
        add(key, coeff)

    # Pin x_i + xbar_i = cap:  w * (x_i + xbar_i - cap)^2
    for i in range(n):
        j = n + i
        add((i, i), pin_weight)
        add((j, j), pin_weight)
        add((i, j), 2.0 * pin_weight)
        add((i,), -2.0 * pin_weight * cap)
        add((j,), -2.0 * pin_weight * cap)
        add((), pin_weight * cap * cap)
        # Binarisation: mu * x_i * xbar_i is 0 at the corners, max in the middle.
        add((i, j), mu)

    terms = {k: v for k, v in terms.items() if v != 0.0}
    total_R = n * cap if R is None else R
    return DiracEncoding(
        terms=terms,
        n_vars=2 * n,
        R=total_R,
        cap=cap,
        scheme="per_stem",
        stem_vars=tuple(range(n)),
        metadata={
            "mu": mu,
            "pin_weight": pin_weight,
            "corner_protected": True,
            "vars_per_stem": 2,
        },
    )


def _encode_global_slack(
    poly, n: int, cap: float, R: float | None, mu: float
) -> DiracEncoding:
    """Encoding B: ``x_i`` at index ``i``, one slack at index ``n``."""
    terms: dict[tuple[int, ...], float] = {}

    def add(key: tuple[int, ...], value: float) -> None:
        key = tuple(sorted(key))
        terms[key] = terms.get(key, 0.0) + value

    for key, coeff in _rescale_terms(poly.terms, cap).items():
        add(key, coeff)

    # Binarisation without a partner variable: mu * x_i * (cap - x_i).
    for i in range(n):
        add((i,), mu * cap)
        add((i, i), -mu)

    terms = {k: v for k, v in terms.items() if v != 0.0}
    # R is a prior on structure density: how many stems may be simultaneously
    # selected. Swept in the R study.
    total_R = (0.5 * n * cap) if R is None else R
    return DiracEncoding(
        terms=terms,
        n_vars=n + 1,
        R=total_R,
        cap=cap,
        scheme="global_slack",
        stem_vars=tuple(range(n)),
        metadata={"mu": mu, "corner_protected": False, "vars_per_stem": 1},
    )


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def decode_continuous(encoding: DiracEncoding, x: np.ndarray) -> tuple[int, ...]:
    """Threshold a continuous solution back to a stem bitstring."""
    return tuple(
        1 if x[v] > 0.5 * encoding.cap else 0 for v in encoding.stem_vars
    )


# --------------------------------------------------------------------------
# Classical simplex simulator
# --------------------------------------------------------------------------


def project_to_simplex(v: np.ndarray, R: float) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum x = R}`` (Duchi et al., 2008)."""
    if R <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - R
    idx = np.arange(1, len(v) + 1)
    cond = u - css / idx > 0
    if not cond.any():
        return np.full_like(v, R / len(v))
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


class DiracSimplexSimulator:
    """Projected-gradient minimiser over the Dirac-3 domain.

    Solves the *same* continuous programme the device solves, classically. This
    is what makes the encoding study, the ``R`` sweep and the decode path
    runnable and testable without credentials, and it is also the honest
    baseline: any advantage claimed for the device must be measured against
    solving its own relaxation on a laptop.
    """

    name = "dirac3_simulated"

    def __init__(
        self,
        scheme: str = "per_stem",
        cap: float = 1.0,
        R: float | None = None,
        n_restarts: int = 20,
        n_steps: int = 600,
        step_size: float = 0.15,
        seed: int = 0,
        mu: float | None = None,
    ) -> None:
        self.scheme = scheme
        self.cap = cap
        self.R = R
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.step_size = step_size
        self.seed = seed
        self.mu = mu

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        enc = encode(
            model,
            scheme=self.scheme,
            cap=self.cap,
            R=self.R,
            mu=self.mu,
            use_penalties=use_penalties,
        )
        poly = model.full if use_penalties else model.objective
        n = poly.n_vars
        if n == 0:
            return SolverResult(
                bitstring=(),
                model_energy=poly.constant,
                wall_time=0.0,
                resource_dict=enc.summary(),
                solver_metadata={"scheme": self.scheme},
                seed=self.seed,
                solver_name=self.name,
            )

        rng = np.random.default_rng(self.seed)
        best_bits: tuple[int, ...] = tuple([0] * n)
        best_energy = math.inf
        corner_hits = 0
        continuous_best = math.inf
        stem_idx = np.asarray(enc.stem_vars, dtype=int)
        shares: list[float] = []

        with timed() as elapsed:
            for _ in range(self.n_restarts):
                x = project_to_simplex(rng.random(enc.n_vars) * enc.cap, enc.R)
                for step in range(self.n_steps):
                    grad = enc.gradient(x)
                    # Normalise the gradient and decay the step. The encoded
                    # objective carries penalty weights an order of magnitude
                    # above the energies, so a raw fixed-size step oscillates
                    # instead of descending.
                    norm = np.linalg.norm(grad)
                    if norm > 0:
                        grad = grad / norm
                    lr = self.step_size * enc.R / (1.0 + step / 50.0)
                    x = project_to_simplex(x - lr * grad, enc.R)
                value = enc.evaluate(x)
                continuous_best = min(continuous_best, value)

                # Concentration diagnostic for the corner problem: the share of
                # total stem mass held by the single largest stem coordinate.
                # Reported as a continuous quantity rather than a boolean flag,
                # because a share of 1.0 is *correct* when the true structure
                # really does contain one helix -- only comparison against the
                # exact solution's stem count can distinguish collapse from a
                # correct sparse answer.
                stem_mass = x[stem_idx]
                total = float(stem_mass.sum())
                share = float(stem_mass.max() / total) if total > 1e-12 else 0.0
                shares.append(share)
                if share > 0.9 and int((stem_mass > 0.5 * enc.cap).sum()) <= 1:
                    corner_hits += 1

                bits = decode_continuous(enc, x)
                e = poly.energy(bits)
                if e < best_energy:
                    best_energy, best_bits = e, bits

        return SolverResult(
            bitstring=best_bits,
            model_energy=best_energy,
            wall_time=elapsed[0],
            resource_dict={
                **enc.summary(),
                "function_evaluations": self.n_restarts * self.n_steps,
            },
            solver_metadata={
                "proven_optimal": False,
                "scheme": self.scheme,
                "R": enc.R,
                "cap": enc.cap,
                "restarts": self.n_restarts,
                "collapsed_runs": corner_hits,
                "collapse_rate": corner_hits / max(self.n_restarts, 1),
                "mean_max_stem_share": (
                    float(np.mean(shares)) if shares else 0.0
                ),
                "n_selected_continuous": sum(best_bits),
                "continuous_objective": continuous_best,
                "use_penalties": use_penalties,
            },
            seed=self.seed,
            solver_name=f"{self.name}[{self.scheme}]",
        )


# --------------------------------------------------------------------------
# Real device
# --------------------------------------------------------------------------


class Dirac3Client:
    """Submits an encoded polynomial to real Dirac-3 hardware.

    Verified against ``qci-client`` **5.0.0**.  Three things about that API are
    easy to get wrong, and each one costs a failed submission:

    1. The token environment variable is ``QCI_TOKEN``, *not* ``QCI_API_TOKEN``.
       ``qci_client`` reads ``QCI_TOKEN`` and ``QCI_API_URL``, and nothing else.
    2. A degree-3 polynomial must be **uploaded as a file** and referenced by
       ``polynomial_file_id``.  Inlining it under a ``"polynomial"`` key in the
       job body -- which older examples show -- is rejected by v5.
    3. The job type is ``sample-hamiltonian`` with ``device_type="dirac-3"``,
       which the client internally rewrites to ``dirac-3_normalized_qudit``
       (problem type NQHO).

    Two device limits are checked in sequence by the API, and the second only
    surfaces once the first passes:

    1. degree <= 3 (free-tier entitlement),
    2. a degree-dependent variable cap (measured: 39 at degree 4; the degree-3
       cap is unmeasured, and finding it is what the hardware run is for).

    Never invoked by ``make all``; the whole project reproduces without
    credentials via :class:`DiracSimplexSimulator`.
    """

    name = "dirac3_device"

    #: QCi production API endpoint.  ``qci-client`` ships **no** default and
    #: raises if neither ``url=`` nor ``QCI_API_URL`` is given, so supplying one
    #: here means a user only has to set the token.
    DEFAULT_API_URL = "https://api.qci-prod.com"

    #: Seconds of device time per sample at ``relaxation_schedule=1``, calibrated
    #: from a previous project measurement (20 samples, schedule 1, about 3 s).
    #: Budgeting only; the API reports true usage after the fact.
    SECONDS_PER_SAMPLE_SCHEDULE1 = 0.15
    #: Fixed per-job overhead in seconds, excluding queue wait.
    JOB_OVERHEAD_SECONDS = 1.0

    #: Measured device-time scaling, fitted to the 2026-08-07 hardware run
    #: (9 accepted degree-3 jobs, 18-78 encoded variables, 107 s total):
    #: ``t = 0.095 * n_vars^1.28`` seconds, R^2 = 0.84. See
    #: ``results/raw/dirac3_device_timing.csv``.
    #:
    #: The flat per-sample estimate this replaces underpredicted the run by 3x,
    #: because device time grows with *encoded dimension*, not just sample count.
    TIME_SCALE = 0.095
    TIME_EXPONENT = 1.28

    def __init__(
        self,
        scheme: str = "per_stem",
        cap: float = 1.0,
        R: float | None = None,
        num_samples: int = 20,
        relaxation_schedule: int = 1,
        seed: int = 0,
        var_limit: int | None = None,
    ) -> None:
        self.scheme = scheme
        self.cap = cap
        self.R = R
        self.num_samples = num_samples
        self.relaxation_schedule = relaxation_schedule
        self.seed = seed
        self.var_limit = var_limit

    # -- environment -------------------------------------------------------

    @staticmethod
    def token() -> str | None:
        """The API token, accepting our older alias as well as the real name.

        ``qci_client`` only ever reads ``QCI_TOKEN``.  ``QCI_API_TOKEN`` is
        accepted here because an earlier revision of this repository told users
        to set that name, and silently ignoring it would be a trap.
        """
        return os.environ.get("QCI_TOKEN") or os.environ.get("QCI_API_TOKEN")

    @classmethod
    def api_url(cls) -> str:
        """API endpoint, falling back to the production default."""
        return os.environ.get("QCI_API_URL") or cls.DEFAULT_API_URL

    @classmethod
    def _client(cls):
        """A configured ``QciClient``. Both url and token are passed explicitly
        so behaviour does not depend on which variables happen to be exported."""
        from qci_client import QciClient  # type: ignore

        return QciClient(url=cls.api_url(), api_token=cls.token())

    @classmethod
    def diagnose(cls) -> dict[str, Any]:
        """Explain *why* the device is or is not reachable.

        A bare ``credentials present: False`` is useless when it could mean
        either a missing package or a missing token.  This separates them.
        """
        try:
            import qci_client  # noqa: F401

            package = True
            version = getattr(qci_client, "__version__", "unknown")
        except ImportError:
            package, version = False, None

        tok = cls.token()
        if os.environ.get("QCI_TOKEN"):
            which = "QCI_TOKEN"
        elif os.environ.get("QCI_API_TOKEN"):
            which = "QCI_API_TOKEN (alias; qci-client itself reads QCI_TOKEN)"
        else:
            which = None

        problems: list[str] = []
        if not package:
            problems.append("qci-client is not installed (pip install qci-client)")
        if not tok:
            problems.append(
                'QCI_TOKEN is not set (PowerShell: $env:QCI_TOKEN = "<token>")'
            )
        return {
            "package_installed": package,
            "package_version": version,
            "token_present": bool(tok),
            "token_variable": which,
            "token_length": len(tok) if tok else 0,
            "api_url": cls.api_url(),
            "api_url_from_env": bool(os.environ.get("QCI_API_URL")),
            "ready": package and bool(tok),
            "problems": problems,
        }

    @classmethod
    def is_available(cls) -> bool:
        return cls.diagnose()["ready"]

    # -- budgeting ---------------------------------------------------------

    def estimated_device_seconds(self, n_vars: int | None = None) -> float:
        """Device time one accepted submission consumes.

        With ``n_vars`` this uses the measured power law
        ``0.095 * n^1.28`` (fitted to the 2026-08-07 run, R^2 = 0.84), scaled
        for sample count and relaxation schedule relative to the 20-sample,
        schedule-1 baseline those measurements were taken at.

        Without ``n_vars`` it falls back to the flat per-sample figure, which is
        known to underpredict by roughly 3x and is kept only so callers that do
        not yet have an encoding can still print *something*.

        Allocation is time-based and **rejected submissions do not consume it**,
        so a preflight failure is free; only accepted jobs are billed.
        """
        sample_factor = (self.num_samples / 20.0) * max(1, self.relaxation_schedule)
        if n_vars is None:
            return (
                self.num_samples
                * max(1, self.relaxation_schedule)
                * self.SECONDS_PER_SAMPLE_SCHEDULE1
                + self.JOB_OVERHEAD_SECONDS
            )
        return self.TIME_SCALE * (n_vars**self.TIME_EXPONENT) * sample_factor

    def allocations(self) -> dict[str, Any]:
        """Remaining allocation as reported by the API. Costs no device time."""

        return self._client().get_allocations()

    # -- submission --------------------------------------------------------

    def preflight(self, encoding: DiracEncoding) -> list[str]:
        """Problems that would make the device reject this job."""
        issues: list[str] = []
        if encoding.degree > FREE_TIER_MAX_DEGREE:
            issues.append(
                f"degree {encoding.degree} exceeds the free-tier limit of "
                f"{FREE_TIER_MAX_DEGREE}"
            )
        if self.var_limit is not None and encoding.n_vars > self.var_limit:
            issues.append(
                f"{encoding.n_vars} variables exceeds the stated limit of "
                f"{self.var_limit}"
            )
        if encoding.R <= 0:
            issues.append(f"sum constraint R={encoding.R} must be positive")
        if not any(k for k in encoding.terms):
            issues.append("polynomial has no non-constant terms")
        return issues

    def polynomial_file(
        self, encoding: DiracEncoding, name: str = "rnaqopt"
    ) -> dict[str, Any]:
        """Build the ``polynomial`` payload for the files API.

        The constant term is dropped: it shifts the objective uniformly and so
        cannot move the minimiser, while keeping it would force ``min_degree=0``
        and an all-zero index tuple.  Reported energies are recomputed locally
        from the decoded bitstring, so nothing is lost.
        """
        orders = [len(k) for k in encoding.terms if k]
        return {
            "file_name": name,
            "file_config": {
                "polynomial": {
                    "min_degree": min(orders),
                    "max_degree": max(orders),
                    "num_variables": encoding.n_vars,
                    "data": _polynomial_elements(encoding),
                }
            },
        }

    def job_params(self, encoding: DiracEncoding) -> dict[str, Any]:
        """Device parameters for the job body."""
        return {
            "device_type": "dirac-3",
            "relaxation_schedule": self.relaxation_schedule,
            "num_samples": self.num_samples,
            "sum_constraint": float(encoding.R),
        }

    def solve(
        self, model: StemModel, use_penalties: bool = True, **kwargs: Any
    ) -> SolverResult:
        diag = self.diagnose()
        if not diag["ready"]:
            raise RuntimeError(
                "Dirac-3 unavailable: "
                + "; ".join(diag["problems"])
                + ". Use DiracSimplexSimulator for the credential-free path."
            )

        enc = encode(
            model,
            scheme=self.scheme,
            cap=self.cap,
            R=self.R,
            use_penalties=use_penalties,
        )
        issues = self.preflight(enc)
        if issues:
            raise ValueError("Dirac-3 preflight failed: " + "; ".join(issues))

        poly = model.full if use_penalties else model.objective
        client = self._client()

        upload = client.upload_file(
            file=self.polynomial_file(
                enc, f"rnaqopt-l{model.level}-{self.scheme}"
            )
        )
        file_id = upload["file_id"]

        job_body = client.build_job_body(
            job_type="sample-hamiltonian",
            job_params=self.job_params(enc),
            polynomial_file_id=file_id,
            job_name=f"rnaqopt-l{model.level}-{self.scheme}",
            job_tags=["rnaqopt", f"level{model.level}", self.scheme],
        )

        with timed() as elapsed:
            response = client.process_job(job_body=job_body)

        results = (response or {}).get("results") or {}
        solutions = results.get("solutions") or []
        energies = results.get("energies") or []

        best_bits: tuple[int, ...] = tuple([0] * poly.n_vars)
        best_energy = math.inf
        for sol in solutions:
            bits = decode_continuous(enc, np.asarray(sol, dtype=float))
            e = poly.energy(bits)
            if e < best_energy:
                best_energy, best_bits = e, bits
        if not solutions:
            best_energy = poly.energy(best_bits)

        job_info = (response or {}).get("job_info") or {}
        return SolverResult(
            bitstring=best_bits,
            model_energy=best_energy,
            wall_time=elapsed[0],
            resource_dict={
                **enc.summary(),
                "device_shots": self.num_samples,
                "relaxation_schedule": self.relaxation_schedule,
                "estimated_device_seconds": self.estimated_device_seconds(),
            },
            solver_metadata={
                "proven_optimal": False,
                "scheme": self.scheme,
                "device_energies": energies,
                "n_solutions": len(solutions),
                "job_id": job_info.get("job_id"),
                "job_status": job_info.get("job_status"),
                "use_penalties": use_penalties,
            },
            seed=self.seed,
            solver_name=self.name,
        )


def _polynomial_elements(encoding: DiracEncoding) -> list[dict[str, Any]]:
    """Dirac-3 polynomial wire format.

    Indices are **1-based**, and every element's ``idx`` list is padded at the
    front with ``0`` up to the polynomial's maximum degree -- that is how the
    API represents a term of order below the overall degree inside a
    fixed-arity list.  The constant term is excluded by the caller.
    """
    orders = [len(k) for k in encoding.terms if k]
    degree = max(orders) if orders else 0
    out: list[dict[str, Any]] = []
    for key, coeff in sorted(encoding.terms.items()):
        if not key:
            continue
        out.append(
            {
                "idx": [0] * (degree - len(key)) + [v + 1 for v in key],
                "val": float(coeff),
            }
        )
    return out
