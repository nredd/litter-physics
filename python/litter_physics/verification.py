"""Bounded spatial and temporal refinement studies against the compiled kernel.

A study runs one research fixture at three or more resolutions along one axis at a
time. The spatial axis varies `research.grid_spacing_m` with a fixed, small `dt_s`;
the temporal axis varies `dt_s` on a fixed grid. Duration, geometry, material and
seed are inherited from the base request and never vary inside an axis, so the two
error sources are not conflated.

Every case is an ordinary run directory under the per-job wall ceiling, and the
whole study is capped by an explicit total budget. Case artifacts are kept whatever
the outcome. The report separates study COMPLETION from criterion SUCCESS and from
any proof of convergence: a passing study means the named observables changed by
less than the threshold between the two finest resolutions, nothing more. This is
numerical verification of the solver; it is never measured validation and never a
reason to promote a response table.

References:
    docs/plan.md (Research numerical model; Acceptance)
    docs/research.md
    docs/verification.md
    https://doi.org/10.1115/1.2960953 (ASME V&V 20 refinement terminology)
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litter_physics.artifacts import OUTPUT_FILE, RunDirectory, RunStatus, write_new_json
from litter_physics.models import (
    MAX_WALL_TIME_S,
    Fidelity,
    Finite,
    Fixture,
    Mode,
    Positive,
    SimulationRequest,
    canonical_json,
    load_document,
    validate_payload,
)
from litter_physics.native import NativeRunner, NativeSimulationError
from litter_physics.provenance import HardwareInfo, build_provenance, hash_request
from litter_physics.runner import BudgetError, RunResult, start_run
from litter_physics.sweep import set_by_path

LOGGER = logging.getLogger(__name__)

MIN_RESOLUTIONS = 3
MAX_RESOLUTIONS = 6
MAX_STUDY_WALL_TIME_S = 12.0 * MAX_WALL_TIME_S
MAX_RELATIVE_CHANGE = 0.05
MIN_CASE_BUDGET_S = 2.0
# The kernel clips the final step of a record interval to the remaining time, which
# can exceed the cap by roundoff; anything beyond this means the limiter governed.
DT_BINDING_REL_TOL = 1e-6
UNIFORM_RATIO_REL_TOL = 1e-9
STEP_COUNT_OBSERVABLE = "step_count"
MAX_DT_OBSERVABLE = "max_dt_s"
REJECTED_STEPS_OBSERVABLE = "rejected_steps"
LIMITED_STEPS_OBSERVABLE = "limited_steps"
COUNT_OBSERVABLES = (STEP_COUNT_OBSERVABLE, REJECTED_STEPS_OBSERVABLE, LIMITED_STEPS_OBSERVABLE)
GRID_SPACING_PATH = "research.grid_spacing_m"
DT_PATH = "dt_s"
SPEC_FILE = "study_spec.json"
REPORT_FILE = "study_report.json"
CASES_DIR = "cases"
# Mirrors of native caps from `core/src/research.rs` and `core/src/mpm/fixtures.rs`.
# The native check is authoritative; these exist so a study fails closed before any
# case starts instead of burning budget on a request the kernel will reject.
NATIVE_MAX_PARTICLES = 100_000
NATIVE_MAX_NODES = 1_000_000
NATIVE_MAX_FRAME_POINTS = 2_000_000
NATIVE_PARTICLES_PER_AXIS = 2
NATIVE_MIN_CELLS_PER_PELLET_RADIUS = 1.5
NATIVE_FRAME_POINT_OVERHEAD = 64
GRID_FIT_REL_TOL = 1e-6
STANDING_CLAIM = (
    "Numerical refinement verification of the compiled research kernel only. Passing "
    "means the named observables changed by less than the threshold between the two "
    "finest resolutions on one axis; it is not proof of convergence, not measured "
    "surrogate or household validation, and not grounds to promote a response table. "
    "No bound on absolute analytic-reference error is established."
)
OMITTED_GATES = (
    "absolute analytic-reference accuracy",
    "domain-size refinement",
    "combined space-time refinement",
    "asymptotic-range (Richardson) error estimation",
    "species residual acceptance",
    "measured clean-surrogate validation",
    "response-table promotion",
)


class StudyPlanError(ValueError):
    """A study cannot start because a case would violate a native constraint."""


class RefinementAxis(StrEnum):
    """Which discretization parameter an axis varies."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class CaseStatus(StrEnum):
    """Outcome of one case, a superset of `RunStatus`."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"
    NOT_STARTED = "not_started"


class ObservableOutcome(StrEnum):
    """Per-observable criterion result."""

    PASSED = "passed"
    UNRESOLVED = "unresolved"
    UNINFORMATIVE = "uninformative"
    MISSING = "missing"
    NONFINITE = "nonfinite"
    NOT_EVALUATED = "not_evaluated"


class ChangeTrend(StrEnum):
    """Shape of the successive absolute changes; a trend is not a convergence proof."""

    DECREASING = "decreasing"
    NON_MONOTONE = "non_monotone"
    UNDETERMINED = "undetermined"


class StudyOutcome(StrEnum):
    """Overall outcome of an axis or a whole study."""

    PASSED = "passed"
    UNRESOLVED = "unresolved"
    INCOMPLETE = "incomplete"


def _strictly_decreasing(values: list[float], name: str) -> None:
    """Require a strictly decreasing refinement sequence.

    Parameters:
        values (list[float]): Resolution parameters, coarse to fine.
        name (str): Field name for the message.

    Raises:
        ValueError: When the sequence is not strictly decreasing.
    """
    for index in range(1, len(values)):
        if values[index] >= values[index - 1]:
            raise ValueError(f"`{name}` must be strictly decreasing, coarse to fine: {values}")


class ObservableSpec(BaseModel):
    """One named observable and its documented near-zero absolute scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, description="Native observable key.")
    near_zero_scale: Positive = Field(
        description="Absolute scale in the observable's units below which the value is "
        "uninformative; also the denominator floor for relative changes."
    )
    note: str = Field(default="", description="Why this scale is appropriate.")


class SpatialAxisSpec(BaseModel):
    """Grid refinement at one fixed, small timestep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_spacings_m: list[Positive] = Field(
        min_length=MIN_RESOLUTIONS,
        max_length=MAX_RESOLUTIONS,
        description="Grid spacings in metres, coarse to fine.",
    )
    dt_s: Positive = Field(description="Fixed timestep cap shared by every spatial case.")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require coarse-to-fine ordering.

        Returns:
            Self: The validated axis.
        """
        _strictly_decreasing(self.grid_spacings_m, "grid_spacings_m")
        return self


class TemporalAxisSpec(BaseModel):
    """Timestep refinement on one fixed grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_spacing_m: Positive = Field(description="Fixed grid spacing shared by every case.")
    dt_values_s: list[Positive] = Field(
        min_length=MIN_RESOLUTIONS,
        max_length=MAX_RESOLUTIONS,
        description="Timestep caps in seconds, coarse to fine.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require coarse-to-fine ordering.

        Returns:
            Self: The validated axis.
        """
        _strictly_decreasing(self.dt_values_s, "dt_values_s")
        return self


class StudySpec(BaseModel):
    """Refinement study definition loaded from YAML or JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_request: str = Field(description="Research request path, relative to the spec.")
    label: str = Field(default="refinement", description="Human label stored in the report.")
    spatial: SpatialAxisSpec | None = Field(default=None, description="Spatial axis.")
    temporal: TemporalAxisSpec | None = Field(default=None, description="Temporal axis.")
    observables: list[ObservableSpec] = Field(
        min_length=1, description="Observables the criterion is evaluated on."
    )
    max_relative_change: float = Field(
        default=MAX_RELATIVE_CHANGE,
        gt=0,
        le=MAX_RELATIVE_CHANGE,
        description="Strict upper bound on the final-two relative change per observable.",
    )
    per_case_wall_budget_s: float = Field(
        gt=0, le=MAX_WALL_TIME_S, description="Wall ceiling for each case, at most one hour."
    )
    total_wall_budget_s: float = Field(
        gt=0,
        le=MAX_STUDY_WALL_TIME_S,
        description="Hard cap on wall time spent starting and running cases.",
    )

    @model_validator(mode="after")
    def _at_least_one_axis(self) -> Self:
        """Require an axis and unique observable names.

        Returns:
            Self: The validated spec.

        Raises:
            ValueError: When no axis is given or observable names repeat.
        """
        if self.spatial is None and self.temporal is None:
            raise ValueError("a study needs `spatial`, `temporal`, or both")
        names = [observable.name for observable in self.observables]
        if len(set(names)) != len(names):
            raise ValueError(f"`observables` names must be unique; got {names}")
        return self


class CaseOutcome(BaseModel):
    """Summary of one executed or skipped case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: RefinementAxis = Field(description="Axis the case belongs to.")
    case_index: int = Field(ge=0, description="Zero-based position, coarse to fine.")
    run_dir: str = Field(description="Run directory holding the case artifacts.")
    grid_spacing_m: float = Field(description="Grid spacing used.")
    dt_s: float = Field(description="Timestep cap used.")
    wall_budget_s: float | None = Field(description="Budget granted; None when not started.")
    status: CaseStatus = Field(description="Case status.")
    duration_s: Finite = Field(description="Requested simulated duration.")
    time_s: Finite | None = Field(description="Simulation time reached.")
    wall_time_s: Finite | None = Field(description="Wall time consumed.")
    fidelity: str | None = Field(description="Native fidelity label of the output.")
    observables: dict[str, Finite] = Field(description="Final native observables.")
    error: str | None = Field(description="Failure message when the case failed.")


class ObservableVerdict(BaseModel):
    """Criterion evaluation for one observable along one axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Observable key.")
    near_zero_scale: Finite = Field(description="Documented absolute scale.")
    values: list[Finite | None] = Field(
        description="Value per case, coarse to fine; None when missing or nonfinite."
    )
    absolute_changes: list[Finite | None] = Field(
        description="Successive absolute changes; None when undefined or overflowed."
    )
    relative_changes: list[Finite | None] = Field(
        description="Successive changes over max(|finer value|, near_zero_scale)."
    )
    final_relative_change: Finite | None = Field(description="Change between the finest two.")
    near_zero: bool = Field(description="Whether the finest value is below the scale.")
    trend: ChangeTrend = Field(description="Shape of successive changes; not a proof.")
    observed_order: float | None = Field(
        description="Apparent order from the last three cases when the refinement ratio "
        "is uniform and both changes are nonzero; an estimate, not an asymptotic claim."
    )
    outcome: ObservableOutcome = Field(description="Criterion result.")
    reasons: list[str] = Field(description="Why the outcome is not `passed`.")


class AxisReport(BaseModel):
    """One refinement axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: RefinementAxis = Field(description="Axis name.")
    varied_parameter: str = Field(description="Dotted request path that varies.")
    values: list[float] = Field(description="Varied parameter values, coarse to fine.")
    fixed_parameter: str = Field(description="Dotted request path held fixed.")
    fixed_value: float = Field(description="Fixed parameter value.")
    cases: list[CaseOutcome] = Field(description="Per-case outcomes.")
    complete: bool = Field(description="Whether every case completed.")
    timestep_cap_binding: bool | None = Field(
        description="Whether `max_dt_s` equalled `dt_s` in every completed case; None "
        "when the axis is incomplete."
    )
    observables: list[ObservableVerdict] = Field(description="Per-observable verdicts.")
    outcome: StudyOutcome = Field(description="Axis outcome.")
    reasons: list[str] = Field(description="Axis-level reasons for not passing.")


class StudyProvenance(BaseModel):
    """Source and build identity of the kernel that produced the study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_version: str = Field(description="`litter_physics` version.")
    python_code_sha256: str = Field(description="Hash of the Python package sources.")
    native_sha256: str | None = Field(description="Hash of the compiled extension.")
    git_commit: str | None = Field(description="Repository commit, when known.")
    git_dirty: bool | None = Field(description="Whether the working tree had changes.")
    python_version: str = Field(description="Interpreter version.")
    dependencies: dict[str, str] = Field(description="Tracked dependency versions.")
    hardware: HardwareInfo = Field(description="Producing host.")


class StudyReport(BaseModel):
    """Machine-readable study outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(description="Study label.")
    created_utc: str = Field(description="ISO-8601 creation time.")
    spec_sha256: str = Field(description="Hash of the canonical study spec.")
    base_request_sha256: str = Field(description="Hash of the canonical base request.")
    fixture: Fixture = Field(description="Research fixture under study.")
    material: str = Field(description="Continuum material name.")
    duration_s: float = Field(description="Simulated duration shared by every case.")
    seed: int = Field(description="Seed shared by every case.")
    fidelity: str | None = Field(
        description="Native fidelity label common to every completed case; None when no "
        "case completed or labels disagree."
    )
    provenance: StudyProvenance = Field(description="Build and source identity.")
    max_relative_change: float = Field(description="Criterion threshold.")
    axes: list[AxisReport] = Field(description="Evaluated axes.")
    study_complete: bool = Field(description="Every case on every axis completed.")
    criterion_met: bool = Field(description="Every observable passed on every axis.")
    outcome: StudyOutcome = Field(description="`passed`, `unresolved`, or `incomplete`.")
    reasons: list[str] = Field(description="Study-level reasons for not passing.")
    total_wall_budget_s: float = Field(description="Study cap.")
    total_wall_time_s: Finite = Field(
        description="Wall time from study start through evaluation, measured just before "
        "this report was assembled; the report write and log tail are not included."
    )
    total_budget_exceeded: bool = Field(
        description="Whether `total_wall_time_s` overran `total_wall_budget_s`. The cap is "
        "checked before each case starts and bounds each case's own budget; it does not "
        "preempt a running case or artifact finalization. Overruns are possible and "
        "are never a pass; there is no guaranteed bound on finalization time."
    )
    omitted_gates: list[str] = Field(description="Acceptance gates this study does not cover.")
    claim: str = Field(description="Standing statement of what a pass means.")


@dataclass(frozen=True)
class PlannedCase:
    """One validated case before execution."""

    axis: RefinementAxis
    case_index: int
    grid_spacing_m: float
    dt_s: float
    request: SimulationRequest


def load_study(path: Path) -> StudySpec:
    """Load and validate a study specification.

    Parameters:
        path (Path): YAML or JSON file.

    Returns:
        StudySpec: Validated specification.
    """
    return validate_payload(StudySpec, load_document(path))


def _near_integer(ratio: float, tolerance: float) -> bool:
    """Whether a ratio is an integer within a relative tolerance.

    Parameters:
        ratio (float): Value to test.
        tolerance (float): Relative tolerance scaled by max(rounded, 1).

    Returns:
        bool: True when close to an integer.
    """
    if not math.isfinite(ratio):
        return False
    rounded = round(ratio)
    return abs(ratio - rounded) <= tolerance * max(rounded, 1.0)


def check_grid_fit(request: SimulationRequest, label: str) -> None:
    """Fail closed on grid, block and resource constraints before any run.

    Parameters:
        request (SimulationRequest): Candidate research request.
        label (str): Case label for messages.

    Raises:
        StudyPlanError: When the kernel would reject the case or exceed a cap.
    """
    research = request.research
    if request.mode is not Mode.RESEARCH or research is None:
        raise StudyPlanError(f"{label}: refinement studies need a research request")
    spacing = research.grid_spacing_m
    sub = spacing / NATIVE_PARTICLES_PER_AXIS
    cells: list[int] = []
    particles_per_axis: list[int] = []
    for axis in range(3):
        domain = research.domain_m[axis]
        block = research.initial_size_m[axis]
        ratio = domain / spacing
        if not _near_integer(ratio, GRID_FIT_REL_TOL) or round(ratio) < 2:
            raise StudyPlanError(
                f"{label}: `domain_m[{axis}]` '{domain}' is not a >=2 integer multiple of "
                f"`grid_spacing_m` '{spacing}'"
            )
        cells.append(round(ratio))
        if block < spacing or block > domain or not _near_integer(block / sub, GRID_FIT_REL_TOL):
            raise StudyPlanError(
                f"{label}: `initial_size_m[{axis}]` '{block}' must span whole particle "
                f"subcells of '{sub}' and at least one cell of '{spacing}'"
            )
        particles_per_axis.append(max(round(block / sub), 1))
    nodes = math.prod(count + 3 for count in cells)
    if nodes > NATIVE_MAX_NODES:
        raise StudyPlanError(
            f"{label}: grid would allocate {nodes} padded nodes, above {NATIVE_MAX_NODES}"
        )
    particles = math.prod(particles_per_axis)
    if particles > NATIVE_MAX_PARTICLES:
        raise StudyPlanError(
            f"{label}: block would allocate {particles} particles, above {NATIVE_MAX_PARTICLES}"
        )
    frames = math.ceil(request.duration_s / request.record_interval_s) + 1
    frame_points = frames * (particles + NATIVE_FRAME_POINT_OVERHEAD)
    if frame_points > NATIVE_MAX_FRAME_POINTS:
        raise StudyPlanError(
            f"{label}: {frame_points} recorded point samples exceed {NATIVE_MAX_FRAME_POINTS}; "
            "increase `record_interval_s`"
        )
    if research.fixture is Fixture.COUPLED_PATCH:
        if len(request.boxes) != 1:
            raise StudyPlanError(f"{label}: `coupled_patch` needs exactly one geometry box")
        radius = request.boxes[0].pellet_radius_m
        if radius < NATIVE_MIN_CELLS_PER_PELLET_RADIUS * spacing:
            raise StudyPlanError(
                f"{label}: pellet radius '{radius}' spans fewer than "
                f"{NATIVE_MIN_CELLS_PER_PELLET_RADIUS} cells of '{spacing}'"
            )


def _case_request(base: dict[str, Any], grid_spacing_m: float, dt_s: float) -> SimulationRequest:
    """Build one case request from the base payload.

    Parameters:
        base (dict[str, Any]): Base request payload.
        grid_spacing_m (float): Grid spacing for the case.
        dt_s (float): Timestep cap for the case.

    Returns:
        SimulationRequest: Validated request.
    """
    payload = copy.deepcopy(base)
    set_by_path(payload, GRID_SPACING_PATH, grid_spacing_m)
    set_by_path(payload, DT_PATH, dt_s)
    return validate_payload(SimulationRequest, payload)


def plan_cases(spec: StudySpec, base: dict[str, Any]) -> list[PlannedCase]:
    """Expand and validate every case before any run starts.

    Parameters:
        spec (StudySpec): Study definition.
        base (dict[str, Any]): Base request payload.

    Returns:
        list[PlannedCase]: Spatial cases then temporal cases, coarse to fine.

    Raises:
        StudyPlanError: When a case is invalid or would violate a native constraint.
    """
    if base.get("mode") != Mode.RESEARCH or not isinstance(base.get("research"), dict):
        raise StudyPlanError("the base request must be a research request")
    pairs: list[tuple[RefinementAxis, int, float, float]] = []
    if spec.spatial is not None:
        pairs.extend(
            (RefinementAxis.SPATIAL, index, spacing, spec.spatial.dt_s)
            for index, spacing in enumerate(spec.spatial.grid_spacings_m)
        )
    if spec.temporal is not None:
        pairs.extend(
            (RefinementAxis.TEMPORAL, index, spec.temporal.grid_spacing_m, dt)
            for index, dt in enumerate(spec.temporal.dt_values_s)
        )
    cases: list[PlannedCase] = []
    for axis, index, spacing, dt in pairs:
        label = f"{axis} case {index} (grid_spacing_m={spacing}, dt_s={dt})"
        try:
            request = _case_request(base, spacing, dt)
        except (ValueError, KeyError, TypeError) as e:
            raise StudyPlanError(f"{label} is invalid: {e}") from e
        check_grid_fit(request, label)
        cases.append(PlannedCase(axis, index, spacing, dt, request))
    return cases


def _read_fidelity(run_dir: Path, segment_index: int) -> str | None:
    """Read the fidelity label committed with a segment.

    Parameters:
        run_dir (Path): Case run directory.
        segment_index (int): Committed segment.

    Returns:
        str | None: The label, or None when unreadable.
    """
    path = RunDirectory(run_dir).segment_dir(segment_index) / OUTPUT_FILE
    try:
        payload = load_document(path)
    except (OSError, ValueError, FileNotFoundError) as e:
        LOGGER.warning(f"cannot read fidelity from '{path}': {e}")
        return None
    fidelity = payload.get("fidelity")
    return fidelity if isinstance(fidelity, str) else None


def _outcome_from_result(
    case: PlannedCase, run_dir: Path, budget: float, result: RunResult
) -> CaseOutcome:
    """Convert a committed run into a case outcome.

    Parameters:
        case (PlannedCase): The planned case.
        run_dir (Path): Run directory.
        budget (float): Budget granted.
        result (RunResult): Committed result.

    Returns:
        CaseOutcome: The record.
    """
    status: RunStatus = result.run_status
    return CaseOutcome(
        axis=case.axis,
        case_index=case.case_index,
        run_dir=str(run_dir),
        grid_spacing_m=case.grid_spacing_m,
        dt_s=case.dt_s,
        wall_budget_s=budget,
        status=CaseStatus(str(status)),
        duration_s=case.request.duration_s,
        time_s=result.time_s,
        wall_time_s=result.wall_time_s,
        fidelity=_read_fidelity(run_dir, result.segment_index),
        observables=dict(result.observables),
        error=None,
    )


def _skipped_outcome(
    case: PlannedCase, run_dir: Path, budget: float | None, status: CaseStatus, error: str | None
) -> CaseOutcome:
    """Record a case that did not commit a completed segment.

    Parameters:
        case (PlannedCase): The planned case.
        run_dir (Path): Run directory that may or may not exist.
        budget (float | None): Budget granted, if any.
        status (CaseStatus): `failed` or `not_started`.
        error (str | None): Failure message.

    Returns:
        CaseOutcome: The record.
    """
    return CaseOutcome(
        axis=case.axis,
        case_index=case.case_index,
        run_dir=str(run_dir),
        grid_spacing_m=case.grid_spacing_m,
        dt_s=case.dt_s,
        wall_budget_s=budget,
        status=status,
        duration_s=case.request.duration_s,
        time_s=None,
        wall_time_s=None,
        fidelity=None,
        observables={},
        error=error,
    )


def _execute_cases(
    cases: list[PlannedCase],
    root: Path,
    spec: StudySpec,
    *,
    runner: NativeRunner | None,
    started: float,
) -> list[CaseOutcome]:
    """Run every case sequentially under the per-case and total caps.

    Parameters:
        cases (list[PlannedCase]): Planned cases.
        root (Path): Study directory.
        spec (StudySpec): Study definition.
        runner (NativeRunner | None): Injected native callable for tests.
        started (float): `time.monotonic()` at study start.

    Returns:
        list[CaseOutcome]: One outcome per case, in plan order.
    """
    outcomes: list[CaseOutcome] = []
    for case in cases:
        run_dir = root / CASES_DIR / str(case.axis) / f"{case.case_index:04d}"
        remaining = spec.total_wall_budget_s - (time.monotonic() - started)
        if remaining < MIN_CASE_BUDGET_S:
            LOGGER.warning(f"study budget exhausted before {case.axis} case {case.case_index}")
            outcomes.append(_skipped_outcome(case, run_dir, None, CaseStatus.NOT_STARTED, None))
            continue
        budget = min(spec.per_case_wall_budget_s, remaining)
        LOGGER.info(
            f"{case.axis} case {case.case_index}: grid_spacing_m={case.grid_spacing_m} "
            f"dt_s={case.dt_s} budget={budget:.1f}s"
        )
        try:
            result = start_run(
                case.request, run_dir, runner=runner, wall_budget_s=budget, recording=False
            )
        except (BudgetError, NativeSimulationError, ValueError, RuntimeError) as e:
            LOGGER.error(f"{case.axis} case {case.case_index} failed: {e}")
            outcomes.append(_skipped_outcome(case, run_dir, budget, CaseStatus.FAILED, str(e)))
            continue
        outcomes.append(_outcome_from_result(case, run_dir, budget, result))
    return outcomes


def _finite(value: float | None) -> bool:
    """Whether a value is present and finite.

    Parameters:
        value (float | None): Candidate.

    Returns:
        bool: True for finite floats.
    """
    return value is not None and math.isfinite(value)


def _observed_order(parameters: list[float], changes: list[float | None]) -> float | None:
    """Estimate the apparent order from the last three resolutions.

    Parameters:
        parameters (list[float]): Refinement parameters, coarse to fine.
        changes (list[float | None]): Successive absolute changes.

    Returns:
        float | None: `log(d_prev / d_last) / log(r)` when the last two refinement ratios
            agree and both changes are positive and finite; otherwise None.
    """
    if len(parameters) < 3 or len(changes) < 2:
        return None
    ratio_prev = parameters[-3] / parameters[-2]
    ratio_last = parameters[-2] / parameters[-1]
    if not all(math.isfinite(ratio) and ratio > 1.0 for ratio in (ratio_prev, ratio_last)):
        return None
    if abs(ratio_prev - ratio_last) > UNIFORM_RATIO_REL_TOL * ratio_last:
        return None
    previous, last = changes[-2], changes[-1]
    if not (_finite(previous) and _finite(last)) or previous is None or last is None:
        return None
    if previous <= 0.0 or last <= 0.0:
        return None
    quotient = previous / last
    if not math.isfinite(quotient) or quotient <= 0.0:
        return None
    order = math.log(quotient) / math.log(ratio_last)
    return order if math.isfinite(order) else None


def evaluate_observable(
    observable: ObservableSpec,
    parameters: list[float],
    values: list[float | None],
    max_relative_change: float,
) -> ObservableVerdict:
    """Apply the final-two relative-change criterion to one observable.

    Parameters:
        observable (ObservableSpec): Name and near-zero scale.
        parameters (list[float]): Refinement parameters, coarse to fine.
        values (list[float | None]): Observable per case; None when absent.
        max_relative_change (float): Strict threshold.

    Returns:
        ObservableVerdict: The verdict with every intermediate quantity.

    Raises:
        ValueError: When the lengths disagree or fewer than three cases are given.
    """
    if len(parameters) != len(values) or len(values) < MIN_RESOLUTIONS:
        raise ValueError(
            f"`{observable.name}` needs at least {MIN_RESOLUTIONS} parameter/value pairs"
        )
    scale = observable.near_zero_scale
    reasons: list[str] = []
    absolute: list[float | None] = []
    relative: list[float | None] = []
    overflowed = False
    for index in range(1, len(values)):
        coarse, fine = values[index - 1], values[index]
        if not (_finite(coarse) and _finite(fine)) or coarse is None or fine is None:
            absolute.append(None)
            relative.append(None)
            continue
        change = abs(fine - coarse)
        ratio = change / max(abs(fine), scale)
        # Finite extrema can still overflow the difference or the quotient; a
        # nonfinite derived quantity must not reach the report or count as a pass.
        if not (math.isfinite(change) and math.isfinite(ratio)):
            overflowed = True
            absolute.append(None)
            relative.append(None)
            continue
        absolute.append(change)
        relative.append(ratio)
    outcome = ObservableOutcome.PASSED
    if any(value is None for value in values):
        outcome = ObservableOutcome.MISSING
        reasons.append("observable is missing from at least one case")
    elif any(not _finite(value) for value in values):
        outcome = ObservableOutcome.NONFINITE
        reasons.append("observable is nonfinite in at least one case")
    elif overflowed:
        outcome = ObservableOutcome.NONFINITE
        reasons.append("a derived change or ratio overflowed to a nonfinite value")
    finest = values[-1]
    near_zero = _finite(finest) and finest is not None and abs(finest) < scale
    final = relative[-1]
    if outcome is ObservableOutcome.PASSED:
        if near_zero:
            outcome = ObservableOutcome.UNINFORMATIVE
            reasons.append(
                f"finest value {finest} is below the near-zero scale {scale}; a relative "
                "change against a near-zero denominator cannot demonstrate resolution"
            )
        elif final is None or not final < max_relative_change:
            outcome = ObservableOutcome.UNRESOLVED
            reasons.append(f"final-two relative change {final} is not below {max_relative_change}")
    trend = ChangeTrend.UNDETERMINED
    finite_changes = [change for change in absolute if change is not None]
    if len(finite_changes) == len(absolute) and len(finite_changes) >= 2:
        decreasing = all(later < earlier for earlier, later in itertools.pairwise(finite_changes))
        trend = ChangeTrend.DECREASING if decreasing else ChangeTrend.NON_MONOTONE
    return ObservableVerdict(
        name=observable.name,
        near_zero_scale=scale,
        values=[value if _finite(value) else None for value in values],
        absolute_changes=absolute,
        relative_changes=relative,
        final_relative_change=final,
        near_zero=near_zero,
        trend=trend,
        observed_order=_observed_order(parameters, absolute),
        outcome=outcome,
        reasons=reasons,
    )


def _timestep_checks(axis: RefinementAxis, cases: list[CaseOutcome]) -> tuple[bool, list[str]]:
    """Check that the requested timestep cap governed EVERY accepted step.

    `max_dt_s == dt_s` only shows the cap bound at least once. The kernel's
    `limited_steps` counts accepted steps where the stability limit undercut the cap
    before final/record clipping; together with `rejected_steps` it must be zero, and
    every diagnostic must be a finite nonnegative integer. Unknown diagnostics fail
    closed.

    Parameters:
        axis (RefinementAxis): Axis under evaluation.
        cases (list[CaseOutcome]): Completed cases, coarse to fine.

    Returns:
        tuple[bool, list[str]]: Whether every cap bound throughout, and the reasons.
    """
    reasons: list[str] = []
    binding = True
    steps: list[float] = []
    for case in cases:
        label = f"{axis} case {case.case_index}"
        counts: dict[str, float] = {}
        for name in COUNT_OBSERVABLES:
            value = case.observables.get(name)
            if not _finite(value) or value is None or value < 0.0 or value != math.floor(value):
                binding = False
                reasons.append(
                    f"{label}: `{name}` is {value!r}, not a finite nonnegative integer; the "
                    "kernel did not report the adaptive-step diagnostics this axis needs"
                )
                continue
            counts[name] = value
        step_count = counts.get(STEP_COUNT_OBSERVABLE)
        if step_count is not None:
            if step_count <= 0.0:
                binding = False
                reasons.append(f"{label}: `{STEP_COUNT_OBSERVABLE}` is zero")
            steps.append(step_count)
        for name in (REJECTED_STEPS_OBSERVABLE, LIMITED_STEPS_OBSERVABLE):
            count = counts.get(name)
            if count is not None and count > 0.0:
                binding = False
                reasons.append(
                    f"{label}: `{name}` {count:g} > 0; the adaptive limiter, not the requested "
                    "cap, governed at least one step, so the axis does not isolate one "
                    "discretization parameter"
                )
        max_dt = case.observables.get(MAX_DT_OBSERVABLE)
        if (
            not _finite(max_dt)
            or max_dt is None
            or abs(max_dt - case.dt_s) > DT_BINDING_REL_TOL * case.dt_s
        ):
            binding = False
            reasons.append(
                f"{label}: `{MAX_DT_OBSERVABLE}` {max_dt!r} does not equal `dt_s` {case.dt_s}; "
                "the requested cap never governed a full step"
            )
    if axis is RefinementAxis.TEMPORAL and len(steps) == len(cases):
        increasing = all(later > earlier for earlier, later in itertools.pairwise(steps))
        if not increasing:
            binding = False
            reasons.append(
                f"temporal `{STEP_COUNT_OBSERVABLE}` did not strictly increase: {steps}"
            )
    return binding, reasons


def _case_integrity(axis: RefinementAxis, cases: list[CaseOutcome]) -> list[str]:
    """Reject completed cases whose output is not a full research run.

    Parameters:
        axis (RefinementAxis): Axis under evaluation.
        cases (list[CaseOutcome]): Completed cases.

    Returns:
        list[str]: Reasons; empty when every case reached its duration with the
            `research_unvalidated` fidelity label.
    """
    reasons: list[str] = []
    for case in cases:
        label = f"{axis} case {case.case_index}"
        if case.time_s is None or case.time_s != case.duration_s:
            reasons.append(
                f"{label}: completed output reached time {case.time_s!r}, not the requested "
                f"duration {case.duration_s}"
            )
        if case.fidelity != Fidelity.RESEARCH_UNVALIDATED:
            reasons.append(
                f"{label}: fidelity {case.fidelity!r} is not "
                f"'{Fidelity.RESEARCH_UNVALIDATED}'; only research kernel output can be refined"
            )
    return reasons


def evaluate_axis(
    axis: RefinementAxis,
    parameters: list[float],
    fixed: tuple[str, float],
    cases: list[CaseOutcome],
    observables: list[ObservableSpec],
    max_relative_change: float,
) -> AxisReport:
    """Evaluate one axis from its case outcomes.

    Parameters:
        axis (RefinementAxis): Axis name.
        parameters (list[float]): Varied parameter values, coarse to fine.
        fixed (tuple[str, float]): Fixed parameter path and value.
        cases (list[CaseOutcome]): Case outcomes in the same order.
        observables (list[ObservableSpec]): Observables to evaluate.
        max_relative_change (float): Strict threshold.

    Returns:
        AxisReport: The axis report.
    """
    varied = GRID_SPACING_PATH if axis is RefinementAxis.SPATIAL else DT_PATH
    complete = len(cases) == len(parameters) and all(
        case.status is CaseStatus.COMPLETED for case in cases
    )
    reasons: list[str] = []
    if not complete:
        incomplete = [
            f"case {case.case_index} {case.status}"
            for case in cases
            if case.status is not CaseStatus.COMPLETED
        ]
        reasons.append(f"axis incomplete: {', '.join(incomplete) or 'cases missing'}")
        verdicts = [
            ObservableVerdict(
                name=observable.name,
                near_zero_scale=observable.near_zero_scale,
                values=[case.observables.get(observable.name) for case in cases],
                absolute_changes=[],
                relative_changes=[],
                final_relative_change=None,
                near_zero=False,
                trend=ChangeTrend.UNDETERMINED,
                observed_order=None,
                outcome=ObservableOutcome.NOT_EVALUATED,
                reasons=["axis incomplete; criterion not evaluated"],
            )
            for observable in observables
        ]
        return AxisReport(
            axis=axis,
            varied_parameter=varied,
            values=parameters,
            fixed_parameter=fixed[0],
            fixed_value=fixed[1],
            cases=cases,
            complete=False,
            timestep_cap_binding=None,
            observables=verdicts,
            outcome=StudyOutcome.INCOMPLETE,
            reasons=reasons,
        )
    integrity_reasons = _case_integrity(axis, cases)
    reasons.extend(integrity_reasons)
    binding, timestep_reasons = _timestep_checks(axis, cases)
    reasons.extend(timestep_reasons)
    verdicts = [
        evaluate_observable(
            observable,
            parameters,
            [case.observables.get(observable.name) for case in cases],
            max_relative_change,
        )
        for observable in observables
    ]
    passed = (
        binding
        and not integrity_reasons
        and all(verdict.outcome is ObservableOutcome.PASSED for verdict in verdicts)
    )
    for verdict in verdicts:
        if verdict.outcome is not ObservableOutcome.PASSED:
            reasons.append(f"`{verdict.name}` {verdict.outcome}")
    return AxisReport(
        axis=axis,
        varied_parameter=varied,
        values=parameters,
        fixed_parameter=fixed[0],
        fixed_value=fixed[1],
        cases=cases,
        complete=True,
        timestep_cap_binding=binding,
        observables=verdicts,
        outcome=StudyOutcome.PASSED if passed else StudyOutcome.UNRESOLVED,
        reasons=reasons,
    )


def _common_fidelity(outcomes: list[CaseOutcome]) -> str | None:
    """Return the fidelity label shared by every completed case.

    Parameters:
        outcomes (list[CaseOutcome]): All case outcomes.

    Returns:
        str | None: The common label, or None when absent or inconsistent.
    """
    labels = {outcome.fidelity for outcome in outcomes if outcome.status is CaseStatus.COMPLETED}
    if len(labels) != 1:
        return None
    return labels.pop()


def build_report(
    spec: StudySpec,
    base: SimulationRequest,
    outcomes: list[CaseOutcome],
    *,
    spec_sha256: str,
    started: float,
) -> StudyReport:
    """Evaluate every axis and assemble the study report.

    Parameters:
        spec (StudySpec): Study definition.
        base (SimulationRequest): Validated base request.
        outcomes (list[CaseOutcome]): Case outcomes from `_execute_cases`.
        spec_sha256 (str): Hash of the canonical spec.
        started (float): `time.monotonic()` at study start; the total is measured
            after evaluation and provenance capture, just before the report model
            is assembled.

    Returns:
        StudyReport: The report.

    Raises:
        ValueError: When the base request is not a research request.
    """
    if base.research is None:
        raise ValueError("the base request must be a research request")
    axes: list[AxisReport] = []
    if spec.spatial is not None:
        axes.append(
            evaluate_axis(
                RefinementAxis.SPATIAL,
                list(spec.spatial.grid_spacings_m),
                (DT_PATH, spec.spatial.dt_s),
                [o for o in outcomes if o.axis is RefinementAxis.SPATIAL],
                spec.observables,
                spec.max_relative_change,
            )
        )
    if spec.temporal is not None:
        axes.append(
            evaluate_axis(
                RefinementAxis.TEMPORAL,
                list(spec.temporal.dt_values_s),
                (GRID_SPACING_PATH, spec.temporal.grid_spacing_m),
                [o for o in outcomes if o.axis is RefinementAxis.TEMPORAL],
                spec.observables,
                spec.max_relative_change,
            )
        )
    complete = all(axis.complete for axis in axes)
    fidelity = _common_fidelity(outcomes)
    reasons: list[str] = []
    if fidelity != Fidelity.RESEARCH_UNVALIDATED:
        reasons.append(f"common fidelity {fidelity!r} is not '{Fidelity.RESEARCH_UNVALIDATED}'")
    provenance = build_provenance(base)
    total_wall_time_s = time.monotonic() - started
    exceeded = total_wall_time_s > spec.total_wall_budget_s
    if exceeded:
        reasons.append(
            f"total wall time {total_wall_time_s:.1f}s exceeded the study cap "
            f"{spec.total_wall_budget_s:.1f}s; the cap only gates case starts and bounds "
            "case budgets, so this overrun is reported rather than prevented"
        )
    criterion = (
        complete and not reasons and all(axis.outcome is StudyOutcome.PASSED for axis in axes)
    )
    if not complete:
        outcome = StudyOutcome.INCOMPLETE
    elif criterion:
        outcome = StudyOutcome.PASSED
    else:
        outcome = StudyOutcome.UNRESOLVED
    return StudyReport(
        label=spec.label,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        spec_sha256=spec_sha256,
        base_request_sha256=hash_request(base),
        fixture=base.research.fixture,
        material=str(base.research.material),
        duration_s=base.duration_s,
        seed=base.seed,
        fidelity=fidelity,
        provenance=StudyProvenance(
            package_version=provenance.package_version,
            python_code_sha256=provenance.python_code_sha256,
            native_sha256=provenance.native_sha256,
            git_commit=provenance.git_commit,
            git_dirty=provenance.git_dirty,
            python_version=provenance.python_version,
            dependencies=provenance.dependencies,
            hardware=provenance.hardware,
        ),
        max_relative_change=spec.max_relative_change,
        axes=axes,
        study_complete=complete,
        criterion_met=criterion,
        outcome=outcome,
        reasons=reasons,
        total_wall_budget_s=spec.total_wall_budget_s,
        total_wall_time_s=total_wall_time_s,
        total_budget_exceeded=exceeded,
        omitted_gates=list(OMITTED_GATES),
        claim=STANDING_CLAIM,
    )


def hash_spec(spec: StudySpec) -> str:
    """Hash the canonical JSON form of a study spec.

    Parameters:
        spec (StudySpec): Validated spec.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(canonical_json(spec.model_dump(mode="json")).encode()).hexdigest()


def run_study(
    spec_path: Path, study_dir: Path, *, runner: NativeRunner | None = None
) -> StudyReport:
    """Plan, execute and evaluate a refinement study.

    Parameters:
        spec_path (Path): Study specification file.
        study_dir (Path): New directory for `cases/<axis>/NNNN` runs and the report.
        runner (NativeRunner | None): Injected native callable for tests.

    Returns:
        StudyReport: The written report.

    Raises:
        FileExistsError: When the study directory already has content.
        StudyPlanError: When any case would violate a constraint; nothing runs.
    """
    started = time.monotonic()
    spec = load_study(spec_path)
    base_path = (Path(str(spec_path)).resolve().parent / spec.base_request).resolve()
    base_payload = load_document(base_path)
    cases = plan_cases(spec, base_payload)
    base = validate_payload(SimulationRequest, base_payload)
    root = Path(str(study_dir)).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Study directory '{root}' is not empty; refusing to overwrite")
    root.mkdir(parents=True, exist_ok=True)
    write_new_json(root / SPEC_FILE, spec.model_dump(mode="json"))
    outcomes = _execute_cases(cases, root, spec, runner=runner, started=started)
    report = build_report(
        spec,
        base,
        outcomes,
        spec_sha256=hash_spec(spec),
        started=started,
    )
    write_new_json(root / REPORT_FILE, report.model_dump(mode="json"))
    LOGGER.info(
        f"study '{spec.label}' {report.outcome}: complete={report.study_complete} "
        f"criterion_met={report.criterion_met} report='{root / REPORT_FILE}'"
    )
    return report
