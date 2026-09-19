"""Parameter sweeps as independent, per-job gated runs.

A sweep specification names a base request and dotted-path overrides. The grid
product produces one job per combination; each job is a normal run directory with
its own one-hour ceiling. A sweep-level wall budget only stops STARTING jobs; it never
extends or shortens a job's own budget, so multi-job scenarios cannot evade per-job
gates.

References:
    docs/plan.md (Scope and claims)
"""

from __future__ import annotations

import copy
import itertools
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from litter_physics.artifacts import RunStatus, write_atomic_json, write_parquet_new
from litter_physics.models import (
    SimulationRequest,
    load_document,
    validate_payload,
)
from litter_physics.native import NativeRunner, NativeSimulationError
from litter_physics.runner import BudgetError, RunResult, start_run

LOGGER = logging.getLogger(__name__)

MAX_SWEEP_JOBS = 10_000
SUMMARY_FILE = "sweep_summary.json"
SUMMARY_TABLE = "sweep_summary.parquet"
JOBS_DIR = "jobs"


class SweepAxis(BaseModel):
    """One swept parameter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        min_length=1,
        description="Dotted path into the request, e.g. `materials.friction` or "
        "`boxes.0.pellet_count`.",
    )
    values: list[Any] = Field(min_length=1, description="Values to substitute, in order.")


class SweepSpec(BaseModel):
    """Sweep definition loaded from YAML or JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_request: str = Field(description="Path to the base request, relative to the spec.")
    axes: list[SweepAxis] = Field(min_length=1, description="Swept axes; grid product.")
    seeds: list[int] = Field(
        default_factory=lambda: [0], description="Seed offsets added to the base seed."
    )
    label: str = Field(default="sweep", description="Human label stored in the summary.")


class JobOutcome(BaseModel):
    """Summary of one sweep job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_index: int = Field(description="Zero-based job index.")
    run_dir: str = Field(description="Run directory path.")
    overrides: dict[str, Any] = Field(description="Applied dotted-path overrides.")
    seed: int = Field(description="Effective seed.")
    status: str = Field(
        description="`completed`, `budget_exhausted`, `deadline_exceeded`, "
        "`failed`, or `not_started`."
    )
    time_s: float | None = Field(description="Simulation time reached.")
    wall_time_s: float | None = Field(description="Wall time consumed.")
    observables: dict[str, float] = Field(description="Observables from the last segment.")
    error: str | None = Field(description="Failure message when the job failed.")


def set_by_path(payload: dict[str, Any], path: str, value: Any) -> None:
    """Assign into a nested JSON-like tree by dotted path.

    Parameters:
        payload (dict[str, Any]): Tree to mutate.
        path (str): Dotted path; integer segments index lists.
        value (Any): Value to assign.

    Raises:
        KeyError: When an intermediate key is missing.
        IndexError: When a list index is out of range.
        TypeError: When a segment does not match the container type.
    """
    parts = path.split(".")
    node: Any = payload
    for part in parts[:-1]:
        node = _descend(node, part, path)
    last = parts[-1]
    if isinstance(node, list):
        node[_list_index(last, path)] = value
    elif isinstance(node, dict):
        if last not in node:
            raise KeyError(f"path '{path}' names unknown key '{last}'")
        node[last] = value
    else:
        raise TypeError(f"path '{path}' descends into a scalar at '{last}'")


def _descend(node: Any, part: str, path: str) -> Any:
    """Step one segment into a tree.

    Parameters:
        node (Any): Current container.
        part (str): Path segment.
        path (str): Full path for messages.

    Returns:
        Any: The child container.

    Raises:
        KeyError: When a mapping key is missing.
        TypeError: When the node is a scalar.
    """
    if isinstance(node, list):
        return node[_list_index(part, path)]
    if isinstance(node, dict):
        if part not in node:
            raise KeyError(f"path '{path}' names unknown key '{part}'")
        return node[part]
    raise TypeError(f"path '{path}' descends into a scalar at '{part}'")


def _list_index(part: str, path: str) -> int:
    """Parse a list index segment.

    Parameters:
        part (str): Segment text.
        path (str): Full path for messages.

    Returns:
        int: The index.

    Raises:
        TypeError: When the segment is not an integer.
    """
    if not part.isdigit():
        raise TypeError(f"path '{path}' indexes a list with non-integer '{part}'")
    return int(part)


def load_spec(path: Path) -> SweepSpec:
    """Load and validate a sweep specification.

    Parameters:
        path (Path): YAML or JSON file.

    Returns:
        SweepSpec: Validated specification.
    """
    return validate_payload(SweepSpec, load_document(path))


def expand_jobs(
    spec: SweepSpec, base: dict[str, Any]
) -> list[tuple[dict[str, Any], int, SimulationRequest]]:
    """Expand the grid into validated requests before any run starts.

    Parameters:
        spec (SweepSpec): Sweep definition.
        base (dict[str, Any]): Base request payload.

    Returns:
        list[tuple[dict[str, Any], int, SimulationRequest]]: Overrides, seed, request.

    Raises:
        ValueError: When the grid is too large or any job is invalid.
    """
    combos = list(itertools.product(*[axis.values for axis in spec.axes]))
    total = len(combos) * len(spec.seeds)
    if total > MAX_SWEEP_JOBS:
        raise ValueError(f"sweep expands to {total} jobs, above the cap {MAX_SWEEP_JOBS}")
    jobs: list[tuple[dict[str, Any], int, SimulationRequest]] = []
    for combo in combos:
        for seed_offset in spec.seeds:
            payload = copy.deepcopy(base)
            overrides = {axis.path: value for axis, value in zip(spec.axes, combo, strict=True)}
            for path, value in overrides.items():
                set_by_path(payload, path, value)
            seed = int(base["seed"]) + seed_offset
            payload["seed"] = seed
            try:
                request = validate_payload(SimulationRequest, payload)
            except ValueError as e:
                raise ValueError(f"sweep job with overrides {overrides} is invalid: {e}") from e
            jobs.append((overrides, seed, request))
    return jobs


def run_sweep(
    spec_path: Path,
    sweep_dir: Path,
    *,
    runner: NativeRunner | None = None,
    total_wall_budget_s: float | None = None,
    recording: bool = False,
) -> list[JobOutcome]:
    """Execute every job sequentially with per-job gates intact.

    Parameters:
        spec_path (Path): Sweep specification file.
        sweep_dir (Path): New directory to hold `jobs/NNNN` run directories.
        runner (NativeRunner | None): Injected native callable for tests.
        total_wall_budget_s (float | None): Stop starting new jobs after this many
            seconds; jobs already running keep their own budget.
        recording (bool): Whether each job writes a Rerun recording.

    Returns:
        list[JobOutcome]: One outcome per expanded job.

    Raises:
        FileExistsError: When the sweep directory already has content.
    """
    started = time.monotonic()
    spec = load_spec(spec_path)
    base_path = (Path(str(spec_path)).resolve().parent / spec.base_request).resolve()
    base = load_document(base_path)
    jobs = expand_jobs(spec, base)
    root = Path(str(sweep_dir)).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Sweep directory '{root}' is not empty; refusing to overwrite")
    root.mkdir(parents=True, exist_ok=True)
    outcomes: list[JobOutcome] = []
    for job_index, (overrides, seed, request) in enumerate(jobs):
        run_dir = root / JOBS_DIR / f"{job_index:04d}"
        elapsed = time.monotonic() - started
        if total_wall_budget_s is not None and elapsed >= total_wall_budget_s:
            outcomes.append(
                _outcome(job_index, run_dir, overrides, seed, "not_started", None, None, {}, None)
            )
            continue
        LOGGER.info(f"sweep job {job_index}/{len(jobs)} overrides={overrides} seed={seed}")
        try:
            result = start_run(request, run_dir, runner=runner, recording=recording)
        except (BudgetError, NativeSimulationError, ValueError, RuntimeError) as e:
            LOGGER.error(f"sweep job {job_index} failed: {e}")
            outcomes.append(
                _outcome(job_index, run_dir, overrides, seed, "failed", None, None, {}, str(e))
            )
            continue
        outcomes.append(_outcome_from_result(job_index, run_dir, overrides, seed, result))
    _write_summary(root, spec, outcomes)
    return outcomes


def _outcome(
    job_index: int,
    run_dir: Path,
    overrides: dict[str, Any],
    seed: int,
    status: str,
    time_s: float | None,
    wall_time_s: float | None,
    observables: dict[str, float],
    error: str | None,
) -> JobOutcome:
    """Construct an outcome record.

    Parameters:
        job_index (int): Job index.
        run_dir (Path): Run directory.
        overrides (dict[str, Any]): Applied overrides.
        seed (int): Effective seed.
        status (str): Outcome status.
        time_s (float | None): Simulation time reached.
        wall_time_s (float | None): Wall time consumed.
        observables (dict[str, float]): Final observables.
        error (str | None): Failure message.

    Returns:
        JobOutcome: The record.
    """
    return JobOutcome(
        job_index=job_index,
        run_dir=str(run_dir),
        overrides=overrides,
        seed=seed,
        status=status,
        time_s=time_s,
        wall_time_s=wall_time_s,
        observables=observables,
        error=error,
    )


def _outcome_from_result(
    job_index: int, run_dir: Path, overrides: dict[str, Any], seed: int, result: RunResult
) -> JobOutcome:
    """Convert a run result into an outcome record.

    Parameters:
        job_index (int): Job index.
        run_dir (Path): Run directory.
        overrides (dict[str, Any]): Applied overrides.
        seed (int): Effective seed.
        result (RunResult): Committed run result.

    Returns:
        JobOutcome: The record.
    """
    status: RunStatus = result.run_status
    return _outcome(
        job_index,
        run_dir,
        overrides,
        seed,
        str(status),
        result.time_s,
        result.wall_time_s,
        result.observables,
        None,
    )


def _write_summary(root: Path, spec: SweepSpec, outcomes: list[JobOutcome]) -> None:
    """Write JSON and Parquet summaries of the sweep.

    Parameters:
        root (Path): Sweep directory.
        spec (SweepSpec): The specification.
        outcomes (list[JobOutcome]): Per-job outcomes.
    """
    payload = {
        "label": spec.label,
        "axes": [axis.model_dump() for axis in spec.axes],
        "jobs": [outcome.model_dump() for outcome in outcomes],
        "counts": {
            status: sum(1 for outcome in outcomes if outcome.status == status)
            for status in sorted({outcome.status for outcome in outcomes})
        },
    }
    write_atomic_json(root / SUMMARY_FILE, payload)
    rows = {
        "job_index": [outcome.job_index for outcome in outcomes],
        "seed": [outcome.seed for outcome in outcomes],
        "status": [outcome.status for outcome in outcomes],
        "time_s": [outcome.time_s for outcome in outcomes],
        "wall_time_s": [outcome.wall_time_s for outcome in outcomes],
        "run_dir": [outcome.run_dir for outcome in outcomes],
    }
    for axis in spec.axes:
        rows[f"override:{axis.path}"] = [
            str(outcome.overrides.get(axis.path)) for outcome in outcomes
        ]
    write_parquet_new(root / SUMMARY_TABLE, pa.table(rows))
    LOGGER.info(f"sweep summary written to '{root / SUMMARY_FILE}'")


def read_summary(root: Path) -> pa.Table:
    """Read the Parquet summary of a sweep.

    Parameters:
        root (Path): Sweep directory.

    Returns:
        pa.Table: Summary rows.
    """
    return pq.read_table(Path(str(root)).resolve() / SUMMARY_TABLE)
