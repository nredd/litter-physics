"""Bounded execution of one native job with resumable, append-only artifacts.

The per-job wall budget covers setup, the native call, and artifact construction.
The native solver receives only the remaining budget minus an output reserve. If the
whole segment overruns the hard deadline, the segment is still committed (so it can
be resumed) but its status is `deadline_exceeded`, never `completed`.

References:
    docs/plan.md (Scope and claims; Artifacts and interfaces)
    docs/wire-contract.md
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from litter_physics.artifacts import (
    PROVENANCE_FILE,
    RESUMABLE_STATUSES,
    ArtifactError,
    RunDirectory,
    RunStatus,
    SegmentRecord,
)
from litter_physics.models import (
    MAX_WALL_TIME_S,
    Checkpoint,
    SimulationOutput,
    SimulationRequest,
    Status,
)
from litter_physics.native import NativeRunner, run_segment
from litter_physics.provenance import Provenance, build_provenance, hash_request
from litter_physics.viewer import recording_file_for, write_segment_recording

LOGGER = logging.getLogger(__name__)

MIN_OUTPUT_RESERVE_S = 1.0
OUTPUT_RESERVE_FRACTION = 0.05
MIN_NATIVE_BUDGET_S = 0.05


class BudgetError(RuntimeError):
    """The job cannot start because its wall budget is already spent."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of one executed segment."""

    run_dir: Path
    segment_index: int
    run_status: RunStatus
    native_status: Status
    time_s: float
    duration_s: float
    wall_time_s: float
    native_wall_time_s: float
    diagnostics: list[str]
    observables: dict[str, float]

    @property
    def complete(self) -> bool:
        """Whether the run finished inside its budget.

        Returns:
            bool: True only for `completed`.
        """
        return self.run_status is RunStatus.COMPLETED

    @property
    def resumable(self) -> bool:
        """Whether a later `resume` can continue the run.

        Returns:
            bool: True for budget or deadline stops.
        """
        return self.run_status in RESUMABLE_STATUSES


def output_reserve_s(budget_s: float) -> float:
    """Wall time held back from the native solver for artifact construction.

    Parameters:
        budget_s (float): Total segment budget.

    Returns:
        float: Reserve seconds, at least `MIN_OUTPUT_RESERVE_S` but never more than
            half the budget.
    """
    return min(max(MIN_OUTPUT_RESERVE_S, OUTPUT_RESERVE_FRACTION * budget_s), budget_s / 2.0)


def _resolve_budget(request: SimulationRequest, wall_budget_s: float | None) -> float:
    """Pick the segment budget, capped at the one-hour ceiling.

    Parameters:
        request (SimulationRequest): Frozen request carrying its own budget.
        wall_budget_s (float | None): Optional override for this segment.

    Returns:
        float: Budget in seconds.

    Raises:
        ValueError: When the override is non-positive or above the ceiling.
    """
    if wall_budget_s is None:
        return request.max_wall_time_s
    if not 0.0 < wall_budget_s <= MAX_WALL_TIME_S:
        raise ValueError(
            f"`wall_budget_s` '{wall_budget_s}' must be in (0, {MAX_WALL_TIME_S}] seconds"
        )
    return wall_budget_s


def _check_resume_provenance(run_dir: RunDirectory, current: Provenance) -> Provenance:
    """Refuse continuation under a different numerical build or altered evidence.

    Parameters:
        run_dir: RunDirectory: Run containing the previous committed segment.
        current: Provenance: Environment about to execute the next segment.
    Returns:
        Provenance: Current record with the original calibration identity preserved.
    Raises:
        ArtifactError: Provenance is corrupt or incompatible with this build.
    """
    manifest = run_dir.read_manifest()
    if manifest is None or manifest.latest is None:
        raise ArtifactError("resume requires a committed provenance record")
    path = run_dir.segment_dir(manifest.latest.index) / PROVENANCE_FILE
    try:
        previous = Provenance.model_validate_json(path.read_text())
    except (OSError, ValueError) as e:
        raise ArtifactError(f"Cannot read committed provenance '{path}': {e}") from e
    if previous.sha256 != manifest.latest.provenance_sha256:
        raise ArtifactError("committed provenance hash disagrees with the manifest")
    if current.calibration_sha256 is None:
        current = current.model_copy(update={"calibration_sha256": previous.calibration_sha256})
    fields = (
        "schema_version",
        "protocol_version",
        "request_sha256",
        "python_code_sha256",
        "native_sha256",
        "dependencies",
        "python_version",
        "calibration_sha256",
    )
    changed = [name for name in fields if getattr(previous, name) != getattr(current, name)]
    if changed:
        raise ArtifactError(f"resume build or calibration changed: {', '.join(changed)}")
    return current


def _execute_segment(
    run_dir: RunDirectory,
    request: SimulationRequest,
    resume: Checkpoint | None,
    *,
    budget_s: float,
    started: float,
    runner: NativeRunner | None,
    calibration_sha256: str | None,
    recording: bool,
) -> RunResult:
    """Run one native segment under a hard deadline and commit its artifacts.

    Parameters:
        run_dir (RunDirectory): Initialized run directory.
        request (SimulationRequest): Frozen request.
        resume (Checkpoint | None): Checkpoint to continue from.
        budget_s (float): Hard wall deadline for this segment.
        started (float): `time.monotonic()` at job start, before setup.
        runner (NativeRunner | None): Injected native callable for tests.
        calibration_sha256 (str | None): Calibration bundle hash for provenance.
        recording (bool): Whether to write a Rerun recording.

    Returns:
        RunResult: The committed outcome.

    Raises:
        BudgetError: When setup already consumed the budget.
    """
    index = run_dir.next_segment_index()
    provenance = build_provenance(request, calibration_sha256=calibration_sha256)
    if resume is not None:
        provenance = _check_resume_provenance(run_dir, provenance)
    setup_elapsed = time.monotonic() - started
    native_budget = budget_s - setup_elapsed - output_reserve_s(budget_s)
    if native_budget < MIN_NATIVE_BUDGET_S:
        raise BudgetError(
            f"segment budget {budget_s:.2f}s leaves {native_budget:.2f}s for the native "
            f"solver after {setup_elapsed:.2f}s of setup; nothing was run"
        )
    native_started = time.monotonic()
    output = run_segment(request.with_wall_budget(native_budget), resume, runner=runner)
    native_wall = time.monotonic() - native_started
    return _commit_output(
        run_dir,
        request,
        output,
        index=index,
        resume=resume,
        provenance_record=provenance,
        budget_s=budget_s,
        started=started,
        native_wall=native_wall,
        recording=recording,
    )


def _commit_output(
    run_dir: RunDirectory,
    request: SimulationRequest,
    output: SimulationOutput,
    *,
    index: int,
    resume: Checkpoint | None,
    provenance_record: Provenance,
    budget_s: float,
    started: float,
    native_wall: float,
    recording: bool,
) -> RunResult:
    """Write the segment and derive the Python-level status.

    Parameters:
        run_dir (RunDirectory): Initialized run directory.
        request (SimulationRequest): Frozen request.
        output (SimulationOutput): Validated native output.
        index (int): Segment index.
        resume (Checkpoint | None): Checkpoint resumed from.
        provenance_record (Provenance): Provenance built before the native call.
        budget_s (float): Hard deadline.
        started (float): Monotonic job start.
        native_wall (float): Seconds spent inside the native call.
        recording (bool): Whether to write a Rerun recording.

    Returns:
        RunResult: The committed outcome.
    """

    def write_recording(directory: Path) -> None:
        if recording:
            write_segment_recording(
                recording_file_for(directory),
                output,
                request,
                recording_id=hashlib.sha256(str(run_dir.root).encode()).hexdigest(),
                start_time_s=0.0 if resume is None else resume.time_s,
            )

    def record_for(status: RunStatus, wall: float) -> SegmentRecord:
        return SegmentRecord(
            index=index,
            native_status=output.status,
            run_status=status,
            start_time_s=0.0 if resume is None else resume.time_s,
            end_time_s=output.time_s,
            frame_count=len(output.frames),
            wall_time_s=wall,
            native_wall_time_s=native_wall,
            provenance_sha256=provenance_record.sha256,
        )

    provisional = (
        RunStatus.COMPLETED if output.status is Status.COMPLETED else RunStatus.BUDGET_EXHAUSTED
    )
    elapsed = time.monotonic() - started
    if elapsed > budget_s:
        provisional = RunStatus.DEADLINE_EXCEEDED
    directory = run_dir.write_segment(
        index,
        output,
        provenance_record,
        record_for(provisional, elapsed),
        extra_writer=write_recording,
    )
    final_elapsed = time.monotonic() - started
    final_status = provisional
    if final_elapsed > budget_s and provisional is RunStatus.COMPLETED:
        final_status = RunStatus.DEADLINE_EXCEEDED
        LOGGER.warning(
            f"segment {index} overran its {budget_s:.1f}s deadline at {final_elapsed:.1f}s "
            "during artifact construction; recorded as `deadline_exceeded`"
        )
        run_dir.amend_latest_status(final_status, final_elapsed)
    LOGGER.info(
        f"segment {index} '{final_status}' time={output.time_s:.3f}s wall={final_elapsed:.2f}s "
        f"dir='{directory}'"
    )
    return RunResult(
        run_dir=run_dir.root,
        segment_index=index,
        run_status=final_status,
        native_status=output.status,
        time_s=output.time_s,
        duration_s=request.duration_s,
        wall_time_s=final_elapsed,
        native_wall_time_s=native_wall,
        diagnostics=list(output.diagnostics),
        observables=dict(output.observables),
    )


def start_run(
    request: SimulationRequest,
    run_dir: Path,
    *,
    runner: NativeRunner | None = None,
    wall_budget_s: float | None = None,
    calibration_sha256: str | None = None,
    recording: bool = True,
) -> RunResult:
    """Freeze a request into a new run directory and execute its first segment.

    Parameters:
        request (SimulationRequest): Validated request.
        run_dir (Path): New or empty directory.
        runner (NativeRunner | None): Injected native callable for tests.
        wall_budget_s (float | None): Override for the segment budget.
        calibration_sha256 (str | None): Calibration bundle hash for provenance.
        recording (bool): Whether to write a Rerun recording.

    Returns:
        RunResult: The committed first segment.
    """
    started = time.monotonic()
    budget_s = _resolve_budget(request, wall_budget_s)
    directory = RunDirectory(run_dir)
    directory.initialize(request)
    return _execute_segment(
        directory,
        request,
        None,
        budget_s=budget_s,
        started=started,
        runner=runner,
        calibration_sha256=calibration_sha256,
        recording=recording,
    )


def resume_run(
    run_dir: Path,
    *,
    runner: NativeRunner | None = None,
    wall_budget_s: float | None = None,
    calibration_sha256: str | None = None,
    recording: bool = True,
) -> RunResult:
    """Continue an incomplete run from its latest committed checkpoint.

    Parameters:
        run_dir (Path): Existing run directory.
        runner (NativeRunner | None): Injected native callable for tests.
        wall_budget_s (float | None): Override for the segment budget.
        calibration_sha256 (str | None): Calibration bundle hash for provenance.
        recording (bool): Whether to write a Rerun recording.

    Returns:
        RunResult: The newly committed segment.

    Raises:
        ArtifactError: When the run is complete, has no segments, or is corrupt.
    """
    started = time.monotonic()
    directory = RunDirectory(run_dir)
    request = directory.read_request()
    manifest = directory.read_manifest()
    if manifest is None or manifest.latest is None:
        raise ArtifactError(
            f"Run '{directory.root}' has no committed segment to resume; start it instead"
        )
    if manifest.status not in RESUMABLE_STATUSES:
        raise ArtifactError(
            f"Run '{directory.root}' has status '{manifest.status}' and cannot be resumed"
        )
    if manifest.request_sha256 != hash_request(request):
        raise ArtifactError("frozen request hash does not match the manifest")
    checkpoint = directory.read_checkpoint(manifest.latest.index)
    if not checkpoint.request.equivalent_except_budget(request):
        raise ArtifactError("latest checkpoint does not match the frozen request")
    if checkpoint.time_s != manifest.time_s:
        raise ArtifactError("latest checkpoint time disagrees with the manifest")
    budget_s = _resolve_budget(request, wall_budget_s)
    return _execute_segment(
        directory,
        request,
        checkpoint,
        budget_s=budget_s,
        started=started,
        runner=runner,
        calibration_sha256=calibration_sha256,
        recording=recording,
    )
