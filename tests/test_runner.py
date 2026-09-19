"""Bounded execution, resume, deadlines, and append-only artifacts (fake core).

References:
    docs/wire-contract.md
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from litter_physics import native
from litter_physics import runner as runner_module
from litter_physics.artifacts import (
    MANIFEST_FILE,
    ArtifactError,
    RunDirectory,
    RunStatus,
    write_new_json,
)
from litter_physics.models import SimulationRequest
from litter_physics.native import (
    NativeContractError,
    NativeSimulationError,
    NativeUnavailableError,
    resolve_runner,
    run_segment,
)
from litter_physics.provenance import Provenance
from litter_physics.runner import BudgetError, resume_run, start_run
from tests.fake_core import FakeCore


def snapshot(root: Path) -> dict[str, bytes]:
    """Capture every file under a directory except the manifest.

    Parameters:
        root (Path): Run directory.

    Returns:
        dict[str, bytes]: Relative path to bytes.
    """
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_FILE
    }


def test_complete_run_writes_segment(household_request: SimulationRequest, tmp_path: Path) -> None:
    """A completed run commits one segment with every artifact."""
    core = FakeCore()
    result = start_run(household_request, tmp_path / "run", runner=core)
    assert result.complete and result.run_status is RunStatus.COMPLETED
    segment = tmp_path / "run" / "segments" / "0000"
    for name in (
        "output.json",
        "frames.parquet",
        "metrics.parquet",
        "checkpoint.json",
        "provenance.json",
        "recording.rrd",
    ):
        assert (segment / name).is_file(), name
    manifest = json.loads((tmp_path / "run" / MANIFEST_FILE).read_text())
    assert manifest["complete"] is True
    assert manifest["segments"][0]["frame_count"] == 21
    provenance = json.loads((segment / "provenance.json").read_text())
    assert provenance["request_sha256"] == manifest["request_sha256"]
    assert provenance["hardware"]["cpu_count"] >= 1
    directory = RunDirectory(tmp_path / "run")
    assert directory.read_frames().num_rows == 21 * 48
    assert directory.read_metrics().num_rows == 21 * 5
    assert directory.read_request() == household_request


def test_resume_appends_suffix_and_preserves_artifacts(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """Budget stops are resumable; earlier segments are byte-identical afterwards."""
    core = FakeCore(max_sim_time_per_call=0.4)
    first = start_run(household_request, tmp_path / "run", runner=core)
    assert first.run_status is RunStatus.BUDGET_EXHAUSTED and first.resumable
    before = snapshot(tmp_path / "run")
    second = resume_run(tmp_path / "run", runner=core)
    assert second.segment_index == 1 and second.time_s == pytest.approx(0.8)
    third = resume_run(tmp_path / "run", runner=core)
    assert third.complete and third.time_s == pytest.approx(1.0)
    after = snapshot(tmp_path / "run")
    for name, content in before.items():
        assert after[name] == content, name
    directory = RunDirectory(tmp_path / "run")
    frames = directory.read_frames()
    times = sorted(set(frames["time_s"].to_pylist()))
    assert len(times) == 21
    assert times == sorted(times)
    manifest = directory.read_manifest()
    assert manifest is not None and [s.index for s in manifest.segments] == [0, 1, 2]
    with pytest.raises(ArtifactError, match="cannot be resumed"):
        resume_run(tmp_path / "run", runner=core)


def test_no_overwrite(household_request: SimulationRequest, tmp_path: Path) -> None:
    """Starting into a used directory is refused."""
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    with pytest.raises(ArtifactError, match="not empty"):
        start_run(household_request, tmp_path / "run", runner=FakeCore())


def test_resume_rejects_mismatch_and_partial(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """Resume refuses a changed request, a stale partial segment, or no segments."""
    core = FakeCore(max_sim_time_per_call=0.4)
    start_run(household_request, tmp_path / "run", runner=core)
    directory = RunDirectory(tmp_path / "run")
    (directory.segment_dir(1)).mkdir()
    with pytest.raises(ArtifactError, match="partial segment"):
        resume_run(tmp_path / "run", runner=core)
    directory.segment_dir(1).rmdir()
    tampered = household_request.model_copy(update={"seed": 99})
    directory.request_path.write_text(json.dumps(tampered.model_dump(mode="json")))
    with pytest.raises(ArtifactError, match="does not match"):
        resume_run(tmp_path / "run", runner=core)
    empty = RunDirectory(tmp_path / "empty")
    empty.initialize(household_request)
    with pytest.raises(ArtifactError, match="no committed segment"):
        resume_run(tmp_path / "empty", runner=core)


def test_deadline_exceeded_is_not_success(
    household_request: SimulationRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native `completed` that overruns the process deadline is `deadline_exceeded`."""
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 40.0
        return clock["now"]

    monkeypatch.setattr(runner_module.time, "monotonic", fake_monotonic)
    result = start_run(household_request, tmp_path / "run", runner=FakeCore(), wall_budget_s=60.0)
    assert result.native_status == "completed"
    assert result.run_status is RunStatus.DEADLINE_EXCEEDED
    assert not result.complete and result.resumable
    manifest = RunDirectory(tmp_path / "run").read_manifest()
    assert manifest is not None and manifest.complete is False
    assert manifest.status is RunStatus.DEADLINE_EXCEEDED


def test_budget_error_before_native_call(
    household_request: SimulationRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup that eats the whole budget never calls the solver."""
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 100.0
        return clock["now"]

    monkeypatch.setattr(runner_module.time, "monotonic", fake_monotonic)
    core = FakeCore()
    with pytest.raises(BudgetError, match="nothing was run"):
        start_run(household_request, tmp_path / "run", runner=core, wall_budget_s=30.0)
    assert core.calls == 0


def test_native_budget_is_reduced(household_request: SimulationRequest, tmp_path: Path) -> None:
    """The solver receives less than the process budget."""
    seen: list[float] = []
    inner = FakeCore()

    def spy(request_json: str, resume_json: str | None) -> str:
        seen.append(json.loads(request_json)["max_wall_time_s"])
        return inner(request_json, resume_json)

    start_run(household_request, tmp_path / "run", runner=spy, wall_budget_s=100.0)
    assert seen and seen[0] < 100.0 - 4.9


def test_native_failures_are_typed(household_request: SimulationRequest) -> None:
    """Solver errors and contract violations surface as distinct exceptions."""
    with pytest.raises(NativeSimulationError, match="boom"):
        run_segment(household_request, None, runner=FakeCore(fail_with="boom"))

    def garbage(_request: str, _resume: str | None) -> str:
        return "{}"

    with pytest.raises(NativeContractError):
        run_segment(household_request, None, runner=garbage)

    def wrong_request(request_json: str, resume: str | None) -> str:
        altered = json.loads(request_json) | {"seed": 5}
        return FakeCore()(json.dumps(altered), resume)

    with pytest.raises(NativeContractError, match="embed"):
        run_segment(household_request, None, runner=wrong_request)


def test_resolve_runner_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `run_json` the boundary raises instead of pretending."""

    monkeypatch.delattr(native._core, "run_json", raising=False)
    with pytest.raises(NativeUnavailableError):
        resolve_runner(None)


def test_wall_budget_override_validation(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """Budget overrides above one hour or non-positive are rejected."""
    with pytest.raises(ValueError, match="wall_budget_s"):
        start_run(household_request, tmp_path / "run", runner=FakeCore(), wall_budget_s=4000.0)
    with pytest.raises(ValueError, match="wall_budget_s"):
        start_run(household_request, tmp_path / "run2", runner=FakeCore(), wall_budget_s=0.0)


def test_deadline_exceeded_during_artifact_writing(
    household_request: SimulationRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overrunning while writing artifacts downgrades the committed manifest status."""
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 40.0
        return clock["now"]

    monkeypatch.setattr(runner_module.time, "monotonic", fake_monotonic)
    result = start_run(household_request, tmp_path / "run", runner=FakeCore(), wall_budget_s=190.0)
    assert result.run_status is RunStatus.DEADLINE_EXCEEDED
    manifest = RunDirectory(tmp_path / "run").read_manifest()
    assert manifest is not None
    assert manifest.status is RunStatus.DEADLINE_EXCEEDED and not manifest.complete
    assert manifest.segments[0].wall_time_s == pytest.approx(200.0)
    assert manifest.cumulative_wall_time_s == pytest.approx(200.0)
    segment_summary = json.loads(
        (tmp_path / "run" / "segments" / "0000" / "output.json").read_text()
    )
    assert segment_summary["run_status"] == "completed"


def test_resume_rejects_changed_native_build(
    household_request: SimulationRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not continue a checkpoint after the native numerical build changes."""

    core = FakeCore(max_sim_time_per_call=0.4)
    start_run(household_request, tmp_path / "run", runner=core)
    original = runner_module.build_provenance

    def changed(
        request: SimulationRequest, *, calibration_sha256: str | None = None
    ) -> Provenance:
        """Simulate a changed build fingerprint without changing numerical state.

        Parameters:
            request: SimulationRequest: Original scenario.
            calibration_sha256: str | None: Calibration identity to retain.
        Returns:
            Provenance: Evidence pointing at a different native binary.
        """
        return original(request, calibration_sha256=calibration_sha256).model_copy(
            update={"native_sha256": "0" * 64}
        )

    monkeypatch.setattr(runner_module, "build_provenance", changed)
    with pytest.raises(ArtifactError, match="native_sha256"):
        resume_run(tmp_path / "run", runner=core)


def test_resume_preserves_calibration_and_detects_provenance_tampering(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """Keep the initial calibration identity and reject edited evidence records."""
    core = FakeCore(max_sim_time_per_call=0.3)
    start_run(household_request, tmp_path / "run", runner=core, calibration_sha256="a" * 64)
    resume_run(tmp_path / "run", runner=core)
    path = RunDirectory(tmp_path / "run").segment_dir(1) / "provenance.json"
    record = json.loads(path.read_text())
    assert record["calibration_sha256"] == "a" * 64
    record["native_sha256"] = "bad"
    path.write_text(json.dumps(record))
    with pytest.raises(ArtifactError, match="provenance hash"):
        resume_run(tmp_path / "run", runner=core)


def test_new_artifacts_are_exclusive_under_concurrent_writers(tmp_path: Path) -> None:
    """Exactly one writer publishes a new artifact without overwriting another."""
    path = tmp_path / "exclusive.json"

    def publish(value: int) -> bool:
        """Attempt one competing publication.

        Parameters:
            value: int: Distinct payload.
        Returns:
            bool: Whether this writer won the exclusive publication.
        """
        try:
            write_new_json(path, {"value": value})
        except ArtifactError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(publish, range(8)))
    assert sum(outcomes) == 1
    assert json.loads(path.read_text())["value"] == outcomes.index(True)
    assert list(tmp_path.iterdir()) == [path]
