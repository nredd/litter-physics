"""Exercise real native kernels, restart state and artifact export.

References:
    docs/wire-contract.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litter_physics.artifacts import RunDirectory, RunStatus
from litter_physics.models import SimulationRequest, load_request, request_to_wire
from litter_physics.native import NativeContractError, resolve_runner, run_segment
from litter_physics.runner import resume_run, start_run
from litter_physics.viewer import recording_entities
from tests.conftest import EXAMPLES

pytestmark = pytest.mark.usefixtures("native_or_skip")


@pytest.fixture
def small_request(household_request: SimulationRequest) -> SimulationRequest:
    """Build a short real-physics fixture; large examples are benchmarked separately.

    Parameters:
        household_request: SimulationRequest: Valid demonstration template.
    Returns:
        SimulationRequest: Four pellets in one box for integration checks.
    """
    data = household_request.model_dump(mode="json")
    data["duration_s"] = 0.02
    data["record_interval_s"] = 0.005
    data["events"] = []
    data["boxes"] = data["boxes"][:1]
    data["boxes"][0]["pellet_count"] = 4
    return SimulationRequest.model_validate_json(json.dumps(data))


def test_household_completes_with_artifacts(
    small_request: SimulationRequest, tmp_path: Path
) -> None:
    """Complete a real fixture, require conservation, and verify its recording."""
    result = start_run(small_request, tmp_path / "run")
    assert result.native_status == "completed"
    assert result.run_status is RunStatus.COMPLETED
    directory = RunDirectory(tmp_path / "run")
    assert directory.read_metrics().num_rows > 0
    assert abs(result.observables["mass_residual_kg"]) <= 1e-12
    assert "/world/particles" in recording_entities(directory.recording_paths()[0])


def test_resume_is_a_suffix(small_request: SimulationRequest, tmp_path: Path) -> None:
    """Force a real native checkpoint deterministically, not by a machine-speed race."""
    call = resolve_runner(None)

    def exhaust_budget(request_json: str, resume_json: str | None) -> str:
        """Run real kernels with a budget exhausted immediately after initialization.

        Parameters:
            request_json: str: Valid request from orchestration.
            resume_json: str | None: Optional full numerical restart state.
        Returns:
            str: Real native output, never a mock particle trajectory.
        """
        request = json.loads(request_json)
        request["max_wall_time_s"] = 1e-12
        return call(json.dumps(request), resume_json)

    partial = start_run(small_request, tmp_path / "run", runner=exhaust_budget)
    assert partial.run_status is RunStatus.BUDGET_EXHAUSTED
    directory = RunDirectory(tmp_path / "run")
    before = directory.read_frames().num_rows
    complete = resume_run(tmp_path / "run")
    assert complete.complete
    assert directory.read_frames().num_rows > before
    times = directory.read_frames()["time_s"].to_pylist()
    assert times == sorted(times)
    full = run_segment(small_request, None)
    assert complete.observables == full.observables


def test_native_rejects_invalid_json() -> None:
    """Rust rejects malformed requests independently of Python validation."""
    with pytest.raises(ValueError, match="mode"):
        resolve_runner(None)('{"schema_version": 1}', None)


def test_native_rejects_mismatched_resume(small_request: SimulationRequest) -> None:
    """Reject request drift in both Python and direct native entry points."""
    output = run_segment(small_request, None)
    other = small_request.model_copy(update={"seed": small_request.seed + 1})
    with pytest.raises(NativeContractError, match="differs"):
        run_segment(other, output.checkpoint)
    with pytest.raises(ValueError, match="differs"):
        resolve_runner(None)(request_to_wire(other), output.checkpoint.model_dump_json())


def test_research_fixture_completes(research_request: SimulationRequest) -> None:
    """Require a real successful fixture; an arbitrary numerical error is NOT a pass."""
    data = research_request.model_dump(mode="json")
    data["duration_s"] = 0.002
    data["dt_s"] = 0.0005
    data["record_interval_s"] = 0.001
    data["research"]["initial_size_m"] = [0.01, 0.01, 0.01]
    request = SimulationRequest.model_validate_json(json.dumps(data))
    output = run_segment(request.with_wall_budget(30.0), None)
    assert output.status == "completed"
    assert output.fidelity == "research_unvalidated"
    assert abs(output.observables["mass_residual_kg"]) < 1e-12
    assert len(output.frames) == 3


def test_coupled_example_exports_wall_energy_and_research_geometry(tmp_path: Path) -> None:
    """Exercise the shipped pellet/paste fixture and its diagnostic-only energy channel."""
    request = load_request(EXAMPLES / "research_coupled_patch.yaml")
    result = start_run(request, tmp_path / "coupled")
    assert result.complete and result.native_status == "completed"
    assert result.observables["pellet_speed_m_s"] > 0.0
    assert result.observables["coupling_grid_energy_j"] != 0.0
    assert "wall_normal_traction_energy_j" in result.observables
    assert abs(result.observables["mass_residual_kg"]) < 1e-12
    directory = RunDirectory(tmp_path / "coupled")
    entities = recording_entities(directory.recording_paths()[0])
    assert "/world/domain" in entities
    assert "/observables/wall_normal_traction_energy_j" in entities
    assert not any(entity.startswith("/world/boxes/") for entity in entities)
