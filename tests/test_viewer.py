"""Recording construction and loopback-only replay.

References:
    https://rerun.io/docs/reference/sdk/operating-modes
"""

from __future__ import annotations

import socket
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from litter_physics import viewer
from litter_physics.artifacts import RunDirectory
from litter_physics.models import SimulationRequest
from litter_physics.native import run_segment
from litter_physics.runner import resume_run, start_run
from litter_physics.viewer import (
    LOOPBACK,
    ViewerError,
    ledger_summary,
    material_colors,
    recording_entities,
    start_loopback_viewer,
)
from tests.fake_core import FakeCore


def free_port() -> int:
    """Pick a currently free loopback TCP port.

    Returns:
        int: Port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def test_recording_is_loadable(household_request: SimulationRequest, tmp_path: Path) -> None:
    """The per-segment `.rrd` loads with the Rerun SDK and has content."""
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    paths = RunDirectory(tmp_path / "run").recording_paths()
    assert len(paths) == 1 and paths[0].stat().st_size > 1000
    entities = recording_entities(paths[0])
    assert entities["/world/particles"] >= 1
    assert any(path.startswith("/ledger/bed") for path in entities)
    assert any(path.startswith("/world/boxes/box-a") for path in entities)
    assert any(path.startswith("/world/events") for path in entities)
    with pytest.raises(ViewerError, match="does not exist"):
        recording_entities(tmp_path / "missing.rrd")
    bogus = tmp_path / "bogus.rrd"
    bogus.write_bytes(b"not a recording")
    with pytest.raises(ViewerError, match="rejected"):
        recording_entities(bogus)


def test_material_colors_and_summary(household_request: SimulationRequest, tmp_path: Path) -> None:
    """Unknown materials get the default color; the ledger summary is per compartment."""
    colors = material_colors(["wood", "mystery"])
    assert colors.shape == (2, 3)
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    summary = ledger_summary(RunDirectory(tmp_path / "run").read_metrics())
    assert set(summary) == {"bed", "drawer", "floor", "removed", "evaporated"}
    assert summary["bed"]["wood_kg"] > 0.0
    assert ledger_summary(RunDirectory(tmp_path / "nothing").read_metrics()) == {}


def test_viewer_requires_recordings(household_request: SimulationRequest, tmp_path: Path) -> None:
    """A run without recordings cannot be viewed."""
    start_run(household_request, tmp_path / "run", runner=FakeCore(), recording=False)
    with pytest.raises(ViewerError, match="no committed recordings"):
        start_loopback_viewer(RunDirectory(tmp_path / "run"))


def test_viewer_binds_loopback_only(household_request: SimulationRequest, tmp_path: Path) -> None:
    """The bundled `rerun` CLI serves on 127.0.0.1 and the HTTP page answers."""
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    grpc_port, web_port = free_port(), free_port()
    handle = start_loopback_viewer(
        RunDirectory(tmp_path / "run"), grpc_port=grpc_port, web_port=web_port
    )
    try:
        with urllib.request.urlopen(handle.web_url, timeout=5.0) as response:
            body = response.read(200)
        assert b"<!doctype html>" in body.lower()
        assert handle.grpc_url == f"rerun+http://{LOOPBACK}:{grpc_port}/proxy"
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(handle.web_url).query)
        assert query.get("url") == [handle.grpc_url], "browser never selects the recording server"
        with pytest.raises(ViewerError, match="already in use"):
            start_loopback_viewer(
                RunDirectory(tmp_path / "run"), grpc_port=grpc_port, web_port=web_port
            )
    finally:
        handle.stop()
    assert handle.process.poll() is not None
    for port in (grpc_port, web_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            assert sock.connect_ex((LOOPBACK, port)) != 0, "viewer child survived stop()"


def test_viewer_keeps_world_coordinates(
    household_request: SimulationRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never add the box origin twice to an already-world-space native frame."""

    output = run_segment(household_request, None, runner=FakeCore())
    captured = []

    def capture(entity: str, archetype: object, **kwargs: object) -> None:
        """Capture the actual Rerun point payload.

        Parameters:
            entity: str: Entity path being logged.
            archetype: object: Rerun archetype.
            kwargs: object: Unused logging options.
        Returns:
            None: Points are retained for assertions.
        """
        if entity == "world/particles" and isinstance(archetype, viewer.rr.Points3D):
            assert archetype.positions is not None
            captured.append(archetype.positions.as_arrow_array().to_pylist())

    monkeypatch.setattr(viewer.rr, "log", capture)
    viewer._log_output(output)
    assert len(captured) == len(output.frames)
    for recorded, frame in zip(captured, output.frames, strict=True):
        for actual, expected in zip(recorded, frame.positions_m, strict=True):
            assert actual == pytest.approx(expected)


def test_restart_segments_share_one_recording_identity(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """Merge restart segments into one timeline, but keep distinct runs separate."""
    core = FakeCore(max_sim_time_per_call=0.4)
    with patch.object(viewer.rr, "RecordingStream", wraps=viewer.rr.RecordingStream) as factory:
        start_run(household_request, tmp_path / "first", runner=core)
        resume_run(tmp_path / "first", runner=core)
        start_run(household_request, tmp_path / "second", runner=core)
    identities = [call.kwargs["recording_id"] for call in factory.call_args_list]
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
