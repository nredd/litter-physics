"""Recording construction and loopback-only replay.

References:
    https://rerun.io/docs/reference/sdk/operating-modes
"""

from __future__ import annotations

import signal
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from litter_physics import cli, viewer
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


def test_recording_includes_replay_guide_and_final_summary(
    household_request: SimulationRequest, tmp_path: Path
) -> None:
    """The actual recording carries explicit model limits and final-value text."""
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    entities = recording_entities(RunDirectory(tmp_path / "run").recording_paths()[0])
    assert "/replay/guide" in entities
    assert "/replay/summary" in entities
    assert "/world" in entities, "the world must declare Z-up coordinates"


def test_research_recording_includes_domain(
    research_request: SimulationRequest, tmp_path: Path
) -> None:
    """Research points need a visible domain even though there are no household boxes."""
    start_run(research_request, tmp_path / "run", runner=FakeCore())
    entities = recording_entities(RunDirectory(tmp_path / "run").recording_paths()[0])
    assert "/world/domain" in entities


def test_view_handles_sigterm_and_releases_ports(
    household_request: SimulationRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM must use the same graceful child cleanup as Ctrl-C."""
    start_run(household_request, tmp_path / "run", runner=FakeCore())
    handlers: dict[int, object] = {}
    delivered: list[int] = []
    original_sleep = cli.time.sleep

    def register(number: int, handler: object) -> object:
        """Intercept handler registration without changing pytest's own signals."""
        previous = handlers.get(number, signal.SIG_DFL)
        handlers[number] = handler
        return previous

    def interrupt(delay: float) -> None:
        """Deliver SIGTERM during startup or serving, after handlers are installed."""
        handler = handlers.get(signal.SIGTERM)
        if callable(handler):
            delivered.append(signal.SIGTERM)
            handler(signal.SIGTERM, None)
        else:
            original_sleep(delay)

    monkeypatch.setattr(cli.signal, "signal", register)
    monkeypatch.setattr(cli.time, "sleep", interrupt)
    grpc_port, web_port = free_port(), free_port()
    assert (
        cli.main(
            [
                "view",
                str(tmp_path / "run"),
                "--grpc-port",
                str(grpc_port),
                "--web-port",
                str(web_port),
                "--duration",
                "0.1",
            ]
        )
        == 0
    )
    assert delivered, "SIGTERM never reached a graceful-stop handler"
    assert handlers[signal.SIGTERM] == signal.SIG_DFL
    assert handlers[signal.SIGINT] == signal.SIG_DFL
    for port in (grpc_port, web_port):
        with socket.socket() as sock:
            assert sock.connect_ex((LOOPBACK, port)) != 0


def test_blueprint_prioritizes_scene_and_selects_real_inventory_paths(
    household_request: SimulationRequest,
) -> None:
    """Explicit entity paths avoid unsupported middle-wildcard filters and empty plots."""
    blueprint = viewer._replay_blueprint(household_request)
    assert blueprint.auto_views is False and blueprint.auto_layout is False
    root = blueprint.root_container
    assert isinstance(root, viewer.rrb.Vertical)
    scene_area, charts = root.contents
    assert isinstance(scene_area, viewer.rrb.Horizontal)
    scene = next(iter(scene_area.contents))
    assert isinstance(scene, viewer.rrb.Spatial3DView)
    assert scene.origin == "/world"
    assert scene.contents == ["$origin/particles", "$origin/boxes/**", "$origin/domain"]
    assert isinstance(charts, viewer.rrb.Horizontal)
    for chart, species in zip(charts.contents, ("wood", "waste", "water"), strict=True):
        assert isinstance(chart, viewer.rrb.TimeSeriesView)
        assert chart.contents == [f"/ledger/{c}/{species}_kg" for c in viewer.Compartment]


def test_summary_is_timed_and_guide_does_not_imply_field_measurements(
    household_request: SimulationRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final results must not be presented as current values while scrubbing earlier frames."""
    output = run_segment(household_request, None, runner=FakeCore())
    current_time = 0.0
    documents: dict[str, tuple[float, bool, str]] = {}

    def set_time(timeline: str, *, duration: float) -> None:
        """Capture the simulation clock used for subsequent logs."""
        nonlocal current_time
        assert timeline == "sim_time"
        current_time = duration

    def capture(entity: str, archetype: object, *, static: bool = False) -> None:
        """Retain text, staticness and the actual timestamp together."""
        if isinstance(archetype, viewer.rr.TextDocument):
            assert archetype.text is not None
            text = archetype.text.as_arrow_array().to_pylist()[0]
            documents[entity] = (current_time, static, text)

    monkeypatch.setattr(viewer.rr, "set_time", set_time)
    monkeypatch.setattr(viewer.rr, "log", capture)
    viewer._log_replay_context(household_request, output)
    viewer._log_output(output)
    _, static, guide = documents["replay/guide"]
    assert static and str(output.fidelity) in guide
    assert "rendering proxies" in guide and "not pressure" in guide
    time_s, static, summary = documents["replay/summary"]
    assert time_s == output.time_s and not static
    assert "segment-final samples" in summary and str(output.status) in summary
