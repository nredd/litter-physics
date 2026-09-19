"""Rerun recordings and loopback-only browser replay.

Recordings are derived views of committed segments, never restart files. Replay uses
the `rerun` CLI bundled with `rerun-sdk` because it accepts `--bind 127.0.0.1`; the
Python `serve_grpc`/`serve_web_viewer` APIs bind every interface and were verified to
do so on the development host, so they are not used here.

References:
    https://rerun.io/docs/reference/sdk/operating-modes
    https://ref.rerun.io/docs/python/stable/common/
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import rerun_cli

from litter_physics.artifacts import RECORDING_FILE, RunDirectory
from litter_physics.models import (
    Compartment,
    SimulationOutput,
    SimulationRequest,
)

LOGGER = logging.getLogger(__name__)

APPLICATION_ID = "litter-physics"
LOOPBACK = "127.0.0.1"
DEFAULT_GRPC_PORT = 9876
DEFAULT_WEB_PORT = 9090
STARTUP_TIMEOUT_S = 20.0
MATERIAL_COLORS: dict[str, tuple[int, int, int]] = {
    "wood": (196, 160, 96),
    "wet_wood": (140, 100, 50),
    "fines": (120, 90, 60),
    "paste": (90, 60, 40),
    "water": (70, 130, 220),
    "urine": (220, 200, 60),
    "stool": (80, 50, 30),
    "waste": (80, 50, 30),
    "paw": (220, 160, 170),
}
DEFAULT_COLOR = (160, 160, 160)


class ViewerError(RuntimeError):
    """The browser viewer could not be started safely."""


def material_colors(materials: list[str]) -> np.ndarray:
    """Map material labels to RGB colors.

    Parameters:
        materials (list[str]): Material label per particle.

    Returns:
        np.ndarray: `(n, 3)` uint8 colors.
    """
    return np.array(
        [MATERIAL_COLORS.get(label, DEFAULT_COLOR) for label in materials], dtype=np.uint8
    ).reshape(-1, 3)


def _log_boxes(request: SimulationRequest) -> None:
    """Log static box and drawer cutaway geometry.

    Parameters:
        request (SimulationRequest): Frozen request with box geometry.
    """
    for box in request.boxes:
        size = np.array(box.size_m, dtype=np.float64)
        origin = np.array(box.origin_m, dtype=np.float64)
        rr.log(
            f"world/boxes/{box.id}/bed",
            rr.Boxes3D(
                mins=[origin],
                sizes=[size],
                colors=[(200, 200, 200)],
                fill_mode="MajorWireframe",
                labels=[box.id],
            ),
            static=True,
        )
        drawer_origin = origin - np.array([0.0, 0.0, box.drawer_depth_m])
        drawer_size = np.array([size[0], size[1], box.drawer_depth_m])
        rr.log(
            f"world/boxes/{box.id}/drawer",
            rr.Boxes3D(
                mins=[drawer_origin],
                sizes=[drawer_size],
                colors=[(120, 120, 160)],
                fill_mode="MajorWireframe",
            ),
            static=True,
        )


def _log_events(request: SimulationRequest, start_time: float, end_time: float) -> None:
    """Log scheduled events as time-stamped text markers.

    Parameters:
        request (SimulationRequest): Frozen request with events.
    """
    boxes = {box.id: box for box in request.boxes}
    for index, event in enumerate(request.events):
        if not start_time <= event.time_s <= end_time:
            continue
        rr.set_time("sim_time", duration=event.time_s)
        box = boxes[event.box_id]
        world = np.array(box.origin_m) + np.array(event.position_m)
        rr.log(
            f"world/events/{index:05d}",
            rr.Points3D(
                [world],
                radii=[event.radius_m],
                colors=[(255, 40, 40)],
                labels=[f"{event.kind}@{box.id}"],
            ),
        )
        rr.log(
            "events/log",
            rr.TextLog(
                f"t={event.time_s:.3f}s {event.kind} box={box.id} amount={event.amount_kg}kg",
                level="INFO",
            ),
        )


def _log_output(output: SimulationOutput) -> None:
    """Log frames and ledger series from one segment.

    Parameters:
        output (SimulationOutput): Validated segment output.
    """
    for frame in output.frames:
        rr.set_time("sim_time", duration=frame.time_s)
        if not frame.positions_m:
            rr.log("world/particles", rr.Clear(recursive=False))
            continue
        positions = np.array(frame.positions_m, dtype=np.float64)
        rr.log(
            "world/particles",
            rr.Points3D(
                positions,
                radii=np.array(frame.radii_m, dtype=np.float32),
                colors=material_colors(frame.materials),
            ),
        )
    for row in output.metrics:
        rr.set_time("sim_time", duration=row.time_s)
        base = f"ledger/{row.compartment}"
        rr.log(f"{base}/wood_kg", rr.Scalars([row.wood_kg]))
        rr.log(f"{base}/waste_kg", rr.Scalars([row.waste_kg]))
        rr.log(f"{base}/water_kg", rr.Scalars([row.water_kg]))
    for key, value in output.observables.items():
        rr.set_time("sim_time", duration=output.time_s)
        rr.log(f"observables/{key}", rr.Scalars([value]))
    rr.set_time("sim_time", duration=output.time_s)
    for line in output.diagnostics:
        rr.log("diagnostics", rr.TextLog(line, level="WARN"))


def write_segment_recording(
    path: Path,
    output: SimulationOutput,
    request: SimulationRequest,
    *,
    recording_id: str,
    start_time_s: float,
) -> Path:
    """Write a `.rrd` recording for one segment.

    Parameters:
        path (Path): Destination file; must not exist.
        output (SimulationOutput): Validated segment output.
        request (SimulationRequest): Frozen request.
        recording_id (str): Stable run identity shared across all restart segments.
        start_time_s (float): Earliest scheduled event belonging to this segment.

    Returns:
        Path: The written recording.

    Raises:
        FileExistsError: When the destination already exists.
    """
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recording '{path}'")
    stream = rr.RecordingStream(APPLICATION_ID, recording_id=recording_id)
    with stream:
        stream.save(path)
        _log_boxes(request)
        _log_events(request, start_time_s, output.time_s)
        _log_output(output)
        stream.flush()
    LOGGER.info(f"wrote recording '{path}'")
    return path


def ledger_summary(metrics: pa.Table) -> dict[str, dict[str, float]]:
    """Summarize the final ledger sample per compartment.

    Parameters:
        metrics (pa.Table): Metric rows across segments.

    Returns:
        dict[str, dict[str, float]]: Compartment to final species masses.
    """
    summary: dict[str, dict[str, float]] = {}
    if metrics.num_rows == 0:
        return summary
    for compartment in Compartment:
        subset = metrics.filter(pc.field("compartment") == str(compartment))
        if subset.num_rows == 0:
            continue
        last = subset.slice(subset.num_rows - 1, 1).to_pylist()[0]
        summary[str(compartment)] = {
            "time_s": float(last["time_s"]),
            "wood_kg": float(last["wood_kg"]),
            "waste_kg": float(last["waste_kg"]),
            "water_kg": float(last["water_kg"]),
        }
    return summary


def rerun_binary() -> Path:
    """Locate the `rerun` CLI shipped with `rerun-sdk`.

    Returns:
        Path: Absolute path to the executable.

    Raises:
        ViewerError: When no bundled binary is found next to the interpreter.
    """
    if rerun_cli.__file__ is None:
        raise ViewerError("`rerun-sdk` has no installed CLI package")
    package = Path(rerun_cli.__file__).resolve().parent
    if sys.platform == "darwin":
        candidates = [package / "Rerun.app/Contents/MacOS/Rerun", package / "rerun"]
    else:
        candidates = [package / ("rerun.exe" if sys.platform == "win32" else "rerun")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ViewerError("bundled `rerun` binary is missing; reinstall `rerun-sdk`")


def _non_loopback_addresses() -> list[str]:
    """List IPv4 addresses of this host that are not loopback.

    Returns:
        list[str]: Addresses to probe when verifying the viewer is loopback-only.
    """
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = str(info[4][0])
            if not address.startswith("127."):
                addresses.add(address)
    except socket.gaierror:
        return []
    return sorted(addresses)


def _port_open(address: str, port: int, timeout_s: float = 0.5) -> bool:
    """Try to open a TCP connection.

    Parameters:
        address (str): Host address.
        port (int): TCP port.
        timeout_s (float): Connect timeout.

    Returns:
        bool: True when the connection succeeded.
    """
    try:
        with socket.create_connection((address, port), timeout=timeout_s):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class ViewerHandle:
    """A running loopback viewer process."""

    process: subprocess.Popen[str]
    web_url: str
    grpc_url: str
    recordings: list[Path]

    def stop(self) -> None:
        """Terminate the viewer process, escalating to kill if needed."""
        if self.process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(self.process.pid, signal.SIGTERM)
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(self.process.pid, signal.SIGKILL)
            else:
                self.process.kill()
            self.process.wait(timeout=5)


def start_loopback_viewer(
    run_dir: RunDirectory,
    *,
    grpc_port: int = DEFAULT_GRPC_PORT,
    web_port: int = DEFAULT_WEB_PORT,
    startup_timeout_s: float = STARTUP_TIMEOUT_S,
) -> ViewerHandle:
    """Serve committed recordings to a browser viewer bound to `127.0.0.1` only.

    Parameters:
        run_dir (RunDirectory): Run with at least one committed recording.
        grpc_port (int): Loopback gRPC port for the data server.
        web_port (int): Loopback HTTP port for the web viewer.
        startup_timeout_s (float): Seconds to wait for the HTTP server.

    Returns:
        ViewerHandle: The running process and its URLs.

    Raises:
        ViewerError: When no recordings exist, a port is taken, the server does not
            start, or it is reachable on a non-loopback address.
    """
    recordings = run_dir.recording_paths()
    if not recordings:
        raise ViewerError(f"Run '{run_dir.root}' has no committed recordings to view")
    for port in (grpc_port, web_port):
        if _port_open(LOOPBACK, port):
            raise ViewerError(f"Port {port} on {LOOPBACK} is already in use")
    command = [
        str(rerun_binary()),
        "--serve-web",
        "--bind",
        LOOPBACK,
        "--port",
        str(grpc_port),
        "--web-viewer-port",
        str(web_port),
        *[str(path) for path in recordings],
    ]
    LOGGER.info(f"starting loopback viewer: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=None,
        text=True,
        start_new_session=True,
    )
    grpc_url = f"rerun+http://{LOOPBACK}:{grpc_port}/proxy"
    query = urllib.parse.urlencode({"url": grpc_url})
    web_url = f"http://{LOOPBACK}:{web_port}/?{query}"
    handle = ViewerHandle(
        process=process, web_url=web_url, grpc_url=grpc_url, recordings=recordings
    )
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise ViewerError(f"`rerun` exited early with code {process.returncode}: {output}")
        try:
            with urllib.request.urlopen(web_url, timeout=1.0) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    else:
        handle.stop()
        raise ViewerError(f"viewer did not answer on '{web_url}' within {startup_timeout_s}s")
    exposed = [
        f"{address}:{port}"
        for address in _non_loopback_addresses()
        for port in (grpc_port, web_port)
        if _port_open(address, port)
    ]
    if exposed:
        handle.stop()
        raise ViewerError(f"viewer is reachable on non-loopback addresses {exposed}; refusing")
    return handle


def recording_entities(path: Path) -> dict[str, int]:
    """Verify a recording with the `rerun` CLI and count chunks per entity path.

    Parameters:
        path (Path): `.rrd` file.

    Returns:
        dict[str, int]: Entity path to chunk count.

    Raises:
        ViewerError: When the file fails `rerun rrd verify` or stats cannot be parsed.
    """
    resolved = Path(str(path)).resolve()
    if not resolved.is_file():
        raise ViewerError(f"Recording does not exist: '{resolved}'")
    binary = str(rerun_binary())
    try:
        subprocess.check_output(
            [binary, "rrd", "verify", str(resolved)], text=True, stderr=subprocess.STDOUT
        )
        stats = subprocess.check_output(
            [binary, "rrd", "stats", str(resolved)], text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise ViewerError(f"`rerun rrd` rejected '{resolved}': {e.output}") from e
    entities: dict[str, int] = {}
    section = False
    for line in stats.splitlines():
        if line.startswith("Num chunks per entity"):
            section = True
            continue
        if section and line.startswith("Num chunks per index"):
            break
        if section and line.startswith("/") and ":" in line:
            entity, _, count = line.rpartition(":")
            entities[entity.strip()] = int(count.strip())
    return entities


def recording_file_for(segment_dir: Path) -> Path:
    """Standard recording path inside a segment directory.

    Parameters:
        segment_dir (Path): Committed or in-progress segment directory.

    Returns:
        Path: `recording.rrd` inside the segment.
    """
    return segment_dir / RECORDING_FILE
