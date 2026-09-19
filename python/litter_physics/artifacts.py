"""Atomic, append-only run artifacts.

A run directory holds one frozen request plus numbered execution segments. Each
segment stores its own output, Parquet tables, checkpoint, provenance, and Rerun
recording. Resuming appends a new segment; nothing already written is modified or
deleted. The manifest is replaced atomically and is the only mutable file.

Layout:
    request.json                 frozen request (written once)
    manifest.json                atomic replace, lists committed segments
    segments/0000/output.json    status, fidelity, observables, diagnostics
    segments/0000/frames.parquet long table of particles per frame
    segments/0000/metrics.parquet ledger rows
    segments/0000/checkpoint.json resumable native checkpoint
    segments/0000/provenance.json provenance at execution time
    segments/0000/recording.rrd  Rerun recording for this segment

References:
    https://arrow.apache.org/docs/python/parquet.html
    https://docs.python.org/3/library/os.html#os.replace
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from litter_physics.models import (
    Checkpoint,
    Frame,
    MetricRow,
    SimulationOutput,
    SimulationRequest,
    Status,
    canonical_json,
    validate_payload,
)
from litter_physics.provenance import Provenance, hash_request

LOGGER = logging.getLogger(__name__)

REQUEST_FILE = "request.json"
MANIFEST_FILE = "manifest.json"
SEGMENTS_DIR = "segments"
OUTPUT_FILE = "output.json"
FRAMES_FILE = "frames.parquet"
METRICS_FILE = "metrics.parquet"
CHECKPOINT_FILE = "checkpoint.json"
PROVENANCE_FILE = "provenance.json"
RECORDING_FILE = "recording.rrd"

FRAME_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64()),
        pa.field("time_s", pa.float64()),
        pa.field("particle_index", pa.int64()),
        pa.field("x_m", pa.float64()),
        pa.field("y_m", pa.float64()),
        pa.field("z_m", pa.float64()),
        pa.field("radius_m", pa.float64()),
        pa.field("material", pa.string()),
        pa.field("box_id", pa.string()),
    ]
)
METRIC_SCHEMA = pa.schema(
    [
        pa.field("time_s", pa.float64()),
        pa.field("compartment", pa.string()),
        pa.field("wood_kg", pa.float64()),
        pa.field("waste_kg", pa.float64()),
        pa.field("water_kg", pa.float64()),
    ]
)


class RunStatus(StrEnum):
    """Python-level run status, stricter than the native status."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"


RESUMABLE_STATUSES = frozenset({RunStatus.BUDGET_EXHAUSTED, RunStatus.DEADLINE_EXCEEDED})


class ArtifactError(RuntimeError):
    """A run directory is missing, corrupt, or would be overwritten."""


class SegmentRecord(BaseModel):
    """Manifest entry for one committed segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0, description="Zero-based segment index.")
    native_status: Status = Field(description="Status reported by the native solver.")
    run_status: RunStatus = Field(description="Status after Python deadline accounting.")
    start_time_s: float = Field(description="Simulation time at segment start.")
    end_time_s: float = Field(description="Simulation time at segment end.")
    frame_count: int = Field(ge=0, description="Frames stored in this segment.")
    wall_time_s: float = Field(ge=0, description="Process wall time consumed by the segment.")
    native_wall_time_s: float = Field(ge=0, description="Wall time inside the native call.")
    provenance_sha256: str = Field(description="Hash of the segment provenance record.")


class Manifest(BaseModel):
    """Mutable run manifest, always replaced atomically."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(description="Wire schema version of the request.")
    request_sha256: str = Field(description="Hash of the frozen request.")
    status: RunStatus = Field(description="Status after the latest committed segment.")
    time_s: float = Field(ge=0, description="Simulation time reached.")
    duration_s: float = Field(gt=0, description="Requested duration.")
    complete: bool = Field(description="True only when the run is fully completed.")
    cumulative_wall_time_s: float = Field(ge=0, description="Wall time across segments.")
    segments: list[SegmentRecord] = Field(description="Committed segments in order.")

    @property
    def latest(self) -> SegmentRecord | None:
        """Return the newest committed segment.

        Returns:
            SegmentRecord | None: The last entry or None for an empty manifest.
        """
        return self.segments[-1] if self.segments else None


def write_atomic_text(path: Path, text: str, *, replace_existing: bool = True) -> None:
    """Write text via a same-directory temp file and `os.replace`.

    Parameters:
        path (Path): Destination path.
        text (str): Content to write.
        replace_existing (bool): Replace an existing destination, otherwise fail atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if replace_existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except FileExistsError as e:
        raise ArtifactError(f"Refusing to overwrite existing artifact '{path}'") from e
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_json(path: Path, payload: object) -> None:
    """Write canonical JSON atomically.

    Parameters:
        path (Path): Destination path.
        payload (object): JSON-like tree.
    """
    write_atomic_text(path, canonical_json(payload) + "\n")


def write_new_json(path: Path, payload: object) -> None:
    """Write JSON only when the file does not already exist.

    Parameters:
        path (Path): Destination path.
        payload (object): JSON-like tree.

    Raises:
        ArtifactError: When the file already exists.
    """
    write_atomic_text(path, canonical_json(payload) + "\n", replace_existing=False)


def frames_table(frames: list[Frame]) -> pa.Table:
    """Flatten frames into a long Arrow table.

    Parameters:
        frames (list[Frame]): Frames from one segment.

    Returns:
        pa.Table: One row per particle per frame.
    """
    columns: dict[str, list[Any]] = {name: [] for name in FRAME_SCHEMA.names}
    for frame_index, frame in enumerate(frames):
        for particle_index, position in enumerate(frame.positions_m):
            columns["frame_index"].append(frame_index)
            columns["time_s"].append(frame.time_s)
            columns["particle_index"].append(particle_index)
            columns["x_m"].append(position[0])
            columns["y_m"].append(position[1])
            columns["z_m"].append(position[2])
            columns["radius_m"].append(frame.radii_m[particle_index])
            columns["material"].append(frame.materials[particle_index])
            columns["box_id"].append(frame.box_ids[particle_index])
    return pa.table(columns, schema=FRAME_SCHEMA)


def metrics_table(metrics: list[MetricRow]) -> pa.Table:
    """Convert ledger rows to an Arrow table.

    Parameters:
        metrics (list[MetricRow]): Ledger samples from one segment.

    Returns:
        pa.Table: One row per sample.
    """
    columns: dict[str, list[Any]] = {name: [] for name in METRIC_SCHEMA.names}
    for row in metrics:
        columns["time_s"].append(row.time_s)
        columns["compartment"].append(str(row.compartment))
        columns["wood_kg"].append(row.wood_kg)
        columns["waste_kg"].append(row.waste_kg)
        columns["water_kg"].append(row.water_kg)
    return pa.table(columns, schema=METRIC_SCHEMA)


def write_parquet_new(path: Path, table: pa.Table) -> None:
    """Write a Parquet file atomically, refusing to overwrite.

    Parameters:
        path (Path): Destination path.
        table (pa.Table): Table to write.

    Raises:
        ArtifactError: When the destination already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as e:
        raise ArtifactError(f"Refusing to overwrite existing artifact '{path}'") from e
    finally:
        temporary.unlink(missing_ok=True)


class RunDirectory:
    """Handle on a run directory with append-only segment semantics."""

    def __init__(self, root: Path) -> None:
        """Bind to a directory without touching the filesystem.

        Parameters:
            root (Path): The run directory.
        """
        self.root = Path(str(root)).resolve()

    @property
    def request_path(self) -> Path:
        """Location of the frozen request.

        Returns:
            Path: `request.json` under the root.
        """
        return self.root / REQUEST_FILE

    @property
    def manifest_path(self) -> Path:
        """Location of the manifest.

        Returns:
            Path: `manifest.json` under the root.
        """
        return self.root / MANIFEST_FILE

    def segment_dir(self, index: int) -> Path:
        """Directory for a segment index.

        Parameters:
            index (int): Zero-based segment index.

        Returns:
            Path: `segments/NNNN` under the root.
        """
        return self.root / SEGMENTS_DIR / f"{index:04d}"

    def exists(self) -> bool:
        """Report whether a frozen request exists here.

        Returns:
            bool: True when `request.json` is present.
        """
        return self.request_path.is_file()

    def initialize(self, request: SimulationRequest) -> None:
        """Freeze a request into a new run directory.

        Parameters:
            request (SimulationRequest): Validated request.

        Raises:
            ArtifactError: When the directory already holds a run or non-empty content.
        """
        if self.root.exists() and any(self.root.iterdir()):
            raise ArtifactError(
                f"Run directory '{self.root}' is not empty; choose a new directory or "
                "pass `--resume` for an incomplete run"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        write_new_json(self.request_path, request.model_dump(mode="json"))
        LOGGER.info(f"froze request into '{self.request_path}'")

    def read_request(self) -> SimulationRequest:
        """Load the frozen request.

        Returns:
            SimulationRequest: The validated frozen request.

        Raises:
            ArtifactError: When the request file is missing or invalid.
        """
        if not self.request_path.is_file():
            raise ArtifactError(f"Run directory '{self.root}' has no '{REQUEST_FILE}'")
        try:
            return validate_payload(
                SimulationRequest, json.loads(self.request_path.read_text(encoding="utf-8"))
            )
        except ValueError as e:
            raise ArtifactError(f"Frozen request '{self.request_path}' is invalid: {e}") from e

    def read_manifest(self) -> Manifest | None:
        """Load the manifest when present.

        Returns:
            Manifest | None: The manifest, or None before the first committed segment.

        Raises:
            ArtifactError: When the manifest is corrupt.
        """
        if not self.manifest_path.is_file():
            return None
        try:
            return validate_payload(
                Manifest, json.loads(self.manifest_path.read_text(encoding="utf-8"))
            )
        except ValueError as e:
            raise ArtifactError(f"Manifest '{self.manifest_path}' is corrupt: {e}") from e

    def read_checkpoint(self, index: int) -> Checkpoint:
        """Load a committed segment checkpoint.

        Parameters:
            index (int): Segment index.

        Returns:
            Checkpoint: The validated checkpoint.

        Raises:
            ArtifactError: When the file is missing or invalid.
        """
        path = self.segment_dir(index) / CHECKPOINT_FILE
        if not path.is_file():
            raise ArtifactError(f"Segment checkpoint missing: '{path}'")
        try:
            return validate_payload(Checkpoint, json.loads(path.read_text(encoding="utf-8")))
        except ValueError as e:
            raise ArtifactError(f"Checkpoint '{path}' is corrupt: {e}") from e

    def read_metrics(self) -> pa.Table:
        """Concatenate metrics from every committed segment.

        Returns:
            pa.Table: Ledger rows in segment order.
        """
        manifest = self.read_manifest()
        tables = [
            pq.read_table(self.segment_dir(segment.index) / METRICS_FILE)
            for segment in (manifest.segments if manifest is not None else [])
        ]
        return pa.concat_tables(tables) if tables else METRIC_SCHEMA.empty_table()

    def read_frames(self) -> pa.Table:
        """Concatenate frames from every committed segment.

        Returns:
            pa.Table: Long particle table in segment order.
        """
        manifest = self.read_manifest()
        tables = [
            pq.read_table(self.segment_dir(segment.index) / FRAMES_FILE)
            for segment in (manifest.segments if manifest is not None else [])
        ]
        return pa.concat_tables(tables) if tables else FRAME_SCHEMA.empty_table()

    def recording_paths(self) -> list[Path]:
        """List committed segment recordings that exist on disk.

        Returns:
            list[Path]: `.rrd` paths in segment order.
        """
        manifest = self.read_manifest()
        if manifest is None:
            return []
        paths = [self.segment_dir(segment.index) / RECORDING_FILE for segment in manifest.segments]
        return [path for path in paths if path.is_file()]

    def next_segment_index(self) -> int:
        """Choose the next segment index, refusing stale partial directories.

        Returns:
            int: One past the last committed segment.

        Raises:
            ArtifactError: When an uncommitted segment directory is in the way.
        """
        manifest = self.read_manifest()
        index = 0 if manifest is None else len(manifest.segments)
        candidate = self.segment_dir(index)
        if candidate.exists():
            raise ArtifactError(
                f"Uncommitted partial segment '{candidate}' exists; inspect and remove it "
                "manually before resuming"
            )
        return index

    def write_segment(
        self,
        index: int,
        output: SimulationOutput,
        provenance: Provenance,
        record: SegmentRecord,
        extra_writer: Callable[[Path], None] | None = None,
    ) -> Path:
        """Write every segment file, then commit it to the manifest.

        Parameters:
            index (int): Segment index from `next_segment_index`.
            output (SimulationOutput): Validated native output.
            provenance (Provenance): Frozen provenance for this segment.
            record (SegmentRecord): Manifest entry describing the segment.
            extra_writer (Callable[[Path], None] | None): Hook that writes additional
                files (e.g. the Rerun recording) into the segment directory before
                the manifest commit.

        Returns:
            Path: The committed segment directory.

        Raises:
            ArtifactError: When the segment directory already exists.
        """
        directory = self.segment_dir(index)
        if directory.exists():
            raise ArtifactError(f"Refusing to overwrite segment directory '{directory}'")
        directory.mkdir(parents=True)
        summary = {
            "status": str(output.status),
            "run_status": str(record.run_status),
            "fidelity": str(output.fidelity),
            "mode": str(output.mode),
            "time_s": output.time_s,
            "observables": output.observables,
            "diagnostics": output.diagnostics,
            "frame_count": len(output.frames),
            "metric_count": len(output.metrics),
        }
        write_new_json(directory / OUTPUT_FILE, summary)
        write_parquet_new(directory / FRAMES_FILE, frames_table(output.frames))
        write_parquet_new(directory / METRICS_FILE, metrics_table(output.metrics))
        write_new_json(directory / CHECKPOINT_FILE, output.checkpoint.model_dump(mode="json"))
        write_new_json(directory / PROVENANCE_FILE, provenance.model_dump(mode="json"))
        if extra_writer is not None:
            extra_writer(directory)
        self._commit(record, output)
        LOGGER.info(f"committed segment {index} to '{directory}'")
        return directory

    def amend_latest_status(self, status: RunStatus, wall_time_s: float) -> None:
        """Downgrade the latest segment status after a post-commit deadline check.

        Only the manifest changes; segment files are never rewritten.

        Parameters:
            status (RunStatus): Replacement status for the latest segment.
            wall_time_s (float): Final measured wall time for that segment.

        Raises:
            ArtifactError: When there is no committed segment.
        """
        manifest = self.read_manifest()
        if manifest is None or manifest.latest is None:
            raise ArtifactError(f"Run '{self.root}' has no segment to amend")
        latest = manifest.latest
        delta = wall_time_s - latest.wall_time_s
        amended = latest.model_copy(update={"run_status": status, "wall_time_s": wall_time_s})
        replacement = manifest.model_copy(
            update={
                "status": status,
                "complete": status is RunStatus.COMPLETED,
                "cumulative_wall_time_s": manifest.cumulative_wall_time_s + delta,
                "segments": [*manifest.segments[:-1], amended],
            }
        )
        write_atomic_json(self.manifest_path, replacement.model_dump(mode="json"))

    def _commit(self, record: SegmentRecord, output: SimulationOutput) -> None:
        """Append a segment record to the manifest atomically.

        Parameters:
            record (SegmentRecord): New segment entry.
            output (SimulationOutput): Output used for time and status.
        """
        previous = self.read_manifest()
        request = self.read_request()
        segments = list(previous.segments) if previous is not None else []
        cumulative = previous.cumulative_wall_time_s if previous is not None else 0.0
        manifest = Manifest(
            schema_version=request.schema_version,
            request_sha256=hash_request(request),
            status=record.run_status,
            time_s=output.time_s,
            duration_s=request.duration_s,
            complete=record.run_status is RunStatus.COMPLETED,
            cumulative_wall_time_s=cumulative + record.wall_time_s,
            segments=[*segments, record],
        )
        write_atomic_json(self.manifest_path, manifest.model_dump(mode="json"))
