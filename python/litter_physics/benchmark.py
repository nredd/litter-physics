"""Wall-clock benchmarks with hardware provenance.

A benchmark repeats one request in throwaway run directories and reports wall time,
native time, and simulated seconds per wall second. Results describe the measured
host; they are not portable performance claims.

References:
    docs/plan.md (Acceptance)
"""

from __future__ import annotations

import logging
import shutil
import statistics
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from litter_physics.artifacts import write_atomic_json
from litter_physics.models import SimulationRequest
from litter_physics.native import NativeRunner
from litter_physics.provenance import HardwareInfo, build_provenance
from litter_physics.runner import start_run

LOGGER = logging.getLogger(__name__)

MAX_REPETITIONS = 100


class BenchmarkSample(BaseModel):
    """One repetition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repetition: int = Field(description="Zero-based repetition index.")
    status: str = Field(description="Run status of the repetition.")
    wall_time_s: float = Field(description="Total segment wall time.")
    native_wall_time_s: float = Field(description="Time inside the native call.")
    time_s: float = Field(description="Simulated time reached.")


class BenchmarkReport(BaseModel):
    """Benchmark summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_sha256: str = Field(description="Hash of the benchmarked request.")
    hardware: HardwareInfo = Field(description="Measured host.")
    native_sha256: str | None = Field(description="Hash of the compiled extension.")
    repetitions: int = Field(description="Number of repetitions executed.")
    samples: list[BenchmarkSample] = Field(description="Per-repetition timings.")
    median_wall_time_s: float = Field(description="Median wall time.")
    median_native_wall_time_s: float = Field(description="Median native time.")
    sim_seconds_per_wall_second: float = Field(description="Median throughput.")
    all_completed: bool = Field(description="Whether every repetition completed.")
    within_one_hour: bool = Field(
        description="Whether the slowest repetition stayed under 3600 s."
    )
    note: str = Field(description="Standing caveat.")


def run_benchmark(
    request: SimulationRequest,
    *,
    repetitions: int,
    runner: NativeRunner | None = None,
    scratch_dir: Path | None = None,
    report_path: Path | None = None,
) -> BenchmarkReport:
    """Repeat a request and summarize its timings.

    Parameters:
        request (SimulationRequest): Validated request.
        repetitions (int): Number of repetitions, 1..100.
        runner (NativeRunner | None): Injected native callable for tests.
        scratch_dir (Path | None): Directory for throwaway runs; a temp dir when None.
        report_path (Path | None): Optional JSON report destination.

    Returns:
        BenchmarkReport: The summary.

    Raises:
        ValueError: When `repetitions` is out of range.
    """
    if not 1 <= repetitions <= MAX_REPETITIONS:
        raise ValueError(f"`repetitions` '{repetitions}' must be in 1..{MAX_REPETITIONS}")
    provenance = build_provenance(request)
    owned_scratch = scratch_dir is None
    scratch = (
        Path(tempfile.mkdtemp(prefix="litter-benchmark-"))
        if scratch_dir is None
        else Path(str(scratch_dir)).resolve()
    )
    samples: list[BenchmarkSample] = []
    try:
        for repetition in range(repetitions):
            result = start_run(
                request, scratch / f"rep-{repetition:03d}", runner=runner, recording=False
            )
            samples.append(
                BenchmarkSample(
                    repetition=repetition,
                    status=str(result.run_status),
                    wall_time_s=result.wall_time_s,
                    native_wall_time_s=result.native_wall_time_s,
                    time_s=result.time_s,
                )
            )
            LOGGER.info(
                f"benchmark rep {repetition}: {result.run_status} wall={result.wall_time_s:.3f}s"
            )
    finally:
        if owned_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    median_wall = statistics.median(sample.wall_time_s for sample in samples)
    median_native = statistics.median(sample.native_wall_time_s for sample in samples)
    median_sim = statistics.median(sample.time_s for sample in samples)
    report = BenchmarkReport(
        request_sha256=provenance.request_sha256,
        hardware=provenance.hardware,
        native_sha256=provenance.native_sha256,
        repetitions=repetitions,
        samples=samples,
        median_wall_time_s=median_wall,
        median_native_wall_time_s=median_native,
        sim_seconds_per_wall_second=median_sim / median_wall if median_wall > 0 else 0.0,
        all_completed=all(sample.status == "completed" for sample in samples),
        within_one_hour=max(sample.wall_time_s for sample in samples) <= 3600.0,
        note=(
            "Timings describe this host only; they are not a fidelity or accuracy claim. "
            "Compilation time is excluded."
        ),
    )
    if report_path is not None:
        write_atomic_json(Path(str(report_path)).resolve(), report.model_dump(mode="json"))
    return report
