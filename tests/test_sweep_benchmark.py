"""Sweeps keep per-job gates; benchmarks report hardware (fake core).

References:
    docs/plan.md (Scope and claims)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litter_physics import sweep as sweep_module
from litter_physics.benchmark import run_benchmark
from litter_physics.models import SimulationRequest, load_document
from litter_physics.sweep import expand_jobs, load_spec, read_summary, run_sweep, set_by_path
from tests.conftest import EXAMPLES
from tests.fake_core import FakeCore

SPEC = EXAMPLES / "sweep_friction.yaml"


def test_set_by_path_errors() -> None:
    """Dotted path assignment reports bad keys and indexes."""
    tree = {"a": {"b": [1, 2]}, "c": 1}
    set_by_path(tree, "a.b.1", 5)
    assert tree["a"]["b"] == [1, 5]
    with pytest.raises(KeyError):
        set_by_path(tree, "a.zzz", 1)
    with pytest.raises(TypeError):
        set_by_path(tree, "a.b.x", 1)
    with pytest.raises(TypeError):
        set_by_path(tree, "c.d", 1)
    with pytest.raises(IndexError):
        set_by_path(tree, "a.b.9", 1)


def test_expand_and_run_sweep(tmp_path: Path) -> None:
    """Every grid point runs in its own directory and the summary counts statuses."""
    spec = load_spec(SPEC)
    jobs = expand_jobs(spec, load_document(EXAMPLES / "household_basic.yaml"))
    assert len(jobs) == 8
    assert {job[1] for job in jobs} == {1, 2}
    core = FakeCore()
    outcomes = run_sweep(SPEC, tmp_path / "sweep", runner=core)
    assert core.calls == 8
    assert all(outcome.status == "completed" for outcome in outcomes)
    table = read_summary(tmp_path / "sweep")
    assert table.num_rows == 8
    assert set(table.column_names) >= {"override:materials.friction", "status"}
    summary = json.loads((tmp_path / "sweep" / "sweep_summary.json").read_text())
    assert summary["counts"] == {"completed": 8}
    assert (tmp_path / "sweep" / "jobs" / "0007" / "manifest.json").is_file()
    with pytest.raises(FileExistsError):
        run_sweep(SPEC, tmp_path / "sweep", runner=core)


def test_sweep_total_budget_stops_starting_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep budget marks unstarted jobs rather than shrinking per-job budgets."""
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 10.0
        return clock["now"]

    monkeypatch.setattr(sweep_module.time, "monotonic", fake_monotonic)
    outcomes = run_sweep(SPEC, tmp_path / "sweep", runner=FakeCore(), total_wall_budget_s=25.0)
    statuses = [outcome.status for outcome in outcomes]
    assert "not_started" in statuses and statuses[0] == "completed"


def test_sweep_records_failures(tmp_path: Path) -> None:
    """A failing job is recorded as failed and the sweep continues."""
    outcomes = run_sweep(SPEC, tmp_path / "sweep", runner=FakeCore(fail_with="solver diverged"))
    assert all(outcome.status == "failed" for outcome in outcomes)
    assert outcomes[0].error is not None and "solver diverged" in outcomes[0].error


def test_invalid_sweep_value_rejected(tmp_path: Path) -> None:
    """Grid points are validated before any job starts."""
    spec = tmp_path / "bad.yaml"
    spec.write_text(
        "base_request: base.yaml\naxes:\n  - path: materials.restitution\n    values: [2.0]\n"
    )
    (tmp_path / "base.yaml").write_text((EXAMPLES / "household_basic.yaml").read_text())
    with pytest.raises(ValueError, match="invalid"):
        run_sweep(spec, tmp_path / "sweep", runner=FakeCore())
    assert not (tmp_path / "sweep").exists()


def test_benchmark_report(household_request: SimulationRequest, tmp_path: Path) -> None:
    """Benchmarks record per-repetition timings and host details."""
    report = run_benchmark(
        household_request,
        repetitions=2,
        runner=FakeCore(),
        scratch_dir=tmp_path / "scratch",
        report_path=tmp_path / "bench.json",
    )
    assert report.repetitions == 2 and report.all_completed and report.within_one_hour
    assert report.hardware.cpu_count >= 1
    assert report.sim_seconds_per_wall_second > 0.0
    assert json.loads((tmp_path / "bench.json").read_text())["repetitions"] == 2
    with pytest.raises(ValueError, match="repetitions"):
        run_benchmark(household_request, repetitions=0, runner=FakeCore())
