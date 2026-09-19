"""CLI behaviour with the fake solver registered as `_core.run_json`.

References:
    docs/cli.md
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from litter_physics import native
from litter_physics.cli import EXIT_FAILURE, EXIT_INCOMPLETE, EXIT_OK, EXIT_USAGE, main
from tests.conftest import EXAMPLES, SCHEMAS
from tests.fake_core import FakeCore

HOUSEHOLD = str(EXAMPLES / "household_basic.yaml")


def test_validate(capsys: pytest.CaptureFixture[str]) -> None:
    """Valid and invalid examples produce the right exit codes."""
    assert main(["validate", HOUSEHOLD]) == EXIT_OK
    assert "VALID" in capsys.readouterr().out
    assert main(["validate", str(EXAMPLES / "invalid_event_order.yaml")]) == EXIT_FAILURE
    assert "INVALID" in capsys.readouterr().out
    assert (
        main(["validate", "--kind", "observations", str(EXAMPLES / "observations_synthetic.json")])
        == EXIT_OK
    )
    assert main(["validate", "--kind", "sweep", str(EXAMPLES / "sweep_friction.yaml")]) == EXIT_OK
    assert (
        main(["validate", "--kind", "profiles", str(EXAMPLES / "cat_profiles_synthetic.yaml")])
        == EXIT_OK
    )
    assert main(["validate", "/nonexistent.yaml"]) == EXIT_FAILURE


def test_schema_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Schema export writes files and `--check` detects staleness."""
    assert main(["schema", "--check", str(SCHEMAS)]) == EXIT_OK
    assert main(["schema", str(tmp_path / "out")]) == EXIT_OK
    assert (tmp_path / "out" / "request.schema.json").is_file()
    (tmp_path / "out" / "request.schema.json").write_text("{}")
    assert main(["schema", "--check", str(tmp_path / "out")]) == EXIT_FAILURE
    assert "STALE" in capsys.readouterr().out


def test_run_without_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without `run_json`, `run` fails loudly and writes nothing."""

    monkeypatch.delattr(native._core, "run_json", raising=False)
    assert main(["run", HOUSEHOLD, "--out", str(tmp_path / "run")]) == EXIT_FAILURE
    assert "UNAVAILABLE" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_run_resume_and_exit_codes(
    tmp_path: Path, patched_native: FakeCore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Incomplete runs exit 3; resume finishes with 0; overwrite is refused."""
    patched_native.max_sim_time_per_call = 0.6
    out = str(tmp_path / "run")
    assert main(["run", HOUSEHOLD, "--out", out]) == EXIT_INCOMPLETE
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "budget_exhausted" and first["resumable"]
    assert main(["run", "--resume", "--out", out]) == EXIT_OK
    second = json.loads(capsys.readouterr().out)
    assert second["complete"] and second["segment_index"] == 1
    assert main(["run", HOUSEHOLD, "--out", out]) == EXIT_FAILURE
    assert "not empty" in capsys.readouterr().err
    assert main(["run", "--resume", "--out", out]) == EXIT_FAILURE


def test_run_usage_error(patched_native: FakeCore, tmp_path: Path) -> None:
    """`run` without a request and without `--resume` is a usage error."""
    with pytest.raises(SystemExit) as info:
        main(["run", "--out", str(tmp_path / "run")])
    assert info.value.code == EXIT_USAGE


def test_run_with_calibration(
    tmp_path: Path, patched_native: FakeCore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Calibrate then run with the bundle; provenance records the bundle hash."""
    bundle = tmp_path / "bundle.json"
    assert (
        main(
            [
                "calibrate",
                "--observations",
                str(EXAMPLES / "observations_synthetic.json"),
                "--template",
                HOUSEHOLD,
                "--out",
                str(bundle),
                "--report",
                str(tmp_path / "report.txt"),
            ]
        )
        == EXIT_OK
    )
    assert "synthetic=True" in capsys.readouterr().out
    assert (tmp_path / "report.txt").is_file()
    out = tmp_path / "run"
    assert (
        main(["run", HOUSEHOLD, "--out", str(out), "--calibration", str(bundle), "--no-recording"])
        == EXIT_OK
    )
    provenance = json.loads((out / "segments" / "0000" / "provenance.json").read_text())
    assert provenance["calibration_sha256"] is not None
    request = json.loads((out / "request.json").read_text())
    assert request["materials"]["water_capacity_ratio"] == pytest.approx(2.8, rel=1e-3)
    assert not (out / "segments" / "0000" / "recording.rrd").exists()
    assert (
        main(
            [
                "calibrate",
                "--observations",
                str(EXAMPLES / "observations_synthetic.json"),
                "--template",
                HOUSEHOLD,
                "--out",
                str(tmp_path / "bundle2.json"),
                "--run",
                str(out),
            ]
        )
        == EXIT_OK
    )
    assert "Household removed mass" in capsys.readouterr().out


def test_sweep_and_benchmark(
    tmp_path: Path, patched_native: FakeCore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sweep and benchmark commands complete with the fake solver."""
    assert (
        main(["sweep", str(EXAMPLES / "sweep_friction.yaml"), "--out", str(tmp_path / "sweep")])
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["counts"] == {"completed": 8}
    assert (
        main(["benchmark", HOUSEHOLD, "--repetitions", "2", "--report", str(tmp_path / "b.json")])
        == EXIT_OK
    )
    report = json.loads(capsys.readouterr().out)
    assert report["repetitions"] == 2 and (tmp_path / "b.json").is_file()


def test_observations_and_visits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Synthesize observations, summarize them, and generate visits from them."""
    observations = tmp_path / "obs.yaml"
    assert (
        main(
            [
                "observations",
                "synthesize",
                "--out",
                str(observations),
                "--cats",
                "1",
                "--visits",
                "6",
            ]
        )
        == EXIT_OK
    )
    assert "SYNTHETIC" in capsys.readouterr().out
    assert main(["observations", "summary", str(observations)]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["visits_per_cat"] == {"cat-0": 6}
    out = tmp_path / "visits.yaml"
    assert (
        main(
            [
                "visits",
                "--template",
                HOUSEHOLD,
                "--observations",
                str(observations),
                "--horizon",
                "43200",
                "--out",
                str(out),
                "--log",
                str(tmp_path / "log.json"),
            ]
        )
        == EXIT_OK
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["synthetic"] and summary["events"] >= 0
    assert main(["validate", str(out)]) == EXIT_OK
    assert (
        main(["visits", "--template", HOUSEHOLD, "--horizon", "3600", "--out", str(out)])
        == EXIT_FAILURE
    )
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert (
        main(
            [
                "visits",
                "--template",
                HOUSEHOLD,
                "--profiles",
                str(EXAMPLES / "cat_profiles_synthetic.yaml"),
                "--horizon",
                "3600",
                "--out",
                str(tmp_path / "p.yaml"),
            ]
        )
        == EXIT_OK
    )
    assert (
        main(
            [
                "visits",
                "--template",
                str(EXAMPLES / "research_slump.yaml"),
                "--horizon",
                "10",
                "--out",
                str(tmp_path / "r.yaml"),
            ]
        )
        == EXIT_FAILURE
    )


def test_view_serves_and_stops(
    tmp_path: Path, patched_native: FakeCore, capsys: pytest.CaptureFixture[str]
) -> None:
    """`view` serves for a bounded duration on free loopback ports and exits 0."""

    def free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    out = tmp_path / "run"
    assert main(["run", HOUSEHOLD, "--out", str(out)]) == EXIT_OK
    capsys.readouterr()
    code = main(
        [
            "view",
            str(out),
            "--grpc-port",
            str(free_port()),
            "--web-port",
            str(free_port()),
            "--duration",
            "1.0",
        ]
    )
    assert code == EXIT_OK
    assert "loopback only" in capsys.readouterr().out
    assert main(["view", str(tmp_path / "nothing")]) == EXIT_FAILURE
