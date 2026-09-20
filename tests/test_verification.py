"""Refinement studies fail closed, never pass silently, and run the real kernel.

Fake-core tests exercise planning, budgets, evaluation and exit codes. The final test
runs a small actual study through the compiled kernel and asserts only what a real
run can honestly show: completion, artifacts, provenance and a machine-readable
outcome that is `passed` or `unresolved`, never a fabricated pass.

References:
    docs/verification.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from litter_physics import native
from litter_physics import verification as verification_module
from litter_physics.cli import EXIT_FAILURE, EXIT_INCOMPLETE, EXIT_OK, main
from litter_physics.models import Fidelity, SimulationRequest, load_request, validate_payload
from litter_physics.sweep import set_by_path
from litter_physics.verification import (
    AxisReport,
    CaseOutcome,
    CaseStatus,
    ChangeTrend,
    ObservableOutcome,
    ObservableSpec,
    ObservableVerdict,
    RefinementAxis,
    StudyOutcome,
    StudyPlanError,
    StudySpec,
    check_grid_fit,
    evaluate_axis,
    evaluate_observable,
    load_study,
    plan_cases,
    run_study,
)
from tests.conftest import EXAMPLES
from tests.fake_core import FakeCore

STUDY = EXAMPLES / "verification_slump.yaml"
BASE = EXAMPLES / "research_slump.yaml"


class RefinementFakeCore:
    """Fake solver whose observables converge like `exact + c * h**order`.

    Not physics. It reports `max_dt_s`, `step_count` and `rejected_steps` the way the
    kernel does so the timestep-binding checks can be exercised.
    """

    def __init__(
        self,
        *,
        order: float = 2.0,
        exact: float = 0.01,
        coefficient: float = 1.0,
        dt_coefficient: float = 1.0,
        binding: bool = True,
        fail_on_call: int | None = None,
        diagnostics: dict[str, float | None] | None = None,
        fidelity: str | None = None,
    ) -> None:
        self.inner = FakeCore()
        self.order = order
        self.exact = exact
        self.coefficient = coefficient
        self.dt_coefficient = dt_coefficient
        self.binding = binding
        self.fail_on_call = fail_on_call
        self.diagnostics = diagnostics or {}
        self.fidelity = fidelity
        self.calls = 0

    def __call__(self, request_json: str, resume_json: str | None) -> str:
        """Mimic `run_json` with resolution-dependent observables.

        Parameters:
            request_json (str): Canonical request JSON.
            resume_json (str | None): Checkpoint JSON or None.

        Returns:
            str: Contract-shaped output JSON.

        Raises:
            ValueError: On the configured failing call.
        """
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise ValueError("fake kernel diverged")
        request = json.loads(request_json)
        output = json.loads(self.inner(request_json, resume_json))
        spacing = float(request["research"]["grid_spacing_m"])
        dt = float(request["dt_s"])
        value = self.exact + self.coefficient * spacing**self.order + self.dt_coefficient * dt
        observables: dict[str, float] = {
            "slump_m": value,
            "mass_residual_kg": 0.0,
            "max_dt_s": dt if self.binding else 0.5 * dt,
            "step_count": float(round(request["duration_s"] / dt)),
            "rejected_steps": 0.0,
            "limited_steps": 0.0,
        }
        for name, override in self.diagnostics.items():
            if override is None:
                observables.pop(name)
            else:
                observables[name] = override
        output["observables"] = observables
        if self.fidelity is not None:
            output["fidelity"] = self.fidelity
        return json.dumps(output)


def _spec_payload(**overrides: Any) -> dict[str, Any]:
    """Build a small study payload pointing at the committed base request.

    Parameters:
        **overrides (Any): Top-level keys to replace.

    Returns:
        dict[str, Any]: Study payload.
    """
    payload: dict[str, Any] = {
        "base_request": str(BASE),
        "label": "test",
        "spatial": {"grid_spacings_m": [0.01, 0.005, 0.0025], "dt_s": 0.0001},
        "temporal": {"grid_spacing_m": 0.01, "dt_values_s": [0.0004, 0.0002, 0.0001]},
        "observables": [{"name": "slump_m", "near_zero_scale": 0.0005}],
        "per_case_wall_budget_s": 60.0,
        "total_wall_budget_s": 300.0,
    }
    return payload | overrides


def _write_spec(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write a study spec next to a copy of the base request.

    Parameters:
        tmp_path (Path): Test directory.
        payload (dict[str, Any]): Study payload.

    Returns:
        Path: The spec path.
    """
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_example_study_loads_and_plans() -> None:
    """The committed example validates and expands to six grid-fitting cases."""
    spec = load_study(STUDY)
    cases = plan_cases(spec, yaml.safe_load(BASE.read_text(encoding="utf-8")))
    assert [case.axis for case in cases] == [RefinementAxis.SPATIAL] * 3 + [
        RefinementAxis.TEMPORAL
    ] * 3
    assert {case.request.dt_s for case in cases[:3]} == {0.00005}
    assert {
        case.request.research.grid_spacing_m for case in cases[3:] if case.request.research
    } == {0.005}
    assert {case.request.seed for case in cases} == {7}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"spatial": {"grid_spacings_m": [0.01, 0.005], "dt_s": 0.0001}}, "at least 3"),
        ({"spatial": {"grid_spacings_m": [0.005, 0.01, 0.0025], "dt_s": 0.0001}}, "decreasing"),
        ({"spatial": None, "temporal": None}, "needs"),
        (
            {
                "observables": [
                    {"name": "slump_m", "near_zero_scale": 1.0},
                    {"name": "slump_m", "near_zero_scale": 1.0},
                ]
            },
            "unique",
        ),
        ({"max_relative_change": 0.1}, "less than or equal"),
        ({"per_case_wall_budget_s": 3601.0}, "less than or equal"),
        ({"observables": [{"name": "slump_m", "near_zero_scale": 0.0}]}, "greater than 0"),
    ],
)
def test_spec_validation(override: dict[str, Any], match: str) -> None:
    """Under-resolved, mis-ordered, duplicate, or over-threshold specs are rejected."""
    with pytest.raises(ValueError, match=match):
        StudySpec.model_validate(_spec_payload(**override))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("research.grid_spacing_m", 0.007, "integer multiple"),
        ("research.initial_size_m", [0.021, 0.02, 0.02], "subcells"),
        ("research.initial_size_m", [0.004, 0.02, 0.02], "at least one cell"),
        ("record_interval_s", 0.0001, "point samples"),
    ],
)
def test_grid_fit_fails_closed(path: str, value: Any, match: str) -> None:
    """Cases the kernel would reject are refused before anything runs."""
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    payload["dt_s"] = 0.0001
    set_by_path(payload, path, value)
    request = validate_payload(SimulationRequest, payload)
    with pytest.raises(StudyPlanError, match=match):
        check_grid_fit(request, "case")


def test_particle_cap_fails_closed() -> None:
    """A block finer than the native particle cap is refused."""
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    payload["research"]["domain_m"] = [0.12, 0.12, 0.12]
    payload["research"]["initial_size_m"] = [0.12, 0.12, 0.12]
    payload["research"]["grid_spacing_m"] = 0.004
    payload["record_interval_s"] = 0.5
    request = validate_payload(SimulationRequest, payload)
    with pytest.raises(StudyPlanError, match="particles"):
        check_grid_fit(request, "case")


def test_hydrostatic_example_plans_below_initial_acoustic_limit() -> None:
    """Keep the diagnostic's axes grid-fitting and below the initial acoustic cap."""
    spec = load_study(EXAMPLES / "verification_hydrostatic_water.yaml")
    base = yaml.safe_load(
        (EXAMPLES / "research_hydrostatic_water.yaml").read_text(encoding="utf-8")
    )
    cases = plan_cases(spec, base)
    assert len(cases) == 6
    counts = []
    for case in cases:
        cfg = case.request.research
        assert cfg is not None
        assert cfg.fixture.value == "hydrostatic"
        assert cfg.material.value == "water"
        bulk = cfg.young_modulus_pa / (3 * (1 - 2 * cfg.poisson_ratio))
        wave_speed = math.sqrt(bulk / cfg.density_kg_m3)
        assert case.request.dt_s < 0.3 * cfg.grid_spacing_m / wave_speed
        counts.append(
            math.prod(round(size / (cfg.grid_spacing_m / 2)) for size in cfg.initial_size_m)
        )
    assert counts == [500, 4000, 32000, 4000, 4000, 4000]
    assert {case.request.dt_s for case in cases[:3]} == {0.000025}
    assert [case.request.dt_s for case in cases[3:]] == [0.00005, 0.000025, 0.0000125]


def test_small_changes_in_large_pressure_errors_are_not_absolute_accuracy() -> None:
    """The relative-change verdict must not be presented as an analytic-error gate."""
    observable = ObservableSpec(name="hydrostatic_max_relative_error", near_zero_scale=0.001)
    result = evaluate_observable(observable, [0.004, 0.002, 0.001], [0.31, 0.30, 0.299], 0.05)
    assert result.outcome is ObservableOutcome.PASSED
    assert result.values[-1] == 0.299  # Still almost 30% error against hydrostatic equilibrium.


def test_plan_rejects_household_base() -> None:
    """Only research requests can be refined."""
    base = yaml.safe_load((EXAMPLES / "household_basic.yaml").read_text(encoding="utf-8"))
    with pytest.raises(StudyPlanError, match="research request"):
        plan_cases(StudySpec.model_validate(_spec_payload()), base)


def test_evaluate_observable_verdicts() -> None:
    """Pass, unresolved, uninformative, missing and nonfinite are all distinct."""
    observable = ObservableSpec(name="x", near_zero_scale=1e-3)
    h = [0.04, 0.02, 0.01]
    passed = evaluate_observable(observable, h, [1.16, 1.04, 1.01], 0.05)
    assert passed.outcome is ObservableOutcome.PASSED
    assert passed.trend is ChangeTrend.DECREASING
    assert passed.observed_order is not None and math.isclose(passed.observed_order, 2.0)
    assert passed.final_relative_change is not None and passed.final_relative_change < 0.05
    unresolved = evaluate_observable(observable, h, [1.0, 1.2, 1.5], 0.05)
    assert unresolved.outcome is ObservableOutcome.UNRESOLVED
    assert unresolved.trend is ChangeTrend.NON_MONOTONE
    tiny = evaluate_observable(observable, h, [1e-5, 2e-6, 1e-6], 0.05)
    assert tiny.outcome is ObservableOutcome.UNINFORMATIVE and tiny.near_zero
    missing = evaluate_observable(observable, h, [1.0, None, 1.0], 0.05)
    assert missing.outcome is ObservableOutcome.MISSING
    assert missing.observed_order is None
    nonfinite = evaluate_observable(observable, h, [1.0, math.nan, 1.0], 0.05)
    assert nonfinite.outcome is ObservableOutcome.NONFINITE
    boundary = evaluate_observable(observable, h, [1.0, 1.0, 1.06], 0.05)
    assert boundary.outcome is ObservableOutcome.UNRESOLVED
    assert boundary.final_relative_change is not None
    assert math.isclose(boundary.final_relative_change, 0.06 / 1.06)
    nonuniform = evaluate_observable(observable, [0.04, 0.02, 0.015], [1.16, 1.04, 1.01], 0.05)
    assert nonuniform.observed_order is None
    with pytest.raises(ValueError, match="at least 3"):
        evaluate_observable(observable, [0.1, 0.05], [1.0, 1.0], 0.05)


def test_study_passes_and_refuses_overwrite(tmp_path: Path) -> None:
    """A convergent fake kernel passes both axes; the study directory is write-once."""
    spec = _write_spec(tmp_path, _spec_payload())
    core = RefinementFakeCore(coefficient=0.1, dt_coefficient=0.1)
    report = run_study(spec, tmp_path / "study", runner=core)
    assert core.calls == 6
    assert report.outcome is StudyOutcome.PASSED
    assert report.study_complete and report.criterion_met
    assert report.fidelity == "research_unvalidated"
    assert [axis.timestep_cap_binding for axis in report.axes] == [True, True]
    assert report.omitted_gates and "validation" in report.claim
    written = json.loads((tmp_path / "study" / "study_report.json").read_text())
    assert written["outcome"] == "passed"
    assert (tmp_path / "study" / "cases" / "spatial" / "0002" / "manifest.json").is_file()
    assert (tmp_path / "study" / "study_spec.json").is_file()
    with pytest.raises(FileExistsError):
        run_study(spec, tmp_path / "study", runner=core)


def test_study_unresolved_when_not_converged(tmp_path: Path) -> None:
    """Large final changes are reported unresolved even though every case completed."""
    spec = _write_spec(tmp_path, _spec_payload(temporal=None))
    report = run_study(spec, tmp_path / "study", runner=RefinementFakeCore(coefficient=200.0))
    assert report.study_complete and not report.criterion_met
    assert report.outcome is StudyOutcome.UNRESOLVED
    verdict = report.axes[0].observables[0]
    assert verdict.outcome is ObservableOutcome.UNRESOLVED
    assert verdict.observed_order is not None and math.isclose(verdict.observed_order, 2.0)


def test_study_unresolved_when_cap_not_binding(tmp_path: Path) -> None:
    """If the adaptive limiter set the step, the axis cannot pass."""
    spec = _write_spec(tmp_path, _spec_payload(spatial=None))
    report = run_study(spec, tmp_path / "study", runner=RefinementFakeCore(binding=False))
    axis = report.axes[0]
    assert axis.complete and axis.timestep_cap_binding is False
    assert axis.outcome is StudyOutcome.UNRESOLVED
    assert any("does not equal `dt_s`" in reason for reason in axis.reasons)


@pytest.mark.parametrize(
    ("diagnostics", "match"),
    [
        ({"limited_steps": None}, "`limited_steps` is None"),
        ({"limited_steps": 1.0}, "`limited_steps` 1 > 0"),
        ({"rejected_steps": 2.0}, "`rejected_steps` 2 > 0"),
        ({"step_count": 12.5}, "not a finite nonnegative integer"),
        ({"step_count": -1.0}, "not a finite nonnegative integer"),
        ({"step_count": 0.0}, "is zero"),
        ({"max_dt_s": None}, "`max_dt_s` None"),
    ],
)
def test_adaptive_diagnostics_fail_closed(
    tmp_path: Path, diagnostics: dict[str, float | None], match: str
) -> None:
    """Missing, nonintegral or nonzero adaptive-step diagnostics make an axis unresolved."""
    spec = _write_spec(tmp_path, _spec_payload(temporal=None))
    core = RefinementFakeCore(coefficient=0.1, diagnostics=diagnostics)
    report = run_study(spec, tmp_path / "study", runner=core)
    axis = report.axes[0]
    assert axis.complete and axis.timestep_cap_binding is False
    assert axis.outcome is StudyOutcome.UNRESOLVED
    assert any(match in reason for reason in axis.reasons), axis.reasons
    assert report.outcome is StudyOutcome.UNRESOLVED


def test_non_research_fidelity_cannot_pass(tmp_path: Path) -> None:
    """A completed study whose outputs carry another fidelity label is unresolved."""
    spec = _write_spec(tmp_path, _spec_payload(temporal=None))
    core = RefinementFakeCore(coefficient=0.1, fidelity=str(Fidelity.PRELIMINARY_HOUSEHOLD))
    report = run_study(spec, tmp_path / "study", runner=core)
    assert report.study_complete and not report.criterion_met
    assert report.outcome is StudyOutcome.UNRESOLVED
    assert report.fidelity == "preliminary_household"
    assert any("fidelity" in reason for reason in report.axes[0].reasons)
    assert any("fidelity" in reason for reason in report.reasons)


def _completed_case(index: int, spacing: float, **overrides: Any) -> CaseOutcome:
    """Build a completed spatial case with binding diagnostics.

    Parameters:
        index (int): Case index.
        spacing (float): Grid spacing.
        **overrides (Any): Fields to replace.

    Returns:
        CaseOutcome: The case.
    """
    payload: dict[str, Any] = {
        "axis": RefinementAxis.SPATIAL,
        "case_index": index,
        "run_dir": f"/tmp/case-{index}",
        "grid_spacing_m": spacing,
        "dt_s": 1e-4,
        "wall_budget_s": 10.0,
        "status": CaseStatus.COMPLETED,
        "duration_s": 0.5,
        "time_s": 0.5,
        "wall_time_s": 1.0,
        "fidelity": str(Fidelity.RESEARCH_UNVALIDATED),
        "observables": {
            "slump_m": 0.01 + 0.1 * spacing**2,
            "max_dt_s": 1e-4,
            "step_count": 5000.0,
            "rejected_steps": 0.0,
            "limited_steps": 0.0,
        },
        "error": None,
    }
    return CaseOutcome.model_validate(payload | overrides)


def test_completed_case_short_of_duration_cannot_pass() -> None:
    """A `completed` case that stopped before `duration_s` is rejected, report kept."""
    spacings = [0.04, 0.02, 0.01]
    cases = [_completed_case(index, spacing) for index, spacing in enumerate(spacings)]
    observable = ObservableSpec(name="slump_m", near_zero_scale=1e-4)
    good: AxisReport = evaluate_axis(
        RefinementAxis.SPATIAL, spacings, ("dt_s", 1e-4), cases, [observable], 0.05
    )
    assert good.outcome is StudyOutcome.PASSED
    cases[1] = _completed_case(1, spacings[1], time_s=0.4)
    short = evaluate_axis(
        RefinementAxis.SPATIAL, spacings, ("dt_s", 1e-4), cases, [observable], 0.05
    )
    assert short.complete and short.outcome is StudyOutcome.UNRESOLVED
    assert any("not the requested duration" in reason for reason in short.reasons)
    assert short.observables[0].outcome is ObservableOutcome.PASSED


def test_derived_overflow_is_nonfinite_and_serializable() -> None:
    """Finite extrema whose difference overflows never enter the report as inf."""
    observable = ObservableSpec(name="x", near_zero_scale=1e-3)
    verdict = evaluate_observable(observable, [0.04, 0.02, 0.01], [1e308, -1e308, 1e308], 0.05)
    assert verdict.outcome is ObservableOutcome.NONFINITE
    assert verdict.absolute_changes == [None, None]
    assert verdict.relative_changes == [None, None]
    assert verdict.observed_order is None
    assert any("overflowed" in reason for reason in verdict.reasons)
    json.loads(verdict.model_dump_json())
    ratio = evaluate_observable(
        ObservableSpec(name="x", near_zero_scale=5e-324),
        [0.04, 0.02, 0.01],
        [1e300, 0.0, 5e-324],
        0.05,
    )
    assert ratio.outcome is ObservableOutcome.NONFINITE
    nonfinite = evaluate_observable(observable, [0.04, 0.02, 0.01], [1.0, math.inf, 1.0], 0.05)
    assert nonfinite.values == [1.0, None, 1.0]
    json.loads(nonfinite.model_dump_json())
    with pytest.raises(ValueError, match="finite"):
        ObservableVerdict.model_validate(
            verdict.model_dump() | {"final_relative_change": math.nan}
        )


def test_subnormal_spacing_fails_closed() -> None:
    """A subnormal spacing that makes `domain / spacing` infinite is a plan error."""
    assert verification_module._near_integer(math.inf, 1e-6) is False
    assert verification_module._near_integer(math.nan, 1e-6) is False
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    spec = StudySpec.model_validate(
        _spec_payload(
            temporal=None, spatial={"grid_spacings_m": [0.01, 0.005, 5e-324], "dt_s": 1e-4}
        )
    )
    with pytest.raises(StudyPlanError):
        plan_cases(spec, base)


def test_total_budget_overrun_is_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete, convergent cases still cannot pass once the study cap was overrun."""
    clock = {"now": 0.0}

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            value = clock["now"]
            clock["now"] += 30.0
            return value

    monkeypatch.setattr(verification_module, "time", FakeTime)
    spec = _write_spec(
        tmp_path,
        _spec_payload(temporal=None, per_case_wall_budget_s=50.0, total_wall_budget_s=100.0),
    )
    report = run_study(spec, tmp_path / "study", runner=RefinementFakeCore(coefficient=0.1))
    assert report.study_complete and report.axes[0].outcome is StudyOutcome.PASSED
    assert report.total_budget_exceeded and report.total_wall_time_s == 120.0
    assert not report.criterion_met and report.outcome is StudyOutcome.UNRESOLVED
    assert any("exceeded the study cap" in reason for reason in report.reasons)
    assert [case.wall_budget_s for case in report.axes[0].cases] == [50.0, 40.0, 10.0]


def test_study_incomplete_keeps_artifacts(tmp_path: Path) -> None:
    """A failing case marks the study incomplete and keeps the other case artifacts."""
    spec = _write_spec(tmp_path, _spec_payload(temporal=None))
    report = run_study(spec, tmp_path / "study", runner=RefinementFakeCore(fail_on_call=2))
    assert report.outcome is StudyOutcome.INCOMPLETE
    assert not report.study_complete and not report.criterion_met
    statuses = [case.status for case in report.axes[0].cases]
    assert statuses == [CaseStatus.COMPLETED, CaseStatus.FAILED, CaseStatus.COMPLETED]
    assert (
        report.axes[0].cases[1].error is not None and "diverged" in report.axes[0].cases[1].error
    )
    assert report.axes[0].observables[0].outcome is ObservableOutcome.NOT_EVALUATED
    assert (tmp_path / "study" / "cases" / "spatial" / "0000" / "manifest.json").is_file()
    assert (tmp_path / "study" / "cases" / "spatial" / "0002" / "manifest.json").is_file()
    assert not (tmp_path / "study" / "cases" / "spatial" / "0001" / "manifest.json").exists()


def test_study_total_budget_caps_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The total cap shrinks a case budget and then stops starting cases."""
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 20.0
        return clock["now"]

    monkeypatch.setattr(verification_module.time, "monotonic", fake_monotonic)
    spec = _write_spec(
        tmp_path,
        _spec_payload(temporal=None, per_case_wall_budget_s=50.0, total_wall_budget_s=100.0),
    )
    report = run_study(spec, tmp_path / "study", runner=RefinementFakeCore())
    cases = report.axes[0].cases
    assert cases[0].wall_budget_s == 50.0
    assert cases[-1].status is CaseStatus.NOT_STARTED and cases[-1].wall_budget_s is None
    assert report.outcome is StudyOutcome.INCOMPLETE
    budgets = [case.wall_budget_s for case in cases if case.wall_budget_s is not None]
    assert all(budget <= 50.0 for budget in budgets)


def test_cli_verify_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`verify` exits 0 only for `passed`, 1 for unresolved, 3 for incomplete."""
    spec = _write_spec(tmp_path, _spec_payload(temporal=None))
    monkeypatch.setattr(
        native._core, "run_json", RefinementFakeCore(coefficient=0.1), raising=False
    )
    assert main(["verify", str(spec), "--out", str(tmp_path / "pass")]) == EXIT_OK
    monkeypatch.setattr(native._core, "run_json", RefinementFakeCore(coefficient=200.0))
    assert main(["verify", str(spec), "--out", str(tmp_path / "unresolved")]) == EXIT_FAILURE
    monkeypatch.setattr(native._core, "run_json", RefinementFakeCore(fail_on_call=1))
    assert main(["verify", str(spec), "--out", str(tmp_path / "incomplete")]) == EXIT_INCOMPLETE
    assert main(["verify", str(spec), "--out", str(tmp_path / "pass")]) == EXIT_FAILURE
    assert main(["validate", "--kind", "study", str(STUDY)]) == EXIT_OK


@pytest.mark.usefixtures("native_or_skip")
def test_native_small_study(tmp_path: Path) -> None:
    """A short real study completes through the compiled kernel and reports honestly."""
    base = tmp_path / "base.yaml"
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    payload["duration_s"] = 0.05
    payload["record_interval_s"] = 0.01
    base.write_text(yaml.safe_dump(payload), encoding="utf-8")
    spec = _write_spec(
        tmp_path,
        _spec_payload(
            base_request=str(base),
            spatial={"grid_spacings_m": [0.01, 0.005, 0.0025], "dt_s": 0.00005},
            temporal={"grid_spacing_m": 0.005, "dt_values_s": [0.0004, 0.0002, 0.0001]},
            observables=[
                {"name": "height_m", "near_zero_scale": 0.0005},
                {"name": "mechanical_energy_j", "near_zero_scale": 0.000001},
            ],
            per_case_wall_budget_s=120.0,
            total_wall_budget_s=600.0,
        ),
    )
    report = run_study(spec, tmp_path / "study")
    assert report.study_complete, [axis.reasons for axis in report.axes]
    assert report.outcome in {StudyOutcome.PASSED, StudyOutcome.UNRESOLVED}
    assert report.criterion_met == (report.outcome is StudyOutcome.PASSED)
    assert report.fidelity == "research_unvalidated"
    assert "absolute analytic-reference accuracy" in report.omitted_gates
    assert "No bound on absolute analytic-reference error is established." in report.claim
    assert report.provenance.native_sha256 is not None
    assert report.total_wall_time_s < 600.0
    assert not report.total_budget_exceeded
    for axis in report.axes:
        assert [case.status for case in axis.cases] == [CaseStatus.COMPLETED] * 3
        assert all(case.observables["limited_steps"] == 0.0 for case in axis.cases)
        assert all(case.observables["rejected_steps"] == 0.0 for case in axis.cases)
        assert axis.timestep_cap_binding is True, axis.reasons
        for verdict in axis.observables:
            assert all(value is not None and math.isfinite(value) for value in verdict.values)
            assert verdict.outcome in {ObservableOutcome.PASSED, ObservableOutcome.UNRESOLVED}
    temporal = report.axes[1]
    steps = [case.observables["step_count"] for case in temporal.cases]
    assert steps == [125.0, 250.0, 500.0]
    spatial = report.axes[0]
    assert [case.observables["particle_count"] for case in spatial.cases] == [64.0, 512.0, 4096.0]
    run = load_request(tmp_path / "study" / "cases" / "spatial" / "0002" / "request.json")
    assert run.dt_s == 0.00005 and run.research is not None
    assert run.research.grid_spacing_m == 0.0025


def test_overflowed_refinement_ratios_do_not_invent_an_order() -> None:
    """Even finite discretizations can overflow when their ratios are formed."""
    verdict = evaluate_observable(
        ObservableSpec(name="height_m", near_zero_scale=1e-6),
        [1e308, 1.0, 5e-324],
        [1.3, 1.1, 1.05],
        0.05,
    )
    assert verdict.observed_order is None
