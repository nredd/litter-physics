"""Check reproducible manuscript evidence, calculations, and build failure paths.

References:
    https://github.com/nredd/litter-physics/blob/9a66a55/docs/verification.md
"""

from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from docs.formal import build

from litter_physics.verification import ObservableSpec, evaluate_observable


def test_archived_evidence_and_generated_artifacts_are_current() -> None:
    """Keep the document's plots tied to unchanged native-run evidence."""
    build.generate(build.ROOT, check=True)
    report = build.load_evidence(build.ROOT)
    assert report.outcome.value == "unresolved"
    assert report.provenance.native_sha256
    for axis in report.axes:
        for recorded in axis.observables:
            computed = evaluate_observable(
                ObservableSpec(name=recorded.name, near_zero_scale=recorded.near_zero_scale),
                axis.values,
                [case.observables[recorded.name] for case in axis.cases],
                report.max_relative_change,
            )
            assert computed == recorded
        for case in axis.cases:
            assert case.observables["limited_steps"] == 0
            assert case.observables["rejected_steps"] == 0
            assert case.observables["mass_residual_kg"] == 0


def test_plot_units_are_converted_from_native_results() -> None:
    """CSV values must use the plotted mm, ms, and microjoule units."""
    report = build.load_evidence(build.ROOT)
    outputs = build.generated_contents(report)
    for axis in report.axes:
        rows = list(csv.DictReader(io.StringIO(outputs[f"{axis.axis.value}.csv"])))
        assert len(rows) == 3
        for row, case in zip(rows, axis.cases, strict=True):
            assert float(row["h_mm"]) == pytest.approx(case.grid_spacing_m * 1000)
            assert float(row["dt_ms"]) == pytest.approx(case.dt_s * 1000)
            assert float(row["slump_mm"]) == pytest.approx(case.observables["slump_m"] * 1000)
            assert float(row["plastic_uj"]) == pytest.approx(
                case.observables["plastic_dissipation_j"] * 1e6
            )


def test_dimensional_examples() -> None:
    """Independently check cylinder properties and continuum potential energy."""
    values = build.worked_values()
    mass = 1100 * math.pi * 0.003**2 * 0.012
    assert values["PelletMassGrams"] == pytest.approx(0.3732212072464674)
    assert values["PelletAxialInertia"] == pytest.approx(mass * 0.003**2 / 2, abs=1e-18)
    assert values["PelletTransverseInertia"] == pytest.approx(
        mass * (3 * 0.003**2 + 0.012**2) / 12, abs=1e-18
    )
    assert values["PelletSubsteps"] == 17
    assert values["BlockMassGrams"] == pytest.approx(8)
    assert values["BlockPotentialMillijoules"] == pytest.approx(0.7848)


def test_constitutive_examples_and_return_energy_gap() -> None:
    """Check the independently bisected root, exact return gap, and water EOS."""
    values = build.constitutive_values()
    shear = 1000 / 2.4
    stress = values["HBReturnedStress"]
    residual = (
        stress
        - 80
        + 3 * shear * 0.0014 * ((stress - math.sqrt(3) * 30) / (3**0.8 * 10)) ** (1 / 0.6)
    )
    assert abs(residual) < 8e-11
    assert stress == pytest.approx(78.0059568837, abs=1e-9)
    assert values["HBElasticDrop"] - values["HBDissipation"] == pytest.approx(
        1.5 * shear * values["HBPlasticIncrement"] ** 2, abs=1e-14
    )
    assert values["WaterBottomPressure"] == pytest.approx(98.186663, rel=0, abs=5e-7)
    assert values["WaterBottomRatio"] == pytest.approx(
        1 / (1 + values["WaterBottomPressure"] / values["WaterBulkModulus"])
    )


def test_generate_check_refuses_stale_or_tampered_data(tmp_path: Path) -> None:
    """Missing/stale generated files and modified source evidence fail closed."""
    shutil.copytree(build.ROOT / "data", tmp_path / "data")
    with pytest.raises(ValueError, match="Stale manuscript data"):
        build.generate(tmp_path, check=True)
    build.generate(tmp_path, check=False)
    build.generate(tmp_path, check=True)
    (tmp_path / "generated/spatial.csv").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale manuscript data"):
        build.generate(tmp_path, check=True)
    report = tmp_path / "data/refinement-report.json"
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build.load_evidence(tmp_path)


def test_missing_compiler_has_actionable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A data-only test suite need not install the LaTeX compiler."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="brew install tectonic"):
        build.compile_pdf(build.ROOT)


@pytest.mark.parametrize("warning", ["Overfull", "undefined references", "Missing character"])
def test_compile_rejects_typesetting_defects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warning: str
) -> None:
    """A successful compiler exit alone must not hide defective typesetting."""
    monkeypatch.setattr(shutil, "which", lambda _: "/fake/tectonic")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "compiler output")
    (tmp_path / "simulation.log").write_text(warning, encoding="utf-8")
    with pytest.raises(ValueError, match="Typesetting defects"):
        build.compile_pdf(tmp_path)
