"""Build manuscript tables/plots from frozen native evidence, then compile LaTeX.

References:
    https://tectonic-typesetting.github.io/en-US/
    https://github.com/nredd/litter-physics/blob/9a66a55/docs/verification.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import math
import shutil
import subprocess
from pathlib import Path

from litter_physics.verification import StudyReport

LOGGER = logging.getLogger(__name__)
REPORT_SHA256 = "aa6030d7ea1af0919343d3c4732855bce3eecc3c69310a7babc116dc0015bce8"
BASELINE = "9a66a559ed478629d514756662a8fd98a8f52f50"
ROOT = Path(__file__).resolve().parent
OBSERVABLES = ("slump_m", "spread_x_m", "plastic_dissipation_j")


def load_evidence(root: Path) -> StudyReport:
    """Read the exact archived report; reject substitution before parsing.

    Parameters:
        root (Path): Manuscript directory containing `data/`.
    Returns:
        StudyReport: Schema-validated, source-pinned native results.
    Raises:
        ValueError: If bytes, provenance, or study status do not match the baseline.
        FileNotFoundError: If the frozen evidence is absent.
    """
    path = root / "data/refinement-report.json"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != REPORT_SHA256:
        raise ValueError(f"Frozen report checksum mismatch: '{path}'")
    report = StudyReport.model_validate_json(content)
    if (
        report.provenance.git_commit != BASELINE
        or report.provenance.git_dirty is not False
        or not report.study_complete
        or report.criterion_met
        or report.fidelity != "research_unvalidated"
        or [axis.axis.value for axis in report.axes] != ["spatial", "temporal"]
    ):
        raise ValueError("Frozen report does not describe the documented unresolved baseline")
    return report


def worked_values() -> dict[str, float]:
    """Evaluate the manuscript's independent dimensional examples in SI units.

    Returns:
        dict[str, float]: Dry-pellet and undeformed slump-block calculations.
    """
    radius, length, density, stiffness, gravity = 0.003, 0.012, 1100.0, 1000.0, 9.80665
    mass = density * math.pi * radius**2 * length
    step = 0.1 * math.sqrt(mass / stiffness)
    return {
        "PelletMassGrams": mass * 1000.0,
        "PelletAxialInertia": 0.5 * mass * radius**2,
        "PelletTransverseInertia": mass * (3.0 * radius**2 + length**2) / 12.0,
        "PelletStepMicroseconds": step * 1e6,
        "PelletWallOverlapMicrons": mass * gravity / stiffness * 1e6,
        "PelletSubsteps": float(math.ceil(0.001 / step)),
        "BlockMassGrams": 1000.0 * 0.02**3 * 1000.0,
        "BlockPotentialMillijoules": 1000.0 * 0.02**3 * 9.81 * 0.01 * 1000.0,
    }


def constitutive_values() -> dict[str, float]:
    """Independently evaluate the fixed HB return and compressible-water examples.

    Returns:
        dict[str, float]: SI values, using bisection rather than the native Newton solve.
    """
    shear = 1000.0 / (2 * 1.2)
    yield_stress = math.sqrt(3) * 30
    consistency = 3**0.8 * 10
    trial, dt = 80.0, 0.0014
    lower, upper = yield_stress, trial
    for _ in range(80):
        stress = (lower + upper) / 2
        residual = (
            stress - trial + 3 * shear * dt * ((stress - yield_stress) / consistency) ** (1 / 0.6)
        )
        if residual > 0:
            upper = stress
        else:
            lower = stress
    stress = (lower + upper) / 2
    increment = (trial - stress) / (3 * shear)
    bulk = 1e5 / (3 * (1 - 2 * 0.2))
    ratio = math.exp(-1000 * 9.81 * 0.01 / bulk)
    sound_speed = math.sqrt(bulk / 1000)
    return {
        "HBReturnedStress": stress,
        "HBPlasticIncrement": increment,
        "HBDissipation": stress * increment,
        "HBElasticDrop": (trial**2 - stress**2) / (6 * shear),
        "WaterBulkModulus": bulk,
        "WaterSoundSpeed": sound_speed,
        "WaterBottomPressure": bulk * (1 / ratio - 1),
        "WaterBottomRatio": ratio,
        "WaterAcousticStep": 0.3 * 0.002 / sound_speed,
    }


def generated_contents(report: StudyReport) -> dict[str, str]:
    """Render deterministic CSV plot data and LaTeX tables, without running physics.

    Parameters:
        report (StudyReport): The validated archived baseline.
    Returns:
        dict[str, str]: Relative filenames mapped to their exact generated text.
    """
    outputs: dict[str, str] = {}
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Axis & $h$ (mm) & $\Delta t$ (ms) & Points & Steps & Slump (mm) & Spread (mm) \\",
        r"\midrule",
    ]
    verdicts = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Final-two change & Slump & Spread & Plastic loss \\",
        r"\midrule",
    ]
    for axis in report.axes:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("h_mm", "dt_ms", "slump_mm", "spread_mm", "plastic_uj", "residual_uj"))
        for case in axis.cases:
            values = case.observables
            writer.writerow(
                (
                    case.grid_spacing_m * 1000.0,
                    case.dt_s * 1000.0,
                    values["slump_m"] * 1000.0,
                    values["spread_x_m"] * 1000.0,
                    values["plastic_dissipation_j"] * 1e6,
                    values["energy_residual_j"] * 1e6,
                )
            )
            rows.append(
                f"{axis.axis.value.capitalize()} & {case.grid_spacing_m * 1000:g} & "
                f"{case.dt_s * 1000:g} & {values['particle_count']:.0f} & "
                f"{values['step_count']:.0f} & {values['slump_m'] * 1000:.4f} & "
                f"{values['spread_x_m'] * 1000:.4f} " + r"\\"
            )
        verdict_map = {item.name: item for item in axis.observables}
        percentages = [
            f"{100 * change:.2f}\\%" if change is not None else "missing"
            for change in (verdict_map[name].final_relative_change for name in OBSERVABLES)
        ]
        verdicts.append(" & ".join([axis.axis.value.capitalize(), *percentages]) + r" \\")
        outputs[f"{axis.axis.value}.csv"] = stream.getvalue()
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    verdicts.extend([r"\bottomrule", r"\end{tabular}", ""])
    outputs["cases.tex"] = "\n".join(rows)
    outputs["verdicts.tex"] = "\n".join(verdicts)
    outputs["calculations.tex"] = "".join(
        f"\\newcommand{{\\{name}}}{{{value:.6g}}}\n"
        for name, value in (worked_values() | constitutive_values()).items()
    )
    return outputs


def generate(root: Path, *, check: bool) -> None:
    """Write derived text or fail if checked-in figures/tables are stale.

    Parameters:
        root (Path): Manuscript directory.
        check (bool): Verify without writing when true.
    Returns:
        None.
    Raises:
        ValueError: If evidence is invalid or a generated file is missing/stale.
    """
    outputs = generated_contents(load_evidence(root))
    destination = root / "generated"
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = destination / name
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Stale manuscript data: '{path}'; run `make manuscript`")
        else:
            path.write_text(content, encoding="utf-8")


def compile_pdf(root: Path) -> None:
    """Compile LaTeX with Tectonic and reject unresolved references or overflow.

    Parameters:
        root (Path): Directory containing `simulation.tex`.
    Returns:
        None.
    Raises:
        FileNotFoundError: If Tectonic is not installed.
        ValueError: If the compiler log reports a typesetting defect.
        subprocess.CalledProcessError: If compilation fails.
    """
    binary = shutil.which("tectonic")
    if binary is None:
        raise FileNotFoundError("Install `tectonic` (macOS: `brew install tectonic`)")
    output = subprocess.check_output(
        [str(Path(binary).resolve()), "--keep-logs", "simulation.tex"],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    )
    LOGGER.info(output.rstrip())
    log = (root / "simulation.log").read_text(encoding="utf-8")
    problems = ("Overfull", "undefined references", "undefined citations", "Missing character")
    if any(problem in log for problem in problems):
        raise ValueError(f"Typesetting defects remain in '{root / 'simulation.log'}'")
    if not (root / "simulation.pdf").is_file():
        raise FileNotFoundError(f"Compiler did not produce '{root / 'simulation.pdf'}'")


def main() -> int:
    """Run the checked-data or complete PDF build command.

    Returns:
        int: Zero on success, one on a data/build failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check generated data without LaTeX")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(ROOT, check=args.check)
        if not args.check:
            compile_pdf(ROOT)
    except subprocess.CalledProcessError as error:
        LOGGER.error(f"Manuscript compilation failed:\n{error.output}")
        return 1
    except (ValueError, OSError) as error:
        LOGGER.error(f"Manuscript build failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
