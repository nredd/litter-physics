"""JSON Schema export for every validated document type.

Committed schemas under `schemas/` are regenerated from the models and checked in the
gate so they cannot drift from the code.

References:
    https://json-schema.org/draft/2020-12
    https://docs.pydantic.dev/latest/concepts/json_schema/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from litter_physics.behavior import CatProfile
from litter_physics.calibration import CalibrationBundle
from litter_physics.models import Checkpoint, SimulationOutput, SimulationRequest, export_schema
from litter_physics.observations import ObservationSet
from litter_physics.sweep import SweepSpec
from litter_physics.verification import StudyReport, StudySpec

LOGGER = logging.getLogger(__name__)

SCHEMA_FILES: dict[str, type[BaseModel]] = {
    "request.schema.json": SimulationRequest,
    "output.schema.json": SimulationOutput,
    "checkpoint.schema.json": Checkpoint,
    "observations.schema.json": ObservationSet,
    "calibration-bundle.schema.json": CalibrationBundle,
    "cat-profile.schema.json": CatProfile,
    "sweep-spec.schema.json": SweepSpec,
    "verification-study.schema.json": StudySpec,
    "verification-report.schema.json": StudyReport,
}


def render_schema(model: type[BaseModel]) -> str:
    """Render a model schema as stable, indented JSON.

    Parameters:
        model (type[BaseModel]): Model class.

    Returns:
        str: JSON text with a trailing newline.
    """
    return json.dumps(export_schema(model), indent=2, sort_keys=True) + "\n"


def write_schemas(directory: Path) -> list[Path]:
    """Write every schema file, overwriting stale copies.

    Parameters:
        directory (Path): Destination directory, created when missing.

    Returns:
        list[Path]: Written paths in `SCHEMA_FILES` order.
    """
    root = Path(str(directory)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_FILES.items():
        path = root / name
        path.write_text(render_schema(model), encoding="utf-8")
        written.append(path)
        LOGGER.info(f"wrote schema '{path}'")
    return written


def check_schemas(directory: Path) -> list[str]:
    """Compare committed schemas against the models.

    Parameters:
        directory (Path): Directory holding committed schemas.

    Returns:
        list[str]: Names of missing or stale files; empty when current.
    """
    root = Path(str(directory)).resolve()
    stale: list[str] = []
    for name, model in SCHEMA_FILES.items():
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != render_schema(model):
            stale.append(name)
    return stale
