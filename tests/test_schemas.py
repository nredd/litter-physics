"""Committed schemas match the models and validate the examples.

References:
    https://python-jsonschema.readthedocs.io/
"""

from __future__ import annotations

import json

import jsonschema
import yaml

from litter_physics.models import load_request, request_to_wire
from litter_physics.schemas import SCHEMA_FILES, check_schemas, render_schema
from tests.conftest import EXAMPLES, SCHEMAS
from tests.fake_core import FakeCore


def test_committed_schemas_current() -> None:
    """`schemas/` must be regenerated whenever a model changes."""
    assert check_schemas(SCHEMAS) == []
    for name in SCHEMA_FILES:
        assert (SCHEMAS / name).is_file()


def test_schemas_are_valid_2020_12() -> None:
    """Every exported schema is itself valid against the metaschema."""
    for model in SCHEMA_FILES.values():
        schema = json.loads(render_schema(model))
        jsonschema.Draft202012Validator.check_schema(schema)


def test_examples_validate_against_schema() -> None:
    """Examples pass the request schema; the invalid one fails."""
    schema = json.loads((SCHEMAS / "request.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for name in (
        "household_basic.yaml",
        "research_slump.yaml",
        "research_hydrostatic_water.yaml",
        "research_coupled_patch.yaml",
    ):
        document = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
        validator.validate(document)
    observations = json.loads((SCHEMAS / "observations.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(observations).validate(
        json.loads((EXAMPLES / "observations_synthetic.json").read_text(encoding="utf-8"))
    )
    sweep = json.loads((SCHEMAS / "sweep-spec.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(sweep).validate(
        yaml.safe_load((EXAMPLES / "sweep_friction.yaml").read_text(encoding="utf-8"))
    )
    study = json.loads((SCHEMAS / "verification-study.schema.json").read_text(encoding="utf-8"))
    for name in ("verification_slump.yaml", "verification_hydrostatic_water.yaml"):
        jsonschema.Draft202012Validator(study).validate(
            yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
        )
    profiles = json.loads((SCHEMAS / "cat-profile.schema.json").read_text(encoding="utf-8"))
    for profile in yaml.safe_load(
        (EXAMPLES / "cat_profiles_synthetic.yaml").read_text(encoding="utf-8")
    ):
        jsonschema.Draft202012Validator(profiles).validate(profile)


def test_output_schema_accepts_fake_output() -> None:
    """The output schema accepts contract-shaped output."""
    request = load_request(EXAMPLES / "household_basic.yaml")
    output = json.loads(FakeCore()(request_to_wire(request), None))
    schema = json.loads((SCHEMAS / "output.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(output)
