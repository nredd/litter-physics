"""Request, output, and checkpoint validation.

References:
    docs/wire-contract.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from litter_physics.models import (
    Checkpoint,
    SimulationOutput,
    SimulationRequest,
    canonical_json,
    load_document,
    load_request,
    parse_output,
    request_to_wire,
    validate_payload,
)
from tests.conftest import EXAMPLES
from tests.fake_core import FakeCore


def payload(request: SimulationRequest) -> dict[str, Any]:
    """Dump a request to a mutable tree.

    Parameters:
        request (SimulationRequest): Validated request.

    Returns:
        dict[str, Any]: JSON-like copy.
    """
    return request.model_dump(mode="json")


def test_examples_load(household_request: SimulationRequest) -> None:
    """The committed examples validate and round-trip through the wire form."""
    assert household_request.mode == "household"
    wire = request_to_wire(household_request)
    assert validate_payload(SimulationRequest, json.loads(wire)) == household_request
    assert load_request(EXAMPLES / "research_slump.yaml").research is not None


def test_invalid_example_fails() -> None:
    """The deliberately invalid example is rejected with a readable message."""
    with pytest.raises(ValueError, match="earlier than its predecessor"):
        load_request(EXAMPLES / "invalid_event_order.yaml")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("unknown", 1), "extra"),
        (lambda p: p.__setitem__("dt_s", 2.0), "dt_s"),
        (lambda p: p.__setitem__("max_wall_time_s", 3601.0), "max_wall_time_s"),
        (lambda p: p.__setitem__("seed", -1), "seed"),
        (lambda p: p.__setitem__("seed", 2**64), "seed"),
        (lambda p: p.__setitem__("schema_version", 2), "schema_version"),
        (lambda p: p["boxes"].append(dict(p["boxes"][0])), "unique"),
        (lambda p: p.__setitem__("boxes", []), "at least one box"),
        (lambda p: p["events"][0].__setitem__("box_id", "nope"), "not a box"),
        (lambda p: p["events"][0].__setitem__("time_s", 5.0), "duration_s"),
        (lambda p: p["events"][0].__setitem__("position_m", [1.0, 0.0, 0.0]), "outside"),
        (lambda p: p["events"][0].__setitem__("direction", [0.0, 0.0, 0.0]), "nonzero"),
        (lambda p: p["events"][1].__setitem__("water_fraction", 1.5), "water_fraction"),
        (lambda p: p["boxes"][0].__setitem__("slot_width_m", 0.05), "slot_width_m"),
        (lambda p: p["boxes"][0].__setitem__("pellet_count", 10**6), "pellet_count"),
        (lambda p: p["materials"].__setitem__("restitution", 1.2), "restitution"),
        (lambda p: p["materials"].__setitem__("friction", "0.4"), "friction"),
        (lambda p: p["materials"].__setitem__("friction", True), "friction"),
        (lambda p: p.__setitem__("research", {"fixture": "slump"}), "research"),
    ],
)
def test_household_rejections(
    household_request: SimulationRequest, mutate: Any, match: str
) -> None:
    """Each contract rule rejects a targeted mutation."""
    tree = payload(household_request)
    mutate(tree)
    with pytest.raises(ValueError, match=match):
        validate_payload(SimulationRequest, tree)


def test_non_finite_rejected(household_request: SimulationRequest) -> None:
    """NaN never reaches the model."""
    tree = payload(household_request)
    tree["materials"]["friction"] = math.nan
    with pytest.raises(ValueError):
        validate_payload(SimulationRequest, tree)
    with pytest.raises(ValueError):
        canonical_json({"x": math.inf})


def test_research_rules(research_request: SimulationRequest) -> None:
    """Research mode needs `research`, forbids events, and bounds the grid."""
    tree = payload(research_request)
    tree["events"] = payload(load_request(EXAMPLES / "household_basic.yaml"))["events"][:1]
    tree["boxes"] = payload(load_request(EXAMPLES / "household_basic.yaml"))["boxes"][:1]
    with pytest.raises(ValueError, match="prohibits household"):
        validate_payload(SimulationRequest, tree)
    tree = payload(research_request)
    tree["research"] = None
    with pytest.raises(ValueError, match="requires `research`"):
        validate_payload(SimulationRequest, tree)
    tree = payload(research_request)
    for spacing in (1e-6, 5e-324):
        tree["research"]["grid_spacing_m"] = spacing
        with pytest.raises(ValueError, match="cells"):
            validate_payload(SimulationRequest, tree)
    tree = payload(research_request)
    tree["research"]["initial_size_m"] = [1.0, 0.02, 0.02]
    with pytest.raises(ValueError, match="exceeds"):
        validate_payload(SimulationRequest, tree)
    tree = payload(research_request)
    tree["research"]["fixture"] = "vortex"
    with pytest.raises(ValueError, match="fixture"):
        validate_payload(SimulationRequest, tree)


def test_equivalent_except_budget(household_request: SimulationRequest) -> None:
    """Budget differences are ignored for resume comparison; anything else is not."""
    other = household_request.with_wall_budget(5.0)
    assert household_request.equivalent_except_budget(other)
    changed = other.model_copy(update={"seed": 2})
    assert not household_request.equivalent_except_budget(changed)


def test_output_parsing(household_request: SimulationRequest) -> None:
    """Fake-core output validates; contradictions are rejected."""
    text = FakeCore()(request_to_wire(household_request), None)
    output = parse_output(text)
    assert output.status == "completed"
    assert output.checkpoint.request == household_request
    tree = output.model_dump(mode="json")
    tree["time_s"] = 0.5
    with pytest.raises(ValueError, match="disagrees"):
        validate_payload(SimulationOutput, tree)
    tree = output.model_dump(mode="json")
    tree["checkpoint"]["time_s"] = 0.5
    tree["time_s"] = 0.5
    with pytest.raises(ValueError, match="short of"):
        validate_payload(SimulationOutput, tree)
    tree = output.model_dump(mode="json")
    tree["frames"][0]["radii_m"].append(0.001)
    with pytest.raises(ValueError, match="unequal"):
        validate_payload(SimulationOutput, tree)
    tree = output.model_dump(mode="json")
    tree["metrics"][0]["compartment"] = "outflow"
    with pytest.raises(ValueError, match="not valid in"):
        validate_payload(SimulationOutput, tree)
    tree = output.model_dump(mode="json")
    tree["checkpoint"]["state"] = {"rng": float("nan")}
    with pytest.raises(ValueError):
        validate_payload(SimulationOutput, tree)
    with pytest.raises(ValueError, match="JSON"):
        parse_output("not json")


def test_checkpoint_rules(household_request: SimulationRequest) -> None:
    """Checkpoint header must agree with the embedded request."""
    text = FakeCore()(request_to_wire(household_request), None)
    checkpoint = parse_output(text).checkpoint
    tree = checkpoint.model_dump(mode="json")
    tree["mode"] = "research"
    with pytest.raises(ValueError, match="mode"):
        validate_payload(Checkpoint, tree)
    tree = checkpoint.model_dump(mode="json")
    tree["schema_version"] = 9
    with pytest.raises(ValueError, match="schema_version"):
        validate_payload(Checkpoint, tree)


def test_load_document_errors(tmp_path: Path) -> None:
    """Document loading reports missing files, bad suffixes, and non-mappings."""
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="suffix"):
        load_document(bad)
    listy = tmp_path / "list.yaml"
    listy.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_document(listy)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_document(broken)


@pytest.mark.parametrize(
    "name,text",
    [
        ("duplicate.yaml", "seed: 1\nseed: 2\n"),
        ("nested.json", '{"a": {"seed": 1, "seed": 2}}'),
        ("alias.yaml", "a: &a [1, 2]\nb: *a\n"),
        ("keys.yaml", "1: wrong\n"),
    ],
)
def test_ambiguous_configuration_is_rejected(tmp_path: Path, name: str, text: str) -> None:
    """Reject duplicate keys and aliases before expanding or validating a scenario."""
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ValueError):
        load_request(path)
