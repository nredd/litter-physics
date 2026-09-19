"""Test validity boundaries of research-derived response tables.

References:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from litter_physics.response_tables import Axis, Evidence, Response, ResponseTable, TableError


def make_table(*, synthetic: bool = True) -> ResponseTable:
    """Construct an explicitly synthetic unit-test table.

    Parameters:
        synthetic: bool: Toggle the evidence flag to test acceptance policy, not real validation.
    Returns:
        ResponseTable: Two-dimensional affine interpolation fixture.
    """
    return ResponseTable(
        name="synthetic-spread",
        axes=(
            Axis(name="moisture", unit="kg/kg", points=(0.0, 1.0)),
            Axis(name="load", unit="N", points=(0.0, 2.0)),
        ),
        responses=(
            Response(
                name="rate", unit="1/s", lower_bound=0.0, upper_bound=4.0, values=(0, 2, 1, 3)
            ),
        ),
        evidence=Evidence(
            artifact_sha256="0" * 64,
            source_revision="0" * 40,
            species_residual=1e-8,
            spatial_change=0.02,
            temporal_change=0.01,
            held_out_error=0.1,
            repeatability_band=0.05,
            near_zero_checked=True,
            domain_boundaries_checked=True,
            synthetic=synthetic,
        ),
        uncertainty_note="Synthetic unit tests, NOT a household calibration.",
    )


def test_interpolation_and_explicit_synthetic_flag() -> None:
    """Check affine interpolation without promoting synthetic evidence.

    Returns:
        None: Assertions report interpolation or validity errors.
    """
    result = make_table().evaluate({"moisture": 0.5, "load": 1.0}, exploratory=True)
    assert result.values["rate"] == pytest.approx(1.5)
    assert not result.validated
    assert result.diagnostics
    assert len(result.table_sha256) == 64
    assert replace(result, validated=False) == result


def test_synthetic_evidence_is_rejected_by_default() -> None:
    """Reject production use of a merely synthetic table.

    Returns:
        None: Assertions check fail-closed behavior.
    """
    with pytest.raises(TableError, match="non-synthetic"):
        make_table().evaluate({"moisture": 0.5, "load": 1.0})


def test_non_synthetic_policy_and_hash() -> None:
    """Test evidence policy with mock evidence, not a physical measurement claim.

    Returns:
        None: Assertions check evidence handling and content addressing.
    """
    table = make_table(synthetic=False)
    result = table.evaluate({"moisture": 0.5, "load": 1.0})
    assert result.validated
    assert not result.diagnostics
    assert result.table_sha256 == table.digest()
    assert result.table_sha256 != make_table().digest()
    assert ResponseTable.model_validate_json(table.model_dump_json()) == table


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values(value: float) -> None:
    """Reject nonfinite runtime coordinates and stored knots.

    Parameters:
        value: float: Invalid coordinate.
    Returns:
        None: Assertions check numerical hygiene.
    """
    with pytest.raises(TableError, match="finite"):
        make_table().evaluate({"moisture": value, "load": 1.0}, exploratory=True)
    with pytest.raises(ValidationError):
        Axis(name="invalid", unit="m", points=(0.0, value))


@pytest.mark.parametrize("inputs", [{}, {"moisture": 0.2}, {"moisture": 0.2, "load": 0, "x": 0}])
def test_coordinate_names(inputs: dict[str, float]) -> None:
    """Reject mismatched named dimensions.

    Parameters:
        inputs: dict[str, float]: Incorrectly named coordinates.
    Returns:
        None: Assertions check strict dimension matching.
    """
    with pytest.raises(TableError, match="axes exactly"):
        make_table().evaluate(inputs, exploratory=True)


def test_extrapolation_requires_opt_in_and_is_bounded() -> None:
    """Reject unrestricted extrapolation even in exploratory mode.

    Returns:
        None: Assertions check both validity and physical clipping.
    """
    table = make_table(synthetic=False)
    with pytest.raises(TableError, match="outside"):
        table.evaluate({"moisture": -0.05, "load": 0.0})
    result = table.evaluate({"moisture": -0.05, "load": 0.0}, exploratory=True)
    assert result.values["rate"] == 0
    assert not result.validated
    assert len(result.diagnostics) == 2
    with pytest.raises(TableError, match="bounded exploration"):
        table.evaluate({"moisture": -0.2, "load": 0.0}, exploratory=True)


def test_evidence_requires_all_acceptance_checks() -> None:
    """Reject table promotion if boundary or near-zero tests are absent.

    Returns:
        None: Assertions check fail-closed acceptance.
    """
    data = make_table(synthetic=False).model_dump()
    data["evidence"]["domain_boundaries_checked"] = False
    table = ResponseTable.model_validate(data)
    with pytest.raises(TableError, match="validation evidence"):
        table.evaluate({"moisture": 0.5, "load": 1.0})
    data["evidence"]["spatial_change"] = 0.05
    with pytest.raises(ValidationError):
        ResponseTable.model_validate(data)
    data["evidence"]["spatial_change"] = 0.01
    data["evidence"]["held_out_error"] = 0.3
    with pytest.raises(ValidationError, match="acceptance threshold"):
        ResponseTable.model_validate(data)


def test_grid_shape_order_and_resource_validation() -> None:
    """Check knot ordering, dimensionality and deterministic sample ordering.

    Returns:
        None: Assertions report malformed-grid acceptance.
    """
    table = make_table()
    assert table.sample_inputs() == (
        {"moisture": 0.0, "load": 0.0},
        {"moisture": 0.0, "load": 2.0},
        {"moisture": 1.0, "load": 0.0},
        {"moisture": 1.0, "load": 2.0},
    )
    with pytest.raises(ValidationError, match="increasing"):
        Axis(name="x", unit="m", points=(1.0, 0.0))
    data = table.model_dump()
    data["responses"][0]["values"] = (0.0,)
    with pytest.raises(ValidationError, match="Cartesian"):
        ResponseTable.model_validate(data)
    data = table.model_dump()
    data["axes"][1]["name"] = "moisture"
    with pytest.raises(ValidationError, match="unique"):
        ResponseTable.model_validate(data)


def test_committed_schema() -> None:
    """Check generated schema drift and validate a synthetic serialized table.

    Returns:
        None: Assertions expose schema/model incompatibilities.
    """
    source_path = Path(__file__).resolve().parents[1] / "schemas/response-table.schema.json"
    schema = json.loads(source_path.read_text())
    assert schema == ResponseTable.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(make_table().model_dump(mode="json"))


def test_physical_bounds_and_unknown_fields() -> None:
    """Reject malformed physical ranges and undocumented schema extensions.

    Returns:
        None: Assertions report schema failures.
    """
    with pytest.raises(ValidationError, match="reversed"):
        Response(name="x", unit="m", lower_bound=1, upper_bound=0, values=(0,))
    with pytest.raises(ValidationError, match="outside physical"):
        Response(name="x", unit="m", lower_bound=0, upper_bound=1, values=(2,))
    data = make_table().model_dump() | {"unchecked": True}
    with pytest.raises(ValidationError, match="Extra inputs"):
        ResponseTable.model_validate(data)


@pytest.mark.parametrize("bound", [0.1, 1e-20, 1e20])
def test_constant_at_bound_remains_validated(bound: float) -> None:
    """Do not mistake floating-point interpolation roundoff for invalid physics.

    Parameters:
        bound: float: A constant response at both physical bounds.
    Returns:
        None: Assertions check roundoff-safe validation and cached preparation.
    """
    table = ResponseTable(
        name="synthetic-constant",
        axes=tuple(Axis(name=name, unit="1", points=(0, 0.5, 1)) for name in ("x", "y", "z")),
        responses=(
            Response(
                name="constant",
                unit="1",
                lower_bound=bound,
                upper_bound=bound,
                values=(bound,) * 27,
            ),
        ),
        evidence=make_table(synthetic=False).evidence,
        uncertainty_note="Mock evidence exercises policy, not actual physical validation.",
    )
    for i in range(1, 100):
        inputs = {"x": i / 100, "y": (i * 31 % 100) / 100, "z": (i * 17 % 100) / 100}
        result = table.evaluate(inputs)
        assert result.validated
        assert result.values["constant"] == bound
    assert table._interpolators is table._interpolators
    assert table.digest() == table.digest()


def test_zero_exploration_margin_and_boolean_rejection() -> None:
    """Reject every excursion at zero margin and prevent boolean coordinates.

    Returns:
        None: Assertions check strict input semantics.
    """
    table = ResponseTable.model_validate(
        make_table().model_dump() | {"exploration_margin_fraction": 0}
    )
    with pytest.raises(TableError, match="bounded exploration"):
        table.evaluate({"moisture": -1e-10, "load": 0}, exploratory=True)
    with pytest.raises(TableError, match="booleans"):
        table.evaluate({"moisture": True, "load": 0}, exploratory=True)
    with pytest.raises(ValidationError, match="span"):
        Axis(name="huge", unit="1", points=(-1e308, 1e308))


def test_resource_cap_before_cartesian_allocation() -> None:
    """Reject a large grid even when only a tiny response array was supplied.

    Returns:
        None: Assertions check the interpolation allocation guard.
    """
    data = make_table().model_dump()
    data["axes"] = [
        {"name": name, "unit": "1", "points": tuple(range(1001))} for name in ("x", "y")
    ]
    with pytest.raises(ValidationError, match="resource limit"):
        ResponseTable.model_validate(data)


def test_copied_table_revalidates_and_invalidates_cached_provenance() -> None:
    """Never evaluate an updated table through the original cached grid or digest.

    Returns:
        None: Assertions verify safe copies and rejection of invalid updates.
    """
    table = make_table(synthetic=False)
    coordinates = {"moisture": 0.5, "load": 1.0}
    original = table.evaluate(coordinates)
    updated_response = Response(
        name="rate", unit="1/s", lower_bound=0, upper_bound=4, values=(4, 4, 4, 4)
    )
    copied = table.model_copy(update={"responses": (updated_response,)})
    result = copied.evaluate(coordinates)
    assert result.values["rate"] == 4
    assert result.table_sha256 != original.table_sha256
    assert table.model_copy().digest() == original.table_sha256
    assert table.model_copy(deep=True).digest() == original.table_sha256
    broken = updated_response.model_copy(update={"values": (0,)})
    with pytest.raises(ValidationError, match="Cartesian"):
        table.model_copy(update={"responses": (broken,)})
    nonfinite = updated_response.model_copy(update={"values": (float("nan"),) * 4})
    with pytest.raises(ValidationError):
        table.model_copy(update={"responses": (nonfinite,)})
