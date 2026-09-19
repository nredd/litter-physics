"""Validate and interpolate research-derived reduced-model responses.

References:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    https://docs.pydantic.dev/latest/concepts/validators/
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import isfinite, prod, ulp
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.interpolate import RegularGridInterpolator


class TableError(ValueError):
    """Reject invalid or scientifically unsupported reduced-model responses."""


class Axis(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    name: str = Field(min_length=1, description="Unique input parameter name.")
    unit: str = Field(min_length=1, description="Explicit physical unit, e.g. kg or s.")
    points: tuple[float, ...] = Field(min_length=2, description="Strictly increasing knots.")

    @model_validator(mode="after")
    def check_order(self) -> Axis:
        """Check interpolation knots.

        Returns:
            Axis: This validated axis.
        Raises:
            ValueError: Knots are not strictly increasing.
        """
        if any(right <= left for left, right in zip(self.points, self.points[1:], strict=False)):
            raise ValueError(f"Axis '{self.name}' requires strictly increasing `points`")
        if not isfinite(self.points[-1] - self.points[0]):
            raise ValueError(f"Axis '{self.name}' has an unrepresentable span")
        return self


class Response(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    name: str = Field(min_length=1, description="Named reduced-model response.")
    unit: str = Field(min_length=1, description="Physical unit of the response.")
    lower_bound: float = Field(description="Physical lower bound, including exploration.")
    upper_bound: float = Field(description="Physical upper bound, including exploration.")
    values: tuple[float, ...] = Field(description="C-order values over the Cartesian axis grid.")

    @model_validator(mode="after")
    def check_bounds(self) -> Response:
        """Check physical response bounds and all supplied samples.

        Returns:
            Response: This validated response.
        Raises:
            ValueError: Bounds or samples are inconsistent.
        """
        if self.upper_bound < self.lower_bound:
            raise ValueError(f"Response '{self.name}' has reversed physical bounds")
        if any(not self.lower_bound <= value <= self.upper_bound for value in self.values):
            raise ValueError(f"Response '{self.name}' contains samples outside physical bounds")
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    artifact_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Evidence artifact digest."
    )
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$", description="Source commit tested.")
    species_residual: float = Field(
        ge=0, le=1e-6, description="Maximum relative species residual."
    )
    spatial_change: float = Field(ge=0, lt=0.05, description="Final spatial refinement change.")
    temporal_change: float = Field(ge=0, lt=0.05, description="Final time refinement change.")
    held_out_error: float = Field(ge=0, description="Held-out surrogate observable error.")
    repeatability_band: float = Field(ge=0, le=1, description="Measured relative repeatability.")
    near_zero_checked: bool = Field(description="Near-zero cases used instrument uncertainty.")
    domain_boundaries_checked: bool = Field(description="Domain size/boundary studies passed.")
    synthetic: bool = Field(
        description="True for synthetic-only evidence; cannot promote a table."
    )

    @model_validator(mode="after")
    def check_accuracy(self) -> Evidence:
        """Check the agreed acceptance thresholds without weakening them.

        Returns:
            Evidence: This validated record.
        Raises:
            ValueError: Held-out errors exceed the acceptance threshold.
        """
        if self.held_out_error > max(0.2, self.repeatability_band):
            raise ValueError("`held_out_error` exceeds the measured acceptance threshold")
        return self


@dataclass(frozen=True)
class Evaluation:
    """Return interpolated values with non-optional validity information."""

    values: dict[str, float]
    validated: bool
    diagnostics: tuple[str, ...]
    table_sha256: str


class ResponseTable(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    schema_version: int = Field(default=1, ge=1, le=1, description="Table format version.")
    name: str = Field(min_length=1, description="Stable response-table identity.")
    axes: tuple[Axis, ...] = Field(min_length=1, max_length=6, description="Ordered input axes.")
    responses: tuple[Response, ...] = Field(min_length=1, description="Output response grids.")
    evidence: Evidence | None = Field(default=None, description="Independent validation evidence.")
    uncertainty_note: str = Field(min_length=1, description="Parameter and model uncertainty.")
    exploration_margin_fraction: float = Field(
        default=0.1, ge=0, le=0.1, description="Maximum explicit extension beyond each axis range."
    )

    @model_validator(mode="after")
    def check_grid(self) -> ResponseTable:
        """Check Cartesian sample counts and uniquely named axes/responses.

        Returns:
            ResponseTable: This validated table.
        Raises:
            ValueError: Names, dimensions or allocation limits are invalid.
        """
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("`axes` must have unique names")
        if len({response.name for response in self.responses}) != len(self.responses):
            raise ValueError("`responses` must have unique names")
        count = prod(len(axis.points) for axis in self.axes)
        if count * len(self.responses) > 1_000_000:
            raise ValueError("Response table exceeds the 1,000,000-value resource limit")
        if any(len(response.values) != count for response in self.responses):
            raise ValueError("Every response requires one value per Cartesian grid point")
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Revalidate a derived table without copying stale hashes or interpolation grids.

        Parameters:
            update: Mapping[str, Any] | None: Field overrides to validate.
            deep: bool: Ignored; all serialized fields are rebuilt and validated.
        Returns:
            Self: A validated copy with empty runtime caches.
        Raises:
            ValueError: Updated fields violate the table contract.
        """
        return type(self).model_validate(self.model_dump() | dict(update or {}))

    @cached_property
    def _content_digest(self) -> str:
        """Cache the digest of this immutable table.

        Returns:
            str: SHA-256 of canonical JSON, excluding cached runtime objects.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @cached_property
    def _interpolators(self) -> tuple[RegularGridInterpolator, ...]:
        """Build immutable interpolation grids once, not once per cell update.

        Returns:
            tuple[RegularGridInterpolator, ...]: Interpolators in response order.
        """
        shape = tuple(len(axis.points) for axis in self.axes)
        knots = tuple(np.asarray(axis.points, dtype=np.float64) for axis in self.axes)
        return tuple(
            RegularGridInterpolator(
                knots,
                np.asarray(response.values, dtype=np.float64).reshape(shape),
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            for response in self.responses
        )

    def digest(self) -> str:
        """Hash the complete table, including units and scientific evidence.

        Returns:
            str: Cached SHA-256 digest of canonical JSON.
        """
        return self._content_digest

    def sample_inputs(self) -> tuple[dict[str, float], ...]:
        """Return the deterministic Cartesian sample order.

        Returns:
            tuple[dict[str, float], ...]: Input points in C order.
        """
        names = tuple(axis.name for axis in self.axes)
        return tuple(
            dict(zip(names, values, strict=True))
            for values in product(*(a.points for a in self.axes))
        )

    def evaluate(self, inputs: dict[str, float], *, exploratory: bool = False) -> Evaluation:
        """Evaluate a table without hiding extrapolation or missing evidence.

        Parameters:
            inputs: dict[str, float]: Named inputs expressed in the axes' declared units.
            exploratory: bool: Permit bounded extrapolation and unvalidated evidence explicitly.
        Returns:
            Evaluation: Values, evidence status, diagnostic reasons and source digest.
        Raises:
            TableError: Inputs, evidence or exploration bounds are unacceptable.
        """
        if set(inputs) != {axis.name for axis in self.axes}:
            raise TableError("`inputs` must match the response table's axes exactly")
        if any(isinstance(value, bool) or not isfinite(value) for value in inputs.values()):
            raise TableError("`inputs` must be finite numbers, not booleans")
        diagnostics: list[str] = []
        evidence = self.evidence
        if (
            evidence is None
            or evidence.synthetic
            or not evidence.near_zero_checked
            or not evidence.domain_boundaries_checked
        ):
            diagnostics.append("Missing non-synthetic independent validation evidence")
        for axis in self.axes:
            value = inputs[axis.name]
            lower, upper = axis.points[0], axis.points[-1]
            if lower <= value <= upper:
                continue
            margin = self.exploration_margin_fraction * (upper - lower)
            if not lower - margin <= value <= upper + margin:
                raise TableError(f"Axis '{axis.name}' exceeds even the bounded exploration domain")
            diagnostics.append(f"Axis '{axis.name}' is outside its calibrated domain")
        if diagnostics and not exploratory:
            raise TableError("; ".join(diagnostics))
        coordinates = np.asarray([[inputs[axis.name] for axis in self.axes]], dtype=np.float64)
        outputs: dict[str, float] = {}
        for response, interpolator in zip(self.responses, self._interpolators, strict=True):
            value = float(interpolator(coordinates)[0])
            if not isfinite(value):
                raise TableError(f"Interpolation of response '{response.name}' is nonfinite")
            bounded = min(response.upper_bound, max(response.lower_bound, value))
            rounding_tolerance = 64 * max(ulp(value), ulp(bounded))
            if abs(bounded - value) > rounding_tolerance:
                diagnostics.append(
                    f"Response '{response.name}' was clipped to its physical bounds"
                )
            outputs[response.name] = bounded
        if diagnostics and not exploratory:
            raise TableError("; ".join(diagnostics))
        return Evaluation(outputs, not diagnostics, tuple(diagnostics), self.digest())
