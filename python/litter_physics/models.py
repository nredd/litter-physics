"""Validated wire-contract models for requests, outputs, and checkpoints.

Every model is strict: unknown fields are rejected, numeric fields must be finite,
and the structural rules of `docs/wire-contract.md` are enforced before any native
call. Rust re-validates independently; Python validation exists so that bad input
fails fast with a readable message and never reaches expensive work.

References:
    docs/wire-contract.md
    https://docs.pydantic.dev/latest/concepts/strict_mode/
    https://json-schema.org/draft/2020-12
"""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = 1
PROTOCOL_VERSION = 1
MAX_WALL_TIME_S = 3600.0
MAX_PELLETS_PER_BOX = 200_000
MAX_EVENTS = 100_000
MAX_BOXES = 16
MAX_RESEARCH_CELLS = 50_000_000
MAX_U64 = 2**64 - 1

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Vec3 = Annotated[list[Finite], Field(min_length=3, max_length=3)]
PositiveVec3 = Annotated[list[Positive], Field(min_length=3, max_length=3)]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")]


class StrictModel(BaseModel):
    """Base for every wire model: frozen, strict, unknown fields rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Mode(StrEnum):
    """Simulation mode selecting the numerical model family."""

    HOUSEHOLD = "household"
    RESEARCH = "research"


class EventKind(StrEnum):
    """Household event kinds accepted by the wire contract."""

    URINATE = "urinate"
    DEFECATE = "defecate"
    DIG = "dig"
    COVER = "cover"
    STIR = "stir"
    SCOOP = "scoop"
    EMPTY_DRAWER = "empty_drawer"
    REFILL = "refill"
    CLEAN = "clean"


MOTION_EVENT_KINDS = frozenset({EventKind.DIG, EventKind.COVER, EventKind.STIR, EventKind.SCOOP})
DEPOSIT_EVENT_KINDS = frozenset({EventKind.URINATE, EventKind.DEFECATE, EventKind.REFILL})


class Fixture(StrEnum):
    """Research fixture names."""

    SLUMP = "slump"
    HYDROSTATIC = "hydrostatic"
    DAM_BREAK = "dam_break"
    COUPLED_PATCH = "coupled_patch"


class Material(StrEnum):
    """Research continuum material names."""

    PASTE = "paste"
    WATER = "water"
    FINES = "fines"


class Status(StrEnum):
    """Native completion status."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Fidelity(StrEnum):
    """Honesty label carried by every native output."""

    PRELIMINARY_HOUSEHOLD = "preliminary_household"
    RESEARCH_UNVALIDATED = "research_unvalidated"


class Compartment(StrEnum):
    """Conservation ledger compartments."""

    BED = "bed"
    DRAWER = "drawer"
    FLOOR = "floor"
    REMOVED = "removed"
    EVAPORATED = "evaporated"
    DOMAIN = "domain"
    OUTFLOW = "outflow"


HOUSEHOLD_COMPARTMENTS = frozenset(
    {
        Compartment.BED,
        Compartment.DRAWER,
        Compartment.FLOOR,
        Compartment.REMOVED,
        Compartment.EVAPORATED,
    }
)
RESEARCH_COMPARTMENTS = HOUSEHOLD_COMPARTMENTS | {Compartment.DOMAIN, Compartment.OUTFLOW}


class BoxCfg(StrictModel):
    """One sifting box with its pellet population."""

    id: Identifier = Field(description="Unique box identifier referenced by events.")
    size_m: PositiveVec3 = Field(description="Inner box dimensions (x, y, z) in metres.")
    origin_m: Vec3 = Field(description="World position of the box minimum corner in metres.")
    slot_width_m: Positive = Field(description="Sifting slot width in metres.")
    slot_length_m: Positive = Field(description="Sifting slot length in metres.")
    slot_pitch_m: Positive = Field(description="Centre-to-centre slot spacing in metres.")
    drawer_depth_m: Positive = Field(description="Drawer cavity depth below the plate in metres.")
    pellet_count: int = Field(ge=0, le=MAX_PELLETS_PER_BOX, description="Intact pellets.")
    pellet_radius_m: Positive = Field(description="Cylindrical pellet radius in metres.")
    pellet_length_m: Positive = Field(description="Cylindrical pellet length in metres.")
    pellet_density_kg_m3: Positive = Field(description="Pellet bulk density in kg/m^3.")

    @model_validator(mode="after")
    def _geometry_fits(self) -> Self:
        """Reject slots and pellets that cannot physically fit the box.

        Returns:
            Self: The validated box.

        Raises:
            ValueError: When a slot or pellet exceeds the box interior.
        """
        size_x, size_y, size_z = self.size_m
        if self.slot_width_m >= self.slot_pitch_m:
            raise ValueError(
                f"`slot_width_m` '{self.slot_width_m}' must be less than `slot_pitch_m` "
                f"'{self.slot_pitch_m}' for box '{self.id}'"
            )
        if self.slot_length_m > max(size_x, size_y):
            raise ValueError(f"`slot_length_m` exceeds the box footprint for box '{self.id}'")
        if self.slot_pitch_m > max(size_x, size_y):
            raise ValueError(f"`slot_pitch_m` exceeds the box footprint for box '{self.id}'")
        diameter = 2.0 * self.pellet_radius_m
        longest = max(diameter, self.pellet_length_m)
        if diameter >= min(size_x, size_y, size_z) or longest > max(size_x, size_y, size_z):
            raise ValueError(f"pellet dimensions do not fit inside box '{self.id}'")
        pellet_volume = math.pi * self.pellet_radius_m**2 * self.pellet_length_m
        if self.pellet_count * pellet_volume > size_x * size_y * size_z:
            raise ValueError(
                f"`pellet_count` '{self.pellet_count}' exceeds the solid volume of box '{self.id}'"
            )
        return self

    @property
    def pellet_mass_kg(self) -> float:
        """Mass of a single intact pellet.

        Returns:
            float: Cylinder volume times density in kilograms.
        """
        return math.pi * self.pellet_radius_m**2 * self.pellet_length_m * self.pellet_density_kg_m3


class MaterialsCfg(StrictModel):
    """Household reduced-model material parameters."""

    friction: NonNegative = Field(description="Coulomb friction coefficient.")
    restitution: UnitInterval = Field(description="Normal restitution coefficient.")
    normal_stiffness_n_m: Positive = Field(description="Contact normal stiffness in N/m.")
    water_capacity_ratio: Positive = Field(
        description="Water mass per dry wood mass at saturation."
    )
    uptake_rate_s: NonNegative = Field(description="First-order water uptake rate in 1/s.")
    breakdown_rate_s: NonNegative = Field(description="Moisture-driven breakup rate in 1/s.")
    evaporation_rate_s: NonNegative = Field(description="Free-water evaporation rate in 1/s.")


class EventCfg(StrictModel):
    """One scheduled household event, positioned in box-local coordinates."""

    time_s: NonNegative = Field(description="Absolute simulation time of the event in seconds.")
    kind: EventKind = Field(description="Event kind.")
    box_id: Identifier = Field(description="Target box identifier.")
    position_m: Vec3 = Field(description="Box-local event position in metres.")
    direction: Vec3 = Field(description="Motion or deposition direction; nonzero for motion.")
    duration_s: NonNegative = Field(description="Event duration in seconds; zero for passive.")
    amount_kg: NonNegative = Field(description="Deposited or removed mass in kilograms.")
    water_fraction: UnitInterval = Field(description="Mass fraction that is mobile water.")
    radius_m: Positive = Field(description="Radius of influence in metres.")

    @model_validator(mode="after")
    def _direction_rules(self) -> Self:
        """Require nonzero directions for motion events.

        Returns:
            Self: The validated event.

        Raises:
            ValueError: When a motion event has a zero direction vector.
        """
        norm = math.sqrt(sum(component * component for component in self.direction))
        if self.kind in MOTION_EVENT_KINDS and norm == 0.0:
            raise ValueError(f"`direction` must be nonzero for motion event kind '{self.kind}'")
        return self


class ResearchCfg(StrictModel):
    """Research fixture configuration for the MPM solver."""

    fixture: Fixture = Field(description="Verification fixture name.")
    grid_spacing_m: Positive = Field(description="Background grid spacing in metres.")
    domain_m: PositiveVec3 = Field(description="Domain extent in metres.")
    material: Material = Field(description="Continuum material model.")
    density_kg_m3: Positive = Field(description="Reference density in kg/m^3.")
    young_modulus_pa: Positive = Field(description="Elastic modulus in pascals.")
    poisson_ratio: Annotated[float, Field(ge=0, lt=0.5, allow_inf_nan=False)] = Field(
        description="Poisson ratio, below the incompressible limit."
    )
    yield_stress_pa: NonNegative = Field(description="Herschel-Bulkley yield stress in Pa.")
    consistency_pa_s_n: NonNegative = Field(description="Consistency index in Pa s^n.")
    flow_index: Positive = Field(description="Flow index n.")
    initial_size_m: PositiveVec3 = Field(description="Initial material block size in metres.")
    initial_velocity_m_s: Vec3 = Field(description="Initial material velocity in m/s.")

    @model_validator(mode="after")
    def _domain_rules(self) -> Self:
        """Bound the grid before any allocation can happen.

        Returns:
            Self: The validated configuration.

        Raises:
            ValueError: When the block leaves the domain or the grid is oversized.
        """
        for axis, (size, domain) in enumerate(
            zip(self.initial_size_m, self.domain_m, strict=True)
        ):
            if size > domain:
                raise ValueError(f"`initial_size_m[{axis}]` exceeds `domain_m[{axis}]`")
        cells = math.prod(math.ceil(extent / self.grid_spacing_m) for extent in self.domain_m)
        if cells > MAX_RESEARCH_CELLS:
            raise ValueError(
                f"research grid would allocate {cells} cells, above the cap "
                f"{MAX_RESEARCH_CELLS}; coarsen `grid_spacing_m` or shrink `domain_m`"
            )
        return self


class SimulationRequest(StrictModel):
    """Complete validated native request."""

    schema_version: int = Field(description="Wire schema version.")
    mode: Mode = Field(description="Simulation mode.")
    seed: int = Field(ge=0, le=MAX_U64, description="Unsigned 64-bit master seed.")
    duration_s: Positive = Field(description="Simulated duration in seconds.")
    dt_s: Positive = Field(description="Nominal timestep in seconds.")
    record_interval_s: Positive = Field(description="Frame recording interval in seconds.")
    max_wall_time_s: Annotated[float, Field(gt=0, le=MAX_WALL_TIME_S, allow_inf_nan=False)] = (
        Field(description="Native wall-clock budget in seconds; at most one hour.")
    )
    boxes: Annotated[list[BoxCfg], Field(max_length=MAX_BOXES)] = Field(
        description="Sifting boxes; at least one in household mode."
    )
    materials: MaterialsCfg = Field(description="Household material parameters.")
    events: Annotated[list[EventCfg], Field(max_length=MAX_EVENTS)] = Field(
        description="Time-ordered household events."
    )
    research: ResearchCfg | None = Field(
        description="Research fixture; required in research mode."
    )

    @model_validator(mode="after")
    def _structural_rules(self) -> Self:
        """Enforce cross-field rules from the wire contract.

        Returns:
            Self: The validated request.

        Raises:
            ValueError: When mode, box, timestep, or event rules are violated.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"`schema_version` '{self.schema_version}' is not supported "
                f"(expected {SCHEMA_VERSION})"
            )
        if self.dt_s > self.duration_s:
            raise ValueError("`dt_s` must not exceed `duration_s`")
        if self.record_interval_s < self.dt_s:
            raise ValueError("`record_interval_s` must be at least `dt_s`")
        box_ids = [box.id for box in self.boxes]
        if len(set(box_ids)) != len(box_ids):
            raise ValueError(f"`boxes` ids must be unique; got {box_ids}")
        boxes = {box.id: box for box in self.boxes}
        if self.mode is Mode.HOUSEHOLD:
            if not self.boxes:
                raise ValueError("household mode requires at least one box")
            if self.research is not None:
                raise ValueError("household mode must not set `research`")
        else:
            if self.research is None:
                raise ValueError("research mode requires `research`")
            if self.events:
                raise ValueError("research mode prohibits household `events`")
        previous = -math.inf
        for index, event in enumerate(self.events):
            if event.time_s < previous:
                raise ValueError(f"`events[{index}]` is earlier than its predecessor")
            previous = event.time_s
            if event.time_s > self.duration_s:
                raise ValueError(f"`events[{index}].time_s` exceeds `duration_s`")
            if event.time_s + event.duration_s > self.duration_s:
                raise ValueError(f"`events[{index}]` runs past `duration_s`")
            box = boxes.get(event.box_id)
            if box is None:
                raise ValueError(f"`events[{index}].box_id` '{event.box_id}' is not a box")
            for axis, (coordinate, extent) in enumerate(
                zip(event.position_m, box.size_m, strict=True)
            ):
                if not 0.0 <= coordinate <= extent:
                    raise ValueError(
                        f"`events[{index}].position_m[{axis}]` '{coordinate}' is outside "
                        f"box '{box.id}' (0..{extent})"
                    )
        return self

    def with_wall_budget(self, max_wall_time_s: float) -> SimulationRequest:
        """Copy the request with a different native wall budget.

        Parameters:
            max_wall_time_s (float): Replacement budget in seconds.

        Returns:
            SimulationRequest: A validated copy.
        """
        return self.model_copy(update={"max_wall_time_s": max_wall_time_s})

    def equivalent_except_budget(self, other: SimulationRequest) -> bool:
        """Compare two requests ignoring the wall budget, as resume requires.

        Parameters:
            other (SimulationRequest): The request to compare against.

        Returns:
            bool: True when every field except `max_wall_time_s` matches.
        """
        budget = self.max_wall_time_s
        return self.with_wall_budget(budget) == other.with_wall_budget(budget)


class Frame(StrictModel):
    """One recorded particle snapshot."""

    time_s: NonNegative = Field(description="Absolute simulation time in seconds.")
    positions_m: list[Vec3] = Field(
        description="World-space particle centres in metres, including box origins."
    )
    radii_m: list[Positive] = Field(description="Particle radii in metres.")
    materials: list[str] = Field(description="Material label per particle.")
    box_ids: list[str] = Field(description="Owning box per particle.")

    @model_validator(mode="after")
    def _equal_lengths(self) -> Self:
        """Require equal-length frame arrays.

        Returns:
            Self: The validated frame.

        Raises:
            ValueError: When per-particle arrays disagree in length.
        """
        lengths = {
            len(self.positions_m),
            len(self.radii_m),
            len(self.materials),
            len(self.box_ids),
        }
        if len(lengths) != 1:
            raise ValueError(f"frame arrays have unequal lengths {sorted(lengths)}")
        return self


class MetricRow(StrictModel):
    """One conservation-ledger sample."""

    time_s: NonNegative = Field(description="Absolute simulation time in seconds.")
    compartment: Compartment = Field(description="Ledger compartment.")
    wood_kg: NonNegative = Field(description="Dry wood solids in kilograms.")
    waste_kg: NonNegative = Field(description="Deposit solids in kilograms.")
    water_kg: NonNegative = Field(description="Water, counted once across phases.")


class Checkpoint(StrictModel):
    """Resumable native checkpoint."""

    schema_version: int = Field(description="Checkpoint schema version.")
    mode: Mode = Field(description="Mode that produced the state.")
    request: SimulationRequest = Field(description="The original complete request.")
    time_s: NonNegative = Field(description="Absolute simulation time reached.")
    state: dict[str, Any] = Field(description="Mode-private opaque state.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Reject checkpoints whose header disagrees with the embedded request.

        Returns:
            Self: The validated checkpoint.

        Raises:
            ValueError: On version, mode, or state inconsistencies.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unknown checkpoint `schema_version` '{self.schema_version}'")
        if self.mode is not self.request.mode:
            raise ValueError("checkpoint `mode` disagrees with `request.mode`")
        if self.time_s > self.request.duration_s:
            raise ValueError("checkpoint `time_s` exceeds `request.duration_s`")
        _require_finite(self.state, "state")
        return self


class SimulationOutput(StrictModel):
    """Complete native output for one execution segment."""

    schema_version: int = Field(description="Wire schema version.")
    mode: Mode = Field(description="Simulation mode.")
    status: Status = Field(description="Completion status; never fakes completion.")
    fidelity: Fidelity = Field(description="Honesty label for the numerical model.")
    time_s: NonNegative = Field(description="Absolute simulation time reached.")
    frames: list[Frame] = Field(description="Recorded frames for this segment only.")
    metrics: list[MetricRow] = Field(description="Ledger samples for this segment only.")
    observables: dict[str, Finite] = Field(description="Scalar summary observables.")
    diagnostics: list[str] = Field(description="Human-readable diagnostics.")
    checkpoint: Checkpoint = Field(description="State to resume from.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Cross-check status, time, compartments, and checkpoint agreement.

        Returns:
            Self: The validated output.

        Raises:
            ValueError: When the output contradicts itself or the contract.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unknown output `schema_version` '{self.schema_version}'")
        if self.mode is not self.checkpoint.mode:
            raise ValueError("output `mode` disagrees with checkpoint `mode`")
        if self.checkpoint.time_s != self.time_s:
            raise ValueError("output `time_s` disagrees with checkpoint `time_s`")
        duration = self.checkpoint.request.duration_s
        if self.status is Status.COMPLETED and self.time_s < duration:
            raise ValueError(
                f"status is `completed` but `time_s` '{self.time_s}' is short of "
                f"`duration_s` '{duration}'"
            )
        allowed = HOUSEHOLD_COMPARTMENTS if self.mode is Mode.HOUSEHOLD else RESEARCH_COMPARTMENTS
        for index, row in enumerate(self.metrics):
            if row.compartment not in allowed:
                raise ValueError(
                    f"`metrics[{index}].compartment` '{row.compartment}' is not valid in "
                    f"'{self.mode}' mode"
                )
        previous = -math.inf
        for index, frame in enumerate(self.frames):
            if frame.time_s < previous:
                raise ValueError(f"`frames[{index}]` is out of time order")
            previous = frame.time_s
            if frame.time_s > self.time_s:
                raise ValueError(f"`frames[{index}].time_s` is later than output `time_s`")
        return self


def _require_finite(value: object, path: str) -> None:
    """Recursively reject non-finite floats in opaque JSON-like data.

    Parameters:
        value (object): A JSON-like tree.
        path (str): Dotted path for error messages.

    Raises:
        ValueError: When any float is NaN or infinite.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at `{path}`")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")


class _UniqueLoader(yaml.SafeLoader):
    """Reject ambiguous YAML mappings rather than silently keeping the last value."""

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[str, Any]:
        """Check mapping keys before constructing their values.

        Parameters:
            node: yaml.Node: Mapping node being decoded.
            deep: bool: Whether to construct nested objects eagerly.
        Returns:
            dict[str, Any]: An unambiguous string-keyed mapping.
        Raises:
            ValueError: A mapping contains a non-string or repeated key.
        """
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("expected a YAML mapping")
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in seen:
                raise ValueError(f"non-string or duplicate mapping key '{key}'")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON keys at every nesting level.

    Parameters:
        pairs: list[tuple[str, Any]]: Decoder-preserved object entries.
    Returns:
        dict[str, Any]: An unambiguous object.
    Raises:
        ValueError: A key appears more than once.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate mapping key '{key}'")
        result[key] = value
    return result


def load_document(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping from disk.

    Parameters:
        path (Path): File ending in `.yaml`, `.yml`, or `.json`.

    Returns:
        dict[str, Any]: The top-level mapping.

    Raises:
        FileNotFoundError: When the file is missing.
        ValueError: When the file is not a mapping or has an unknown suffix.
    """
    resolved = Path(str(path)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document does not exist: '{resolved}'")
    if resolved.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported document suffix '{resolved.suffix}' for '{resolved}'")
    if resolved.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Document '{resolved}' exceeds the 32 MiB input limit")
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix in {".yaml", ".yml"}:
        try:
            if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text)):
                raise ValueError("YAML aliases are not supported; use explicit bounded values")
            loaded = yaml.load(text, Loader=_UniqueLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML '{resolved}': {e}") from e
    elif resolved.suffix == ".json":
        try:
            loaded = json.loads(text, object_pairs_hook=_unique_json_pairs)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON '{resolved}': {e}") from e
    else:
        raise ValueError(f"Unsupported document suffix '{resolved.suffix}' for '{resolved}'")
    if not isinstance(loaded, dict):
        raise ValueError(f"Document '{resolved}' must be a mapping at the top level")
    return loaded


def validate_payload[ModelT: BaseModel](model: type[ModelT], payload: object) -> ModelT:
    """Validate JSON-like data strictly, via the JSON validation path.

    Strict Python-mode validation refuses plain strings for enums; strict JSON-mode
    validation accepts them while still rejecting string-to-number coercion, booleans
    as numbers, unknown fields, and non-finite values.

    Parameters:
        model (type[ModelT]): Target model class.
        payload (object): JSON-like tree from YAML, JSON, or native output.

    Returns:
        ModelT: The validated instance.

    Raises:
        ValueError: On any validation failure, with the Pydantic message.
    """
    try:
        return model.model_validate_json(canonical_json(payload))
    except ValidationError as e:
        raise ValueError(str(e)) from e


def load_request(path: Path) -> SimulationRequest:
    """Load and validate a request document.

    Parameters:
        path (Path): YAML or JSON request.

    Returns:
        SimulationRequest: The validated request.
    """
    return validate_payload(SimulationRequest, load_document(path))


def request_to_wire(request: SimulationRequest) -> str:
    """Serialize a request to canonical JSON for the native call.

    Parameters:
        request (SimulationRequest): Validated request.

    Returns:
        str: Compact JSON with sorted keys.
    """
    return canonical_json(request.model_dump(mode="json"))


def canonical_json(payload: object) -> str:
    """Serialize JSON-like data deterministically.

    Parameters:
        payload (object): JSON-like tree.

    Returns:
        str: Sorted-key JSON without NaN/Infinity tokens.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_output(text: str) -> SimulationOutput:
    """Parse native output JSON into a validated model.

    Parameters:
        text (str): JSON produced by `run_json`.

    Returns:
        SimulationOutput: Validated output.

    Raises:
        ValueError: When the text is not valid JSON or violates the contract.
    """
    try:
        return SimulationOutput.model_validate_json(text)
    except ValidationError as e:
        raise ValueError(str(e)) from e


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "request": SimulationRequest,
    "output": SimulationOutput,
    "checkpoint": Checkpoint,
}


def export_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a JSON Schema document for a model.

    Parameters:
        model (type[BaseModel]): A Pydantic model class.

    Returns:
        dict[str, Any]: Draft 2020-12 JSON Schema with a `$schema` key.
    """
    schema = model.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema"} | schema
