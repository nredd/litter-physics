"""Observation import: measured material tests, visit logs, and maintenance logs.

Every observation file must say whether it is synthetic. Synthetic files exercise the
pipeline and can never satisfy a measured validation claim. Measurement groups carry
units, provenance, uncertainty, and a `measured` / `literature_prior` / `assumed`
status so downstream reports can say exactly what each number rests on.

References:
    docs/plan.md (Cats and calibration)
    docs/calibration.md
"""

from __future__ import annotations

import hashlib
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litter_physics.models import (
    Identifier,
    NonNegative,
    Positive,
    canonical_json,
    load_document,
    validate_payload,
)

LOGGER = logging.getLogger(__name__)

OBSERVATION_SCHEMA_VERSION = 1


class Provenance(StrEnum):
    """Where a number comes from."""

    MEASURED = "measured"
    LITERATURE_PRIOR = "literature_prior"
    ASSUMED = "assumed"


class StrictModel(BaseModel):
    """Strict frozen base for observation models."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MeasurementGroup(StrictModel):
    """Metadata shared by a group of samples."""

    status: Provenance = Field(description="Provenance status of the samples.")
    source: str = Field(min_length=1, description="Instrument, protocol, or citation.")
    instrument_uncertainty: NonNegative = Field(
        description="Absolute instrument uncertainty in the group's primary unit."
    )


class DryPourSample(StrictModel):
    """Poured pellet mass, volume, and repose angle."""

    mass_kg: Positive = Field(description="Poured dry mass in kilograms.")
    volume_m3: Positive = Field(description="Settled bulk volume in cubic metres.")
    repose_angle_deg: Annotated[float, Field(gt=0, lt=90, allow_inf_nan=False)] = Field(
        description="Angle of repose in degrees."
    )


class DryPourGroup(MeasurementGroup):
    """Dry pour and repose measurements."""

    samples: list[DryPourSample] = Field(min_length=1, description="Repeat pours.")


class UptakeSample(StrictModel):
    """Water uptake time series point."""

    time_s: NonNegative = Field(description="Immersion or exposure time in seconds.")
    water_mass_ratio: NonNegative = Field(description="Absorbed water mass per dry mass.")
    swelling_ratio: NonNegative | None = Field(
        default=None, description="Volume ratio to dry volume when measured."
    )


class UptakeGroup(MeasurementGroup):
    """Uptake and swelling series."""

    samples: list[UptakeSample] = Field(min_length=2, description="Time-ordered points.")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require nondecreasing times.

        Returns:
            Self: The validated group.

        Raises:
            ValueError: When samples are not time ordered.
        """
        times = [sample.time_s for sample in self.samples]
        if times != sorted(times):
            raise ValueError("uptake samples must be ordered by `time_s`")
        return self


class BreakupSample(StrictModel):
    """Disturbance-driven breakup measurement."""

    exposure_s: NonNegative = Field(description="Wet exposure time in seconds.")
    water_mass_ratio: NonNegative = Field(description="Water per dry mass during exposure.")
    broken_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = Field(
        description="Mass fraction below the pellet threshold after standardized agitation."
    )


class BreakupGroup(MeasurementGroup):
    """Breakup measurements."""

    samples: list[BreakupSample] = Field(min_length=2, description="Repeatable trials.")


class SlumpSample(StrictModel):
    """Single slump test result."""

    initial_height_m: Positive = Field(description="Cylinder height in metres.")
    final_height_m: Positive = Field(description="Slumped height in metres.")
    spread_m: Positive = Field(description="Final spread diameter in metres.")


class SlumpGroup(MeasurementGroup):
    """Slump tests on the clean surrogate paste."""

    samples: list[SlumpSample] = Field(min_length=1, description="Repeat tests.")


class FlowSample(StrictModel):
    """Steady shear rheometry point."""

    shear_rate_1_s: Positive = Field(description="Applied shear rate in 1/s.")
    shear_stress_pa: Positive = Field(description="Measured shear stress in Pa.")


class FlowGroup(MeasurementGroup):
    """Multi-rate flow curve for Herschel-Bulkley fitting."""

    samples: list[FlowSample] = Field(min_length=4, description="At least four rates.")


class AdhesionSample(StrictModel):
    """Traction-separation measurement against a substrate."""

    substrate: Identifier = Field(description="Substrate label, e.g. `steel` or `wood`.")
    peak_traction_pa: Positive = Field(description="Peak traction in Pa.")
    separation_m: Positive = Field(description="Separation at full detachment in metres.")


class AdhesionGroup(MeasurementGroup):
    """Adhesion measurements."""

    samples: list[AdhesionSample] = Field(min_length=1, description="Trials.")


class MaterialObservations(StrictModel):
    """All material test groups; any may be absent."""

    dry_pour: DryPourGroup | None = Field(default=None, description="Dry pour tests.")
    uptake: UptakeGroup | None = Field(default=None, description="Uptake series.")
    breakup: BreakupGroup | None = Field(default=None, description="Breakup trials.")
    slump: SlumpGroup | None = Field(default=None, description="Slump tests.")
    flow_curve: FlowGroup | None = Field(default=None, description="Rheometry points.")
    adhesion: AdhesionGroup | None = Field(default=None, description="Adhesion tests.")


class EliminationKind(StrEnum):
    """Observed deposit kind."""

    URINE = "urine"
    FIRM_STOOL = "firm_stool"
    SOFT_STOOL = "soft_stool"
    LIQUID_STOOL = "liquid_stool"
    NONE = "none"


class VisitObservation(StrictModel):
    """One annotated visit."""

    cat_id: Identifier = Field(description="Observed cat identifier.")
    box_id: Identifier = Field(description="Box used.")
    start_s: NonNegative = Field(description="Visit start relative to the log origin.")
    inspect_s: NonNegative = Field(description="Time inspecting before digging.")
    dig_s: NonNegative = Field(description="Total digging time.")
    dig_strokes: int = Field(ge=0, description="Counted digging strokes.")
    elimination: EliminationKind = Field(description="Deposit kind, `none` if aborted.")
    eliminate_s: NonNegative = Field(description="Elimination duration.")
    cover_s: NonNegative = Field(description="Covering duration; zero when skipped.")
    cover_strokes: int = Field(ge=0, description="Counted covering strokes.")
    abandoned: bool = Field(description="True when the cat left before eliminating.")
    deposit_mass_kg: NonNegative | None = Field(
        default=None, description="Measured deposit mass when available."
    )

    @property
    def duration_s(self) -> float:
        """Total in-box time.

        Returns:
            float: Sum of stage durations.
        """
        return self.inspect_s + self.dig_s + self.eliminate_s + self.cover_s


class MaintenanceKind(StrEnum):
    """Maintenance action kinds."""

    SCOOP = "scoop"
    STIR = "stir"
    EMPTY_DRAWER = "empty_drawer"
    REFILL = "refill"
    CLEAN = "clean"


class MaintenanceObservation(StrictModel):
    """One maintenance action with weighed masses."""

    time_s: NonNegative = Field(description="Action time relative to the log origin.")
    kind: MaintenanceKind = Field(description="Action kind.")
    box_id: Identifier = Field(description="Box acted on.")
    removed_mass_kg: NonNegative = Field(description="Mass removed, weighed.")
    clean_litter_loss_kg: NonNegative = Field(description="Clean pellets removed alongside.")
    added_mass_kg: NonNegative = Field(description="Mass added for refill or clean.")


class ObservationSet(StrictModel):
    """Top-level observation file."""

    schema_version: int = Field(description="Observation schema version.")
    synthetic: bool = Field(description="True when generated, never measured.")
    description: str = Field(min_length=1, description="What was observed and how.")
    materials: MaterialObservations = Field(description="Material tests.")
    visits: list[VisitObservation] = Field(description="Annotated visits.")
    maintenance: list[MaintenanceObservation] = Field(description="Weighed maintenance.")

    @model_validator(mode="after")
    def _versioned(self) -> Self:
        """Check the schema version and visit ordering.

        Returns:
            Self: The validated set.

        Raises:
            ValueError: On unknown versions or unordered logs.
        """
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"unknown observation `schema_version` '{self.schema_version}'")
        starts = [visit.start_s for visit in self.visits]
        if starts != sorted(starts):
            raise ValueError("visits must be ordered by `start_s`")
        times = [item.time_s for item in self.maintenance]
        if times != sorted(times):
            raise ValueError("maintenance must be ordered by `time_s`")
        return self

    @property
    def cat_ids(self) -> list[str]:
        """Distinct cat identifiers in visit order.

        Returns:
            list[str]: Order-preserving unique cat ids.
        """
        return list(dict.fromkeys(visit.cat_id for visit in self.visits))

    @property
    def sha256(self) -> str:
        """Hash of the canonical observation JSON.

        Returns:
            str: Hex digest.
        """
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


def load_observations(path: Path) -> ObservationSet:
    """Load and validate an observation file.

    Parameters:
        path (Path): YAML or JSON file.

    Returns:
        ObservationSet: The validated observations.
    """
    observations = validate_payload(ObservationSet, load_document(path))
    if observations.synthetic:
        LOGGER.warning(f"observations '{path}' are SYNTHETIC; no measured claim is possible")
    return observations


def synthesize_observations(
    *, seed: int, cats: int, visits_per_cat: int, box_ids: list[str]
) -> ObservationSet:
    """Generate a clean synthetic observation set for pipeline exercise.

    The material groups are internally consistent with simple closed-form models so the
    calibration fits have a known answer. Nothing here is a cat.

    Parameters:
        seed (int): Generator seed.
        cats (int): Number of synthetic cats, at least 1.
        visits_per_cat (int): Visits per cat, at least 1.
        box_ids (list[str]): Boxes to distribute visits across.

    Returns:
        ObservationSet: Synthetic observations flagged `synthetic=True`.

    Raises:
        ValueError: When counts are non-positive or no boxes are given.
    """
    if cats < 1 or visits_per_cat < 1:
        raise ValueError("`cats` and `visits_per_cat` must be positive")
    if not box_ids:
        raise ValueError("`box_ids` must not be empty")
    rng = np.random.default_rng(seed)
    capacity = 2.8
    uptake_rate = 0.12
    breakdown_rate = 0.004
    uptake_times = np.linspace(0.0, 60.0, 13)
    uptake = [
        UptakeSample(
            time_s=float(t),
            water_mass_ratio=float(capacity * (1.0 - np.exp(-uptake_rate * t))),
            swelling_ratio=float(1.0 + 0.3 * (1.0 - np.exp(-uptake_rate * t))),
        )
        for t in uptake_times
    ]
    breakup = [
        BreakupSample(
            exposure_s=float(t),
            water_mass_ratio=2.0,
            broken_fraction=float(min(1.0, 1.0 - np.exp(-breakdown_rate * t))),
        )
        for t in np.linspace(0.0, 900.0, 7)
    ]
    rates = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    flow = [
        FlowSample(shear_rate_1_s=float(rate), shear_stress_pa=float(30.0 + 10.0 * rate**0.6))
        for rate in rates
    ]
    visits: list[VisitObservation] = []
    for cat_index in range(cats):
        cat_id = f"cat-{cat_index}"
        interval_mu = np.log(6.0 * 3600.0) + 0.2 * cat_index
        clock = float(rng.uniform(0.0, 3600.0))
        for _ in range(visits_per_cat):
            clock += float(rng.lognormal(interval_mu, 0.3))
            abandoned = bool(rng.uniform() < 0.05)
            elimination = (
                EliminationKind.NONE
                if abandoned
                else EliminationKind(
                    rng.choice(["urine", "firm_stool", "soft_stool"], p=[0.7, 0.2, 0.1])
                )
            )
            visits.append(
                VisitObservation(
                    cat_id=cat_id,
                    box_id=box_ids[int(rng.integers(len(box_ids)))],
                    start_s=clock,
                    inspect_s=float(rng.lognormal(np.log(4.0), 0.4)),
                    dig_s=0.0 if abandoned else float(rng.lognormal(np.log(5.0 + cat_index), 0.4)),
                    dig_strokes=0 if abandoned else int(rng.poisson(6 + 2 * cat_index)),
                    elimination=elimination,
                    eliminate_s=0.0 if abandoned else float(rng.lognormal(np.log(15.0), 0.3)),
                    cover_s=0.0
                    if abandoned or rng.uniform() < 0.1
                    else float(rng.lognormal(np.log(6.0), 0.4)),
                    cover_strokes=0 if abandoned else int(rng.poisson(5)),
                    abandoned=abandoned,
                    deposit_mass_kg=None if abandoned else float(rng.lognormal(np.log(0.03), 0.3)),
                )
            )
    visits.sort(key=lambda visit: visit.start_s)
    horizon = visits[-1].start_s if visits else 0.0
    maintenance = [
        MaintenanceObservation(
            time_s=float(day * 86400.0),
            kind=MaintenanceKind.SCOOP,
            box_id=box_ids[day % len(box_ids)],
            removed_mass_kg=float(rng.lognormal(np.log(0.08), 0.2)),
            clean_litter_loss_kg=float(rng.lognormal(np.log(0.01), 0.3)),
            added_mass_kg=0.0,
        )
        for day in range(1, int(horizon // 86400.0) + 1)
    ]
    return ObservationSet(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        synthetic=True,
        description=(
            "SYNTHETIC observations generated by `synthesize_observations`; closed-form "
            f"truth: capacity={capacity}, uptake_rate={uptake_rate}, "
            f"breakdown_rate={breakdown_rate}, HB(30, 10, 0.6)"
        ),
        materials=MaterialObservations(
            dry_pour=DryPourGroup(
                status=Provenance.ASSUMED,
                source="synthetic",
                instrument_uncertainty=0.0005,
                samples=[
                    DryPourSample(mass_kg=0.5, volume_m3=0.00083, repose_angle_deg=33.0),
                    DryPourSample(mass_kg=0.5, volume_m3=0.00085, repose_angle_deg=34.0),
                ],
            ),
            uptake=UptakeGroup(
                status=Provenance.ASSUMED,
                source="synthetic",
                instrument_uncertainty=0.01,
                samples=uptake,
            ),
            breakup=BreakupGroup(
                status=Provenance.ASSUMED,
                source="synthetic",
                instrument_uncertainty=0.02,
                samples=breakup,
            ),
            slump=SlumpGroup(
                status=Provenance.ASSUMED,
                source="synthetic",
                instrument_uncertainty=0.001,
                samples=[SlumpSample(initial_height_m=0.05, final_height_m=0.03, spread_m=0.09)],
            ),
            flow_curve=FlowGroup(
                status=Provenance.ASSUMED,
                source="synthetic",
                instrument_uncertainty=0.5,
                samples=flow,
            ),
            adhesion=None,
        ),
        visits=visits,
        maintenance=maintenance,
    )


def observations_to_payload(observations: ObservationSet) -> dict[str, Any]:
    """Dump observations to a JSON-like tree.

    Parameters:
        observations (ObservationSet): Validated observations.

    Returns:
        dict[str, Any]: JSON-compatible mapping.
    """
    return observations.model_dump(mode="json")
