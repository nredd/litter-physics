"""Per-cat semi-Markov visit generation with independent seed streams.

States: approach -> enter -> inspect -> dig -> posture -> eliminate -> cover/abandon
-> exit. Stages may be skipped or repeated according to profile probabilities; sojourn
times are lognormal. Box choice applies occupancy exclusion only; cleanliness effects
are deliberately absent until evidence exists. Profiles are empirical summaries of
observed visits, or explicitly synthetic.

Missing on purpose: RL, inferred emotions, developmental laws, paw force inference
from motion (paw events carry a radius and direction; force limits belong to the
compliant contact model in Rust).

References:
    docs/plan.md (Cats and calibration)
    https://numpy.org/doc/stable/reference/random/parallel.html
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litter_physics.models import (
    BoxCfg,
    EventCfg,
    EventKind,
    Identifier,
    NonNegative,
    Positive,
    SimulationRequest,
    UnitInterval,
    validate_payload,
)
from litter_physics.observations import EliminationKind, ObservationSet, VisitObservation

LOGGER = logging.getLogger(__name__)

PROFILE_SCHEMA_VERSION = 1
MAX_VISITS_PER_CAT = 10_000
STROKE_DURATION_S = 0.35
MIN_LOG_SIGMA = 0.05


class Stage(StrEnum):
    """Semi-Markov visit stages."""

    APPROACH = "approach"
    ENTER = "enter"
    INSPECT = "inspect"
    DIG = "dig"
    POSTURE = "posture"
    ELIMINATE = "eliminate"
    COVER = "cover"
    ABANDON = "abandon"
    EXIT = "exit"


class LogNormal(BaseModel):
    """Lognormal sojourn or count distribution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    log_mean: Annotated[float, Field(allow_inf_nan=False)] = Field(
        description="Mean of the natural log."
    )
    log_sigma: Annotated[float, Field(gt=0, allow_inf_nan=False)] = Field(
        description="Standard deviation of the natural log."
    )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one value.

        Parameters:
            rng (np.random.Generator): The cat's private generator.

        Returns:
            float: A positive sample.
        """
        return float(rng.lognormal(self.log_mean, self.log_sigma))


class CatProfile(BaseModel):
    """Empirical or synthetic behaviour profile for one cat."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    cat_id: Identifier = Field(description="Cat identifier.")
    synthetic: bool = Field(description="True when not derived from observed visits.")
    observed_visits: int = Field(ge=0, description="Visits the profile was built from.")
    paw_radius_m: Positive = Field(description="Paw contact proxy radius in metres.")
    source_height_m: Positive = Field(description="Elimination source height above the bed.")
    visit_interval: LogNormal = Field(description="Time between visit starts, seconds.")
    inspect: LogNormal = Field(description="Inspect stage duration, seconds.")
    dig_strokes: LogNormal = Field(description="Dig stroke count (rounded, >= 1).")
    eliminate: LogNormal = Field(description="Eliminate stage duration, seconds.")
    cover_strokes: LogNormal = Field(description="Cover stroke count (rounded, >= 1).")
    deposit_mass: LogNormal = Field(description="Deposit mass in kilograms.")
    p_abandon: UnitInterval = Field(description="Abandon after inspect.")
    p_skip_dig: UnitInterval = Field(description="Skip the dig stage.")
    p_repeat_dig: UnitInterval = Field(description="Repeat the dig stage once more.")
    p_no_cover: UnitInterval = Field(description="Skip covering.")
    p_urine: UnitInterval = Field(description="Urination versus defecation.")
    p_soft_stool: UnitInterval = Field(description="Soft stool given defecation.")
    stool_water_fraction: UnitInterval = Field(description="Water fraction of firm stool.")
    box_weights: dict[str, NonNegative] = Field(description="Relative box preference.")
    location_fraction: Annotated[list[UnitInterval], Field(min_length=2, max_length=2)] = Field(
        description="Preferred (x, y) position as fractions of the box footprint."
    )
    location_spread_m: NonNegative = Field(description="Positional spread in metres.")

    @model_validator(mode="after")
    def _weights(self) -> Self:
        """Require at least one positive box weight.

        Returns:
            Self: The validated profile.

        Raises:
            ValueError: When every box weight is zero.
        """
        if not self.box_weights or all(weight == 0.0 for weight in self.box_weights.values()):
            raise ValueError(f"profile '{self.cat_id}' needs a positive `box_weights` entry")
        return self


class VisitRecord(BaseModel):
    """Generated visit, for logs and validation replay comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cat_id: str = Field(description="Cat identifier.")
    box_id: str = Field(description="Box used.")
    start_s: float = Field(description="Visit start time.")
    end_s: float = Field(description="Visit end time.")
    stages: list[str] = Field(description="Stage sequence actually taken.")
    elimination: str = Field(description="Deposit kind or `none`.")
    waited_s: float = Field(description="Time waited for an occupied box.")


class GeneratedVisits(BaseModel):
    """Output of the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic: bool = Field(description="True when any profile is synthetic.")
    visits: list[VisitRecord] = Field(description="Visits in start order.")
    events: list[EventCfg] = Field(description="Wire events in time order.")
    truncated: int = Field(description="Visits dropped because they overran the horizon.")


def _fit_lognormal(values: list[float], fallback: LogNormal) -> LogNormal:
    """Fit a lognormal from positive samples, falling back when too few.

    Parameters:
        values (list[float]): Samples; non-positive values are dropped.
        fallback (LogNormal): Used when fewer than two samples remain.

    Returns:
        LogNormal: The fitted or fallback distribution.
    """
    positive = np.array([value for value in values if value > 0.0])
    if positive.size < 2:
        return fallback
    logs = np.log(positive)
    return LogNormal(
        log_mean=float(logs.mean()), log_sigma=float(max(logs.std(ddof=1), MIN_LOG_SIGMA))
    )


def _rate(numerator: int, denominator: int, fallback: float) -> float:
    """Compute a probability with a fallback for empty denominators.

    Parameters:
        numerator (int): Count of successes.
        denominator (int): Count of trials.
        fallback (float): Value when there are no trials.

    Returns:
        float: Probability in [0, 1].
    """
    return numerator / denominator if denominator > 0 else fallback


def synthetic_profile(cat_id: str, box_ids: list[str], *, variant: int = 0) -> CatProfile:
    """Build a clearly synthetic profile.

    Parameters:
        cat_id (str): Cat identifier.
        box_ids (list[str]): Boxes available.
        variant (int): Small integer to differentiate synthetic cats.

    Returns:
        CatProfile: A synthetic profile.
    """
    return CatProfile(
        cat_id=cat_id,
        synthetic=True,
        observed_visits=0,
        paw_radius_m=0.012,
        source_height_m=0.05,
        visit_interval=LogNormal(log_mean=float(np.log(6.0 * 3600.0)), log_sigma=0.4),
        inspect=LogNormal(log_mean=float(np.log(4.0)), log_sigma=0.4),
        dig_strokes=LogNormal(log_mean=float(np.log(6.0 + variant)), log_sigma=0.4),
        eliminate=LogNormal(log_mean=float(np.log(15.0)), log_sigma=0.3),
        cover_strokes=LogNormal(log_mean=float(np.log(5.0)), log_sigma=0.4),
        deposit_mass=LogNormal(log_mean=float(np.log(0.03)), log_sigma=0.3),
        p_abandon=0.05,
        p_skip_dig=0.1,
        p_repeat_dig=0.2,
        p_no_cover=0.1,
        p_urine=0.7,
        p_soft_stool=0.3,
        stool_water_fraction=0.7,
        box_weights={box_id: 1.0 for box_id in box_ids},
        location_fraction=[0.5, 0.5],
        location_spread_m=0.02,
    )


def profile_from_observations(
    observations: ObservationSet, cat_id: str, box_ids: list[str]
) -> CatProfile:
    """Summarize a cat's observed visits into a profile.

    Parameters:
        observations (ObservationSet): Observations containing visits.
        cat_id (str): Cat to summarize.
        box_ids (list[str]): Boxes that exist in the scenario.

    Returns:
        CatProfile: Empirical profile; `synthetic` mirrors the observation flag.

    Raises:
        ValueError: When the cat has no visits.
    """
    visits: list[VisitObservation] = [v for v in observations.visits if v.cat_id == cat_id]
    if not visits:
        raise ValueError(f"no visits for cat '{cat_id}'")
    base = synthetic_profile(cat_id, box_ids)
    starts = [visit.start_s for visit in visits]
    intervals = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    completed = [visit for visit in visits if not visit.abandoned]
    defecations = [
        visit
        for visit in completed
        if visit.elimination
        in {EliminationKind.FIRM_STOOL, EliminationKind.SOFT_STOOL, EliminationKind.LIQUID_STOOL}
    ]
    box_counts = {box_id: 0.0 for box_id in box_ids}
    for visit in visits:
        if visit.box_id in box_counts:
            box_counts[visit.box_id] += 1.0
    if all(count == 0.0 for count in box_counts.values()):
        box_counts = {box_id: 1.0 for box_id in box_ids}
    return base.model_copy(
        update={
            "synthetic": observations.synthetic,
            "observed_visits": len(visits),
            "visit_interval": _fit_lognormal(intervals, base.visit_interval),
            "inspect": _fit_lognormal([v.inspect_s for v in visits], base.inspect),
            "dig_strokes": _fit_lognormal(
                [float(v.dig_strokes) for v in visits], base.dig_strokes
            ),
            "eliminate": _fit_lognormal([v.eliminate_s for v in completed], base.eliminate),
            "cover_strokes": _fit_lognormal(
                [float(v.cover_strokes) for v in completed], base.cover_strokes
            ),
            "deposit_mass": _fit_lognormal(
                [v.deposit_mass_kg for v in completed if v.deposit_mass_kg is not None],
                base.deposit_mass,
            ),
            "p_abandon": _rate(sum(1 for v in visits if v.abandoned), len(visits), 0.0),
            "p_skip_dig": _rate(
                sum(1 for v in completed if v.dig_strokes == 0), len(completed), 0.0
            ),
            "p_no_cover": _rate(
                sum(1 for v in completed if v.cover_s == 0.0), len(completed), 0.0
            ),
            "p_urine": _rate(
                sum(1 for v in completed if v.elimination is EliminationKind.URINE),
                len(completed),
                0.7,
            ),
            "p_soft_stool": _rate(
                sum(1 for v in defecations if v.elimination is not EliminationKind.FIRM_STOOL),
                len(defecations),
                0.3,
            ),
            "box_weights": box_counts,
        }
    )


def _pick_box(
    profile: CatProfile,
    rng: np.random.Generator,
    boxes: dict[str, BoxCfg],
    busy_until: dict[str, float],
    now: float,
) -> tuple[str, float]:
    """Choose a box with occupancy exclusion.

    Parameters:
        profile (CatProfile): The cat's preferences.
        rng (np.random.Generator): The cat's generator.
        boxes (dict[str, BoxCfg]): Available boxes.
        busy_until (dict[str, float]): Time each box becomes free.
        now (float): Current time.

    Returns:
        tuple[str, float]: Chosen box and seconds waited.
    """
    ids = [box_id for box_id in boxes if profile.box_weights.get(box_id, 0.0) > 0.0]
    weights = np.array([profile.box_weights[box_id] for box_id in ids])
    free = [box_id for box_id in ids if busy_until.get(box_id, 0.0) <= now]
    if free:
        free_weights = np.array([profile.box_weights[box_id] for box_id in free])
        return str(rng.choice(free, p=free_weights / free_weights.sum())), 0.0
    chosen = str(rng.choice(ids, p=weights / weights.sum()))
    return chosen, max(0.0, busy_until[chosen] - now)


def _position(
    profile: CatProfile, rng: np.random.Generator, box: BoxCfg, height: float
) -> list[float]:
    """Sample a box-local position near the preferred location, clamped inside.

    Parameters:
        profile (CatProfile): Preferred location and spread.
        rng (np.random.Generator): The cat's generator.
        box (BoxCfg): Target box.
        height (float): Height above the box floor.

    Returns:
        list[float]: `[x, y, z]` inside the box.
    """
    x = profile.location_fraction[0] * box.size_m[0] + rng.normal(0.0, profile.location_spread_m)
    y = profile.location_fraction[1] * box.size_m[1] + rng.normal(0.0, profile.location_spread_m)
    margin = profile.paw_radius_m
    return [
        float(np.clip(x, margin, box.size_m[0] - margin)),
        float(np.clip(y, margin, box.size_m[1] - margin)),
        float(np.clip(height, 0.0, box.size_m[2])),
    ]


def _stroke_events(
    kind: EventKind,
    profile: CatProfile,
    rng: np.random.Generator,
    box: BoxCfg,
    start: float,
    strokes: int,
) -> tuple[list[EventCfg], float]:
    """Emit one motion event per stroke with alternating direction.

    Parameters:
        kind (EventKind): `dig` or `cover`.
        profile (CatProfile): Paw geometry.
        rng (np.random.Generator): The cat's generator.
        box (BoxCfg): Target box.
        start (float): Stage start time.
        strokes (int): Stroke count, at least 1.

    Returns:
        tuple[list[EventCfg], float]: Events and the stage end time.
    """
    events: list[EventCfg] = []
    clock = start
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    for stroke in range(strokes):
        sign = 1.0 if stroke % 2 == 0 else -1.0
        direction = [sign * float(np.cos(angle)), sign * float(np.sin(angle)), -0.3]
        events.append(
            EventCfg(
                time_s=clock,
                kind=kind,
                box_id=box.id,
                position_m=_position(profile, rng, box, profile.paw_radius_m),
                direction=direction,
                duration_s=STROKE_DURATION_S,
                amount_kg=0.0,
                water_fraction=0.0,
                radius_m=profile.paw_radius_m,
            )
        )
        clock += STROKE_DURATION_S
    return events, clock


def _eliminate_event(
    profile: CatProfile, rng: np.random.Generator, box: BoxCfg, start: float, duration: float
) -> tuple[EventCfg, str]:
    """Emit a deposit event.

    Parameters:
        profile (CatProfile): Deposit probabilities and masses.
        rng (np.random.Generator): The cat's generator.
        box (BoxCfg): Target box.
        start (float): Stage start time.
        duration (float): Stage duration.

    Returns:
        tuple[EventCfg, str]: The event and the elimination label.
    """
    mass = profile.deposit_mass.sample(rng)
    if rng.uniform() < profile.p_urine:
        kind, label, water = EventKind.URINATE, "urine", 1.0
    elif rng.uniform() < profile.p_soft_stool:
        kind, label, water = (
            EventKind.DEFECATE,
            "soft_stool",
            min(1.0, profile.stool_water_fraction + 0.15),
        )
    else:
        kind, label, water = EventKind.DEFECATE, "firm_stool", profile.stool_water_fraction
    event = EventCfg(
        time_s=start,
        kind=kind,
        box_id=box.id,
        position_m=_position(profile, rng, box, profile.source_height_m),
        direction=[0.0, 0.0, -1.0],
        duration_s=duration,
        amount_kg=mass,
        water_fraction=water,
        radius_m=0.015 if kind is EventKind.URINATE else 0.01,
    )
    return event, label


def _one_visit(
    profile: CatProfile,
    rng: np.random.Generator,
    box: BoxCfg,
    start: float,
) -> tuple[list[EventCfg], list[str], str, float]:
    """Walk the semi-Markov chain for one visit.

    Parameters:
        profile (CatProfile): Stage distributions and probabilities.
        rng (np.random.Generator): The cat's generator.
        box (BoxCfg): Chosen box.
        start (float): Visit start.

    Returns:
        tuple[list[EventCfg], list[str], str, float]: Events, stage path, elimination
            label, and end time.
    """
    events: list[EventCfg] = []
    stages = [Stage.APPROACH, Stage.ENTER, Stage.INSPECT]
    clock = start + profile.inspect.sample(rng)
    if rng.uniform() < profile.p_abandon:
        stages.extend([Stage.ABANDON, Stage.EXIT])
        return events, [str(stage) for stage in stages], "none", clock
    if rng.uniform() >= profile.p_skip_dig:
        repeats = 2 if rng.uniform() < profile.p_repeat_dig else 1
        for _ in range(repeats):
            strokes = max(1, round(profile.dig_strokes.sample(rng)))
            new_events, clock = _stroke_events(EventKind.DIG, profile, rng, box, clock, strokes)
            events.extend(new_events)
            stages.append(Stage.DIG)
    stages.append(Stage.POSTURE)
    duration = profile.eliminate.sample(rng)
    event, label = _eliminate_event(profile, rng, box, clock, duration)
    events.append(event)
    clock += duration
    stages.append(Stage.ELIMINATE)
    if rng.uniform() >= profile.p_no_cover:
        strokes = max(1, round(profile.cover_strokes.sample(rng)))
        new_events, clock = _stroke_events(EventKind.COVER, profile, rng, box, clock, strokes)
        events.extend(new_events)
        stages.append(Stage.COVER)
    stages.append(Stage.EXIT)
    return events, [str(stage) for stage in stages], label, clock


def generate_visits(
    profiles: list[CatProfile],
    boxes: list[BoxCfg],
    *,
    master_seed: int,
    horizon_s: float,
    max_visits_per_cat: int = MAX_VISITS_PER_CAT,
) -> GeneratedVisits:
    """Generate interleaved visits for every cat with independent seed streams.

    Parameters:
        profiles (list[CatProfile]): One profile per cat; ids must be unique.
        boxes (list[BoxCfg]): Boxes in the scenario.
        master_seed (int): Seed spawning one child stream per cat, in profile order.
        horizon_s (float): Generation horizon; visits that would end after it are
            dropped and counted in `truncated`.
        max_visits_per_cat (int): Hard cap per cat.

    Returns:
        GeneratedVisits: Visits and time-ordered wire events.

    Raises:
        ValueError: On duplicate cat ids, no boxes, or a non-positive horizon.
    """
    if not boxes:
        raise ValueError("`boxes` must not be empty")
    if horizon_s <= 0.0:
        raise ValueError(f"`horizon_s` '{horizon_s}' must be positive")
    ids = [profile.cat_id for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError(f"profile cat ids must be unique; got {ids}")
    box_map = {box.id: box for box in boxes}
    for profile in profiles:
        unknown = set(profile.box_weights) - set(box_map)
        if unknown:
            raise ValueError(
                f"profile '{profile.cat_id}' references unknown boxes {sorted(unknown)}"
            )
    streams = np.random.SeedSequence(master_seed).spawn(len(profiles))
    generators = [np.random.default_rng(stream) for stream in streams]
    pending: list[tuple[float, int]] = []
    for index, rng in enumerate(generators):
        pending.append((float(rng.uniform(0.0, min(horizon_s, 3600.0))), index))
    counts = [0 for _ in profiles]
    busy_until: dict[str, float] = {box_id: 0.0 for box_id in box_map}
    visits: list[VisitRecord] = []
    events: list[EventCfg] = []
    truncated = 0
    while pending:
        pending.sort()
        start, index = pending.pop(0)
        profile, rng = profiles[index], generators[index]
        if start > horizon_s or counts[index] >= max_visits_per_cat:
            continue
        box_id, waited = _pick_box(profile, rng, box_map, busy_until, start)
        actual_start = start + waited
        new_events, stages, label, end = _one_visit(profile, rng, box_map[box_id], actual_start)
        counts[index] += 1
        next_start = start + profile.visit_interval.sample(rng)
        pending.append((next_start, index))
        if end > horizon_s:
            truncated += 1
            continue
        busy_until[box_id] = end
        visits.append(
            VisitRecord(
                cat_id=profile.cat_id,
                box_id=box_id,
                start_s=actual_start,
                end_s=end,
                stages=stages,
                elimination=label,
                waited_s=waited,
            )
        )
        events.extend(new_events)
    events.sort(key=lambda event: event.time_s)
    visits.sort(key=lambda visit: visit.start_s)
    LOGGER.info(
        f"generated {len(visits)} visits / {len(events)} events for {len(profiles)} cats "
        f"(truncated={truncated})"
    )
    return GeneratedVisits(
        synthetic=any(profile.synthetic for profile in profiles),
        visits=visits,
        events=events,
        truncated=truncated,
    )


def request_with_events(
    template: SimulationRequest, generated: GeneratedVisits, *, horizon_s: float
) -> SimulationRequest:
    """Copy a household request with generated events and matching duration.

    Parameters:
        template (SimulationRequest): Household request supplying boxes and materials.
        generated (GeneratedVisits): Generated events.
        horizon_s (float): New `duration_s`.

    Returns:
        SimulationRequest: Validated request.
    """
    payload = template.model_dump(mode="json") | {
        "duration_s": horizon_s,
        "events": [event.model_dump(mode="json") for event in generated.events],
    }
    return validate_payload(SimulationRequest, payload)


def load_profiles(payload: object) -> list[CatProfile]:
    """Validate a list of profile payloads.

    Parameters:
        payload (object): JSON-like list of profiles.

    Returns:
        list[CatProfile]: Validated profiles.

    Raises:
        ValueError: When the payload is not a list.
    """
    if not isinstance(payload, list):
        raise ValueError("profiles document must be a list of profiles")
    return [validate_payload(CatProfile, item) for item in payload]
