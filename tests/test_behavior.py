"""Semi-Markov visit generation.

References:
    docs/plan.md (Cats and calibration)
"""

from __future__ import annotations

import pytest

from litter_physics.behavior import (
    CatProfile,
    generate_visits,
    load_profiles,
    profile_from_observations,
    request_with_events,
    synthetic_profile,
)
from litter_physics.models import SimulationRequest
from litter_physics.observations import synthesize_observations


def test_generation_is_deterministic_per_cat(household_request: SimulationRequest) -> None:
    """Same seeds give identical output; adding a cat leaves the first cat's stream alone."""
    boxes = list(household_request.boxes)
    ids = [box.id for box in boxes]
    solo = [synthetic_profile("a", ids)]
    pair = [synthetic_profile("a", ids), synthetic_profile("b", ids, variant=1)]
    first = generate_visits(solo, boxes, master_seed=1, horizon_s=86400.0)
    again = generate_visits(solo, boxes, master_seed=1, horizon_s=86400.0)
    assert first == again
    both = generate_visits(pair, boxes, master_seed=1, horizon_s=86400.0)
    cat_a_solo = [visit.start_s for visit in first.visits]
    cat_a_pair = [visit.start_s for visit in both.visits if visit.cat_id == "a"]
    assert cat_a_pair[0] == pytest.approx(cat_a_solo[0])
    other = generate_visits(solo, boxes, master_seed=2, horizon_s=86400.0)
    assert other != first


def test_events_are_valid_wire_events(household_request: SimulationRequest) -> None:
    """Generated events sort by time, stay inside boxes, and validate as a request."""
    boxes = list(household_request.boxes)
    profiles = [synthetic_profile("a", [box.id for box in boxes])]
    generated = generate_visits(profiles, boxes, master_seed=3, horizon_s=86400.0)
    assert generated.synthetic
    times = [event.time_s for event in generated.events]
    assert times == sorted(times)
    request = request_with_events(household_request, generated, horizon_s=86400.0)
    assert request.duration_s == 86400.0 and len(request.events) == len(generated.events)
    kinds = {str(event.kind) for event in generated.events}
    assert kinds <= {"dig", "urinate", "defecate", "cover"}
    for visit in generated.visits:
        assert visit.stages[0] == "approach" and visit.stages[-1] == "exit"


def test_occupancy_exclusion(household_request: SimulationRequest) -> None:
    """Two cats wanting the same single box never overlap."""
    box = household_request.boxes[0]
    profile_a = synthetic_profile("a", [box.id]).model_copy(
        update={"visit_interval": synthetic_profile("a", [box.id]).inspect}
    )
    profile_b = profile_a.model_copy(update={"cat_id": "b"})
    generated = generate_visits([profile_a, profile_b], [box], master_seed=9, horizon_s=600.0)
    intervals = sorted((visit.start_s, visit.end_s) for visit in generated.visits)
    for (_, end), (start, _) in zip(intervals, intervals[1:], strict=False):
        assert start >= end - 1e-9
    assert any(visit.waited_s > 0.0 for visit in generated.visits)


def test_profile_from_observations() -> None:
    """Empirical profiles summarize visit statistics per cat."""
    observations = synthesize_observations(seed=1, cats=2, visits_per_cat=25, box_ids=["box-a"])
    profile = profile_from_observations(observations, "cat-1", ["box-a"])
    assert profile.observed_visits == 25 and profile.synthetic
    assert profile.box_weights == {"box-a": 25.0}
    assert 0.0 <= profile.p_abandon <= 0.5
    with pytest.raises(ValueError, match="no visits"):
        profile_from_observations(observations, "cat-9", ["box-a"])


def test_generation_rejects_bad_inputs(household_request: SimulationRequest) -> None:
    """Duplicate ids, unknown boxes, and empty boxes are errors."""
    boxes = list(household_request.boxes)
    ids = [box.id for box in boxes]
    with pytest.raises(ValueError, match="unique"):
        generate_visits(
            [synthetic_profile("a", ids), synthetic_profile("a", ids)],
            boxes,
            master_seed=1,
            horizon_s=10.0,
        )
    with pytest.raises(ValueError, match="unknown boxes"):
        generate_visits([synthetic_profile("a", ["ghost"])], boxes, master_seed=1, horizon_s=10.0)
    with pytest.raises(ValueError, match="empty"):
        generate_visits([synthetic_profile("a", ids)], [], master_seed=1, horizon_s=10.0)
    with pytest.raises(ValueError, match="profiles document"):
        load_profiles({"cat_id": "a"})
    with pytest.raises(ValueError, match="box_weights"):
        CatProfile.model_validate(
            synthetic_profile("a", ids).model_dump() | {"box_weights": {"box-a": 0.0}}
        )
