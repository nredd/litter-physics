"""Observation import and calibration reporting on clean synthetic data.

References:
    docs/calibration.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from litter_physics.artifacts import METRIC_SCHEMA
from litter_physics.calibration import (
    apply_bundle,
    build_bundle,
    fit_rheology,
    load_bundle,
    render_report,
    score_household_masses,
    score_personalization,
    write_bundle,
)
from litter_physics.models import SimulationRequest, load_request
from litter_physics.observations import (
    MaterialObservations,
    ObservationSet,
    load_observations,
    synthesize_observations,
)
from tests.conftest import EXAMPLES


def test_fits_recover_synthetic_truth(household_request: SimulationRequest) -> None:
    """Closed-form synthetic curves are recovered, yet everything stays `assumed`."""
    observations = synthesize_observations(seed=2, cats=1, visits_per_cat=8, box_ids=["box-a"])
    bundle = build_bundle(observations, household_request.materials)
    assert bundle.synthetic
    assert bundle.parameters["water_capacity_ratio"].value == pytest.approx(2.8, rel=1e-3)
    assert bundle.parameters["uptake_rate_s"].value == pytest.approx(0.12, rel=1e-3)
    assert bundle.parameters["breakdown_rate_s"].value == pytest.approx(0.004, rel=1e-3)
    assert bundle.rheology["yield_stress_pa"].value == pytest.approx(30.0, rel=1e-3)
    assert bundle.rheology["flow_index"].value == pytest.approx(0.6, rel=1e-3)
    assert all(p.status == "assumed" for p in bundle.parameters.values())
    assert all(p.status == "assumed" for p in bundle.rheology.values())
    assert bundle.validation_claim.startswith("none")
    assert any("SYNTHETIC" in line for line in bundle.diagnostics)
    applied = apply_bundle(household_request, bundle)
    assert applied.materials.water_capacity_ratio == pytest.approx(2.8, rel=1e-3)


def test_slump_only_is_unidentifiable(household_request: SimulationRequest) -> None:
    """Without a flow curve, rheology is not fitted and the report says why."""
    observations = synthesize_observations(seed=2, cats=1, visits_per_cat=3, box_ids=["box-a"])
    materials = observations.materials.model_copy(update={"flow_curve": None})
    slump_only = observations.model_copy(update={"materials": materials})
    rheology, notes = fit_rheology(slump_only)
    assert rheology == {}
    assert any("NOT identifiable from slump alone" in note for note in notes)
    bundle = build_bundle(slump_only, household_request.materials)
    assert bundle.rheology == {}
    assert "not identified" in render_report(bundle, None)


def test_personalization_requires_evidence() -> None:
    """Synthetic visits never establish supported household personalization."""
    few = synthesize_observations(seed=4, cats=2, visits_per_cat=4, box_ids=["box-a"])
    scores = score_personalization(few)
    assert all(score.evidence == "insufficient" and not score.supported for score in scores)
    many = synthesize_observations(seed=4, cats=2, visits_per_cat=40, box_ids=["box-a"])
    scores = score_personalization(many)
    assert all(score.evidence == "synthetic" and not score.supported for score in scores)
    assert all(score.per_cat_log_likelihood is not None for score in scores)
    # Mock the provenance flag to exercise the separate measured-data policy branch.
    measured_policy = many.model_copy(update={"synthetic": False})
    assert all(score.evidence == "strong" for score in score_personalization(measured_policy))


def test_household_mass_score() -> None:
    """Removed-mass comparison uses the larger of 20% and repeatability."""
    observations = synthesize_observations(seed=5, cats=1, visits_per_cat=40, box_ids=["box-a"])
    measured = sum(item.removed_mass_kg for item in observations.maintenance)
    table = pa.table(
        {
            "time_s": [0.0, 1.0],
            "compartment": ["bed", "removed"],
            "wood_kg": [1.0, measured * 0.9],
            "waste_kg": [0.0, 0.0],
            "water_kg": [0.0, 0.0],
        },
        schema=METRIC_SCHEMA,
    )
    score = score_household_masses(table, observations)
    assert score is not None and score.within_tolerance
    assert "synthetic" in score.claim
    assert score_household_masses(METRIC_SCHEMA.empty_table(), observations) is None
    none = observations.model_copy(update={"maintenance": []})
    assert score_household_masses(table, none) is None


def test_bundle_round_trip(household_request: SimulationRequest, tmp_path: Path) -> None:
    """Bundles serialize, reload, and keep their hash."""
    observations = load_observations(EXAMPLES / "observations_synthetic.json")
    bundle = build_bundle(observations, household_request.materials)
    write_bundle(bundle, tmp_path / "bundle.json")
    assert load_bundle(tmp_path / "bundle.json").sha256 == bundle.sha256


def test_observation_validation_rules() -> None:
    """Unordered logs and unknown versions are rejected."""
    observations = synthesize_observations(seed=1, cats=1, visits_per_cat=3, box_ids=["box-a"])
    tree = observations.model_dump(mode="json")
    tree["visits"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        ObservationSet.model_validate_json(json.dumps(tree))
    tree = observations.model_dump(mode="json")
    tree["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version"):
        ObservationSet.model_validate_json(json.dumps(tree))
    with pytest.raises(ValueError):
        synthesize_observations(seed=1, cats=0, visits_per_cat=3, box_ids=["box-a"])
    with pytest.raises(ValueError):
        synthesize_observations(seed=1, cats=1, visits_per_cat=3, box_ids=[])
    empty = observations.model_copy(update={"materials": MaterialObservations()})
    bundle = build_bundle(empty, load_household().materials)
    assert "No uptake series" in " ".join(bundle.diagnostics)


def load_household() -> SimulationRequest:
    """Load the household example.

    Returns:
        SimulationRequest: Validated request.
    """
    return load_request(EXAMPLES / "household_basic.yaml")
