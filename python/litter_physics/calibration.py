"""Calibration bundles and reports from observations.

Only identifiable parameters are fitted. Uptake series give capacity and rate;
breakup trials give a breakdown rate; a multi-rate flow curve gives Herschel-Bulkley
parameters. A slump test alone does NOT identify yield stress, consistency, and flow
index together, so slump-only inputs leave rheology `assumed` with an explicit
diagnostic. Synthetic observations produce a bundle whose every parameter is labelled
`assumed`; no household accuracy claim can come from them.

References:
    docs/plan.md (Cats and calibration; Acceptance)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize, stats

from litter_physics.artifacts import write_atomic_json
from litter_physics.models import (
    MaterialsCfg,
    SimulationRequest,
    canonical_json,
    load_document,
    validate_payload,
)
from litter_physics.observations import (
    MaintenanceKind,
    ObservationSet,
    Provenance,
    VisitObservation,
)

LOGGER = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION = 1
MIN_VISITS_STRONG = 20
MIN_VISITS_WEAK = 5
HOLDOUT_FRACTION = 0.3
HOUSEHOLD_MASS_TOLERANCE = 0.20
NEAR_ZERO_KG = 1e-4


class Parameter(BaseModel):
    """One calibrated or assumed parameter with full provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(description="Parameter value in `unit`.")
    unit: str = Field(description="SI unit string.")
    uncertainty: float | None = Field(description="One-sigma uncertainty when estimated.")
    status: Provenance = Field(description="`measured`, `literature_prior`, or `assumed`.")
    provenance: str = Field(description="How the value was obtained.")
    note: str | None = Field(default=None, description="Identifiability or caveat note.")


class PersonalizationScore(BaseModel):
    """Held-out comparison of a per-cat model against the generic baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cat_id: str = Field(description="Cat identifier.")
    train_visits: int = Field(description="Visits used to fit the per-cat model.")
    holdout_visits: int = Field(description="Contiguous held-out visits.")
    per_cat_log_likelihood: float | None = Field(description="Mean held-out log-likelihood.")
    generic_log_likelihood: float | None = Field(description="Generic baseline value.")
    supported: bool = Field(
        description="Measured per-cat evidence beats generic with at least 20 training visits."
    )
    evidence: str = Field(description="`strong`, `weak`, `insufficient`, or `synthetic`.")


class HouseholdMassScore(BaseModel):
    """Comparison of simulated removed mass against weighed maintenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measured_removed_kg: float = Field(description="Sum of weighed removed masses.")
    simulated_removed_kg: float = Field(description="Final `removed` ledger mass.")
    relative_error: float | None = Field(description="Relative error, None when measured ~0.")
    absolute_error_kg: float = Field(description="Absolute error in kilograms.")
    tolerance_kg: float = Field(description="Accepted absolute band.")
    within_tolerance: bool = Field(description="Whether the error is inside the band.")
    claim: str = Field(description="What this comparison does and does not establish.")


class CalibrationBundle(BaseModel):
    """Versioned calibration output consumed by runs and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(description="Bundle schema version.")
    synthetic: bool = Field(description="True when derived from synthetic observations.")
    observations_sha256: str = Field(description="Hash of the observation set.")
    parameters: dict[str, Parameter] = Field(description="Household material parameters.")
    rheology: dict[str, Parameter] = Field(description="Herschel-Bulkley parameters.")
    personalization: list[PersonalizationScore] = Field(description="Per-cat held-out scores.")
    diagnostics: list[str] = Field(description="Identifiability and coverage notes.")
    validation_claim: str = Field(description="Explicit statement of what is established.")

    @property
    def sha256(self) -> str:
        """Hash the bundle for provenance.

        Returns:
            str: Hex digest of canonical JSON.
        """
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


def _status_for(observations: ObservationSet, measured: Provenance) -> Provenance:
    """Downgrade any status to `assumed` for synthetic observations.

    Parameters:
        observations (ObservationSet): Source observations.
        measured (Provenance): Status declared by the group.

    Returns:
        Provenance: `assumed` when synthetic, otherwise the group's status.
    """
    return Provenance.ASSUMED if observations.synthetic else measured


def _uptake_model(t: np.ndarray, capacity: float, rate: float) -> np.ndarray:
    """First-order uptake curve.

    Parameters:
        t (np.ndarray): Times in seconds.
        capacity (float): Saturation water ratio.
        rate (float): Rate constant in 1/s.

    Returns:
        np.ndarray: Water ratio at each time.
    """
    return capacity * (1.0 - np.exp(-rate * t))


def _breakup_model(t: np.ndarray, rate: float) -> np.ndarray:
    """First-order breakup fraction.

    Parameters:
        t (np.ndarray): Exposure times in seconds.
        rate (float): Rate constant in 1/s.

    Returns:
        np.ndarray: Broken mass fraction.
    """
    return 1.0 - np.exp(-rate * t)


def _herschel_bulkley(rate: np.ndarray, yield_stress: float, k: float, n: float) -> np.ndarray:
    """Herschel-Bulkley flow curve.

    Parameters:
        rate (np.ndarray): Shear rates in 1/s.
        yield_stress (float): Yield stress in Pa.
        k (float): Consistency in Pa s^n.
        n (float): Flow index.

    Returns:
        np.ndarray: Shear stress in Pa.
    """
    return yield_stress + k * np.power(rate, n)


def fit_uptake(observations: ObservationSet) -> tuple[Parameter, Parameter] | None:
    """Fit capacity and uptake rate from an uptake series.

    Parameters:
        observations (ObservationSet): Observations with an optional uptake group.

    Returns:
        tuple[Parameter, Parameter] | None: Capacity and rate, or None without data.
    """
    group = observations.materials.uptake
    if group is None:
        return None
    t = np.array([sample.time_s for sample in group.samples])
    w = np.array([sample.water_mass_ratio for sample in group.samples])
    guess = (max(float(w.max()), 1e-3), 1.0 / max(float(t.max()), 1e-3))
    try:
        popt, pcov = optimize.curve_fit(
            _uptake_model, t, w, p0=guess, bounds=([1e-6, 1e-9], [np.inf, np.inf])
        )
    except (RuntimeError, ValueError) as e:
        LOGGER.warning(f"uptake fit failed: {e}")
        return None
    sigma = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    status = _status_for(observations, group.status)
    provenance = f"curve_fit of first-order uptake to {len(t)} points from '{group.source}'"
    capacity = Parameter(
        value=float(popt[0]),
        unit="kg/kg",
        uncertainty=float(max(sigma[0], group.instrument_uncertainty)),
        status=status,
        provenance=provenance,
    )
    rate = Parameter(
        value=float(popt[1]),
        unit="1/s",
        uncertainty=float(sigma[1]) if math.isfinite(sigma[1]) else None,
        status=status,
        provenance=provenance,
    )
    return capacity, rate


def fit_breakup(observations: ObservationSet) -> Parameter | None:
    """Fit the breakdown rate from breakup trials.

    Parameters:
        observations (ObservationSet): Observations with an optional breakup group.

    Returns:
        Parameter | None: Rate, or None without data.
    """
    group = observations.materials.breakup
    if group is None:
        return None
    t = np.array([sample.exposure_s for sample in group.samples])
    f = np.array([sample.broken_fraction for sample in group.samples])
    try:
        popt, pcov = optimize.curve_fit(
            _breakup_model, t, f, p0=(1.0 / max(float(t.max()), 1e-3),), bounds=([1e-9], [np.inf])
        )
    except (RuntimeError, ValueError) as e:
        LOGGER.warning(f"breakup fit failed: {e}")
        return None
    sigma = float(np.sqrt(max(float(pcov[0][0]), 0.0)))
    return Parameter(
        value=float(popt[0]),
        unit="1/s",
        uncertainty=sigma if math.isfinite(sigma) else None,
        status=_status_for(observations, group.status),
        provenance=f"curve_fit of first-order breakup to {len(t)} trials from '{group.source}'",
        note="Fitted at the trials' moisture and agitation only; not a general law.",
    )


def friction_prior(observations: ObservationSet) -> Parameter | None:
    """Derive a friction prior from the repose angle.

    Parameters:
        observations (ObservationSet): Observations with an optional dry pour group.

    Returns:
        Parameter | None: Friction prior, or None without data.
    """
    group = observations.materials.dry_pour
    if group is None:
        return None
    angles = np.array([sample.repose_angle_deg for sample in group.samples])
    mu = float(np.tan(np.radians(angles.mean())))
    spread = float(np.tan(np.radians(angles.max())) - np.tan(np.radians(angles.min())))
    return Parameter(
        value=mu,
        unit="1",
        uncertainty=max(spread / 2.0, 0.02),
        status=Provenance.ASSUMED,
        provenance=f"tan(mean repose angle) over {len(angles)} pours from '{group.source}'",
        note=(
            "Repose angle constrains friction plus rolling resistance jointly; tan(angle) "
            "is a prior, not an identification. Confirm against DEM repose tests."
        ),
    )


def fit_rheology(observations: ObservationSet) -> tuple[dict[str, Parameter], list[str]]:
    """Fit Herschel-Bulkley parameters when a multi-rate flow curve exists.

    Parameters:
        observations (ObservationSet): Observations with optional flow and slump groups.

    Returns:
        tuple[dict[str, Parameter], list[str]]: Parameters and diagnostics. Slump-only
            inputs return no parameters and an explicit identifiability diagnostic.
    """
    diagnostics: list[str] = []
    flow = observations.materials.flow_curve
    slump = observations.materials.slump
    if flow is None:
        if slump is not None:
            diagnostics.append(
                "Slump tests present but no multi-rate flow curve: yield stress, consistency "
                "and flow index are NOT identifiable from slump alone; rheology stays assumed."
            )
        else:
            diagnostics.append("No rheology measurements; rheology stays assumed.")
        return {}, diagnostics
    rate = np.array([sample.shear_rate_1_s for sample in flow.samples])
    stress = np.array([sample.shear_stress_pa for sample in flow.samples])
    guess = (max(float(stress.min()) * 0.5, 1e-3), 1.0, 0.5)
    try:
        popt, pcov = optimize.curve_fit(
            _herschel_bulkley,
            rate,
            stress,
            p0=guess,
            bounds=([0.0, 1e-9, 0.05], [np.inf, np.inf, 2.0]),
        )
    except (RuntimeError, ValueError) as e:
        diagnostics.append(f"Herschel-Bulkley fit failed: {e}")
        return {}, diagnostics
    sigma = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    status = _status_for(observations, flow.status)
    provenance = f"Herschel-Bulkley curve_fit to {len(rate)} rates from '{flow.source}'"
    residual = float(np.sqrt(np.mean((_herschel_bulkley(rate, *popt) - stress) ** 2)))
    diagnostics.append(f"Herschel-Bulkley fit RMS residual {residual:.3g} Pa")
    units = {"yield_stress_pa": "Pa", "consistency_pa_s_n": "Pa s^n", "flow_index": "1"}
    parameters = {
        name: Parameter(
            value=float(value),
            unit=unit,
            uncertainty=float(unc) if math.isfinite(unc) else None,
            status=status,
            provenance=provenance,
            note="Steady shear only; elastic stiffness and moisture dependence unmeasured.",
        )
        for (name, unit), value, unc in zip(units.items(), popt, sigma, strict=True)
    }
    if slump is not None:
        diagnostics.append(
            "Slump tests are available as an independent check of the fitted rheology; "
            "they were not used in the fit."
        )
    return parameters, diagnostics


def _lognormal_fit(values: np.ndarray) -> tuple[float, float]:
    """Fit a lognormal by log-moments with a floor on sigma.

    Parameters:
        values (np.ndarray): Positive samples.

    Returns:
        tuple[float, float]: Log-mean and log-sigma.
    """
    logs = np.log(np.clip(values, 1e-6, None))
    return float(logs.mean()), float(max(logs.std(ddof=1) if logs.size > 1 else 0.5, 0.05))


def _visit_features(visits: list[VisitObservation]) -> np.ndarray:
    """Extract the scored feature: total in-box duration.

    Parameters:
        visits (list[VisitObservation]): Visits.

    Returns:
        np.ndarray: Durations in seconds.
    """
    return np.array([max(visit.duration_s, 1e-3) for visit in visits])


def score_personalization(observations: ObservationSet) -> list[PersonalizationScore]:
    """Score per-cat visit-duration models against a pooled baseline on held-out visits.

    Parameters:
        observations (ObservationSet): Observations with visits.

    Returns:
        list[PersonalizationScore]: One score per cat.
    """
    scores: list[PersonalizationScore] = []
    by_cat = {
        cat_id: [visit for visit in observations.visits if visit.cat_id == cat_id]
        for cat_id in observations.cat_ids
    }
    for cat_id, visits in by_cat.items():
        n = len(visits)
        holdout_n = int(math.floor(n * HOLDOUT_FRACTION))
        train_n = n - holdout_n
        if train_n < MIN_VISITS_WEAK or holdout_n < 1:
            scores.append(
                PersonalizationScore(
                    cat_id=cat_id,
                    train_visits=train_n,
                    holdout_visits=holdout_n,
                    per_cat_log_likelihood=None,
                    generic_log_likelihood=None,
                    supported=False,
                    evidence="insufficient",
                )
            )
            continue
        train = _visit_features(visits[:train_n])
        holdout = _visit_features(visits[train_n:])
        generic_pool = [
            visit
            for other_id, other in by_cat.items()
            for visit in (other[: len(other) - int(math.floor(len(other) * HOLDOUT_FRACTION))])
        ]
        generic = _visit_features(generic_pool)
        mu_cat, sigma_cat = _lognormal_fit(train)
        mu_gen, sigma_gen = _lognormal_fit(generic)
        ll_cat = float(stats.lognorm.logpdf(holdout, s=sigma_cat, scale=math.exp(mu_cat)).mean())
        ll_gen = float(stats.lognorm.logpdf(holdout, s=sigma_gen, scale=math.exp(mu_gen)).mean())
        strong = train_n >= MIN_VISITS_STRONG
        scores.append(
            PersonalizationScore(
                cat_id=cat_id,
                train_visits=train_n,
                holdout_visits=holdout_n,
                per_cat_log_likelihood=ll_cat,
                generic_log_likelihood=ll_gen,
                supported=not observations.synthetic and strong and ll_cat > ll_gen,
                evidence="synthetic"
                if observations.synthetic
                else ("strong" if strong else "weak"),
            )
        )
    return scores


def build_bundle(observations: ObservationSet, defaults: MaterialsCfg) -> CalibrationBundle:
    """Build a calibration bundle, fitting only what the observations identify.

    Parameters:
        observations (ObservationSet): Validated observations.
        defaults (MaterialsCfg): Fallback material values marked `assumed`.

    Returns:
        CalibrationBundle: The bundle with diagnostics and an explicit claim.
    """
    diagnostics: list[str] = []
    parameters: dict[str, Parameter] = {}
    assumed = "default from the request template; no observation constrains it"
    for name, unit in (
        ("friction", "1"),
        ("restitution", "1"),
        ("normal_stiffness_n_m", "N/m"),
        ("water_capacity_ratio", "kg/kg"),
        ("uptake_rate_s", "1/s"),
        ("breakdown_rate_s", "1/s"),
        ("evaporation_rate_s", "1/s"),
    ):
        parameters[name] = Parameter(
            value=float(getattr(defaults, name)),
            unit=unit,
            uncertainty=None,
            status=Provenance.ASSUMED,
            provenance=assumed,
        )
    uptake = fit_uptake(observations)
    if uptake is not None:
        parameters["water_capacity_ratio"], parameters["uptake_rate_s"] = uptake
    else:
        diagnostics.append("No uptake series; capacity and uptake rate stay assumed.")
    breakup = fit_breakup(observations)
    if breakup is not None:
        parameters["breakdown_rate_s"] = breakup
    else:
        diagnostics.append("No breakup trials; breakdown rate stays assumed.")
    friction = friction_prior(observations)
    if friction is not None:
        parameters["friction"] = friction
    diagnostics.append(
        "Restitution, stiffness and evaporation rate have no measurement protocol in the "
        "observation schema and remain assumed."
    )
    rheology, rheology_notes = fit_rheology(observations)
    diagnostics.extend(rheology_notes)
    personalization = score_personalization(observations)
    if observations.synthetic:
        diagnostics.append(
            "Observations are SYNTHETIC: every parameter is labelled assumed regardless of "
            "fit quality, and no household or material validation claim is made."
        )
        claim = "none: synthetic observations exercise the pipeline only"
    else:
        claim = (
            "material parameters fitted from the listed measurements; household accuracy "
            "requires a completed run compared against weighed maintenance (see "
            "`score_household_masses`)"
        )
    return CalibrationBundle(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        synthetic=observations.synthetic,
        observations_sha256=observations.sha256,
        parameters=parameters,
        rheology=rheology,
        personalization=personalization,
        diagnostics=diagnostics,
        validation_claim=claim,
    )


def apply_bundle(request: SimulationRequest, bundle: CalibrationBundle) -> SimulationRequest:
    """Copy a request with material parameters replaced by bundle values.

    Parameters:
        request (SimulationRequest): Template request.
        bundle (CalibrationBundle): Calibration bundle.

    Returns:
        SimulationRequest: Validated copy with updated `materials`.
    """
    materials = request.materials.model_dump() | {
        name: parameter.value for name, parameter in bundle.parameters.items()
    }
    payload = request.model_dump(mode="json") | {"materials": materials}
    return validate_payload(SimulationRequest, payload)


def score_household_masses(
    metrics: pa.Table, observations: ObservationSet
) -> HouseholdMassScore | None:
    """Compare the final `removed` ledger mass with weighed maintenance removals.

    Parameters:
        metrics (pa.Table): Ledger rows across a run's segments.
        observations (ObservationSet): Observations with maintenance logs.

    Returns:
        HouseholdMassScore | None: The comparison, or None without maintenance data.
    """
    removals = [
        item
        for item in observations.maintenance
        if item.kind
        in {MaintenanceKind.SCOOP, MaintenanceKind.EMPTY_DRAWER, MaintenanceKind.CLEAN}
    ]
    if not removals or metrics.num_rows == 0:
        return None
    measured = float(sum(item.removed_mass_kg for item in removals))
    removed_rows = metrics.filter(pc.field("compartment") == "removed")
    if removed_rows.num_rows == 0:
        return None
    last = removed_rows.slice(removed_rows.num_rows - 1, 1).to_pylist()[0]
    simulated = float(last["wood_kg"] + last["waste_kg"] + last["water_kg"])
    absolute = abs(simulated - measured)
    repeatability = float(np.std([item.removed_mass_kg for item in removals], ddof=0))
    tolerance = max(HOUSEHOLD_MASS_TOLERANCE * measured, repeatability, NEAR_ZERO_KG)
    relative = absolute / measured if measured > NEAR_ZERO_KG else None
    claim = (
        "synthetic observations: comparison exercises the scoring path only"
        if observations.synthetic
        else "single held-out aggregate; not a per-visit validation"
    )
    return HouseholdMassScore(
        measured_removed_kg=measured,
        simulated_removed_kg=simulated,
        relative_error=relative,
        absolute_error_kg=absolute,
        tolerance_kg=tolerance,
        within_tolerance=absolute <= tolerance,
        claim=claim,
    )


def write_bundle(bundle: CalibrationBundle, path: Path) -> None:
    """Write a bundle as JSON.

    Parameters:
        bundle (CalibrationBundle): Bundle to write.
        path (Path): Destination.
    """
    write_atomic_json(Path(str(path)).resolve(), bundle.model_dump(mode="json"))


def load_bundle(path: Path) -> CalibrationBundle:
    """Load and validate a bundle.

    Parameters:
        path (Path): YAML or JSON bundle file.

    Returns:
        CalibrationBundle: Validated bundle.
    """
    return validate_payload(CalibrationBundle, load_document(path))


def render_report(bundle: CalibrationBundle, mass_score: HouseholdMassScore | None) -> str:
    """Render a plain-text calibration report.

    Parameters:
        bundle (CalibrationBundle): Bundle to describe.
        mass_score (HouseholdMassScore | None): Optional household comparison.

    Returns:
        str: Multi-line report.
    """
    lines = [
        f"Calibration bundle (synthetic={bundle.synthetic}) sha256={bundle.sha256[:12]}",
        f"Validation claim: {bundle.validation_claim}",
        "Parameters:",
    ]
    for name, parameter in bundle.parameters.items():
        unc = "n/a" if parameter.uncertainty is None else f"{parameter.uncertainty:.3g}"
        lines.append(
            f"  {name} = {parameter.value:.6g} {parameter.unit} +/- {unc} [{parameter.status}]"
        )
    lines.append("Rheology:")
    if not bundle.rheology:
        lines.append("  (not identified)")
    for name, parameter in bundle.rheology.items():
        lines.append(f"  {name} = {parameter.value:.6g} {parameter.unit} [{parameter.status}]")
    lines.append("Personalization (held-out visit duration):")
    for score in bundle.personalization:
        lines.append(
            f"  {score.cat_id}: train={score.train_visits} holdout={score.holdout_visits} "
            f"evidence={score.evidence} supported={score.supported}"
        )
    if mass_score is not None:
        lines.append(
            f"Household removed mass: measured={mass_score.measured_removed_kg:.4g} kg "
            f"simulated={mass_score.simulated_removed_kg:.4g} kg "
            f"within_tolerance={mass_score.within_tolerance} ({mass_score.claim})"
        )
    lines.append("Diagnostics:")
    lines.extend(f"  - {line}" for line in bundle.diagnostics)
    return "\n".join(lines)


def bundle_payload(bundle: CalibrationBundle) -> dict[str, Any]:
    """Dump a bundle to a JSON-like tree.

    Parameters:
        bundle (CalibrationBundle): Bundle.

    Returns:
        dict[str, Any]: JSON-compatible mapping.
    """
    return bundle.model_dump(mode="json")
