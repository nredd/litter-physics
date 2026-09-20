//! Versioned research fixtures, conservative recordings and restart state.
//!
//! These are unvalidated single-phase numerical fixtures, not a coupled wet-litter
//! model. Porous fines, absorption, fragmentation and adhesion remain unsupported.
//! References: <https://doi.org/10.1145/3197517.3201293> and `docs/research.md`.

use std::time::Instant;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::household::request::Request;
use crate::mpm::constitutive::Material;
use crate::mpm::fixtures::{Fixture, FixtureSpec, PelletSpec, ResourceLimits};
use crate::mpm::grid::GridLayout;
use crate::mpm::particles::ParticleSet;
use crate::mpm::rigid::{ContactParams, Pellet};
use crate::mpm::solver::{Ledger, Simulation, SolverConfig};

const MAX_PARTICLES: usize = 100_000;
const MAX_NODES: usize = 1_000_000;
const MAX_FRAME_POINTS: f64 = 2_000_000.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    fixture: String,
    grid_spacing_m: f64,
    domain_m: [f64; 3],
    material: String,
    density_kg_m3: f64,
    young_modulus_pa: f64,
    poisson_ratio: f64,
    yield_stress_pa: f64,
    consistency_pa_s_n: f64,
    flow_index: f64,
    initial_size_m: [f64; 3],
    initial_velocity_m_s: [f64; 3],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    particles: ParticleSet,
    pellet: Option<Pellet>,
    ledger: Ledger,
    step_count: u64,
    limited_steps: u64,
    rejected_steps: u64,
    dt_scale: f64,
    min_dt_used: Option<f64>,
    max_dt_used: f64,
    next_record_index: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema_version: u32,
    mode: String,
    request: Value,
    time_s: f64,
    state: Snapshot,
}

/// Execute a supported research fixture or continue a complete numerical checkpoint.
///
/// # Errors
///
/// Rejects unsupported physics, invalid/oversized inputs, corrupt checkpoints or
/// numerical failure. Budget expiry returns an explicitly incomplete result.
pub fn run(request: &Value, resume: Option<&Value>) -> Result<Value, String> {
    let started = Instant::now();
    let typed: Request = serde_json::from_value(request.clone()).map_err(|e| e.to_string())?;
    let spec = specification(&typed)?;
    let (mut sim, _) = spec.build()?;
    let mut next_record = 1;
    let mut frames = Vec::new();
    let mut metrics = Vec::new();
    if let Some(value) = resume {
        next_record = restore(&mut sim, &spec, &typed, value)?;
    } else {
        record(&sim, &spec, &mut frames, &mut metrics);
    }
    let mut completed = sim.time >= typed.duration_s;
    while !completed {
        let target = (f64::from(next_record) * typed.record_interval_s).min(typed.duration_s);
        let reached = sim
            .advance_while(target, || {
                started.elapsed().as_secs_f64() < typed.max_wall_time_s
            })
            .map_err(|e| e.to_string())?;
        if !reached {
            break;
        }
        check_closed_state(&sim)?;
        record(&sim, &spec, &mut frames, &mut metrics);
        next_record += 1;
        completed = sim.time >= typed.duration_s;
    }
    // Budget stops may occur between record boundaries; validate that state too.
    check_closed_state(&sim)?;
    let checkpoint = Checkpoint {
        schema_version: 1,
        mode: "research".into(),
        request: request.clone(),
        time_s: sim.time,
        state: snapshot(&sim, next_record),
    };
    let observables = spec.observables(&sim);
    if observables.values().any(|value| !value.is_finite()) {
        return Err("research observables became nonfinite".into());
    }
    let mut output = json!({
        "schema_version": 1, "mode": "research",
        "status": if completed { "completed" } else { "budget_exhausted" },
        "fidelity": "research_unvalidated", "time_s": sim.time,
        "frames": frames, "metrics": metrics, "observables": observables,
        "checkpoint": checkpoint,
        "diagnostics": [
            "Unvalidated numerical fixture; no measured surrogate or household calibration",
            "Single-phase paste mass is booked as waste; no mobile/bound-water split",
            "No porous fines, wet fragmentation, absorption, adhesion or response-table generation",
            "Dense padded grid with active-node updates; no sparse-block storage",
            "Grid-level coupling conserves impulses; leakage and full coupled convergence remain unverified",
            "Wall lattice completion is an approximate contact treatment; its normal grid work is unledgered and energy closure remains open"
        ]
    });
    if started.elapsed().as_secs_f64() >= typed.max_wall_time_s {
        output["status"] = json!("budget_exhausted");
    }
    Ok(output)
}

/// Check the shared envelope before any research allocation.
fn validate_envelope(request: &Request) -> Result<(), String> {
    if request.schema_version != 1 || request.mode != "research" || !request.events.is_empty() {
        return Err(
            "research requires schema_version=1, mode=research and no household events".into(),
        );
    }
    for value in [
        request.duration_s,
        request.dt_s,
        request.record_interval_s,
        request.max_wall_time_s,
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err("research times and budgets must be finite and positive".into());
        }
    }
    if request.dt_s > request.duration_s || request.max_wall_time_s > 3600.0 {
        return Err("research dt exceeds duration or budget exceeds 3600 seconds".into());
    }
    if request.duration_s / request.record_interval_s > 10_000.0
        || request.duration_s / request.dt_s > 10_000_000.0
    {
        return Err("research recording or nominal-step resource cap exceeded".into());
    }
    if request.boxes.len() > 1 {
        return Err("research fixtures support at most one box supplying pellet geometry".into());
    }
    for spec in &request.boxes {
        spec.validate()?;
    }
    request.materials.validate()
}

/// Convert a validated wire request into an independently bounded fixture.
fn specification(request: &Request) -> Result<FixtureSpec, String> {
    validate_envelope(request)?;
    let params: Parameters = serde_json::from_value(
        request
            .research
            .clone()
            .ok_or("research parameters are required")?,
    )
    .map_err(|e| e.to_string())?;
    validate_parameters(&params)?;
    let fixture = match params.fixture.as_str() {
        "slump" => Fixture::Slump,
        "hydrostatic" => Fixture::Hydrostatic,
        "dam_break" => Fixture::DamBreak,
        "coupled_patch" => Fixture::CoupledPatch,
        other => return Err(format!("unsupported research fixture '{other}'")),
    };
    if fixture != Fixture::CoupledPatch && !request.boxes.is_empty() {
        return Err("only coupled_patch accepts a pellet geometry box".into());
    }
    let material = material(&params)?;
    let spec = FixtureSpec {
        fixture,
        material,
        grid_spacing: params.grid_spacing_m,
        domain: params.domain_m,
        initial_size: params.initial_size_m,
        initial_velocity: Vector3::from(params.initial_velocity_m_s),
        seed: request.seed,
        pellet: request.boxes.first().map(|box_spec| PelletSpec {
            radius: box_spec.pellet_radius_m,
            length: box_spec.pellet_length_m,
            density: box_spec.pellet_density_kg_m3,
        }),
        config: SolverConfig {
            gravity: Vector3::new(0.0, 0.0, -9.81),
            wall_friction: request.materials.friction,
            cfl: 0.3,
            max_dt: request.dt_s,
            min_dt: 1e-12,
            contact: ContactParams {
                normal_stiffness: request.materials.normal_stiffness_n_m,
                restitution: request.materials.restitution,
                friction: request.materials.friction,
            },
        },
        limits: ResourceLimits {
            max_particles: MAX_PARTICLES,
            max_nodes: MAX_NODES,
        },
    };
    GridLayout::new(spec.domain, spec.grid_spacing, MAX_NODES)?;
    spec.validate()?;
    let frame_count = (request.duration_s / request.record_interval_s).ceil() + 1.0;
    let count = u32::try_from(spec.particle_count()).map_err(|e| e.to_string())?;
    if frame_count * f64::from(count.saturating_add(64)) > MAX_FRAME_POINTS {
        return Err(
            "research frame-point resource cap exceeded; reduce recording frequency".into(),
        );
    }
    Ok(spec)
}

/// Reject nonfinite and physically invalid material, velocity and geometry inputs.
fn validate_parameters(params: &Parameters) -> Result<(), String> {
    let positive = [
        params.grid_spacing_m,
        params.density_kg_m3,
        params.young_modulus_pa,
        params.flow_index,
    ];
    if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || params
            .domain_m
            .iter()
            .chain(&params.initial_size_m)
            .any(|v| !v.is_finite() || *v <= 0.0)
        || params.initial_velocity_m_s.iter().any(|v| !v.is_finite())
    {
        return Err(
            "research geometry, density, stiffness and flow index must be finite and positive"
                .into(),
        );
    }
    if !params.poisson_ratio.is_finite()
        || !(0.0..0.5).contains(&params.poisson_ratio)
        || !params.yield_stress_pa.is_finite()
        || params.yield_stress_pa < 0.0
        || !params.consistency_pa_s_n.is_finite()
        || params.consistency_pa_s_n < 0.0
    {
        return Err("invalid Poisson ratio, yield stress or consistency".into());
    }
    for (extent, domain) in params.initial_size_m.iter().zip(params.domain_m) {
        let ratio = extent / (0.5 * params.grid_spacing_m);
        if *extent > domain || !ratio.is_finite() || (ratio - ratio.round()).abs() > 1e-6 {
            return Err("initial block must fit domain and span whole particle subcells".into());
        }
    }
    Ok(())
}

/// Interpret input rheology in measured shear units, not equivalent-stress units.
fn material(params: &Parameters) -> Result<Material, String> {
    let result = match params.material.as_str() {
        "paste" => Material::paste_from_shear_rheology(
            params.density_kg_m3,
            params.young_modulus_pa,
            params.poisson_ratio,
            params.yield_stress_pa,
            params.consistency_pa_s_n,
            params.flow_index,
        ),
        "water" if params.yield_stress_pa == 0.0 && (params.flow_index - 1.0).abs() < 1e-12 => {
            Material::liquid(
                params.density_kg_m3,
                params.young_modulus_pa,
                params.poisson_ratio,
                params.consistency_pa_s_n,
            )
        }
        "water" => return Err("water requires zero yield stress and flow_index=1".into()),
        "fines" => return Err("porous Drucker-Prager fines are not implemented".into()),
        other => return Err(format!("unsupported research material '{other}'")),
    };
    if !result.wave_speed().is_finite()
        || result.wave_speed() <= 0.0
        || !result.consistency.is_finite()
        || !result.yield_stress.is_finite()
        || !result.diffusivity().is_finite()
    {
        return Err("derived material properties are nonfinite".into());
    }
    Ok(result)
}

/// Snapshot every non-scratch state variable, preserving adaptive-step history.
fn snapshot(sim: &Simulation, next_record_index: u32) -> Snapshot {
    Snapshot {
        particles: sim.particles.clone(),
        pellet: sim.pellet.clone(),
        ledger: sim.ledger,
        step_count: sim.step_count,
        limited_steps: sim.limited_steps,
        rejected_steps: sim.rejected_steps,
        dt_scale: sim.dt_scale,
        min_dt_used: sim.min_dt_used.is_finite().then_some(sim.min_dt_used),
        max_dt_used: sim.max_dt_used,
        next_record_index,
    }
}

/// Restore independently checked state into a freshly rebuilt compatible grid.
fn restore(
    sim: &mut Simulation,
    spec: &FixtureSpec,
    request: &Request,
    value: &Value,
) -> Result<u32, String> {
    let cp: Checkpoint = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if cp.schema_version != 1
        || cp.mode != "research"
        || !cp.time_s.is_finite()
        || cp.time_s < 0.0
        || cp.time_s > request.duration_s
    {
        return Err("invalid research checkpoint version, mode or time".into());
    }
    let mut original: Request = serde_json::from_value(cp.request).map_err(|e| e.to_string())?;
    original.max_wall_time_s = request.max_wall_time_s;
    if original != *request {
        return Err("research checkpoint request differs from resume request".into());
    }
    let state = cp.state;
    state.particles.validate()?;
    if state.particles.len() > spec.particle_count()
        || state
            .particles
            .id
            .iter()
            .any(|id| usize::try_from(*id).map_or(true, |id| id >= spec.particle_count()))
    {
        return Err("research checkpoint particle identities are incompatible".into());
    }
    validate_snapshot(&state, cp.time_s, request, sim)?;
    sim.particles = state.particles;
    sim.pellet = state.pellet;
    sim.ledger = state.ledger;
    sim.time = cp.time_s;
    sim.step_count = state.step_count;
    sim.limited_steps = state.limited_steps;
    sim.rejected_steps = state.rejected_steps;
    sim.dt_scale = state.dt_scale;
    sim.min_dt_used = state.min_dt_used.unwrap_or(f64::INFINITY);
    sim.max_dt_used = state.max_dt_used;
    check_closed_state(sim)?;
    Ok(state.next_record_index)
}

/// Validate restart histories, physical invariants and geometry before assignment.
fn validate_snapshot(
    state: &Snapshot,
    time: f64,
    request: &Request,
    initial: &Simulation,
) -> Result<(), String> {
    if !state.dt_scale.is_finite()
        || !(0.0..=1.0).contains(&state.dt_scale)
        || state.dt_scale == 0.0
        || !state.max_dt_used.is_finite()
        || state.max_dt_used < 0.0
        || state
            .min_dt_used
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > state.max_dt_used)
        || state.next_record_index == 0
        || state.next_record_index > 10_002
        || state.step_count > 1_000_000_000_000
        || state.limited_steps > state.step_count
        || state.rejected_steps > 1_000_000_000_000
    {
        return Err("invalid research checkpoint integration history".into());
    }
    let next_time = f64::from(state.next_record_index) * request.record_interval_s;
    let previous = f64::from(state.next_record_index - 1) * request.record_interval_s;
    if time < request.duration_s && (time < previous - 1e-12 || time >= next_time) {
        return Err("research checkpoint recording schedule is inconsistent".into());
    }
    if state
        .particles
        .deformation
        .iter()
        .any(|f| !f.determinant().is_finite() || f.determinant() <= 0.0)
    {
        return Err("research checkpoint contains inverted deformation".into());
    }
    let ledger_value = serde_json::to_value(state.ledger).map_err(|e| e.to_string())?;
    if contains_null(&ledger_value)
        || state.ledger.outflow_mass < 0.0
        || (state.ledger.initial_mass - initial.ledger.initial_mass).abs() > 1e-12
        || state.ledger.initial_mechanical_energy.to_bits()
            != initial.ledger.initial_mechanical_energy.to_bits()
        || (state.ledger.initial_momentum - initial.ledger.initial_momentum).norm() > 1e-12
    {
        return Err("research checkpoint initial inventory or ledger is inconsistent".into());
    }
    match (&state.pellet, &initial.pellet) {
        (None, None) => {}
        (Some(pellet), Some(reference))
            if pellet.is_finite()
                && pellet.mass.to_bits() == reference.mass.to_bits()
                && pellet.radius.to_bits() == reference.radius.to_bits()
                && pellet.length.to_bits() == reference.length.to_bits()
                && pellet.inertia_body == reference.inertia_body
                && pellet.sphere_offsets == reference.sphere_offsets
                && (pellet.orientation.norm() - 1.0).abs() < 1e-10 => {}
        _ => return Err("research checkpoint pellet geometry is incompatible".into()),
    }
    Ok(())
}

/// Detect nonfinite numbers converted by JSON serialization to null.
fn contains_null(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Array(values) => values.iter().any(contains_null),
        Value::Object(values) => values.values().any(contains_null),
        _ => false,
    }
}

/// Check representable state and closed-fixture inventory, including on restart.
fn check_closed_state(sim: &Simulation) -> Result<(), String> {
    sim.validate().map_err(|e| e.to_string())?;
    if !sim.ledger.outflow_mass.is_finite() || sim.ledger.outflow_mass.abs() > 0.0 {
        return Err(format!(
            "closed research fixture leaked '{}' kg beyond the padded grid",
            sim.ledger.outflow_mass
        ));
    }
    let residual = sim.mass_residual();
    if !residual.is_finite() || residual.abs() > (1e-6 * sim.ledger.initial_mass).max(1e-12) {
        return Err(format!(
            "research mass conservation failed: '{residual}' kg"
        ));
    }
    Ok(())
}

/// Record material points and pellet sphere proxies with explicit compartment metrics.
fn record(sim: &Simulation, spec: &FixtureSpec, frames: &mut Vec<Value>, metrics: &mut Vec<Value>) {
    let water = spec.material.kind == crate::mpm::constitutive::MaterialKind::Liquid;
    let label = if water { "water" } else { "waste" };
    let mut positions: Vec<[f64; 3]> = sim.particles.position.iter().map(|v| (*v).into()).collect();
    let mut radii = vec![spec.grid_spacing / 4.0; positions.len()];
    let mut materials = vec![label; positions.len()];
    if let Some(pellet) = &sim.pellet {
        for point in pellet.sphere_centers() {
            positions.push(point.into());
            radii.push(pellet.radius);
            materials.push("wood");
        }
    }
    frames.push(json!({
        "time_s": sim.time, "positions_m": positions, "radii_m": radii,
        "materials": materials, "box_ids": vec!["research"; positions.len()]
    }));
    for (compartment, mass) in [
        ("domain", sim.particles.total_mass()),
        ("outflow", sim.ledger.outflow_mass),
    ] {
        metrics.push(json!({
            "time_s": sim.time, "compartment": compartment,
            "wood_kg": if compartment == "domain" { sim.pellet.as_ref().map_or(0.0, |p| p.mass) } else { 0.0 },
            "water_kg": if water { mass } else { 0.0 },
            "waste_kg": if water { 0.0 } else { mass }
        }));
    }
}

#[cfg(test)]
mod tests;
