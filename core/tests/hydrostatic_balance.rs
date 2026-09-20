//! Time-evolved hydrostatic balance of the liquid column against the domain walls.
//!
//! The fixture starts from the exact equation-of-state profile, so any error is
//! produced by the discrete dynamics. Every check here drives the real
//! [`Simulation::advance_to`] with the current Rust sources; nothing is mocked.
//!
//! References: `docs/hydrostatic-balance.md`; MLS-MPM force in
//! <https://doi.org/10.1145/3197517.3201293>.

use std::collections::BTreeMap;

use _core::mpm::constitutive::{Material, liquid_pressure};
use _core::mpm::fixtures::{Fixture, FixtureSpec, ResourceLimits};
use _core::mpm::rigid::ContactParams;
use _core::mpm::solver::{Simulation, SolverConfig};
use nalgebra::Vector3;

const GRAVITY: f64 = 9.81;

/// Water column spec matching the research request that reproduced the error.
fn water_column(domain: [f64; 3], height: f64, spacing: f64, max_dt: f64) -> FixtureSpec {
    FixtureSpec {
        fixture: Fixture::Hydrostatic,
        material: Material::liquid(1000.0, 1e5, 0.2, 1e-3),
        grid_spacing: spacing,
        domain,
        initial_size: [domain[0], domain[1], height],
        initial_velocity: Vector3::zeros(),
        seed: 7,
        pellet: None,
        config: SolverConfig {
            gravity: Vector3::new(0.0, 0.0, -GRAVITY),
            wall_friction: 0.4,
            cfl: 0.3,
            max_dt,
            min_dt: 1e-9,
            contact: ContactParams {
                normal_stiffness: 1000.0,
                restitution: 0.2,
                friction: 0.4,
            },
        },
        limits: ResourceLimits {
            max_particles: 100_000,
            max_nodes: 1_000_000,
        },
    }
}

/// Advance the native kernel and check conservation and fixed-step execution.
fn run(spec: &FixtureSpec, duration: f64) -> (Simulation, BTreeMap<String, f64>) {
    let (mut sim, _) = spec.build().expect("fixture builds");
    sim.advance_to(duration).expect("advance");
    assert_eq!(sim.rejected_steps, 0, "no adaptive rejection allowed");
    assert_eq!(sim.limited_steps, 0, "step cap must bind");
    let obs = spec.observables(&sim);
    assert!(obs["outflow_kg"].abs() < 1e-18);
    assert!(obs["mass_residual_kg"].abs() < 1e-12 * sim.ledger.initial_mass);
    assert!(obs["momentum_residual_kg_m_s"] < 1e-12);
    (sim, obs)
}

/// Signed relative pressure error of every particle plus its distance to the
/// nearest wall in cells (floor and the four side walls).
fn particle_errors(spec: &FixtureSpec, sim: &Simulation) -> Vec<(f64, f64)> {
    let h0 = spec.initial_size[2];
    let scale = spec.material.density * GRAVITY * h0;
    let dx = spec.grid_spacing;
    (0..sim.particles.len())
        .map(|p| {
            let x = sim.particles.position[p];
            let depth = (h0 - x.z).max(0.0);
            let reference = spec.hydrostatic_reference(depth);
            let measured = liquid_pressure(&spec.material, sim.particles.volume_ratio[p]);
            let wall_cells = [x.z, x.x, spec.domain[0] - x.x, x.y, spec.domain[1] - x.y]
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                / dx;
            ((measured - reference) / scale, wall_cells)
        })
        .collect()
}

/// Largest absolute value in the supplied diagnostic sample.
fn max_abs(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Grid accelerations in units of `g` after exactly one step from rest, keyed by
/// padded node coordinates. Wall nodes are excluded (their normal velocity is
/// projected, so they carry no acceleration signal).
fn first_step_accelerations(
    spec: &FixtureSpec,
    dt: f64,
) -> BTreeMap<(usize, usize, usize), Vector3<f64>> {
    let (mut sim, _) = spec.build().expect("fixture builds");
    sim.advance_to(dt).expect("one step");
    assert_eq!(sim.step_count, 1);
    let layout = sim.grid.layout;
    let mut out = BTreeMap::new();
    for &index in &sim.grid.active {
        let (i, j, k) = layout.coords(index);
        if (0..3).any(|axis| layout.wall_side(axis, [i, j, k][axis]).is_some()) {
            continue;
        }
        out.insert((i, j, k), sim.grid.momentum[index] / (dt * GRAVITY));
    }
    out
}

/// Cells from the nearest wall for a padded node coordinate along one axis.
fn cells_from_wall(layout_cells: usize, coord: usize) -> usize {
    (coord - 1).min(layout_cells + 1 - coord)
}

#[test]
fn reduced_column_pressure_error_after_forty_steps() {
    // 2x2 cell footprint: every particle is symmetric about the side-wall
    // nodes, so only the floor truncation acts. 160 particles, 40 steps.
    let spec = water_column([0.004, 0.004, 0.02], 0.01, 0.002, 5e-5);
    let (sim, obs) = run(&spec, 0.002);
    assert_eq!(sim.particles.len(), 160);
    assert_eq!(sim.step_count, 40);
    let max = obs["hydrostatic_max_relative_error"];
    let rms = obs["hydrostatic_rms_relative_error"];
    eprintln!("reduced 0.002 s: max={max} rms={rms}");
    assert!(
        max < 0.05,
        "max relative pressure error {max} >= 0.05 (rms {rms})"
    );
}

#[test]
fn original_column_pressure_error_after_ten_milliseconds() {
    let spec = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 5e-5);
    let (sim, obs) = run(&spec, 0.01);
    assert_eq!(sim.particles.len(), 4000);
    let max = obs["hydrostatic_max_relative_error"];
    let rms = obs["hydrostatic_rms_relative_error"];
    let errors = particle_errors(&spec, &sim);
    let interior = max_abs(errors.iter().filter(|e| e.1 > 1.0).map(|e| e.0));
    let layer = max_abs(errors.iter().filter(|e| e.1 <= 1.0).map(|e| e.0));
    eprintln!(
        "original 0.01 s: max={max} rms={rms} interior(>1 cell)={interior} wall layer={layer}"
    );
    assert!(
        max < 0.05,
        "max relative pressure error {max} >= 0.05 (rms {rms})"
    );
    // Strict invariant: the walls no longer leave a pressure layer. The floor
    // layer (corners included) stays within 1e-4 of the exact profile, and
    // side-wall particles more than two cells below the free surface within
    // 2e-3; before the wall traction term the same sets carried the 0.297
    // corner maximum. The free-surface error decays downward into the column.
    let h0 = spec.initial_size[2];
    let dx = spec.grid_spacing;
    let floor_layer = max_abs(
        errors
            .iter()
            .zip(&sim.particles.position)
            .filter(|(_, x)| x.z < dx)
            .map(|(e, _)| e.0),
    );
    assert!(floor_layer < 1e-4, "floor-layer error {floor_layer}");
    let submerged_wall_layer = max_abs(
        errors
            .iter()
            .zip(&sim.particles.position)
            .filter(|(e, x)| e.1 <= 1.0 && x.z < h0 - 2.0 * dx)
            .map(|(e, _)| e.0),
    );
    assert!(
        submerged_wall_layer < 2e-3,
        "submerged wall-layer error {submerged_wall_layer}"
    );
    // What remains is the free-surface truncation: the node above the
    // surface holds 1/32 of a particle mass under a one-sided stencil.
    let surface = max_abs(
        errors
            .iter()
            .zip(&sim.particles.position)
            .filter(|(_, x)| x.z >= h0 - spec.grid_spacing)
            .map(|(e, _)| e.0),
    );
    assert!(
        (surface - max).abs() < 1e-12,
        "surface {surface} vs max {max}"
    );
}

#[test]
fn paste_column_pressure_error_stays_small_after_forty_steps() {
    // Confined-compression paste column (the manuscript's resting column):
    // the traction extrapolates only the mirrored normal component, so the
    // balance is first-order rather than exact.
    let mut spec = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 4e-5);
    spec.material = Material::paste(1000.0, 1e5, 0.3, 1e4, 1.0, 1.0);
    let (sim, obs) = run(&spec, 0.002);
    let max = obs["hydrostatic_max_relative_error"];
    eprintln!(
        "paste 0.002 s: max={max} rms={}",
        obs["hydrostatic_rms_relative_error"]
    );
    assert!(max < 0.05, "paste max relative stress error {max}");
    assert!(sim.ledger.wall_impulse.z > 0.0);
}

#[test]
fn first_step_from_exact_profile_is_balanced_at_every_free_node() {
    // Exact prestress at rest: the discrete momentum equation must give zero
    // acceleration at every material-covered node not projected by a wall.
    // Before the wall traction term this was 0.40 g one cell above the floor,
    // 0.32 g one cell inside a side wall and 0.60 g in the corner.
    let spec = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 5e-5);
    let accelerations = first_step_accelerations(&spec, 5e-5);
    let layout = spec.build().expect("build").0.grid.layout;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // small positive ratio.
    let top_node = 1 + (spec.initial_size[2] / spec.grid_spacing).round() as usize;
    let mut by_layer: BTreeMap<(usize, usize), (f64, usize)> = BTreeMap::new();
    let mut wall_layer = 0.0f64;
    let mut covered = 0.0f64;
    for (&(i, j, k), a) in &accelerations {
        if k > top_node {
            // The node above the free surface holds 1/32 of a particle mass
            // and a traction-free truncated stencil; it is not a wall effect.
            continue;
        }
        let floor = cells_from_wall(layout.cells[2], k);
        let side = cells_from_wall(layout.cells[0], i).min(cells_from_wall(layout.cells[1], j));
        let entry = by_layer
            .entry((floor.min(2), side.min(2)))
            .or_insert((0.0, 0));
        entry.0 = entry.0.max(a.norm());
        entry.1 += 1;
        covered = covered.max(a.norm());
        if (floor == 1 || side == 1) && k + 1 < top_node {
            wall_layer = wall_layer.max(a.norm());
        }
    }
    for ((floor, side), (max, count)) in &by_layer {
        eprintln!("floor>={floor} side>={side} cells: nodes={count} max|a|/g={max}");
    }
    assert!(
        wall_layer < 1e-5,
        "node one cell inside a wall accelerates at {wall_layer} g"
    );
    // One cell below the free surface the lattice row above the surface is
    // absent, so the linear pressure profile is truncated: 0.004 g at this
    // resolution, independent of the walls.
    assert!(covered < 1e-2, "covered node acceleration {covered} g");
}

/// Matched spatial and temporal controls on the original footprint, printing
/// the absolutely normalised error (scale `rho g H`) so the numbers can be set
/// against `docs/hydrostatic-balance.md`; every case must also satisfy the
/// absolute error bound. This is not a refinement-convergence assertion.
#[test]
fn refinement_and_timestep_study() {
    let cases = [
        ("spatial", 0.004, 2.5e-5),
        ("spatial", 0.002, 2.5e-5),
        ("spatial", 0.001, 2.5e-5),
        ("temporal", 0.002, 5e-5),
        ("temporal", 0.002, 1.25e-5),
        ("footprint", 0.002, 5e-5),
    ];
    for (label, spacing, max_dt) in cases {
        let footprint = if label == "footprint" { 0.004 } else { 0.02 };
        let spec = water_column([footprint, footprint, 0.02], 0.01, spacing, max_dt);
        let (sim, obs) = run(&spec, 0.01);
        assert!(
            obs["hydrostatic_max_relative_error"] < 0.05,
            "{label}: absolute pressure error"
        );
        let errors = particle_errors(&spec, &sim);
        let layer = max_abs(errors.iter().filter(|e| e.1 <= 1.0).map(|e| e.0));
        let interior = max_abs(errors.iter().filter(|e| e.1 > 1.0).map(|e| e.0));
        eprintln!(
            "[study] {label} h={spacing} dt={max_dt} particles={} steps={} max={} rms={} wall_layer={} interior={} projection_j={} traction_impulse_z={}",
            sim.particles.len(),
            sim.step_count,
            obs["hydrostatic_max_relative_error"],
            obs["hydrostatic_rms_relative_error"],
            layer,
            interior,
            obs["wall_normal_projection_energy_j"],
            sim.ledger.wall_impulse.z,
        );
    }
}

#[test]
fn free_fall_parallel_to_frictionless_sidewalls_has_no_extra_acceleration() {
    let mut spec = water_column([0.02, 0.02, 0.04], 0.004, 0.002, 5e-5);
    spec.config.wall_friction = 0.0;
    let (mut sim, _) = spec.build().expect("fixture");
    for position in &mut sim.particles.position {
        position.z += 0.01;
    }
    sim.particles.kirchhoff.fill(nalgebra::Matrix3::zeros());
    sim.particles.volume_ratio.fill(1.0);
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    let dt = 5e-5;
    sim.advance_to(dt).expect("step");
    let expected = spec.config.gravity * dt;
    let defect = sim
        .particles
        .velocity
        .iter()
        .map(|velocity| (velocity - expected).norm())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "parallel freefall max velocity defect={defect:e}; wall impulse={:?}",
        sim.ledger.wall_impulse
    );
    assert!(
        defect < 1e-12,
        "frictionless sidewalls altered tangential free fall: {defect:e}"
    );
}

/// Max relative pressure error against the exact profile for an arbitrary
/// depth function (scale `rho g H`).
fn max_error_with_depth(
    spec: &FixtureSpec,
    sim: &Simulation,
    depth: impl Fn(&Vector3<f64>) -> f64,
) -> f64 {
    let scale = spec.material.density * GRAVITY * spec.initial_size[2];
    (0..sim.particles.len())
        .map(|p| {
            let reference = spec.hydrostatic_reference(depth(&sim.particles.position[p]).max(0.0));
            let measured = liquid_pressure(&spec.material, sim.particles.volume_ratio[p]);
            ((measured - reference) / scale).abs()
        })
        .fold(0.0f64, f64::max)
}

#[test]
fn column_resting_on_the_high_x_wall_under_rotated_gravity_matches_upright() {
    // Same column, coordinates rotated so gravity points along +x and the
    // "floor" is the x = L face: the traction must be face- and side-agnostic.
    let upright = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 5e-5);
    let (mut sim_up, _) = upright.build().expect("build");
    sim_up.advance_to(0.002).expect("advance");
    let h0 = upright.initial_size[2];
    let error_up = max_error_with_depth(&upright, &sim_up, |x| h0 - x.z);

    let mut rotated = upright.clone();
    rotated.config.gravity = Vector3::new(GRAVITY, 0.0, 0.0);
    let (mut sim_rot, _) = upright.build().expect("build");
    sim_rot.config = rotated.config;
    let length = upright.domain[0];
    for x in &mut sim_rot.particles.position {
        *x = Vector3::new(length - x.z, x.y, x.x);
    }
    sim_rot.ledger.initial_mechanical_energy = sim_rot.mechanical_energy();
    sim_rot.advance_to(0.002).expect("advance");
    assert_eq!(sim_rot.rejected_steps, 0);
    let error_rot = max_error_with_depth(&rotated, &sim_rot, |x| x.x - (length - h0));
    eprintln!("rotated gravity: upright={error_up} rotated={error_rot}");
    assert!(error_rot < 0.01, "rotated column error {error_rot}");
    assert!(
        (error_rot - error_up).abs() < 1e-9,
        "rotation changed the error: {error_up} vs {error_rot}"
    );
    assert!(sim_rot.ledger.wall_impulse.x < 0.0);
    assert!(sim_rot.ledger.wall_impulse.z.abs() < 1e-15);
    assert!(sim_rot.momentum_residual().norm() < 1e-12);
}

/// Stress-free block lifted off the floor and spanning the side walls.
fn lifted_stress_free_block(
    gravity: Vector3<f64>,
    velocity: Vector3<f64>,
) -> (FixtureSpec, Simulation) {
    let mut spec = water_column([0.02, 0.02, 0.04], 0.004, 0.002, 5e-5);
    spec.config.wall_friction = 0.0;
    spec.config.gravity = gravity;
    let (mut sim, _) = spec.build().expect("build");
    for x in &mut sim.particles.position {
        x.z += 0.01;
    }
    sim.particles.kirchhoff.fill(nalgebra::Matrix3::zeros());
    sim.particles.volume_ratio.fill(1.0);
    sim.particles.velocity.fill(velocity);
    sim.ledger.initial_momentum = sim.total_momentum();
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    (spec, sim)
}

#[test]
fn tangential_motion_past_frictionless_walls_is_pure_free_fall() {
    let dt = 5e-5;
    for (gravity, velocity) in [
        (
            Vector3::new(0.0, 0.0, -GRAVITY),
            Vector3::new(0.0, 0.0, 0.3),
        ),
        (
            Vector3::new(0.0, 0.0, -GRAVITY),
            Vector3::new(0.0, 0.0, -0.2),
        ),
        (Vector3::zeros(), Vector3::new(0.0, 0.0, 0.25)),
    ] {
        let (_, mut sim) = lifted_stress_free_block(gravity, velocity);
        sim.advance_to(dt).expect("step");
        let expected = velocity + gravity * dt;
        let defect = sim
            .particles
            .velocity
            .iter()
            .map(|v| (v - expected).norm())
            .fold(0.0f64, f64::max);
        eprintln!("tangential g={gravity:?} v={velocity:?}: defect={defect:e}");
        assert!(defect < 1e-12, "defect {defect:e}");
        assert_eq!(sim.ledger.wall_impulse, Vector3::zeros());
        assert!(sim.momentum_residual().norm() < 1e-15);
    }
}

#[test]
fn confined_uniform_pressure_does_not_accelerate_toward_side_walls() {
    // Uniform compression, no gravity, no friction, lifted off the floor. The
    // side walls confine it: the truncated stencil alone pulled the first
    // free node toward each wall at `0.0794 p / (rho dx)`; the traction must
    // cancel that exactly while the free top and bottom expand.
    let (spec, mut sim) = lifted_stress_free_block(Vector3::zeros(), Vector3::zeros());
    let ratio = 0.999;
    let pressure = liquid_pressure(&spec.material, ratio);
    sim.particles.volume_ratio.fill(ratio);
    sim.particles
        .kirchhoff
        .fill(nalgebra::Matrix3::identity() * (-pressure * ratio));
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    sim.advance_to(5e-5).expect("step");
    let lateral = sim
        .particles
        .velocity
        .iter()
        .map(|v| v.xy().norm())
        .fold(0.0f64, f64::max);
    let vertical = sim
        .particles
        .velocity
        .iter()
        .map(|v| v.z.abs())
        .fold(0.0f64, f64::max);
    let unbalanced = 0.0794 * pressure / (spec.material.density * spec.grid_spacing) * 5e-5;
    eprintln!(
        "confined pressure: lateral={lateral:e} vertical={vertical:e} untreated node scale={unbalanced:e}"
    );
    assert!(vertical > 1e-4, "free faces must expand");
    assert!(lateral < 1e-6 * unbalanced, "lateral drift {lateral:e}");
    assert!(sim.momentum_residual().norm() < 1e-15);
}

#[test]
fn subcell_gap_does_not_create_attraction_or_excess_contact_force() {
    // Existing contact-range limitation, now with the traction on top of it.
    // Particles represent `dx/2` cells, so a touching lattice has its lowest
    // centres at `dx/4`; here the lowest centres sit at `0.4 dx`, a represented
    // gap of `0.15 dx` between the cell faces and the floor. The wall-plane
    // node still carries mass and an approaching trial velocity from the
    // pressure gradient, so the constraint is active in the baseline too: the
    // projection alone reacts `9.156e-7 kg m/s` on this state at `a378fc8`,
    // the traction brings it to `9.311e-7`, both under the full contact value
    // `p A dt = 1.112e-6`. Contact range, not adhesion: the same state with
    // the block released is covered by the separation tests.
    let (spec, mut sim) = lifted_stress_free_block(Vector3::zeros(), Vector3::zeros());
    let ratio = 0.999;
    let pressure = liquid_pressure(&spec.material, ratio);
    sim.particles.volume_ratio.fill(ratio);
    sim.particles
        .kirchhoff
        .fill(nalgebra::Matrix3::identity() * (-pressure * ratio));
    let dx = spec.grid_spacing;
    let bottom = sim
        .particles
        .position
        .iter()
        .map(|x| x.z)
        .fold(f64::INFINITY, f64::min);
    for x in &mut sim.particles.position {
        x.z -= bottom - 0.4 * dx;
    }
    let represented_gap = sim
        .particles
        .position
        .iter()
        .map(|x| x.z)
        .fold(f64::INFINITY, f64::min)
        - 0.25 * dx;
    assert!((represented_gap - 0.15 * dx).abs() < 1e-15);
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    let dt = 5e-5;
    sim.advance_to(dt).expect("step");
    let face_force_impulse = pressure * spec.domain[0] * spec.domain[1] * dt;
    let impulse = sim.ledger.wall_impulse.z;
    eprintln!(
        "subcell gap {represented_gap:e} m: wall impulse z={impulse:e}, full-contact p A dt={face_force_impulse:e}"
    );
    assert!(
        impulse >= 0.0,
        "a wall must not attract material across a sub-cell gap"
    );
    assert!(
        impulse <= face_force_impulse * (1.0 + 1e-9),
        "impulse {impulse:e} exceeds the contact face force {face_force_impulse:e}"
    );
    assert!(sim.momentum_residual().norm() < 1e-15);
}

/// Stress-free block touching one or two faces, released from or approaching
/// them: release must leave pure free fall and zero wall impulse; approach
/// must act only along the inward normals of the touched faces.
#[test]
fn release_and_approach_on_rotated_faces_and_a_corner() {
    let dt = 5e-5;
    let g = GRAVITY;
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    // (gravity, unit outward direction from the touched faces, touched normals)
    let cases = [
        (
            Vector3::new(0.0, 0.0, -g),
            Vector3::new(0.0, 0.0, 1.0),
            [false, false, true],
        ),
        (
            Vector3::new(g, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            [true, false, false],
        ),
        (
            Vector3::new(g * s2, 0.0, -g * s2),
            Vector3::new(-s2, 0.0, s2),
            [true, false, true],
        ),
    ];
    for (gravity, outward, touched) in cases {
        for approach in [false, true] {
            let speed = if approach { -0.3 } else { 0.3 };
            let velocity = outward * speed;
            let (spec, mut sim) = lifted_stress_free_block(gravity, velocity);
            // The fixture spans the width; keep the right half so the x = 0
            // face is free, then put the block in contact with the floor
            // and/or the x = L face.
            if touched[0] {
                let left: Vec<usize> = (0..sim.particles.len())
                    .filter(|&p| sim.particles.position[p].x < 0.5 * spec.domain[0])
                    .collect();
                sim.particles.remove_sorted(&left);
                sim.ledger.initial_mass = sim.particles.total_mass();
            }
            let bottom = sim
                .particles
                .position
                .iter()
                .map(|x| x.z)
                .fold(f64::INFINITY, f64::min);
            let right = sim
                .particles
                .position
                .iter()
                .map(|x| x.x)
                .fold(f64::NEG_INFINITY, f64::max);
            for x in &mut sim.particles.position {
                if touched[2] {
                    x.z -= bottom - 0.25 * spec.grid_spacing;
                }
                if touched[0] {
                    x.x += spec.domain[0] - 0.25 * spec.grid_spacing - right;
                }
            }
            sim.ledger.initial_momentum = sim.total_momentum();
            sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
            sim.advance_to(dt).expect("step");
            let impulse = sim.ledger.wall_impulse;
            let expected = velocity + gravity * dt;
            let defect = sim
                .particles
                .velocity
                .iter()
                .map(|v| (v - expected).norm())
                .fold(0.0f64, f64::max);
            eprintln!(
                "faces {touched:?} approach={approach}: defect={defect:e} impulse={:?}",
                impulse.as_slice()
            );
            assert!(sim.momentum_residual().norm() < 1e-15);
            if approach {
                // Wall projection plus traction, only along the touched inward normals.
                assert!(!touched[2] || impulse.z > 0.0);
                assert!(!touched[0] || impulse.x < 0.0);
                assert!(impulse.y.abs() < 1e-18);
                assert!(touched[0] || impulse.x.abs() < 1e-18);
            } else {
                assert!(defect < 1e-12, "release altered free fall: {defect:e}");
                assert_eq!(impulse, Vector3::zeros());
            }
        }
    }
}

#[test]
fn stress_free_material_separating_from_floor_has_no_wall_force() {
    let gravity = Vector3::new(0.0, 0.0, -GRAVITY);
    let velocity = Vector3::new(0.0, 0.0, 0.3);
    let (_, mut sim) = lifted_stress_free_block(gravity, velocity);
    for position in &mut sim.particles.position {
        position.z -= 0.01;
    }
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    let dt = 5e-5;
    sim.advance_to(dt).expect("step");
    let expected = velocity + gravity * dt;
    let defect = sim
        .particles
        .velocity
        .iter()
        .map(|v| (v - expected).norm())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "separating stress-free block: defect={defect:e}, wall impulse={:?}",
        sim.ledger.wall_impulse
    );
    assert!(
        defect < 1e-12,
        "a nonadhesive wall acted on stress-free separating material"
    );
}

#[test]
fn hydrostatic_column_remains_accurate_after_longer_evolution() {
    let spec = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 5e-5);
    let (sim, obs) = run(&spec, 0.1);
    eprintln!(
        "longer hydro: steps={} max={} rms={}",
        sim.step_count,
        obs["hydrostatic_max_relative_error"],
        obs["hydrostatic_rms_relative_error"]
    );
    assert!(obs["hydrostatic_max_relative_error"] < 0.05);
}

#[test]
fn hydrostatic_accuracy_does_not_require_friction_to_damp_the_error() {
    let mut spec = water_column([0.02, 0.02, 0.02], 0.01, 0.002, 5e-5);
    spec.config.wall_friction = 0.0;
    let (_, obs) = run(&spec, 0.1);
    let error = obs["hydrostatic_max_relative_error"];
    eprintln!("frictionless hydrostatic 0.1 s: max={error}");
    assert!(error < 0.05, "frictionless hydrostatic error {error}");
}
