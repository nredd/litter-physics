//! Independent analytic and variance references for transient audit measurements.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::audit::EnergyStages;
use super::constitutive::Material;
use super::grid::{Grid, GridLayout};
use super::particles::ParticleSet;
use super::rigid::{ContactParams, Pellet};
use super::solver::{Simulation, SolverConfig};

/// Interior one-particle fixture with full quadratic support.
fn isolated() -> Simulation {
    let material = Material::paste(1000.0, 1000.0, 0.3, 1e6, 0.0, 1.0);
    let mut particles = ParticleSet::with_capacity(2);
    particles.push(Vector3::new(0.47, 0.52, 0.55), Vector3::zeros(), 2.0, 0.002);
    Simulation::new(
        material,
        particles,
        Grid::new(GridLayout::new([1.0; 3], 0.1, 10000).unwrap()),
        None,
        SolverConfig {
            gravity: Vector3::zeros(),
            wall_friction: 0.4,
            cfl: 0.3,
            max_dt: 1e-3,
            min_dt: 1e-9,
            contact: ContactParams {
                normal_stiffness: 1000.0,
                restitution: 0.2,
                friction: 0.4,
            },
        },
    )
}

/// Absolute tolerance for analytic fixtures with order-one energies.
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-12,
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn translation_and_affine_norm_are_analytic_and_nonmutating() {
    let mut sim = isolated();
    sim.particles.velocity[0] = Vector3::new(1.0, -2.0, 0.5);
    sim.particles.affine[0] = Matrix3::new(0.2, 0.3, -0.1, 0.5, -0.4, 0.7, 0.1, 0.2, 0.4);
    let original = sim.clone();
    let audit = sim.audit_step(1e-4).unwrap();
    close(audit.before.translation, 5.25);
    // h² m / 8 times sum of the nine explicitly supplied squares.
    let expected_affine = 0.0025 * (0.04 + 0.09 + 0.01 + 0.25 + 0.16 + 0.49 + 0.01 + 0.04 + 0.16);
    close(audit.before.affine, expected_affine);
    close(audit.stages.grid_transport, 5.25 + expected_affine);
    close(audit.after.translation, 5.25);
    close(audit.after.affine, expected_affine);
    close(audit.terms.p2g_transfer_loss, 0.0);
    close(audit.terms.g2p_transfer_loss, 0.0);
    close(audit.terms.telescoping_error, 0.0);
    assert_eq!(sim, original);
    // The old physical diagnostic intentionally excludes the APIC norm.
    close(sim.kinetic_energy(), audit.before.translation);
}

#[test]
fn colliding_velocities_have_exact_variance_loss() {
    let mut sim = isolated();
    let position = sim.particles.position[0];
    sim.particles.velocity[0] = Vector3::x();
    sim.particles.push(position, -Vector3::x(), 2.0, 0.002);
    let report = sim.audit_step(1e-4).unwrap();
    close(report.before.translation, 2.0);
    close(report.stages.grid_transport, 0.0);
    close(report.terms.p2g_transfer_loss, 2.0);
    close(report.terms.g2p_transfer_loss, 0.0);
}

#[test]
fn transfers_match_independent_stencil_variance_identities() {
    let mut sim = isolated();
    sim.particles.velocity[0] = Vector3::new(0.7, -0.2, 0.3);
    sim.particles.affine[0] = Matrix3::identity() * 0.3;
    sim.particles.push(
        Vector3::new(0.51, 0.49, 0.58),
        Vector3::new(-0.4, 0.8, -0.1),
        3.0,
        0.003,
    );
    sim.particles.affine[1] = Matrix3::new(0.1, -0.2, 0.3, 0.4, 0.2, -0.1, 0.0, 0.3, -0.2);
    let report = sim.audit_step(1e-4).unwrap();
    let mut trial = sim.clone();
    trial
        .try_step_captured(1e-4, Some(&mut EnergyStages::default()))
        .unwrap();
    let layout = sim.grid.layout;
    // No forces: final grid is the pure transport velocity. Reconstruct local
    // residuals directly, rather than subtracting two kinetic-energy totals.
    let mut p2g_variance = 0.0;
    let mut g2p_variance = 0.0;
    for p in 0..sim.particles.len() {
        let stencil = layout.stencil(&sim.particles.position[p]).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let index = layout.index(
                        stencil.base[0] + a,
                        stencil.base[1] + b,
                        stencil.base[2] + c,
                    );
                    let distance = stencil.distance(a, b, c);
                    let grid_velocity = trial.grid.momentum[index];
                    let old = sim.particles.velocity[p] + sim.particles.affine[p] * distance;
                    let reconstructed =
                        trial.particles.velocity[p] + trial.particles.affine[p] * distance;
                    let half_mass_weight = 0.5 * sim.particles.mass[p] * stencil.weight(a, b, c);
                    p2g_variance += half_mass_weight * (old - grid_velocity).norm_squared();
                    g2p_variance +=
                        half_mass_weight * (grid_velocity - reconstructed).norm_squared();
                }
            }
        }
    }
    assert!(p2g_variance > 0.0 && g2p_variance > 0.0);
    close(report.terms.p2g_transfer_loss, p2g_variance);
    close(report.terms.g2p_transfer_loss, g2p_variance);
}

#[test]
fn freefall_has_semiimplicit_gravity_potential_defect() {
    let mut sim = isolated();
    sim.config.gravity.z = -9.81;
    let dt = 1e-3;
    let whole = sim.audit_step(dt).unwrap();
    let half = sim.audit_step(dt / 2.0).unwrap();
    let expected = -0.5 * 2.0 * 9.81_f64.powi(2) * dt * dt;
    close(
        whole.terms.uncoupled_gravity_potential_mismatch.unwrap(),
        expected,
    );
    close(
        half.terms.uncoupled_gravity_potential_mismatch.unwrap(),
        expected / 4.0,
    );
    assert!(
        whole
            .terms
            .coupled_grid_gravity_total_potential_change
            .is_none()
    );
}

#[test]
fn resting_and_prestressed_material_expose_storage_mismatch() {
    let mut sim = isolated();
    let rest = sim.audit_step(1e-4).unwrap();
    close(rest.terms.augmented_energy_change, 0.0);
    sim.material = Material::liquid(1000.0, 1000.0, 0.3, 0.0);
    sim.particles.volume_ratio[0] = 0.95;
    sim.particles.kirchhoff[0] =
        Matrix3::identity() * (sim.material.bulk_modulus * (sim.particles.volume_ratio[0] - 1.0));
    let stressed = sim.audit_step(1e-4).unwrap();
    assert!(stressed.before.elastic > 0.0);
    assert!(stressed.stages.grid_after_stress > stressed.stages.grid_transport);
    assert!(stressed.terms.stress_storage_plastic_mismatch.abs() > 1e-10);
    close(stressed.terms.telescoping_error, 0.0);
}

#[test]
fn signed_wall_stages_match_existing_histories() {
    let mut sim = isolated();
    sim.particles.position[0].z = 0.025;
    sim.particles.velocity[0] = Vector3::new(0.1, 0.0, -0.5);
    sim.particles.kirchhoff[0] = Matrix3::identity() * -1000.0;
    sim.config.gravity.z = -9.81;
    let report = sim.audit_step(1e-4).unwrap();
    assert!(report.increments.wall_normal_projection_energy < 0.0);
    assert!(report.increments.wall_friction_dissipation > 0.0);
    assert!(report.increments.wall_normal_traction_energy < 0.0);
    close(
        report.terms.traction_change,
        report.increments.wall_normal_traction_energy,
    );
    close(
        report.terms.walls_change,
        report.increments.wall_normal_projection_energy
            - report.increments.wall_friction_dissipation,
    );
    close(report.terms.telescoping_error, 0.0);
    // Same prestressed particle at rest yields a positive traction jump:
    // traction accelerates already upward stress momentum before gravity.
    sim.particles.velocity[0] = Vector3::zeros();
    let resting = sim.audit_step(1e-4).unwrap();
    assert!(resting.increments.wall_normal_traction_energy > 0.0);
    close(
        resting.terms.traction_change,
        resting.increments.wall_normal_traction_energy,
    );
}

#[test]
fn coupled_stages_are_combined_not_fake_gravity_work() {
    // Exercise light and heavy bodies without requiring the known light-body
    // energy-creation defect to persist when the coupling solver is corrected.
    for density in [1.0, 500.0] {
        let mut sim = isolated();
        sim.particles.position[0] = Vector3::new(0.5, 0.5, 0.5);
        sim.particles.velocity[0] = Vector3::new(0.0, 0.0, 1.0);
        sim.config.gravity.z = -9.81;
        sim.pellet = Some(
            Pellet::new(
                0.15,
                0.3,
                density,
                Vector3::new(0.5, 0.5, 0.65),
                UnitQuaternion::identity(),
            )
            .unwrap(),
        );
        let report = sim.audit_step(1e-4).unwrap();
        assert!(report.terms.uncoupled_gravity_potential_mismatch.is_none());
        assert!(
            report
                .terms
                .coupled_grid_gravity_total_potential_change
                .is_some()
        );
        assert!(report.increments.coupling_grid_energy < 0.0);
        assert!(report.increments.coupling_pellet_energy > 0.0);
        close(
            report.terms.coupling_change,
            report.increments.coupling_grid_energy + report.increments.coupling_pellet_energy,
        );
        close(
            report.terms.pellet_integrator_combined_change,
            report.after.pellet - report.stages.pellet_after_coupling,
        );
        close(report.terms.telescoping_error, 0.0);
        let initial_momentum = sim.total_momentum();
        sim.advance_to(1e-4).unwrap();
        close(
            (sim.total_momentum() - initial_momentum - sim.ledger.gravity_impulse).norm(),
            0.0,
        );
    }
}

#[test]
fn invalid_unstable_escaped_and_rejected_trials_do_not_mutate() {
    let sim = isolated();
    let original = sim.clone();
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12, 1.0] {
        assert!(sim.audit_step(dt).is_err());
        assert_eq!(sim, original);
    }
    let mut recovery_limited = sim.clone();
    recovery_limited.dt_scale = 0.001;
    assert!(recovery_limited.audit_step(1e-4).is_err());
    let mut overflow = sim.clone();
    overflow.particles.affine[0] = Matrix3::identity() * 1e200;
    let original = overflow.clone();
    assert!(overflow.audit_step(1e-4).is_err());
    assert_eq!(overflow, original);
    let mut escaped = sim.clone();
    escaped.particles.position[0].x = -10.0;
    let original = escaped.clone();
    assert!(escaped.audit_step(1e-4).is_err());
    assert_eq!(escaped, original);
    let mut rejected = sim;
    // Finite state, stable dt from particle speed, but enormous stress causes
    // the actual trial kernel to reject displacement or constitutive update.
    rejected.particles.kirchhoff[0] = Matrix3::identity() * 1e20;
    let original = rejected.clone();
    assert!(rejected.audit_step(1e-4).is_err());
    assert_eq!(rejected, original);
}

#[test]
fn report_serializes_finite_measurements_and_optional_scope() {
    let sim = isolated();
    let report = sim.audit_step(1e-4).unwrap();
    let encoded = serde_json::to_string(&report).unwrap();
    let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert!(value["before"]["affine"].is_number());
    assert!(value["terms"]["uncoupled_gravity_potential_mismatch"].is_number());
    assert!(value["terms"]["coupled_grid_gravity_total_potential_change"].is_null());
    assert!(value["increments"]["wall_normal_traction_energy"].is_number());
}

#[test]
fn plastic_increment_is_history_difference_not_total_or_invented_dissipation() {
    let mut sim = isolated();
    sim.material = Material::paste(1000.0, 1000.0, 0.3, 0.1, 0.0, 1.0);
    sim.particles.affine[0][(0, 1)] = 50.0;
    sim.ledger.plastic_dissipation = 3.0;
    let audit = sim.audit_step(1e-3).unwrap();
    sim.advance_to(1e-3).unwrap();
    assert!(audit.increments.plastic_dissipation > 0.0);
    close(
        audit.increments.plastic_dissipation,
        sim.ledger.plastic_dissipation - 3.0,
    );
    let expected = audit.stages.grid_after_stress - audit.stages.grid_transport
        + audit.after.elastic
        - audit.before.elastic
        + audit.stages.plastic_dissipation;
    close(audit.terms.stress_storage_plastic_mismatch, expected);
    close(audit.terms.telescoping_error, 0.0);
}

#[test]
fn local_plastic_work_survives_cumulative_history_rounding() {
    let mut sim = isolated();
    sim.material = Material::paste(1000.0, 1000.0, 0.3, 0.1, 0.0, 1.0);
    sim.particles.affine[0][(0, 1)] = 50.0;
    sim.ledger.plastic_dissipation = 1e100;
    let original = sim.clone();
    let audit = sim.audit_step(1e-3).unwrap();
    assert!(audit.stages.plastic_dissipation > 0.0);
    assert_eq!(
        audit.increments.plastic_dissipation.to_bits(),
        0.0_f64.to_bits()
    );
    close(
        audit.terms.stress_storage_plastic_mismatch,
        audit.stages.grid_after_stress - audit.stages.grid_transport + audit.after.elastic
            - audit.before.elastic
            + audit.stages.plastic_dissipation,
    );
    assert_eq!(sim, original);
}

#[test]
fn rigid_gravity_contact_rotation_stay_one_combined_term() {
    let mut sim = isolated();
    sim.config.gravity.z = -9.81;
    let mut pellet = Pellet::new(
        0.1,
        0.3,
        500.0,
        Vector3::new(0.5, 0.5, 0.095),
        UnitQuaternion::from_euler_angles(0.0, 0.03, 0.0),
    )
    .unwrap();
    pellet.velocity = Vector3::new(0.1, 0.0, -0.2);
    pellet.angular_momentum = Vector3::new(0.01, 0.02, 0.03);
    sim.pellet = Some(pellet);
    let original = sim.clone();
    let audit = sim.audit_step(1e-4).unwrap();
    assert!(audit.increments.pellet_wall_work.abs() > 0.0);
    assert!(audit.terms.uncoupled_gravity_potential_mismatch.is_none());
    assert!(audit.terms.pellet_integrator_combined_change.abs() > 0.0);
    // Approximate contact work is not an exact kinetic stage change, and does
    // not include gravity/rotation integration or recoverable spring storage.
    assert!(
        (audit.terms.pellet_integrator_combined_change - audit.increments.pellet_wall_work).abs()
            > 1e-9
    );
    close(audit.terms.telescoping_error, 0.0);
    assert_eq!(sim, original);
}

#[test]
fn audit_matches_actual_advance_endpoint_and_existing_histories() {
    let mut sim = isolated();
    sim.particles.velocity[0] = Vector3::new(0.1, -0.2, 0.3);
    sim.config.gravity.z = -9.81;
    let audit = sim.audit_step(1e-4).unwrap();
    sim.advance_to(1e-4).unwrap();
    close(audit.after.translation, sim.particles.kinetic_energy());
    close(audit.after.elastic, sim.elastic_energy());
    close(audit.after.potential, sim.potential_energy());
    close(
        audit.increments.plastic_dissipation,
        sim.ledger.plastic_dissipation,
    );
}
