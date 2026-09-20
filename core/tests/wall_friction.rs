//! Adversarial check of the total normal/tangential wall impulse during sliding.
//!
//! References: `docs/research.md`; Coulomb sliding relation `|I_t| = mu I_n`.

use _core::mpm::constitutive::Material;
use _core::mpm::fixtures::{Fixture, FixtureSpec, ResourceLimits};
use _core::mpm::rigid::ContactParams;
use _core::mpm::solver::SolverConfig;
use nalgebra::{Matrix3, Vector3};

/// Uniaxially prestressed paste block on the floor, centred away from the side
/// walls, with the requested initial velocity and floor friction.
fn prestressed_block(
    friction: f64,
    velocity: Vector3<f64>,
) -> (FixtureSpec, _core::mpm::solver::Simulation) {
    let spec = FixtureSpec {
        fixture: Fixture::Slump,
        material: Material::paste(1000.0, 1e5, 0.0, 1e6, 1.0, 1.0),
        grid_spacing: 0.002,
        domain: [0.04, 0.04, 0.03],
        initial_size: [0.02, 0.02, 0.01],
        initial_velocity: velocity,
        seed: 7,
        pellet: None,
        config: SolverConfig {
            gravity: Vector3::new(0.0, 0.0, -9.81),
            wall_friction: friction,
            cfl: 0.3,
            max_dt: 5e-5,
            min_dt: 1e-9,
            contact: ContactParams {
                normal_stiffness: 1000.0,
                restitution: 0.2,
                friction,
            },
        },
        limits: ResourceLimits {
            max_particles: 100_000,
            max_nodes: 1_000_000,
        },
    };
    let (mut sim, _) = spec.build().expect("fixture");
    for p in 0..sim.particles.len() {
        let depth = spec.initial_size[2] - sim.particles.position[p].z;
        // Zero Poisson ratio permits uniaxial compression without lateral stress.
        // All wall-node tangential velocities then point exactly along +x.
        let pressure = spec.material.density * 9.81 * depth;
        sim.particles.deformation[p] =
            Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, (-pressure / 1e5).exp()));
        sim.particles.kirchhoff[p] = Matrix3::from_diagonal(&Vector3::new(0.0, 0.0, -pressure));
    }
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    (spec, sim)
}

#[test]
fn sliding_floor_impulse_obeys_coulomb_limit_for_total_normal_reaction() {
    let friction = 0.4;
    let (_, mut sim) = prestressed_block(friction, Vector3::new(1.0, 0.0, 0.0));
    sim.advance_to(5e-5).expect("step");
    assert_eq!(sim.limited_steps, 0);
    let impulse = sim.ledger.wall_impulse;
    let ratio = impulse.xy().norm() / (friction * impulse.z);
    eprintln!("wall impulse={impulse:?}; tangential/(mu*normal)={ratio:.12}");
    assert!(impulse.z > 0.0);
    assert!(
        (ratio - 1.0).abs() < 1e-12,
        "sliding friction no longer uses the total normal reaction"
    );
}

#[test]
fn slow_block_sticks_within_the_coulomb_budget() {
    // Tangential speed far below `mu N / m` at every floor-loaded node:
    // friction must stop the floor layer in one step without reversing it,
    // with the total tangential impulse strictly inside the cone. Only the
    // bottom half cell sees wall or budget nodes on its whole stencil, so the
    // rest of the block still carries its momentum after one step.
    let friction = 0.4;
    let speed = 1e-5;
    let (spec, mut sim) = prestressed_block(friction, Vector3::new(speed, 0.0, 0.0));
    sim.advance_to(5e-5).expect("step");
    let impulse = sim.ledger.wall_impulse;
    let ratio = impulse.xy().norm() / (friction * impulse.z);
    let floor_layer_speed = (0..sim.particles.len())
        .filter(|&p| sim.particles.position[p].z < 0.5 * spec.grid_spacing)
        .map(|p| sim.particles.velocity[p].xy().norm())
        .fold(0.0f64, f64::max);
    eprintln!(
        "stick: impulse={impulse:?} ratio={ratio:.6} floor-layer tangential speed={floor_layer_speed:e}"
    );
    assert!(impulse.z > 0.0);
    assert!(impulse.x < 0.0 && ratio < 0.5, "ratio {ratio}");
    assert!(
        floor_layer_speed < 1e-15,
        "floor layer still slides at {floor_layer_speed:e}"
    );
    // No reversal anywhere.
    assert!(sim.particles.velocity.iter().all(|v| v.x >= -1e-15));
    assert!(sim.ledger.wall_friction_dissipation > 0.0);
    sim.validate().expect("ledger signs");
}

#[test]
fn release_with_friction_receives_no_normal_or_tangential_impulse() {
    // Stress-free block leaving the floor at 0.3 m/s while sliding at 1 m/s:
    // an inactive contact must not generate friction from the image budget.
    let friction = 0.4;
    let (_, mut sim) = prestressed_block(friction, Vector3::new(1.0, 0.0, 0.3));
    sim.particles.kirchhoff.fill(Matrix3::zeros());
    sim.particles.deformation.fill(Matrix3::identity());
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    let dt = 5e-5;
    sim.advance_to(dt).expect("step");
    let expected = Vector3::new(1.0, 0.0, 0.3 - 9.81 * dt);
    let defect = sim
        .particles
        .velocity
        .iter()
        .map(|v| (v - expected).norm())
        .fold(0.0f64, f64::max);
    eprintln!(
        "release with friction: defect={defect:e} impulse={:?}",
        sim.ledger.wall_impulse
    );
    assert!(defect < 1e-12, "defect {defect:e}");
    assert_eq!(sim.ledger.wall_impulse, Vector3::zeros());
    assert!(sim.ledger.wall_friction_dissipation <= 0.0);
    sim.validate().expect("non-negative dissipation");
}

#[test]
fn vanishing_contact_does_not_turn_negative_image_weight_into_attraction() {
    // A near-zero inward trial speed gives almost no projection reaction.
    // At d/h=0.05 the image's gravity term exceeds its continued-stress force;
    // that negative correction must not overwhelm the unilateral constraint.
    let dt = 5e-5;
    let inward_speed = 1e-7;
    let velocity = Vector3::new(1.0, 0.0, 9.81 * dt - inward_speed);
    let (spec, mut sim) = prestressed_block(0.4, velocity);
    sim.particles.kirchhoff.fill(Matrix3::zeros());
    sim.particles.deformation.fill(Matrix3::identity());
    for position in &mut sim.particles.position {
        if position.z < 0.5 * spec.grid_spacing {
            position.z = 0.05 * spec.grid_spacing;
        }
    }
    sim.ledger.initial_mechanical_energy = sim.mechanical_energy();
    sim.advance_to(dt).expect("step");
    let constrained_mass: f64 = sim
        .grid
        .active
        .iter()
        .filter_map(|&index| {
            let (_, _, k) = sim.grid.layout.coords(index);
            sim.grid
                .layout
                .wall_side(2, k)
                .map(|_| sim.grid.mass[index])
        })
        .sum();
    let expected_normal = constrained_mass * inward_speed;
    let impulse = sim.ledger.wall_impulse;
    assert!(impulse.z >= 0.0);
    assert!((impulse.z - expected_normal).abs() < 1e-16);
    assert!((impulse.x + 0.4 * expected_normal).abs() < 1e-16);
    assert!(sim.momentum_residual().norm() < 1e-15);
    sim.validate().expect("valid state and ledger");
}

#[test]
fn zero_friction_preserves_even_a_tiny_tangential_velocity() {
    let speed = 1e-16;
    let (_, mut sim) = prestressed_block(0.0, Vector3::new(speed, 0.0, 0.0));
    sim.advance_to(5e-5).expect("step");
    assert_eq!(sim.ledger.wall_impulse.x.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        sim.ledger.wall_friction_dissipation.to_bits(),
        0.0_f64.to_bits()
    );
    assert!(
        sim.particles
            .velocity
            .iter()
            .all(|v| (v.x - speed).abs() < 1e-29)
    );
}
