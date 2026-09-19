//! Regression checks for zero-weight stencil entries and wall impulse accounting.
//!
//! References: `docs/research.md`; quadratic B-spline transfers in
//! <https://doi.org/10.1145/3197517.3201293>.

use _core::mpm::constitutive::Material;
use _core::mpm::grid::{Grid, GridLayout};
use _core::mpm::particles::ParticleSet;
use _core::mpm::rigid::ContactParams;
use _core::mpm::solver::{Simulation, SolverConfig};
use nalgebra::Vector3;

#[test]
fn zero_weight_deposits_do_not_register_active_nodes() {
    let layout = GridLayout::new([0.02; 3], 0.002, 1_000_000).expect("layout");
    let mut grid = Grid::new(layout);
    grid.deposit(0, 0.0, Vector3::zeros());
    grid.deposit(0, 0.0, Vector3::zeros());
    assert!(grid.active.is_empty());
    grid.deposit(0, 1.0, Vector3::new(1.0, 0.0, 0.0));
    grid.deposit(0, 0.0, Vector3::zeros());
    grid.deposit(0, 2.0, Vector3::new(2.0, 0.0, 0.0));
    assert_eq!(grid.active, vec![0]);
    assert!((grid.total_momentum().x - 3.0).abs() < 1e-15);
    grid.clear();
    assert!(grid.active.is_empty());
    assert!(grid.mass[0].abs() < 1e-15);
}

#[test]
fn exact_half_cell_particle_preserves_unique_nodes_and_wall_momentum() {
    let spacing = 0.002;
    let material = Material::paste(1000.0, 1.0e4, 0.3, 1.0e6, 0.0, 1.0);
    let layout = GridLayout::new([0.02; 3], spacing, 1_000_000).expect("layout");
    let cfg = SolverConfig {
        gravity: Vector3::zeros(),
        wall_friction: 0.0,
        cfl: 0.3,
        max_dt: 1e-4,
        min_dt: 1e-12,
        contact: ContactParams {
            normal_stiffness: 1000.0,
            restitution: 0.2,
            friction: 0.4,
        },
    };
    for offset in [0.0, 1e-7] {
        let mut particles = ParticleSet::with_capacity(2);
        for x in [0.02 - 0.5 * spacing - offset, 0.02 - 0.25 * spacing] {
            particles.push(
                Vector3::new(x, 0.01, 0.01),
                Vector3::new(0.5, 0.0, 0.0),
                1e-6,
                1e-9,
            );
        }
        let mut sim = Simulation::new(material, particles, Grid::new(layout), None, cfg);
        sim.advance_to(1e-4).expect("advance");
        let mut nodes = sim.grid.active.clone();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes.len(), sim.grid.active.len(), "duplicate active nodes");
        assert!(
            sim.momentum_residual().norm() < 1e-18,
            "wall impulses counted incorrectly: {:?}",
            sim.momentum_residual()
        );
    }
}
