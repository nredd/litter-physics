//! Bounded actual-kernel energy audit fixtures, emitted as one JSON document.
//!
//! Run `CARGO_BUILD_JOBS=2 cargo run -p litter-physics-core --example energy_audit`.
//! Fixed cases: 4000-particle and 20000-node allocation caps, two snapshots per
//! fixture, three local timesteps from each IDENTICAL snapshot. Integration has
//! a 3000-trial budget checked before every attempted integration step. This is a local-step comparison,
//! not a physical closure or a global accuracy/convergence certification.

use std::error::Error;
use std::io::{self, BufWriter, Write};

use _core::mpm::audit::EnergyAudit;
use _core::mpm::constitutive::{Material, update_paste};
use _core::mpm::fixtures::{Fixture, FixtureSpec, PelletSpec, ResourceLimits};
use _core::mpm::grid::{Grid, GridLayout};
use _core::mpm::particles::ParticleSet;
use _core::mpm::rigid::{ContactParams, Pellet};
use _core::mpm::solver::{Simulation, SolverConfig};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use serde::Serialize;

/// One nonmutating local-step measurement from a shared evolved snapshot.
#[derive(Serialize)]
struct Sample {
    /// Stable fixture label.
    case: &'static str,
    /// Requested production timestep cap in s.
    timestep_cap: f64,
    /// Actual material point count, excluding the optional pellet.
    particle_count: usize,
    /// Grid spacing in m for interpreting the affine kinetic norm.
    grid_spacing_m: f64,
    /// Number of real accepted steps preceding the audit.
    accepted_steps: u64,
    /// Actual energy measurements, all energies in J.
    audit: EnergyAudit,
}

/// Document scope is deliberately explicit in the machine-readable output.
#[derive(Serialize)]
struct Evidence {
    schema_version: u8,
    interpretation: &'static str,
    unresolved: [&'static str; 6],
    samples: Vec<Sample>,
}

/// Shared production configuration, independently rebuilt for each fixture.
fn config(dt: f64) -> SolverConfig {
    SolverConfig {
        gravity: Vector3::new(0.0, 0.0, -9.81),
        wall_friction: 0.4,
        cfl: 0.3,
        max_dt: dt,
        min_dt: 1e-9,
        contact: ContactParams {
            normal_stiffness: 1000.0,
            restitution: 0.2,
            friction: 0.4,
        },
    }
}

/// Few-particle analytic fixtures with support entirely in the interior.
fn isolated(case: &str, dt: f64) -> Result<Simulation, Box<dyn Error>> {
    let material = Material::paste(1000.0, 1e4, 0.3, 100.0, 1.0, 0.6);
    let mut particles = ParticleSet::with_capacity(2);
    particles.push(
        Vector3::new(0.02, 0.02, 0.03),
        Vector3::zeros(),
        0.001,
        1e-6,
    );
    let mut cfg = config(dt);
    if case != "freefall" {
        cfg.gravity = Vector3::zeros();
    }
    match case {
        "translation" => particles.velocity[0].x = 0.1,
        "affine" => {
            particles.affine[0] = Matrix3::new(0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        "collision" => {
            particles.velocity[0].x = 0.1;
            particles.push(
                particles.position[0],
                Vector3::new(-0.1, 0.0, 0.0),
                0.001,
                1e-6,
            );
        }
        "prestressed" => {
            let initial = update_paste(&(Matrix3::identity() * 0.999), &material, dt)?;
            particles.deformation[0] = initial.deformation;
            particles.kirchhoff[0] = initial.kirchhoff;
        }
        "rest" | "freefall" => {}
        _ => return Err(format!("unknown isolated fixture '{case}'").into()),
    }
    let grid = Grid::new(GridLayout::new([0.04, 0.04, 0.06], 0.004, 20000)?);
    Ok(Simulation::new(material, particles, grid, None, cfg))
}

/// Small canonical research fixtures, retaining their real initializer and solver.
fn research(fixture: Fixture, dt: f64) -> Result<Simulation, Box<dyn Error>> {
    let hydro = fixture == Fixture::Hydrostatic;
    let spec = FixtureSpec {
        fixture,
        material: if hydro {
            Material::liquid(1000.0, 1e4, 0.3, 0.001)
        } else {
            Material::paste(1000.0, 1e4, 0.3, 100.0, 1.0, 0.6)
        },
        grid_spacing: 0.004,
        domain: [0.032, 0.032, 0.048],
        initial_size: if hydro {
            [0.032, 0.032, 0.012]
        } else {
            [0.016, 0.016, 0.012]
        },
        initial_velocity: Vector3::zeros(),
        seed: 42,
        pellet: (fixture == Fixture::CoupledPatch).then_some(PelletSpec {
            radius: 0.006,
            length: 0.016,
            density: 800.0,
        }),
        config: config(dt),
        limits: ResourceLimits {
            max_particles: 4000,
            max_nodes: 20000,
        },
    };
    let (mut simulation, _) = spec.build()?;
    // A downward launch reaches the patch within the bounded 6ms window, so the
    // later sample measures actual coupling rather than an untouched gap.
    if let Some(pellet) = &mut simulation.pellet {
        pellet.velocity.z = -2.0;
    }
    // Rebuild initial histories after the synthetic launch, rather than leaving
    // momentum and mechanical-energy baselines tied to the stationary pellet.
    Ok(Simulation::new(
        simulation.material,
        simulation.particles,
        simulation.grid,
        simulation.pellet,
        simulation.config,
    ))
}

/// Deliberately light body: exposes coupling energy creation despite zero gravity,
/// stress and wall work. This is an inertial-ratio probe, NOT a pine-pellet model.
fn light_body(dt: f64) -> Result<Simulation, Box<dyn Error>> {
    let mut particles = ParticleSet::with_capacity(1);
    particles.push(
        Vector3::new(0.5, 0.5, 0.5),
        Vector3::new(0.0, 0.0, 1.0),
        2.0,
        0.002,
    );
    let pellet = Pellet::new(
        0.15,
        0.3,
        1.0,
        Vector3::new(0.5, 0.5, 0.65),
        UnitQuaternion::identity(),
    )?;
    let mut cfg = config(dt);
    cfg.gravity = Vector3::zeros();
    cfg.wall_friction = 0.0;
    cfg.contact.friction = 0.0;
    Ok(Simulation::new(
        Material::paste(1000.0, 1000.0, 0.3, 1e6, 0.0, 1.0),
        particles,
        Grid::new(GridLayout::new([1.0; 3], 0.1, 20000)?),
        Some(pellet),
        cfg,
    ))
}

/// Integrate between samples, with a hard work bound and exact nonmutation check.
fn sample_run(
    case: &'static str,
    mut sim: Simulation,
    cap: f64,
    later: f64,
    samples: &mut Vec<Sample>,
) -> Result<(), Box<dyn Error>> {
    let mut attempts = 0;
    for time in [0.0, later] {
        let complete = sim.advance_while(time, || {
            let allowed = attempts < 3000;
            attempts += usize::from(allowed);
            allowed
        })?;
        if !complete {
            return Err(format!("bounded example exhausted trial allowance for '{case}'").into());
        }
        let original = sim.clone();
        let dt = (sim.dt_scale * sim.stability_limit()).min(cap);
        for divisor in [1.0, 2.0, 4.0] {
            let audit = sim.audit_step(dt / divisor)?;
            if sim != original {
                return Err("audit mutated original simulation".into());
            }
            if case == "coupled_patch"
                && time > 0.0
                && (audit.increments.coupling_grid_energy == 0.0
                    || audit.increments.coupling_pellet_energy == 0.0)
            {
                return Err("evolved coupled fixture has no actual coupling".into());
            }
            samples.push(Sample {
                case,
                timestep_cap: cap,
                particle_count: sim.particles.len(),
                grid_spacing_m: sim.grid.layout.spacing,
                accepted_steps: sim.step_count,
                audit,
            });
        }
    }
    Ok(())
}

/// Emit bounded real measurements, with errors propagated to the process status.
fn main() -> Result<(), Box<dyn Error>> {
    let mut samples = Vec::new();
    let cap = 4e-5;
    for case in [
        "rest",
        "translation",
        "affine",
        "collision",
        "freefall",
        "prestressed",
    ] {
        sample_run(case, isolated(case, cap)?, cap, 2e-4, &mut samples)?;
    }
    sample_run(
        "light_body_coupling",
        light_body(cap)?,
        cap,
        2e-4,
        &mut samples,
    )?;
    for fixture in [Fixture::Hydrostatic, Fixture::Slump, Fixture::CoupledPatch] {
        sample_run(
            fixture.name(),
            research(fixture, cap)?,
            cap,
            0.006,
            &mut samples,
        )?;
    }
    let evidence = Evidence {
        schema_version: 1,
        interpretation: "Algebraic stage telescope, NOT physical energy closure. APIC affine norm is discretization-specific; legacy mechanical_energy is unchanged. Each dt/dt2/dt4 triple uses the same snapshot: local-step comparison, NOT global convergence.",
        unresolved: [
            "Stress/storage/plastic mismatch includes constitutive lag, viscous effects and approximate plastic work; not invented dissipation.",
            "Pellet integration combines gravity/contact/rotation; recoverable contact-spring storage is not measured.",
            "Signed wall traction and projection are discrete grid changes, not stationary-wall work.",
            "Quadratic full-support transfers only; escaped particles and unstable requested timesteps are rejected, not clipped.",
            "Canned synthetic fixtures, not the shipped YAML presets: coupled pellet starts at -2 m/s vertically to exercise impact within 6 ms.",
            "Light-body coupling probe demonstrates artificial kinetic-energy creation; coupled energy stability is not established.",
        ],
        samples,
    };
    let mut output = BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, &evidence)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}
