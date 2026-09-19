//! Research fixture construction and observables.
//!
//! Fixtures (coordinates: `z` up, gravity `-z`, domain `[0, L]^3`):
//!
//! - `slump`: block centred horizontally, resting on the floor, released from
//!   rest (or with the requested velocity). Observables include the final height,
//!   spread, and the Pashias et al. 1996 cylinder-slump estimate as a sanity
//!   reference (not validation).
//! - `hydrostatic`: column spanning the horizontal domain, initialised with the
//!   exact static stress profile (liquid: `p = K (exp(rho g (H - z) / K) - 1)`;
//!   paste: confined uniaxial compression). Observables report the deviation of
//!   the particle pressure / vertical stress from the exact profile.
//! - `dam_break`: block in the `x = 0`, `y = 0` corner released from rest.
//!   Observables report the front position and the residual column height, and
//!   the Ritter inviscid front bound `x <= a + 2 sqrt(g h0) t` for comparison.
//! - `coupled_patch`: paste block centred on the floor with an oriented pellet
//!   (from the request's first box) resting just above it, falling under gravity
//!   with two-way coupling. Observables report the pellet kinematics, the
//!   cumulative coupling impulses and the momentum balance residual.
//!
//! References:
//! - Pashias et al. 1996, <https://doi.org/10.1122/1.550780>
//! - Martin & Moyce 1952, <https://doi.org/10.1098/rsta.1952.0006>
//! - Ritter 1892 dam-break solution, as summarised in Stoker, Water Waves (1957)

use std::collections::BTreeMap;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::constitutive::{Material, MaterialKind, liquid_pressure};
use super::grid::{Grid, GridLayout};
use super::particles::ParticleSet;
use super::rigid::Pellet;
use super::rng::SplitMix64;
use super::solver::{Simulation, SolverConfig};

/// Particles per cell per axis.
const PARTICLES_PER_AXIS: usize = 2;
/// Fraction of the sub-cell spacing used as placement jitter.
const JITTER_FRACTION: f64 = 0.1;
/// Grid cells required across a pellet radius for the coupling to resolve it.
const MIN_CELLS_PER_PELLET_RADIUS: f64 = 1.5;
/// Gap between the block top and the pellet surface in cells.
const PELLET_GAP_CELLS: f64 = 2.0;

/// Fixture identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fixture {
    /// Gravity slump of a block.
    Slump,
    /// Static column with exact initial stress.
    Hydrostatic,
    /// Collapse of a corner column.
    DamBreak,
    /// Paste patch with a falling oriented pellet.
    CoupledPatch,
}

impl Fixture {
    /// Wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Slump => "slump",
            Self::Hydrostatic => "hydrostatic",
            Self::DamBreak => "dam_break",
            Self::CoupledPatch => "coupled_patch",
        }
    }
}

/// Pellet geometry from the request's box definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PelletSpec {
    /// Radius in m.
    pub radius: f64,
    /// Length in m.
    pub length: f64,
    /// Density in kg/m^3.
    pub density: f64,
}

/// Allocation caps enforced before allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Maximum particle count.
    pub max_particles: usize,
    /// Maximum grid node count.
    pub max_nodes: usize,
}

/// Resolved fixture specification (validated by the caller).
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureSpec {
    /// Which fixture.
    pub fixture: Fixture,
    /// Material.
    pub material: Material,
    /// Grid spacing in m.
    pub grid_spacing: f64,
    /// Domain extents in m.
    pub domain: [f64; 3],
    /// Initial block size in m.
    pub initial_size: [f64; 3],
    /// Initial block velocity in m/s.
    pub initial_velocity: Vector3<f64>,
    /// Placement seed.
    pub seed: u64,
    /// Pellet geometry (required for `coupled_patch`, rejected otherwise).
    pub pellet: Option<PelletSpec>,
    /// Solver configuration.
    pub config: SolverConfig,
    /// Allocation caps.
    pub limits: ResourceLimits,
}

impl FixtureSpec {
    /// Particle count the fixture would allocate.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.axis_counts()
            .iter()
            .copied()
            .fold(1, usize::saturating_mul)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // rounded positive ratios.
    fn axis_counts(&self) -> [usize; 3] {
        #[allow(clippy::cast_precision_loss)]
        let sub = self.grid_spacing / PARTICLES_PER_AXIS as f64;
        self.initial_size
            .map(|size| (size / sub).round().max(1.0) as usize)
    }

    /// Block origin (minimum corner) in m.
    fn block_origin(&self) -> Vector3<f64> {
        match self.fixture {
            Fixture::Slump | Fixture::CoupledPatch => Vector3::new(
                0.5 * (self.domain[0] - self.initial_size[0]),
                0.5 * (self.domain[1] - self.initial_size[1]),
                0.0,
            ),
            Fixture::Hydrostatic | Fixture::DamBreak => Vector3::zeros(),
        }
    }

    /// Validate fixture-specific geometry and material compatibility.
    ///
    /// # Errors
    ///
    /// descriptive message naming the offending field.
    pub fn validate(&self) -> Result<(), String> {
        for axis in 0..3 {
            if self.initial_size[axis] > self.domain[axis] + 1e-12 {
                return Err(format!(
                    "`initial_size_m[{axis}]`='{}' exceeds `domain_m[{axis}]`='{}'",
                    self.initial_size[axis], self.domain[axis]
                ));
            }
            if self.initial_size[axis] < self.grid_spacing {
                return Err(format!(
                    "`initial_size_m[{axis}]`='{}' must span at least one cell of '{}'",
                    self.initial_size[axis], self.grid_spacing
                ));
            }
        }
        let count = self.particle_count();
        if count > self.limits.max_particles {
            return Err(format!(
                "fixture would allocate '{count}' particles, exceeding the cap of '{}'",
                self.limits.max_particles
            ));
        }
        match self.fixture {
            Fixture::Hydrostatic => {
                for axis in 0..2 {
                    if (self.initial_size[axis] - self.domain[axis]).abs() > 1e-9 {
                        return Err(format!(
                            "`hydrostatic` requires `initial_size_m[{axis}]` to equal `domain_m[{axis}]` so the column spans the domain"
                        ));
                    }
                }
                if self.initial_velocity.norm() > 0.0 {
                    return Err("`hydrostatic` requires zero `initial_velocity_m_s`".to_string());
                }
                if self.initial_size[2] >= self.domain[2] {
                    return Err("`hydrostatic` requires free space above the column".to_string());
                }
            }
            Fixture::CoupledPatch => {
                let Some(pellet) = self.pellet else {
                    return Err(
                        "`coupled_patch` requires exactly one box supplying pellet geometry"
                            .to_string(),
                    );
                };
                if self.material.kind != MaterialKind::Paste {
                    return Err("`coupled_patch` is implemented for `paste` only".to_string());
                }
                if pellet.radius < MIN_CELLS_PER_PELLET_RADIUS * self.grid_spacing {
                    return Err(format!(
                        "`pellet_radius_m`='{}' is under-resolved: coupling needs at least {MIN_CELLS_PER_PELLET_RADIUS} cells per radius at `grid_spacing_m`='{}'",
                        pellet.radius, self.grid_spacing
                    ));
                }
                let top = self.initial_size[2]
                    + PELLET_GAP_CELLS * self.grid_spacing
                    + 2.0 * pellet.radius;
                if top > self.domain[2]
                    || pellet.length > self.domain[0]
                    || 2.0 * pellet.radius > self.domain[1]
                {
                    return Err(
                        "`coupled_patch` pellet does not fit above the block inside `domain_m`"
                            .to_string(),
                    );
                }
            }
            Fixture::Slump | Fixture::DamBreak => {
                if self.pellet.is_some() {
                    return Err(format!(
                        "`{}` does not accept boxes; only `coupled_patch` uses pellet geometry",
                        self.fixture.name()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Build the simulation.
    ///
    /// Returns: `(Simulation, SplitMix64)` with the generator state after placement.
    ///
    /// # Errors
    ///
    /// layout or geometry errors as `String`.
    pub fn build(&self) -> Result<(Simulation, SplitMix64), String> {
        self.validate()?;
        let layout = GridLayout::new(self.domain, self.grid_spacing, self.limits.max_nodes)?;
        let mut rng = SplitMix64::new(self.seed);
        let particles = self.place_particles(&mut rng);
        let pellet = match (self.fixture, self.pellet) {
            (Fixture::CoupledPatch, Some(spec)) => {
                let center = Vector3::new(
                    0.5 * self.domain[0],
                    0.5 * self.domain[1],
                    self.initial_size[2] + PELLET_GAP_CELLS * self.grid_spacing + spec.radius,
                );
                Some(Pellet::new(
                    spec.radius,
                    spec.length,
                    spec.density,
                    center,
                    UnitQuaternion::identity(),
                )?)
            }
            _ => None,
        };
        let simulation = Simulation::new(
            self.material,
            particles,
            Grid::new(layout),
            pellet,
            self.config,
        );
        Ok((simulation, rng))
    }

    /// Place the block particles, pre-stressed for the hydrostatic fixture.
    #[allow(clippy::cast_precision_loss)] // lattice indices are small integers.
    fn place_particles(&self, rng: &mut SplitMix64) -> ParticleSet {
        let counts = self.axis_counts();
        let sub = self.grid_spacing / PARTICLES_PER_AXIS as f64;
        let origin = self.block_origin();
        let jitter = if self.fixture == Fixture::Hydrostatic {
            0.0
        } else {
            JITTER_FRACTION * sub
        };
        let mut particles = ParticleSet::with_capacity(self.particle_count());
        let height = self.initial_size[2];
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let mut position = origin
                        + Vector3::new(
                            (i as f64 + 0.5) * sub,
                            (j as f64 + 0.5) * sub,
                            (k as f64 + 0.5) * sub,
                        );
                    if jitter > 0.0 {
                        for axis in 0..3 {
                            position[axis] += (rng.next_unit() - 0.5) * 2.0 * jitter;
                        }
                    }
                    let current_volume = sub * sub * sub;
                    let (volume_ratio, deformation, kirchhoff) =
                        if self.fixture == Fixture::Hydrostatic {
                            self.hydrostatic_state(height - position.z)
                        } else {
                            (1.0, Matrix3::identity(), Matrix3::zeros())
                        };
                    let volume0 = current_volume / volume_ratio;
                    particles.push(
                        position,
                        self.initial_velocity,
                        self.material.density * volume0,
                        volume0,
                    );
                    let index = particles.len() - 1;
                    particles.volume_ratio[index] = volume_ratio;
                    particles.deformation[index] = deformation;
                    particles.kirchhoff[index] = kirchhoff;
                }
            }
        }
        particles
    }

    /// Exact static state at a depth below the free surface: `(J, F, tau)`.
    fn hydrostatic_state(&self, depth: f64) -> (f64, Matrix3<f64>, Matrix3<f64>) {
        let gravity = self.config.gravity.norm();
        let material = &self.material;
        match material.kind {
            MaterialKind::Liquid => {
                let volume_ratio =
                    (-material.density * gravity * depth / material.bulk_modulus).exp();
                let pressure = liquid_pressure(material, volume_ratio);
                (
                    volume_ratio,
                    Matrix3::identity(),
                    Matrix3::identity() * (-pressure * volume_ratio),
                )
            }
            MaterialKind::Paste => {
                let modulus = material.lame_lambda + 2.0 * material.shear_modulus;
                let strain = -material.density * gravity * depth / modulus;
                let stretch = strain.exp();
                let deformation = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, stretch));
                let kirchhoff = Matrix3::from_diagonal(&Vector3::new(
                    material.lame_lambda * strain,
                    material.lame_lambda * strain,
                    modulus * strain,
                ));
                (stretch, deformation, kirchhoff)
            }
        }
    }

    /// Exact hydrostatic reference at a depth: pressure (liquid) or vertical
    /// compressive stress magnitude (paste), in Pa.
    #[must_use]
    pub fn hydrostatic_reference(&self, depth: f64) -> f64 {
        let gravity = self.config.gravity.norm();
        let material = &self.material;
        match material.kind {
            MaterialKind::Liquid => {
                material.bulk_modulus
                    * ((material.density * gravity * depth / material.bulk_modulus).exp() - 1.0)
            }
            MaterialKind::Paste => material.density * gravity * depth,
        }
    }

    /// Fixture observables as a flat, sorted map of finite numbers.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // counts are reported as f64 for JSON.
    pub fn observables(&self, sim: &Simulation) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        let particles = &sim.particles;
        let sub = self.grid_spacing / PARTICLES_PER_AXIS as f64;
        out.insert("mass_residual_kg".into(), sim.mass_residual());
        out.insert(
            "momentum_residual_kg_m_s".into(),
            sim.momentum_residual().norm(),
        );
        out.insert("kinetic_energy_j".into(), sim.kinetic_energy());
        out.insert("potential_energy_j".into(), sim.potential_energy());
        out.insert("elastic_energy_j".into(), sim.elastic_energy());
        out.insert("mechanical_energy_j".into(), sim.mechanical_energy());
        out.insert(
            "plastic_dissipation_j".into(),
            sim.ledger.plastic_dissipation,
        );
        out.insert("wall_work_j".into(), sim.ledger.wall_work);
        out.insert("outflow_kg".into(), sim.ledger.outflow_mass);
        out.insert("particle_count".into(), particles.len() as f64);
        out.insert("step_count".into(), sim.step_count as f64);
        out.insert("rejected_steps".into(), sim.rejected_steps as f64);
        out.insert(
            "min_dt_s".into(),
            if sim.min_dt_used.is_finite() {
                sim.min_dt_used
            } else {
                0.0
            },
        );
        out.insert("max_dt_s".into(), sim.max_dt_used);
        let max_speed = particles
            .velocity
            .iter()
            .fold(0.0f64, |m, v| m.max(v.norm()));
        out.insert("max_speed_m_s".into(), max_speed);
        let max_plastic = particles
            .plastic_strain
            .iter()
            .fold(0.0f64, |m, &e| m.max(e));
        out.insert("max_plastic_strain".into(), max_plastic);
        let yielded = particles
            .plastic_strain
            .iter()
            .filter(|&&e| e > 0.0)
            .count();
        out.insert(
            "yielded_fraction".into(),
            if particles.is_empty() {
                0.0
            } else {
                yielded as f64 / particles.len() as f64
            },
        );
        let (height, max_x) = self.shape_observables(sim, &mut out);
        match self.fixture {
            Fixture::Slump => self.slump_observables(sim, height, &mut out),
            Fixture::Hydrostatic => self.hydrostatic_observables(sim, &mut out),
            Fixture::DamBreak => {
                let front = if particles.is_empty() {
                    0.0
                } else {
                    max_x + 0.5 * sub
                };
                out.insert("front_x_m".into(), front);
                let g = self.config.gravity.norm();
                let ritter =
                    self.initial_size[0] + 2.0 * (g * self.initial_size[2]).sqrt() * sim.time;
                out.insert("ritter_front_bound_m".into(), ritter.min(self.domain[0]));
                let column = particles
                    .position
                    .iter()
                    .filter(|x| x.x < self.grid_spacing)
                    .fold(0.0f64, |m, x| m.max(x.z + 0.5 * sub));
                out.insert("column_height_m".into(), column);
            }
            Fixture::CoupledPatch => Self::coupled_observables(sim, self.initial_size[2], &mut out),
        }
        out
    }

    /// Record observable extent without mistaking particle centres for surfaces.
    fn shape_observables(&self, sim: &Simulation, out: &mut BTreeMap<String, f64>) -> (f64, f64) {
        let particles = &sim.particles;
        let sub = self.grid_spacing / 2.0;
        let (min_x, max_x, min_y, max_y, max_z) = particles.position.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ),
            |acc, x| {
                (
                    acc.0.min(x.x),
                    acc.1.max(x.x),
                    acc.2.min(x.y),
                    acc.3.max(x.y),
                    acc.4.max(x.z),
                )
            },
        );
        let height = if particles.is_empty() {
            0.0
        } else {
            max_z + 0.5 * sub
        };
        out.insert("height_m".into(), height);
        out.insert(
            "spread_x_m".into(),
            if particles.is_empty() {
                0.0
            } else {
                max_x - min_x + sub
            },
        );
        out.insert(
            "spread_y_m".into(),
            if particles.is_empty() {
                0.0
            } else {
                max_y - min_y + sub
            },
        );
        (height, max_x)
    }

    fn slump_observables(&self, sim: &Simulation, height: f64, out: &mut BTreeMap<String, f64>) {
        let h0 = self.initial_size[2];
        out.insert("slump_m".into(), h0 - height);
        let g = self.config.gravity.norm();
        // Pashias et al. 1996 cylinder estimate with the shear yield stress.
        let shear_yield = self.material.yield_stress / 3f64.sqrt();
        let tau_prime = shear_yield / (self.material.density * g * h0);
        out.insert("dimensionless_yield_stress".into(), tau_prime);
        let pashias = if tau_prime >= 0.5 {
            0.0
        } else if tau_prime <= 0.0 {
            h0
        } else {
            h0 * (1.0 - 2.0 * tau_prime * (1.0 - (2.0 * tau_prime).ln()))
        };
        out.insert("pashias_slump_m".into(), pashias);
        let _ = sim;
    }

    fn hydrostatic_observables(&self, sim: &Simulation, out: &mut BTreeMap<String, f64>) {
        let particles = &sim.particles;
        let h0 = self.initial_size[2];
        let scale = self.material.density * self.config.gravity.norm() * h0;
        let mut sum_sq = 0.0;
        let mut max_err: f64 = 0.0;
        for p in 0..particles.len() {
            let depth = (h0 - particles.position[p].z).max(0.0);
            let reference = self.hydrostatic_reference(depth);
            let measured = match self.material.kind {
                MaterialKind::Liquid => liquid_pressure(&self.material, particles.volume_ratio[p]),
                // Kirchhoff vertical stress; the confined-compression reference is the
                // small-strain solution, exact to O(strain).
                MaterialKind::Paste => -particles.kirchhoff[p][(2, 2)],
            };
            let error = (measured - reference) / scale;
            sum_sq += error * error;
            max_err = max_err.max(error.abs());
        }
        #[allow(clippy::cast_precision_loss)]
        let rms = if particles.is_empty() {
            0.0
        } else {
            (sum_sq / particles.len() as f64).sqrt()
        };
        out.insert("hydrostatic_rms_relative_error".into(), rms);
        out.insert("hydrostatic_max_relative_error".into(), max_err);
        out.insert(
            "reference_bottom_stress_pa".into(),
            self.hydrostatic_reference(h0),
        );
        out.insert("wave_speed_m_s".into(), self.material.wave_speed());
    }

    fn coupled_observables(sim: &Simulation, block_height: f64, out: &mut BTreeMap<String, f64>) {
        let Some(pellet) = &sim.pellet else {
            return;
        };
        out.insert("pellet_x_m".into(), pellet.position.x);
        out.insert("pellet_y_m".into(), pellet.position.y);
        out.insert("pellet_z_m".into(), pellet.position.z);
        out.insert("pellet_speed_m_s".into(), pellet.velocity.norm());
        out.insert(
            "pellet_angular_speed_rad_s".into(),
            pellet.angular_velocity().norm(),
        );
        out.insert("pellet_kinetic_energy_j".into(), pellet.kinetic_energy());
        out.insert(
            "pellet_penetration_m".into(),
            (block_height + pellet.radius - pellet.position.z).max(0.0),
        );
        let axis = pellet.orientation * Vector3::x();
        out.insert("pellet_axis_tilt_rad".into(), axis.z.abs().min(1.0).asin());
        let ledger = &sim.ledger;
        for (name, value) in [
            ("coupling_impulse_x_kg_m_s", ledger.coupling_impulse.x),
            ("coupling_impulse_y_kg_m_s", ledger.coupling_impulse.y),
            ("coupling_impulse_z_kg_m_s", ledger.coupling_impulse.z),
            (
                "coupling_angular_impulse_x_kg_m2_s",
                ledger.coupling_angular_impulse.x,
            ),
            (
                "coupling_angular_impulse_y_kg_m2_s",
                ledger.coupling_angular_impulse.y,
            ),
            (
                "coupling_angular_impulse_z_kg_m2_s",
                ledger.coupling_angular_impulse.z,
            ),
            ("pellet_wall_impulse_z_kg_m_s", ledger.pellet_wall_impulse.z),
            ("pellet_wall_work_j", ledger.pellet_wall_work),
        ] {
            out.insert(name.into(), value);
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::{Fixture, FixtureSpec, PelletSpec, ResourceLimits};
    use crate::mpm::constitutive::Material;
    use crate::mpm::rigid::ContactParams;
    use crate::mpm::solver::SolverConfig;
    use nalgebra::Vector3;

    fn spec(fixture: Fixture, material: Material) -> FixtureSpec {
        FixtureSpec {
            fixture,
            material,
            grid_spacing: 0.002,
            domain: [0.02, 0.02, 0.02],
            initial_size: [0.02, 0.02, 0.01],
            initial_velocity: Vector3::zeros(),
            seed: 1,
            pellet: None,
            config: SolverConfig {
                gravity: Vector3::new(0.0, 0.0, -9.81),
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
            limits: ResourceLimits {
                max_particles: 100_000,
                max_nodes: 1_000_000,
            },
        }
    }

    #[test]
    fn particle_count_and_caps_are_checked_before_allocation() {
        let mut s = spec(
            Fixture::Slump,
            Material::paste(1000.0, 1e4, 0.3, 10.0, 1.0, 1.0),
        );
        s.initial_size = [0.01, 0.01, 0.01];
        assert_eq!(s.particle_count(), 1000);
        s.limits.max_particles = 999;
        assert!(s.validate().unwrap_err().contains("exceeding the cap"));
    }

    #[test]
    fn hydrostatic_requires_full_width_column_and_rest() {
        let material = Material::liquid(1000.0, 1e5, 0.2, 1e-3);
        let mut s = spec(Fixture::Hydrostatic, material);
        assert!(s.validate().is_ok());
        s.initial_size[0] = 0.01;
        assert!(s.validate().unwrap_err().contains("spans the domain"));
        let mut s = spec(Fixture::Hydrostatic, material);
        s.initial_velocity = Vector3::new(0.0, 0.0, 0.1);
        assert!(s.validate().unwrap_err().contains("zero"));
    }

    #[test]
    fn hydrostatic_initial_state_matches_exact_profile() {
        let material = Material::liquid(1000.0, 1e5, 0.2, 1e-3);
        let s = spec(Fixture::Hydrostatic, material);
        let (sim, _) = s.build().unwrap_or_else(|e| panic!("{e}"));
        let obs = s.observables(&sim);
        assert!(obs["hydrostatic_max_relative_error"] < 1e-12);
        assert!(obs["mass_residual_kg"].abs() < 1e-18);
        let paste = Material::paste(1000.0, 1e5, 0.3, 1e4, 1.0, 1.0);
        let s = spec(Fixture::Hydrostatic, paste);
        let (sim, _) = s.build().unwrap_or_else(|e| panic!("{e}"));
        let obs = s.observables(&sim);
        assert!(obs["hydrostatic_max_relative_error"] < 1e-9);
    }

    #[test]
    fn coupled_patch_needs_paste_and_resolved_pellet() {
        let pellet = PelletSpec {
            radius: 0.003,
            length: 0.012,
            density: 1100.0,
        };
        let mut s = spec(
            Fixture::CoupledPatch,
            Material::liquid(1000.0, 1e5, 0.2, 1e-3),
        );
        s.domain = [0.03, 0.03, 0.03];
        s.initial_size = [0.02, 0.02, 0.008];
        assert!(
            s.validate()
                .unwrap_err()
                .contains("requires exactly one box")
        );
        s.pellet = Some(pellet);
        assert!(s.validate().unwrap_err().contains("`paste` only"));
        s.material = Material::paste(1000.0, 1e4, 0.3, 10.0, 1.0, 1.0);
        assert!(s.validate().is_ok());
        s.grid_spacing = 0.003;
        assert!(s.validate().unwrap_err().contains("under-resolved"));
        let mut slump = spec(
            Fixture::Slump,
            Material::paste(1000.0, 1e4, 0.3, 10.0, 1.0, 1.0),
        );
        slump.pellet = Some(pellet);
        assert!(
            slump
                .validate()
                .unwrap_err()
                .contains("does not accept boxes")
        );
    }

    #[test]
    fn pashias_reference_matches_closed_form() {
        // tau' = 0.1 -> s/H = 1 - 0.2 (1 - ln 0.2) = 0.4781...
        let h0 = 0.02;
        let shear_yield = 0.1 * 1000.0 * 9.81 * h0;
        let material = Material::paste(1000.0, 1e4, 0.3, shear_yield * 3f64.sqrt(), 1.0, 1.0);
        let mut s = spec(Fixture::Slump, material);
        s.initial_size = [0.01, 0.01, h0];
        let (sim, _) = s.build().unwrap_or_else(|e| panic!("{e}"));
        let obs = s.observables(&sim);
        let expected = h0 * (1.0 - 0.2 * (1.0 - 0.2f64.ln()));
        assert!((obs["pashias_slump_m"] - expected).abs() < 1e-12);
        assert!((obs["dimensionless_yield_stress"] - 0.1).abs() < 1e-12);
    }
}
