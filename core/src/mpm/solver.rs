//! MLS-MPM time stepping with conservative ledgers and step rejection.
//!
//! One step is: particle-to-grid (APIC affine momentum plus the MLS stress
//! force), grid update (gravity, Coulomb walls, pellet coupling), pellet
//! integration, grid-to-particle (velocity, affine matrix, advection,
//! constitutive update). Every transfer is a fixed-order reduction so results are
//! bit-reproducible for a given thread-independent particle order: P2G is
//! serial over sorted particles, grid and G2P work is particle/node parallel
//! with per-element outputs committed in index order.
//!
//! A step is attempted into scratch buffers and committed only if every
//! constitutive update succeeded and no particle moved more than one cell;
//! otherwise it is rejected, `dt` halves, and the step is retried down to a
//! minimum below which the solver aborts with diagnostics.
//!
//! References:
//! - Hu et al. 2018, <https://doi.org/10.1145/3197517.3201293>
//! - Jiang et al. 2015, <https://doi.org/10.1145/2766996>

use std::fmt;

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use super::constitutive::{
    ConstitutiveError, Material, MaterialKind, symmetric_eigen3, update_liquid, update_paste,
};
use super::grid::Grid;
use super::particles::ParticleSet;
use super::rigid::{ContactParams, Pellet, couple_grid};

/// Steps between cache-locality particle sorts.
const SORT_INTERVAL: u64 = 32;
/// Growth factor of the step scale after an accepted step.
const DT_RECOVERY: f64 = 1.25;

/// Solver configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Gravitational acceleration in m/s^2.
    pub gravity: Vector3<f64>,
    /// Coulomb friction coefficient of the domain walls.
    pub wall_friction: f64,
    /// CFL number applied to `dx / (c + |v|max)`.
    pub cfl: f64,
    /// Largest permitted step in s (wire `dt_s`).
    pub max_dt: f64,
    /// Smallest permitted step in s before aborting.
    pub min_dt: f64,
    /// Spring-dashpot parameters for pellet/wall contact.
    pub contact: ContactParams,
}

/// Conservation ledger accumulated across the run.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ledger {
    /// Initial material mass in kg.
    pub initial_mass: f64,
    /// Mass removed to the outflow compartment in kg.
    pub outflow_mass: f64,
    /// Initial momentum of material plus pellet in kg m/s.
    pub initial_momentum: Vector3<f64>,
    /// Cumulative gravity impulse on material plus pellet in kg m/s.
    pub gravity_impulse: Vector3<f64>,
    /// Cumulative impulse from the walls on the material in kg m/s.
    pub wall_impulse: Vector3<f64>,
    /// Cumulative impulse from the walls on the pellet in kg m/s.
    pub pellet_wall_impulse: Vector3<f64>,
    /// Cumulative coupling impulse delivered to the pellet in kg m/s.
    pub coupling_impulse: Vector3<f64>,
    /// Cumulative coupling angular impulse about the pellet centre in kg m^2/s.
    pub coupling_angular_impulse: Vector3<f64>,
    /// Work done by the walls on the material in J (non-positive).
    pub wall_work: f64,
    /// Work done by the walls on the pellet in J.
    pub pellet_wall_work: f64,
    /// Plastic dissipation in J.
    pub plastic_dissipation: f64,
    /// Cumulative momentum lost with outflow particles in kg m/s.
    pub outflow_momentum: Vector3<f64>,
}

/// Solver failure. Numerical failures carry diagnostics for the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// The step size fell below the minimum after repeated rejections.
    MinimumStepReached {
        /// Simulation time in s at failure.
        time: f64,
        /// Last attempted step in s.
        dt: f64,
        /// Rejections so far.
        rejected_steps: u64,
        /// Reason of the last rejection.
        reason: String,
    },
    /// State validation failed.
    InvalidState(String),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinimumStepReached {
                time,
                dt,
                rejected_steps,
                reason,
            } => write!(
                f,
                "solver aborted at t='{time}' s: step '{dt}' s below minimum after '{rejected_steps}' rejections; last failure: {reason}"
            ),
            Self::InvalidState(message) => write!(f, "invalid solver state: {message}"),
        }
    }
}

impl std::error::Error for SolverError {}

/// Reason a trial step was rejected.
#[derive(Debug, Clone, PartialEq)]
enum StepFailure {
    Constitutive(ConstitutiveError),
    NonFiniteGrid,
    Displacement { max_cells: f64 },
    PelletNonFinite,
}

impl fmt::Display for StepFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constitutive(error) => write!(f, "constitutive update failed: {error}"),
            Self::NonFiniteGrid => write!(f, "non-finite grid velocity"),
            Self::Displacement { max_cells } => {
                write!(f, "particle moved '{max_cells}' cells in one step")
            }
            Self::PelletNonFinite => write!(f, "pellet state became non-finite"),
        }
    }
}

/// Per-particle trial result of grid-to-particle plus constitutive update.
#[derive(Debug, Clone)]
struct ParticleTrial {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    affine: Matrix3<f64>,
    deformation: Matrix3<f64>,
    volume_ratio: f64,
    kirchhoff: Matrix3<f64>,
    plastic_increment: f64,
    dissipation: f64,
    displacement_cells: f64,
}

/// Per-node trial result of the grid update.
#[derive(Debug, Clone, Copy)]
struct NodeTrial {
    velocity: Vector3<f64>,
    wall_impulse: Vector3<f64>,
    wall_work: f64,
}

/// Simulation state: material, particles, grid, optional pellet, ledgers.
#[derive(Debug, Clone, PartialEq)]
pub struct Simulation {
    /// Material parameters.
    pub material: Material,
    /// Material points.
    pub particles: ParticleSet,
    /// Background grid (scratch between steps).
    pub grid: Grid,
    /// Optional rigid pellet.
    pub pellet: Option<Pellet>,
    /// Solver configuration.
    pub config: SolverConfig,
    /// Conservation ledger.
    pub ledger: Ledger,
    /// Simulation time in s.
    pub time: f64,
    /// Accepted steps.
    pub step_count: u64,
    /// Accepted steps whose stability/recovery limit was below the requested cap,
    /// evaluated before endpoint clipping. This is not inferred from `max_dt_used`.
    pub limited_steps: u64,
    /// Rejected steps.
    pub rejected_steps: u64,
    /// Current step scale in `(0, 1]` applied to the stability limit.
    pub dt_scale: f64,
    /// Smallest accepted step in s.
    pub min_dt_used: f64,
    /// Largest accepted step in s.
    pub max_dt_used: f64,
}

impl Simulation {
    /// Assemble a simulation and seed its ledger from the current state.
    #[must_use]
    pub fn new(
        material: Material,
        particles: ParticleSet,
        grid: Grid,
        pellet: Option<Pellet>,
        config: SolverConfig,
    ) -> Self {
        let mut momentum = particles.linear_momentum();
        if let Some(pellet) = &pellet {
            momentum += pellet.velocity * pellet.mass;
        }
        let ledger = Ledger {
            initial_mass: particles.total_mass(),
            initial_momentum: momentum,
            ..Ledger::default()
        };
        Self {
            material,
            particles,
            grid,
            pellet,
            config,
            ledger,
            time: 0.0,
            step_count: 0,
            limited_steps: 0,
            rejected_steps: 0,
            dt_scale: 1.0,
            min_dt_used: f64::INFINITY,
            max_dt_used: 0.0,
        }
    }

    /// Advance to `target_time` with adaptive substeps.
    ///
    /// Parameters:
    /// - `target_time` (`f64`): absolute simulation time in s, at or after `self.time`.
    ///
    /// Returns: `Result<(), SolverError>`.
    ///
    /// # Errors
    ///
    /// [`SolverError::MinimumStepReached`] after unrecoverable rejections.
    pub fn advance_to(&mut self, target_time: f64) -> Result<(), SolverError> {
        self.advance_while(target_time, || true).map(|_| ())
    }

    /// Advance while the caller's budget permits another adaptive substep.
    ///
    /// Returns `false` at a safe resumable boundary when the budget expires.
    ///
    /// # Errors
    ///
    /// Rejects invalid times or unrecoverable numerical steps.
    pub fn advance_while(
        &mut self,
        target_time: f64,
        mut should_continue: impl FnMut() -> bool,
    ) -> Result<bool, SolverError> {
        if !target_time.is_finite() || target_time < self.time {
            return Err(SolverError::InvalidState("invalid target time".into()));
        }
        while self.time < target_time {
            if !should_continue() {
                return Ok(false);
            }
            let remaining = target_time - self.time;
            let limit = self.stability_limit();
            let scaled_limit = self.dt_scale * limit;
            let limited = scaled_limit < self.config.max_dt;
            let mut dt = scaled_limit.min(self.config.max_dt).min(remaining);
            if remaining - dt < 1e-12 * target_time.max(1.0) {
                dt = remaining;
            }
            if !dt.is_finite()
                || dt <= 0.0
                || (dt < self.config.min_dt && remaining > self.config.min_dt)
            {
                return Err(SolverError::MinimumStepReached {
                    time: self.time,
                    dt,
                    rejected_steps: self.rejected_steps,
                    reason: "stability limit is nonfinite, nonpositive or below minimum".into(),
                });
            }
            match self.try_step(dt) {
                Ok(()) => {
                    self.time += dt;
                    self.step_count += 1;
                    self.limited_steps += u64::from(limited);
                    self.min_dt_used = self.min_dt_used.min(dt);
                    self.max_dt_used = self.max_dt_used.max(dt);
                    self.dt_scale = (self.dt_scale * DT_RECOVERY).min(1.0);
                    if self.step_count.is_multiple_of(SORT_INTERVAL) {
                        self.sort_particles();
                    }
                }
                Err(failure) => {
                    self.rejected_steps += 1;
                    self.dt_scale *= 0.5;
                    if self.dt_scale * limit < self.config.min_dt {
                        return Err(SolverError::MinimumStepReached {
                            time: self.time,
                            dt,
                            rejected_steps: self.rejected_steps,
                            reason: failure.to_string(),
                        });
                    }
                }
            }
        }
        self.time = target_time;
        Ok(true)
    }

    /// Stability-limited step in s from the continuum CFL, viscous, and contact
    /// restrictions.
    #[must_use]
    pub fn stability_limit(&self) -> f64 {
        let dx = self.grid.layout.spacing;
        let max_speed = self
            .particles
            .velocity
            .iter()
            .fold(0.0f64, |m, v| m.max(v.norm()));
        let wave = self.material.wave_speed();
        let mut limit = self.config.cfl * dx / (wave + max_speed);
        let diffusivity = self.material.diffusivity();
        if diffusivity > 0.0 {
            limit = limit.min(0.25 * dx * dx / diffusivity);
        }
        if let Some(pellet) = &self.pellet {
            limit = limit.min(self.config.contact.stable_dt(pellet.mass));
            let pellet_speed = pellet.velocity.norm();
            if pellet_speed > 0.0 {
                limit = limit.min(self.config.cfl * dx / pellet_speed);
            }
        }
        limit
    }

    /// Sort particles by grid cell for cache locality (deterministic).
    fn sort_particles(&mut self) {
        let layout = self.grid.layout;
        let mut order: Vec<usize> = (0..self.particles.len()).collect();
        let key = |index: usize| -> usize {
            let position = self.particles.position[index];
            let mut cell = 0usize;
            for axis in 0..3 {
                let scaled = (position[axis] / layout.spacing).floor().max(0.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped >= 0.
                let coordinate = (scaled as usize).min(layout.cells[axis]);
                cell = cell * (layout.cells[axis] + 1) + coordinate;
            }
            cell
        };
        order.sort_by_key(|&index| (key(index), self.particles.id[index]));
        self.particles.permute(&order);
    }

    /// Remove particles that left the padded grid, crediting the outflow ledger.
    fn sweep_outflow(&mut self) {
        let layout = self.grid.layout;
        let escaped: Vec<usize> = (0..self.particles.len())
            .filter(|&i| layout.stencil(&self.particles.position[i]).is_none())
            .collect();
        if escaped.is_empty() {
            return;
        }
        for &i in &escaped {
            self.ledger.outflow_momentum += self.particles.velocity[i] * self.particles.mass[i];
        }
        self.ledger.outflow_mass += self.particles.remove_sorted(&escaped);
    }

    /// Attempt one step of size `dt`; commit only on success.
    fn try_step(&mut self, dt: f64) -> Result<(), StepFailure> {
        self.sweep_outflow();
        self.particle_to_grid(dt);
        let node_trials = self.grid_update(dt)?;
        let mut grid_wall_impulse = Vector3::zeros();
        let mut grid_wall_work = 0.0;
        for (slot, trial) in self.grid.active.iter().zip(&node_trials) {
            self.grid.momentum[*slot] = trial.velocity;
            grid_wall_impulse += trial.wall_impulse;
            grid_wall_work += trial.wall_work;
        }
        let mut pellet_trial = self.pellet.clone();
        let mut coupling = None;
        if let Some(pellet) = pellet_trial.as_mut() {
            let result = couple_grid(&mut self.grid, pellet, self.config.wall_friction);
            pellet.apply_impulse(&result.impulse, &result.angular_impulse);
            let extents = self.grid.layout.extents();
            let wall = pellet.integrate(dt, &self.config.gravity, &extents, &self.config.contact);
            if !pellet.is_finite() {
                return Err(StepFailure::PelletNonFinite);
            }
            coupling = Some((result, wall));
        }
        let trials = self.grid_to_particle(dt)?;
        let max_cells = trials
            .iter()
            .fold(0.0f64, |m, t| m.max(t.displacement_cells));
        if max_cells > 1.0 {
            return Err(StepFailure::Displacement { max_cells });
        }
        self.commit(trials, dt, grid_wall_impulse, grid_wall_work);
        if let (Some(pellet), Some((result, wall))) = (pellet_trial, coupling) {
            self.ledger.gravity_impulse += self.config.gravity * (pellet.mass * dt);
            self.ledger.pellet_wall_impulse += wall.impulse;
            self.ledger.pellet_wall_work += wall.work;
            self.ledger.coupling_impulse += result.impulse;
            self.ledger.coupling_angular_impulse += result.angular_impulse;
            self.pellet = Some(pellet);
        }
        Ok(())
    }

    /// APIC/MLS particle-to-grid transfer (serial, fixed order).
    fn particle_to_grid(&mut self, dt: f64) {
        self.grid.clear();
        let layout = self.grid.layout;
        let dx2_inv = 4.0 / (layout.spacing * layout.spacing);
        for p in 0..self.particles.len() {
            let position = self.particles.position[p];
            let Some(stencil) = layout.stencil(&position) else {
                continue;
            };
            let mass = self.particles.mass[p];
            let velocity = self.particles.velocity[p];
            let affine_momentum = self.particles.affine[p] * mass
                - self.particles.kirchhoff[p] * (dt * dx2_inv * self.particles.volume0[p]);
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let weight = stencil.weight(a, b, c);
                        let distance = stencil.distance(a, b, c);
                        let index = layout.index(
                            stencil.base[0] + a,
                            stencil.base[1] + b,
                            stencil.base[2] + c,
                        );
                        let momentum = (velocity * mass + affine_momentum * distance) * weight;
                        self.grid.deposit(index, weight * mass, momentum);
                    }
                }
            }
        }
    }

    /// Grid velocity update with gravity and Coulomb walls (node parallel).
    fn grid_update(&self, dt: f64) -> Result<Vec<NodeTrial>, StepFailure> {
        let layout = self.grid.layout;
        let gravity = self.config.gravity;
        let friction = self.config.wall_friction;
        let (masses, momenta) = (&self.grid.mass, &self.grid.momentum);
        let trials: Vec<Option<NodeTrial>> = self
            .grid
            .active
            .par_iter()
            .map(|&index| {
                let mass = masses[index];
                if mass <= 0.0 {
                    return Some(NodeTrial {
                        velocity: Vector3::zeros(),
                        wall_impulse: Vector3::zeros(),
                        wall_work: 0.0,
                    });
                }
                let mut velocity = momenta[index] / mass + gravity * dt;
                if !velocity.iter().all(|v| v.is_finite()) {
                    return None;
                }
                let before = velocity;
                let (i, j, k) = layout.coords(index);
                for (axis, coordinate) in [(0, i), (1, j), (2, k)] {
                    let Some(sign) = layout.wall_side(axis, coordinate) else {
                        continue;
                    };
                    let normal_speed = velocity[axis] * sign;
                    if normal_speed >= 0.0 {
                        continue;
                    }
                    velocity[axis] = 0.0;
                    let mut tangential = velocity;
                    tangential[axis] = 0.0;
                    let tangential_speed = tangential.norm();
                    let reduction = friction * (-normal_speed);
                    if tangential_speed <= reduction || tangential_speed < 1e-14 {
                        velocity = Vector3::zeros();
                    } else {
                        velocity = tangential * ((tangential_speed - reduction) / tangential_speed);
                    }
                }
                let wall_impulse = (velocity - before) * mass;
                let wall_work = wall_impulse.dot(&((velocity + before) * 0.5));
                Some(NodeTrial {
                    velocity,
                    wall_impulse,
                    wall_work,
                })
            })
            .collect();
        trials
            .into_iter()
            .collect::<Option<Vec<NodeTrial>>>()
            .ok_or(StepFailure::NonFiniteGrid)
    }

    /// Grid-to-particle transfer, advection and constitutive update (particle parallel).
    fn grid_to_particle(&self, dt: f64) -> Result<Vec<ParticleTrial>, StepFailure> {
        let layout = self.grid.layout;
        let dx2_inv = 4.0 / (layout.spacing * layout.spacing);
        let material = self.material;
        let velocities = &self.grid.momentum;
        let particles = &self.particles;
        let trials: Vec<Result<ParticleTrial, StepFailure>> = (0..particles.len())
            .into_par_iter()
            .map(|p| {
                let position = particles.position[p];
                let Some(stencil) = layout.stencil(&position) else {
                    // Swept before the step; keep the particle frozen for this trial.
                    return Ok(ParticleTrial {
                        position,
                        velocity: particles.velocity[p],
                        affine: particles.affine[p],
                        deformation: particles.deformation[p],
                        volume_ratio: particles.volume_ratio[p],
                        kirchhoff: particles.kirchhoff[p],
                        plastic_increment: 0.0,
                        dissipation: 0.0,
                        displacement_cells: 0.0,
                    });
                };
                let mut velocity = Vector3::zeros();
                let mut affine = Matrix3::zeros();
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            let weight = stencil.weight(a, b, c);
                            let distance = stencil.distance(a, b, c);
                            let index = layout.index(
                                stencil.base[0] + a,
                                stencil.base[1] + b,
                                stencil.base[2] + c,
                            );
                            let node_velocity = velocities[index] * weight;
                            velocity += node_velocity;
                            affine += node_velocity * distance.transpose();
                        }
                    }
                }
                affine *= dx2_inv;
                let new_position = position + velocity * dt;
                let displacement_cells = velocity.norm() * dt / layout.spacing;
                match material.kind {
                    MaterialKind::Paste => {
                        let trial = (Matrix3::identity() + affine * dt) * particles.deformation[p];
                        let update = update_paste(&trial, &material, dt)
                            .map_err(StepFailure::Constitutive)?;
                        Ok(ParticleTrial {
                            position: new_position,
                            velocity,
                            affine,
                            deformation: update.deformation,
                            volume_ratio: 1.0,
                            kirchhoff: update.kirchhoff,
                            plastic_increment: update.plastic_increment,
                            dissipation: update.dissipation_density * particles.volume0[p],
                            displacement_cells,
                        })
                    }
                    MaterialKind::Liquid => {
                        let update =
                            update_liquid(particles.volume_ratio[p], &affine, &material, dt)
                                .map_err(StepFailure::Constitutive)?;
                        Ok(ParticleTrial {
                            position: new_position,
                            velocity,
                            affine,
                            deformation: Matrix3::identity(),
                            volume_ratio: update.volume_ratio,
                            kirchhoff: update.kirchhoff,
                            plastic_increment: 0.0,
                            dissipation: 0.0,
                            displacement_cells,
                        })
                    }
                }
            })
            .collect();
        trials.into_iter().collect()
    }

    /// Commit accepted particle trials and ledger contributions.
    fn commit(
        &mut self,
        trials: Vec<ParticleTrial>,
        dt: f64,
        wall_impulse: Vector3<f64>,
        wall_work: f64,
    ) {
        let mut dissipation = 0.0;
        for (p, trial) in trials.into_iter().enumerate() {
            self.particles.position[p] = trial.position;
            self.particles.velocity[p] = trial.velocity;
            self.particles.affine[p] = trial.affine;
            self.particles.deformation[p] = trial.deformation;
            self.particles.volume_ratio[p] = trial.volume_ratio;
            self.particles.kirchhoff[p] = trial.kirchhoff;
            self.particles.plastic_strain[p] += trial.plastic_increment;
            dissipation += trial.dissipation;
        }
        self.ledger.plastic_dissipation += dissipation;
        self.ledger.wall_impulse += wall_impulse;
        self.ledger.wall_work += wall_work;
        self.ledger.gravity_impulse += self.config.gravity * (self.particles.total_mass() * dt);
    }

    /// Total linear momentum of material plus pellet in kg m/s.
    #[must_use]
    pub fn total_momentum(&self) -> Vector3<f64> {
        let mut momentum = self.particles.linear_momentum();
        if let Some(pellet) = &self.pellet {
            momentum += pellet.velocity * pellet.mass;
        }
        momentum
    }

    /// Momentum balance residual in kg m/s:
    /// `P(t) - P(0) - gravity - walls - pellet walls + outflow`.
    #[must_use]
    pub fn momentum_residual(&self) -> Vector3<f64> {
        let ledger = &self.ledger;
        self.total_momentum() - ledger.initial_momentum - ledger.gravity_impulse
            + ledger.outflow_momentum
            - ledger.wall_impulse
            - ledger.pellet_wall_impulse
    }

    /// Mass balance residual in kg: `m(t) + outflow - m(0)`.
    #[must_use]
    pub fn mass_residual(&self) -> f64 {
        self.particles.total_mass() + self.ledger.outflow_mass - self.ledger.initial_mass
    }

    /// Gravitational potential energy of material plus pellet in J (zero at `z = 0`).
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        let g = self.config.gravity;
        let mut energy: f64 = self
            .particles
            .mass
            .iter()
            .zip(&self.particles.position)
            .map(|(m, x)| -m * g.dot(x))
            .sum();
        if let Some(pellet) = &self.pellet {
            energy -= pellet.mass * g.dot(&pellet.position);
        }
        energy
    }

    /// Stored elastic energy in J (Hencky for paste, `K (J - 1 - ln J)` for liquid).
    #[must_use]
    pub fn elastic_energy(&self) -> f64 {
        let material = &self.material;
        (0..self.particles.len())
            .map(|p| {
                let v0 = self.particles.volume0[p];
                match material.kind {
                    MaterialKind::Paste => {
                        let f = self.particles.deformation[p];
                        let b = f * f.transpose();
                        let Some((eigenvalues, _)) = symmetric_eigen3(&b) else {
                            return f64::NAN;
                        };
                        let hencky = eigenvalues.map(|l| 0.5 * l.max(1e-300).ln());
                        let trace = hencky.sum();
                        v0 * (material.shear_modulus * hencky.norm_squared()
                            + 0.5 * material.lame_lambda * trace * trace)
                    }
                    MaterialKind::Liquid => {
                        let j = self.particles.volume_ratio[p];
                        v0 * material.bulk_modulus * (j - 1.0 - j.ln())
                    }
                }
            })
            .sum()
    }

    /// Kinetic energy of material plus pellet in J.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.particles.kinetic_energy() + self.pellet.as_ref().map_or(0.0, Pellet::kinetic_energy)
    }

    /// Total mechanical energy in J.
    #[must_use]
    pub fn mechanical_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy() + self.elastic_energy()
    }

    /// Validate the whole state (used after resume).
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidState`] with the first inconsistency.
    pub fn validate(&self) -> Result<(), SolverError> {
        self.particles
            .validate()
            .map_err(SolverError::InvalidState)?;
        if !self.time.is_finite() || self.time < 0.0 {
            return Err(SolverError::InvalidState(format!(
                "time '{}' invalid",
                self.time
            )));
        }
        if !(self.dt_scale.is_finite() && self.dt_scale > 0.0 && self.dt_scale <= 1.0) {
            return Err(SolverError::InvalidState(format!(
                "dt_scale '{}' invalid",
                self.dt_scale
            )));
        }
        if let Some(pellet) = &self.pellet
            && !pellet.is_finite()
        {
            return Err(SolverError::InvalidState(
                "pellet state non-finite".to_string(),
            ));
        }
        let layout = self.grid.layout;
        for (i, position) in self.particles.position.iter().enumerate() {
            if layout.stencil(position).is_none() {
                return Err(SolverError::InvalidState(format!(
                    "particle id '{}' lies outside the grid",
                    self.particles.id[i]
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::{Ledger, Simulation, SolverConfig};
    use crate::mpm::constitutive::Material;
    use crate::mpm::grid::{Grid, GridLayout};
    use crate::mpm::particles::ParticleSet;
    use crate::mpm::rigid::ContactParams;
    use nalgebra::Vector3;

    fn config() -> SolverConfig {
        SolverConfig {
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
        }
    }

    fn block(material: Material, spacing: f64, size: [f64; 3], origin: [f64; 3]) -> ParticleSet {
        let mut set = ParticleSet::with_capacity(1024);
        let sub = spacing / 2.0;
        let volume = sub * sub * sub;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let counts = [
            (size[0] / sub).round() as usize,
            (size[1] / sub).round() as usize,
            (size[2] / sub).round() as usize,
        ];
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let position = Vector3::new(
                        origin[0] + (i as f64 + 0.5) * sub,
                        origin[1] + (j as f64 + 0.5) * sub,
                        origin[2] + (k as f64 + 0.5) * sub,
                    );
                    set.push(
                        position,
                        Vector3::zeros(),
                        material.density * volume,
                        volume,
                    );
                }
            }
        }
        set
    }

    #[test]
    fn free_fall_block_matches_kinematics_and_conserves_momentum() {
        let material = Material::paste(1000.0, 1.0e4, 0.3, 1.0e6, 0.0, 1.0);
        let layout =
            GridLayout::new([0.02, 0.02, 0.04], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        let particles = block(
            material,
            0.002,
            [0.008, 0.008, 0.008],
            [0.006, 0.006, 0.028],
        );
        let mut sim = Simulation::new(material, particles, Grid::new(layout), None, config());
        let mass = sim.particles.total_mass();
        let z0 =
            sim.particles.position.iter().map(|x| x.z).sum::<f64>() / sim.particles.len() as f64;
        sim.advance_to(0.03).unwrap_or_else(|e| panic!("{e}"));
        let z =
            sim.particles.position.iter().map(|x| x.z).sum::<f64>() / sim.particles.len() as f64;
        // Free fall until touching the floor at ~ z=0: block bottom at 0.028 - 4.4mm > 0.
        let expected_drop = 0.5 * 9.81 * 0.03f64.powi(2);
        assert!(
            (z0 - z - expected_drop).abs() < 0.03 * expected_drop,
            "drop={}",
            z0 - z
        );
        let momentum = sim.total_momentum();
        assert!((momentum.z + mass * 9.81 * 0.03).abs() < 1e-9 * mass * 9.81 * 0.03);
        assert!(sim.momentum_residual().norm() < 1e-12);
        assert!(sim.mass_residual().abs() < 1e-18);
        assert_eq!(sim.ledger.outflow_mass, 0.0);
        assert_eq!(sim.rejected_steps, 0);
        // No deformation in free fall: stress stays negligible.
        let max_stress = sim
            .particles
            .kirchhoff
            .iter()
            .fold(0.0f64, |m, s| m.max(s.norm()));
        assert!(max_stress < 1e-6, "stress={max_stress}");
    }

    #[test]
    fn block_landing_on_floor_logs_wall_impulse_and_balances_momentum() {
        let material = Material::paste(1000.0, 1.0e5, 0.3, 1.0e6, 0.0, 1.0);
        let layout =
            GridLayout::new([0.02, 0.02, 0.02], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        let particles = block(
            material,
            0.002,
            [0.008, 0.008, 0.006],
            [0.006, 0.006, 0.006],
        );
        let mut sim = Simulation::new(material, particles, Grid::new(layout), None, config());
        sim.advance_to(0.15).unwrap_or_else(|e| panic!("{e}"));
        assert!(sim.ledger.wall_impulse.z > 0.0);
        assert!(sim.ledger.wall_work <= 1e-12);
        let residual = sim.momentum_residual();
        let scale = sim.ledger.gravity_impulse.norm();
        assert!(
            residual.norm() < 1e-9 * scale,
            "residual={residual:?} scale={scale}"
        );
        let energy = sim.mechanical_energy();
        assert!(energy.is_finite());
        assert!(sim.particles.position.iter().all(|x| x.z >= -1e-3));
    }

    #[test]
    fn ledger_default_is_zero() {
        let ledger = Ledger::default();
        assert_eq!(ledger.initial_mass, 0.0);
        assert_eq!(ledger.wall_impulse, Vector3::zeros());
    }
}
