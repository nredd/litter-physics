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
//! Projection/coupling ledgers record discrete kinetic-energy changes, alongside
//! approximate contact-force and plastic-dissipation terms. These are diagnostics,
//! not a closed balance: transfer and time-discretisation losses stay unledgered and
//! appear in [`Simulation::energy_residual`]. See `docs/energy-ledgers.md`.
//!
//! References:
//! - Hu et al. 2018, <https://doi.org/10.1145/3197517.3201293>
//! - Jiang et al. 2015, <https://doi.org/10.1145/2766996>

use std::fmt;

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use super::audit::{EnergyStages, grid_kinetic};
use super::constitutive::{
    ConstitutiveError, Material, MaterialKind, symmetric_eigen3, update_liquid, update_paste,
};
use super::grid::Grid;
use super::particles::ParticleSet;
use super::rigid::{ContactParams, Pellet, couple_grid};

/// Steps between cache-locality particle sorts.
const SORT_INTERVAL: u64 = 32;
/// Wall distance, in cells, inside which a particle's mirror image reaches a
/// free node: the image at `-d` has support `(-d - 1.5 dx, -d + 1.5 dx)`, which
/// contains the first free node at `dx` only for `d < dx / 2`.
const TRACTION_HALF_WIDTH_CELLS: f64 = 0.5;
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
    /// Initial mechanical energy of material plus pellet in J.
    pub initial_mechanical_energy: f64,
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
    /// Kinetic energy removed from the grid by the no-penetration wall
    /// projection in J (non-positive): `sum_i m_i / 2 (|v_i'|^2 - |v_i|^2)` over
    /// the normal component zeroed at each wall node. The stationary walls do no
    /// external work; this lumps inelastic wall impact with the operator-
    /// splitting loss of reacting gravity and stress at wall nodes every step,
    /// which is `O(dt)` for material at rest on a wall. NOT physical work.
    pub wall_normal_projection_energy: f64,
    /// Kinetic energy removed from the grid by the Coulomb tangential reduction
    /// at wall nodes in J (non-negative, like `plastic_dissipation`):
    /// `sum_i m_i / 2 (|t_i|^2 - |t_i'|^2)` over reduced tangential velocities,
    /// equal to minus the friction impulse dotted with the midpoint velocity.
    pub wall_friction_dissipation: f64,
    /// Grid kinetic-energy change from the face-normal wall lattice-completion
    /// deposits of `Simulation::wall_traction_to_grid` in J (signed):
    /// `sum dp (p_a / m + dp / (2 m))` per deposit, evaluated on the momentum
    /// the receiving node holds at the deposit, i.e. after the APIC/MLS
    /// transfer and BEFORE `Simulation::grid_update` adds gravity. Negative
    /// when the reaction cancels momentum already heading into the wall,
    /// positive when the completed lattice adds momentum along the node's
    /// existing normal momentum (a resting column, whose free nodes carry the
    /// upward stress-force momentum that gravity then removes). Sequential
    /// grid-level bookkeeping, NOT physical work of the stationary wall.
    pub wall_normal_traction_energy: f64,
    /// Work of the wall contact force on the pellet in J: `sum F . v_contact dt`
    /// with the start-of-step force and contact-point velocity. Includes
    /// recoverable spring energy, so it is neither dissipation nor external
    /// work of the stationary wall (which is zero).
    pub pellet_wall_work: f64,
    /// Grid kinetic-energy change from the pellet coupling projection in J:
    /// `sum_i m_i / 2 (|v_i'|^2 - |v_i|^2)` over constrained nodes.
    pub coupling_grid_energy: f64,
    /// Pellet kinetic-energy change from the coupling impulses in J, evaluated
    /// exactly before and after [`Pellet::apply_impulse`].
    pub coupling_pellet_energy: f64,
    /// Plastic dissipation in J (non-negative).
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

impl Ledger {
    /// Sum of every ledgered energy exchange in J: wall projection loss, wall
    /// traction jump, pellet wall work and both coupling terms, minus wall
    /// friction and plastic dissipation.
    #[must_use]
    pub fn ledgered_energy(&self) -> f64 {
        self.wall_normal_projection_energy - self.wall_friction_dissipation
            + self.wall_normal_traction_energy
            + self.pellet_wall_work
            + self.coupling_grid_energy
            + self.coupling_pellet_energy
            - self.plastic_dissipation
    }

    /// Check finiteness and the sign invariants each ledger carries by construction.
    ///
    /// # Errors
    ///
    /// [`SolverError::InvalidState`] naming the first violated field.
    pub fn validate(&self) -> Result<(), SolverError> {
        let scalars = [
            ("initial_mass", self.initial_mass),
            ("outflow_mass", self.outflow_mass),
            ("initial_mechanical_energy", self.initial_mechanical_energy),
            (
                "wall_normal_projection_energy",
                self.wall_normal_projection_energy,
            ),
            ("wall_friction_dissipation", self.wall_friction_dissipation),
            (
                "wall_normal_traction_energy",
                self.wall_normal_traction_energy,
            ),
            ("pellet_wall_work", self.pellet_wall_work),
            ("coupling_grid_energy", self.coupling_grid_energy),
            ("coupling_pellet_energy", self.coupling_pellet_energy),
            ("plastic_dissipation", self.plastic_dissipation),
        ];
        for (name, value) in scalars {
            if !value.is_finite() {
                return Err(SolverError::InvalidState(format!(
                    "ledger `{name}` is nonfinite: '{value}'"
                )));
            }
        }
        let vectors = [
            ("initial_momentum", self.initial_momentum),
            ("gravity_impulse", self.gravity_impulse),
            ("wall_impulse", self.wall_impulse),
            ("pellet_wall_impulse", self.pellet_wall_impulse),
            ("coupling_impulse", self.coupling_impulse),
            ("coupling_angular_impulse", self.coupling_angular_impulse),
            ("outflow_momentum", self.outflow_momentum),
        ];
        for (name, value) in vectors {
            if !value.iter().all(|v| v.is_finite()) {
                return Err(SolverError::InvalidState(format!(
                    "ledger `{name}` is nonfinite: '{value:?}'"
                )));
            }
        }
        let non_negative = [
            ("initial_mass", self.initial_mass),
            ("outflow_mass", self.outflow_mass),
            ("plastic_dissipation", self.plastic_dissipation),
            ("wall_friction_dissipation", self.wall_friction_dissipation),
        ];
        for (name, value) in non_negative {
            if value < 0.0 {
                return Err(SolverError::InvalidState(format!(
                    "ledger `{name}` must be non-negative, got '{value}'"
                )));
            }
        }
        if self.wall_normal_projection_energy > 0.0 {
            return Err(SolverError::InvalidState(format!(
                "ledger `wall_normal_projection_energy` must be non-positive, got '{}'",
                self.wall_normal_projection_energy
            )));
        }
        Ok(())
    }
}

/// Reason a trial step was rejected.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum StepFailure {
    Constitutive(ConstitutiveError),
    NonFiniteGrid,
    Displacement {
        max_cells: f64,
    },
    PelletNonFinite,
    /// The traction transfer's energy increment or accumulated history became
    /// non-finite; reject before committing any particle or ledger changes.
    NonFiniteTractionEnergy,
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
            Self::NonFiniteTractionEnergy => {
                write!(f, "wall traction kinetic-energy jump became non-finite")
            }
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
#[derive(Debug, Clone, Copy, Default)]
struct NodeTrial {
    velocity: Vector3<f64>,
    wall_impulse: Vector3<f64>,
    /// Kinetic energy removed by zeroing approaching normal components (`<= 0`).
    normal_projection_energy: f64,
    /// Kinetic energy removed by the tangential Coulomb reduction (`>= 0`).
    friction_dissipation: f64,
}

/// Wall contributions of one accepted step, summed over active nodes.
#[derive(Debug, Clone, Copy, Default)]
struct WallStep {
    impulse: Vector3<f64>,
    normal_projection_energy: f64,
    friction_dissipation: f64,
    /// Signed grid kinetic-energy jump of the traction deposits.
    normal_traction_energy: f64,
}

/// Totals of one wall-traction transfer, accumulated deposit by deposit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct TractionTransfer {
    /// Reaction impulse delivered to the grid in kg m/s.
    impulse: Vector3<f64>,
    /// Signed grid kinetic-energy jump of the deposits in J, see
    /// [`Grid::deposit_reaction`].
    energy: f64,
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
        let mut simulation = Self {
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
        };
        simulation.ledger.initial_mechanical_energy = simulation.mechanical_energy();
        simulation
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
        self.try_step_captured(dt, None)
    }

    /// Shared trial kernel; snapshots are evaluated only for a transient audit.
    pub(super) fn try_step_captured(
        &mut self,
        dt: f64,
        mut audit: Option<&mut EnergyStages>,
    ) -> Result<(), StepFailure> {
        self.sweep_outflow();
        let traction = self.particle_to_grid(dt, audit.as_deref_mut());
        if !traction.energy.is_finite()
            || !(self.ledger.wall_normal_traction_energy + traction.energy).is_finite()
        {
            return Err(StepFailure::NonFiniteTractionEnergy);
        }
        let node_trials = self.grid_update(dt)?;
        let mut wall = WallStep {
            impulse: traction.impulse,
            normal_traction_energy: traction.energy,
            ..WallStep::default()
        };
        for (slot, trial) in self.grid.active.iter().zip(&node_trials) {
            self.grid.momentum[*slot] = trial.velocity;
            wall.impulse += trial.wall_impulse;
            wall.normal_projection_energy += trial.normal_projection_energy;
            wall.friction_dissipation += trial.friction_dissipation;
        }
        if let Some(stages) = audit.as_deref_mut() {
            stages.grid_after_walls = grid_kinetic(&self.grid, true, Vector3::zeros());
        }
        let mut pellet_trial = self.pellet.clone();
        let mut coupling = None;
        if let Some(pellet) = pellet_trial.as_mut() {
            let result = couple_grid(&mut self.grid, pellet, self.config.wall_friction);
            let kinetic_before = pellet.kinetic_energy();
            pellet.apply_impulse(&result.impulse, &result.angular_impulse);
            let pellet_energy = pellet.kinetic_energy() - kinetic_before;
            if let Some(stages) = audit.as_deref_mut() {
                stages.pellet_after_coupling = pellet.kinetic_energy();
            }
            let extents = self.grid.layout.extents();
            let wall = pellet.integrate(dt, &self.config.gravity, &extents, &self.config.contact);
            if !pellet.is_finite() || !pellet_energy.is_finite() {
                return Err(StepFailure::PelletNonFinite);
            }
            coupling = Some((result, pellet_energy, wall));
        }
        if let Some(stages) = audit.as_deref_mut() {
            stages.grid_after_coupling = grid_kinetic(&self.grid, true, Vector3::zeros());
        }
        let trials = self.grid_to_particle(dt)?;
        if let Some(stages) = audit {
            stages.plastic_dissipation = trials
                .iter()
                .fold(0.0, |sum, trial| sum + trial.dissipation);
        }
        let max_cells = trials
            .iter()
            .fold(0.0f64, |m, t| m.max(t.displacement_cells));
        if max_cells > 1.0 {
            return Err(StepFailure::Displacement { max_cells });
        }
        self.commit(trials, dt, &wall);
        if let (Some(pellet), Some((result, pellet_energy, wall))) = (pellet_trial, coupling) {
            self.ledger.gravity_impulse += self.config.gravity * (pellet.mass * dt);
            self.ledger.pellet_wall_impulse += wall.impulse;
            self.ledger.pellet_wall_work += wall.work;
            self.ledger.coupling_impulse += result.impulse;
            self.ledger.coupling_angular_impulse += result.angular_impulse;
            self.ledger.coupling_grid_energy += result.grid_energy;
            self.ledger.coupling_pellet_energy += pellet_energy;
            self.pellet = Some(pellet);
        }
        Ok(())
    }

    /// APIC/MLS particle-to-grid transfer (serial, fixed order), followed by
    /// the wall-traction transfer of [`Self::wall_traction_to_grid`].
    ///
    /// Returns: the wall-traction totals delivered to the grid.
    fn particle_to_grid(&mut self, dt: f64, audit: Option<&mut EnergyStages>) -> TractionTransfer {
        self.deposit_particles(dt);
        if let Some(stages) = audit {
            stages.grid_after_stress = grid_kinetic(&self.grid, false, Vector3::zeros());
            let traction = self.wall_traction_to_grid(dt);
            stages.grid_after_traction = grid_kinetic(&self.grid, false, Vector3::zeros());
            stages.grid_after_gravity = grid_kinetic(&self.grid, false, self.config.gravity * dt);
            traction
        } else {
            self.wall_traction_to_grid(dt)
        }
    }

    /// Clear the scratch grid and deposit APIC momentum plus the MLS stress
    /// impulse of every particle (serial, fixed order).
    fn deposit_particles(&mut self, dt: f64) {
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

    /// Deposit the face-normal wall traction on free nodes whose particle
    /// stencil is truncated by a domain wall.
    ///
    /// The MLS force `f_i = -(4/dx^2) sum_p V0_p tau_p w_ip (x_i - x_p)` is a
    /// quadrature surrogate of `-∫ σ:∇N_i dV`; on the two-per-cell lattice its
    /// balance along axis `a` at node `i` needs the `a`-column sum
    /// `sum_p V_p w_ip (x_i - x_p)_a` to vanish, which fails by one missing
    /// lattice column at the node one cell inside a wall (see
    /// `docs/hydrostatic-balance.md`). The deficit is a lattice-completion
    /// problem, not the continuum face integral `∫_Γ N_i σ n dA`, so it is
    /// repaired in the surrogate's own terms: every particle within
    /// [`TRACTION_HALF_WIDTH_CELLS`] of a wall face emits one image mirrored in
    /// that face alone, and the image deposits only the face-normal component
    /// of its MLS stress force plus its weight. The image stress is the
    /// particle's normal stress `V σ_nn` continued across the face with the
    /// normal momentum balance of material in contact with a stationary wall,
    /// `∂σ_nn/∂n = -ρ g_n` (so `a_n = 0`). A tensile continuation emits no
    /// image at all (no stress, no weight): a wall cannot pull. The deposit is
    /// made only while the constraint is active, i.e. while the wall-plane
    /// node with the same tangential index has an inward trial normal velocity
    /// `p_i / m_i + g_n dt`, the same strict test [`Self::grid_update`] uses to
    /// project it; that node is loaded by the same parent and receives no
    /// normal component from any image, so the test is well defined and
    /// separating material receives nothing. Nothing tangential is ever
    /// deposited: a stress-free body falling or sliding past a wall receives
    /// exactly nothing. An entry whose continued stress force plus image
    /// weight points toward the wall is skipped: a pulling image is not a
    /// reaction, so every booked reaction is inward and the total per node is
    /// unilateral. Each booked reaction is recorded in [`Grid::traction`] and
    /// [`Self::grid_update`] applies the Coulomb law with that budget on the
    /// receiving node's tangential plane, so friction sees the total normal
    /// reaction, projection plus images. Corners
    /// need no multi-face image: the `a`-force lattice at a corner node is
    /// real particles plus `a`-images, symmetric row by row, while the other
    /// axis truncates stress and mass by the same factor.
    ///
    /// Images only reach nodes their parent already loaded, so no massless
    /// node receives momentum, and the component is dropped at nodes that are
    /// wall nodes along that axis: there the no-penetration projection is the
    /// reaction, and the padding node one cell behind the wall carries
    /// `w(1.25)` of a particle mass against the image's `w(0.75)` momentum.
    ///
    /// Limits: this is a nodal surrogate of the contact reaction, not an exact
    /// traction. Material within half a cell of an approaching wall node is
    /// treated as in contact (the projection already does so), stress-free
    /// material arriving at a wall gets the half-cell layer weight as image
    /// stress before real stress builds, and `∂σ_nn/∂n = -ρ g_n` drops the
    /// tangential shear gradients. The normal balance holds for a liquid at
    /// rest; its linear continuation still approximates a varying density.
    ///
    /// Energy: each deposit reports its signed grid kinetic-energy jump
    /// ([`Grid::deposit_reaction`]) and the sum is booked as
    /// [`Ledger::wall_normal_traction_energy`]. It is evaluated exactly where
    /// the code applies the impulse, on the post-APIC momentum before
    /// [`Self::grid_update`] adds `g dt`. It is the jump of only the traction
    /// substep within P2G -> traction -> gravity -> projection, not the jump
    /// a gravity-inclusive velocity would give. Potential energy is included
    /// in mechanical energy, but discrete gravity/transfer consistency remains
    /// unclosed. This is not physical work of the stationary wall.
    ///
    /// Returns: total traction impulse in kg m/s, booked as wall impulse, and
    /// the signed kinetic-energy jump in J.
    fn wall_traction_to_grid(&mut self, dt: f64) -> TractionTransfer {
        let layout = self.grid.layout;
        let extents = layout.extents();
        let half_width = TRACTION_HALF_WIDTH_CELLS * layout.spacing;
        let force_scale = -dt * 4.0 / (layout.spacing * layout.spacing);
        let gravity = self.config.gravity;
        let mut transfer = TractionTransfer::default();
        for p in 0..self.particles.len() {
            let position = self.particles.position[p];
            let mass = self.particles.mass[p];
            let volume0 = self.particles.volume0[p];
            for axis in 0..3 {
                let face = if position[axis] < half_width {
                    0.0
                } else if position[axis] > extents[axis] - half_width {
                    extents[axis]
                } else {
                    continue;
                };
                let mut image = position;
                image[axis] = 2.0 * face - position[axis];
                let Some(stencil) = layout.stencil(&image) else {
                    continue;
                };
                // Wall-plane node along this axis and the inward normal sign.
                let (plane, inward) = if face == 0.0 {
                    (1, 1.0)
                } else {
                    (layout.cells[axis] + 1, -1.0)
                };
                // `V σ_nn` continued to the image point: `V0 τ_nn - ρ V g_n (x_g - x_p)_n`.
                let normal_stress = self.particles.kirchhoff[p][(axis, axis)] * volume0
                    - mass * gravity[axis] * (image[axis] - position[axis]);
                if normal_stress >= 0.0 {
                    // Tensile continuation: a wall cannot pull, so there is no
                    // image material along this face and no image weight either.
                    continue;
                }
                let weight_momentum = mass * gravity[axis] * dt;
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            let coords = [
                                stencil.base[0] + a,
                                stencil.base[1] + b,
                                stencil.base[2] + c,
                            ];
                            if layout.wall_side(axis, coords[axis]).is_some() {
                                continue;
                            }
                            let weight = stencil.weight(a, b, c);
                            if weight == 0.0 {
                                continue;
                            }
                            // Active set: the constraint at the wall-plane node with
                            // the same tangential coordinates must be approaching,
                            // exactly as `grid_update` decides to project it. Its
                            // normal momentum is untouched by image deposits.
                            let mut plane_coords = coords;
                            plane_coords[axis] = plane;
                            let plane_index =
                                layout.index(plane_coords[0], plane_coords[1], plane_coords[2]);
                            let plane_mass = self.grid.mass[plane_index];
                            if plane_mass <= 0.0 {
                                continue;
                            }
                            let trial = self.grid.momentum[plane_index][axis] / plane_mass
                                + gravity[axis] * dt;
                            if trial * inward >= 0.0 {
                                continue;
                            }
                            let distance = stencil.distance(a, b, c)[axis];
                            // Inward-signed reaction; a net pull (weight beating a
                            // vanishing continued stress) is not a wall reaction.
                            let reaction = (normal_stress * distance * force_scale
                                + weight_momentum)
                                * weight
                                * inward;
                            if reaction <= 0.0 {
                                continue;
                            }
                            let index = layout.index(coords[0], coords[1], coords[2]);
                            if let Some(jump) =
                                self.grid.deposit_reaction(index, axis, inward, reaction)
                            {
                                transfer.impulse[axis] += reaction * inward;
                                transfer.energy += jump;
                            }
                        }
                    }
                }
            }
        }
        transfer
    }

    /// Grid velocity update with gravity and Coulomb walls (node parallel).
    fn grid_update(&self, dt: f64) -> Result<Vec<NodeTrial>, StepFailure> {
        let layout = self.grid.layout;
        let gravity = self.config.gravity;
        let friction = self.config.wall_friction;
        let (masses, momenta, tractions) =
            (&self.grid.mass, &self.grid.momentum, &self.grid.traction);
        let trials: Vec<Option<NodeTrial>> = self
            .grid
            .active
            .par_iter()
            .map(|&index| {
                let mass = masses[index];
                if mass <= 0.0 {
                    return Some(NodeTrial::default());
                }
                let mut velocity = momenta[index] / mass + gravity * dt;
                if !velocity.iter().all(|v| v.is_finite()) {
                    return None;
                }
                let before = velocity;
                let mut normal_projection_energy = 0.0;
                let mut friction_dissipation = 0.0;
                let (i, j, k) = layout.coords(index);
                for (axis, coordinate) in [(0, i), (1, j), (2, k)] {
                    let Some(sign) = layout.wall_side(axis, coordinate) else {
                        continue;
                    };
                    let normal_speed = velocity[axis] * sign;
                    if normal_speed >= 0.0 {
                        continue;
                    }
                    // Zeroing the approaching component removes `m v_n^2 / 2` exactly.
                    normal_projection_energy -= 0.5 * mass * normal_speed * normal_speed;
                    velocity[axis] = 0.0;
                    let tangential = velocity;
                    velocity = reduce_tangential_velocity(tangential, friction * (-normal_speed));
                    // The Coulomb reduction shortens the tangential vector, so the
                    // impulse dotted with the midpoint velocity is the exact loss.
                    friction_dissipation +=
                        0.5 * mass * (tangential.norm_squared() - velocity.norm_squared());
                }
                // Coulomb budget of the wall reaction delivered by the traction
                // transfer: the same law as above, on this node's tangential plane.
                for axis in 0..3 {
                    let reaction = tractions[index][axis];
                    if reaction <= 0.0 {
                        continue;
                    }
                    let normal = velocity[axis];
                    let mut tangential = velocity;
                    tangential[axis] = 0.0;
                    let reduced =
                        reduce_tangential_velocity(tangential, friction * reaction / mass);
                    friction_dissipation +=
                        0.5 * mass * (tangential.norm_squared() - reduced.norm_squared());
                    velocity = reduced;
                    velocity[axis] = normal;
                }
                Some(NodeTrial {
                    velocity,
                    wall_impulse: (velocity - before) * mass,
                    normal_projection_energy,
                    friction_dissipation,
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
    fn commit(&mut self, trials: Vec<ParticleTrial>, dt: f64, wall: &WallStep) {
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
        self.ledger.wall_impulse += wall.impulse;
        self.ledger.wall_normal_projection_energy += wall.normal_projection_energy;
        self.ledger.wall_friction_dissipation += wall.friction_dissipation;
        self.ledger.wall_normal_traction_energy += wall.normal_traction_energy;
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

    /// Algebraic diagnostic `E(t) - E(0) - ledgered_energy` in J.
    ///
    /// This mixes particle mechanical energy with grid-level projection and
    /// traction terms and is NOT a physical unexplained-energy closure:
    /// in a resting column, transient grid energy can be introduced and removed
    /// between transfers without appearing in particle mechanical energy.
    /// The resulting nonzero residual is not a measure of physical wall work.
    /// Particle/grid transfer losses, constitutive and pellet
    /// time-discretisation errors, contact spring energy and outflow energy are
    /// not ledgered either.
    #[must_use]
    pub fn energy_residual(&self) -> f64 {
        self.mechanical_energy()
            - self.ledger.initial_mechanical_energy
            - self.ledger.ledgered_energy()
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
        self.ledger.validate()?;
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

/// Apply a nonnegative Coulomb speed budget without reversal or an artificial
/// small-velocity cutoff. A zero budget leaves the tangent exactly unchanged.
fn reduce_tangential_velocity(tangential: Vector3<f64>, reduction: f64) -> Vector3<f64> {
    if reduction == 0.0 {
        return tangential;
    }
    let speed = tangential.norm();
    if speed <= reduction {
        Vector3::zeros()
    } else {
        tangential * ((speed - reduction) / speed)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::{Ledger, Simulation, SolverConfig, StepFailure};
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
        assert!(sim.ledger.wall_normal_projection_energy < 0.0);
        assert!(sim.ledger.wall_friction_dissipation >= 0.0);
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
        assert_eq!(ledger.ledgered_energy(), 0.0);
        ledger.validate().unwrap_or_else(|e| panic!("{e}"));
    }

    /// Empty simulation on a small grid so `grid_update` can be probed node by node.
    fn empty_sim(gravity: Vector3<f64>) -> Simulation {
        let material = Material::paste(1000.0, 1.0e5, 0.3, 1.0e6, 0.0, 1.0);
        let layout =
            GridLayout::new([0.02, 0.02, 0.02], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        let mut cfg = config();
        cfg.gravity = gravity;
        Simulation::new(
            material,
            ParticleSet::with_capacity(0),
            Grid::new(layout),
            None,
            cfg,
        )
    }

    #[test]
    fn wall_decomposition_equals_grid_kinetic_jump_for_slide_stick_and_corner() {
        let mut sim = empty_sim(Vector3::zeros());
        let layout = sim.grid.layout;
        let mass = 1e-6;
        // Floor node, sliding: tangential 1.0 > mu * normal 0.4 * 0.5.
        let slide = layout.index(5, 5, 0);
        // Floor node, sticking: tangential 0.1 < 0.4 * 0.5.
        let stick = layout.index(6, 5, 0);
        // Corner node on the low x and low z faces, approaching both.
        let corner = layout.index(0, 5, 0);
        // Floor node moving away: untouched.
        let free = layout.index(7, 5, 0);
        let velocities = [
            (slide, Vector3::new(1.0, 0.0, -0.5)),
            (stick, Vector3::new(0.1, 0.0, -0.5)),
            (corner, Vector3::new(-0.3, 0.2, -0.4)),
            (free, Vector3::new(0.3, 0.0, 0.2)),
        ];
        for (index, velocity) in velocities {
            sim.grid.deposit(index, mass, velocity * mass);
        }
        let trials = sim.grid_update(1e-3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(trials.len(), 4);
        for ((index, velocity), trial) in velocities.iter().zip(&trials) {
            let jump = 0.5 * mass * (trial.velocity.norm_squared() - velocity.norm_squared());
            let decomposed = trial.normal_projection_energy - trial.friction_dissipation;
            assert!(
                (jump - decomposed).abs() <= 1e-15 * jump.abs().max(1e-12),
                "node {index}: jump={jump} decomposed={decomposed}"
            );
            assert!(trial.normal_projection_energy <= 0.0);
            assert!(trial.friction_dissipation >= 0.0);
            let impulse = (trial.velocity - velocity) * mass;
            assert!((impulse - trial.wall_impulse).norm() < 1e-20);
        }
        // Sliding: normal loss m v_n^2 / 2, tangential reduced by mu v_n.
        assert!((trials[0].normal_projection_energy + 0.5 * mass * 0.25).abs() < 1e-20);
        let expected = 0.5 * mass * (1.0 - 0.8f64.powi(2));
        assert!((trials[0].friction_dissipation - expected).abs() < 1e-20);
        assert!((trials[0].velocity - Vector3::new(0.8, 0.0, 0.0)).norm() < 1e-15);
        // Sticking: the whole tangential kinetic energy is friction dissipation.
        assert!((trials[1].friction_dissipation - 0.5 * mass * 0.01).abs() < 1e-20);
        assert_eq!(trials[1].velocity, Vector3::zeros());
        // Corner: both approaching components are removed; the x projection
        // happens first, and friction shrinks the z approach before its turn.
        assert!(trials[2].velocity.x == 0.0 && trials[2].velocity.z == 0.0);
        assert!(trials[2].normal_projection_energy <= -0.5 * mass * 0.09 + 1e-20);
        assert!(trials[2].normal_projection_energy > -0.5 * mass * (0.09 + 0.16));
        assert!(trials[2].friction_dissipation > 0.0);
        // Receding node: nothing happens.
        assert_eq!(trials[3].normal_projection_energy, 0.0);
        assert_eq!(trials[3].friction_dissipation, 0.0);
        assert_eq!(trials[3].wall_impulse, Vector3::zeros());
    }

    /// Paste column at rest with exact hydrostatic prestress under a fixed step cap.
    fn resting_column(max_dt: f64) -> (crate::mpm::fixtures::FixtureSpec, Simulation) {
        use crate::mpm::fixtures::{Fixture, FixtureSpec, ResourceLimits};
        let mut cfg = config();
        cfg.max_dt = max_dt;
        let spec = FixtureSpec {
            fixture: Fixture::Hydrostatic,
            material: Material::paste(1000.0, 1e5, 0.3, 1e4, 1.0, 1.0),
            grid_spacing: 0.002,
            domain: [0.02, 0.02, 0.02],
            initial_size: [0.02, 0.02, 0.01],
            initial_velocity: Vector3::zeros(),
            seed: 1,
            pellet: None,
            config: cfg,
            limits: ResourceLimits {
                max_particles: 100_000,
                max_nodes: 1_000_000,
            },
        };
        let (sim, _) = spec.build().unwrap_or_else(|e| panic!("{e}"));
        (spec, sim)
    }

    #[test]
    fn resting_column_projection_loss_halves_with_dt_and_is_not_physical_work() {
        let mut losses = Vec::new();
        for max_dt in [4e-5, 2e-5] {
            let (spec, mut sim) = resting_column(max_dt);
            sim.advance_to(0.02).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(sim.limited_steps, 0, "cap must bind for the scaling claim");
            assert_eq!(sim.rejected_steps, 0);
            // The material barely moves, so the stationary walls do no physical
            // work, yet the projection term is orders of magnitude above the KE.
            assert!(sim.ledger.wall_normal_projection_energy < 0.0);
            assert!(
                sim.kinetic_energy() < 1e-3 * sim.ledger.wall_normal_projection_energy.abs(),
                "KE={} loss={}",
                sim.kinetic_energy(),
                sim.ledger.wall_normal_projection_energy
            );
            assert!(sim.ledger.wall_friction_dissipation >= 0.0);
            // Momentum accounting stays exact regardless.
            assert!(sim.momentum_residual().norm() < 1e-9 * sim.ledger.gravity_impulse.norm());
            let obs = spec.observables(&sim);
            assert_eq!(
                obs["wall_normal_projection_energy_j"],
                sim.ledger.wall_normal_projection_energy
            );
            assert!(!obs.contains_key("wall_work_j"));
            // The residual is the grid-level loss that never reached particle energy.
            assert!(obs["energy_residual_j"] > 0.0);
            losses.push(sim.ledger.wall_normal_projection_energy);
        }
        let ratio = losses[0] / losses[1];
        assert!((ratio - 2.0).abs() < 0.1, "ratio={ratio} losses={losses:?}");
    }

    /// Grid kinetic energy `sum |p|^2 / (2 m)` of the momentum-carrying scratch
    /// grid, computed by a full scan independent of the incremental bookkeeping.
    fn grid_momentum_energy(grid: &Grid) -> f64 {
        grid.active
            .iter()
            .map(|&index| 0.5 * grid.momentum[index].norm_squared() / grid.mass[index])
            .sum()
    }

    /// Full before/after grid kinetic-energy jump of the traction transfer on
    /// one trial step, against the incremental sum the transfer reports.
    fn assert_traction_energy_matches_full_scan(sim: &Simulation, dt: f64) -> f64 {
        let mut reference = sim.clone();
        reference.sweep_outflow();
        reference.deposit_particles(dt);
        let before = grid_momentum_energy(&reference.grid);
        let transfer = reference.wall_traction_to_grid(dt);
        let jump = grid_momentum_energy(&reference.grid) - before;
        assert!(transfer.energy.is_finite());
        assert!(
            (transfer.energy - jump).abs() <= 1e-12 * jump.abs().max(1e-30),
            "incremental {} vs full scan {jump}",
            transfer.energy
        );
        transfer.energy
    }

    #[test]
    fn resting_column_books_traction_energy_equal_to_independent_grid_jump() {
        let (spec, mut sim) = resting_column(4e-5);
        let dt = 4e-5;
        let expected = assert_traction_energy_matches_full_scan(&sim, dt);
        assert!(
            expected != 0.0,
            "the resting column must exercise the transfer"
        );
        // The transfer runs before gravity is added in `grid_update`: the free
        // node's pre-gravity momentum already points inward (the stress force
        // reacting gravity), so completing the lattice adds grid kinetic energy.
        assert!(expected > 0.0, "expected a positive jump, got {expected}");
        sim.try_step(dt).expect("accepted step");
        assert_eq!(sim.ledger.wall_normal_traction_energy, expected);
        assert_eq!(
            sim.ledger.ledgered_energy(),
            expected + sim.ledger.wall_normal_projection_energy
                - sim.ledger.wall_friction_dissipation
                - sim.ledger.plastic_dissipation
        );
        // Accumulates only over accepted steps and is exported as an observable.
        let mut total = expected;
        for _ in 0..5 {
            total += assert_traction_energy_matches_full_scan(&sim, dt);
            sim.try_step(dt).expect("accepted step");
        }
        assert_eq!(sim.ledger.wall_normal_traction_energy, total);
        let obs = spec.observables(&sim);
        assert_eq!(obs["wall_normal_traction_energy_j"], total);
        assert_eq!(obs["energy_residual_j"], sim.energy_residual());
    }

    #[test]
    fn coupled_patch_books_traction_energy_from_accepted_steps() {
        use crate::mpm::fixtures::{Fixture, FixtureSpec, PelletSpec, ResourceLimits};
        let mut cfg = config();
        cfg.max_dt = 5e-5;
        let spec = FixtureSpec {
            fixture: Fixture::CoupledPatch,
            material: Material::paste(1000.0, 1e4, 0.3, 10.0, 1.0, 1.0),
            grid_spacing: 0.002,
            domain: [0.03, 0.03, 0.03],
            initial_size: [0.02, 0.02, 0.008],
            initial_velocity: Vector3::zeros(),
            seed: 7,
            pellet: Some(PelletSpec {
                radius: 0.003,
                length: 0.012,
                density: 1100.0,
            }),
            config: cfg,
            limits: ResourceLimits {
                max_particles: 100_000,
                max_nodes: 1_000_000,
            },
        };
        let (mut sim, _) = spec.build().unwrap_or_else(|e| panic!("{e}"));
        sim.advance_to(2e-3).unwrap_or_else(|e| panic!("{e}"));
        assert!(sim.pellet.is_some());
        assert!(sim.ledger.wall_normal_traction_energy.is_finite());
        assert!(sim.ledger.wall_normal_traction_energy != 0.0);
        let before = sim.ledger.wall_normal_traction_energy;
        let dt = 5e-5;
        let expected = assert_traction_energy_matches_full_scan(&sim, dt);
        sim.try_step(dt).expect("accepted coupled step");
        assert_eq!(sim.ledger.wall_normal_traction_energy, before + expected);
        let obs = spec.observables(&sim);
        assert_eq!(
            obs["wall_normal_traction_energy_j"],
            sim.ledger.wall_normal_traction_energy
        );
        assert!(obs.contains_key("coupling_grid_energy_j"));
    }

    /// One compressed particle within half a cell of the floor (and optionally
    /// the low `x` wall), so the traction transfer emits images.
    fn contact_particle(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        axes: &[usize],
    ) -> Simulation {
        let mut sim = empty_sim(Vector3::new(0.0, 0.0, -9.81));
        let volume = 1e-9;
        sim.particles
            .push(position, velocity, sim.material.density * volume, volume);
        let mut stress = nalgebra::Matrix3::zeros();
        for &axis in axes {
            stress[(axis, axis)] = -1000.0;
        }
        sim.particles.kirchhoff[0] = stress;
        sim
    }

    #[test]
    fn traction_energy_sign_follows_the_pre_gravity_normal_momentum() {
        let dt = 1e-5;
        let floor = Vector3::new(0.0105, 0.0105, 0.0005);
        // At rest the free node above carries the upward stress-force momentum
        // only, so the inward reaction adds kinetic energy.
        let mut resting = contact_particle(floor, Vector3::zeros(), &[2]);
        let positive = assert_traction_energy_matches_full_scan(&resting, dt);
        assert!(positive > 0.0, "{positive}");
        resting.try_step(dt).expect("accepted");
        assert_eq!(resting.ledger.wall_normal_traction_energy, positive);
        // Approaching the floor the nodes carry downward momentum that the
        // upward reaction cancels: kinetic energy is removed.
        let mut falling = contact_particle(floor, Vector3::new(0.0, 0.0, -0.1), &[2]);
        let negative = assert_traction_energy_matches_full_scan(&falling, dt);
        assert!(negative < 0.0, "{negative}");
        falling.try_step(dt).expect("accepted");
        assert_eq!(falling.ledger.wall_normal_traction_energy, negative);
        assert!(falling.ledger.wall_impulse.z > 0.0);
        // Separating faster than one gravity step: no image, no energy, no impulse.
        let mut leaving = contact_particle(floor, Vector3::new(0.0, 0.0, 0.1), &[2]);
        assert_eq!(assert_traction_energy_matches_full_scan(&leaving, dt), 0.0);
        leaving.try_step(dt).expect("accepted");
        assert_eq!(leaving.ledger.wall_normal_traction_energy, 0.0);
        assert_eq!(leaving.ledger.wall_impulse, Vector3::zeros());
        // A corner particle emits images across two faces onto shared nodes;
        // the per-deposit sum still equals the full before/after jump.
        let corner = Vector3::new(0.0005, 0.0105, 0.0005);
        let mut cornered = contact_particle(corner, Vector3::new(-0.05, 0.0, -0.05), &[0, 2]);
        let both = assert_traction_energy_matches_full_scan(&cornered, dt);
        cornered.try_step(dt).expect("accepted");
        assert_eq!(cornered.ledger.wall_normal_traction_energy, both);
        assert!(cornered.ledger.wall_impulse.x > 0.0 && cornered.ledger.wall_impulse.z > 0.0);
    }

    #[test]
    fn nonfinite_traction_energy_rejects_the_step_before_commit() {
        let dt = 1e-5;
        let floor = Vector3::new(0.0105, 0.0105, 0.0005);
        let mut sim = contact_particle(floor, Vector3::zeros(), &[2]);
        // A finite but absurd compressive stress yields a finite reaction whose
        // squared momentum overflows: the diagnostic must not commit as `inf`.
        sim.particles.kirchhoff[0][(2, 2)] = -1e300;
        let reference = sim.clone();
        let failure = sim.try_step(dt).expect_err("must reject");
        assert!(
            matches!(failure, StepFailure::NonFiniteTractionEnergy),
            "{failure}"
        );
        assert_eq!(sim.ledger, reference.ledger);
        assert_eq!(sim.particles, reference.particles);
        sim.ledger.validate().expect("ledger stays finite");
    }

    #[test]
    fn finite_traction_increment_cannot_overflow_accumulated_history() {
        let dt = 1e-5;
        let mut sim =
            contact_particle(Vector3::new(0.0105, 0.0105, 0.0005), Vector3::zeros(), &[2]);
        sim.particles.kirchhoff[0][(2, 2)] = -1e160;
        sim.ledger.wall_normal_traction_energy = f64::MAX;
        let reference = sim.clone();
        let transfer = sim.particle_to_grid(dt, None);
        assert!(transfer.energy.is_finite() && transfer.energy > 0.0);
        assert!(!(sim.ledger.wall_normal_traction_energy + transfer.energy).is_finite());
        assert!(matches!(
            sim.try_step(dt),
            Err(StepFailure::NonFiniteTractionEnergy)
        ));
        assert_eq!(sim.ledger, reference.ledger);
        assert_eq!(sim.particles, reference.particles);
    }

    #[test]
    fn coupling_energy_ledgers_match_direct_kinetic_changes() {
        use crate::mpm::rigid::{Pellet, couple_grid};
        use nalgebra::UnitQuaternion;
        let mut sim = empty_sim(Vector3::zeros());
        let layout = sim.grid.layout;
        let mut pellet = Pellet::new(
            0.003,
            0.012,
            1100.0,
            Vector3::new(0.01, 0.01, 0.01),
            UnitQuaternion::identity(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        pellet.velocity = Vector3::new(0.0, 0.0, -0.1);
        pellet.angular_momentum = Vector3::new(0.0, 1e-7, 0.0);
        for i in 0..layout.nodes[0] {
            for j in 0..layout.nodes[1] {
                for k in 0..layout.nodes[2] {
                    let node = layout.node_position(i, j, k);
                    if pellet.signed_distance(&node).0 <= 0.0 {
                        let velocity = Vector3::new(0.2 * (node.x - 0.01) / 0.006, 0.05, 0.3);
                        sim.grid
                            .deposit(layout.index(i, j, k), 1e-6, velocity * 1e-6);
                    }
                }
            }
        }
        // Grid holds momentum; convert to velocities as `grid_update` would.
        let grid_ke = |grid: &Grid| -> f64 {
            grid.active
                .iter()
                .map(|&n| 0.5 * grid.mass[n] * grid.momentum[n].norm_squared())
                .sum()
        };
        for &n in &sim.grid.active {
            sim.grid.momentum[n] /= sim.grid.mass[n];
        }
        let before = grid_ke(&sim.grid);
        let result = couple_grid(&mut sim.grid, &pellet, 0.4);
        assert!(result.constrained_nodes > 0);
        let grid_change = grid_ke(&sim.grid) - before;
        assert!(
            (grid_change - result.grid_energy).abs() < 1e-15 * grid_change.abs().max(1e-12),
            "grid {grid_change} vs {}",
            result.grid_energy
        );
        let pellet_before = pellet.kinetic_energy();
        pellet.apply_impulse(&result.impulse, &result.angular_impulse);
        let pellet_change = pellet.kinetic_energy() - pellet_before;
        assert!(pellet_change.is_finite() && pellet_change != 0.0);
        // The solver books exactly these two numbers.
        let mut ledger = Ledger::default();
        ledger.coupling_grid_energy += result.grid_energy;
        ledger.coupling_pellet_energy += pellet_change;
        assert_eq!(ledger.ledgered_energy(), result.grid_energy + pellet_change);
        ledger.validate().unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn accepted_step_books_both_coupling_energy_changes() {
        use crate::mpm::rigid::{Pellet, couple_grid};
        use nalgebra::UnitQuaternion;
        let base = empty_sim(Vector3::zeros());
        let mut particles = block(base.material, 0.002, [0.008; 3], [0.006; 3]);
        for velocity in &mut particles.velocity {
            *velocity = Vector3::new(0.05, 0.0, 0.1);
        }
        let pellet = Pellet::new(
            0.003,
            0.012,
            1100.0,
            Vector3::new(0.01, 0.01, 0.01),
            UnitQuaternion::identity(),
        )
        .expect("pellet");
        let mut sim = Simulation::new(
            base.material,
            particles,
            base.grid,
            Some(pellet.clone()),
            base.config,
        );
        let mut reference = sim.clone();
        let dt = 1e-5;
        let _ = reference.particle_to_grid(dt, None);
        let trials = reference.grid_update(dt).expect("grid update");
        for (slot, trial) in reference.grid.active.iter().zip(trials) {
            reference.grid.momentum[*slot] = trial.velocity;
        }
        let exchange = couple_grid(&mut reference.grid, &pellet, reference.config.wall_friction);
        let mut changed_pellet = pellet.clone();
        changed_pellet.apply_impulse(&exchange.impulse, &exchange.angular_impulse);
        let pellet_energy = changed_pellet.kinetic_energy() - pellet.kinetic_energy();
        assert!(exchange.grid_energy.abs() > 1e-12);
        assert!(pellet_energy.abs() > 1e-12);
        sim.try_step(dt).expect("accepted coupled step");
        assert_eq!(sim.ledger.coupling_grid_energy, exchange.grid_energy);
        assert_eq!(sim.ledger.coupling_pellet_energy, pellet_energy);
    }

    #[test]
    fn rejected_trial_step_preserves_ledger_and_state() {
        let material = Material::paste(1000.0, 1.0e5, 0.3, 1.0e6, 0.0, 1.0);
        let layout =
            GridLayout::new([0.02, 0.02, 0.02], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        let mut particles = block(material, 0.002, [0.008, 0.008, 0.006], [0.006, 0.006, 0.0]);
        for v in &mut particles.velocity {
            *v = Vector3::new(1.0, 0.0, -1.0);
        }
        let mut sim = Simulation::new(material, particles, Grid::new(layout), None, config());
        sim.advance_to(2e-4).unwrap_or_else(|e| panic!("{e}"));
        let reference = sim.clone();
        assert!(reference.ledger.wall_normal_projection_energy < 0.0);
        assert!(reference.ledger.wall_normal_traction_energy != 0.0);
        // 1 m/s over 5 ms crosses 2.5 cells and inverts the trial deformation:
        // rejected (constitutive or displacement) before any commit.
        let failure = sim.try_step(5e-3).expect_err("oversized step must reject");
        assert!(
            matches!(
                failure,
                StepFailure::Displacement { .. } | StepFailure::Constitutive(_)
            ),
            "{failure}"
        );
        assert_eq!(sim.ledger, reference.ledger);
        assert_eq!(sim.particles, reference.particles);
        assert_eq!(sim.time, reference.time);
    }

    #[test]
    fn ledger_round_trips_through_json_and_rejects_bad_signs() {
        let (_, mut sim) = resting_column(4e-5);
        sim.advance_to(2e-4).unwrap_or_else(|e| panic!("{e}"));
        let ledger = sim.ledger;
        assert!(ledger.initial_mechanical_energy > 0.0);
        let json = serde_json::to_string(&ledger).unwrap_or_else(|e| panic!("{e}"));
        assert!(json.contains("wall_normal_projection_energy"));
        assert!(json.contains("\"wall_normal_traction_energy\":"));
        assert!(ledger.wall_normal_traction_energy != 0.0);
        assert!(!json.contains("\"wall_work\""));
        let restored: Ledger = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(restored, ledger);
        assert_eq!(
            restored.wall_normal_traction_energy.to_bits(),
            ledger.wall_normal_traction_energy.to_bits()
        );
        // Missing history is a hard error, never a silent zero.
        let value: serde_json::Value =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}"));
        let mut missing = value.clone();
        missing
            .as_object_mut()
            .expect("object")
            .remove("wall_normal_traction_energy");
        assert!(serde_json::from_value::<Ledger>(missing).is_err());
        let mut null = value;
        null["wall_normal_traction_energy"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<Ledger>(null).is_err());
        let mut bad = ledger;
        bad.wall_normal_traction_energy = f64::INFINITY;
        assert!(
            bad.validate()
                .expect_err("infinite traction energy")
                .to_string()
                .contains("`wall_normal_traction_energy` is nonfinite")
        );
        assert!(
            serde_json::from_str::<Ledger>(
                &json.replace("wall_normal_projection_energy", "wall_work")
            )
            .is_err()
        );
        let mut bad = ledger;
        bad.wall_normal_projection_energy = 1e-9;
        let message = bad
            .validate()
            .expect_err("positive projection energy")
            .to_string();
        assert!(
            message.contains("non-positive, got '0.000000001'"),
            "{message}"
        );
        let mut bad = ledger;
        bad.wall_friction_dissipation = -1e-9;
        let message = bad
            .validate()
            .expect_err("negative dissipation")
            .to_string();
        assert!(
            message.contains("non-negative, got '-0.000000001'"),
            "{message}"
        );
        let mut bad = ledger;
        bad.coupling_grid_energy = f64::NAN;
        assert!(
            bad.validate()
                .unwrap_err()
                .to_string()
                .contains("nonfinite")
        );
        assert!(sim.validate().is_ok());
        sim.ledger.plastic_dissipation = -1.0;
        assert!(sim.validate().is_err());
    }
}
