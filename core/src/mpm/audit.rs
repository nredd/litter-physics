//! Transient substep energy measurements, NOT a physical energy closure.
//!
//! APIC affine energy uses the quadratic full-support second moment `h² I / 4`.
//! It is a discretization-specific kinetic norm, not part of the solver's legacy
//! mechanical energy. No audit data enters checkpoints or persistent ledgers.
//! Stress/storage mismatch includes constitutive lag, viscosity and plastic-work
//! approximation; it is not synthesized dissipation. Pellet integration combines
//! gravity, contact and rotation and omits recoverable contact-spring storage.

use nalgebra::Vector3;
use serde::Serialize;

use super::grid::Grid;
use super::rigid::Pellet;
use super::solver::{Ledger, Simulation, SolverError};

/// Particle and rigid-body energies at one endpoint, in J.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EnergyState {
    /// Material translational kinetic energy.
    pub translation: f64,
    /// Quadratic APIC norm `h²/8 sum m ||C||²`, not physical stored energy.
    pub affine: f64,
    /// Constitutive elastic storage (the solver's existing energy model).
    pub elastic: f64,
    /// Material plus pellet gravitational potential energy.
    pub potential: f64,
    /// Pellet translational plus rotational kinetic energy, zero without pellet.
    pub pellet: f64,
}

impl EnergyState {
    /// Sum including the discretization-specific affine norm.
    #[must_use]
    pub fn augmented_total(self) -> f64 {
        self.translation + self.affine + self.elastic + self.potential + self.pellet
    }

    /// Measure an endpoint without mutating any state.
    fn measure(sim: &Simulation) -> Self {
        Self {
            translation: sim.particles.kinetic_energy(),
            affine: sim.grid.layout.spacing.powi(2) / 8.0
                * sim
                    .particles
                    .mass
                    .iter()
                    .zip(&sim.particles.affine)
                    .map(|(mass, affine)| mass * affine.norm_squared())
                    .sum::<f64>(),
            elastic: sim.elastic_energy(),
            potential: sim.potential_energy(),
            pellet: sim.pellet.as_ref().map_or(0.0, Pellet::kinetic_energy),
        }
    }

    /// Scalars checked individually, so cancellation cannot hide overflow.
    fn values(self) -> [f64; 5] {
        [
            self.translation,
            self.affine,
            self.elastic,
            self.potential,
            self.pellet,
        ]
    }
}

/// Kinetic snapshots and local plastic work of the actual operator sequence, in J.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct EnergyStages {
    /// Independent no-stress APIC shadow deposition (diagnostic only).
    pub grid_transport: f64,
    /// Actual combined APIC plus MLS stress deposition.
    pub grid_after_stress: f64,
    /// After signed face-normal wall lattice-completion impulses.
    pub grid_after_traction: f64,
    /// After gravity, before projection and friction.
    pub grid_after_gravity: f64,
    /// After normal projection and Coulomb friction.
    pub grid_after_walls: f64,
    /// After grid/pellet coupling, before G2P.
    pub grid_after_coupling: f64,
    /// Pellet kinetic energy after coupling, before combined integration.
    pub pellet_after_coupling: f64,
    /// Actual trial plastic-work sum before cumulative-history rounding.
    /// This retains the solver's constitutive approximation, not exact physical work.
    pub plastic_dissipation: f64,
}

impl EnergyStages {
    /// Scalars checked individually for finiteness.
    fn values(self) -> [f64; 8] {
        [
            self.grid_transport,
            self.grid_after_stress,
            self.grid_after_traction,
            self.grid_after_gravity,
            self.grid_after_walls,
            self.grid_after_coupling,
            self.pellet_after_coupling,
            self.plastic_dissipation,
        ]
    }
}

/// Differences of existing energy histories across the accepted trial, in J.
///
/// Cumulative-history rounding can erase a small contribution; these differences
/// are not replacements for independently measured stage changes.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EnergyIncrements {
    /// Existing nonnegative plastic-work approximation.
    pub plastic_dissipation: f64,
    /// Existing nonnegative grid wall-friction loss.
    pub wall_friction_dissipation: f64,
    /// Existing signed wall-normal projection change.
    pub wall_normal_projection_energy: f64,
    /// Existing signed wall-normal traction change.
    pub wall_normal_traction_energy: f64,
    /// Existing signed grid coupling change.
    pub coupling_grid_energy: f64,
    /// Existing signed pellet coupling change.
    pub coupling_pellet_energy: f64,
    /// Existing contact-force work approximation, NOT a separate exact stage.
    pub pellet_wall_work: f64,
}

impl EnergyIncrements {
    /// Subtract actual histories; never infer or default missing history.
    fn between(before: Ledger, after: Ledger) -> Self {
        Self {
            plastic_dissipation: after.plastic_dissipation - before.plastic_dissipation,
            wall_friction_dissipation: after.wall_friction_dissipation
                - before.wall_friction_dissipation,
            wall_normal_projection_energy: after.wall_normal_projection_energy
                - before.wall_normal_projection_energy,
            wall_normal_traction_energy: after.wall_normal_traction_energy
                - before.wall_normal_traction_energy,
            coupling_grid_energy: after.coupling_grid_energy - before.coupling_grid_energy,
            coupling_pellet_energy: after.coupling_pellet_energy - before.coupling_pellet_energy,
            pellet_wall_work: after.pellet_wall_work - before.pellet_wall_work,
        }
    }

    /// Scalars checked individually for finiteness.
    fn values(self) -> [f64; 7] {
        [
            self.plastic_dissipation,
            self.wall_friction_dissipation,
            self.wall_normal_projection_energy,
            self.wall_normal_traction_energy,
            self.coupling_grid_energy,
            self.coupling_pellet_energy,
            self.pellet_wall_work,
        ]
    }
}

/// Algebraic decomposition, NOT physical closure; every quantity is in J.
///
/// Signed losses are never clamped. Roundoff can make a transfer loss negative.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EnergyTerms {
    /// Particle APIC norm minus shadow transport grid energy.
    pub p2g_transfer_loss: f64,
    /// Post-coupling grid energy minus final particle APIC norm.
    pub g2p_transfer_loss: f64,
    /// Stress grid jump + elastic storage change + existing plastic increment.
    /// Contains unresolved constitutive/time-discretization/viscous effects.
    pub stress_storage_plastic_mismatch: f64,
    /// Grid gravity jump plus total potential change, ONLY without a pellet.
    pub uncoupled_gravity_potential_mismatch: Option<f64>,
    /// Grid gravity jump plus material/pellet potential change when coupled.
    /// Not gravity-only: positions reflect coupling/contact/combined integration.
    pub coupled_grid_gravity_total_potential_change: Option<f64>,
    /// Pellet post-integration minus post-coupling kinetic energy. Combined
    /// gravity/contact/rotation; no claim of separated physical contributions.
    pub pellet_integrator_combined_change: f64,
    /// Signed wall-traction kinetic stage jump (not stationary-wall work).
    pub traction_change: f64,
    /// Signed projection plus friction kinetic stage jump.
    pub walls_change: f64,
    /// Signed grid plus pellet coupling kinetic stage jump.
    pub coupling_change: f64,
    /// Change of total endpoint energy INCLUDING the APIC affine norm.
    pub augmented_energy_change: f64,
    /// Sum of the signed stage terms; should equal `augmented_energy_change`.
    pub telescoping_sum: f64,
    /// Floating-point telescoping discrepancy; NOT unexplained physical energy.
    pub telescoping_error: f64,
}

impl EnergyTerms {
    /// Form algebraic terms independently of cumulative-history roundoff.
    fn measure(before: EnergyState, after: EnergyState, s: EnergyStages, coupled: bool) -> Self {
        let p2g = before.translation + before.affine - s.grid_transport;
        let g2p = s.grid_after_coupling - after.translation - after.affine;
        let stress = s.grid_after_stress - s.grid_transport + after.elastic - before.elastic
            + s.plastic_dissipation;
        let gravity =
            s.grid_after_gravity - s.grid_after_traction + after.potential - before.potential;
        let integrator = after.pellet - s.pellet_after_coupling;
        let traction = s.grid_after_traction - s.grid_after_stress;
        let walls = s.grid_after_walls - s.grid_after_gravity;
        let coupling =
            s.grid_after_coupling - s.grid_after_walls + s.pellet_after_coupling - before.pellet;
        let change = after.augmented_total() - before.augmented_total();
        let sum = -p2g - g2p + stress - s.plastic_dissipation
            + gravity
            + integrator
            + traction
            + walls
            + coupling;
        Self {
            p2g_transfer_loss: p2g,
            g2p_transfer_loss: g2p,
            stress_storage_plastic_mismatch: stress,
            uncoupled_gravity_potential_mismatch: (!coupled).then_some(gravity),
            coupled_grid_gravity_total_potential_change: coupled.then_some(gravity),
            pellet_integrator_combined_change: integrator,
            traction_change: traction,
            walls_change: walls,
            coupling_change: coupling,
            augmented_energy_change: change,
            telescoping_sum: sum,
            telescoping_error: change - sum,
        }
    }

    /// Scalars checked individually for finiteness, including optional terms.
    fn values(self) -> impl Iterator<Item = f64> {
        [
            self.p2g_transfer_loss,
            self.g2p_transfer_loss,
            self.stress_storage_plastic_mismatch,
            self.pellet_integrator_combined_change,
            self.traction_change,
            self.walls_change,
            self.coupling_change,
            self.augmented_energy_change,
            self.telescoping_sum,
            self.telescoping_error,
        ]
        .into_iter()
        .chain(self.uncoupled_gravity_potential_mismatch)
        .chain(self.coupled_grid_gravity_total_potential_change)
    }
}

/// One finite, accepted, nonmutating trial measurement. No persistent schema change.
#[derive(Debug, Clone, Serialize)]
pub struct EnergyAudit {
    /// Start time in s (the original simulation is not advanced).
    pub time: f64,
    /// Requested and actually attempted step in s, never clipped or retried.
    pub dt: f64,
    /// Endpoint before the trial.
    pub before: EnergyState,
    /// Endpoint after the accepted trial.
    pub after: EnergyState,
    /// Grid and pellet kinetic stages.
    pub stages: EnergyStages,
    /// Accepted differences of the existing ledger histories.
    pub increments: EnergyIncrements,
    /// Algebraic decomposition with explicit unresolved terms.
    pub terms: EnergyTerms,
}

impl Simulation {
    /// Audit exactly one trial using the production kernel on a clone.
    ///
    /// The caller bounds sample count; work is one trial plus shadow deposition
    /// and active-node/particle scans. Normal stepping performs none of these
    /// extra scans. No retries, clipping, original-state mutations, checkpoint
    /// fields or persistent histories are introduced.
    ///
    /// # Errors
    ///
    /// Rejects invalid state, escaped particles, nonfinite energies, a rejected
    /// trial, or `dt` outside `[min_dt, min(max_dt, dt_scale * stability_limit)]`.
    pub fn audit_step(&self, dt: f64) -> Result<EnergyAudit, SolverError> {
        self.validate()?;
        let stability = self.stability_limit();
        if !self.config.gravity.iter().all(|v| v.is_finite())
            || !self.config.cfl.is_finite()
            || self.config.cfl <= 0.0
            || !self.config.wall_friction.is_finite()
            || self.config.wall_friction < 0.0
            || !self.config.min_dt.is_finite()
            || self.config.min_dt <= 0.0
            || !self.config.max_dt.is_finite()
            || self.config.max_dt < self.config.min_dt
            || !stability.is_finite()
            || stability <= 0.0
        {
            return Err(SolverError::InvalidState(
                "invalid audit step configuration or stability limit".into(),
            ));
        }
        let limit = (self.dt_scale * stability).min(self.config.max_dt);
        if !dt.is_finite()
            || dt <= 0.0
            || dt < self.config.min_dt
            || !limit.is_finite()
            || dt > limit
        {
            return Err(SolverError::InvalidState(format!(
                "audit `dt`='{dt}' outside finite accepted range ending at '{limit}'"
            )));
        }
        let before = EnergyState::measure(self);
        if !before.values().into_iter().all(f64::is_finite) {
            return Err(SolverError::InvalidState(
                "nonfinite initial audit energy".into(),
            ));
        }
        let mut stages = EnergyStages {
            grid_transport: transport_energy(self),
            ..EnergyStages::default()
        };
        let mut trial = self.clone();
        trial
            .try_step_captured(dt, Some(&mut stages))
            .map_err(|error| SolverError::InvalidState(format!("audit trial rejected: {error}")))?;
        trial.validate()?;
        let after = EnergyState::measure(&trial);
        let increments = EnergyIncrements::between(self.ledger, trial.ledger);
        let terms = EnergyTerms::measure(before, after, stages, self.pellet.is_some());
        if !before
            .values()
            .into_iter()
            .chain(after.values())
            .chain(stages.values())
            .chain(increments.values())
            .chain(terms.values())
            .all(f64::is_finite)
        {
            return Err(SolverError::InvalidState(
                "audit energy became nonfinite".into(),
            ));
        }
        Ok(EnergyAudit {
            time: self.time,
            dt,
            before,
            after,
            stages,
            increments,
            terms,
        })
    }
}

/// Active-node kinetic energy. Momentum storage changes to velocity after walls.
pub(super) fn grid_kinetic(grid: &Grid, velocities: bool, kick: Vector3<f64>) -> f64 {
    grid.active
        .iter()
        .filter(|&&i| grid.mass[i] > 0.0)
        .map(|&i| {
            let velocity = if velocities {
                grid.momentum[i]
            } else {
                grid.momentum[i] / grid.mass[i]
            };
            0.5 * grid.mass[i] * (velocity + kick).norm_squared()
        })
        .sum()
}

/// Independent APIC-only deposition: never reuses real stress-laden momentum.
fn transport_energy(sim: &Simulation) -> f64 {
    let layout = sim.grid.layout;
    let mut shadow = Grid::new(layout);
    for p in 0..sim.particles.len() {
        let position = sim.particles.position[p];
        let Some(stencil) = layout.stencil(&position) else {
            return f64::NAN;
        };
        let mass = sim.particles.mass[p];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let weight = stencil.weight(a, b, c);
                    let velocity = sim.particles.velocity[p]
                        + sim.particles.affine[p] * stencil.distance(a, b, c);
                    let index = layout.index(
                        stencil.base[0] + a,
                        stencil.base[1] + b,
                        stencil.base[2] + c,
                    );
                    shadow.deposit(index, weight * mass, velocity * (weight * mass));
                }
            }
        }
    }
    grid_kinetic(&shadow, false, Vector3::zeros())
}
