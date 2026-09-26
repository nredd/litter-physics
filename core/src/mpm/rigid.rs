//! Rigid oriented pellet and conservative two-way grid coupling.
//!
//! The pellet is a rigid body whose mass and inertia come from the cylinder fit
//! (`m = rho pi r^2 L`, `I_axis = m r^2 / 2`, `I_perp = m (3 r^2 + L^2) / 12`)
//! and whose collision shape is the multisphere proxy used by the DEM side of the
//! project: spheres of radius `r` spaced at most `r` apart along the axis.
//!
//! Coupling is grid-based: every active grid node inside the pellet has its
//! velocity constrained against the pellet surface velocity (no penetration,
//! Coulomb friction on the tangential relative velocity). The momentum removed
//! from the grid at each node, `dp_i = m_i (v_i' - v_i)`, is applied with the
//! opposite sign to the pellet as a linear impulse `-sum dp_i` and an angular
//! impulse `-sum r_i x dp_i` about the pellet centre, which conserves linear and
//! angular momentum of the grid + pellet system exactly (see
//! [`couple_grid`] and its tests). Contact and drag are the same mechanism, so
//! nothing is double counted.
//!
//! References:
//! - Hu et al. 2018, <https://doi.org/10.1145/3197517.3201293> (grid rigid coupling)
//! - Stomakhin et al. 2013, <https://doi.org/10.1145/2461912.2461948> (collision nodes)

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use rayon::prelude::*;

use super::grid::Grid;

/// Rigid pellet state and geometry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pellet {
    /// Cylinder radius in m.
    pub radius: f64,
    /// Cylinder length in m.
    pub length: f64,
    /// Mass in kg.
    pub mass: f64,
    /// Principal inertia in the body frame (axis along body x) in kg m^2.
    pub inertia_body: Vector3<f64>,
    /// Centre of mass in m.
    pub position: Vector3<f64>,
    /// Orientation (body to world).
    pub orientation: UnitQuaternion<f64>,
    /// Linear velocity in m/s.
    pub velocity: Vector3<f64>,
    /// Angular momentum in the world frame in kg m^2/s.
    pub angular_momentum: Vector3<f64>,
    /// Sphere centre offsets along the body axis in m.
    pub sphere_offsets: Vec<f64>,
}

/// Impulses exchanged with the domain walls during one pellet integration step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WallContact {
    /// Linear impulse delivered by the walls to the pellet in kg m/s.
    pub impulse: Vector3<f64>,
    /// Work of the wall contact forces on the pellet in J: `F . v_contact dt`
    /// with the start-of-step force and contact-point velocity. The stationary
    /// wall itself does no external work; this term includes the recoverable
    /// spring energy of the penalty contact, so a negative value is NOT all
    /// dissipation.
    pub work: f64,
    /// Number of sphere/face pairs in contact.
    pub contacts: usize,
}

/// Spring-dashpot wall contact parameters (shared DEM material from the request).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactParams {
    /// Normal stiffness in N/m.
    pub normal_stiffness: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

impl ContactParams {
    /// Normal damping coefficient in N s/m derived from the restitution
    /// (`zeta = -ln e / sqrt(pi^2 + ln^2 e)`, critical damping at `e = 0`).
    #[must_use]
    pub fn damping(&self, mass: f64) -> f64 {
        let zeta = if self.restitution <= 0.0 {
            1.0
        } else if self.restitution >= 1.0 {
            0.0
        } else {
            let log = self.restitution.ln();
            -log / (std::f64::consts::PI.powi(2) + log * log).sqrt()
        };
        2.0 * zeta * (self.normal_stiffness * mass).sqrt()
    }

    /// Contact stability limit in s: `0.2 sqrt(m / k)`.
    #[must_use]
    pub fn stable_dt(&self, mass: f64) -> f64 {
        0.2 * (mass / self.normal_stiffness).sqrt()
    }
}

impl Pellet {
    /// Build a pellet at rest.
    ///
    /// Parameters:
    /// - `radius` (`f64`): cylinder radius in m, positive.
    /// - `length` (`f64`): cylinder length in m, at least `2 * radius`.
    /// - `density` (`f64`): density in kg/m^3, positive.
    /// - `position` (`Vector3<f64>`): centre of mass in m.
    /// - `orientation` (`UnitQuaternion<f64>`): body-to-world rotation.
    ///
    /// Returns: `Result<Pellet, String>`.
    ///
    /// # Errors
    ///
    /// on non-positive or non-finite geometry.
    pub fn new(
        radius: f64,
        length: f64,
        density: f64,
        position: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
    ) -> Result<Self, String> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(format!("pellet radius must be positive, got '{radius}'"));
        }
        if !(length.is_finite() && length >= 2.0 * radius) {
            return Err(format!(
                "pellet length must be at least twice the radius, got '{length}' for radius '{radius}'"
            ));
        }
        if !(density.is_finite() && density > 0.0) {
            return Err(format!("pellet density must be positive, got '{density}'"));
        }
        let mass = density * std::f64::consts::PI * radius * radius * length;
        let inertia_body = Vector3::new(
            0.5 * mass * radius * radius,
            mass * (3.0 * radius * radius + length * length) / 12.0,
            mass * (3.0 * radius * radius + length * length) / 12.0,
        );
        if !mass.is_finite()
            || mass <= 0.0
            || inertia_body.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || position.iter().any(|v| !v.is_finite())
            || orientation.coords.iter().any(|v| !v.is_finite())
        {
            return Err("pellet mass, inertia or pose is not representable".into());
        }
        let half_span = 0.5 * length - radius;
        let count = if half_span <= 0.0 {
            1
        } else {
            // Spacing at most `radius`; ceil(2 * half_span / radius) + 1 spheres.
            let ratio = (2.0 * half_span / radius).ceil();
            if !ratio.is_finite() || ratio > 4095.0 {
                return Err("pellet proxy exceeds the sphere allocation cap".into());
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // ratio >= 1, bounded.
            let count = ratio as usize + 1;
            count
        };
        #[allow(clippy::cast_precision_loss)] // sphere counts are small.
        let sphere_offsets: Vec<f64> = (0..count)
            .map(|i| {
                if count == 1 {
                    0.0
                } else {
                    -half_span + 2.0 * half_span * (i as f64) / ((count - 1) as f64)
                }
            })
            .collect();
        Ok(Self {
            radius,
            length,
            mass,
            inertia_body,
            position,
            orientation,
            velocity: Vector3::zeros(),
            angular_momentum: Vector3::zeros(),
            sphere_offsets,
        })
    }

    /// World-frame inertia tensor.
    #[must_use]
    pub fn inertia_world(&self) -> Matrix3<f64> {
        let rotation = self.orientation.to_rotation_matrix().into_inner();
        rotation * Matrix3::from_diagonal(&self.inertia_body) * rotation.transpose()
    }

    /// World-frame angular velocity in rad/s.
    #[must_use]
    pub fn angular_velocity(&self) -> Vector3<f64> {
        let rotation = self.orientation.to_rotation_matrix().into_inner();
        let body_momentum = rotation.transpose() * self.angular_momentum;
        let body_omega = body_momentum.component_div(&self.inertia_body);
        rotation * body_omega
    }

    /// World positions of the multisphere centres.
    #[must_use]
    pub fn sphere_centers(&self) -> Vec<Vector3<f64>> {
        let axis = self.orientation * Vector3::x();
        self.sphere_offsets
            .iter()
            .map(|&offset| self.position + axis * offset)
            .collect()
    }

    /// Signed distance to the multisphere surface and the outward normal.
    #[must_use]
    pub fn signed_distance(&self, point: &Vector3<f64>) -> (f64, Vector3<f64>) {
        let axis = self.orientation * Vector3::x();
        let mut best = (f64::INFINITY, Vector3::z());
        for &offset in &self.sphere_offsets {
            let center = self.position + axis * offset;
            let delta = point - center;
            let distance = delta.norm();
            let phi = distance - self.radius;
            if phi < best.0 {
                let normal = if distance > 1e-12 {
                    delta / distance
                } else {
                    Vector3::z()
                };
                best = (phi, normal);
            }
        }
        best
    }

    /// Velocity of the rigid body at a world point.
    #[must_use]
    pub fn point_velocity(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.velocity + self.angular_velocity().cross(&(point - self.position))
    }

    /// Conservative axis-aligned bounding box of the multisphere surface.
    #[must_use]
    pub fn bounding_box(&self) -> (Vector3<f64>, Vector3<f64>) {
        let axis = self.orientation * Vector3::x();
        let half_span = self
            .sphere_offsets
            .iter()
            .fold(0.0f64, |m, o| m.max(o.abs()));
        let reach = axis.abs() * half_span + Vector3::repeat(self.radius);
        (self.position - reach, self.position + reach)
    }

    /// Apply a linear impulse and an angular impulse about the centre of mass.
    pub fn apply_impulse(&mut self, impulse: &Vector3<f64>, angular_impulse: &Vector3<f64>) {
        self.velocity += impulse / self.mass;
        self.angular_momentum += angular_impulse;
    }

    /// Translational plus rotational kinetic energy in J.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let omega = self.angular_velocity();
        0.5 * self.mass * self.velocity.norm_squared() + 0.5 * omega.dot(&self.angular_momentum)
    }

    /// Angular momentum about the world origin in kg m^2/s.
    #[must_use]
    pub fn angular_momentum_about_origin(&self) -> Vector3<f64> {
        self.position.cross(&(self.velocity * self.mass)) + self.angular_momentum
    }

    /// Integrate the pellet through `dt` with gravity and spring-dashpot contact
    /// against the six domain walls `[0, extents]`.
    ///
    /// Parameters:
    /// - `dt` (`f64`): time step in s.
    /// - `gravity` (`&Vector3<f64>`): gravitational acceleration in m/s^2.
    /// - `extents` (`&Vector3<f64>`): domain extents in m.
    /// - `contact` (`&ContactParams`): spring-dashpot parameters.
    ///
    /// Returns: `WallContact` impulse ledger for this step.
    pub fn integrate(
        &mut self,
        dt: f64,
        gravity: &Vector3<f64>,
        extents: &Vector3<f64>,
        contact: &ContactParams,
    ) -> WallContact {
        let damping = contact.damping(self.mass);
        let mut force = Vector3::zeros();
        let mut torque = Vector3::zeros();
        let mut ledger = WallContact::default();
        let omega = self.angular_velocity();
        for center in self.sphere_centers() {
            for axis in 0..3 {
                for (face_position, sign) in [(0.0, 1.0), (extents[axis], -1.0)] {
                    let gap = (center[axis] - face_position) * sign;
                    let penetration = self.radius - gap;
                    if penetration <= 0.0 {
                        continue;
                    }
                    let mut normal = Vector3::zeros();
                    normal[axis] = sign;
                    let contact_point = center - normal * self.radius;
                    let lever = contact_point - self.position;
                    let point_velocity = self.velocity + omega.cross(&lever);
                    let normal_speed = point_velocity.dot(&normal);
                    let normal_force =
                        (contact.normal_stiffness * penetration - damping * normal_speed).max(0.0);
                    let tangential = point_velocity - normal * normal_speed;
                    let tangential_speed = tangential.norm();
                    let friction_force = if tangential_speed > 1e-12 {
                        // Impulse-limited Coulomb friction: never reverses sliding in one step.
                        let limit = (contact.friction * normal_force)
                            .min(self.mass * tangential_speed / dt);
                        -tangential / tangential_speed * limit
                    } else {
                        Vector3::zeros()
                    };
                    let total = normal * normal_force + friction_force;
                    force += total;
                    torque += lever.cross(&total);
                    ledger.work += total.dot(&point_velocity) * dt;
                    ledger.contacts += 1;
                }
            }
        }
        let impulse = force * dt;
        ledger.impulse = impulse;
        self.velocity += gravity * dt + impulse / self.mass;
        self.angular_momentum += torque * dt;
        self.position += self.velocity * dt;
        let omega_new = self.angular_velocity();
        let rotation_increment = UnitQuaternion::from_scaled_axis(omega_new * dt);
        self.orientation = rotation_increment * self.orientation;
        ledger
    }

    /// Whether every state component is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.velocity.iter().all(|v| v.is_finite())
            && self.angular_momentum.iter().all(|v| v.is_finite())
            && self.orientation.coords.iter().all(|v| v.is_finite())
    }
}

/// Per-node coupling update: new velocity, momentum change, lever arm,
/// kinetic-energy change.
type NodeCoupling = (Vector3<f64>, Vector3<f64>, Vector3<f64>, f64);

/// Result of a grid/pellet coupling pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CouplingResult {
    /// Linear impulse applied to the pellet in kg m/s.
    pub impulse: Vector3<f64>,
    /// Angular impulse about the pellet centre in kg m^2/s.
    pub angular_impulse: Vector3<f64>,
    /// Grid kinetic-energy change from the constraint in J:
    /// `sum_i m_i / 2 (|v_i'|^2 - |v_i|^2)`. Not sign-definite: the constraint
    /// targets the pellet surface velocity, which can add energy to the grid.
    pub grid_energy: f64,
    /// Number of constrained nodes.
    pub constrained_nodes: usize,
}

/// Constrain grid node velocities against the pellet and return the equal and
/// opposite impulses for the pellet.
///
/// The grid `momentum` array must hold node velocities on entry (after the grid
/// update divided momentum by mass). Nodes with signed distance `<= 0` are
/// constrained: approaching normal relative velocity is removed and the
/// tangential relative velocity is reduced by Coulomb friction with coefficient
/// `friction` times the removed normal speed.
///
/// All nodes target the old body velocity; the accumulated reaction is applied
/// afterward. This preserves impulses but can create joint kinetic energy for
/// light bodies. It is NOT an energy-stable finite-inertia constraint solve;
/// the `energy_audit` example retains a zero-external-work counterexample.
///
/// Parameters:
/// - `grid` (`&mut Grid`): grid holding node velocities.
/// - `pellet` (`&Pellet`): rigid body (velocity from the start of the step).
/// - `friction` (`f64`): Coulomb friction coefficient.
///
/// Returns: `CouplingResult` with the momentum removed from the grid, negated.
pub fn couple_grid(grid: &mut Grid, pellet: &Pellet, friction: f64) -> CouplingResult {
    let (low, high) = pellet.bounding_box();
    let layout = grid.layout;
    let (masses, velocities, active) = (&grid.mass, &grid.momentum, &grid.active);
    let updates: Vec<Option<NodeCoupling>> = active
        .par_iter()
        .map(|&index| {
            let mass = masses[index];
            if mass <= 0.0 {
                return None;
            }
            let (i, j, k) = layout.coords(index);
            let node = layout.node_position(i, j, k);
            if (0..3).any(|axis| node[axis] < low[axis] || node[axis] > high[axis]) {
                return None;
            }
            let (phi, normal) = pellet.signed_distance(&node);
            if phi > 0.0 {
                return None;
            }
            let velocity = velocities[index];
            let body_velocity = pellet.point_velocity(&node);
            let relative = velocity - body_velocity;
            let normal_speed = relative.dot(&normal);
            if normal_speed >= 0.0 {
                return None;
            }
            let tangential = relative - normal * normal_speed;
            let tangential_speed = tangential.norm();
            let friction_reduction = friction * (-normal_speed);
            let tangential_new =
                if tangential_speed <= friction_reduction || tangential_speed < 1e-14 {
                    Vector3::zeros()
                } else {
                    tangential * ((tangential_speed - friction_reduction) / tangential_speed)
                };
            let new_velocity = body_velocity + tangential_new;
            let delta_momentum = (new_velocity - velocity) * mass;
            let energy = 0.5 * mass * (new_velocity.norm_squared() - velocity.norm_squared());
            Some((new_velocity, delta_momentum, node - pellet.position, energy))
        })
        .collect();
    let mut result = CouplingResult::default();
    for (slot, update) in grid.active.iter().zip(updates) {
        if let Some((new_velocity, delta_momentum, lever, energy)) = update {
            grid.momentum[*slot] = new_velocity;
            result.impulse -= delta_momentum;
            result.angular_impulse -= lever.cross(&delta_momentum);
            result.grid_energy += energy;
            result.constrained_nodes += 1;
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::{ContactParams, Pellet, couple_grid};
    use crate::mpm::grid::{Grid, GridLayout};
    use nalgebra::{UnitQuaternion, Vector3};

    fn pellet_at(position: Vector3<f64>) -> Pellet {
        Pellet::new(0.003, 0.012, 1100.0, position, UnitQuaternion::identity())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn geometry_matches_cylinder_fit() {
        let pellet = pellet_at(Vector3::zeros());
        let expected_mass = 1100.0 * std::f64::consts::PI * 0.003f64.powi(2) * 0.012;
        assert!((pellet.mass - expected_mass).abs() < 1e-12);
        assert!(pellet.sphere_offsets.len() >= 3);
        assert!(
            pellet
                .sphere_offsets
                .windows(2)
                .all(|w| w[1] - w[0] <= 0.003 + 1e-12)
        );
        let (phi, normal) = pellet.signed_distance(&Vector3::new(0.0, 0.0, 0.004));
        assert!((phi - 0.001).abs() < 1e-12);
        assert!((normal - Vector3::z()).norm() < 1e-12);
        let (phi_end, _) = pellet.signed_distance(&Vector3::new(0.006, 0.0, 0.0));
        assert!(phi_end.abs() < 1e-12);
        assert!(
            Pellet::new(
                0.003,
                0.004,
                1100.0,
                Vector3::zeros(),
                UnitQuaternion::identity()
            )
            .is_err()
        );
    }

    #[test]
    fn coupling_conserves_linear_and_angular_momentum_exactly() {
        let layout = GridLayout::new([0.02, 0.02, 0.02], 0.001, 10_000_000)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut grid = Grid::new(layout);
        let mut pellet = pellet_at(Vector3::new(0.01, 0.01, 0.01));
        pellet.velocity = Vector3::new(0.0, 0.0, -0.1);
        pellet.angular_momentum = Vector3::new(0.0, 1e-7, 0.0);
        // Populate nodes around the pellet with an off-centre upward flow.
        for i in 0..layout.nodes[0] {
            for j in 0..layout.nodes[1] {
                for k in 0..layout.nodes[2] {
                    let node = layout.node_position(i, j, k);
                    let (phi, _) = pellet.signed_distance(&node);
                    if phi <= 0.001 {
                        let velocity = Vector3::new(0.2 * (node.x - 0.01) / 0.006, 0.05, 0.3);
                        grid.deposit(layout.index(i, j, k), 1e-6, velocity);
                    }
                }
            }
        }
        assert!(!grid.active.is_empty());
        let grid_momentum_before = grid.active.iter().fold(Vector3::zeros(), |acc, &n| {
            acc + grid.momentum[n] * grid.mass[n]
        });
        let grid_angular_before = grid.active.iter().fold(Vector3::zeros(), |acc, &n| {
            let (i, j, k) = layout.coords(n);
            acc + layout
                .node_position(i, j, k)
                .cross(&(grid.momentum[n] * grid.mass[n]))
        });
        let body_momentum_before = pellet.velocity * pellet.mass;
        let body_angular_before = pellet.angular_momentum_about_origin();

        let result = couple_grid(&mut grid, &pellet, 0.4);
        assert!(result.constrained_nodes > 0, "no nodes constrained");
        pellet.apply_impulse(&result.impulse, &result.angular_impulse);

        let grid_momentum_after = grid.active.iter().fold(Vector3::zeros(), |acc, &n| {
            acc + grid.momentum[n] * grid.mass[n]
        });
        let grid_angular_after = grid.active.iter().fold(Vector3::zeros(), |acc, &n| {
            let (i, j, k) = layout.coords(n);
            acc + layout
                .node_position(i, j, k)
                .cross(&(grid.momentum[n] * grid.mass[n]))
        });
        let total_before = grid_momentum_before + body_momentum_before;
        let total_after = grid_momentum_after + pellet.velocity * pellet.mass;
        assert!(
            (total_before - total_after).norm() < 1e-12 * total_before.norm().max(1e-9),
            "linear momentum drift {:?}",
            total_before - total_after
        );
        let angular_before = grid_angular_before + body_angular_before;
        let angular_after = grid_angular_after + pellet.angular_momentum_about_origin();
        assert!(
            (angular_before - angular_after).norm() < 1e-12 * angular_before.norm().max(1e-12),
            "angular momentum drift {:?}",
            angular_before - angular_after
        );
        // The upward flow pushes the pellet up; the momentum removed from the grid is non-trivial.
        assert!(result.impulse.z > 0.0);
        // Constrained nodes no longer approach the pellet.
        for &n in &grid.active {
            let (i, j, k) = layout.coords(n);
            let node = layout.node_position(i, j, k);
            let (phi, normal) = pellet.signed_distance(&node);
            if phi <= 0.0 {
                // Pellet velocity here is the pre-impulse value the constraint used.
                let mut reference = pellet.clone();
                reference.apply_impulse(&(-result.impulse), &(-result.angular_impulse));
                let relative = grid.momentum[n] - reference.point_velocity(&node);
                assert!(relative.dot(&normal) >= -1e-12);
            }
        }
    }

    #[test]
    fn off_centre_impulse_produces_torque_in_expected_direction() {
        let layout = GridLayout::new([0.03, 0.02, 0.02], 0.001, 10_000_000)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut grid = Grid::new(layout);
        let pellet = pellet_at(Vector3::new(0.015, 0.01, 0.01));
        // Upward flow only at the +x end of the pellet.
        for i in 0..layout.nodes[0] {
            for j in 0..layout.nodes[1] {
                for k in 0..layout.nodes[2] {
                    let node = layout.node_position(i, j, k);
                    let (phi, _) = pellet.signed_distance(&node);
                    if phi <= 0.0 && node.x > 0.018 {
                        grid.deposit(layout.index(i, j, k), 1e-6, Vector3::new(0.0, 0.0, 0.5));
                    }
                }
            }
        }
        let result = couple_grid(&mut grid, &pellet, 0.0);
        assert!(result.impulse.z > 0.0);
        // Upward push at +x lever produces torque about -y... r = (+x, 0, 0), F = (0,0,+z): r x F = (0, -x z, 0).
        assert!(
            result.angular_impulse.y < 0.0,
            "{:?}",
            result.angular_impulse
        );
        assert!(result.angular_impulse.x.abs() < 1e-15);
    }

    #[test]
    fn pellet_settles_on_floor_with_spring_dashpot_contact() {
        let contact = ContactParams {
            normal_stiffness: 1000.0,
            restitution: 0.2,
            friction: 0.4,
        };
        let extents = Vector3::new(0.05, 0.05, 0.05);
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let mut pellet = pellet_at(Vector3::new(0.025, 0.025, 0.01));
        let dt = contact.stable_dt(pellet.mass) * 0.5;
        let steps = (1.0 / dt).ceil();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let steps = steps as usize;
        let mut total_impulse = Vector3::zeros();
        for _ in 0..steps {
            let ledger = pellet.integrate(dt, &gravity, &extents, &contact);
            total_impulse += ledger.impulse;
            assert!(pellet.is_finite());
        }
        assert!(pellet.velocity.norm() < 1e-3, "v={:?}", pellet.velocity);
        // Static equilibrium: penetration = m g / (k * spheres in contact).
        let spheres = pellet.sphere_offsets.len();
        #[allow(clippy::cast_precision_loss)]
        let expected_penetration = pellet.mass * 9.81 / (1000.0 * spheres as f64);
        let penetration = pellet.radius - pellet.position.z;
        assert!(
            (penetration - expected_penetration).abs() < 0.05 * expected_penetration,
            "penetration={penetration} expected={expected_penetration}"
        );
        // Total wall impulse balances gravity impulse plus the initial momentum change.
        #[allow(clippy::cast_precision_loss)]
        let gravity_impulse = pellet.mass * 9.81 * dt * steps as f64;
        assert!((total_impulse.z - gravity_impulse).abs() < 0.02 * gravity_impulse);
    }

    #[test]
    fn free_rotation_keeps_angular_momentum_and_energy() {
        let contact = ContactParams {
            normal_stiffness: 1000.0,
            restitution: 0.2,
            friction: 0.4,
        };
        let extents = Vector3::new(1.0, 1.0, 1.0);
        let mut pellet = pellet_at(Vector3::new(0.5, 0.5, 0.5));
        // |omega| ~ 20 rad/s, so omega * dt ~ 2e-3 rad per step (first-order rotation update).
        let momentum = Vector3::new(5e-8, 1e-7, 2.5e-8);
        pellet.angular_momentum = momentum;
        let energy0 = pellet.kinetic_energy();
        for _ in 0..2000 {
            pellet.integrate(1e-4, &Vector3::zeros(), &extents, &contact);
        }
        assert!((pellet.angular_momentum - momentum).norm() < 1e-20);
        let energy = pellet.kinetic_energy();
        assert!(
            (energy - energy0).abs() < 1e-2 * energy0,
            "{energy} vs {energy0}"
        );
    }
}
