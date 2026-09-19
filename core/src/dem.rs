//! Dry multisphere discrete element mechanics for pellets, box boundaries and
//! driven tool proxies.
//!
//! Intact pellets are oriented rigid bodies built from overlapping spheres fitted
//! to a cylinder of measured radius and length; mass and inertia come from the
//! solid cylinder, not from the sphere union. Contacts use a linear spring-dashpot
//! normal law, a tangential spring with Coulomb sliding, and constant-directional
//! rolling resistance. The slot plate is an explicit boundary: sphere-to-edge
//! contacts decide whether an oriented pellet can pass a slot, and no equal-volume
//! single-sphere shortcut exists anywhere in this module.
//!
//! Every reduction is executed in a fixed order so that stepping is bit-for-bit
//! deterministic across thread counts. Rayon parallelises independent pair
//! evaluations; accumulation into bodies happens sequentially in pair order.
//!
//! References:
//! - Cundall and Strack (1979), <https://doi.org/10.1680/geot.1979.29.1.47>
//! - Ai et al. (2011), rolling resistance models, <https://doi.org/10.1016/j.powtec.2010.09.030>
//! - Kruggel-Emden et al. (2008), multisphere shape representation,
//!   <https://doi.org/10.1016/j.powtec.2008.04.037>

use std::collections::BTreeMap;

use nalgebra::{UnitQuaternion, Vector3};
use rayon::prelude::*;

/// World-space vector type used throughout the mechanics modules.
pub type Vec3 = Vector3<f64>;
/// Orientation type mapping body-frame vectors into world-frame vectors.
pub type Quat = UnitQuaternion<f64>;

/// Gravitational acceleration magnitude in m/s^2.
pub const GRAVITY_M_S2: f64 = 9.806_65;
/// Maximum spheres allowed per multisphere pellet before the fit is rejected.
pub const MAX_SPHERES_PER_PELLET: usize = 16;
/// Safety factor applied to the linear spring contact period for stability.
pub const STABLE_DT_FACTOR: f64 = 0.1;
/// Fraction of normal stiffness used for the tangential spring.
const TANGENTIAL_STIFFNESS_RATIO: f64 = 2.0 / 7.0;

/// Identifier of a fixed box boundary surface.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Wall {
    /// Slotted sifting plate at local `z = 0`.
    Plate,
    /// Solid drawer floor at local `z = -drawer_depth`.
    DrawerFloor,
    /// Wall at local `x = 0`.
    XMin,
    /// Wall at local `x = size.x`.
    XMax,
    /// Wall at local `y = 0`.
    YMin,
    /// Wall at local `y = size.y`.
    YMax,
}

/// One side of a contact, used as a stable key for tangential history.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum ContactPartner {
    /// A sphere of a pellet, identified by pellet id and sphere index.
    Pellet {
        /// Stable pellet identifier.
        id: u64,
        /// Index of the sphere along the pellet axis.
        sphere: u32,
    },
    /// A driven tool sphere.
    Tool {
        /// Stable tool identifier.
        id: u64,
    },
    /// A fixed box boundary.
    Wall(Wall),
}

/// Ordered contact key; the first partner is always a pellet sphere.
pub type ContactKey = (ContactPartner, ContactPartner);

/// Persisted tangential spring displacement for one contact.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContactHistory {
    /// Contact identity.
    pub key: ContactKey,
    /// Accumulated tangential displacement in metres, world frame.
    pub tangential_m: [f64; 3],
}

/// Contact law parameters shared by all bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactParams {
    /// Linear normal stiffness in N/m.
    pub normal_stiffness_n_m: f64,
    /// Coulomb sliding friction coefficient.
    pub friction: f64,
    /// Coefficient of restitution in `[0, 1]` used to derive normal damping.
    pub restitution: f64,
    /// Rolling resistance coefficient (dimensionless, times effective radius).
    pub rolling_friction: f64,
}

impl ContactParams {
    /// Damping ratio matching the requested restitution for a linear dashpot.
    ///
    /// A restitution of zero maps to critical damping; values are clamped to
    /// `[0, 1]` before use.
    #[must_use]
    pub fn damping_ratio(&self) -> f64 {
        let e = self.restitution.clamp(0.0, 1.0);
        if e <= f64::EPSILON {
            return 1.0;
        }
        let ln_e = e.ln();
        (-ln_e / (std::f64::consts::PI.powi(2) + ln_e * ln_e).sqrt()).min(1.0)
    }

    /// Normal dashpot coefficient for an effective mass, in N s/m.
    #[must_use]
    pub fn normal_damping(&self, effective_mass_kg: f64) -> f64 {
        2.0 * self.damping_ratio() * (self.normal_stiffness_n_m * effective_mass_kg).sqrt()
    }
}

/// Box interior geometry expressed in box-local coordinates.
///
/// The bed occupies `[0, size]` above the plate; the drawer occupies
/// `z` in `[-drawer_depth, 0]` with the same footprint. Slots are rectangular
/// through-holes of `slot_length` along `x` and `slot_width` along `y`, arranged
/// in rows with `slot_pitch` along `y`. Along `x`, slots repeat with a solid bar
/// of `slot_width` between segments starting at `x = slot_width`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxGeometry {
    /// Interior size in metres.
    pub size_m: Vec3,
    /// Slot width across `y` in metres.
    pub slot_width_m: f64,
    /// Slot length along `x` in metres.
    pub slot_length_m: f64,
    /// Row pitch along `y` in metres.
    pub slot_pitch_m: f64,
    /// Drawer depth below the plate in metres.
    pub drawer_depth_m: f64,
}

impl BoxGeometry {
    /// Rectangle `[x0, x1] x [y0, y1]` of the slot under the point, if any.
    #[must_use]
    pub fn slot_under(&self, x: f64, y: f64) -> Option<[f64; 4]> {
        if !(0.0..=self.size_m.x).contains(&x) || !(0.0..=self.size_m.y).contains(&y) {
            return None;
        }
        let row = (y / self.slot_pitch_m).floor();
        let center_y = row.mul_add(self.slot_pitch_m, 0.5 * self.slot_pitch_m);
        let y0 = center_y - 0.5 * self.slot_width_m;
        let y1 = center_y + 0.5 * self.slot_width_m;
        if y < y0 || y > y1 || y1 > self.size_m.y {
            return None;
        }
        let period = self.slot_length_m + self.slot_width_m;
        let segment = ((x - self.slot_width_m) / period).floor();
        let x0 = segment.mul_add(period, self.slot_width_m);
        let x1 = x0 + self.slot_length_m;
        if x < x0 || x > x1 || x1 > self.size_m.x {
            return None;
        }
        Some([x0, x1, y0, y1])
    }

    /// Fraction of the plate area that is open, computed by regular sampling.
    #[must_use]
    pub fn open_fraction(&self) -> f64 {
        const SAMPLES: u32 = 200;
        let mut open = 0u32;
        for i in 0..SAMPLES {
            for j in 0..SAMPLES {
                let x = (f64::from(i) + 0.5) / f64::from(SAMPLES) * self.size_m.x;
                let y = (f64::from(j) + 0.5) / f64::from(SAMPLES) * self.size_m.y;
                if self.slot_under(x, y).is_some() {
                    open += 1;
                }
            }
        }
        f64::from(open) / f64::from(SAMPLES * SAMPLES)
    }

    /// Closest point on the solid plate to a sphere centre, with outward normal.
    fn plate_contact(&self, centre: Vec3, radius: f64) -> Option<Contact> {
        let (point, normal) = match self.slot_under(centre.x, centre.y) {
            None => {
                let sign = if centre.z >= 0.0 { 1.0 } else { -1.0 };
                (
                    Vec3::new(centre.x, centre.y, 0.0),
                    Vec3::new(0.0, 0.0, sign),
                )
            }
            Some([x0, x1, y0, y1]) => {
                let dx = (centre.x - x0).min(x1 - centre.x);
                let dy = (centre.y - y0).min(y1 - centre.y);
                let edge = if dx < dy {
                    let ex = if centre.x - x0 < x1 - centre.x {
                        x0
                    } else {
                        x1
                    };
                    Vec3::new(ex, centre.y, 0.0)
                } else {
                    let ey = if centre.y - y0 < y1 - centre.y {
                        y0
                    } else {
                        y1
                    };
                    Vec3::new(centre.x, ey, 0.0)
                };
                let offset = centre - edge;
                let distance = offset.norm();
                if distance <= f64::EPSILON {
                    return None;
                }
                (edge, offset / distance)
            }
        };
        let gap = (centre - point).dot(&normal);
        let overlap = radius - gap;
        (overlap > 0.0).then(|| Contact {
            normal,
            overlap,
            point: centre - normal * (radius - 0.5 * overlap),
        })
    }

    /// Contact of a sphere with one planar boundary, if overlapping.
    fn wall_contact(&self, wall: Wall, centre: Vec3, radius: f64) -> Option<Contact> {
        let (normal, gap) = match wall {
            Wall::Plate => return self.plate_contact(centre, radius),
            Wall::DrawerFloor => (Vec3::new(0.0, 0.0, 1.0), centre.z + self.drawer_depth_m),
            Wall::XMin => (Vec3::new(1.0, 0.0, 0.0), centre.x),
            Wall::XMax => (Vec3::new(-1.0, 0.0, 0.0), self.size_m.x - centre.x),
            Wall::YMin => (Vec3::new(0.0, 1.0, 0.0), centre.y),
            Wall::YMax => (Vec3::new(0.0, -1.0, 0.0), self.size_m.y - centre.y),
        };
        if wall != Wall::DrawerFloor && centre.z > self.size_m.z {
            return None;
        }
        let overlap = radius - gap;
        (overlap > 0.0).then(|| Contact {
            normal,
            overlap,
            point: centre - normal * (radius - 0.5 * overlap),
        })
    }
}

/// Geometric description of one overlap.
#[derive(Clone, Copy, Debug)]
struct Contact {
    /// Unit normal pointing from the second body toward the first.
    normal: Vec3,
    /// Overlap depth in metres.
    overlap: f64,
    /// Contact point in world coordinates.
    point: Vec3,
}

/// Rigid multisphere pellet.
#[derive(Clone, Debug, PartialEq)]
pub struct Pellet {
    /// Stable identifier used in contact history and ledgers.
    pub id: u64,
    /// Centre of mass in world coordinates.
    pub position: Vec3,
    /// Body-to-world orientation; the pellet axis is body `z`.
    pub orientation: Quat,
    /// Linear velocity in m/s.
    pub velocity: Vec3,
    /// Angular velocity in rad/s, world frame.
    pub angular_velocity: Vec3,
    /// Total mass in kg.
    pub mass: f64,
    /// Principal moments of inertia in the body frame, kg m^2.
    pub inertia_body: Vec3,
    /// Current sphere radius in metres (may swell).
    pub radius: f64,
    /// Sphere centre offsets along the body axis in metres.
    pub offsets: Vec<f64>,
}

impl Pellet {
    /// Fit a multisphere to a cylinder of the given radius, length and density.
    ///
    /// Spheres are spaced at most one radius apart so the surface stays smooth;
    /// mass and inertia are those of the solid cylinder.
    ///
    /// # Errors
    ///
    /// Returns an error when inputs are non-positive, non-finite, or when the
    /// aspect ratio would need more than [`MAX_SPHERES_PER_PELLET`] spheres.
    pub fn fit_cylinder(
        id: u64,
        radius_m: f64,
        length_m: f64,
        density_kg_m3: f64,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("radius_m", radius_m),
            ("length_m", length_m),
            ("density_kg_m3", density_kg_m3),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "`{name}` must be finite and positive, got '{value}'"
                ));
            }
        }
        let span = (length_m - 2.0 * radius_m).max(0.0);
        let segments = (span / radius_m).ceil();
        let count = if span <= 0.0 { 1.0 } else { segments + 1.0 };
        if count > count_to_f64(MAX_SPHERES_PER_PELLET) {
            return Err(format!(
                "pellet aspect ratio needs more than {MAX_SPHERES_PER_PELLET} spheres \
                 (radius '{radius_m}', length '{length_m}')"
            ));
        }
        let offsets = if span <= 0.0 {
            vec![0.0]
        } else {
            let spacing = span / segments;
            let mut offsets = Vec::new();
            let mut k = 0.0;
            while k <= segments + 0.5 {
                offsets.push(k.mul_add(spacing, -0.5 * span));
                k += 1.0;
            }
            offsets
        };
        let mass = density_kg_m3 * std::f64::consts::PI * radius_m * radius_m * length_m;
        let axial = 0.5 * mass * radius_m * radius_m;
        let transverse = mass * (3.0 * radius_m * radius_m + length_m * length_m) / 12.0;
        if !mass.is_finite()
            || mass <= 0.0
            || !axial.is_finite()
            || axial <= 0.0
            || !transverse.is_finite()
            || transverse <= 0.0
        {
            return Err("pellet mass or inertia is not representable".into());
        }
        Ok(Self {
            id,
            position: Vec3::zeros(),
            orientation: Quat::identity(),
            velocity: Vec3::zeros(),
            angular_velocity: Vec3::zeros(),
            mass,
            inertia_body: Vec3::new(transverse, transverse, axial),
            radius: radius_m,
            offsets,
        })
    }

    /// World-space centre of the sphere at `index`.
    #[must_use]
    pub fn sphere_centre(&self, index: usize) -> Vec3 {
        let offset = self.offsets.get(index).copied().unwrap_or(0.0);
        self.position + self.orientation * Vec3::new(0.0, 0.0, offset)
    }

    /// Radius of the smallest sphere centred on the pellet enclosing all spheres.
    #[must_use]
    pub fn bounding_radius(&self) -> f64 {
        self.offsets.iter().fold(0.0_f64, |acc, o| acc.max(o.abs())) + self.radius
    }

    /// Velocity of the material point at `point` in world coordinates.
    #[must_use]
    pub fn point_velocity(&self, point: Vec3) -> Vec3 {
        self.velocity + self.angular_velocity.cross(&(point - self.position))
    }

    /// Translational kinetic energy plus rotational energy in joules.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let omega_body = self.orientation.inverse() * self.angular_velocity;
        let rotational = 0.5
            * omega_body
                .component_mul(&self.inertia_body)
                .dot(&omega_body);
        0.5 * self.mass * self.velocity.norm_squared() + rotational
    }

    /// Advance the rigid body under a resultant force and torque.
    fn integrate(&mut self, force: Vec3, torque: Vec3, dt: f64) {
        self.velocity += force / self.mass * dt;
        self.position += self.velocity * dt;
        let inverse = self.orientation.inverse();
        let omega_body = inverse * self.angular_velocity;
        let torque_body = inverse * torque;
        let inertia_omega = self.inertia_body.component_mul(&omega_body);
        let gyroscopic = omega_body.cross(&inertia_omega);
        let omega_dot = (torque_body - gyroscopic).component_div(&self.inertia_body);
        let omega_body = omega_body + omega_dot * dt;
        self.angular_velocity = self.orientation * omega_body;
        let rotation = Quat::from_scaled_axis(self.angular_velocity * dt);
        self.orientation = rotation * self.orientation;
        self.orientation.renormalize();
    }
}

/// Driven tool sphere (paw or stir tool proxy).
///
/// The tool integrates its own mass under `drive_force` plus contact reactions,
/// so a caller can implement compliant, force-limited motion by choosing the
/// drive each substep.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    /// Stable identifier used in contact history.
    pub id: u64,
    /// Centre in world coordinates.
    pub position: Vec3,
    /// Velocity in m/s.
    pub velocity: Vec3,
    /// Sphere radius in metres.
    pub radius: f64,
    /// Tool mass in kg (used only for its own integration).
    pub mass: f64,
    /// Force applied by the driver during the next step, in newtons.
    pub drive_force: Vec3,
}

/// Per-step diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepReport {
    /// Total contact force exerted by pellets on each tool, in tool order.
    pub tool_reaction_n: Vec<Vec3>,
    /// Work done by tools on pellets during the step, in joules.
    pub tool_work_j: f64,
    /// Largest sphere overlap observed, in metres.
    pub max_overlap_m: f64,
    /// Number of active pellet-pellet contacts.
    pub pellet_contacts: usize,
}

/// Simulation world holding pellets, tools, boundaries and contact history.
#[derive(Clone, Debug)]
pub struct World {
    /// Box geometry in local coordinates (pellet positions are local).
    pub geometry: BoxGeometry,
    /// Contact parameters.
    pub params: ContactParams,
    /// Pellets in stable id order.
    pub pellets: Vec<Pellet>,
    /// Driven tools.
    pub tools: Vec<Tool>,
    /// Gravitational acceleration vector in m/s^2 (defaults to `-z`).
    pub gravity: Vec3,
    /// Tangential spring displacements for active contacts.
    history: BTreeMap<ContactKey, Vec3>,
}

/// One evaluated contact ready for sequential accumulation.
struct PairForce {
    key: ContactKey,
    first: usize,
    second: Option<usize>,
    force: Vec3,
    torque_first: Vec3,
    torque_second: Vec3,
    tangential: Vec3,
    overlap: f64,
}

/// Relative motion and effective properties at one contact.
#[derive(Clone, Copy)]
struct Kinematics {
    relative_velocity: Vec3,
    effective_mass: f64,
    effective_radius: f64,
    relative_omega: Vec3,
}

/// Body index of the second contact partner.
#[derive(Clone, Copy)]
enum Other {
    Pellet(usize),
    Tool(usize),
    Wall,
}

impl World {
    /// Create an empty world.
    #[must_use]
    pub fn new(geometry: BoxGeometry, params: ContactParams) -> Self {
        Self {
            geometry,
            params,
            pellets: Vec::new(),
            tools: Vec::new(),
            gravity: Vec3::new(0.0, 0.0, -GRAVITY_M_S2),
            history: BTreeMap::new(),
        }
    }

    /// Largest stable explicit step for the current bodies, or `None` when
    /// there are no pellets.
    #[must_use]
    pub fn stable_dt(&self) -> Option<f64> {
        let min_mass = self
            .pellets
            .iter()
            .map(|p| p.mass)
            .fold(f64::INFINITY, f64::min);
        min_mass
            .is_finite()
            .then(|| STABLE_DT_FACTOR * (min_mass / self.params.normal_stiffness_n_m).sqrt())
    }

    /// Persisted contact history in key order.
    #[must_use]
    pub fn history(&self) -> Vec<ContactHistory> {
        self.history
            .iter()
            .map(|(key, t)| ContactHistory {
                key: *key,
                tangential_m: [t.x, t.y, t.z],
            })
            .collect()
    }

    /// Replace the contact history (used on resume).
    pub fn set_history(&mut self, entries: &[ContactHistory]) {
        self.history = entries
            .iter()
            .map(|e| {
                let t = e.tangential_m;
                (e.key, Vec3::new(t[0], t[1], t[2]))
            })
            .collect();
    }

    /// Total linear momentum of pellets in kg m/s.
    #[must_use]
    pub fn momentum(&self) -> Vec3 {
        self.pellets
            .iter()
            .fold(Vec3::zeros(), |acc, p| acc + p.velocity * p.mass)
    }

    /// Total pellet kinetic energy in joules.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.pellets.iter().map(Pellet::kinetic_energy).sum()
    }

    /// Remove pellets whose ids satisfy `predicate`, returning them in order.
    pub fn drain_pellets(&mut self, predicate: impl Fn(&Pellet) -> bool) -> Vec<Pellet> {
        let (removed, kept): (Vec<_>, Vec<_>) = self.pellets.drain(..).partition(|p| predicate(p));
        self.pellets = kept;
        let removed_ids: Vec<u64> = removed.iter().map(|p| p.id).collect();
        self.history.retain(|(a, b), _| {
            let hit = |p: &ContactPartner| match p {
                ContactPartner::Pellet { id, .. } => removed_ids.contains(id),
                _ => false,
            };
            !hit(a) && !hit(b)
        });
        removed
    }

    /// Advance the world by one explicit substep of `dt` seconds.
    ///
    /// Callers must keep `dt` at or below [`World::stable_dt`].
    pub fn step(&mut self, dt: f64) -> StepReport {
        let spheres = self.collect_spheres();
        let pairs = Self::find_pairs(&spheres);
        let mut forces: Vec<PairForce> = pairs
            .par_iter()
            .filter_map(|&(a, b)| self.pellet_pair_force(&spheres, a, b, dt))
            .collect();
        let boundary: Vec<PairForce> = spheres
            .par_iter()
            .flat_map_iter(|s| self.boundary_forces(s, dt))
            .collect();
        forces.extend(boundary);
        self.accumulate(&forces, dt)
    }

    /// Flatten all pellet spheres in deterministic order.
    fn collect_spheres(&self) -> Vec<Sphere> {
        let mut out = Vec::new();
        for (body, pellet) in self.pellets.iter().enumerate() {
            for (index, _) in pellet.offsets.iter().enumerate() {
                out.push(Sphere {
                    body,
                    index,
                    centre: pellet.sphere_centre(index),
                    radius: pellet.radius,
                });
            }
        }
        out
    }

    /// Candidate sphere pairs from a uniform grid, sorted by sphere index.
    fn find_pairs(spheres: &[Sphere]) -> Vec<(usize, usize)> {
        let max_radius = spheres.iter().map(|s| s.radius).fold(0.0_f64, f64::max);
        if spheres.is_empty() || max_radius <= 0.0 {
            return Vec::new();
        }
        let cell = 2.0 * max_radius * 1.01;
        let mut keyed: Vec<(CellKey, usize)> = spheres
            .iter()
            .enumerate()
            .map(|(i, s)| (cell_key(s.centre, cell), i))
            .collect();
        keyed.sort_unstable();
        let keys: Vec<CellKey> = keyed.iter().map(|k| k.0).collect();
        let pairs: Vec<Vec<(usize, usize)>> = spheres
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let mut local = Vec::new();
                let base = cell_key(s.centre, cell);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let key = (base.0 + dx, base.1 + dy, base.2 + dz);
                            let start = keys.partition_point(|k| *k < key);
                            let end = keys.partition_point(|k| *k <= key);
                            for &(_, j) in &keyed[start..end] {
                                if j > i && spheres[j].body != s.body {
                                    let gap = (spheres[j].centre - s.centre).norm();
                                    if gap < s.radius + spheres[j].radius {
                                        local.push((i, j));
                                    }
                                }
                            }
                        }
                    }
                }
                local.sort_unstable();
                local
            })
            .collect();
        pairs.into_iter().flatten().collect()
    }

    /// Evaluate a pellet-pellet sphere contact.
    fn pellet_pair_force(
        &self,
        spheres: &[Sphere],
        a: usize,
        b: usize,
        dt: f64,
    ) -> Option<PairForce> {
        let (sa, sb) = (&spheres[a], &spheres[b]);
        let offset = sa.centre - sb.centre;
        let distance = offset.norm();
        let overlap = sa.radius + sb.radius - distance;
        if overlap <= 0.0 || distance <= f64::EPSILON {
            return None;
        }
        let normal = offset / distance;
        let contact = Contact {
            normal,
            overlap,
            point: sa.centre - normal * (sa.radius - 0.5 * overlap),
        };
        let (pa, pb) = (&self.pellets[sa.body], &self.pellets[sb.body]);
        let key = (
            ContactPartner::Pellet {
                id: pa.id,
                sphere: index_to_u32(sa.index),
            },
            ContactPartner::Pellet {
                id: pb.id,
                sphere: index_to_u32(sb.index),
            },
        );
        let relative = pa.point_velocity(contact.point) - pb.point_velocity(contact.point);
        let effective_mass = pa.mass * pb.mass / (pa.mass + pb.mass);
        let radius = sa.radius * sb.radius / (sa.radius + sb.radius);
        let omega = pa.angular_velocity - pb.angular_velocity;
        let kinematics = Kinematics {
            relative_velocity: relative,
            effective_mass,
            effective_radius: radius,
            relative_omega: omega,
        };
        let (force, tangential, rolling) = self.contact_force(key, &contact, &kinematics, dt);
        Some(PairForce {
            key,
            first: sa.body,
            second: Some(sb.body),
            force,
            torque_first: (contact.point - pa.position).cross(&force) + rolling,
            torque_second: (contact.point - pb.position).cross(&-force) - rolling,
            tangential,
            overlap,
        })
    }

    /// Evaluate one sphere against walls and tools.
    fn boundary_forces(&self, sphere: &Sphere, dt: f64) -> Vec<PairForce> {
        let pellet = &self.pellets[sphere.body];
        let me = ContactPartner::Pellet {
            id: pellet.id,
            sphere: index_to_u32(sphere.index),
        };
        let mut out = Vec::new();
        let walls = [
            Wall::Plate,
            Wall::DrawerFloor,
            Wall::XMin,
            Wall::XMax,
            Wall::YMin,
            Wall::YMax,
        ];
        for wall in walls {
            let Some(contact) = self
                .geometry
                .wall_contact(wall, sphere.centre, sphere.radius)
            else {
                continue;
            };
            let key = (me, ContactPartner::Wall(wall));
            let relative = pellet.point_velocity(contact.point);
            let omega = pellet.angular_velocity;
            let kinematics = Kinematics {
                relative_velocity: relative,
                effective_mass: pellet.mass,
                effective_radius: sphere.radius,
                relative_omega: omega,
            };
            let (force, tangential, rolling) = self.contact_force(key, &contact, &kinematics, dt);
            out.push(PairForce {
                key,
                first: sphere.body,
                second: None,
                force,
                torque_first: (contact.point - pellet.position).cross(&force) + rolling,
                torque_second: Vec3::zeros(),
                tangential,
                overlap: contact.overlap,
            });
        }
        for (t, tool) in self.tools.iter().enumerate() {
            let offset = sphere.centre - tool.position;
            let distance = offset.norm();
            let overlap = sphere.radius + tool.radius - distance;
            if overlap <= 0.0 || distance <= f64::EPSILON {
                continue;
            }
            let normal = offset / distance;
            let contact = Contact {
                normal,
                overlap,
                point: sphere.centre - normal * (sphere.radius - 0.5 * overlap),
            };
            let key = (me, ContactPartner::Tool { id: tool.id });
            let relative = pellet.point_velocity(contact.point) - tool.velocity;
            let radius = sphere.radius * tool.radius / (sphere.radius + tool.radius);
            let kinematics = Kinematics {
                relative_velocity: relative,
                effective_mass: pellet.mass,
                effective_radius: radius,
                relative_omega: pellet.angular_velocity,
            };
            let (force, tangential, rolling) = self.contact_force(key, &contact, &kinematics, dt);
            out.push(PairForce {
                key,
                first: sphere.body,
                second: Some(usize::MAX - t),
                force,
                torque_first: (contact.point - pellet.position).cross(&force) + rolling,
                torque_second: Vec3::zeros(),
                tangential,
                overlap,
            });
        }
        out
    }

    /// Spring-dashpot normal force, tangential spring with Coulomb limit and
    /// rolling resistance torque. Returns (force on first body, new tangential
    /// displacement, rolling torque on first body).
    fn contact_force(
        &self,
        key: ContactKey,
        contact: &Contact,
        kinematics: &Kinematics,
        dt: f64,
    ) -> (Vec3, Vec3, Vec3) {
        let params = self.params;
        let Kinematics {
            relative_velocity,
            effective_mass,
            effective_radius,
            relative_omega,
        } = *kinematics;
        let n = contact.normal;
        let vn = relative_velocity.dot(&n);
        let normal_magnitude = (params.normal_stiffness_n_m * contact.overlap
            - params.normal_damping(effective_mass) * vn)
            .max(0.0);
        let vt = relative_velocity - n * vn;
        let previous = self.history.get(&key).copied().unwrap_or_else(Vec3::zeros);
        let previous = previous - n * previous.dot(&n);
        let mut xi = previous + vt * dt;
        let kt = TANGENTIAL_STIFFNESS_RATIO * params.normal_stiffness_n_m;
        let ct = TANGENTIAL_STIFFNESS_RATIO.sqrt() * params.normal_damping(effective_mass);
        let mut ft = -(xi * kt) - vt * ct;
        let limit = params.friction * normal_magnitude;
        let ft_norm = ft.norm();
        if ft_norm > limit {
            ft = if ft_norm > f64::EPSILON {
                ft * (limit / ft_norm)
            } else {
                Vec3::zeros()
            };
            xi = -ft / kt;
        }
        let omega_norm = relative_omega.norm();
        let rolling = if omega_norm > f64::EPSILON {
            -relative_omega / omega_norm
                * (params.rolling_friction * effective_radius * normal_magnitude)
        } else {
            Vec3::zeros()
        };
        (n * normal_magnitude + ft, xi, rolling)
    }

    /// Apply forces sequentially, integrate bodies and rebuild history.
    fn accumulate(&mut self, forces: &[PairForce], dt: f64) -> StepReport {
        let gravity = self.gravity;
        let mut pellet_force: Vec<Vec3> = self.pellets.iter().map(|p| gravity * p.mass).collect();
        let mut pellet_torque = vec![Vec3::zeros(); self.pellets.len()];
        let mut tool_force = vec![Vec3::zeros(); self.tools.len()];
        let mut report = StepReport::default();
        let mut history = BTreeMap::new();
        for pair in forces {
            pellet_force[pair.first] += pair.force;
            pellet_torque[pair.first] += pair.torque_first;
            match classify(pair.second, self.tools.len()) {
                Other::Pellet(j) => {
                    pellet_force[j] -= pair.force;
                    pellet_torque[j] += pair.torque_second;
                    report.pellet_contacts += 1;
                }
                Other::Tool(t) => tool_force[t] -= pair.force,
                Other::Wall => {}
            }
            report.max_overlap_m = report.max_overlap_m.max(pair.overlap);
            history.insert(pair.key, pair.tangential);
        }
        self.history = history;
        for (pellet, (force, torque)) in self
            .pellets
            .iter_mut()
            .zip(pellet_force.into_iter().zip(pellet_torque))
        {
            pellet.integrate(force, torque, dt);
        }
        for (tool, reaction) in self.tools.iter_mut().zip(tool_force) {
            report.tool_work_j += -reaction.dot(&tool.velocity) * dt;
            report.tool_reaction_n.push(reaction);
            tool.velocity += (tool.drive_force + reaction) / tool.mass * dt;
            tool.position += tool.velocity * dt;
        }
        report
    }
}

/// Map the packed `second` index back to its body class.
fn classify(second: Option<usize>, tool_count: usize) -> Other {
    match second {
        None => Other::Wall,
        Some(j) if j > usize::MAX - tool_count => Other::Tool(usize::MAX - j),
        Some(j) => Other::Pellet(j),
    }
}

/// Flattened sphere used for neighbour search.
#[derive(Clone, Copy, Debug)]
struct Sphere {
    body: usize,
    index: usize,
    centre: Vec3,
    radius: f64,
}

type CellKey = (i64, i64, i64);

/// Integer grid cell containing a point.
fn cell_key(p: Vec3, cell: f64) -> CellKey {
    (
        floor_to_i64(p.x / cell),
        floor_to_i64(p.y / cell),
        floor_to_i64(p.z / cell),
    )
}

/// Floor a finite value into `i64`, saturating at the `i32` range.
///
/// Positions are validated finite and bounded by box geometry before they
/// reach here, so the clamp never engages in practice; it exists to keep the
/// cast total for non-finite garbage.
#[expect(
    clippy::cast_possible_truncation,
    reason = "value is floored and clamped into the i32 range, so the cast is exact"
)]
pub(crate) fn floor_to_i64(value: f64) -> i64 {
    let clamped = value
        .floor()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX));
    if clamped.is_nan() {
        return 0;
    }
    clamped as i64
}

/// Convert a bounded count into `f64` without a lossy cast.
pub(crate) fn count_to_f64(n: usize) -> f64 {
    u32::try_from(n).map_or(f64::INFINITY, f64::from)
}

/// Convert a sphere index into the compact `u32` used in contact keys.
fn index_to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> BoxGeometry {
        BoxGeometry {
            size_m: Vec3::new(0.12, 0.10, 0.08),
            slot_width_m: 0.005,
            slot_length_m: 0.015,
            slot_pitch_m: 0.02,
            drawer_depth_m: 0.03,
        }
    }

    fn params(restitution: f64, friction: f64) -> ContactParams {
        ContactParams {
            normal_stiffness_n_m: 2000.0,
            friction,
            restitution,
            rolling_friction: 0.05,
        }
    }

    fn settle(world: &mut World, seconds: f64) -> f64 {
        let dt = world.stable_dt().unwrap_or(1e-4);
        let steps = (seconds / dt).ceil();
        let mut k = 0.0;
        while k < steps {
            world.step(dt);
            k += 1.0;
        }
        dt
    }

    #[test]
    fn cylinder_fit_matches_measured_mass_and_covers_length() {
        let pellet = Pellet::fit_cylinder(1, 0.003, 0.012, 1100.0).unwrap();
        let expected = 1100.0 * std::f64::consts::PI * 0.003 * 0.003 * 0.012;
        assert!((pellet.mass - expected).abs() < 1e-12);
        assert_eq!(pellet.offsets.len(), 3);
        assert!((pellet.bounding_radius() - 0.006).abs() < 1e-12);
        assert!(pellet.inertia_body.x > pellet.inertia_body.z);
        assert!(Pellet::fit_cylinder(1, 0.001, 0.5, 1100.0).is_err());
        assert!(Pellet::fit_cylinder(1, -0.001, 0.01, 1100.0).is_err());
    }

    #[test]
    fn restitution_sets_bounce_ratio() {
        let mut measured = Vec::new();
        for e in [0.2, 0.6] {
            let mut world = World::new(geometry(), params(e, 0.0));
            let mut pellet = Pellet::fit_cylinder(1, 0.003, 0.003, 1100.0).unwrap();
            pellet.position = Vec3::new(0.0125, 0.0075, 0.02);
            world.pellets.push(pellet);
            let dt = world.stable_dt().unwrap();
            let mut impact = 0.0_f64;
            let mut rebound = 0.0_f64;
            let mut hit = false;
            for _ in 0..40_000 {
                let vz = world.pellets[0].velocity.z;
                world.step(dt);
                let z = world.pellets[0].position.z;
                if !hit && z < 0.003 {
                    hit = true;
                    impact = vz.abs();
                }
                if hit {
                    rebound = rebound.max(world.pellets[0].velocity.z);
                    if world.pellets[0].velocity.z < 0.0 && rebound > 0.0 {
                        break;
                    }
                }
            }
            assert!(hit, "pellet never reached the plate");
            let ratio = rebound / impact;
            // A non-cohesive dashpot (normal force clamped at zero) restitutes
            // slightly more than the linear theory; see docs/dem.md.
            assert!(
                ratio >= e - 0.02 && ratio <= e + 0.15,
                "e={e} measured={ratio}"
            );
            measured.push(ratio);
        }
        assert!(
            measured[1] > measured[0],
            "restitution not monotonic: {measured:?}"
        );
    }

    #[test]
    fn sliding_sphere_decelerates_at_mu_g() {
        let mu = 0.3;
        let mut world = World::new(geometry(), params(0.0, mu));
        let mut pellet = Pellet::fit_cylinder(1, 0.002, 0.002, 1100.0).unwrap();
        pellet.position = Vec3::new(0.005, 0.0075, 0.002 - 1e-6);
        pellet.velocity = Vec3::new(0.5, 0.0, 0.0);
        world.pellets.push(pellet);
        world.params.rolling_friction = 0.0;
        let dt = world.stable_dt().unwrap();
        let mut elapsed = 0.0;
        settle(&mut world, 5.0 * dt);
        let v0 = world.pellets[0].velocity.x;
        while elapsed < 0.02 {
            world.step(dt);
            elapsed += dt;
        }
        let v1 = world.pellets[0].velocity.x;
        let decel = (v0 - v1) / elapsed;
        assert!(decel > 0.0, "no deceleration");
        assert!(
            decel <= mu * GRAVITY_M_S2 * 1.05,
            "decel {decel} exceeds Coulomb limit"
        );
    }

    #[test]
    fn neighbour_grid_matches_brute_force() {
        let mut world = World::new(geometry(), params(0.2, 0.4));
        let mut state = 7.0_f64;
        let mut next = move || {
            state = state.mul_add(9301.0, 49_297.0) % 233_280.0;
            state / 233_280.0
        };
        for id in 0..40u32 {
            let mut pellet = Pellet::fit_cylinder(u64::from(id), 0.003, 0.012, 1100.0).unwrap();
            let (a, b, c) = (next(), next(), next());
            pellet.position = Vec3::new(0.01 + a * 0.1, 0.01 + b * 0.08, 0.01 + c * 0.04);
            pellet.orientation = Quat::from_scaled_axis(Vec3::new(a, b, c) * 3.0);
            world.pellets.push(pellet);
        }
        let spheres = world.collect_spheres();
        let grid = World::find_pairs(&spheres);
        let mut brute = Vec::new();
        for i in 0..spheres.len() {
            for j in (i + 1)..spheres.len() {
                if spheres[i].body != spheres[j].body {
                    let gap = (spheres[i].centre - spheres[j].centre).norm();
                    if gap < spheres[i].radius + spheres[j].radius {
                        brute.push((i, j));
                    }
                }
            }
        }
        assert!(!brute.is_empty(), "test needs overlapping pellets");
        assert_eq!(grid, brute);
    }

    #[test]
    fn thin_pellet_passes_slot_only_when_aligned() {
        let geometry = geometry();
        let slot = geometry.slot_under(0.0125, 0.01).expect("slot present");
        assert!((slot[0] - 0.005).abs() < 1e-12);
        assert!((slot[3] - slot[2] - 0.005).abs() < 1e-12);
        for (aligned, expect_through) in [(true, true), (false, false)] {
            let mut world = World::new(geometry, params(0.1, 0.4));
            let mut pellet = Pellet::fit_cylinder(1, 0.0015, 0.012, 1100.0).unwrap();
            pellet.position = Vec3::new(0.0125, 0.01, 0.02);
            pellet.orientation = if aligned {
                Quat::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2)
            } else {
                Quat::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2)
            };
            world.pellets.push(pellet);
            settle(&mut world, 0.6);
            let z = world.pellets[0].position.z;
            assert_eq!(z < -0.005, expect_through, "aligned={aligned} z={z}");
            assert!(z > -geometry.drawer_depth_m - 1e-3);
        }
    }

    #[test]
    fn packed_bed_settles_inside_box_with_bounded_overlap() {
        let geometry = BoxGeometry {
            size_m: Vec3::new(0.05, 0.05, 0.08),
            ..geometry()
        };
        let mut world = World::new(geometry, params(0.2, 0.4));
        world.params.normal_stiffness_n_m = 500.0;
        let mut id = 0u32;
        for iz in 0..8 {
            for ix in 0..4 {
                for iy in 0..4 {
                    let mut pellet =
                        Pellet::fit_cylinder(u64::from(id), 0.003, 0.009, 1100.0).unwrap();
                    pellet.position = Vec3::new(
                        0.007 + f64::from(ix) * 0.012 + f64::from(iz % 2) * 0.003,
                        0.007 + f64::from(iy) * 0.012,
                        0.006 + f64::from(iz) * 0.009,
                    );
                    pellet.orientation =
                        Quat::from_axis_angle(
                            &Vector3::y_axis(),
                            std::f64::consts::FRAC_PI_2 + f64::from(id) * 0.4,
                        ) * Quat::from_axis_angle(&Vector3::z_axis(), f64::from(id) * 0.9);
                    world.pellets.push(pellet);
                    id += 1;
                }
            }
        }
        settle(&mut world, 0.8);
        let report = world.step(world.stable_dt().unwrap());
        assert!(report.pellet_contacts > 0);
        assert!(
            report.max_overlap_m < 0.3 * 0.003,
            "overlap {}",
            report.max_overlap_m
        );
        let mut top = 0.0_f64;
        for pellet in &world.pellets {
            assert!(pellet.position.x > 0.0 && pellet.position.x < 0.05);
            assert!(pellet.position.y > 0.0 && pellet.position.y < 0.05);
            assert!(
                pellet.position.z > -0.001,
                "intact pellet fell through slot"
            );
            top = top.max(pellet.position.z);
        }
        assert!(top < 0.05, "bed did not settle, top {top}");
        assert!(world.kinetic_energy() < 1e-6 * count_to_f64(world.pellets.len()));
        let solid = count_to_f64(world.pellets.len()) * world.pellets[0].mass / 1100.0;
        let packing = solid / (0.05 * 0.05 * (top + 0.003));
        assert!(
            packing > 0.3 && packing < 0.75,
            "packing fraction {packing}"
        );
    }

    #[test]
    fn pellet_on_slope_holds_below_friction_angle_and_slides_above() {
        let mu = 0.5_f64;
        for (tilt_deg, expect_slide) in [(10.0, false), (40.0, true)] {
            let mut world = World::new(geometry(), params(0.1, mu));
            let tilt = f64::to_radians(tilt_deg);
            world.gravity = Vec3::new(GRAVITY_M_S2 * tilt.sin(), 0.0, -GRAVITY_M_S2 * tilt.cos());
            let mut pellet = Pellet::fit_cylinder(1, 0.003, 0.012, 1100.0).unwrap();
            pellet.position = Vec3::new(0.02, 0.05, 0.003);
            pellet.orientation =
                Quat::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2);
            world.pellets.push(pellet);
            settle(&mut world, 0.5);
            let moved = world.pellets[0].position.x - 0.02;
            assert_eq!(moved > 0.005, expect_slide, "tilt={tilt_deg} moved={moved}");
        }
    }

    #[test]
    fn tool_pushes_pellets_and_reports_reaction_and_work() {
        let mut world = World::new(geometry(), params(0.2, 0.4));
        let mut pellet = Pellet::fit_cylinder(1, 0.003, 0.012, 1100.0).unwrap();
        pellet.position = Vec3::new(0.06, 0.05, 0.003);
        pellet.orientation = Quat::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        world.pellets.push(pellet);
        world.tools.push(Tool {
            id: 1,
            position: Vec3::new(0.045, 0.05, 0.005),
            velocity: Vec3::new(0.2, 0.0, 0.0),
            radius: 0.01,
            mass: 0.05,
            drive_force: Vec3::zeros(),
        });
        let dt = world.stable_dt().unwrap();
        let mut total_work = 0.0;
        let mut max_reaction = 0.0_f64;
        for _ in 0..2000 {
            world.tools[0].velocity = Vec3::new(0.2, 0.0, 0.0);
            let report = world.step(dt);
            total_work += report.tool_work_j;
            max_reaction = max_reaction.max(report.tool_reaction_n[0].norm());
        }
        assert!(max_reaction > 0.0);
        assert!(total_work > 0.0);
        assert!(world.pellets[0].position.x > 0.06);
    }

    #[test]
    fn history_round_trips_and_drain_removes_entries() {
        let mut world = World::new(geometry(), params(0.2, 0.4));
        let mut pellet = Pellet::fit_cylinder(3, 0.003, 0.012, 1100.0).unwrap();
        pellet.position = Vec3::new(0.06, 0.05, 0.0029);
        pellet.orientation = Quat::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        pellet.velocity = Vec3::new(0.1, 0.0, 0.0);
        world.pellets.push(pellet);
        world.step(1e-4);
        let history = world.history();
        assert!(!history.is_empty());
        let mut copy = World::new(geometry(), params(0.2, 0.4));
        copy.pellets.clone_from(&world.pellets);
        copy.set_history(&history);
        assert_eq!(copy.history(), history);
        let removed = world.drain_pellets(|p| p.id == 3);
        assert_eq!(removed.len(), 1);
        assert!(world.history().is_empty());
    }

    #[test]
    fn open_fraction_is_bounded() {
        let open = geometry().open_fraction();
        assert!(open > 0.05 && open < 0.5, "open fraction {open}");
        assert!(geometry().slot_under(0.002, 0.01).is_none());
        assert!(geometry().slot_under(0.5, 0.01).is_none());
    }
}
