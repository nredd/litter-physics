//! Checkpoint records, ledgers and the deterministic RNG for household runs.
//!
//! Everything a run needs to resume identically lives here as plain serde
//! types: pellet kinematics and moisture, contact history, field arrays, event
//! progress, ledgers and the RNG state. Non-serde runtime objects are rebuilt
//! from these records on resume and validated for consistency.
//!
//! References: `docs/wire-contract.md`, `docs/household.md`.

use serde::{Deserialize, Serialize};

use super::field::Field;
use crate::dem::{ContactHistory, Pellet, Quat, Vec3};

/// Species masses in one compartment, kg.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compartment {
    /// Wood solids (intact pellets plus fines).
    pub wood: f64,
    /// Deposit solids.
    pub waste: f64,
    /// Water in all phases.
    pub water: f64,
}

impl Compartment {
    /// Add species from another compartment.
    pub fn add(&mut self, other: Self) {
        self.wood += other.wood;
        self.waste += other.waste;
        self.water += other.water;
    }

    /// Move everything out, returning what was moved.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Whether all species are finite and non-negative.
    #[must_use]
    pub fn is_sane(&self) -> bool {
        [self.wood, self.waste, self.water]
            .iter()
            .all(|v| v.is_finite() && *v >= -1e-15)
    }
}

/// Deterministic `splitmix64` generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng(pub u64);

impl Rng {
    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, 1)` built from the top 32 bits.
    pub fn next_f64(&mut self) -> f64 {
        let top = u32::try_from(self.next_u64() >> 32).unwrap_or(0);
        f64::from(top) / 4_294_967_296.0
    }

    /// Uniformly random orientation.
    pub fn orientation(&mut self) -> Quat {
        let axis = Vec3::new(
            self.next_f64().mul_add(2.0, -1.0),
            self.next_f64().mul_add(2.0, -1.0),
            self.next_f64().mul_add(2.0, -1.0),
        );
        let angle = self.next_f64() * std::f64::consts::TAU;
        let axis = nalgebra::Unit::try_new(axis, 1e-9).unwrap_or_else(Vec3::z_axis);
        Quat::from_axis_angle(&axis, angle)
    }
}

/// Persisted pellet: kinematics plus moisture and damage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PelletRecord {
    pub id: u64,
    pub position_m: [f64; 3],
    pub orientation_wijk: [f64; 4],
    pub velocity_m_s: [f64; 3],
    pub angular_velocity_rad_s: [f64; 3],
    pub water_kg: f64,
    pub damage: f64,
}

impl PelletRecord {
    /// Snapshot a live pellet with its moisture state.
    #[must_use]
    pub fn from_pellet(pellet: &Pellet, moisture: &Moisture) -> Self {
        let q = pellet.orientation.quaternion();
        Self {
            id: pellet.id,
            position_m: pellet.position.into(),
            orientation_wijk: [q.w, q.i, q.j, q.k],
            velocity_m_s: pellet.velocity.into(),
            angular_velocity_rad_s: pellet.angular_velocity.into(),
            water_kg: moisture.water_kg,
            damage: moisture.damage,
        }
    }

    /// Rebuild a live pellet from the reference shape.
    ///
    /// # Errors
    ///
    /// Returns an error when any value is non-finite, negative where it must
    /// not be, or when the orientation is degenerate.
    pub fn to_pellet(&self, reference: &Pellet) -> Result<(Pellet, Moisture), String> {
        let all = self
            .position_m
            .iter()
            .chain(&self.orientation_wijk)
            .chain(&self.velocity_m_s)
            .chain(&self.angular_velocity_rad_s)
            .chain([&self.water_kg, &self.damage]);
        for value in all {
            if !value.is_finite() {
                return Err(format!("pellet {} holds a non-finite value", self.id));
            }
        }
        if self.water_kg < 0.0 || self.damage < 0.0 {
            return Err(format!("pellet {} has negative water or damage", self.id));
        }
        let [w, i, j, k] = self.orientation_wijk;
        let raw = nalgebra::Quaternion::new(w, i, j, k);
        if raw.norm() < 1e-6 {
            return Err(format!("pellet {} has a degenerate orientation", self.id));
        }
        let mut pellet = reference.clone();
        pellet.id = self.id;
        pellet.position = Vec3::from(self.position_m);
        pellet.orientation = Quat::from_quaternion(raw);
        pellet.velocity = Vec3::from(self.velocity_m_s);
        pellet.angular_velocity = Vec3::from(self.angular_velocity_rad_s);
        Ok((
            pellet,
            Moisture {
                water_kg: self.water_kg,
                damage: self.damage,
            },
        ))
    }
}

/// Moisture and damage state of one pellet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moisture {
    /// Absorbed water, kg.
    pub water_kg: f64,
    /// Accumulated breakup damage; the pellet disintegrates at 1.
    pub damage: f64,
}

/// Persisted driven tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRecord {
    pub event_index: usize,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
}

/// Persisted per-box state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxRecord {
    pub pellets: Vec<PelletRecord>,
    pub history: Vec<ContactHistory>,
    pub field: Field,
    pub drawer: Compartment,
    pub tool: Option<ToolRecord>,
}

/// Progress of one scheduled event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventProgress {
    pub started: bool,
    pub finished: bool,
    pub applied_kg: f64,
}

/// Mode-private checkpoint state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub step_index: u64,
    pub next_record_index: u64,
    pub rng: Rng,
    pub next_pellet_id: u64,
    pub boxes: Vec<BoxRecord>,
    pub events: Vec<EventProgress>,
    pub supplied: Compartment,
    pub floor: Compartment,
    pub removed: Compartment,
    pub evaporated: Compartment,
    pub tool_work_j: f64,
    pub scoop_clean_wood_kg: f64,
    pub breakups: u64,
    pub max_overlap_m: f64,
}

/// Checkpoint envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub mode: String,
    pub request: serde_json::Value,
    pub time_s: f64,
    pub state: State,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut a = Rng(42);
        let mut b = Rng(42);
        for _ in 0..100 {
            let x = a.next_f64();
            assert!((0.0..1.0).contains(&x));
            assert!((x - b.next_f64()).abs() < f64::EPSILON);
        }
        let q = a.orientation();
        assert!((q.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pellet_record_round_trips() {
        let reference = Pellet::fit_cylinder(0, 0.003, 0.012, 1100.0).unwrap();
        let mut pellet = reference.clone();
        pellet.id = 9;
        pellet.position = Vec3::new(0.1, 0.2, 0.3);
        pellet.orientation = Quat::from_axis_angle(&Vec3::x_axis(), 0.7);
        let moisture = Moisture {
            water_kg: 1e-4,
            damage: 0.2,
        };
        let record = PelletRecord::from_pellet(&pellet, &moisture);
        let (back, m) = record.to_pellet(&reference).unwrap();
        assert_eq!(back, pellet);
        assert_eq!(m, moisture);
        let mut bad = record.clone();
        bad.orientation_wijk = [0.0; 4];
        assert!(bad.to_pellet(&reference).is_err());
        bad = record;
        bad.water_kg = -1.0;
        assert!(bad.to_pellet(&reference).is_err());
    }
}
