//! Typed household request models and independent validation.
//!
//! Python validates first, but Rust must reject invalid input on its own. The
//! models use `deny_unknown_fields` so contract drift fails loudly. Validation
//! also derives the resource footprint (pellets, spheres, substeps, frames) and
//! rejects requests that exceed the documented caps before any allocation.
//!
//! References: `docs/wire-contract.md`.

use serde::{Deserialize, Serialize};

use crate::dem::{self, MAX_SPHERES_PER_PELLET, Pellet};

/// Wire schema version this module understands.
pub const SCHEMA_VERSION: u32 = 1;
/// Mode string handled by this module.
pub const MODE: &str = "household";
/// Fidelity label reported for every household result.
pub const FIDELITY: &str = "preliminary_household";
/// Finite-volume field spacing in metres.
pub const FIELD_SPACING_M: f64 = 0.005;

/// Maximum pellets alive at once across all boxes, including refills.
pub const MAX_PELLETS: usize = 10_000;
/// Maximum DEM substeps per request step before the step is rejected.
pub const MAX_SUBSTEPS_PER_STEP: u64 = 1_000;
/// Maximum request steps (`duration_s / dt_s`).
pub const MAX_STEPS: u64 = 10_000_000;
/// Maximum recorded frames.
pub const MAX_FRAMES: u64 = 100_000;
/// Maximum recorded sphere positions across all frames.
pub const MAX_FRAME_ELEMENTS: u64 = 20_000_000;
/// Maximum field cells across all boxes.
pub const MAX_FIELD_CELLS: usize = 2_000_000;
/// Maximum events per request.
pub const MAX_EVENTS: usize = 10_000;
/// Maximum wall budget in seconds.
pub const MAX_WALL_TIME_S: f64 = 3600.0;

/// Complete household request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub mode: String,
    pub seed: u64,
    pub duration_s: f64,
    pub dt_s: f64,
    pub record_interval_s: f64,
    pub max_wall_time_s: f64,
    pub boxes: Vec<BoxSpec>,
    pub materials: Materials,
    pub events: Vec<Event>,
    pub research: Option<serde_json::Value>,
}

/// One sifting box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxSpec {
    pub id: String,
    pub size_m: [f64; 3],
    pub origin_m: [f64; 3],
    pub slot_width_m: f64,
    pub slot_length_m: f64,
    pub slot_pitch_m: f64,
    pub drawer_depth_m: f64,
    pub pellet_count: u64,
    pub pellet_radius_m: f64,
    pub pellet_length_m: f64,
    pub pellet_density_kg_m3: f64,
}

/// Shared material parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Materials {
    pub friction: f64,
    pub restitution: f64,
    pub normal_stiffness_n_m: f64,
    pub water_capacity_ratio: f64,
    pub uptake_rate_s: f64,
    pub breakdown_rate_s: f64,
    pub evaporation_rate_s: f64,
}

/// Household event kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Urinate,
    Defecate,
    Dig,
    Cover,
    Stir,
    Scoop,
    EmptyDrawer,
    Refill,
    Clean,
}

impl EventKind {
    /// Whether the event drives a tool through the bed.
    #[must_use]
    pub fn is_motion(self) -> bool {
        matches!(self, Self::Dig | Self::Cover | Self::Stir)
    }

    /// Whether the event adds deposit material over its duration.
    #[must_use]
    pub fn is_deposit(self) -> bool {
        matches!(self, Self::Urinate | Self::Defecate)
    }
}

/// One scheduled event, positions box-local.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub time_s: f64,
    pub kind: EventKind,
    pub box_id: String,
    pub position_m: [f64; 3],
    pub direction: [f64; 3],
    pub duration_s: f64,
    pub amount_kg: f64,
    pub water_fraction: f64,
    pub radius_m: f64,
}

/// Derived sizes established during validation.
#[derive(Clone, Debug, PartialEq)]
pub struct Footprint {
    /// Request steps needed to reach `duration_s`.
    pub steps: u64,
    /// DEM substeps per request step.
    pub substeps: u64,
    /// Substep length in seconds.
    pub substep_dt_s: f64,
    /// Frames that will be recorded.
    pub frames: u64,
    /// Field cells across all boxes.
    pub cells: usize,
    /// Upper bound on pellets alive at once.
    pub max_pellets: usize,
}

/// Check that a value is finite.
fn finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("`{name}` must be finite, got '{value}'"))
    }
}

/// Check that a value is finite and strictly positive.
fn positive(name: &str, value: f64) -> Result<(), String> {
    finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(format!("`{name}` must be positive, got '{value}'"))
    }
}

/// Check that a value is finite and non-negative.
fn non_negative(name: &str, value: f64) -> Result<(), String> {
    finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(format!("`{name}` must be non-negative, got '{value}'"))
    }
}

/// Check that a value lies inside a closed interval.
fn within(name: &str, value: f64, low: f64, high: f64) -> Result<(), String> {
    finite(name, value)?;
    if (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "`{name}` must be in [{low}, {high}], got '{value}'"
        ))
    }
}

impl Materials {
    /// Validate ranges from the wire contract.
    ///
    /// # Errors
    ///
    /// Returns the first violated constraint.
    pub fn validate(&self) -> Result<(), String> {
        non_negative("materials.friction", self.friction)?;
        within("materials.restitution", self.restitution, 0.0, 1.0)?;
        positive("materials.normal_stiffness_n_m", self.normal_stiffness_n_m)?;
        positive("materials.water_capacity_ratio", self.water_capacity_ratio)?;
        non_negative("materials.uptake_rate_s", self.uptake_rate_s)?;
        non_negative("materials.breakdown_rate_s", self.breakdown_rate_s)?;
        non_negative("materials.evaporation_rate_s", self.evaporation_rate_s)
    }

    /// Contact parameters for the DEM world.
    #[must_use]
    pub fn contact_params(&self) -> dem::ContactParams {
        dem::ContactParams {
            normal_stiffness_n_m: self.normal_stiffness_n_m,
            friction: self.friction,
            restitution: self.restitution,
            rolling_friction: super::ROLLING_FRICTION,
        }
    }
}

impl BoxSpec {
    /// Validate geometry and pellet fit.
    ///
    /// # Errors
    ///
    /// Returns the first violated constraint, prefixed with the box id.
    pub fn validate(&self) -> Result<(), String> {
        let tag = format!("boxes['{}']", self.id);
        if self.id.is_empty() {
            return Err("box `id` must not be empty".to_string());
        }
        for (axis, value) in ["x", "y", "z"].iter().zip(self.size_m) {
            positive(&format!("{tag}.size_m.{axis}"), value)?;
        }
        for (axis, value) in ["x", "y", "z"].iter().zip(self.origin_m) {
            finite(&format!("{tag}.origin_m.{axis}"), value)?;
        }
        positive(&format!("{tag}.slot_width_m"), self.slot_width_m)?;
        positive(&format!("{tag}.slot_length_m"), self.slot_length_m)?;
        positive(&format!("{tag}.slot_pitch_m"), self.slot_pitch_m)?;
        positive(&format!("{tag}.drawer_depth_m"), self.drawer_depth_m)?;
        positive(&format!("{tag}.pellet_radius_m"), self.pellet_radius_m)?;
        positive(&format!("{tag}.pellet_length_m"), self.pellet_length_m)?;
        positive(
            &format!("{tag}.pellet_density_kg_m3"),
            self.pellet_density_kg_m3,
        )?;
        if self.slot_width_m >= self.slot_pitch_m {
            return Err(format!(
                "{tag}: `slot_width_m` must be smaller than `slot_pitch_m`"
            ));
        }
        if self.slot_pitch_m > self.size_m[1] {
            return Err(format!("{tag}: `slot_pitch_m` exceeds the box `size_m.y`"));
        }
        if self.slot_length_m + 2.0 * self.slot_width_m > self.size_m[0] {
            return Err(format!(
                "{tag}: one slot plus its bars does not fit `size_m.x`"
            ));
        }
        let bounding = self.pellet_length_m.max(2.0 * self.pellet_radius_m);
        if bounding >= self.size_m[0].min(self.size_m[1]) || bounding >= self.size_m[2] {
            return Err(format!("{tag}: a pellet does not fit inside the box"));
        }
        if self.pellet_length_m / self.pellet_radius_m > dem::count_to_f64(MAX_SPHERES_PER_PELLET) {
            return Err(format!(
                "{tag}: pellet aspect ratio exceeds {MAX_SPHERES_PER_PELLET} spheres"
            ));
        }
        let per_layer = self.lattice_per_layer();
        if per_layer == 0 {
            return Err(format!("{tag}: no pellet lattice site fits the footprint"));
        }
        let layers = self.pellet_count.div_ceil(per_layer);
        if dem::count_to_f64(usize::try_from(layers).unwrap_or(usize::MAX)) * bounding
            > self.size_m[2]
        {
            return Err(format!(
                "{tag}: `pellet_count` {} does not fit the box height as a loose lattice",
                self.pellet_count
            ));
        }
        Ok(())
    }

    /// Reference pellet fitted to this box's pellet dimensions.
    ///
    /// # Errors
    ///
    /// Propagates the multisphere fit error.
    pub fn reference_pellet(&self) -> Result<Pellet, String> {
        Pellet::fit_cylinder(
            0,
            self.pellet_radius_m,
            self.pellet_length_m,
            self.pellet_density_kg_m3,
        )
    }

    /// Loose lattice sites per layer used for initial placement and refills.
    #[must_use]
    pub fn lattice_per_layer(&self) -> u64 {
        let pitch = self.lattice_pitch();
        let nx = (self.size_m[0] / pitch).floor();
        let ny = (self.size_m[1] / pitch).floor();
        // Floor values are non-negative and bounded by the footprint check.
        let sites = nx * ny;
        if (0.0..1e12).contains(&sites) {
            dem::floor_to_i64(sites).unsigned_abs()
        } else {
            0
        }
    }

    /// Lattice pitch large enough for any pellet orientation.
    #[must_use]
    pub fn lattice_pitch(&self) -> f64 {
        self.pellet_length_m.max(2.0 * self.pellet_radius_m) * 1.05
    }

    /// Geometry in local coordinates.
    #[must_use]
    pub fn geometry(&self) -> dem::BoxGeometry {
        dem::BoxGeometry {
            size_m: dem::Vec3::new(self.size_m[0], self.size_m[1], self.size_m[2]),
            slot_width_m: self.slot_width_m,
            slot_length_m: self.slot_length_m,
            slot_pitch_m: self.slot_pitch_m,
            drawer_depth_m: self.drawer_depth_m,
        }
    }

    /// Field cell count for the bed volume at [`FIELD_SPACING_M`].
    #[must_use]
    pub fn field_dims(&self) -> [usize; 3] {
        let dim = |extent: f64| {
            let n = (extent / FIELD_SPACING_M).ceil().max(1.0);
            usize::try_from(dem::floor_to_i64(n)).unwrap_or(usize::MAX)
        };
        [
            dim(self.size_m[0]),
            dim(self.size_m[1]),
            dim(self.size_m[2]),
        ]
    }
}

impl Event {
    /// Validate one event against the request's boxes.
    ///
    /// # Errors
    ///
    /// Returns the first violated constraint, prefixed with the event index.
    pub fn validate(&self, index: usize, duration_s: f64, boxes: &[BoxSpec]) -> Result<(), String> {
        let tag = format!("events[{index}]");
        within(&format!("{tag}.time_s"), self.time_s, 0.0, duration_s)?;
        non_negative(&format!("{tag}.duration_s"), self.duration_s)?;
        non_negative(&format!("{tag}.amount_kg"), self.amount_kg)?;
        within(
            &format!("{tag}.water_fraction"),
            self.water_fraction,
            0.0,
            1.0,
        )?;
        positive(&format!("{tag}.radius_m"), self.radius_m)?;
        for (axis, value) in ["x", "y", "z"].iter().zip(self.position_m) {
            finite(&format!("{tag}.position_m.{axis}"), value)?;
        }
        let mut norm = 0.0;
        for (axis, value) in ["x", "y", "z"].iter().zip(self.direction) {
            finite(&format!("{tag}.direction.{axis}"), value)?;
            norm += value * value;
        }
        if self.kind.is_motion() && norm <= 0.0 {
            return Err(format!("{tag}: motion events need a nonzero `direction`"));
        }
        let Some(spec) = boxes.iter().find(|b| b.id == self.box_id) else {
            return Err(format!("{tag}: unknown `box_id` '{}'", self.box_id));
        };
        for (axis, (value, size)) in ["x", "y", "z"]
            .iter()
            .zip(self.position_m.iter().zip(spec.size_m))
        {
            if *value < -spec.drawer_depth_m || *value > 2.0 * size {
                return Err(format!(
                    "{tag}.position_m.{axis} '{value}' lies far outside box '{}'",
                    spec.id
                ));
            }
        }
        Ok(())
    }
}

impl Request {
    /// Validate the full request and derive its resource footprint.
    ///
    /// # Errors
    ///
    /// Returns the first violated constraint or exceeded resource cap.
    pub fn validate(&self) -> Result<Footprint, String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "unsupported `schema_version` {}, expected {SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        if self.mode != MODE {
            return Err(format!("`mode` must be '{MODE}', got '{}'", self.mode));
        }
        if self.research.is_some() {
            return Err("household mode requires `research` to be null".to_string());
        }
        positive("duration_s", self.duration_s)?;
        positive("dt_s", self.dt_s)?;
        positive("record_interval_s", self.record_interval_s)?;
        within(
            "max_wall_time_s",
            self.max_wall_time_s,
            f64::MIN_POSITIVE,
            MAX_WALL_TIME_S,
        )?;
        if self.dt_s > self.duration_s {
            return Err("`dt_s` must not exceed `duration_s`".to_string());
        }
        if self.boxes.is_empty() {
            return Err("household mode requires at least one box".to_string());
        }
        self.materials.validate()?;
        let mut ids = std::collections::BTreeSet::new();
        for spec in &self.boxes {
            spec.validate()?;
            if !ids.insert(spec.id.as_str()) {
                return Err(format!("duplicate box id '{}'", spec.id));
            }
        }
        if self.events.len() > MAX_EVENTS {
            return Err(format!(
                "too many events: {} > {MAX_EVENTS}",
                self.events.len()
            ));
        }
        let mut last = 0.0_f64;
        for (index, event) in self.events.iter().enumerate() {
            event.validate(index, self.duration_s, &self.boxes)?;
            if event.time_s < last {
                return Err(format!(
                    "events[{index}] is out of order (time {})",
                    event.time_s
                ));
            }
            last = event.time_s;
        }
        self.footprint()
    }

    /// Derive step counts, substeps, frames and pellet caps.
    fn footprint(&self) -> Result<Footprint, String> {
        let steps_f = (self.duration_s / self.dt_s).ceil();
        if steps_f > dem::count_to_f64(usize::try_from(MAX_STEPS).unwrap_or(usize::MAX)) {
            return Err(format!("`duration_s / dt_s` exceeds {MAX_STEPS} steps"));
        }
        let steps = dem::floor_to_i64(steps_f).unsigned_abs().max(1);
        let frames_f = (self.duration_s / self.record_interval_s).floor() + 1.0;
        if frames_f > dem::count_to_f64(usize::try_from(MAX_FRAMES).unwrap_or(usize::MAX)) {
            return Err(format!("recorded frames exceed {MAX_FRAMES}"));
        }
        let frames = dem::floor_to_i64(frames_f).unsigned_abs();
        let mut cells = 0usize;
        let mut max_pellets = 0usize;
        let mut spheres = 0u64;
        let mut min_mass = f64::INFINITY;
        for spec in &self.boxes {
            let dims = spec.field_dims();
            cells = cells.saturating_add(dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]));
            let reference = spec.reference_pellet()?;
            min_mass = min_mass.min(reference.mass);
            let refills: f64 = self
                .events
                .iter()
                .filter(|e| e.kind == EventKind::Refill && e.box_id == spec.id)
                .map(|e| (e.amount_kg / reference.mass).floor())
                .sum();
            let total = dem::count_to_f64(usize::try_from(spec.pellet_count).unwrap_or(usize::MAX))
                + refills;
            if total > dem::count_to_f64(MAX_PELLETS) {
                return Err(format!(
                    "box '{}' may hold {total} pellets, above the cap of {MAX_PELLETS}",
                    spec.id
                ));
            }
            let total = usize::try_from(dem::floor_to_i64(total)).unwrap_or(usize::MAX);
            max_pellets = max_pellets.saturating_add(total);
            spheres = spheres.saturating_add(
                u64::try_from(total.saturating_mul(reference.offsets.len())).unwrap_or(u64::MAX),
            );
        }
        if cells > MAX_FIELD_CELLS {
            return Err(format!(
                "field cells {cells} exceed the cap of {MAX_FIELD_CELLS}"
            ));
        }
        if max_pellets > MAX_PELLETS {
            return Err(format!(
                "total pellets {max_pellets} exceed the cap of {MAX_PELLETS}"
            ));
        }
        if frames.saturating_mul(spheres) > MAX_FRAME_ELEMENTS {
            return Err(format!(
                "frames x spheres ({frames} x {spheres}) exceed {MAX_FRAME_ELEMENTS}"
            ));
        }
        let stiffness = self
            .materials
            .normal_stiffness_n_m
            .max(super::PAW_STIFFNESS_N_M);
        let stable = dem::STABLE_DT_FACTOR * (min_mass / stiffness).sqrt();
        let substeps_f = (self.dt_s / stable).ceil().max(1.0);
        if substeps_f
            > dem::count_to_f64(usize::try_from(MAX_SUBSTEPS_PER_STEP).unwrap_or(usize::MAX))
        {
            return Err(format!(
                "`dt_s` {} needs {substeps_f} DEM substeps (stable {stable:.3e} s), above {MAX_SUBSTEPS_PER_STEP}",
                self.dt_s
            ));
        }
        let substeps = dem::floor_to_i64(substeps_f).unsigned_abs();
        Ok(Footprint {
            steps,
            substeps,
            substep_dt_s: self.dt_s
                / dem::count_to_f64(usize::try_from(substeps).unwrap_or(usize::MAX)),
            frames,
            cells,
            max_pellets,
        })
    }
}
