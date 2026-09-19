//! Preliminary reduced household model: intact multisphere pellets, compliant
//! force-limited paw/stir proxies, a coarse conservative field for fines, waste
//! and water, and stateful maintenance events.
//!
//! This stage is explicitly `preliminary_household` fidelity. Paste rheology,
//! adhesion, calibrated slot flux laws and research response tables are NOT
//! implemented; the constants below are documented placeholders, not measured
//! values. Every species transfer goes through explicit ledgers and the run
//! fails, rather than completing, when the species residual exceeds tolerance.
//!
//! References: `docs/household.md`, `docs/wire-contract.md`, `docs/plan.md`.

mod field;
pub(crate) mod request;
mod state;

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;

use crate::dem::{self, Pellet, Tool, Vec3, World};
use field::Field;
use request::{BoxSpec, Event, EventKind, Footprint, Request};
use state::{
    BoxRecord, Checkpoint, Compartment, EventProgress, Moisture, PelletRecord, Rng, State,
    ToolRecord,
};

/// Rolling resistance coefficient for pellet contacts (placeholder).
pub const ROLLING_FRICTION: f64 = 0.05;
/// Spring stiffness pulling a paw/stir proxy toward its target path, N/m.
pub const PAW_STIFFNESS_N_M: f64 = 200.0;
/// Upper bound on the drive force a paw/stir proxy can apply, N.
pub const PAW_FORCE_LIMIT_N: f64 = 2.0;
/// Mass of the paw/stir proxy, kg.
pub const PAW_MASS_KG: f64 = 0.02;
/// Speed of the target path for motion events, m/s.
pub const STROKE_SPEED_M_S: f64 = 0.15;
/// Pellet speed above which breakup counts as disturbance-driven, m/s.
pub const DISTURBANCE_SPEED_M_S: f64 = 0.01;
/// Multiplier on the breakup rate while a pellet is disturbed.
pub const DISTURBANCE_BREAKUP_FACTOR: f64 = 2.0;
/// Relative species residual tolerance.
pub const RESIDUAL_RELATIVE_TOLERANCE: f64 = 1e-6;
/// Absolute species residual tolerance near zero, kg.
pub const RESIDUAL_ABSOLUTE_TOLERANCE_KG: f64 = 1e-12;

/// Run or resume a household simulation.
///
/// # Errors
///
/// Returns a message when the request or checkpoint is invalid, when a
/// resource cap is exceeded, or when the simulation violates conservation.
pub fn run(
    request: &serde_json::Value,
    resume: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let started = Instant::now();
    let typed: Request =
        serde_json::from_value(request.clone()).map_err(|e| format!("invalid request: {e}"))?;
    let footprint = typed.validate()?;
    let mut sim = match resume {
        None => Simulation::new(typed, footprint, request.clone())?,
        Some(checkpoint) => Simulation::resume(typed, footprint, request.clone(), checkpoint)?,
    };
    let status = sim.advance(started)?;
    sim.output(status)
}

/// Per-box runtime state.
struct BoxSim {
    spec: BoxSpec,
    reference: Pellet,
    world: World,
    moisture: BTreeMap<u64, Moisture>,
    field: Field,
    drawer: Compartment,
    tool: Option<ActiveTool>,
}

/// Runtime tool bound to its motion event.
struct ActiveTool {
    event_index: usize,
}

/// Whole-run state.
struct Simulation {
    request: Request,
    request_value: serde_json::Value,
    footprint: Footprint,
    boxes: Vec<BoxSim>,
    events: Vec<EventProgress>,
    step_index: u64,
    next_record_index: u64,
    rng: Rng,
    next_pellet_id: u64,
    supplied: Compartment,
    floor: Compartment,
    removed: Compartment,
    evaporated: Compartment,
    tool_work_j: f64,
    scoop_clean_wood_kg: f64,
    breakups: u64,
    max_overlap_m: f64,
    frames: Vec<Frame>,
    metrics: Vec<Metric>,
}

#[derive(Serialize)]
struct Frame {
    time_s: f64,
    positions_m: Vec<[f64; 3]>,
    radii_m: Vec<f64>,
    materials: Vec<&'static str>,
    box_ids: Vec<String>,
}

#[derive(Serialize)]
struct Metric {
    time_s: f64,
    compartment: &'static str,
    wood_kg: f64,
    waste_kg: f64,
    water_kg: f64,
}

#[derive(Serialize)]
struct Output {
    schema_version: u32,
    mode: &'static str,
    status: &'static str,
    fidelity: &'static str,
    time_s: f64,
    frames: Vec<Frame>,
    metrics: Vec<Metric>,
    observables: BTreeMap<&'static str, f64>,
    diagnostics: Vec<String>,
    checkpoint: Checkpoint,
}

/// Terminal status of `advance`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Completed,
    BudgetExhausted,
}

impl BoxSim {
    /// Build an empty box from its specification.
    fn new(spec: &BoxSpec, materials: &request::Materials) -> Result<Self, String> {
        let reference = spec.reference_pellet()?;
        let geometry = spec.geometry();
        Ok(Self {
            spec: spec.clone(),
            reference,
            world: World::new(geometry, materials.contact_params()),
            moisture: BTreeMap::new(),
            field: Field::new(spec.field_dims(), &geometry),
            drawer: Compartment::default(),
            tool: None,
        })
    }

    /// Water capacity of one pellet, kg.
    fn capacity_kg(&self, ratio: f64) -> f64 {
        ratio * self.reference.mass
    }

    /// Update pellet mass, inertia and swollen radius from moisture.
    fn apply_swelling(&mut self) {
        let density = self.spec.pellet_density_kg_m3;
        for pellet in &mut self.world.pellets {
            let water = self.moisture.get(&pellet.id).map_or(0.0, |m| m.water_kg);
            let dry = self.reference.mass;
            let volume_ratio = 1.0 + water * density / (dry * field::WATER_DENSITY_KG_M3);
            let scale = volume_ratio.cbrt();
            pellet.mass = dry + water;
            pellet.radius = self.reference.radius * scale;
            pellet.inertia_body = self.reference.inertia_body * (pellet.mass / dry) * scale * scale;
        }
    }

    /// Refresh field occupancy from pellet sphere positions.
    fn update_occupancy(&mut self) {
        let field = &mut self.field;
        field.occupancy.iter_mut().for_each(|o| *o = 0.0);
        let volume_per_sphere = self.reference.mass
            / self.spec.pellet_density_kg_m3
            / dem::count_to_f64(self.reference.offsets.len());
        let cell_volume = field.spacing_m.powi(3);
        for pellet in &self.world.pellets {
            if pellet.position.z < 0.0 {
                continue;
            }
            for index in 0..pellet.offsets.len() {
                let c = pellet.sphere_centre(index);
                let [ix, iy, iz] = field.cell_of(c);
                let i = field.index(ix, iy, iz);
                field.occupancy[i] = (field.occupancy[i] + volume_per_sphere / cell_volume)
                    .min(field::MAX_OCCUPANCY);
            }
        }
    }

    /// Species held by pellets, split into bed (`z >= 0`) and drawer.
    fn pellet_compartments(&self) -> (Compartment, Compartment) {
        let mut bed = Compartment::default();
        let mut drawer = Compartment::default();
        for pellet in &self.world.pellets {
            let water = self.moisture.get(&pellet.id).map_or(0.0, |m| m.water_kg);
            let target = if pellet.position.z >= 0.0 {
                &mut bed
            } else {
                &mut drawer
            };
            target.wood += self.reference.mass;
            target.water += water;
        }
        (bed, drawer)
    }

    /// Species removed with a set of pellets.
    fn pellet_mass(&self, pellets: &[Pellet]) -> Compartment {
        let mut out = Compartment::default();
        for pellet in pellets {
            out.wood += self.reference.mass;
            out.water += self.moisture.get(&pellet.id).map_or(0.0, |m| m.water_kg);
        }
        out
    }

    /// Remove pellets matching a predicate and forget their moisture.
    fn drain(&mut self, predicate: impl Fn(&Pellet) -> bool) -> Compartment {
        let removed = self.world.drain_pellets(predicate);
        let mass = self.pellet_mass(&removed);
        for pellet in &removed {
            self.moisture.remove(&pellet.id);
        }
        mass
    }

    /// Place `count` pellets on a loose lattice above `base_z` within
    /// `radius` of `centre`; the nearest site is used when none is inside.
    fn spawn(
        &mut self,
        count: u64,
        centre: Vec3,
        radius: f64,
        base_z: f64,
        rng: &mut Rng,
        next_id: &mut u64,
    ) {
        let pitch = self.spec.lattice_pitch();
        let nx = self.spec.size_m[0] / pitch;
        let ny = self.spec.size_m[1] / pitch;
        let mut sites = Vec::new();
        let mut y = 0.5 * pitch;
        while y < ny * pitch {
            let mut x = 0.5 * pitch;
            while x < nx * pitch {
                if (x - centre.x).hypot(y - centre.y) <= radius {
                    sites.push((x, y));
                }
                x += pitch;
            }
            y += pitch;
        }
        if sites.is_empty() {
            let x = centre.x.clamp(0.5 * pitch, (nx - 0.5).max(0.5) * pitch);
            let y = centre.y.clamp(0.5 * pitch, (ny - 0.5).max(0.5) * pitch);
            sites.push((x, y));
        }
        let mut placed = 0u64;
        let mut layer = 0.0;
        while placed < count {
            for &(x, y) in &sites {
                if placed >= count {
                    break;
                }
                let mut pellet = self.reference.clone();
                let id = *next_id;
                pellet.id = id;
                *next_id += 1;
                let jitter = 0.1 * pitch;
                pellet.position = Vec3::new(
                    x + (rng.next_f64() - 0.5) * jitter,
                    y + (rng.next_f64() - 0.5) * jitter,
                    base_z + (layer + 0.5) * pitch,
                );
                pellet.orientation = rng.orientation();
                self.world.pellets.push(pellet);
                self.moisture.insert(id, Moisture::default());
                placed += 1;
            }
            layer += 1.0;
        }
    }

    /// Highest pellet sphere top in the bed, or zero.
    fn bed_top(&self) -> f64 {
        self.world
            .pellets
            .iter()
            .map(|p| p.position.z + p.bounding_radius())
            .fold(0.0_f64, f64::max)
    }

    /// Snapshot for the checkpoint.
    fn record(&self) -> BoxRecord {
        BoxRecord {
            pellets: self
                .world
                .pellets
                .iter()
                .map(|p| {
                    let m = self.moisture.get(&p.id).copied().unwrap_or_default();
                    PelletRecord::from_pellet(p, &m)
                })
                .collect(),
            history: self.world.history(),
            field: self.field.clone(),
            drawer: self.drawer,
            tool: self.tool.as_ref().and_then(|t| {
                self.world.tools.first().map(|tool| ToolRecord {
                    event_index: t.event_index,
                    position_m: tool.position.into(),
                    velocity_m_s: tool.velocity.into(),
                })
            }),
        }
    }

    /// Rebuild from a checkpoint record.
    fn restore(&mut self, record: BoxRecord, events: &[Event]) -> Result<(), String> {
        let tag = format!("box '{}'", self.spec.id);
        let geometry = self.spec.geometry();
        let mut ids = std::collections::BTreeSet::new();
        for pr in &record.pellets {
            let (pellet, moisture) = pr.to_pellet(&self.reference)?;
            let p = pellet.position;
            if p.x < -0.1
                || p.x > geometry.size_m.x + 0.1
                || p.y < -0.1
                || p.y > geometry.size_m.y + 0.1
                || p.z < -geometry.drawer_depth_m - 0.1
                || p.z > geometry.size_m.z + 1.0
            {
                return Err(format!("{tag}: pellet {} lies outside the box", pellet.id));
            }
            if !ids.insert(pellet.id) {
                return Err(format!("{tag}: duplicate pellet id {}", pellet.id));
            }
            self.moisture.insert(pellet.id, moisture);
            self.world.pellets.push(pellet);
        }
        self.world.set_history(&record.history);
        let mut field = record.field;
        field
            .check(self.spec.field_dims())
            .map_err(|e| format!("{tag}: {e}"))?;
        if field.spacing_m.to_bits() != self.field.spacing_m.to_bits()
            || field.slot_open != self.field.slot_open
        {
            return Err(format!(
                "{tag}: checkpoint field geometry differs from request"
            ));
        }
        self.field = field;
        if !record.drawer.is_sane() {
            return Err(format!("{tag}: drawer ledger is not finite"));
        }
        self.drawer = record.drawer;
        if let Some(tool) = record.tool {
            let Some(event) = events.get(tool.event_index) else {
                return Err(format!(
                    "{tag}: tool references unknown event {}",
                    tool.event_index
                ));
            };
            if !event.kind.is_motion() {
                return Err(format!("{tag}: tool bound to a non-motion event"));
            }
            if tool
                .position_m
                .iter()
                .chain(&tool.velocity_m_s)
                .any(|v| !v.is_finite())
            {
                return Err(format!("{tag}: tool state is not finite"));
            }
            self.world.tools.push(Tool {
                id: u64::try_from(tool.event_index).unwrap_or(u64::MAX),
                position: Vec3::from(tool.position_m),
                velocity: Vec3::from(tool.velocity_m_s),
                radius: event.radius_m,
                mass: PAW_MASS_KG,
                drive_force: Vec3::zeros(),
            });
            self.tool = Some(ActiveTool {
                event_index: tool.event_index,
            });
        }
        self.apply_swelling();
        Ok(())
    }
}

impl Simulation {
    /// Fresh simulation with seeded initial packing.
    fn new(
        request: Request,
        footprint: Footprint,
        request_value: serde_json::Value,
    ) -> Result<Self, String> {
        let mut rng = Rng(request.seed);
        let mut next_pellet_id = 1u64;
        let mut boxes = Vec::with_capacity(request.boxes.len());
        let mut supplied = Compartment::default();
        for spec in &request.boxes {
            let mut b = BoxSim::new(spec, &request.materials)?;
            let centre = Vec3::new(0.5 * spec.size_m[0], 0.5 * spec.size_m[1], 0.0);
            let radius = spec.size_m[0].hypot(spec.size_m[1]);
            b.spawn(
                spec.pellet_count,
                centre,
                radius,
                0.0,
                &mut rng,
                &mut next_pellet_id,
            );
            supplied.wood += b.reference.mass * dem::count_to_f64(b.world.pellets.len());
            boxes.push(b);
        }
        let events = vec![EventProgress::default(); request.events.len()];
        let mut sim = Self {
            request,
            request_value,
            footprint,
            boxes,
            events,
            step_index: 0,
            next_record_index: 0,
            rng,
            next_pellet_id,
            supplied,
            floor: Compartment::default(),
            removed: Compartment::default(),
            evaporated: Compartment::default(),
            tool_work_j: 0.0,
            scoop_clean_wood_kg: 0.0,
            breakups: 0,
            max_overlap_m: 0.0,
            frames: Vec::new(),
            metrics: Vec::new(),
        };
        for b in &mut sim.boxes {
            b.update_occupancy();
        }
        sim.record_if_due();
        Ok(sim)
    }

    /// Resume from a checkpoint after validating compatibility.
    fn resume(
        request: Request,
        footprint: Footprint,
        request_value: serde_json::Value,
        checkpoint: &serde_json::Value,
    ) -> Result<Self, String> {
        let cp: Checkpoint = serde_json::from_value(checkpoint.clone())
            .map_err(|e| format!("invalid checkpoint: {e}"))?;
        if cp.schema_version != request::SCHEMA_VERSION {
            return Err(format!(
                "unsupported checkpoint schema_version {}",
                cp.schema_version
            ));
        }
        if cp.mode != request::MODE {
            return Err(format!(
                "checkpoint mode '{}' is not '{}'",
                cp.mode,
                request::MODE
            ));
        }
        let mut original: Request = serde_json::from_value(cp.request)
            .map_err(|e| format!("checkpoint request is invalid: {e}"))?;
        original.max_wall_time_s = request.max_wall_time_s;
        if original != request {
            return Err("checkpoint request differs from the resume request".to_string());
        }
        let state = cp.state;
        if state.boxes.len() != request.boxes.len() || state.events.len() != request.events.len() {
            return Err("checkpoint box/event counts do not match the request".to_string());
        }
        let expected_time = step_time(&request, &footprint, state.step_index);
        if !cp.time_s.is_finite() || (cp.time_s - expected_time).abs() > 1e-9 {
            return Err(format!(
                "checkpoint time_s '{}' is inconsistent with step {}",
                cp.time_s, state.step_index
            ));
        }
        for ledger in [state.supplied, state.floor, state.removed, state.evaporated] {
            if !ledger.is_sane() {
                return Err("checkpoint ledger holds a non-finite or negative mass".to_string());
            }
        }
        for (i, e) in state.events.iter().enumerate() {
            if !e.applied_kg.is_finite() || e.applied_kg < 0.0 || (e.finished && !e.started) {
                return Err(format!("checkpoint event progress {i} is inconsistent"));
            }
        }
        if !(state.tool_work_j.is_finite()
            && state.scoop_clean_wood_kg.is_finite()
            && state.max_overlap_m.is_finite())
        {
            return Err("checkpoint diagnostics are not finite".to_string());
        }
        let mut boxes = Vec::with_capacity(request.boxes.len());
        for (spec, record) in request.boxes.iter().zip(state.boxes) {
            let mut b = BoxSim::new(spec, &request.materials)?;
            b.restore(record, &request.events)?;
            boxes.push(b);
        }
        let mut sim = Self {
            request,
            request_value,
            footprint,
            boxes,
            events: state.events,
            step_index: state.step_index,
            next_record_index: state.next_record_index,
            rng: state.rng,
            next_pellet_id: state.next_pellet_id,
            supplied: state.supplied,
            floor: state.floor,
            removed: state.removed,
            evaporated: state.evaporated,
            tool_work_j: state.tool_work_j,
            scoop_clean_wood_kg: state.scoop_clean_wood_kg,
            breakups: state.breakups,
            max_overlap_m: state.max_overlap_m,
            frames: Vec::new(),
            metrics: Vec::new(),
        };
        for b in &mut sim.boxes {
            b.update_occupancy();
        }
        sim.check_conservation()?;
        Ok(sim)
    }

    /// Current absolute simulation time.
    fn time(&self) -> f64 {
        step_time(&self.request, &self.footprint, self.step_index)
    }

    /// Step until completion or budget exhaustion.
    fn advance(&mut self, started: Instant) -> Result<Status, String> {
        while self.step_index < self.footprint.steps {
            if started.elapsed().as_secs_f64() >= self.request.max_wall_time_s {
                return Ok(Status::BudgetExhausted);
            }
            self.step()?;
        }
        Ok(Status::Completed)
    }

    /// One request step: events, mechanics, moisture, transport, recording.
    fn step(&mut self) -> Result<(), String> {
        let t0 = self.time();
        let t1 = step_time(&self.request, &self.footprint, self.step_index + 1);
        let dt = t1 - t0;
        let last = self.step_index + 1 == self.footprint.steps;
        self.apply_events(t0, t1, last);
        self.mechanics(t0, dt)?;
        self.moisture(dt);
        self.step_index += 1;
        self.finish_events(t1, last);
        self.check_conservation()?;
        self.record_if_due();
        Ok(())
    }

    /// Start due events and apply their per-step share.
    fn apply_events(&mut self, t0: f64, t1: f64, last: bool) {
        for index in 0..self.request.events.len() {
            let event = self.request.events[index].clone();
            let due = event.time_s < t1 || last;
            if self.events[index].finished || !due {
                continue;
            }
            let Some(bi) = self.request.boxes.iter().position(|b| b.id == event.box_id) else {
                continue;
            };
            if !self.events[index].started {
                self.events[index].started = true;
                self.start_event(index, bi, &event);
            }
            if event.kind.is_deposit() {
                let end = event.time_s + event.duration_s;
                let overlap = (t1.min(end) - t0.max(event.time_s)).max(0.0);
                let share = if event.duration_s > 0.0 && t1 < end {
                    event.amount_kg * overlap / event.duration_s
                } else {
                    event.amount_kg - self.events[index].applied_kg
                };
                let waste = share * (1.0 - event.water_fraction);
                let water = share * event.water_fraction;
                self.boxes[bi].field.deposit(
                    Vec3::from(event.position_m),
                    event.radius_m,
                    waste,
                    water,
                );
                self.events[index].applied_kg += share;
                self.supplied.waste += waste;
                self.supplied.water += water;
            }
        }
    }

    /// Apply the instantaneous part of an event.
    fn start_event(&mut self, index: usize, bi: usize, event: &Event) {
        let position = Vec3::from(event.position_m);
        match event.kind {
            EventKind::Urinate | EventKind::Defecate => {}
            EventKind::Dig | EventKind::Cover | EventKind::Stir => {
                let b = &mut self.boxes[bi];
                b.world.tools.clear();
                // Enter above the current bed rather than teleporting a tool into
                // existing pellets or through the steel floor. The compliant drive
                // then approaches the requested stroke path at bounded force.
                let entry = Vec3::new(
                    position.x,
                    position.y,
                    b.bed_top().max(position.z).max(0.0) + event.radius_m,
                );
                b.world.tools.push(Tool {
                    id: u64::try_from(index).unwrap_or(u64::MAX),
                    position: entry,
                    velocity: Vec3::zeros(),
                    radius: event.radius_m,
                    mass: PAW_MASS_KG,
                    drive_force: Vec3::zeros(),
                });
                b.tool = Some(ActiveTool { event_index: index });
            }
            EventKind::Scoop => {
                let b = &mut self.boxes[bi];
                let radius = event.radius_m;
                let removed =
                    b.drain(|p| p.position.z >= 0.0 && (p.position - position).norm() <= radius);
                let (fines, waste, water) = b.field.remove_within(position, radius);
                self.scoop_clean_wood_kg += removed.wood + fines;
                self.removed.add(removed);
                self.removed.add(Compartment {
                    wood: fines,
                    waste,
                    water,
                });
            }
            EventKind::EmptyDrawer => {
                let b = &mut self.boxes[bi];
                let pellets = b.drain(|p| p.position.z < 0.0);
                self.removed.add(pellets);
                self.removed.add(b.drawer.take());
            }
            EventKind::Refill => {
                let b = &mut self.boxes[bi];
                let count_f = (event.amount_kg / b.reference.mass).floor();
                let count = dem::floor_to_i64(count_f).unsigned_abs();
                let remainder = event.amount_kg - count_f * b.reference.mass;
                let base = b.bed_top().max(position.z);
                b.spawn(
                    count,
                    position,
                    event.radius_m,
                    base,
                    &mut self.rng,
                    &mut self.next_pellet_id,
                );
                b.field.add_fines(position, remainder);
                self.supplied.wood += event.amount_kg;
            }
            EventKind::Clean => {
                let b = &mut self.boxes[bi];
                let pellets = b.drain(|_| true);
                self.removed.add(pellets);
                let (fines, waste, water) = b.field.clear();
                self.removed.add(Compartment {
                    wood: fines,
                    waste,
                    water,
                });
                self.removed.add(b.drawer.take());
            }
        }
    }

    /// Mark events whose span ended and retire their tools.
    fn finish_events(&mut self, t1: f64, last: bool) {
        for index in 0..self.request.events.len() {
            let event = &self.request.events[index];
            let progress = &mut self.events[index];
            if !progress.started || progress.finished {
                continue;
            }
            let ended = t1 >= event.time_s + event.duration_s || last;
            if !ended {
                continue;
            }
            progress.finished = true;
            if !event.kind.is_motion() {
                continue;
            }
            if let Some(b) = self.boxes.iter_mut().find(|b| b.spec.id == event.box_id)
                && b.tool.as_ref().is_some_and(|t| t.event_index == index)
            {
                b.tool = None;
                b.world.tools.clear();
            }
        }
    }

    /// Advance DEM substeps with compliant tool drives; move escaped pellets
    /// to the floor ledger.
    fn mechanics(&mut self, t0: f64, dt: f64) -> Result<(), String> {
        let substeps = self.footprint.substeps;
        let sub = dt / dem::count_to_f64(usize::try_from(substeps).unwrap_or(usize::MAX));
        for b in &mut self.boxes {
            b.apply_swelling();
            let mut k = 0u64;
            while k < substeps {
                let tau = t0 + dem::count_to_f64(usize::try_from(k).unwrap_or(usize::MAX)) * sub;
                if let Some(active) = &b.tool {
                    let event = &self.request.events[active.event_index];
                    if let Some(tool) = b.world.tools.first_mut() {
                        tool.drive_force = tool_drive(tool, event, tau);
                    }
                }
                let report = b.world.step(sub);
                self.tool_work_j += report.tool_work_j;
                self.max_overlap_m = self.max_overlap_m.max(report.max_overlap_m);
                k += 1;
            }
            let size = b.world.geometry.size_m;
            let depth = b.world.geometry.drawer_depth_m;
            for p in &b.world.pellets {
                if !(p.position.iter().all(|v| v.is_finite())
                    && p.velocity.iter().all(|v| v.is_finite()))
                {
                    return Err(format!(
                        "box '{}': pellet {} became non-finite",
                        b.spec.id, p.id
                    ));
                }
            }
            let escaped = b.drain(|p| {
                p.position.x < 0.0
                    || p.position.x > size.x
                    || p.position.y < 0.0
                    || p.position.y > size.y
                    || p.position.z < -depth
            });
            self.floor.add(escaped);
        }
        Ok(())
    }

    /// Absorption, breakup, drainage, sifting and evaporation for one step.
    fn moisture(&mut self, dt: f64) {
        let m = &self.request.materials;
        let uptake = 1.0 - (-m.uptake_rate_s * dt).exp();
        let evaporate = 1.0 - (-m.evaporation_rate_s * dt).exp();
        for b in &mut self.boxes {
            b.update_occupancy();
            let capacity = b.capacity_kg(m.water_capacity_ratio);
            let mut broken = Vec::new();
            for pellet in &b.world.pellets {
                let Some(moist) = b.moisture.get_mut(&pellet.id) else {
                    continue;
                };
                let in_bed = pellet.position.z >= 0.0;
                if in_bed {
                    let cell = b.field.cell_of(pellet.position);
                    let room = (capacity - moist.water_kg).max(0.0) * uptake;
                    moist.water_kg += b.field.take_water(cell, room);
                }
                let saturation = (moist.water_kg / capacity).min(1.0);
                let disturbed = pellet.velocity.norm() > DISTURBANCE_SPEED_M_S;
                let factor = if disturbed {
                    DISTURBANCE_BREAKUP_FACTOR
                } else {
                    1.0
                };
                moist.damage += m.breakdown_rate_s * saturation * factor * dt;
                let evaporated = moist.water_kg * evaporate;
                moist.water_kg -= evaporated;
                self.evaporated.water += evaporated;
                if moist.damage >= 1.0 {
                    broken.push((pellet.id, in_bed, b.field.cell_of(pellet.position)));
                }
            }
            for (id, in_bed, cell) in broken {
                let removed = b.drain(|p| p.id == id);
                if in_bed {
                    b.field.add_to_cell(cell, removed.wood, removed.water);
                } else {
                    b.drawer.add(removed);
                }
                self.breakups += 1;
            }
            let outflow = b.field.transport(dt);
            b.drawer.wood += outflow.wood_kg;
            b.drawer.water += outflow.water_kg;
            self.evaporated.water += b.field.evaporate(m.evaporation_rate_s, dt);
            let drawer_evaporated = b.drawer.water * evaporate;
            b.drawer.water -= drawer_evaporated;
            self.evaporated.water += drawer_evaporated;
        }
        let floor_evaporated = self.floor.water * evaporate;
        self.floor.water -= floor_evaporated;
        self.evaporated.water += floor_evaporated;
    }

    /// Compartment totals: bed, drawer, floor, removed, evaporated.
    fn compartments(&self) -> [(&'static str, Compartment); 5] {
        let mut bed = Compartment::default();
        let mut drawer = Compartment::default();
        for b in &self.boxes {
            let (pb, pd) = b.pellet_compartments();
            bed.add(pb);
            drawer.add(pd);
            let (fines, waste, water) = b.field.totals();
            bed.add(Compartment {
                wood: fines,
                waste,
                water,
            });
            drawer.add(b.drawer);
        }
        [
            ("bed", bed),
            ("drawer", drawer),
            ("floor", self.floor),
            ("removed", self.removed),
            ("evaporated", self.evaporated),
        ]
    }

    /// Species residuals (wood, waste, water) against supplied totals.
    fn residuals(&self) -> [f64; 3] {
        let mut total = Compartment::default();
        for (_, c) in self.compartments() {
            total.add(c);
        }
        [
            total.wood - self.supplied.wood,
            total.waste - self.supplied.waste,
            total.water - self.supplied.water,
        ]
    }

    /// Fail when any species residual exceeds tolerance.
    fn check_conservation(&self) -> Result<(), String> {
        let supplied = [self.supplied.wood, self.supplied.waste, self.supplied.water];
        for ((name, residual), reference) in ["wood", "waste", "water"]
            .iter()
            .zip(self.residuals())
            .zip(supplied)
        {
            let limit = RESIDUAL_ABSOLUTE_TOLERANCE_KG.max(RESIDUAL_RELATIVE_TOLERANCE * reference);
            if !residual.is_finite() || residual.abs() > limit {
                return Err(format!(
                    "conservation violated for `{name}` at t={}: residual {residual:e} kg exceeds {limit:e}",
                    self.time()
                ));
            }
        }
        Ok(())
    }

    /// Record frames and metrics for every record index reached.
    fn record_if_due(&mut self) {
        let time = self.time();
        while dem::count_to_f64(usize::try_from(self.next_record_index).unwrap_or(usize::MAX))
            * self.request.record_interval_s
            <= time + 1e-12
            && self.next_record_index < self.footprint.frames
        {
            self.frames.push(self.frame(time));
            for (name, c) in self.compartments() {
                self.metrics.push(Metric {
                    time_s: time,
                    compartment: name,
                    wood_kg: c.wood,
                    waste_kg: c.waste,
                    water_kg: c.water,
                });
            }
            self.next_record_index += 1;
        }
    }

    /// Visual snapshot: pellet spheres, waste cells and active tools.
    fn frame(&self, time: f64) -> Frame {
        let mut frame = Frame {
            time_s: time,
            positions_m: Vec::new(),
            radii_m: Vec::new(),
            materials: Vec::new(),
            box_ids: Vec::new(),
        };
        for b in &self.boxes {
            let origin = Vec3::from(b.spec.origin_m);
            for p in &b.world.pellets {
                for index in 0..p.offsets.len() {
                    frame
                        .positions_m
                        .push((origin + p.sphere_centre(index)).into());
                    frame.radii_m.push(p.radius);
                    frame.materials.push("wood");
                    frame.box_ids.push(b.spec.id.clone());
                }
            }
            for iz in 0..b.field.dims[2] {
                for iy in 0..b.field.dims[1] {
                    for ix in 0..b.field.dims[0] {
                        if b.field.waste_kg[b.field.index(ix, iy, iz)] > 0.0 {
                            frame
                                .positions_m
                                .push((origin + b.field.centre(ix, iy, iz)).into());
                            frame.radii_m.push(0.5 * b.field.spacing_m);
                            frame.materials.push("waste");
                            frame.box_ids.push(b.spec.id.clone());
                        }
                    }
                }
            }
            for tool in &b.world.tools {
                frame.positions_m.push((origin + tool.position).into());
                frame.radii_m.push(tool.radius);
                frame.materials.push("paw");
                frame.box_ids.push(b.spec.id.clone());
            }
        }
        frame
    }

    /// Serialize the full checkpoint state.
    fn state(&self) -> State {
        State {
            step_index: self.step_index,
            next_record_index: self.next_record_index,
            rng: self.rng,
            next_pellet_id: self.next_pellet_id,
            boxes: self.boxes.iter().map(BoxSim::record).collect(),
            events: self.events.clone(),
            supplied: self.supplied,
            floor: self.floor,
            removed: self.removed,
            evaporated: self.evaporated,
            tool_work_j: self.tool_work_j,
            scoop_clean_wood_kg: self.scoop_clean_wood_kg,
            breakups: self.breakups,
            max_overlap_m: self.max_overlap_m,
        }
    }

    /// Build the wire output.
    fn output(self, status: Status) -> Result<serde_json::Value, String> {
        let residuals = self.residuals();
        let intact: usize = self.boxes.iter().map(|b| b.world.pellets.len()).sum();
        let kinetic: f64 = self.boxes.iter().map(|b| b.world.kinetic_energy()).sum();
        let mut observables = BTreeMap::new();
        observables.insert(
            "mass_residual_kg",
            residuals.iter().fold(0.0_f64, |a, r| a.max(r.abs())),
        );
        observables.insert("wood_residual_kg", residuals[0]);
        observables.insert("waste_residual_kg", residuals[1]);
        observables.insert("water_residual_kg", residuals[2]);
        observables.insert("tool_work_j", self.tool_work_j);
        observables.insert("scoop_clean_wood_kg", self.scoop_clean_wood_kg);
        observables.insert("intact_pellets", dem::count_to_f64(intact));
        observables.insert(
            "pellet_breakups",
            dem::count_to_f64(usize::try_from(self.breakups).unwrap_or(usize::MAX)),
        );
        observables.insert("max_contact_overlap_m", self.max_overlap_m);
        observables.insert("pellet_kinetic_energy_j", kinetic);
        for value in observables.values() {
            if !value.is_finite() {
                return Err("output observable is not finite".to_string());
            }
        }
        let checkpoint = Checkpoint {
            schema_version: request::SCHEMA_VERSION,
            mode: request::MODE.to_string(),
            request: self.request_value.clone(),
            time_s: self.time(),
            state: self.state(),
        };
        let output = Output {
            schema_version: request::SCHEMA_VERSION,
            mode: request::MODE,
            status: match status {
                Status::Completed => "completed",
                Status::BudgetExhausted => "budget_exhausted",
            },
            fidelity: request::FIDELITY,
            time_s: self.time(),
            frames: self.frames,
            metrics: self.metrics,
            observables,
            diagnostics: diagnostics(),
            checkpoint,
        };
        serde_json::to_value(output).map_err(|e| format!("output serialisation failed: {e}"))
    }
}

/// Absolute time at the start of a step index (clamped to the duration).
fn step_time(request: &Request, footprint: &Footprint, step_index: u64) -> f64 {
    if step_index >= footprint.steps {
        return request.duration_s;
    }
    (dem::count_to_f64(usize::try_from(step_index).unwrap_or(usize::MAX)) * request.dt_s)
        .min(request.duration_s)
}

/// Compliant, force-limited drive toward the event's straight target path.
fn tool_drive(tool: &Tool, event: &Event, time: f64) -> Vec3 {
    let direction = Vec3::from(event.direction);
    let unit = direction.try_normalize(1e-12).unwrap_or_else(Vec3::zeros);
    let tau = (time - event.time_s).clamp(0.0, event.duration_s);
    let target = Vec3::from(event.position_m) + unit * (STROKE_SPEED_M_S * tau);
    let damping = 2.0 * (PAW_STIFFNESS_N_M * PAW_MASS_KG).sqrt();
    let force = (target - tool.position) * PAW_STIFFNESS_N_M - tool.velocity * damping;
    let magnitude = force.norm();
    if magnitude > PAW_FORCE_LIMIT_N {
        force * (PAW_FORCE_LIMIT_N / magnitude)
    } else {
        force
    }
}

/// Fixed diagnostics describing what this stage does not do.
fn diagnostics() -> Vec<String> {
    [
        "Synthetic, uncalibrated parameters",
        "preliminary_household: no measured validation, no research response tables",
        "Not implemented: paste spread/yield/adhesion; deposit solids do not move",
        "Not implemented: calibrated slot flux laws; sifting uses fixed placeholder rates",
        "Not implemented: capillary transport; drainage is gravity-only first-order",
        "Not implemented: box entrance/enclosure/pads; walls are open-topped planes",
        "Not implemented: pre-settled initial bed; pellets start on a loose seeded lattice",
        "Not implemented: research-grade breakup; damage is a moisture-time-disturbance rate",
        "Motion kinds dig/cover/stir share one force-limited sphere proxy model",
        "Scoop and clean ignore amount_kg; scoop removes a spherical region",
        "Refill amount below one pellet mass is added as fines",
        "Evaporation is a first-order sink without humidity or exposure dependence",
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

#[cfg(test)]
mod tests;
