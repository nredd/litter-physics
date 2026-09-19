//! Protocol, conservation, checkpoint, budget and physical behaviour tests.

use serde_json::{Value, json};

use super::*;

type Mutation = Box<dyn Fn(&mut Value)>;

/// Small but complete request following the wire contract example.
fn base_request() -> Value {
    json!({
        "schema_version": 1,
        "mode": "household",
        "seed": 7,
        "duration_s": 0.2,
        "dt_s": 0.002,
        "record_interval_s": 0.05,
        "max_wall_time_s": 600.0,
        "boxes": [{
            "id": "box-a",
            "size_m": [0.12, 0.10, 0.08],
            "origin_m": [0.5, 0.0, 0.0],
            "slot_width_m": 0.005,
            "slot_length_m": 0.015,
            "slot_pitch_m": 0.02,
            "drawer_depth_m": 0.03,
            "pellet_count": 12,
            "pellet_radius_m": 0.003,
            "pellet_length_m": 0.012,
            "pellet_density_kg_m3": 1100.0
        }],
        "materials": {
            "friction": 0.4,
            "restitution": 0.2,
            "normal_stiffness_n_m": 1000.0,
            "water_capacity_ratio": 3.0,
            "uptake_rate_s": 0.1,
            "breakdown_rate_s": 0.01,
            "evaporation_rate_s": 0.00001
        },
        "events": [],
        "research": null
    })
}

fn with_events(events: Value) -> Value {
    let mut request = base_request();
    request["events"] = events;
    request
}

fn urinate(time: f64) -> Value {
    json!({
        "time_s": time,
        "kind": "urinate",
        "box_id": "box-a",
        "position_m": [0.06, 0.05, 0.02],
        "direction": [0.0, 0.0, 0.0],
        "duration_s": 0.02,
        "amount_kg": 0.002,
        "water_fraction": 1.0,
        "radius_m": 0.015
    })
}

fn instant(kind: &str, time: f64, amount: f64) -> Value {
    json!({
        "time_s": time,
        "kind": kind,
        "box_id": "box-a",
        "position_m": [0.06, 0.05, 0.01],
        "direction": [0.0, 0.0, 0.0],
        "duration_s": 0.0,
        "amount_kg": amount,
        "water_fraction": 0.0,
        "radius_m": 0.03
    })
}

fn run_ok(request: &Value) -> Value {
    match run(request, None) {
        Ok(v) => v,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn metric(output: &Value, compartment: &str, species: &str) -> f64 {
    let rows = output["metrics"].as_array().expect("metrics array");
    let last_time = rows.last().expect("rows")["time_s"].as_f64().expect("time");
    rows.iter()
        .filter(|r| r["compartment"] == compartment && r["time_s"] == last_time)
        .map(|r| r[species].as_f64().expect("mass"))
        .sum()
}

fn observable(output: &Value, name: &str) -> f64 {
    output["observables"][name].as_f64().expect("observable")
}

fn assert_finite_output(output: &Value) {
    fn walk(v: &Value) {
        match v {
            Value::Number(n) => assert!(n.as_f64().is_some_and(f64::is_finite), "non-finite {n}"),
            Value::Array(a) => a.iter().for_each(walk),
            Value::Object(o) => o.values().for_each(walk),
            _ => {}
        }
    }
    walk(output);
    for frame in output["frames"].as_array().expect("frames") {
        let n = frame["positions_m"].as_array().expect("positions").len();
        assert_eq!(frame["radii_m"].as_array().expect("radii").len(), n);
        assert_eq!(frame["materials"].as_array().expect("materials").len(), n);
        assert_eq!(frame["box_ids"].as_array().expect("box ids").len(), n);
    }
}

#[test]
fn baseline_request_completes_with_conserved_species() {
    let output = run_ok(&base_request());
    assert_eq!(output["status"], "completed");
    assert_eq!(output["fidelity"], request::FIDELITY);
    assert_eq!(output["mode"], "household");
    assert_eq!(output["schema_version"], 1);
    assert!((output["time_s"].as_f64().expect("time") - 0.2).abs() < 1e-12);
    assert_eq!(output["frames"].as_array().expect("frames").len(), 5);
    assert_finite_output(&output);
    let expected = 12.0 * 1100.0 * std::f64::consts::PI * 0.003 * 0.003 * 0.012;
    assert!((metric(&output, "bed", "wood_kg") - expected).abs() < 1e-12);
    assert!(observable(&output, "mass_residual_kg") <= 1e-12);
    assert_eq!(output["checkpoint"]["request"], base_request());
    let first = &output["frames"][0];
    assert!(
        first["positions_m"][0][0].as_f64().expect("x") >= 0.5,
        "origin offset missing"
    );
    assert!(output["diagnostics"].as_array().expect("diagnostics").len() > 3);
}

#[test]
fn identical_requests_produce_identical_output() {
    let request = with_events(json!([urinate(0.05), instant("scoop", 0.15, 0.0)]));
    let a = run_ok(&request);
    let b = run_ok(&request);
    assert_eq!(a, b);
    let mut other = request.clone();
    other["seed"] = json!(8);
    let c = run_ok(&other);
    assert_ne!(a["frames"], c["frames"]);
}

#[test]
fn checkpoint_midway_resumes_to_identical_result() {
    let request = with_events(json!([urinate(0.03), instant("scoop", 0.12, 0.0)]));
    let full = run_ok(&request);
    let typed: Request = serde_json::from_value(request.clone()).expect("typed");
    let footprint = typed.validate().expect("valid");
    let mut sim = Simulation::new(typed, footprint, request.clone()).expect("sim");
    for _ in 0..40 {
        sim.step().expect("step");
    }
    let partial = sim.output(Status::BudgetExhausted).expect("partial output");
    assert_eq!(partial["status"], "budget_exhausted");
    let resumed = match run(&request, Some(&partial["checkpoint"])) {
        Ok(v) => v,
        Err(e) => panic!("resume failed: {e}"),
    };
    assert_eq!(resumed["status"], "completed");
    let mut frames = partial["frames"].as_array().expect("frames").clone();
    frames.extend(
        resumed["frames"]
            .as_array()
            .expect("frames")
            .iter()
            .cloned(),
    );
    assert_eq!(Value::Array(frames), full["frames"]);
    assert_eq!(resumed["checkpoint"]["state"], full["checkpoint"]["state"]);
    assert_eq!(resumed["observables"], full["observables"]);
}

#[test]
fn tiny_wall_budget_reports_exhaustion_and_resumes() {
    let mut request = base_request();
    request["max_wall_time_s"] = json!(1e-9);
    let output = run_ok(&request);
    assert_eq!(output["status"], "budget_exhausted");
    assert_eq!(output["checkpoint"]["mode"], "household");
    assert_eq!(output["frames"].as_array().expect("frames").len(), 1);
    let mut resume_request = request.clone();
    resume_request["max_wall_time_s"] = json!(600.0);
    let resumed = run(&resume_request, Some(&output["checkpoint"])).expect("resume");
    assert_eq!(resumed["status"], "completed");
    assert_eq!(resumed["frames"].as_array().expect("frames").len(), 4);
    assert_eq!(
        resumed,
        run_ok(&resume_request).clone_without_frames_prefix(1)
    );
}

trait FrameSuffix {
    fn clone_without_frames_prefix(&self, skip: usize) -> Value;
}

impl FrameSuffix for Value {
    fn clone_without_frames_prefix(&self, skip: usize) -> Value {
        let mut out = self.clone();
        let frames: Vec<Value> = self["frames"].as_array().expect("frames")[skip..].to_vec();
        out["frames"] = Value::Array(frames);
        let time = frames_min_time(&out["frames"]);
        let metrics: Vec<Value> = self["metrics"]
            .as_array()
            .expect("metrics")
            .iter()
            .filter(|m| m["time_s"].as_f64().expect("time") >= time)
            .cloned()
            .collect();
        out["metrics"] = Value::Array(metrics);
        out
    }
}

fn frames_min_time(frames: &Value) -> f64 {
    frames
        .as_array()
        .expect("frames")
        .iter()
        .map(|f| f["time_s"].as_f64().expect("time"))
        .fold(f64::INFINITY, f64::min)
}

fn assert_rejected(cases: Vec<(&str, Mutation)>) {
    for (name, mutate) in cases {
        let mut request = base_request();
        mutate(&mut request);
        assert!(
            run(&request, None).is_err(),
            "case '{name}' should be rejected"
        );
    }
}

#[test]
fn top_level_request_fields_are_validated() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "unknown top-level field",
            Box::new(|r| r["extra"] = json!(1)),
        ),
        ("wrong schema", Box::new(|r| r["schema_version"] = json!(2))),
        ("wrong mode", Box::new(|r| r["mode"] = json!("research"))),
        ("research present", Box::new(|r| r["research"] = json!({}))),
        ("dt above duration", Box::new(|r| r["dt_s"] = json!(1.0))),
        ("negative dt", Box::new(|r| r["dt_s"] = json!(-0.001))),
        (
            "zero record interval",
            Box::new(|r| r["record_interval_s"] = json!(0.0)),
        ),
        (
            "wall budget too large",
            Box::new(|r| r["max_wall_time_s"] = json!(3601.0)),
        ),
        (
            "zero wall budget",
            Box::new(|r| r["max_wall_time_s"] = json!(0.0)),
        ),
        ("no boxes", Box::new(|r| r["boxes"] = json!([]))),
        ("negative seed", Box::new(|r| r["seed"] = json!(-1))),
    ];
    assert_rejected(cases);
    assert!(run(&json!("not an object"), None).is_err());
    assert!(run(&json!({"schema_version": 1}), None).is_err());
}

#[test]
fn material_fields_are_validated() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "restitution above one",
            Box::new(|r| r["materials"]["restitution"] = json!(1.5)),
        ),
        (
            "zero stiffness",
            Box::new(|r| r["materials"]["normal_stiffness_n_m"] = json!(0.0)),
        ),
        (
            "zero capacity",
            Box::new(|r| r["materials"]["water_capacity_ratio"] = json!(0.0)),
        ),
        (
            "negative rate",
            Box::new(|r| r["materials"]["uptake_rate_s"] = json!(-1.0)),
        ),
        (
            "unknown material field",
            Box::new(|r| r["materials"]["cohesion"] = json!(1.0)),
        ),
    ];
    assert_rejected(cases);
}

#[test]
fn box_fields_and_resource_caps_are_validated() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "slot wider than pitch",
            Box::new(|r| r["boxes"][0]["slot_width_m"] = json!(0.03)),
        ),
        (
            "pellet larger than box",
            Box::new(|r| r["boxes"][0]["pellet_length_m"] = json!(0.2)),
        ),
        (
            "too many pellets for height",
            Box::new(|r| r["boxes"][0]["pellet_count"] = json!(5000)),
        ),
        (
            "pellet cap",
            Box::new(|r| {
                r["boxes"][0]["size_m"] = json!([2.0, 2.0, 2.0]);
                r["boxes"][0]["pellet_count"] = json!(20_000);
            }),
        ),
        (
            "field cell cap",
            Box::new(|r| r["boxes"][0]["size_m"] = json!([10.0, 10.0, 10.0])),
        ),
        (
            "empty box id",
            Box::new(|r| r["boxes"][0]["id"] = json!("")),
        ),
        (
            "duplicate box ids",
            Box::new(|r| {
                let b = r["boxes"][0].clone();
                r["boxes"].as_array_mut().expect("boxes").push(b);
            }),
        ),
        (
            "substep cap",
            Box::new(|r| {
                r["dt_s"] = json!(0.2);
                r["materials"]["normal_stiffness_n_m"] = json!(1e9);
            }),
        ),
        (
            "frame cap",
            Box::new(|r| {
                r["duration_s"] = json!(3000.0);
                r["record_interval_s"] = json!(0.001);
            }),
        ),
    ];
    assert_rejected(cases);
}

#[test]
fn event_fields_are_validated() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "event out of order",
            Box::new(|r| r["events"] = json!([urinate(0.1), urinate(0.05)])),
        ),
        (
            "event after duration",
            Box::new(|r| r["events"] = json!([urinate(0.5)])),
        ),
        (
            "event unknown box",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["box_id"] = json!("nope");
                r["events"] = json!([e]);
            }),
        ),
        (
            "motion without direction",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["kind"] = json!("dig");
                r["events"] = json!([e]);
            }),
        ),
        (
            "unknown event kind",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["kind"] = json!("nap");
                r["events"] = json!([e]);
            }),
        ),
        (
            "water fraction above one",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["water_fraction"] = json!(1.2);
                r["events"] = json!([e]);
            }),
        ),
        (
            "negative amount",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["amount_kg"] = json!(-0.1);
                r["events"] = json!([e]);
            }),
        ),
        (
            "event missing field",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e.as_object_mut().expect("event").remove("radius_m");
                r["events"] = json!([e]);
            }),
        ),
        (
            "event far outside box",
            Box::new(|r| {
                let mut e = urinate(0.1);
                e["position_m"] = json!([5.0, 0.0, 0.0]);
                r["events"] = json!([e]);
            }),
        ),
    ];
    assert_rejected(cases);
}

#[test]
fn corrupt_or_incompatible_checkpoints_are_rejected() {
    let request = base_request();
    let output = run_ok(&request);
    let checkpoint = output["checkpoint"].clone();
    assert!(
        run(&request, Some(&checkpoint)).is_ok(),
        "completed checkpoint resumes"
    );
    let cases: Vec<(&str, Mutation)> = vec![
        ("schema", Box::new(|c| c["schema_version"] = json!(99))),
        ("mode", Box::new(|c| c["mode"] = json!("research"))),
        ("unknown field", Box::new(|c| c["bogus"] = json!(1))),
        (
            "request drift",
            Box::new(|c| c["request"]["seed"] = json!(99)),
        ),
        ("missing state", Box::new(|c| c["state"] = json!(null))),
        (
            "negative water",
            Box::new(|c| c["state"]["boxes"][0]["pellets"][0]["water_kg"] = json!(-1.0)),
        ),
        (
            "degenerate orientation",
            Box::new(|c| {
                c["state"]["boxes"][0]["pellets"][0]["orientation_wijk"] =
                    json!([0.0, 0.0, 0.0, 0.0]);
            }),
        ),
        (
            "pellet outside box",
            Box::new(|c| {
                c["state"]["boxes"][0]["pellets"][0]["position_m"] = json!([5.0, 0.0, 0.0]);
            }),
        ),
        (
            "duplicate pellet id",
            Box::new(|c| {
                let id = c["state"]["boxes"][0]["pellets"][0]["id"].clone();
                c["state"]["boxes"][0]["pellets"][1]["id"] = id;
            }),
        ),
        (
            "field size drift",
            Box::new(|c| {
                c["state"]["boxes"][0]["field"]["water_kg"]
                    .as_array_mut()
                    .expect("array")
                    .pop();
            }),
        ),
        (
            "field negative",
            Box::new(|c| c["state"]["boxes"][0]["field"]["fines_kg"][0] = json!(-1.0)),
        ),
        ("time inconsistent", Box::new(|c| c["time_s"] = json!(0.05))),
        (
            "ledger corrupt",
            Box::new(|c| c["state"]["removed"]["wood"] = json!(-5.0)),
        ),
        (
            "event progress corrupt",
            Box::new(|c| {
                c["state"]["events"] =
                    json!([{"started": false, "finished": true, "applied_kg": 0.0}]);
            }),
        ),
        (
            "supplied drift breaks conservation",
            Box::new(|c| c["state"]["supplied"]["wood"] = json!(1.0)),
        ),
        (
            "tool with unknown event",
            Box::new(|c| {
                c["state"]["boxes"][0]["tool"] = json!({"event_index": 7, "position_m": [0.0, 0.0, 0.0], "velocity_m_s": [0.0, 0.0, 0.0]});
            }),
        ),
        (
            "wrong box count",
            Box::new(|c| c["state"]["boxes"] = json!([])),
        ),
    ];
    for (name, mutate) in cases {
        let mut bad = checkpoint.clone();
        mutate(&mut bad);
        assert!(
            run(&request, Some(&bad)).is_err(),
            "checkpoint case '{name}' should be rejected"
        );
    }
    let mut changed = request.clone();
    changed["duration_s"] = json!(0.3);
    assert!(
        run(&changed, Some(&checkpoint)).is_err(),
        "changed request must not resume"
    );
}

#[test]
fn urine_is_absorbed_drained_and_evaporated_conservatively() {
    let mut request = with_events(json!([urinate(0.02)]));
    request["materials"]["uptake_rate_s"] = json!(20.0);
    request["materials"]["evaporation_rate_s"] = json!(0.5);
    let output = run_ok(&request);
    assert_finite_output(&output);
    let bed_water = metric(&output, "bed", "water_kg");
    let drawer_water = metric(&output, "drawer", "water_kg");
    let evaporated = metric(&output, "evaporated", "water_kg");
    assert!(bed_water > 0.0, "pellets/field hold no water");
    assert!(evaporated > 0.0, "nothing evaporated");
    assert!((bed_water + drawer_water + evaporated - 0.002).abs() <= 1e-6 * 0.002);
    assert!(observable(&output, "water_residual_kg").abs() <= 1e-6 * 0.002);
    let rows = output["metrics"].as_array().expect("metrics");
    let waters: Vec<f64> = rows
        .iter()
        .filter(|r| r["compartment"] == "evaporated")
        .map(|r| r["water_kg"].as_f64().expect("water"))
        .collect();
    assert!(
        waters.windows(2).all(|w| w[1] >= w[0]),
        "evaporation must be monotonic"
    );
}

#[test]
fn wet_pellets_break_into_fines_that_sift_to_the_drawer() {
    let mut request = with_events(json!([urinate(0.01)]));
    request["duration_s"] = json!(1.0);
    request["record_interval_s"] = json!(0.25);
    request["materials"]["uptake_rate_s"] = json!(50.0);
    request["materials"]["breakdown_rate_s"] = json!(200.0);
    let output = run_ok(&request);
    assert!(
        observable(&output, "pellet_breakups") >= 1.0,
        "no breakup occurred"
    );
    assert!(observable(&output, "intact_pellets") < 12.0);
    assert!(
        metric(&output, "drawer", "wood_kg") > 0.0,
        "fines did not sift"
    );
    assert!(observable(&output, "mass_residual_kg") <= 1e-9);
    let dry = run_ok(&base_request());
    assert!(
        (observable(&dry, "pellet_breakups")).abs() < f64::EPSILON,
        "dry pellets must not break"
    );
}

#[test]
fn scoop_refill_empty_drawer_and_clean_are_stateful() {
    let mut request = with_events(json!([
        urinate(0.01),
        instant("scoop", 0.05, 0.0),
        instant("scoop", 0.06, 0.0),
        instant("refill", 0.08, 0.004),
        instant("empty_drawer", 0.12, 0.0),
        instant("clean", 0.18, 0.0)
    ]));
    request["materials"]["uptake_rate_s"] = json!(50.0);
    request["materials"]["breakdown_rate_s"] = json!(200.0);
    let output = run_ok(&request);
    assert_finite_output(&output);
    let rows = output["metrics"].as_array().expect("metrics");
    let at = |time: f64, compartment: &str, species: &str| -> f64 {
        rows.iter()
            .filter(|r| {
                r["compartment"] == compartment
                    && (r["time_s"].as_f64().expect("t") - time).abs() < 1e-9
            })
            .map(|r| r[species].as_f64().expect("mass"))
            .sum()
    };
    let removed_after_scoop = at(0.10, "removed", "wood_kg");
    assert!(removed_after_scoop > 0.0, "scoop removed nothing");
    assert!(observable(&output, "scoop_clean_wood_kg") > 0.0);
    let pellet_mass = 1100.0 * std::f64::consts::PI * 0.003 * 0.003 * 0.012;
    let bed_after_refill = at(0.10, "bed", "wood_kg") + at(0.10, "drawer", "wood_kg");
    assert!(
        bed_after_refill + removed_after_scoop > 12.0 * pellet_mass,
        "refill added nothing"
    );
    // The drawer is emptied at 0.12 s, but the wet bed keeps draining afterwards.
    // Inspect removed mass, not an unjustified zero inventory 0.03 s later.
    assert!(
        at(0.15, "removed", "wood_kg") + at(0.15, "removed", "water_kg")
            > at(0.10, "removed", "wood_kg") + at(0.10, "removed", "water_kg"),
        "emptying the drawer removed no material"
    );
    assert!(
        at(0.20, "bed", "wood_kg") < 1e-15,
        "clean left wood in the bed"
    );
    assert!(
        at(0.20, "bed", "water_kg") < 1e-15,
        "clean left water in the bed"
    );
    let supplied = 12.0 * pellet_mass + 0.004;
    let removed = at(0.20, "removed", "wood_kg");
    assert!(
        (removed - supplied).abs() <= 1e-6 * supplied,
        "removed {removed} vs supplied {supplied}"
    );
    assert!(observable(&output, "mass_residual_kg") <= 1e-9);
}

#[test]
fn dig_event_moves_pellets_with_bounded_force_and_logged_work() {
    let dig = json!({
        "time_s": 0.02,
        "kind": "dig",
        "box_id": "box-a",
        "position_m": [0.02, 0.05, 0.004],
        "direction": [1.0, 0.0, 0.0],
        "duration_s": 0.15,
        "amount_kg": 0.0,
        "water_fraction": 0.0,
        "radius_m": 0.012
    });
    let request = with_events(json!([dig]));
    let output = run_ok(&request);
    assert_finite_output(&output);
    assert!(observable(&output, "tool_work_j") > 0.0, "tool did no work");
    let frames = output["frames"].as_array().expect("frames");
    let paw_frames = frames
        .iter()
        .filter(|f| {
            f["materials"]
                .as_array()
                .expect("materials")
                .iter()
                .any(|m| m == "paw")
        })
        .count();
    assert!(paw_frames >= 2, "paw proxy never appeared in frames");
    let last = frames.last().expect("frame");
    assert!(
        !last["materials"]
            .as_array()
            .expect("materials")
            .iter()
            .any(|m| m == "paw")
    );
    let quiet = run_ok(&base_request());
    assert_ne!(quiet["frames"][4]["positions_m"], last["positions_m"]);
    assert!(observable(&output, "pellet_kinetic_energy_j") < 1.0);
    let tool = Tool {
        id: 0,
        position: Vec3::new(0.0, 0.0, 0.0),
        velocity: Vec3::zeros(),
        radius: 0.01,
        mass: PAW_MASS_KG,
        drive_force: Vec3::zeros(),
    };
    let event: Event = serde_json::from_value(request["events"][0].clone()).expect("event");
    let drive = tool_drive(&tool, &event, 10.0);
    assert!(
        drive.norm() <= PAW_FORCE_LIMIT_N + 1e-12,
        "drive force exceeds limit"
    );
}

#[test]
fn defecation_deposits_solids_that_stay_in_bed_until_scooped() {
    let mut defecate = urinate(0.01);
    defecate["kind"] = json!("defecate");
    defecate["water_fraction"] = json!(0.25);
    defecate["amount_kg"] = json!(0.004);
    let request = with_events(json!([defecate, instant("scoop", 0.15, 0.0)]));
    let output = run_ok(&request);
    let rows = output["metrics"].as_array().expect("metrics");
    let bed_waste_before: f64 = rows
        .iter()
        .filter(|r| {
            r["compartment"] == "bed" && (r["time_s"].as_f64().expect("t") - 0.1).abs() < 1e-9
        })
        .map(|r| r["waste_kg"].as_f64().expect("waste"))
        .sum();
    assert!(
        (bed_waste_before - 0.003).abs() < 1e-12,
        "bed waste {bed_waste_before}"
    );
    assert!(
        metric(&output, "removed", "waste_kg") > 0.0,
        "scoop removed no waste"
    );
    let waste_markers = output["frames"][2]["materials"]
        .as_array()
        .expect("materials")
        .iter()
        .filter(|m| *m == "waste")
        .count();
    assert!(waste_markers > 0, "waste cells not rendered");
}

#[test]
fn two_boxes_keep_independent_state() {
    let mut request = base_request();
    let mut second = request["boxes"][0].clone();
    second["id"] = json!("box-b");
    second["origin_m"] = json!([1.0, 0.0, 0.0]);
    request["boxes"].as_array_mut().expect("boxes").push(second);
    let mut event = urinate(0.02);
    event["box_id"] = json!("box-b");
    request["events"] = json!([event]);
    request["materials"]["uptake_rate_s"] = json!(20.0);
    let output = run_ok(&request);
    let state = &output["checkpoint"]["state"]["boxes"];
    let water = |i: usize| -> f64 {
        state[i]["pellets"]
            .as_array()
            .expect("pellets")
            .iter()
            .map(|p| p["water_kg"].as_f64().expect("water"))
            .sum::<f64>()
            + state[i]["field"]["water_kg"]
                .as_array()
                .expect("field")
                .iter()
                .map(|w| w.as_f64().expect("w"))
                .sum::<f64>()
    };
    assert!(water(0).abs() < f64::EPSILON, "box-a received water");
    assert!(water(1) > 0.0, "box-b received no water");
    assert!(observable(&output, "mass_residual_kg") <= 1e-9);
}

#[test]
fn wire_contract_example_runs_to_completion() {
    let request = json!({
        "schema_version": 1,
        "mode": "household",
        "seed": 1,
        "duration_s": 1.0,
        "dt_s": 0.001,
        "record_interval_s": 0.05,
        "max_wall_time_s": 3600.0,
        "boxes": [{
            "id": "box-a",
            "size_m": [0.12, 0.1, 0.08],
            "origin_m": [0.0, 0.0, 0.0],
            "slot_width_m": 0.005,
            "slot_length_m": 0.015,
            "slot_pitch_m": 0.02,
            "drawer_depth_m": 0.03,
            "pellet_count": 24,
            "pellet_radius_m": 0.003,
            "pellet_length_m": 0.012,
            "pellet_density_kg_m3": 1100.0
        }],
        "materials": {
            "friction": 0.4,
            "restitution": 0.2,
            "normal_stiffness_n_m": 1000.0,
            "water_capacity_ratio": 3.0,
            "uptake_rate_s": 0.1,
            "breakdown_rate_s": 0.01,
            "evaporation_rate_s": 0.00001
        },
        "events": [{
            "time_s": 0.1,
            "kind": "urinate",
            "box_id": "box-a",
            "position_m": [0.06, 0.05, 0.02],
            "direction": [1.0, 0.0, 0.0],
            "duration_s": 0.1,
            "amount_kg": 0.002,
            "water_fraction": 1.0,
            "radius_m": 0.015
        }],
        "research": null
    });
    let output = run_ok(&request);
    assert_eq!(output["status"], "completed");
    assert_eq!(output["frames"].as_array().expect("frames").len(), 21);
    assert_finite_output(&output);
    assert!(observable(&output, "mass_residual_kg") <= 1e-9);
    assert!(observable(&output, "intact_pellets") > 0.0);
    assert!(
        observable(&output, "pellet_kinetic_energy_j") < 1e-4,
        "bed did not settle"
    );
    assert!(metric(&output, "bed", "water_kg") > 0.0);
}

#[test]
fn empty_drawer_preserves_the_bed_and_transfers_current_inventory() {
    let value = with_events(json!([instant("empty_drawer", 0.0, 0.0)]));
    let request: Request = serde_json::from_value(value.clone()).expect("valid request");
    let footprint = request.validate().expect("valid footprint");
    let mut sim = Simulation::new(request, footprint, value).expect("valid simulation");
    let inventory = Compartment {
        wood: 0.002,
        waste: 0.001,
        water: 0.003,
    };
    sim.boxes[0].drawer = inventory;
    sim.supplied.add(inventory);
    let before = sim.boxes[0].record();
    let event = sim.request.events[0].clone();
    sim.start_event(0, 0, &event);
    assert!(sim.boxes[0].drawer.wood.abs() < 1e-15);
    assert!(sim.boxes[0].drawer.waste.abs() < 1e-15);
    assert!(sim.boxes[0].drawer.water.abs() < 1e-15);
    assert_eq!(sim.boxes[0].record().pellets, before.pellets);
    assert_eq!(sim.boxes[0].record().field, before.field);
    assert!((sim.removed.wood - inventory.wood).abs() < 1e-15);
    assert!((sim.removed.waste - inventory.waste).abs() < 1e-15);
    assert!((sim.removed.water - inventory.water).abs() < 1e-15);
    sim.check_conservation().expect("all species conserved");
}

#[test]
fn paw_enters_above_the_bed_instead_of_teleporting_into_it() {
    let mut event = instant("dig", 0.0, 0.0);
    event["direction"] = json!([1.0, 0.0, 0.0]);
    event["duration_s"] = json!(0.1);
    let value = with_events(json!([event]));
    let request: Request = serde_json::from_value(value.clone()).expect("request");
    let footprint = request.validate().expect("valid footprint");
    let mut sim = Simulation::new(request, footprint, value).expect("simulation");
    let bed_top = sim.boxes[0].bed_top();
    let event = sim.request.events[0].clone();
    sim.start_event(0, 0, &event);
    let tool = &sim.boxes[0].world.tools[0];
    assert!(tool.position.z - tool.radius >= bed_top - 1e-12);
    assert!(tool_drive(tool, &event, 0.0).norm() <= PAW_FORCE_LIMIT_N + 1e-12);
}

#[test]
fn checkpoint_cannot_change_field_geometry() {
    let mut value = base_request();
    value["boxes"][0]["pellet_count"] = json!(0);
    value["duration_s"] = json!(0.002);
    let output = run_ok(&value);
    let mut checkpoint = output["checkpoint"].clone();
    checkpoint["state"]["boxes"][0]["field"]["spacing_m"] = json!(0.01);
    assert!(run(&value, Some(&checkpoint)).is_err());
    let mut checkpoint = output["checkpoint"].clone();
    checkpoint["state"]["boxes"][0]["field"]["slot_open"][0] = json!(2.0);
    assert!(run(&value, Some(&checkpoint)).is_err());
}
