//! Independent wire, restart and rejection tests for research fixtures.
use super::*;

fn request() -> Value {
    json!({
        "schema_version": 1, "mode": "research", "seed": 3,
        "duration_s": 0.002, "dt_s": 0.0005, "record_interval_s": 0.001,
        "max_wall_time_s": 60.0, "boxes": [], "events": [],
        "materials": {
            "friction": 0.3, "restitution": 0.1, "normal_stiffness_n_m": 1000.0,
            "water_capacity_ratio": 3.0, "uptake_rate_s": 0.1,
            "breakdown_rate_s": 0.01, "evaporation_rate_s": 0.0
        },
        "research": {
            "fixture": "slump", "grid_spacing_m": 0.005,
            "domain_m": [0.03, 0.03, 0.03], "material": "paste",
            "density_kg_m3": 1000.0, "young_modulus_pa": 1000.0,
            "poisson_ratio": 0.2, "yield_stress_pa": 30.0,
            "consistency_pa_s_n": 10.0, "flow_index": 0.6,
            "initial_size_m": [0.01, 0.01, 0.01], "initial_velocity_m_s": [0.0, 0.0, 0.0]
        }
    })
}

#[test]
fn completes_a_real_fixture_without_claiming_validation() {
    let output = run(&request(), None).expect("fixture runs");
    assert_eq!(output["status"], "completed");
    assert_eq!(output["fidelity"], "research_unvalidated");
    assert_eq!(output["frames"].as_array().expect("frames").len(), 3);
    assert!(
        output["diagnostics"]
            .as_array()
            .expect("diagnostics")
            .iter()
            .any(|message| message
                .as_str()
                .is_some_and(|text| text.contains("normal grid work is unledgered")))
    );
    assert!(
        output["observables"]["mass_residual_kg"]
            .as_f64()
            .expect("residual")
            .abs()
            < 1e-12
    );
}

#[test]
fn limiter_history_distinguishes_endpoint_clips_from_adaptive_reductions() {
    let typed: Request = serde_json::from_value(request()).expect("request");
    let spec = specification(&typed).expect("spec");
    let (mut sim, _) = spec.build().expect("build");
    sim.config.gravity = Vector3::zeros();
    let cap = sim.stability_limit() * 0.25;
    sim.config.max_dt = cap;
    sim.advance_to(2.5 * cap)
        .expect("includes an endpoint clip");
    assert_eq!(spec.observables(&sim).get("limited_steps"), Some(&0.0));
    assert!((sim.max_dt_used - cap).abs() < 1e-15);

    // A previous rejection can leave recovery scaling active even after the run
    // has already reached the requested cap. The maximum alone misses this.
    sim.dt_scale = 0.1;
    sim.advance_to(sim.time + cap).expect("recovery steps");
    assert!(spec.observables(&sim)["limited_steps"] > 0.0);
    assert!((sim.max_dt_used - cap).abs() < 1e-15);
}

#[test]
fn unsupported_fines_and_bad_parameters_fail_closed() {
    for (key, value) in [
        ("material", json!("fines")),
        ("material", json!("unknown")),
        ("poisson_ratio", json!(0.5)),
        ("flow_index", json!(0)),
        ("grid_spacing_m", json!(1e-300)),
        ("initial_size_m", json!([0.011, 0.01, 0.01])),
    ] {
        let mut input = request();
        input["research"][key] = value;
        assert!(run(&input, None).is_err(), "accepted invalid field {key}");
    }
    let mut input = request();
    input["unknown"] = json!(1);
    assert!(run(&input, None).is_err());
}

#[test]
fn full_state_json_round_trip_matches_uninterrupted_trajectory() {
    let mut input = request();
    input["dt_s"] = json!(0.002);
    let typed: Request = serde_json::from_value(input.clone()).expect("request");
    let spec = specification(&typed).expect("spec");
    let (mut sim, _) = spec.build().expect("build");
    sim.advance_to(0.001).expect("advance to record boundary");
    let checkpoint = Checkpoint {
        schema_version: 1,
        mode: "research".into(),
        request: input.clone(),
        time_s: sim.time,
        state: snapshot(&sim, 2),
    };
    let encoded = serde_json::to_string(&checkpoint).expect("serialize checkpoint");
    let restored = serde_json::from_str(&encoded).expect("parse checkpoint");
    let resumed = run(&input, Some(&restored)).expect("resume");
    let full = run(&input, None).expect("full");
    assert_eq!(resumed["checkpoint"], full["checkpoint"]);
    assert_eq!(resumed["observables"], full["observables"]);
    assert!(
        full["observables"]["limited_steps"]
            .as_f64()
            .expect("count")
            > 0.0
    );
    assert_eq!(resumed["frames"].as_array().expect("frames").len(), 1);
    assert_eq!(resumed["frames"][0], full["frames"][2]);
}

#[test]
fn corrupt_checkpoint_histories_and_requests_are_rejected() {
    let input = request();
    let full = run(&input, None).expect("run");
    for (key, value) in [
        ("dt_scale", json!(0)),
        ("next_record_index", json!(0)),
        ("limited_steps", json!(u64::MAX)),
    ] {
        let mut checkpoint = full["checkpoint"].clone();
        checkpoint["state"][key] = value;
        assert!(run(&input, Some(&checkpoint)).is_err());
    }
    let mut checkpoint = full["checkpoint"].clone();
    checkpoint["state"]["particles"]["mass"][0] = json!(-1);
    assert!(run(&input, Some(&checkpoint)).is_err());
    let mut checkpoint = full["checkpoint"].clone();
    checkpoint["state"]["particles"]["position"][0] = json!([100.0, 0.0, 0.0]);
    assert!(
        run(&input, Some(&checkpoint)).is_err(),
        "accepted a completed checkpoint with a particle outside the grid"
    );
    let mut checkpoint = full["checkpoint"].clone();
    checkpoint["state"]["ledger"]["initial_mechanical_energy"] = json!(1.0);
    assert!(
        run(&input, Some(&checkpoint)).is_err(),
        "accepted a forged initial-energy baseline"
    );
    let mut changed = input.clone();
    changed["seed"] = json!(99);
    assert!(run(&changed, Some(&full["checkpoint"])).is_err());
}

#[test]
fn budget_expiry_and_restoration_are_explicit() {
    let mut input = request();
    input["max_wall_time_s"] = json!(1e-12);
    let partial = run(&input, None).expect("partial checkpoint");
    assert_eq!(partial["status"], "budget_exhausted");
    input["max_wall_time_s"] = json!(60);
    let complete = run(&input, Some(&partial["checkpoint"])).expect("resume");
    assert_eq!(complete["status"], "completed");
}

#[test]
fn balanced_outflow_is_not_valid_for_closed_research_fixtures() {
    let typed: Request = serde_json::from_value(request()).expect("request");
    let spec = specification(&typed).expect("spec");
    let (mut sim, _) = spec.build().expect("build");
    sim.ledger.outflow_mass += sim.particles.remove_sorted(&[0]);
    assert!(sim.mass_residual().abs() < 1e-12);
    assert!(
        check_closed_state(&sim).is_err(),
        "balanced ledger concealed a wall leak"
    );
}

#[test]
fn cancelled_adaptive_advance_preserves_state() {
    let typed: Request = serde_json::from_value(request()).expect("request");
    let spec = specification(&typed).expect("spec");
    let (mut sim, _) = spec.build().expect("build");
    let original = sim.clone();
    assert!(!sim.advance_while(0.001, || false).expect("cancel"));
    assert_eq!(sim, original);
    assert!(sim.advance_to(f64::NAN).is_err());
    assert!(sim.advance_to(-1.0).is_err());
}
