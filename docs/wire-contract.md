# Native wire contract v1

The Python extension exposes `protocol_version() -> int` and
`run_json(request_json: str, resume_json: str | None = None) -> str`.
Python validates before calling; Rust must independently reject invalid input.
Native numerical errors become Python `ValueError` with contextual messages.
Python does not step particles. Top-level unknown fields are errors.

Rust modules implement `pub fn run(request: &serde_json::Value,
resume: Option<&serde_json::Value>) -> Result<serde_json::Value, String>`.
Integration owns binding/dispatch in `lib.rs`; each module uses private typed
serde models with deny_unknown_fields. Errors, not panics, on invalid input.

## Request

All fields shown below are REQUIRED unless stated. SI units. Finite values only.

```json
{
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
}
```

`boxes` IDs must be unique. Household requires >=1 box, research allows empty boxes.
`pellet_count` >=0, dimensions/density/stiffness positive, restitution/friction >=0
(restitution <=1), rates >=0, capacity >0. Fit pellets/slots to the box. Seed is an
unsigned 64-bit integer. Positive dt <= duration. Positive recording interval.
Wall-time limit is >0 and <=3600. Prevent unbounded allocation before allocating.

Event kinds: `urinate`, `defecate`, `dig`, `cover`, `stir`, `scoop`, `empty_drawer`,
`refill`, `clean`. Directions are finite vectors; nonzero for motion events.
`amount_kg` >=0, `water_fraction` in [0,1], `radius_m` positive, event duration >=0.
Events are ordered nondecreasing by time, time in [0,duration], box references valid.
Event positions are box-local. Passive events can have zero duration. Each event is
applied exactly once; if deposition/motion spans time, checkpoint its progress.

Research mode requires `research`, prohibits household events, and uses:

```json
{
  "fixture": "slump",
  "grid_spacing_m": 0.005,
  "domain_m": [0.06, 0.06, 0.06],
  "material": "paste",
  "density_kg_m3": 1000.0,
  "young_modulus_pa": 1000.0,
  "poisson_ratio": 0.2,
  "yield_stress_pa": 30.0,
  "consistency_pa_s_n": 10.0,
  "flow_index": 0.6,
  "initial_size_m": [0.02, 0.02, 0.02],
  "initial_velocity_m_s": [0.0, 0.0, 0.0]
}
```

Fixture enum: `slump`, `hydrostatic`, `dam_break`, `coupled_patch`.
Material enum: `paste`, `water`, `fines`.
Parameters must be finite/physically bounded. An unimplemented fixture/material is
an explicit error, not a silent alias to another model. Research implementer may
support only a documented verified subset; outstanding scope must be recorded.
No research run claims physical validation without external measurements.

## Output

```json
{
  "schema_version": 1,
  "mode": "household",
  "status": "completed",
  "fidelity": "preliminary_household",
  "time_s": 1.0,
  "frames": [{
    "time_s": 0.0,
    "positions_m": [[0.01,0.01,0.02]],
    "radii_m": [0.003],
    "materials": ["wood"],
    "box_ids": ["box-a"]
  }],
  "metrics": [{
    "time_s": 0.0,
    "compartment": "bed",
    "wood_kg": 0.001,
    "waste_kg": 0.0,
    "water_kg": 0.0
  }],
  "observables": {"mass_residual_kg": 0.0},
  "diagnostics": ["Synthetic, uncalibrated parameters"],
  "checkpoint": {
    "schema_version": 1,
    "mode": "household",
    "request": {},
    "time_s": 1.0,
    "state": {}
  }
}
```

Checkpoint `request` is the ORIGINAL complete request, not the empty example.
State is mode-private and contains ALL history, ledger/event progress/RNG needed to
resume identically on the same build. Resume compares request exactly except wall
budget, and rejects unknown checkpoint versions/mode, corrupt/missing/nonfinite state.
Simulation time is absolute since initialization. Resumed frames are a suffix, not a
replacement for earlier artifacts. Mode-private checkpoint shape is versioned by
schema_version; breaking changes increment it.

Status enum: `completed`, `budget_exhausted`. Numerical errors return an error, not
fake completion. Fidelity: `preliminary_household` or `research_unvalidated` until
explicit release evidence exists. All output arrays must be finite. Frame arrays have
equal lengths. Metrics use compartments `bed`, `drawer`, `floor`, `removed`, `evaporated`;
research may use `domain` and `outflow`. Include zeros where useful. Water in solid and
liquid phases must not be counted twice. Numerical models document omitted outputs.

Budget checks include initial setup and output construction. At a safe step boundary,
checkpoint before exhaustion. Python owns total process budget/artifact overhead.
Runtime resource caps reject oversized inputs, never change physics silently.

## Ownership

- Integration: root manifests/gate, `lib.rs`, extension stubs, shared contracts/status.
- Mechanics agent: `core/src/household.rs`, `core/src/dem.rs`, their tests/docs.
- Research agent: `core/src/research.rs`, `core/src/mpm/`, their tests/docs.
- Python agent: `python/litter_physics/` except `_core.pyi`, `tests/` except foundation,
  `examples/`, `schemas/`, Python CLI/schema/replay documentation.

Agents may temporarily edit `lib.rs` in their worktree for testing but MUST restore it
before committing, and provide integration registration instructions. No edits to
root manifests without proposing the required change to integration first.
