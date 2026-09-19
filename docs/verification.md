# Refinement verification studies

`litter-physics verify STUDY --out DIR` runs one research fixture at three or more
resolutions per axis through the real compiled kernel and writes a machine-readable
report. This is numerical verification of the solver. It is NOT measured surrogate
validation, NOT household calibration, and NEVER grounds to promote a response table;
every case output stays `research_unvalidated` and the report says so in `claim`.

## Study specification

`examples/verification_slump.yaml`, schema `schemas/verification-study.schema.json`:

- `base_request`: research request, relative to the spec. Duration, geometry,
  material and seed come from it and never vary inside a study
- `spatial`: `grid_spacings_m` (3..6 values, strictly decreasing) at one fixed `dt_s`.
  Pick `dt_s` below the CFL limit of the FINEST grid or the axis cannot pass
- `temporal`: `dt_values_s` (3..6 values, strictly decreasing) on one fixed
  `grid_spacing_m`
- `observables`: native observable names, each with a documented `near_zero_scale`
  in its own units. The scale is the denominator floor for relative changes and the
  threshold below which a value is uninformative
- `max_relative_change`: strict bound on the final-two relative change, at most 0.05
- `per_case_wall_budget_s` (<= 3600) and `total_wall_budget_s` (<= 12 h)

At least one axis is required. The two axes are evaluated SEPARATELY so grid and
timestep error are not conflated.

## What runs

`plan_cases` expands every case and validates it before anything starts. In addition
to the request model, `check_grid_fit` mirrors the kernel's grid-fit and resource
rules (domain an integer multiple of spacing with >= 2 cells per axis, block spanning
whole particle subcells, 100,000 particles, 1,000,000 padded nodes, 2,000,000 recorded
point samples, coupled pellet radius >= 1.5 cells) and raises `StudyPlanError` so a
study that would be rejected fails closed with no artifacts. The native check remains
authoritative; a native rejection during execution marks that case `failed`.

Each case is a normal run directory (`DIR/cases/<axis>/NNNN`) started through
`runner.start_run` with `wall_budget_s = min(per_case_wall_budget_s, remaining total)`
and no Rerun recording. When under `MIN_CASE_BUDGET_S` (2 s) of the total remains,
later cases are `not_started`. Case artifacts, including checkpoints of incomplete
cases, are never deleted. `DIR` must be new or empty; the spec copy and report are
written once and never overwritten.

The total cap is checked before each case starts and bounds each case's own budget.
It does NOT preempt a running case: the runner's own deadline accounting ends a case,
and artifact finalization, evaluation and provenance capture can overrun the study
budget. There is no guaranteed upper bound on that finalization tail. `total_wall_time_s` is measured after evaluation, just before the report model is
assembled; the report write and log tail are not included. Any overrun sets
`total_budget_exceeded`, is listed in `reasons`, and makes the outcome `unresolved`
even when every axis passed.

## Evaluation

Per axis:

- `complete` is true only when every case is `completed`. Otherwise the axis is
  `incomplete`, the criterion is `not_evaluated`, and available values are still listed
- Case integrity: every `completed` case must have reached exactly `duration_s` and
  carry the `research_unvalidated` fidelity label; otherwise the axis is `unresolved`
  and the report is still written
- `timestep_cap_binding`: `max_dt_s == dt_s` (within 1e-6 relative) only shows the
  cap bound at least once, so it is necessary, not sufficient. Every completed case
  must also report `step_count` (> 0), `rejected_steps` and `limited_steps` as finite
  nonnegative integers, with both counts zero. `limited_steps` is the kernel's count
  of accepted steps where `stability_limit * dt_scale < dt_s` before final/record
  clipping, i.e. the adaptive limiter governed. Missing or malformed diagnostics fail
  closed. Temporal `step_count` must strictly increase. The kernel count is
  conservative: endpoint clipping alone does not increment it, but an active limiter
  is counted even if the endpoint subsequently clips that step further
- Per observable: `absolute_changes`, `relative_changes` over
  `max(|finer value|, near_zero_scale)`, `final_relative_change`, `trend`
  (`decreasing` / `non_monotone` / `undetermined`) and `observed_order`
  (`log(d_prev / d_last) / log(r)` when the refinement ratio is uniform and both
  changes are positive). Outcomes: `passed`, `unresolved`, `uninformative` (finest
  value below the scale), `missing`, `nonfinite`. A difference or ratio that overflows
  from finite extrema is `nonfinite` and stored as `null`; the report models reject
  any nonfinite float, so a report is either fully finite or not written

Study level: `study_complete` (every case completed), `criterion_met` (every
observable passed on every axis, common fidelity `research_unvalidated`, total cap not
exceeded), `outcome` in `passed` / `unresolved` / `incomplete`, `reasons`.
Exit codes: 0 `passed`, 1 `unresolved`, 3 `incomplete`. An `unresolved` axis is
reported with its trends; a decreasing trend is an error trend, not proof of
convergence, and a `passed` axis is a threshold check at the final two resolutions,
not an asymptotic claim.

The report (`schemas/verification-report.schema.json`) carries the spec and base
request hashes, fixture, material, duration, seed, the common native `fidelity`,
Python/native/git/dependency/hardware provenance, the per-case outcomes and
observables, `omitted_gates` and `claim`.

## Omitted gates

This facility does not cover: domain-size refinement, combined space-time refinement,
Richardson or asymptotic-range error estimation, species residual acceptance, measured
clean-surrogate validation, or response-table promotion. Those remain open in
`docs/status.md`.

## Executed evidence

Apple M4 Mac mini, 16 GB, macOS 27.0, Rust 1.98.1, Python 3.13.15, release build via
`uv sync --locked`. `litter-physics verify examples/verification_slump.yaml`,
six real kernel cases, total wall 14.6 s, every case `completed` at 0.5 s, every
timestep cap binding with `rejected_steps` 0 and `limited_steps` 0, exit 1:

```text
spatial (dt_s=5e-5; grid 0.01 -> 0.005 -> 0.0025 m; 64 -> 512 -> 4096 points)
  slump_m               passed      0.00398975 0.00436233 0.00451651  rel 0.0854 0.0341
  spread_x_m            unresolved  0.0226655  0.0235512  0.0259463   rel 0.0376 0.0923
  plastic_dissipation_j unresolved  0.00012974 0.00015538 0.000226479 rel 0.1650 0.3139
temporal (grid 0.005 m; dt_s 4e-4 -> 2e-4 -> 1e-4; steps 1250 -> 2500 -> 5000)
  slump_m               passed      0.00473759 0.00460771 0.00448748  rel 0.0282 0.0268
  spread_x_m            passed      0.0250541  0.0246164  0.0241053   rel 0.0178 0.0212
  plastic_dissipation_j unresolved  0.000209528 0.000191709 0.000173886 rel 0.0929 0.1025
outcome=unresolved study_complete=true criterion_met=false fidelity=research_unvalidated
```

That is the honest result: the slump fixture at these resolutions is NOT resolved in
spread or dissipation, and the spatial spread change grows under refinement. No
acceptance follows from this study. `tests/test_verification.py::test_native_small_study`
runs a shorter real study (0.05 s, same grids and steps) on every gate run and asserts
completion, provenance and a `passed`-or-`unresolved` outcome only.

References:
- ASME V&V 20 refinement terminology: [here](https://doi.org/10.1115/1.2960953)
- docs/plan.md (Acceptance), docs/research.md
