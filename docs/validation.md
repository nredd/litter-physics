# Verification evidence

This records executed checks, not intended acceptance. Nothing here establishes
cat-specific material properties or held-out household accuracy.

## Foundation and response tables

Environment: Apple M4 Mac mini, 16 GB RAM; Rust 1.98.1; Python 3.13.15.

- Native extension loads and exposes protocol version 1.
- Full `make gate` checks formatting, pedantic Clippy, Rust/Python types and tests,
  Rust documentation and committed JSON Schema agreement.
- Wheel/sdist installation and then-current tests also exercised Python 3.12.13 and
  3.14.7 using `uv run tox -e py312` and `uv run tox -e py314`.
- Response-table tests cover affine interpolation, evidence refusal, named inputs,
  nonfinite values, bounded exploration, constant physical bounds, allocation limits,
  schema drift and updated-copy cache/provenance invalidation.
- Independent Claude review found roundoff incorrectly marking valid interpolation
  unvalidated, repeated preparation overhead, and stale caches in Pydantic copies.
  Fixes use ULP-aware bound handling, a final fail-closed gate, cached preparation,
  fully revalidated copies and regression tests. No gate exceptions were added.

Synthetic interpolation microbenchmark on this Mac, 781,250 response values:

```text
cold_evaluation_s=0.079585
warm_evaluation_s=0.000104
```

This is Python table preparation/reference-evaluation throughput, NOT DEM/MPM or
household throughput. Python is not intended to evaluate every field cell per step;
a future native table consumer must preserve the same domain/evidence contract.

## Outstanding evidence

No measured geometry, pellet tests, cat observations, surrogate rheology, held-out
maintenance cycle, converged MPM study, or full simulation runtime acceptance has
been supplied or accepted yet. Numerical modules are undergoing separate integration.

## Initial native integration

Executed on the same M4/16 GB host, macOS 27.0:

```text
maintenance_smoke: 2 tiny boxes, 8 initial pellets, 7 events, 0.2 simulated seconds
wall_time_s=0.291861417  mass_residual_kg=6.505213e-19
household_basic: 2 tiny boxes, 48 initial pellets, 1 simulated second
wall_time_s=2.857846542  mass_residual_kg=3.469447e-18
research_slump: 512 material points, 5000 steps, 0.5 simulated seconds
wall_time_s=0.874982125  mass_residual_kg=0
momentum_residual_kg_m_s=8.326673e-17
```

These are actual compiled-kernel CLI runs, not mock tests. They establish execution,
artifact generation and ledger checks only. The contact-overlap outputs are nonzero
and have NOT met a stiffness/refinement acceptance study. No actual-size prediction
or measured-material accuracy follows from the above timings/residuals.

Real integration tests require both modes to complete, exercise real native checkpoint
and JSON round trips, reject request drift, and verify exported Rerun files. Viewer
checks fetch the loopback HTTP page and assert both ports close after shutdown. The
browser's visual rendering has not been manually inspected. Synthetic calibration was
run through the CLI and remained explicitly non-measured.

Integration fixes:
- Replace an incorrect permanent-empty drawer assertion with immediate inventory and
  removed-mass conservation checks; subsequent wet-bed drainage is expected.
- Start paw proxies above the bed rather than inside particles/floor.
- Disable release stripping because Rust's stripping step generated a misaligned
  Mach-O LINKEDIT string pool rejected by macOS 27. The unmodified linker image loads.
  Upstream issue: [here](https://github.com/rust-lang/rust/issues/157750).
- Enforce world coordinates and continuous recording identity, reap viewer children,
  reject build/provenance drift on resume, and atomically refuse artifact overwrites.
- Reject ambiguous/expanding configuration syntax and forged checkpoint field geometry.

Native tests use an optimized test profile with debug assertions and overflow checks
still enabled. This changes runtime cost, not numerical tolerances or gate coverage.

Final integration gate: 73 native tests and 107 Python tests passed, no skips. Python
coverage was 95%. Formatting, pedantic Clippy, ty, Rust docs and all schema checks were
clean. Installed sdist/wheel tests also passed on Python 3.12, 3.13 and 3.14.

The separate constitutive microbenchmark executed 10,000 real implicit paste updates:
`mean_update_s=0.000000343`. This is a kernel-only timing, not a visit benchmark.

CI pins the immutable commits for `actions/checkout` v7.0.1 and
`astral-sh/setup-uv` v10.1.0, with read-only repository permissions. The latter
release has no floating `v10` tag; the initial floating-tag CI attempt failed
before executing tests, so release commits are resolved and pinned explicitly. `tox` also requires the native extension explicitly,
so a packaging regression cannot turn real-kernel checks into skipped tests.

## Energy-ledger correction

The old `wall_work_j` was a grid projection kinetic-energy change, not physical
boundary work. The corrected semantics, signs and deliberately incomplete algebraic
balance are specified in `energy-ledgers.md`. No trajectory law was altered to hide
the residual. Exact initial-energy restart identity is now checked as well.

Integrated gate: 81 Rust unit tests, 2 Rust integration tests and 142 Python tests
passed, including coupled-step ledger booking, rejected-trial rollback and forged
restart-baseline rejection. An actual compiled CLI slump run completed and reported:

```text
wall_normal_projection_energy_j = -1.0937867722540651e-4
wall_friction_dissipation_j     =  3.548583755138611e-5
plastic_dissipation_j           =  1.7388556177281976e-4
energy_residual_j               =  8.447562054407812e-5
```

The nonzero residual is reported, not relabelled as an energy-conservation pass.
