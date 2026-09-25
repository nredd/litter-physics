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
checks fetch the loopback HTTP page and assert both ports close after shutdown. At that initial milestone,
browser rendering had not been visually inspected; see the later replay milestone below. Synthetic calibration was
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

## Formal manuscript evidence

`docs/formal/simulation.pdf` and its LaTeX source pin the scientific implementation
at `9a66a55`. The manuscript archives the original six-case report and generates
its plots/tables from checksum-verified evidence. A fresh compiled-kernel study
reproduced every archived native observable and refinement verdict exactly; it
completed but remained unresolved, with the expected exit status 1.

The manuscript gate passed 81 Rust unit tests, 4 Rust integration tests and 151
Python tests, with formatting/lint/types, Rust docs and schema checks clean.
The installed-package tests also passed all 151 cases on Python 3.12 and 3.14.
Two new native tests verify the worked paste return/energy-gap calculation and
liquid EOS/first-order volume update, including inverted-volume rejection.
Python tests independently evaluate the scalar return and archived verdicts and
exercise tampered evidence, stale generated data and typesetting-error detection.
Tectonic 0.17.0 compiles the PDF without warnings; rendered pages were inspected.
These checks validate the document's calculations and provenance, not the missing
physical energy closure or measured material/household accuracy.


## Wall-contact correction

The wall-adjacent MLS quadrature lacked normal lattice support. Face-normal
image reactions now use the wall-plane active set, reject pulling reactions,
and contribute their own Coulomb budget. This is a nodal boundary approximation,
not an exact contact solution; see `hydrostatic-balance.md` for the derivation,
rejected variants, retained counterexamples and limits. At that milestone, native
output diagnostics flagged omitted normal grid work. The later signed grid-energy
accounting below replaces that warning; physical energy closure remains missing.

Integrated gate: **105 Rust tests and 153 Python tests**, no skips, 95% Python
coverage, formatting/lint/types, Rust docs and schema checks clean. Installed-package
tests passed all 153 cases on Python 3.12 and 3.14 as well as the gate's 3.13.
The matched-resolution native regression is part of the normal gate, not ignored.
Independent tests cover release, rotated walls/corners, slip/stick, vanishing
normal reaction and exactly zero friction without an artificial velocity cutoff.

Actual compiled CLI runs, original water column (`20 x 20 x 10 mm`, `h=2 mm`,
`dt=50 us`, 4000 points, 10 ms simulated):

```text
max normalized pressure error: 0.2969891369 before -> 0.0115138566 after
rms normalized pressure error: 0.0646332251 before -> 0.0034430221 after
mass_residual_kg             = -8.673617379884035e-19
momentum_residual_kg_m_s      =  4.770490026058286e-18
energy_residual_j            =  1.4356728261000955e-6
limited_steps = rejected_steps = 0
wall_time_s                 =  0.218
```

Error uses **every** particle, normalized by `rho0 g H`; no smoothing or discarded
wall layers. At 0.1 s the native maximum stays at 1.27%; with zero wall friction
it is 1.00%, so the improvement does not require frictional damping.

Both six-case CLI studies completed with zero limited/rejected steps and exit
**1 (`unresolved`)**, not a convergence pass. Hydrostatic study: 3.48 s wall time;
slump study: 11.83 s. Final-two relative changes:

| study/axis | observable | relative change |
| --- | --- | --- |
| hydrostatic/spatial | max pressure error | 57.17% |
| hydrostatic/spatial | rms pressure error | 101.11% |
| hydrostatic/temporal | max pressure error | 30.23% |
| hydrostatic/temporal | rms pressure error | 39.13% |
| slump/spatial | slump height | 9.09% |
| slump/spatial | spread | 9.72% |
| slump/spatial | plastic dissipation | 36.56% |
| slump/temporal | slump height | 2.67% |
| slump/temporal | spread | 1.67% |
| slump/temporal | plastic dissipation | 9.45% |

Local reports: `outputs/wall-contact-final-hydrostatic-study/study_report.json`
and `outputs/wall-contact-final-slump-study/study_report.json`. These are working-tree
execution artifacts with provenance, not replacements for the manuscript archive.
Reproduce with the committed `verification_hydrostatic_water.yaml` and
`verification_slump.yaml` examples in fresh output directories.

Free-surface pressure accuracy, contact range, pellet-surface coupling, energy
closure and measured validation remain open. The `9a66a55` manuscript equations,
frozen report and PDF are unchanged; its generated-data check still passes.


## Browser replay inspection

The actual household (48 pellets, 1 s) and research slump (512 points, 0.5 s) runs
were rendered in Chromium 153 / WebGPU, not merely fetched over HTTP. Initial
states, moving samples, inventory charts, event/warning tabs and final summaries
were inspected; numeric-time scrubbing reached both final states. Screenshots and
exact reproduction instructions are in `replay.md`. No JavaScript page errors were
captured. Safari/Firefox and resumed multi-segment browser rendering remain uninspected.

The prior auto-layout reduced the scene to a small tile surrounded by final-value
plots. New recordings use a geometry-bounded Z-up scene, explicit ledger selections,
compartment series names/colors and a visible fidelity/proxy guide. Final observables
remain logged numerically, but the default Summary tab presents them as segment-final
samples, timestamped at the segment end. Existing artifacts are not rewritten.

The visual probe also exposed a real lifecycle bug: SIGTERM orphaned the separate
Rerun process group. The CLI now installs both termination handlers before startup,
stops the child in cleanup and restores previous handlers. Separate-process SIGTERM
checks during startup and while serving both exited 0 and closed HTTP/gRPC ports.
No browser tooling was added to runtime dependencies and no numerical law changed.

Replay milestone gate: 105 Rust tests and 158 Python tests passed, 95% Python
coverage; formatting, Clippy, type checks, Rust docs and schema checks clean.
Installed-package tests also passed all 158 cases on Python 3.12 and 3.14.
The historical formal manuscript and frozen evidence remain unchanged.

## Normal wall-transfer energy accounting

`wall_normal_traction_energy_j` records the actual face-normal deposit's signed grid
KE change before gravity, not physical external wall work. Per-deposit accumulation
is checked against independent full-grid KE differences, including positive,
negative, separating, shared-node/corner and coupled cases. Massless deposits do
nothing. Rejected trials preserve histories; nonfinite increments and cumulative
overflow fail before commit. See `energy-ledgers.md` for the exact definition.

Actual CLI comparisons against `4a243b7`, using the shipped hydrostatic-water,
slump and new coupled-patch examples:

| fixture | new signed channel (J) | old residual (J) | new residual (J) |
|---|---:|---:|---:|
| Hydrostatic, 10 ms | 1.0219846729e-8 | 1.4356728261e-6 | 1.4254529794e-6 |
| Slump, 500 ms | -9.7731734973e-6 | 6.3996422367e-5 | 7.3769595864e-5 |
| Coupled pellet, 50 ms | -3.4963973432e-6 | -1.0620311823e-5 | -7.1239144793e-6 |

Canonical JSON comparisons found identical recorded particle rows, species metrics,
physical checkpoint state, previous ledger histories and every old observable
except `energy_residual_j`. The only excluded old checkpoint field was the
wall-clock-dependent remaining `request.max_wall_time_s`. The residual changes by
minus the new channel, up to roundoff. The slump residual **increases** in magnitude:
this is improved accounting, not reduced physical error or energy acceptance.
Local artifacts: `outputs/wall-energy-{before,after}-{hydro,slump,coupled}` and
`outputs/wall-energy-review/comparison.json`.

A real 50 ms native wall budget stopped the coupled run after 43 accepted steps
(2.15 ms simulated), with `-4.5405977455e-9 J` of nonzero traction history. After a
JSON checkpoint round trip, the resumed run matched every uninterrupted final
observable exactly. Removing the new history produced `missing field
wall_normal_traction_energy`; replacing it with null produced `invalid type: null,
expected f64`. Older checkpoints are rejected, not given fabricated zero history.
These runtime-dependent stop timings are observations, not timing assertions in CI.

The 8000-point coupled example completed 1000 steps and exports research-domain
replay geometry. It remains a synthetic, single-pellet fixture: no porous transport,
absorption, wet fragmentation, adhesion or accepted coupled refinement. No force
law or acceptance threshold changed. The historical manuscript/archive are untouched.

Milestone gate: **111 Rust tests and 159 Python tests**, no skips, 95% Python
coverage; formatting, Clippy, type checks, Rust docs and schema checks clean.
Installed-package tests passed all 159 cases on Python 3.12 and 3.14; the gate used
3.13. `docs/formal/build.py --check` also passed without rebaselining frozen evidence.
