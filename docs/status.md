# Implementation status

This is the first executable milestone, NOT completion of `plan.md`.

Implemented and exercised:
- Private repository, native extension, locked dependencies, shared wire contract,
  strict schemas and non-mutating lint/type/test gates.
- Dry oriented-pellet mechanics and conservative preliminary maintenance runs.
- Single-phase paste/liquid MPM primitives and a two-way single-pellet fixture.
- CLI, JSON checkpoints, immutable artifact segments, Parquet, loopback browser serving,
  recording verification, build-compatible resume and explicit budget/deadline outcomes.
- Scene-first Rerun layouts with explicit Z-up geometry, research-domain bounds,
  inventory charts, model/material guide and timestamped segment summaries. Real
  household/slump browser playback and scrubbing inspected (`replay.md`); no cat
  mesh, fluid-surface reconstruction or pressure/velocity coloring yet.
- Synthetic observation import, identifiable curve fits, duration-only personalization
  scoring, visit-stage generation, sweeps and benchmarks.
- Bounded, separately varied spatial/time refinement studies using the real research
  kernel. The committed slump example completes but remains `unresolved`: spatial
  slump height, spread and plastic dissipation have not met the 5% final-two-change criterion.
  Missing diagnostics, adaptive limiting, incomplete cases and deadline overruns
  cannot pass. This facility does not establish measured validation.
- Face-normal wall lattice completion with unilateral activation and a Coulomb
  budget for both image and projection reactions. Native all-particle pressure
  regressions and release/slip/stick checks pass (`hydrostatic-balance.md`).
- Standalone response-table validation/interpolation with bounded exploration and
  fail-closed evidence policy. No physical response tables are shipped.

Required work still outstanding:
- Complete research porous fines, absorption, wet breakup and adhesion with independent
  verification, refinement studies and coupled momentum/work/leakage acceptance.
- Generate and validate research response tables; consume them in the native household
  model. Current household paste does not spread or push pellets.
- Add calibrated geometry (entrance/enclosure/pads), settled beds, actual stroke replay,
  behavior-to-native event generation, exposure/burial metrics and adaptive idle evolution.
- Validate material/surrogate measurements, held-out maintenance cycles, per-cat profiles,
  numerical convergence and actual-size-box runtime on the intended hardware.

Input blockers: actual box/pellet measurements, cats' identities/habits/videos,
maintenance logs and clean-surrogate experiments. See `measurements.md`.

Runtime results for the tiny synthetic demos are recorded in `validation.md`. They are
not representative full-box benchmarks. Native feasibility caps remain deliberately
explicit. The first independent MPM review found active-grid bookkeeping and
checkpoint-domain gaps and misleading energy-ledger semantics, now regression-tested
(`energy-ledgers.md`); energy acceptance remains open. The hydrostatic wall correction reduces the original all-particle maximum
pressure error from 29.7% to 1.15% at 10 ms, and stays at 1.27% at 100 ms.
This is a bounded numerical improvement, not general liquid-pressure acceptance:
free-surface error, contact range and both hydrostatic/slump refinement studies
remain unresolved. The signed grid KE jump from normal wall-image deposits is
now ledgered before gravity, with accepted-step accumulation and required restart
history. This is not physical wall work or energy closure. Hydrostatic, slump and
single-pellet CLI comparisons retain identical physical state and recorded frames;
the new diagnostic changes only its channel and the algebraic energy residual.
`examples/research_coupled_patch.yaml` makes the existing single-pellet/paste fixture
runnable directly. A bounded review and conservation tests do not establish
research-grade accuracy.

The new nonmutating substep audit (`energy-audit.md`) measures APIC affine energy,
transfer losses, stress/storage mismatch and gravity/coupling stages without changing
normal runs or checkpoints. Sixty bounded measurements reproduce the analytic
free-fall integration defect and expose a coupling failure: a light-body probe
creates `0.5223 J` during the grid/body constraint, independent of the sampled
timestep. This is a real unresolved energy-stability defect, not merely an absent
ledger channel. Finite-inertia coupling repair is the next priority; the audit's
algebraic telescope is not physical acceptance.

The formal manuscript (`formal/simulation.pdf`, scientific baseline `9a66a55`)
maps equations to implementation and preserves the unresolved refinement data.
Its derivation also makes model discrepancies explicit: household swelling leaves
sphere offsets fixed and deposits dry-reference occupancy; incoming-water momentum
is not resolved; and the fitted exponential breakup fraction is not the runtime's
accumulated-damage threshold law. None is repaired by documenting it. The isolated
plastic-return energy gap is derived and tested, not mistaken for full energy closure.
