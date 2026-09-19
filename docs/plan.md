# Litter Physics: implementation contract

Build a local two-scale simulator of individual cats using two independently stateful
pine-pellet sifting boxes. Research fixtures resolve small material interactions;
household runs use conservative reduced rules. The primary physical validation target
is the entire maintenance cycle, not visual plausibility.

## Scope and claims

- Model enter/inspect/dig/deposit/cover/exit, including aborted visits, no-cover events,
  wall hits, and escaped material. Use paw/body/source proxies, not animal biomechanics.
- Include urine, firm/soft/liquid stool, moisture-driven pellet breakup, fines, slots,
  drawer contents, optional pads, scoop/stir/empty/refill/full-clean maintenance.
- Research-grade viscoplastic flow and two-way pellet coupling are required deliverables,
  not features that may be replaced silently with one-way or purely decorative physics.
- Use clean surrogates for validation. Actual cat-waste material parameters stay uncertain.
- Run locally on CPU with Python CLI and a loopback-only Rerun browser viewer. No mandatory
  GPU/cloud. Each research fixture or detailed visit has a one-hour wall-clock ceiling,
  including initialization/output but excluding one-time compilation. Budget expiry is
  resumable and explicitly incomplete. Multi-job scenarios do not evade per-job gates.
- Research scenes are independent fixtures, not hidden moving high-resolution regions
  inside a household run. A full-box directly resolved research run under an hour is
  not promised. Rendering decimation is separate from numerical resolution.

## Architecture

Rust owns numerical state, stepping, conservative ledgers and full checkpoints. Python
owns Pydantic configuration, observation import, calibration, sweeps, provenance,
artifacts and CLI. Bind bulk operations with PyO3/maturin, not individual particle calls.
Use deterministic sorting/fixed-order reductions with Rayon where parallelism helps.
Use current compatible stable versions of uv, NumPy/SciPy, Pydantic, Rerun and Rust.

Data flow: measurements/observations -> calibration bundle -> seeded scenario -> state,
events, metrics, replay. Research produces versioned reduced-model tables with units,
input ranges, uncertainty and independent verification/validation evidence. Refuse
out-of-domain table inputs by default. Explicit exploratory extrapolation must be
bounded and permanently mark affected results unvalidated.

Separate core/scenarios/behavior/calibration/reporting responsibilities without building
a general-purpose physics platform. Expose validate/calibrate/run/sweep/benchmark/view
CLI operations. Export JSON Schema from validated YAML models. Use SI internally.

## Geometry and materials

Actual measured inputs: inner dimensions, entrance/walls, slot width/length/pitch, plate
thickness, drawer clearance, enclosure/pad use, pellet size/mass distributions, fill
mass/depth and initial moisture. Advertised Sorstrem dimensions mix variants and are
provisional, not measured interior geometry. Cat count, identities and habits are
unknown inputs, not assumed household facts. Clearly label synthetic demos.

Intact pellets are oriented rigid multispheres fitted to cylindrical shape and measured
mass/inertia. Use spring-dashpot normal contact, tangential friction and rolling
resistance. Validate shape near slots; never substitute equal-volume spheres merely to
meet runtime. Degrading pellets retain identity, dry mass, absorbed water, swelling,
strength and damage. Fines use a porous continuum in research and coarse fields in
household runs. Deposits retain separate solids and mobile/bound water.

Track wood solids, deposit solids and water in bed/drawer/pad/floor/removed/evaporated
compartments. Transformations must conserve species, including fragments below particle
thresholds. Prescribed boundaries contribute logged external work.

## Research numerical model

Use MLS-MPM with conservative two-way rigid/DEM coupling, sparse active blocks, sorted
particles and timestep control from continuum CFL and DEM contact stability.

- Firm through liquid stool: elastoviscoplastic Herschel-Bulkley material, local implicit
  constitutive update; yield stress, consistency, exponent and elastic stiffness are
  measured or explicitly uncertain, with documented moisture dependence.
- Water: weakly compressible liquid MPM with density and artificial-wave-speed sensitivity
  checks. No claim of resolved atomization or fine splashes.
- Fines: Drucker-Prager porous elastoplastic material, separate liquid/solid momentum
  fields with conservative drag.
- Wetting: capillary transport and capacity-limited uptake. Breakup depends on moisture,
  exposure time and mechanical damage; allow passive and disturbance-driven breakup.
- Adhesion: bounded traction-separation laws, measured against clean paste/steel/wood,
  not inferred from viscosity.
- Coupling: equal-and-opposite forces/torques, no double counting drag and solid contact.
- Fragmentation: conserve dry mass/water/momentum and record fracture dissipation;
  swelling changes occupied volume, not material mass.

Failed constitutive solves reject the step, retry with smaller dt, and abort with
state/diagnostics at the minimum. Start fixtures near 1 mm spacing but require spatial,
temporal and domain-size refinement. A few cells across a slot do not prove pore-scale
resolution. No research-derived table can be promoted before independent solver tests.

## Household model

Resolve intact pellets and compliant paw contact. Use a conservative 3D finite-volume
field (initially 5 mm) for fines, water and deposits, porosity from pellet occupancy,
and refinement checks for spatial metrics. Gravity/capillary transport, absorption,
overflow and drawer pooling stay explicit. Paste spread/yield/adhesion and reactions
use calibrated research response tables; label reduced forces honestly.

Slots are explicit pellet boundaries; fines/paste crossing uses calibrated flux laws
conditioned on size, moisture, blockage and disturbance. Stir follows a tool path.
Scoop uses a selected removal region with explicit clean-litter loss. Empty drawer
without resetting bed; refill adds measured species; clean removes accounted material.

Between visits settle mechanics and advance drying/redistribution/breakup adaptively.
Evaporation is a ledger sink. Never freeze flowing liquid or creeping paste because it
is briefly slow. Changed support/strength triggers re-equilibration. Restrict floor
tracking to a bounded local region; exclude whole-home tracking, odor chemistry,
microbes, ammonia and health predictions.

## Cats and calibration

Use semi-Markov visit states: approach -> enter -> inspect -> dig -> posture -> eliminate
-> cover/abandon -> exit. Permit skipped/repeated stages. Profiles encode observable
body/paw dimensions, stance/source position, entry/exit, location/orientation, stroke
counts/durations/directions/depth/speed, deposit scenarios, box choice and visit timing.
Use compliant force-limited paw trajectories; motion alone cannot identify paw force.

Replay annotated visits for primary physical validation. Generate from per-cat empirical
profiles and independent seed streams; occupancy exclusion first, cleanliness effects
only with evidence. No RL, inferred emotions or invented developmental laws.

Measure dry pour/repose, uptake/swelling, repeatable disturbance/breakup, sifted mass,
surrogate slump/spread/multi-rate flow/adhesion, event and maintenance logs, top-up,
scooped/drawer/floor masses. Slump alone does not identify every rheology parameter.
Prefer 20-40 visits/cat, accept less with weaker evidence; hold out contiguous maintenance
periods. Every parameter has units/provenance/uncertainty and measured/literature_prior/
assumed status. Separate numerical verification, surrogate validation and household
calibration claims. Sensitivity envelopes are not confidence intervals.

## Artifacts and interfaces

Each run stores frozen configuration, schema/version/seeds/code/build/dependencies/
hardware/calibration hashes/fidelity, full checkpoints (contact/material history, events,
ledger and RNG), structured events, Parquet metrics and Rerun recordings. Recordings
are not restart files. Include explicit completion and extrapolation/failure diagnostics.
Viewer: boxes/drawer cutaway, paw/source proxies, material/moisture colors, event markers,
accumulation/moisture/clean-loss/exposure/top-up plots. Burial means standardized top-view
surface exposure, never odor reduction. Bind loopback only.

## Acceptance

Test DEM restitution/friction/repose/packing/neighbors/slot orientation; hydrostatics,
dam break, yield-stress channel/slump, frictional column collapse; coupled momentum,
torque, work, leakage, adhesion and drag; absorption/overflow/breakup/drying/maintenance;
checkpoint equivalence, seeds, incompatible state rejection; idle vs fine-step references
including creep and drying collapse.

- Species residual <= 1e-6 relative to supplied species, with absolute near-zero tolerance.
- Target observable changes <5% across final two spatial refinements and separately
  temporal refinements; otherwise mark unresolved.
- Held-out material/household mass errors <=20% or measured repeatability band, whichever
  is larger; use absolute instrument uncertainty near zero.
- Per-cat distributions must beat generic baseline on held-out scores before claims of
  supported personalization.
- Representative accepted research fixtures and reduced visits complete within one hour
  on measured hardware. If shrinking domain/time cannot retain valid boundaries and
  convergence, report a blocker rather than coarsen/disconnect coupling to pass.

Full gate: rustfmt, pedantic Clippy, Rust tests/docs, Ruff, ty, pytest, schema and CLI/
browser integration tests. Run real inputs/failure paths. Synthetic data cannot satisfy
measured validation. Run independent adversarial review before acceptance.

## Delivery and collaboration

Workspace ~/code/litter-physics, private nredd/litter-physics. Claude Fable 5.1 subagents
own bounded implementation and independent reviews in isolated worktrees. Main agent
owns shared contracts/integration/gates, inspects real diffs, and documents/commits/pushes
each milestone. Disclose unavailable models rather than silently substitute them.

Milestones: foundation/feasibility -> dry mechanics -> preliminary maintenance baseline
-> research multiphysics -> validated reduced tables -> personalization/held-out studies.
The baseline is not project completion. Commit docs alongside behavior and keep videos,
large outputs, measurements and credentials out of git. Report blocked deliverables.

References:
- https://doi.org/10.1145/3197517.3201293 (MLS-MPM/rigid coupling)
- https://doi.org/10.1145/2751541 (Herschel-Bulkley MPM)
- https://doi.org/10.1145/3072959.3073651 (porous multiphase MPM)
- https://rerun.io/docs/reference/sdk/operating-modes (viewer operation)

These establish methods, not cat-specific material parameters.
