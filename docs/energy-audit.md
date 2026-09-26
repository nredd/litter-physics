# Discrete energy audit

This audit is a diagnostic of the implemented step, not a physical energy-closure
certificate. The existing `mechanical_energy_j` and `energy_residual_j` retain their
meaning; no missing work is manufactured by defining it as the residual.

## Run the bounded probe

```sh
mkdir -p outputs
CARGO_BUILD_JOBS=2 cargo run --release -p litter-physics-core --example energy_audit \
  > outputs/energy-audit.json
```

The JSON contains 60 measurements: ten synthetic cases, two snapshots each, and
three requested timesteps from each identical snapshot. Cases cover rest,
translation, affine motion, opposed velocities, free fall, prestress, a light-body
coupling counterexample, hydrostatic liquid, slump and a launched coupled pellet.
The 40/20/10 us triples are **local-step comparisons**, not global convergence.
The last three cases use a 4 mm grid and different parameters from the YAML demos.
The coupled pellet is deliberately launched downward at 2 m/s to exercise contact
within 6 ms; its initial momentum/energy histories are rebuilt after that launch.
The light-body case is an inertial-ratio stress test, not pine material calibration.

The example caps allocations at 4000 material points and 20000 grid nodes, and
checks a 3000-attempt integration budget before every trial, including rejected
trials. Exhaustion is an error, not a silently shortened successful report.

Rust API: `Simulation::audit_step(dt) -> Result<EnergyAudit, SolverError>`.
It measures one production trial on a clone, without retries or timestep clipping.
The original state, normal solver trajectories and checkpoint layout are unchanged.
Invalid state, escaped particles, inadmissible timesteps, rejected trials and
nonfinite measurements return errors. Extra particle/grid scans occur only when
this diagnostic is requested. This is a developer probe, not a new Python CLI mode
or automatic per-step recording feature.

The report separates `before`/`after` endpoint energy, `stages`, ledger-history
`increments`, and derived `terms`. All energy fields are joules. Optional gravity
fields distinguish uncoupled and coupled scope. Local plastic work is captured
before accumulation: a large old history can round away a small increment, so
`stages.plastic_dissipation` and the rounded history difference are both retained.

## Which kinetic energy?

The ordinary particle kinetic energy contains translational velocities. APIC also
carries the affine velocity matrix `C_p`. For the implemented quadratic spline with
a complete stencil, its second moment is `D_p = h^2 I / 4`. The corresponding
extended discrete kinetic norm is

```text
K_particle* = sum_p m_p |v_p|^2 / 2 + K_affine
K_affine    = sum_p m_p tr(C_p D_p C_p^T) / 2
            = h^2 / 8 sum_p m_p ||C_p||_F^2
```

This is a grid-spacing-dependent APIC norm, not a newly validated physical
subparticle inertia. It is useful for distinguishing transfer loss from energy
moved between the translational and affine representation. It does not silently
replace the historical mechanical-energy observable.

With positive stencil weights and complete support, pure APIC particle-to-grid
transfer is a mass-weighted average. Its energy loss has an independent variance
identity:

```text
K_particle* - K_grid,transport
  = sum_p,i m_p w_ip |v_p + C_p (x_i - x_p) - v_i,transport|^2 / 2
```

Grid-to-particle transfer is a weighted affine least-squares projection. Using the
pre-advection stencil and the newly transferred `v'_p`, `C'_p`:

```text
K_grid,final - K_particle,after*
  = sum_p,i m_p w_ip |v_i,final - v'_p - C'_p (x_i - x_p)|^2 / 2
```

These identities supply independent references for transfer tests. A telescoping
sum alone cannot detect a wrong energy definition. Negative roundoff in a measured
loss must not be clamped away or relabeled as a physical source.

## Separate the stages

The MLS implementation combines APIC transport and the stress impulse in one
scatter. A diagnostic-only stress-free shadow scatter supplies the transport
reference without changing that actual scatter. The kinetic chain then separates:

1. Particle translation plus affine norm before the step.
2. Pure APIC transport on the grid.
3. Actual MLS transport plus stress.
4. Face-normal wall-image impulses, before gravity.
5. Grid gravity update.
6. Wall normal projection and Coulomb friction.
7. Grid/pellet coupling.
8. Particle translation plus affine norm after transfer.

Elastic storage, gravitational potential, plastic dissipation and pellet kinetic
changes must be tracked alongside this chain. Summing adjacent kinetic differences
must reproduce the endpoint difference by algebra. A small telescope residual
checks accounting consistency, **not** the first law.

The combined stress/storage/plastic mismatch is diagnostic too: a grid stress
kick is not automatically the constitutive work conjugate to the actual strain
increment. Explicit force timing, the finite-strain update, plastic-return
integration and liquid viscosity can all contribute. Later gravity/wall/coupling
updates also influence the velocity gradient used by the constitutive update;
this is not a clean isolation of material-model error. It must not be booked as
invented material dissipation.

## An independently predictable defect

For an isolated, stress-free translating particle with uniform gravity, no walls
or pellet, the current update is

```text
v_new = v_old + g dt
x_new = x_old + v_new dt
Delta K_gravity + Delta U = -m |g|^2 dt^2 / 2
```

Thus halving the timestep quarters this *one-step* defect. Over a fixed duration
with constant steps, the accumulated defect is first order in timestep. This is
a numerical integration effect even with perfect transfers; it is not physical
dissipation and cannot be repaired by renaming a ledger channel.

## Measured local-step behavior

The evolved free-fall sample (1 g, 0.2 ms) gives gravity/potential defects of
`-7.698888e-11`, `-1.924722e-11`, and `-4.811805e-12 J` at 40/20/10 us,
matching the independent formula above.

At the 6 ms snapshots, the 40 us measurements include:

| case | pure P2G loss (J) | G2P loss (J) | stress/storage/plastic mismatch (J) |
|---|---:|---:|---:|
| Hydrostatic, 1536 points | 3.1322e-14 | 2.0923e-13 | 6.6030e-9 |
| Slump, 384 points | 4.1856e-10 | 7.3974e-10 | 1.9753e-9 |
| Launched pellet, 384 points | 1.2871e-6 | 1.5852e-6 | 1.2470e-7 |

Pure P2G loss is identical across each timestep triple: its input snapshot and
transport map are identical. Reducing timestep alone does not remove that transfer
loss. The launched coupled case also has joint grid/body coupling change
`-1.02684e-6 J` at 40 us. These observations neither establish spatial convergence
nor identify every term as physical dissipation.

## Retained coupling counterexample

The `light_body_coupling` case has one 2 kg stress-free material point moving
upward at 1 m/s and a stationary body with radius 0.15 m, length 0.3 m and density
1 kg/m^3. Grid spacing is 0.1 m. There is no gravity, friction, initial elastic
storage or wall contact. These are deliberate algorithmic stress-test parameters.

The first step reports:

```text
grid coupling change           -0.079904513889 J
pellet coupling change         +0.602205342991 J
joint coupling change          +0.522300829103 J
G2P transfer loss                0.047207353380 J
augmented endpoint energy gain +0.475093476868 J  (dt=10 us)
```

The joint coupling gain is identical at 40/20/10 us. This is **artificial energy
creation**, not merely missing bookkeeping. Transfer losses partly mask it.
A zero telescope discrepancy here does not make the dynamics acceptable.

`rigid.rs::couple_grid` constrains all nodes against the old body velocity, then
`solver.rs` applies their accumulated reaction to the finite-mass body. That split
can overshoot. In the scalar analogue, projecting fluid mass `m` to old body
velocity `V`, then kicking body mass `M`, changes joint kinetic energy by
`m (u-V)^2 (m/M-1) / 2`: positive when `m > M`.

The fixture remains runnable, but tests do not require energy creation to persist.
The next physics repair must account for finite translational and rotational body
inertia in the coupled constraints, with independent energy-passivity and impulse
checks. Clipping energy or excluding light bodies is not a repair. This milestone
measures and reports the defect; it does not fix it.

## Coupled and physical limits

The pellet's impulse update and subsequent integration are distinct stages. The
latter mixes gravity, wall contact and rotation. Without separately establishing
spring storage, damping/friction work and integration consistency, that combined
kinetic change must not be called gravity-only work or dissipation.

A physical closure claim still requires consistent stored energy, external work
and genuine dissipation, then independent space/time refinement of the resulting
balance. Numerical transfer losses and these stage mismatches identify where to
investigate; they are not an automatic acceptance budget. Measured surrogate and
household validation remain separate requirements.
