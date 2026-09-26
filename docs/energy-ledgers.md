# Energy ledgers

`core/src/mpm/solver.rs::Ledger` records discrete projection/coupling kinetic-energy
changes, normal wall-transfer grid energy, and contact-force/plastic-dissipation
estimates. These are diagnostics.
There is NO closed energy balance and no energy acceptance gate.
The [discrete energy audit](energy-audit.md) separates transfer, stress, gravity,
wall and coupling stages without changing these historical observables.

## What was wrong

The removed `wall_work` ledger (observable `wall_work_j`) accumulated
`dp_i . (v_i + v_i') / 2` per wall node, documented as "work done by the walls on the
material". That quantity is exactly the grid kinetic-energy change of the wall
projection, `m_i / 2 (|v_i'|^2 - |v_i|^2)`. It is NOT physical work: the walls are
stationary, so their external work is identically zero. For material at rest on a
wall, every step deposits the reacted gravity and stress momentum `W dt` on the wall
nodes and the projection removes it again, losing `W^2 dt^2 / (2 m_wall)` per step.
That loss is first order in `dt`, lives on the scratch grid and never reaches
particle state. The following measurements predate the wall lattice completion
and belong to the `9a66a55` energy-ledger baseline:

```
paste column at rest, 0.02 s, cap-limited steps (limited_steps = 0)
dt=4e-5  wall_normal_projection_energy=-1.67e-6 J
dt=2e-5  wall_normal_projection_energy=-0.83e-6 J   (ratio 2.0)
hydrostatic liquid 0.3 s: KE=2.9e-15 J, PE=1.96e-4 J, projection=-7.8e-5 J
```

Evaluating the impulse against the post-projection velocity instead would zero the
normal term and under-count sticking friction. It was not adopted.

## Definitions

Per wall node, in axis order, with `v_n` the approaching normal speed and `t`, `t'`
the tangential velocity before and after the Coulomb reduction:

- `wall_normal_projection_energy -= m v_n^2 / 2` (non-positive). Lumps inelastic
  wall impact with the `O(dt)` splitting loss above. Not physical work
- `wall_friction_dissipation += m (|t|^2 - |t'|^2) / 2` (non-negative, same sign
  convention as `plastic_dissipation`). Equals minus the friction impulse dotted with
  the midpoint tangential velocity. On the sliding-block probe it reproduces the lost
  kinetic energy to under 1% (`7.73e-6` vs `KE0 = 7.68e-6` J at rest)
- Normal projection energy MINUS friction dissipation equals the old midpoint
  value (up to roundoff), i.e. the projection/friction grid KE jump. The
  energy-ledger split itself did not change `wall_impulse`

The subsequent wall lattice completion (`hydrostatic-balance.md`) adds normal
reaction impulses to nearby free nodes before this grid update. Those impulses
are included in `wall_impulse`; their Coulomb reductions contribute to
`wall_friction_dissipation` by the same nonnegative KE-loss formula.

`wall_normal_traction_energy` now records the signed grid KE change of these
normal deposits. For each deposit, with signed normal impulse `dp`, existing
normal momentum `p`, and receiving-node mass `m`:

```text
Delta K = dp (p / m + dp / (2 m))
        = ((p + dp)^2 - p^2) / (2 m)
```

The increment avoids subtracting two large energies. Sequential deposits on the
same node telescope; the total equals a full before/after grid KE difference.
Massless nodes receive neither impulse nor energy. No extra whole-grid scan is
needed at runtime. Negative increments are retained: `m=2`, `p=-3`, `dp=2` gives
`-2 J`, not a positive dissipation channel.

The sampling point matters: momentum is read **after APIC/MLS transfer and before
gravity**, exactly where the impulse is applied. Using gravity-shifted momentum
instead would add `dt g . I_traction` to this channel. In a resting column, stress
forces can already give free nodes upward momentum before gravity, so the normal
reaction can add transient grid KE without doing physical external wall work.
The projection/friction identity above covers only that later substep, not the
whole transfer/gravity/wall sequence.

This accounting addition does not change force or trajectory calculations.
Nonfinite increments or accumulated traction history reject the trial before
particle/energy updates are committed; accepted steps alone accumulate history.

Pellet and coupling:

- `pellet_wall_work += F . v_contact dt` with the start-of-step penalty force and
  contact-point velocity. Includes recoverable spring energy, so it is neither
  dissipation nor external work, and is `O(dt)` inconsistent with the symplectic
  Euler kinetic-energy change
- `coupling_grid_energy += sum m_i / 2 (|v_i'|^2 - |v_i|^2)` over nodes constrained
  by `couple_grid`. Not sign-definite: the constraint targets the moving pellet
  surface and can add energy to the grid
- `coupling_pellet_energy += KE_pellet(after) - KE_pellet(before)` around
  `apply_impulse`, exact

`initial_mechanical_energy` is seeded after fixture placement, so a prestressed
hydrostatic column starts with its elastic energy counted.

## Residual

`energy_residual = E(t) - E(0) - ledgered_energy` with
`ledgered_energy = wall_normal_projection_energy - wall_friction_dissipation
+ wall_normal_traction_energy + pellet_wall_work + coupling_grid_energy
+ coupling_pellet_energy - plastic_dissipation`.

This is an algebraic mixed grid/particle diagnostic, NOT an unexplained-energy
closure. In the historical projection-only resting-column example, the positive
residual included scratch-grid loss absent from particle energy. Booking another
grid substep does not resolve that mismatch. APIC particle/grid transfer losses,
discrete gravity and constitutive integration consistency, pellet integration
error, contact spring energy and outflow energy remain unclosed. A larger or
smaller residual after adding the new channel does not imply changed physics.

## Observables

Removed: `wall_work_j`. Added: `wall_normal_projection_energy_j`,
`wall_friction_dissipation_j`, `wall_normal_traction_energy_j`, `energy_residual_j`,
and for `coupled_patch` `coupling_grid_energy_j`, `coupling_pellet_energy_j`.
`Ledger::validate` rejects nonfinite values and wrong projection/friction signs;
traction energy is deliberately signed. Checkpoints carrying the old `wall_work`
field are rejected by `deny_unknown_fields`.

The new traction history is required in native checkpoints. Missing/null history
is rejected, never defaulted to zero. Older recordings remain readable, but old
checkpoints cannot be resumed with this build. JSON restart preserves the new
history exactly; neither scratch transfers nor rejected steps may double-book it.

Restart validates `initial_mechanical_energy` against the deterministically rebuilt
initial state using exact float bits. JSON float round trips preserve that baseline;
forged energy offsets are rejected rather than silently shifting the diagnostic.
