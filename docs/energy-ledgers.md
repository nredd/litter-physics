# Energy ledgers

`core/src/mpm/solver.rs::Ledger` records discrete projection/coupling kinetic-energy
changes plus contact-force and plastic-dissipation estimates. These are diagnostics.
There is NO closed energy balance and no energy acceptance gate.

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
`wall_friction_dissipation` by the same nonnegative KE-loss formula. The KE
change from the added normal impulses is **not** included in the ledger. The
projection/friction identity above therefore does not cover the whole wall
treatment, and unchanged energy-channel names do not imply unchanged trajectories.

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
+ pellet_wall_work + coupling_grid_energy + coupling_pellet_energy - plastic_dissipation`.

This is an algebraic mixed grid/particle diagnostic. It is NOT an unexplained-energy
closure: the resting column reports a positive residual equal to the projection loss
because that energy was never in the particles. Also unledgered: APIC particle/grid
transfer losses, constitutive and pellet time-discretisation error, contact spring
energy, the normal lattice-completion impulse contribution, and the energy of
outflow particles.

## Observables

Removed: `wall_work_j`. Added: `wall_normal_projection_energy_j`,
`wall_friction_dissipation_j`, `energy_residual_j`, and for `coupled_patch`
`coupling_grid_energy_j`, `coupling_pellet_energy_j`. `Ledger::validate` rejects
nonfinite values and wrong signs; checkpoints carrying the old `wall_work` field are
rejected by `deny_unknown_fields`.

Restart validates `initial_mechanical_energy` against the deterministically rebuilt
initial state using exact float bits. JSON float round trips preserve that baseline;
forged energy offsets are rejected rather than silently shifting the diagnostic.
