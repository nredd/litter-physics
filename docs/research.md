# Research fixtures

`core/src/mpm/` contains deterministic CPU MLS-MPM/APIC kernels. `core/src/research.rs`
validates requests, runs fixtures with a wall budget, records observations and checkpoints
all non-scratch state. All results remain `research_unvalidated`.

Implemented:
- Hencky-elastic, J2 Herschel-Bulkley paste with a bracketed implicit scalar return.
  Wire parameters are shear-rheometer values. The kernel converts shear yield stress
  and consistency to equivalent-stress units; these conventions must not be mixed.
- Weakly compressible Newtonian water with an artificial bulk modulus derived from
  `young_modulus_pa` and `poisson_ratio`. Water requires zero yield stress and unit
  flow index. Study density error and artificial-wave-speed sensitivity before use.
- Quadratic B-spline transfers, APIC affine state, gravity/wall impulses and adaptive
  CFL/contact/viscous limits. Failed steps reduce dt; failure below the minimum aborts.
- Face-normal lattice completion at active domain-wall contacts, with no tensile
  image reactions and Coulomb budgets for both images and normal projections.
  This is a nodal contact surrogate, not an exact continuum boundary integral.
  Its derivation, regressions and limitations are in `hydrostatic-balance.md`.
- A two-way single-oriented-pellet coupling fixture with equal/opposite linear and
  angular impulses. The grid impulse test is NOT a full leakage/convergence validation.
- `slump`, `hydrostatic`, `dam_break`, and `coupled_patch` fixtures. Coupled patches require
  exactly one geometry box and paste. Other fixtures reject geometry boxes.

`fines` is explicitly rejected. Porous multiphase flow, absorption, wet fragmentation,
adhesion, multiple interacting research pellets and research-response-table generation
are not implemented. Paste is currently a single surrogate phase whose mass is booked
to `waste_kg`, without a separate mobile/bound-water inventory.

Geometry must fit a grid-aligned domain and whole particle subcells. Grid storage is
DENSE with active-node updates, not sparse active blocks. Native limits are 100,000
continuum points, 1,000,000 padded grid nodes and 2,000,000 recorded point samples.
Coupled pellet radius must span at least 1.5 cells, which is only an under-resolution
rejection rule, not proof of adequate spatial resolution.

The checkpoint retains particle identities, position/velocity, APIC and elastic history,
Kirchhoff stress, plastic strain, volumes/masses, pellet pose/momentum/geometry, ledgers,
adaptive-step history and recording index. Scratch grids are rebuilt. Fixture placement
uses the request seed; no further random draws occur after initialization. JSON parsing
uses round-trip float handling, tested against uninterrupted trajectories.

Current verification includes constitutive residuals, steady shear, compression,
hydrostatic initialization, transfer consistency, free fall, wall/gravity bookkeeping,
rigid impulse balance, restart identity and failure paths. A block-slump comparison to
a cylinder-slump formula is only a sanity reference because the shapes differ.

Not accepted yet:
- End-to-end grid/time/domain convergence studies, measured clean-surrogate validation,
  or actual cat-stool properties.
- Yield-stress channel-flow and frictional-fines collapse acceptance.
- General two-way coupled momentum/work/leakage accuracy at refined resolutions.
- Actual-size-box or multi-day runtime guarantees.

References:
- MLS-MPM and rigid coupling: [here](https://doi.org/10.1145/3197517.3201293)
- Herschel-Bulkley MPM: [here](https://doi.org/10.1145/2751541)
- APIC: [here](https://doi.org/10.1145/2766996)
- Future porous mixtures: [here](https://doi.org/10.1145/3072959.3073651)

### Timestep evidence

`limited_steps` counts accepted steps whose stability/recovery limit was below the
requested `dt_s`, evaluated BEFORE clipping to recording or final-time boundaries.
Endpoint clipping alone does not increment it. The count is conservative: a limiter
may be below the requested cap even when an endpoint clips the step further.

`max_dt_s == dt_s` only establishes that the cap was reached at least once; it does
not establish that adaptive limiting stayed inactive afterward. A separated fixed-
timestep refinement study must reject nonzero `limited_steps` or `rejected_steps`.
The counter is preserved in checkpoints and cannot exceed accepted `step_count`.
Checkpoints lacking this history are rejected, not assigned a fabricated zero;
Python also refuses restart across changed numerical builds.

### Grid and restart integrity

Zero-weight stencil entries do not register active nodes. Previously an exact half-cell
particle could register a zero-mass node, then another particle registered it again;
wall impulses were counted twice. `core/tests/grid_transfer.rs` reproduces the exact
alignment and its perturbed control and checks unique nodes and momentum residual
below 1e-18 kg m/s.

All current wire fixtures are closed. Restored particles must have valid grid stencils,
even when resuming a checkpoint already at its final time. At record boundaries and
budget stops, the full particle/grid state is validated and nonzero outflow inventory
is rejected. A balanced ledger does not make a wall leak acceptable. This guard catches
escape beyond the padded grid; it does NOT establish watertight subcell boundaries.

The first independent review also reported nonconvergent maximum hydrostatic errors
at the bottom boundary and per-particle liquid pressure noise insensitive to timestep
refinement. Time-evolved all-particle tests now exercise the wall correction, including
matched grid/time controls and a 0.1-second hold. The original maximum error
drops from 29.7% to 1.15% at 10 ms, but the separate refinement study still
reports `unresolved`. No pressure smoothing, interior-only metric substitution,
or material-law change has been used; liquid-pressure and full energy acceptance remain
open. The former `wall_work_j` ledger was the grid projection kinetic-energy loss, not
physical work; it is now split into `wall_normal_projection_energy_j` and
`wall_friction_dissipation_j` with coupling energy terms and an algebraic
`energy_residual_j`, none of which is an energy acceptance gate. See `energy-ledgers.md`.
