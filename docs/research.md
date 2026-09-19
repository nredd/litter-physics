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
