# Preliminary household model

Every household run is labelled `preliminary_household`. It is a conservative baseline
for repeated events and maintenance, not the completed predictive model in `plan.md`.

Implemented state:
- Oriented intact pellets with dry mass, absorbed water and a moisture/damage state.
- A 5 mm field holding wood fines, deposit solids and mobile water.
- Separate bed/drawer/floor/removed/evaporated ledgers for wood, waste and water.
- Per-event progress, rigid contact history, deterministic RNG and full checkpoint state.

Water drains downwards, pools and redistributes overflow with capacity-limited transfers.
It is NOT a capillary or free-surface flow solution. Uptake, breakup, fines settling,
sifting and evaporation use explicit phenomenological rates. Deposit solids remain
stationary until scooped/cleaned: no household paste spreading, adhesion or waste/pellet
reaction is implemented. Slot fluxes for fines are placeholder laws, not fitted response
tables. Strong numerical conservation is necessary but cannot validate these laws.

Maintenance semantics:
- `scoop` removes a spherical region, including any clean litter caught there; `amount_kg`
  does not select a target scoop mass.
- `empty_drawer` removes current drawer contents only. The wet bed can immediately
  drain more material into it. Empty is an event, not a permanent zero-content state.
- `refill` adds full reference pellets and books the fractional-pellet mass remainder
  as fines. This conserves mass but is not a measured pellet-size distribution.
- `clean` accounts for removed bed and drawer contents instead of silently resetting them.
- `dig`, `cover`, and `stir` share the compliant sphere proxy described in `dem.md`.

Native caps: 10,000 pellets including refills across all boxes, 2,000,000 field cells,
10,000 events, 10,000,000 nominal steps, 1,000 DEM substeps per nominal step, 100,000
frames and 20,000,000 recorded sphere positions. Some Python schema caps are broader;
valid schema syntax does not guarantee a request fits native feasibility limits.

Frames are world-space coordinates; event positions remain box-local. Restart requests
must match except for wall budget, field geometry cannot be changed in the checkpoint,
and Python also checks build/dependency/calibration provenance. A budget stop is not a
completed simulation. Artifact finalization can exceed its reserve; such a run is
labelled `deadline_exceeded`, not quietly reported successful.

Outstanding: pre-settled beds, box entrances/enclosures/pads, research-derived constitutive
rules, physical moisture/fracture calibration, actual cat motion profiles, spatial
exposure/burial metrics and adaptive idle-time integration. Multi-day scenarios still
use fine stepping and may exceed resource caps; no multi-day runtime claim is made.
