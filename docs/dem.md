# Dry mechanics

`core/src/dem.rs` represents each intact pellet as a rigid oriented multisphere body.
Mass and principal inertia come from a cylindrical fit, not from adding overlapping
sphere volumes. Sphere centres form the collision proxy; their radii/spacing approximate
rather than exactly reproduce a sharp-edged cylinder.

The solver uses spring-dashpot normal forces, history-dependent tangential friction,
rolling resistance, spatial neighbor search and stable mechanical substeps. Sifter
bars/slots, side walls and drawer floor are explicit boundaries. Slot passage depends
on shape AND orientation. Open-topped walls allow escaped material to be accounted for.

Paw/stir proxies have mass, bounded compliant drive and collision response. They enter
above the bed, then approach their target path; they do not teleport into buried
pellets. These are not measured paw forces or animal biomechanics. All three motion
kinds currently share one sphere/tool model with fixed speed and drive constants.

Tests exercise cylinder mass/shape, neighbor lookup against brute force, restitution,
sliding friction, inclined-plane motion, orientation-dependent slot passage, contact
history round trips, packed-bed settling and tool reaction/work.

Validation limits:
- Tests do not establish measured friction, repose angle or pellet packing accuracy.
- Contact overlap is reported and must be studied under stiffness/time refinement;
  a conserved mass ledger alone does not establish mechanically acceptable overlap.
- Initial beds use a seeded loose lattice, not a separately calibrated settled packing.
- Swollen pellet radius/inertia is a preliminary approximation; exact changing-shape
  inertia and fracture momentum/work require further validation.

Reference: [here](https://doi.org/10.1145/3197517.3201293) for the separate research
rigid/continuum coupling; the household contact solver is an independent DEM model.
