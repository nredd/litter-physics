# Reduced response tables

`litter_physics.response_tables` supplies versioned, unit-labelled multilinear
interpolation over a complete Cartesian grid. It does not generate a physics model,
fit rheology, or establish that a measured result is correct.

Each axis has a unique name, units and strictly increasing finite knots. Each response
has explicit physical bounds and C-order sample values. Table digests cover values,
units, uncertainty notes and evidence. Unsupported dimensions, unknown fields,
nonfinite numbers, duplicate names and oversized grids fail before interpolation.

Default evaluation refuses out-of-domain inputs and absent/synthetic/incomplete
validation evidence. Explicit `exploratory=True` permits an extension of at most 10%
of an axis range. Outputs stay within declared physical bounds; every affected result
carries diagnostics and `validated=False`. Larger excursions fail even in exploratory
mode. There is no nearest-neighbor fallback or unlimited extrapolation.

Evidence records include the source commit and artifact SHA-256, species-ledger error,
separate spatial/temporal refinement changes, held-out surrogate error/repeatability,
and domain/near-zero checks. Thresholds implement the acceptance policy in `plan.md`.
These are evidence ATTESTATIONS: the interpolation library does not authenticate the
artifact or rerun experiments. The integrator must review the referenced artifact
before promotion. A forged or mistaken metadata record is not scientific proof.

Current state: the table library is tested but not connected to a verified research
solver or household constitutive rules. No calibrated physical response tables are
shipped. Unit tests use synthetic numbers, including mock acceptance metadata to
exercise policy branches; those fixtures must never become household calibration.

Schema: `schemas/response-table.schema.json`. The integration test checks that the
committed schema matches the model exactly.
