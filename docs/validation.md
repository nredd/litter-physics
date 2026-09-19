# Verification evidence

This records executed checks, not intended acceptance. Nothing here establishes
cat-specific material properties or held-out household accuracy.

## Foundation and response tables

Environment: Apple M4 Mac mini, 16 GB RAM; Rust 1.98.1; Python 3.13.15.

- Native extension loads and exposes protocol version 1.
- Full `make gate` checks formatting, pedantic Clippy, Rust/Python types and tests,
  Rust documentation and committed JSON Schema agreement.
- Wheel/sdist installation and then-current tests also exercised Python 3.12.13 and
  3.14.7 using `uv run tox -e py312` and `uv run tox -e py314`.
- Response-table tests cover affine interpolation, evidence refusal, named inputs,
  nonfinite values, bounded exploration, constant physical bounds, allocation limits,
  schema drift and updated-copy cache/provenance invalidation.
- Independent Claude review found roundoff incorrectly marking valid interpolation
  unvalidated, repeated preparation overhead, and stale caches in Pydantic copies.
  Fixes use ULP-aware bound handling, a final fail-closed gate, cached preparation,
  fully revalidated copies and regression tests. No gate exceptions were added.

Synthetic interpolation microbenchmark on this Mac, 781,250 response values:

```text
cold_evaluation_s=0.079585
warm_evaluation_s=0.000104
```

This is Python table preparation/reference-evaluation throughput, NOT DEM/MPM or
household throughput. Python is not intended to evaluate every field cell per step;
a future native table consumer must preserve the same domain/evidence contract.

## Outstanding evidence

No measured geometry, pellet tests, cat observations, surrogate rheology, held-out
maintenance cycle, converged MPM study, or full simulation runtime acceptance has
been supplied or accepted yet. Numerical modules are undergoing separate integration.
