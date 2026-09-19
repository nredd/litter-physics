# Litter Physics

Local pine-pellet litter simulation: Rust mechanics and research fixtures, Python
experiments/calibration, and browser replay. Private repository: `nredd/litter-physics`.

**Working preliminary implementation, not a faithful household predictor yet.**
Examples use assumed geometry/materials and synthetic cats. Research fixtures are
single-phase and unvalidated; the reduced household model does not yet consume
research response tables. The complete design is in [docs/plan.md](docs/plan.md).

## Run

Requires Rust 1.98+, Python 3.12-3.14, and `uv`. The development host is an M4 Mac mini
with 16 GB RAM. No GPU or cloud account is required.

```sh
uv sync --locked
uv run litter-physics validate examples/maintenance_smoke.yaml
uv run litter-physics run examples/maintenance_smoke.yaml --out outputs/demo
uv run litter-physics view outputs/demo --open-browser
```

The viewer binds only `127.0.0.1`. Ctrl-C stops it. Existing run directories are never
overwritten. For an incomplete run:

```sh
uv run litter-physics run --resume --out outputs/demo
```

Research and calibration examples:

```sh
uv run litter-physics run examples/research_slump.yaml --out outputs/slump --wall-budget 30
uv run litter-physics calibrate --observations examples/observations_synthetic.json \
  --template examples/maintenance_smoke.yaml --out outputs/calibration.json
uv run litter-physics benchmark examples/maintenance_smoke.yaml --repetitions 3
```

Synthetic calibration never establishes measured parameters or supported household
personalization. Collect real observations using [docs/measurements.md](docs/measurements.md).

## Implemented

- Oriented multisphere pellets, frictional contacts, explicit slot geometry and compliant
  paw/stir proxies.
- Stateful two-box events, conservative wood/waste/water bookkeeping, preliminary
  uptake/breakup/sifting/drying, and scoop/stir/empty/refill/clean maintenance.
- Small MLS-MPM paste/liquid fixtures, implicit Herschel-Bulkley updates, adaptive
  timesteps, and a two-way single-pellet coupling fixture.
- Strict configuration/schema checks, bounded jobs, full native checkpoints, build-
  compatible resume, immutable segment artifacts, Parquet metrics and Rerun replay.
- Observation import, simple material fits, synthetic visit profiles, sweeps and benchmarks.
- Response-table validation/interpolation library, not yet connected to the native model.

## Not implemented or accepted

- Research porous fines, wet fragmentation, adhesion and resolved liquid/pellet absorption.
- Validated research-to-household response tables or realistic household paste transport.
- Adaptive between-visit evolution for multi-day runs, actual box entrances/pads, and
  per-cat stroke trajectories connected to native events.
- Full convergence studies, measured surrogate validation, your cats' calibration, or
  an actual-size-box runtime acceptance study.

Outputs carry these limitations. Small demo timings must not be extrapolated to a
full box. See [implementation status](docs/status.md) and [verification evidence](docs/validation.md).

## Develop

```sh
make gate
uv run tox -e py312,py313,py314
cargo bench --bench constitutive
```

The gate rebuilds the native extension, requires real-kernel integration tests, checks
Rust/Python formatting/lint/types, runs tests, builds Rust docs and validates schemas.
Release stripping is disabled for the macOS 27 LINKEDIT issue documented in
[docs/validation.md](docs/validation.md).

Keep measurements in ignored `measurements/`, recordings in `recordings/`, and generated
runs in `outputs/`. Document, test, commit and push each milestone. Claude work is
integrated only after checking the actual code and running the shared gate.

Details: [CLI](docs/cli.md), [mechanics](docs/dem.md), [household](docs/household.md),
[research](docs/research.md), [calibration](docs/calibration.md),
[response tables](docs/response-tables.md), [wire contract](docs/wire-contract.md).
