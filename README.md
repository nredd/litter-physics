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
uv run litter-physics run examples/research_coupled_patch.yaml --out outputs/coupled
uv run litter-physics view outputs/coupled --open-browser
uv run litter-physics calibrate --observations examples/observations_synthetic.json \
  --template examples/maintenance_smoke.yaml --out outputs/calibration.json
uv run litter-physics benchmark examples/maintenance_smoke.yaml --repetitions 3
uv run litter-physics verify examples/verification_slump.yaml --out outputs/refinement
```

The refinement example currently exits **1 (`unresolved`)**, not 0: spatial slump height, spread
and plastic dissipation have not met the 5% final-two-change criterion. Its complete
case artifacts and report remain available. See [verification studies](docs/verification.md).

The coupled example is one rigid pellet interacting with synthetic paste, not
absorption, fragmentation or a validated litter bed. Signed wall-transfer energy
is now recorded without changing trajectories; [energy closure remains open](docs/energy-ledgers.md).

Synthetic calibration never establishes measured parameters or supported household
personalization. Collect real observations using [docs/measurements.md](docs/measurements.md).

## Replay

New recordings open with a large Z-up simulation view, material/fidelity guide,
three inventory charts, events and segment-final summaries. Household and slump
playback have been inspected in Chromium/WebGPU. These are particle/proxy views,
not photorealistic cats or reconstructed fluid surfaces.
[Controls, screenshots and limitations](docs/replay.md).

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
- Separate spatial/time refinement studies with per-case artifacts, total budgeting,
  adaptive-step diagnostics and fail-closed acceptance of named observables.
- Response-table validation/interpolation library, not yet connected to the native model.

## Not implemented or accepted

- Research porous fines, wet fragmentation, adhesion and resolved liquid/pellet absorption.
- Validated research-to-household response tables or realistic household paste transport.
- Adaptive between-visit evolution for multi-day runs, actual box entrances/pads, and
  per-cat stroke trajectories connected to native events.
- Accepted convergence studies, measured surrogate validation, your cats' calibration, or
  an actual-size-box runtime acceptance study.

Outputs carry these limitations. Small demo timings must not be extrapolated to a
full box. See [implementation status](docs/status.md) and [verification evidence](docs/validation.md).

## Formal model

The [LaTeX manuscript](docs/formal/simulation.tex) and [PDF](docs/formal/simulation.pdf)
derive the implemented mechanics, constitutive laws, transfers and ledgers, with worked
calculations, vector figures, actual refinement plots and equation-to-code mappings.
The scientific baseline is pinned to `9a66a55`; unresolved accuracy and planned physics
are explicit. [Build instructions and evidence provenance](docs/formal/README.md).

```sh
brew install tectonic  # macOS; optional unless rebuilding the PDF
make manuscript
```

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
[research](docs/research.md), [wall-contact diagnosis](docs/hydrostatic-balance.md),
[calibration](docs/calibration.md),
[response tables](docs/response-tables.md), [wire contract](docs/wire-contract.md).
