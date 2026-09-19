# Python CLI and artifacts

`litter-physics` orchestrates the native solver. Python validates, freezes, budgets,
stores, and replays; it never steps particles. Until integration registers
`_core.run_json`, `run`, `sweep`, and `benchmark` exit 1 with `UNAVAILABLE`; every
other command works today.

Exit codes: `0` success, `1` failure, `2` usage, `3` incomplete (budget or deadline
stop, resumable). An incomplete run is never reported as success.

## Commands

`validate PATH... [--kind request|observations|bundle|sweep|profiles]`
Strict validation of YAML/JSON documents. Unknown fields, non-finite numbers,
string-to-number coercion, and every structural rule in `docs/wire-contract.md` fail
with a message naming the field. Examples: `examples/household_basic.yaml` passes,
`examples/invalid_event_order.yaml` fails on event ordering.

`schema OUT_DIR [--check]`
Export JSON Schema (draft 2020-12) for request, output, checkpoint, observations,
calibration bundle, cat profile, and sweep spec. `--check` fails when the committed
`schemas/` differ from the models; `tests/test_schemas.py` enforces this, so edit a
model and re-run `litter-physics schema schemas`.

`run REQUEST --out DIR [--wall-budget S] [--calibration BUNDLE] [--no-recording]`
`run --resume --out DIR [--wall-budget S]`
Freeze the request into `DIR/request.json`, execute one native segment under a hard
process deadline, commit artifacts, and print a JSON result. `--resume` continues from
the latest checkpoint and appends a new segment. `--wall-budget` overrides the
request's `max_wall_time_s` for this segment only (still capped at 3600 s).
`--calibration` applies a bundle's material parameters and records its hash in
provenance; a synthetic bundle logs a warning.

`sweep SPEC --out DIR [--total-wall-budget S] [--recording]`
Grid product of dotted-path overrides times seed offsets, each job a normal run
directory under `DIR/jobs/NNNN` with its own budget. `--total-wall-budget` only stops
STARTING jobs (reported `not_started`); it never trims a running job's budget. Every
job request is validated before the first one starts. Summary in `sweep_summary.json`
and `sweep_summary.parquet`. Exit 3 unless every job completed.

`benchmark REQUEST [--repetitions N] [--report PATH]`
Repeat a request in throwaway directories; report median wall/native time, simulated
seconds per wall second, and the host description. Host timings only, not a fidelity
claim.

`view RUN_DIR [--grpc-port P] [--web-port P] [--open-browser] [--duration S]`
Serve committed `.rrd` recordings to the Rerun web viewer on `127.0.0.1` only. Uses
the `rerun` CLI bundled with `rerun-sdk` with `--bind 127.0.0.1`; the Python
`serve_grpc`/`serve_web_viewer` API binds all interfaces (verified with `lsof` on the
development host) and is deliberately not used. After start, the command probes every
non-loopback address of the host and refuses to continue if any answers. Runs until
Ctrl-C or `--duration`.

`calibrate --observations FILE --template REQUEST --out BUNDLE [--run DIR] [--report PATH]`
Fit identifiable parameters and write a bundle plus text report. See
`docs/calibration.md` for what is and is not identified. `--run` compares the run's
final `removed` ledger mass against weighed maintenance removals.

`observations synthesize --out FILE [--seed N] [--cats N] [--visits N] [--box-ids ...]`
`observations summary FILE`
Write a clearly labelled SYNTHETIC observation set with closed-form material curves
(for pipeline exercise only), or summarize an observation file.

`visits --template REQUEST --horizon S --out REQUEST [--seed N] [--log PATH]
        (--profiles FILE | --observations FILE | --synthetic-cats N)`
Generate per-cat semi-Markov visits with independent seed streams and write a
household request whose `events` are the generated strokes and deposits. `--log`
stores the visit records (stage path, box, wait time, elimination).

## Run directory layout

```
DIR/
  request.json               frozen request, written once, never rewritten
  manifest.json              the only mutable file; replaced atomically
  segments/0000/output.json  status, fidelity, observables, diagnostics
  segments/0000/frames.parquet   long table: frame_index, time_s, particle_index, x/y/z, radius, material, box_id
  segments/0000/metrics.parquet  time_s, compartment, wood_kg, waste_kg, water_kg
  segments/0000/checkpoint.json  native checkpoint embedding the request
  segments/0000/provenance.json  request/code/native hashes, git, deps, hardware, calibration hash
  segments/0000/recording.rrd    Rerun recording for this segment only
  segments/0001/...              appended by each resume
```

Guarantees:

- Every file is written to a temp name and `os.replace`d; segment files are never
  overwritten and a stale uncommitted segment directory blocks resume until removed
- Resume compares the frozen request with the checkpoint's embedded request
  (ignoring only `max_wall_time_s`) and rejects mismatches, unknown checkpoint
  versions, or non-finite state before calling the solver
- Frames from a resumed segment are a suffix; earlier segments stay byte-identical
- The manifest carries `status`, `complete`, `time_s`, cumulative wall time, and one
  record per segment. `segments/NNNN/output.json` keeps the status at commit time; the
  manifest is authoritative

## Budgets and deadlines

The per-job budget is `max_wall_time_s` (or `--wall-budget`), at most 3600 s. It covers
setup, the native call, and artifact construction. The solver receives the remaining
budget minus a reserve (`max(1 s, 5%)`). If setup already ate the budget the solver is
not called (`BudgetError`). If the segment's total wall time exceeds the budget, its
status is `deadline_exceeded` even when the solver reported `completed`; the
checkpoint is kept so `--resume` can finish it. Native `budget_exhausted` maps to
`budget_exhausted`. Both are exit 3 and resumable.

## Limitations

- No native solver is registered yet; `tests/test_native_integration.py` skips with
  an explicit reason and fails under `pytest --require-native`. Fake-core tests are
  not acceptance evidence
- Sweeps run sequentially; there is no job-level parallelism
- Recordings are derived views, never restart inputs; `rerun rrd verify` checks them
- Loopback verification probes the host's non-loopback IPv4 addresses only
- Coordinates in recordings are box origin plus box-local positions; there is no
  rotation support in `BoxCfg`

Integration guarantees:
- `make gate` rebuilds the current native sources and requires real-kernel integration
  tests. Unsupported research materials fail explicitly.
- Frame coordinates already include box origins. Restart recordings share one run ID
  and timeline; separate runs have separate IDs. Future events are not logged as if
  they occurred in an incomplete segment.
- The bundled Rerun executable is launched directly, not through its Python launcher.
  Stopping the viewer terminates its process group and closes both ports. Output is
  not left in an unread pipe that could block a long-lived server.
- Resume verifies frozen-request and provenance hashes and refuses changed native/Python
  builds, dependencies or calibration identity. Ordinary documentation-only commits do
  not invalidate unchanged numerical binaries and package code.
- New artifact publication is exclusive and atomic, including competing writers.
- Configuration files are limited to 32 MiB. Repeated JSON/YAML keys, non-string YAML
  mapping keys and YAML aliases are rejected rather than silently overriding inputs
  or expanding unbounded structures.
- Schema validation is not physical validation or a promise that all native resource
  caps/fixture-specific constraints will pass.

The advertised browser URL includes an encoded `url` query parameter selecting the
loopback recording server. Opening only the bare HTTP root serves the viewer shell,
not the selected recording. Keep the full link printed by `view`, including its query.
