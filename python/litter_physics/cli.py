"""Command-line entry point.

Exit codes: 0 success, 1 failure, 2 usage error, 3 incomplete (budget or deadline
stop; resumable). An incomplete run is never reported as success.

References:
    https://docs.python.org/3/library/argparse.html
    docs/cli.md
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from litter_physics import __version__
from litter_physics.artifacts import ArtifactError, RunDirectory
from litter_physics.behavior import (
    CatProfile,
    generate_visits,
    load_profiles,
    profile_from_observations,
    request_with_events,
    synthetic_profile,
)
from litter_physics.benchmark import run_benchmark
from litter_physics.calibration import (
    CalibrationBundle,
    apply_bundle,
    build_bundle,
    load_bundle,
    render_report,
    score_household_masses,
    write_bundle,
)
from litter_physics.models import (
    Mode,
    SimulationRequest,
    load_document,
    load_request,
    validate_payload,
)
from litter_physics.native import (
    NativeContractError,
    NativeSimulationError,
    NativeUnavailableError,
    native_available,
)
from litter_physics.observations import (
    ObservationSet,
    load_observations,
    synthesize_observations,
)
from litter_physics.runner import BudgetError, RunResult, resume_run, start_run
from litter_physics.schemas import SCHEMA_FILES, check_schemas, write_schemas
from litter_physics.sweep import SweepSpec, run_sweep
from litter_physics.verification import StudyOutcome, load_study, run_study
from litter_physics.viewer import ViewerError, ledger_summary, start_loopback_viewer

LOGGER = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3

VALIDATE_KINDS: dict[str, Callable[[Path], object]] = {
    "request": load_request,
    "observations": load_observations,
    "bundle": load_bundle,
    "sweep": lambda path: validate_payload(SweepSpec, load_document(path)),
    "profiles": lambda path: load_profiles(_load_list(path)),
    "study": load_study,
}


class CliError(RuntimeError):
    """A command failed for a reason the user can act on."""


def _load_list(path: Path) -> object:
    """Load a YAML or JSON list document.

    Parameters:
        path (Path): File to load.

    Returns:
        object: The parsed document.

    Raises:
        FileNotFoundError: When the file is missing.
    """
    resolved = Path(str(path)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document does not exist: '{resolved}'")
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


def _write_document(path: Path, payload: object) -> None:
    """Write YAML or JSON depending on the suffix, refusing to overwrite.

    Parameters:
        path (Path): Destination ending in `.yaml`, `.yml`, or `.json`.
        payload (object): JSON-like tree.

    Raises:
        CliError: When the destination exists or has an unknown suffix.
    """
    resolved = Path(str(path)).resolve()
    if resolved.exists():
        raise CliError(f"Refusing to overwrite '{resolved}'")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(payload, sort_keys=False)
    elif resolved.suffix == ".json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        raise CliError(f"Unsupported output suffix '{resolved.suffix}'")
    resolved.write_text(text, encoding="utf-8")


def _configure_logging(verbose: bool) -> None:
    """Set up stderr logging.

    Parameters:
        verbose (bool): Enable DEBUG level.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_result(result: RunResult) -> None:
    """Print a run outcome as JSON on stdout.

    Parameters:
        result (RunResult): Committed segment result.
    """
    payload = {
        "run_dir": str(result.run_dir),
        "segment_index": result.segment_index,
        "status": str(result.run_status),
        "native_status": str(result.native_status),
        "complete": result.complete,
        "resumable": result.resumable,
        "time_s": result.time_s,
        "duration_s": result.duration_s,
        "wall_time_s": result.wall_time_s,
        "native_wall_time_s": result.native_wall_time_s,
        "observables": result.observables,
        "diagnostics": result.diagnostics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate documents of one kind.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    loader = VALIDATE_KINDS[args.kind]
    failures = 0
    for path in args.paths:
        try:
            loaded = loader(Path(path))
        except (ValueError, FileNotFoundError) as e:
            failures += 1
            print(f"INVALID {path}: {e}")
            continue
        summary = ""
        if isinstance(loaded, SimulationRequest):
            summary = (
                f" mode={loaded.mode} boxes={len(loaded.boxes)} events={len(loaded.events)} "
                f"duration_s={loaded.duration_s}"
            )
        elif isinstance(loaded, ObservationSet):
            summary = (
                f" synthetic={loaded.synthetic} cats={len(loaded.cat_ids)} "
                f"visits={len(loaded.visits)}"
            )
        print(f"VALID {path}{summary}")
    return EXIT_FAILURE if failures else EXIT_OK


def cmd_schema(args: argparse.Namespace) -> int:
    """Export or check JSON schemas.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    out = Path(args.out).resolve()
    if args.check:
        stale = check_schemas(out)
        if stale:
            print(f"STALE schemas: {', '.join(stale)}; run `litter-physics schema {out}`")
            return EXIT_FAILURE
        print(f"schemas in '{out}' are current ({len(SCHEMA_FILES)} files)")
        return EXIT_OK
    written = write_schemas(out)
    for path in written:
        print(path)
    return EXIT_OK


def _require_native() -> None:
    """Fail clearly when the native solver is not registered.

    Raises:
        NativeUnavailableError: When `run_json` is missing.
    """
    if not native_available():
        raise NativeUnavailableError(
            "native `run_json` is not registered in this build; numerical modules are not "
            "integrated yet, so `run`, `sweep`, and `benchmark` cannot execute"
        )


def cmd_run(args: argparse.Namespace) -> int:
    """Start or resume a bounded run.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    _require_native()
    calibration_sha256: str | None = None
    if args.resume:
        result = resume_run(
            Path(args.out), wall_budget_s=args.wall_budget, recording=not args.no_recording
        )
    else:
        request = load_request(Path(args.request))
        if args.calibration is not None:
            bundle = load_bundle(Path(args.calibration))
            request = apply_bundle(request, bundle)
            calibration_sha256 = bundle.sha256
            if bundle.synthetic:
                LOGGER.warning("calibration bundle is SYNTHETIC; results are uncalibrated")
        result = start_run(
            request,
            Path(args.out),
            wall_budget_s=args.wall_budget,
            calibration_sha256=calibration_sha256,
            recording=not args.no_recording,
        )
    _print_result(result)
    return EXIT_OK if result.complete else EXIT_INCOMPLETE


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a parameter sweep.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    _require_native()
    outcomes = run_sweep(
        Path(args.spec),
        Path(args.out),
        total_wall_budget_s=args.total_wall_budget,
        recording=args.recording,
    )
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    print(json.dumps({"jobs": len(outcomes), "counts": counts}, indent=2, sort_keys=True))
    return EXIT_OK if counts.keys() <= {"completed"} else EXIT_INCOMPLETE


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark a request.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    _require_native()
    request = load_request(Path(args.request))
    report = run_benchmark(
        request,
        repetitions=args.repetitions,
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return EXIT_OK if report.all_completed else EXIT_INCOMPLETE


def cmd_verify(args: argparse.Namespace) -> int:
    """Run a bounded refinement study against the compiled kernel.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: `EXIT_OK` only for `passed`; `EXIT_INCOMPLETE` when any case did not
            complete; `EXIT_FAILURE` when complete but unresolved.
    """
    _require_native()
    report = run_study(Path(args.study), Path(args.out))
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.outcome is StudyOutcome.PASSED:
        return EXIT_OK
    if report.outcome is StudyOutcome.INCOMPLETE:
        return EXIT_INCOMPLETE
    return EXIT_FAILURE


def cmd_view(args: argparse.Namespace) -> int:
    """Serve a run's recordings to a loopback-only browser viewer.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    run_dir = RunDirectory(Path(args.run_dir))
    manifest = run_dir.read_manifest()
    if manifest is None:
        raise ArtifactError(f"Run '{run_dir.root}' has no committed segments")
    summary = ledger_summary(run_dir.read_metrics())
    print(json.dumps({"status": str(manifest.status), "ledger": summary}, indent=2))
    stop = False

    def on_signal(_signum: int, _frame: object) -> None:
        """Request graceful shutdown without interrupting child-process cleanup."""
        nonlocal stop
        stop = True

    # Install handlers before starting the child: termination during readiness
    # checks must not orphan its separate process group either.
    previous = {
        number: signal.signal(number, on_signal) for number in (signal.SIGINT, signal.SIGTERM)
    }
    handle = None
    try:
        handle = start_loopback_viewer(run_dir, grpc_port=args.grpc_port, web_port=args.web_port)
        print(f"viewer: {handle.web_url} (loopback only)")
        if args.open_browser and not stop:
            import webbrowser  # local import: deferred, only used when opening a browser

            webbrowser.open(handle.web_url)
        deadline = None if args.duration is None else time.monotonic() + args.duration
        while not stop and (deadline is None or time.monotonic() < deadline):
            if handle.process.poll() is not None:
                raise ViewerError(f"viewer exited with code {handle.process.returncode}")
            time.sleep(0.25)
    finally:
        try:
            if handle is not None:
                handle.stop()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Build a calibration bundle and report.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    observations = load_observations(Path(args.observations))
    template = load_request(Path(args.template))
    bundle: CalibrationBundle = build_bundle(observations, template.materials)
    mass_score = None
    if args.run is not None:
        run_dir = RunDirectory(Path(args.run))
        if run_dir.read_manifest() is None:
            raise ArtifactError(f"Run '{run_dir.root}' has no committed segments")
        mass_score = score_household_masses(run_dir.read_metrics(), observations)
    write_bundle(bundle, Path(args.out))
    report = render_report(bundle, mass_score)
    if args.report is not None:
        Path(args.report).resolve().write_text(report + "\n", encoding="utf-8")
    print(report)
    return EXIT_OK


def cmd_observations(args: argparse.Namespace) -> int:
    """Synthesize or summarize observations.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    if args.observations_command == "synthesize":
        observations = synthesize_observations(
            seed=args.seed,
            cats=args.cats,
            visits_per_cat=args.visits,
            box_ids=list(args.box_ids),
        )
        _write_document(Path(args.out), observations.model_dump(mode="json"))
        print(f"wrote SYNTHETIC observations to '{args.out}'")
        return EXIT_OK
    observations = load_observations(Path(args.path))
    per_cat = {
        cat_id: sum(1 for visit in observations.visits if visit.cat_id == cat_id)
        for cat_id in observations.cat_ids
    }
    groups = {
        name: group is not None for name, group in observations.materials.model_dump().items()
    }
    print(
        json.dumps(
            {
                "synthetic": observations.synthetic,
                "visits_per_cat": per_cat,
                "maintenance": len(observations.maintenance),
                "material_groups": groups,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def cmd_visits(args: argparse.Namespace) -> int:
    """Generate visits and write a request with the resulting events.

    Parameters:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Exit code.
    """
    template = load_request(Path(args.template))
    if template.mode is not Mode.HOUSEHOLD:
        raise CliError("visit generation needs a household template")
    box_ids = [box.id for box in template.boxes]
    profiles: list[CatProfile]
    if args.profiles is not None:
        profiles = load_profiles(_load_list(Path(args.profiles)))
    elif args.observations is not None:
        observations = load_observations(Path(args.observations))
        profiles = [
            profile_from_observations(observations, cat_id, box_ids)
            for cat_id in observations.cat_ids
        ]
    else:
        profiles = [
            synthetic_profile(f"synthetic-cat-{index}", box_ids, variant=index)
            for index in range(args.synthetic_cats)
        ]
    generated = generate_visits(
        profiles, list(template.boxes), master_seed=args.seed, horizon_s=args.horizon
    )
    request = request_with_events(template, generated, horizon_s=args.horizon)
    _write_document(Path(args.out), request.model_dump(mode="json"))
    if args.log is not None:
        _write_document(Path(args.log), generated.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "synthetic": generated.synthetic,
                "cats": len(profiles),
                "visits": len(generated.visits),
                "events": len(generated.events),
                "truncated": generated.truncated,
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: Parser with every subcommand registered.
    """
    parser = argparse.ArgumentParser(
        prog="litter-physics", description="Local two-scale litter physics"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate documents")
    validate.add_argument("paths", nargs="+")
    validate.add_argument("--kind", choices=sorted(VALIDATE_KINDS), default="request")
    validate.set_defaults(func=cmd_validate)

    schema = subparsers.add_parser("schema", help="export or check JSON schemas")
    schema.add_argument("out", help="schema directory")
    schema.add_argument("--check", action="store_true", help="fail if committed schemas are stale")
    schema.set_defaults(func=cmd_schema)

    run = subparsers.add_parser("run", help="start or resume a bounded run")
    run.add_argument("request", nargs="?", help="request YAML/JSON (omit with --resume)")
    run.add_argument("--out", required=True, help="run directory")
    run.add_argument("--resume", action="store_true", help="continue an incomplete run")
    run.add_argument("--wall-budget", type=float, default=None, help="segment budget seconds")
    run.add_argument("--calibration", default=None, help="calibration bundle JSON")
    run.add_argument("--no-recording", action="store_true", help="skip the Rerun recording")
    run.set_defaults(func=cmd_run)

    sweep = subparsers.add_parser("sweep", help="run a parameter sweep")
    sweep.add_argument("spec", help="sweep specification YAML/JSON")
    sweep.add_argument("--out", required=True, help="sweep directory")
    sweep.add_argument("--total-wall-budget", type=float, default=None)
    sweep.add_argument("--recording", action="store_true", help="write recordings per job")
    sweep.set_defaults(func=cmd_sweep)

    benchmark = subparsers.add_parser("benchmark", help="time a request on this host")
    benchmark.add_argument("request")
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--report", default=None, help="JSON report path")
    benchmark.set_defaults(func=cmd_benchmark)

    verify = subparsers.add_parser("verify", help="bounded spatial/temporal refinement study")
    verify.add_argument("study", help="study specification YAML/JSON")
    verify.add_argument("--out", required=True, help="new study directory")
    verify.set_defaults(func=cmd_verify)

    view = subparsers.add_parser("view", help="loopback-only browser replay")
    view.add_argument("run_dir")
    view.add_argument("--grpc-port", type=int, default=9876)
    view.add_argument("--web-port", type=int, default=9090)
    view.add_argument("--open-browser", action="store_true")
    view.add_argument("--duration", type=float, default=None, help="serve for N seconds")
    view.set_defaults(func=cmd_view)

    calibrate = subparsers.add_parser("calibrate", help="fit identifiable parameters")
    calibrate.add_argument("--observations", required=True)
    calibrate.add_argument("--template", required=True, help="request supplying defaults")
    calibrate.add_argument("--out", required=True, help="bundle JSON path")
    calibrate.add_argument("--run", default=None, help="completed run for mass comparison")
    calibrate.add_argument("--report", default=None, help="text report path")
    calibrate.set_defaults(func=cmd_calibrate)

    observations = subparsers.add_parser("observations", help="observation utilities")
    observation_sub = observations.add_subparsers(dest="observations_command", required=True)
    synthesize = observation_sub.add_parser("synthesize", help="write SYNTHETIC observations")
    synthesize.add_argument("--out", required=True)
    synthesize.add_argument("--seed", type=int, default=0)
    synthesize.add_argument("--cats", type=int, default=2)
    synthesize.add_argument("--visits", type=int, default=30)
    synthesize.add_argument("--box-ids", nargs="+", default=["box-a", "box-b"])
    summary = observation_sub.add_parser("summary", help="summarize an observation file")
    summary.add_argument("path")
    observations.set_defaults(func=cmd_observations)

    visits = subparsers.add_parser("visits", help="generate per-cat semi-Markov visits")
    visits.add_argument("--template", required=True, help="household request template")
    visits.add_argument("--out", required=True, help="request with generated events")
    visits.add_argument("--log", default=None, help="visit log JSON/YAML path")
    visits.add_argument("--horizon", type=float, required=True, help="seconds to generate")
    visits.add_argument("--seed", type=int, default=0)
    source = visits.add_mutually_exclusive_group()
    source.add_argument("--profiles", default=None, help="profile list YAML/JSON")
    source.add_argument("--observations", default=None, help="observation file")
    source.add_argument("--synthetic-cats", type=int, default=1)
    visits.set_defaults(func=cmd_visits)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Parameters:
        argv (Sequence[str] | None): Arguments without the program name.

    Returns:
        int: Exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "run" and not args.resume and args.request is None:
        parser.error("`run` needs a request path unless `--resume` is given")
    func: Callable[[argparse.Namespace], int] = args.func
    try:
        return func(args)
    except NativeUnavailableError as e:
        print(f"UNAVAILABLE: {e}", file=sys.stderr)
        return EXIT_FAILURE
    except (
        ArtifactError,
        BudgetError,
        CliError,
        FileExistsError,
        FileNotFoundError,
        NativeContractError,
        NativeSimulationError,
        ValueError,
        ViewerError,
    ) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_FAILURE
