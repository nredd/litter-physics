"""Thin, validated boundary around the compiled `_core` extension.

Python never steps particles. This module serializes a validated request, calls the
native `run_json`, and validates the returned JSON before anything else touches it.
The native callable is injectable so orchestration can be tested before the numerical
modules are registered, without pretending that the mock is the real solver.

References:
    docs/wire-contract.md
    https://pyo3.rs/
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from litter_physics import _core
from litter_physics.models import (
    PROTOCOL_VERSION,
    Checkpoint,
    SimulationOutput,
    SimulationRequest,
    canonical_json,
    parse_output,
    request_to_wire,
)

LOGGER = logging.getLogger(__name__)

NativeRunner = Callable[[str, str | None], str]


class NativeUnavailableError(RuntimeError):
    """The compiled extension does not expose `run_json` yet."""


class NativeSimulationError(RuntimeError):
    """The native solver rejected the input or failed numerically."""


class NativeContractError(RuntimeError):
    """The native output violated the wire contract."""


def native_available() -> bool:
    """Report whether the installed extension exposes `run_json`.

    Returns:
        bool: True when `run_json` is registered on the extension.
    """
    return callable(getattr(_core, "run_json", None))


def resolve_runner(runner: NativeRunner | None) -> NativeRunner:
    """Pick the native callable, defaulting to the compiled extension.

    Parameters:
        runner (NativeRunner | None): Injected callable for tests, or None.

    Returns:
        NativeRunner: A callable with the `run_json` signature.

    Raises:
        NativeUnavailableError: When no injection is given and the extension lacks
            `run_json` or reports a different protocol version.
    """
    if runner is not None:
        return runner
    version = _core.protocol_version()
    if version != PROTOCOL_VERSION:
        raise NativeUnavailableError(
            f"native protocol version '{version}' does not match Python '{PROTOCOL_VERSION}'"
        )
    registered: NativeRunner | None = getattr(_core, "run_json", None)
    if registered is None or not callable(registered):
        raise NativeUnavailableError(
            "native `run_json` is not registered in `litter_physics._core`; the numerical "
            "modules have not been integrated into this build"
        )
    return registered


def run_segment(
    request: SimulationRequest,
    resume: Checkpoint | None,
    *,
    runner: NativeRunner | None = None,
) -> SimulationOutput:
    """Execute one native segment and validate its output against the contract.

    Parameters:
        request (SimulationRequest): Validated request; its wall budget is the native
            budget for this segment only.
        resume (Checkpoint | None): Checkpoint to continue from, or None for a fresh run.
        runner (NativeRunner | None): Injected native callable for tests.

    Returns:
        SimulationOutput: Validated output whose frames are a suffix after `resume`.

    Raises:
        NativeSimulationError: When the native call raises.
        NativeContractError: When the output disagrees with the request or checkpoint.
    """
    if resume is not None and not resume.request.equivalent_except_budget(request):
        raise NativeContractError("resume checkpoint request differs from the supplied request")
    call = resolve_runner(runner)
    request_json = request_to_wire(request)
    resume_json = canonical_json(resume.model_dump(mode="json")) if resume is not None else None
    LOGGER.info(
        f"native segment mode='{request.mode}' budget={request.max_wall_time_s:.1f}s "
        f"resume_from={resume.time_s if resume is not None else 'start'}"
    )
    try:
        text = call(request_json, resume_json)
    except (ValueError, RuntimeError) as e:
        raise NativeSimulationError(f"native `run_json` failed: {e}") from e
    if not isinstance(text, str):
        raise NativeContractError(f"native `run_json` returned {type(text).__name__}, not str")
    try:
        output = parse_output(text)
    except ValueError as e:
        raise NativeContractError(f"native output violates the wire contract: {e}") from e
    _check_output_against_request(output, request, resume)
    return output


def _check_output_against_request(
    output: SimulationOutput, request: SimulationRequest, resume: Checkpoint | None
) -> None:
    """Cross-check native output against what was asked for.

    Parameters:
        output (SimulationOutput): Validated output.
        request (SimulationRequest): The request that produced it.
        resume (Checkpoint | None): The checkpoint resumed from, if any.

    Raises:
        NativeContractError: When mode, embedded request, or time monotonicity disagree.
    """
    if output.mode is not request.mode:
        raise NativeContractError(
            f"native output mode '{output.mode}' differs from request mode '{request.mode}'"
        )
    if not output.checkpoint.request.equivalent_except_budget(request):
        raise NativeContractError("native checkpoint does not embed the original request")
    start = 0.0 if resume is None else resume.time_s
    if output.time_s < start:
        raise NativeContractError(
            f"native output time '{output.time_s}' regressed below resume time '{start}'"
        )
    if resume is not None and output.frames and output.frames[0].time_s < start:
        raise NativeContractError("resumed frames must be a suffix after the checkpoint time")
    if output.time_s > request.duration_s:
        raise NativeContractError("native output overran `duration_s`")
