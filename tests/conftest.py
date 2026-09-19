"""Shared fixtures.

`--require-native` turns missing-native skips into failures so integration cannot
pass on mocks by accident.

References:
    https://docs.pytest.org/en/stable/how-to/fixtures.html
"""

from __future__ import annotations

from pathlib import Path

import pytest

from litter_physics import native
from litter_physics.models import SimulationRequest, load_request
from tests.fake_core import FakeCore

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SCHEMAS = REPO / "schemas"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--require-native` flag.

    Parameters:
        parser (pytest.Parser): Pytest option parser.
    """
    parser.addoption(
        "--require-native",
        action="store_true",
        default=False,
        help="fail instead of skip when native `run_json` is unavailable",
    )


@pytest.fixture
def household_request() -> SimulationRequest:
    """Load the committed household example.

    Returns:
        SimulationRequest: Validated request.
    """
    return load_request(EXAMPLES / "household_basic.yaml")


@pytest.fixture
def research_request() -> SimulationRequest:
    """Load the committed research example.

    Returns:
        SimulationRequest: Validated request.
    """
    return load_request(EXAMPLES / "research_slump.yaml")


@pytest.fixture
def fake_core() -> FakeCore:
    """Provide a fresh fake solver.

    Returns:
        FakeCore: Contract-shaped fake.
    """
    return FakeCore()


@pytest.fixture
def patched_native(monkeypatch: pytest.MonkeyPatch, fake_core: FakeCore) -> FakeCore:
    """Register the fake solver as `_core.run_json` for CLI tests.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch.
        fake_core (FakeCore): Fake solver.

    Returns:
        FakeCore: The registered fake.
    """
    monkeypatch.setattr(native._core, "run_json", fake_core, raising=False)
    return fake_core


@pytest.fixture
def native_or_skip(request: pytest.FixtureRequest) -> None:
    """Skip, or fail under `--require-native`, when the real solver is missing.

    Parameters:
        request (pytest.FixtureRequest): Fixture request for option access.
    """
    if native.native_available():
        return
    message = "native `run_json` is not registered; integration pending"
    if request.config.getoption("--require-native"):
        pytest.fail(message)
    pytest.skip(message)
