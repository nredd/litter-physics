"""Exercise the installed extension rather than just parsing its source.

References:
    https://pyo3.rs/
"""

from __future__ import annotations

from litter_physics import _core


def test_protocol_version() -> None:
    """Verify the native extension and Python protocol agree.

    Returns:
        None: Assertions report incompatibility.
    """
    assert _core.protocol_version() == 1
