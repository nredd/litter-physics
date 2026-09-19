"""Command-line entry point.

References:
    https://docs.python.org/3/library/argparse.html
"""

from __future__ import annotations

import argparse

from litter_physics import __version__


def main() -> None:
    """Parse the command line.

    Returns:
        None: The command exits normally on success.
    """
    parser = argparse.ArgumentParser(description="Local two-scale litter physics")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()
