"""Provenance capture for reproducible run artifacts.

Every run freezes the request hash, package code hash, native extension hash, git
revision, dependency versions, interpreter, and hardware description. These are
observations about the producing machine, not performance claims.

References:
    docs/plan.md (Artifacts and interfaces)
    https://docs.python.org/3/library/importlib.metadata.html
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from litter_physics import __version__
from litter_physics.models import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SimulationRequest,
    canonical_json,
    request_to_wire,
)

LOGGER = logging.getLogger(__name__)

TRACKED_DISTRIBUTIONS = ("pydantic", "numpy", "scipy", "pyarrow", "PyYAML", "rerun-sdk")
PACKAGE_DIR = Path(__file__).resolve().parent
GIT_BINARY = Path("/usr/bin/git")
SYSCTL_BINARY = Path("/usr/sbin/sysctl")


class HardwareInfo(BaseModel):
    """Description of the producing host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    machine: str = Field(description="CPU architecture string.")
    system: str = Field(description="Operating system name.")
    release: str = Field(description="Operating system release.")
    processor: str = Field(description="Processor description, possibly empty.")
    cpu_count: int = Field(description="Logical CPUs visible to the process.")
    memory_bytes: int | None = Field(description="Physical memory when discoverable.")


class Provenance(BaseModel):
    """Frozen provenance record stored alongside every run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    created_utc: str = Field(description="ISO-8601 creation time.")
    package_version: str = Field(description="`litter_physics` version.")
    schema_version: int = Field(description="Wire schema version.")
    protocol_version: int = Field(description="Native protocol version expected by Python.")
    request_sha256: str = Field(description="Hash of the canonical request JSON.")
    python_code_sha256: str = Field(description="Hash of the Python package sources.")
    native_sha256: str | None = Field(description="Hash of the compiled extension, if built.")
    git_commit: str | None = Field(description="Repository commit when running from git.")
    git_dirty: bool | None = Field(description="Whether the working tree had changes.")
    dependencies: dict[str, str] = Field(description="Installed versions of tracked packages.")
    python_version: str = Field(description="Interpreter version string.")
    hardware: HardwareInfo = Field(description="Producing host description.")
    calibration_sha256: str | None = Field(description="Hash of the calibration bundle used.")
    fidelity_note: str = Field(description="Standing caveat about validation status.")

    @property
    def sha256(self) -> str:
        """Hash the whole provenance record.

        Returns:
            str: Hex digest of the canonical JSON.
        """
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


def hash_request(request: SimulationRequest) -> str:
    """Hash the canonical wire form of a request.

    Parameters:
        request (SimulationRequest): Validated request.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(request_to_wire(request).encode()).hexdigest()


def hash_paths(paths: list[Path]) -> str:
    """Hash file contents in sorted path order.

    Parameters:
        paths (list[Path]): Files to include; missing files are skipped.

    Returns:
        str: Hex SHA-256 digest over `relative_path\\0contents` records.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def python_code_hash() -> str:
    """Hash every `.py` and `.pyi` file in the package.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hash_paths(sorted(PACKAGE_DIR.glob("*.py")) + sorted(PACKAGE_DIR.glob("*.pyi")))


def native_hash() -> str | None:
    """Hash the compiled extension when present.

    Returns:
        str | None: Hex digest, or None when no extension binary is installed.
    """
    binaries = sorted(PACKAGE_DIR.glob("_core*.so")) + sorted(PACKAGE_DIR.glob("_core*.pyd"))
    if not binaries:
        return None
    return hash_paths(binaries)


def git_state(root: Path) -> tuple[str | None, bool | None]:
    """Read the git commit and dirty flag for a directory.

    Parameters:
        root (Path): Directory inside a repository, or not.

    Returns:
        tuple[str | None, bool | None]: Commit and dirty flag, or Nones outside git.
    """
    if not GIT_BINARY.is_file():
        return None, None
    try:
        commit = subprocess.check_output(
            [str(GIT_BINARY), "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            [str(GIT_BINARY), "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        LOGGER.debug(f"git provenance unavailable for '{root}': {e}")
        return None, None
    return commit, bool(status.strip())


def dependency_versions() -> dict[str, str]:
    """Collect installed versions for tracked distributions.

    Returns:
        dict[str, str]: Distribution name to version, `missing` when not installed.
    """
    versions: dict[str, str] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def physical_memory_bytes() -> int | None:
    """Discover physical memory without third-party helpers.

    Returns:
        int | None: Bytes, or None when the platform offers no cheap answer.
    """
    if sys.platform == "darwin" and SYSCTL_BINARY.is_file():
        try:
            text = subprocess.check_output(
                [str(SYSCTL_BINARY), "-n", "hw.memsize"], text=True, stderr=subprocess.STDOUT
            )
            return int(text.strip())
        except (subprocess.CalledProcessError, OSError, ValueError):
            return None
    names = ("SC_PAGE_SIZE", "SC_PHYS_PAGES")
    if all(name in os.sysconf_names for name in names):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def hardware_info() -> HardwareInfo:
    """Describe the producing host.

    Returns:
        HardwareInfo: Architecture, OS, CPU count, and memory.
    """
    return HardwareInfo(
        machine=platform.machine(),
        system=platform.system(),
        release=platform.release(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        memory_bytes=physical_memory_bytes(),
    )


def build_provenance(
    request: SimulationRequest, *, calibration_sha256: str | None = None
) -> Provenance:
    """Freeze provenance for a request on this machine.

    Parameters:
        request (SimulationRequest): Validated request.
        calibration_sha256 (str | None): Hash of a calibration bundle when one was used.

    Returns:
        Provenance: The frozen record.
    """
    commit, dirty = git_state(PACKAGE_DIR)
    return Provenance(
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        package_version=__version__,
        schema_version=SCHEMA_VERSION,
        protocol_version=PROTOCOL_VERSION,
        request_sha256=hash_request(request),
        python_code_sha256=python_code_hash(),
        native_sha256=native_hash(),
        git_commit=commit,
        git_dirty=dirty,
        dependencies=dependency_versions(),
        python_version=platform.python_version(),
        hardware=hardware_info(),
        calibration_sha256=calibration_sha256,
        fidelity_note=(
            "Synthetic or uncalibrated inputs produce no household validation claim; "
            "see docs/calibration.md"
        ),
    )
