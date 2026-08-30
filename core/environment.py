"""Software/environment snapshot.

Extracted from ``marketdata/store.py`` (manager review of Unit 6,
``marketdata/provenance.py``): ``provenance.py`` needs this same snapshot
for the envelope's environment evidence, and importing it from
``marketdata.store`` would create a future import cycle once a later
storage unit needs to import ``marketdata.provenance``. Living in ``core``,
below both ``marketdata/store.py`` and ``marketdata/provenance.py``, lets
each import it independently with no cycle possible.

Behaviour is moved verbatim from ``marketdata/store.py`` -- not redesigned.
"""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Packages whose versions can change how data is parsed, normalised or stored,
# and therefore belong in provenance.
_TRACKED_PACKAGES = ("pandas", "numpy", "pyarrow", "fyers-apiv3")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def git_revision() -> str:
    """Short git commit of the working tree, or a marker if unavailable.

    Recorded so a dataset can be traced to the exact code that produced it.
    A dirty tree is flagged, because a hash from uncommitted code is not
    reproducible from the repository alone.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rev.returncode != 0:
            return "not-a-git-repo"
        commit = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return f"{commit}-dirty" if dirty.stdout.strip() else commit
    except Exception:  # noqa: BLE001 - provenance must never break a write
        return "unknown"


def software_versions() -> dict:
    """Snapshot of the software that produced a dataset."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": git_revision(),
        **{name: _package_version(name) for name in _TRACKED_PACKAGES},
    }
