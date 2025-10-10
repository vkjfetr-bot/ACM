"""Pipeline package initializer for ACM vNext implementation."""

from __future__ import annotations

from pathlib import Path

MAJOR_VERSION = 3
MINOR_VERSION = 0
PATCH_VERSION = 0

__all__ = [
    "MAJOR_VERSION",
    "MINOR_VERSION",
    "PATCH_VERSION",
    "__version__",
    "PACKAGE_ROOT",
    "default_config_path",
]

__version__ = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"

PACKAGE_ROOT = Path(__file__).resolve().parent


def default_config_path() -> Path:
    """Return the default configuration file path."""
    return PACKAGE_ROOT.parent / "acm_config.yaml"
