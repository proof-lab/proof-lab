"""Proof Lab – quantitative research and algorithmic trading platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("prooflab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
