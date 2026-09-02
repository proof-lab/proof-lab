"""Configuration system for Proof Lab.

Loading order (later wins):
  1. config/default.yaml
  2. config/<PROOFLAB_ENV>.yaml   (PROOFLAB_ENV defaults to "development")
  3. Environment variables        (prefix PROOFLAB_, delimiter __)

Usage::

    from prooflab.config import get_settings

    settings = get_settings()
    print(settings.log.level)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root: src/prooflab/config.py -> src/prooflab -> src -> root
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR: Path = _REPO_ROOT / "config"

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_merged_yaml() -> dict[str, Any]:
    """Merge default.yaml with the environment-specific YAML overlay."""
    env_name = os.getenv("PROOFLAB_ENV", "development")
    base = _load_yaml(_CONFIG_DIR / "default.yaml")
    overlay = _load_yaml(_CONFIG_DIR / f"{env_name}.yaml")
    return _deep_merge(base, overlay)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply PROOFLAB_* environment variables on top of YAML data.

    Supports single-level nesting via the __ delimiter:
      PROOFLAB_LOG__LEVEL=DEBUG  ->  data["log"]["level"] = "DEBUG"
      PROOFLAB_LIVE_TRADING_ENABLED=true -> data["live_trading_enabled"] = "true"
    """
    prefix = "PROOFLAB_"
    result: dict[str, Any] = _deep_merge(data, {})  # copy
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if "__" in stripped:
            outer, inner = stripped.split("__", 1)
            outer_k = outer.lower()
            inner_k = inner.lower()
            if outer_k not in result:
                result[outer_k] = {}
            if isinstance(result[outer_k], dict):
                result[outer_k][inner_k] = value
        else:
            result[stripped.lower()] = value
    return result


# ---------------------------------------------------------------------------
# Nested config models (plain Pydantic BaseModel, no BaseSettings)
# ---------------------------------------------------------------------------


class LogSettings(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["text", "json"] = "text"


class DataSettings(BaseModel):
    """Data directory paths."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"


class DbSettings(BaseModel):
    """Database connection settings."""

    url: str = "duckdb:///data/prooflab.duckdb"


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class ProofLabSettings(BaseModel):
    """Root configuration object for Proof Lab.

    Sources (in ascending priority): YAML files -> environment variables.

    Environment variables use the prefix `PROOFLAB_` and the delimiter
    `__` for nested keys, e.g.::

        PROOFLAB_LOG__LEVEL=DEBUG
        PROOFLAB_LIVE_TRADING_ENABLED=true   # will be rejected by the validator
    """

    env: str = "development"
    log: LogSettings = LogSettings()
    data: DataSettings = DataSettings()
    db: DbSettings = DbSettings()
    live_trading_enabled: bool = False

    @field_validator("live_trading_enabled", mode="before")
    @classmethod
    def _block_live_trading(cls, value: object) -> bool:
        """Reject any attempt to enable live trading through configuration.

        Live trading requires an explicit programmatic gate that sits outside
        the configuration system.
        """
        truthy = str(value).strip().lower() in ("true", "1", "yes")
        if truthy:
            raise ValueError(
                "live_trading_enabled cannot be set to True via configuration. "
                "Use the explicit live-trading gate."
            )
        return False


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def _build_settings() -> ProofLabSettings:
    yaml_data = load_merged_yaml()
    merged = _apply_env_overrides(yaml_data)
    return ProofLabSettings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> ProofLabSettings:
    """Return the singleton ProofLabSettings instance.

    The result is cached after the first call.  In tests, call
    `get_settings.cache_clear()` between cases to force a fresh load.
    """
    return _build_settings()
