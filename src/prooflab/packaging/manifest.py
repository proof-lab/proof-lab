"""Manifest and compatibility schemas for portable .plb strategy packages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class CompatibilityDeclaration(BaseModel):
    """Execution environment requirements declared by a strategy package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    feature_names: list[str] = Field(min_length=1)
    feature_parameters: dict[str, Any] = Field(default_factory=dict)
    min_app_version: str = Field(default="0.1.0")
    target_pips: float = Field(gt=0.0)
    stop_pips: float = Field(gt=0.0)
    horizon_bars: int = Field(gt=0)
    extra_parameters: dict[str, Any] = Field(default_factory=dict)


class PackageManifest(BaseModel):
    """Canonical manifest.json metadata inside a .plb strategy package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: str = Field(default="1.0.0")
    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    feature_version: str = Field(default="1.0.0")
    model_version: str = Field(default="1.0.0")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    app_version: str = Field(default="0.1.0")
    min_app_version: str = Field(default="0.1.0")
    compatibility: CompatibilityDeclaration
    models: list[str] = Field(default_factory=list)
    description: str = Field(default="")
    author: str = Field(default="")
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> PackageManifest:
        """Parse and validate manifest from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageManifest:
        """Validate and construct manifest from dictionary."""
        return cls.model_validate(data)
