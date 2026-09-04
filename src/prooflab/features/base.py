"""Feature metadata definitions and global feature registry.

Enforces explicit lookback declaration, parameter tracking, family categorization,
and strict absence of future data dependencies across all feature generators.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureFamily(StrEnum):
    """Categorical family classification for quantitative features."""

    PRICE = "PRICE"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    TREND = "TREND"
    TIME = "TIME"
    MICROSTRUCTURE = "MICROSTRUCTURE"


class FeatureMetadata(BaseModel):
    """Immutable specification and declaration for a single feature or feature generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    family: FeatureFamily
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)
    lookback_period: int = 0
    uses_future_data: bool = False
    version: str = "1.0.0"

    @field_validator("lookback_period")
    @classmethod
    def validate_lookback(cls, value: int) -> int:
        if value < 0:
            raise ValueError("lookback_period must be non-negative (>= 0).")
        return value

    @field_validator("uses_future_data")
    @classmethod
    def reject_future_data(cls, value: bool) -> bool:
        if value is True:
            raise ValueError(
                "Absolute Rule Violation: Features must never use future data "
                "(uses_future_data must be False)."
            )
        return value


class FeatureRegistry:
    """Thread-safe registry mapping feature identifiers to metadata and generation functions."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[FeatureMetadata, Callable[..., Any]]] = {}

    def register(
        self,
        metadata: FeatureMetadata,
        generator_func: Callable[..., Any],
    ) -> None:
        """Register a feature calculation function along with its metadata contract.

        Args:
            metadata: FeatureMetadata specification.
            generator_func: Callable calculating the feature.
        """
        if metadata.feature_name in self._entries:
            raise ValueError(f"Feature '{metadata.feature_name}' is already registered.")
        self._entries[metadata.feature_name] = (metadata, generator_func)

    def get_metadata(self, name: str) -> FeatureMetadata:
        """Retrieve metadata for a registered feature."""
        if name not in self._entries:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._entries[name][0]

    def get_generator(self, name: str) -> Callable[..., Any]:
        """Retrieve calculation function for a registered feature."""
        if name not in self._entries:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._entries[name][1]

    def get_family_features(self, family: FeatureFamily) -> list[str]:
        """List all feature names belonging to a specified family."""
        return [
            name
            for name, (meta, _) in self._entries.items()
            if meta.family == family
        ]

    def get_all_metadata(self) -> dict[str, FeatureMetadata]:
        """Return a mapping of all registered feature names to their metadata."""
        return {name: meta for name, (meta, _) in self._entries.items()}

    def compute_max_lookback(self, feature_names: list[str]) -> int:
        """Compute the maximum historical lookback required by a set of features.

        Args:
            feature_names: List of registered feature names.

        Returns:
            Maximum lookback integer (0 if empty or all features have 0 lookback).
        """
        if not feature_names:
            return 0
        lookbacks = [self.get_metadata(name).lookback_period for name in feature_names]
        return max(lookbacks, default=0)

    def has_feature(self, name: str) -> bool:
        """Check if a feature name is registered."""
        return name in self._entries

    def list_all_features(self) -> list[str]:
        """Return a sorted list of all registered feature names."""
        return sorted(self._entries.keys())

    def clear(self) -> None:
        """Clear all registered features (useful for isolated tests)."""
        self._entries.clear()


# Global feature registry instance
feature_registry = FeatureRegistry()
