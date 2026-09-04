"""Setup configuration schemas for Predictive Setup Classification.

Defines target, stop, horizon, direction, unit, and ambiguity resolution policies.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class Direction(StrEnum):
    """Trading setup direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class DistanceUnit(StrEnum):
    """Units used to specify target and stop barrier distances."""

    POINTS = "POINTS"
    PIPS = "PIPS"
    PERCENT = "PERCENT"
    ATR = "ATR"


class AmbiguityPolicy(StrEnum):
    """Policy for resolving bars where both target and stop are touched."""

    CONSERVATIVE = "CONSERVATIVE"  # Assume adverse barrier hit first (Default)
    OPTIMISTIC = "OPTIMISTIC"      # Assume favorable barrier hit first
    EXCLUDE = "EXCLUDE"            # Mark bar as ambiguous/excluded from classification


class SetupConfig(BaseModel):
    """Immutable configuration defining a quantitative setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Direction
    target_distance: float
    stop_distance: float
    unit: DistanceUnit = DistanceUnit.POINTS
    horizon_bars: int = 10
    ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.CONSERVATIVE
    point_value: float = 1.0
    entry_price_col: str = "close"

    @field_validator("target_distance", "stop_distance")
    @classmethod
    def validate_positive_distance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Distance must be strictly positive (> 0).")
        return value

    @field_validator("horizon_bars")
    @classmethod
    def validate_horizon(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Horizon bars must be at least 1.")
        return value

    @field_validator("point_value")
    @classmethod
    def validate_point_value(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Point value must be strictly positive (> 0).")
        return value

    def calculate_barriers(
        self,
        entry_price: float,
        atr_value: float | None = None,
    ) -> tuple[float, float]:
        """Calculate the absolute target and stop price barriers.

        Args:
            entry_price: The baseline entry price.
            atr_value: Optional Average True Range value required if unit is ATR.

        Returns:
            Tuple of (target_price, stop_price).
        """
        if entry_price <= 0:
            raise ValueError("Entry price must be strictly positive.")

        if self.unit in (DistanceUnit.POINTS, DistanceUnit.PIPS):
            t_offset = self.target_distance * self.point_value
            s_offset = self.stop_distance * self.point_value
        elif self.unit == DistanceUnit.PERCENT:
            t_offset = entry_price * (self.target_distance / 100.0)
            s_offset = entry_price * (self.stop_distance / 100.0)
        elif self.unit == DistanceUnit.ATR:
            if atr_value is None or atr_value <= 0:
                raise ValueError("Positive atr_value is required when unit is ATR.")
            t_offset = self.target_distance * atr_value
            s_offset = self.stop_distance * atr_value
        else:
            raise ValueError(f"Unsupported distance unit: {self.unit}")

        if self.direction == Direction.LONG:
            target_price = entry_price + t_offset
            stop_price = entry_price - s_offset
        elif self.direction == Direction.SHORT:
            target_price = entry_price - t_offset
            stop_price = entry_price + s_offset
        else:
            raise ValueError(f"Unsupported direction: {self.direction}")

        return target_price, stop_price
