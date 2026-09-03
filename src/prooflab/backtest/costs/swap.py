"""Swap and overnight financing cost model for leveraged instrument holding."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class SwapModelConfig(BaseModel):
    """Configuration governing overnight financing swap calculations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    long_swap_pips_per_day: float = Field(default=-0.5)  # e.g., -0.5 pips cost for long overnight
    short_swap_pips_per_day: float = Field(default=-0.2)  # e.g., -0.2 pips cost for short overnight
    pip_size: float = Field(default=0.0001, gt=0.0)
    lot_size: float = Field(default=100000.0, gt=0.0)
    rollover_hour_utc: int = Field(default=22, ge=0, le=23)
    triple_swap_weekday: int = Field(default=2, ge=0, le=6)  # Wednesday (2) triple rollover


class SwapModel:
    """Calculates financing charges when positions cross overnight rollover boundaries."""

    def __init__(self, config: SwapModelConfig | None = None) -> None:
        self.config = config or SwapModelConfig()

    def crosses_rollover(
        self,
        prev_time: AwareDatetime | None,
        curr_time: AwareDatetime,
    ) -> bool:
        """Check if time stepped across the configured daily rollover hour."""
        if prev_time is None:
            return False

        # If date advanced or hour transitioned past rollover hour
        if curr_time.date() > prev_time.date():
            return True
        if prev_time.hour < self.config.rollover_hour_utc <= curr_time.hour:
            return True
        return False

    def calculate_swap(
        self,
        side: Literal["BUY", "SELL"],
        quantity: float,
        prev_time: AwareDatetime | None,
        curr_time: AwareDatetime,
        *,
        point_value: float = 1.0,
    ) -> float:
        """Calculate the swap cost for the time step (positive value = expense/cost).

        Applies 3x multiplier on triple swap day (Wednesday) to cover weekend settlement.

        Returns:
            Monetary swap cost (positive = debit/cost to trader, negative = credit/earning).
        """
        if not self.crosses_rollover(prev_time, curr_time):
            return 0.0

        # Determine if triple swap applies
        # If transitioning from Wednesday (weekday 2), 3 days of swap are billed
        multiplier = 3.0 if curr_time.weekday() == self.config.triple_swap_weekday else 1.0

        # Long swap or short swap pips
        pips = (
            self.config.long_swap_pips_per_day
            if side == "BUY"
            else self.config.short_swap_pips_per_day
        )
        # In FX conventions: negative swap points = fee to trader
        # Monetary charge: -1 * pips * pip_size * quantity * point_value * multiplier
        cost = (
            -1.0
            * (pips * self.config.pip_size)
            * float(quantity)
            * float(point_value)
            * multiplier
        )
        return float(cost)
