"""Spread cost modelling supporting historical, fixed, multiplier, and stress modes.

Provides deterministic spread evaluations under Normal, Conservative, and Stress
macro/market scenarios with independent inspectability.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

SpreadScenario = Literal["normal", "conservative", "stress"]
SpreadMode = Literal["historical", "fixed", "multiplier", "stress"]


class SpreadModelConfig(BaseModel):
    """Configuration governing spread calculation modes and stress scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: SpreadMode = "fixed"
    scenario: SpreadScenario = "normal"
    fixed_pips: float = Field(default=1.0, ge=0.0)
    pip_size: float = Field(default=0.0001, gt=0.0)
    multiplier: float = Field(default=1.0, gt=0.0)
    conservative_multiplier: float = Field(default=1.5, ge=1.0)
    stress_multiplier: float = Field(default=2.5, ge=1.0)
    stress_additive_pips: float = Field(default=2.0, ge=0.0)
    min_spread_pips: float = Field(default=0.1, ge=0.0)


class SpreadModel:
    """Calculates spread in price units and transaction costs across market conditions."""

    def __init__(self, config: SpreadModelConfig | None = None) -> None:
        self.config = config or SpreadModelConfig()

    def calculate_spread(
        self,
        bar: dict[str, Any] | pd.Series | None = None,
        *,
        atr: float | None = None,
    ) -> float:
        """Calculate the current bid/ask spread in price units.

        Args:
            bar: Optional bar record or Series containing historical bid/ask or spread fields.
            atr: Optional current Average True Range for volatility stress adjustments.

        Returns:
            Spread in price units (e.g., 0.00012 for 1.2 pips).
        """
        pip_size = self.config.pip_size
        base_spread_price: float

        if self.config.mode == "historical" and bar is not None:
            # Check for direct bid/ask columns
            bid = bar.get("bid") if isinstance(bar, dict) else bar.get("bid", None)
            ask = bar.get("ask") if isinstance(bar, dict) else bar.get("ask", None)
            spread_val = bar.get("spread") if isinstance(bar, dict) else bar.get("spread", None)

            if bid is not None and ask is not None and ask >= bid:
                base_spread_price = float(ask - bid)
            elif spread_val is not None and spread_val >= 0:
                base_spread_price = float(spread_val)
            else:
                base_spread_price = self.config.fixed_pips * pip_size
        elif self.config.mode == "multiplier":
            base_spread_price = (self.config.fixed_pips * pip_size) * self.config.multiplier
        elif self.config.mode == "stress":
            base_spread_price = (
                (self.config.fixed_pips * pip_size * self.config.stress_multiplier)
                + (self.config.stress_additive_pips * pip_size)
            )
            if atr is not None and atr > 0:
                base_spread_price += atr * 0.05
        else:  # "fixed"
            base_spread_price = self.config.fixed_pips * pip_size

        # Apply Scenario Multipliers
        scenario_factor = 1.0
        if self.config.scenario == "conservative":
            scenario_factor = self.config.conservative_multiplier
        elif self.config.scenario == "stress":
            scenario_factor = self.config.stress_multiplier

        final_spread_price = base_spread_price * scenario_factor
        min_spread_price = self.config.min_spread_pips * pip_size

        return max(final_spread_price, min_spread_price)

    def calculate_side_spread_cost(
        self,
        quantity: float,
        spread_price: float,
        point_value: float = 1.0,
    ) -> float:
        """Calculate the monetary spread cost incurred for a single side execution (half spread).

        Args:
            quantity: Trade volume (units or contracts).
            spread_price: Total spread in price units.
            point_value: Monetary value per price point.
        """
        return (spread_price / 2.0) * float(quantity) * float(point_value)
