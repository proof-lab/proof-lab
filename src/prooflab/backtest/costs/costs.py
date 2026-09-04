"""Unified execution cost coordinator combining spread, commission, slippage, and swap."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from prooflab.backtest.costs.commission import CommissionModel, CommissionModelConfig
from prooflab.backtest.costs.slippage import SlippageModel, SlippageModelConfig
from prooflab.backtest.costs.spread import SpreadModel, SpreadModelConfig
from prooflab.backtest.costs.swap import SwapModel, SwapModelConfig


class ExecutionCostConfig(BaseModel):
    """Configuration bundling all execution friction components."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread: SpreadModelConfig = Field(default_factory=SpreadModelConfig)
    commission: CommissionModelConfig = Field(default_factory=CommissionModelConfig)
    slippage: SlippageModelConfig = Field(default_factory=SlippageModelConfig)
    swap: SwapModelConfig = Field(default_factory=SwapModelConfig)
    execution_delay_bars: int = Field(default=1, ge=0)  # Default 1-bar delay (fill next bar open)


class ExecutionCostModel:
    """Coordinates and computes individual and aggregated execution costs."""

    def __init__(self, config: ExecutionCostConfig | None = None) -> None:
        self.config = config or ExecutionCostConfig()
        self.spread_model = SpreadModel(self.config.spread)
        self.commission_model = CommissionModel(self.config.commission)
        self.slippage_model = SlippageModel(self.config.slippage)
        self.swap_model = SwapModel(self.config.swap)

    def calculate_entry_execution(
        self,
        side: Literal["BUY", "SELL"],
        requested_price: float,
        quantity: float,
        bar: dict[str, Any] | pd.Series | None = None,
        *,
        atr: float | None = None,
        point_value: float = 1.0,
    ) -> dict[str, float]:
        """Compute execution fill price and individual entry friction costs.

        Returns:
            Dictionary with:
                - fill_price: Effective price after slippage.
                - spread_price: Incurred spread in price units.
                - spread_cost: Monetary spread fee.
                - commission_cost: Broker commission fee.
                - slippage_cost: Monetary slippage degradation.
                - total_entry_friction: Sum of all entry costs.
        """
        fill_price, slip_price = self.slippage_model.calculate_slippage_price(
            side, requested_price, atr=atr
        )
        spread_price = self.spread_model.calculate_spread(bar, atr=atr)

        spread_cost = self.spread_model.calculate_side_spread_cost(
            quantity, spread_price, point_value=point_value
        )
        comm_cost = self.commission_model.calculate_commission(
            quantity, fill_price, point_value=point_value
        )
        slip_cost = self.slippage_model.calculate_slippage_cost(
            quantity, slip_price, point_value=point_value
        )

        return {
            "fill_price": fill_price,
            "spread_price": spread_price,
            "spread_cost": spread_cost,
            "commission_cost": comm_cost,
            "slippage_cost": slip_cost,
            "total_entry_friction": spread_cost + comm_cost + slip_cost,
        }

    def calculate_exit_execution(
        self,
        side: Literal["BUY", "SELL"],
        requested_price: float,
        quantity: float,
        bar: dict[str, Any] | pd.Series | None = None,
        *,
        atr: float | None = None,
        point_value: float = 1.0,
    ) -> dict[str, float]:
        """Compute exit execution fill price and individual exit friction costs."""
        # Closing side is opposite of position side: Long position closes via SELL, etc.
        exit_side: Literal["BUY", "SELL"] = "SELL" if side == "BUY" else "BUY"
        fill_price, slip_price = self.slippage_model.calculate_slippage_price(
            exit_side, requested_price, atr=atr
        )
        spread_price = self.spread_model.calculate_spread(bar, atr=atr)

        spread_cost = self.spread_model.calculate_side_spread_cost(
            quantity, spread_price, point_value=point_value
        )
        comm_cost = self.commission_model.calculate_commission(
            quantity, fill_price, point_value=point_value
        )
        slip_cost = self.slippage_model.calculate_slippage_cost(
            quantity, slip_price, point_value=point_value
        )

        return {
            "fill_price": fill_price,
            "spread_price": spread_price,
            "spread_cost": spread_cost,
            "commission_cost": comm_cost,
            "slippage_cost": slip_cost,
            "total_exit_friction": spread_cost + comm_cost + slip_cost,
        }

    def calculate_holding_swap(
        self,
        side: Literal["BUY", "SELL"],
        quantity: float,
        prev_time: AwareDatetime | None,
        curr_time: AwareDatetime,
        *,
        point_value: float = 1.0,
    ) -> float:
        """Calculate financing swap for positions held across rollover boundaries."""
        return self.swap_model.calculate_swap(
            side, quantity, prev_time, curr_time, point_value=point_value
        )
