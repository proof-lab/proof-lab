"""Slippage cost model supporting fixed pips and volatility-dependent execution degradation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SlippageMode = Literal["fixed_pips", "volatility_dependent"]


class SlippageModelConfig(BaseModel):
    """Configuration governing execution slippage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: SlippageMode = "fixed_pips"
    fixed_pips: float = Field(default=0.5, ge=0.0)  # 0.5 pip default slippage
    pip_size: float = Field(default=0.0001, gt=0.0)
    atr_fraction: float = Field(default=0.05, ge=0.0)  # 5% of ATR when volatility-dependent
    min_slippage_pips: float = Field(default=0.0, ge=0.0)


class SlippageModel:
    """Simulates market impact and execution latency slippage on market fills."""

    def __init__(self, config: SlippageModelConfig | None = None) -> None:
        self.config = config or SlippageModelConfig()

    def calculate_slippage_price(
        self,
        side: Literal["BUY", "SELL"],
        requested_price: float,
        *,
        atr: float | None = None,
    ) -> tuple[float, float]:
        """Calculate the adjusted fill price and incurred slippage in price units.

        BUY orders slip higher (worse fill price).
        SELL orders slip lower (worse fill price).

        Returns:
            Tuple of (effective_fill_price, slippage_price_delta).
        """
        pip_size = self.config.pip_size
        req_p = float(requested_price)

        min_slip = self.config.min_slippage_pips * pip_size
        if self.config.mode == "volatility_dependent" and atr is not None and atr > 0:
            slip_price = max(atr * self.config.atr_fraction, min_slip)
        else:
            slip_price = max(self.config.fixed_pips * pip_size, min_slip)

        if side == "BUY":
            fill_price = req_p + slip_price
        else:
            fill_price = req_p - slip_price

        return fill_price, slip_price

    def calculate_slippage_cost(
        self,
        quantity: float,
        slippage_price: float,
        point_value: float = 1.0,
    ) -> float:
        """Calculate total monetary cost caused by slippage."""
        return float(quantity) * float(slippage_price) * float(point_value)
