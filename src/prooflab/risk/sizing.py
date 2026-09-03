"""Risk-based position sizing engine with broker constraint and leverage enforcement."""

from __future__ import annotations

import json
import math

from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.portfolio import BrokerLimitsConfig


class PositionSizingResult(BaseModel):
    """Immutable audit record containing calculated position sizing results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    calculated_lots: float = Field(default=0.0, ge=0.0)
    calculated_units: float = Field(default=0.0, ge=0.0)
    risk_amount_dollars: float = Field(default=0.0, ge=0.0)
    nominal_exposure_dollars: float = Field(default=0.0, ge=0.0)
    implied_leverage: float = Field(default=0.0, ge=0.0)
    rejection_reason: str | None = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class RiskPositionSizer:
    """Calculates risk-proportional position sizing strictly respecting broker constraints."""

    def __init__(self, broker_limits: BrokerLimitsConfig | None = None) -> None:
        self.limits = broker_limits or BrokerLimitsConfig()

    def calculate_position_size(
        self,
        account_equity: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float,
        point_value: float = 1.0,
        contract_size: float = 100000.0,
    ) -> PositionSizingResult:
        """Calculate exact lot and unit sizing given account equity and stop loss distance."""
        if account_equity <= 0:
            return PositionSizingResult(
                is_valid=False,
                rejection_reason="Account equity must be positive",
            )

        if risk_per_trade_pct <= 0 or risk_per_trade_pct > 1.0:
            return PositionSizingResult(
                is_valid=False,
                rejection_reason=f"Invalid risk_per_trade_pct ({risk_per_trade_pct})",
            )

        stop_dist = abs(entry_price - stop_loss_price)
        if stop_dist <= 1e-9:
            return PositionSizingResult(
                is_valid=False,
                rejection_reason="Stop loss price cannot equal entry price",
            )

        # 1. Base dollar risk amount
        risk_amount = account_equity * risk_per_trade_pct

        # 2. Raw units calculation: risk_amount / (stop_dist * point_value)
        # For standard FX (point_value=1.0 on 1 unit), risk per unit = stop_dist * point_value
        raw_units = risk_amount / (stop_dist * point_value)
        raw_lots = raw_units / contract_size

        # 3. Step size quantization (floor to prevent risk overshooting)
        step = self.limits.lot_step
        stepped_lots = round(math.floor(round(raw_lots / step, 8)) * step, 4)

        # 4. Check minimum lot boundary
        if stepped_lots < self.limits.min_lot_size:
            return PositionSizingResult(
                is_valid=False,
                risk_amount_dollars=round(risk_amount, 2),
                rejection_reason=(
                    f"Calculated size ({stepped_lots:.4f} lots) below broker minimum "
                    f"({self.limits.min_lot_size:.4f} lots)"
                ),
            )

        # 5. Clamp to maximum lot boundary
        clamped_lots = min(stepped_lots, self.limits.max_lot_size)
        final_units = clamped_lots * contract_size

        # 6. Leverage Ceiling Enforcement
        nominal_exposure = final_units * entry_price
        implied_leverage = nominal_exposure / account_equity

        if implied_leverage > self.limits.max_leverage:
            # Reduce lots to strictly satisfy leverage cap
            max_allowed_units = (account_equity * self.limits.max_leverage) / entry_price
            max_allowed_lots = max_allowed_units / contract_size
            stepped_lev_lots = round(math.floor(round(max_allowed_lots / step, 8)) * step, 4)

            if stepped_lev_lots < self.limits.min_lot_size:
                return PositionSizingResult(
                    is_valid=False,
                    risk_amount_dollars=round(risk_amount, 2),
                    rejection_reason=(
                        f"Size needed to satisfy max leverage {self.limits.max_leverage}x "
                        f"falls below broker minimum lot size"
                    ),
                )

            clamped_lots = stepped_lev_lots
            final_units = clamped_lots * contract_size
            nominal_exposure = final_units * entry_price
            implied_leverage = nominal_exposure / account_equity

        return PositionSizingResult(
            is_valid=True,
            calculated_lots=round(clamped_lots, 4),
            calculated_units=round(final_units, 2),
            risk_amount_dollars=round(risk_amount, 2),
            nominal_exposure_dollars=round(nominal_exposure, 2),
            implied_leverage=round(implied_leverage, 2),
        )
