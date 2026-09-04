"""Commission cost model supporting multiple fee structures.

Supports per-lot, per-unit, per-transaction, and percentage structures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CommissionType = Literal["per_lot", "per_unit", "per_transaction", "percentage"]


class CommissionModelConfig(BaseModel):
    """Configuration governing broker commission calculations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commission_type: CommissionType = "per_lot"
    rate: float = Field(default=3.50, ge=0.0)  # e.g., .50 per lot per side ( roundturn)
    lot_size: float = Field(default=100000.0, gt=0.0)
    min_commission: float = Field(default=0.0, ge=0.0)


class CommissionModel:
    """Calculates broker commission fees across supported fee structures."""

    def __init__(self, config: CommissionModelConfig | None = None) -> None:
        self.config = config or CommissionModelConfig()

    def calculate_commission(
        self,
        quantity: float,
        price: float,
        *,
        point_value: float = 1.0,
    ) -> float:
        """Calculate the commission charge for a single execution side.

        Args:
            quantity: Order volume in units.
            price: Executed fill price.
            point_value: Currency conversion point value.
        """
        qty = float(quantity)
        fill_p = float(price)

        if self.config.commission_type == "per_lot":
            lots = qty / self.config.lot_size
            comm = lots * self.config.rate
        elif self.config.commission_type == "per_unit":
            comm = qty * self.config.rate
        elif self.config.commission_type == "per_transaction":
            comm = self.config.rate
        elif self.config.commission_type == "percentage":
            notional = qty * fill_p * float(point_value)
            comm = notional * (self.config.rate / 100.0)
        else:
            comm = 0.0

        return max(float(comm), self.config.min_commission)
