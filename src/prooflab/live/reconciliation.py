"""Restart recovery and broker position reconciliation engine."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prooflab.live.base import BrokerAdapter, BrokerPosition
from prooflab.live.orders import LiveOrder, LiveOrderState
from prooflab.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


class DiscrepancyType(StrEnum):
    """Types of discrepancies between local tracking state and broker reality."""

    MATCHED = "MATCHED"
    ORPHAN_BROKER_POSITION = "ORPHAN_BROKER_POSITION"
    MISSING_BROKER_POSITION = "MISSING_BROKER_POSITION"
    VOLUME_MISMATCH = "VOLUME_MISMATCH"
    SIDE_MISMATCH = "SIDE_MISMATCH"


class PositionDiscrepancy(BaseModel):
    """Details of an individual position discrepancy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    symbol: str
    discrepancy_type: DiscrepancyType
    broker_volume: float | None = None
    local_volume: float | None = None
    broker_side: str | None = None
    local_side: str | None = None
    description: str


class ReconciliationReport(BaseModel):
    """Complete immutable audit report of post-restart position reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    broker_positions_count: int
    local_positions_count: int
    is_consistent: bool
    discrepancies: list[PositionDiscrepancy] = Field(default_factory=list)
    account_balance: float
    account_equity: float
    restored_exposure: float
    summary: str


class ReconciliationEngine:
    """Performs state synchronization between local order records and live broker positions."""

    @staticmethod
    def reconcile(
        adapter: BrokerAdapter,
        local_orders: list[LiveOrder] | dict[str, LiveOrder],
        risk_engine: RiskEngine | None = None,
    ) -> ReconciliationReport:
        """Query broker, compare with local active orders, restore risk state, and produce report."""
        if not adapter.is_connected():
            raise RuntimeError("BrokerAdapter must be connected prior to reconciliation.")

        broker_positions = adapter.get_positions()
        account_info = adapter.get_account()

        orders_list = list(local_orders.values()) if isinstance(local_orders, dict) else list(local_orders)
        active_local_orders = [o for o in orders_list if o.status == LiveOrderState.FILLED]

        broker_pos_by_id: dict[str, BrokerPosition] = {p.position_id: p for p in broker_positions}
        # Also index by ticket if position_id is formatted as POS_<ticket>
        broker_pos_by_ticket: dict[int, BrokerPosition] = {}
        for p in broker_positions:
            if p.position_id.startswith("POS_") and p.position_id[4:].isdigit():
                broker_pos_by_ticket[int(p.position_id[4:])] = p

        local_by_ticket: dict[int, LiveOrder] = {}
        for o in active_local_orders:
            if o.broker_ticket is not None:
                local_by_ticket[o.broker_ticket] = o

        discrepancies: list[PositionDiscrepancy] = []
        matched_tickets: set[int] = set()

        # 1. Check local active orders against broker
        for order in active_local_orders:
            ticket = order.broker_ticket
            if ticket is None or ticket not in broker_pos_by_ticket:
                discrepancies.append(
                    PositionDiscrepancy(
                        position_id=order.order_id,
                        symbol=order.symbol,
                        discrepancy_type=DiscrepancyType.MISSING_BROKER_POSITION,
                        local_volume=order.filled_quantity or order.quantity,
                        local_side=order.side,
                        description=f"Local active order {order.order_id} (ticket {ticket}) is no longer open on broker.",
                    )
                )
            else:
                broker_pos = broker_pos_by_ticket[ticket]
                matched_tickets.add(ticket)

                # Check volume and side consistency
                local_vol = order.filled_quantity or order.quantity
                if abs(broker_pos.volume - local_vol) > 1e-4:
                    discrepancies.append(
                        PositionDiscrepancy(
                            position_id=broker_pos.position_id,
                            symbol=order.symbol,
                            discrepancy_type=DiscrepancyType.VOLUME_MISMATCH,
                            broker_volume=broker_pos.volume,
                            local_volume=local_vol,
                            description=f"Volume mismatch on ticket {ticket}: broker={broker_pos.volume}, local={local_vol}",
                        )
                    )
                if broker_pos.side != order.side:
                    discrepancies.append(
                        PositionDiscrepancy(
                            position_id=broker_pos.position_id,
                            symbol=order.symbol,
                            discrepancy_type=DiscrepancyType.SIDE_MISMATCH,
                            broker_side=broker_pos.side,
                            local_side=order.side,
                            description=f"Side mismatch on ticket {ticket}: broker={broker_pos.side}, local={order.side}",
                        )
                    )

        # 2. Check for orphan broker positions
        for p in broker_positions:
            ticket = None
            if p.position_id.startswith("POS_") and p.position_id[4:].isdigit():
                ticket = int(p.position_id[4:])

            if ticket is None or ticket not in matched_tickets:
                discrepancies.append(
                    PositionDiscrepancy(
                        position_id=p.position_id,
                        symbol=p.symbol,
                        discrepancy_type=DiscrepancyType.ORPHAN_BROKER_POSITION,
                        broker_volume=p.volume,
                        broker_side=p.side,
                        description=f"Orphan position {p.position_id} on broker has no corresponding local active order.",
                    )
                )

        total_exposure = sum(p.volume * p.open_price * 100000.0 for p in broker_positions)
        is_consistent = len(discrepancies) == 0

        # 3. Restore risk state if risk engine provided
        if risk_engine is not None:
            # Sync start of day baseline and current exposure
            risk_engine.limits_evaluator.start_of_day_equity = account_info.equity
            risk_engine.limits_evaluator.start_of_week_equity = account_info.equity
            logger.info(
                "Restored risk engine state: equity=%.2f, balance=%.2f, open_positions=%d, exposure=%.2f",
                account_info.equity,
                account_info.balance,
                len(broker_positions),
                total_exposure,
            )

        summary = (
            f"Reconciliation {'PASSED (Consistent)' if is_consistent else 'FAILED (Discrepancies found)'}: "
            f"{len(broker_positions)} broker positions, {len(active_local_orders)} local active orders, "
            f"{len(discrepancies)} discrepancies."
        )

        logger.info(summary)
        return ReconciliationReport(
            broker_positions_count=len(broker_positions),
            local_positions_count=len(active_local_orders),
            is_consistent=is_consistent,
            discrepancies=discrepancies,
            account_balance=account_info.balance,
            account_equity=account_info.equity,
            restored_exposure=round(total_exposure, 2),
            summary=summary,
        )
