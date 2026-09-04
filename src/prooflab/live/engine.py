"""Live execution coordinator enforcing governance, risk checks, deduplication, and broker dispatch."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from prooflab.live.base import BrokerAdapter, BrokerPosition
from prooflab.live.deduplication import DuplicateSignalError, SignalDeduplicator
from prooflab.live.orders import LiveOrder, LiveOrderState, OrderStateMachine
from prooflab.live.reconciliation import ReconciliationEngine, ReconciliationReport
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState
from prooflab.risk.engine import RiskEngine
from prooflab.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class LiveTradingDisabledError(Exception):
    """Raised when an order submission is attempted while live trading is disabled."""


class LiveExecutionEngine:
    """Coordinates real-time order lifecycle, risk controls, deduplication, and broker adapters."""

    def __init__(
        self,
        adapter: BrokerAdapter,
        lifecycle_manager: StrategyLifecycleManager,
        risk_engine: RiskEngine,
        kill_switch: KillSwitch,
        deduplicator: SignalDeduplicator | None = None,
    ) -> None:
        self.adapter = adapter
        self.lifecycle_manager = lifecycle_manager
        self.risk_engine = risk_engine
        self.kill_switch = kill_switch
        self.deduplicator = deduplicator or SignalDeduplicator()

        self._orders: dict[str, LiveOrder] = {}
        self._last_reconciliation: ReconciliationReport | None = None

    @property
    def is_live_enabled(self) -> bool:
        """Check if live trading is explicitly enabled under lifecycle rules."""
        return self.lifecycle_manager.current_state == StrategyLifecycleState.LIVE_ENABLED

    @property
    def orders(self) -> list[LiveOrder]:
        """Return list of all managed orders."""
        return list(self._orders.values())

    @property
    def active_orders(self) -> list[LiveOrder]:
        """Return list of currently open/active orders."""
        return [o for o in self._orders.values() if o.is_active]

    def startup_and_reconcile(self) -> ReconciliationReport:
        """Connect to broker adapter and perform initial position & risk reconciliation."""
        if not self.adapter.is_connected():
            logger.info("Connecting to broker adapter during startup...")
            self.adapter.connect()

        report = ReconciliationEngine.reconcile(
            adapter=self.adapter,
            local_orders=self._orders,
            risk_engine=self.risk_engine,
        )
        self._last_reconciliation = report
        return report

    def process_signal(
        self,
        signal_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        current_time: datetime | None = None,
    ) -> LiveOrder:
        """Execute signal under governance, deduplication, risk evaluation, and broker submission."""
        now = current_time or datetime.now(UTC)

        # 1. Gate: Live trading must be explicitly enabled
        if not self.is_live_enabled:
            raise LiveTradingDisabledError(
                f"Live trading is disabled. Current strategy lifecycle state: '{self.lifecycle_manager.current_state}'. "
                "Explicit human approval and transition to LIVE_ENABLED is required."
            )

        # 2. Gate: Kill switch must not be active
        if self.kill_switch.is_active:
            raise PermissionError("Kill switch is active. Live order execution is paused.")

        # 3. Gate: Deduplication check
        order_id = f"ORD_{signal_id}"
        self.deduplicator.register_signal(signal_id, order_id, symbol, side)

        # 4. Gate: Risk engine evaluation
        risk_decision = self.risk_engine.evaluate_signal(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss_price=stop_loss or (price * 0.99 if side == "BUY" else price * 1.01),
            current_time=now,
        )

        order = LiveOrder(
            order_id=order_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            created_at=now,
            updated_at=now,
        )
        self._orders[order_id] = order

        if not risk_decision.is_approved:
            reason = risk_decision.message or "; ".join(risk_decision.rejection_reasons) or "Risk engine rejected"
            OrderStateMachine.transition(
                order,
                LiveOrderState.REJECTED,
                reason=f"Risk engine rejected: {reason}",
            )
            logger.warning("Signal %s rejected by risk engine: %s", signal_id, reason)
            return order

        # 5. Broker submission
        try:
            executed_order = self.adapter.submit_order(order)
            self._orders[order_id] = executed_order
            return executed_order
        except Exception as exc:
            OrderStateMachine.transition(
                order,
                LiveOrderState.FAILED,
                reason=f"Broker submission exception: {exc}",
            )
            logger.exception("Failed to submit order %s to broker", order_id)
            return order

    def close_position(self, position_id: str, reason: str = "Manual close") -> bool:
        """Close an open position on the broker and update local records."""
        success = self.adapter.close_position(position_id)
        if success:
            logger.info("Successfully closed position %s (reason: %s)", position_id, reason)
        return success

    def close_all_positions(self, reason: str = "Emergency close all") -> int:
        """Close all open positions on the broker."""
        positions = self.adapter.get_positions()
        closed_count = 0
        for pos in positions:
            if self.adapter.close_position(pos.position_id):
                closed_count += 1
        logger.info("Closed %d/%d open positions (reason: %s)", closed_count, len(positions), reason)
        return closed_count
