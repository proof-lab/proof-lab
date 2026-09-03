"""End-to-end paper trading execution coordinator."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.orders import OrderSide
from prooflab.paper.consumer import LiveBar, MarketDataConsumer
from prooflab.paper.execution import PaperExecutionEngine
from prooflab.paper.features import LiveFeatureCalculator
from prooflab.paper.inference import InferencePrediction, LiveInferenceEngine
from prooflab.paper.lifecycle import StrategyLifecycleManager
from prooflab.risk.engine import RiskDecision, RiskEngine


class PaperBarEventResult(BaseModel):
    """Structured report detailing engine actions on an incoming market bar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: datetime
    symbol: str
    data_quality_ok: bool
    data_quality_issues: list[str] = Field(default_factory=list)
    closed_orders: list[str] = Field(default_factory=list)
    feature_ok: bool = False
    prediction: InferencePrediction | None = None
    risk_decision: RiskDecision | None = None
    executed_order_id: str | None = None
    engine_status: str = "PROCESSED"
    notes: str | None = None


class PaperTradingEngine:
    """Orchestrates market ingestion, features, inference, risk gates, and paper execution."""

    def __init__(
        self,
        consumer: MarketDataConsumer,
        feature_calculator: LiveFeatureCalculator,
        inference_engine: LiveInferenceEngine,
        risk_engine: RiskEngine,
        execution_engine: PaperExecutionEngine,
        lifecycle: StrategyLifecycleManager,
        target_pips: float = 30.0,
        stop_pips: float = 20.0,
        risk_pct: float = 0.01,
        pip_size: float = 0.0001,
    ) -> None:
        self.consumer = consumer
        self.feature_calculator = feature_calculator
        self.inference_engine = inference_engine
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.lifecycle = lifecycle
        self.target_pips = target_pips
        self.stop_pips = stop_pips
        self.risk_pct = risk_pct
        self.pip_size = pip_size

    def process_incoming_bar(
        self,
        bar: LiveBar,
        wall_clock_utc: datetime | None = None,
    ) -> PaperBarEventResult:
        """Process a newly completed OHLCV bar through the forward-testing pipeline."""
        clock = wall_clock_utc or bar.timestamp_utc

        # 1. Market Data Quality Check
        is_data_ok, quality_issues = self.consumer.process_bar(bar, wall_clock_utc=clock)
        if not is_data_ok:
            return PaperBarEventResult(
                timestamp_utc=bar.timestamp_utc,
                symbol=bar.symbol,
                data_quality_ok=False,
                data_quality_issues=[issue.value for issue in quality_issues],
                engine_status="DATA_REJECTED",
                notes="Market data quality check failed; bar not appended to history.",
            )

        # 2. Update existing open positions on execution engine
        closed_orders = self.execution_engine.update_on_bar(bar)
        closed_ids = [o.order_id for o in closed_orders]

        # Sync realized PnL from closed orders with Risk Engine state tracker
        for closed_order in closed_orders:
            self.risk_engine.state_tracker.record_closed_trade(closed_order.net_pnl)

        # 3. Check Strategy Lifecycle
        if not self.lifecycle.is_paper_trading_allowed:
            return PaperBarEventResult(
                timestamp_utc=bar.timestamp_utc,
                symbol=bar.symbol,
                data_quality_ok=True,
                closed_orders=closed_ids,
                engine_status="LIFECYCLE_BLOCKED",
                notes=f"Lifecycle state '{self.lifecycle.current_state}' disallows trading.",
            )

        # 4. Live Feature Calculation
        history_df = self.consumer.get_bars_dataframe(bar.symbol)
        feature_res = self.feature_calculator.compute_live_features(history_df)
        if not feature_res.is_valid:
            return PaperBarEventResult(
                timestamp_utc=bar.timestamp_utc,
                symbol=bar.symbol,
                data_quality_ok=True,
                closed_orders=closed_ids,
                feature_ok=False,
                engine_status="WARMUP_OR_FEATURE_ERROR",
                notes=feature_res.rejection_reason,
            )

        # 5. Live Model Inference
        pred = self.inference_engine.predict_live(feature_res.features)
        if not pred.is_valid or pred.signal_direction == 0:
            return PaperBarEventResult(
                timestamp_utc=bar.timestamp_utc,
                symbol=bar.symbol,
                data_quality_ok=True,
                closed_orders=closed_ids,
                feature_ok=True,
                prediction=pred,
                engine_status="NO_SIGNAL",
                notes="Prediction is neutral or invalid.",
            )

        side: OrderSide = "BUY" if pred.signal_direction == 1 else "SELL"
        stop_dist = self.stop_pips * self.pip_size
        target_dist = self.target_pips * self.pip_size

        entry_price = bar.close
        stop_price = (
            (entry_price - stop_dist) if side == "BUY" else (entry_price + stop_dist)
        )
        target_price = (
            (entry_price + target_dist) if side == "BUY" else (entry_price - target_dist)
        )

        # 6. Sovereign Risk Engine Signal Interception & Sizing Gate
        # Sync risk engine state with current portfolio equity and open positions
        self.risk_engine.state_tracker.current_equity = self.execution_engine.current_equity
        self.risk_engine.state_tracker.set_open_positions(
            self.execution_engine.get_open_position_records()
        )

        risk_decision = self.risk_engine.evaluate_signal(
            symbol=bar.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            current_time=bar.timestamp_utc,
            risk_per_trade_pct=self.risk_pct,
            signal_id=f"SIG-{bar.timestamp_utc.isoformat()}",
            bar_timestamp=bar.timestamp_utc,
            data_timestamp=bar.timestamp_utc,
            features=feature_res.features,
            current_spread_pips=bar.spread / self.pip_size,
            model_confidence=pred.confidence,
        )

        if risk_decision.action != "APPROVED" or risk_decision.approved_units <= 0:
            return PaperBarEventResult(
                timestamp_utc=bar.timestamp_utc,
                symbol=bar.symbol,
                data_quality_ok=True,
                closed_orders=closed_ids,
                feature_ok=True,
                prediction=pred,
                risk_decision=risk_decision,
                engine_status="RISK_REJECTED",
                notes=risk_decision.message or "Rejected by sovereign risk engine",
            )

        # 7. Paper Execution
        order = self.execution_engine.execute_order(
            symbol=bar.symbol,
            side=side,
            quantity=risk_decision.approved_units,
            current_price=entry_price,
            timestamp=bar.timestamp_utc,
            stop_loss=stop_price,
            take_profit=target_price,
            spread_pips=bar.spread / self.pip_size,
        )

        # Update risk tracker with newly opened position
        self.risk_engine.state_tracker.set_open_positions(
            self.execution_engine.get_open_position_records()
        )

        return PaperBarEventResult(
            timestamp_utc=bar.timestamp_utc,
            symbol=bar.symbol,
            data_quality_ok=True,
            closed_orders=closed_ids,
            feature_ok=True,
            prediction=pred,
            risk_decision=risk_decision,
            executed_order_id=order.order_id,
            engine_status="ORDER_EXECUTED",
            notes=f"Paper order {order.order_id} filled for {risk_decision.approved_units} units.",
        )
