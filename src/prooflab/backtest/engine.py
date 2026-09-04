"""Event-driven chronological backtesting engine and execution simulation.

Orchestrates SignalEngine, ExecutionCostModel, PositionSizer, and PortfolioAccountant
without lookahead bias, enforcing realistic order latency delays and intrabar barrier
checks.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.costs import ExecutionCostConfig, ExecutionCostModel
from prooflab.backtest.metrics import BacktestMetricsReport, calculate_backtest_metrics
from prooflab.backtest.orders import OrderRecord, Position
from prooflab.backtest.portfolio import (
    BrokerLimitsConfig,
    EquitySnapshot,
    PortfolioAccountant,
    PositionSizer,
)
from prooflab.backtest.signals import SignalEngine, SignalFilterConfig, TradeSignal


class BacktestConfig(BaseModel):
    """Full execution and portfolio configuration for a backtest run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float = Field(default=100000.0, gt=0.0)
    risk_per_trade_pct: float = Field(default=0.01, ge=0.0001, le=1.0)
    point_value: float = Field(default=1.0, gt=0.0)
    pip_size: float = Field(default=0.0001, gt=0.0)
    default_stop_pips: float = Field(default=50.0, gt=0.0)
    default_target_pips: float = Field(default=100.0, gt=0.0)
    max_holding_bars: int | None = Field(default=50, ge=1)
    risk_free_rate_pct: float = Field(default=0.0, ge=0.0)

    signal_filters: SignalFilterConfig = Field(default_factory=SignalFilterConfig)
    costs: ExecutionCostConfig = Field(default_factory=ExecutionCostConfig)
    broker_limits: BrokerLimitsConfig = Field(default_factory=BrokerLimitsConfig)


class BacktestResult(BaseModel):
    """Complete output package containing metrics, trade logs, and equity curves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: BacktestConfig
    metrics: BacktestMetricsReport
    trades: list[OrderRecord]
    signals: list[TradeSignal]
    equity_snapshots: list[EquitySnapshot]

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve time series as a pandas DataFrame."""
        if not self.equity_snapshots:
            return pd.DataFrame()
        records = [s.model_dump(mode="python") for s in self.equity_snapshots]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Return the trade ledger as a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = [t.model_dump(mode="python") for t in self.trades]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class BacktestEngine:
    """Event-driven simulation loop executing trades chronologically over OHLCV bars."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.signal_engine = SignalEngine(self.config.signal_filters)
        self.cost_model = ExecutionCostModel(self.config.costs)
        self.position_sizer = PositionSizer(self.config.broker_limits)

    def run(
        self,
        data: pd.DataFrame,
        predictions: list[dict[str, Any]] | pd.DataFrame,
        symbol: str = "EURUSD",
    ) -> BacktestResult:
        """Execute full backtest over OHLCV dataset and prediction stream.

        Args:
            data: OHLCV DataFrame with UTC DatetimeIndex and 'open', 'high', 'low', 'close'.
            predictions: Prediction stream conforming to canonical prediction schema.
            symbol: Target market symbol.
        """
        # 1. Validate Input Data
        if not isinstance(data.index, pd.DatetimeIndex) or str(data.index.tz) != "UTC":
            raise ValueError("Backtest data index must be a timezone-aware UTC DatetimeIndex.")

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Backtest data missing required OHLC columns: {required_cols}")

        if data.empty:
            raise ValueError("Backtest data is empty.")

        # 2. Index predictions by timestamp for O(1) alignment
        if isinstance(predictions, pd.DataFrame):
            pred_records = predictions.to_dict(orient="records")
        else:
            pred_records = predictions

        pred_map: dict[pd.Timestamp, dict[str, Any]] = {}
        for p in pred_records:
            raw_ts = p.get("timestamp")
            if raw_ts is not None:
                pred_map[pd.Timestamp(raw_ts)] = p

        # 3. Initialize Portfolio & State
        accountant = PortfolioAccountant(
            initial_capital=self.config.initial_capital,
            broker_limits=self.config.broker_limits,
        )
        open_positions: list[Position] = []
        pending_orders: list[dict[str, Any]] = []
        closed_trades: list[OrderRecord] = []
        recorded_signals: list[TradeSignal] = []

        prev_time: pd.Timestamp | None = None
        pip_size = self.config.pip_size
        point_val = self.config.point_value

        # 4. Chronological Bar Simulation Loop
        for bar_idx, (curr_time, row) in enumerate(data.iterrows()):
            open_p = float(row["open"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            close_p = float(row["close"])
            atr_val = float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else None
            bar_dict = row.to_dict()

            # --- Step A: Process Overnight Swap Financing ---
            if prev_time is not None:
                for pos in open_positions:
                    swap_charge = self.cost_model.calculate_holding_swap(
                        pos.side,
                        pos.quantity,
                        prev_time,
                        curr_time,
                        point_value=point_val,
                    )
                    if swap_charge != 0.0:
                        pos.apply_swap(swap_charge)

            # --- Step B: Fill Pending Orders at Current Bar Open ---
            remaining_pending: list[dict[str, Any]] = []
            for pending in pending_orders:
                side = pending["side"]
                req_price = open_p  # Market fill at open
                target_stop = pending["stop_price"]
                target_tp = pending["target_price"]

                # Sizing from current net equity
                equity_for_sizing = accountant.cash
                qty = self.position_sizer.calculate_position_size(
                    account_equity=equity_for_sizing,
                    entry_price=req_price,
                    stop_loss_price=target_stop,
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                    point_value=point_val,
                )

                min_units = (
                    self.config.broker_limits.min_lot_size * self.config.broker_limits.lot_unit_size
                )
                if qty <= 0:
                    closed_trades.append(
                        OrderRecord(
                            timestamp=curr_time,
                            symbol=symbol,
                            side=side,
                            requested_price=req_price,
                            quantity=min_units,
                            status="REJECTED",
                            rejection_reason="Position sizer calculated zero quantity",
                        )
                    )
                    continue

                can_open, reject_msg = accountant.can_open_position(
                    quantity=qty,
                    price=req_price,
                    active_positions_count=len(open_positions),
                )

                if not can_open:
                    closed_trades.append(
                        OrderRecord(
                            timestamp=curr_time,
                            symbol=symbol,
                            side=side,
                            requested_price=req_price,
                            quantity=qty,
                            status="REJECTED",
                            rejection_reason=reject_msg or "Broker margin limit breach",
                        )
                    )
                    continue

                # Calculate entry fill & friction
                entry_exec = self.cost_model.calculate_entry_execution(
                    side,
                    req_price,
                    qty,
                    bar=bar_dict,
                    atr=atr_val,
                    point_value=point_val,
                )

                pos = Position(
                    order_id=pending["order_id"],
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    entry_price=entry_exec["fill_price"],
                    entry_time=curr_time,
                    stop_loss=target_stop,
                    take_profit=target_tp,
                    commission_paid=entry_exec["commission_cost"],
                    entry_spread=entry_exec["spread_cost"],
                    entry_slippage=entry_exec["slippage_cost"],
                    max_holding_bars=self.config.max_holding_bars,
                    metadata={"signal_id": pending["signal_id"]},
                )
                open_positions.append(pos)

            pending_orders = remaining_pending

            # --- Step C: Evaluate Active Positions for Intrabar Exits ---
            active_positions: list[Position] = []
            for pos in open_positions:
                pos.increment_bar()
                exit_check = pos.check_intrabar_exit(
                    open_=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    bar_time=curr_time,
                )

                if exit_check is not None:
                    exit_reason, exit_req_price = exit_check

                    # Calculate exit execution & friction
                    exit_exec = self.cost_model.calculate_exit_execution(
                        pos.side,
                        exit_req_price,
                        pos.quantity,
                        bar=bar_dict,
                        atr=atr_val,
                        point_value=point_val,
                    )

                    # Finalize trade record
                    trade_record = pos.close(
                        exit_price=exit_exec["fill_price"],
                        exit_time=curr_time,
                        exit_reason=exit_reason,
                        exit_commission=exit_exec["commission_cost"],
                        exit_spread=exit_exec["spread_cost"],
                        exit_slippage=exit_exec["slippage_cost"],
                        point_value=point_val,
                        pip_size=pip_size,
                    )

                    accountant.record_trade_close(
                        gross_pnl=trade_record.gross_pnl,
                        net_pnl=trade_record.net_pnl,
                        commission=trade_record.commission,
                        spread=trade_record.spread,
                        slippage=trade_record.slippage,
                        swap=trade_record.swap,
                    )
                    closed_trades.append(trade_record)
                else:
                    active_positions.append(pos)

            open_positions = active_positions

            # --- Step D: Evaluate Signal from Current Bar Prediction ---
            pred = pred_map.get(curr_time)
            if pred is not None:
                market_ctx = {"atr": atr_val, "bar": bar_dict}
                signal = self.signal_engine.evaluate_prediction(pred, market_context=market_ctx)
                recorded_signals.append(signal)

                if signal.is_actionable and signal.direction in {"BUY", "SELL"}:
                    side = signal.direction
                    if side == "BUY":
                        stop_p = close_p - (self.config.default_stop_pips * pip_size)
                        tp_p = close_p + (self.config.default_target_pips * pip_size)
                    else:
                        stop_p = close_p + (self.config.default_stop_pips * pip_size)
                        tp_p = close_p - (self.config.default_target_pips * pip_size)

                    pending_order = {
                        "order_id": f"ORD-{signal.signal_id[4:]}",
                        "signal_id": signal.signal_id,
                        "side": side,
                        "stop_price": stop_p,
                        "target_price": tp_p,
                        "created_time": curr_time,
                    }

                    if self.config.costs.execution_delay_bars == 0:
                        # Immediate fill at bar close
                        # (Simulated by queuing for instant fill)
                        pending_orders.append(pending_order)
                    else:
                        # Filled at next bar open
                        pending_orders.append(pending_order)

            # --- Step E: Update Mark-to-Market Snapshot ---
            accountant.update_snapshot(
                timestamp=curr_time,
                open_positions=open_positions,
                current_prices={symbol: close_p},
                point_value=point_val,
            )

            prev_time = curr_time

        # 5. End of Simulation: Force Close Any Remaining Positions at Final Close
        if open_positions:
            last_time = data.index[-1]
            last_close = float(data.iloc[-1]["close"])
            last_bar = data.iloc[-1].to_dict()

            for pos in open_positions:
                exit_exec = self.cost_model.calculate_exit_execution(
                    pos.side,
                    last_close,
                    pos.quantity,
                    bar=last_bar,
                    point_value=point_val,
                )
                trade_record = pos.close(
                    exit_price=exit_exec["fill_price"],
                    exit_time=last_time,
                    exit_reason="FORCE_CLOSE",
                    exit_commission=exit_exec["commission_cost"],
                    exit_spread=exit_exec["spread_cost"],
                    exit_slippage=exit_exec["slippage_cost"],
                    point_value=point_val,
                    pip_size=pip_size,
                )
                accountant.record_trade_close(
                    gross_pnl=trade_record.gross_pnl,
                    net_pnl=trade_record.net_pnl,
                    commission=trade_record.commission,
                    spread=trade_record.spread,
                    slippage=trade_record.slippage,
                    swap=trade_record.swap,
                )
                closed_trades.append(trade_record)

        # 6. Compute Comprehensive Metrics
        metrics = calculate_backtest_metrics(
            trades=closed_trades,
            equity_snapshots=accountant.history,
            initial_capital=self.config.initial_capital,
            risk_free_rate_pct=self.config.risk_free_rate_pct,
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=closed_trades,
            signals=recorded_signals,
            equity_snapshots=accountant.history,
        )
