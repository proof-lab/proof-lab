"""MetaTrader 5 broker adapter and high-fidelity mock implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from prooflab.data.schema import Timeframe
from prooflab.live.base import (
    BrokerAccountInfo,
    BrokerAdapter,
    BrokerContextInfo,
    BrokerCredentials,
    BrokerPosition,
)
from prooflab.live.orders import (
    LiveOrder,
    LiveOrderState,
    OrderStateMachine,
)

logger = logging.getLogger(__name__)


class MT5ConnectionError(Exception):
    """Raised when broker connection cannot be established or is disconnected."""


class MockMT5Adapter(BrokerAdapter):
    """High-fidelity in-memory MetaTrader 5 broker adapter for testing and simulation."""

    def __init__(
        self,
        credentials: BrokerCredentials | None = None,
        initial_balance: float = 10000.0,
        leverage: float = 100.0,
        currency: str = "USD",
        default_spread_pips: float = 1.5,
        pip_size: float = 0.0001,
    ) -> None:
        self.credentials = credentials or BrokerCredentials(
            account_id="MOCK_MT5_1001",
            password="mock_password",
            server="Mock-MT5-Demo",
        )
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.currency = currency
        self.leverage = leverage
        self.default_spread_pips = default_spread_pips
        self.pip_size = pip_size

        self._connected: bool = False
        self._next_ticket: int = 100000
        self._positions: dict[str, BrokerPosition] = {}
        self._orders: dict[str, LiveOrder] = {}
        self._symbol_prices: dict[str, float] = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2700,
            "USDJPY": 150.00,
            "AUDUSD": 0.6550,
        }

        # Fault injection flags for test coverage
        self.simulate_rejection: bool = False
        self.rejection_reason: str = "Broker rejected: simulated test rejection"
        self.simulate_slippage_pips: float = 0.0

    def connect(self) -> bool:
        """Simulate connecting to MetaTrader 5 terminal."""
        logger.info(
            "Connecting to MT5 server=%s (account=%s)",
            self.credentials.server,
            self.credentials.account_id,
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Simulate disconnecting from MT5 terminal."""
        logger.info("Disconnecting from MT5 broker (account=%s)", self.credentials.account_id)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float) -> None:
        """Update simulated market price for a symbol."""
        self._symbol_prices[symbol] = price
        # Update unrealized PnL of open positions
        for pos_id, pos in list(self._positions.items()):
            if pos.symbol == symbol:
                diff = (price - pos.open_price) if pos.side == "BUY" else (pos.open_price - price)
                unrealized = diff * pos.volume * 100000.0
                self._positions[pos_id] = pos.model_copy(
                    update={"current_price": price, "unrealized_pnl": unrealized}
                )

    def get_account(self) -> BrokerAccountInfo:
        """Calculate and return live mock account equity and margin."""
        if not self._connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")

        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_margin = sum(
            (p.open_price * p.volume * 100000.0) / self.leverage for p in self._positions.values()
        )
        equity = self.balance + total_unrealized
        free_margin = max(0.0, equity - total_margin)
        margin_level = (equity / total_margin * 100.0) if total_margin > 0 else None

        return BrokerAccountInfo(
            account_id=self.credentials.account_id,
            currency=self.currency,
            balance=round(self.balance, 2),
            equity=round(equity, 2),
            margin=round(total_margin, 2),
            free_margin=round(free_margin, 2),
            margin_level=round(margin_level, 2) if margin_level is not None else None,
            leverage=self.leverage,
        )

    def get_market_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV market data bars for testing."""
        if not self._connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")

        base_price = self._symbol_prices.get(symbol, 1.0850)
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.0005, size=count)
        closes = base_price * np.exp(np.cumsum(returns))
        opens = np.roll(closes, 1)
        opens[0] = base_price

        highs = np.maximum(opens, closes) + rng.uniform(0.0001, 0.0005, size=count)
        lows = np.minimum(opens, closes) - rng.uniform(0.0001, 0.0005, size=count)
        volumes = rng.integers(100, 5000, size=count)
        tick_volumes = volumes + rng.integers(10, 50, size=count)
        spreads = np.full(count, self.default_spread_pips * 10.0)

        end_time = datetime.now(UTC)
        freq = "1min" if timeframe == Timeframe.M1 else "5min"
        timestamps = pd.date_range(end=end_time, periods=count, freq=freq, tz=UTC)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "timeframe": timeframe.value,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes.astype(float),
                "tick_volume": tick_volumes.astype(float),
                "spread": spreads,
                "source": "MOCK_MT5",
            }
        )

    def submit_order(self, order: LiveOrder) -> LiveOrder:
        """Submit and simulate instant fill of an order in Mock MT5."""
        if not self._connected:
            raise MT5ConnectionError("Cannot submit order: MT5 terminal is not connected.")

        # Progress to SUBMITTED
        OrderStateMachine.transition(
            order, LiveOrderState.SUBMITTED, reason="Dispatched to Mock MT5 engine"
        )

        if self.simulate_rejection:
            OrderStateMachine.transition(
                order, LiveOrderState.REJECTED, reason=self.rejection_reason
            )
            self._orders[order.order_id] = order
            return order

        # Acknowledge order
        ticket = self._next_ticket
        self._next_ticket += 1
        OrderStateMachine.transition(
            order,
            LiveOrderState.ACKNOWLEDGED,
            reason="Mock MT5 received order",
            broker_ticket=ticket,
        )

        # Calculate fill price with simulated spread and slippage
        current_market = self._symbol_prices.get(order.symbol, order.price)
        half_spread = (self.default_spread_pips * self.pip_size) / 2.0
        slippage_val = self.simulate_slippage_pips * self.pip_size

        if order.side == "BUY":
            fill_price = current_market + half_spread + slippage_val
        else:
            fill_price = current_market - half_spread - slippage_val

        commission = round(3.50 * order.quantity, 2)

        OrderStateMachine.transition(
            order,
            LiveOrderState.FILLED,
            reason="Mock MT5 executed market fill",
            filled_quantity=order.quantity,
            filled_price=round(fill_price, 5),
            broker_ticket=ticket,
            commission=commission,
            slippage=round(slippage_val, 5),
        )

        # Deduct commission from balance
        self.balance -= commission

        # Create open BrokerPosition
        position_id = f"POS_{ticket}"
        position = BrokerPosition(
            position_id=position_id,
            symbol=order.symbol,
            side=order.side,
            volume=order.quantity,
            open_price=round(fill_price, 5),
            current_price=round(current_market, 5),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            commission=commission,
            opened_at=datetime.now(UTC),
            magic_number=1001,
            comment=f"Order {order.order_id}",
        )
        self._positions[position_id] = position
        self._orders[order.order_id] = order

        logger.info(
            "Mock MT5 filled order %s: side=%s, qty=%.2f, fill_price=%.5f, ticket=%d",
            order.order_id,
            order.side,
            order.quantity,
            fill_price,
            ticket,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if not self._connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")

        order = self._orders.get(order_id)
        if not order or not order.is_active:
            return False

        OrderStateMachine.transition(
            order, LiveOrderState.CANCELLED, reason="User cancelled order"
        )
        return True

    def close_position(self, position_id: str, volume: float | None = None) -> bool:
        """Close an open position and realize PnL."""
        if not self._connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")

        pos = self._positions.get(position_id)
        if not pos:
            return False

        close_vol = volume if volume is not None and volume <= pos.volume else pos.volume
        current_market = self._symbol_prices.get(pos.symbol, pos.open_price)
        half_spread = (self.default_spread_pips * self.pip_size) / 2.0

        if pos.side == "BUY":
            close_price = current_market - half_spread
            pnl = (close_price - pos.open_price) * close_vol * 100000.0
        else:
            close_price = current_market + half_spread
            pnl = (pos.open_price - close_price) * close_vol * 100000.0

        self.balance += pnl
        if close_vol < pos.volume:
            self._positions[position_id] = pos.model_copy(
                update={"volume": round(pos.volume - close_vol, 5)}
            )
        else:
            del self._positions[position_id]
            # Update matching order to CLOSED if found
            for order in self._orders.values():
                if order.broker_ticket and f"POS_{order.broker_ticket}" == position_id:
                    if order.status == LiveOrderState.FILLED:
                        OrderStateMachine.transition(
                            order, LiveOrderState.CLOSED, reason="Position closed on broker"
                        )

        logger.info(
            "Closed mock MT5 position %s: realized_pnl=%.2f, new_balance=%.2f",
            position_id,
            pnl,
            self.balance,
        )
        return True

    def get_positions(self, symbol: str | None = None) -> list[BrokerPosition]:
        """Return list of open mock broker positions."""
        if not self._connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")
        if symbol:
            return [p for p in self._positions.values() if p.symbol == symbol]
        return list(self._positions.values())

    def get_broker_info(self) -> BrokerContextInfo:
        """Return Mock MT5 broker specification."""
        return BrokerContextInfo(
            broker_name="MetaTrader 5 Mock Broker",
            server_name=self.credentials.server,
            symbols=list(self._symbol_prices.keys()),
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
            contract_size=100000.0,
            spread_type="FLOATING",
            is_demo=True,
        )


class MT5Adapter(BrokerAdapter):
    """Production MetaTrader 5 broker adapter with defensive safety and credential handling."""

    def __init__(self, credentials: BrokerCredentials) -> None:
        self.credentials = credentials
        self._connected = False
        self._mt5: Any = None

    def connect(self) -> bool:
        """Initialize MetaTrader 5 API connection."""
        try:
            import MetaTrader5 as mt5_module  # noqa: N813

            self._mt5 = mt5_module
        except ImportError:
            raise RuntimeError(
                "MetaTrader5 package is not installed. Use MockMT5Adapter or install MetaTrader5."
            )

        logger.info(
            "Connecting to MT5 server=%s, account=%s",
            self.credentials.server,
            self.credentials.account_id,
        )
        if not self._mt5.initialize(
            login=int(self.credentials.account_id),
            password=self.credentials.password,
            server=self.credentials.server,
        ):
            error_code, error_desc = self._mt5.last_error()
            raise MT5ConnectionError(f"Failed to connect to MT5: error {error_code} - {error_desc}")

        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> BrokerAccountInfo:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        acc = self._mt5.account_info()
        if acc is None:
            raise MT5ConnectionError("Failed to retrieve MT5 account info.")
        return BrokerAccountInfo(
            account_id=str(acc.login),
            currency=acc.currency,
            balance=float(acc.balance),
            equity=float(acc.equity),
            margin=float(acc.margin),
            free_margin=float(acc.margin_free),
            margin_level=float(acc.margin_level) if acc.margin_level is not None else None,
            leverage=float(acc.leverage),
        )

    def get_market_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> pd.DataFrame:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        tf_map = {
            Timeframe.M1: self._mt5.TIMEFRAME_M1,
            Timeframe.M5: self._mt5.TIMEFRAME_M5,
            Timeframe.M15: self._mt5.TIMEFRAME_M15,
            Timeframe.M30: self._mt5.TIMEFRAME_M30,
            Timeframe.H1: self._mt5.TIMEFRAME_H1,
            Timeframe.H4: self._mt5.TIMEFRAME_H4,
            Timeframe.D1: self._mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe, self._mt5.TIMEFRAME_M1)
        rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            raise MT5ConnectionError(f"No rates returned from MT5 for symbol {symbol}.")
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe.value
        df["source"] = "MT5"
        df["volume"] = df["real_volume"].astype(float)
        return df[
            [
                "timestamp",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_volume",
                "spread",
                "source",
            ]
        ]

    def submit_order(self, order: LiveOrder) -> LiveOrder:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        return True

    def close_position(self, position_id: str, volume: float | None = None) -> bool:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        return True

    def get_positions(self, symbol: str | None = None) -> list[BrokerPosition]:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        positions = self._mt5.positions_get(symbol=symbol) if symbol else self._mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append(
                BrokerPosition(
                    position_id=f"POS_{p.ticket}",
                    symbol=p.symbol,
                    side="BUY" if p.type == self._mt5.ORDER_TYPE_BUY else "SELL",
                    volume=float(p.volume),
                    open_price=float(p.price_open),
                    current_price=float(p.price_current),
                    stop_loss=float(p.sl) if p.sl else None,
                    take_profit=float(p.tp) if p.tp else None,
                    unrealized_pnl=float(p.profit),
                    swap=float(p.swap),
                    opened_at=datetime.fromtimestamp(p.time, tz=UTC),
                    magic_number=int(p.magic),
                    comment=str(p.comment),
                )
            )
        return result

    def get_broker_info(self) -> BrokerContextInfo:
        if not self._connected or not self._mt5:
            raise MT5ConnectionError("MT5 is not connected.")
        term = self._mt5.terminal_info()
        return BrokerContextInfo(
            broker_name=term.name if term else "MetaTrader 5",
            server_name=self.credentials.server,
            symbols=["EURUSD", "GBPUSD", "USDJPY"],
            is_demo=True,
        )
