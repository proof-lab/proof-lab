"""Broker adapter interface and canonical live execution data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.data.schema import Timeframe


class BrokerCredentials(BaseModel):
    """Secure broker credentials container with automatic sensitive string masking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: str
    password: str = Field(repr=False)
    server: str
    api_token: str | None = Field(default=None, repr=False)

    def __str__(self) -> str:
        return f"BrokerCredentials(account_id='{self.account_id}', server='{self.server}', password='***')"

    def __repr__(self) -> str:
        return self.__str__()


class BrokerAccountInfo(BaseModel):
    """Live account balance, equity, and margin metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: str
    currency: str = "USD"
    balance: float
    equity: float
    margin: float = 0.0
    free_margin: float
    margin_level: float | None = None
    leverage: float = 100.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BrokerPosition(BaseModel):
    """Open market position reported directly by the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    volume: float
    open_price: float
    current_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    opened_at: datetime
    magic_number: int = 0
    comment: str = ""


class BrokerContextInfo(BaseModel):
    """Metadata describing broker contract specifications and trading environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_name: str
    server_name: str
    symbols: list[str]
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    contract_size: float = 100000.0
    spread_type: str = "FLOATING"
    is_demo: bool = True
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BrokerAdapter(ABC):
    """Narrow interface abstracting all broker-specific communication."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker terminal/API."""

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection to broker is currently active and healthy."""

    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> pd.DataFrame:
        """Fetch historical or recent OHLCV bars from the broker."""

    @abstractmethod
    def get_account(self) -> BrokerAccountInfo:
        """Query current live broker account balance and margin state."""

    @abstractmethod
    def submit_order(self, order: Any) -> Any:
        """Submit an order to the broker for live execution."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing pending/active order."""

    @abstractmethod
    def close_position(self, position_id: str, volume: float | None = None) -> bool:
        """Close an open broker position."""

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[BrokerPosition]:
        """Query all open positions currently held on the broker."""

    @abstractmethod
    def get_broker_info(self) -> BrokerContextInfo:
        """Retrieve broker specifications and environment details."""
