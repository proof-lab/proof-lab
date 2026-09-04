"""Unit tests for BrokerAdapter interface, models, and credential safety."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from prooflab.data.schema import Timeframe
from prooflab.live.base import (
    BrokerAccountInfo,
    BrokerAdapter,
    BrokerContextInfo,
    BrokerCredentials,
    BrokerPosition,
)


class DummyBrokerAdapter(BrokerAdapter):
    """Concrete mock adapter to verify ABC contract."""

    def __init__(self, connected: bool = True) -> None:
        self._connected = connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_market_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> pd.DataFrame:
        return pd.DataFrame({"close": [1.1000] * count})

    def get_account(self) -> BrokerAccountInfo:
        return BrokerAccountInfo(
            account_id="ACC_123",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            free_margin=10000.0,
        )

    def submit_order(self, order: object) -> object:
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def close_position(self, position_id: str, volume: float | None = None) -> bool:
        return True

    def get_positions(self, symbol: str | None = None) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                position_id="POS_1",
                symbol="EURUSD",
                side="BUY",
                volume=0.1,
                open_price=1.0850,
                current_price=1.0860,
                opened_at=datetime.now(UTC),
            )
        ]

    def get_broker_info(self) -> BrokerContextInfo:
        return BrokerContextInfo(
            broker_name="MockBroker",
            server_name="MockDemo",
            symbols=["EURUSD", "GBPUSD"],
        )


def test_broker_credentials_protection() -> None:
    """Ensure sensitive credentials never appear in string representations."""
    creds = BrokerCredentials(
        account_id="123456",
        password="SuperSecretPassword123!",
        server="BrokerDemo01",
        api_token="tok_secret_abc123",
    )

    str_repr = str(creds)
    repr_str = repr(creds)

    assert "SuperSecretPassword123!" not in str_repr
    assert "SuperSecretPassword123!" not in repr_str
    assert "tok_secret_abc123" not in str_repr
    assert "tok_secret_abc123" not in repr_str
    assert "123456" in str_repr
    assert "BrokerDemo01" in str_repr


def test_broker_adapter_abc_contract() -> None:
    """Ensure BrokerAdapter cannot be instantiated directly without implementations."""
    with pytest.raises(TypeError):
        BrokerAdapter()  # type: ignore[abstract]

    adapter = DummyBrokerAdapter()
    assert adapter.is_connected()
    assert adapter.connect()
    account = adapter.get_account()
    assert account.balance == 10000.0
    positions = adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "EURUSD"
    info = adapter.get_broker_info()
    assert info.broker_name == "MockBroker"
    adapter.disconnect()
    assert not adapter.is_connected()
