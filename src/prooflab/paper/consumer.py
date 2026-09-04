"""Live market data consumer enforcing real-time quality, staleness, and gap checks."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DataQualityIssue(StrEnum):
    """Classification of live market data quality anomalies."""

    STALE_DATA = "STALE_DATA"
    OUT_OF_ORDER_TIMESTAMP = "OUT_OF_ORDER_TIMESTAMP"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    PRICE_GAP_EXCESSIVE = "PRICE_GAP_EXCESSIVE"
    TIMESTAMP_GAP_EXCESSIVE = "TIMESTAMP_GAP_EXCESSIVE"
    ABNORMAL_SPREAD = "ABNORMAL_SPREAD"
    INVALID_PRICE_OR_VOLUME = "INVALID_PRICE_OR_VOLUME"


class LiveTick(BaseModel):
    """Immutable real-time market tick."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp_utc: datetime
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    volume: float = Field(default=0.0, ge=0.0)
    pip_size: float = Field(default=0.0001, gt=0.0)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pips(self) -> float:
        return (self.ask - self.bid) / self.pip_size


class LiveBar(BaseModel):
    """Immutable completed OHLCV bar from market feed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp_utc: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(default=0.0, ge=0.0)
    spread: float = Field(default=0.0001, ge=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "spread": self.spread,
        }


class ConsumerConfig(BaseModel):
    """Thresholds governing live data health and anomaly detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_staleness_seconds: float = Field(default=300.0, ge=1.0)
    max_spread_pips: float = Field(default=5.0, ge=0.1)
    max_price_gap_pips: float = Field(default=30.0, ge=1.0)
    max_time_gap_seconds: float = Field(default=7200.0, ge=1.0)
    pip_size: float = Field(default=0.0001, gt=0.0)


class MarketDataConsumer:
    """Consumes live ticks and bars while flagging data corruption and anomalies."""

    def __init__(self, config: ConsumerConfig | None = None) -> None:
        self.config = config or ConsumerConfig()
        self._last_ticks: dict[str, LiveTick] = {}
        self._last_bars: dict[str, LiveBar] = {}
        self._bar_history: dict[str, list[LiveBar]] = {}

    def process_tick(
        self,
        tick: LiveTick,
        wall_clock_utc: datetime | None = None,
    ) -> tuple[bool, list[DataQualityIssue]]:
        """Validate an incoming tick against temporal and price integrity rules."""
        issues: list[DataQualityIssue] = []
        clock = wall_clock_utc or datetime.now(UTC)

        # 1. Invalid Bid/Ask
        if tick.bid <= 0 or tick.ask <= 0 or tick.ask < tick.bid:
            issues.append(DataQualityIssue.INVALID_PRICE_OR_VOLUME)

        # 2. Staleness Check
        staleness = (clock - tick.timestamp_utc).total_seconds()
        if staleness > self.config.max_staleness_seconds:
            issues.append(DataQualityIssue.STALE_DATA)

        # 3. Abnormal Spread Check
        if tick.spread_pips > self.config.max_spread_pips:
            issues.append(DataQualityIssue.ABNORMAL_SPREAD)

        # 4. Sequence & Gap Checks vs Previous Tick
        prev_tick = self._last_ticks.get(tick.symbol)
        if prev_tick:
            # Duplicate
            if tick.timestamp_utc == prev_tick.timestamp_utc:
                issues.append(DataQualityIssue.DUPLICATE_TIMESTAMP)
            # Out of order
            elif tick.timestamp_utc < prev_tick.timestamp_utc:
                issues.append(DataQualityIssue.OUT_OF_ORDER_TIMESTAMP)
            else:
                # Excessive Time Gap
                time_gap = (tick.timestamp_utc - prev_tick.timestamp_utc).total_seconds()
                if time_gap > self.config.max_time_gap_seconds:
                    issues.append(DataQualityIssue.TIMESTAMP_GAP_EXCESSIVE)

                # Excessive Price Gap
                price_gap_pips = abs(tick.mid_price - prev_tick.mid_price) / self.config.pip_size
                if price_gap_pips > self.config.max_price_gap_pips:
                    issues.append(DataQualityIssue.PRICE_GAP_EXCESSIVE)

        is_valid = len(issues) == 0
        if is_valid or DataQualityIssue.ABNORMAL_SPREAD in issues:
            # We record tick if timestamp is monotonic
            if not prev_tick or tick.timestamp_utc > prev_tick.timestamp_utc:
                self._last_ticks[tick.symbol] = tick

        return is_valid, issues

    def process_bar(
        self,
        bar: LiveBar,
        wall_clock_utc: datetime | None = None,
    ) -> tuple[bool, list[DataQualityIssue]]:
        """Validate an incoming bar and append to rolling historical window."""
        issues: list[DataQualityIssue] = []
        clock = wall_clock_utc or datetime.now(UTC)

        # 1. OHLC Sanity
        if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
            issues.append(DataQualityIssue.INVALID_PRICE_OR_VOLUME)

        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0 or bar.volume < 0:
            issues.append(DataQualityIssue.INVALID_PRICE_OR_VOLUME)

        # 2. Staleness Check
        staleness = (clock - bar.timestamp_utc).total_seconds()
        if staleness > self.config.max_staleness_seconds:
            issues.append(DataQualityIssue.STALE_DATA)

        # 3. Sequence & Gap Checks vs Previous Bar
        prev_bar = self._last_bars.get(bar.symbol)
        if prev_bar:
            if bar.timestamp_utc == prev_bar.timestamp_utc:
                issues.append(DataQualityIssue.DUPLICATE_TIMESTAMP)
            elif bar.timestamp_utc < prev_bar.timestamp_utc:
                issues.append(DataQualityIssue.OUT_OF_ORDER_TIMESTAMP)
            else:
                price_gap_pips = abs(bar.open - prev_bar.close) / self.config.pip_size
                if price_gap_pips > self.config.max_price_gap_pips:
                    issues.append(DataQualityIssue.PRICE_GAP_EXCESSIVE)

        is_valid = len(issues) == 0
        if is_valid:
            self._last_bars[bar.symbol] = bar
            if bar.symbol not in self._bar_history:
                self._bar_history[bar.symbol] = []
            self._bar_history[bar.symbol].append(bar)

        return is_valid, issues

    def get_last_tick(self, symbol: str) -> LiveTick | None:
        return self._last_ticks.get(symbol)

    def get_last_bar(self, symbol: str) -> LiveBar | None:
        return self._last_bars.get(symbol)

    def get_bars_dataframe(self, symbol: str) -> pd.DataFrame:
        """Return history buffer for symbol formatted as a canonical DataFrame."""
        bars = self._bar_history.get(symbol, [])
        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "spread"]
            )
        data = [b.to_dict() for b in bars]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
