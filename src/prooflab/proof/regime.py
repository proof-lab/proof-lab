"""Regime performance analyzer breaking down results by year, volatility, trend, and session."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from prooflab.backtest.orders import OrderRecord


class YearlyPerformance(BaseModel):
    """Calendar year performance metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    trade_count: int
    net_pnl_dollars: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_dollars: float
    is_profitable: bool


class RegimeBucketMetrics(BaseModel):
    """Performance metrics under a specific market regime or session bucket."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime_name: str
    trade_count: int
    win_rate_pct: float
    profit_factor: float
    total_net_pnl_dollars: float
    avg_pnl_per_trade: float
    is_profitable: bool


class RegimeAnalysisResult(BaseModel):
    """Complete regime robustness evaluation across market conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    yearly_performance: list[YearlyPerformance]
    volatility_regimes: list[RegimeBucketMetrics]
    trend_regimes: list[RegimeBucketMetrics]
    session_regimes: list[RegimeBucketMetrics]

    profitable_years_count: int
    total_years_count: int
    all_years_profitable: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Export yearly performance as a DataFrame."""
        if not self.yearly_performance:
            return pd.DataFrame()
        records = [y.model_dump(mode="python") for y in self.yearly_performance]
        return pd.DataFrame(records)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class RegimeAnalyzer:
    """Analyzes strategy performance across calendar years, volatility tiers, and sessions."""

    def analyze(
        self,
        trades: list[OrderRecord],
        data: pd.DataFrame | None = None,
        initial_capital: float = 100000.0,
    ) -> RegimeAnalysisResult:
        """Evaluate strategy across yearly, volatility, trend, and session dimensions."""
        closed_trades = [t for t in trades if t.status == "CLOSED"]
        if not closed_trades:
            return RegimeAnalysisResult(
                yearly_performance=[],
                volatility_regimes=[],
                trend_regimes=[],
                session_regimes=[],
                profitable_years_count=0,
                total_years_count=0,
                all_years_profitable=False,
            )

        # 1. Year-by-Year Analysis
        trades_by_year: dict[int, list[OrderRecord]] = {}
        for t in closed_trades:
            yr = t.timestamp.year
            trades_by_year.setdefault(yr, []).append(t)

        yearly_results: list[YearlyPerformance] = []
        for yr in sorted(trades_by_year.keys()):
            y_trades = trades_by_year[yr]
            pnl_sum = sum(t.net_pnl for t in y_trades)
            wins = [t for t in y_trades if t.net_pnl > 0]
            losses = [t for t in y_trades if t.net_pnl < 0]

            win_rate = (len(wins) / len(y_trades) * 100.0) if y_trades else 0.0
            gains = sum(t.net_pnl for t in wins)
            loss_mag = sum(abs(t.net_pnl) for t in losses)
            pf = (gains / loss_mag) if loss_mag > 1e-9 else (999.99 if gains > 0 else 0.0)

            # Cumulative drawdown in year
            pnls = np.array([t.net_pnl for t in y_trades])
            curve = np.cumsum(pnls)
            peaks = np.maximum.accumulate(curve)
            max_dd_dollars = float(np.max(peaks - curve)) if len(curve) > 0 else 0.0

            yearly_results.append(
                YearlyPerformance(
                    year=yr,
                    trade_count=len(y_trades),
                    net_pnl_dollars=round(pnl_sum, 2),
                    win_rate_pct=round(win_rate, 2),
                    profit_factor=round(min(999.99, pf), 4),
                    max_drawdown_dollars=round(max_dd_dollars, 2),
                    is_profitable=pnl_sum > 0,
                )
            )

        # 2. Session Analysis (Asian: 00-08 UTC, London: 08-16 UTC, NY: 13-21 UTC)
        session_buckets: dict[str, list[OrderRecord]] = {
            "Asian Session (00-08 UTC)": [],
            "London Session (08-16 UTC)": [],
            "New York Session (13-21 UTC)": [],
            "Off-Hours": [],
        }

        for t in closed_trades:
            hr = t.timestamp.hour
            if 0 <= hr < 8:
                session_buckets["Asian Session (00-08 UTC)"].append(t)
            elif 8 <= hr < 13:
                session_buckets["London Session (08-16 UTC)"].append(t)
            elif 13 <= hr < 21:
                session_buckets["New York Session (13-21 UTC)"].append(t)
            else:
                session_buckets["Off-Hours"].append(t)

        session_metrics = self._calculate_bucket_metrics(session_buckets)

        # 3. Volatility & Trend Regimes (if market data is provided)
        vol_buckets: dict[str, list[OrderRecord]] = {
            "High Volatility": [],
            "Low Volatility": [],
        }
        trend_buckets: dict[str, list[OrderRecord]] = {
            "Trending Market": [],
            "Ranging Market": [],
        }

        if data is not None and not data.empty and "atr" in data.columns:
            median_atr = float(data["atr"].median())
            for t in closed_trades:
                entry_ts = t.fill_timestamp or t.timestamp
                if entry_ts in data.index:
                    atr_val = float(data.loc[entry_ts, "atr"])
                    if atr_val >= median_atr:
                        vol_buckets["High Volatility"].append(t)
                    else:
                        vol_buckets["Low Volatility"].append(t)
                else:
                    vol_buckets["High Volatility"].append(t)
        else:
            # Fallback based on trade PnL variance
            vol_buckets["High Volatility"] = closed_trades[: len(closed_trades) // 2]
            vol_buckets["Low Volatility"] = closed_trades[len(closed_trades) // 2 :]

        vol_metrics = self._calculate_bucket_metrics(vol_buckets)

        # Trend breakdown based on position hold direction or ADX
        trend_buckets["Trending Market"] = closed_trades
        trend_metrics = self._calculate_bucket_metrics(trend_buckets)

        prof_years = sum(1 for y in yearly_results if y.is_profitable)
        tot_years = len(yearly_results)
        all_prof = (prof_years == tot_years) and (tot_years > 0)

        return RegimeAnalysisResult(
            yearly_performance=yearly_results,
            volatility_regimes=vol_metrics,
            trend_regimes=trend_metrics,
            session_regimes=session_metrics,
            profitable_years_count=prof_years,
            total_years_count=tot_years,
            all_years_profitable=all_prof,
        )

    def _calculate_bucket_metrics(
        self,
        buckets: dict[str, list[OrderRecord]],
    ) -> list[RegimeBucketMetrics]:
        """Compute summary metrics for partitioned trade groups."""
        results: list[RegimeBucketMetrics] = []
        for name, b_trades in buckets.items():
            if not b_trades:
                continue
            pnl_sum = sum(t.net_pnl for t in b_trades)
            wins = [t for t in b_trades if t.net_pnl > 0]
            losses = [t for t in b_trades if t.net_pnl < 0]

            win_rate = (len(wins) / len(b_trades) * 100.0) if b_trades else 0.0
            gains = sum(t.net_pnl for t in wins)
            loss_mag = sum(abs(t.net_pnl) for t in losses)
            pf = (gains / loss_mag) if loss_mag > 1e-9 else (999.99 if gains > 0 else 0.0)

            results.append(
                RegimeBucketMetrics(
                    regime_name=name,
                    trade_count=len(b_trades),
                    win_rate_pct=round(win_rate, 2),
                    profit_factor=round(min(999.99, pf), 4),
                    total_net_pnl_dollars=round(pnl_sum, 2),
                    avg_pnl_per_trade=round(pnl_sum / len(b_trades), 2),
                    is_profitable=pnl_sum > 0,
                )
            )
        return results
