"""Monte Carlo trade-order reshuffling and bootstrap simulation risk analytics."""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.orders import OrderRecord


class MonteCarloConfig(BaseModel):
    """Configuration governing Monte Carlo sequence reshuffling and ruin analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_simulations: int = Field(default=1000, ge=100)  # Standard 1,000, high-precision up to 10,000
    initial_capital: float = Field(default=100000.0, gt=0.0)
    ruin_drawdown_threshold_pct: float = Field(default=30.0, ge=1.0, le=100.0)
    random_seed: int = Field(default=42)
    resampling_mode: Literal["reshuffle", "bootstrap"] = "reshuffle"


class MonteCarloResult(BaseModel):
    """Statistical distribution of strategy returns and drawdown under sequence randomness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_simulations: int
    trade_count: int
    resampling_mode: str

    median_return_pct: float
    percentile_5_return_pct: float
    percentile_95_return_pct: float

    median_max_drawdown_pct: float
    percentile_95_max_drawdown_pct: float

    probability_of_loss_pct: float
    probability_of_ruin_pct: float

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class MonteCarloEngine:
    """Executes vectorized Monte Carlo trade-sequence reshuffling and ruin analysis."""

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self.config = config or MonteCarloConfig()

    def run_simulation(
        self,
        trades: list[OrderRecord] | list[float] | np.ndarray,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation across trade net PnL distributions."""
        if isinstance(trades, np.ndarray):
            pnl_arr = trades.astype(float)
        elif isinstance(trades, list):
            if trades and isinstance(trades[0], OrderRecord):
                # Extract net PnL from closed OrderRecords
                pnl_arr = np.array(
                    [t.net_pnl for t in trades if getattr(t, "status", None) == "CLOSED"],
                    dtype=float,
                )
            else:
                pnl_arr = np.array(trades, dtype=float)
        else:
            pnl_arr = np.array([], dtype=float)

        n_trades = len(pnl_arr)
        if n_trades == 0:
            return MonteCarloResult(
                n_simulations=self.config.n_simulations,
                trade_count=0,
                resampling_mode=self.config.resampling_mode,
                median_return_pct=0.0,
                percentile_5_return_pct=0.0,
                percentile_95_return_pct=0.0,
                median_max_drawdown_pct=0.0,
                percentile_95_max_drawdown_pct=0.0,
                probability_of_loss_pct=100.0,
                probability_of_ruin_pct=0.0,
            )

        rng = np.random.default_rng(self.config.random_seed)
        n_sims = self.config.n_simulations
        init_cap = self.config.initial_capital
        ruin_thresh = self.config.ruin_drawdown_threshold_pct

        # Matrix shape: (n_sims, n_trades)
        if self.config.resampling_mode == "bootstrap":
            # Sample with replacement
            indices = rng.integers(0, n_trades, size=(n_sims, n_trades))
            sim_pnl_matrix = pnl_arr[indices]
        else:
            # Reshuffle without replacement
            sim_pnl_matrix = np.tile(pnl_arr, (n_sims, 1))
            for i in range(n_sims):
                rng.shuffle(sim_pnl_matrix[i])

        # Cumulative equity curves: prepend initial capital column
        curves = np.hstack([
            np.full((n_sims, 1), init_cap),
            init_cap + np.cumsum(sim_pnl_matrix, axis=1),
        ])

        # Final returns
        final_equity = curves[:, -1]
        sim_returns_pct = (final_equity - init_cap) / init_cap * 100.0

        # Maximum Drawdown for each path
        running_peaks = np.maximum.accumulate(curves, axis=1)
        drawdowns_dollars = running_peaks - curves
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns_pct = np.where(
                running_peaks > 0, (drawdowns_dollars / running_peaks) * 100.0, 0.0
            )
        max_drawdowns_pct = np.max(drawdowns_pct, axis=1)

        # Statistical Metrics
        med_return = float(np.median(sim_returns_pct))
        p5_return = float(np.percentile(sim_returns_pct, 5.0))
        p95_return = float(np.percentile(sim_returns_pct, 95.0))

        med_dd = float(np.median(max_drawdowns_pct))
        p95_dd = float(np.percentile(max_drawdowns_pct, 95.0))

        loss_count = np.sum(sim_returns_pct <= 0.0)
        prob_loss = float(loss_count / n_sims * 100.0)

        ruin_count = np.sum(max_drawdowns_pct >= ruin_thresh)
        prob_ruin = float(ruin_count / n_sims * 100.0)

        return MonteCarloResult(
            n_simulations=n_sims,
            trade_count=n_trades,
            resampling_mode=self.config.resampling_mode,
            median_return_pct=round(med_return, 4),
            percentile_5_return_pct=round(p5_return, 4),
            percentile_95_return_pct=round(p95_return, 4),
            median_max_drawdown_pct=round(med_dd, 4),
            percentile_95_max_drawdown_pct=round(p95_dd, 4),
            probability_of_loss_pct=round(prob_loss, 2),
            probability_of_ruin_pct=round(prob_ruin, 2),
        )
