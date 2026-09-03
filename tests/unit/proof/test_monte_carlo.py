"""Unit tests for prooflab.proof.monte_carlo (Monte Carlo Sequence Reshuffling)."""

import numpy as np

from prooflab.proof.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
)


def test_monte_carlo_reshuffle_and_bootstrap() -> None:
    # 50 trades with positive net expectancy (avg +, with noise)
    rng = np.random.default_rng(42)
    pnls = rng.normal(loc=100.0, scale=300.0, size=50)

    # 1. Reshuffle (without replacement)
    engine_reshuffle = MonteCarloEngine(
        MonteCarloConfig(n_simulations=1000, resampling_mode="reshuffle", random_seed=42)
    )
    res_reshuffle = engine_reshuffle.run_simulation(pnls)

    assert isinstance(res_reshuffle, MonteCarloResult)
    assert res_reshuffle.n_simulations == 1000
    assert res_reshuffle.trade_count == 50
    assert res_reshuffle.median_return_pct > 0
    p5 = res_reshuffle.percentile_5_return_pct
    p95 = res_reshuffle.percentile_95_return_pct
    assert p5 <= res_reshuffle.median_return_pct <= p95
    assert res_reshuffle.median_max_drawdown_pct >= 0
    assert res_reshuffle.percentile_95_max_drawdown_pct >= res_reshuffle.median_max_drawdown_pct

    # 2. Bootstrap (with replacement)
    engine_boot = MonteCarloEngine(
        MonteCarloConfig(n_simulations=1000, resampling_mode="bootstrap", random_seed=42)
    )
    res_boot = engine_boot.run_simulation(pnls)

    assert isinstance(res_boot, MonteCarloResult)
    assert res_boot.n_simulations == 1000
    assert res_boot.probability_of_loss_pct < 50.0  # Positive edge strategy

    # JSON export
    json_str = res_reshuffle.to_json()
    assert "median_return_pct" in json_str
    assert "probability_of_ruin_pct" in json_str


def test_monte_carlo_empty_trades() -> None:
    engine = MonteCarloEngine()
    res = engine.run_simulation([])

    assert res.trade_count == 0
    assert res.probability_of_loss_pct == 100.0
