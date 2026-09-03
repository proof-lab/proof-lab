"""Unit tests for prooflab.backtest.costs.spread (Spread Modelling)."""

import pytest

from prooflab.backtest.costs.spread import SpreadModel, SpreadModelConfig


def test_fixed_spread_normal_scenario() -> None:
    config = SpreadModelConfig(mode="fixed", scenario="normal", fixed_pips=1.5, pip_size=0.0001)
    model = SpreadModel(config)

    spread_price = model.calculate_spread()
    assert pytest.approx(spread_price) == 0.00015

    # Side cost for 100,000 units (half spread = 0.75 pips = .50)
    side_cost = model.calculate_side_spread_cost(100000.0, spread_price)
    assert pytest.approx(side_cost) == 7.50


def test_historical_spread_modes() -> None:
    config = SpreadModelConfig(mode="historical", scenario="normal", pip_size=0.0001)
    model = SpreadModel(config)

    # 1. Bar with bid/ask
    bar_bid_ask = {"bid": 1.10000, "ask": 1.10018}
    spread = model.calculate_spread(bar_bid_ask)
    assert pytest.approx(spread) == 0.00018

    # 2. Bar with spread column
    bar_spread = {"spread": 0.00012}
    spread2 = model.calculate_spread(bar_spread)
    assert pytest.approx(spread2) == 0.00012

    # 3. Bar missing spread fields falls back to fixed_pips
    spread_fallback = model.calculate_spread({})
    assert pytest.approx(spread_fallback) == 0.00010


def test_multiplier_and_stress_modes() -> None:
    # Multiplier: 1.0 pip * 2.0 = 2.0 pips
    model_mult = SpreadModel(SpreadModelConfig(mode="multiplier", multiplier=2.0, fixed_pips=1.0))
    assert pytest.approx(model_mult.calculate_spread()) == 0.00020

    # Stress mode: (1.0 pip * 2.5) + 2.0 pips = 4.5 pips
    model_stress = SpreadModel(
        SpreadModelConfig(
            mode="stress",
            fixed_pips=1.0,
            stress_multiplier=2.5,
            stress_additive_pips=2.0,
        )
    )
    assert pytest.approx(model_stress.calculate_spread()) == 0.00045

    # Stress mode with ATR expansion
    spread_with_atr = model_stress.calculate_spread(atr=0.0020)
    assert spread_with_atr > 0.00045


def test_scenario_scaling() -> None:
    # Base: 1.0 pip
    # Conservative scenario: 1.0 pip * 1.5 = 1.5 pips
    model_cons = SpreadModel(
        SpreadModelConfig(
            mode="fixed",
            scenario="conservative",
            fixed_pips=1.0,
            conservative_multiplier=1.5,
        )
    )
    assert pytest.approx(model_cons.calculate_spread()) == 0.00015

    # Stress scenario: 1.0 pip * 2.5 = 2.5 pips
    model_stress_scen = SpreadModel(
        SpreadModelConfig(
            mode="fixed",
            scenario="stress",
            fixed_pips=1.0,
            stress_multiplier=2.5,
        )
    )
    assert pytest.approx(model_stress_scen.calculate_spread()) == 0.00025
