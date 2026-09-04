"""Unit tests for prooflab.backtest.costs (Commission, Slippage, Swap, ExecutionCostModel)."""

from datetime import UTC, datetime

import pytest

from prooflab.backtest.costs import (
    CommissionModel,
    CommissionModelConfig,
    ExecutionCostConfig,
    ExecutionCostModel,
    SlippageModel,
    SlippageModelConfig,
    SwapModel,
    SwapModelConfig,
)


def test_commission_models() -> None:
    # 1. Per lot ($3.50 per 100k): 200,000 units -> $7.00
    m_lot = CommissionModel(
        CommissionModelConfig(commission_type="per_lot", rate=3.50, lot_size=100000.0)
    )
    assert pytest.approx(m_lot.calculate_commission(200000.0, 1.1000)) == 7.00

    # 2. Per unit ($0.01 per share): 500 shares -> $5.00
    m_unit = CommissionModel(CommissionModelConfig(commission_type="per_unit", rate=0.01))
    assert pytest.approx(m_unit.calculate_commission(500.0, 150.0)) == 5.00

    # 3. Per transaction ($4.95 flat)
    m_tx = CommissionModel(CommissionModelConfig(commission_type="per_transaction", rate=4.95))
    assert pytest.approx(m_tx.calculate_commission(100.0, 50.0)) == 4.95

    # 4. Percentage (0.05% of notional): 100,000 * 1.1000 = $110,000 -> $55.00
    m_pct = CommissionModel(CommissionModelConfig(commission_type="percentage", rate=0.05))
    assert pytest.approx(m_pct.calculate_commission(100000.0, 1.1000)) == 55.00


def test_slippage_models() -> None:
    # Fixed pips (0.5 pip = 0.00005)
    m_slip = SlippageModel(SlippageModelConfig(mode="fixed_pips", fixed_pips=0.5, pip_size=0.0001))

    # BUY orders slip up
    fill_buy, slip_buy = m_slip.calculate_slippage_price("BUY", 1.10000)
    assert pytest.approx(fill_buy) == 1.10005
    assert pytest.approx(slip_buy) == 0.00005

    # SELL orders slip down
    fill_sell, slip_sell = m_slip.calculate_slippage_price("SELL", 1.10000)
    assert pytest.approx(fill_sell) == 1.09995
    assert pytest.approx(slip_sell) == 0.00005

    # Slippage monetary cost for 100,000 units
    cost = m_slip.calculate_slippage_cost(100000.0, slip_buy)
    assert pytest.approx(cost) == 5.00

    # Volatility-dependent slippage (5% of ATR 0.0020 = 0.00010 = 1.0 pip)
    m_vol = SlippageModel(
        SlippageModelConfig(mode="volatility_dependent", atr_fraction=0.05, pip_size=0.0001)
    )
    fill_v, slip_v = m_vol.calculate_slippage_price("BUY", 1.10000, atr=0.0020)
    assert pytest.approx(slip_v) == 0.00010
    assert pytest.approx(fill_v) == 1.10010


def test_swap_model_rollover_and_wednesday_triple() -> None:
    # Long swap = -0.5 pips/day (/lot), Short swap = -0.2 pips/day (/lot)
    m_swap = SwapModel(SwapModelConfig(
        long_swap_pips_per_day=-0.5,
        short_swap_pips_per_day=-0.2,
        pip_size=0.0001,
        rollover_hour_utc=22,
        triple_swap_weekday=2,  # Wednesday
    ))

    # 1. No rollover crossed during the day
    t1 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)  # Monday 10:00
    t2 = datetime(2026, 3, 2, 18, 0, tzinfo=UTC)  # Monday 18:00
    assert m_swap.calculate_swap("BUY", 100000.0, t1, t2) == 0.0

    # 2. Normal rollover crossed (Monday 18:00 to Tuesday 02:00)
    t3 = datetime(2026, 3, 3, 2, 0, tzinfo=UTC)  # Tuesday 02:00
    cost_long = m_swap.calculate_swap("BUY", 100000.0, t2, t3)
    assert pytest.approx(cost_long) == 5.00

    cost_short = m_swap.calculate_swap("SELL", 100000.0, t2, t3)
    assert pytest.approx(cost_short) == 2.00

    # 3. Wednesday triple swap (Wednesday 21:00 to Thursday 01:00)
    t_wed = datetime(2026, 3, 4, 21, 0, tzinfo=UTC)
    t_wed_roll = datetime(2026, 3, 4, 23, 0, tzinfo=UTC)  # Wednesday night
    triple_cost = m_swap.calculate_swap("BUY", 100000.0, t_wed, t_wed_roll)
    assert pytest.approx(triple_cost) == 15.00  #  * 3 =


def test_unified_execution_cost_model() -> None:
    exec_model = ExecutionCostModel(ExecutionCostConfig())

    # Entry evaluation for BUY order
    entry_res = exec_model.calculate_entry_execution("BUY", 1.10000, 100000.0)
    assert "fill_price" in entry_res
    assert "spread_cost" in entry_res
    assert "commission_cost" in entry_res
    assert "slippage_cost" in entry_res
    assert entry_res["total_entry_friction"] > 0

    # Exit evaluation for Long position (closes via SELL)
    exit_res = exec_model.calculate_exit_execution("BUY", 1.10500, 100000.0)
    assert exit_res["fill_price"] < 1.10500  # Slips downwards
    assert exit_res["total_exit_friction"] > 0
