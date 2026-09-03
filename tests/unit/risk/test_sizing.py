"""Unit tests for prooflab.risk.sizing (Risk-Based Position Sizer)."""

import pytest

from prooflab.backtest.portfolio import BrokerLimitsConfig
from prooflab.risk.sizing import PositionSizingResult, RiskPositionSizer


def test_position_sizing_standard() -> None:
    sizer = RiskPositionSizer(
        BrokerLimitsConfig(min_lot_size=0.01, max_lot_size=10.0, lot_step=0.01, max_leverage=30.0)
    )

    # 100k equity, 1% risk (,000), 50 pip stop (0.0050 on EURUSD)
    # raw units = 1000 / 0.0050 = 200,000 units = 2.0 lots
    res = sizer.calculate_position_size(
        account_equity=100000.0,
        risk_per_trade_pct=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        point_value=1.0,
        contract_size=100000.0,
    )

    assert isinstance(res, PositionSizingResult)
    assert res.is_valid is True
    assert res.calculated_lots == 2.0
    assert res.calculated_units == 200000.0
    assert res.risk_amount_dollars == 1000.0
    assert pytest.approx(res.nominal_exposure_dollars) == 220000.0
    assert pytest.approx(res.implied_leverage) == 2.2


def test_position_sizing_below_min_lot_rejection() -> None:
    sizer = RiskPositionSizer(BrokerLimitsConfig(min_lot_size=0.1, lot_step=0.01))

    # Very small account (), 1% risk (.00), 50 pip stop (.00/0.1 lot) -> raw lots = 0.02 < 0.1
    res = sizer.calculate_position_size(
        account_equity=100.0,
        risk_per_trade_pct=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0950,
    )

    assert res.is_valid is False
    assert "below broker minimum" in str(res.rejection_reason)


def test_position_sizing_leverage_clamp() -> None:
    # Max leverage 2.0x on $10,000 equity = max $20,000 exposure = ~0.18 lots
    # With a tight 10 pip stop, raw size would exceed 10x leverage
    broker_cfg = BrokerLimitsConfig(min_lot_size=0.01, max_leverage=2.0, lot_step=0.01)
    sizer = RiskPositionSizer(broker_cfg)

    res = sizer.calculate_position_size(
        account_equity=10000.0,
        risk_per_trade_pct=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0990,
    )

    assert res.is_valid is True
    assert res.implied_leverage <= 2.0
    assert res.calculated_lots < 1.0
