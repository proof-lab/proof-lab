"""Unit tests for UI view models, presentation transformations, and live safety gates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from prooflab.data.health import HealthReport
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.status import ProofStatus
from prooflab.ui.views import (
    DatasetHealthSummaryView,
    FeatureGroupSelectionView,
    LiveDeploymentConfirmation,
    ModelSelectionView,
    ProofEngineViewResponse,
    SafeguardsConfigView,
    SetupDefinitionView,
    TrainingProgressStage,
    TrainingProgressView,
    WarningItemView,
)


def test_dataset_health_summary_view_from_domain() -> None:
    """Test transformation of domain HealthReport into Data Studio view model."""
    report = HealthReport(
        symbols=["EURUSD"],
        timeframes=["H1"],
        sources=["MT5"],
        row_count=5000,
        missing_rows=0,
        duplicate_rows=2,
        invalid_rows=8,
        disordered_rows=0,
        start_time=datetime(2023, 1, 1, tzinfo=UTC),
        end_time=datetime(2023, 6, 30, tzinfo=UTC),
        missing_intervals=0,
        median_spread=0.00014,
        max_spread=0.00050,
        completeness=0.995,
        is_valid=True,
    )

    view = DatasetHealthSummaryView.from_health_report(
        report=report,
        total_fetched=5010,
        rejection_reasons=["8 rows failed high/low check"],
    )

    assert view.symbol == "EURUSD"
    assert view.timeframe == "H1"
    assert view.total_rows_fetched == 5010
    assert view.rows_retained == 5000
    assert view.rows_rejected == 10
    assert view.duplicate_rows == 2
    assert view.health_status == "HEALTHY"
    assert view.completeness_pct == 99.5
    assert len(view.rejection_reasons) == 1


def test_proof_engine_view_response_from_domain() -> None:
    """Test transformation of domain ProofScorecard into Proof Engine presentation response."""
    scorecard = ProofScorecard(
        initial_capital=100000.0,
        final_net_equity=142500.0,
        total_net_return_pct=42.5,
        annualized_return_pct=42.5,
        cagr_pct=42.5,
        profit_factor=2.05,
        sharpe_ratio=2.10,
        sortino_ratio=2.65,
        calmar_ratio=3.10,
        max_drawdown_net_pct=5.2,
        max_drawdown_net_dollars=5200.0,
        expectancy_dollars=0.55,
        win_rate_pct=62.0,
        loss_rate_pct=38.0,
        trade_count=520,
        winning_trades=322,
        losing_trades=198,
        total_costs_paid=1500.0,
        total_spread_paid=1000.0,
        total_commission_paid=400.0,
        total_slippage_paid=100.0,
        total_swap_paid=0.0,
    )
    warning = WarningItemView(
        code="REGIME_SENSITIVITY",
        severity="WARNING",
        title="Regime Sensitivity",
        message="Lower profit factor observed during low volatility periods.",
    )

    view = ProofEngineViewResponse.from_scorecard(
        strategy_id="EURUSD_M15_BREAKOUT",
        scorecard=scorecard,
        status=ProofStatus.ROBUST,
        status_summary="Proof criteria fully satisfied across all walk-forward splits.",
        warnings=[warning],
    )

    assert view.strategy_id == "EURUSD_M15_BREAKOUT"
    assert view.proof_status == ProofStatus.ROBUST
    assert view.scorecard.sharpe_ratio == 2.10
    assert view.scorecard.win_rate == 0.62
    assert view.scorecard.max_drawdown_pct == 5.2
    assert len(view.warnings) == 1
    assert view.warnings[0].code == "REGIME_SENSITIVITY"


def test_live_deployment_confirmation_validation() -> None:
    """Test explicit safety gate validation for Live Auto-Pilot confirmation."""
    # Valid confirmation
    valid_conf = LiveDeploymentConfirmation(
        strategy_id="EURUSD_ALPHA",
        operator_name="lead_quant",
        acknowledged_proof_status=ProofStatus.ROBUST,
        paper_trading_confirmed_days=14,
        max_allowed_drawdown_pct=10.0,
        explicit_live_risk_acknowledgement=True,
    )
    valid_conf.validate_for_live()  # Should not raise

    # Invalid: missing risk acknowledgement
    with pytest.raises(PermissionError, match="Explicit acknowledgement of live trading risk"):
        LiveDeploymentConfirmation(
            strategy_id="EURUSD_ALPHA",
            operator_name="lead_quant",
            acknowledged_proof_status=ProofStatus.ROBUST,
            paper_trading_confirmed_days=14,
            max_allowed_drawdown_pct=10.0,
            explicit_live_risk_acknowledgement=False,
        ).validate_for_live()

    # Invalid: strategy has NOT_PROVEN proof status
    with pytest.raises(PermissionError, match="unproven Proof Status cannot be deployed"):
        LiveDeploymentConfirmation(
            strategy_id="EURUSD_ALPHA",
            operator_name="lead_quant",
            acknowledged_proof_status=ProofStatus.NOT_PROVEN,
            paper_trading_confirmed_days=14,
            max_allowed_drawdown_pct=10.0,
            explicit_live_risk_acknowledgement=True,
        ).validate_for_live()

    # Invalid: insufficient paper trading track record (< 7 days)
    with pytest.raises(PermissionError, match="Minimum 7 days of paper trading"):
        LiveDeploymentConfirmation(
            strategy_id="EURUSD_ALPHA",
            operator_name="lead_quant",
            acknowledged_proof_status=ProofStatus.ROBUST,
            paper_trading_confirmed_days=3,
            max_allowed_drawdown_pct=10.0,
            explicit_live_risk_acknowledgement=True,
        ).validate_for_live()


def test_quant_lab_and_safeguards_views() -> None:
    """Test instantiation and defaults of Quant Lab and Safeguards view models."""
    setup = SetupDefinitionView()
    assert setup.instrument == "EURUSD"
    assert setup.target_pips == 20.0
    assert setup.horizon_bars == 12

    feats = FeatureGroupSelectionView()
    assert feats.price
    assert feats.momentum
    assert not feats.volume

    models = ModelSelectionView()
    assert models.use_xgboost
    assert models.calibration_method == "PLATT"

    safeguards = SafeguardsConfigView()
    assert safeguards.max_daily_loss_pct == 3.0
    assert not safeguards.kill_switch_active
    assert safeguards.news_blackout_enabled

    progress = TrainingProgressView(
        job_id="job_123",
        strategy_id="strat_1",
        current_stage=TrainingProgressStage.MODEL_TRAINING,
        progress_pct=60.0,
        elapsed_seconds=8.5,
    )
    assert progress.current_stage == TrainingProgressStage.MODEL_TRAINING
    assert progress.progress_pct == 60.0
