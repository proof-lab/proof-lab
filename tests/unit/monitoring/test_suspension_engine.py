"""Unit tests for automatic strategy suspension engine and champion/challenger retraining governance."""

from __future__ import annotations

import pytest

from prooflab.monitoring.audit import AuditEventType, AuditLogger
from prooflab.monitoring.drift import DriftReport, DriftStatus
from prooflab.monitoring.health import ComponentHealth, ComponentStatus, SystemHealthReport
from prooflab.monitoring.suspension import (
    AutomaticSuspensionEngine,
    CandidateStatus,
    ChampionChallengerRegistry,
    SuspensionRuleConfig,
    SuspensionTrigger,
)
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState


def test_suspension_engine_nominal_conditions() -> None:
    """Test that nominal operating conditions do not trigger strategy suspension."""
    engine = AutomaticSuspensionEngine()
    decision = engine.evaluate(
        daily_loss_pct=0.01,
        current_drawdown_pct=0.05,
        current_spread_pips=1.2,
    )
    assert not decision.should_suspend
    assert len(decision.triggers) == 0
    assert "nominal" in decision.reason.lower()


def test_suspension_engine_daily_loss_trigger() -> None:
    """Test that daily loss exceeding threshold triggers suspension."""
    engine = AutomaticSuspensionEngine(SuspensionRuleConfig(max_daily_loss_pct=0.03))
    decision = engine.evaluate(daily_loss_pct=0.035)

    assert decision.should_suspend
    assert SuspensionTrigger.DAILY_LOSS_LIMIT in decision.triggers
    assert "Daily loss" in decision.reason


def test_suspension_engine_drawdown_trigger() -> None:
    """Test that maximum drawdown breach triggers suspension."""
    engine = AutomaticSuspensionEngine(SuspensionRuleConfig(max_drawdown_pct=0.10))
    decision = engine.evaluate(current_drawdown_pct=0.12)

    assert decision.should_suspend
    assert SuspensionTrigger.MAX_DRAWDOWN in decision.triggers
    assert "Drawdown" in decision.reason


def test_suspension_engine_excessive_spread_trigger() -> None:
    """Test that abnormal spread spikes trigger suspension."""
    engine = AutomaticSuspensionEngine(SuspensionRuleConfig(max_spread_pips=5.0))
    decision = engine.evaluate(current_spread_pips=6.2)

    assert decision.should_suspend
    assert SuspensionTrigger.EXCESSIVE_SPREAD in decision.triggers
    assert "Abnormal spread" in decision.reason


def test_suspension_engine_drift_and_health_triggers() -> None:
    """Test critical drift report and component health failure triggers."""
    engine = AutomaticSuspensionEngine()

    drift_report = DriftReport(
        overall_status=DriftStatus.SUSPENDED,
        summary="Severe statistical divergence across key features",
    )

    health_report = SystemHealthReport(
        overall_status=ComponentStatus.FAILED,
        components={
            "broker": ComponentHealth(
                component_name="broker",
                status=ComponentStatus.FAILED,
                message="Lost connection to MT5 terminal",
            )
        },
        summary="Critical failure in broker component",
    )

    decision = engine.evaluate(drift_report=drift_report, health_report=health_report)
    assert decision.should_suspend
    assert SuspensionTrigger.FEATURE_DRIFT in decision.triggers
    assert SuspensionTrigger.BROKER_DISCONNECT in decision.triggers


def test_suspension_enforcement_on_lifecycle_manager(tmp_path) -> None:
    """Test that suspension engine transitions lifecycle state and emits audit logs."""
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_file=audit_file)

    lifecycle = StrategyLifecycleManager(strategy_id="strat_alpha")
    lifecycle.transition_to(StrategyLifecycleState.VALIDATED, reason="Passed", authorized_by="qa")
    lifecycle.transition_to(
        StrategyLifecycleState.PAPER_TRADING, reason="Passed", authorized_by="qa"
    )
    lifecycle.transition_to(
        StrategyLifecycleState.APPROVED,
        reason="Passed",
        authorized_by="gov",
        explicit_human_approval=True,
    )
    lifecycle.transition_to(
        StrategyLifecycleState.LIVE_ENABLED,
        reason="Passed",
        authorized_by="gov",
        explicit_human_approval=True,
    )

    engine = AutomaticSuspensionEngine()
    decision = engine.evaluate(daily_loss_pct=0.05)
    assert decision.should_suspend

    applied = engine.enforce_suspension(
        decision=decision,
        lifecycle_manager=lifecycle,
        audit_logger=logger,
        actor="circuit_breaker",
    )
    assert applied
    assert lifecycle.current_state == StrategyLifecycleState.SUSPENDED

    records = logger.records
    assert len(records) == 1
    assert records[0].event_type == AuditEventType.MODEL_SUSPENDED
    assert records[0].strategy_id == "strat_alpha"




def test_champion_challenger_governance_requires_human_approval() -> None:
    """Test champion/challenger retraining governance strictly enforces human approval."""
    registry = ChampionChallengerRegistry()
    registry.register_champion(strategy_id="strat_101", champion_model_id="v1_champion")

    # Register newly retrained candidate as challenger
    challenger = registry.register_challenger(
        strategy_id="strat_101",
        challenger_model_id="v2_candidate",
        validation_metrics={"sharpe": 2.1, "win_rate": 0.62},
    )
    assert challenger.challenger_status == CandidateStatus.VALIDATED
    assert challenger.champion_model_id == "v1_champion"

    # Attempting to promote without explicit human approval MUST raise PermissionError
    with pytest.raises(PermissionError, match="Explicit human approval is required"):
        registry.promote_challenger(strategy_id="strat_101", explicit_human_approval=False)

    # Explicit human approval successfully promotes challenger
    promoted = registry.promote_challenger(
        strategy_id="strat_101",
        explicit_human_approval=True,
        actor="chief_risk_officer",
    )
    assert promoted.champion_model_id == "v2_candidate"
    assert promoted.challenger_model_id is None
    assert promoted.challenger_status == CandidateStatus.PROMOTED
