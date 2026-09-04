"""Automatic strategy suspension engine and Champion/Challenger retraining governance."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prooflab.monitoring.audit import AuditEventType, AuditLogger, AuditSeverity
from prooflab.monitoring.drift import DriftReport, DriftStatus
from prooflab.monitoring.health import ComponentStatus, SystemHealthReport
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState

logger = logging.getLogger(__name__)


class SuspensionTrigger(StrEnum):
    """Classification of automatic suspension causes."""

    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    FEATURE_DRIFT = "FEATURE_DRIFT"
    PREDICTION_ANOMALY = "PREDICTION_ANOMALY"
    DATA_FEED_FAILURE = "DATA_FEED_FAILURE"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"
    EXCESSIVE_SPREAD = "EXCESSIVE_SPREAD"
    MANUAL_OPERATOR = "MANUAL_OPERATOR"


class SuspensionRuleConfig(BaseModel):
    """Threshold configurations governing automatic suspension triggers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_spread_pips: float = 10.0
    suspend_on_critical_drift: bool = True
    suspend_on_health_failure: bool = True


class SuspensionDecision(BaseModel):
    """Outcome of evaluating suspension rules against live telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    should_suspend: bool
    triggers: list[SuspensionTrigger] = Field(default_factory=list)
    reason: str = "Operating within normal safety parameters"
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class AutomaticSuspensionEngine:
    """Evaluates live market, risk, drift, and health telemetry to enforce strategy suspension."""

    def __init__(self, config: SuspensionRuleConfig | None = None) -> None:

        self.config = config or SuspensionRuleConfig()

    def evaluate(
        self,
        daily_loss_pct: float = 0.0,
        current_drawdown_pct: float = 0.0,
        current_spread_pips: float = 1.0,
        drift_report: DriftReport | None = None,
        health_report: SystemHealthReport | None = None,
    ) -> SuspensionDecision:
        """Evaluate telemetry across drawdown, loss, spread, drift, and component health."""
        triggers: list[SuspensionTrigger] = []
        reasons: list[str] = []

        # 1. Daily Loss Check
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            triggers.append(SuspensionTrigger.DAILY_LOSS_LIMIT)
            reasons.append(
                f"Daily loss ({daily_loss_pct * 100:.1f}%) reached limit "
                f"({self.config.max_daily_loss_pct * 100:.1f}%)"
            )

        # 2. Max Drawdown Check
        if current_drawdown_pct >= self.config.max_drawdown_pct:
            triggers.append(SuspensionTrigger.MAX_DRAWDOWN)
            reasons.append(
                f"Drawdown ({current_drawdown_pct * 100:.1f}%) breached limit "
                f"({self.config.max_drawdown_pct * 100:.1f}%)"
            )

        # 3. Excessive Spread Check
        if current_spread_pips >= self.config.max_spread_pips:
            triggers.append(SuspensionTrigger.EXCESSIVE_SPREAD)
            reasons.append(
                f"Abnormal spread ({current_spread_pips:.1f} pips) exceeds limit "
                f"({self.config.max_spread_pips:.1f} pips)"
            )

        # 4. Critical Drift Check
        if self.config.suspend_on_critical_drift and drift_report:
            if drift_report.overall_status == DriftStatus.SUSPENDED:
                triggers.append(SuspensionTrigger.FEATURE_DRIFT)
                reasons.append(f"Critical statistical drift detected: {drift_report.summary}")

        # 5. Component Health Failure Check
        if self.config.suspend_on_health_failure and health_report:
            if health_report.overall_status == ComponentStatus.FAILED:
                triggers.append(SuspensionTrigger.BROKER_DISCONNECT)
                reasons.append(f"Critical health failure: {health_report.summary}")

        should_suspend = len(triggers) > 0
        reason_text = "; ".join(reasons) if should_suspend else "All conditions nominal"

        return SuspensionDecision(
            should_suspend=should_suspend,
            triggers=triggers,
            reason=reason_text,
            details={
                "daily_loss_pct": daily_loss_pct,
                "current_drawdown_pct": current_drawdown_pct,
                "current_spread_pips": current_spread_pips,
            },
        )

    def enforce_suspension(
        self,
        decision: SuspensionDecision,
        lifecycle_manager: StrategyLifecycleManager,
        audit_logger: AuditLogger | None = None,
        actor: str = "system",
    ) -> bool:
        """Apply suspension transition to lifecycle manager and emit audit event."""
        if not decision.should_suspend:
            return False

        if lifecycle_manager.current_state in {
            StrategyLifecycleState.LIVE_ENABLED,
            StrategyLifecycleState.PAPER_TRADING,
            StrategyLifecycleState.APPROVED,
        }:
            lifecycle_manager.transition_to(
                StrategyLifecycleState.SUSPENDED,
                reason=decision.reason,
                authorized_by=actor,
            )
            logger.warning(
                "Strategy %s automatically SUSPENDED: %s",
                lifecycle_manager.strategy_id,
                decision.reason,
            )

            if audit_logger:
                audit_logger.log(
                    event_type=AuditEventType.MODEL_SUSPENDED,
                    message=f"Automatic strategy suspension: {decision.reason}",
                    severity=AuditSeverity.CRITICAL,
                    actor=actor,
                    strategy_id=lifecycle_manager.strategy_id,
                    metadata={"triggers": [t.value for t in decision.triggers]},
                )
            return True

        return False


class CandidateStatus(StrEnum):
    """Candidate model approval progression status."""

    TRAINING = "TRAINING"
    VALIDATED = "VALIDATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class ChampionChallengerModel(BaseModel):
    """Champion/Challenger deployment model tracking active live and retrained candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    champion_model_id: str
    challenger_model_id: str | None = None
    challenger_status: CandidateStatus = CandidateStatus.TRAINING
    challenger_metrics: dict[str, Any] = Field(default_factory=dict)
    updated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChampionChallengerRegistry:
    """Ensures retrained challenger models cannot trade live without explicit human approval."""

    def __init__(self) -> None:
        self._registry: dict[str, ChampionChallengerModel] = {}

    def register_champion(self, strategy_id: str, champion_model_id: str) -> None:
        """Register the baseline production champion model."""
        self._registry[strategy_id] = ChampionChallengerModel(
            strategy_id=strategy_id,
            champion_model_id=champion_model_id,
        )

    def register_challenger(
        self,
        strategy_id: str,
        challenger_model_id: str,
        validation_metrics: dict[str, Any] | None = None,
    ) -> ChampionChallengerModel:
        """Register a newly retrained candidate model as a challenger."""
        existing = self._registry.get(strategy_id)
        champion_id = existing.champion_model_id if existing else "initial_champion"

        record = ChampionChallengerModel(
            strategy_id=strategy_id,
            champion_model_id=champion_id,
            challenger_model_id=challenger_model_id,
            challenger_status=CandidateStatus.VALIDATED,
            challenger_metrics=validation_metrics or {},
        )
        self._registry[strategy_id] = record
        logger.info(
            "Registered retrained challenger %s for strategy %s (Champion remains %s)",
            challenger_model_id,
            strategy_id,
            champion_id,
        )
        return record

    def promote_challenger(
        self,
        strategy_id: str,
        explicit_human_approval: bool = False,
        actor: str = "human_operator",
        audit_logger: AuditLogger | None = None,
    ) -> ChampionChallengerModel:
        """Promote challenger to champion; strictly requires explicit human approval."""
        if not explicit_human_approval:
            raise PermissionError(
                "Automatic promotion of retrained candidate models to live execution is "
                "prohibited. Explicit human approval is required to promote a "
                "Challenger to Champion."
            )



        record = self._registry.get(strategy_id)
        if not record or not record.challenger_model_id:
            raise ValueError(f"No challenger model registered for strategy {strategy_id}.")

        new_champion = record.challenger_model_id
        promoted = ChampionChallengerModel(
            strategy_id=strategy_id,
            champion_model_id=new_champion,
            challenger_model_id=None,
            challenger_status=CandidateStatus.PROMOTED,
            challenger_metrics=record.challenger_metrics,
        )
        self._registry[strategy_id] = promoted

        logger.info(
            "Promoted challenger %s to champion for strategy %s by %s",
            new_champion,
            strategy_id,
            actor,
        )

        if audit_logger:
            audit_logger.log(
                event_type=AuditEventType.MODEL_DEPLOYED,
                message=f"Promoted challenger {new_champion} to production champion",
                severity=AuditSeverity.INFO,
                actor=actor,
                strategy_id=strategy_id,
            )
        return promoted

    def get_model(self, strategy_id: str) -> ChampionChallengerModel | None:
        """Retrieve champion/challenger tracking record."""
        return self._registry.get(strategy_id)
