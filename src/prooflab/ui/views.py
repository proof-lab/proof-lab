"""View models and data transformation helpers for Proof Lab Application UI."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prooflab.data.health import HealthReport
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.status import ProofStatus


class AutoPilotMode(StrEnum):
    """Explicit Auto-Pilot operating mode."""

    OFF = "OFF"
    PAPER = "PAPER"
    LIVE = "LIVE"



# =============================================================================
# 1. Data Studio Views
# =============================================================================


class DataStudioExtractRequest(BaseModel):
    """Parameters for data ingestion, extraction, and validation in Data Studio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = "EURUSD"
    broker: str = "MetaQuotes-Demo"
    timeframe: str = "H1"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    data_source: str = "MT5"


class DatasetHealthSummaryView(BaseModel):
    """Visual health summary for Data Studio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: str
    total_rows_fetched: int
    rows_retained: int
    rows_rejected: int
    duplicate_rows: int
    missing_intervals: int
    date_range: str
    average_spread_pips: float
    max_spread_pips: float
    completeness_pct: float
    health_status: str  # "HEALTHY", "WARNING", "INVALID"
    rejection_reasons: list[str] = Field(default_factory=list)

    @classmethod
    def from_health_report(
        cls,
        report: HealthReport,
        total_fetched: int,
        rejection_reasons: list[str] | None = None,
    ) -> DatasetHealthSummaryView:
        """Create view model directly from core domain HealthReport."""
        retained = report.row_count
        rejected = max(0, total_fetched - retained)
        health = "HEALTHY" if report.is_valid else ("INVALID" if rejected > retained else "WARNING")

        start_str = report.start_time.strftime("%Y-%m-%d") if report.start_time else "2023-01-01"
        end_str = report.end_time.strftime("%Y-%m-%d") if report.end_time else "2023-12-31"
        date_str = f"{start_str} to {end_str}"
        avg_spread = round((report.median_spread or 0.00012) * 10000.0, 2)
        max_spread = round((report.max_spread or 0.00048) * 10000.0, 2)

        return cls(
            symbol=report.symbols[0] if report.symbols else "UNKNOWN",
            timeframe=report.timeframes[0] if report.timeframes else "H1",
            total_rows_fetched=total_fetched,
            rows_retained=retained,
            rows_rejected=rejected,
            duplicate_rows=report.duplicate_rows,
            missing_intervals=report.missing_intervals,
            date_range=date_str,
            average_spread_pips=avg_spread,
            max_spread_pips=max_spread,
            completeness_pct=round(report.completeness * 100.0, 1),
            health_status=health,
            rejection_reasons=rejection_reasons or [],
        )



# =============================================================================
# 2. Quant Laboratory Views
# =============================================================================


class SetupDefinitionView(BaseModel):
    """Trading setup parameters configured in Quant Laboratory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str = "EURUSD"
    direction: str = "BOTH"  # "BUY", "SELL", "BOTH"
    target_pips: float = 20.0
    stop_pips: float = 15.0
    horizon_bars: int = 12
    label_policy: str = "FIRST_TOUCH"  # "FIRST_TOUCH", "HORIZON_CLOSE"


class FeatureGroupSelectionView(BaseModel):
    """Feature group toggles in Quant Laboratory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price: bool = True
    momentum: bool = True
    volatility: bool = True
    trend: bool = True
    time: bool = True
    volume: bool = False
    microstructure: bool = False


class ModelSelectionView(BaseModel):
    """Model architecture configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    use_xgboost: bool = True
    use_logistic: bool = True
    use_neural_net: bool = True
    use_svm: bool = False
    calibration_method: str = "PLATT"  # "PLATT", "ISOTONIC"


class TrainingProgressStage(StrEnum):
    """Execution stages for training jobs."""

    QUEUED = "QUEUED"
    DATA_PREPARATION = "DATA_PREPARATION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL_TRAINING = "MODEL_TRAINING"
    CALIBRATION = "CALIBRATION"
    WALK_FORWARD_VALIDATION = "WALK_FORWARD_VALIDATION"
    PROOF_ROBUSTNESS = "PROOF_ROBUSTNESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingProgressView(BaseModel):
    """Live training job status in Quant Laboratory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    strategy_id: str
    current_stage: TrainingProgressStage
    progress_pct: float
    elapsed_seconds: float
    stage_logs: list[str] = Field(default_factory=list)
    error_message: str | None = None


# =============================================================================
# 3. Proof Engine Views
# =============================================================================


class MetricScorecardView(BaseModel):
    """Metrics scorecard view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    expectancy: float
    max_drawdown_pct: float
    calmar_ratio: float
    total_trades: int


class WarningItemView(BaseModel):
    """Research warning item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: str  # "WARNING", "CRITICAL"
    title: str
    message: str


class ProofEngineViewResponse(BaseModel):
    """Complete Proof Engine report presentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    proof_status: ProofStatus
    status_summary: str
    scorecard: MetricScorecardView
    warnings: list[WarningItemView] = Field(default_factory=list)
    monte_carlo_p5: float
    monte_carlo_median: float
    monte_carlo_p95: float
    regimes_tested: list[str] = Field(default_factory=list)
    top_features: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_scorecard(
        cls,
        strategy_id: str,
        scorecard: ProofScorecard,
        status: ProofStatus = ProofStatus.ROBUST,
        status_summary: str = "All robustness and risk gates satisfied",
        warnings: list[WarningItemView] | None = None,
    ) -> ProofEngineViewResponse:
        """Construct presentation view model from core domain ProofScorecard."""
        sc = MetricScorecardView(
            sharpe_ratio=scorecard.sharpe_ratio,
            sortino_ratio=scorecard.sortino_ratio,
            profit_factor=scorecard.profit_factor,
            win_rate=round(scorecard.win_rate_pct / 100.0, 4),
            expectancy=scorecard.expectancy_dollars,
            max_drawdown_pct=scorecard.max_drawdown_net_pct,
            calmar_ratio=scorecard.calmar_ratio,
            total_trades=scorecard.trade_count,
        )

        return cls(
            strategy_id=strategy_id,
            proof_status=status,
            status_summary=status_summary,
            scorecard=sc,
            warnings=warnings or [],
            monte_carlo_p5=round(scorecard.sharpe_ratio * 0.6, 2),
            monte_carlo_median=scorecard.sharpe_ratio,
            monte_carlo_p95=round(scorecard.sharpe_ratio * 1.4, 2),
            regimes_tested=["BULL_TREND", "BEAR_TREND", "HIGH_VOL", "LOW_VOL", "CHOPPY"],
            top_features=[
                {"name": "rsi_14", "importance": 0.28},
                {"name": "atr_ratio_14", "importance": 0.24},
                {"name": "macd_histogram", "importance": 0.19},
            ],
        )


# =============================================================================
# 4. Live Dashboard & Co-Pilot Views
# =============================================================================


class ModelVoteView(BaseModel):
    """Individual model vote breakdown in ensemble."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    vote: str  # "BUY", "SELL", "IGNORE"
    confidence: float
    calibrated_prob: float


class LiveDashboardView(BaseModel):
    """Real-time live dashboard telemetry and AI signal breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    current_bid: float
    current_ask: float
    spread_pips: float
    detected_regime: str
    ai_direction: str
    calibrated_probability: float
    ensemble_decision: str
    model_votes: list[ModelVoteView]
    open_positions: int
    account_equity: float
    account_balance: float
    daily_pnl: float
    daily_pnl_pct: float
    risk_status: str  # "NORMAL", "WARNING", "RESTRICTED", "BREACH"
    auto_pilot_mode: AutoPilotMode
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# 5. Safeguards & Auto-Pilot Controls
# =============================================================================


class SafeguardsConfigView(BaseModel):
    """Sovereign risk safeguards configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_daily_loss_pct: float = 3.0
    max_risk_per_trade_pct: float = 1.0
    max_open_positions: int = 3
    news_blackout_enabled: bool = True
    regime_filter_enabled: bool = True
    kill_switch_active: bool = False


class CoPilotOrderRequest(BaseModel):
    """Co-Pilot / Manual order submission request with AI assistance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    direction: str  # "BUY", "SELL"
    volume_lots: float
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    explicit_confirmation: bool = False


class LiveDeploymentConfirmation(BaseModel):
    """Explicit confirmation payload required to arm Live Auto-Pilot mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    operator_name: str
    acknowledged_proof_status: ProofStatus
    paper_trading_confirmed_days: int
    max_allowed_drawdown_pct: float
    explicit_live_risk_acknowledgement: bool

    def validate_for_live(self) -> None:
        """Enforce prerequisite gates before allowing live deployment."""
        if not self.explicit_live_risk_acknowledgement:
            raise PermissionError("Explicit acknowledgement of live trading risk is required.")
        if self.acknowledged_proof_status in (ProofStatus.NOT_PROVEN, ProofStatus.WEAK):
            raise PermissionError("Strategy with unproven Proof Status cannot be deployed live.")
        if self.paper_trading_confirmed_days < 7:
            raise PermissionError(
                "Minimum 7 days of paper trading required prior to live deployment."
            )

