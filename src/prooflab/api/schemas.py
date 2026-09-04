"""Pydantic request and response schemas for the Proof Lab REST API."""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from prooflab.api.jobs import JobStatus


class SystemHealthResponse(BaseModel):
    """System health and operational status response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = "OK"
    version: str = "0.1.0"
    timestamp_utc: AwareDatetime
    active_jobs: int = 0
    environment: str = "production"


class SystemVersionResponse(BaseModel):
    """Application version and component build information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "0.1.0"
    api_version: str = "v1"
    quantitative_core: str = "0.1.0"


class JobResponse(BaseModel):
    """Response payload detailing a background quantitative job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    job_type: str
    status: JobStatus
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None


# --- Data Schemas ---
class DatasetSummaryResponse(BaseModel):
    """Summary record of a persisted historical dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    symbol: str
    timeframe: str
    row_count: int
    start_time: AwareDatetime
    end_time: AwareDatetime
    checksum: str


class ValidateDataRequest(BaseModel):
    """Request payload to validate a series of OHLCV bars."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: str
    bars: list[dict[str, Any]]


class DataHealthResponse(BaseModel):
    """Health diagnostic report for a historical dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    is_valid: bool
    total_records: int
    quality_score: float
    issues: list[str] = Field(default_factory=list)


# --- Feature Schemas ---
class FeatureItemResponse(BaseModel):
    """Registered feature metadata specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    family: str
    description: str
    lookback_period: int = 0


class FeaturePresetResponse(BaseModel):
    """Standard predefined feature configuration set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preset_name: str
    feature_count: int
    features: list[str]


# --- Experiments & Training Schemas ---
class TrainModelRequest(BaseModel):
    """Parameters to initiate an asynchronous model training job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(default="majority_classifier")
    dataset_id: str
    feature_preset: str = Field(default="PRICE_ONLY")
    target_pips: float = Field(default=20.0, gt=0.0)
    stop_pips: float = Field(default=10.0, gt=0.0)
    horizon_bars: int = Field(default=5, gt=0)
    model_params: dict[str, Any] = Field(default_factory=dict)


class ModelSummaryResponse(BaseModel):
    """Information summary for a trained model artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    model_name: str
    created_at: AwareDatetime
    features: list[str]
    classes: list[int]
    is_calibrated: bool = False


# --- Backtest & Proof Schemas ---
class RunBacktestRequest(BaseModel):
    """Parameters to execute a strategy backtesting simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    dataset_id: str
    initial_capital: float = Field(default=100000.0, gt=0.0)
    slippage_pips: float = Field(default=0.1, ge=0.0)
    commission_per_unit: float = Field(default=0.00001, ge=0.0)


class ProofScorecardRequest(BaseModel):
    """Parameters to generate a multi-dimensional proof robustness report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    dataset_id: str


# --- Strategy Packaging Schemas ---
class ExportStrategyRequest(BaseModel):
    """Parameters to export a strategy package into portable .plb format."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    symbol: str
    timeframe: str
    target_pips: float = Field(default=25.0, gt=0.0)
    stop_pips: float = Field(default=15.0, gt=0.0)
    horizon_bars: int = Field(default=5, gt=0)
    output_filename: str = "strategy.plb"
    author: str = ""
    description: str = ""
    embed_dataset: bool = False


# --- Risk & Live Governance Schemas ---
class EvaluateSignalRequest(BaseModel):
    """Order parameters to evaluate against the sovereign Risk Engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: str
    entry_price: float = Field(gt=0.0)
    stop_loss_price: float = Field(gt=0.0)
    confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    risk_per_trade_pct: float | None = None


class RiskDecisionResponse(BaseModel):
    """Risk engine decision for a candidate order signal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: str
    is_approved: bool
    symbol: str
    order_side: str
    approved_units: float
    message: str | None = None
    rejection_reasons: list[str] = Field(default_factory=list)


class KillSwitchActivateRequest(BaseModel):
    """Emergency kill switch trigger request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=3)
    actor: str = "RiskOfficer"
    policy: str = "HOLD_OPEN"


class KillSwitchResetRequest(BaseModel):
    """Emergency kill switch reset request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=3)
    actor: str = "RiskOfficer"


class LiveEnableRequest(BaseModel):
    """Explicit human approval request to enable live trading mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    confirm: bool = True
    authorized_by: str = Field(min_length=3)
    reason: str = Field(min_length=3)
