"""Safety condition monitoring and automated trading pause coordinator."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SafetyPauseReason(StrEnum):
    """Specific cause triggering an automated trading pause."""

    MARKET_DATA_STALE = "MARKET_DATA_STALE"
    BROKER_CONNECTION_LOST = "BROKER_CONNECTION_LOST"
    INVALID_MODEL_ARTIFACT = "INVALID_MODEL_ARTIFACT"
    FEATURE_CALCULATION_FAILURE = "FEATURE_CALCULATION_FAILURE"
    UNEXPECTED_SPREAD_BLOWOUT = "UNEXPECTED_SPREAD_BLOWOUT"
    RISK_LIMITS_EXCEEDED = "RISK_LIMITS_EXCEEDED"
    NEWS_BLACKOUT_ACTIVE = "NEWS_BLACKOUT_ACTIVE"
    LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
    SYSTEM_CLOCK_INVALID = "SYSTEM_CLOCK_INVALID"
    DUPLICATE_SIGNAL_DETECTED = "DUPLICATE_SIGNAL_DETECTED"


class SafetyCheckConfig(BaseModel):
    """Thresholds governing safety pause triggers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_data_staleness_seconds: float = Field(default=300.0, ge=1.0)
    max_spread_pips: float = Field(default=5.0, ge=0.1)
    max_spread_multiplier: float = Field(default=3.0, ge=1.0)
    min_model_confidence: float = Field(default=0.55, ge=0.5, le=1.0)
    max_clock_skew_seconds: float = Field(default=5.0, ge=0.1)
    news_blackout_pre_minutes: int = Field(default=15, ge=0)
    news_blackout_post_minutes: int = Field(default=15, ge=0)


class SafetyCheckResult(BaseModel):
    """Evaluation output detailing whether conditions permit trading or mandate a pause."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_safe: bool
    pause_reasons: list[SafetyPauseReason] = Field(default_factory=list)
    rejection_message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class SafetyMonitor:
    """Monitors live environment health and pauses trading upon detecting hazardous conditions."""

    def __init__(self, config: SafetyCheckConfig | None = None) -> None:
        self.config = config or SafetyCheckConfig()
        self._processed_signal_keys: set[str] = set()

    def check_safety(
        self,
        current_time: datetime,
        data_timestamp: datetime | None = None,
        is_broker_connected: bool = True,
        last_broker_heartbeat: datetime | None = None,
        is_model_valid: bool = True,
        model_error: str | None = None,
        features: dict[str, float] | None = None,
        current_spread_pips: float | None = None,
        normal_spread_pips: float = 1.0,
        scheduled_news_events: list[datetime] | None = None,
        model_confidence: float | None = None,
        reference_time: datetime | None = None,
        symbol: str | None = None,
        signal_id: str | None = None,
        bar_timestamp: datetime | None = None,
    ) -> SafetyCheckResult:
        """Run all safety condition checks and return aggregate safety assessment."""
        reasons: list[SafetyPauseReason] = []
        messages: list[str] = []
        details: dict[str, Any] = {}

        # Ensure current_time is timezone-aware UTC
        if current_time.tzinfo is None:
            reasons.append(SafetyPauseReason.SYSTEM_CLOCK_INVALID)
            messages.append("Current time is timezone-naive; UTC required")

        # 1. System Clock Skew Check
        if reference_time:
            skew = abs((current_time - reference_time).total_seconds())
            if skew > self.config.max_clock_skew_seconds:
                reasons.append(SafetyPauseReason.SYSTEM_CLOCK_INVALID)
                messages.append(
                    f"Clock skew ({skew:.1f}s) exceeds threshold "
                    f"({self.config.max_clock_skew_seconds}s)"
                )
                details["clock_skew_seconds"] = skew

        # 2. Market Data Staleness
        if data_timestamp:
            if data_timestamp.tzinfo is None:
                data_timestamp = data_timestamp.replace(tzinfo=UTC)
            staleness = (current_time - data_timestamp).total_seconds()
            if staleness > self.config.max_data_staleness_seconds:
                reasons.append(SafetyPauseReason.MARKET_DATA_STALE)
                messages.append(
                    f"Market data is stale ({staleness:.0f}s > "
                    f"{self.config.max_data_staleness_seconds:.0f}s)"
                )
                details["data_staleness_seconds"] = staleness

        # 3. Broker Connection
        if not is_broker_connected:
            reasons.append(SafetyPauseReason.BROKER_CONNECTION_LOST)
            messages.append("Broker connection is lost")
        elif last_broker_heartbeat:
            hb_lag = (current_time - last_broker_heartbeat).total_seconds()
            if hb_lag > 60.0:
                reasons.append(SafetyPauseReason.BROKER_CONNECTION_LOST)
                messages.append(f"Broker heartbeat expired ({hb_lag:.0f}s ago)")
                details["broker_heartbeat_lag"] = hb_lag

        # 4. Model Artifact Validity
        if not is_model_valid:
            reasons.append(SafetyPauseReason.INVALID_MODEL_ARTIFACT)
            messages.append(f"Model artifact invalid: {model_error or 'Unknown validation error'}")

        # 5. Feature Calculation Integrity
        if features is not None:
            for feat_name, feat_val in features.items():
                if math.isnan(feat_val) or math.isinf(feat_val):
                    reasons.append(SafetyPauseReason.FEATURE_CALCULATION_FAILURE)
                    messages.append(f"Non-finite feature detected in '{feat_name}' ({feat_val})")
                    details["corrupted_feature"] = feat_name
                    break

        # 6. Unexpected Spread Blowout
        if current_spread_pips is not None:
            spread_mult = current_spread_pips / max(normal_spread_pips, 1e-4)
            if (
                current_spread_pips > self.config.max_spread_pips
                or spread_mult > self.config.max_spread_multiplier
            ):
                reasons.append(SafetyPauseReason.UNEXPECTED_SPREAD_BLOWOUT)
                messages.append(
                    f"Spread blowout: {current_spread_pips:.1f} pips ({spread_mult:.1f}x baseline)"
                )
                details["current_spread_pips"] = current_spread_pips
                details["spread_multiplier"] = spread_mult

        # 7. News Blackout
        if scheduled_news_events:
            pre_delta = timedelta(minutes=self.config.news_blackout_pre_minutes)
            post_delta = timedelta(minutes=self.config.news_blackout_post_minutes)
            for event_time in scheduled_news_events:
                if (event_time - pre_delta) <= current_time <= (event_time + post_delta):
                    reasons.append(SafetyPauseReason.NEWS_BLACKOUT_ACTIVE)
                    messages.append(
                        f"News blackout active for event at {event_time.isoformat()}"
                    )
                    details["news_event_time"] = event_time.isoformat()
                    break

        # 8. Model Confidence Threshold
        if model_confidence is not None:
            if model_confidence < self.config.min_model_confidence:
                reasons.append(SafetyPauseReason.LOW_MODEL_CONFIDENCE)
                messages.append(
                    f"Model confidence ({model_confidence:.2f}) below threshold "
                    f"({self.config.min_model_confidence:.2f})"
                )
                details["model_confidence"] = model_confidence

        # 9. Duplicate Signal Detection
        if symbol and (signal_id or bar_timestamp):
            ts_str = bar_timestamp.isoformat() if bar_timestamp is not None else ""
            key = f"{symbol}:{signal_id or ts_str}"
            if key in self._processed_signal_keys:
                reasons.append(SafetyPauseReason.DUPLICATE_SIGNAL_DETECTED)
                messages.append(f"Duplicate signal detected for key: {key}")
                details["duplicate_key"] = key
            else:
                self._processed_signal_keys.add(key)

        is_safe = len(reasons) == 0
        rejection_msg = "; ".join(messages) if messages else None

        return SafetyCheckResult(
            is_safe=is_safe,
            pause_reasons=reasons,
            rejection_message=rejection_msg,
            details=details,
        )
