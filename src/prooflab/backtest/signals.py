"""Signal engine translating model predictions into qualified trading decisions.

Evaluates calibrated probabilities, directional constraints, regime filters, and
time/news blackout rules against incoming prediction records without ever modifying
the original predictions.
"""

from __future__ import annotations

import copy
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class SignalFilterConfig(BaseModel):
    """Configuration governing when a model prediction qualifies as an actionable signal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_probability: float = Field(default=0.50, ge=0.0, le=1.0)
    allowed_directions: tuple[Literal["BUY", "SELL"], ...] = ("BUY", "SELL")
    min_agreement_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    require_unanimous_vote: bool = False
    blackout_hours_utc: tuple[tuple[int, int], ...] = ()
    blackout_weekdays: tuple[int, ...] = ()
    regime_filter: Literal["none", "trend_only", "ranging_only"] = "none"
    min_atr: float | None = Field(default=None, ge=0.0)
    max_atr: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_atr_bounds(self) -> SignalFilterConfig:
        if self.min_atr is not None and self.max_atr is not None:
            if self.min_atr > self.max_atr:
                raise ValueError("min_atr cannot be greater than max_atr.")
        for start_hour, end_hour in self.blackout_hours_utc:
            if not (0 <= start_hour <= 24 and 0 <= end_hour <= 24):
                raise ValueError("Blackout hours must be between 0 and 24.")
        for day in self.blackout_weekdays:
            if not (0 <= day <= 6):
                raise ValueError("Blackout weekdays must be between 0 (Monday) and 6 (Sunday).")
        return self


class TradeSignal(BaseModel):
    """Immutable qualified signal decision produced by the signal engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(default_factory=lambda: f"SIG-{uuid4().hex[:8].upper()}")
    timestamp: AwareDatetime
    symbol: str
    direction: Literal["BUY", "SELL", "IGNORE"]
    is_actionable: bool
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    model_votes: dict[str, str] = Field(default_factory=dict)
    filter_audit: dict[str, bool] = Field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalEngine:
    """Evaluates prediction records through configurable risk, probability, and timing filters."""

    def __init__(self, config: SignalFilterConfig | None = None) -> None:
        self.config = config or SignalFilterConfig()

    def evaluate_prediction(
        self,
        prediction: dict[str, Any],
        *,
        market_context: dict[str, Any] | None = None,
    ) -> TradeSignal:
        """Evaluate a single prediction dictionary and return a qualified TradeSignal.

        Guarantees that the incoming prediction object is never mutated.

        Args:
            prediction: Dictionary conforming to the canonical prediction schema:
                {
                    "timestamp": pd.Timestamp | datetime | str,
                    "symbol": str,
                    "prediction": "BUY" | "SELL" | "IGNORE",
                    "probabilities": {"BUY": float, "SELL": float, "IGNORE": float},
                    "model_votes": dict[str, str] (optional)
                }
            market_context: Optional context dictionary containing ATR, regime indicators, etc.
        """
        # 1. Defend immutability: deep-copy prediction input
        pred_copy = copy.deepcopy(prediction)
        context = copy.deepcopy(market_context) if market_context is not None else {}

        # 2. Extract and validate required fields
        raw_ts = pred_copy.get("timestamp")
        if raw_ts is None:
            raise ValueError("Prediction missing required 'timestamp' field.")
        ts = pd.Timestamp(raw_ts)
        if ts.tz is None or str(ts.tz) != "UTC":
            raise ValueError("Prediction timestamp must be timezone-aware UTC.")

        symbol = str(pred_copy.get("symbol", "")).strip().upper()
        if not symbol:
            raise ValueError("Prediction missing required 'symbol' field.")

        raw_pred = str(pred_copy.get("prediction", "IGNORE")).strip().upper()
        if raw_pred not in {"BUY", "SELL", "IGNORE"}:
            raise ValueError(f"Invalid prediction direction: '{raw_pred}'.")

        probs = pred_copy.get("probabilities", {})
        if not isinstance(probs, dict):
            raise ValueError("Prediction 'probabilities' must be a dictionary.")

        raw_prob = float(probs.get(raw_pred, 0.0))
        model_votes = pred_copy.get("model_votes", {})
        if not isinstance(model_votes, dict):
            model_votes = {}

        # 3. Filter Auditing
        audit: dict[str, bool] = {}
        rejections: list[str] = []

        # If model predicted IGNORE, it is inherently non-actionable
        if raw_pred == "IGNORE":
            audit["is_action_class"] = False
            rejections.append("Prediction is IGNORE")
            return TradeSignal(
                timestamp=ts,
                symbol=symbol,
                direction="IGNORE",
                is_actionable=False,
                calibrated_probability=float(probs.get("IGNORE", 1.0)),
                model_votes=model_votes,
                filter_audit=audit,
                rejection_reasons=tuple(rejections),
                metadata={"original_prediction": raw_pred},
            )

        audit["is_action_class"] = True

        # Filter: Directional enablement
        dir_ok = raw_pred in self.config.allowed_directions
        audit["direction_allowed"] = dir_ok
        if not dir_ok:
            rejections.append(f"Direction '{raw_pred}' not in allowed_directions")

        # Filter: Minimum calibrated probability
        prob_ok = raw_prob >= self.config.min_probability
        audit["min_probability_met"] = prob_ok
        if not prob_ok:
            rejections.append(
                f"Calibrated probability {raw_prob:.4f} below threshold "
                f"{self.config.min_probability:.4f}"
            )

        # Filter: Model agreement / unanimity
        if model_votes:
            total_votes = len(model_votes)
            agreeing_votes = sum(1 for v in model_votes.values() if v == raw_pred)
            agreement_fraction = agreeing_votes / total_votes if total_votes > 0 else 0.0

            if self.config.require_unanimous_vote and agreement_fraction < 1.0:
                audit["unanimous_vote_met"] = False
                rejections.append(
                    f"Unanimous vote required: only {agreeing_votes}/{total_votes} agree"
                )
            else:
                audit["unanimous_vote_met"] = True

            if agreement_fraction < self.config.min_agreement_fraction:
                audit["agreement_fraction_met"] = False
                rejections.append(
                    f"Agreement fraction {agreement_fraction:.2f} below required "
                    f"{self.config.min_agreement_fraction:.2f}"
                )
            else:
                audit["agreement_fraction_met"] = True

        # Filter: Blackout Hours UTC
        hour = ts.hour
        in_hour_blackout = any(
            start <= hour < end for start, end in self.config.blackout_hours_utc
        )
        audit["blackout_hours_passed"] = not in_hour_blackout
        if in_hour_blackout:
            rejections.append(f"Timestamp hour {hour} falls within blackout hours")

        # Filter: Blackout Weekdays
        weekday = ts.weekday()
        in_day_blackout = weekday in self.config.blackout_weekdays
        audit["blackout_weekdays_passed"] = not in_day_blackout
        if in_day_blackout:
            rejections.append(f"Timestamp weekday {weekday} falls within blackout weekdays")

        # Filter: Market Context (ATR & Regime)
        atr = context.get("atr")
        if atr is not None:
            if self.config.min_atr is not None and atr < self.config.min_atr:
                audit["min_atr_passed"] = False
                rejections.append(f"ATR {atr:.5f} below minimum threshold {self.config.min_atr}")
            else:
                audit["min_atr_passed"] = True

            if self.config.max_atr is not None and atr > self.config.max_atr:
                audit["max_atr_passed"] = False
                rejections.append(f"ATR {atr:.5f} above maximum threshold {self.config.max_atr}")
            else:
                audit["max_atr_passed"] = True

        regime = context.get("regime")
        if self.config.regime_filter != "none" and regime is not None:
            if self.config.regime_filter == "trend_only" and regime != "trend":
                audit["regime_passed"] = False
                rejections.append(f"Regime filter requires 'trend' (current: '{regime}')")
            elif self.config.regime_filter == "ranging_only" and regime != "ranging":
                audit["regime_passed"] = False
                rejections.append(f"Regime filter requires 'ranging' (current: '{regime}')")
            else:
                audit["regime_passed"] = True

        is_actionable = len(rejections) == 0
        final_direction = raw_pred if is_actionable else "IGNORE"

        return TradeSignal(
            timestamp=ts,
            symbol=symbol,
            direction=final_direction,  # type: ignore[arg-type]
            is_actionable=is_actionable,
            calibrated_probability=raw_prob,
            model_votes=model_votes,
            filter_audit=audit,
            rejection_reasons=tuple(rejections),
            metadata={
                "original_prediction": raw_pred,
                "market_context": context,
            },
        )

    def evaluate_batch(
        self,
        predictions: list[dict[str, Any]] | pd.DataFrame,
        market_contexts: list[dict[str, Any]] | None = None,
    ) -> list[TradeSignal]:
        """Evaluate a batch of predictions sequentially."""
        if isinstance(predictions, pd.DataFrame):
            pred_list = predictions.to_dict(orient="records")
        else:
            pred_list = predictions

        signals: list[TradeSignal] = []
        for i, pred in enumerate(pred_list):
            ctx = market_contexts[i] if market_contexts and i < len(market_contexts) else None
            signals.append(self.evaluate_prediction(pred, market_context=ctx))
        return signals
