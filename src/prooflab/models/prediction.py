"""Exact M05 prediction records from one coherent ensemble evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from prooflab.models.ensemble import DirectionalEnsemble

if TYPE_CHECKING:
    from prooflab.models.calibration import CalibratedEnsemble

Action = Literal["BUY", "SELL", "IGNORE"]
_ACTIONS: dict[int, Action] = {1: "BUY", -1: "SELL", 0: "IGNORE"}


class ClassProbabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    BUY: float = Field(ge=0, le=1)
    SELL: float = Field(ge=0, le=1)
    IGNORE: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def directional_distribution(self) -> ClassProbabilities:
        if not np.isclose(self.BUY + self.SELL + self.IGNORE, 1, atol=1e-6, rtol=0):
            raise ValueError("Probabilities must sum to one.")
        if self.BUY != 0 and self.SELL != 0:
            raise ValueError("A directional prediction must assign zero to the opposite action.")
        return self


class EnsemblePrediction(BaseModel):
    """Exactly the five fields required by the M05 contract; no confidence field."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)
    timestamp: AwareDatetime
    symbol: str = Field(min_length=1)
    prediction: Action
    probabilities: ClassProbabilities
    model_votes: dict[str, Action] = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def normalize_utc(cls, value: AwareDatetime) -> AwareDatetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def consistent_direction(self) -> EnsemblePrediction:
        if any(not name.strip() for name in self.model_votes):
            raise ValueError("Model votes require nonempty names.")
        actions: set[str] = {vote for vote in self.model_votes.values() if vote != "IGNORE"}
        actions.update(name for name, value in self.probabilities.model_dump().items()
                       if name != "IGNORE" and value > 0)
        if self.prediction != "IGNORE":
            actions.add(self.prediction)
        if len(actions) > 1:
            raise ValueError("Predictions and votes cannot mix setup directions.")
        return self


def predict_records(
    model: DirectionalEnsemble | CalibratedEnsemble, features: pd.DataFrame, *, symbol: str,
) -> list[EnsemblePrediction]:
    """Produce pre-blind research predictions after all model fitting information.

    Generation requires explicit ordered UTC timestamps. M05 does not open the
    blind period for evaluation. Generic wrapper prediction remains available
    for inspecting fitted behavior, but cannot bypass this research-output gate.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("A nonempty symbol is required.")
    if (
        not isinstance(features.index, pd.DatetimeIndex) or str(features.index.tz) != "UTC"
        or features.empty or not features.index.is_unique
        or not features.index.is_monotonic_increasing or features.index.hasnans
    ):
        raise ValueError("Prediction records require unique ordered UTC timestamps.")
    blind_start = (model.config.blind_start if isinstance(model, DirectionalEnsemble)
                   else model._ensemble.config.blind_start)
    if features.index[0] <= model.information_end or features.index[-1] >= blind_start:
        raise ValueError("Prediction records must follow fitting information and precede blind.")
    batch = model.evaluate(features)
    records = []
    for row, timestamp in enumerate(features.index):
        probabilities: dict[str, float] = {"BUY": 0., "SELL": 0., "IGNORE": 0.}
        for column, cls in enumerate(model.classes_):
            probabilities[_ACTIONS[cls]] = float(batch.probabilities[row, column])
        records.append(EnsemblePrediction(
            timestamp=timestamp, symbol=symbol, prediction=_ACTIONS[int(batch.predictions[row])],
            probabilities=ClassProbabilities.model_validate(probabilities),
            model_votes={name: _ACTIONS[int(votes[row])]
                         for name, votes in batch.model_votes.items()},
        ))
    return records


@dataclass(frozen=True)
class PredictionResult:
    """Diagnostics kept outside the exact five-field serialized record schema.

    Confidence is unavailable without formal calibration. Agreement is always
    the fraction of member votes matching the emitted prediction, not confidence.
    """

    records: list[EnsemblePrediction]
    confidence: np.ndarray | None
    agreement: np.ndarray
    probability_kind: Literal["raw", "calibrated"]


def predict_with_confidence(
    model: DirectionalEnsemble | CalibratedEnsemble, features: pd.DataFrame, *, symbol: str,
) -> PredictionResult:
    """Return calibrated P(emitted class), even when hard voting disagrees with argmax."""
    records = predict_records(model, features, symbol=symbol)
    calibrated = False
    if not isinstance(model, DirectionalEnsemble):
        from prooflab.models.calibration import CalibratedEnsemble
        calibrated = isinstance(model, CalibratedEnsemble)
    confidence = (np.array([record.probabilities.model_dump()[record.prediction]
                            for record in records]) if calibrated else None)
    agreement = np.array([
        sum(vote == record.prediction for vote in record.model_votes.values())
        / len(record.model_votes)
        for record in records
    ])
    return PredictionResult(records, confidence, agreement, "calibrated" if calibrated else "raw")
