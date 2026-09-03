"""Chronological research plans; blind observations are never materialized.

Inputs contain only ordered pre-blind timestamps. The blind interval is reserved
from dataset metadata, with a default start two calendar years before its end.
All research windows are half-open. dataset_end is the inclusive metadata end.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    validation_start: AwareDatetime
    blind_start: AwareDatetime | None = None
    blind_years: int = Field(default=2, ge=1, strict=True)
    max_label_horizon: int = Field(ge=1, strict=True)


class FoldPlan(BaseModel):
    """Research row positions plus blind metadata, never blind row values."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    fold_id: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_start: AwareDatetime
    validation_start: AwareDatetime
    validation_end: AwareDatetime
    blind_start: AwareDatetime
    dataset_end: AwareDatetime
    timeline_checksum: str
    configuration: dict[str, Any]


class WalkForwardConfig(SplitConfig):
    """Window lengths count observations, not elapsed days across market closures."""

    mode: Literal["expanding", "rolling"] = "expanding"
    validation_bars: int = Field(ge=1, strict=True)
    step_bars: int | None = Field(default=None, ge=1, strict=True)
    training_bars: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_windows(self) -> WalkForwardConfig:
        if self.mode == "rolling" and self.training_bars is None:
            raise ValueError("Rolling validation requires training_bars.")
        if self.step_bars is not None and self.step_bars < self.validation_bars:
            raise ValueError("Walk-forward validation windows must not overlap.")
        return self


def blind_boundary(config: SplitConfig, dataset_end: pd.Timestamp) -> pd.Timestamp:
    end = pd.Timestamp(dataset_end)
    if pd.isna(end) or end.tzinfo is None:
        raise ValueError("Dataset end must be timezone-aware.")
    boundary = (pd.Timestamp(config.blind_start) if config.blind_start is not None else
                end - pd.DateOffset(years=config.blind_years))
    if not pd.Timestamp(config.validation_start) < boundary <= end:
        raise ValueError("Validation must precede the blind interval within the dataset span.")
    return boundary.tz_convert("UTC")


def validate_timeline(index: pd.DatetimeIndex, blind_start: pd.Timestamp) -> None:
    if (
        not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
        or index.empty or index.hasnans or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Research timestamps must be nonempty, unique, ordered UTC values.")
    if index[-1] >= blind_start:
        raise ValueError("Research input reaches blind; supply pre-blind timestamps only.")


def _plan(
    index: pd.DatetimeIndex, config: SplitConfig, dataset_end: pd.Timestamp,
    blind: pd.Timestamp, train_start: int, validation_start: int, validation_stop: int,
    fold_id: int,
) -> FoldPlan:
    if not train_start < validation_start < validation_stop:
        raise ValueError("Every fold requires nonempty training and validation windows.")
    return FoldPlan(
        fold_id=fold_id, train_indices=tuple(range(train_start, validation_start)),
        validation_indices=tuple(range(validation_start, validation_stop)),
        train_start=index[train_start], validation_start=index[validation_start],
        validation_end=index[validation_stop] if validation_stop < len(index) else blind,
        blind_start=blind, dataset_end=dataset_end,
        timeline_checksum=hashlib.sha256(index.as_unit("ns").asi8.tobytes()).hexdigest(),
        configuration=config.model_dump(mode="json"),
    )


def chronological_split(
    index: pd.DatetimeIndex, config: SplitConfig, *, dataset_end: pd.Timestamp,
) -> FoldPlan:
    """Plan training then validation, reserving the final blind interval separately."""
    blind = blind_boundary(config, dataset_end)
    validate_timeline(index, blind)
    boundary = int(index.searchsorted(pd.Timestamp(config.validation_start)))
    return _plan(index, config, dataset_end, blind, 0, boundary, len(index), 0)


def walk_forward(
    index: pd.DatetimeIndex, config: WalkForwardConfig, *, dataset_end: pd.Timestamp,
) -> tuple[FoldPlan, ...]:
    """Generate complete expanding/rolling research windows; omit an incomplete final window."""
    blind = blind_boundary(config, dataset_end)
    validate_timeline(index, blind)
    first = int(index.searchsorted(pd.Timestamp(config.validation_start)))
    initial = 0 if config.training_bars is None else first - config.training_bars
    if initial < 0 or first == 0:
        raise ValueError("Insufficient history for the requested training window.")
    plans: list[FoldPlan] = []
    for start in range(first, len(index) - config.validation_bars + 1,
                       config.step_bars or config.validation_bars):
        train_start = initial
        if config.mode == "rolling":
            assert config.training_bars is not None
            train_start = start - config.training_bars
        plans.append(_plan(index, config, dataset_end, blind, train_start, start,
                           start + config.validation_bars, len(plans)))
    if not plans:
        raise ValueError("No complete walk-forward validation window fits before blind.")
    return tuple(plans)
