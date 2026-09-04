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
    embargo_bars: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def sufficient_embargo(self) -> SplitConfig:
        if self.embargo_bars is not None and self.embargo_bars < self.max_label_horizon:
            raise ValueError("Embargo length must be at least the maximum label horizon.")
        return self


class EmbargoInterval(BaseModel):
    """Half-open UTC interval anchored to positions in the original bar timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    start: AwareDatetime
    end: AwareDatetime
    start_position: int
    stop_position: int
    reason: Literal["before_validation", "before_blind", "after_validation"]
    configured_bars: int


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
    purged_training: tuple[int, ...] = ()
    purged_validation: tuple[int, ...] = ()
    horizon_bars: tuple[int, ...] = ()
    embargo_bars: int = 0
    embargo_intervals: tuple[EmbargoInterval, ...] = ()
    embargoed_training: tuple[int, ...] = ()
    embargoed_validation: tuple[int, ...] = ()


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
    horizon_bars: pd.Series | None = None,
) -> FoldPlan:
    """Plan training then validation, reserving the final blind interval separately."""
    blind = blind_boundary(config, dataset_end)
    validate_timeline(index, blind)
    boundary = int(index.searchsorted(pd.Timestamp(config.validation_start)))
    plan = _plan(index, config, dataset_end, blind, 0, boundary, len(index), 0)
    return apply_embargo(purge(plan, index, horizon_bars=horizon_bars), index)


def walk_forward(
    index: pd.DatetimeIndex, config: WalkForwardConfig, *, dataset_end: pd.Timestamp,
    horizon_bars: pd.Series | None = None,
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
    return tuple(apply_embargo(purge(plan, index, horizon_bars=horizon_bars), index,
                              previous_windows=tuple(plans[:position]))
                 for position, plan in enumerate(plans))


def purge(
    plan: FoldPlan, index: pd.DatetimeIndex, *, horizon_bars: pd.Series | None = None,
) -> FoldPlan:
    """Exclude full horizons reaching the next partition, even for early barrier hits.

    Horizons count actual observations, preserving weekend/gap semantics. A
    sample with insufficient future observations is excluded, never shortened.
    Optional per-row horizons must be known setup configuration, not realized exits.
    """
    validate_timeline(index, pd.Timestamp(plan.blind_start))
    checksum = hashlib.sha256(index.as_unit("ns").asi8.tobytes()).hexdigest()
    if checksum != plan.timeline_checksum:
        raise ValueError("Fold timeline differs from its recorded checksum.")
    maximum = int(plan.configuration["max_label_horizon"])
    if horizon_bars is None:
        horizons = [maximum] * len(index)
    else:
        if (
            not horizon_bars.index.equals(index) or horizon_bars.dtype.kind not in "iu"
            or not horizon_bars.between(1, maximum).all()
        ):
            raise ValueError("Horizon bars must align with the timeline and stay within maximum.")
        horizons = [int(value) for value in horizon_bars]

    def retained(positions: tuple[int, ...], end: AwareDatetime) -> tuple[int, ...]:
        return tuple(i for i in positions if i + horizons[i] < len(index)
                     and index[i + horizons[i]] < end)

    train = retained(plan.train_indices, plan.validation_start)
    validation = retained(plan.validation_indices, plan.validation_end)
    if not train or not validation:
        raise ValueError("Purging leaves an empty training or validation partition.")
    train_set, validation_set = set(train), set(validation)
    return plan.model_copy(update={
        "train_indices": train, "validation_indices": validation,
        "purged_training": tuple(i for i in plan.train_indices if i not in train_set),
        "purged_validation": tuple(i for i in plan.validation_indices if i not in validation_set),
        "horizon_bars": tuple(horizons),
    })


def apply_embargo(
    plan: FoldPlan, index: pd.DatetimeIndex, *, previous_windows: tuple[FoldPlan, ...] = (),
) -> FoldPlan:
    """Apply pre-boundary buffers and prior validation post-buffers in later training.

    The masks are independent of purging and may overlap it. Before-blind
    buffers exclude validation samples too. Post-validation buffers only exclude
    training entries. Intervals are clipped at blind, which is never available.
    """
    validate_timeline(index, pd.Timestamp(plan.blind_start))
    if hashlib.sha256(index.as_unit("ns").asi8.tobytes()).hexdigest() != plan.timeline_checksum:
        raise ValueError("Fold timeline differs from its recorded checksum.")
    length = int(plan.configuration.get("embargo_bars") or plan.configuration["max_label_horizon"])
    if length < int(plan.configuration["max_label_horizon"]):
        raise ValueError("Embargo length is below the maximum horizon.")
    boundary = int(index.searchsorted(pd.Timestamp(plan.validation_start)))
    intervals: list[EmbargoInterval] = []

    def add(start: int, stop: int, reason: Literal[
        "before_validation", "before_blind", "after_validation",
    ]) -> None:
        start, stop = max(0, start), min(len(index), stop)
        if start >= stop:
            return
        intervals.append(EmbargoInterval(
            start=index[start], end=index[stop] if stop < len(index) else plan.blind_start,
            start_position=start, stop_position=stop, reason=reason, configured_bars=length,
        ))

    add(boundary - length, boundary, "before_validation")
    add(len(index) - length, len(index), "before_blind")
    training_start = int(index.searchsorted(pd.Timestamp(plan.train_start)))
    for previous in previous_windows:
        if previous.timeline_checksum != plan.timeline_checksum:
            raise ValueError("Previous validation window belongs to a different timeline.")
        start = int(index.searchsorted(pd.Timestamp(previous.validation_end)))
        if start < boundary and start + length > training_start:
            add(start, start + length, "after_validation")
    blocked_train: set[int] = set()
    blocked_validation: set[int] = set()
    for interval in intervals:
        positions = range(interval.start_position, interval.stop_position)
        blocked_train.update(positions)
        if interval.reason == "before_blind":
            blocked_validation.update(positions)
    train = tuple(i for i in plan.train_indices if i not in blocked_train)
    validation = tuple(i for i in plan.validation_indices if i not in blocked_validation)
    if not train or not validation:
        raise ValueError("Embargo leaves an empty training or validation partition.")
    return plan.model_copy(update={
        "train_indices": train, "validation_indices": validation, "embargo_bars": length,
        "embargo_intervals": tuple(intervals),
        "embargoed_training": tuple(i for i in plan.train_indices if i in blocked_train),
        "embargoed_validation": tuple(
            i for i in plan.validation_indices if i in blocked_validation),
    })
