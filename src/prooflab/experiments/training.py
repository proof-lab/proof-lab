"""Minimal chronological training, using only a versioned pre-blind snapshot.

No blind dataset is accepted, opened, labeled, transformed, or evaluated here.
Split timestamps are configuration, never selected using observed outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from prooflab.data.health import generate_health_report
from prooflab.data.repository import DataRepository
from prooflab.data.schema import OHLCV_COLUMNS
from prooflab.data.validator import DataValidator
from prooflab.data.versioning import DatasetMetadata
from prooflab.features.base import FeatureMetadata
from prooflab.features.pipeline import FeaturePipeline
from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import DistanceUnit, SetupConfig
from prooflab.labels.outcome import BarrierOutcome, LabelMatrix, RichLabelOutcome
from prooflab.labels.quality import generate_quality_report
from prooflab.models.artifacts import TrainingMetadata, save_artifact
from prooflab.models.base import BaseModelWrapper
from prooflab.models.baselines import MajorityClassifier, RandomClassifier


class ModelSpec(BaseModel):
    """One independently configured estimator, with no parameter search."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["random", "majority", "logistic", "xgboost", "neural", "svm", "simple_rule"]
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Versioned single-direction training contract and fixed time boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    setup: SetupConfig
    feature_names: tuple[str, ...] = Field(min_length=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)
    validation_start: AwareDatetime
    blind_start: AwareDatetime
    atr_feature: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> TrainingConfig:
        if self.validation_start >= self.blind_start:
            raise ValueError("validation_start must precede blind_start.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique.")
        if len({spec.kind for spec in self.models}) != len(self.models):
            raise ValueError("Configure each model kind only once per run; no implicit search.")
        if self.setup.entry_price_col != "close":
            raise ValueError("M04 bar-close features require entry_price_col='close'.")
        if not np.isfinite([
            self.setup.target_distance, self.setup.stop_distance, self.setup.point_value,
        ]).all():
            raise ValueError("Setup distances and point value must be finite.")
        if self.setup.unit == DistanceUnit.ATR and (
            self.atr_feature is None or self.atr_feature not in self.feature_names
        ):
            raise ValueError("ATR setups require an explicitly selected atr_feature.")
        if self.atr_feature is not None and self.atr_feature != "atr_14":
            raise ValueError("The feature registry supports atr_14 for ATR label distances.")
        if self.atr_feature is not None and self.atr_feature not in self.feature_names:
            raise ValueError("atr_feature must be included in feature_names.")
        return self


@dataclass(frozen=True)
class TrainingPartition:
    features: pd.DataFrame
    labels: pd.Series
    horizon_end_times: pd.Series
    report: dict[str, Any]


@dataclass(frozen=True)
class PreparedTrainingData:
    metadata: DatasetMetadata
    feature_metadata: list[FeatureMetadata]
    training: TrainingPartition
    validation: TrainingPartition
    health: dict[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    models: dict[str, BaseModelWrapper]
    artifacts: dict[str, Path]
    report_path: Path


def _make_model(spec: ModelSpec, config: TrainingConfig) -> BaseModelWrapper:
    params = spec.parameters
    if spec.kind == "random":
        return RandomClassifier(**params)
    if spec.kind == "majority":
        if params:
            raise ValueError("Majority baseline accepts no parameters.")
        return MajorityClassifier()
    if spec.kind == "simple_rule":
        from prooflab.models.rule import SimpleRuleConfig, SimpleRuleStrategy
        rule = SimpleRuleConfig.model_validate(params)
        if rule.direction != config.setup.direction:
            raise ValueError("Rule direction must match the run's setup direction.")
        if rule.feature_col not in config.feature_names:
            raise ValueError("Rule feature must be explicitly selected in feature_names.")
        return SimpleRuleStrategy(rule)
    if spec.kind == "logistic":
        from prooflab.models.logistic import LogisticRegressionBaseline, LogisticRegressionConfig
        return LogisticRegressionBaseline(LogisticRegressionConfig.model_validate(params))
    if spec.kind == "xgboost":
        from prooflab.models.xgboost import XGBoostConfig, XGBoostModel
        return XGBoostModel(XGBoostConfig.model_validate(params))
    if spec.kind == "neural":
        from prooflab.models.neural import NeuralNetworkConfig, NeuralNetworkModel
        return NeuralNetworkModel(NeuralNetworkConfig.model_validate(params))
    from prooflab.models.svm import SVMConfig, SVMModel
    svm = SVMConfig.model_validate(params)
    if svm.training_end != config.validation_start:
        raise ValueError("SVM training_end must equal the main validation_start.")
    return SVMModel(svm)


def _load_research_data(repository: DataRepository, config: TrainingConfig) -> tuple[
    pd.DataFrame, DatasetMetadata,
]:
    metadata = repository.get_metadata(config.dataset_id)
    if metadata.dataset_id != config.dataset_id:
        raise ValueError("Repository metadata does not match the requested dataset id.")
    if metadata.start_time.tzinfo is None or metadata.end_time.tzinfo is None:
        raise ValueError("Dataset metadata must have timezone-aware endpoints.")
    if metadata.end_time >= config.blind_start:
        raise ValueError("Dataset reaches the blind period; supply a separate pre-blind snapshot.")
    if not metadata.start_time < config.validation_start <= metadata.end_time:
        raise ValueError("Dataset must include observations in both research partitions.")
    # Only after metadata establishes that this is a pre-blind snapshot do we
    # invoke the repository's integrity-checked data loader.
    frame, loaded_metadata = repository.load_dataset(config.dataset_id)
    if loaded_metadata != metadata:
        raise ValueError("Dataset metadata changed during loading.")
    frame = frame.reset_index(drop=True)
    if frame.empty or not frame.columns.is_unique or set(OHLCV_COLUMNS) - set(frame.columns):
        raise ValueError("Training requires nonempty canonical OHLCV data with unique columns.")
    timestamps = frame["timestamp"]
    if (
        not isinstance(timestamps.dtype, pd.DatetimeTZDtype) or str(timestamps.dt.tz) != "UTC"
        or timestamps.isna().any() or not timestamps.is_monotonic_increasing
        or not timestamps.is_unique
    ):
        raise ValueError("Dataset timestamps must be unique, ordered UTC values.")
    if (
        len(frame) != metadata.row_count or timestamps.iloc[0] != metadata.start_time
        or timestamps.iloc[-1] != metadata.end_time
    ):
        raise ValueError("Dataset rows and time span do not match version metadata.")
    for column, expected in [("symbol", metadata.symbol), ("timeframe", metadata.timeframe.value),
                             ("source", metadata.source)]:
        if not (frame[column] == expected).all():
            raise ValueError(f"Dataset {column} must match the single versioned series.")
    numeric = frame[["open", "high", "low", "close", "volume", "tick_volume", "spread"]]
    if any(dtype.kind not in "iuf" for dtype in numeric.dtypes):
        raise ValueError("OHLCV values must be numeric.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Training prices must be positive.")
    return frame, metadata


def _prepare_partition(
    frame: pd.DataFrame, features: pd.DataFrame, config: TrainingConfig,
    start: int, stop: int, lookback: int,
) -> TrainingPartition:
    horizon = config.setup.horizon_bars
    counts = {"raw_rows": stop - start, "full_horizon_exclusions": 0,
              "warmup_exclusions": 0, "nonfinite_feature_exclusions": 0,
              "policy_exclusions": 0}
    entries: list[int] = []
    outcomes: list[RichLabelOutcome] = []
    evaluator = BarrierEvaluator()
    permitted_frame = frame.iloc[:stop]
    for entry in range(start, stop):
        if entry + horizon >= stop:
            counts["full_horizon_exclusions"] += 1
            continue
        if entry < lookback:
            counts["warmup_exclusions"] += 1
            continue
        if not np.isfinite(features.iloc[entry].to_numpy(dtype=float)).all():
            counts["nonfinite_feature_exclusions"] += 1
            continue
        atr = float(features.iloc[entry][config.atr_feature]) if config.atr_feature else None
        if config.setup.unit == DistanceUnit.ATR and (atr is None or atr <= 0):
            raise ValueError("Label ATR values must be positive.")
        outcome = evaluator.evaluate_bar(permitted_frame, entry, config.setup, atr_value=atr)
        if outcome.barrier_outcome == BarrierOutcome.EXCLUDED:
            counts["policy_exclusions"] += 1
            continue
        entries.append(entry)
        outcomes.append(outcome)
    if not entries:
        raise ValueError("No eligible samples remain in a chronological partition.")
    selected = features.iloc[entries].copy()
    selected.index = pd.DatetimeIndex(frame["timestamp"].iloc[entries], name="timestamp")
    labels = pd.Series([int(item.canonical_label) for item in outcomes], index=selected.index)
    ends = pd.Series(pd.DatetimeIndex(frame["timestamp"].iloc[np.array(entries) + horizon]),
                     index=selected.index)
    quality = generate_quality_report(LabelMatrix(
        outcomes=outcomes, setup_config=config.setup, dataset_id=config.dataset_id,
    ))
    return TrainingPartition(selected, labels, ends, {
        **counts, "retained_rows": len(entries),
        "first_entry": selected.index[0].isoformat(),
        "last_entry": selected.index[-1].isoformat(),
        "last_complete_horizon": ends.max().isoformat(),
        "label_quality": quality.model_dump(mode="json"),
    })


def prepare_training_data(
    repository: DataRepository, config: TrainingConfig,
) -> PreparedTrainingData:
    """Load only pre-blind data; validate and generate eligible directional samples."""
    frame, metadata = _load_research_data(repository, config)
    validation = DataValidator().validate(frame, timeframe=metadata.timeframe)
    if not validation.is_valid:
        rules = [issue.rule for issue in validation.issues]
        raise ValueError(f"Dataset validation failed: {rules}")
    pipeline = FeaturePipeline(list(config.feature_names), include_raw_columns=False)
    features = pipeline.transform(frame, drop_warmup=False)
    if (
        list(features.columns) != pipeline.get_feature_names()
        or not features.index.equals(frame.index)
    ):
        raise ValueError("Feature generator output does not match the declared feature schema.")
    boundary = int((frame["timestamp"] < config.validation_start).sum())
    lookback = pipeline.get_max_lookback()
    return PreparedTrainingData(
        metadata=metadata, feature_metadata=pipeline.get_metadata_summary(),
        training=_prepare_partition(frame, features, config, 0, boundary, lookback),
        validation=_prepare_partition(frame, features, config, boundary, len(frame), lookback),
        health=generate_health_report(frame, metadata.timeframe, validation).model_dump(
            mode="json"),
    )


def run_training(
    repository: DataRepository, config: TrainingConfig, output_dir: Path | str,
) -> TrainingResult:
    """Fit each configured model independently, then persist artifacts and a report.

    The output directory must be new; a failed fit produces no output artifacts.
    There is no blind inference, parameter selection, ensembling, or calibration API.
    """
    config = config.model_copy(deep=True)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Training output directory must be new.")
    models: dict[str, BaseModelWrapper] = {
        spec.kind: _make_model(spec, config) for spec in config.models
    }
    data = prepare_training_data(repository, config)
    train, val = data.training, data.validation
    for model in models.values():
        model.fit(train.features, train.labels, (val.features, val.labels),
                  horizon_end_times=train.horizon_end_times)
    training_metadata = TrainingMetadata(
        dataset_id=data.metadata.dataset_id, dataset_checksum=data.metadata.checksum,
        setup_config=config.setup.model_dump(mode="json"),
        train_start=train.features.index[0], train_end=train.features.index[-1],
        train_rows=len(train.features), validation_start=val.features.index[0],
        validation_end=val.features.index[-1], validation_rows=len(val.features),
        details={"pipeline_version": 1, "training_config": config.model_dump(mode="json"),
                 "training_partition": train.report, "validation_partition": val.report,
                 "blind_accessed": False},
    )
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Path] = {}
    for name, model in models.items():
        path = output / f"{name}-{config.setup.direction.value.lower()}.plmodel"
        save_artifact(model, path, training=training_metadata,
                      feature_metadata=data.feature_metadata)
        artifacts[name] = path
    report = {
        "pipeline_version": 1, "configuration": config.model_dump(mode="json"),
        "dataset_id": data.metadata.dataset_id, "dataset_checksum": data.metadata.checksum,
        "health": data.health, "training": train.report, "validation": val.report,
        "feature_order": list(train.features.columns), "blind_accessed": False,
        "artifacts": {name: path.name for name, path in artifacts.items()},
        "model_fit_details": {name: model.fit_details_ for name, model in models.items()},
    }
    report_path = output / "training-report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return TrainingResult(models=models, artifacts=artifacts, report_path=report_path)
