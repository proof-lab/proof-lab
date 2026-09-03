"""Real versioned-data integration and causal boundaries for M04 training."""

import json

import numpy as np
import pandas as pd
import pytest

from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe
from prooflab.experiments.training import (
    ModelSpec,
    TrainingConfig,
    prepare_training_data,
    run_training,
)
from prooflab.features.base import feature_registry
from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import SetupConfig
from prooflab.models.artifacts import load_artifact


@pytest.fixture
def raw():
    count = 150
    close = 100 + 2 * np.sin(np.arange(count) * np.pi / 4)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=count, freq="min", tz="UTC"),
        "symbol": "SYNTH", "timeframe": "M1", "source": "synthetic-test",
        "open": close, "high": close + 0.15, "low": close - 0.15, "close": close,
        "volume": 100., "tick_volume": 10., "spread": 0.01,
    })


def store(tmp_path, raw):
    repository = ParquetRepository(tmp_path / "data")
    metadata = repository.save_dataset(raw, "synthetic-test", "SYNTH", Timeframe.M1)
    return repository, metadata


def configuration(metadata, direction="LONG", **updates):
    return TrainingConfig(**{
        "dataset_id": metadata.dataset_id,
        "setup": SetupConfig(direction=direction, target_distance=0.9, stop_distance=1,
                             horizon_bars=3),
        "feature_names": ("return_1", "range_1"),
        "models": (ModelSpec(kind="random"), ModelSpec(kind="majority")),
        "validation_start": pd.Timestamp("2020-01-01 01:40", tz="UTC"),
        "blind_start": pd.Timestamp("2020-01-01 02:30", tz="UTC"),
        **updates,
    })


@pytest.mark.parametrize("direction,action", [("LONG", 1), ("SHORT", -1)])
def test_directional_alignment_and_complete_horizons(tmp_path, raw, direction, action, monkeypatch):
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata, direction)
    original = BarrierEvaluator.evaluate_bar

    def guarded(self, frame, entry, setup, **kwargs):
        assert entry + setup.horizon_bars < len(frame)
        assert frame.timestamp.max() < config.blind_start
        if frame.timestamp.iloc[entry] < config.validation_start:
            assert frame.timestamp.max() < config.validation_start
        return original(self, frame, entry, setup, **kwargs)

    monkeypatch.setattr(BarrierEvaluator, "evaluate_bar", guarded)
    prepared = prepare_training_data(repo, config)
    train, val = prepared.training, prepared.validation
    assert set(train.labels) == {0, action}
    assert set(val.labels) == {0, action}
    assert len(train.features) == 96  # 100 raw - 3 full horizons - 1 warm-up.
    assert len(val.features) == 47
    assert train.features.index.equals(train.labels.index)
    assert list(train.features.columns) == ["range_1", "return_1"]
    assert (train.horizon_end_times < config.validation_start).all()
    assert (val.horizon_end_times < config.blind_start).all()
    assert train.report["full_horizon_exclusions"] == 3
    assert val.report["full_horizon_exclusions"] == 3


def test_blind_snapshot_is_rejected_before_any_data_load(tmp_path, raw, monkeypatch):
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata)
    metadata = metadata.model_copy(update={"end_time": config.blind_start})
    monkeypatch.setattr(repo, "get_metadata", lambda _: metadata)
    monkeypatch.setattr(repo, "load_dataset", lambda _: pytest.fail("Blind observations opened"))
    with pytest.raises(ValueError, match="blind period"):
        run_training(repo, config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_core_training_and_artifact_reload(tmp_path, raw):
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata)
    prepared = prepare_training_data(repo, config)
    result = run_training(repo, config, tmp_path / "run")
    report = json.loads(result.report_path.read_text())
    assert report["blind_accessed"] is False
    assert "blind_metrics" not in report
    assert report["dataset_checksum"] == metadata.checksum
    for name, path in result.artifacts.items():
        artifact = load_artifact(path, trusted=True)
        np.testing.assert_array_equal(result.models[name].predict_proba(prepared.validation.features),
                                      artifact.model.predict_proba(prepared.validation.features))
        assert artifact.manifest.training.details["blind_accessed"] is False
        assert artifact.manifest.training.setup_config["direction"] == "LONG"
    original = result.report_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_training(repo, config, tmp_path / "run")
    assert result.report_path.read_bytes() == original


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_all_models_on_one_direction_without_blind_data(tmp_path, raw, direction):
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    pytest.importorskip("torch")
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata, direction, models=(
        ModelSpec(kind="random"), ModelSpec(kind="majority"), ModelSpec(kind="logistic"),
        ModelSpec(kind="xgboost", parameters={"n_estimators": 3}),
        ModelSpec(kind="neural", parameters={"epochs": 2, "hidden_units": [4]}),
        ModelSpec(kind="svm", parameters={
            "probability_start": "2020-01-01T01:00:00Z",
            "training_end": "2020-01-01T01:40:00Z",
        }),
        ModelSpec(kind="simple_rule", parameters={
            "feature_col": "return_1", "lower_threshold": -0.005, "upper_threshold": 0.005,
            "direction": direction, "mode": "momentum",
        }),
    ))
    result = run_training(repo, config, tmp_path / "all-models")
    assert len(result.artifacts) == 7
    prepared = prepare_training_data(repo, config)
    for name, path in result.artifacts.items():
        artifact = load_artifact(path, trusted=True)
        np.testing.assert_allclose(artifact.model.predict_proba(prepared.validation.features),
                                   result.models[name].predict_proba(prepared.validation.features))
    assert result.models["svm"].fit_details_["purged_rows"] == 3


def test_later_validation_observations_cannot_change_training_inputs(tmp_path, raw):
    repo, original = store(tmp_path, raw)
    first = prepare_training_data(repo, configuration(original))
    modified = raw.copy()
    modified.loc[100:, ["open", "high", "low", "close"]] += 20
    modified.loc[100:, "volume"] *= 10
    _, changed = store(tmp_path, modified)
    second = prepare_training_data(repo, configuration(changed))
    pd.testing.assert_frame_equal(first.training.features, second.training.features)
    pd.testing.assert_series_equal(first.training.labels, second.training.labels)
    pd.testing.assert_series_equal(first.training.horizon_end_times,
                                   second.training.horizon_end_times)


def test_atr_setup_and_explicit_warmup(tmp_path, raw):
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata, setup=SetupConfig(
        direction="LONG", target_distance=1, stop_distance=1, unit="ATR", horizon_bars=3,
    ), feature_names=("atr_14", "return_1"), atr_feature="atr_14")
    prepared = prepare_training_data(repo, config)
    assert prepared.training.report["warmup_exclusions"] == 14
    assert prepared.training.features.index[0] == raw.timestamp.iloc[14]


def test_excluded_ambiguity_and_nonfinite_rows_are_recorded(tmp_path, raw, monkeypatch):
    raw.loc[20, "high"] += 10
    raw.loc[20, "low"] -= 10
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata, setup=SetupConfig(
        direction="LONG", target_distance=0.9, stop_distance=1, horizon_bars=3,
        ambiguity_policy="EXCLUDE",
    ))
    feature_metadata, generator = feature_registry._entries["return_1"]

    def with_missing(frame):
        series = generator(frame)
        series.iloc[40] = np.nan
        return series

    monkeypatch.setitem(feature_registry._entries, "return_1", (feature_metadata, with_missing))
    prepared = prepare_training_data(repo, config)
    assert prepared.training.report["policy_exclusions"] >= 1
    assert prepared.training.report["nonfinite_feature_exclusions"] == 1
    assert raw.timestamp.iloc[40] not in prepared.training.features.index
    assert prepared.training.horizon_end_times.loc[raw.timestamp.iloc[39]] == raw.timestamp.iloc[42]


@pytest.mark.parametrize("problem", ["bad_ohlc", "duplicate", "negative", "mixed_symbol"])
def test_dirty_data_rejected_without_output(tmp_path, raw, problem):
    if problem == "bad_ohlc":
        raw.loc[10, "high"] = 1
    elif problem == "duplicate":
        raw.loc[10, "timestamp"] = raw.loc[9, "timestamp"]
    elif problem == "negative":
        raw.loc[10, "volume"] = -1
    else:
        raw.loc[10, "symbol"] = "OTHER"
    repo, metadata = store(tmp_path, raw)
    with pytest.raises(ValueError):
        run_training(repo, configuration(metadata), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_insufficient_history_unknown_features_and_bad_model_params(tmp_path, raw):
    repo, metadata = store(tmp_path, raw)
    with pytest.raises(ValueError, match="No eligible"):
        prepare_training_data(repo, configuration(metadata, validation_start=raw.timestamp.iloc[2]))
    with pytest.raises(KeyError, match="not found"):
        prepare_training_data(repo, configuration(metadata, feature_names=("canonical_label",)))
    with pytest.raises(ValueError, match="no parameters"):
        run_training(repo, configuration(metadata, models=(ModelSpec(
            kind="majority", parameters={"tune": True}),)), tmp_path / "run")


@pytest.mark.parametrize("updates", [
    {"feature_names": ("return_1", "return_1")},
    {"validation_start": "2020-02-01T00:00:00Z"},
    {"blind_start": "2020-01-01"},
    {"models": (ModelSpec(kind="majority"), ModelSpec(kind="majority"))},
])
def test_invalid_run_configuration(tmp_path, raw, updates):
    _, metadata = store(tmp_path, raw)
    with pytest.raises(ValueError):
        configuration(metadata, **updates)


def test_svm_cannot_move_training_end_into_validation(tmp_path, raw, monkeypatch):
    pytest.importorskip("sklearn")
    repo, metadata = store(tmp_path, raw)
    config = configuration(metadata, models=(ModelSpec(kind="svm", parameters={
        "probability_start": "2020-01-01T01:00:00Z",
        "training_end": "2020-01-01T02:00:00Z",
    }),))
    monkeypatch.setattr(repo, "load_dataset", lambda _: pytest.fail("Invalid run loaded data"))
    with pytest.raises(ValueError, match="must equal"):
        run_training(repo, config, tmp_path / "run")


def test_entry_timing_and_atr_contracts(tmp_path, raw):
    _, metadata = store(tmp_path, raw)
    with pytest.raises(ValueError, match="bar-close"):
        configuration(metadata, setup=SetupConfig(direction="LONG", target_distance=1,
                      stop_distance=1, entry_price_col="open"))
    with pytest.raises(ValueError, match="selected atr_feature"):
        configuration(metadata, setup=SetupConfig(direction="LONG", target_distance=1,
                      stop_distance=1, unit="ATR"))
    with pytest.raises(ValueError, match="supports atr_14"):
        configuration(metadata, atr_feature="return_1")
