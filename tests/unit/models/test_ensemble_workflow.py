"""M04-to-M05 integration with real fits, immutable synthetic snapshots, and reloads."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")
pytest.importorskip("xgboost")

from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe
from prooflab.experiments.training import (
    ModelSpec,
    TrainingConfig,
    prepare_training_data,
    run_training,
)
from prooflab.labels.config import SetupConfig
from prooflab.models.artifacts import load_artifact, save_artifact
from prooflab.models.calibration import CalibratedEnsemble, CalibrationConfig
from prooflab.models.ensemble import DirectionalEnsemble, EnsembleConfig
from prooflab.models.prediction import predict_with_confidence


@pytest.fixture(scope="module", params=["LONG", "SHORT"])
def workflow(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"ensemble-{request.param}")
    count = 300
    close = 100 + 2 * np.sin(np.arange(count) * np.pi / 4)
    raw = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=count, freq="min", tz="UTC"),
        "symbol": "SYNTH", "timeframe": "M1", "source": "synthetic-test",
        "open": close, "high": close + 0.15, "low": close - 0.15, "close": close,
        "volume": 100., "tick_volume": 10., "spread": 0.01,
    })
    repo = ParquetRepository(root / "data")
    member_data = repo.save_dataset(raw.iloc[:120], "synthetic-test", "SYNTH", Timeframe.M1)
    research = repo.save_dataset(raw, "synthetic-test", "SYNTH", Timeframe.M1)
    setup = SetupConfig(direction=request.param, target_distance=0.9, stop_distance=1,
                        horizon_bars=3)
    blind = pd.Timestamp("2020-01-01T05:00:00Z")
    train_config = TrainingConfig(
        dataset_id=member_data.dataset_id, setup=setup,
        feature_names=("return_1", "range_1"),
        validation_start=raw.timestamp.iloc[80], blind_start=blind,
        models=(ModelSpec(kind="random"), ModelSpec(kind="majority"), ModelSpec(kind="logistic"),
                ModelSpec(kind="xgboost", parameters={"n_estimators": 3}),
                ModelSpec(kind="neural", parameters={"epochs": 2, "hidden_units": [4]}),
                ModelSpec(kind="svm", parameters={
                    "probability_start": raw.timestamp.iloc[50],
                    "training_end": raw.timestamp.iloc[80]}),
                ModelSpec(kind="simple_rule", parameters={
                    "feature_col": "return_1", "lower_threshold": -0.005,
                    "upper_threshold": 0.005, "direction": request.param, "mode": "momentum"})),
    )
    trained = run_training(repo, train_config, root / "members")
    members = {name: load_artifact(path, trusted=True) for name, path in trained.artifacts.items()}
    # The existing chronological preparer purges full horizons at the later split.
    prepared = prepare_training_data(repo, train_config.model_copy(update={
        "dataset_id": research.dataset_id, "validation_start": raw.timestamp.iloc[220],
    }))
    calibration_config = CalibrationConfig(start=raw.timestamp.iloc[130],
                                            end=raw.timestamp.iloc[220])
    mask = prepared.training.features.index >= calibration_config.start
    return (members, prepared, mask, calibration_config, blind, request.param)


@pytest.mark.parametrize("method", ["hard_vote", "probability_average", "weighted_average"])
@pytest.mark.parametrize("calibration", ["platt", "isotonic"])
def test_all_combinations_with_both_calibrators_and_directions(workflow, method, calibration,
                                                              tmp_path):
    members, data, mask, cal_config, blind, direction = workflow
    config = EnsembleConfig(direction=direction, blind_start=blind, method=method,
                            weights={name: i + 1 for i, name in enumerate(members)}
                            if method == "weighted_average" else None)
    ensemble = DirectionalEnsemble(members, config)
    assert ensemble.information_end < cal_config.start
    calibrated = CalibratedEnsemble(ensemble, cal_config.model_copy(update={"method": calibration}))
    calibrated.fit(data.training.features.loc[mask], data.training.labels.loc[mask],
                   horizon_end_times=data.training.horizon_end_times.loc[mask])
    report = calibrated.evaluate_quality(data.validation.features, data.validation.labels,
                                         horizon_end_times=data.validation.horizon_end_times)
    json.dumps(report, allow_nan=False)
    assert report["blind_accessed"] is False
    assert report["calibrated"]["rows"] == len(data.validation.features)
    assert calibrated.information_end < data.validation.features.index[0]
    assert data.validation.horizon_end_times.max() < blind
    # Persist and load the whole graph, including all scalers and both SVM probability links.
    first = next(iter(members.values())).manifest
    path = tmp_path / "ensemble.plmodel"
    manifest = save_artifact(calibrated, path, training=first.training,
                             feature_metadata=data.feature_metadata)
    restored = load_artifact(path, trusted=True).model
    for package in ["torch", "xgboost", "scikit-learn", "scipy"]:
        assert package in manifest.dependencies
    expected = predict_with_confidence(calibrated, data.validation.features, symbol="SYNTH")
    actual = predict_with_confidence(restored, data.validation.features, symbol="SYNTH")
    assert actual.records == expected.records
    np.testing.assert_array_equal(actual.confidence, expected.confidence)
    opposite = "SELL" if direction == "LONG" else "BUY"
    assert all(record.probabilities.model_dump()[opposite] == 0 for record in actual.records)
    assert all(set(record.model_votes) == set(members) for record in actual.records)


def test_member_early_stopping_partition_cannot_be_reused_for_calibration(workflow):
    members, _, _, _, blind, direction = workflow
    ensemble = DirectionalEnsemble(members, EnsembleConfig(direction=direction, blind_start=blind))
    with pytest.raises(ValueError, match="training and validation horizons"):
        CalibratedEnsemble(ensemble, CalibrationConfig(start="2020-01-01T01:30:00Z",
                                                      end="2020-01-01T03:00:00Z"))
