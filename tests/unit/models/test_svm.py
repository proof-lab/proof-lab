"""SVM probability isolation across explicit chronological, full-horizon splits."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import TrainingMetadata, load_artifact, save_artifact
from prooflab.models.svm import SVMConfig, SVMModel


@pytest.fixture
def data():
    index = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    x = pd.DataFrame({"signal": np.tile([-3., -1., 1., 3.], 10)}, index=index)
    y = (x.signal > 0).astype(int)
    ends = pd.Series(index + pd.Timedelta(hours=2), index=index)
    config = SVMConfig(probability_start=index[24], training_end=ends.iloc[-1] + pd.Timedelta("1h"))
    return x, y, ends, config


def fit(data):
    x, y, ends, config = data
    return SVMModel(config).fit(x, y, horizon_end_times=ends)


@pytest.mark.parametrize("action", [-1, 1])
def test_directional_mapping_and_purge(data, action):
    x, y, ends, config = data
    model = fit((x, y * action, ends, config))
    assert model.classes_ == sorted([0, action])
    assert model.fit_details_["purged_rows"] == 2
    assert model.fit_details_["svm_fit_rows"] == 22
    assert model.fit_details_["probability_rows"] == 16
    assert pd.Timestamp(model.fit_details_["svm_fit_last_horizon"]) < config.probability_start
    assert not hasattr(model.pipeline["model"], "predict_proba")
    np.testing.assert_array_equal(model.predict(x), y * action)
    proba = model.predict_proba(x)
    assert proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert np.mean(proba[np.arange(40), np.searchsorted(model.classes_, y * action)]) > 0.7
    assert model._horizon_end_times is None


def test_later_rows_cannot_refit_scaling_or_support_vectors(data):
    x, y, ends, config = data
    original = fit(data)
    modified_x, modified_y = x.copy(), y.copy()
    modified_x.iloc[24:] *= 10
    modified_y.iloc[22:24] = 1 - modified_y.iloc[22:24]  # Purged labels have no role.
    modified_y.iloc[24:] = 1 - modified_y.iloc[24:]
    modified = fit((modified_x, modified_y, ends, config))
    np.testing.assert_array_equal(original.pipeline["scaler"].mean_,
                                  modified.pipeline["scaler"].mean_)
    np.testing.assert_array_equal(original.pipeline["model"].support_vectors_,
                                  modified.pipeline["model"].support_vectors_)
    np.testing.assert_array_equal(original.pipeline["model"].dual_coef_,
                                  modified.pipeline["model"].dual_coef_)
    assert original.probability_coefficients_ != modified.probability_coefficients_


def test_validation_is_not_used_and_fit_is_repeatable(data):
    x, y, ends, config = data
    a = fit(data)
    b = SVMModel(config).fit(x, y, (x * 1000, 1 - y), horizon_end_times=ends)
    np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))


@pytest.mark.parametrize("change,match", [
    ("no_context", "horizon_end_times"), ("outer_crossing", "reaches training_end"),
    ("unaligned", "aligned UTC"), ("naive", "ordered UTC"),
    ("single_early", "Earlier"), ("single_later", "Later"),
    ("three_classes", "single direction"),
])
def test_invalid_partitions(data, change, match):
    x, y, ends, config = data
    if change == "no_context":
        ends = None
    elif change == "outer_crossing":
        ends.iloc[-1] = config.training_end
    elif change == "unaligned":
        ends = ends.iloc[::-1]
    elif change == "naive":
        x.index = x.index.tz_localize(None)
        y.index = x.index
    elif change == "single_early":
        y.iloc[:24] = 0
    elif change == "single_later":
        y.iloc[24:] = 0
    else:
        y.iloc[0] = -1
    with pytest.raises(ValueError, match=match):
        SVMModel(config).fit(x, y, horizon_end_times=ends)


def test_probability_disabled_and_parameters(data):
    x, y, ends, original = data
    config = SVMConfig(training_end=original.training_end, probability=False,
                       kernel="linear", c_param=2, gamma=0.2, class_weight={0: 1, 1: 2})
    model = SVMModel(config).fit(x, y, horizon_end_times=ends)
    assert model.pipeline["model"].C == 2
    assert model.pipeline["model"].kernel == "linear"
    assert model.fit_details_["svm_fit_rows"] == 40
    with pytest.raises(NotImplementedError, match="disabled"):
        model.predict_proba(x)


@pytest.mark.parametrize("params", [
    {"gamma": -1}, {"c_param": 0}, {"class_weight": {3: 1}},
    {"class_weight": {1: float("nan")}}, {"probability_start": None}, {"kernel": "unknown"},
])
def test_invalid_config(data, params):
    config = data[-1].model_dump()
    config.update(params)
    with pytest.raises(ValueError):
        SVMConfig(**config)


def test_round_trip_preserves_probability_link_and_partition_evidence(data, tmp_path):
    x, y, ends, config = data
    model = fit(data)
    path = tmp_path / "svm.plmodel"
    manifest = save_artifact(model, path, training=TrainingMetadata(
        dataset_id="synthetic", dataset_checksum="a" * 64, setup_config={"direction": "LONG"},
        train_start=x.index[0], train_end=x.index[-1], train_rows=len(x),
    ), feature_metadata=[FeatureMetadata(feature_name="signal", family=FeatureFamily.PRICE,
                                         description="Synthetic test signal")])
    loaded = load_artifact(path, trusted=True)
    assert manifest.fit_details["m04_compatibility_exception"] is True
    assert loaded.manifest.fit_details == model.fit_details_
    np.testing.assert_array_equal(model.predict(x), loaded.model.predict(x))
    np.testing.assert_array_equal(model.predict_proba(x), loaded.model.predict_proba(x))
