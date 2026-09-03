"""Formal calibration, isolation, and persistence on synthetic score distributions."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from test_ensemble import config

from prooflab.models.artifacts import load_artifact, save_artifact
from prooflab.models.calibration import CalibratedEnsemble, CalibrationConfig
from prooflab.models.ensemble import DirectionalEnsemble


@pytest.fixture
def calibration_data():
    index = pd.date_range("2020-01-05", periods=40, freq="h", tz="UTC")
    x = pd.DataFrame({"signal": np.repeat([0.1, 0.3, 0.7, 0.9], 10)}, index=index)
    y = pd.Series(np.tile([0, 0, 0, 1, 1, 0, 1, 1, 1, 1], 4), index=index)
    return x, y, pd.Series(index + pd.Timedelta("2h"), index=index)


def calibration_config(**updates):
    return CalibrationConfig(**{"start": "2020-01-04T00:00:00Z",
                                "end": "2020-01-10T00:00:00Z", **updates})


@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_platt_fit_direction_and_frozen_members(
    member_factory, calibration_data, action, direction,
):
    x, y, ends = calibration_data
    ensemble = DirectionalEnsemble({"a": member_factory(action, probability=None)},
                                  config(direction, method="probability_average"))
    before = ensemble.predict_proba(x)
    calibrated = CalibratedEnsemble(ensemble, calibration_config()).fit(
        x, y * action, horizon_end_times=ends)
    after = calibrated.predict_proba(x)
    np.testing.assert_array_equal(ensemble.predict_proba(x), before)
    assert not np.allclose(after, before)
    np.testing.assert_allclose(after.sum(axis=1), 1)
    assert np.isfinite(after).all()
    assert calibrated.fit_details_["calibration_framework"] == "m05_v1"
    assert calibrated.information_end == ends.max()
    assert calibrated._horizon_end_times is None


def test_platt_constant_score_has_smoothed_prior(member_factory, calibration_data):
    x, y, ends = calibration_data
    ensemble = DirectionalEnsemble({"a": member_factory()}, config())
    model = CalibratedEnsemble(ensemble, calibration_config()).fit(x, y, horizon_end_times=ends)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    expected = (n_pos * (n_pos + 1) / (n_pos + 2) + n_neg / (n_neg + 2)) / len(y)
    np.testing.assert_allclose(model.predict_proba(x)[:, 1], expected, atol=1e-6)
    np.testing.assert_array_equal(model.predict(x), 1)  # Hard vote decision is preserved.


@pytest.mark.parametrize("change", ["overlap", "blind", "reversed", "naive"])
def test_invalid_calibration_windows(member_factory, change):
    ensemble = DirectionalEnsemble({"a": member_factory()}, config())
    updates = {"overlap": {"start": "2020-01-03T00:00:00Z"},
               "blind": {"end": "2020-02-02T00:00:00Z"},
               "reversed": {"end": "2020-01-02T00:00:00Z"},
               "naive": {"start": "2020-01-04"}}[change]
    with pytest.raises(ValueError):
        CalibratedEnsemble(ensemble, calibration_config(**updates))


@pytest.mark.parametrize("change", ["early", "crossing", "blind", "missing", "single", "opposite",
                                   "val_data", "unaligned"])
def test_fit_rejects_invalid_data_before_member_prediction(
    member_factory, calibration_data, change, monkeypatch,
):
    x, y, ends = calibration_data
    model = CalibratedEnsemble(DirectionalEnsemble({"a": member_factory()}, config()),
                               calibration_config())
    monkeypatch.setattr(model._ensemble, "evaluate", lambda *_: pytest.fail("Invalid data scored"))
    kwargs = {}
    if change == "early":
        x.index = x.index - pd.Timedelta("3D")
        y.index = ends.index = x.index
    elif change in {"crossing", "blind"}:
        ends.iloc[-1] = pd.Timestamp("2020-01-10" if change == "crossing" else "2020-02-01",
                                     tz="UTC")
    elif change == "missing":
        ends = None
    elif change == "single":
        y[:] = 0
    elif change == "opposite":
        y = -y
    elif change == "val_data":
        kwargs["val_data"] = (x, y)
    else:
        ends = ends.iloc[::-1]
    with pytest.raises(ValueError):
        model.fit(x, y, horizon_end_times=ends, **kwargs)
    assert not model.is_fitted


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrated_artifact_round_trip(member_factory, calibration_data, tmp_path, method):
    x, y, ends = calibration_data
    member = member_factory(probability=None)
    ensemble = DirectionalEnsemble({"a": member}, config(method="probability_average"))
    model = CalibratedEnsemble(ensemble, calibration_config(method=method)).fit(
        x, y, horizon_end_times=ends)
    path = tmp_path / "calibrated.plmodel"
    save_artifact(model, path, training=member.manifest.training,
                  feature_metadata=member.manifest.feature_metadata)
    restored = load_artifact(path, trusted=True)
    np.testing.assert_array_equal(model.predict_proba(x), restored.model.predict_proba(x))
    assert restored.manifest.preprocessing == "ensemble"
    assert "scipy" in restored.manifest.dependencies
    assert restored.manifest.fit_details["last_complete_horizon"] == ends.max().isoformat()


def test_failed_refit_invalidates_calibrator(member_factory, calibration_data):
    x, y, ends = calibration_data
    model = CalibratedEnsemble(DirectionalEnsemble({"a": member_factory()}, config()),
                               calibration_config()).fit(x, y, horizon_end_times=ends)
    with pytest.raises(ValueError):
        model.fit(x, y * 0, horizon_end_times=ends)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba(x)


@pytest.mark.parametrize("invalid", [False, True])
def test_quality_uses_later_preblind_rows(member_factory, calibration_data, invalid, monkeypatch):
    x, y, ends = calibration_data
    model = CalibratedEnsemble(DirectionalEnsemble({"a": member_factory()}, config()),
                               calibration_config()).fit(x, y, horizon_end_times=ends)
    if invalid:
        monkeypatch.setattr(model._ensemble, "evaluate", lambda *_: pytest.fail("In-sample score"))
        with pytest.raises(ValueError, match="after calibration"):
            model.evaluate_quality(x, y, horizon_end_times=ends)
        return
    x.index += pd.Timedelta("10D")
    y.index = ends.index = x.index
    ends += pd.Timedelta("10D")
    report = model.evaluate_quality(x, y, horizon_end_times=ends, n_bins=5)
    assert report["calibrated"]["brier_score"] < report["raw"]["brier_score"]
    assert report["blind_accessed"] is False
    ends.iloc[-1] = pd.Timestamp("2020-02-01", tz="UTC")
    with pytest.raises(ValueError, match="pre-blind"):
        model.evaluate_quality(x, y, horizon_end_times=ends)


def test_metrics_match_hand_calculation_and_empty_bins():
    from prooflab.validation.calibration import probability_quality
    result = probability_quality(np.array([0., 0.25, 0.75, 1.]), np.array([0, 1, 0, 1]), n_bins=2)
    assert result["brier_score"] == pytest.approx(0.28125)
    assert result["log_loss"] == pytest.approx(-np.log(0.25) / 2)
    assert result["expected_calibration_error"] == pytest.approx(0.375)
    assert [row["count"] for row in result["calibration_curve"]] == [2, 2]
    empty = probability_quality(np.array([0., 1.]), np.array([0, 1]), n_bins=3)
    assert empty["calibration_curve"][1]["mean_probability"] is None
    assert np.isfinite(probability_quality(np.array([0., 1.]), np.array([1, 0]))["log_loss"])


@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_isotonic_monotonicity_endpoint_clipping_and_ties(member_factory, action, direction):
    index = pd.date_range("2020-01-05", periods=8, freq="h", tz="UTC")
    x = pd.DataFrame({"signal": [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8]}, index=index)
    y = pd.Series(np.array([0, 0, 0, 1, 0, 1, 1, 1]) * action, index=index)
    ends = pd.Series(index + pd.Timedelta("1h"), index=index)
    ensemble = DirectionalEnsemble({"a": member_factory(action, probability=None)},
                                  config(direction, method="probability_average"))
    model = CalibratedEnsemble(ensemble, calibration_config(method="isotonic")).fit(
        x, y, horizon_end_times=ends)
    probe = pd.DataFrame({"signal": [0., 0.2, 0.4, 0.6, 0.8, 1.]})
    p = model.predict_proba(probe)[:, model.classes_.index(action)]
    np.testing.assert_allclose(p, [0, 0, 0.5, 0.5, 1, 1])
    assert (np.diff(p) >= 0).all()
    np.testing.assert_array_equal(model.predict(probe), [0, 0, 0, 0, action, action])


def test_isotonic_constant_score_is_observed_frequency(member_factory, calibration_data):
    x, y, ends = calibration_data
    ensemble = DirectionalEnsemble({"a": member_factory()}, config())
    model = CalibratedEnsemble(ensemble, calibration_config(method="isotonic")).fit(
        x, y, horizon_end_times=ends)
    np.testing.assert_allclose(model.predict_proba(x)[:, 1], y.mean())
