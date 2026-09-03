"""Exact directional JSON records and a single coherent draw per voter."""

import json

import numpy as np
import pandas as pd
import pytest
from test_ensemble import config

from prooflab.models.ensemble import DirectionalEnsemble
from prooflab.models.prediction import EnsemblePrediction, predict_records


@pytest.mark.parametrize("action,direction,opposite", [(1, "LONG", "SELL"), (-1, "SHORT", "BUY")])
def test_exact_prediction_json(member_factory, action, direction, opposite):
    model = DirectionalEnsemble({"xgboost": member_factory(action)}, config(direction))
    x = pd.DataFrame({"signal": [0.1, 0.9]},
                     index=pd.date_range("2020-01-05", periods=2, freq="h", tz="UTC"))
    records = predict_records(model, x, symbol="SYNTH")
    for record in records:
        payload = json.loads(record.model_dump_json())
        assert set(payload) == {"timestamp", "symbol", "prediction", "probabilities", "model_votes"}
        assert set(payload["probabilities"]) == {"BUY", "SELL", "IGNORE"}
        assert payload["probabilities"][opposite] == 0
        assert payload["model_votes"]["xgboost"] == payload["prediction"]
        assert payload["timestamp"].endswith("Z")
        assert EnsemblePrediction.model_validate(payload) == record


def test_single_vote_snapshot(member_factory, monkeypatch):
    model = DirectionalEnsemble({"random": member_factory()}, config())
    seen = []

    def alternating(features):
        seen.append(1)
        return np.full(len(features), len(seen) % 2)

    monkeypatch.setattr(model._members["random"].model, "predict", alternating)
    x = pd.DataFrame({"signal": [1.]}, index=pd.DatetimeIndex(["2020-01-05"], tz="UTC"))
    record = predict_records(model, x, symbol="SYNTH")[0]
    assert len(seen) == 1
    assert record.prediction == record.model_votes["random"] == "BUY"
    assert record.probabilities.BUY == 1


@pytest.mark.parametrize("change", ["blind", "past", "naive", "empty_symbol", "duplicates"])
def test_invalid_record_requests_do_not_predict(member_factory, monkeypatch, change):
    model = DirectionalEnsemble({"a": member_factory()}, config())
    monkeypatch.setattr(model, "evaluate", lambda *_: pytest.fail("Invalid request was scored"))
    index = pd.DatetimeIndex(["2020-01-05"], tz="UTC")
    if change == "blind":
        index = pd.DatetimeIndex(["2020-02-01"], tz="UTC")
    elif change == "past":
        index = pd.DatetimeIndex(["2020-01-03"], tz="UTC")
    elif change == "naive":
        index = index.tz_localize(None)
    elif change == "duplicates":
        index = index.append(index)
    with pytest.raises(ValueError):
        predict_records(model, pd.DataFrame({"signal": np.ones(len(index))}, index=index),
                        symbol=" " if change == "empty_symbol" else "SYNTH")


@pytest.mark.parametrize("change", ["mixed", "sum", "nan", "extra", "votes"])
def test_invalid_prediction_schema(change):
    payload = {"timestamp": "2020-01-05T00:00:00Z", "symbol": "SYNTH", "prediction": "BUY",
               "probabilities": {"BUY": 0.7, "SELL": 0, "IGNORE": 0.3},
               "model_votes": {"a": "BUY"}}
    if change == "mixed":
        payload["probabilities"] = {"BUY": 0.5, "SELL": 0.2, "IGNORE": 0.3}
    elif change == "sum":
        payload["probabilities"]["BUY"] = 0.8
    elif change == "nan":
        payload["probabilities"]["BUY"] = float("nan")
    elif change == "extra":
        payload["confidence"] = 0.7
    else:
        payload["model_votes"]["a"] = "SELL"
    with pytest.raises(ValueError):
        EnsemblePrediction.model_validate(payload)


@pytest.mark.parametrize("method", ["hard_vote", "probability_average", "weighted_average"])
def test_uncalibrated_confidence_is_unavailable(member_factory, method):
    from prooflab.models.prediction import predict_with_confidence
    settings = {"weights": {"a": 1}} if method == "weighted_average" else {}
    model = DirectionalEnsemble({"a": member_factory()}, config(method=method, **settings))
    x = pd.DataFrame({"signal": [1.]}, index=pd.DatetimeIndex(["2020-01-15"], tz="UTC"))
    result = predict_with_confidence(model, x, symbol="SYNTH")
    assert result.confidence is None
    assert result.agreement[0] == 1
    assert result.probability_kind == "raw"


@pytest.mark.parametrize("method", ["platt", "isotonic"])
@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_confidence_is_probability_of_emitted_class_not_votes(member_factory, method, action,
                                                             direction):
    pytest.importorskip("sklearn")
    from prooflab.models.calibration import CalibratedEnsemble, CalibrationConfig
    from prooflab.models.prediction import predict_with_confidence
    model = DirectionalEnsemble({"a": member_factory(action)}, config(direction))
    index = pd.date_range("2020-01-05", periods=10, freq="h", tz="UTC")
    x = pd.DataFrame({"signal": np.ones(10)}, index=index)
    y = pd.Series([0] * 8 + [action] * 2, index=index)
    ends = pd.Series(index + pd.Timedelta("1h"), index=index)
    calibrated = CalibratedEnsemble(model, CalibrationConfig(
        method=method, start="2020-01-04T00:00:00Z", end="2020-01-10T00:00:00Z",
    )).fit(x, y, horizon_end_times=ends)
    x.index += pd.Timedelta("10D")
    result = predict_with_confidence(calibrated, x, symbol="SYNTH")
    assert result.probability_kind == "calibrated"
    np.testing.assert_array_equal(result.agreement, 1)
    assert (result.confidence < 0.5).all()  # Agreement can be 100% with low confidence.
    for row, record in enumerate(result.records):
        assert record.prediction == ("BUY" if action == 1 else "SELL")
        assert result.confidence[row] == record.probabilities.model_dump()[record.prediction]
        assert "confidence" not in record.model_dump()
