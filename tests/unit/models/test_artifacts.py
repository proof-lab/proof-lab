"""Native model round trips, provenance, and safe inspection boundaries."""

import json
from datetime import UTC, datetime
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import (
    TrainingMetadata,
    inspect_artifact,
    load_artifact,
    save_artifact,
)
from prooflab.models.baselines import MajorityClassifier, RandomClassifier


@pytest.fixture
def data():
    return pd.DataFrame({"signal": [-3., -2., -1., 1., 2., 3.]}), np.array([-1, -1, -1, 1, 1, 1])


@pytest.fixture
def metadata():
    return {
        "training": TrainingMetadata(
            dataset_id="synthetic-test", dataset_checksum="a" * 64,
            setup_config={"direction": "long", "horizon_bars": 1},
            train_start=datetime(2020, 1, 1, tzinfo=UTC),
            train_end=datetime(2020, 1, 6, tzinfo=UTC), train_rows=6,
            validation_start=datetime(2020, 1, 7, tzinfo=UTC),
            validation_end=datetime(2020, 1, 12, tzinfo=UTC), validation_rows=6,
        ),
        "feature_metadata": [FeatureMetadata(
            feature_name="signal", family=FeatureFamily.PRICE,
            description="Synthetic unit-test feature",
            required_columns=["close"], lookback_period=1,
        )],
    }


def make_model(name):
    if name == "random":
        return RandomClassifier()
    if name == "majority":
        return MajorityClassifier()
    pytest.importorskip("sklearn")
    if name == "logistic":
        from prooflab.models.logistic import LogisticRegressionBaseline, LogisticRegressionConfig
        return LogisticRegressionBaseline(LogisticRegressionConfig(class_weight={-1: 2, 1: 1}))
    if name == "xgboost":
        pytest.importorskip("xgboost")
        from prooflab.models.xgboost import XGBoostConfig, XGBoostModel
        return XGBoostModel(XGBoostConfig(n_estimators=2))
    pytest.importorskip("torch")
    from prooflab.models.neural import NeuralNetworkConfig, NeuralNetworkModel
    return NeuralNetworkModel(NeuralNetworkConfig(hidden_units=(4,), epochs=2))


@pytest.mark.parametrize("name", ["random", "majority", "logistic", "xgboost", "neural"])
def test_complete_round_trip(name, data, metadata, tmp_path):
    x, y = data
    model = make_model(name).fit(x, y, (x + 0.1, y))
    path = tmp_path / f"{name}.plmodel"
    manifest = save_artifact(model, path, **metadata)
    loaded = load_artifact(path, trusted=True, expected_feature_order=["signal"])
    np.testing.assert_array_equal(model.predict(x), loaded.model.predict(x))
    np.testing.assert_allclose(model.predict_proba(x), loaded.model.predict_proba(x))
    assert manifest.feature_schema == {"signal": "float64"}
    assert loaded.manifest.training.dataset_checksum == "a" * 64
    assert loaded.manifest.feature_metadata == metadata["feature_metadata"]
    assert loaded.manifest.classes == [-1, 1]
    assert inspect_artifact(path) == loaded.manifest
    if name == "neural":
        assert loaded.manifest.best_epoch is not None
        assert loaded.manifest.training_history == model.history_


def test_immutable_and_unfitted(data, metadata, tmp_path):
    x, y = data
    path = tmp_path / "model.plmodel"
    with pytest.raises(ValueError, match="unfitted"):
        save_artifact(MajorityClassifier(), path, **metadata)
    assert not path.exists()
    model = MajorityClassifier().fit(x, y)
    save_artifact(model, path, **metadata)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_artifact(model, path, **metadata)
    assert path.read_bytes() == original


def test_trust_and_package_boundary(data, metadata, tmp_path, monkeypatch):
    x, y = data
    path = tmp_path / "model.plmodel"
    save_artifact(MajorityClassifier().fit(x, y), path, **metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("Deserialization must not occur")

    monkeypatch.setattr("prooflab.models.artifacts.joblib.load", forbidden)
    inspect_artifact(path)
    with pytest.raises(ValueError, match="trusted=True"):
        load_artifact(path)
    with pytest.raises(ValueError, match="unsupported"):
        load_artifact(tmp_path / "strategy.plb", trusted=True)
    with pytest.raises(ValueError, match="feature order"):
        load_artifact(path, trusted=True, expected_feature_order=["other"])


def rewrite(path, *, manifest_update=None, payload_update=None):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("model.joblib")
    if manifest_update:
        manifest_update(manifest)
    if payload_update:
        payload = payload_update(payload)
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.joblib", payload)


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(format_version=99), "format_version"),
    (lambda m: m["dependencies"].update(python="0.0"), "dependency mismatch"),
    (lambda m: m.update(feature_order=["wrong"]), "schema"),
    (lambda m: m.update(model_name="wrong"), "does not match"),
])
def test_manifest_rejection(change, match, data, metadata, tmp_path):
    x, y = data
    path = tmp_path / "model.plmodel"
    save_artifact(MajorityClassifier().fit(x, y), path, **metadata)
    rewrite(path, manifest_update=change)
    with pytest.raises(ValueError, match=match):
        load_artifact(path, trusted=True)


def test_corruption_before_deserialization(data, metadata, tmp_path, monkeypatch):
    x, y = data
    path = tmp_path / "model.plmodel"
    save_artifact(MajorityClassifier().fit(x, y), path, **metadata)
    rewrite(path, payload_update=lambda data: data + b"corruption")
    monkeypatch.setattr("prooflab.models.artifacts.joblib.load", lambda *_: pytest.fail("Unpickle"))
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(path, trusted=True)


def test_missing_metadata_and_overlap(data, metadata, tmp_path):
    x, y = data
    with pytest.raises(ValueError, match="schema"):
        save_artifact(MajorityClassifier().fit(x, y), tmp_path / "model.plmodel",
                      training=metadata["training"], feature_metadata=[])
    invalid = metadata["training"].model_dump()
    invalid["validation_start"] = invalid["train_start"]
    with pytest.raises(ValueError, match="overlap"):
        TrainingMetadata(**invalid)


def test_missing_archive_component(tmp_path):
    path = tmp_path / "incomplete.plmodel"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="exactly"):
        inspect_artifact(path)
