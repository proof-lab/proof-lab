"""Unit tests for prooflab.paper.inference (Live Inference & Artifact Completeness)."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import ArtifactManifest, ModelArtifact, TrainingMetadata
from prooflab.models.baselines import MajorityClassifier
from prooflab.paper.inference import InferencePrediction, LiveInferenceEngine


def _make_dummy_artifact(is_fitted: bool = True) -> ModelArtifact:
    model = MajorityClassifier()
    if is_fitted:
        train_df = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [0.1, 0.2, 0.3]})
        y_series = pd.Series([1, 1, 0])
        model.fit(train_df, y_series)

    meta_a = FeatureMetadata(
        feature_name="feat_a",
        family=FeatureFamily.PRICE,
        lookback_period=1,
        description="Feature A",
        required_columns=["close"],
    )
    meta_b = FeatureMetadata(
        feature_name="feat_b",
        family=FeatureFamily.VOLATILITY,
        lookback_period=1,
        description="Feature B",
        required_columns=["close"],
    )

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    training_meta = TrainingMetadata(
        dataset_id="ds-test",
        dataset_checksum="a" * 64,
        setup_config={"target": 20, "stop": 20},
        train_start=t0,
        train_end=t1,
        train_rows=100,
    )

    deps = {
        "python": "3.12",
        "prooflab": "0.1",
        "numpy": "1.26",
        "pandas": "2.2",
        "joblib": "1.3",
        "pydantic": "2.6",
    }

    manifest = ArtifactManifest(
        created_at=datetime.now(UTC),
        model_name="majority_classifier",
        model_type="baseline",
        model_params={},
        feature_order=["feat_a", "feat_b"],
        feature_schema={"feat_a": "float64", "feat_b": "float64"},
        feature_metadata=[meta_a, meta_b],
        classes=[0, 1],
        preprocessing="identity",
        training=training_meta,
        dependencies=deps,
        payload_checksum="b" * 64,
    )

    return ModelArtifact(model=model, manifest=manifest)


def test_live_inference_valid_prediction() -> None:
    artifact = _make_dummy_artifact(is_fitted=True)
    engine = LiveInferenceEngine(artifact=artifact, min_confidence_threshold=0.50)

    # Correct features
    pred = engine.predict_live({"feat_a": 1.5, "feat_b": 0.25})

    assert isinstance(pred, InferencePrediction)
    assert pred.is_valid is True
    assert pred.model_name == "majority_classifier"
    assert pred.signal_direction == 1
    assert 1 in pred.probabilities


def test_live_inference_missing_feature_rejection() -> None:
    artifact = _make_dummy_artifact(is_fitted=True)
    engine = LiveInferenceEngine(artifact=artifact)

    # Missing feat_b
    pred = engine.predict_live({"feat_a": 1.5})

    assert pred.is_valid is False
    assert "Missing required features" in str(pred.rejection_reason)


def test_live_inference_unfitted_model_rejection() -> None:
    unfitted_artifact = _make_dummy_artifact(is_fitted=False)
    with pytest.raises(ValueError, match="Artifact model is not fitted"):
        LiveInferenceEngine(artifact=unfitted_artifact)


def test_live_inference_series_and_dataframe_inputs() -> None:
    artifact = _make_dummy_artifact(is_fitted=True)
    engine = LiveInferenceEngine(artifact=artifact, min_confidence_threshold=0.50)

    # Series
    s = pd.Series({"feat_a": 1.5, "feat_b": 0.25})
    pred_s = engine.predict_live(s)
    assert pred_s.is_valid is True

    # DataFrame
    df = pd.DataFrame([{"feat_a": 1.5, "feat_b": 0.25}])
    pred_df = engine.predict_live(df)
    assert pred_df.is_valid is True

    # Series missing feature
    s_bad = pd.Series({"feat_a": 1.5})
    pred_s_bad = engine.predict_live(s_bad)
    assert pred_s_bad.is_valid is False

    # DataFrame missing feature
    df_bad = pd.DataFrame([{"feat_a": 1.5}])
    pred_df_bad = engine.predict_live(df_bad)
    assert pred_df_bad.is_valid is False

    # Unsupported type
    pred_bad_type = engine.predict_live([1.5, 0.25])  # type: ignore[arg-type]
    assert pred_bad_type.is_valid is False
    assert "Unsupported features container" in str(pred_bad_type.rejection_reason)


def test_live_inference_unloaded_and_missing_file() -> None:
    empty_engine = LiveInferenceEngine()
    pred = empty_engine.predict_live({"feat_a": 1.0})
    assert pred.is_valid is False
    assert "No model artifact loaded" in str(pred.rejection_reason)

    with pytest.raises(FileNotFoundError, match="Model artifact file not found"):
        LiveInferenceEngine.from_file("nonexistent_model.plmodel")
