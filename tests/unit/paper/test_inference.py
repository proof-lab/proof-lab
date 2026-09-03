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
