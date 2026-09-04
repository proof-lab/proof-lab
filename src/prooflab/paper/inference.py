"""Live model inference engine enforcing complete artifact validation and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.models.artifacts import ModelArtifact, load_artifact


class InferencePrediction(BaseModel):
    """Result of live model inference on a single feature vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    model_name: str
    signal_direction: int = 0  # -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
    probabilities: dict[int, float] = Field(default_factory=dict)
    confidence: float = 0.0
    rejection_reason: str | None = None


class LiveInferenceEngine:
    """Executes live model inference while rejecting incomplete or untrusted artifacts."""

    def __init__(
        self,
        artifact: ModelArtifact | None = None,
        min_confidence_threshold: float = 0.55,
    ) -> None:
        self.artifact = artifact
        self.min_confidence_threshold = min_confidence_threshold
        if self.artifact:
            self._validate_artifact(self.artifact)

    @classmethod
    def from_file(
        cls,
        artifact_path: Path | str,
        trusted: bool = True,
        min_confidence_threshold: float = 0.55,
    ) -> LiveInferenceEngine:
        """Load and strictly validate a model artifact from disk."""
        target = Path(artifact_path)
        if not target.exists():
            raise FileNotFoundError(f"Model artifact file not found: {target}")

        artifact = load_artifact(target, trusted=trusted)
        return cls(artifact=artifact, min_confidence_threshold=min_confidence_threshold)

    def _validate_artifact(self, artifact: ModelArtifact) -> None:
        """Enforce completeness of artifact manifest, model state, and feature schema."""
        manifest = artifact.manifest
        if not manifest.feature_order:
            raise ValueError("Artifact is incomplete: missing feature_order")

        if not manifest.feature_schema:
            raise ValueError("Artifact is incomplete: missing feature_schema")

        if not manifest.classes:
            raise ValueError("Artifact is incomplete: missing classes")

        if not getattr(artifact.model, "is_fitted", False):
            raise ValueError("Artifact model is not fitted")

    def predict_live(
        self,
        features: dict[str, float] | pd.DataFrame | pd.Series,
    ) -> InferencePrediction:
        """Execute single-observation inference against the validated model artifact."""
        if self.artifact is None:
            return InferencePrediction(
                is_valid=False,
                model_name="unloaded",
                rejection_reason="No model artifact loaded in inference engine",
            )

        manifest = self.artifact.manifest
        model = self.artifact.model

        # 1. Feature completeness and ordering check
        if isinstance(features, dict):
            missing_feats = [f for f in manifest.feature_order if f not in features]
            if missing_feats:
                return InferencePrediction(
                    is_valid=False,
                    model_name=manifest.model_name,
                    rejection_reason=f"Missing required features: {missing_feats}",
                )
            row_data = {f: [features[f]] for f in manifest.feature_order}
            df = pd.DataFrame(row_data)
        elif isinstance(features, pd.Series):
            missing_feats = [f for f in manifest.feature_order if f not in features.index]
            if missing_feats:
                return InferencePrediction(
                    is_valid=False,
                    model_name=manifest.model_name,
                    rejection_reason=f"Missing required features: {missing_feats}",
                )
            row_data = {f: [features[f]] for f in manifest.feature_order}
            df = pd.DataFrame(row_data)
        elif isinstance(features, pd.DataFrame):
            missing_feats = [f for f in manifest.feature_order if f not in features.columns]
            if missing_feats:
                return InferencePrediction(
                    is_valid=False,
                    model_name=manifest.model_name,
                    rejection_reason=f"Missing required features: {missing_feats}",
                )
            df = features[manifest.feature_order].copy()
        else:
            return InferencePrediction(
                is_valid=False,
                model_name=manifest.model_name,
                rejection_reason=f"Unsupported features container type: {type(features)}",
            )

        # 2. Model Prediction
        try:
            proba_arr = model.predict_proba(df)
            pred_classes = getattr(model, "classes_", manifest.classes)
            # Probability map
            prob_map: dict[int, float] = {}
            for i, cls_val in enumerate(pred_classes):
                prob_map[int(cls_val)] = float(proba_arr[0, i])

            # Determine signal direction
            # If binary (e.g. classes = [0, 1] or [-1, 1])
            best_class = int(pred_classes[int(np.argmax(proba_arr[0]))])
            best_prob = float(np.max(proba_arr[0]))

            direction = best_class if best_prob >= self.min_confidence_threshold else 0

            return InferencePrediction(
                is_valid=True,
                model_name=manifest.model_name,
                signal_direction=direction,
                probabilities=prob_map,
                confidence=round(best_prob, 4),
            )
        except Exception as e:
            return InferencePrediction(
                is_valid=False,
                model_name=manifest.model_name,
                rejection_reason=f"Inference execution failed: {e}",
            )
