"""Feature engine endpoints for feature catalog and preset inspection."""

from __future__ import annotations

from fastapi import APIRouter

from prooflab.api.schemas import FeatureItemResponse, FeaturePresetResponse
from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset

router = APIRouter(prefix="/api/features", tags=["Features"])


@router.get("", response_model=list[FeatureItemResponse])
async def list_features() -> list[FeatureItemResponse]:
    """List all available quantitative features with descriptions and lookback windows."""
    pipeline = FeaturePipeline(features=FeatureSetPreset.ALL_STANDARD)
    return [
        FeatureItemResponse(
            feature_name=meta.feature_name,
            family=meta.family.value,
            description=meta.description,
            lookback_period=meta.lookback_period,
        )
        for meta in pipeline.get_metadata_summary()
    ]


@router.get("/presets", response_model=list[FeaturePresetResponse])
async def list_feature_presets() -> list[FeaturePresetResponse]:
    """List available feature set presets and member feature names."""
    presets = []
    for preset in FeatureSetPreset:
        p = FeaturePipeline(features=preset)
        names = p.get_feature_names()
        presets.append(
            FeaturePresetResponse(
                preset_name=preset.name,
                feature_count=len(names),
                features=names,
            )
        )
    return presets
