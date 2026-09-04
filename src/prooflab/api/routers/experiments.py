"""Experiments and model training endpoints using non-blocking background workers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, status

from prooflab.api.dependencies import get_dataset_repository, get_job_manager
from prooflab.api.jobs import JobManager
from prooflab.api.schemas import JobResponse, TrainModelRequest
from prooflab.data.repository import ParquetRepository
from prooflab.experiments.training import ModelSpec, TrainingConfig, run_training
from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset
from prooflab.labels.config import Direction, DistanceUnit, SetupConfig

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


def _run_training_task(
    repo: ParquetRepository,
    req: TrainModelRequest,
) -> dict[str, Any]:
    """Background worker task for model training without blocking API loop."""
    meta = repo.get_metadata(req.dataset_id)
    preset_enum = getattr(FeatureSetPreset, req.feature_preset, FeatureSetPreset.PRICE_ONLY)
    pipeline = FeaturePipeline(features=preset_enum)
    feature_names = tuple(pipeline.get_feature_names())

    # Time boundaries
    delta = meta.end_time - meta.start_time
    val_start = meta.start_time + delta * 0.7
    blind_start = meta.end_time + (delta * 0.5)

    setup = SetupConfig(
        direction=Direction.LONG,
        target_distance=req.target_pips,
        stop_distance=req.stop_pips,
        horizon_bars=req.horizon_bars,
        unit=DistanceUnit.PIPS,
        point_value=0.0001,
        entry_price_col="close",
    )

    kind_map = {
        "majority_classifier": "majority",
        "random_classifier": "random",
        "logistic_regression": "logistic",
        "majority": "majority",
        "random": "random",
        "logistic": "logistic",
        "xgboost": "xgboost",
        "neural": "neural",
        "svm": "svm",
    }
    model_kind = kind_map.get(req.model_name.lower(), "majority")
    valid_kind = cast(
        Literal["random", "majority", "logistic", "xgboost", "neural", "svm", "simple_rule"],
        model_kind,
    )

    config = TrainingConfig(
        dataset_id=req.dataset_id,
        setup=setup,
        feature_names=feature_names,
        models=(ModelSpec(kind=valid_kind, parameters=req.model_params),),
        validation_start=val_start,
        blind_start=blind_start,
    )

    out_dir = Path(tempfile.mkdtemp()) / f"train_{req.dataset_id}"
    res = run_training(repository=repo, config=config, output_dir=out_dir)

    return {
        "dataset_id": req.dataset_id,
        "models_trained": list(res.models.keys()),
        "report_file": str(res.report_path),
        "feature_count": len(feature_names),
    }


@router.post("/train", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training_experiment(
    req: TrainModelRequest,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> JobResponse:
    """Submit a model training experiment to the background job queue."""
    try:
        repo.get_metadata(req.dataset_id)
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {req.dataset_id} not found in repository",
        ) from err

    job = job_manager.submit_job(
        job_type="MODEL_TRAINING",
        func=_run_training_task,
        params=req.model_dump(),
        repo=repo,
        req=req,
    )

    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )
