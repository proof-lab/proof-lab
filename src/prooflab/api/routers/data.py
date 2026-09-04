"""Data engine endpoints for dataset inspection, validation, and health diagnostics."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from prooflab.api.dependencies import get_dataset_repository
from prooflab.api.schemas import (
    DataHealthResponse,
    DatasetSummaryResponse,
    ValidateDataRequest,
)
from prooflab.data.health import generate_health_report
from prooflab.data.repository import ParquetRepository
from prooflab.data.validator import DataValidator

router = APIRouter(prefix="/api/data", tags=["Data"])


@router.get("/datasets", response_model=list[DatasetSummaryResponse])
async def list_datasets(
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> list[DatasetSummaryResponse]:
    """List all registered historical datasets in the repository."""
    metadata_list = repo.list_datasets()
    return [
        DatasetSummaryResponse(
            dataset_id=m.dataset_id,
            symbol=m.symbol,
            timeframe=m.timeframe.value if hasattr(m.timeframe, "value") else str(m.timeframe),
            row_count=m.row_count,
            start_time=m.start_time,
            end_time=m.end_time,
            checksum=m.checksum,
        )
        for m in metadata_list
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetSummaryResponse)
async def get_dataset(
    dataset_id: str,
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> DatasetSummaryResponse:
    """Retrieve metadata for a specific dataset ID."""
    try:
        m = repo.get_metadata(dataset_id)
        return DatasetSummaryResponse(
            dataset_id=m.dataset_id,
            symbol=m.symbol,
            timeframe=m.timeframe.value if hasattr(m.timeframe, "value") else str(m.timeframe),
            row_count=m.row_count,
            start_time=m.start_time,
            end_time=m.end_time,
            checksum=m.checksum,
        )
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        ) from err


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_raw_data(
    req: ValidateDataRequest,
) -> dict[str, Any]:
    """Validate candidate OHLCV bar records against canonical data engine rules."""
    if not req.bars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot validate empty bar list",
        )
    df = pd.DataFrame(req.bars)
    validator = DataValidator()
    report = validator.validate(df)
    return {
        "is_valid": report.is_valid,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "total_records": len(df),
        "issues": [str(i) for i in report.issues],
    }


@router.get("/health/{dataset_id}", response_model=DataHealthResponse)
async def get_dataset_health(
    dataset_id: str,
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> DataHealthResponse:
    """Generate comprehensive health report for a stored dataset."""
    try:
        df, meta = repo.load_dataset(dataset_id)
        report = generate_health_report(df, timeframe=meta.timeframe)
        return DataHealthResponse(
            dataset_id=dataset_id,
            is_valid=report.is_valid,
            total_records=report.row_count,
            quality_score=report.completeness,
            issues=[str(i) for i in report.issues],
        )
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        ) from err
