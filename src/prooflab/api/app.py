"""FastAPI application factory and middleware configuration for Proof Lab."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prooflab.api.dependencies import get_job_manager
from prooflab.api.routers import (
    backtests_router,
    data_router,
    experiments_router,
    features_router,
    live_router,
    risk_router,
    strategies_router,
    system_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and graceful shutdown."""
    job_mgr = get_job_manager()
    yield
    job_mgr.shutdown(wait=False)


def create_app() -> FastAPI:
    """Construct and configure the canonical Proof Lab FastAPI application."""
    app = FastAPI(
        title="Proof Lab Quantitative API",
        description=(
            "Research-first quantitative trading and validation platform API. "
            "Exposes data ingestion, feature generation, ML training, backtesting, "
            "proof robustness evaluation, sovereign risk controls, and strategy packaging."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS for client frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Domain Routers
    app.include_router(system_router)
    app.include_router(data_router)
    app.include_router(features_router)
    app.include_router(experiments_router)
    app.include_router(backtests_router)
    app.include_router(strategies_router)
    app.include_router(risk_router)
    app.include_router(live_router)

    return app


app = create_app()
