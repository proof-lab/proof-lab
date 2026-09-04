"""Proof Lab API domain routers package."""

from prooflab.api.routers.backtests import router as backtests_router
from prooflab.api.routers.data import router as data_router
from prooflab.api.routers.experiments import router as experiments_router
from prooflab.api.routers.features import router as features_router
from prooflab.api.routers.live import router as live_router
from prooflab.api.routers.risk import router as risk_router
from prooflab.api.routers.strategies import router as strategies_router
from prooflab.api.routers.system import router as system_router

__all__ = [
    "backtests_router",
    "data_router",
    "experiments_router",
    "features_router",
    "live_router",
    "risk_router",
    "strategies_router",
    "system_router",
]
