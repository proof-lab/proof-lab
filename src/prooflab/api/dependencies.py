"""FastAPI dependency injection providers for quantitative engines and state."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from prooflab.api.jobs import JobManager
from prooflab.data.repository import ParquetRepository
from prooflab.paper.lifecycle import StrategyLifecycleManager
from prooflab.risk.engine import RiskEngine
from prooflab.risk.kill_switch import KillSwitch

_GLOBAL_JOB_MANAGER: JobManager | None = None
_GLOBAL_KILL_SWITCH: KillSwitch | None = None
_GLOBAL_RISK_ENGINE: RiskEngine | None = None
_GLOBAL_REPOSITORY: ParquetRepository | None = None
_GLOBAL_LIFECYCLE: StrategyLifecycleManager | None = None


def get_job_manager() -> JobManager:
    """Provide singleton instance of the background JobManager."""
    global _GLOBAL_JOB_MANAGER
    if _GLOBAL_JOB_MANAGER is None:
        _GLOBAL_JOB_MANAGER = JobManager()
    return _GLOBAL_JOB_MANAGER


def get_kill_switch() -> KillSwitch:
    """Provide singleton instance of the emergency KillSwitch."""
    global _GLOBAL_KILL_SWITCH
    if _GLOBAL_KILL_SWITCH is None:
        _GLOBAL_KILL_SWITCH = KillSwitch()
    return _GLOBAL_KILL_SWITCH


def get_risk_engine(
    kill_switch: Annotated[KillSwitch, Depends(get_kill_switch)],
) -> RiskEngine:
    """Provide singleton instance of the sovereign RiskEngine."""
    global _GLOBAL_RISK_ENGINE
    if _GLOBAL_RISK_ENGINE is None:
        _GLOBAL_RISK_ENGINE = RiskEngine(kill_switch=kill_switch)
    return _GLOBAL_RISK_ENGINE


def get_dataset_repository() -> ParquetRepository:
    """Provide dataset repository instance."""
    global _GLOBAL_REPOSITORY
    if _GLOBAL_REPOSITORY is None:
        default_dir = Path(tempfile.gettempdir()) / "prooflab_data"
        default_dir.mkdir(parents=True, exist_ok=True)
        _GLOBAL_REPOSITORY = ParquetRepository(base_dir=default_dir)
    return _GLOBAL_REPOSITORY


def get_lifecycle_manager() -> StrategyLifecycleManager:
    """Provide strategy lifecycle manager instance."""
    global _GLOBAL_LIFECYCLE
    if _GLOBAL_LIFECYCLE is None:
        _GLOBAL_LIFECYCLE = StrategyLifecycleManager(strategy_id="default-strat")
    return _GLOBAL_LIFECYCLE
