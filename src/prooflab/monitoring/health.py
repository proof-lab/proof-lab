"""System health monitoring tracking data feeds, database, models, features, risk, and broker."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ComponentStatus(StrEnum):
    """Health status classification for individual platform subsystems."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ComponentHealth(BaseModel):
    """Detailed health status report for a single system component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_name: str
    status: ComponentStatus
    message: str = "Operating normally"
    last_checked_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = 0.0
    is_critical: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemHealthReport(BaseModel):
    """Aggregated system-wide telemetry and health evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_status: ComponentStatus
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    summary: str


class HealthMonitor:
    """Central health telemetry engine running diagnostic checkers across all platform domains."""

    def __init__(self) -> None:
        self._checkers: dict[str, tuple[Callable[[], ComponentHealth], bool]] = {}

    def register_checker(
        self,
        component_name: str,
        checker: Callable[[], ComponentHealth],
        is_critical: bool = True,
    ) -> None:
        """Register a diagnostic health checking callback for a component."""
        self._checkers[component_name] = (checker, is_critical)

    def evaluate_health(self) -> SystemHealthReport:
        """Execute all diagnostic checks and assemble the unified system health report."""
        components_health: dict[str, ComponentHealth] = {}
        has_critical_failure = False
        has_degraded = False

        for name, (checker, is_critical) in self._checkers.items():
            start = time.perf_counter()
            try:
                health = checker()
                latency = round((time.perf_counter() - start) * 1000.0, 2)
                # Ensure latency and name are attached
                health = health.model_copy(
                    update={"latency_ms": latency, "is_critical": is_critical}
                )
            except Exception as exc:
                latency = round((time.perf_counter() - start) * 1000.0, 2)
                logger.exception("Health check failed for component %s", name)
                health = ComponentHealth(
                    component_name=name,
                    status=ComponentStatus.FAILED,
                    message=f"Diagnostic check exception: {exc}",
                    latency_ms=latency,
                    is_critical=is_critical,
                )

            components_health[name] = health

            if health.status == ComponentStatus.FAILED:
                if is_critical:
                    has_critical_failure = True
                else:
                    has_degraded = True
            elif health.status == ComponentStatus.DEGRADED:
                has_degraded = True

        if has_critical_failure:
            overall_status = ComponentStatus.FAILED
        elif has_degraded:
            overall_status = ComponentStatus.DEGRADED
        else:
            overall_status = ComponentStatus.HEALTHY

        failed_names = [
            k for k, v in components_health.items() if v.status == ComponentStatus.FAILED
        ]
        degraded_names = [
            k for k, v in components_health.items() if v.status == ComponentStatus.DEGRADED
        ]


        if overall_status == ComponentStatus.HEALTHY:
            summary = f"All {len(components_health)} monitored components are HEALTHY."
        elif overall_status == ComponentStatus.DEGRADED:
            summary = (
                f"System DEGRADED. Degraded components: {degraded_names or 'none'}, "
                f"non-critical failures: {failed_names or 'none'}."
            )
        else:
            summary = f"System FAILED. Critical component failures detected: {failed_names}."

        logger.info("Health evaluation completed: overall=%s (%s)", overall_status.value, summary)
        return SystemHealthReport(
            overall_status=overall_status,
            components=components_health,
            summary=summary,
        )
