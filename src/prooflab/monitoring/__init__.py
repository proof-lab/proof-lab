"""Observability, telemetry, drift monitoring, and security hardening."""

from prooflab.monitoring.audit import (
    AuditEventType,
    AuditLogger,
    AuditRecord,
    AuditSeverity,
)
from prooflab.monitoring.health import (
    ComponentHealth,
    ComponentStatus,
    HealthMonitor,
    SystemHealthReport,
)

__all__ = [
    "AuditEventType",
    "AuditLogger",
    "AuditRecord",
    "AuditSeverity",
    "ComponentHealth",
    "ComponentStatus",
    "HealthMonitor",
    "SystemHealthReport",
]
