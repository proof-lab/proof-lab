"""Unit tests for system health monitoring telemetry."""

from __future__ import annotations

from prooflab.monitoring.health import (
    ComponentHealth,
    ComponentStatus,
    HealthMonitor,
)


def test_health_monitor_all_healthy() -> None:
    """Test health evaluation when all components report healthy."""
    monitor = HealthMonitor()

    monitor.register_checker(
        "database",
        lambda: ComponentHealth(component_name="database", status=ComponentStatus.HEALTHY),
    )
    monitor.register_checker(
        "broker_adapter",
        lambda: ComponentHealth(component_name="broker_adapter", status=ComponentStatus.HEALTHY),
    )
    monitor.register_checker(
        "risk_engine",
        lambda: ComponentHealth(component_name="risk_engine", status=ComponentStatus.HEALTHY),
    )

    report = monitor.evaluate_health()
    assert report.overall_status == ComponentStatus.HEALTHY
    assert len(report.components) == 3
    assert "All 3 monitored components are HEALTHY" in report.summary


def test_health_monitor_degraded_state() -> None:
    """Test health evaluation with degraded component."""
    monitor = HealthMonitor()

    monitor.register_checker(
        "data_feed",
        lambda: ComponentHealth(
            component_name="data_feed",
            status=ComponentStatus.DEGRADED,
            message="High latency detected (150ms)",
        ),
    )
    monitor.register_checker(
        "model_engine",
        lambda: ComponentHealth(component_name="model_engine", status=ComponentStatus.HEALTHY),
    )

    report = monitor.evaluate_health()
    assert report.overall_status == ComponentStatus.DEGRADED
    assert "System DEGRADED" in report.summary


def test_health_monitor_critical_failure_and_exception() -> None:
    """Test critical failure and unhandled exception in checker callback."""
    monitor = HealthMonitor()

    def throwing_checker() -> ComponentHealth:
        raise ConnectionResetError("Broker connection reset")

    monitor.register_checker("broker", throwing_checker, is_critical=True)
    monitor.register_checker(
        "news_feed",
        lambda: ComponentHealth(component_name="news_feed", status=ComponentStatus.HEALTHY),
        is_critical=False,
    )

    report = monitor.evaluate_health()
    assert report.overall_status == ComponentStatus.FAILED
    assert report.components["broker"].status == ComponentStatus.FAILED
    assert "Broker connection reset" in report.components["broker"].message
