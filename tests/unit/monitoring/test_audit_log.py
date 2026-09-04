"""Unit tests for structured audit logging system."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from prooflab.monitoring.audit import (
    AuditEventType,
    AuditLogger,
    AuditSeverity,
)


def test_audit_logger_in_memory() -> None:
    """Test recording and querying audit records in memory."""
    audit = AuditLogger()

    # Log events
    rec1 = audit.log(
        event_type=AuditEventType.MODEL_TRAINED,
        message="Trained XGBoost ensemble model",
        severity=AuditSeverity.INFO,
        strategy_id="STRAT_001",
        metadata={"trees": 100, "val_loss": 0.32},
    )

    rec2 = audit.log(
        event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
        message="Emergency stop activated by operator",
        severity=AuditSeverity.CRITICAL,
        actor="human_operator",
        strategy_id="STRAT_001",
    )

    rec3 = audit.log(
        event_type=AuditEventType.ORDER_FILLED,
        message="Order filled on broker",
        severity=AuditSeverity.INFO,
        strategy_id="STRAT_002",
        symbol="EURUSD",
        metadata={"ticket": 123456, "fill_price": 1.0850},
    )

    assert len(audit.records) == 3

    # Query by event_type
    ks_records = audit.query(event_type=AuditEventType.KILL_SWITCH_ACTIVATED)
    assert len(ks_records) == 1
    assert ks_records[0].event_id == rec2.event_id
    assert ks_records[0].severity == AuditSeverity.CRITICAL

    # Query by strategy_id
    strat1_records = audit.query(strategy_id="STRAT_001")
    assert len(strat1_records) == 2

    # Query by symbol
    sym_records = audit.query(symbol="EURUSD")
    assert len(sym_records) == 1
    assert sym_records[0].event_id == rec3.event_id


def test_audit_logger_file_persistence(tmp_path: Path) -> None:
    """Test persisting audit records to JSONL file and reloading them."""
    log_path = tmp_path / "audit.jsonl"
    logger1 = AuditLogger(log_file=log_path)

    logger1.log(
        event_type=AuditEventType.RISK_LIMIT_TRIGGERED,
        message="Daily loss threshold reached",
        severity=AuditSeverity.WARNING,
        strategy_id="STRAT_RISK_1",
    )
    logger1.log(
        event_type=AuditEventType.ORDER_SUBMITTED,
        message="Order submitted to broker",
        severity=AuditSeverity.INFO,
        strategy_id="STRAT_RISK_1",
        symbol="GBPUSD",
    )

    assert log_path.exists()

    # Create new logger pointing to same file -> records should reload
    logger2 = AuditLogger(log_file=log_path)
    assert len(logger2.records) == 2
    assert logger2.records[0].event_type == AuditEventType.RISK_LIMIT_TRIGGERED
    assert logger2.records[1].symbol == "GBPUSD"
