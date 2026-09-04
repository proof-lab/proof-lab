"""Structured audit logging system for all consequential system actions."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AuditEventType(StrEnum):
    """Classification of all consequential events across Proof Lab."""

    MODEL_TRAINED = "MODEL_TRAINED"
    MODEL_APPROVED = "MODEL_APPROVED"
    MODEL_IMPORTED = "MODEL_IMPORTED"
    MODEL_DEPLOYED = "MODEL_DEPLOYED"
    MODEL_SUSPENDED = "MODEL_SUSPENDED"
    MODEL_RETIRED = "MODEL_RETIRED"

    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"

    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACKNOWLEDGED = "ORDER_ACKNOWLEDGED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_CLOSED = "ORDER_CLOSED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FAILED = "ORDER_FAILED"

    RISK_LIMIT_TRIGGERED = "RISK_LIMIT_TRIGGERED"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_RESET = "KILL_SWITCH_RESET"

    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    DATA_STALE = "DATA_STALE"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"
    POSITION_RECONCILED = "POSITION_RECONCILED"
    LIFECYCLE_TRANSITION = "LIFECYCLE_TRANSITION"
    SYSTEM_HEALTH_CHANGE = "SYSTEM_HEALTH_CHANGE"
    SECURITY_ALERT = "SECURITY_ALERT"


class AuditSeverity(StrEnum):
    """Severity classification for audit records."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditRecord(BaseModel):
    """Immutable structured record capturing a consequential platform action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    actor: str = "system"  # User ID, "system", or "human_operator"
    strategy_id: str | None = None
    symbol: str | None = None
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize record to JSON string with ISO datetime format."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "timestamp_utc": self.timestamp_utc.isoformat(),
                "event_type": self.event_type.value,
                "severity": self.severity.value,
                "actor": self.actor,
                "strategy_id": self.strategy_id,
                "symbol": self.symbol,
                "message": self.message,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> AuditRecord:
        """Deserialize record from JSON string."""
        data = json.loads(json_str)
        data["timestamp_utc"] = datetime.fromisoformat(data["timestamp_utc"])
        data["event_type"] = AuditEventType(data["event_type"])
        data["severity"] = AuditSeverity(data["severity"])
        return cls(**data)


class AuditLogger:
    """Append-only structured audit logger with in-memory caching and JSONL persistence."""

    def __init__(self, log_file: Path | str | None = None) -> None:
        self.log_file = Path(log_file) if log_file else None
        self._records: list[AuditRecord] = []

        if self.log_file and self.log_file.exists():
            self._load_records()

    def _load_records(self) -> None:
        """Load historical records from JSONL file."""
        if not self.log_file or not self.log_file.exists():
            return
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._records.append(AuditRecord.from_json(line))
                    except Exception as exc:
                        logger.warning("Failed to parse audit log entry: %s", exc)

    def log(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        actor: str = "system",
        strategy_id: str | None = None,
        symbol: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditRecord:
        """Record and persist an audit event."""
        record = AuditRecord(
            timestamp_utc=timestamp or datetime.now(UTC),
            event_type=event_type,
            severity=severity,
            actor=actor,
            strategy_id=strategy_id,
            symbol=symbol,
            message=message,
            metadata=metadata or {},
        )

        self._records.append(record)

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")

        log_level = logging.INFO
        if severity == AuditSeverity.WARNING:
            log_level = logging.WARNING
        elif severity in (AuditSeverity.ERROR, AuditSeverity.CRITICAL):
            log_level = logging.ERROR

        logger.log(
            log_level,
            "AUDIT [%s] %s (actor=%s, strat=%s, sym=%s): %s",
            severity.value,
            event_type.value,
            actor,
            strategy_id or "-",
            symbol or "-",
            message,
        )
        return record

    def query(
        self,
        event_type: AuditEventType | None = None,
        severity: AuditSeverity | None = None,
        strategy_id: str | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AuditRecord]:
        """Query audit records matching filter criteria."""
        results = self._records
        if event_type:
            results = [r for r in results if r.event_type == event_type]
        if severity:
            results = [r for r in results if r.severity == severity]
        if strategy_id:
            results = [r for r in results if r.strategy_id == strategy_id]
        if symbol:
            results = [r for r in results if r.symbol == symbol]
        if start_time:
            results = [r for r in results if r.timestamp_utc >= start_time]
        if end_time:
            results = [r for r in results if r.timestamp_utc <= end_time]
        return list(results)

    @property
    def records(self) -> list[AuditRecord]:
        """Return all in-memory audit records."""
        return list(self._records)
