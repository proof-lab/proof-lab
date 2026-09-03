"""Emergency kill switch system with state persistence and immutable audit logging."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KillSwitchPolicy(StrEnum):
    """Position liquidation policy upon emergency kill switch activation."""

    CLOSE_ALL = "CLOSE_ALL"
    CLOSE_LOSING = "CLOSE_LOSING"
    HOLD_OPEN = "HOLD_OPEN"


class KillSwitchState(BaseModel):
    """Persistent schema representing current kill switch operational state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_active: bool = False
    activated_at_utc: str | None = None
    triggered_by: str | None = None
    reason: str | None = None
    policy: KillSwitchPolicy = KillSwitchPolicy.HOLD_OPEN

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> KillSwitchState:
        return cls.model_validate(json.loads(json_str))


class KillSwitchAuditEvent(BaseModel):
    """Immutable audit record emitted during kill switch activation or reset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: f"ks-{uuid.uuid4().hex[:12]}")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    action: str  # "ACTIVATED" | "RESET"
    actor: str
    reason: str
    policy: KillSwitchPolicy
    cancelled_orders: list[str] = Field(default_factory=list)
    liquidated_positions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class KillSwitch:
    """Sovereign emergency control system blocking new orders and enforcing liquidation."""

    def __init__(
        self,
        state_file: Path | str | None = None,
        default_policy: KillSwitchPolicy = KillSwitchPolicy.CLOSE_ALL,
    ) -> None:
        self.state_file = Path(state_file) if state_file else None
        self.default_policy = default_policy
        self._state = KillSwitchState(policy=default_policy)
        self._audit_log: list[KillSwitchAuditEvent] = []

        if self.state_file and self.state_file.exists():
            self.load_state()

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def audit_history(self) -> list[KillSwitchAuditEvent]:
        return list(self._audit_log)

    def activate(
        self,
        actor: str,
        reason: str,
        policy: KillSwitchPolicy | None = None,
        pending_order_ids: list[str] | None = None,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> KillSwitchAuditEvent:
        """Activate the kill switch, halt new orders, and emit an audit event."""
        chosen_policy = policy or self.default_policy
        now_utc = datetime.now(UTC).isoformat()

        self._state = KillSwitchState(
            is_active=True,
            activated_at_utc=now_utc,
            triggered_by=actor,
            reason=reason,
            policy=chosen_policy,
        )

        cancelled = list(pending_order_ids or [])
        liquidated: list[str] = []

        if open_positions:
            for pos in open_positions:
                pos_id = str(pos.get("position_id", pos.get("symbol", "unknown")))
                unrealized_pnl = float(pos.get("unrealized_pnl", 0.0))

                if chosen_policy == KillSwitchPolicy.CLOSE_ALL:
                    liquidated.append(pos_id)
                elif chosen_policy == KillSwitchPolicy.CLOSE_LOSING and unrealized_pnl < 0:
                    liquidated.append(pos_id)

        event = KillSwitchAuditEvent(
            action="ACTIVATED",
            actor=actor,
            reason=reason,
            policy=chosen_policy,
            cancelled_orders=cancelled,
            liquidated_positions=liquidated,
        )
        self._audit_log.append(event)
        self.save_state()
        return event

    def reset(self, actor: str, reason: str) -> KillSwitchAuditEvent:
        """Reset the kill switch back to normal operations."""
        event = KillSwitchAuditEvent(
            action="RESET",
            actor=actor,
            reason=reason,
            policy=self.default_policy,
        )
        self._state = KillSwitchState(is_active=False, policy=self.default_policy)
        self._audit_log.append(event)
        self.save_state()
        return event

    def save_state(self) -> None:
        """Persist state to disk if a state file path was provided."""
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(self._state.to_json(), encoding="utf-8")

    def load_state(self) -> None:
        """Load state from disk."""
        if self.state_file and self.state_file.exists():
            content = self.state_file.read_text(encoding="utf-8")
            self._state = KillSwitchState.from_json(content)
