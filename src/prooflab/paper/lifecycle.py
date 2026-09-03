"""Strategy lifecycle state machine enforcing explicit human approval gates."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class StrategyLifecycleState(StrEnum):
    """Explicit governance lifecycle progression for trading strategies."""

    RESEARCH = "RESEARCH"
    VALIDATED = "VALIDATED"
    PAPER_TRADING = "PAPER_TRADING"
    APPROVED = "APPROVED"
    LIVE_ENABLED = "LIVE_ENABLED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class LifecycleTransitionEvent(BaseModel):
    """Immutable audit record of a lifecycle state change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: datetime
    from_state: StrategyLifecycleState
    to_state: StrategyLifecycleState
    reason: str
    authorized_by: str = "system"


ALLOWED_TRANSITIONS: dict[StrategyLifecycleState, set[StrategyLifecycleState]] = {
    StrategyLifecycleState.RESEARCH: {
        StrategyLifecycleState.VALIDATED,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.VALIDATED: {
        StrategyLifecycleState.PAPER_TRADING,
        StrategyLifecycleState.RESEARCH,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.PAPER_TRADING: {
        StrategyLifecycleState.APPROVED,
        StrategyLifecycleState.SUSPENDED,
        StrategyLifecycleState.RESEARCH,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.APPROVED: {
        StrategyLifecycleState.LIVE_ENABLED,
        StrategyLifecycleState.SUSPENDED,
        StrategyLifecycleState.PAPER_TRADING,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.LIVE_ENABLED: {
        StrategyLifecycleState.SUSPENDED,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.SUSPENDED: {
        StrategyLifecycleState.PAPER_TRADING,
        StrategyLifecycleState.APPROVED,
        StrategyLifecycleState.RETIRED,
    },
    StrategyLifecycleState.RETIRED: set(),
}


class StrategyLifecycleManager:
    """Manages strategy approval transitions with hard gates against auto-live enablement."""

    def __init__(
        self,
        strategy_id: str,
        initial_state: StrategyLifecycleState = StrategyLifecycleState.RESEARCH,
    ) -> None:
        self.strategy_id = strategy_id
        self._current_state = initial_state
        self._history: list[LifecycleTransitionEvent] = []

    @property
    def current_state(self) -> StrategyLifecycleState:
        return self._current_state

    @property
    def is_paper_trading_allowed(self) -> bool:
        return self._current_state in {
            StrategyLifecycleState.PAPER_TRADING,
            StrategyLifecycleState.APPROVED,
            StrategyLifecycleState.LIVE_ENABLED,
        }

    @property
    def is_live_trading_allowed(self) -> bool:
        return self._current_state == StrategyLifecycleState.LIVE_ENABLED

    @property
    def history(self) -> list[LifecycleTransitionEvent]:
        return list(self._history)

    def transition_to(
        self,
        target_state: StrategyLifecycleState,
        reason: str,
        explicit_human_approval: bool = False,
        authorized_by: str = "system",
        timestamp: datetime | None = None,
    ) -> LifecycleTransitionEvent:
        """Execute state transition if valid under lifecycle rules."""
        if target_state == self._current_state:
            raise ValueError(f"Strategy is already in state: {self._current_state}")

        valid_targets = ALLOWED_TRANSITIONS.get(self._current_state, set())
        if target_state not in valid_targets:
            raise ValueError(
                f"Illegal transition: {self._current_state} -> {target_state}. "
                f"Allowed targets: {valid_targets}"
            )

        # Gate: Never enable live trading automatically
        if target_state == StrategyLifecycleState.LIVE_ENABLED and not explicit_human_approval:
            raise PermissionError(
                "Live trading activation requires explicit human approval. "
                "Automated live enablement is strictly prohibited."
            )

        event = LifecycleTransitionEvent(
            timestamp_utc=timestamp or datetime.now(UTC),
            from_state=self._current_state,
            to_state=target_state,
            reason=reason,
            authorized_by=authorized_by,
        )

        self._current_state = target_state
        self._history.append(event)
        return event
