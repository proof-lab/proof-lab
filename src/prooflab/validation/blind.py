"""Blind test set protection and out-of-sample audit gates.

Ensures the final blind test holdout is strictly isolated from casual inspection,
tuning, model selection, and repeated evaluation abuse. Requires explicit operator
confirmation and persists immutable audit trail records.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from prooflab.models.base import BaseModelWrapper
from prooflab.validation.calibration import probability_quality


class BlindAccessViolationError(PermissionError):
    """Raised when blind test set evaluation is attempted without explicit operator confirmation."""


class BlindMultipleTestingWarning(UserWarning):
    """Emitted when repeated evaluations are performed on the same blind holdout period."""


class BlindEvaluationAudit(BaseModel):
    """Immutable audit entry tracking a formal blind test set evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_id: str
    dataset_id: str
    blind_start: AwareDatetime
    dataset_end: AwareDatetime
    model_name: str
    evaluated_at: AwareDatetime
    operator_confirmed: bool
    confirmation_reason: str
    rows_evaluated: int
    metrics: dict[str, Any]

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> BlindEvaluationAudit:
        return cls.model_validate(json.loads(json_str))


class BlindEvaluationGate:
    """Gatekeeper enforcing blind test set isolation and audit logging."""

    def __init__(
        self,
        ledger_path: Path | str | None = None,
        max_permitted_evaluations: int = 1,
    ) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.max_permitted_evaluations = max_permitted_evaluations
        self._audits: list[BlindEvaluationAudit] = []

        if self.ledger_path is not None and self.ledger_path.exists():
            self._load_ledger()

    def _load_ledger(self) -> None:
        if self.ledger_path is None or not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        audit = BlindEvaluationAudit.from_json(line)
                        self._audits.append(audit)
                    except Exception:
                        pass

    def count_blind_evaluations(
        self,
        dataset_id: str,
        blind_start: AwareDatetime,
        dataset_end: AwareDatetime,
    ) -> int:
        """Count how many times this specific blind period has been evaluated."""
        count = 0
        for audit in self._audits:
            if (
                audit.dataset_id == dataset_id
                and audit.blind_start == blind_start
                and audit.dataset_end == dataset_end
            ):
                count += 1
        return count

    def evaluate(
        self,
        model: BaseModelWrapper,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        experiment_id: str,
        dataset_id: str,
        blind_start: AwareDatetime,
        dataset_end: AwareDatetime,
        confirm_blind_evaluation: bool = False,
        confirmation_reason: str = "",
    ) -> dict[str, Any]:
        """Formally evaluate a frozen model on blind test data under strict audit protocol.

        Args:
            model: Fitted, frozen BaseModelWrapper instance.
            features: Feature DataFrame containing blind test rows.
            labels: Canonical target labels for blind test rows.
            experiment_id: ID of the experiment being evaluated.
            dataset_id: ID of the target dataset.
            blind_start: Start timestamp of the blind interval.
            dataset_end: End timestamp of the dataset.
            confirm_blind_evaluation: Must be explicitly True to unlock blind evaluation.
            confirmation_reason: Non-empty rationale justifying blind unlock.
        """
        # 1. Gate check: Confirmation and reason required
        if not confirm_blind_evaluation:
            raise BlindAccessViolationError(
                "Blind test set evaluation is strictly protected from casual inspection. "
                "Explicit confirmation (confirm_blind_evaluation=True) and a formal "
                "confirmation_reason are required before blind test evaluation."
            )

        clean_reason = confirmation_reason.strip()
        if len(clean_reason) < 10:
            raise ValueError(
                "A meaningful confirmation_reason (at least 10 characters) must be supplied "
                "to justify unlocking the blind test evaluation."
            )

        # 2. Timeline validation
        if not isinstance(features.index, pd.DatetimeIndex) or str(features.index.tz) != "UTC":
            raise ValueError("Blind features index must be a timezone-aware UTC DatetimeIndex.")

        if features.empty:
            raise ValueError("Blind features cannot be empty.")

        if features.index[0] < blind_start:
            raise ValueError(
                f"Blind features contain timestamps before blind start ({blind_start})."
            )

        if features.index[-1] > dataset_end:
            raise ValueError(
                f"Blind features contain timestamps after dataset end ({dataset_end})."
            )

        y_arr = np.asarray(labels)
        if len(features) != len(y_arr):
            raise ValueError(
                f"Length mismatch: features has {len(features)} rows, labels has {len(y_arr)}."
            )

        # 3. Multiple testing check on blind holdout
        prior_evals = self.count_blind_evaluations(dataset_id, blind_start, dataset_end)
        if prior_evals >= self.max_permitted_evaluations:
            warnings.warn(
                f"Repeated blind test evaluation alert (eval count={prior_evals + 1}) for dataset "
                f"'{dataset_id}' on window [{blind_start} to {dataset_end}]. "
                f"Out-of-sample integrity is at risk from repeated evaluation.",
                BlindMultipleTestingWarning,
                stacklevel=2,
            )

        # 4. Execute evaluation
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        accuracy = float(np.mean(predictions == y_arr))

        # Binary action probability metrics if applicable
        action_metrics: dict[str, Any] = {}
        if 1 in model.classes_:
            action_idx = model.classes_.index(1)
            action_probs = probabilities[:, action_idx]
            binary_labels = (y_arr == 1).astype(int)
            action_metrics = probability_quality(action_probs, binary_labels)

        metrics: dict[str, Any] = {
            "rows": len(features),
            "accuracy": accuracy,
            "classes": model.classes_,
            "predicted_distribution": {
                int(c): int((predictions == c).sum()) for c in model.classes_
            },
            "action_calibration": action_metrics,
        }

        # 5. Record immutable audit entry
        now = pd.Timestamp.now(tz="UTC")
        audit = BlindEvaluationAudit(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            blind_start=blind_start,
            dataset_end=dataset_end,
            model_name=model.model_name,
            evaluated_at=now,
            operator_confirmed=True,
            confirmation_reason=clean_reason,
            rows_evaluated=len(features),
            metrics=metrics,
        )

        self._audits.append(audit)

        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(audit.to_json(indent=None) + "\n")

        return {
            "audit_id": audit.audit_id,
            "evaluated_at": now.isoformat(),
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "metrics": metrics,
        }

    def get_audit_history(
        self,
        dataset_id: str | None = None,
    ) -> list[BlindEvaluationAudit]:
        """Return all historical blind evaluation audit records."""
        if dataset_id is not None:
            return [a for a in self._audits if a.dataset_id == dataset_id]
        return list(self._audits)
