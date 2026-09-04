"""Structural and behavioral leakage checks, without blind evaluation.

No finite behavioral test can prove arbitrary code causal. Prefix invariance,
availability evidence, and fit-row provenance are complementary checks; high
predictive accuracy alone is deliberately not treated as proof of leakage.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prooflab.validation.splits import (
    FoldPlan,
    SplitConfig,
    WalkForwardConfig,
    chronological_split,
    validate_timeline,
    walk_forward,
)


@dataclass(frozen=True)
class LeakageReport:
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def require_pass(self) -> None:
        if self.issues:
            raise ValueError("Leakage detected: " + "; ".join(self.issues))


def audit_plan(plan: FoldPlan, index: pd.DatetimeIndex) -> LeakageReport:
    """Rebuild the permitted fold, including preceding walk-forward embargoes."""
    try:
        horizons = pd.Series(plan.horizon_bars, index=index, dtype=int)
        if "mode" in plan.configuration:
            candidates = walk_forward(index, WalkForwardConfig.model_validate(plan.configuration),
                                      dataset_end=pd.Timestamp(plan.dataset_end),
                                      horizon_bars=horizons)
            expected = candidates[plan.fold_id]
        else:
            expected = chronological_split(index, SplitConfig.model_validate(plan.configuration),
                                           dataset_end=pd.Timestamp(plan.dataset_end),
                                           horizon_bars=horizons)
    except (ValueError, IndexError, TypeError, KeyError) as exc:
        return LeakageReport((f"invalid_split_provenance: {exc}",))
    if expected != plan:
        return LeakageReport(("split_or_embargo_mismatch: fold differs from its causal plan",))
    return LeakageReport(())


def audit_fit_rows(fitted_rows: pd.Index, permitted_training_rows: pd.Index) -> LeakageReport:
    """Use for scalers, feature selection, threshold fitting, or estimator provenance."""
    if fitted_rows.empty or not fitted_rows.is_unique or not fitted_rows.isin(
        permitted_training_rows,
    ).all():
        return LeakageReport(("fit_outside_training: fitted rows are not a training-only subset",))
    return LeakageReport(())


def audit_availability(availability: pd.DataFrame) -> LeakageReport:
    """Each cell records when its feature value became available, indexed by entry UTC."""
    index = availability.index
    if (
        not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
        or not index.is_unique or not index.is_monotonic_increasing or index.empty
        or not availability.columns.is_unique or availability.shape[1] == 0
    ):
        return LeakageReport(("invalid_availability_schema",))
    issues = []
    for name in availability.columns:
        times = availability[name]
        if (not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC"
                or times.isna().any()):
            issues.append(f"unknown_availability: {name}")
        elif (times > index).any():
            issues.append(f"future_feature_availability: {name}")
    return LeakageReport(tuple(issues))


def audit_feature_causality(
    history: pd.DataFrame, transform: Callable[[pd.DataFrame], pd.DataFrame], *,
    blind_start: pd.Timestamp, cutoffs: tuple[int, ...] | None = None,
) -> LeakageReport:
    """Compare each historical prefix against the corresponding full-history output.

    The default checks every prefix; callers may specify audited cutoff positions
    for expensive generators. Record chosen cutoffs in experiment configuration.
    Transforms receive copies, so mutations cannot contaminate the source history.
    """
    validate_timeline(history.index, blind_start)
    points = cutoffs if cutoffs is not None else tuple(range(1, len(history)))
    if not points or any(isinstance(i, bool) or not isinstance(i, int)
                         or not 0 < i < len(history) for i in points):
        raise ValueError("Causality cutoffs must lie inside the research history.")
    full = transform(history.copy(deep=True))
    if (
        not isinstance(full, pd.DataFrame) or not full.index.equals(history.index)
        or not full.columns.is_unique or full.shape[1] == 0
        or any(dtype.kind not in "iuf" for dtype in full.dtypes)
    ):
        return LeakageReport(("invalid_feature_output_schema",))
    issues: list[str] = []
    for end in points:
        prefix = transform(history.iloc[:end].copy(deep=True))
        if (not isinstance(prefix, pd.DataFrame) or not prefix.index.equals(history.index[:end])
                or not prefix.columns.equals(full.columns)
                or any(dtype.kind not in "iuf" for dtype in prefix.dtypes)):
            issues.append(f"prefix_schema_changed: cutoff={end}")
            continue
        for name in full.columns:
            if not np.allclose(full[name].iloc[:end].to_numpy(dtype=float),
                               prefix[name].to_numpy(dtype=float),
                               atol=1e-12, rtol=1e-10, equal_nan=True):
                issues.append(f"future_sensitive_feature: {name}, cutoff={end}")
    return LeakageReport(tuple(issues))
