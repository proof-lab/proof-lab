"""Binary action-event calibration metrics with explicit uniform-bin semantics."""

from typing import Any

import numpy as np


def probability_quality(
    probabilities: np.ndarray, outcomes: np.ndarray, *, n_bins: int = 10,
) -> dict[str, Any]:
    """Measure supplied predictions; callers must supply held-out observations.

    Brier score is mean (P(action) - action_occurred)**2, on the binary [0,1]
    scale. ECE uses the same action event, not top-class accuracy. Bins are
    [lower, upper), with 1 included in the final bin. Empty bins remain explicit.
    Log loss clips only for logarithms, using float64 machine epsilon.
    """
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    if (
        p.ndim != 1 or not len(p) or p.shape != y.shape or p.dtype.kind not in "iuf"
        or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
        or y.dtype.kind not in "biuf" or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Metrics require aligned finite binary outcomes and probabilities.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or not 1 <= n_bins <= 1000:
        raise ValueError("n_bins must be an integer between 1 and 1000.")
    p, y = p.astype(float), y.astype(float)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    edges = np.linspace(0, 1, n_bins + 1)
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    curve = []
    ece = 0.0
    for index in range(n_bins):
        mask = indices == index
        count = int(mask.sum())
        mean = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if mean is not None and observed is not None:
            ece += count / len(p) * abs(mean - observed)
        curve.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                      "count": count, "mean_probability": mean, "observed_frequency": observed})
    return {
        "rows": len(p), "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "expected_calibration_error": ece, "calibration_curve": curve,
        "n_bins": n_bins, "binning": "uniform", "event": "configured_action",
    }
