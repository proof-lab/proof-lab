"""Property and invariance tests proving absence of look-ahead bias across all feature families."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset


@pytest.fixture
def baseline_dataset() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 60 historical bars with spread and tick volume
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(60)],
            "open": [100.0 + np.sin(i * 0.1) * 5.0 for i in range(60)],
            "high": [102.0 + np.sin(i * 0.1) * 5.0 for i in range(60)],
            "low": [98.0 + np.sin(i * 0.1) * 5.0 for i in range(60)],
            "close": [101.0 + np.sin(i * 0.1) * 5.0 for i in range(60)],
            "volume": [1000 + (i * 20) for i in range(60)],
            "spread": [1.5 + (i % 3) * 0.5 for i in range(60)],
            "tick_volume": [150 + (i * 5) for i in range(60)],
        }
    )


@pytest.mark.parametrize(
    "preset",
    [
        FeatureSetPreset.PRICE_ONLY,
        FeatureSetPreset.PRICE_VOLATILITY,
        FeatureSetPreset.PRICE_MOMENTUM,
        FeatureSetPreset.PRICE_VOLATILITY_MOMENTUM,
        FeatureSetPreset.ALL_STANDARD,
        FeatureSetPreset.ALL_STANDARD_MICROSTRUCTURE,
    ],
)
def test_future_data_invariance(baseline_dataset: pd.DataFrame, preset: FeatureSetPreset) -> None:
    """Property test: Appending future data MUST NOT alter historical feature calculations."""
    pipeline = FeaturePipeline(features=preset, include_raw_columns=False)

    # 1. Transform baseline historical dataset (T=60)
    baseline_features = pipeline.transform(baseline_dataset, drop_warmup=False)

    # 2. Construct future dataset with extreme market spikes / drops (T=60..90)
    base_future = baseline_dataset["timestamp"].iloc[-1] + timedelta(minutes=1)
    future_bars = pd.DataFrame(
        {
            "timestamp": [base_future + timedelta(minutes=i) for i in range(30)],
            "open": [250.0 + (i * 10.0) for i in range(30)],     # Extreme upward jump
            "high": [300.0 + (i * 10.0) for i in range(30)],
            "low": [200.0 + (i * 10.0) for i in range(30)],
            "close": [280.0 + (i * 10.0) for i in range(30)],
            "volume": [50000 for _ in range(30)],
            "spread": [10.0 for _ in range(30)],
            "tick_volume": [5000 for _ in range(30)],
        }
    )
    extended_dataset = pd.concat([baseline_dataset, future_bars], ignore_index=True)

    # 3. Transform extended dataset (T=90)
    extended_features = pipeline.transform(extended_dataset, drop_warmup=False)

    # 4. Slice extended features back to the baseline period (0..60)
    recalculated_historical_slice = extended_features.iloc[: len(baseline_dataset)]

    # 5. Assert byte-for-byte numerical equality across every feature column
    for col in baseline_features.columns:
        base_col = baseline_features[col].to_numpy()
        ext_col = recalculated_historical_slice[col].to_numpy()

        # Compare non-NaN elements
        valid_mask = ~np.isnan(base_col)
        assert np.array_equal(valid_mask, ~np.isnan(ext_col)), f"NaN mismatch in column {col}"

        np.testing.assert_allclose(
            base_col[valid_mask],
            ext_col[valid_mask],
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future leakage detected in feature '{col}'!",
        )


def test_live_inference_single_bar_equivalence(baseline_dataset: pd.DataFrame) -> None:
    """Test that calculating features in streaming/live mode matches batch calculation."""
    pipeline = FeaturePipeline(features=FeatureSetPreset.ALL_STANDARD, include_raw_columns=False)
    batch_features = pipeline.transform(baseline_dataset, drop_warmup=False)

    # Simulate streaming live inference for the final 5 bars
    for idx in range(55, 60):
        rolling_window = baseline_dataset.iloc[: idx + 1]
        live_features = pipeline.transform(rolling_window, drop_warmup=False)

        # The last row of live_features must match batch_features at idx
        live_last_row = live_features.iloc[-1].to_numpy()
        batch_row = batch_features.iloc[idx].to_numpy()

        valid = ~np.isnan(batch_row)
        np.testing.assert_allclose(
            live_last_row[valid],
            batch_row[valid],
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Live streaming inference mismatch at index {idx}",
        )
