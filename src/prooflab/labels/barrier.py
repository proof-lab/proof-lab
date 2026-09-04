"""Barrier evaluation engine for predictive setup classification.

Evaluates forward barrier trajectories for Long and Short setups and resolves
same-bar target/stop ambiguity according to explicit policies.
"""

from __future__ import annotations

import pandas as pd

from prooflab.labels.config import AmbiguityPolicy, Direction, SetupConfig
from prooflab.labels.outcome import (
    BarrierOutcome,
    CanonicalLabel,
    LabelMatrix,
    RichLabelOutcome,
)


class BarrierEvaluator:
    """Evaluates triple-barrier outcomes across market price series."""

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        idx: int,
        config: SetupConfig,
        atr_value: float | None = None,
    ) -> RichLabelOutcome:
        """Evaluate the setup barrier trajectory starting at a single bar index.

        Args:
            df: Historical price DataFrame (must include timestamp, high, low, close).
            idx: Entry bar integer index.
            config: Setup configuration specifying targets, stops, horizon, policy.
            atr_value: Optional ATR value for ATR-scaled barrier distances.

        Returns:
            RichLabelOutcome capturing canonical label and complete trajectory details.
        """
        entry_row = df.iloc[idx]
        entry_time = pd.to_datetime(entry_row["timestamp"]).to_pydatetime()
        entry_price = float(entry_row[config.entry_price_col])

        target_price, stop_price = config.calculate_barriers(
            entry_price=entry_price,
            atr_value=atr_value,
        )

        future_slice = df.iloc[idx + 1 : idx + 1 + config.horizon_bars]

        if future_slice.empty:
            return RichLabelOutcome(
                entry_index=idx,
                entry_time=entry_time,
                entry_price=entry_price,
                target_price=target_price,
                stop_price=stop_price,
                canonical_label=CanonicalLabel.IGNORE,
                barrier_outcome=BarrierOutcome.TIMEOUT,
                exit_index=None,
                exit_time=None,
                exit_price=None,
                bars_held=0,
                return_at_exit=0.0,
                was_ambiguous=False,
                ambiguity_policy_used=config.ambiguity_policy.value,
            )

        # Iterate forward through the future horizon window
        for step, (future_idx, bar) in enumerate(future_slice.iterrows(), start=1):
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            bar_time = pd.to_datetime(bar["timestamp"]).to_pydatetime()
            bar_idx_int = int(future_idx) if isinstance(future_idx, int) else idx + step

            if config.direction == Direction.LONG:
                target_hit = bar_high >= target_price
                stop_hit = bar_low <= stop_price

                if target_hit and stop_hit:
                    # Same-bar ambiguity
                    if config.ambiguity_policy == AmbiguityPolicy.CONSERVATIVE:
                        # Adverse barrier (Stop) hit first
                        return_ret = (stop_price - entry_price) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.IGNORE,
                            barrier_outcome=BarrierOutcome.STOP_FIRST,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=stop_price,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )
                    elif config.ambiguity_policy == AmbiguityPolicy.OPTIMISTIC:
                        # Favorable barrier (Target) hit first
                        return_ret = (target_price - entry_price) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.BUY,
                            barrier_outcome=BarrierOutcome.TARGET_FIRST,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=target_price,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )
                    elif config.ambiguity_policy == AmbiguityPolicy.EXCLUDE:
                        return_ret = (bar_close - entry_price) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.IGNORE,
                            barrier_outcome=BarrierOutcome.EXCLUDED,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=bar_close,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )

                elif target_hit:
                    return_ret = (target_price - entry_price) / entry_price
                    return RichLabelOutcome(
                        entry_index=idx,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        canonical_label=CanonicalLabel.BUY,
                        barrier_outcome=BarrierOutcome.TARGET_FIRST,
                        exit_index=bar_idx_int,
                        exit_time=bar_time,
                        exit_price=target_price,
                        bars_held=step,
                        return_at_exit=return_ret,
                        was_ambiguous=False,
                        ambiguity_policy_used=config.ambiguity_policy.value,
                    )

                elif stop_hit:
                    return_ret = (stop_price - entry_price) / entry_price
                    return RichLabelOutcome(
                        entry_index=idx,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        canonical_label=CanonicalLabel.IGNORE,
                        barrier_outcome=BarrierOutcome.STOP_FIRST,
                        exit_index=bar_idx_int,
                        exit_time=bar_time,
                        exit_price=stop_price,
                        bars_held=step,
                        return_at_exit=return_ret,
                        was_ambiguous=False,
                        ambiguity_policy_used=config.ambiguity_policy.value,
                    )

            elif config.direction == Direction.SHORT:
                target_hit = bar_low <= target_price
                stop_hit = bar_high >= stop_price

                if target_hit and stop_hit:
                    # Same-bar ambiguity
                    if config.ambiguity_policy == AmbiguityPolicy.CONSERVATIVE:
                        # Adverse barrier (Stop) hit first
                        return_ret = (entry_price - stop_price) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.IGNORE,
                            barrier_outcome=BarrierOutcome.STOP_FIRST,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=stop_price,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )
                    elif config.ambiguity_policy == AmbiguityPolicy.OPTIMISTIC:
                        # Favorable barrier (Target) hit first
                        return_ret = (entry_price - target_price) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.SELL,
                            barrier_outcome=BarrierOutcome.TARGET_FIRST,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=target_price,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )
                    elif config.ambiguity_policy == AmbiguityPolicy.EXCLUDE:
                        return_ret = (entry_price - bar_close) / entry_price
                        return RichLabelOutcome(
                            entry_index=idx,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_price=stop_price,
                            canonical_label=CanonicalLabel.IGNORE,
                            barrier_outcome=BarrierOutcome.EXCLUDED,
                            exit_index=bar_idx_int,
                            exit_time=bar_time,
                            exit_price=bar_close,
                            bars_held=step,
                            return_at_exit=return_ret,
                            was_ambiguous=True,
                            ambiguity_policy_used=config.ambiguity_policy.value,
                        )

                elif target_hit:
                    return_ret = (entry_price - target_price) / entry_price
                    return RichLabelOutcome(
                        entry_index=idx,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        canonical_label=CanonicalLabel.SELL,
                        barrier_outcome=BarrierOutcome.TARGET_FIRST,
                        exit_index=bar_idx_int,
                        exit_time=bar_time,
                        exit_price=target_price,
                        bars_held=step,
                        return_at_exit=return_ret,
                        was_ambiguous=False,
                        ambiguity_policy_used=config.ambiguity_policy.value,
                    )

                elif stop_hit:
                    return_ret = (entry_price - stop_price) / entry_price
                    return RichLabelOutcome(
                        entry_index=idx,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        canonical_label=CanonicalLabel.IGNORE,
                        barrier_outcome=BarrierOutcome.STOP_FIRST,
                        exit_index=bar_idx_int,
                        exit_time=bar_time,
                        exit_price=stop_price,
                        bars_held=step,
                        return_at_exit=return_ret,
                        was_ambiguous=False,
                        ambiguity_policy_used=config.ambiguity_policy.value,
                    )

        # Horizon expired without hitting either barrier -> TIMEOUT
        last_bar = future_slice.iloc[-1]
        last_close = float(last_bar["close"])
        last_time = pd.to_datetime(last_bar["timestamp"]).to_pydatetime()
        last_idx = (
            int(future_slice.index[-1])
            if isinstance(future_slice.index[-1], int)
            else idx + len(future_slice)
        )

        if config.direction == Direction.LONG:
            ret = (last_close - entry_price) / entry_price
        else:
            ret = (entry_price - last_close) / entry_price

        return RichLabelOutcome(
            entry_index=idx,
            entry_time=entry_time,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            canonical_label=CanonicalLabel.IGNORE,
            barrier_outcome=BarrierOutcome.TIMEOUT,
            exit_index=last_idx,
            exit_time=last_time,
            exit_price=last_close,
            bars_held=len(future_slice),
            return_at_exit=ret,
            was_ambiguous=False,
            ambiguity_policy_used=config.ambiguity_policy.value,
        )

    def generate_labels(
        self,
        df: pd.DataFrame,
        config: SetupConfig,
        atr_series: pd.Series | None = None,
        dataset_id: str | None = None,
    ) -> LabelMatrix:
        """Generate predictive setup classification labels across a full DataFrame.

        Args:
            df: Historical price DataFrame (must be chronologically sorted).
            config: Setup configuration.
            atr_series: Optional ATR values matching df index.
            dataset_id: Optional dataset version identifier.

        Returns:
            LabelMatrix containing rich outcome records.
        """
        if df.empty:
            return LabelMatrix(outcomes=[], setup_config=config, dataset_id=dataset_id)

        outcomes: list[RichLabelOutcome] = []
        for idx in range(len(df)):
            atr_val = float(atr_series.iloc[idx]) if atr_series is not None else None
            outcome = self.evaluate_bar(
                df=df,
                idx=idx,
                config=config,
                atr_value=atr_val,
            )
            outcomes.append(outcome)

        return LabelMatrix(
            outcomes=outcomes,
            setup_config=config,
            dataset_id=dataset_id,
        )
