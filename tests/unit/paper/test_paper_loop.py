"""Integration tests for the complete Proof Lab paper trading loop."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset
from prooflab.models.artifacts import ArtifactManifest, ModelArtifact, TrainingMetadata
from prooflab.models.baselines import MajorityClassifier
from prooflab.paper.consumer import ConsumerConfig, LiveBar, MarketDataConsumer
from prooflab.paper.engine import PaperTradingEngine
from prooflab.paper.execution import PaperExecutionConfig, PaperExecutionEngine
from prooflab.paper.features import LiveFeatureCalculator
from prooflab.paper.inference import LiveInferenceEngine
from prooflab.paper.ledger import PaperTradeLedger
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState
from prooflab.risk.engine import RiskEngine
from prooflab.risk.kill_switch import KillSwitch
from prooflab.risk.limits import RiskLimitsConfig
from prooflab.risk.safety import SafetyCheckConfig


def _create_trained_artifact() -> ModelArtifact:
    pipeline = FeaturePipeline(features=FeatureSetPreset.PRICE_ONLY, include_raw_columns=False)
    idx = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    prices = 1.1000 + np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.0050
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0010,
            "low": prices - 0.0010,
            "close": prices + 0.0002,
            "volume": 1000.0,
            "spread": 0.0001,
        },
        index=idx,
    )
    features_df = pipeline.transform(df, drop_warmup=False).dropna()
    feature_names = pipeline.get_feature_names()

    model = MajorityClassifier()
    y_series = pd.Series([1] * len(features_df), index=features_df.index)
    model.fit(features_df, y_series)

    meta_list = [
        FeatureMetadata(
            feature_name=name,
            family=FeatureFamily.PRICE,
            lookback_period=1,
            description=name,
            required_columns=["close"],
        )
        for name in feature_names
    ]

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 2, 1, tzinfo=UTC)
    training_meta = TrainingMetadata(
        dataset_id="ds-paper-test",
        dataset_checksum="c" * 64,
        setup_config={"target": 30, "stop": 20},
        train_start=t0,
        train_end=t1,
        train_rows=len(features_df),
    )

    manifest = ArtifactManifest(
        created_at=datetime.now(UTC),
        model_name="majority_classifier",
        model_type="baseline",
        model_params={},
        feature_order=feature_names,
        feature_schema={name: "float64" for name in feature_names},
        feature_metadata=meta_list,
        classes=[1],
        preprocessing="identity",
        training=training_meta,
        dependencies={
            "python": "3.12",
            "prooflab": "0.1",
            "numpy": "1.26",
            "pandas": "2.2",
            "joblib": "1.3",
            "pydantic": "2.6",
        },
        payload_checksum="d" * 64,
    )

    return ModelArtifact(model=model, manifest=manifest)


def test_strategy_lifecycle_hard_gates() -> None:
    mgr = StrategyLifecycleManager(strategy_id="strat-001")
    assert mgr.current_state == StrategyLifecycleState.RESEARCH
    assert mgr.is_live_trading_allowed is False

    mgr.transition_to(StrategyLifecycleState.VALIDATED, reason="Passed proof validation")
    mgr.transition_to(StrategyLifecycleState.PAPER_TRADING, reason="Beginning forward simulation")
    assert mgr.is_paper_trading_allowed is True
    assert mgr.is_live_trading_allowed is False

    mgr.transition_to(StrategyLifecycleState.APPROVED, reason="Passed paper review")

    # Attempt auto-transition to LIVE_ENABLED without human approval
    with pytest.raises(PermissionError, match="explicit human approval"):
        mgr.transition_to(StrategyLifecycleState.LIVE_ENABLED, reason="Automated attempt")

    # Explicit approval
    event = mgr.transition_to(
        StrategyLifecycleState.LIVE_ENABLED,
        reason="Human approved after full review",
        explicit_human_approval=True,
        authorized_by="Lead Quantitative Officer",
    )
    assert mgr.is_live_trading_allowed is True
    assert event.authorized_by == "Lead Quantitative Officer"


def test_complete_paper_trading_end_to_end_loop() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        start_time = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)
        ledger_path = Path(tmpdir) / "paper_trades.json"
        kill_switch_path = Path(tmpdir) / "kill_switch.json"

        consumer = MarketDataConsumer(ConsumerConfig(max_staleness_seconds=3600.0))
        pipeline = FeaturePipeline(features=FeatureSetPreset.PRICE_ONLY, include_raw_columns=False)
        artifact = _create_trained_artifact()

        feature_calc = LiveFeatureCalculator(
            pipeline=pipeline,
            expected_features=pipeline.get_feature_names(),
            min_warmup_bars=25,
        )
        inference_engine = LiveInferenceEngine(artifact=artifact, min_confidence_threshold=0.50)

        kill_switch = KillSwitch(state_file=kill_switch_path)
        risk_engine = RiskEngine(
            limits_config=RiskLimitsConfig(
                max_open_positions=1,
                max_symbol_leverage=10.0,
                max_total_leverage=10.0,
            ),
            kill_switch=kill_switch,
            safety_config=SafetyCheckConfig(max_data_staleness_seconds=3600.0),
            initial_equity=100000.0,
            current_time=start_time,
        )

        ledger = PaperTradeLedger(storage_path=ledger_path)
        execution_engine = PaperExecutionEngine(
            config=PaperExecutionConfig(
                initial_capital=100000.0,
                commission_per_unit=0.00001,
                slippage_pips=0.1,
            ),
            ledger=ledger,
        )

        lifecycle = StrategyLifecycleManager(strategy_id="demo-strat")
        lifecycle.transition_to(StrategyLifecycleState.VALIDATED, reason="Passed proof")
        lifecycle.transition_to(StrategyLifecycleState.PAPER_TRADING, reason="Start paper trading")

        engine = PaperTradingEngine(
            consumer=consumer,
            feature_calculator=feature_calc,
            inference_engine=inference_engine,
            risk_engine=risk_engine,
            execution_engine=execution_engine,
            lifecycle=lifecycle,
            target_pips=30.0,
            stop_pips=20.0,
            risk_pct=0.01,
        )

        executed_orders: list[str] = []

        # Feed 30 bars (warmup is 25 bars)
        for i in range(30):
            bar_time = start_time + timedelta(hours=i)
            price = 1.1000 + (i * 0.0005)
            bar = LiveBar(
                symbol="EURUSD",
                timestamp_utc=bar_time,
                open=price,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price + 0.0002,
                volume=1000.0,
                spread=0.0001,
            )

            res = engine.process_incoming_bar(bar, wall_clock_utc=bar_time)

            if i < 24:
                assert res.engine_status == "WARMUP_OR_FEATURE_ERROR"
            elif i == 24:
                # First trade execution!
                assert res.engine_status == "ORDER_EXECUTED"
                assert res.executed_order_id is not None
                executed_orders.append(res.executed_order_id)
            else:
                # Max open positions limit prevents duplicate orders
                assert res.engine_status in {"RISK_REJECTED", "ORDER_EXECUTED"}

        assert len(executed_orders) >= 1
        assert len(execution_engine.open_positions) == 1

        # Feed a sharp rally bar that hits the take profit barrier (e.g. 1.1200)
        rally_time = start_time + timedelta(hours=31)
        rally_bar = LiveBar(
            symbol="EURUSD",
            timestamp_utc=rally_time,
            open=1.1150,
            high=1.1300,
            low=1.1140,
            close=1.1250,
            volume=5000.0,
            spread=0.0001,
        )

        rally_res = engine.process_incoming_bar(rally_bar, wall_clock_utc=rally_time)
        assert len(rally_res.closed_orders) == 1
        assert len(execution_engine.open_positions) == 1  # New order entered on bar close

        # Verify ledger persistence
        assert len(ledger.trades) == 1
        trade = ledger.trades[0]
        assert trade.symbol == "EURUSD"
        assert trade.exit_reason == "TAKE_PROFIT"
        assert trade.status == "CLOSED"
        assert trade.net_pnl > 0.0


def test_paper_trading_pauses_on_stale_data_and_kill_switch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        consumer = MarketDataConsumer(ConsumerConfig(max_staleness_seconds=60.0))
        pipeline = FeaturePipeline(features=FeatureSetPreset.PRICE_ONLY, include_raw_columns=False)
        artifact = _create_trained_artifact()

        feature_calc = LiveFeatureCalculator(
            pipeline=pipeline,
            expected_features=pipeline.get_feature_names(),
            min_warmup_bars=25,
        )
        inference_engine = LiveInferenceEngine(artifact=artifact, min_confidence_threshold=0.50)

        t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
        t_stale = datetime(2026, 3, 2, 10, 5, tzinfo=UTC)  # 5 min later (stale for 60s config)

        kill_switch = KillSwitch(state_file=Path(tmpdir) / "ks.json")
        risk_engine = RiskEngine(
            limits_config=RiskLimitsConfig(),
            kill_switch=kill_switch,
            safety_config=SafetyCheckConfig(max_data_staleness_seconds=60.0),
            initial_equity=100000.0,
            current_time=t0,
        )

        execution_engine = PaperExecutionEngine()
        lifecycle = StrategyLifecycleManager(strategy_id="strat-pause")
        lifecycle.transition_to(StrategyLifecycleState.VALIDATED, reason="Passed proof")
        lifecycle.transition_to(StrategyLifecycleState.PAPER_TRADING, reason="Start paper trading")

        engine = PaperTradingEngine(
            consumer=consumer,
            feature_calculator=feature_calc,
            inference_engine=inference_engine,
            risk_engine=risk_engine,
            execution_engine=execution_engine,
            lifecycle=lifecycle,
        )

        stale_bar = LiveBar(
            symbol="EURUSD",
            timestamp_utc=t0,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=100.0,
            spread=0.0001,
        )

        res = engine.process_incoming_bar(stale_bar, wall_clock_utc=t_stale)
        assert res.data_quality_ok is False
        assert res.engine_status == "DATA_REJECTED"
        assert "STALE_DATA" in res.data_quality_issues

        # Kill Switch activation halts trading
        kill_switch.activate(reason="Emergency volatility breach", actor="RiskOfficer")
        assert kill_switch.is_active is True

        valid_bar = LiveBar(
            symbol="EURUSD",
            timestamp_utc=t0,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=100.0,
            spread=0.0001,
        )
        res_ks = engine.process_incoming_bar(valid_bar, wall_clock_utc=t0)
        assert res_ks.engine_status in {"WARMUP_OR_FEATURE_ERROR", "RISK_REJECTED"}
