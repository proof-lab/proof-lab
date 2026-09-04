"""Unit tests for package manifest, compatibility declarations, and strategy config schemas."""


import pytest

from prooflab.packaging.manifest import CompatibilityDeclaration, PackageManifest
from prooflab.packaging.strategy_config import StrategyPackageConfig


def test_compatibility_declaration_validation() -> None:
    compat = CompatibilityDeclaration(
        symbol="EURUSD",
        timeframe="H1",
        feature_names=["ret_1", "vol_10"],
        feature_parameters={"fast": 12, "slow": 26},
        min_app_version="0.1.0",
        target_pips=30.0,
        stop_pips=20.0,
        horizon_bars=5,
    )
    assert compat.symbol == "EURUSD"
    assert compat.target_pips == 30.0
    assert compat.stop_pips == 20.0

    # Invalid values
    with pytest.raises(ValueError):
        CompatibilityDeclaration(
            symbol="EURUSD",
            timeframe="H1",
            feature_names=[],  # empty
            target_pips=30.0,
            stop_pips=20.0,
            horizon_bars=5,
        )


def test_package_manifest_json_round_trip() -> None:
    compat = CompatibilityDeclaration(
        symbol="EURUSD",
        timeframe="H1",
        feature_names=["ret_1"],
        target_pips=25.0,
        stop_pips=15.0,
        horizon_bars=10,
    )
    manifest = PackageManifest(
        strategy_id="strat-alpha-1",
        symbol="EURUSD",
        timeframe="H1",
        compatibility=compat,
        description="Alpha strategy for EURUSD H1",
        author="ProofLab Quant",
        dataset_metadata={"dataset_id": "eurusd-h1-v1", "dataset_checksum": "a" * 64},
    )

    json_str = manifest.to_json()
    loaded = PackageManifest.from_json(json_str)

    assert loaded.strategy_id == "strat-alpha-1"
    assert loaded.symbol == "EURUSD"
    assert loaded.compatibility.target_pips == 25.0
    assert loaded.dataset_metadata["dataset_id"] == "eurusd-h1-v1"


def test_strategy_package_config_yaml_round_trip() -> None:
    cfg = StrategyPackageConfig(
        strategy_id="strat-demo",
        symbol="GBPUSD",
        timeframe="M15",
        target_pips=20.0,
        stop_pips=10.0,
        horizon_bars=8,
        risk_per_trade_pct=0.015,
        min_confidence=0.60,
        feature_preset="PRICE_VOLATILITY",
        parameters={"rsi_period": 14},
        notes="Production breakout strategy",
    )

    yaml_str = cfg.to_yaml()
    loaded = StrategyPackageConfig.from_yaml(yaml_str)

    assert loaded.strategy_id == "strat-demo"
    assert loaded.symbol == "GBPUSD"
    assert loaded.risk_per_trade_pct == 0.015
    assert loaded.parameters["rsi_period"] == 14
