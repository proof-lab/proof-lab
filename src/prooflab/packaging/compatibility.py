"""Environment and parameter compatibility validation for strategy packages."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset
from prooflab.packaging.manifest import PackageManifest


class EnvironmentContext(BaseModel):
    """Runtime environment specifications for compatibility assessment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_version: str = Field(default="0.1.0")
    supported_symbols: list[str] = Field(
        default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    )
    supported_timeframes: list[str] = Field(
        default_factory=lambda: ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    )
    available_features: list[str] = Field(default_factory=list)


class CompatibilityReport(BaseModel):
    """Result of strategy package compatibility check against an execution environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_compatible: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """Render a readable summary of compatibility assessment."""
        if self.is_compatible:
            status = "COMPATIBLE"
            warn_str = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return (
                f"[{status}] Strategy package is fully compatible with "
                f"execution environment{warn_str}."
            )
        issues_str = "\n  - ".join(self.issues)
        return f"[INCOMPATIBLE] Strategy package cannot be loaded:\n  - {issues_str}"


def _parse_version(v_str: str) -> tuple[int, ...]:
    """Extract numeric components from a semantic version string."""
    nums = re.findall(r"\d+", v_str)
    return tuple(int(n) for n in nums) if nums else (0,)


class CompatibilityChecker:
    """Verifies that a strategy package matches target runtime requirements."""

    def __init__(self, default_context: EnvironmentContext | None = None) -> None:
        if default_context is None:
            # Populate with all standard pipeline feature names
            pipeline = FeaturePipeline(
                features=FeatureSetPreset.ALL_STANDARD,
                include_raw_columns=False,
            )
            self.context = EnvironmentContext(
                app_version="0.1.0",
                available_features=pipeline.get_feature_names(),
            )
        else:
            self.context = default_context

    def check_compatibility(
        self,
        manifest: PackageManifest,
        target_environment: EnvironmentContext | None = None,
        target_symbol: str | None = None,
        target_timeframe: str | None = None,
    ) -> CompatibilityReport:
        """Evaluate strategy manifest requirements against target runtime environment."""
        env = target_environment or self.context
        issues: list[str] = []
        warnings: list[str] = []

        compat = manifest.compatibility

        # 1. Application Version Check
        app_ver = _parse_version(env.app_version)
        min_ver = _parse_version(compat.min_app_version)
        if app_ver < min_ver:
            issues.append(
                f"App version {env.app_version} is lower than "
                f"required minimum {compat.min_app_version}"
            )

        # 2. Symbol Compatibility Check
        if target_symbol and target_symbol.upper() != compat.symbol.upper():
            issues.append(
                f"Target symbol {target_symbol} does not match "
                f"strategy symbol {compat.symbol}"
            )
        elif compat.symbol.upper() not in [s.upper() for s in env.supported_symbols]:
            warnings.append(
                f"Strategy symbol {compat.symbol} is not in "
                f"standard supported symbols: {env.supported_symbols}"
            )

        # 3. Timeframe Compatibility Check
        if target_timeframe and target_timeframe.upper() != compat.timeframe.upper():
            issues.append(
                f"Target timeframe {target_timeframe} does not match "
                f"strategy timeframe {compat.timeframe}"
            )
        elif compat.timeframe.upper() not in [t.upper() for t in env.supported_timeframes]:
            warnings.append(
                f"Strategy timeframe {compat.timeframe} is not in "
                f"standard supported timeframes: {env.supported_timeframes}"
            )

        # 4. Feature Requirements Check
        if env.available_features:
            missing_feats = [
                f for f in compat.feature_names if f not in env.available_features
            ]
            if missing_feats:
                issues.append(
                    f"Required strategy features are unavailable "
                    f"in runtime environment: {missing_feats}"
                )

        # 5. Barrier Parameter Validation
        if compat.target_pips <= 0:
            issues.append(f"Invalid target_pips: {compat.target_pips} (must be > 0)")
        if compat.stop_pips <= 0:
            issues.append(f"Invalid stop_pips: {compat.stop_pips} (must be > 0)")
        if compat.horizon_bars <= 0:
            issues.append(f"Invalid horizon_bars: {compat.horizon_bars} (must be > 0)")

        is_compat = len(issues) == 0
        return CompatibilityReport(
            is_compatible=is_compat,
            issues=issues,
            warnings=warnings,
            details={
                "strategy_id": manifest.strategy_id,
                "strategy_symbol": compat.symbol,
                "strategy_timeframe": compat.timeframe,
                "feature_count": len(compat.feature_names),
            },
        )
