"""Strategy configuration schema for human-readable strategy/strategy.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrategyPackageConfig(BaseModel):
    """Strategy configuration persisted inside .plb packages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    target_pips: float = Field(gt=0.0)
    stop_pips: float = Field(gt=0.0)
    horizon_bars: int = Field(gt=0)
    risk_per_trade_pct: float = Field(default=0.01, gt=0.0, le=0.10)
    min_confidence: float = Field(default=0.55, ge=0.50, le=1.0)
    feature_preset: str = Field(default="PRICE_ONLY")
    parameters: dict[str, Any] = Field(default_factory=dict)
    notes: str = Field(default="")

    def to_yaml(self) -> str:
        """Serialize configuration to human-readable YAML string."""
        return str(yaml.dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> StrategyPackageConfig:
        """Parse configuration safely from YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Invalid strategy YAML format: expected dictionary")
        return cls.model_validate(data)
