"""Backtesting and execution simulation engine for Proof Lab.

Provides signal filtering, order and position lifecycle tracking, execution cost
modelling (spread, commission, slippage, swap), portfolio accounting, and quantitative
performance analytics.
"""

from prooflab.backtest.costs import (
    CommissionModel,
    CommissionModelConfig,
    CommissionType,
    ExecutionCostConfig,
    ExecutionCostModel,
    SlippageMode,
    SlippageModel,
    SlippageModelConfig,
    SpreadMode,
    SpreadModel,
    SpreadModelConfig,
    SpreadScenario,
    SwapModel,
    SwapModelConfig,
)
from prooflab.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from prooflab.backtest.metrics import (
    BacktestMetricsReport,
    CostMetrics,
    ReturnsMetrics,
    RiskAdjustedMetrics,
    RiskMetrics,
    TradingMetrics,
    calculate_backtest_metrics,
)
from prooflab.backtest.orders import (
    ExitReason,
    OrderRecord,
    OrderSide,
    OrderStatus,
    Position,
)
from prooflab.backtest.portfolio import (
    BrokerLimitsConfig,
    EquitySnapshot,
    PortfolioAccountant,
    PositionSizer,
)
from prooflab.backtest.signals import (
    SignalEngine,
    SignalFilterConfig,
    TradeSignal,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetricsReport",
    "BacktestResult",
    "BrokerLimitsConfig",
    "CommissionModel",
    "CommissionModelConfig",
    "CommissionType",
    "CostMetrics",
    "EquitySnapshot",
    "ExecutionCostConfig",
    "ExecutionCostModel",
    "ExitReason",
    "OrderRecord",
    "OrderSide",
    "OrderStatus",
    "PortfolioAccountant",
    "Position",
    "PositionSizer",
    "ReturnsMetrics",
    "RiskAdjustedMetrics",
    "RiskMetrics",
    "SignalEngine",
    "SignalFilterConfig",
    "SlippageMode",
    "SlippageModel",
    "SlippageModelConfig",
    "SpreadMode",
    "SpreadModel",
    "SpreadModelConfig",
    "SpreadScenario",
    "SwapModel",
    "SwapModelConfig",
    "TradeSignal",
    "TradingMetrics",
    "calculate_backtest_metrics",
]
