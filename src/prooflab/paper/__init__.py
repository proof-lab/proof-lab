"""Paper trading simulation and live forward testing engine for Proof Lab."""

from prooflab.paper.consumer import (
    ConsumerConfig,
    DataQualityIssue,
    LiveBar,
    LiveTick,
    MarketDataConsumer,
)
from prooflab.paper.engine import (
    PaperBarEventResult,
    PaperTradingEngine,
)
from prooflab.paper.execution import (
    PaperExecutionConfig,
    PaperExecutionEngine,
    PaperPosition,
)
from prooflab.paper.features import (
    LiveFeatureCalculator,
    LiveFeatureResult,
)
from prooflab.paper.inference import (
    InferencePrediction,
    LiveInferenceEngine,
)
from prooflab.paper.ledger import (
    PaperTradeLedger,
)
from prooflab.paper.lifecycle import (
    ALLOWED_TRANSITIONS,
    LifecycleTransitionEvent,
    StrategyLifecycleManager,
    StrategyLifecycleState,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ConsumerConfig",
    "DataQualityIssue",
    "InferencePrediction",
    "LifecycleTransitionEvent",
    "LiveBar",
    "LiveFeatureCalculator",
    "LiveFeatureResult",
    "LiveInferenceEngine",
    "LiveTick",
    "MarketDataConsumer",
    "PaperBarEventResult",
    "PaperExecutionConfig",
    "PaperExecutionEngine",
    "PaperPosition",
    "PaperTradeLedger",
    "PaperTradingEngine",
    "StrategyLifecycleManager",
    "StrategyLifecycleState",
]
