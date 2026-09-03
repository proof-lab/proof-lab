"""Proof Engine and robustness stress-testing framework for Proof Lab."""

from prooflab.proof.importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceEntry,
    FeatureImportanceResult,
)
from prooflab.proof.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
)
from prooflab.proof.regime import (
    RegimeAnalysisResult,
    RegimeAnalyzer,
    RegimeBucketMetrics,
    YearlyPerformance,
)
from prooflab.proof.report import (
    ProofEngine,
    ProofReport,
)
from prooflab.proof.scorecard import (
    EquityCurveData,
    ProofScorecard,
)
from prooflab.proof.sensitivity import (
    ParameterSensitivityAnalyzer,
    ParameterSensitivityConfig,
    ParameterSensitivityResult,
    SensitivityGridCell,
)
from prooflab.proof.status import (
    ProofStatus,
    ProofStatusEvaluation,
    ProofStatusEvaluator,
    ProofStatusThresholds,
    RuleEvaluationGate,
)
from prooflab.proof.stress import (
    ExecutionStressAnalyzer,
    ExecutionStressConfig,
    ExecutionStressResult,
    StressScenarioResult,
)
from prooflab.proof.warnings import (
    ResearchWarning,
    ResearchWarningCode,
    ResearchWarningDetector,
)

__all__ = [
    "EquityCurveData",
    "ExecutionStressAnalyzer",
    "ExecutionStressConfig",
    "ExecutionStressResult",
    "FeatureImportanceAnalyzer",
    "FeatureImportanceEntry",
    "FeatureImportanceResult",
    "MonteCarloConfig",
    "MonteCarloEngine",
    "MonteCarloResult",
    "ParameterSensitivityAnalyzer",
    "ParameterSensitivityConfig",
    "ParameterSensitivityResult",
    "ProofEngine",
    "ProofReport",
    "ProofScorecard",
    "ProofStatus",
    "ProofStatusEvaluation",
    "ProofStatusEvaluator",
    "ProofStatusThresholds",
    "RegimeAnalysisResult",
    "RegimeAnalyzer",
    "RegimeBucketMetrics",
    "ResearchWarning",
    "ResearchWarningCode",
    "ResearchWarningDetector",
    "RuleEvaluationGate",
    "SensitivityGridCell",
    "StressScenarioResult",
    "YearlyPerformance",
]
