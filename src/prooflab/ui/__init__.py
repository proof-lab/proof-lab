"""Proof Lab Graphical User Interface presentation and control layer."""

from prooflab.ui.router import ui_router
from prooflab.ui.views import (
    AutoPilotMode,
    CoPilotOrderRequest,
    DatasetHealthSummaryView,
    DataStudioExtractRequest,
    FeatureGroupSelectionView,
    LiveDashboardView,
    LiveDeploymentConfirmation,
    MetricScorecardView,
    ModelSelectionView,
    ModelVoteView,
    ProofEngineViewResponse,
    SafeguardsConfigView,
    SetupDefinitionView,
    TrainingProgressStage,
    TrainingProgressView,
    WarningItemView,
)

__all__ = [
    "AutoPilotMode",
    "CoPilotOrderRequest",
    "DataStudioExtractRequest",
    "DatasetHealthSummaryView",
    "FeatureGroupSelectionView",
    "LiveDashboardView",
    "LiveDeploymentConfirmation",
    "MetricScorecardView",
    "ModelSelectionView",
    "ModelVoteView",
    "ProofEngineViewResponse",
    "SafeguardsConfigView",
    "SetupDefinitionView",
    "TrainingProgressStage",
    "TrainingProgressView",
    "WarningItemView",
    "ui_router",
]
