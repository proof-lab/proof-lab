"""Unit tests for prooflab.proof.status (Explicit Deterministic Proof Status Rules)."""

import pytest

from prooflab.proof.monte_carlo import MonteCarloResult
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.sensitivity import ParameterSensitivityResult
from prooflab.proof.status import (
    ProofStatus,
    ProofStatusEvaluation,
    ProofStatusEvaluator,
)
from prooflab.proof.stress import ExecutionStressResult


@pytest.fixture
def mock_scorecard() -> ProofScorecard:
    return ProofScorecard(
        initial_capital=100000.0,
        final_net_equity=115000.0,
        total_net_return_pct=15.0,
        annualized_return_pct=15.0,
        cagr_pct=15.0,
        profit_factor=1.85,
        sharpe_ratio=1.45,
        sortino_ratio=2.10,
        calmar_ratio=2.50,
        max_drawdown_net_pct=6.0,
        max_drawdown_net_dollars=6000.0,
        expectancy_dollars=125.0,
        win_rate_pct=58.0,
        loss_rate_pct=42.0,
        trade_count=120,
        winning_trades=70,
        losing_trades=50,
        total_costs_paid=450.0,
        total_spread_paid=250.0,
        total_commission_paid=120.0,
        total_slippage_paid=50.0,
        total_swap_paid=30.0,
    )


@pytest.fixture
def mock_stress_survives() -> ExecutionStressResult:
    return ExecutionStressResult(
        normal_return_pct=15.0,
        conservative_return_pct=10.0,
        stress_return_pct=4.0,
        extreme_return_pct=1.0,
        scenarios=[],
        survives_conservative=True,
        survives_stress=True,
        survives_extreme=True,
        depends_on_low_spread=False,
    )


@pytest.fixture
def mock_sensitivity_stable() -> ParameterSensitivityResult:
    return ParameterSensitivityResult(
        base_stop_pips=25.0,
        base_target_pips=50.0,
        base_net_return_pct=15.0,
        grid_cells=[],
        profitable_cells_pct=100.0,
        avg_perturbed_return_pct=14.0,
        worst_perturbed_return_pct=10.0,
        return_std_pct=1.5,
        has_cliff_effect=False,
        is_fragile=False,
    )


@pytest.fixture
def mock_mc_safe() -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        trade_count=120,
        resampling_mode="reshuffle",
        median_return_pct=15.0,
        percentile_5_return_pct=8.0,
        percentile_95_return_pct=22.0,
        median_max_drawdown_pct=6.5,
        percentile_95_max_drawdown_pct=10.5,
        probability_of_loss_pct=0.0,
        probability_of_ruin_pct=0.0,
    )


def test_proof_status_robust(
    mock_scorecard: ProofScorecard,
    mock_stress_survives: ExecutionStressResult,
    mock_sensitivity_stable: ParameterSensitivityResult,
    mock_mc_safe: MonteCarloResult,
) -> None:
    evaluator = ProofStatusEvaluator()
    res = evaluator.evaluate(
        scorecard=mock_scorecard,
        sensitivity_result=mock_sensitivity_stable,
        stress_result=mock_stress_survives,
        monte_carlo_result=mock_mc_safe,
        has_leakage=False,
        blind_test_completed=True,
    )

    assert isinstance(res, ProofStatusEvaluation)
    assert res.status == ProofStatus.ROBUST
    assert res.is_validated is True
    assert all(g.passed for g in res.rule_gates)


def test_proof_status_not_proven_on_leakage(mock_scorecard: ProofScorecard) -> None:
    evaluator = ProofStatusEvaluator()
    res = evaluator.evaluate(
        scorecard=mock_scorecard,
        has_leakage=True,
    )

    assert res.status == ProofStatus.NOT_PROVEN
    assert res.is_validated is False
    assert "Data leakage" in res.status_reason


def test_proof_status_weak_on_fragile_sensitivity(
    mock_scorecard: ProofScorecard,
    mock_stress_survives: ExecutionStressResult,
    mock_mc_safe: MonteCarloResult,
) -> None:
    fragile_sens = ParameterSensitivityResult(
        base_stop_pips=25.0,
        base_target_pips=50.0,
        base_net_return_pct=15.0,
        grid_cells=[],
        profitable_cells_pct=40.0,
        avg_perturbed_return_pct=-2.0,
        worst_perturbed_return_pct=-8.0,
        return_std_pct=5.5,
        has_cliff_effect=True,
        is_fragile=True,
    )

    evaluator = ProofStatusEvaluator()
    res = evaluator.evaluate(
        scorecard=mock_scorecard,
        sensitivity_result=fragile_sens,
        stress_result=mock_stress_survives,
        monte_carlo_result=mock_mc_safe,
    )

    assert res.status == ProofStatus.WEAK
    assert res.is_validated is False
    assert "parameter sensitivity" in res.status_reason.lower()
