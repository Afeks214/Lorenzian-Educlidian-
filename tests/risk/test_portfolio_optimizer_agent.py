"""
Comprehensive Test Suite for Portfolio Optimizer Agent

This module implements comprehensive tests for the Portfolio Optimizer Agent (π₄)
including unit tests, integration tests, performance tests, and edge case validation.

Test Coverage:
- Portfolio optimization algorithms (ERC, HRP, Multi-objective)
- Correlation management and regime detection
- Performance attribution and risk decomposition
- Dynamic rebalancing with <10ms response time validation
- Error handling and edge cases
- Event system integration
- Performance benchmarking
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time

from src.risk.agents.portfolio_optimizer_agent import (
    PortfolioOptimizerAgent, StrategyPerformance, PortfolioState, OptimizationResult
)
from src.risk.agents.portfolio_correlation_manager import (
    PortfolioCorrelationManager, CorrelationRiskMetrics, DiversificationMetrics
)
from src.risk.agents.multi_objective_optimizer import (
    MultiObjectiveOptimizer, OptimizationObjectives, OptimizationConstraints,
    OptimizationMethod, ParetoSolution
)
from src.risk.agents.risk_parity_engine import (
    RiskParityEngine, RiskParityMethod, RiskBudget, RiskParityResult
)
from src.risk.agents.performance_attribution import (
    PerformanceAttributionEngine, StrategyPerformanceMetrics, AttributionMethod
)
from src.risk.agents.dynamic_rebalancing_engine import (
    DynamicRebalancingEngine, RebalanceConfig, RebalanceTrigger, RebalanceUrgency
)
from src.risk.agents.base_risk_agent import RiskState
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.risk.core.var_calculator import VaRCalculator
from src.core.events import EventBus, EventType


class TestPortfolioOptimizerAgent:
    """Test suite for Portfolio Optimizer Agent"""
    
    @pytest.fixture
    def event_bus(self):
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def correlation_tracker(self):
        tracker = Mock(spec=CorrelationTracker)
        tracker.get_correlation_matrix.return_value = np.eye(5)
        tracker.current_regime = CorrelationRegime.NORMAL
        return tracker
    
    @pytest.fixture
    def var_calculator(self):
        return Mock(spec=VaRCalculator)
    
    @pytest.fixture
    def optimizer_config(self):
        return {
            'n_strategies': 5,
            'target_volatility': 0.15,
            'rebalance_threshold': 0.05,
            'objective_weights': {
                'sharpe': 0.4,
                'diversification': 0.3,
                'drawdown': 0.2,
                'tail_risk': 0.1
            }
        }
    
    @pytest.fixture
    def portfolio_optimizer(self, optimizer_config, correlation_tracker, var_calculator, event_bus):
        return PortfolioOptimizerAgent(
            config=optimizer_config,
            correlation_tracker=correlation_tracker,
            var_calculator=var_calculator,
            event_bus=event_bus
        )
    
    def test_initialization(self, portfolio_optimizer):
        """Test proper initialization of portfolio optimizer"""
        assert portfolio_optimizer.n_strategies == 5
        assert len(portfolio_optimizer.current_weights) == 5
        assert np.allclose(np.sum(portfolio_optimizer.current_weights), 1.0)
        assert portfolio_optimizer.target_volatility == 0.15
        assert portfolio_optimizer.rebalance_threshold == 0.05
    
    def test_risk_state_processing(self, portfolio_optimizer):
        """Test processing of risk state vector"""
        # Create risk state
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.3,
            correlation_risk=0.4,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.3,
            time_of_day_risk=0.5,
            market_stress_level=0.2,
            liquidity_conditions=0.8
        )
        
        # Test risk action calculation
        action, confidence = portfolio_optimizer.calculate_risk_action(risk_state)
        
        assert isinstance(action, np.ndarray)
        assert len(action) == 5
        assert np.allclose(np.sum(action), 1.0, atol=1e-6)
        assert 0.0 <= confidence <= 1.0
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)
    
    def test_constraint_validation(self, portfolio_optimizer):
        """Test risk constraint validation"""
        # Valid risk state
        valid_risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=3,
            volatility_regime=0.2,
            correlation_risk=0.3,
            var_estimate_5pct=0.015,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.2,
            time_of_day_risk=0.3,
            market_stress_level=0.1,
            liquidity_conditions=0.9
        )
        
        assert portfolio_optimizer.validate_risk_constraints(valid_risk_state) == True
        
        # Invalid risk state (high volatility)
        invalid_risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.95,  # Very high volatility
            correlation_risk=0.4,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.3,
            time_of_day_risk=0.5,
            market_stress_level=0.2,
            liquidity_conditions=0.8
        )
        
        # Should return False due to high volatility regime
        assert portfolio_optimizer.validate_risk_constraints(invalid_risk_state) == False
    
    def test_strategy_performance_update(self, portfolio_optimizer):
        """Test strategy performance updating"""
        strategy_performance = StrategyPerformance(
            strategy_id='Strategy_0',
            returns=[0.01, 0.02, -0.01, 0.015],
            volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            var_95=0.02,
            correlation_with_portfolio=0.6,
            current_weight=0.2
        )
        
        portfolio_optimizer.update_strategy_performance('Strategy_0', strategy_performance)
        
        assert 'Strategy_0' in portfolio_optimizer.strategy_performances
        stored_perf = portfolio_optimizer.strategy_performances['Strategy_0']
        assert stored_perf.sharpe_ratio == 1.2
        assert stored_perf.volatility == 0.15
    
    def test_optimization_performance(self, portfolio_optimizer):
        """Test optimization performance requirements"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.3,
            correlation_risk=0.4,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.3,
            time_of_day_risk=0.5,
            market_stress_level=0.2,
            liquidity_conditions=0.8
        )
        
        # Measure optimization time
        start_time = time.perf_counter()
        action, confidence = portfolio_optimizer.calculate_risk_action(risk_state)
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        # Should complete within 10ms target
        assert optimization_time < 10.0, f"Optimization took {optimization_time:.2f}ms, exceeds 10ms target"
    
    def test_emergency_stop(self, portfolio_optimizer):
        """Test emergency stop functionality"""
        result = portfolio_optimizer.emergency_stop("Test emergency")
        
        assert result == True
        # Should set equal weights in emergency
        assert np.allclose(portfolio_optimizer.current_weights, 0.2, atol=1e-6)
    
    def test_portfolio_metrics(self, portfolio_optimizer):
        """Test portfolio metrics calculation"""
        # Add some strategy performances
        for i in range(5):
            perf = StrategyPerformance(
                strategy_id=f'Strategy_{i}',
                returns=[0.01, 0.02, -0.01],
                volatility=0.15 + i * 0.02,
                sharpe_ratio=1.0 + i * 0.1,
                max_drawdown=0.05,
                var_95=0.02,
                correlation_with_portfolio=0.5 + i * 0.05,
                current_weight=0.2
            )
            portfolio_optimizer.update_strategy_performance(f'Strategy_{i}', perf)
        
        metrics = portfolio_optimizer.get_portfolio_metrics()
        
        assert 'current_weights' in metrics
        assert 'strategy_performances' in metrics
        assert len(metrics['strategy_performances']) == 5


class TestCorrelationManager:
    """Test suite for Portfolio Correlation Manager"""
    
    @pytest.fixture
    def correlation_tracker(self):
        tracker = Mock(spec=CorrelationTracker)
        tracker.get_correlation_matrix.return_value = np.array([
            [1.0, 0.3, 0.2, 0.4, 0.1],
            [0.3, 1.0, 0.5, 0.2, 0.3],
            [0.2, 0.5, 1.0, 0.1, 0.4],
            [0.4, 0.2, 0.1, 1.0, 0.2],
            [0.1, 0.3, 0.4, 0.2, 1.0]
        ])
        tracker.current_regime = CorrelationRegime.NORMAL
        return tracker
    
    @pytest.fixture
    def event_bus(self):
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def correlation_manager(self, correlation_tracker, event_bus):
        return PortfolioCorrelationManager(
            correlation_tracker=correlation_tracker,
            event_bus=event_bus,
            n_strategies=5
        )
    
    def test_correlation_risk_analysis(self, correlation_manager, correlation_tracker):
        """Test correlation risk analysis"""
        correlation_matrix = correlation_tracker.get_correlation_matrix.return_value
        
        risk_metrics = correlation_manager.analyze_correlation_risk(correlation_matrix)
        
        assert isinstance(risk_metrics, CorrelationRiskMetrics)
        assert 0 <= risk_metrics.average_correlation <= 1
        assert 0 <= risk_metrics.correlation_risk_score <= 1
        assert risk_metrics.effective_bets > 0
    
    def test_diversification_metrics(self, correlation_manager, correlation_tracker):
        """Test diversification metrics calculation"""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        volatilities = np.array([0.15, 0.18, 0.12, 0.20, 0.16])
        correlation_matrix = correlation_tracker.get_correlation_matrix.return_value
        
        div_metrics = correlation_manager.calculate_diversification_metrics(
            weights, volatilities, correlation_matrix
        )
        
        assert isinstance(div_metrics, DiversificationMetrics)
        assert div_metrics.diversification_ratio > 1.0  # Should be diversified
        assert div_metrics.effective_strategies > 0
        assert 0 <= div_metrics.concentration_index <= 1
    
    def test_diversification_optimization(self, correlation_manager, correlation_tracker):
        """Test diversification optimization"""
        expected_returns = np.array([0.08, 0.10, 0.06, 0.12, 0.09])
        volatilities = np.array([0.15, 0.18, 0.12, 0.20, 0.16])
        correlation_matrix = correlation_tracker.get_correlation_matrix.return_value
        
        optimal_weights, div_metrics = correlation_manager.optimize_for_diversification(
            expected_returns, volatilities, correlation_matrix
        )
        
        assert len(optimal_weights) == 5
        assert np.allclose(np.sum(optimal_weights), 1.0, atol=1e-6)
        assert np.all(optimal_weights >= 0.05)  # Minimum weight constraint
        assert np.all(optimal_weights <= 0.5)   # Maximum weight constraint
        assert div_metrics.diversification_ratio > 1.0


class TestMultiObjectiveOptimizer:
    """Test suite for Multi-Objective Optimizer"""
    
    @pytest.fixture
    def multi_optimizer(self):
        return MultiObjectiveOptimizer(n_strategies=5)
    
    @pytest.fixture
    def test_data(self):
        expected_returns = np.array([0.08, 0.10, 0.06, 0.12, 0.09])
        covariance_matrix = np.array([
            [0.0225, 0.0054, 0.0036, 0.0072, 0.0027],
            [0.0054, 0.0324, 0.0108, 0.0072, 0.0081],
            [0.0036, 0.0108, 0.0144, 0.0024, 0.0057],
            [0.0072, 0.0072, 0.0024, 0.0400, 0.0048],
            [0.0027, 0.0081, 0.0057, 0.0048, 0.0256]
        ])
        return expected_returns, covariance_matrix
    
    def test_weighted_combination_optimization(self, multi_optimizer, test_data):
        """Test weighted combination optimization"""
        expected_returns, covariance_matrix = test_data
        
        objectives = OptimizationObjectives()
        constraints = OptimizationConstraints()
        
        result = multi_optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objectives=objectives,
            constraints=constraints,
            method=OptimizationMethod.WEIGHTED_COMBINATION
        )
        
        assert result.convergence_status == "SUCCESS"
        assert len(result.best_solution.weights) == 5
        assert np.allclose(np.sum(result.best_solution.weights), 1.0, atol=1e-6)
        assert result.best_solution.sharpe_ratio > 0
        assert result.best_solution.volatility > 0
    
    def test_pareto_frontier_optimization(self, multi_optimizer, test_data):
        """Test Pareto frontier optimization"""
        expected_returns, covariance_matrix = test_data
        
        objectives = OptimizationObjectives()
        constraints = OptimizationConstraints()
        
        result = multi_optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objectives=objectives,
            constraints=constraints,
            method=OptimizationMethod.PARETO_FRONTIER
        )
        
        assert len(result.pareto_frontier) > 0
        assert all(isinstance(sol, ParetoSolution) for sol in result.pareto_frontier)
        
        # Check that solutions are non-dominated
        for i, sol1 in enumerate(result.pareto_frontier):
            for j, sol2 in enumerate(result.pareto_frontier):
                if i != j:
                    # No solution should strictly dominate another
                    assert not (sol1.sharpe_ratio > sol2.sharpe_ratio and 
                               sol1.volatility < sol2.volatility)
    
    def test_constraint_handling(self, multi_optimizer, test_data):
        """Test constraint handling in optimization"""
        expected_returns, covariance_matrix = test_data
        
        # Tight constraints
        constraints = OptimizationConstraints(
            min_weight=0.1,
            max_weight=0.3,
            max_concentration=0.6
        )
        
        result = multi_optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            constraints=constraints
        )
        
        weights = result.best_solution.weights
        assert np.all(weights >= 0.1 - 1e-6)  # Min weight constraint
        assert np.all(weights <= 0.3 + 1e-6)  # Max weight constraint
        
        # Check concentration constraint
        sorted_weights = np.sort(weights)[::-1]
        assert np.sum(sorted_weights[:3]) <= 0.6 + 1e-6


class TestRiskParityEngine:
    """Test suite for Risk Parity Engine"""
    
    @pytest.fixture
    def risk_parity_engine(self):
        return RiskParityEngine(n_strategies=5)
    
    @pytest.fixture
    def test_covariance(self):
        return np.array([
            [0.0225, 0.0054, 0.0036, 0.0072, 0.0027],
            [0.0054, 0.0324, 0.0108, 0.0072, 0.0081],
            [0.0036, 0.0108, 0.0144, 0.0024, 0.0057],
            [0.0072, 0.0072, 0.0024, 0.0400, 0.0048],
            [0.0027, 0.0081, 0.0057, 0.0048, 0.0256]
        ])
    
    def test_equal_risk_contribution(self, risk_parity_engine, test_covariance):
        """Test Equal Risk Contribution optimization"""
        result = risk_parity_engine.optimize_equal_risk_contribution(test_covariance)
        
        assert isinstance(result, RiskParityResult)
        assert result.method == RiskParityMethod.EQUAL_RISK_CONTRIBUTION
        assert len(result.weights) == 5
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        
        # Check that risk contributions are approximately equal
        risk_contribs = [rc.risk_contribution_pct for rc in result.risk_contributions]
        target_contrib = 1.0 / 5  # 20% each
        
        # Should be close to equal risk contribution
        for contrib in risk_contribs:
            assert abs(contrib - target_contrib) < 0.1  # Within 10% of target
    
    def test_hierarchical_risk_parity(self, risk_parity_engine, test_covariance):
        """Test Hierarchical Risk Parity"""
        result = risk_parity_engine.optimize_hierarchical_risk_parity(test_covariance)
        
        assert isinstance(result, RiskParityResult)
        assert result.method == RiskParityMethod.HIERARCHICAL_RISK_PARITY
        assert len(result.weights) == 5
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert result.diversification_ratio > 1.0
    
    def test_risk_budgeting(self, risk_parity_engine, test_covariance):
        """Test risk budgeting optimization"""
        # Define custom risk budgets
        risk_budgets = [
            RiskBudget(strategy_id='0', target_risk_contribution=0.3),
            RiskBudget(strategy_id='1', target_risk_contribution=0.25),
            RiskBudget(strategy_id='2', target_risk_contribution=0.2),
            RiskBudget(strategy_id='3', target_risk_contribution=0.15),
            RiskBudget(strategy_id='4', target_risk_contribution=0.1)
        ]
        
        result = risk_parity_engine.optimize_risk_budgeting(test_covariance, risk_budgets)
        
        assert isinstance(result, RiskParityResult)
        assert result.method == RiskParityMethod.RISK_BUDGETING
        assert len(result.weights) == 5
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        
        # Check that risk contributions match targets approximately
        risk_contribs = [rc.risk_contribution_pct for rc in result.risk_contributions]
        targets = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for actual, target in zip(risk_contribs, targets):
            assert abs(actual - target) < 0.15  # Within 15% of target
    
    def test_inverse_volatility(self, risk_parity_engine, test_covariance):
        """Test inverse volatility weighting"""
        result = risk_parity_engine.optimize_inverse_volatility(test_covariance)
        
        assert isinstance(result, RiskParityResult)
        assert result.method == RiskParityMethod.INVERSE_VOLATILITY
        assert len(result.weights) == 5
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        
        # Check inverse relationship with volatilities
        volatilities = np.sqrt(np.diag(test_covariance))
        for i, weight in enumerate(result.weights):
            # Higher volatility should have lower weight
            relative_vol = volatilities[i] / np.mean(volatilities)
            relative_weight = weight / np.mean(result.weights)
            # Inverse relationship (not perfect due to normalization)
            assert relative_vol * relative_weight < 2.0


class TestPerformanceAttribution:
    """Test suite for Performance Attribution Engine"""
    
    @pytest.fixture
    def attribution_engine(self):
        return PerformanceAttributionEngine(n_strategies=5)
    
    def test_strategy_performance_calculation(self, attribution_engine):
        """Test strategy performance metrics calculation"""
        # Add sample return data
        strategy_id = 'Strategy_0'
        returns_data = []
        for i in range(100):
            date = datetime.now() - timedelta(days=100-i)
            return_val = np.random.normal(0.0008, 0.02)  # Daily return
            returns_data.append((date, return_val))
        
        attribution_engine.strategy_returns[strategy_id] = returns_data
        
        # Add corresponding benchmark returns
        benchmark_returns = []
        for i in range(100):
            date = datetime.now() - timedelta(days=100-i)
            return_val = np.random.normal(0.0005, 0.015)  # Benchmark return
            benchmark_returns.append((date, return_val))
        
        attribution_engine.benchmark_returns = benchmark_returns
        attribution_engine.portfolio_returns = benchmark_returns  # Simplified
        
        metrics = attribution_engine.calculate_strategy_performance_metrics(strategy_id)
        
        assert isinstance(metrics, StrategyPerformanceMetrics)
        assert metrics.strategy_id == strategy_id
        assert metrics.volatility > 0
        assert -1 <= metrics.sharpe_ratio <= 5  # Reasonable range
        assert -1 <= metrics.max_drawdown <= 0  # Drawdown should be negative
        assert 0 <= metrics.hit_ratio <= 1  # Hit ratio is a percentage
    
    def test_portfolio_attribution(self, attribution_engine):
        """Test portfolio attribution analysis"""
        # Setup sample data
        n_periods = 50
        
        # Portfolio returns
        for i in range(n_periods):
            date = datetime.now() - timedelta(days=n_periods-i)
            portfolio_return = np.random.normal(0.001, 0.02)
            attribution_engine.portfolio_returns.append((date, portfolio_return))
        
        # Benchmark returns
        for i in range(n_periods):
            date = datetime.now() - timedelta(days=n_periods-i)
            benchmark_return = np.random.normal(0.0008, 0.015)
            attribution_engine.benchmark_returns.append((date, benchmark_return))
        
        # Strategy returns
        for strategy_id in ['Strategy_0', 'Strategy_1', 'Strategy_2', 'Strategy_3', 'Strategy_4']:
            strategy_returns = []
            for i in range(n_periods):
                date = datetime.now() - timedelta(days=n_periods-i)
                return_val = np.random.normal(0.001, 0.02)
                strategy_returns.append((date, return_val))
            attribution_engine.strategy_returns[strategy_id] = strategy_returns
        
        # Weights history
        for i in range(n_periods):
            date = datetime.now() - timedelta(days=n_periods-i)
            weights = np.random.dirichlet([1, 1, 1, 1, 1])  # Random weights that sum to 1
            attribution_engine.weights_history.append((date, weights))
        
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        assert attribution is not None
        assert attribution.method == AttributionMethod.BRINSON_FACHLER
        assert len(attribution.strategy_contributions) == 5
        assert -1 <= attribution.information_ratio <= 5  # Reasonable range
        assert 0 <= attribution.attribution_r_squared <= 1


class TestDynamicRebalancingEngine:
    """Test suite for Dynamic Rebalancing Engine"""
    
    @pytest.fixture
    def event_bus(self):
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def rebalance_config(self):
        return RebalanceConfig(
            drift_threshold=0.05,
            max_response_time_ms=10.0,
            use_async_processing=True
        )
    
    @pytest.fixture
    def rebalancing_engine(self, event_bus, rebalance_config):
        return DynamicRebalancingEngine(
            event_bus=event_bus,
            n_strategies=5,
            config=rebalance_config
        )
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, rebalancing_engine):
        """Test <10ms response time requirement"""
        new_weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
        
        # Measure response time
        start_time = time.perf_counter()
        
        execution = await rebalancing_engine.process_rebalance_signal(
            new_target_weights=new_weights,
            trigger=RebalanceTrigger.MANUAL,
            urgency=RebalanceUrgency.HIGH
        )
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        # Should meet <10ms target
        assert response_time < 10.0, f"Response time {response_time:.2f}ms exceeds 10ms target"
        
        if execution:
            assert execution.success
            assert len(execution.executed_weights) == 5
            assert np.allclose(np.sum(execution.executed_weights), 1.0, atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_critical_rebalancing(self, rebalancing_engine):
        """Test critical rebalancing execution"""
        new_weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        
        execution = await rebalancing_engine.process_rebalance_signal(
            new_target_weights=new_weights,
            trigger=RebalanceTrigger.EMERGENCY,
            urgency=RebalanceUrgency.CRITICAL,
            reason="Emergency risk breach"
        )
        
        assert execution is not None
        assert execution.success
        assert execution.execution_time_ms < 10.0  # Should be very fast
        
        # Weights should be applied (with possible emergency constraints)
        assert len(execution.executed_weights) == 5
        assert np.allclose(np.sum(execution.executed_weights), 1.0, atol=1e-6)
    
    def test_drift_threshold_detection(self, rebalancing_engine):
        """Test drift threshold detection"""
        # Set current weights
        rebalancing_engine.current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Small drift - should not trigger
        small_drift_weights = np.array([0.22, 0.21, 0.19, 0.19, 0.19])
        max_drift = np.max(np.abs(small_drift_weights - rebalancing_engine.current_weights))
        
        should_rebalance = rebalancing_engine._should_rebalance(
            max_drift, RebalanceTrigger.DRIFT_THRESHOLD, RebalanceUrgency.LOW
        )
        assert not should_rebalance  # Below threshold
        
        # Large drift - should trigger
        large_drift_weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1])
        max_drift = np.max(np.abs(large_drift_weights - rebalancing_engine.current_weights))
        
        should_rebalance = rebalancing_engine._should_rebalance(
            max_drift, RebalanceTrigger.DRIFT_THRESHOLD, RebalanceUrgency.MEDIUM
        )
        assert should_rebalance  # Above threshold
    
    def test_transaction_cost_estimation(self, rebalancing_engine):
        """Test transaction cost estimation"""
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        target_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Use asyncio.run for async function
        async def run_cost_calculation():
            return await rebalancing_engine._calculate_transaction_costs(
                current_weights, target_weights
            )
        
        cost = asyncio.run(run_cost_calculation())
        
        assert cost >= 0  # Costs should be non-negative
        assert cost < 0.1  # Should be reasonable (less than 10%)
        
        # Higher turnover should result in higher costs
        high_turnover_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        
        async def run_high_cost_calculation():
            return await rebalancing_engine._calculate_transaction_costs(
                current_weights, high_turnover_weights
            )
        
        high_cost = asyncio.run(run_high_cost_calculation())
        assert high_cost > cost  # Higher turnover should cost more
    
    def test_emergency_mode(self, rebalancing_engine):
        """Test emergency mode functionality"""
        # Enable emergency mode
        rebalancing_engine.enable_emergency_mode("Test emergency")
        assert rebalancing_engine.emergency_mode == True
        
        # Test emergency constraints
        risky_weights = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
        constrained_weights = rebalancing_engine._apply_emergency_constraints(risky_weights)
        
        # Should limit maximum position size
        assert np.max(constrained_weights) <= rebalancing_engine.emergency_protocols['max_position_size']
        assert np.allclose(np.sum(constrained_weights), 1.0, atol=1e-6)
        
        # Disable emergency mode
        rebalancing_engine.disable_emergency_mode("Test completed")
        assert rebalancing_engine.emergency_mode == False
    
    def test_performance_metrics(self, rebalancing_engine):
        """Test performance metrics collection"""
        # Add some mock performance data
        rebalancing_engine.response_times = [5.2, 7.8, 6.1, 8.9, 4.3]
        rebalancing_engine.execution_times = [3.1, 4.2, 3.8, 5.1, 2.9]
        rebalancing_engine.rebalance_count = 5
        rebalancing_engine.performance_violations = 1
        
        metrics = rebalancing_engine.get_performance_metrics()
        
        assert metrics['rebalance_count'] == 5
        assert metrics['avg_response_time_ms'] == np.mean([5.2, 7.8, 6.1, 8.9, 4.3])
        assert metrics['target_response_time_ms'] == 10.0
        assert metrics['response_time_violations'] == 1
        assert 'success_rate' in metrics
        assert 'current_weights' in metrics


class TestIntegration:
    """Integration tests for the complete Portfolio Optimizer system"""
    
    @pytest.fixture
    def complete_system(self):
        """Setup complete portfolio optimization system"""
        event_bus = Mock(spec=EventBus)
        
        # Setup correlation tracker
        correlation_tracker = Mock(spec=CorrelationTracker)
        correlation_tracker.get_correlation_matrix.return_value = np.eye(5)
        correlation_tracker.current_regime = CorrelationRegime.NORMAL
        
        # Setup VaR calculator
        var_calculator = Mock(spec=VaRCalculator)
        
        # Setup portfolio optimizer
        config = {
            'n_strategies': 5,
            'target_volatility': 0.15,
            'rebalance_threshold': 0.05
        }
        
        portfolio_optimizer = PortfolioOptimizerAgent(
            config=config,
            correlation_tracker=correlation_tracker,
            var_calculator=var_calculator,
            event_bus=event_bus
        )
        
        # Setup rebalancing engine
        rebalance_config = RebalanceConfig(max_response_time_ms=10.0)
        rebalancing_engine = DynamicRebalancingEngine(
            event_bus=event_bus,
            n_strategies=5,
            config=rebalance_config
        )
        
        return {
            'portfolio_optimizer': portfolio_optimizer,
            'rebalancing_engine': rebalancing_engine,
            'correlation_tracker': correlation_tracker,
            'event_bus': event_bus
        }
    
    def test_end_to_end_optimization_flow(self, complete_system):
        """Test complete optimization flow"""
        portfolio_optimizer = complete_system['portfolio_optimizer']
        rebalancing_engine = complete_system['rebalancing_engine']
        
        # Create risk state
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.4,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.03,
            margin_usage_pct=0.25,
            time_of_day_risk=0.5,
            market_stress_level=0.2,
            liquidity_conditions=0.8
        )
        
        # Get optimization result
        start_time = time.perf_counter()
        new_weights, confidence = portfolio_optimizer.calculate_risk_action(risk_state)
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        # Verify optimization results
        assert isinstance(new_weights, np.ndarray)
        assert len(new_weights) == 5
        assert np.allclose(np.sum(new_weights), 1.0, atol=1e-6)
        assert 0.0 <= confidence <= 1.0
        assert optimization_time < 10.0  # Performance requirement
        
        # Test constraint validation
        constraints_valid = portfolio_optimizer.validate_risk_constraints(risk_state)
        assert isinstance(constraints_valid, bool)
    
    @pytest.mark.asyncio
    async def test_rebalancing_integration(self, complete_system):
        """Test integration between optimizer and rebalancing engine"""
        portfolio_optimizer = complete_system['portfolio_optimizer']
        rebalancing_engine = complete_system['rebalancing_engine']
        
        # Get new weights from optimizer
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.3,
            correlation_risk=0.4,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.3,
            time_of_day_risk=0.5,
            market_stress_level=0.2,
            liquidity_conditions=0.8
        )
        
        new_weights, confidence = portfolio_optimizer.calculate_risk_action(risk_state)
        
        # Execute rebalancing
        execution = await rebalancing_engine.process_rebalance_signal(
            new_target_weights=new_weights,
            trigger=RebalanceTrigger.MANUAL,
            urgency=RebalanceUrgency.MEDIUM,
            reason="Integration test"
        )
        
        # Verify execution
        if execution:
            assert execution.success
            assert execution.execution_time_ms < 10.0
            assert len(execution.executed_weights) == 5
            assert np.allclose(np.sum(execution.executed_weights), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])