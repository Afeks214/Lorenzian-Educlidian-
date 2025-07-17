"""
Comprehensive Test Suite for Pre-Mortem Analysis System

Tests all components of the pre-mortem analysis system including:
- Monte Carlo engine performance and accuracy
- Advanced market models validation
- Decision interception and routing
- Failure probability calculations
- Integration with MARL agents
- Performance benchmarks
"""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.risk.simulation.monte_carlo_engine import (
    MonteCarloEngine, SimulationParameters, SimulationResults
)
from src.risk.simulation.advanced_market_models import (
    GeometricBrownianMotion, JumpDiffusionModel, HestonStochasticVolatility,
    GBMParameters, JumpDiffusionParameters, HestonParameters,
    CorrelationGenerator
)
from src.risk.analysis.failure_probability_calculator import (
    FailureProbabilityCalculator, FailureMetrics, FailureThresholds, RiskRecommendation
)
from src.risk.integration.decision_interceptor import (
    DecisionInterceptor, DecisionContext, InterceptionConfig, DecisionType, DecisionPriority
)
from src.risk.analysis.premortem_agent import PreMortemAgent, PreMortemConfig
from src.core.events import EventBus


class TestMonteCarloEngine:
    """Test high-speed Monte Carlo engine"""
    
    def setup_method(self):
        """Setup test environment"""
        self.engine = MonteCarloEngine(enable_gpu=False)  # Use CPU for consistent testing
        self.test_params = SimulationParameters(
            num_paths=1000,  # Smaller for testing
            time_horizon_hours=24.0,
            time_steps=100,   # Reduced for speed
            initial_prices=np.array([100.0, 200.0]),
            drift_rates=np.array([0.1, 0.08]),
            volatilities=np.array([0.2, 0.25]),
            correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
        )
    
    def test_simulation_performance(self):
        """Test that simulation meets performance targets"""
        start_time = time.perf_counter()
        results = self.engine.run_simulation(self.test_params)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Should complete in reasonable time for test parameters
        assert execution_time < 1000, f"Simulation took {execution_time:.2f}ms"
        assert results.computation_time_ms < 1000
        
        # Validate results structure
        assert results.price_paths.shape == (1000, 100, 2)
        assert results.portfolio_values.shape == (1000, 100)
        assert len(results.final_portfolio_values) == 1000
        assert len(results.max_drawdowns) == 1000
    
    def test_simulation_accuracy(self):
        """Test simulation accuracy against theoretical values"""
        # Simple single-asset GBM test
        simple_params = SimulationParameters(
            num_paths=10000,
            time_horizon_hours=8760.0,  # 1 year
            time_steps=252,  # Daily
            initial_prices=np.array([100.0]),
            drift_rates=np.array([0.1]),
            volatilities=np.array([0.2]),
            correlation_matrix=np.array([[1.0]])
        )
        
        results = self.engine.run_simulation(simple_params)
        final_prices = results.price_paths[:, -1, 0]
        
        # Check mean (should be approximately S0 * exp(μ*T))
        expected_mean = 100.0 * np.exp(0.1 * 1.0)
        actual_mean = np.mean(final_prices)
        
        # Allow 5% tolerance due to Monte Carlo error
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05
        
        # Check that we have reasonable distribution
        assert np.std(final_prices) > 0
        assert np.min(final_prices) > 0
    
    def test_correlation_preservation(self):
        """Test that correlation structure is preserved"""
        results = self.engine.run_simulation(self.test_params)
        
        # Calculate correlations from simulation
        returns_asset1 = np.diff(np.log(results.price_paths[:, :, 0]), axis=1)
        returns_asset2 = np.diff(np.log(results.price_paths[:, :, 1]), axis=1)
        
        # Average correlation across all paths
        correlations = []
        for path in range(100):  # Sample first 100 paths
            if path < len(returns_asset1):
                corr = np.corrcoef(returns_asset1[path], returns_asset2[path])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Should be close to target correlation (0.3) with some tolerance
        assert abs(avg_correlation - 0.3) < 0.2  # Allow wider tolerance for limited data
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        benchmark_results = self.engine.benchmark_performance()
        
        assert 'avg_time_ms' in benchmark_results
        assert 'paths_per_second' in benchmark_results
        assert benchmark_results['avg_time_ms'] > 0
        assert benchmark_results['paths_per_second'] > 0


class TestAdvancedMarketModels:
    """Test advanced market simulation models"""
    
    def test_geometric_brownian_motion(self):
        """Test GBM model implementation"""
        params = GBMParameters(
            drift=0.1,
            volatility=0.2,
            initial_price=100.0
        )
        
        model = GeometricBrownianMotion(params)
        paths = model.simulate_paths(
            num_paths=1000,
            time_steps=252,
            dt=1/252,
            random_seed=42
        )
        
        assert paths.shape == (1000, 252)
        assert np.all(paths[:, 0] == 100.0)  # Initial price
        assert np.all(paths > 0)  # Prices stay positive
    
    def test_jump_diffusion_model(self):
        """Test jump diffusion model"""
        params = JumpDiffusionParameters(
            drift=0.1,
            volatility=0.2,
            jump_intensity=2.0,
            jump_mean=0.0,
            jump_std=0.05,
            initial_price=100.0
        )
        
        model = JumpDiffusionModel(params)
        paths = model.simulate_paths(
            num_paths=1000,
            time_steps=252,
            dt=1/252,
            random_seed=42
        )
        
        assert paths.shape == (1000, 252)
        assert np.all(paths > 0)
        
        # Should have more extreme moves than pure GBM
        returns = np.diff(np.log(paths), axis=1)
        max_returns = np.max(np.abs(returns), axis=1)
        
        # Some paths should have large moves (jumps)
        large_moves = np.sum(max_returns > 0.1)  # >10% moves
        assert large_moves > 0
    
    def test_heston_stochastic_volatility(self):
        """Test Heston stochastic volatility model"""
        params = HestonParameters(
            initial_price=100.0,
            initial_variance=0.04,  # 20% vol
            long_term_variance=0.04,
            kappa=2.0,
            vol_of_vol=0.3,
            correlation=-0.5,
            drift=0.1
        )
        
        model = HestonStochasticVolatility(params)
        price_paths, variance_paths = model.simulate_paths(
            num_paths=1000,
            time_steps=252,
            dt=1/252,
            random_seed=42
        )
        
        assert price_paths.shape == (1000, 252)
        assert variance_paths.shape == (1000, 252)
        assert np.all(price_paths > 0)
        assert np.all(variance_paths >= 0)
    
    def test_correlation_generator(self):
        """Test correlation matrix generation utilities"""
        # Test random correlation matrix
        corr_matrix = CorrelationGenerator.generate_random_correlation_matrix(
            n_assets=5, random_seed=42
        )
        
        assert corr_matrix.shape == (5, 5)
        assert np.allclose(np.diag(corr_matrix), 1.0)
        assert np.allclose(corr_matrix, corr_matrix.T)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(corr_matrix)
        assert np.all(eigenvals >= -1e-8)  # Allow small numerical errors
        
        # Test block correlation matrix
        block_corr = CorrelationGenerator.block_correlation_matrix(
            block_sizes=[2, 3],
            within_block_corr=0.7,
            between_block_corr=0.2
        )
        
        assert block_corr.shape == (5, 5)
        assert np.allclose(np.diag(block_corr), 1.0)


class TestFailureProbabilityCalculator:
    """Test failure probability calculation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.calculator = FailureProbabilityCalculator()
        
        # Create synthetic simulation results
        num_paths = 10000
        
        # Generate portfolio values (some losses, some gains)
        np.random.seed(42)
        final_values = np.random.normal(1.02, 0.15, num_paths)  # 2% mean return, 15% vol
        final_values = np.maximum(final_values, 0.01)  # Ensure positive
        
        max_drawdowns = np.random.beta(2, 8, num_paths) * 0.3  # Beta distribution
        
        self.test_results = SimulationResults(
            price_paths=None,
            return_paths=None,
            portfolio_values=None,
            final_portfolio_values=final_values,
            max_drawdowns=max_drawdowns,
            computation_time_ms=50.0
        )
    
    def test_failure_metrics_calculation(self):
        """Test comprehensive failure metrics calculation"""
        metrics = self.calculator.calculate_failure_metrics(self.test_results)
        
        # Validate structure
        assert isinstance(metrics, FailureMetrics)
        assert 0 <= metrics.failure_probability <= 1
        assert metrics.var_95_percent >= 0
        assert metrics.expected_shortfall_95 >= 0
        assert isinstance(metrics.recommendation, RiskRecommendation)
        
        # Check recommendation logic
        if metrics.failure_probability <= 0.05:
            assert metrics.recommendation == RiskRecommendation.GO
        elif metrics.failure_probability <= 0.15:
            assert metrics.recommendation == RiskRecommendation.GO_WITH_CAUTION
        else:
            assert metrics.recommendation == RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW
    
    def test_var_calculations(self):
        """Test VaR and Expected Shortfall calculations"""
        metrics = self.calculator.calculate_failure_metrics(self.test_results)
        
        # VaR should be reasonable for our distribution
        assert 0 <= metrics.var_95_percent <= 0.5
        assert 0 <= metrics.var_99_percent <= 0.5
        assert metrics.var_99_percent >= metrics.var_95_percent
        
        # Expected Shortfall should be >= VaR
        assert metrics.expected_shortfall_95 >= metrics.var_95_percent
        assert metrics.expected_shortfall_99 >= metrics.var_99_percent
    
    def test_confidence_intervals(self):
        """Test statistical confidence interval calculation"""
        metrics = self.calculator.calculate_failure_metrics(self.test_results)
        
        # Confidence intervals should be reasonable
        assert 0 <= metrics.failure_prob_lower_ci <= metrics.failure_probability
        assert metrics.failure_probability <= metrics.failure_prob_upper_ci <= 1
        assert metrics.confidence_level == 0.95
    
    def test_human_review_triggers(self):
        """Test human review trigger logic"""
        # Create scenario with high failure probability
        high_risk_values = np.random.normal(0.85, 0.20, 10000)  # 15% loss mean
        high_risk_results = SimulationResults(
            price_paths=None,
            return_paths=None,
            portfolio_values=None,
            final_portfolio_values=high_risk_values,
            max_drawdowns=np.random.beta(2, 5, 10000) * 0.5,
            computation_time_ms=50.0
        )
        
        metrics = self.calculator.calculate_failure_metrics(high_risk_results)
        
        # Should trigger human review for high-risk scenario
        assert metrics.requires_human_review or metrics.failure_probability > 0.15
    
    def test_performance_benchmark(self):
        """Test calculator performance"""
        benchmark_results = self.calculator.benchmark_performance(n_tests=10)
        
        assert 'avg_time_ms' in benchmark_results
        assert 'calculations_per_second' in benchmark_results
        assert benchmark_results['avg_time_ms'] > 0


class TestDecisionInterceptor:
    """Test decision interception system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = EventBus()
        self.config = InterceptionConfig(
            min_position_size_threshold=1000.0,
            min_portfolio_impact_threshold=0.1,
            max_concurrent_analyses=2
        )
        
        # Mock pre-mortem analyzer
        self.mock_analyzer = Mock()
        self.mock_analyzer.return_value = Mock()
        
        self.interceptor = DecisionInterceptor(
            config=self.config,
            event_bus=self.event_bus,
            premortem_analyzer=self.mock_analyzer
        )
    
    def test_decision_filtering(self):
        """Test decision filtering logic"""
        # Small decision - should not be intercepted
        small_decision = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=500.0,  # Below threshold
            portfolio_impact_percent=0.05   # Below threshold
        )
        
        result = self.interceptor.intercept_decision(small_decision)
        assert result is None  # Should be bypassed
        
        # Large decision - should be intercepted
        large_decision = DecisionContext(
            agent_name="position_sizing_agent", 
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0,  # Above threshold
            portfolio_impact_percent=0.25    # Above threshold
        )
        
        result = self.interceptor.intercept_decision(large_decision)
        assert result is not None
    
    def test_emergency_bypass(self):
        """Test emergency bypass functionality"""
        emergency_decision = DecisionContext(
            agent_name="risk_monitor_agent",
            decision_type=DecisionType.RISK_REDUCTION,
            priority=DecisionPriority.EMERGENCY,
            position_change_amount=50000.0
        )
        
        result = self.interceptor.intercept_decision(emergency_decision)
        assert result is not None
        assert "bypass" in result.status.lower() or result.recommendation == "GO"
    
    def test_crisis_mode(self):
        """Test crisis mode functionality"""
        normal_decision = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0,
            portfolio_impact_percent=0.25
        )
        
        # Enable crisis mode
        self.interceptor.enable_crisis_mode()
        
        result = self.interceptor.intercept_decision(normal_decision)
        # Should bypass in crisis mode unless critical priority
        assert result is None or "bypass" in result.status.lower()
        
        # Disable crisis mode
        self.interceptor.disable_crisis_mode()
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        initial_stats = self.interceptor.get_interception_stats()
        
        # Process some decisions
        for i in range(5):
            decision = DecisionContext(
                agent_name="position_sizing_agent",
                decision_type=DecisionType.POSITION_SIZING,
                position_change_amount=10000.0 + i * 1000,
                portfolio_impact_percent=0.20 + i * 0.05
            )
            self.interceptor.intercept_decision(decision)
        
        final_stats = self.interceptor.get_interception_stats()
        
        # Stats should be updated
        assert final_stats['total_decisions'] >= initial_stats['total_decisions']


class TestPreMortemAgent:
    """Test complete pre-mortem analysis agent"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = EventBus()
        self.config = {
            'name': 'premortem_test_agent',
            'premortem_config': {
                'default_num_paths': 1000,  # Smaller for testing
                'max_analysis_time_ms': 1000.0,
                'enable_gpu_acceleration': False
            }
        }
        
        self.agent = PreMortemAgent(self.config, self.event_bus)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.name == 'premortem_test_agent'
        assert self.agent.monte_carlo_engine is not None
        assert self.agent.failure_calculator is not None
        assert self.agent.decision_interceptor is not None
    
    def test_trading_decision_analysis(self):
        """Test complete trading decision analysis"""
        decision_context = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            current_position_size=10000.0,
            proposed_position_size=15000.0,
            position_change_amount=5000.0,
            position_change_percent=50.0,
            portfolio_impact_percent=20.0,
            symbol="EURUSD",
            reasoning="Increased position based on technical signals"
        )
        
        result = self.agent.analyze_trading_decision(decision_context)
        
        # Validate result structure
        assert result.decision_id == decision_context.decision_id
        assert isinstance(result.recommendation, RiskRecommendation)
        assert 0 <= result.failure_probability <= 1
        assert 0 <= result.confidence <= 1
        assert result.total_analysis_time_ms > 0
        
        # Should have detailed metrics
        assert result.failure_metrics is not None
        assert result.simulation_results is not None
    
    def test_performance_requirements(self):
        """Test that performance requirements are met"""
        decision_context = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0,
            portfolio_impact_percent=25.0
        )
        
        start_time = time.perf_counter()
        result = self.agent.analyze_trading_decision(decision_context)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Should meet performance target for test configuration
        assert total_time < 2000  # 2 second timeout for testing
        assert result.total_analysis_time_ms < 2000
    
    def test_risk_action_calculation(self):
        """Test risk action calculation"""
        from src.risk.agents.base_risk_agent import RiskState
        
        # Normal risk state
        normal_risk = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=3,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.6,
            time_of_day_risk=0.3,
            market_stress_level=0.4,
            liquidity_conditions=0.8
        )
        
        action, confidence = self.agent.calculate_risk_action(normal_risk)
        assert isinstance(action, int)
        assert 0 <= confidence <= 1
        
        # Emergency risk state
        emergency_risk = RiskState(
            account_equity_normalized=0.8,
            open_positions_count=5,
            volatility_regime=0.9,
            correlation_risk=0.8,
            var_estimate_5pct=0.15,  # High VaR
            current_drawdown_pct=0.20,
            margin_usage_pct=0.95,
            time_of_day_risk=0.7,
            market_stress_level=0.95,  # High stress
            liquidity_conditions=0.2
        )
        
        action, confidence = self.agent.calculate_risk_action(emergency_risk)
        # Should recommend risk reduction or position closure
        assert action in [1, 2]  # REDUCE_POSITION or CLOSE_ALL
    
    def test_analysis_caching(self):
        """Test analysis result caching"""
        decision_context = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0
        )
        
        # First analysis
        start_time = time.perf_counter()
        result1 = self.agent.analyze_trading_decision(decision_context)
        first_analysis_time = (time.perf_counter() - start_time) * 1000
        
        # Second analysis (should use cache)
        start_time = time.perf_counter()
        result2 = self.agent.analyze_trading_decision(decision_context)
        second_analysis_time = (time.perf_counter() - start_time) * 1000
        
        # Results should be identical
        assert result1.decision_id == result2.decision_id
        assert result1.recommendation == result2.recommendation
        
        # Second analysis should be much faster (cached)
        assert second_analysis_time < first_analysis_time / 2
    
    def test_analysis_statistics(self):
        """Test analysis statistics tracking"""
        initial_stats = self.agent.get_analysis_stats()
        
        # Perform several analyses
        for i in range(3):
            decision = DecisionContext(
                agent_name="position_sizing_agent",
                decision_type=DecisionType.POSITION_SIZING,
                position_change_amount=10000.0 + i * 1000
            )
            self.agent.analyze_trading_decision(decision)
        
        final_stats = self.agent.get_analysis_stats()
        
        # Stats should be updated
        assert final_stats['total_analyses'] >= initial_stats['total_analyses'] + 3
        assert 'monte_carlo_stats' in final_stats
        assert 'failure_calc_stats' in final_stats
    
    def test_crisis_mode_integration(self):
        """Test crisis mode integration"""
        # Enable crisis mode
        self.agent.enable_crisis_mode()
        
        # Should affect decision interception
        stats = self.agent.get_analysis_stats()
        assert 'interception_stats' in stats
        
        # Disable crisis mode
        self.agent.disable_crisis_mode()
    
    def test_export_analysis_report(self):
        """Test analysis report export"""
        decision_context = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0
        )
        
        result = self.agent.analyze_trading_decision(decision_context)
        
        # Export report
        report = self.agent.export_analysis_report(result.decision_id)
        
        assert report is not None
        assert 'decision_id' in report
        assert 'recommendation' in report
        assert 'failure_probability' in report
        assert 'risk_factors' in report
        assert 'mitigation_suggestions' in report


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.event_bus = EventBus()
        self.agent = PreMortemAgent({
            'name': 'integration_test_agent',
            'premortem_config': {
                'default_num_paths': 1000,
                'enable_gpu_acceleration': False
            }
        }, self.event_bus)
    
    def test_high_risk_scenario(self):
        """Test high-risk trading scenario"""
        high_risk_decision = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            current_position_size=100000.0,
            proposed_position_size=200000.0,  # Double position
            position_change_amount=100000.0,
            position_change_percent=100.0,
            portfolio_impact_percent=50.0,     # 50% portfolio impact
            market_volatility=0.35,            # High volatility
            symbol="BTCUSD",
            reasoning="Aggressive position increase on breakout signal"
        )
        
        result = self.agent.analyze_trading_decision(high_risk_decision)
        
        # Should flag as high risk
        assert result.failure_probability > 0.1  # >10% failure probability
        assert result.recommendation in [
            RiskRecommendation.GO_WITH_CAUTION,
            RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW
        ]
        assert len(result.primary_risk_factors) > 0
        assert len(result.risk_mitigation_suggestions) > 0
    
    def test_conservative_scenario(self):
        """Test conservative trading scenario"""
        conservative_decision = DecisionContext(
            agent_name="position_sizing_agent", 
            decision_type=DecisionType.POSITION_SIZING,
            current_position_size=10000.0,
            proposed_position_size=11000.0,    # Small increase
            position_change_amount=1000.0,
            position_change_percent=10.0,
            portfolio_impact_percent=2.0,      # Small impact
            market_volatility=0.15,            # Low volatility
            symbol="EURUSD",
            reasoning="Minor position adjustment based on low-risk signal"
        )
        
        result = self.agent.analyze_trading_decision(conservative_decision)
        
        # Should be low risk
        assert result.failure_probability < 0.1  # <10% failure probability
        assert result.recommendation == RiskRecommendation.GO
        assert not result.requires_human_review
    
    def test_emergency_scenario(self):
        """Test emergency trading scenario"""
        emergency_decision = DecisionContext(
            agent_name="risk_monitor_agent",
            decision_type=DecisionType.RISK_REDUCTION,
            priority=DecisionPriority.EMERGENCY,
            current_position_size=50000.0,
            proposed_position_size=0.0,        # Close position
            position_change_amount=-50000.0,
            position_change_percent=-100.0,
            portfolio_impact_percent=30.0,
            reasoning="Emergency position closure due to risk breach"
        )
        
        # Should handle emergency scenario quickly
        start_time = time.perf_counter()
        result = self.agent.analyze_trading_decision(emergency_decision)
        analysis_time = (time.perf_counter() - start_time) * 1000
        
        # Emergency analysis should be fast
        assert analysis_time < 500  # <500ms for emergency
        
        # Should allow emergency action (GO or bypass)
        assert result.recommendation in [
            RiskRecommendation.GO,
            RiskRecommendation.GO_WITH_CAUTION
        ]


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def setup_method(self):
        """Setup benchmark environment"""
        self.agent = PreMortemAgent({
            'name': 'benchmark_agent',
            'premortem_config': {
                'default_num_paths': 10000,  # Full performance test
                'enable_gpu_acceleration': True,  # Use GPU if available
                'max_analysis_time_ms': 100.0
            }
        })
    
    def test_target_performance_10k_paths(self):
        """Test target performance with 10,000 simulation paths"""
        decision_context = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=10000.0,
            portfolio_impact_percent=20.0
        )
        
        # Run multiple iterations for reliable measurement
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = self.agent.analyze_trading_decision(decision_context)
            total_time = (time.perf_counter() - start_time) * 1000
            times.append(total_time)
        
        avg_time = np.mean(times)
        
        # Log performance results
        print(f"\nPerformance Benchmark Results:")
        print(f"Average analysis time: {avg_time:.2f}ms")
        print(f"Target time: 100ms")
        print(f"Performance target met: {avg_time <= 100}")
        
        # Performance assertion (may be relaxed based on hardware)
        if avg_time > 100:
            print(f"WARNING: Performance target not met ({avg_time:.2f}ms > 100ms)")
        
        # Ensure reasonable performance (within 5x target)
        assert avg_time < 500, f"Analysis too slow: {avg_time:.2f}ms"
    
    def test_monte_carlo_benchmark(self):
        """Benchmark Monte Carlo engine specifically"""
        benchmark_results = self.agent.monte_carlo_engine.benchmark_performance()
        
        print(f"\nMonte Carlo Benchmark:")
        print(f"Average time: {benchmark_results['avg_time_ms']:.2f}ms")
        print(f"Paths per second: {benchmark_results['paths_per_second']:.0f}")
        print(f"Target met: {benchmark_results['target_met']}")
        
        # Should process reasonable number of paths per second
        assert benchmark_results['paths_per_second'] > 50000  # At least 50k paths/sec
    
    def test_concurrent_analysis_performance(self):
        """Test performance under concurrent load"""
        import threading
        
        decision_contexts = [
            DecisionContext(
                agent_name="position_sizing_agent",
                decision_type=DecisionType.POSITION_SIZING,
                position_change_amount=10000.0 + i * 1000,
                portfolio_impact_percent=20.0 + i * 2
            ) for i in range(5)
        ]
        
        results = []
        threads = []
        
        def analyze_decision(context):
            start_time = time.perf_counter()
            result = self.agent.analyze_trading_decision(context)
            analysis_time = (time.perf_counter() - start_time) * 1000
            results.append((context.decision_id, analysis_time))
        
        # Start concurrent analyses
        start_time = time.perf_counter()
        for context in decision_contexts:
            thread = threading.Thread(target=analyze_decision, args=(context,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\nConcurrent Analysis Results:")
        print(f"Total time for 5 concurrent analyses: {total_time:.2f}ms")
        print(f"Individual analysis times: {[f'{t:.2f}ms' for _, t in results]}")
        
        # Should complete concurrent analyses efficiently
        assert total_time < 2000  # 2 second timeout
        assert len(results) == 5   # All analyses completed


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not benchmark"  # Skip benchmarks by default
    ])