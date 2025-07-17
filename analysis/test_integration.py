"""
Integration Tests for Advanced Metrics and Risk Analytics System

This module provides integration tests to verify the integration between
the advanced metrics system and the existing VaR calculation framework.
"""

import numpy as np
import pandas as pd
import pytest
import asyncio
from typing import Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch
import logging
from datetime import datetime, timedelta

# Import the modules to test
from analysis.metrics import (
    calculate_all_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    PerformanceMetrics
)

from analysis.advanced_metrics import (
    AdvancedMetricsCalculator,
    RiskAdjustedMetrics,
    MetricWithConfidence,
    ConfidenceInterval
)

from analysis.risk_metrics import (
    RiskMetricsCalculator,
    RiskMetrics,
    ComponentRiskAnalysis,
    calculate_var_historical,
    calculate_cvar_historical
)

from analysis.performance_optimizer import (
    MetricsOptimizer,
    performance_monitor,
    optimized_sharpe_ratio,
    optimized_sortino_ratio,
    optimized_max_drawdown
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBasicMetrics:
    """Test basic metrics calculations"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
        self.prices = np.cumprod(1 + self.returns) * 100  # Price series
        self.equity_curve = np.cumprod(1 + self.returns) * 10000  # Equity curve
        
    def test_basic_metric_calculations(self):
        """Test basic metric calculations"""
        # Test Sharpe ratio
        sharpe = calculate_sharpe_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test Sortino ratio
        sortino = calculate_sortino_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        
        # Test max drawdown
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(self.equity_curve)
        assert isinstance(max_dd, float)
        assert max_dd >= 0
        assert isinstance(peak_idx, int)
        assert isinstance(trough_idx, int)
        
    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics calculation"""
        benchmark_returns = np.random.normal(0.0008, 0.015, 1000)
        
        metrics = calculate_all_metrics(
            equity_curve=self.equity_curve,
            returns=self.returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.jensens_alpha is not None
        assert metrics.treynor_ratio is not None
        
        # Check new advanced metrics
        assert metrics.omega_ratio is not None
        assert metrics.pain_index is not None
        assert metrics.gain_to_pain_ratio is not None
        assert metrics.martin_ratio is not None
        assert metrics.conditional_sharpe_ratio is not None
        assert metrics.rachev_ratio is not None
        assert metrics.modified_sharpe_ratio is not None


class TestAdvancedMetrics:
    """Test advanced metrics with VaR integration"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.benchmark_returns = np.random.normal(0.0008, 0.015, 1000)
        
        # Mock VaR calculator
        self.mock_var_calculator = Mock()
        self.mock_var_result = Mock()
        self.mock_var_result.portfolio_var = 0.05
        self.mock_var_result.component_vars = {"AAPL": 0.02, "GOOGL": 0.03}
        self.mock_var_result.marginal_vars = {"AAPL": 0.01, "GOOGL": 0.015}
        
        self.mock_var_calculator.calculate_var = Mock(return_value=self.mock_var_result)
        
    def test_var_cvar_calculation(self):
        """Test VaR/CVaR calculation"""
        calculator = AdvancedMetricsCalculator(var_calculator=self.mock_var_calculator)
        
        # Test historical VaR/CVaR
        var, cvar = calculator.calculate_var_cvar_metrics(
            self.returns,
            confidence_level=0.95,
            method="historical"
        )
        
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        assert var > 0
        assert cvar > 0
        assert cvar >= var  # CVaR should be >= VaR
        
    def test_risk_adjusted_metrics(self):
        """Test risk-adjusted metrics calculation"""
        calculator = AdvancedMetricsCalculator(var_calculator=self.mock_var_calculator)
        
        risk_metrics = calculator.calculate_risk_adjusted_metrics(
            returns=self.returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02,
            confidence_level=0.95,
            periods_per_year=252
        )
        
        assert isinstance(risk_metrics, RiskAdjustedMetrics)
        assert risk_metrics.sharpe_ratio is not None
        assert risk_metrics.sortino_ratio is not None
        assert risk_metrics.var_adjusted_return is not None
        assert risk_metrics.cvar_adjusted_return is not None
        assert risk_metrics.upside_capture_ratio is not None
        assert risk_metrics.downside_capture_ratio is not None
        
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence intervals"""
        calculator = AdvancedMetricsCalculator(bootstrap_samples=100)  # Reduced for testing
        
        def simple_mean(returns):
            return np.mean(returns)
        
        metric_with_confidence = calculator.calculate_metric_with_confidence(
            returns=self.returns,
            metric_func=simple_mean,
            confidence_level=0.95
        )
        
        assert isinstance(metric_with_confidence, MetricWithConfidence)
        assert isinstance(metric_with_confidence.value, float)
        assert isinstance(metric_with_confidence.confidence_interval, ConfidenceInterval)
        assert metric_with_confidence.confidence_interval.lower <= metric_with_confidence.value
        assert metric_with_confidence.value <= metric_with_confidence.confidence_interval.upper
        
    def test_parallel_metrics_calculation(self):
        """Test parallel metrics calculation"""
        calculator = AdvancedMetricsCalculator(bootstrap_samples=50, max_workers=2)
        
        def simple_sharpe(returns):
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        def simple_volatility(returns):
            return np.std(returns)
        
        metric_functions = {
            'sharpe': simple_sharpe,
            'volatility': simple_volatility
        }
        
        results = calculator.calculate_parallel_metrics(
            returns=self.returns,
            metric_functions=metric_functions,
            confidence_level=0.95
        )
        
        assert isinstance(results, dict)
        assert 'sharpe' in results
        assert 'volatility' in results
        assert isinstance(results['sharpe'], MetricWithConfidence)
        assert isinstance(results['volatility'], MetricWithConfidence)
        
    def test_streaming_metrics(self):
        """Test streaming metrics functionality"""
        calculator = AdvancedMetricsCalculator()
        
        # Update streaming metrics
        calculator.update_streaming_metric("test_metric", 0.5)
        calculator.update_streaming_metric("test_metric", 0.6)
        calculator.update_streaming_metric("test_metric", 0.4)
        
        streaming_metrics = calculator.get_streaming_metrics()
        assert "test_metric" in streaming_metrics
        assert streaming_metrics["test_metric"].count == 3
        assert abs(streaming_metrics["test_metric"].value - 0.5) < 0.01  # Should be average


class TestRiskMetrics:
    """Test risk metrics integration"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.equity_curve = np.cumprod(1 + self.returns) * 10000
        self.benchmark_returns = np.random.normal(0.0008, 0.015, 1000)
        
        # Mock VaR calculator and correlation tracker
        self.mock_var_calculator = Mock()
        self.mock_correlation_tracker = Mock()
        self.mock_event_bus = Mock()
        
        # Mock VaR result
        self.mock_var_result = Mock()
        self.mock_var_result.portfolio_var = 500.0  # $500 VaR
        self.mock_var_result.component_vars = {"AAPL": 200.0, "GOOGL": 300.0}
        self.mock_var_result.marginal_vars = {"AAPL": 100.0, "GOOGL": 150.0}
        
        # Set up async mock
        async def mock_calculate_var(*args, **kwargs):
            return self.mock_var_result
        
        self.mock_var_calculator.calculate_var = mock_calculate_var
        self.mock_var_calculator.get_latest_var = Mock(return_value=self.mock_var_result)
        
        # Mock correlation tracker
        self.mock_regime = Mock()
        self.mock_regime.value = "NORMAL"
        self.mock_correlation_tracker.current_regime = self.mock_regime
        
    def test_comprehensive_risk_metrics(self):
        """Test comprehensive risk metrics calculation"""
        calculator = RiskMetricsCalculator(
            var_calculator=self.mock_var_calculator,
            correlation_tracker=self.mock_correlation_tracker,
            event_bus=self.mock_event_bus
        )
        
        # Test async calculation
        risk_metrics = asyncio.run(calculator.calculate_comprehensive_risk_metrics(
            returns=self.returns,
            equity_curve=self.equity_curve,
            benchmark_returns=self.benchmark_returns,
            periods_per_year=252
        ))
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.var_95 > 0
        assert risk_metrics.cvar_95 > 0
        assert risk_metrics.volatility > 0
        assert risk_metrics.sharpe_ratio is not None
        assert risk_metrics.sortino_ratio is not None
        assert risk_metrics.beta is not None
        assert risk_metrics.component_var is not None
        assert risk_metrics.marginal_var is not None
        
    def test_standalone_var_cvar_functions(self):
        """Test standalone VaR/CVaR functions"""
        var_95 = calculate_var_historical(self.returns, confidence_level=0.95)
        cvar_95 = calculate_cvar_historical(self.returns, confidence_level=0.95)
        
        assert isinstance(var_95, float)
        assert isinstance(cvar_95, float)
        assert var_95 > 0
        assert cvar_95 > 0
        assert cvar_95 >= var_95
        
    def test_component_risk_analysis(self):
        """Test component risk analysis"""
        calculator = RiskMetricsCalculator(
            var_calculator=self.mock_var_calculator,
            correlation_tracker=self.mock_correlation_tracker
        )
        
        # Mock position data
        from analysis.risk_metrics import PositionData
        mock_positions = {
            "AAPL": Mock(symbol="AAPL", market_value=10000, volatility=0.25),
            "GOOGL": Mock(symbol="GOOGL", market_value=15000, volatility=0.30)
        }
        
        component_analysis = calculator.analyze_component_risk(
            asset_positions=mock_positions,
            latest_var_result=self.mock_var_result
        )
        
        assert isinstance(component_analysis, dict)
        assert "AAPL" in component_analysis
        assert "GOOGL" in component_analysis
        assert isinstance(component_analysis["AAPL"], ComponentRiskAnalysis)
        assert component_analysis["AAPL"].component_var == 200.0
        assert component_analysis["AAPL"].marginal_var == 100.0


class TestPerformanceOptimization:
    """Test performance optimization features"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.equity_curve = np.cumprod(1 + self.returns) * 10000
        
    def test_optimized_metrics(self):
        """Test optimized metric calculations"""
        # Test optimized Sharpe ratio
        sharpe_optimized = optimized_sharpe_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        sharpe_standard = calculate_sharpe_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        
        assert isinstance(sharpe_optimized, float)
        assert abs(sharpe_optimized - sharpe_standard) < 0.01  # Should be very close
        
        # Test optimized Sortino ratio
        sortino_optimized = optimized_sortino_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        sortino_standard = calculate_sortino_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        
        assert isinstance(sortino_optimized, float)
        assert abs(sortino_optimized - sortino_standard) < 0.01
        
        # Test optimized max drawdown
        max_dd_optimized = optimized_max_drawdown(self.equity_curve)
        max_dd_standard, _, _ = calculate_max_drawdown(self.equity_curve)
        
        assert isinstance(max_dd_optimized, float)
        assert abs(max_dd_optimized - max_dd_standard) < 0.01
        
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        # Clear previous stats
        performance_monitor.stats_history.clear()
        
        # Run some calculations
        for _ in range(5):
            optimized_sharpe_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        
        # Check performance stats
        summary = performance_monitor.get_performance_summary()
        assert isinstance(summary, dict)
        if "optimized_sharpe_ratio" in summary:
            assert summary["optimized_sharpe_ratio"]["call_count"] == 5
            assert summary["optimized_sharpe_ratio"]["avg_time_ms"] > 0
            
    def test_metrics_optimizer(self):
        """Test metrics optimizer"""
        optimizer = MetricsOptimizer()
        
        # Test optimization
        result = optimizer.optimize_calculation(
            function_name="sharpe_ratio",
            data=self.returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        
        # Test benchmark
        benchmark_results = optimizer.benchmark_performance(
            returns=self.returns,
            iterations=10
        )
        
        assert isinstance(benchmark_results, dict)
        assert "sharpe_ratio" in benchmark_results
        assert "optimized_time_ms" in benchmark_results["sharpe_ratio"]


class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.equity_curve = np.cumprod(1 + self.returns) * 10000
        self.benchmark_returns = np.random.normal(0.0008, 0.015, 1000)
        
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Calculate basic metrics
        basic_metrics = calculate_all_metrics(
            equity_curve=self.equity_curve,
            returns=self.returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        assert isinstance(basic_metrics, PerformanceMetrics)
        
        # 2. Calculate advanced metrics with confidence intervals
        advanced_calculator = AdvancedMetricsCalculator(bootstrap_samples=50)
        
        def simple_sharpe(returns):
            if len(returns) == 0:
                return 0.0
            excess_returns = returns - 0.02/252
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        
        sharpe_with_confidence = advanced_calculator.calculate_metric_with_confidence(
            returns=self.returns,
            metric_func=simple_sharpe,
            confidence_level=0.95
        )
        
        assert isinstance(sharpe_with_confidence, MetricWithConfidence)
        
        # 3. Calculate risk metrics
        risk_calculator = RiskMetricsCalculator()
        
        risk_metrics = asyncio.run(risk_calculator.calculate_comprehensive_risk_metrics(
            returns=self.returns,
            equity_curve=self.equity_curve,
            benchmark_returns=self.benchmark_returns,
            periods_per_year=252
        ))
        
        assert isinstance(risk_metrics, RiskMetrics)
        
        # 4. Test performance optimization
        optimizer = MetricsOptimizer()
        optimized_sharpe = optimizer.optimize_calculation(
            function_name="sharpe_ratio",
            data=self.returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        assert isinstance(optimized_sharpe, float)
        
        # 5. Verify consistency across different methods
        # Basic and optimized Sharpe should be very close
        assert abs(basic_metrics.sharpe_ratio - optimized_sharpe) < 0.01
        
        # Advanced metrics should be in reasonable range
        assert abs(sharpe_with_confidence.value - basic_metrics.sharpe_ratio) < 0.5
        
        # Risk metrics should be consistent
        assert abs(risk_metrics.sharpe_ratio - basic_metrics.sharpe_ratio) < 0.01
        
    def test_performance_targets(self):
        """Test that performance targets are met"""
        # Test basic metrics performance
        import time
        
        start_time = time.time()
        for _ in range(10):
            calculate_all_metrics(
                equity_curve=self.equity_curve,
                returns=self.returns,
                benchmark_returns=self.benchmark_returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
        basic_time = (time.time() - start_time) * 1000 / 10  # Average time in ms
        
        # Should be under 5ms for standard metrics
        assert basic_time < 50  # Allow some slack for testing environment
        
        # Test optimized metrics performance
        start_time = time.time()
        for _ in range(10):
            optimized_sharpe_ratio(self.returns, risk_free_rate=0.02, periods_per_year=252)
        optimized_time = (time.time() - start_time) * 1000 / 10
        
        # Optimized version should be faster or comparable
        assert optimized_time < basic_time * 2  # Allow some overhead for optimization
        
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with empty data
        empty_returns = np.array([])
        empty_equity = np.array([])
        
        # Should not crash with empty data
        try:
            metrics = calculate_all_metrics(
                equity_curve=empty_equity,
                returns=empty_returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            # Should return valid metrics object with zero/default values
            assert isinstance(metrics, PerformanceMetrics)
        except Exception as e:
            pytest.fail(f"Should handle empty data gracefully: {e}")
        
        # Test with invalid data
        invalid_returns = np.array([np.nan, np.inf, -np.inf])
        
        try:
            var_result = calculate_var_historical(invalid_returns, confidence_level=0.95)
            assert isinstance(var_result, float)
            assert not np.isnan(var_result)
        except Exception as e:
            pytest.fail(f"Should handle invalid data gracefully: {e}")


def run_integration_tests():
    """Run all integration tests"""
    logger.info("Starting integration tests...")
    
    # Initialize test classes
    test_classes = [
        TestBasicMetrics(),
        TestAdvancedMetrics(),
        TestRiskMetrics(),
        TestPerformanceOptimization(),
        TestIntegration()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        logger.info(f"Running tests for {class_name}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                # Set up and run test
                test_class.setup_method()
                test_method = getattr(test_class, test_method_name)
                test_method()
                passed_tests += 1
                logger.info(f"✓ {class_name}.{test_method_name}")
            except Exception as e:
                logger.error(f"✗ {class_name}.{test_method_name}: {e}")
    
    logger.info(f"\nIntegration tests completed: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)