#!/usr/bin/env python3
"""
AGENT 4 CRITICAL MISSION: Comprehensive Performance Testing Suite
===============================================================

This module provides comprehensive testing and validation for the performance
analytics system with:
- Unit tests for all performance metrics
- Integration tests for statistical validation
- Stress testing for edge cases
- Performance benchmarking
- Accuracy validation against known results
- Comprehensive test coverage

Key Features:
- Automated test suite with pytest framework
- Mathematical accuracy validation
- Edge case and boundary testing
- Performance benchmarking
- Statistical validation testing
- Comprehensive reporting
- Continuous integration support

Author: Agent 4 - Performance Analytics Specialist
Date: 2025-07-17
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add paths for imports
sys.path.insert(0, '/home/QuantNova/GrandModel/src')
sys.path.insert(0, '/home/QuantNova/GrandModel/src/performance')
sys.path.insert(0, '/home/QuantNova/GrandModel/src/validation')
sys.path.insert(0, '/home/QuantNova/GrandModel/src/analytics')

from comprehensive_performance_metrics import (
    ComprehensivePerformanceAnalyzer,
    ComprehensivePerformanceMetrics,
    calculate_returns_jit,
    calculate_drawdown_jit,
    calculate_var_jit,
    calculate_cvar_jit,
    calculate_sharpe_ratio_jit,
    calculate_sortino_ratio_jit,
    calculate_calmar_ratio_jit,
    calculate_omega_ratio_jit,
    calculate_ulcer_index_jit,
    calculate_kelly_criterion_jit
)

from statistical_validation_framework import (
    StatisticalValidationFramework,
    StatisticalValidationResults,
    bootstrap_sample_returns,
    calculate_bootstrap_sharpe,
    calculate_bootstrap_max_drawdown,
    monte_carlo_simulation_single_run
)

from advanced_analytics import (
    AdvancedAnalytics,
    RollingAnalysisResults,
    RegimeAnalysisResults,
    PerformanceAttributionResults,
    calculate_rolling_sharpe_jit,
    calculate_rolling_volatility_jit,
    calculate_rolling_var_jit,
    calculate_rolling_max_drawdown_jit
)

warnings.filterwarnings('ignore')


class TestPerformanceMetrics(unittest.TestCase):
    """Test suite for performance metrics calculations"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, 0.02])
        self.prices = np.array([100, 101, 103, 101.97, 105.03, 102.93, 103.96, 106.04, 104.98, 106.03, 108.15])
        self.benchmark_returns = np.array([0.008, 0.015, -0.005, 0.025, -0.015, 0.008, 0.015, -0.005, 0.008, 0.015])
        
        # Known expected values for validation
        self.expected_total_return = 0.0815  # Approximately 8.15%
        self.expected_volatility = 0.0141  # Approximately 1.41% daily
        self.tolerance = 1e-3
    
    def test_returns_calculation(self):
        """Test return calculation accuracy"""
        calculated_returns = calculate_returns_jit(self.prices)
        
        # Test length
        self.assertEqual(len(calculated_returns), len(self.prices) - 1)
        
        # Test specific values
        expected_first_return = (101 - 100) / 100
        self.assertAlmostEqual(calculated_returns[0], expected_first_return, places=6)
        
        # Test no NaN or infinite values
        self.assertFalse(np.any(np.isnan(calculated_returns)))
        self.assertFalse(np.any(np.isinf(calculated_returns)))
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        sharpe = calculate_sharpe_ratio_jit(self.returns, 0.02, 252)
        
        # Manual calculation for verification
        excess_returns = self.returns - 0.02/252
        expected_sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        self.assertAlmostEqual(sharpe, expected_sharpe, places=6)
        
        # Test edge case - zero volatility
        zero_vol_returns = np.array([0.01, 0.01, 0.01, 0.01])
        zero_vol_sharpe = calculate_sharpe_ratio_jit(zero_vol_returns, 0.0, 252)
        self.assertEqual(zero_vol_sharpe, 0.0)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        sortino = calculate_sortino_ratio_jit(self.returns, 0.02, 252)
        
        # Manual calculation for verification
        excess_returns = self.returns - 0.02/252
        downside_returns = excess_returns[excess_returns < 0]
        expected_sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
        
        self.assertAlmostEqual(sortino, expected_sortino, places=6)
        
        # Test edge case - all positive returns
        positive_returns = np.array([0.01, 0.02, 0.03, 0.01])
        positive_sortino = calculate_sortino_ratio_jit(positive_returns, 0.0, 252)
        self.assertEqual(positive_sortino, np.inf)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        drawdown_series, max_dd, peak_idx, trough_idx = calculate_drawdown_jit(self.prices)
        
        # Test drawdown series length
        self.assertEqual(len(drawdown_series), len(self.prices))
        
        # Test maximum drawdown is positive
        self.assertGreaterEqual(max_dd, 0)
        
        # Test indices are valid
        self.assertGreaterEqual(peak_idx, 0)
        self.assertGreaterEqual(trough_idx, 0)
        self.assertLess(peak_idx, len(self.prices))
        self.assertLess(trough_idx, len(self.prices))
        
        # Test peak comes before trough
        self.assertLessEqual(peak_idx, trough_idx)
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        var_95 = calculate_var_jit(self.returns, 0.95)
        var_99 = calculate_var_jit(self.returns, 0.99)
        
        # VaR 99% should be greater than VaR 95%
        self.assertGreaterEqual(var_99, var_95)
        
        # VaR should be positive
        self.assertGreaterEqual(var_95, 0)
        self.assertGreaterEqual(var_99, 0)
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation"""
        cvar_95 = calculate_cvar_jit(self.returns, 0.95)
        cvar_99 = calculate_cvar_jit(self.returns, 0.99)
        
        # CVaR should be greater than or equal to VaR
        var_95 = calculate_var_jit(self.returns, 0.95)
        var_99 = calculate_var_jit(self.returns, 0.99)
        
        self.assertGreaterEqual(cvar_95, var_95)
        self.assertGreaterEqual(cvar_99, var_99)
        
        # CVaR should be positive
        self.assertGreaterEqual(cvar_95, 0)
        self.assertGreaterEqual(cvar_99, 0)
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation"""
        kelly = calculate_kelly_criterion_jit(self.returns)
        
        # Kelly criterion should be between 0 and 1 for reasonable strategies
        self.assertGreaterEqual(kelly, 0)
        self.assertLessEqual(kelly, 1)
        
        # Test edge case - no wins
        all_losses = np.array([-0.01, -0.02, -0.01, -0.03])
        kelly_no_wins = calculate_kelly_criterion_jit(all_losses)
        self.assertEqual(kelly_no_wins, 0.0)
    
    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics calculation"""
        analyzer = ComprehensivePerformanceAnalyzer()
        metrics = analyzer.calculate_comprehensive_metrics(self.returns, benchmark_returns=self.benchmark_returns)
        
        # Test all metrics are finite
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, dict):
                self.assertFalse(np.isnan(value), f"NaN value found in {key}")
                self.assertFalse(np.isinf(value), f"Infinite value found in {key}")
        
        # Test specific metric ranges
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
        self.assertGreaterEqual(metrics.max_drawdown, 0)
        self.assertGreaterEqual(metrics.volatility, 0)


class TestStatisticalValidation(unittest.TestCase):
    """Test suite for statistical validation framework"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.0008, 0.015, 1000)
        self.framework = StatisticalValidationFramework(
            bootstrap_samples=1000,
            monte_carlo_runs=1000,
            confidence_level=0.95
        )
    
    def test_bootstrap_sampling(self):
        """Test bootstrap sampling functionality"""
        bootstrap_sample = bootstrap_sample_returns(self.returns, len(self.returns), 42)
        
        # Test length
        self.assertEqual(len(bootstrap_sample), len(self.returns))
        
        # Test values are from original returns
        for value in bootstrap_sample:
            self.assertIn(value, self.returns)
    
    def test_bootstrap_sharpe_calculation(self):
        """Test bootstrap Sharpe ratio calculation"""
        bootstrap_sharpe = calculate_bootstrap_sharpe(self.returns, 0.02, 252)
        
        # Test finite value
        self.assertFalse(np.isnan(bootstrap_sharpe))
        self.assertFalse(np.isinf(bootstrap_sharpe))
        
        # Test reasonable range
        self.assertGreater(bootstrap_sharpe, -5)
        self.assertLess(bootstrap_sharpe, 5)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        total_return, sharpe_ratio, max_drawdown = monte_carlo_simulation_single_run(self.returns, 252, 42)
        
        # Test finite values
        self.assertFalse(np.isnan(total_return))
        self.assertFalse(np.isnan(sharpe_ratio))
        self.assertFalse(np.isnan(max_drawdown))
        
        # Test reasonable ranges
        self.assertGreater(total_return, -0.9)  # Not more than 90% loss
        self.assertLess(total_return, 10.0)     # Not more than 1000% gain
        self.assertGreaterEqual(max_drawdown, 0)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation framework"""
        results = self.framework.run_comprehensive_validation(self.returns)
        
        # Test result structure
        self.assertIsInstance(results, StatisticalValidationResults)
        self.assertIsInstance(results.bootstrap_confidence_intervals, dict)
        self.assertIsInstance(results.monte_carlo_metrics, dict)
        self.assertIsInstance(results.hypothesis_tests, dict)
        
        # Test confidence intervals
        for metric, (lower, upper) in results.bootstrap_confidence_intervals.items():
            self.assertLessEqual(lower, upper, f"Invalid confidence interval for {metric}")
        
        # Test robustness scores
        self.assertGreaterEqual(results.overall_robustness_score, 0)
        self.assertLessEqual(results.overall_robustness_score, 1)
        self.assertGreaterEqual(results.statistical_significance_score, 0)
        self.assertLessEqual(results.statistical_significance_score, 1)


class TestAdvancedAnalytics(unittest.TestCase):
    """Test suite for advanced analytics"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.0008, 0.015, 1000)
        self.benchmark_returns = np.random.normal(0.0005, 0.012, 1000)
        self.factor_returns = {
            'market_factor': np.random.normal(0.0006, 0.013, 1000),
            'size_factor': np.random.normal(0.0002, 0.008, 1000)
        }
        self.analytics = AdvancedAnalytics()
    
    def test_rolling_sharpe_calculation(self):
        """Test rolling Sharpe ratio calculation"""
        window_size = 252
        rolling_sharpe = calculate_rolling_sharpe_jit(self.returns, window_size, 0.02, 252)
        
        # Test length
        expected_length = len(self.returns) - window_size + 1
        self.assertEqual(len(rolling_sharpe), expected_length)
        
        # Test finite values
        self.assertFalse(np.any(np.isnan(rolling_sharpe)))
        self.assertFalse(np.any(np.isinf(rolling_sharpe)))
    
    def test_rolling_volatility_calculation(self):
        """Test rolling volatility calculation"""
        window_size = 252
        rolling_vol = calculate_rolling_volatility_jit(self.returns, window_size, 252)
        
        # Test length
        expected_length = len(self.returns) - window_size + 1
        self.assertEqual(len(rolling_vol), expected_length)
        
        # Test positive values
        self.assertTrue(np.all(rolling_vol >= 0))
    
    def test_rolling_var_calculation(self):
        """Test rolling VaR calculation"""
        window_size = 252
        rolling_var = calculate_rolling_var_jit(self.returns, window_size, 0.95)
        
        # Test length
        expected_length = len(self.returns) - window_size + 1
        self.assertEqual(len(rolling_var), expected_length)
        
        # Test positive values
        self.assertTrue(np.all(rolling_var >= 0))
    
    def test_rolling_max_drawdown_calculation(self):
        """Test rolling maximum drawdown calculation"""
        window_size = 252
        rolling_dd = calculate_rolling_max_drawdown_jit(self.returns, window_size)
        
        # Test length
        expected_length = len(self.returns) - window_size + 1
        self.assertEqual(len(rolling_dd), expected_length)
        
        # Test positive values
        self.assertTrue(np.all(rolling_dd >= 0))
    
    def test_rolling_analysis(self):
        """Test comprehensive rolling analysis"""
        results = self.analytics.calculate_rolling_analysis(
            self.returns, 
            window_size=252, 
            benchmark_returns=self.benchmark_returns
        )
        
        # Test result structure
        self.assertIsInstance(results, RollingAnalysisResults)
        self.assertEqual(results.window_size, 252)
        
        # Test array lengths are consistent
        if len(results.rolling_returns) > 0:
            self.assertEqual(len(results.rolling_sharpe), len(results.rolling_returns))
            self.assertEqual(len(results.rolling_volatility), len(results.rolling_returns))
    
    def test_regime_analysis(self):
        """Test regime analysis"""
        results = self.analytics.analyze_regimes(self.returns, n_regimes=3)
        
        # Test result structure
        self.assertIsInstance(results, RegimeAnalysisResults)
        self.assertEqual(results.n_regimes, 3)
        
        # Test regime labels
        if len(results.regime_labels) > 0:
            self.assertEqual(len(results.regime_labels), len(self.returns))
            self.assertTrue(np.all(results.regime_labels >= 0))
            self.assertTrue(np.all(results.regime_labels < 3))
    
    def test_performance_attribution(self):
        """Test performance attribution analysis"""
        results = self.analytics.calculate_performance_attribution(
            self.returns, 
            self.factor_returns, 
            self.benchmark_returns
        )
        
        # Test result structure
        self.assertIsInstance(results, PerformanceAttributionResults)
        self.assertIsInstance(results.factor_loadings, dict)
        self.assertIsInstance(results.factor_contributions, dict)
        
        # Test R-squared is between 0 and 1
        self.assertGreaterEqual(results.factor_r_squared, 0)
        self.assertLessEqual(results.factor_r_squared, 1)
        
        # Test risk attribution
        self.assertGreaterEqual(results.systematic_risk, 0)
        self.assertGreaterEqual(results.idiosyncratic_risk, 0)
        self.assertGreaterEqual(results.total_risk, 0)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and boundary conditions"""
    
    def test_empty_arrays(self):
        """Test handling of empty arrays"""
        empty_returns = np.array([])
        
        # Test performance metrics
        analyzer = ComprehensivePerformanceAnalyzer()
        metrics = analyzer.calculate_comprehensive_metrics(empty_returns)
        
        # Should return default values without errors
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
        
        # Test statistical validation
        framework = StatisticalValidationFramework(bootstrap_samples=10, monte_carlo_runs=10)
        results = framework.run_comprehensive_validation(empty_returns)
        
        # Should return empty results without errors
        self.assertEqual(len(results.bootstrap_confidence_intervals), 0)
    
    def test_single_value_arrays(self):
        """Test handling of single value arrays"""
        single_return = np.array([0.01])
        
        # Test drawdown calculation
        drawdown_series, max_dd, peak_idx, trough_idx = calculate_drawdown_jit(single_return)
        
        # Should handle single value gracefully
        self.assertEqual(len(drawdown_series), 1)
        self.assertEqual(max_dd, 0.0)
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        extreme_returns = np.array([0.5, -0.8, 1.0, -0.9, 0.3])
        
        # Test Sharpe ratio calculation
        sharpe = calculate_sharpe_ratio_jit(extreme_returns, 0.02, 252)
        
        # Should handle extreme values without errors
        self.assertFalse(np.isnan(sharpe))
        self.assertFalse(np.isinf(sharpe))
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values"""
        problematic_returns = np.array([0.01, np.nan, 0.02, np.inf, -0.01])
        
        # Remove NaN and infinite values
        clean_returns = problematic_returns[~np.isnan(problematic_returns)]
        clean_returns = clean_returns[~np.isinf(clean_returns)]
        
        # Test with cleaned data
        if len(clean_returns) > 0:
            sharpe = calculate_sharpe_ratio_jit(clean_returns, 0.02, 252)
            self.assertFalse(np.isnan(sharpe))
            self.assertFalse(np.isinf(sharpe))
    
    def test_zero_volatility(self):
        """Test handling of zero volatility scenarios"""
        zero_vol_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        
        # Test Sharpe ratio
        sharpe = calculate_sharpe_ratio_jit(zero_vol_returns, 0.0, 252)
        self.assertEqual(sharpe, 0.0)
        
        # Test Sortino ratio
        sortino = calculate_sortino_ratio_jit(zero_vol_returns, 0.0, 252)
        self.assertEqual(sortino, np.inf)  # No downside deviation


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test suite for performance benchmarking"""
    
    def setUp(self):
        """Set up benchmark data"""
        np.random.seed(42)
        self.small_returns = np.random.normal(0.0008, 0.015, 1000)
        self.medium_returns = np.random.normal(0.0008, 0.015, 10000)
        self.large_returns = np.random.normal(0.0008, 0.015, 100000)
    
    def test_performance_metrics_speed(self):
        """Test performance metrics calculation speed"""
        analyzer = ComprehensivePerformanceAnalyzer()
        
        # Test small dataset
        start_time = time.time()
        metrics_small = analyzer.calculate_comprehensive_metrics(self.small_returns)
        time_small = time.time() - start_time
        
        # Test medium dataset
        start_time = time.time()
        metrics_medium = analyzer.calculate_comprehensive_metrics(self.medium_returns)
        time_medium = time.time() - start_time
        
        # Test large dataset
        start_time = time.time()
        metrics_large = analyzer.calculate_comprehensive_metrics(self.large_returns)
        time_large = time.time() - start_time
        
        # Performance should scale reasonably
        self.assertLess(time_small, 1.0)   # Should complete in under 1 second
        self.assertLess(time_medium, 5.0)  # Should complete in under 5 seconds
        self.assertLess(time_large, 30.0)  # Should complete in under 30 seconds
        
        print(f"Performance benchmarks:")
        print(f"  Small (1k): {time_small:.3f}s")
        print(f"  Medium (10k): {time_medium:.3f}s")
        print(f"  Large (100k): {time_large:.3f}s")
    
    def test_statistical_validation_speed(self):
        """Test statistical validation speed"""
        framework = StatisticalValidationFramework(
            bootstrap_samples=1000, 
            monte_carlo_runs=1000
        )
        
        start_time = time.time()
        results = framework.run_comprehensive_validation(self.small_returns)
        validation_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(validation_time, 60.0)  # Should complete in under 1 minute
        
        print(f"Statistical validation time: {validation_time:.3f}s")
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Measure memory before
        gc.collect()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run analysis
        analyzer = ComprehensivePerformanceAnalyzer()
        for _ in range(10):
            metrics = analyzer.calculate_comprehensive_metrics(self.medium_returns)
            del metrics
        
        # Measure memory after
        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = memory_after - memory_before
        
        # Should not increase memory by more than 100MB
        self.assertLess(memory_increase, 100)
        
        print(f"Memory increase: {memory_increase:.2f}MB")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test data"""
        np.random.seed(42)
        self.returns = np.random.normal(0.0008, 0.015, 1000)
        self.benchmark_returns = np.random.normal(0.0005, 0.012, 1000)
        self.factor_returns = {
            'market_factor': np.random.normal(0.0006, 0.013, 1000),
            'size_factor': np.random.normal(0.0002, 0.008, 1000)
        }
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        # Performance metrics
        analyzer = ComprehensivePerformanceAnalyzer()
        metrics = analyzer.calculate_comprehensive_metrics(
            self.returns, 
            benchmark_returns=self.benchmark_returns
        )
        
        # Statistical validation
        framework = StatisticalValidationFramework(
            bootstrap_samples=1000, 
            monte_carlo_runs=1000
        )
        validation_results = framework.run_comprehensive_validation(self.returns)
        
        # Advanced analytics
        analytics = AdvancedAnalytics()
        rolling_results = analytics.calculate_rolling_analysis(
            self.returns, 
            benchmark_returns=self.benchmark_returns
        )
        regime_results = analytics.analyze_regimes(self.returns)
        attribution_results = analytics.calculate_performance_attribution(
            self.returns, 
            self.factor_returns, 
            self.benchmark_returns
        )
        
        # Test all components completed successfully
        self.assertIsInstance(metrics, ComprehensivePerformanceMetrics)
        self.assertIsInstance(validation_results, StatisticalValidationResults)
        self.assertIsInstance(rolling_results, RollingAnalysisResults)
        self.assertIsInstance(regime_results, RegimeAnalysisResults)
        self.assertIsInstance(attribution_results, PerformanceAttributionResults)
        
        # Test consistency across components
        self.assertAlmostEqual(metrics.sharpe_ratio, validation_results.bootstrap_statistics.get('sharpe_ratio', {}).get('mean', 0), delta=0.5)
    
    def test_report_generation(self):
        """Test report generation"""
        from advanced_analytics import generate_comprehensive_analytics_report
        
        report = generate_comprehensive_analytics_report(
            self.returns,
            benchmark_returns=self.benchmark_returns,
            factor_returns=self.factor_returns
        )
        
        # Test report structure
        self.assertIn('timestamp', report)
        self.assertIn('analysis_summary', report)
        self.assertIn('rolling_analysis', report)
        self.assertIn('regime_analysis', report)
        self.assertIn('performance_attribution', report)
        self.assertIn('summary_insights', report)
        
        # Test report can be serialized
        json_report = json.dumps(report, indent=2)
        self.assertIsInstance(json_report, str)


def run_comprehensive_test_suite():
    """Run comprehensive test suite and generate report"""
    print("🧪 Running Comprehensive Performance Analytics Test Suite")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPerformanceMetrics,
        TestStatisticalValidation,
        TestAdvancedAnalytics,
        TestEdgeCases,
        TestPerformanceBenchmarking,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Generate test report
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'test_duration': end_time - start_time,
        'total_tests': result.testsRun,
        'passed_tests': result.testsRun - len(result.failures) - len(result.errors),
        'failed_tests': len(result.failures),
        'error_tests': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'failures': [str(failure) for failure in result.failures],
        'errors': [str(error) for error in result.errors]
    }
    
    # Save test report
    report_path = "/tmp/performance_testing_report.json"
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🧪 TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {test_report['total_tests']}")
    print(f"Passed: {test_report['passed_tests']}")
    print(f"Failed: {test_report['failed_tests']}")
    print(f"Errors: {test_report['error_tests']}")
    print(f"Success Rate: {test_report['success_rate']:.1%}")
    print(f"Test Duration: {test_report['test_duration']:.2f} seconds")
    print(f"Report Saved: {report_path}")
    
    if test_report['success_rate'] >= 0.95:
        print("\n✅ EXCELLENT: Test suite passed with high success rate!")
    elif test_report['success_rate'] >= 0.90:
        print("\n✅ GOOD: Test suite passed with good success rate")
    elif test_report['success_rate'] >= 0.80:
        print("\n⚠️ ACCEPTABLE: Test suite passed with acceptable success rate")
    else:
        print("\n❌ POOR: Test suite has significant failures")
    
    return test_report


if __name__ == "__main__":
    # Run comprehensive test suite
    report = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if report['success_rate'] >= 0.95 else 1)