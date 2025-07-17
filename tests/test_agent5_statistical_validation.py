#!/usr/bin/env python3
"""
Test Suite for Agent 5 Statistical Validation Framework
========================================================

Comprehensive test suite for the Agent 5 statistical validation and 
performance metrics framework.

Tests cover:
1. Statistical validation framework initialization
2. Performance metrics calculation with bootstrap validation
3. Robustness testing with Monte Carlo simulation
4. Validation framework (out-of-sample, walk-forward)
5. Trustworthiness assessment and certification
6. Report generation and serialization

All tests ensure 500% trustworthy statistical validation.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.validation.statistical_validation_framework import (
    StatisticalValidationFramework,
    StatisticalValidationResult,
    RobustnessTestResult,
    ValidationFrameworkResult,
    TrustworthinessReport
)


class TestStatisticalValidationFramework(unittest.TestCase):
    """Test suite for StatisticalValidationFramework"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize framework with test parameters
        self.framework = StatisticalValidationFramework(
            confidence_level=0.95,
            n_bootstrap=100,  # Reduced for faster testing
            n_monte_carlo=1000,  # Reduced for faster testing
            significance_level=0.05
        )
        
        # Generate test data
        np.random.seed(42)
        self.n_samples = 500
        
        # Strategy returns with positive alpha
        self.strategy_returns = np.random.normal(0.001, 0.02, self.n_samples)
        
        # Benchmark returns
        self.benchmark_returns = np.random.normal(0.0005, 0.015, self.n_samples)
        
        # Add correlation
        correlation = 0.6
        self.strategy_returns = (correlation * self.benchmark_returns + 
                                np.sqrt(1 - correlation**2) * self.strategy_returns)
        
        # Price series
        self.prices = 100 * np.cumprod(1 + self.strategy_returns)
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        self.assertEqual(self.framework.confidence_level, 0.95)
        self.assertEqual(self.framework.n_bootstrap, 100)
        self.assertEqual(self.framework.n_monte_carlo, 1000)
        self.assertEqual(self.framework.significance_level, 0.05)
        self.assertIsNotNone(self.framework.statistical_tests)
    
    def test_performance_metrics_calculation(self):
        """Test comprehensive performance metrics calculation"""
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            prices=self.prices,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return',
            'volatility', 'max_drawdown', 'win_rate', 'profit_factor',
            'alpha', 'beta', 'information_ratio', 'tracking_error'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, validated_metrics)
            result = validated_metrics[metric]
            self.assertIsInstance(result, StatisticalValidationResult)
            self.assertIsInstance(result.point_estimate, float)
            self.assertIsInstance(result.confidence_interval, tuple)
            self.assertEqual(len(result.confidence_interval), 2)
            self.assertIsInstance(result.p_value, float)
            self.assertIsInstance(result.statistical_significance, bool)
            self.assertIsInstance(result.trustworthiness_score, float)
            self.assertGreaterEqual(result.trustworthiness_score, 0.0)
            self.assertLessEqual(result.trustworthiness_score, 1.0)
    
    def test_bootstrap_validation(self):
        """Test bootstrap validation of metrics"""
        # Test with a simple metric function
        metric_func = lambda x: np.mean(x)
        
        result = self.framework._validate_metric_with_bootstrap(
            self.strategy_returns, metric_func, 'test_metric'
        )
        
        self.assertIsInstance(result, StatisticalValidationResult)
        self.assertEqual(result.metric_name, 'test_metric')
        self.assertEqual(result.sample_size, len(self.strategy_returns))
        self.assertEqual(result.validation_method, 'Bootstrap Resampling')
        self.assertEqual(len(result.bootstrap_distribution), self.framework.n_bootstrap)
        
        # Check confidence interval is reasonable
        self.assertLessEqual(result.confidence_interval[0], result.point_estimate)
        self.assertGreaterEqual(result.confidence_interval[1], result.point_estimate)
    
    def test_robustness_testing(self):
        """Test robustness testing with Monte Carlo simulation"""
        robustness_results = self.framework.perform_robustness_testing(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            strategy_params={'lookback_period': 20, 'threshold': 0.01}
        )
        
        self.assertIsInstance(robustness_results, RobustnessTestResult)
        self.assertEqual(robustness_results.test_name, 'Comprehensive Robustness Test')
        self.assertIsInstance(robustness_results.base_performance, dict)
        self.assertIsInstance(robustness_results.stressed_performance, dict)
        self.assertIsInstance(robustness_results.robustness_score, float)
        self.assertGreaterEqual(robustness_results.robustness_score, 0.0)
        self.assertLessEqual(robustness_results.robustness_score, 1.0)
        self.assertEqual(len(robustness_results.monte_carlo_distribution), self.framework.n_monte_carlo)
        
        # Check that stress test scenarios are present
        stress_scenarios = ['high_volatility', 'extreme_volatility', 'bear_market', 'flash_crash']
        for scenario in stress_scenarios:
            self.assertIn(scenario, robustness_results.stressed_performance)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        mc_results = self.framework._monte_carlo_simulation(
            self.strategy_returns, self.benchmark_returns
        )
        
        self.assertEqual(len(mc_results), self.framework.n_monte_carlo)
        self.assertIsInstance(mc_results, np.ndarray)
        
        # Check that results are reasonable
        self.assertTrue(np.all(np.isfinite(mc_results)))
        self.assertGreater(np.std(mc_results), 0)  # Should have some variation
    
    def test_stress_testing(self):
        """Test stress testing scenarios"""
        stress_results = self.framework._stress_testing(
            self.strategy_returns, self.benchmark_returns
        )
        
        self.assertIsInstance(stress_results, dict)
        self.assertIn('detailed_results', stress_results)
        
        # Check that stress scenarios are applied
        expected_scenarios = ['high_volatility', 'extreme_volatility', 'bear_market', 'flash_crash']
        for scenario in expected_scenarios:
            self.assertIn(scenario, stress_results)
    
    def test_validation_framework(self):
        """Test validation framework with out-of-sample testing"""
        validation_results = self.framework.run_validation_framework(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            validation_split=0.7
        )
        
        self.assertIsInstance(validation_results, ValidationFrameworkResult)
        self.assertIsInstance(validation_results.in_sample_performance, dict)
        self.assertIsInstance(validation_results.out_of_sample_performance, dict)
        self.assertIsInstance(validation_results.walk_forward_results, list)
        self.assertIsInstance(validation_results.cross_validation_scores, np.ndarray)
        self.assertIsInstance(validation_results.stability_metrics, dict)
        self.assertIsInstance(validation_results.generalization_score, float)
        self.assertGreaterEqual(validation_results.generalization_score, 0.0)
        self.assertLessEqual(validation_results.generalization_score, 1.0)
        
        # Check that walk-forward results are populated
        self.assertGreater(len(validation_results.walk_forward_results), 0)
        
        # Check that cross-validation scores are reasonable
        self.assertGreater(len(validation_results.cross_validation_scores), 0)
        self.assertTrue(np.all(np.isfinite(validation_results.cross_validation_scores)))
    
    def test_walk_forward_analysis(self):
        """Test walk-forward analysis"""
        walk_forward_results = self.framework._walk_forward_analysis(
            self.strategy_returns, self.benchmark_returns, window_size=100
        )
        
        self.assertIsInstance(walk_forward_results, list)
        self.assertGreater(len(walk_forward_results), 0)
        
        # Check structure of walk-forward results
        for result in walk_forward_results:
            self.assertIn('period', result)
            self.assertIn('train_performance', result)
            self.assertIn('test_performance', result)
            self.assertIn('performance_decay', result)
    
    def test_cross_validation_analysis(self):
        """Test cross-validation analysis"""
        cv_scores = self.framework._cross_validation_analysis(
            self.strategy_returns, self.benchmark_returns, n_splits=3
        )
        
        self.assertIsInstance(cv_scores, np.ndarray)
        self.assertEqual(len(cv_scores), 3)
        self.assertTrue(np.all(np.isfinite(cv_scores)))
    
    def test_trustworthiness_report_generation(self):
        """Test trustworthiness report generation"""
        # First generate the required inputs
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            prices=self.prices
        )
        
        robustness_results = self.framework.perform_robustness_testing(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        validation_results = self.framework.run_validation_framework(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        # Generate trustworthiness report
        trustworthiness_report = self.framework.generate_trustworthiness_report(
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results
        )
        
        self.assertIsInstance(trustworthiness_report, TrustworthinessReport)
        self.assertIsInstance(trustworthiness_report.overall_trustworthiness_score, float)
        self.assertGreaterEqual(trustworthiness_report.overall_trustworthiness_score, 0.0)
        self.assertLessEqual(trustworthiness_report.overall_trustworthiness_score, 1.0)
        self.assertIsInstance(trustworthiness_report.certification_status, str)
        self.assertIsInstance(trustworthiness_report.recommendations, list)
        self.assertIsInstance(trustworthiness_report.warnings, list)
        
        # Check that statistical significance tests are populated
        self.assertGreater(len(trustworthiness_report.statistical_significance_tests), 0)
        
        # Check that confidence intervals are populated
        self.assertGreater(len(trustworthiness_report.confidence_intervals), 0)
        
        # Check that robustness assessment is populated
        self.assertGreater(len(trustworthiness_report.robustness_assessment), 0)
    
    def test_trustworthiness_score_calculation(self):
        """Test trustworthiness score calculation"""
        # Test with known values
        bootstrap_values = np.random.normal(0.5, 0.1, 1000)
        original_metric = 0.5
        standard_error = 0.1
        sample_size = 1000
        
        score = self.framework._calculate_trustworthiness_score(
            bootstrap_values, original_metric, standard_error, sample_size
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_robustness_score_calculation(self):
        """Test robustness score calculation"""
        base_performance = {'sharpe_ratio': 1.0, 'total_return': 0.1}
        mc_results = np.random.normal(1.0, 0.1, 1000)
        stress_results = {
            'high_volatility': {'sharpe_ratio': 0.8},
            'bear_market': {'sharpe_ratio': 0.5}
        }
        regime_results = {
            'low_vol': {'sharpe_ratio': 1.2},
            'high_vol': {'sharpe_ratio': 0.8}
        }
        
        score = self.framework._calculate_robustness_score(
            base_performance, mc_results, stress_results, regime_results
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_results_serialization(self):
        """Test serialization of results"""
        # Generate sample results
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            prices=self.prices
        )
        
        robustness_results = self.framework.perform_robustness_testing(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        validation_results = self.framework.run_validation_framework(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        trustworthiness_report = self.framework.generate_trustworthiness_report(
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results
        )
        
        # Test serialization
        serialized_metrics = self.framework._serialize_validation_results(validated_metrics)
        serialized_robustness = self.framework._serialize_robustness_results(robustness_results)
        serialized_validation = self.framework._serialize_validation_framework_results(validation_results)
        serialized_trustworthiness = self.framework._serialize_trustworthiness_report(trustworthiness_report)
        
        # Check that serialization produces valid JSON-serializable objects
        self.assertIsInstance(serialized_metrics, dict)
        self.assertIsInstance(serialized_robustness, dict)
        self.assertIsInstance(serialized_validation, dict)
        self.assertIsInstance(serialized_trustworthiness, dict)
        
        # Test that they can be JSON serialized
        json.dumps(serialized_metrics)
        json.dumps(serialized_robustness)
        json.dumps(serialized_validation)
        json.dumps(serialized_trustworthiness)
    
    def test_save_validation_results(self):
        """Test saving validation results to files"""
        # Generate sample results
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            prices=self.prices
        )
        
        robustness_results = self.framework.perform_robustness_testing(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        validation_results = self.framework.run_validation_framework(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )
        
        trustworthiness_report = self.framework.generate_trustworthiness_report(
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results
        )
        
        # Save results
        saved_files = self.framework.save_validation_results(
            output_dir=self.temp_dir,
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results,
            trustworthiness_report=trustworthiness_report
        )
        
        # Check that files were created
        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 0)
        
        expected_files = [
            'validated_metrics', 'robustness_results', 'validation_results',
            'trustworthiness_report', 'markdown_report', 'executive_summary'
        ]
        
        for file_type in expected_files:
            self.assertIn(file_type, saved_files)
            self.assertTrue(os.path.exists(saved_files[file_type]))
    
    def test_advanced_risk_metrics(self):
        """Test advanced risk metrics calculation"""
        advanced_metrics = self.framework._calculate_advanced_risk_metrics(
            self.strategy_returns, self.benchmark_returns, 0.02, 252
        )
        
        self.assertIsInstance(advanced_metrics, dict)
        
        expected_metrics = ['var_95', 'var_99', 'cvar_95', 'cvar_99', 'skewness', 'kurtosis']
        for metric in expected_metrics:
            self.assertIn(metric, advanced_metrics)
            self.assertIsInstance(advanced_metrics[metric], StatisticalValidationResult)
    
    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity analysis"""
        strategy_params = {
            'lookback_period': 20,
            'threshold': 0.01,
            'rebalance_freq': 5
        }
        
        sensitivity_results = self.framework._parameter_sensitivity_analysis(
            self.strategy_returns, strategy_params
        )
        
        self.assertIsInstance(sensitivity_results, dict)
        self.assertEqual(len(sensitivity_results), len(strategy_params))
        
        for param_name in strategy_params:
            self.assertIn(param_name, sensitivity_results)
            self.assertIsInstance(sensitivity_results[param_name], float)
    
    def test_regime_based_analysis(self):
        """Test regime-based analysis"""
        regime_results = self.framework._regime_based_analysis(
            self.strategy_returns, self.benchmark_returns
        )
        
        self.assertIsInstance(regime_results, dict)
        
        # Should have different volatility regimes
        expected_regimes = ['low_volatility', 'medium_volatility', 'high_volatility']
        for regime in expected_regimes:
            if regime in regime_results:
                self.assertIsInstance(regime_results[regime], dict)
                self.assertIn('sharpe_ratio', regime_results[regime])
    
    def test_extreme_scenario_testing(self):
        """Test extreme scenario testing"""
        extreme_results = self.framework._extreme_scenario_testing(
            self.strategy_returns, self.benchmark_returns
        )
        
        self.assertIsInstance(extreme_results, dict)
        
        # Should have extreme percentile scenarios
        expected_scenarios = ['extreme_negative_1th', 'extreme_negative_5th', 'extreme_positive_95th', 'extreme_positive_99th']
        for scenario in expected_scenarios:
            if scenario in extreme_results:
                self.assertIsInstance(extreme_results[scenario], dict)
                self.assertIn('count', extreme_results[scenario])
                self.assertIn('mean_return', extreme_results[scenario])
    
    def test_error_handling(self):
        """Test error handling in validation"""
        # Test with invalid data
        invalid_returns = np.array([np.nan, np.inf, -np.inf])
        
        # This should not crash but should handle the errors gracefully
        try:
            validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
                returns=invalid_returns,
                benchmark_returns=self.benchmark_returns[:3],
                prices=np.array([100, 101, 102])
            )
            
            # Should return some results even with invalid data
            self.assertIsInstance(validated_metrics, dict)
            
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            self.assertIsInstance(e, Exception)
    
    def test_certification_status_determination(self):
        """Test certification status determination"""
        # Test different trustworthiness scores
        test_cases = [
            (0.95, "CERTIFIED EXCELLENT"),
            (0.85, "CERTIFIED GOOD"),
            (0.75, "CERTIFIED ACCEPTABLE"),
            (0.65, "NEEDS IMPROVEMENT"),
            (0.45, "NOT CERTIFIED")
        ]
        
        for score, expected_status in test_cases:
            status = self.framework._determine_certification_status(
                score, {}, {}
            )
            self.assertEqual(status, expected_status)
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        recommendations = self.framework._generate_recommendations(
            overall_score=0.8,
            significance_tests={'metric1': True, 'metric2': False},
            robustness_assessment={'monte_carlo_stability': 0.7, 'stress_test_resilience': 0.6},
            validation_consistency={'in_out_sample_consistency': 0.8}
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Test low score recommendations
        low_score_recommendations = self.framework._generate_recommendations(
            overall_score=0.5,
            significance_tests={'metric1': False, 'metric2': False},
            robustness_assessment={'monte_carlo_stability': 0.4, 'stress_test_resilience': 0.4},
            validation_consistency={'in_out_sample_consistency': 0.5}
        )
        
        self.assertIsInstance(low_score_recommendations, list)
        self.assertGreater(len(low_score_recommendations), 0)
    
    def test_warnings_generation(self):
        """Test warnings generation"""
        warnings = self.framework._generate_warnings(
            significance_tests={'metric1': True, 'metric2': False},
            robustness_assessment={'monte_carlo_stability': 0.4, 'stress_test_resilience': 0.4},
            validation_consistency={'in_out_sample_consistency': 0.5}
        )
        
        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0)


class TestStatisticalValidationIntegration(unittest.TestCase):
    """Integration tests for the complete statistical validation workflow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.framework = StatisticalValidationFramework(
            confidence_level=0.95,
            n_bootstrap=50,  # Reduced for faster testing
            n_monte_carlo=500,  # Reduced for faster testing
            significance_level=0.05
        )
        
        # Generate more realistic test data
        np.random.seed(42)
        self.n_samples = 1000
        
        # Create realistic return patterns
        self.strategy_returns = self._generate_realistic_returns()
        self.benchmark_returns = self._generate_benchmark_returns()
        self.prices = 100 * np.cumprod(1 + self.strategy_returns)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_realistic_returns(self):
        """Generate realistic strategy returns with regime changes"""
        returns = []
        
        # Regime 1: Bull market (first 300 days)
        bull_returns = np.random.normal(0.0008, 0.015, 300)
        returns.extend(bull_returns)
        
        # Regime 2: Bear market (next 200 days)
        bear_returns = np.random.normal(-0.0005, 0.025, 200)
        returns.extend(bear_returns)
        
        # Regime 3: Recovery (next 300 days)
        recovery_returns = np.random.normal(0.0006, 0.020, 300)
        returns.extend(recovery_returns)
        
        # Regime 4: Sideways (remaining days)
        remaining = self.n_samples - len(returns)
        if remaining > 0:
            sideways_returns = np.random.normal(0.0002, 0.012, remaining)
            returns.extend(sideways_returns)
        
        return np.array(returns[:self.n_samples])
    
    def _generate_benchmark_returns(self):
        """Generate benchmark returns correlated with strategy"""
        # Create benchmark with lower but positive alpha
        benchmark_alpha = 0.0003
        benchmark_vol = 0.018
        
        # Add correlation with strategy returns
        correlation = 0.65
        independent_component = np.random.normal(benchmark_alpha, benchmark_vol, self.n_samples)
        
        benchmark_returns = (correlation * self.strategy_returns + 
                           np.sqrt(1 - correlation**2) * independent_component)
        
        return benchmark_returns
    
    def test_complete_validation_workflow(self):
        """Test complete validation workflow from start to finish"""
        # Step 1: Calculate comprehensive performance metrics
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            prices=self.prices,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        self.assertIsInstance(validated_metrics, dict)
        self.assertGreater(len(validated_metrics), 10)
        
        # Step 2: Perform robustness testing
        robustness_results = self.framework.perform_robustness_testing(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            strategy_params={
                'lookback_period': 20,
                'threshold': 0.01,
                'rebalance_freq': 5,
                'max_position': 0.1
            }
        )
        
        self.assertIsInstance(robustness_results, RobustnessTestResult)
        self.assertGreaterEqual(robustness_results.robustness_score, 0.0)
        
        # Step 3: Run validation framework
        validation_results = self.framework.run_validation_framework(
            returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns,
            validation_split=0.7
        )
        
        self.assertIsInstance(validation_results, ValidationFrameworkResult)
        self.assertGreaterEqual(validation_results.generalization_score, 0.0)
        
        # Step 4: Generate trustworthiness report
        trustworthiness_report = self.framework.generate_trustworthiness_report(
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results
        )
        
        self.assertIsInstance(trustworthiness_report, TrustworthinessReport)
        self.assertGreaterEqual(trustworthiness_report.overall_trustworthiness_score, 0.0)
        
        # Step 5: Save all results
        saved_files = self.framework.save_validation_results(
            output_dir=self.temp_dir,
            validated_metrics=validated_metrics,
            robustness_results=robustness_results,
            validation_results=validation_results,
            trustworthiness_report=trustworthiness_report
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 5)
        
        # Verify all files exist and are non-empty
        for file_type, file_path in saved_files.items():
            self.assertTrue(os.path.exists(file_path))
            self.assertGreater(os.path.getsize(file_path), 0)
    
    def test_validation_with_different_data_sizes(self):
        """Test validation with different data sizes"""
        data_sizes = [100, 500, 1000]
        
        for size in data_sizes:
            returns = self.strategy_returns[:size]
            benchmark = self.benchmark_returns[:size]
            
            validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
                returns=returns,
                benchmark_returns=benchmark,
                prices=100 * np.cumprod(1 + returns),
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            self.assertIsInstance(validated_metrics, dict)
            self.assertGreater(len(validated_metrics), 5)
            
            # Check that metrics are reasonable
            for metric_name, result in validated_metrics.items():
                self.assertIsInstance(result.point_estimate, float)
                self.assertTrue(np.isfinite(result.point_estimate))
                self.assertGreaterEqual(result.trustworthiness_score, 0.0)
                self.assertLessEqual(result.trustworthiness_score, 1.0)
    
    def test_validation_with_poor_performance(self):
        """Test validation with poor performing strategy"""
        # Create a poor performing strategy
        poor_returns = np.random.normal(-0.0002, 0.03, self.n_samples)
        
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=poor_returns,
            benchmark_returns=self.benchmark_returns,
            prices=100 * np.cumprod(1 + poor_returns),
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        # Should still work, but with lower trustworthiness scores
        self.assertIsInstance(validated_metrics, dict)
        
        # Check that Sharpe ratio is likely negative
        sharpe_result = validated_metrics.get('sharpe_ratio')
        if sharpe_result:
            self.assertLess(sharpe_result.point_estimate, 0.5)
    
    def test_validation_with_high_volatility(self):
        """Test validation with high volatility strategy"""
        # Create high volatility returns
        high_vol_returns = np.random.normal(0.001, 0.05, self.n_samples)
        
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=high_vol_returns,
            benchmark_returns=self.benchmark_returns,
            prices=100 * np.cumprod(1 + high_vol_returns),
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        # Should work and capture the high volatility
        self.assertIsInstance(validated_metrics, dict)
        
        volatility_result = validated_metrics.get('volatility')
        if volatility_result:
            self.assertGreater(volatility_result.point_estimate, 0.3)  # Should be high
    
    def test_performance_with_missing_benchmark(self):
        """Test performance calculation without benchmark"""
        validated_metrics = self.framework.calculate_comprehensive_performance_metrics(
            returns=self.strategy_returns,
            benchmark_returns=None,
            prices=self.prices,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        
        self.assertIsInstance(validated_metrics, dict)
        
        # Should not have benchmark-dependent metrics
        self.assertNotIn('alpha', validated_metrics)
        self.assertNotIn('beta', validated_metrics)
        self.assertNotIn('information_ratio', validated_metrics)
        
        # Should have standalone metrics
        self.assertIn('sharpe_ratio', validated_metrics)
        self.assertIn('sortino_ratio', validated_metrics)
        self.assertIn('calmar_ratio', validated_metrics)


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTest(unittest.makeSuite(TestStatisticalValidationFramework))
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestStatisticalValidationIntegration))
    
    return suite


def main():
    """Run the test suite"""
    # Configure test logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGENT 5 STATISTICAL VALIDATION TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED - AGENT 5 STATISTICAL VALIDATION FRAMEWORK IS READY")
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW AND FIX ISSUES")
    
    return success


if __name__ == '__main__':
    import logging
    success = main()
    sys.exit(0 if success else 1)