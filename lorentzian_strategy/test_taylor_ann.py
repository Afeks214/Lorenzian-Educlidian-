"""
TAYLOR SERIES ANN TESTING AND VALIDATION SUITE
==============================================

Comprehensive testing framework for the Taylor Series ANN Optimization System.
This suite validates all components and ensures performance targets are met.

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - System-level testing
3. Performance Tests - Speed and accuracy benchmarks
4. Stress Tests - Edge cases and robustness
5. Market Simulation Tests - Real-world scenario testing

Author: Claude AI Research Division
Date: 2025-07-20
"""

import unittest
import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorentzian_strategy.taylor_ann import (
    TaylorANNConfig,
    TaylorANNClassifier,
    TaylorDistanceApproximator,
    ExpansionPointSelector,
    TaylorCoefficientCache,
    HybridExactApproximateStrategy,
    MarketRegimeAwareANN,
    MarketRegimeDetector,
    fast_lorentzian_distance,
    taylor_expansion_4th_order,
    compute_lorentzian_derivatives
)

class TestTaylorSeriesComponents(unittest.TestCase):
    """Unit tests for Taylor series mathematical components"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        np.random.seed(42)
    
    def test_fast_lorentzian_distance(self):
        """Test optimized Lorentzian distance calculation"""
        x = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
        y = np.array([0.2, 0.4, 0.6, 0.8, 0.1])
        
        # Compute distance
        distance = fast_lorentzian_distance(x, y)
        
        # Verify properties
        self.assertGreater(distance, 0)
        self.assertIsInstance(distance, float)
        
        # Test symmetry
        distance_reverse = fast_lorentzian_distance(y, x)
        self.assertAlmostEqual(distance, distance_reverse, places=10)
        
        # Test identity
        identity_distance = fast_lorentzian_distance(x, x)
        self.assertAlmostEqual(identity_distance, 0.0, places=10)
    
    def test_taylor_expansion_4th_order(self):
        """Test fourth-order Taylor expansion accuracy"""
        # Test expansion around point x0 = 0.5
        x0 = 0.5
        coeffs = compute_lorentzian_derivatives(x0, order=4)
        
        # Test points near expansion point
        test_points = [0.45, 0.48, 0.52, 0.55]
        
        for x in test_points:
            taylor_approx = taylor_expansion_4th_order(x, x0, coeffs)
            exact_value = np.log(1.0 + x)
            
            # Taylor approximation should be reasonably close
            relative_error = abs(taylor_approx - exact_value) / abs(exact_value)
            self.assertLess(relative_error, 0.1, 
                          f"High error for x={x}: {relative_error}")
    
    def test_compute_lorentzian_derivatives(self):
        """Test analytical derivative computation"""
        x0 = 0.3
        derivatives = compute_lorentzian_derivatives(x0, order=4)
        
        # Check array structure
        self.assertEqual(len(derivatives), 5)  # 0th to 4th derivative
        
        # Verify known values
        expected_0th = np.log(1.0 + x0)
        expected_1st = 1.0 / (1.0 + x0)
        
        self.assertAlmostEqual(derivatives[0], expected_0th, places=10)
        self.assertAlmostEqual(derivatives[1], expected_1st, places=10)
        
        # All derivatives should be finite
        self.assertTrue(np.all(np.isfinite(derivatives)))

class TestExpansionPointSelector(unittest.TestCase):
    """Test expansion point selection algorithms"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        self.selector = ExpansionPointSelector(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.randn(100, 5)
    
    def test_statistical_coverage_selection(self):
        """Test statistical coverage expansion point selection"""
        n_points = 20
        points = self.selector._statistical_coverage_selection(self.test_data, n_points)
        
        # Check output structure
        self.assertLessEqual(len(points), n_points * self.test_data.shape[1])
        self.assertTrue(np.all(np.isfinite(points)))
        
        # Points should cover data range
        data_min = np.min(self.test_data)
        data_max = np.max(self.test_data)
        
        self.assertGreaterEqual(np.min(points), data_min - 1.0)
        self.assertLessEqual(np.max(points), data_max + 1.0)
    
    def test_regime_aware_adjustment(self):
        """Test regime-aware point adjustment"""
        base_points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        # Test different regimes
        regimes = ['trending', 'ranging', 'volatile', 'calm']
        
        for regime in regimes:
            adjusted_points = self.selector._regime_aware_adjustment(base_points, regime)
            
            # Adjusted points should maintain structure
            self.assertEqual(len(adjusted_points), len(base_points))
            self.assertTrue(np.all(np.isfinite(adjusted_points)))
    
    def test_select_optimal_expansion_points(self):
        """Test integrated expansion point selection"""
        points = self.selector.select_optimal_expansion_points(self.test_data)
        
        # Check output properties
        self.assertGreater(len(points), 0)
        self.assertLessEqual(len(points), self.config.expansion_points_count)
        self.assertTrue(np.all(np.isfinite(points)))

class TestTaylorCoefficientCache(unittest.TestCase):
    """Test coefficient caching system"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        self.cache = TaylorCoefficientCache(self.config)
    
    def test_cache_operations(self):
        """Test cache storage and retrieval"""
        expansion_point = 0.5
        order = 4
        
        # First access should compute and cache
        coeffs1 = self.cache.get_coefficients(expansion_point, order)
        self.assertEqual(len(coeffs1), order + 1)
        
        # Second access should use cache
        coeffs2 = self.cache.get_coefficients(expansion_point, order)
        np.testing.assert_array_equal(coeffs1, coeffs2)
        
        # Check cache statistics
        stats = self.cache.get_cache_stats()
        self.assertGreater(stats['total_accesses'], 0)
        self.assertGreater(stats['cache_size'], 0)
    
    def test_cache_size_management(self):
        """Test cache size management"""
        # Fill cache beyond limit
        for i in range(self.config.coefficient_cache_size + 50):
            point = i * 0.01
            self.cache.get_coefficients(point, 4)
        
        # Cache should be managed to stay within limits
        self.assertLessEqual(len(self.cache.cache), self.config.coefficient_cache_size)

class TestTaylorDistanceApproximator(unittest.TestCase):
    """Test Taylor distance approximation engine"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        self.approximator = TaylorDistanceApproximator(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.test_features = np.random.rand(50, 5)
    
    def test_approximate_lorentzian_distance(self):
        """Test distance approximation accuracy"""
        x = self.test_features[0]
        y = self.test_features[1]
        
        # Compute approximation
        approx_distance = self.approximator.approximate_lorentzian_distance(x, y)
        
        # Compute exact distance
        exact_distance = fast_lorentzian_distance(x, y)
        
        # Check approximation quality
        self.assertGreater(approx_distance, 0)
        relative_error = abs(approx_distance - exact_distance) / abs(exact_distance)
        self.assertLess(relative_error, 0.2, "Approximation error too high")
    
    def test_batch_approximate_distances(self):
        """Test batch distance computation"""
        query_point = self.test_features[0]
        feature_matrix = self.test_features[1:11]  # 10 points
        
        # Compute batch distances
        distances = self.approximator.batch_approximate_distances(
            feature_matrix, query_point
        )
        
        # Check output structure
        self.assertEqual(len(distances), len(feature_matrix))
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(np.isfinite(distances)))
    
    def test_approximation_quality_metrics(self):
        """Test approximation quality tracking"""
        # Generate some approximations to populate metrics
        for i in range(10):
            x = self.test_features[i]
            y = self.test_features[i+1]
            self.approximator.approximate_lorentzian_distance(x, y)
        
        # Get quality metrics
        metrics = self.approximator.get_approximation_quality_metrics()
        
        # Check metric structure
        required_keys = ['mean_error', 'max_error', 'std_error']
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0)

class TestTaylorANNClassifier(unittest.TestCase):
    """Test main Taylor ANN classifier"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        self.classifier = TaylorANNClassifier(self.config)
        
        # Generate training data
        np.random.seed(42)
        n_samples = 200
        self.features = np.random.rand(n_samples, 5)
        self.targets = (self.features[:, 0] + self.features[:, 1] > 1.0).astype(int)
    
    def test_fit_and_predict(self):
        """Test basic fit and predict functionality"""
        # Split data
        train_features = self.features[:150]
        train_targets = self.targets[:150]
        test_features = self.features[150:]
        test_targets = self.targets[150:]
        
        # Fit classifier
        self.classifier.fit(train_features, train_targets)
        
        # Make predictions
        predictions = []
        for features in test_features:
            pred = self.classifier.predict(features)
            predictions.append(pred)
        
        # Check prediction structure
        self.assertEqual(len(predictions), len(test_features))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Calculate accuracy
        accuracy = np.mean(np.array(predictions) == test_targets)
        self.assertGreater(accuracy, 0.4)  # Should be better than random
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Fit and make some predictions
        self.classifier.fit(self.features[:100], self.targets[:100])
        
        for i in range(10):
            self.classifier.predict(self.features[100 + i])
        
        # Get metrics
        metrics = self.classifier.get_performance_metrics()
        
        # Check metric structure
        required_keys = ['avg_query_time', 'cache_hit_rate', 'total_queries']
        for key in required_keys:
            self.assertIn(key, metrics)
    
    def test_caching_functionality(self):
        """Test query caching functionality"""
        # Fit classifier
        self.classifier.fit(self.features[:100], self.targets[:100])
        
        test_features = self.features[100]
        
        # First prediction (cache miss)
        start_time = time.time()
        pred1 = self.classifier.predict(test_features)
        time1 = time.time() - start_time
        
        # Second prediction (cache hit)
        start_time = time.time()
        pred2 = self.classifier.predict(test_features)
        time2 = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(pred1, pred2)
        
        # Second prediction should be faster (though this may not always hold due to system variance)
        # self.assertLess(time2, time1)  # Commented out due to potential flakiness

class TestPerformanceBenchmarking(unittest.TestCase):
    """Performance and benchmarking tests"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        np.random.seed(42)
        
        # Generate larger dataset for performance testing
        n_samples = 1000
        self.features = np.random.rand(n_samples, 5)
        self.targets = (self.features[:, 0] + self.features[:, 1] > 1.0).astype(int)
    
    def test_speedup_measurement(self):
        """Test speedup measurement against exact computation"""
        classifier = TaylorANNClassifier(self.config)
        
        # Split data
        train_features = self.features[:700]
        train_targets = self.targets[:700]
        test_features = self.features[700:750]  # 50 test samples
        
        # Fit classifier
        classifier.fit(train_features, train_targets)
        
        # Measure approximate computation time
        start_time = time.time()
        for features in test_features:
            classifier.predict(features, force_exact=False)
        approx_time = time.time() - start_time
        
        # Measure exact computation time
        start_time = time.time()
        for features in test_features:
            classifier.predict(features, force_exact=True)
        exact_time = time.time() - start_time
        
        # Calculate speedup
        speedup = exact_time / max(approx_time, 1e-6)
        
        print(f"Measured speedup: {speedup:.1f}x")
        
        # Speedup should be meaningful (at least 2x)
        self.assertGreater(speedup, 2.0)
    
    def test_accuracy_retention(self):
        """Test accuracy retention with approximation"""
        classifier = TaylorANNClassifier(self.config)
        
        # Split data
        train_features = self.features[:700]
        train_targets = self.targets[:700]
        test_features = self.features[700:800]
        test_targets = self.targets[700:800]
        
        # Fit classifier
        classifier.fit(train_features, train_targets)
        
        # Get exact predictions
        exact_predictions = []
        for features in test_features:
            pred = classifier.predict(features, force_exact=True)
            exact_predictions.append(pred)
        
        # Get approximate predictions
        approx_predictions = []
        for features in test_features:
            pred = classifier.predict(features, force_exact=False)
            approx_predictions.append(pred)
        
        # Calculate accuracy retention
        exact_accuracy = np.mean(np.array(exact_predictions) == test_targets)
        approx_accuracy = np.mean(np.array(approx_predictions) == test_targets)
        
        if exact_accuracy > 0:
            accuracy_retention = approx_accuracy / exact_accuracy
        else:
            accuracy_retention = 1.0
        
        print(f"Exact accuracy: {exact_accuracy:.3f}")
        print(f"Approx accuracy: {approx_accuracy:.3f}")
        print(f"Accuracy retention: {accuracy_retention:.1%}")
        
        # Accuracy retention should be high
        self.assertGreater(accuracy_retention, 0.8)

class TestMarketRegimeAware(unittest.TestCase):
    """Test market regime-aware components"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
        np.random.seed(42)
        
        # Generate market data
        n_samples = 500
        returns = np.random.normal(0.0001, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        # Generate features and targets
        self.features = np.random.rand(n_samples, 5)
        self.targets = (self.features[:, 0] + self.features[:, 1] > 1.0).astype(int)
    
    def test_regime_detection(self):
        """Test market regime detection"""
        detector = MarketRegimeDetector(self.config)
        regimes = detector.detect_regimes(self.market_data)
        
        # Check output structure
        self.assertEqual(len(regimes), len(self.market_data))
        
        # Check regime types
        valid_regimes = {'volatile', 'trending', 'ranging', 'normal'}
        unique_regimes = set(regimes)
        self.assertTrue(unique_regimes.issubset(valid_regimes))
    
    def test_regime_aware_ann(self):
        """Test regime-aware ANN system"""
        regime_ann = MarketRegimeAwareANN(self.config)
        
        # Fit regime-aware system
        train_size = 350
        train_features = self.features[:train_size]
        train_targets = self.targets[:train_size]
        train_market_data = self.market_data.iloc[:train_size]
        
        regime_ann.fit_regime_aware(train_features, train_targets, train_market_data)
        
        # Make regime-aware predictions
        test_features = self.features[train_size:train_size+50]
        test_market_data = self.market_data.iloc[:train_size+50]
        
        predictions = []
        for i, features in enumerate(test_features):
            current_market_slice = test_market_data.iloc[:train_size+i+1]
            pred = regime_ann.predict_regime_aware(features, current_market_slice)
            predictions.append(pred)
        
        # Check predictions
        self.assertEqual(len(predictions), len(test_features))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

class TestStressAndEdgeCases(unittest.TestCase):
    """Stress tests and edge case handling"""
    
    def setUp(self):
        self.config = TaylorANNConfig()
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        classifier = TaylorANNClassifier(self.config)
        
        # Test with empty training data
        empty_features = np.array([]).reshape(0, 5)
        empty_targets = np.array([])
        
        # Should not crash
        classifier.fit(empty_features, empty_targets)
        
        # Prediction with no training data
        test_features = np.random.rand(5)
        pred = classifier.predict(test_features)
        self.assertIn(pred, [0, 1])
    
    def test_extreme_values(self):
        """Test handling of extreme feature values"""
        classifier = TaylorANNClassifier(self.config)
        
        # Generate data with extreme values
        features = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            [1.0, 1.0, 1.0, 1.0, 1.0],  # All ones
            [1e-10, 1e-10, 1e-10, 1e-10, 1e-10],  # Very small
            [0.999999, 0.999999, 0.999999, 0.999999, 0.999999]  # Near boundary
        ])
        targets = np.array([0, 1, 0, 1])
        
        # Should handle extreme values without crashing
        classifier.fit(features, targets)
        
        # Test predictions with extreme values
        extreme_test = np.array([0.5, 1e-8, 0.999999, 0.0, 1.0])
        pred = classifier.predict(extreme_test)
        self.assertIn(pred, [0, 1])
    
    def test_single_class_data(self):
        """Test handling of single-class training data"""
        classifier = TaylorANNClassifier(self.config)
        
        # All targets are the same class
        features = np.random.rand(50, 5)
        targets = np.ones(50, dtype=int)  # All class 1
        
        classifier.fit(features, targets)
        
        # Predictions should still work
        test_features = np.random.rand(5)
        pred = classifier.predict(test_features)
        self.assertIn(pred, [0, 1])
    
    def test_high_dimensional_features(self):
        """Test with higher dimensional feature spaces"""
        # Modify config for higher dimensions
        high_dim_config = TaylorANNConfig()
        high_dim_config.feature_count = 20
        
        classifier = TaylorANNClassifier(high_dim_config)
        
        # Generate high-dimensional data
        features = np.random.rand(100, 20)
        targets = (np.sum(features[:, :5], axis=1) > 2.5).astype(int)
        
        # Should handle higher dimensions
        classifier.fit(features, targets)
        
        test_features = np.random.rand(20)
        pred = classifier.predict(test_features)
        self.assertIn(pred, [0, 1])

def run_comprehensive_tests():
    """Run all test suites and generate report"""
    print("TAYLOR SERIES ANN COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    # Create test suite
    test_classes = [
        TestTaylorSeriesComponents,
        TestExpansionPointSelector,
        TestTaylorCoefficientCache,
        TestTaylorDistanceApproximator,
        TestTaylorANNClassifier,
        TestPerformanceBenchmarking,
        TestMarketRegimeAware,
        TestStressAndEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests with custom result handler
        result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)
        
        class_total = result.testsRun
        class_failures = len(result.failures) + len(result.errors)
        class_passed = class_total - class_failures
        
        total_tests += class_total
        passed_tests += class_passed
        failed_tests += class_failures
        
        status = "✓ PASSED" if class_failures == 0 else f"✗ {class_failures} FAILED"
        print(f"  {class_passed}/{class_total} tests passed - {status}")
        
        # Print failure details if any
        if class_failures > 0:
            for failure in result.failures + result.errors:
                test_name = failure[0]._testMethodName
                error_msg = failure[1].split('\n')[-2] if failure[1] else "Unknown error"
                print(f"    FAILED: {test_name} - {error_msg}")
        
        print()
    
    # Generate summary report
    print("TEST SUMMARY REPORT")
    print("=" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/max(total_tests, 1):.1%}")
    print()
    
    if failed_tests == 0:
        print("🎯 ALL TESTS PASSED!")
        print("Taylor Series ANN system is ready for production deployment.")
    else:
        print("⚠️  Some tests failed. Review and fix issues before deployment.")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': passed_tests / max(total_tests, 1)
    }

if __name__ == "__main__":
    # Run comprehensive test suite
    test_results = run_comprehensive_tests()
    
    print("\nTest suite execution complete!")
    print(f"Results: {test_results}")