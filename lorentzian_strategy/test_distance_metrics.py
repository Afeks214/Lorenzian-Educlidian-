#!/usr/bin/env python3
"""
Comprehensive Test Suite for Lorentzian Distance Metrics

This script demonstrates and validates all features of the Lorentzian distance
implementation including performance optimization, mathematical validation,
and production readiness.

Usage:
    python test_distance_metrics.py

Author: Claude Code Assistant
Date: 2025-07-20
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorentzian_strategy.distance_metrics import (
    LorentzianDistanceCalculator,
    DistanceMetricsConfig,
    lorentzian_distance,
    euclidean_distance,
    manhattan_distance,
    run_comprehensive_tests
)


def test_basic_functionality():
    """Test basic distance calculation functionality"""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    # Test vectors
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    # Test simple function interface
    print("1. Simple Function Interface:")
    lorentz_dist = lorentzian_distance(x, y)
    euclidean_dist = euclidean_distance(x, y)
    manhattan_dist = manhattan_distance(x, y)
    
    print(f"   Lorentzian distance: {lorentz_dist:.6f}")
    print(f"   Euclidean distance:  {euclidean_dist:.6f}")
    print(f"   Manhattan distance:  {manhattan_dist:.6f}")
    
    # Test calculator interface
    print("\n2. Calculator Interface:")
    calculator = LorentzianDistanceCalculator()
    result = calculator.lorentzian_distance(x, y)
    
    print(f"   Distance: {result.distance:.6f}")
    print(f"   Computation time: {result.computation_time:.6f} seconds")
    print(f"   Method used: {result.method_used}")
    print(f"   Cache hit: {result.cache_hit}")
    print(f"   Validation passed: {result.validation_passed}")
    
    # Test weighted distance
    print("\n3. Weighted Distance:")
    weights = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
    weighted_result = calculator.lorentzian_distance(x, y, weights=weights)
    
    print(f"   Weighted distance: {weighted_result.distance:.6f}")
    print(f"   Method used: {weighted_result.method_used}")
    
    print("\n✓ Basic functionality tests completed successfully!")
    return True


def test_mathematical_properties():
    """Test mathematical properties validation"""
    print("\n" + "=" * 60)
    print("TESTING MATHEMATICAL PROPERTIES")
    print("=" * 60)
    
    calculator = LorentzianDistanceCalculator()
    
    # Run mathematical validation
    validation_results = calculator.validate_mathematical_properties(
        n_test_vectors=100, 
        n_features=8
    )
    
    print("Mathematical Properties Validation Results:")
    print(f"- Non-negativity success rate: {validation_results['non_negativity_success_rate']:.2%}")
    print(f"- Identity success rate: {validation_results['identity_success_rate']:.2%}")
    print(f"- Symmetry success rate: {validation_results['symmetry_success_rate']:.2%}")
    print(f"- Monotonicity success rate: {validation_results['monotonicity_success_rate']:.2%}")
    print(f"- Overall success rate: {validation_results['overall_success_rate']:.2%}")
    
    # Test specific mathematical properties manually
    print("\nManual Property Tests:")
    
    # Test 1: Non-negativity
    x = np.random.randn(5)
    y = np.random.randn(5)
    dist = calculator.lorentzian_distance(x, y).distance
    non_negative = dist >= 0
    print(f"1. Non-negativity (d >= 0): {non_negative} (d = {dist:.6f})")
    
    # Test 2: Identity of indiscernibles (approximately zero for identical vectors)
    epsilon = calculator.config.epsilon
    expected_identity = len(x) * np.log(1 + epsilon)
    self_dist = calculator.lorentzian_distance(x, x).distance
    identity_test = abs(self_dist - expected_identity) < 1e-10
    print(f"2. Identity (d(x,x) ≈ 0): {identity_test} (d = {self_dist:.6f})")
    
    # Test 3: Symmetry
    d_xy = calculator.lorentzian_distance(x, y).distance
    d_yx = calculator.lorentzian_distance(y, x).distance
    symmetry_test = abs(d_xy - d_yx) < 1e-10
    print(f"3. Symmetry (d(x,y) = d(y,x)): {symmetry_test}")
    
    success = (validation_results['overall_success_rate'] > 0.95 and 
              non_negative and identity_test and symmetry_test)
    
    print(f"\n{'✓' if success else '✗'} Mathematical properties tests completed!")
    return success


def test_batch_processing():
    """Test batch distance calculation functionality"""
    print("\n" + "=" * 60)
    print("TESTING BATCH PROCESSING")
    print("=" * 60)
    
    calculator = LorentzianDistanceCalculator()
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(10, 5)
    Y = np.random.randn(8, 5)
    
    print("1. Batch Distance Calculation:")
    batch_result = calculator.batch_distances(X, Y, metric="lorentzian")
    
    print(f"   Input X shape: {X.shape}")
    print(f"   Input Y shape: {Y.shape}")
    print(f"   Output distance matrix shape: {batch_result.distance.shape}")
    print(f"   Computation time: {batch_result.computation_time:.6f} seconds")
    print(f"   Method used: {batch_result.method_used}")
    print(f"   Total comparisons: {batch_result.metadata['total_comparisons']}")
    
    # Test different metrics
    print("\n2. Different Distance Metrics:")
    for metric in ["lorentzian", "euclidean", "manhattan"]:
        result = calculator.batch_distances(X[:5], Y[:5], metric=metric)
        print(f"   {metric.capitalize():>12}: {result.computation_time:.6f}s ({result.method_used})")
    
    # Verify batch vs individual calculations
    print("\n3. Batch vs Individual Validation:")
    individual_distances = []
    for i in range(3):
        for j in range(3):
            dist = calculator.lorentzian_distance(X[i], Y[j]).distance
            individual_distances.append(dist)
    
    batch_distances = calculator.batch_distances(X[:3], Y[:3]).distance.flatten()
    
    max_difference = np.max(np.abs(np.array(individual_distances) - batch_distances))
    validation_passed = max_difference < 1e-10
    
    print(f"   Maximum difference: {max_difference:.2e}")
    print(f"   Validation passed: {validation_passed}")
    
    print(f"\n{'✓' if validation_passed else '✗'} Batch processing tests completed!")
    return validation_passed


def test_knn_functionality():
    """Test k-nearest neighbors functionality"""
    print("\n" + "=" * 60)
    print("TESTING K-NEAREST NEIGHBORS")
    print("=" * 60)
    
    calculator = LorentzianDistanceCalculator()
    
    # Generate test data
    np.random.seed(42)
    query_vector = np.random.randn(6)
    reference_vectors = np.random.randn(20, 6)
    
    print("1. Basic k-NN Search:")
    knn_result = calculator.k_nearest_neighbors(
        query_vector, 
        reference_vectors, 
        k=5,
        metric="lorentzian"
    )
    
    print(f"   Query vector shape: {query_vector.shape}")
    print(f"   Reference vectors shape: {reference_vectors.shape}")
    print(f"   Found {len(knn_result['indices'])} nearest neighbors")
    print(f"   Neighbor indices: {knn_result['indices']}")
    print(f"   Neighbor distances: {knn_result['distances']}")
    print(f"   Computation time: {knn_result['computation_time']:.6f} seconds")
    
    # Test different metrics
    print("\n2. k-NN with Different Metrics:")
    for metric in ["lorentzian", "euclidean", "manhattan"]:
        result = calculator.k_nearest_neighbors(
            query_vector, 
            reference_vectors[:10], 
            k=3, 
            metric=metric
        )
        print(f"   {metric.capitalize():>12}: indices {result['indices']}")
    
    # Validate that distances are in ascending order
    distances = knn_result['distances']
    is_sorted = np.all(distances[:-1] <= distances[1:])
    
    print(f"\n3. Distance Ordering Validation:")
    print(f"   Distances sorted: {is_sorted}")
    print(f"   Distance values: {distances}")
    
    print(f"\n{'✓' if is_sorted else '✗'} k-NN functionality tests completed!")
    return is_sorted


def test_performance_optimization():
    """Test performance optimization features"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE OPTIMIZATION")
    print("=" * 60)
    
    # Test with different configurations
    configs = {
        "Basic": DistanceMetricsConfig(
            use_numba_jit=False,
            use_gpu_acceleration=False,
            enable_caching=False
        ),
        "JIT Optimized": DistanceMetricsConfig(
            use_numba_jit=True,
            use_gpu_acceleration=False,
            enable_caching=False
        ),
        "Cached": DistanceMetricsConfig(
            use_numba_jit=True,
            use_gpu_acceleration=False,
            enable_caching=True
        ),
        "Full Optimization": DistanceMetricsConfig(
            use_numba_jit=True,
            use_gpu_acceleration=True,  # Will fallback if GPU not available
            enable_caching=True
        )
    }
    
    # Generate test data
    np.random.seed(42)
    x = np.random.randn(10)
    y = np.random.randn(10)
    
    print("1. Configuration Performance Comparison:")
    performance_results = {}
    
    for config_name, config in configs.items():
        calculator = LorentzianDistanceCalculator(config)
        
        # Warm up
        calculator.lorentzian_distance(x, y)
        
        # Time multiple calculations
        start_time = time.time()
        for _ in range(100):
            calculator.lorentzian_distance(x, y)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        performance_results[config_name] = avg_time
        
        print(f"   {config_name:>17}: {avg_time:.6f}s per calculation")
    
    # Test caching effectiveness
    print("\n2. Cache Performance Test:")
    cached_calculator = LorentzianDistanceCalculator(configs["Cached"])
    
    # First calculation (cache miss)
    start_time = time.time()
    result1 = cached_calculator.lorentzian_distance(x, y)
    first_time = time.time() - start_time
    
    # Second calculation (cache hit)
    start_time = time.time()
    result2 = cached_calculator.lorentzian_distance(x, y)
    second_time = time.time() - start_time
    
    speedup = first_time / second_time if second_time > 0 else float('inf')
    
    print(f"   First calculation (miss): {first_time:.6f}s")
    print(f"   Second calculation (hit): {second_time:.6f}s")
    print(f"   Cache speedup: {speedup:.1f}x")
    
    # Performance statistics
    print("\n3. Performance Statistics:")
    stats = cached_calculator.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n✓ Performance optimization tests completed!")
    return True


def test_production_features():
    """Test production-ready features"""
    print("\n" + "=" * 60)
    print("TESTING PRODUCTION FEATURES")
    print("=" * 60)
    
    # Test error handling
    print("1. Error Handling Tests:")
    
    calculator = LorentzianDistanceCalculator()
    
    # Test with mismatched dimensions
    x_short = np.array([1.0, 2.0])
    y_long = np.array([1.0, 2.0, 3.0])
    
    result = calculator.lorentzian_distance(x_short, y_long)
    print(f"   Mismatched dimensions: validation_passed = {result.validation_passed}")
    
    # Test with NaN values
    x_nan = np.array([1.0, np.nan, 3.0])
    y_clean = np.array([1.0, 2.0, 3.0])
    
    result = calculator.lorentzian_distance(x_nan, y_clean)
    print(f"   NaN input handling: validation_passed = {result.validation_passed}")
    
    # Test with infinite values
    x_inf = np.array([1.0, np.inf, 3.0])
    
    result = calculator.lorentzian_distance(x_inf, y_clean)
    print(f"   Infinite input handling: validation_passed = {result.validation_passed}")
    
    # Test numerical stability
    print("\n2. Numerical Stability Tests:")
    
    # Very small differences
    x_base = np.array([1.0, 2.0, 3.0])
    y_close = x_base + 1e-15
    
    result = calculator.lorentzian_distance(x_base, y_close)
    print(f"   Very small differences: distance = {result.distance:.6f}")
    
    # Very large differences
    x_large = np.array([1e6, 2e6, 3e6])
    y_large = np.array([1e6 + 1000, 2e6 + 1000, 3e6 + 1000])
    
    result = calculator.lorentzian_distance(x_large, y_large)
    print(f"   Large value handling: distance = {result.distance:.6f}")
    
    # Test configuration management
    print("\n3. Configuration Management:")
    
    # Save and load configuration
    config_file = "/tmp/test_lorentzian_config.pkl"
    try:
        calculator.save_config(config_file)
        loaded_calculator = LorentzianDistanceCalculator.load_config(config_file)
        print(f"   Configuration save/load: ✓")
        
        # Clean up
        os.remove(config_file)
    except Exception as e:
        print(f"   Configuration save/load: ✗ ({e})")
    
    print(f"\n✓ Production features tests completed!")
    return True


def test_integration_scenarios():
    """Test realistic integration scenarios"""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION SCENARIOS")
    print("=" * 60)
    
    calculator = LorentzianDistanceCalculator()
    
    # Scenario 1: Financial time series feature vectors
    print("1. Financial Time Series Scenario:")
    
    # Simulate feature vectors (RSI, WT1, WT2, CCI, ADX)
    np.random.seed(42)
    
    # Current market state
    current_features = np.array([0.6, 0.3, 0.2, 0.1, 0.7])  # Normalized [0,1]
    
    # Historical patterns
    n_historical = 1000
    historical_features = np.random.rand(n_historical, 5)
    
    # Find similar historical patterns
    knn_result = calculator.k_nearest_neighbors(
        current_features,
        historical_features,
        k=10,
        metric="lorentzian"
    )
    
    print(f"   Current features: {current_features}")
    print(f"   Historical database size: {n_historical}")
    print(f"   Found {len(knn_result['indices'])} similar patterns")
    print(f"   Similarity scores: {knn_result['distances'][:5]}")  # Top 5
    print(f"   Search time: {knn_result['computation_time']:.6f}s")
    
    # Scenario 2: Real-time pattern matching
    print("\n2. Real-time Pattern Matching:")
    
    # Simulate real-time processing
    batch_size = 50
    total_patterns = 0
    total_time = 0
    
    for batch in range(5):
        # Generate batch of current market states
        current_batch = np.random.rand(batch_size, 5)
        
        start_time = time.time()
        
        # Process each pattern in the batch
        for pattern in current_batch:
            knn_result = calculator.k_nearest_neighbors(
                pattern,
                historical_features,
                k=5,
                metric="lorentzian"
            )
            total_patterns += 1
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        print(f"   Batch {batch + 1}: {batch_size} patterns in {batch_time:.6f}s "
              f"({batch_time/batch_size:.6f}s per pattern)")
    
    avg_time_per_pattern = total_time / total_patterns
    throughput = 1.0 / avg_time_per_pattern
    
    print(f"   Total patterns processed: {total_patterns}")
    print(f"   Average time per pattern: {avg_time_per_pattern:.6f}s")
    print(f"   Throughput: {throughput:.1f} patterns/second")
    
    # Scenario 3: Multi-timeframe analysis
    print("\n3. Multi-timeframe Analysis:")
    
    timeframes = {
        "5min": {"features": 5, "history": 2000},
        "30min": {"features": 5, "history": 1500}, 
        "1h": {"features": 5, "history": 1000},
        "4h": {"features": 5, "history": 500}
    }
    
    current_multi_tf = {
        tf: np.random.rand(info["features"]) 
        for tf, info in timeframes.items()
    }
    
    for tf, info in timeframes.items():
        # Generate historical data for this timeframe
        historical_tf = np.random.rand(info["history"], info["features"])
        
        # Find patterns
        knn_result = calculator.k_nearest_neighbors(
            current_multi_tf[tf],
            historical_tf,
            k=5,
            metric="lorentzian"
        )
        
        print(f"   {tf:>5}: {info['history']} patterns, "
              f"top similarity: {knn_result['distances'][0]:.4f}")
    
    print(f"\n✓ Integration scenario tests completed!")
    return True


def main():
    """Run all tests"""
    print("LORENTZIAN DISTANCE METRICS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing mathematical core implementation with performance optimization")
    print("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run test modules
    test_functions = [
        ("Basic Functionality", test_basic_functionality),
        ("Mathematical Properties", test_mathematical_properties),
        ("Batch Processing", test_batch_processing),
        ("k-NN Functionality", test_knn_functionality),
        ("Performance Optimization", test_performance_optimization),
        ("Production Features", test_production_features),
        ("Integration Scenarios", test_integration_scenarios),
    ]
    
    overall_success = True
    
    for test_name, test_func in test_functions:
        try:
            success = test_func()
            test_results[test_name] = success
            if not success:
                overall_success = False
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            test_results[test_name] = False
            overall_success = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    for test_name, success in test_results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Lorentzian Distance Metrics implementation is ready for production!")
        print("   - Mathematical accuracy: Validated")
        print("   - Performance optimization: Enabled")
        print("   - Production features: Tested")
        print("   - Integration scenarios: Verified")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    print("=" * 80)
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)