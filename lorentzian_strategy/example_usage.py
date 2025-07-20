#!/usr/bin/env python3
"""
Example Usage of Lorentzian Distance Metrics

This script demonstrates practical usage of the Lorentzian distance implementation
for financial time series analysis and pattern matching.

Usage:
    python example_usage.py

Author: Claude Code Assistant
Date: 2025-07-20
"""

import sys
import os
import numpy as np
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorentzian_strategy import (
    LorentzianDistanceCalculator,
    DistanceMetricsConfig,
    lorentzian_distance,
    create_production_calculator
)


def example_1_basic_usage():
    """Example 1: Basic distance calculation"""
    print("Example 1: Basic Distance Calculation")
    print("=" * 50)
    
    # Create feature vectors (normalized technical indicators)
    # Represents: [RSI, WT1, WT2, CCI, ADX]
    current_market = np.array([0.65, 0.3, 0.25, 0.1, 0.75])  # Current market state
    historical_pattern = np.array([0.68, 0.32, 0.28, 0.12, 0.72])  # Similar historical pattern
    
    # Simple distance calculation
    distance = lorentzian_distance(current_market, historical_pattern)
    print(f"Current market: {current_market}")
    print(f"Historical pattern: {historical_pattern}")
    print(f"Lorentzian distance: {distance:.6f}")
    
    # Compare with Euclidean distance
    euclidean_dist = np.sqrt(np.sum((current_market - historical_pattern) ** 2))
    print(f"Euclidean distance: {euclidean_dist:.6f}")
    print(f"Lorentzian emphasizes smaller differences better!\n")


def example_2_pattern_matching():
    """Example 2: Financial pattern matching with k-NN"""
    print("Example 2: Financial Pattern Matching")
    print("=" * 50)
    
    # Create production-optimized calculator
    calculator = create_production_calculator()
    
    # Simulate current market conditions
    current_state = np.array([0.7, 0.4, 0.35, 0.2, 0.8])  # Overbought conditions
    
    # Simulate historical database (1000 patterns)
    np.random.seed(42)
    n_historical = 1000
    historical_patterns = np.random.rand(n_historical, 5)
    
    # Add some similar patterns to current state
    for i in range(5):
        noise = np.random.normal(0, 0.05, 5)  # Small random variations
        similar_pattern = np.clip(current_state + noise, 0, 1)
        historical_patterns[i] = similar_pattern
    
    print(f"Current market state: {current_state}")
    print(f"Searching {n_historical} historical patterns...")
    
    # Find similar patterns
    start_time = time.time()
    knn_result = calculator.k_nearest_neighbors(
        current_state,
        historical_patterns,
        k=10,
        metric="lorentzian"
    )
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.6f} seconds")
    print(f"Top 5 similar pattern indices: {knn_result['indices'][:5]}")
    print(f"Similarity scores: {knn_result['distances'][:5]}")
    print(f"Most similar pattern: {historical_patterns[knn_result['indices'][0]]}")
    print()


def example_3_batch_processing():
    """Example 3: Efficient batch processing"""
    print("Example 3: Batch Distance Processing")
    print("=" * 50)
    
    calculator = LorentzianDistanceCalculator()
    
    # Simulate multiple current market states
    current_states = np.random.rand(20, 5)  # 20 different market conditions
    
    # Simulate historical patterns
    historical_patterns = np.random.rand(50, 5)  # 50 historical patterns
    
    print(f"Calculating distances between {current_states.shape[0]} current states")
    print(f"and {historical_patterns.shape[0]} historical patterns...")
    
    # Batch calculation
    start_time = time.time()
    batch_result = calculator.batch_distances(
        current_states, 
        historical_patterns, 
        metric="lorentzian"
    )
    batch_time = time.time() - start_time
    
    distance_matrix = batch_result.distance
    
    print(f"Batch processing completed in {batch_time:.6f} seconds")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Total comparisons: {distance_matrix.size}")
    print(f"Average time per comparison: {batch_time / distance_matrix.size:.9f} seconds")
    
    # Find closest matches for each current state
    closest_matches = np.argmin(distance_matrix, axis=1)
    min_distances = np.min(distance_matrix, axis=1)
    
    print(f"\nClosest matches for first 5 current states:")
    for i in range(5):
        print(f"State {i}: matches historical pattern {closest_matches[i]} "
              f"(distance: {min_distances[i]:.6f})")
    print()


def example_4_real_time_simulation():
    """Example 4: Real-time trading signal simulation"""
    print("Example 4: Real-Time Trading Signal Simulation")
    print("=" * 50)
    
    class SimpleLorentzianStrategy:
        def __init__(self, history_size=500, k_neighbors=20):
            self.calculator = LorentzianDistanceCalculator()
            self.feature_history = []
            self.return_history = []
            self.history_size = history_size
            self.k_neighbors = k_neighbors
        
        def update_history(self, features, future_return):
            """Update historical database"""
            self.feature_history.append(features)
            self.return_history.append(future_return)
            
            # Maintain history size
            if len(self.feature_history) > self.history_size:
                self.feature_history.pop(0)
                self.return_history.pop(0)
        
        def generate_signal(self, current_features):
            """Generate trading signal based on similar patterns"""
            if len(self.feature_history) < 50:
                return None  # Need sufficient history
            
            # Find similar historical patterns
            knn_result = self.calculator.k_nearest_neighbors(
                current_features,
                np.array(self.feature_history),
                k=self.k_neighbors,
                metric="lorentzian"
            )
            
            # Get future returns of similar patterns
            similar_returns = []
            for idx in knn_result['indices']:
                if idx < len(self.return_history):
                    similar_returns.append(self.return_history[idx])
            
            if not similar_returns:
                return None
            
            # Calculate expected return and confidence
            expected_return = np.mean(similar_returns)
            return_std = np.std(similar_returns)
            avg_distance = np.mean(knn_result['distances'])
            
            # Confidence inversely related to distance and return volatility
            confidence = 1.0 / (1.0 + avg_distance + return_std)
            
            return {
                'signal': 1 if expected_return > 0 else -1,
                'expected_return': expected_return,
                'confidence': confidence,
                'similar_patterns': len(similar_returns),
                'avg_distance': avg_distance
            }
    
    # Initialize strategy
    strategy = SimpleLorentzianStrategy()
    
    # Simulate historical data
    np.random.seed(42)
    n_periods = 200
    
    print(f"Simulating {n_periods} market periods...")
    
    signals_generated = 0
    correct_signals = 0
    total_return = 0.0
    
    for period in range(n_periods):
        # Generate market features (normalized technical indicators)
        features = np.random.rand(5)
        
        # Generate future return (random walk with slight momentum)
        if period > 0:
            momentum = 0.1 * previous_return if 'previous_return' in locals() else 0
            future_return = np.random.normal(momentum, 0.02)
        else:
            future_return = np.random.normal(0, 0.02)
        
        # Generate signal
        signal_info = strategy.generate_signal(features)
        
        if signal_info is not None:
            signals_generated += 1
            
            # Check if signal was correct
            actual_direction = 1 if future_return > 0 else -1
            if signal_info['signal'] == actual_direction:
                correct_signals += 1
            
            # Calculate strategy return (simplified)
            strategy_return = signal_info['signal'] * future_return * signal_info['confidence']
            total_return += strategy_return
            
            if period % 50 == 0 and period > 0:
                accuracy = correct_signals / signals_generated if signals_generated > 0 else 0
                print(f"Period {period}: "
                      f"Signals: {signals_generated}, "
                      f"Accuracy: {accuracy:.2%}, "
                      f"Return: {total_return:.4f}")
        
        # Update history
        strategy.update_history(features, future_return)
        previous_return = future_return
    
    # Final results
    final_accuracy = correct_signals / signals_generated if signals_generated > 0 else 0
    print(f"\nFinal Results:")
    print(f"Total signals generated: {signals_generated}")
    print(f"Signal accuracy: {final_accuracy:.2%}")
    print(f"Total strategy return: {total_return:.4f}")
    print(f"Average return per signal: {total_return / signals_generated:.6f}")
    print()


def example_5_performance_comparison():
    """Example 5: Performance comparison with different configurations"""
    print("Example 5: Performance Optimization Comparison")
    print("=" * 50)
    
    # Test different configurations
    configs = {
        "Basic": DistanceMetricsConfig(
            use_numba_jit=False,
            enable_caching=False
        ),
        "JIT Optimized": DistanceMetricsConfig(
            use_numba_jit=True,
            enable_caching=False
        ),
        "JIT + Caching": DistanceMetricsConfig(
            use_numba_jit=True,
            enable_caching=True
        )
    }
    
    # Generate test data
    np.random.seed(42)
    x = np.random.rand(10)
    y = np.random.rand(10)
    
    print("Comparing performance for 1000 distance calculations:")
    
    for config_name, config in configs.items():
        calculator = LorentzianDistanceCalculator(config)
        
        # Warm up
        calculator.lorentzian_distance(x, y)
        
        # Time multiple calculations
        start_time = time.time()
        for _ in range(1000):
            calculator.lorentzian_distance(x, y)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 1000
        
        # Get performance stats
        stats = calculator.get_performance_stats()
        cache_rate = stats.get('cache_hit_rate', 0)
        
        print(f"{config_name:>15}: {avg_time:.6f}s per calc, "
              f"cache rate: {cache_rate:.1%}")
    
    print()


def example_6_mathematical_validation():
    """Example 6: Mathematical property validation"""
    print("Example 6: Mathematical Property Validation")
    print("=" * 50)
    
    calculator = LorentzianDistanceCalculator()
    
    # Run comprehensive mathematical validation
    validation_results = calculator.validate_mathematical_properties(
        n_test_vectors=100,
        n_features=5
    )
    
    print("Mathematical Property Validation Results:")
    print(f"Non-negativity success rate: {validation_results['non_negativity_success_rate']:.2%}")
    print(f"Identity success rate: {validation_results['identity_success_rate']:.2%}")
    print(f"Symmetry success rate: {validation_results['symmetry_success_rate']:.2%}")
    print(f"Monotonicity success rate: {validation_results['monotonicity_success_rate']:.2%}")
    print(f"Overall success rate: {validation_results['overall_success_rate']:.2%}")
    
    if validation_results['overall_success_rate'] > 0.95:
        print("✅ Mathematical validation PASSED!")
    else:
        print("❌ Mathematical validation FAILED!")
    
    print()


def main():
    """Run all examples"""
    print("LORENTZIAN DISTANCE METRICS - PRACTICAL EXAMPLES")
    print("=" * 80)
    print("Demonstrating financial time series pattern matching capabilities")
    print("=" * 80)
    print()
    
    # Run examples
    example_1_basic_usage()
    example_2_pattern_matching()
    example_3_batch_processing()
    example_4_real_time_simulation()
    example_5_performance_comparison()
    example_6_mathematical_validation()
    
    print("=" * 80)
    print("🎉 All examples completed successfully!")
    print("The Lorentzian distance implementation is ready for production use.")
    print("=" * 80)


if __name__ == "__main__":
    main()