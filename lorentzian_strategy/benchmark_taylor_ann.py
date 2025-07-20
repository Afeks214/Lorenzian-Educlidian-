"""
TAYLOR SERIES ANN PERFORMANCE BENCHMARKING SUITE
===============================================

Comprehensive benchmarking framework to validate the Taylor Series ANN system
against the research targets:
- 25x speedup over traditional KNN
- 90% accuracy retention
- Real-time trading compatibility

This script provides detailed performance analysis, comparison studies,
and production readiness validation.

Author: Claude AI Research Division
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorentzian_strategy.taylor_ann import (
    TaylorANNConfig,
    TaylorANNClassifier,
    MarketRegimeAwareANN,
    fast_lorentzian_distance
)

warnings.filterwarnings('ignore')

class TaylorANNBenchmarkSuite:
    """
    Comprehensive benchmarking suite for Taylor Series ANN system
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Benchmark configuration
        self.config = TaylorANNConfig(
            k_neighbors=8,
            taylor_order=4,
            expansion_points_count=50,
            speedup_target=25.0,
            accuracy_target=0.90,
            parallel_threads=4
        )
        
        # Results storage
        self.benchmark_results = {}
        self.performance_history = []
        
    def generate_synthetic_market_data(self, n_samples: int = 5000, 
                                     n_features: int = 5) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate realistic synthetic market data for benchmarking
        """
        print(f"Generating {n_samples} samples with {n_features} features...")
        
        # Reset random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate base features with market-like characteristics
        features = np.random.randn(n_samples, n_features)
        
        # Add realistic market correlations
        for i in range(1, n_features):
            # Auto-correlation
            features[1:, i] += 0.3 * features[:-1, i]
            # Cross-correlation with previous feature
            features[:, i] += 0.2 * features[:, i-1]
        
        # Add regime changes
        regime_changes = [n_samples//4, n_samples//2, 3*n_samples//4]
        for change_point in regime_changes:
            # Change volatility regime
            volatility_factor = np.random.uniform(0.5, 2.0)
            features[change_point:] *= volatility_factor
        
        # Normalize features to [0, 1] range
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        
        # Generate targets with some structure (more realistic than random)
        # Combine multiple indicators for target
        feature_sum = np.sum(features[:, :3], axis=1)
        feature_trend = np.diff(np.concatenate([[feature_sum[0]], feature_sum]))
        momentum = np.convolve(feature_trend, np.ones(5)/5, mode='same')  # Moving average
        
        # Create targets based on multiple conditions
        targets = ((feature_sum > np.median(feature_sum)) & 
                  (momentum > 0)).astype(int)
        
        # Generate corresponding market data (prices, volume)
        returns = np.random.normal(0.0001, 0.02, n_samples)
        returns += 0.01 * (targets - 0.5)  # Targets influence returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add volatility clustering
        for i in range(1, n_samples):
            vol_persistence = 0.7
            returns[i] *= (1 + vol_persistence * abs(returns[i-1]))
        
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='1H'),
            'open': np.roll(prices, 1),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        return features, targets, market_data
    
    def benchmark_against_sklearn_knn(self, features: np.ndarray, targets: np.ndarray,
                                    test_size: float = 0.3) -> Dict[str, float]:
        """
        Benchmark Taylor ANN against scikit-learn KNN implementation
        """
        print("\\nBenchmarking against scikit-learn KNN...")
        
        # Split data
        split_idx = int(len(features) * (1 - test_size))
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        test_features = features[split_idx:]
        test_targets = targets[split_idx:]
        
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples: {len(test_features)}")
        
        # Initialize classifiers
        taylor_ann = TaylorANNClassifier(self.config)
        sklearn_knn = KNeighborsClassifier(
            n_neighbors=self.config.k_neighbors,
            metric='manhattan',  # Closest to Lorentzian
            algorithm='auto'
        )
        
        # Train both classifiers
        print("Training Taylor ANN...")
        start_time = time.time()
        taylor_ann.fit(train_features, train_targets)
        taylor_train_time = time.time() - start_time
        
        print("Training sklearn KNN...")
        start_time = time.time()
        sklearn_knn.fit(train_features, train_targets)
        sklearn_train_time = time.time() - start_time
        
        # Benchmark inference speed
        print("Benchmarking inference speed...")
        
        # Taylor ANN predictions (approximate)
        taylor_predictions = []
        start_time = time.time()
        for features_row in test_features:
            pred = taylor_ann.predict(features_row, force_exact=False)
            taylor_predictions.append(pred)
        taylor_inference_time = time.time() - start_time
        
        # Taylor ANN predictions (exact)
        taylor_exact_predictions = []
        start_time = time.time()
        for features_row in test_features:
            pred = taylor_ann.predict(features_row, force_exact=True)
            taylor_exact_predictions.append(pred)
        taylor_exact_time = time.time() - start_time
        
        # sklearn KNN predictions
        sklearn_predictions = []
        start_time = time.time()
        sklearn_preds = sklearn_knn.predict(test_features)
        sklearn_inference_time = time.time() - start_time
        sklearn_predictions = sklearn_preds.tolist()
        
        # Calculate metrics
        taylor_accuracy = accuracy_score(test_targets, taylor_predictions)
        taylor_exact_accuracy = accuracy_score(test_targets, taylor_exact_predictions)
        sklearn_accuracy = accuracy_score(test_targets, sklearn_predictions)
        
        # Additional metrics
        taylor_precision = precision_score(test_targets, taylor_predictions, average='weighted')
        taylor_recall = recall_score(test_targets, taylor_predictions, average='weighted')
        taylor_f1 = f1_score(test_targets, taylor_predictions, average='weighted')
        
        sklearn_precision = precision_score(test_targets, sklearn_predictions, average='weighted')
        sklearn_recall = recall_score(test_targets, sklearn_predictions, average='weighted')
        sklearn_f1 = f1_score(test_targets, sklearn_predictions, average='weighted')
        
        # Calculate speedups
        speedup_vs_sklearn = sklearn_inference_time / max(taylor_inference_time, 1e-6)
        speedup_vs_exact = taylor_exact_time / max(taylor_inference_time, 1e-6)
        accuracy_retention = taylor_accuracy / max(taylor_exact_accuracy, 0.001)
        
        results = {
            # Training times
            'taylor_train_time': taylor_train_time,
            'sklearn_train_time': sklearn_train_time,
            
            # Inference times
            'taylor_inference_time': taylor_inference_time,
            'taylor_exact_time': taylor_exact_time,
            'sklearn_inference_time': sklearn_inference_time,
            
            # Accuracy metrics
            'taylor_accuracy': taylor_accuracy,
            'taylor_exact_accuracy': taylor_exact_accuracy,
            'sklearn_accuracy': sklearn_accuracy,
            'accuracy_retention': accuracy_retention,
            
            # Additional metrics
            'taylor_precision': taylor_precision,
            'taylor_recall': taylor_recall,
            'taylor_f1': taylor_f1,
            'sklearn_precision': sklearn_precision,
            'sklearn_recall': sklearn_recall,
            'sklearn_f1': sklearn_f1,
            
            # Speedup metrics
            'speedup_vs_sklearn': speedup_vs_sklearn,
            'speedup_vs_exact': speedup_vs_exact,
            
            # Target achievement
            'target_speedup_achieved': speedup_vs_exact >= self.config.speedup_target,
            'target_accuracy_achieved': accuracy_retention >= self.config.accuracy_target,
            
            # Sample information
            'n_train_samples': len(train_features),
            'n_test_samples': len(test_features),
            'n_features': features.shape[1]
        }
        
        return results
    
    def scalability_benchmark(self, base_features: np.ndarray, base_targets: np.ndarray) -> Dict[str, List]:
        """
        Test scalability with different dataset sizes
        """
        print("\\nRunning scalability benchmark...")
        
        # Different dataset sizes to test
        sizes = [500, 1000, 2000, 3000, 4000, 5000]
        sizes = [s for s in sizes if s <= len(base_features)]
        
        results = {
            'dataset_sizes': [],
            'taylor_times': [],
            'exact_times': [],
            'sklearn_times': [],
            'taylor_accuracies': [],
            'exact_accuracies': [],
            'sklearn_accuracies': [],
            'speedups_vs_exact': [],
            'accuracy_retentions': []
        }
        
        for size in sizes:
            print(f"  Testing with {size} samples...")
            
            # Subsample data
            indices = np.random.choice(len(base_features), size, replace=False)
            features = base_features[indices]
            targets = base_targets[indices]
            
            # Split data
            split_idx = int(size * 0.7)
            train_features = features[:split_idx]
            train_targets = targets[:split_idx]
            test_features = features[split_idx:split_idx+100]  # Fixed test size
            test_targets = targets[split_idx:split_idx+100]
            
            if len(test_features) == 0:
                continue
            
            # Initialize classifiers
            taylor_ann = TaylorANNClassifier(self.config)
            sklearn_knn = KNeighborsClassifier(n_neighbors=self.config.k_neighbors, metric='manhattan')
            
            # Train
            taylor_ann.fit(train_features, train_targets)
            sklearn_knn.fit(train_features, train_targets)
            
            # Time predictions
            start_time = time.time()
            taylor_preds = [taylor_ann.predict(f, force_exact=False) for f in test_features]
            taylor_time = time.time() - start_time
            
            start_time = time.time()
            exact_preds = [taylor_ann.predict(f, force_exact=True) for f in test_features]
            exact_time = time.time() - start_time
            
            start_time = time.time()
            sklearn_preds = sklearn_knn.predict(test_features).tolist()
            sklearn_time = time.time() - start_time
            
            # Calculate metrics
            taylor_acc = accuracy_score(test_targets, taylor_preds)
            exact_acc = accuracy_score(test_targets, exact_preds)
            sklearn_acc = accuracy_score(test_targets, sklearn_preds)
            
            speedup = exact_time / max(taylor_time, 1e-6)
            retention = taylor_acc / max(exact_acc, 0.001)
            
            # Store results
            results['dataset_sizes'].append(size)
            results['taylor_times'].append(taylor_time)
            results['exact_times'].append(exact_time)
            results['sklearn_times'].append(sklearn_time)
            results['taylor_accuracies'].append(taylor_acc)
            results['exact_accuracies'].append(exact_acc)
            results['sklearn_accuracies'].append(sklearn_acc)
            results['speedups_vs_exact'].append(speedup)
            results['accuracy_retentions'].append(retention)
        
        return results
    
    def regime_awareness_benchmark(self, features: np.ndarray, targets: np.ndarray,
                                 market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Benchmark regime-aware Taylor ANN system
        """
        print("\\nBenchmarking regime-aware system...")
        
        # Split data
        split_idx = int(len(features) * 0.7)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        train_market_data = market_data.iloc[:split_idx]
        
        test_features = features[split_idx:]
        test_targets = targets[split_idx:]
        test_market_data = market_data.iloc[:len(features)]
        
        # Initialize systems
        standard_ann = TaylorANNClassifier(self.config)
        regime_ann = MarketRegimeAwareANN(self.config)
        
        # Train systems
        print("  Training standard Taylor ANN...")
        standard_ann.fit(train_features, train_targets)
        
        print("  Training regime-aware Taylor ANN...")
        regime_ann.fit_regime_aware(train_features, train_targets, train_market_data)
        
        # Make predictions
        print("  Making predictions...")
        standard_predictions = []
        regime_predictions = []
        standard_times = []
        regime_times = []
        
        for i, features_row in enumerate(test_features[:200]):  # Limit for speed
            # Standard prediction
            start_time = time.time()
            std_pred = standard_ann.predict(features_row)
            standard_times.append(time.time() - start_time)
            standard_predictions.append(std_pred)
            
            # Regime-aware prediction
            current_market_slice = test_market_data.iloc[:split_idx + i + 50]
            start_time = time.time()
            regime_pred = regime_ann.predict_regime_aware(features_row, current_market_slice)
            regime_times.append(time.time() - start_time)
            regime_predictions.append(regime_pred)
        
        # Calculate metrics
        test_targets_subset = test_targets[:len(standard_predictions)]
        
        standard_accuracy = accuracy_score(test_targets_subset, standard_predictions)
        regime_accuracy = accuracy_score(test_targets_subset, regime_predictions)
        
        results = {
            'standard_accuracy': standard_accuracy,
            'regime_accuracy': regime_accuracy,
            'regime_improvement': regime_accuracy - standard_accuracy,
            'avg_standard_time': np.mean(standard_times),
            'avg_regime_time': np.mean(regime_times),
            'regime_overhead': np.mean(regime_times) / max(np.mean(standard_times), 1e-6)
        }
        
        return results
    
    def memory_efficiency_benchmark(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Benchmark memory efficiency and compression
        """
        print("\\nBenchmarking memory efficiency...")
        
        import psutil
        import os
        
        # Get memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with compression enabled
        compressed_config = TaylorANNConfig()
        compressed_config.compress_features = True
        compressed_config.enable_caching = True
        
        compressed_ann = TaylorANNClassifier(compressed_config)
        compressed_ann.fit(features, targets)
        
        memory_after_compressed = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test without compression
        uncompressed_config = TaylorANNConfig()
        uncompressed_config.compress_features = False
        uncompressed_config.enable_caching = True
        
        uncompressed_ann = TaylorANNClassifier(uncompressed_config)
        uncompressed_ann.fit(features, targets)
        
        memory_after_uncompressed = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate efficiency metrics
        compressed_memory_usage = memory_after_compressed - memory_before
        uncompressed_memory_usage = memory_after_uncompressed - memory_after_compressed
        
        memory_savings = max(0, uncompressed_memory_usage - compressed_memory_usage)
        compression_ratio = compressed_memory_usage / max(uncompressed_memory_usage, 1)
        
        results = {
            'baseline_memory_mb': memory_before,
            'compressed_memory_usage_mb': compressed_memory_usage,
            'uncompressed_memory_usage_mb': uncompressed_memory_usage,
            'memory_savings_mb': memory_savings,
            'compression_ratio': compression_ratio,
            'memory_efficiency_gain': (1 - compression_ratio) * 100  # Percentage
        }
        
        return results
    
    def real_time_performance_benchmark(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Benchmark real-time performance requirements
        """
        print("\\nBenchmarking real-time performance...")
        
        # Split data
        train_features = features[:int(len(features) * 0.8)]
        train_targets = targets[:int(len(targets) * 0.8)]
        test_features = features[int(len(features) * 0.8):]
        
        # Initialize classifier
        taylor_ann = TaylorANNClassifier(self.config)
        taylor_ann.fit(train_features, train_targets)
        
        # Simulate real-time prediction requirements
        print("  Simulating real-time predictions...")
        
        prediction_times = []
        predictions = []
        
        # Test single predictions (typical real-time scenario)
        for i in range(min(1000, len(test_features))):
            start_time = time.time()
            pred = taylor_ann.predict(test_features[i])
            pred_time = time.time() - start_time
            
            prediction_times.append(pred_time)
            predictions.append(pred)
        
        # Calculate real-time metrics
        avg_prediction_time = np.mean(prediction_times)
        max_prediction_time = np.max(prediction_times)
        percentile_95_time = np.percentile(prediction_times, 95)
        
        # Real-time requirements (typical for high-frequency trading)
        latency_1ms = np.sum(np.array(prediction_times) <= 0.001) / len(prediction_times)
        latency_10ms = np.sum(np.array(prediction_times) <= 0.01) / len(prediction_times)
        latency_100ms = np.sum(np.array(prediction_times) <= 0.1) / len(prediction_times)
        
        # Throughput test
        print("  Testing throughput...")
        start_time = time.time()
        batch_predictions = [taylor_ann.predict(f) for f in test_features[:100]]
        batch_time = time.time() - start_time
        throughput = 100 / batch_time  # predictions per second
        
        results = {
            'avg_prediction_time_ms': avg_prediction_time * 1000,
            'max_prediction_time_ms': max_prediction_time * 1000,
            'percentile_95_time_ms': percentile_95_time * 1000,
            'latency_under_1ms_pct': latency_1ms * 100,
            'latency_under_10ms_pct': latency_10ms * 100,
            'latency_under_100ms_pct': latency_100ms * 100,
            'throughput_predictions_per_sec': throughput,
            'real_time_ready': latency_10ms > 0.95  # 95% under 10ms
        }
        
        return results
    
    def run_comprehensive_benchmark(self, n_samples: int = 5000) -> Dict[str, any]:
        """
        Run comprehensive benchmark suite
        """
        print("TAYLOR SERIES ANN COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        print(f"Target: {self.config.speedup_target}x speedup with {self.config.accuracy_target:.0%} accuracy retention")
        print("=" * 60)
        
        # Generate data
        features, targets, market_data = self.generate_synthetic_market_data(n_samples)
        
        # Run all benchmarks
        benchmark_results = {}
        
        # 1. Basic performance benchmark
        print("\\n1. BASIC PERFORMANCE BENCHMARK")
        print("-" * 40)
        basic_results = self.benchmark_against_sklearn_knn(features, targets)
        benchmark_results['basic_performance'] = basic_results
        
        # 2. Scalability benchmark
        print("\\n2. SCALABILITY BENCHMARK")
        print("-" * 40)
        scalability_results = self.scalability_benchmark(features, targets)
        benchmark_results['scalability'] = scalability_results
        
        # 3. Regime awareness benchmark
        print("\\n3. REGIME AWARENESS BENCHMARK")
        print("-" * 40)
        regime_results = self.regime_awareness_benchmark(features, targets, market_data)
        benchmark_results['regime_awareness'] = regime_results
        
        # 4. Memory efficiency benchmark
        print("\\n4. MEMORY EFFICIENCY BENCHMARK")
        print("-" * 40)
        memory_results = self.memory_efficiency_benchmark(features, targets)
        benchmark_results['memory_efficiency'] = memory_results
        
        # 5. Real-time performance benchmark
        print("\\n5. REAL-TIME PERFORMANCE BENCHMARK")
        print("-" * 40)
        realtime_results = self.real_time_performance_benchmark(features, targets)
        benchmark_results['real_time_performance'] = realtime_results
        
        # Generate summary
        summary = self.generate_benchmark_summary(benchmark_results)
        benchmark_results['summary'] = summary
        
        # Store results
        self.benchmark_results = benchmark_results
        
        return benchmark_results
    
    def generate_benchmark_summary(self, results: Dict[str, any]) -> Dict[str, any]:
        """
        Generate benchmark summary and target achievement analysis
        """
        basic = results['basic_performance']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'speedup_target': self.config.speedup_target,
                'accuracy_target': self.config.accuracy_target,
                'k_neighbors': self.config.k_neighbors,
                'taylor_order': self.config.taylor_order
            },
            'key_metrics': {
                'speedup_achieved': basic['speedup_vs_exact'],
                'accuracy_retention': basic['accuracy_retention'],
                'speedup_target_met': basic['target_speedup_achieved'],
                'accuracy_target_met': basic['target_accuracy_achieved']
            },
            'performance_summary': {
                'taylor_accuracy': basic['taylor_accuracy'],
                'exact_accuracy': basic['taylor_exact_accuracy'],
                'sklearn_accuracy': basic['sklearn_accuracy'],
                'taylor_vs_sklearn_speedup': basic['speedup_vs_sklearn'],
                'avg_prediction_time_ms': results['real_time_performance']['avg_prediction_time_ms']
            },
            'advanced_features': {
                'regime_improvement': results['regime_awareness']['regime_improvement'],
                'memory_savings_pct': results['memory_efficiency']['memory_efficiency_gain'],
                'real_time_ready': results['real_time_performance']['real_time_ready']
            }
        }
        
        # Overall system assessment
        targets_met = (basic['target_speedup_achieved'] and 
                      basic['target_accuracy_achieved'])
        
        summary['overall_assessment'] = {
            'targets_achieved': targets_met,
            'production_ready': (targets_met and 
                               results['real_time_performance']['real_time_ready']),
            'recommendation': 'APPROVED' if targets_met else 'NEEDS_OPTIMIZATION'
        }
        
        return summary
    
    def print_benchmark_report(self, results: Dict[str, any]):
        """
        Print comprehensive benchmark report
        """
        print("\\n" + "=" * 60)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 60)
        
        summary = results['summary']
        basic = results['basic_performance']
        
        # Key Results
        print("\\nKEY RESULTS:")
        print("-" * 20)
        print(f"Speedup Achieved: {summary['key_metrics']['speedup_achieved']:.1f}x")
        print(f"Speedup Target: {summary['config']['speedup_target']}x")
        print(f"Speedup Target Met: {'✓ YES' if summary['key_metrics']['speedup_target_met'] else '✗ NO'}")
        print()
        print(f"Accuracy Retention: {summary['key_metrics']['accuracy_retention']:.1%}")
        print(f"Accuracy Target: {summary['config']['accuracy_target']:.0%}")
        print(f"Accuracy Target Met: {'✓ YES' if summary['key_metrics']['accuracy_target_met'] else '✗ NO'}")
        
        # Performance Comparison
        print("\\nPERFORMANCE COMPARISON:")
        print("-" * 30)
        print(f"Taylor ANN Accuracy: {basic['taylor_accuracy']:.3f}")
        print(f"Exact Computation Accuracy: {basic['taylor_exact_accuracy']:.3f}")
        print(f"Sklearn KNN Accuracy: {basic['sklearn_accuracy']:.3f}")
        print(f"Taylor vs Sklearn Speedup: {basic['speedup_vs_sklearn']:.1f}x")
        
        # Real-time Performance
        realtime = results['real_time_performance']
        print("\\nREAL-TIME PERFORMANCE:")
        print("-" * 25)
        print(f"Avg Prediction Time: {realtime['avg_prediction_time_ms']:.2f}ms")
        print(f"95th Percentile Time: {realtime['percentile_95_time_ms']:.2f}ms")
        print(f"Predictions < 10ms: {realtime['latency_under_10ms_pct']:.1f}%")
        print(f"Throughput: {realtime['throughput_predictions_per_sec']:.0f} pred/sec")
        print(f"Real-time Ready: {'✓ YES' if realtime['real_time_ready'] else '✗ NO'}")
        
        # Advanced Features
        regime = results['regime_awareness']
        memory = results['memory_efficiency']
        print("\\nADVANCED FEATURES:")
        print("-" * 20)
        print(f"Regime Awareness Improvement: {regime['regime_improvement']:.3f}")
        print(f"Memory Efficiency Gain: {memory['memory_efficiency_gain']:.1f}%")
        print(f"Compression Ratio: {memory['compression_ratio']:.2f}")
        
        # Overall Assessment
        assessment = summary['overall_assessment']
        print("\\nOVERALL ASSESSMENT:")
        print("-" * 20)
        print(f"Research Targets Achieved: {'✓ YES' if assessment['targets_achieved'] else '✗ NO'}")
        print(f"Production Ready: {'✓ YES' if assessment['production_ready'] else '✗ NO'}")
        print(f"Recommendation: {assessment['recommendation']}")
        
        # Scalability Analysis
        if 'scalability' in results and results['scalability']['dataset_sizes']:
            print("\\nSCALABILITY ANALYSIS:")
            print("-" * 22)
            scalability = results['scalability']
            sizes = scalability['dataset_sizes']
            speedups = scalability['speedups_vs_exact']
            
            print(f"Dataset sizes tested: {sizes}")
            print(f"Average speedup across sizes: {np.mean(speedups):.1f}x")
            print(f"Speedup stability: {np.std(speedups):.2f} (lower is better)")
        
        print("\\n" + "=" * 60)
        
        if assessment['targets_achieved']:
            print("🎯 MISSION ACCOMPLISHED!")
            print("Taylor Series ANN system successfully achieved research targets!")
        else:
            print("⚠️  Research targets not fully met. Further optimization needed.")
        
        print("=" * 60)
    
    def save_benchmark_results(self, results: Dict[str, any], filename: str = None):
        """
        Save benchmark results to JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"taylor_ann_benchmark_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\\nBenchmark results saved to: {filename}")

def main():
    """
    Main benchmark execution
    """
    print("Initializing Taylor Series ANN Benchmark Suite...")
    
    # Create benchmark suite
    benchmark_suite = TaylorANNBenchmarkSuite(random_seed=42)
    
    # Run comprehensive benchmarks
    results = benchmark_suite.run_comprehensive_benchmark(n_samples=3000)
    
    # Print detailed report
    benchmark_suite.print_benchmark_report(results)
    
    # Save results
    benchmark_suite.save_benchmark_results(results)
    
    return results

if __name__ == "__main__":
    results = main()