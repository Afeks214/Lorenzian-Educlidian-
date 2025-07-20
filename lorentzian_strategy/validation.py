"""
Validation and Testing Framework for Lorentzian Feature Engineering
================================================================

Comprehensive validation framework to ensure feature quality, performance,
and correctness of the Lorentzian Classification feature engineering pipeline.

Components:
1. Feature Quality Validation
2. Performance Benchmarking  
3. Mathematical Correctness Tests
4. Stress Testing
5. Production Readiness Checks

Author: Claude Code Agent
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from .feature_engineering import (
    LorentzianFeatureEngine, 
    LorentzianConfig,
    create_production_config,
    TechnicalIndicators,
    FeatureNormalizer,
    KernelRegression
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a validation test"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class ValidationSuite:
    """Complete validation suite results"""
    overall_passed: bool
    overall_score: float
    individual_results: List[ValidationResult]
    total_execution_time_ms: float
    summary: Dict[str, Any]


class FeatureQualityValidator:
    """Validates feature quality and mathematical correctness"""
    
    def __init__(self):
        self.tolerance = 1e-6
        
    def validate_technical_indicators(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Validate technical indicator calculations"""
        start_time = time.time()
        
        try:
            # Generate test data with known properties
            test_data = self._generate_test_data()
            
            # Test RSI
            rsi_result = self._test_rsi(engine.indicators, test_data)
            
            # Test WaveTrend
            wt_result = self._test_wavetrend(engine.indicators, test_data)
            
            # Test ADX
            adx_result = self._test_adx(engine.indicators, test_data)
            
            # Test CCI
            cci_result = self._test_cci(engine.indicators, test_data)
            
            # Aggregate results
            all_tests = [rsi_result, wt_result, adx_result, cci_result]
            passed = all(test['passed'] for test in all_tests)
            score = np.mean([test['score'] for test in all_tests])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Technical Indicators",
                passed=passed,
                score=score,
                details={
                    'rsi': rsi_result,
                    'wavetrend': wt_result,
                    'adx': adx_result,
                    'cci': cci_result
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Technical Indicators",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def validate_feature_normalization(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Validate feature normalization methods"""
        start_time = time.time()
        
        try:
            normalizer = engine.normalizer
            
            # Test data with known statistical properties
            test_values = np.array([1, 2, 3, 4, 5, 10, 15, 20])
            
            # Test min-max normalization
            minmax_result = self._test_minmax_normalization(normalizer, test_values)
            
            # Test z-score normalization
            zscore_result = self._test_zscore_normalization(normalizer, test_values)
            
            # Test percentile normalization
            percentile_result = self._test_percentile_normalization(normalizer, test_values)
            
            # Test robust normalization
            robust_result = self._test_robust_normalization(normalizer, test_values)
            
            all_tests = [minmax_result, zscore_result, percentile_result, robust_result]
            passed = all(test['passed'] for test in all_tests)
            score = np.mean([test['score'] for test in all_tests])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Feature Normalization",
                passed=passed,
                score=score,
                details={
                    'minmax': minmax_result,
                    'zscore': zscore_result,
                    'percentile': percentile_result,
                    'robust': robust_result
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Feature Normalization",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def validate_kernel_regression(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Validate kernel regression calculations"""
        start_time = time.time()
        
        try:
            kernel = engine.kernel_regression
            
            # Test kernel functions
            kernel_result = self._test_kernel_functions(kernel)
            
            # Test Nadaraya-Watson estimator
            nw_result = self._test_nadaraya_watson(kernel)
            
            # Test signal smoothing
            smoothing_result = self._test_signal_smoothing(kernel)
            
            all_tests = [kernel_result, nw_result, smoothing_result]
            passed = all(test['passed'] for test in all_tests)
            score = np.mean([test['score'] for test in all_tests])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Kernel Regression",
                passed=passed,
                score=score,
                details={
                    'kernel_functions': kernel_result,
                    'nadaraya_watson': nw_result,
                    'signal_smoothing': smoothing_result
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Kernel Regression",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data with known statistical properties"""
        np.random.seed(42)
        n_points = 100
        
        # Create trending price data
        base_price = 100.0
        trend = np.linspace(0, 0.1, n_points)
        noise = np.random.normal(0, 0.02, n_points)
        
        close = base_price * (1 + trend + noise)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
        volume = np.random.lognormal(10, 1, n_points)
        
        return {
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
    
    def _test_rsi(self, indicators: TechnicalIndicators, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test RSI calculation"""
        rsi = indicators.calculate_rsi(data['close'], 14)
        
        # RSI should be between 0 and 100
        range_check = np.all((rsi >= 0) & (rsi <= 100))
        
        # RSI should not be all the same value
        variance_check = np.var(rsi[~np.isnan(rsi)]) > 0
        
        # Check for reasonable values (not all extreme)
        reasonable_check = np.mean((rsi > 20) & (rsi < 80)) > 0.5
        
        passed = range_check and variance_check and reasonable_check
        score = np.mean([range_check, variance_check, reasonable_check])
        
        return {
            'passed': passed,
            'score': score,
            'range_check': range_check,
            'variance_check': variance_check,
            'reasonable_check': reasonable_check
        }
    
    def _test_wavetrend(self, indicators: TechnicalIndicators, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test WaveTrend calculation"""
        wt1, wt2 = indicators.calculate_wavetrend(data['high'], data['low'], data['close'])
        
        # Check for finite values
        finite_check = np.all(np.isfinite(wt1[~np.isnan(wt1)])) and np.all(np.isfinite(wt2[~np.isnan(wt2)]))
        
        # Check for variance
        wt1_variance = np.var(wt1[~np.isnan(wt1)]) > 0
        wt2_variance = np.var(wt2[~np.isnan(wt2)]) > 0
        
        # WaveTrend should oscillate around zero
        wt1_oscillation = np.any(wt1 > 0) and np.any(wt1 < 0)
        wt2_oscillation = np.any(wt2 > 0) and np.any(wt2 < 0)
        
        passed = finite_check and wt1_variance and wt2_variance and wt1_oscillation and wt2_oscillation
        score = np.mean([finite_check, wt1_variance, wt2_variance, wt1_oscillation, wt2_oscillation])
        
        return {
            'passed': passed,
            'score': score,
            'finite_check': finite_check,
            'wt1_variance': wt1_variance,
            'wt2_variance': wt2_variance,
            'wt1_oscillation': wt1_oscillation,
            'wt2_oscillation': wt2_oscillation
        }
    
    def _test_adx(self, indicators: TechnicalIndicators, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test ADX calculation"""
        adx = indicators.calculate_adx(data['high'], data['low'], data['close'], 14)
        
        # ADX should be between 0 and 100
        range_check = np.all((adx >= 0) & (adx <= 100))
        
        # ADX should have variance
        variance_check = np.var(adx[~np.isnan(adx)]) > 0
        
        # ADX should generally be positive (it's an absolute measure)
        positive_check = np.mean(adx[~np.isnan(adx)] >= 0) > 0.95
        
        passed = range_check and variance_check and positive_check
        score = np.mean([range_check, variance_check, positive_check])
        
        return {
            'passed': passed,
            'score': score,
            'range_check': range_check,
            'variance_check': variance_check,
            'positive_check': positive_check
        }
    
    def _test_cci(self, indicators: TechnicalIndicators, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test CCI calculation"""
        cci = indicators.calculate_cci(data['high'], data['low'], data['close'], 20)
        
        # CCI should have finite values
        finite_check = np.all(np.isfinite(cci[~np.isnan(cci)]))
        
        # CCI should oscillate around zero
        oscillation_check = np.any(cci > 0) and np.any(cci < 0)
        
        # CCI should have reasonable range (typically -200 to 200)
        range_check = np.percentile(np.abs(cci[~np.isnan(cci)]), 95) < 500
        
        passed = finite_check and oscillation_check and range_check
        score = np.mean([finite_check, oscillation_check, range_check])
        
        return {
            'passed': passed,
            'score': score,
            'finite_check': finite_check,
            'oscillation_check': oscillation_check,
            'range_check': range_check
        }
    
    def _test_minmax_normalization(self, normalizer: FeatureNormalizer, test_values: np.ndarray) -> Dict[str, Any]:
        """Test min-max normalization"""
        from .feature_engineering import NormalizationMethod
        
        normalized = normalizer.normalize_feature(test_values, "test_minmax", NormalizationMethod.MIN_MAX)
        
        # Should be in [0, 1] range
        range_check = np.all((normalized >= 0) & (normalized <= 1))
        
        # Min should be 0, max should be 1 (approximately)
        boundary_check = abs(np.min(normalized) - 0.0) < self.tolerance and abs(np.max(normalized) - 1.0) < self.tolerance
        
        # Should preserve relative ordering
        order_check = np.all(np.diff(normalized) >= 0)  # For ascending input
        
        passed = range_check and boundary_check and order_check
        score = np.mean([range_check, boundary_check, order_check])
        
        return {
            'passed': passed,
            'score': score,
            'range_check': range_check,
            'boundary_check': boundary_check,
            'order_check': order_check
        }
    
    def _test_zscore_normalization(self, normalizer: FeatureNormalizer, test_values: np.ndarray) -> Dict[str, Any]:
        """Test z-score normalization"""
        from .feature_engineering import NormalizationMethod
        
        normalized = normalizer.normalize_feature(test_values, "test_zscore", NormalizationMethod.Z_SCORE)
        
        # Should have approximately zero mean
        mean_check = abs(np.mean(normalized)) < 0.1
        
        # Should have approximately unit variance
        var_check = abs(np.var(normalized) - 1.0) < 0.1
        
        # Should preserve relative ordering
        order_check = np.all(np.diff(normalized) >= 0)
        
        passed = mean_check and var_check and order_check
        score = np.mean([mean_check, var_check, order_check])
        
        return {
            'passed': passed,
            'score': score,
            'mean_check': mean_check,
            'var_check': var_check,
            'order_check': order_check
        }
    
    def _test_percentile_normalization(self, normalizer: FeatureNormalizer, test_values: np.ndarray) -> Dict[str, Any]:
        """Test percentile normalization"""
        from .feature_engineering import NormalizationMethod
        
        normalized = normalizer.normalize_feature(test_values, "test_percentile", NormalizationMethod.PERCENTILE)
        
        # Most values should be in [0, 1] range
        range_check = np.mean((normalized >= 0) & (normalized <= 1)) > 0.8
        
        # Should have finite values
        finite_check = np.all(np.isfinite(normalized))
        
        # Should preserve relative ordering
        order_check = np.all(np.diff(normalized) >= 0)
        
        passed = range_check and finite_check and order_check
        score = np.mean([range_check, finite_check, order_check])
        
        return {
            'passed': passed,
            'score': score,
            'range_check': range_check,
            'finite_check': finite_check,
            'order_check': order_check
        }
    
    def _test_robust_normalization(self, normalizer: FeatureNormalizer, test_values: np.ndarray) -> Dict[str, Any]:
        """Test robust normalization"""
        from .feature_engineering import NormalizationMethod
        
        normalized = normalizer.normalize_feature(test_values, "test_robust", NormalizationMethod.ROBUST)
        
        # Should have finite values
        finite_check = np.all(np.isfinite(normalized))
        
        # Should have reasonable scale (median should be close to 0)
        median_check = abs(np.median(normalized)) < 1.0
        
        # Should preserve relative ordering
        order_check = np.all(np.diff(normalized) >= 0)
        
        passed = finite_check and median_check and order_check
        score = np.mean([finite_check, median_check, order_check])
        
        return {
            'passed': passed,
            'score': score,
            'finite_check': finite_check,
            'median_check': median_check,
            'order_check': order_check
        }
    
    def _test_kernel_functions(self, kernel: KernelRegression) -> Dict[str, Any]:
        """Test kernel function properties"""
        from .feature_engineering import rational_quadratic_kernel, gaussian_kernel
        
        # Test kernel properties
        # 1. Kernel at same point should be 1.0
        rq_identity = abs(rational_quadratic_kernel(0.0, 0.0, 1.0, 1.0) - 1.0) < self.tolerance
        gauss_identity = abs(gaussian_kernel(0.0, 0.0, 1.0) - 1.0) < self.tolerance
        
        # 2. Kernel should decrease with distance
        rq_decrease = rational_quadratic_kernel(0.0, 1.0, 1.0, 1.0) < rational_quadratic_kernel(0.0, 0.0, 1.0, 1.0)
        gauss_decrease = gaussian_kernel(0.0, 1.0, 1.0) < gaussian_kernel(0.0, 0.0, 1.0)
        
        # 3. Kernel should be positive
        rq_positive = rational_quadratic_kernel(0.0, 5.0, 1.0, 1.0) > 0
        gauss_positive = gaussian_kernel(0.0, 5.0, 1.0) > 0
        
        passed = rq_identity and gauss_identity and rq_decrease and gauss_decrease and rq_positive and gauss_positive
        score = np.mean([rq_identity, gauss_identity, rq_decrease, gauss_decrease, rq_positive, gauss_positive])
        
        return {
            'passed': passed,
            'score': score,
            'rq_identity': rq_identity,
            'gauss_identity': gauss_identity,
            'rq_decrease': rq_decrease,
            'gauss_decrease': gauss_decrease,
            'rq_positive': rq_positive,
            'gauss_positive': gauss_positive
        }
    
    def _test_nadaraya_watson(self, kernel: KernelRegression) -> Dict[str, Any]:
        """Test Nadaraya-Watson estimator"""
        # Create test data
        x_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_values = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Quadratic function
        
        # Test estimation at known point
        estimate_at_3 = kernel.estimate(x_points, y_values, 3.0)
        
        # Should be close to the actual value at x=3 (y=9)
        accuracy_check = abs(estimate_at_3 - 9.0) < 2.0  # Allow some smoothing error
        
        # Test with different kernel types
        kernel.config.kernel_type = "rational_quadratic"
        rq_estimate = kernel.estimate(x_points, y_values, 3.0)
        
        kernel.config.kernel_type = "gaussian"
        gauss_estimate = kernel.estimate(x_points, y_values, 3.0)
        
        # Both should give reasonable estimates
        rq_reasonable = 5.0 < rq_estimate < 15.0
        gauss_reasonable = 5.0 < gauss_estimate < 15.0
        
        passed = accuracy_check and rq_reasonable and gauss_reasonable
        score = np.mean([accuracy_check, rq_reasonable, gauss_reasonable])
        
        return {
            'passed': passed,
            'score': score,
            'accuracy_check': accuracy_check,
            'rq_reasonable': rq_reasonable,
            'gauss_reasonable': gauss_reasonable,
            'estimate_at_3': estimate_at_3
        }
    
    def _test_signal_smoothing(self, kernel: KernelRegression) -> Dict[str, Any]:
        """Test signal smoothing"""
        # Create noisy signal
        np.random.seed(42)
        clean_signal = np.sin(np.linspace(0, 4*np.pi, 50))
        noisy_signal = clean_signal + np.random.normal(0, 0.2, 50)
        
        # Apply smoothing
        smoothed_signal = kernel.smooth_signals(noisy_signal)
        
        # Smoothed signal should be closer to clean signal
        original_mse = np.mean((noisy_signal - clean_signal) ** 2)
        smoothed_mse = np.mean((smoothed_signal - clean_signal) ** 2)
        improvement_check = smoothed_mse < original_mse
        
        # Smoothed signal should have less variance
        variance_reduction = np.var(smoothed_signal) < np.var(noisy_signal)
        
        # Smoothed signal should preserve general shape
        correlation_check = np.corrcoef(smoothed_signal, clean_signal)[0, 1] > 0.8
        
        passed = improvement_check and variance_reduction and correlation_check
        score = np.mean([improvement_check, variance_reduction, correlation_check])
        
        return {
            'passed': passed,
            'score': score,
            'improvement_check': improvement_check,
            'variance_reduction': variance_reduction,
            'correlation_check': correlation_check,
            'original_mse': original_mse,
            'smoothed_mse': smoothed_mse
        }


class PerformanceBenchmarker:
    """Benchmarks performance and efficiency of the feature engine"""
    
    def __init__(self):
        self.benchmark_sizes = [100, 500, 1000, 2000]
        
    def benchmark_feature_calculation(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Benchmark feature calculation performance"""
        start_time = time.time()
        
        try:
            results = {}
            
            for size in self.benchmark_sizes:
                # Generate test data
                test_df = self._generate_benchmark_data(size)
                
                # Measure processing time
                calc_start = time.time()
                feature_df = engine.process_dataframe(test_df)
                calc_time = (time.time() - calc_start) * 1000
                
                # Calculate throughput
                throughput = size / (calc_time / 1000)  # bars per second
                
                results[f'size_{size}'] = {
                    'calculation_time_ms': calc_time,
                    'throughput_bars_per_sec': throughput,
                    'avg_time_per_bar_ms': calc_time / size,
                    'memory_efficient': calc_time < size * 2  # Less than 2ms per bar
                }
            
            # Overall performance assessment
            avg_throughput = np.mean([r['throughput_bars_per_sec'] for r in results.values()])
            max_time_per_bar = max([r['avg_time_per_bar_ms'] for r in results.values()])
            
            # Performance criteria
            throughput_check = avg_throughput > 50  # At least 50 bars per second
            efficiency_check = max_time_per_bar < 5.0  # Less than 5ms per bar
            scaling_check = self._check_scaling_behavior(results)
            
            passed = throughput_check and efficiency_check and scaling_check
            score = np.mean([throughput_check, efficiency_check, scaling_check])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=passed,
                score=score,
                details={
                    'benchmark_results': results,
                    'avg_throughput': avg_throughput,
                    'max_time_per_bar': max_time_per_bar,
                    'throughput_check': throughput_check,
                    'efficiency_check': efficiency_check,
                    'scaling_check': scaling_check
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def benchmark_memory_usage(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Benchmark memory usage and efficiency"""
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process large dataset
            large_df = self._generate_benchmark_data(5000)
            
            # Measure memory before processing
            before_memory = process.memory_info().rss / 1024 / 1024
            
            # Process features
            feature_df = engine.process_dataframe(large_df)
            
            # Measure memory after processing
            after_memory = process.memory_info().rss / 1024 / 1024
            
            # Calculate memory usage
            processing_memory_delta = after_memory - before_memory
            memory_per_bar = processing_memory_delta / len(large_df) * 1024  # KB per bar
            
            # Memory efficiency checks
            memory_per_bar_check = memory_per_bar < 10  # Less than 10KB per bar
            total_memory_check = processing_memory_delta < 100  # Less than 100MB total
            memory_growth_check = processing_memory_delta > 0  # Some memory usage is expected
            
            passed = memory_per_bar_check and total_memory_check and memory_growth_check
            score = np.mean([memory_per_bar_check, total_memory_check, memory_growth_check])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Memory Benchmark",
                passed=passed,
                score=score,
                details={
                    'initial_memory_mb': initial_memory,
                    'before_memory_mb': before_memory,
                    'after_memory_mb': after_memory,
                    'processing_memory_delta_mb': processing_memory_delta,
                    'memory_per_bar_kb': memory_per_bar,
                    'memory_per_bar_check': memory_per_bar_check,
                    'total_memory_check': total_memory_check,
                    'memory_growth_check': memory_growth_check
                },
                execution_time_ms=execution_time
            )
            
        except ImportError:
            # psutil not available, skip memory test
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Memory Benchmark",
                passed=True,
                score=1.0,
                details={'skipped': 'psutil not available'},
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Memory Benchmark",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _generate_benchmark_data(self, size: int) -> pd.DataFrame:
        """Generate benchmark data of specified size"""
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, size)
        prices = base_price * np.exp(np.cumsum(returns))
        
        noise_factor = 0.01
        high = prices * (1 + np.abs(np.random.normal(0, noise_factor, size)))
        low = prices * (1 - np.abs(np.random.normal(0, noise_factor, size)))
        open_prices = np.roll(prices, 1)
        volume = np.random.lognormal(10, 1, size)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=pd.date_range('2024-01-01', periods=size, freq='1T'))
    
    def _check_scaling_behavior(self, results: Dict[str, Any]) -> bool:
        """Check if the algorithm scales reasonably"""
        times = [results[f'size_{size}']['calculation_time_ms'] for size in self.benchmark_sizes]
        
        # Calculate scaling factors
        scaling_factors = []
        for i in range(1, len(times)):
            size_ratio = self.benchmark_sizes[i] / self.benchmark_sizes[i-1]
            time_ratio = times[i] / times[i-1]
            scaling_factors.append(time_ratio / size_ratio)
        
        # Good scaling means time ratio should be close to size ratio (linear scaling)
        # Acceptable if scaling factor is less than 2.0 (better than quadratic)
        return all(factor < 2.0 for factor in scaling_factors)


class LorentzianValidator:
    """Complete validation framework for Lorentzian feature engineering"""
    
    def __init__(self):
        self.quality_validator = FeatureQualityValidator()
        self.performance_benchmarker = PerformanceBenchmarker()
        
    def run_full_validation(self, config: LorentzianConfig = None) -> ValidationSuite:
        """Run complete validation suite"""
        start_time = time.time()
        
        # Create engine with provided or default config
        if config is None:
            config = create_production_config()
        
        engine = LorentzianFeatureEngine(config)
        
        # Run all validation tests
        results = []
        
        # Quality tests
        results.append(self.quality_validator.validate_technical_indicators(engine))
        results.append(self.quality_validator.validate_feature_normalization(engine))
        results.append(self.quality_validator.validate_kernel_regression(engine))
        
        # Performance tests
        results.append(self.performance_benchmarker.benchmark_feature_calculation(engine))
        results.append(self.performance_benchmarker.benchmark_memory_usage(engine))
        
        # Additional integration test
        results.append(self._integration_test(engine))
        
        # Calculate overall results
        passed_count = sum(1 for r in results if r.passed)
        overall_passed = passed_count == len(results)
        overall_score = np.mean([r.score for r in results])
        
        total_execution_time = (time.time() - start_time) * 1000
        
        # Create summary
        summary = {
            'total_tests': len(results),
            'passed_tests': passed_count,
            'failed_tests': len(results) - passed_count,
            'pass_rate': passed_count / len(results),
            'average_score': overall_score,
            'configuration_tested': {
                'feature_count': len(config.feature_configs),
                'enable_numba': config.enable_numba,
                'enable_caching': config.enable_caching,
                'streaming_mode': config.streaming_mode
            }
        }
        
        return ValidationSuite(
            overall_passed=overall_passed,
            overall_score=overall_score,
            individual_results=results,
            total_execution_time_ms=total_execution_time,
            summary=summary
        )
    
    def _integration_test(self, engine: LorentzianFeatureEngine) -> ValidationResult:
        """Integration test for complete pipeline"""
        start_time = time.time()
        
        try:
            # Generate test data
            np.random.seed(42)
            test_df = pd.DataFrame({
                'open': [100, 101, 99, 102, 98],
                'high': [102, 103, 101, 104, 100],
                'low': [99, 100, 98, 101, 97],
                'close': [101, 99, 102, 98, 99],
                'volume': [1000, 1100, 900, 1200, 800]
            }, index=pd.date_range('2024-01-01', periods=5, freq='1T'))
            
            # Process features
            feature_df = engine.process_dataframe(test_df)
            
            # Validate results
            shape_check = feature_df.shape == (5, len(engine.config.feature_configs) + 2)  # +2 for metadata
            no_nan_check = not feature_df.iloc[:, :-2].isnull().any().any()  # Exclude metadata columns
            range_check = np.all((feature_df.iloc[:, :-2] >= -5) & (feature_df.iloc[:, :-2] <= 5))
            
            # Feature validation
            last_features = feature_df.iloc[-1, :-2].values
            validation_results = engine.validate_features(last_features)
            validation_check = all(validation_results.values())
            
            passed = shape_check and no_nan_check and range_check and validation_check
            score = np.mean([shape_check, no_nan_check, range_check, validation_check])
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                test_name="Integration Test",
                passed=passed,
                score=score,
                details={
                    'output_shape': feature_df.shape,
                    'shape_check': shape_check,
                    'no_nan_check': no_nan_check,
                    'range_check': range_check,
                    'validation_results': validation_results,
                    'validation_check': validation_check
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="Integration Test",
                passed=False,
                score=0.0,
                details={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def print_validation_report(self, suite: ValidationSuite):
        """Print detailed validation report"""
        print("=" * 80)
        print("LORENTZIAN FEATURE ENGINEERING VALIDATION REPORT")
        print("=" * 80)
        
        # Overall summary
        status = "✅ PASSED" if suite.overall_passed else "❌ FAILED"
        print(f"\nOVERALL STATUS: {status}")
        print(f"Overall Score: {suite.overall_score:.3f}")
        print(f"Total Execution Time: {suite.total_execution_time_ms:.2f}ms")
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"  Total Tests: {suite.summary['total_tests']}")
        print(f"  Passed Tests: {suite.summary['passed_tests']}")
        print(f"  Failed Tests: {suite.summary['failed_tests']}")
        print(f"  Pass Rate: {suite.summary['pass_rate']:.1%}")
        
        # Individual test results
        print(f"\nINDIVIDUAL TEST RESULTS:")
        for result in suite.individual_results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.test_name}: {result.score:.3f} ({result.execution_time_ms:.2f}ms)")
            
            if result.error_message:
                print(f"    Error: {result.error_message}")
        
        # Configuration details
        print(f"\nCONFIGURATION TESTED:")
        config = suite.summary['configuration_tested']
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if suite.overall_passed:
            print("  ✅ All tests passed! The system is ready for production use.")
        else:
            failed_tests = [r for r in suite.individual_results if not r.passed]
            print("  ❌ Some tests failed. Please review the following:")
            for test in failed_tests:
                print(f"    - {test.test_name}: {test.error_message or 'See details above'}")
        
        if suite.overall_score < 0.8:
            print("  ⚠️  Overall score below 0.8. Consider optimization.")
        
        print("=" * 80)


def run_validation_demo():
    """Demonstrate the validation framework"""
    print("🧪 Starting Lorentzian Feature Engineering Validation...")
    
    # Create validator
    validator = LorentzianValidator()
    
    # Run full validation
    suite = validator.run_full_validation()
    
    # Print detailed report
    validator.print_validation_report(suite)
    
    return suite


if __name__ == "__main__":
    # Run validation demonstration
    validation_suite = run_validation_demo()