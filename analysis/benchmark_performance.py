"""
Performance Benchmarking and Validation Module

This module provides comprehensive benchmarking for the advanced metrics
and risk analytics system to validate performance improvements and ensure
targets are met.
"""

import numpy as np
import pandas as pd
import time
import psutil
import logging
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps

# Import modules to benchmark
from analysis.metrics import (
    calculate_all_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_jensens_alpha,
    calculate_treynor_ratio,
    calculate_omega_ratio
)

from analysis.advanced_metrics import (
    AdvancedMetricsCalculator,
    calculate_sharpe_with_confidence,
    calculate_sortino_with_confidence
)

from analysis.risk_metrics import (
    RiskMetricsCalculator,
    calculate_var_historical,
    calculate_cvar_historical
)

from analysis.performance_optimizer import (
    MetricsOptimizer,
    optimized_sharpe_ratio,
    optimized_sortino_ratio,
    optimized_max_drawdown,
    performance_monitor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    function_name: str
    data_size: int
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: str = ""
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.function_name} (n={self.data_size}): {self.execution_time_ms:.2f}ms, {self.memory_usage_mb:.1f}MB"


@dataclass
class ComparisonResult:
    """Result of comparing two implementations"""
    function_name: str
    original_time_ms: float
    optimized_time_ms: float
    speedup_ratio: float
    memory_original_mb: float
    memory_optimized_mb: float
    accuracy_difference: float
    
    def __str__(self) -> str:
        return f"{self.function_name}: {self.speedup_ratio:.2f}x speedup, {self.accuracy_difference:.6f} accuracy diff"


class BenchmarkSuite:
    """Comprehensive benchmark suite for metrics calculations"""
    
    def __init__(self, warmup_iterations: int = 3, test_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results: List[BenchmarkResult] = []
        self.comparison_results: List[ComparisonResult] = []
        self.process = psutil.Process()
        
    def _measure_performance(self, func: Callable, *args, **kwargs) -> Tuple[float, float, float, Any]:
        """Measure performance of a function"""
        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except (ConnectionError, OSError, ValueError) as e:
                logger.error(f'Error occurred: {e}')
        
        # Clear caches
        if hasattr(func, 'cache_clear'):
            func.cache_clear()
        
        # Measure
        start_time = time.time()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()
        
        results = []
        for _ in range(self.test_iterations):
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                return 0.0, 0.0, 0.0, f"Error: {e}"
        
        end_time = time.time()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        cpu_after = self.process.cpu_percent()
        
        avg_time_ms = (end_time - start_time) * 1000 / self.test_iterations
        memory_usage_mb = memory_after - memory_before
        cpu_usage_percent = cpu_after - cpu_before
        
        return avg_time_ms, memory_usage_mb, cpu_usage_percent, results[0] if results else None
    
    def benchmark_function(
        self,
        func: Callable,
        function_name: str,
        data_size: int,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a single function"""
        logger.info(f"Benchmarking {function_name} with data size {data_size}")
        
        try:
            time_ms, memory_mb, cpu_percent, result = self._measure_performance(
                func, *args, **kwargs
            )
            
            if isinstance(result, str) and result.startswith("Error"):
                return BenchmarkResult(
                    function_name=function_name,
                    data_size=data_size,
                    execution_time_ms=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message=result
                )
            
            benchmark_result = BenchmarkResult(
                function_name=function_name,
                data_size=data_size,
                execution_time_ms=time_ms,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                success=True
            )
            
            self.results.append(benchmark_result)
            return benchmark_result
            
        except Exception as e:
            error_result = BenchmarkResult(
                function_name=function_name,
                data_size=data_size,
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def compare_implementations(
        self,
        original_func: Callable,
        optimized_func: Callable,
        function_name: str,
        data_size: int,
        *args,
        **kwargs
    ) -> ComparisonResult:
        """Compare original and optimized implementations"""
        logger.info(f"Comparing implementations for {function_name} with data size {data_size}")
        
        # Benchmark original
        original_time, original_memory, _, original_result = self._measure_performance(
            original_func, *args, **kwargs
        )
        
        # Benchmark optimized
        optimized_time, optimized_memory, _, optimized_result = self._measure_performance(
            optimized_func, *args, **kwargs
        )
        
        # Calculate speedup
        speedup_ratio = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Calculate accuracy difference
        accuracy_difference = 0.0
        if (isinstance(original_result, (int, float)) and 
            isinstance(optimized_result, (int, float))):
            if original_result != 0:
                accuracy_difference = abs(original_result - optimized_result) / abs(original_result)
            else:
                accuracy_difference = abs(optimized_result)
        
        comparison_result = ComparisonResult(
            function_name=function_name,
            original_time_ms=original_time,
            optimized_time_ms=optimized_time,
            speedup_ratio=speedup_ratio,
            memory_original_mb=original_memory,
            memory_optimized_mb=optimized_memory,
            accuracy_difference=accuracy_difference
        )
        
        self.comparison_results.append(comparison_result)
        return comparison_result
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        # Test data sizes
        data_sizes = [100, 500, 1000, 2500, 5000, 10000]
        
        # Generate test data
        test_data = {}
        for size in data_sizes:
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, size)
            equity_curve = np.cumprod(1 + returns) * 10000
            benchmark_returns = np.random.normal(0.0008, 0.015, size)
            
            test_data[size] = {
                'returns': returns,
                'equity_curve': equity_curve,
                'benchmark_returns': benchmark_returns
            }
        
        # Benchmark basic metrics
        self._benchmark_basic_metrics(test_data)
        
        # Benchmark advanced metrics
        self._benchmark_advanced_metrics(test_data)
        
        # Benchmark risk metrics
        self._benchmark_risk_metrics(test_data)
        
        # Benchmark optimized vs original
        self._benchmark_optimized_vs_original(test_data)
        
        # Generate summary
        summary = self._generate_benchmark_summary()
        
        logger.info("Comprehensive benchmark completed")
        return summary
    
    def _benchmark_basic_metrics(self, test_data: Dict[int, Dict[str, np.ndarray]]):
        """Benchmark basic metrics calculations"""
        logger.info("Benchmarking basic metrics")
        
        for size, data in test_data.items():
            returns = data['returns']
            equity_curve = data['equity_curve']
            benchmark_returns = data['benchmark_returns']
            
            # Sharpe ratio
            self.benchmark_function(
                calculate_sharpe_ratio,
                f"sharpe_ratio",
                size,
                returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            # Sortino ratio
            self.benchmark_function(
                calculate_sortino_ratio,
                f"sortino_ratio",
                size,
                returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            # Max drawdown
            self.benchmark_function(
                calculate_max_drawdown,
                f"max_drawdown",
                size,
                equity_curve
            )
            
            # Jensen's Alpha
            beta = 1.0  # Simplified for benchmarking
            self.benchmark_function(
                calculate_jensens_alpha,
                f"jensens_alpha",
                size,
                returns,
                benchmark_returns,
                beta,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            # Comprehensive metrics
            self.benchmark_function(
                calculate_all_metrics,
                f"all_metrics",
                size,
                equity_curve,
                returns,
                benchmark_returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
    
    def _benchmark_advanced_metrics(self, test_data: Dict[int, Dict[str, np.ndarray]]):
        """Benchmark advanced metrics calculations"""
        logger.info("Benchmarking advanced metrics")
        
        # Test smaller sizes for computationally intensive operations
        test_sizes = [100, 500, 1000, 2500]
        
        for size in test_sizes:
            if size not in test_data:
                continue
                
            returns = test_data[size]['returns']
            benchmark_returns = test_data[size]['benchmark_returns']
            
            # Advanced metrics calculator
            calculator = AdvancedMetricsCalculator(
                bootstrap_samples=100,  # Reduced for benchmarking
                max_workers=2
            )
            
            # VaR/CVaR metrics
            self.benchmark_function(
                calculator.calculate_var_cvar_metrics,
                f"var_cvar_metrics",
                size,
                returns,
                confidence_level=0.95,
                method="historical"
            )
            
            # Risk-adjusted metrics
            self.benchmark_function(
                calculator.calculate_risk_adjusted_metrics,
                f"risk_adjusted_metrics",
                size,
                returns,
                benchmark_returns,
                risk_free_rate=0.02,
                confidence_level=0.95,
                periods_per_year=252
            )
            
            # Confidence intervals (smaller sample for speed)
            if size <= 1000:
                self.benchmark_function(
                    calculate_sharpe_with_confidence,
                    f"sharpe_with_confidence",
                    size,
                    returns,
                    risk_free_rate=0.02,
                    periods_per_year=252,
                    confidence_level=0.95,
                    bootstrap_samples=50
                )
    
    def _benchmark_risk_metrics(self, test_data: Dict[int, Dict[str, np.ndarray]]):
        """Benchmark risk metrics calculations"""
        logger.info("Benchmarking risk metrics")
        
        for size, data in test_data.items():
            returns = data['returns']
            equity_curve = data['equity_curve']
            benchmark_returns = data['benchmark_returns']
            
            # Historical VaR
            self.benchmark_function(
                calculate_var_historical,
                f"var_historical",
                size,
                returns,
                confidence_level=0.95
            )
            
            # Historical CVaR
            self.benchmark_function(
                calculate_cvar_historical,
                f"cvar_historical",
                size,
                returns,
                confidence_level=0.95
            )
            
            # Comprehensive risk metrics (async)
            risk_calculator = RiskMetricsCalculator()
            
            async def calculate_risk_metrics_wrapper():
                return await risk_calculator.calculate_comprehensive_risk_metrics(
                    returns=returns,
                    equity_curve=equity_curve,
                    benchmark_returns=benchmark_returns,
                    periods_per_year=252
                )
            
            # Synchronous wrapper for benchmarking
            def sync_risk_metrics():
                return asyncio.run(calculate_risk_metrics_wrapper())
            
            self.benchmark_function(
                sync_risk_metrics,
                f"comprehensive_risk_metrics",
                size
            )
    
    def _benchmark_optimized_vs_original(self, test_data: Dict[int, Dict[str, np.ndarray]]):
        """Benchmark optimized vs original implementations"""
        logger.info("Benchmarking optimized vs original implementations")
        
        for size, data in test_data.items():
            returns = data['returns']
            equity_curve = data['equity_curve']
            
            # Sharpe ratio comparison
            self.compare_implementations(
                calculate_sharpe_ratio,
                optimized_sharpe_ratio,
                f"sharpe_ratio_comparison",
                size,
                returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            # Sortino ratio comparison
            self.compare_implementations(
                calculate_sortino_ratio,
                optimized_sortino_ratio,
                f"sortino_ratio_comparison",
                size,
                returns,
                risk_free_rate=0.02,
                periods_per_year=252
            )
            
            # Max drawdown comparison
            def original_max_drawdown_wrapper(equity_curve):
                max_dd, _, _ = calculate_max_drawdown(equity_curve)
                return max_dd
            
            self.compare_implementations(
                original_max_drawdown_wrapper,
                optimized_max_drawdown,
                f"max_drawdown_comparison",
                size,
                equity_curve
            )
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_iterations": self.test_iterations,
            "warmup_iterations": self.warmup_iterations,
            "total_tests": len(self.results),
            "successful_tests": len([r for r in self.results if r.success]),
            "failed_tests": len([r for r in self.results if not r.success]),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": psutil.sys.version
            }
        }
        
        # Performance targets validation
        targets = {
            "basic_metrics_target_ms": 5.0,
            "advanced_metrics_target_ms": 100.0,
            "risk_metrics_target_ms": 100.0
        }
        
        # Analyze results by category
        categories = {
            "basic_metrics": ["sharpe_ratio", "sortino_ratio", "max_drawdown", "jensens_alpha"],
            "advanced_metrics": ["var_cvar_metrics", "risk_adjusted_metrics", "sharpe_with_confidence"],
            "risk_metrics": ["var_historical", "cvar_historical", "comprehensive_risk_metrics"],
            "comprehensive": ["all_metrics"]
        }
        
        category_stats = {}
        for category, functions in categories.items():
            category_results = [r for r in self.results if any(func in r.function_name for func in functions)]
            
            if category_results:
                avg_time = np.mean([r.execution_time_ms for r in category_results if r.success])
                max_time = np.max([r.execution_time_ms for r in category_results if r.success])
                avg_memory = np.mean([r.memory_usage_mb for r in category_results if r.success])
                success_rate = len([r for r in category_results if r.success]) / len(category_results)
                
                category_stats[category] = {
                    "avg_time_ms": avg_time,
                    "max_time_ms": max_time,
                    "avg_memory_mb": avg_memory,
                    "success_rate": success_rate,
                    "test_count": len(category_results)
                }
        
        summary["category_stats"] = category_stats
        
        # Optimization analysis
        if self.comparison_results:
            optimization_stats = {
                "avg_speedup": np.mean([r.speedup_ratio for r in self.comparison_results]),
                "max_speedup": np.max([r.speedup_ratio for r in self.comparison_results]),
                "avg_accuracy_diff": np.mean([r.accuracy_difference for r in self.comparison_results]),
                "max_accuracy_diff": np.max([r.accuracy_difference for r in self.comparison_results])
            }
            summary["optimization_stats"] = optimization_stats
        
        # Performance target compliance
        target_compliance = {}
        for target_name, target_value in targets.items():
            category = target_name.replace("_target_ms", "")
            if category in category_stats:
                avg_time = category_stats[category]["avg_time_ms"]
                target_compliance[target_name] = {
                    "target_ms": target_value,
                    "actual_ms": avg_time,
                    "meets_target": avg_time <= target_value,
                    "margin_ms": target_value - avg_time
                }
        
        summary["target_compliance"] = target_compliance
        
        return summary
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        summary = self._generate_benchmark_summary()
        
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {summary['timestamp']}")
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Successful: {summary['successful_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append("")
        
        # System info
        report.append("SYSTEM INFORMATION")
        report.append("-" * 40)
        sys_info = summary["system_info"]
        report.append(f"CPU Cores: {sys_info['cpu_count']}")
        report.append(f"Memory: {sys_info['memory_gb']:.1f} GB")
        report.append("")
        
        # Category performance
        report.append("CATEGORY PERFORMANCE")
        report.append("-" * 40)
        for category, stats in summary["category_stats"].items():
            report.append(f"{category.upper()}:")
            report.append(f"  Average Time: {stats['avg_time_ms']:.2f}ms")
            report.append(f"  Maximum Time: {stats['max_time_ms']:.2f}ms")
            report.append(f"  Average Memory: {stats['avg_memory_mb']:.1f}MB")
            report.append(f"  Success Rate: {stats['success_rate']:.1%}")
            report.append(f"  Test Count: {stats['test_count']}")
            report.append("")
        
        # Optimization results
        if "optimization_stats" in summary:
            report.append("OPTIMIZATION RESULTS")
            report.append("-" * 40)
            opt_stats = summary["optimization_stats"]
            report.append(f"Average Speedup: {opt_stats['avg_speedup']:.2f}x")
            report.append(f"Maximum Speedup: {opt_stats['max_speedup']:.2f}x")
            report.append(f"Average Accuracy Difference: {opt_stats['avg_accuracy_diff']:.6f}")
            report.append(f"Maximum Accuracy Difference: {opt_stats['max_accuracy_diff']:.6f}")
            report.append("")
        
        # Target compliance
        if "target_compliance" in summary:
            report.append("TARGET COMPLIANCE")
            report.append("-" * 40)
            for target_name, compliance in summary["target_compliance"].items():
                status = "✓ PASS" if compliance["meets_target"] else "✗ FAIL"
                report.append(f"{target_name}: {status}")
                report.append(f"  Target: {compliance['target_ms']:.1f}ms")
                report.append(f"  Actual: {compliance['actual_ms']:.1f}ms")
                report.append(f"  Margin: {compliance['margin_ms']:.1f}ms")
                report.append("")
        
        # Top performers and problem areas
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            # Fastest functions
            fastest = sorted(successful_results, key=lambda x: x.execution_time_ms)[:5]
            report.append("TOP PERFORMERS (Fastest)")
            report.append("-" * 40)
            for result in fastest:
                report.append(f"{result.function_name} (n={result.data_size}): {result.execution_time_ms:.2f}ms")
            report.append("")
            
            # Slowest functions
            slowest = sorted(successful_results, key=lambda x: x.execution_time_ms, reverse=True)[:5]
            report.append("PERFORMANCE CONCERNS (Slowest)")
            report.append("-" * 40)
            for result in slowest:
                report.append(f"{result.function_name} (n={result.data_size}): {result.execution_time_ms:.2f}ms")
            report.append("")
        
        # Failed tests
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            report.append("FAILED TESTS")
            report.append("-" * 40)
            for result in failed_results:
                report.append(f"{result.function_name} (n={result.data_size}): {result.error_message}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        import json
        
        results_data = {
            "summary": self._generate_benchmark_summary(),
            "detailed_results": [
                {
                    "function_name": r.function_name,
                    "data_size": r.data_size,
                    "execution_time_ms": r.execution_time_ms,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ],
            "comparison_results": [
                {
                    "function_name": r.function_name,
                    "original_time_ms": r.original_time_ms,
                    "optimized_time_ms": r.optimized_time_ms,
                    "speedup_ratio": r.speedup_ratio,
                    "memory_original_mb": r.memory_original_mb,
                    "memory_optimized_mb": r.memory_optimized_mb,
                    "accuracy_difference": r.accuracy_difference
                }
                for r in self.comparison_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")


def run_full_benchmark():
    """Run full benchmark suite and generate report"""
    logger.info("Starting full benchmark suite...")
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(warmup_iterations=3, test_iterations=10)
    
    # Run comprehensive benchmark
    summary = benchmark.run_comprehensive_benchmark()
    
    # Generate and print report
    report = benchmark.generate_performance_report()
    print(report)
    
    # Save results
    benchmark.save_results("benchmark_results.json")
    
    # Return summary for programmatic use
    return summary


def validate_performance_targets():
    """Validate that performance targets are met"""
    logger.info("Validating performance targets...")
    
    # Quick benchmark with smaller dataset
    benchmark = BenchmarkSuite(warmup_iterations=1, test_iterations=5)
    
    # Generate test data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    equity_curve = np.cumprod(1 + returns) * 10000
    
    # Test critical functions
    targets = {
        "sharpe_ratio": 5.0,  # 5ms target
        "sortino_ratio": 5.0,
        "max_drawdown": 5.0,
        "all_metrics": 50.0,  # 50ms target for comprehensive metrics
    }
    
    results = {}
    
    # Test each function
    for func_name, target_ms in targets.items():
        if func_name == "sharpe_ratio":
            result = benchmark.benchmark_function(
                calculate_sharpe_ratio, func_name, 1000,
                returns, risk_free_rate=0.02, periods_per_year=252
            )
        elif func_name == "sortino_ratio":
            result = benchmark.benchmark_function(
                calculate_sortino_ratio, func_name, 1000,
                returns, risk_free_rate=0.02, periods_per_year=252
            )
        elif func_name == "max_drawdown":
            result = benchmark.benchmark_function(
                calculate_max_drawdown, func_name, 1000,
                equity_curve
            )
        elif func_name == "all_metrics":
            result = benchmark.benchmark_function(
                calculate_all_metrics, func_name, 1000,
                equity_curve, returns, risk_free_rate=0.02, periods_per_year=252
            )
        
        meets_target = result.success and result.execution_time_ms <= target_ms
        results[func_name] = {
            "target_ms": target_ms,
            "actual_ms": result.execution_time_ms,
            "meets_target": meets_target,
            "success": result.success
        }
        
        status = "✓ PASS" if meets_target else "✗ FAIL"
        logger.info(f"{func_name}: {status} ({result.execution_time_ms:.2f}ms / {target_ms}ms)")
    
    # Overall validation
    all_pass = all(r["meets_target"] for r in results.values())
    logger.info(f"Overall validation: {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    return all_pass, results


if __name__ == "__main__":
    # Run full benchmark
    summary = run_full_benchmark()
    
    # Validate targets
    print("\n" + "="*80)
    print("PERFORMANCE TARGET VALIDATION")
    print("="*80)
    
    success, validation_results = validate_performance_targets()
    
    if success:
        print("\n✓ All performance targets met!")
        exit(0)
    else:
        print("\n✗ Some performance targets not met!")
        exit(1)