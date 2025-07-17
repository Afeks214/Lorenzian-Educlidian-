"""
GrandModel Performance Testing Module
====================================

Comprehensive performance testing and benchmarking framework for the GrandModel
trading system, including latency testing, throughput measurement, and load testing.
"""

import asyncio
import logging
import time
import statistics
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from memory_profiler import profile
import sys
import tracemalloc


class PerformanceTestType(Enum):
    """Performance test type enumeration"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    LOAD = "load"
    STRESS = "stress"
    ENDURANCE = "endurance"
    MEMORY = "memory"
    CPU = "cpu"
    SCALABILITY = "scalability"


class PerformanceMetric(Enum):
    """Performance metric enumeration"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    MEMORY_PEAK = "memory_peak"
    GC_TIME = "gc_time"


@dataclass
class PerformanceResult:
    """Performance test result"""
    test_id: str
    test_name: str
    test_type: PerformanceTestType
    metrics: Dict[PerformanceMetric, float]
    duration: float
    success: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceBenchmark:
    """Performance benchmark configuration"""
    name: str
    metric: PerformanceMetric
    target_value: float
    threshold_warning: float
    threshold_critical: float
    higher_is_better: bool = False
    unit: str = "ms"


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    name: str
    target_function: Callable
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 0
    ramp_down_seconds: int = 0
    think_time_seconds: float = 0.1
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PerformanceTester:
    """
    Comprehensive performance testing framework
    
    Features:
    - Latency and throughput testing
    - Load and stress testing
    - Memory and CPU profiling
    - Scalability testing
    - Performance regression detection
    - Real-time monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.results: List[PerformanceResult] = []
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.baseline_results: Dict[str, PerformanceResult] = {}
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.system_metrics = []
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "performance_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize benchmarks
        self._initialize_benchmarks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("PerformanceTester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_benchmarks(self):
        """Initialize default performance benchmarks"""
        self.benchmarks = {
            "var_calculation": PerformanceBenchmark(
                name="VaR Calculation",
                metric=PerformanceMetric.RESPONSE_TIME,
                target_value=5.0,  # 5ms target
                threshold_warning=10.0,
                threshold_critical=25.0,
                unit="ms"
            ),
            "correlation_update": PerformanceBenchmark(
                name="Correlation Matrix Update",
                metric=PerformanceMetric.RESPONSE_TIME,
                target_value=2.0,  # 2ms target
                threshold_warning=5.0,
                threshold_critical=10.0,
                unit="ms"
            ),
            "signal_generation": PerformanceBenchmark(
                name="Signal Generation",
                metric=PerformanceMetric.RESPONSE_TIME,
                target_value=1.0,  # 1ms target
                threshold_warning=2.0,
                threshold_critical=5.0,
                unit="ms"
            ),
            "throughput_test": PerformanceBenchmark(
                name="System Throughput",
                metric=PerformanceMetric.THROUGHPUT,
                target_value=1000.0,  # 1000 ops/sec
                threshold_warning=500.0,
                threshold_critical=100.0,
                higher_is_better=True,
                unit="ops/sec"
            ),
            "memory_usage": PerformanceBenchmark(
                name="Memory Usage",
                metric=PerformanceMetric.MEMORY_USAGE,
                target_value=512.0,  # 512 MB
                threshold_warning=1024.0,
                threshold_critical=2048.0,
                unit="MB"
            ),
            "cpu_usage": PerformanceBenchmark(
                name="CPU Usage",
                metric=PerformanceMetric.CPU_USAGE,
                target_value=50.0,  # 50%
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit="%"
            )
        }
    
    def register_benchmark(self, name: str, benchmark: PerformanceBenchmark):
        """Register a performance benchmark"""
        self.benchmarks[name] = benchmark
        self.logger.info(f"Registered benchmark: {name}")
    
    def start_monitoring(self, interval: float = 0.1):
        """Start system performance monitoring"""
        self.monitoring_active = True
        self.system_metrics = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    timestamp = time.time()
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    self.system_metrics.append({
                        "timestamp": timestamp,
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_mb": memory.used / (1024 * 1024),
                        "memory_available_mb": memory.available / (1024 * 1024)
                    })
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    async def test_latency(self, name: str, target_function: Callable, 
                          iterations: int = 1000, warmup: int = 100) -> PerformanceResult:
        """
        Test function latency
        
        Args:
            name: Test name
            target_function: Function to test
            iterations: Number of test iterations
            warmup: Number of warmup iterations
            
        Returns:
            PerformanceResult object
        """
        self.logger.info(f"Starting latency test: {name}")
        
        # Warmup
        for _ in range(warmup):
            try:
                if asyncio.iscoroutinefunction(target_function):
                    await target_function()
                else:
                    target_function()
            except Exception as e:
                self.logger.warning(f"Warmup error: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Actual test
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                iter_start = time.perf_counter()
                
                if asyncio.iscoroutinefunction(target_function):
                    await target_function()
                else:
                    target_function()
                
                iter_end = time.perf_counter()
                latency = (iter_end - iter_start) * 1000  # Convert to ms
                latencies.append(latency)
                
            except Exception as e:
                errors += 1
                self.logger.error(f"Test iteration {i} failed: {e}")
        
        end_time = time.time()
        
        # Calculate metrics
        if latencies:
            metrics = {
                PerformanceMetric.RESPONSE_TIME: statistics.mean(latencies),
                PerformanceMetric.LATENCY_P50: statistics.median(latencies),
                PerformanceMetric.LATENCY_P95: np.percentile(latencies, 95),
                PerformanceMetric.LATENCY_P99: np.percentile(latencies, 99),
                PerformanceMetric.ERROR_RATE: (errors / iterations) * 100
            }
        else:
            metrics = {
                PerformanceMetric.RESPONSE_TIME: float('inf'),
                PerformanceMetric.ERROR_RATE: 100.0
            }
        
        result = PerformanceResult(
            test_id=f"latency_{name}_{int(time.time())}",
            test_name=name,
            test_type=PerformanceTestType.LATENCY,
            metrics=metrics,
            duration=end_time - start_time,
            success=errors == 0,
            error_message=f"{errors} errors out of {iterations} iterations" if errors > 0 else None,
            details={
                "iterations": iterations,
                "warmup": warmup,
                "errors": errors,
                "latencies": latencies[:100] if len(latencies) > 100 else latencies  # Store sample
            }
        )
        
        self.results.append(result)
        self.logger.info(f"Latency test completed: {name}")
        return result
    
    async def test_throughput(self, name: str, target_function: Callable,
                             duration_seconds: int = 10, concurrent_workers: int = 4) -> PerformanceResult:
        """
        Test function throughput
        
        Args:
            name: Test name
            target_function: Function to test
            duration_seconds: Test duration
            concurrent_workers: Number of concurrent workers
            
        Returns:
            PerformanceResult object
        """
        self.logger.info(f"Starting throughput test: {name}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Shared state
        completed_operations = 0
        errors = 0
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async def worker():
            nonlocal completed_operations, errors
            
            while time.time() < end_time:
                try:
                    if asyncio.iscoroutinefunction(target_function):
                        await target_function()
                    else:
                        target_function()
                    completed_operations += 1
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Worker error: {e}")
        
        # Run workers
        tasks = [worker() for _ in range(concurrent_workers)]
        await asyncio.gather(*tasks)
        
        actual_duration = time.time() - start_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate metrics
        throughput = completed_operations / actual_duration
        error_rate = (errors / (completed_operations + errors)) * 100 if (completed_operations + errors) > 0 else 0
        
        # Calculate system metrics
        avg_cpu = statistics.mean([m["cpu_percent"] for m in self.system_metrics]) if self.system_metrics else 0
        avg_memory = statistics.mean([m["memory_used_mb"] for m in self.system_metrics]) if self.system_metrics else 0
        peak_memory = max([m["memory_used_mb"] for m in self.system_metrics]) if self.system_metrics else 0
        
        metrics = {
            PerformanceMetric.THROUGHPUT: throughput,
            PerformanceMetric.ERROR_RATE: error_rate,
            PerformanceMetric.CPU_USAGE: avg_cpu,
            PerformanceMetric.MEMORY_USAGE: avg_memory,
            PerformanceMetric.MEMORY_PEAK: peak_memory
        }
        
        result = PerformanceResult(
            test_id=f"throughput_{name}_{int(time.time())}",
            test_name=name,
            test_type=PerformanceTestType.THROUGHPUT,
            metrics=metrics,
            duration=actual_duration,
            success=error_rate < 5.0,  # Less than 5% error rate
            error_message=f"{error_rate:.1f}% error rate" if error_rate >= 5.0 else None,
            details={
                "completed_operations": completed_operations,
                "errors": errors,
                "concurrent_workers": concurrent_workers,
                "system_metrics": self.system_metrics[-100:] if len(self.system_metrics) > 100 else self.system_metrics
            }
        )
        
        self.results.append(result)
        self.logger.info(f"Throughput test completed: {name}")
        return result
    
    async def test_load(self, config: LoadTestConfig) -> PerformanceResult:
        """
        Run load test with configurable parameters
        
        Args:
            config: Load test configuration
            
        Returns:
            PerformanceResult object
        """
        self.logger.info(f"Starting load test: {config.name}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Test state
        completed_operations = 0
        errors = 0
        response_times = []
        
        # Calculate timing
        total_duration = config.duration_seconds + config.ramp_up_seconds + config.ramp_down_seconds
        start_time = time.time()
        
        async def user_session():
            nonlocal completed_operations, errors, response_times
            
            while time.time() < start_time + total_duration:
                try:
                    # Calculate current load level
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Ramp up phase
                    if elapsed < config.ramp_up_seconds:
                        load_factor = elapsed / config.ramp_up_seconds
                    # Steady state
                    elif elapsed < config.ramp_up_seconds + config.duration_seconds:
                        load_factor = 1.0
                    # Ramp down phase
                    else:
                        remaining = total_duration - elapsed
                        load_factor = remaining / config.ramp_down_seconds if config.ramp_down_seconds > 0 else 0
                    
                    # Skip if ramping down
                    if load_factor <= 0:
                        break
                    
                    # Execute operation
                    operation_start = time.perf_counter()
                    
                    if asyncio.iscoroutinefunction(config.target_function):
                        await config.target_function(**config.parameters)
                    else:
                        config.target_function(**config.parameters)
                    
                    operation_end = time.perf_counter()
                    
                    # Record metrics
                    response_time = (operation_end - operation_start) * 1000  # ms
                    response_times.append(response_time)
                    completed_operations += 1
                    
                    # Think time
                    adjusted_think_time = config.think_time_seconds * load_factor
                    if adjusted_think_time > 0:
                        await asyncio.sleep(adjusted_think_time)
                    
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Load test operation failed: {e}")
        
        # Start user sessions
        tasks = [user_session() for _ in range(config.concurrent_users)]
        await asyncio.gather(*tasks)
        
        actual_duration = time.time() - start_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate metrics
        throughput = completed_operations / actual_duration
        error_rate = (errors / (completed_operations + errors)) * 100 if (completed_operations + errors) > 0 else 0
        
        # Response time metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        # System metrics
        avg_cpu = statistics.mean([m["cpu_percent"] for m in self.system_metrics]) if self.system_metrics else 0
        avg_memory = statistics.mean([m["memory_used_mb"] for m in self.system_metrics]) if self.system_metrics else 0
        peak_memory = max([m["memory_used_mb"] for m in self.system_metrics]) if self.system_metrics else 0
        
        metrics = {
            PerformanceMetric.THROUGHPUT: throughput,
            PerformanceMetric.ERROR_RATE: error_rate,
            PerformanceMetric.RESPONSE_TIME: avg_response_time,
            PerformanceMetric.LATENCY_P95: p95_response_time,
            PerformanceMetric.LATENCY_P99: p99_response_time,
            PerformanceMetric.CPU_USAGE: avg_cpu,
            PerformanceMetric.MEMORY_USAGE: avg_memory,
            PerformanceMetric.MEMORY_PEAK: peak_memory
        }
        
        result = PerformanceResult(
            test_id=f"load_{config.name}_{int(time.time())}",
            test_name=config.name,
            test_type=PerformanceTestType.LOAD,
            metrics=metrics,
            duration=actual_duration,
            success=error_rate < 5.0 and throughput > 0,
            error_message=f"{error_rate:.1f}% error rate" if error_rate >= 5.0 else None,
            details={
                "config": {
                    "concurrent_users": config.concurrent_users,
                    "duration_seconds": config.duration_seconds,
                    "ramp_up_seconds": config.ramp_up_seconds,
                    "ramp_down_seconds": config.ramp_down_seconds,
                    "think_time_seconds": config.think_time_seconds
                },
                "completed_operations": completed_operations,
                "errors": errors,
                "response_times_sample": response_times[-100:] if len(response_times) > 100 else response_times,
                "system_metrics": self.system_metrics[-100:] if len(self.system_metrics) > 100 else self.system_metrics
            }
        )
        
        self.results.append(result)
        self.logger.info(f"Load test completed: {config.name}")
        return result
    
    async def test_memory_usage(self, name: str, target_function: Callable,
                               iterations: int = 100) -> PerformanceResult:
        """
        Test memory usage of function
        
        Args:
            name: Test name
            target_function: Function to test
            iterations: Number of iterations
            
        Returns:
            PerformanceResult object
        """
        self.logger.info(f"Starting memory test: {name}")
        
        # Start memory tracing
        tracemalloc.start()
        
        # Force garbage collection
        gc.collect()
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Run test
        start_time = time.time()
        errors = 0
        
        for i in range(iterations):
            try:
                if asyncio.iscoroutinefunction(target_function):
                    await target_function()
                else:
                    target_function()
            except Exception as e:
                errors += 1
                self.logger.error(f"Memory test iteration {i} failed: {e}")
        
        end_time = time.time()
        
        # Get final memory
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Get memory trace
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        memory_used = final_memory - initial_memory
        memory_peak = peak / (1024 * 1024)  # MB
        
        metrics = {
            PerformanceMetric.MEMORY_USAGE: memory_used,
            PerformanceMetric.MEMORY_PEAK: memory_peak,
            PerformanceMetric.ERROR_RATE: (errors / iterations) * 100
        }
        
        result = PerformanceResult(
            test_id=f"memory_{name}_{int(time.time())}",
            test_name=name,
            test_type=PerformanceTestType.MEMORY,
            metrics=metrics,
            duration=end_time - start_time,
            success=errors == 0,
            error_message=f"{errors} errors out of {iterations} iterations" if errors > 0 else None,
            details={
                "iterations": iterations,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "traced_current_mb": current / (1024 * 1024),
                "traced_peak_mb": memory_peak
            }
        )
        
        self.results.append(result)
        self.logger.info(f"Memory test completed: {name}")
        return result
    
    async def test_scalability(self, name: str, target_function: Callable,
                              load_levels: List[int], duration_per_level: int = 30) -> PerformanceResult:
        """
        Test scalability across different load levels
        
        Args:
            name: Test name
            target_function: Function to test
            load_levels: List of concurrent user levels to test
            duration_per_level: Duration for each load level
            
        Returns:
            PerformanceResult object
        """
        self.logger.info(f"Starting scalability test: {name}")
        
        scalability_results = []
        total_start_time = time.time()
        
        for load_level in load_levels:
            self.logger.info(f"Testing load level: {load_level}")
            
            # Create load test config
            config = LoadTestConfig(
                name=f"{name}_load_{load_level}",
                target_function=target_function,
                concurrent_users=load_level,
                duration_seconds=duration_per_level,
                ramp_up_seconds=5,
                ramp_down_seconds=5
            )
            
            # Run load test
            result = await self.test_load(config)
            
            # Extract key metrics
            throughput = result.metrics.get(PerformanceMetric.THROUGHPUT, 0)
            response_time = result.metrics.get(PerformanceMetric.RESPONSE_TIME, 0)
            error_rate = result.metrics.get(PerformanceMetric.ERROR_RATE, 0)
            cpu_usage = result.metrics.get(PerformanceMetric.CPU_USAGE, 0)
            memory_usage = result.metrics.get(PerformanceMetric.MEMORY_USAGE, 0)
            
            scalability_results.append({
                "load_level": load_level,
                "throughput": throughput,
                "response_time": response_time,
                "error_rate": error_rate,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "success": result.success
            })
        
        total_duration = time.time() - total_start_time
        
        # Calculate scalability metrics
        throughputs = [r["throughput"] for r in scalability_results]
        max_throughput = max(throughputs) if throughputs else 0
        
        # Find the point where performance starts to degrade
        degradation_point = None
        for i in range(1, len(scalability_results)):
            current = scalability_results[i]
            previous = scalability_results[i-1]
            
            # Check if throughput decreased or response time increased significantly
            if (current["throughput"] < previous["throughput"] * 0.9 or 
                current["response_time"] > previous["response_time"] * 1.5):
                degradation_point = current["load_level"]
                break
        
        # Calculate overall score
        success_rate = sum(1 for r in scalability_results if r["success"]) / len(scalability_results)
        
        metrics = {
            PerformanceMetric.THROUGHPUT: max_throughput,
            PerformanceMetric.RESPONSE_TIME: statistics.mean([r["response_time"] for r in scalability_results]),
            PerformanceMetric.ERROR_RATE: statistics.mean([r["error_rate"] for r in scalability_results]),
            PerformanceMetric.CPU_USAGE: statistics.mean([r["cpu_usage"] for r in scalability_results]),
            PerformanceMetric.MEMORY_USAGE: statistics.mean([r["memory_usage"] for r in scalability_results])
        }
        
        result = PerformanceResult(
            test_id=f"scalability_{name}_{int(time.time())}",
            test_name=name,
            test_type=PerformanceTestType.SCALABILITY,
            metrics=metrics,
            duration=total_duration,
            success=success_rate >= 0.8,
            error_message=f"Success rate: {success_rate:.1f}" if success_rate < 0.8 else None,
            details={
                "load_levels": load_levels,
                "duration_per_level": duration_per_level,
                "max_throughput": max_throughput,
                "degradation_point": degradation_point,
                "scalability_results": scalability_results
            }
        )
        
        self.results.append(result)
        self.logger.info(f"Scalability test completed: {name}")
        return result
    
    def compare_with_baseline(self, test_name: str, baseline_result: PerformanceResult) -> Dict[str, Any]:
        """
        Compare current test results with baseline
        
        Args:
            test_name: Name of the test
            baseline_result: Baseline performance result
            
        Returns:
            Comparison analysis
        """
        # Find latest result for this test
        current_results = [r for r in self.results if r.test_name == test_name]
        if not current_results:
            return {"error": "No current results found"}
        
        current_result = current_results[-1]  # Latest result
        
        comparison = {
            "test_name": test_name,
            "baseline_timestamp": baseline_result.timestamp,
            "current_timestamp": current_result.timestamp,
            "metric_changes": {}
        }
        
        # Compare metrics
        for metric in baseline_result.metrics:
            if metric in current_result.metrics:
                baseline_value = baseline_result.metrics[metric]
                current_value = current_result.metrics[metric]
                
                if baseline_value != 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    change_percent = 0
                
                comparison["metric_changes"][metric.value] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_percent": change_percent,
                    "improved": self._is_improvement(metric, change_percent)
                }
        
        return comparison
    
    def _is_improvement(self, metric: PerformanceMetric, change_percent: float) -> bool:
        """Check if metric change is an improvement"""
        # Metrics where lower is better
        lower_is_better = {
            PerformanceMetric.RESPONSE_TIME,
            PerformanceMetric.LATENCY_P50,
            PerformanceMetric.LATENCY_P95,
            PerformanceMetric.LATENCY_P99,
            PerformanceMetric.ERROR_RATE,
            PerformanceMetric.CPU_USAGE,
            PerformanceMetric.MEMORY_USAGE,
            PerformanceMetric.MEMORY_PEAK
        }
        
        if metric in lower_is_better:
            return change_percent < 0  # Negative change is improvement
        else:
            return change_percent > 0  # Positive change is improvement
    
    def check_benchmarks(self, result: PerformanceResult) -> Dict[str, Any]:
        """
        Check result against benchmarks
        
        Args:
            result: Performance result to check
            
        Returns:
            Benchmark analysis
        """
        analysis = {
            "test_name": result.test_name,
            "benchmark_results": {}
        }
        
        # Check if we have benchmarks for this test
        if result.test_name in self.benchmarks:
            benchmark = self.benchmarks[result.test_name]
            
            if benchmark.metric in result.metrics:
                value = result.metrics[benchmark.metric]
                
                # Determine status
                if benchmark.higher_is_better:
                    if value >= benchmark.target_value:
                        status = "PASS"
                    elif value >= benchmark.threshold_warning:
                        status = "WARNING"
                    else:
                        status = "CRITICAL"
                else:
                    if value <= benchmark.target_value:
                        status = "PASS"
                    elif value <= benchmark.threshold_warning:
                        status = "WARNING"
                    else:
                        status = "CRITICAL"
                
                analysis["benchmark_results"][benchmark.name] = {
                    "metric": benchmark.metric.value,
                    "value": value,
                    "target": benchmark.target_value,
                    "status": status,
                    "unit": benchmark.unit
                }
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"performance_report_{timestamp}.json"
        
        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        # Group results by test type
        by_type = {}
        for result in self.results:
            test_type = result.test_type.value
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(result)
        
        # Generate benchmark checks
        benchmark_analysis = []
        for result in self.results:
            analysis = self.check_benchmarks(result)
            if analysis["benchmark_results"]:
                benchmark_analysis.append(analysis)
        
        report_data = {
            "timestamp": timestamp,
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "tests_by_type": {k: len(v) for k, v in by_type.items()}
            },
            "benchmark_analysis": benchmark_analysis,
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "test_type": r.test_type.value,
                    "metrics": {k.value: v for k, v in r.metrics.items()},
                    "duration": r.duration,
                    "success": r.success,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(report_file)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance test summary"""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in PerformanceMetric:
            values = [r.metrics.get(metric, 0) for r in self.results if metric in r.metrics]
            if values:
                avg_metrics[metric.value] = statistics.mean(values)
        
        # Latest results
        latest_results = {}
        for result in self.results:
            if result.test_name not in latest_results or result.timestamp > latest_results[result.test_name].timestamp:
                latest_results[result.test_name] = result
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            "average_metrics": avg_metrics,
            "latest_results": {k: v.test_id for k, v in latest_results.items()},
            "total_duration": sum(r.duration for r in self.results)
        }


# Example test functions for demonstration
def sample_fast_function():
    """Sample fast function for testing"""
    time.sleep(0.001)  # 1ms
    return sum(range(100))


def sample_slow_function():
    """Sample slow function for testing"""
    time.sleep(0.01)  # 10ms
    return sum(range(1000))


def sample_memory_intensive_function():
    """Sample memory intensive function for testing"""
    data = [i for i in range(100000)]
    return sum(data)


async def sample_async_function():
    """Sample async function for testing"""
    await asyncio.sleep(0.001)
    return "async result"


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize performance tester
        tester = PerformanceTester()
        
        # Test latency
        await tester.test_latency("fast_function", sample_fast_function, iterations=1000)
        await tester.test_latency("slow_function", sample_slow_function, iterations=100)
        await tester.test_latency("async_function", sample_async_function, iterations=500)
        
        # Test throughput
        await tester.test_throughput("fast_function", sample_fast_function, duration_seconds=5)
        
        # Test memory usage
        await tester.test_memory_usage("memory_function", sample_memory_intensive_function, iterations=50)
        
        # Test load
        load_config = LoadTestConfig(
            name="sample_load_test",
            target_function=sample_fast_function,
            concurrent_users=10,
            duration_seconds=15,
            ramp_up_seconds=5,
            ramp_down_seconds=5
        )
        await tester.test_load(load_config)
        
        # Test scalability
        await tester.test_scalability("scalability_test", sample_fast_function, [1, 5, 10, 20, 50])
        
        # Generate report
        report_file = tester.generate_performance_report()
        
        # Print summary
        summary = tester.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Successful Tests: {summary.get('successful_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Total Duration: {summary.get('total_duration', 0):.2f}s")
        print(f"Report: {report_file}")
    
    asyncio.run(main())