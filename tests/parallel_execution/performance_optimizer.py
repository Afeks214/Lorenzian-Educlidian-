"""
Performance Optimization Scripts and Validation Tests
Agent 2 Mission: Advanced Performance Optimization and Validation

This module provides comprehensive performance optimization scripts, validation
tests, benchmarking tools, and automated performance tuning for parallel test
execution systems.
"""

import os
import time
import psutil
import threading
import subprocess
import statistics
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import concurrent.futures
import multiprocessing
from contextlib import contextmanager
import tempfile
import shutil

# Import our parallel execution components
from .test_executor import ParallelTestExecutor
from .profiling_system import TestExecutionProfiler
from .resource_manager import AdvancedResourceManager, ResourceLimits
from .monitoring_system import RealTimeMonitoringSystem
from .load_balancer import AdvancedLoadBalancer, WorkerCapacity

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, float]
    speedup: float
    efficiency: float
    resource_utilization: Dict[str, float]
    recommendations: List[str]
    validation_passed: bool
    optimization_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    test_suite: str
    execution_time: float
    success_rate: float
    throughput: float  # tests per second
    memory_peak: float
    cpu_utilization: float
    worker_count: int
    configuration: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceOptimizer:
    """Advanced performance optimizer for parallel test execution"""
    
    def __init__(self, test_directory: str = "tests"):
        self.test_directory = Path(test_directory)
        self.optimization_history: List[OptimizationResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.system_baseline = self._collect_system_baseline()
        
        # Optimization parameters
        self.optimization_targets = {
            'execution_time': 'minimize',
            'memory_usage': 'minimize',
            'cpu_efficiency': 'maximize',
            'success_rate': 'maximize',
            'throughput': 'maximize'
        }
        
        self.configuration_space = {
            'max_workers': [1, 2, 4, 8, 16],
            'distribution_strategy': ['loadfile', 'loadscope', 'worksteal'],
            'memory_limit_mb': [256, 512, 1024, 2048],
            'cpu_affinity': [True, False],
            'batch_size': [1, 5, 10, 20],
            'timeout_seconds': [60, 120, 300, 600]
        }
    
    def _collect_system_baseline(self) -> Dict[str, Any]:
        """Collect system baseline metrics"""
        return {
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_free_gb': shutil.disk_usage('/').free / (1024**3),
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'platform': os.uname().system if hasattr(os, 'uname') else 'unknown'
        }
    
    def discover_tests(self) -> List[str]:
        """Discover all test files in the test directory"""
        test_files = []
        
        for pattern in ['test_*.py', '*_test.py']:
            test_files.extend(self.test_directory.glob(f'**/{pattern}'))
        
        return [str(f.relative_to(self.test_directory)) for f in test_files]
    
    def run_benchmark(self, configuration: Dict[str, Any], 
                     test_subset: Optional[List[str]] = None) -> BenchmarkResult:
        """Run benchmark with specific configuration"""
        start_time = time.time()
        
        # Prepare test list
        if test_subset is None:
            test_files = self.discover_tests()[:10]  # Use first 10 tests for benchmarking
        else:
            test_files = test_subset
        
        if not test_files:
            raise ValueError("No test files found for benchmarking")
        
        # Configure parallel executor
        executor = ParallelTestExecutor(configuration.get('max_workers', 4))
        
        # Monitor resources during execution
        resource_monitor = AdvancedResourceManager()
        
        # Set up monitoring
        monitoring_system = RealTimeMonitoringSystem()
        monitoring_system.start_monitoring()
        
        try:
            # Execute tests
            results = executor.run_parallel_tests(
                test_files,
                configuration.get('distribution_strategy', 'loadscope')
            )
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            total_tests = results['execution_summary']['total_tests']
            success_rate = results['execution_summary']['success_rate']
            throughput = total_tests / execution_time if execution_time > 0 else 0
            
            # Get resource usage
            system_metrics = monitoring_system._get_system_metrics()
            
            benchmark_result = BenchmarkResult(
                test_suite=f"{len(test_files)} files",
                execution_time=execution_time,
                success_rate=success_rate,
                throughput=throughput,
                memory_peak=system_metrics.system_memory_usage,
                cpu_utilization=system_metrics.system_cpu_usage,
                worker_count=configuration.get('max_workers', 4),
                configuration=configuration
            )
            
            self.benchmark_results.append(benchmark_result)
            
            return benchmark_result
            
        finally:
            monitoring_system.stop_monitoring()
    
    def optimize_configuration(self, optimization_rounds: int = 5) -> OptimizationResult:
        """Optimize configuration using multiple strategies"""
        logger.info(f"Starting configuration optimization with {optimization_rounds} rounds")
        
        best_configuration = None
        best_score = float('-inf')
        best_metrics = None
        
        optimization_results = []
        
        for round_num in range(optimization_rounds):
            logger.info(f"Optimization round {round_num + 1}/{optimization_rounds}")
            
            # Generate configuration candidates
            if round_num == 0:
                # Start with baseline configurations
                candidates = self._generate_baseline_configurations()
            else:
                # Generate candidates based on previous results
                candidates = self._generate_smart_configurations(optimization_results)
            
            # Evaluate each candidate
            for config in candidates:
                try:
                    logger.info(f"Evaluating configuration: {config}")
                    
                    benchmark_result = self.run_benchmark(config)
                    score = self._calculate_optimization_score(benchmark_result)
                    
                    optimization_results.append({
                        'configuration': config,
                        'benchmark_result': benchmark_result,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_configuration = config
                        best_metrics = benchmark_result
                        
                        logger.info(f"New best configuration found: score={score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate configuration {config}: {e}")
                    continue
        
        # Calculate final results
        if best_configuration is None:
            raise RuntimeError("No valid configuration found during optimization")
        
        # Calculate speedup compared to baseline
        baseline_time = self._get_baseline_execution_time()
        speedup = baseline_time / best_metrics.execution_time if best_metrics.execution_time > 0 else 1.0
        
        # Calculate efficiency
        efficiency = speedup / best_configuration.get('max_workers', 1)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            optimization_results, best_configuration
        )
        
        # Validate optimized configuration
        validation_passed = self._validate_configuration(best_configuration)
        
        result = OptimizationResult(
            configuration=best_configuration,
            performance_metrics={
                'execution_time': best_metrics.execution_time,
                'throughput': best_metrics.throughput,
                'success_rate': best_metrics.success_rate,
                'memory_peak': best_metrics.memory_peak,
                'cpu_utilization': best_metrics.cpu_utilization
            },
            speedup=speedup,
            efficiency=efficiency,
            resource_utilization={
                'cpu': best_metrics.cpu_utilization,
                'memory': best_metrics.memory_peak,
                'workers': best_metrics.worker_count
            },
            recommendations=recommendations,
            validation_passed=validation_passed
        )
        
        self.optimization_history.append(result)
        
        return result
    
    def _generate_baseline_configurations(self) -> List[Dict[str, Any]]:
        """Generate baseline configurations for initial evaluation"""
        configurations = []
        
        # Single worker baseline
        configurations.append({
            'max_workers': 1,
            'distribution_strategy': 'loadfile',
            'memory_limit_mb': 1024,
            'cpu_affinity': False,
            'batch_size': 1,
            'timeout_seconds': 300
        })
        
        # CPU-count based configurations
        cpu_count = self.system_baseline['cpu_count']
        for workers in [cpu_count // 2, cpu_count, cpu_count * 2]:
            if workers > 0:
                configurations.append({
                    'max_workers': workers,
                    'distribution_strategy': 'loadscope',
                    'memory_limit_mb': 1024,
                    'cpu_affinity': True,
                    'batch_size': 5,
                    'timeout_seconds': 300
                })
        
        # Memory-optimized configuration
        configurations.append({
            'max_workers': cpu_count,
            'distribution_strategy': 'worksteal',
            'memory_limit_mb': 2048,
            'cpu_affinity': True,
            'batch_size': 10,
            'timeout_seconds': 600
        })
        
        return configurations
    
    def _generate_smart_configurations(self, previous_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate smart configurations based on previous results"""
        configurations = []
        
        # Find top performing configurations
        top_results = sorted(previous_results, key=lambda x: x['score'], reverse=True)[:3]
        
        for result in top_results:
            base_config = result['configuration']
            
            # Generate variations
            for param, values in self.configuration_space.items():
                if param in base_config:
                    current_value = base_config[param]
                    
                    # Try neighboring values
                    if isinstance(current_value, (int, float)) and isinstance(values, list):
                        try:
                            current_idx = values.index(current_value)
                            for offset in [-1, 1]:
                                new_idx = current_idx + offset
                                if 0 <= new_idx < len(values):
                                    new_config = base_config.copy()
                                    new_config[param] = values[new_idx]
                                    configurations.append(new_config)
                        except ValueError:
                            pass
                    
                    # Try other values for categorical parameters
                    elif isinstance(values, list) and current_value in values:
                        for value in values:
                            if value != current_value:
                                new_config = base_config.copy()
                                new_config[param] = value
                                configurations.append(new_config)
        
        # Remove duplicates
        unique_configs = []
        for config in configurations:
            if config not in unique_configs:
                unique_configs.append(config)
        
        return unique_configs[:10]  # Limit to 10 configurations per round
    
    def _calculate_optimization_score(self, benchmark_result: BenchmarkResult) -> float:
        """Calculate optimization score for a benchmark result"""
        # Weighted scoring based on optimization targets
        weights = {
            'execution_time': 0.3,
            'throughput': 0.3,
            'success_rate': 0.2,
            'memory_efficiency': 0.1,
            'cpu_efficiency': 0.1
        }
        
        # Normalize metrics (0-1 scale)
        normalized_metrics = {}
        
        # Execution time (lower is better)
        baseline_time = self._get_baseline_execution_time()
        normalized_metrics['execution_time'] = min(1.0, baseline_time / benchmark_result.execution_time)
        
        # Throughput (higher is better)
        max_theoretical_throughput = self.system_baseline['cpu_count'] * 10  # Rough estimate
        normalized_metrics['throughput'] = min(1.0, benchmark_result.throughput / max_theoretical_throughput)
        
        # Success rate (higher is better)
        normalized_metrics['success_rate'] = benchmark_result.success_rate
        
        # Memory efficiency (lower usage is better)
        memory_total = self.system_baseline['memory_total_gb'] * 1024  # Convert to MB
        normalized_metrics['memory_efficiency'] = 1.0 - (benchmark_result.memory_peak / memory_total)
        
        # CPU efficiency (balanced utilization is better)
        optimal_cpu = 80.0  # 80% utilization is optimal
        cpu_diff = abs(benchmark_result.cpu_utilization - optimal_cpu)
        normalized_metrics['cpu_efficiency'] = 1.0 - (cpu_diff / 100.0)
        
        # Calculate weighted score
        score = sum(normalized_metrics[metric] * weights[metric] for metric in weights)
        
        return score
    
    def _get_baseline_execution_time(self) -> float:
        """Get baseline execution time for comparison"""
        # Use single-worker execution time as baseline
        baseline_results = [r for r in self.benchmark_results if r.worker_count == 1]
        
        if baseline_results:
            return min(r.execution_time for r in baseline_results)
        else:
            return 60.0  # Default baseline of 60 seconds
    
    def _generate_optimization_recommendations(self, results: List[Dict[str, Any]], 
                                             best_config: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Worker count recommendation
        worker_scores = defaultdict(list)
        for result in results:
            worker_count = result['configuration'].get('max_workers', 1)
            worker_scores[worker_count].append(result['score'])
        
        best_worker_count = max(worker_scores, key=lambda x: statistics.mean(worker_scores[x]))
        
        if best_worker_count != best_config.get('max_workers'):
            recommendations.append(f"Consider using {best_worker_count} workers for optimal performance")
        
        # Distribution strategy recommendation
        strategy_scores = defaultdict(list)
        for result in results:
            strategy = result['configuration'].get('distribution_strategy', 'loadscope')
            strategy_scores[strategy].append(result['score'])
        
        if strategy_scores:
            best_strategy = max(strategy_scores, key=lambda x: statistics.mean(strategy_scores[x]))
            recommendations.append(f"Best distribution strategy: {best_strategy}")
        
        # Memory optimization
        memory_usage = [r['benchmark_result'].memory_peak for r in results]
        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            available_memory = self.system_baseline['memory_available_gb'] * 1024
            
            if avg_memory > available_memory * 0.8:
                recommendations.append("Consider increasing memory limits or reducing parallel workers")
        
        # CPU optimization
        cpu_usage = [r['benchmark_result'].cpu_utilization for r in results]
        if cpu_usage:
            avg_cpu = statistics.mean(cpu_usage)
            
            if avg_cpu < 50:
                recommendations.append("CPU underutilized - consider increasing worker count")
            elif avg_cpu > 90:
                recommendations.append("CPU overutilized - consider decreasing worker count")
        
        return recommendations
    
    def _validate_configuration(self, configuration: Dict[str, Any]) -> bool:
        """Validate optimized configuration"""
        try:
            # Test configuration with small test suite
            test_files = self.discover_tests()[:3]  # Use only 3 tests for validation
            
            if not test_files:
                return False
            
            # Run validation benchmark
            validation_result = self.run_benchmark(configuration, test_files)
            
            # Validation criteria
            validation_checks = [
                validation_result.success_rate >= 0.95,  # 95% success rate
                validation_result.execution_time > 0,    # Valid execution time
                validation_result.throughput > 0,       # Valid throughput
                validation_result.memory_peak > 0,      # Valid memory usage
                validation_result.cpu_utilization > 0   # Valid CPU usage
            ]
            
            return all(validation_checks)
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def run_stress_test(self, configuration: Dict[str, Any], 
                       duration_minutes: int = 10) -> Dict[str, Any]:
        """Run stress test with given configuration"""
        logger.info(f"Starting stress test for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        stress_results = []
        resource_samples = []
        
        # Set up monitoring
        monitoring_system = RealTimeMonitoringSystem()
        monitoring_system.start_monitoring()
        
        try:
            while time.time() < end_time:
                # Run benchmark
                benchmark_result = self.run_benchmark(configuration)
                stress_results.append(benchmark_result)
                
                # Collect resource samples
                resource_sample = {
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                }
                resource_samples.append(resource_sample)
                
                # Brief pause between iterations
                time.sleep(5)
                
        finally:
            monitoring_system.stop_monitoring()
        
        # Analyze stress test results
        if not stress_results:
            return {"error": "No stress test results collected"}
        
        execution_times = [r.execution_time for r in stress_results]
        success_rates = [r.success_rate for r in stress_results]
        throughputs = [r.throughput for r in stress_results]
        
        return {
            'test_duration_minutes': duration_minutes,
            'iterations_completed': len(stress_results),
            'performance_stability': {
                'execution_time': {
                    'mean': statistics.mean(execution_times),
                    'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'min': min(execution_times),
                    'max': max(execution_times)
                },
                'success_rate': {
                    'mean': statistics.mean(success_rates),
                    'std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                    'min': min(success_rates),
                    'max': max(success_rates)
                },
                'throughput': {
                    'mean': statistics.mean(throughputs),
                    'std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                    'min': min(throughputs),
                    'max': max(throughputs)
                }
            },
            'resource_usage': {
                'cpu_usage': statistics.mean([s['cpu_usage'] for s in resource_samples]),
                'memory_usage': statistics.mean([s['memory_usage'] for s in resource_samples]),
                'load_average': statistics.mean([s['load_average'] for s in resource_samples])
            },
            'stability_score': self._calculate_stability_score(stress_results)
        }
    
    def _calculate_stability_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate stability score for stress test results"""
        if len(results) < 2:
            return 0.0
        
        # Calculate coefficient of variation for key metrics
        execution_times = [r.execution_time for r in results]
        success_rates = [r.success_rate for r in results]
        throughputs = [r.throughput for r in results]
        
        def cv(values):
            """Calculate coefficient of variation"""
            mean_val = statistics.mean(values)
            if mean_val == 0:
                return 0
            std_val = statistics.stdev(values)
            return std_val / mean_val
        
        cv_execution = cv(execution_times)
        cv_success = cv(success_rates)
        cv_throughput = cv(throughputs)
        
        # Stability score (lower CV = higher stability)
        stability = 1.0 - (cv_execution * 0.4 + cv_success * 0.3 + cv_throughput * 0.3)
        
        return max(0.0, min(1.0, stability))
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        latest_optimization = self.optimization_history[-1]
        
        # Performance comparison
        if len(self.benchmark_results) > 1:
            baseline_result = min(self.benchmark_results, key=lambda x: x.worker_count)
            optimized_result = max(self.benchmark_results, key=lambda x: x.throughput)
            
            performance_improvement = {
                'execution_time_improvement': (
                    (baseline_result.execution_time - optimized_result.execution_time) / 
                    baseline_result.execution_time * 100
                ),
                'throughput_improvement': (
                    (optimized_result.throughput - baseline_result.throughput) / 
                    baseline_result.throughput * 100
                ),
                'success_rate_improvement': (
                    (optimized_result.success_rate - baseline_result.success_rate) * 100
                )
            }
        else:
            performance_improvement = {}
        
        # Resource efficiency analysis
        resource_efficiency = {
            'cpu_efficiency': latest_optimization.resource_utilization.get('cpu', 0) / 100,
            'memory_efficiency': latest_optimization.resource_utilization.get('memory', 0) / 100,
            'worker_efficiency': latest_optimization.efficiency
        }
        
        return {
            'optimization_timestamp': latest_optimization.optimization_timestamp.isoformat(),
            'system_baseline': self.system_baseline,
            'optimized_configuration': latest_optimization.configuration,
            'performance_metrics': latest_optimization.performance_metrics,
            'speedup': latest_optimization.speedup,
            'efficiency': latest_optimization.efficiency,
            'performance_improvement': performance_improvement,
            'resource_efficiency': resource_efficiency,
            'recommendations': latest_optimization.recommendations,
            'validation_status': latest_optimization.validation_passed,
            'optimization_history_count': len(self.optimization_history),
            'benchmark_history_count': len(self.benchmark_results)
        }
    
    def export_optimization_data(self, output_path: str = "optimization_results.json"):
        """Export optimization data to JSON file"""
        data = {
            'system_baseline': self.system_baseline,
            'optimization_history': [
                {
                    'configuration': opt.configuration,
                    'performance_metrics': opt.performance_metrics,
                    'speedup': opt.speedup,
                    'efficiency': opt.efficiency,
                    'resource_utilization': opt.resource_utilization,
                    'recommendations': opt.recommendations,
                    'validation_passed': opt.validation_passed,
                    'timestamp': opt.optimization_timestamp.isoformat()
                }
                for opt in self.optimization_history
            ],
            'benchmark_results': [
                {
                    'test_suite': bench.test_suite,
                    'execution_time': bench.execution_time,
                    'success_rate': bench.success_rate,
                    'throughput': bench.throughput,
                    'memory_peak': bench.memory_peak,
                    'cpu_utilization': bench.cpu_utilization,
                    'worker_count': bench.worker_count,
                    'configuration': bench.configuration,
                    'timestamp': bench.timestamp.isoformat()
                }
                for bench in self.benchmark_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported optimization data to {output_path}")


class ValidationTestSuite:
    """Validation test suite for parallel execution system"""
    
    def __init__(self):
        self.test_results = []
        
    def run_validation_tests(self) -> Dict[str, Any]:
        """Run comprehensive validation tests"""
        logger.info("Starting validation test suite")
        
        validation_results = {
            'basic_functionality': self._test_basic_functionality(),
            'resource_management': self._test_resource_management(),
            'load_balancing': self._test_load_balancing(),
            'monitoring_system': self._test_monitoring_system(),
            'error_handling': self._test_error_handling(),
            'performance_consistency': self._test_performance_consistency(),
            'scalability': self._test_scalability()
        }
        
        # Calculate overall validation score
        passed_tests = sum(1 for result in validation_results.values() if result.get('passed', False))
        total_tests = len(validation_results)
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_results': validation_results,
            'validation_passed': overall_score >= 0.8  # 80% threshold
        }
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic parallel execution functionality"""
        try:
            # Test basic parallel execution
            executor = ParallelTestExecutor(max_workers=2)
            test_names = ['dummy_test_1', 'dummy_test_2', 'dummy_test_3']
            
            results = executor.run_parallel_tests(test_names, 'loadscope')
            
            # Validation checks
            checks = [
                results is not None,
                'execution_summary' in results,
                results['execution_summary']['total_tests'] > 0,
                results['execution_summary']['num_workers'] == 2
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 4
            }
    
    def _test_resource_management(self) -> Dict[str, Any]:
        """Test resource management functionality"""
        try:
            resource_manager = AdvancedResourceManager()
            
            # Test worker allocation
            limits = ResourceLimits(memory_mb=512, cpu_cores=[0, 1])
            allocation = resource_manager.allocate_resources('test_worker', limits)
            
            # Test monitoring
            monitoring_info = resource_manager.monitor_worker_resources('test_worker')
            
            # Test release
            resource_manager.release_worker_resources('test_worker')
            
            checks = [
                allocation is not None,
                allocation.worker_id == 'test_worker',
                allocation.memory_limit_mb == 512,
                monitoring_info is not None,
                'worker_id' in monitoring_info
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'allocation': allocation.__dict__ if allocation else None,
                    'monitoring': monitoring_info
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 5
            }
    
    def _test_load_balancing(self) -> Dict[str, Any]:
        """Test load balancing functionality"""
        try:
            balancer = AdvancedLoadBalancer()
            
            # Register test workers
            for i in range(3):
                capacity = WorkerCapacity(
                    worker_id=f'test_worker_{i}',
                    max_concurrent_tests=2,
                    current_load=0,
                    cpu_capacity=100.0,
                    memory_capacity=1024.0,
                    current_cpu_usage=0.0,
                    current_memory_usage=0.0,
                    performance_score=85.0,
                    specialty_tags=set()
                )
                balancer.register_worker(f'test_worker_{i}', capacity)
            
            # Test task assignment
            from .load_balancer import TestTask
            task = TestTask(
                test_id='test_task_1',
                test_name='dummy_test',
                estimated_duration=1.0,
                priority=1,
                dependencies=[],
                resource_requirements={}
            )
            
            assigned_worker = balancer.assign_task(task)
            
            # Test load balance report
            report = balancer.get_load_balance_report()
            
            checks = [
                assigned_worker is not None,
                assigned_worker.startswith('test_worker_'),
                report is not None,
                'total_workers' in report,
                report['total_workers'] == 3
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'assigned_worker': assigned_worker,
                    'report': report
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 5
            }
    
    def _test_monitoring_system(self) -> Dict[str, Any]:
        """Test monitoring system functionality"""
        try:
            monitoring = RealTimeMonitoringSystem()
            
            # Test worker registration
            monitoring.worker_tracker.register_worker('test_worker', 12345)
            
            # Test health update
            monitoring.worker_tracker.update_worker_health('test_worker', {
                'cpu_usage': 50.0,
                'memory_usage': 256.0,
                'tests_executed': 5,
                'tests_passed': 5,
                'tests_failed': 0,
                'response_time': 0.5
            })
            
            # Test health retrieval
            health = monitoring.worker_tracker.get_worker_health('test_worker')
            
            # Test dashboard data
            dashboard_data = monitoring.get_monitoring_dashboard_data()
            
            checks = [
                health is not None,
                health.worker_id == 'test_worker',
                health.cpu_usage == 50.0,
                dashboard_data is not None,
                'worker_health' in dashboard_data,
                len(dashboard_data['worker_health']) == 1
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'health': health.__dict__ if health else None,
                    'dashboard_keys': list(dashboard_data.keys()) if dashboard_data else []
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 6
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        try:
            # Test invalid configuration handling
            optimizer = PerformanceOptimizer()
            
            # Test with invalid configuration
            invalid_config = {
                'max_workers': -1,  # Invalid
                'distribution_strategy': 'invalid_strategy',
                'memory_limit_mb': 0
            }
            
            error_handled = False
            try:
                optimizer.run_benchmark(invalid_config)
            except Exception:
                error_handled = True
            
            # Test resource limit enforcement
            resource_manager = AdvancedResourceManager()
            
            # Test invalid worker ID
            invalid_monitoring = resource_manager.monitor_worker_resources('nonexistent_worker')
            
            checks = [
                error_handled,  # Invalid config should raise error
                invalid_monitoring is not None,
                'error' in invalid_monitoring or 'worker_id' in invalid_monitoring
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'error_handled': error_handled,
                    'invalid_monitoring': invalid_monitoring
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 3
            }
    
    def _test_performance_consistency(self) -> Dict[str, Any]:
        """Test performance consistency across multiple runs"""
        try:
            optimizer = PerformanceOptimizer()
            
            # Run multiple benchmarks with same configuration
            config = {
                'max_workers': 2,
                'distribution_strategy': 'loadscope',
                'memory_limit_mb': 1024,
                'cpu_affinity': False,
                'batch_size': 1,
                'timeout_seconds': 60
            }
            
            results = []
            for i in range(3):
                # Use minimal test set for consistency testing
                result = optimizer.run_benchmark(config, ['dummy_test.py'])
                results.append(result)
            
            # Check consistency
            execution_times = [r.execution_time for r in results]
            success_rates = [r.success_rate for r in results]
            
            time_cv = statistics.stdev(execution_times) / statistics.mean(execution_times)
            success_consistency = all(rate >= 0.8 for rate in success_rates)
            
            checks = [
                len(results) == 3,
                time_cv < 0.3,  # Less than 30% variation
                success_consistency,
                all(r.execution_time > 0 for r in results)
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'execution_times': execution_times,
                    'success_rates': success_rates,
                    'time_cv': time_cv,
                    'success_consistency': success_consistency
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 4
            }
    
    def _test_scalability(self) -> Dict[str, Any]:
        """Test scalability with different worker counts"""
        try:
            optimizer = PerformanceOptimizer()
            
            # Test with different worker counts
            worker_counts = [1, 2, 4]
            scalability_results = []
            
            for workers in worker_counts:
                config = {
                    'max_workers': workers,
                    'distribution_strategy': 'loadscope',
                    'memory_limit_mb': 1024,
                    'cpu_affinity': False,
                    'batch_size': 1,
                    'timeout_seconds': 60
                }
                
                result = optimizer.run_benchmark(config, ['dummy_test.py'])
                scalability_results.append((workers, result))
            
            # Check scalability
            throughputs = [result.throughput for _, result in scalability_results]
            
            # Throughput should generally increase with more workers
            throughput_increases = [
                throughputs[1] >= throughputs[0] * 0.8,  # Allow some variance
                throughputs[2] >= throughputs[1] * 0.8
            ]
            
            checks = [
                len(scalability_results) == len(worker_counts),
                all(result.success_rate > 0.8 for _, result in scalability_results),
                sum(throughput_increases) >= 1,  # At least one improvement
                all(result.execution_time > 0 for _, result in scalability_results)
            ]
            
            return {
                'passed': all(checks),
                'checks_passed': sum(checks),
                'total_checks': len(checks),
                'details': {
                    'worker_counts': worker_counts,
                    'throughputs': throughputs,
                    'throughput_increases': throughput_increases,
                    'scalability_results': [
                        {
                            'workers': workers,
                            'execution_time': result.execution_time,
                            'throughput': result.throughput,
                            'success_rate': result.success_rate
                        }
                        for workers, result in scalability_results
                    ]
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'checks_passed': 0,
                'total_checks': 4
            }


if __name__ == "__main__":
    # Demo usage
    import sys
    
    # Create performance optimizer
    optimizer = PerformanceOptimizer()
    
    # Run optimization
    print("Starting performance optimization...")
    optimization_result = optimizer.optimize_configuration(optimization_rounds=3)
    
    print(f"\nOptimization Results:")
    print(f"Best Configuration: {optimization_result.configuration}")
    print(f"Speedup: {optimization_result.speedup:.2f}x")
    print(f"Efficiency: {optimization_result.efficiency:.2f}")
    print(f"Validation Passed: {optimization_result.validation_passed}")
    
    print(f"\nRecommendations:")
    for rec in optimization_result.recommendations:
        print(f"- {rec}")
    
    # Run validation tests
    print("\nRunning validation tests...")
    validator = ValidationTestSuite()
    validation_results = validator.run_validation_tests()
    
    print(f"\nValidation Results:")
    print(f"Overall Score: {validation_results['overall_score']:.2%}")
    print(f"Tests Passed: {validation_results['passed_tests']}/{validation_results['total_tests']}")
    print(f"Validation Passed: {validation_results['validation_passed']}")
    
    # Export results
    optimizer.export_optimization_data("optimization_results.json")
    print("\nOptimization data exported to optimization_results.json")