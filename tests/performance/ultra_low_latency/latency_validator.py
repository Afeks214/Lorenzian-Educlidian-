"""
Latency Validation Framework
==========================

Comprehensive latency validation framework for ultra-low latency systems.
Validates latency requirements, performs stress testing, and provides
detailed performance analysis.
"""

import logging


import time
import threading
import statistics
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .nanosecond_timer import NanosecondTimer
from .hardware_profiler import HardwareProfiler
from .rdma_simulator import RDMASimulator, RDMAOperation
from .performance_monitor import PerformanceMonitor


class ValidationLevel(Enum):
    """Validation test levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STRESS = "stress"
    EXTREME = "extreme"


@dataclass
class LatencyRequirement:
    """Latency requirement specification"""
    operation: str
    max_latency_ns: int
    percentile_requirements: Dict[str, int] = field(default_factory=dict)
    throughput_requirement: Optional[float] = None
    error_rate_threshold: float = 0.01
    test_duration_seconds: int = 30


@dataclass
class ValidationResult:
    """Validation test result"""
    operation: str
    requirement: LatencyRequirement
    passed: bool
    actual_latency_ns: float
    percentile_results: Dict[str, float]
    throughput_achieved: float
    error_rate: float
    violations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestConfig:
    """Stress test configuration"""
    concurrent_threads: int = 16
    operations_per_thread: int = 1000
    ramp_up_time_seconds: int = 5
    test_duration_seconds: int = 60
    operation_mix: Dict[str, float] = field(default_factory=dict)


class LatencyValidator:
    """
    Comprehensive latency validation framework
    
    Features:
    - Requirements-based validation
    - Multi-level testing (basic to extreme)
    - Stress testing with concurrent load
    - Hardware-aware validation
    - RDMA performance validation
    - Detailed reporting and analysis
    """
    
    def __init__(self, timer: NanosecondTimer, hardware_profiler: HardwareProfiler,
                 rdma_simulator: RDMASimulator, performance_monitor: PerformanceMonitor):
        self.timer = timer
        self.hardware_profiler = hardware_profiler
        self.rdma_simulator = rdma_simulator
        self.performance_monitor = performance_monitor
        
        # Default requirements for common operations
        self.default_requirements = {
            'order_processing': LatencyRequirement(
                operation='order_processing',
                max_latency_ns=1_000_000,  # 1ms
                percentile_requirements={'p95': 2_000_000, 'p99': 5_000_000},
                throughput_requirement=10000.0,
                error_rate_threshold=0.001
            ),
            'market_data_processing': LatencyRequirement(
                operation='market_data_processing',
                max_latency_ns=500_000,  # 500µs
                percentile_requirements={'p95': 1_000_000, 'p99': 2_000_000},
                throughput_requirement=50000.0,
                error_rate_threshold=0.0001
            ),
            'risk_calculation': LatencyRequirement(
                operation='risk_calculation',
                max_latency_ns=2_000_000,  # 2ms
                percentile_requirements={'p95': 5_000_000, 'p99': 10_000_000},
                throughput_requirement=5000.0,
                error_rate_threshold=0.01
            ),
            'rdma_write': LatencyRequirement(
                operation='rdma_write',
                max_latency_ns=100_000,  # 100µs
                percentile_requirements={'p95': 200_000, 'p99': 500_000},
                throughput_requirement=100000.0,
                error_rate_threshold=0.0001
            )
        }
        
        # Executor for concurrent testing
        self.executor = ThreadPoolExecutor(max_workers=32)
        
        # Test operation registry
        self.test_operations: Dict[str, Callable] = {
            'order_processing': self._simulate_order_processing,
            'market_data_processing': self._simulate_market_data_processing,
            'risk_calculation': self._simulate_risk_calculation,
            'rdma_write': self._simulate_rdma_write,
            'rdma_read': self._simulate_rdma_read
        }
    
    def validate_latency_requirements(self, requirements: List[LatencyRequirement],
                                    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> List[ValidationResult]:
        """Validate latency requirements"""
        results = []
        
        for requirement in requirements:
            result = self._validate_single_requirement(requirement, validation_level)
            results.append(result)
        
        return results
    
    def _validate_single_requirement(self, requirement: LatencyRequirement,
                                   validation_level: ValidationLevel) -> ValidationResult:
        """Validate a single latency requirement"""
        
        # Configure test parameters based on validation level
        test_params = self._get_test_parameters(validation_level)
        
        # Clear previous results
        self.timer.clear_results(requirement.operation)
        
        violations = []
        
        try:
            # Run the test
            test_results = self._run_latency_test(requirement, test_params)
            
            # Analyze results
            stats = self.timer.get_statistics(requirement.operation)
            
            if not stats:
                return ValidationResult(
                    operation=requirement.operation,
                    requirement=requirement,
                    passed=False,
                    actual_latency_ns=0,
                    percentile_results={},
                    throughput_achieved=0,
                    error_rate=1.0,
                    violations=["No test results available"]
                )
            
            # Check mean latency
            if stats.mean_ns > requirement.max_latency_ns:
                violations.append(
                    f"Mean latency {stats.mean_ns}ns exceeds requirement {requirement.max_latency_ns}ns"
                )
            
            # Check percentile requirements
            percentile_results = {}
            for percentile, max_latency in requirement.percentile_requirements.items():
                if percentile == 'p95':
                    actual_latency = stats.p95_ns
                elif percentile == 'p99':
                    actual_latency = stats.p99_ns
                elif percentile == 'p999':
                    actual_latency = stats.p999_ns
                else:
                    continue
                
                percentile_results[percentile] = actual_latency
                
                if actual_latency > max_latency:
                    violations.append(
                        f"{percentile} latency {actual_latency}ns exceeds requirement {max_latency}ns"
                    )
            
            # Calculate throughput and error rate
            throughput_achieved = test_results.get('throughput', 0)
            error_rate = test_results.get('error_rate', 0)
            
            # Check throughput requirement
            if requirement.throughput_requirement and throughput_achieved < requirement.throughput_requirement:
                violations.append(
                    f"Throughput {throughput_achieved} ops/sec below requirement {requirement.throughput_requirement} ops/sec"
                )
            
            # Check error rate
            if error_rate > requirement.error_rate_threshold:
                violations.append(
                    f"Error rate {error_rate} exceeds threshold {requirement.error_rate_threshold}"
                )
            
            return ValidationResult(
                operation=requirement.operation,
                requirement=requirement,
                passed=len(violations) == 0,
                actual_latency_ns=stats.mean_ns,
                percentile_results=percentile_results,
                throughput_achieved=throughput_achieved,
                error_rate=error_rate,
                violations=violations,
                metadata={
                    'validation_level': validation_level.value,
                    'test_params': test_params,
                    'statistics': stats
                }
            )
            
        except Exception as e:
            return ValidationResult(
                operation=requirement.operation,
                requirement=requirement,
                passed=False,
                actual_latency_ns=0,
                percentile_results={},
                throughput_achieved=0,
                error_rate=1.0,
                violations=[f"Test execution failed: {str(e)}"]
            )
    
    def _get_test_parameters(self, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Get test parameters based on validation level"""
        params = {
            ValidationLevel.BASIC: {
                'iterations': 1000,
                'threads': 1,
                'duration_seconds': 10,
                'warmup_iterations': 100
            },
            ValidationLevel.COMPREHENSIVE: {
                'iterations': 10000,
                'threads': 4,
                'duration_seconds': 30,
                'warmup_iterations': 1000
            },
            ValidationLevel.STRESS: {
                'iterations': 100000,
                'threads': 16,
                'duration_seconds': 60,
                'warmup_iterations': 10000
            },
            ValidationLevel.EXTREME: {
                'iterations': 1000000,
                'threads': 32,
                'duration_seconds': 300,
                'warmup_iterations': 100000
            }
        }
        
        return params.get(validation_level, params[ValidationLevel.COMPREHENSIVE])
    
    def _run_latency_test(self, requirement: LatencyRequirement, 
                         test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run latency test for a specific requirement"""
        
        operation = requirement.operation
        iterations = test_params['iterations']
        threads = test_params['threads']
        warmup_iterations = test_params['warmup_iterations']
        
        # Get test function
        test_func = self.test_operations.get(operation)
        if not test_func:
            raise ValueError(f"No test function available for operation: {operation}")
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                test_func()
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.error(f'Error occurred: {e}')
        
        # Clear warmup results
        self.timer.clear_results(operation)
        
        # Run concurrent test
        start_time = time.time()
        successful_operations = 0
        failed_operations = 0
        
        futures = []
        operations_per_thread = iterations // threads
        
        for _ in range(threads):
            future = self.executor.submit(
                self._run_thread_test, test_func, operation, operations_per_thread
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                success, failures = future.result()
                successful_operations += success
                failed_operations += failures
            except Exception as e:
                failed_operations += operations_per_thread
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_operations = successful_operations + failed_operations
        throughput = successful_operations / total_time if total_time > 0 else 0
        error_rate = failed_operations / total_operations if total_operations > 0 else 0
        
        return {
            'throughput': throughput,
            'error_rate': error_rate,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': failed_operations,
            'total_time': total_time
        }
    
    def _run_thread_test(self, test_func: Callable, operation: str, iterations: int) -> Tuple[int, int]:
        """Run test in a single thread"""
        successful = 0
        failed = 0
        
        for _ in range(iterations):
            try:
                with self.timer.measure(operation):
                    test_func()
                successful += 1
            except Exception:
                failed += 1
        
        return successful, failed
    
    def _simulate_order_processing(self):
        """Simulate order processing operation"""
        # Simulate order validation
        order_data = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'side': 'BUY'
        }
        
        # Simulate processing time
        time.sleep(0.0001)  # 100µs
        
        # Simulate validation logic
        if order_data['quantity'] <= 0:
            raise ValueError("Invalid quantity")
        
        return order_data
    
    def _simulate_market_data_processing(self):
        """Simulate market data processing operation"""
        # Simulate market data parsing
        market_data = {
            'symbol': 'AAPL',
            'bid': 149.95,
            'ask': 150.05,
            'timestamp': time.time_ns()
        }
        
        # Simulate processing time
        time.sleep(0.00005)  # 50µs
        
        # Simulate data validation
        if market_data['bid'] >= market_data['ask']:
            raise ValueError("Invalid bid/ask spread")
        
        return market_data
    
    def _simulate_risk_calculation(self):
        """Simulate risk calculation operation"""
        # Simulate portfolio risk calculation
        portfolio = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
                {'symbol': 'GOOGL', 'quantity': 50, 'price': 2500.0}
            ]
        }
        
        # Simulate calculation time
        time.sleep(0.0005)  # 500µs
        
        # Calculate portfolio value
        total_value = sum(pos['quantity'] * pos['price'] for pos in portfolio['positions'])
        
        return {'total_value': total_value, 'risk_metrics': {'var': total_value * 0.05}}
    
    def _simulate_rdma_write(self):
        """Simulate RDMA write operation"""
        # Create a test connection if not exists
        connection_id = 1
        if connection_id not in self.rdma_simulator.connections:
            self.rdma_simulator.create_connection(
                connection_id=connection_id,
                local_id=0,
                remote_id=1,
                latency_profile='local_network'
            )
        
        # Simulate RDMA write
        test_data = b'test_data_64_bytes' * 4  # 64 bytes
        result = self.rdma_simulator.simulate_rdma_operation(
            connection_id, RDMAOperation.WRITE, test_data
        )
        
        return result
    
    def _simulate_rdma_read(self):
        """Simulate RDMA read operation"""
        # Create a test connection if not exists
        connection_id = 1
        if connection_id not in self.rdma_simulator.connections:
            self.rdma_simulator.create_connection(
                connection_id=connection_id,
                local_id=0,
                remote_id=1,
                latency_profile='local_network'
            )
        
        # Simulate RDMA read
        test_data = b'test_data_64_bytes' * 4  # 64 bytes
        result = self.rdma_simulator.simulate_rdma_operation(
            connection_id, RDMAOperation.READ, test_data
        )
        
        return result
    
    def run_stress_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run comprehensive stress test"""
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Ramp up phase
            self._run_ramp_up(config)
            
            # Main stress test
            stress_results = self._run_main_stress_test(config)
            
            # Ramp down phase
            self._run_ramp_down(config)
            
            # Collect performance metrics
            performance_summary = self.performance_monitor.get_performance_summary()
            
            return {
                'stress_test_results': stress_results,
                'performance_summary': performance_summary,
                'system_health': self._assess_system_health(),
                'config': config
            }
            
        finally:
            self.performance_monitor.stop_monitoring()
    
    def _run_ramp_up(self, config: StressTestConfig):
        """Run ramp up phase"""
        ramp_up_steps = 5
        step_duration = config.ramp_up_time_seconds / ramp_up_steps
        
        for step in range(ramp_up_steps):
            threads = int(config.concurrent_threads * (step + 1) / ramp_up_steps)
            ops_per_thread = config.operations_per_thread // 10  # Reduced ops during ramp up
            
            futures = []
            for _ in range(threads):
                future = self.executor.submit(
                    self._run_mixed_operations, config.operation_mix, ops_per_thread
                )
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
            
            time.sleep(step_duration)
    
    def _run_main_stress_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run main stress test"""
        
        start_time = time.time()
        end_time = start_time + config.test_duration_seconds
        
        total_operations = 0
        successful_operations = 0
        failed_operations = 0
        
        while time.time() < end_time:
            futures = []
            
            # Submit work for all threads
            for _ in range(config.concurrent_threads):
                future = self.executor.submit(
                    self._run_mixed_operations, config.operation_mix, config.operations_per_thread
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    success, failures = future.result()
                    successful_operations += success
                    failed_operations += failures
                except Exception:
                    failed_operations += config.operations_per_thread
                
                total_operations += config.operations_per_thread
        
        actual_duration = time.time() - start_time
        
        return {
            'duration_seconds': actual_duration,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': failed_operations,
            'operations_per_second': total_operations / actual_duration,
            'error_rate': failed_operations / total_operations if total_operations > 0 else 0
        }
    
    def _run_ramp_down(self, config: StressTestConfig):
        """Run ramp down phase"""
        # Gradual reduction of load
        time.sleep(config.ramp_up_time_seconds / 2)
    
    def _run_mixed_operations(self, operation_mix: Dict[str, float], 
                             operations_count: int) -> Tuple[int, int]:
        """Run mixed operations based on operation mix"""
        
        successful = 0
        failed = 0
        
        for _ in range(operations_count):
            # Select operation based on mix
            operation = self._select_operation_from_mix(operation_mix)
            
            try:
                test_func = self.test_operations.get(operation)
                if test_func:
                    with self.timer.measure(operation):
                        test_func()
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        return successful, failed
    
    def _select_operation_from_mix(self, operation_mix: Dict[str, float]) -> str:
        """Select operation based on weighted mix"""
        if not operation_mix:
            return 'order_processing'  # Default operation
        
        # Simple weighted selection
        import random
        rand_val = random.random()
        cumulative = 0.0
        
        for operation, weight in operation_mix.items():
            cumulative += weight
            if rand_val <= cumulative:
                return operation
        
        # Fallback to first operation
        return list(operation_mix.keys())[0]
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess system health after stress test"""
        
        # Get current metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        active_alerts = self.performance_monitor.get_active_alerts()
        
        # Calculate health score
        health_score = 100.0
        
        # Penalize for active alerts
        for alert in active_alerts.values():
            if alert.level.value == 'critical':
                health_score -= 30
            elif alert.level.value == 'error':
                health_score -= 20
            elif alert.level.value == 'warning':
                health_score -= 10
        
        health_score = max(0, health_score)
        
        # Determine health status
        if health_score >= 90:
            health_status = 'excellent'
        elif health_score >= 70:
            health_status = 'good'
        elif health_score >= 50:
            health_status = 'degraded'
        else:
            health_status = 'poor'
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'active_alerts': len(active_alerts),
            'system_recommendations': self._generate_system_recommendations(active_alerts)
        }
    
    def _generate_system_recommendations(self, active_alerts: Dict[str, Any]) -> List[str]:
        """Generate system recommendations based on alerts"""
        recommendations = []
        
        for alert in active_alerts.values():
            if 'latency' in alert.metric_name:
                recommendations.append(
                    f"Consider optimizing {alert.metric_name} - current value: {alert.actual_value}"
                )
            elif 'throughput' in alert.metric_name:
                recommendations.append(
                    f"Investigate throughput degradation in {alert.metric_name}"
                )
            elif 'error_rate' in alert.metric_name:
                recommendations.append(
                    f"Address error rate issues in {alert.metric_name}"
                )
        
        if not recommendations:
            recommendations.append("System performing within acceptable parameters")
        
        return recommendations
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        passed_tests = [r for r in results if r.passed]
        failed_tests = [r for r in results if not r.passed]
        
        # Calculate summary statistics
        all_latencies = [r.actual_latency_ns for r in results if r.actual_latency_ns > 0]
        avg_latency = statistics.mean(all_latencies) if all_latencies else 0
        
        all_throughputs = [r.throughput_achieved for r in results if r.throughput_achieved > 0]
        avg_throughput = statistics.mean(all_throughputs) if all_throughputs else 0
        
        return {
            'timestamp': time.time(),
            'summary': {
                'total_tests': len(results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(passed_tests) / len(results) if results else 0,
                'average_latency_ns': avg_latency,
                'average_throughput': avg_throughput
            },
            'detailed_results': [
                {
                    'operation': r.operation,
                    'passed': r.passed,
                    'actual_latency_ns': r.actual_latency_ns,
                    'requirement_max_latency_ns': r.requirement.max_latency_ns,
                    'percentile_results': r.percentile_results,
                    'throughput_achieved': r.throughput_achieved,
                    'error_rate': r.error_rate,
                    'violations': r.violations
                }
                for r in results
            ],
            'recommendations': self._generate_optimization_recommendations(results)
        }
    
    def _generate_optimization_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate optimization recommendations based on validation results"""
        recommendations = []
        
        failed_results = [r for r in results if not r.passed]
        
        for result in failed_results:
            for violation in result.violations:
                if 'latency' in violation.lower():
                    recommendations.append(
                        f"Optimize {result.operation} to reduce latency below {result.requirement.max_latency_ns}ns"
                    )
                elif 'throughput' in violation.lower():
                    recommendations.append(
                        f"Scale {result.operation} to achieve required throughput"
                    )
                elif 'error' in violation.lower():
                    recommendations.append(
                        f"Improve error handling for {result.operation}"
                    )
        
        # Hardware recommendations
        hw_recommendations = self.hardware_profiler.optimize_for_hardware()
        recommendations.extend([
            f"Consider CPU affinity optimization: {hw_recommendations.get('cpu_affinity', [])}",
            f"Enable hardware optimizations: {hw_recommendations.get('memory_allocation', {})}"
        ])
        
        return list(set(recommendations))  # Remove duplicates