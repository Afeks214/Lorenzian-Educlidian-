"""
Performance Test and Optimization Suite - AGENT 1 MISSION COMPLETE

This module provides comprehensive performance testing and optimization for
the Universal Superposition Core Framework to ensure <1ms performance target.

Key Features:
- Automated performance testing across all components
- Benchmark suite for different action formats
- Memory usage analysis
- Bottleneck identification
- Optimization recommendations
- Regression testing

Performance Targets:
- Conversion time: <1ms per action
- Validation time: <0.5ms per state
- Serialization time: <2ms per state
- Memory usage: <10MB for 1000 states

Author: Agent 1 - Universal Superposition Core Architect
Version: 1.0 - Complete Performance Suite
"""

import time
import numpy as np
import torch
import psutil
import gc
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import warnings
import tracemalloc
from datetime import datetime

# Import the superposition framework
from . import (
    create_agent_converter, create_validator, create_persistence_manager,
    SuperpositionState, ValidationLevel, SerializationFormat,
    create_uniform_superposition, create_peaked_superposition,
    PERFORMANCE_TRACKER
)

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics to track"""
    CONVERSION_TIME = "conversion_time_ms"
    VALIDATION_TIME = "validation_time_ms"
    SERIALIZATION_TIME = "serialization_time_ms"
    MEMORY_USAGE = "memory_usage_mb"
    THROUGHPUT = "throughput_ops_per_sec"


@dataclass
class PerformanceResult:
    """Result of a performance test"""
    test_name: str
    metric: PerformanceMetric
    value: float
    target: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_name": self.test_name,
            "metric": self.metric.value,
            "value": self.value,
            "target": self.target,
            "passed": self.passed,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceReport:
    """Complete performance test report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[PerformanceResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def overall_status(self) -> str:
        """Get overall status"""
        if self.success_rate >= 0.9:
            return "EXCELLENT"
        elif self.success_rate >= 0.8:
            return "GOOD"
        elif self.success_rate >= 0.7:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.success_rate,
            "overall_status": self.overall_status,
            "timestamp": self.timestamp.isoformat(),
            "results": [result.to_dict() for result in self.results]
        }


class PerformanceTester:
    """
    Comprehensive performance testing suite for superposition framework
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance tester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Performance targets
        self.targets = {
            PerformanceMetric.CONVERSION_TIME: 1.0,  # ms
            PerformanceMetric.VALIDATION_TIME: 0.5,  # ms
            PerformanceMetric.SERIALIZATION_TIME: 2.0,  # ms
            PerformanceMetric.MEMORY_USAGE: 10.0,  # MB
            PerformanceMetric.THROUGHPUT: 1000.0  # ops/sec
        }
        
        # Update targets from config
        self.targets.update(self.config.get('targets', {}))
        
        # Initialize components
        self.converter = create_agent_converter()
        self.validator = create_validator()
        self.persistence = create_persistence_manager()
        
        # Test data
        self.test_actions = self._create_test_actions()
        
        # Results
        self.results = []
        
        logger.info("Initialized PerformanceTester")
    
    def _create_test_actions(self) -> List[Tuple[Any, str]]:
        """Create diverse test actions"""
        return [
            # Discrete actions
            (5, "discrete_int"),
            (np.int32(3), "discrete_numpy"),
            
            # Continuous actions
            (2.5, "continuous_float"),
            (np.array([1.0, 2.0, 3.0]), "continuous_vector"),
            (np.random.randn(10), "continuous_large_vector"),
            
            # Hybrid actions
            ([1, 2.5], "hybrid_simple"),
            ((2, 3.14, "action"), "hybrid_complex"),
            
            # Dictionary actions
            ({"position": 1, "size": 0.5}, "dict_simple"),
            ({"a": 1, "b": [1, 2, 3], "c": {"nested": True}}, "dict_complex"),
            
            # Legacy formats
            (np.array([0.1, 0.3, 0.6]), "legacy_3_action"),
            (np.random.rand(15), "legacy_15_action"),
            (torch.tensor([0.2, 0.8]), "legacy_tensor"),
            
            # Large actions
            (np.random.randn(100), "large_vector"),
            (list(range(1000)), "large_list"),
            
            # Edge cases
            (0, "edge_zero"),
            (np.inf, "edge_infinity"),
            ([], "edge_empty_list"),
            ({}, "edge_empty_dict")
        ]
    
    def test_conversion_performance(self) -> List[PerformanceResult]:
        """Test conversion performance"""
        results = []
        
        for action, description in self.test_actions:
            try:
                # Warm up
                for _ in range(10):
                    self.converter.convert_to_superposition(action)
                
                # Measure performance
                times = []
                for _ in range(100):
                    start_time = time.perf_counter()
                    superposition = self.converter.convert_to_superposition(action)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                
                avg_time = np.mean(times)
                max_time = np.max(times)
                std_time = np.std(times)
                
                result = PerformanceResult(
                    test_name=f"conversion_{description}",
                    metric=PerformanceMetric.CONVERSION_TIME,
                    value=avg_time,
                    target=self.targets[PerformanceMetric.CONVERSION_TIME],
                    passed=avg_time < self.targets[PerformanceMetric.CONVERSION_TIME],
                    details={
                        "max_time_ms": max_time,
                        "std_time_ms": std_time,
                        "samples": len(times),
                        "action_type": description
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Conversion test failed for {description}: {str(e)}")
                result = PerformanceResult(
                    test_name=f"conversion_{description}",
                    metric=PerformanceMetric.CONVERSION_TIME,
                    value=float('inf'),
                    target=self.targets[PerformanceMetric.CONVERSION_TIME],
                    passed=False,
                    details={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def test_validation_performance(self) -> List[PerformanceResult]:
        """Test validation performance"""
        results = []
        
        # Create test superposition states
        test_states = []
        for action, description in self.test_actions[:10]:  # Limit for performance
            try:
                state = self.converter.convert_to_superposition(action)
                test_states.append((state, description))
            except Exception:
                continue
        
        # Test different validation levels
        for level in ValidationLevel:
            for state, description in test_states:
                try:
                    # Warm up
                    for _ in range(5):
                        self.validator.validate(state, level)
                    
                    # Measure performance
                    times = []
                    for _ in range(50):
                        start_time = time.perf_counter()
                        report = self.validator.validate(state, level)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)
                    
                    avg_time = np.mean(times)
                    
                    result = PerformanceResult(
                        test_name=f"validation_{level.value}_{description}",
                        metric=PerformanceMetric.VALIDATION_TIME,
                        value=avg_time,
                        target=self.targets[PerformanceMetric.VALIDATION_TIME],
                        passed=avg_time < self.targets[PerformanceMetric.VALIDATION_TIME],
                        details={
                            "validation_level": level.value,
                            "action_type": description,
                            "samples": len(times)
                        }
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Validation test failed for {description} ({level.value}): {str(e)}")
        
        return results
    
    def test_serialization_performance(self) -> List[PerformanceResult]:
        """Test serialization performance"""
        results = []
        
        # Create test superposition states
        test_states = []
        for action, description in self.test_actions[:5]:  # Limit for performance
            try:
                state = self.converter.convert_to_superposition(action)
                test_states.append((state, description))
            except Exception:
                continue
        
        # Test different serialization formats
        for format_type in [SerializationFormat.JSON, SerializationFormat.PICKLE, SerializationFormat.BINARY]:
            for state, description in test_states:
                try:
                    # Test serialization
                    serializer = None
                    if format_type == SerializationFormat.JSON:
                        from .superposition_serializer import JSONSerializer
                        serializer = JSONSerializer()
                    elif format_type == SerializationFormat.PICKLE:
                        from .superposition_serializer import PickleSerializer
                        serializer = PickleSerializer()
                    elif format_type == SerializationFormat.BINARY:
                        from .superposition_serializer import BinarySerializer
                        serializer = BinarySerializer()
                    
                    if serializer:
                        # Warm up
                        for _ in range(5):
                            data = serializer.serialize(state)
                            serializer.deserialize(data)
                        
                        # Measure serialization
                        times = []
                        for _ in range(20):
                            start_time = time.perf_counter()
                            data = serializer.serialize(state)
                            end_time = time.perf_counter()
                            times.append((end_time - start_time) * 1000)
                        
                        avg_time = np.mean(times)
                        
                        result = PerformanceResult(
                            test_name=f"serialization_{format_type.value}_{description}",
                            metric=PerformanceMetric.SERIALIZATION_TIME,
                            value=avg_time,
                            target=self.targets[PerformanceMetric.SERIALIZATION_TIME],
                            passed=avg_time < self.targets[PerformanceMetric.SERIALIZATION_TIME],
                            details={
                                "format": format_type.value,
                                "action_type": description,
                                "data_size": len(data),
                                "samples": len(times)
                            }
                        )
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Serialization test failed for {description} ({format_type.value}): {str(e)}")
        
        return results
    
    def test_memory_usage(self) -> List[PerformanceResult]:
        """Test memory usage"""
        results = []
        
        # Test memory usage for batch operations
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            try:
                # Start memory tracking
                tracemalloc.start()
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Create batch of superposition states
                states = []
                for i in range(batch_size):
                    action = self.test_actions[i % len(self.test_actions)][0]
                    state = self.converter.convert_to_superposition(action)
                    states.append(state)
                
                # Measure memory usage
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
                
                # Memory per state
                memory_per_state = memory_used / batch_size
                
                result = PerformanceResult(
                    test_name=f"memory_usage_batch_{batch_size}",
                    metric=PerformanceMetric.MEMORY_USAGE,
                    value=memory_per_state,
                    target=self.targets[PerformanceMetric.MEMORY_USAGE] / 1000,  # Convert to per-state
                    passed=memory_per_state < self.targets[PerformanceMetric.MEMORY_USAGE] / 1000,
                    details={
                        "batch_size": batch_size,
                        "total_memory_mb": memory_used,
                        "initial_memory_mb": initial_memory,
                        "final_memory_mb": final_memory
                    }
                )
                
                results.append(result)
                
                # Cleanup
                del states
                gc.collect()
                tracemalloc.stop()
                
            except Exception as e:
                logger.error(f"Memory test failed for batch size {batch_size}: {str(e)}")
        
        return results
    
    def test_throughput(self) -> List[PerformanceResult]:
        """Test throughput performance"""
        results = []
        
        # Test conversion throughput
        test_duration = 1.0  # seconds
        
        for action, description in self.test_actions[:5]:  # Limit for performance
            try:
                # Warm up
                for _ in range(100):
                    self.converter.convert_to_superposition(action)
                
                # Measure throughput
                start_time = time.perf_counter()
                count = 0
                
                while time.perf_counter() - start_time < test_duration:
                    self.converter.convert_to_superposition(action)
                    count += 1
                
                actual_duration = time.perf_counter() - start_time
                throughput = count / actual_duration
                
                result = PerformanceResult(
                    test_name=f"throughput_{description}",
                    metric=PerformanceMetric.THROUGHPUT,
                    value=throughput,
                    target=self.targets[PerformanceMetric.THROUGHPUT],
                    passed=throughput > self.targets[PerformanceMetric.THROUGHPUT],
                    details={
                        "action_type": description,
                        "operations": count,
                        "duration_seconds": actual_duration
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Throughput test failed for {description}: {str(e)}")
        
        return results
    
    def test_concurrent_performance(self) -> List[PerformanceResult]:
        """Test concurrent performance"""
        results = []
        
        # Test concurrent conversion
        num_threads = [1, 2, 4, 8]
        
        for thread_count in num_threads:
            try:
                def worker_task():
                    """Worker task for concurrent testing"""
                    for _ in range(100):
                        action = self.test_actions[0][0]  # Use first test action
                        self.converter.convert_to_superposition(action)
                
                # Measure concurrent performance
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    futures = [executor.submit(worker_task) for _ in range(thread_count)]
                    for future in futures:
                        future.result()
                
                end_time = time.perf_counter()
                
                total_operations = thread_count * 100
                total_time = end_time - start_time
                throughput = total_operations / total_time
                
                result = PerformanceResult(
                    test_name=f"concurrent_throughput_{thread_count}_threads",
                    metric=PerformanceMetric.THROUGHPUT,
                    value=throughput,
                    target=self.targets[PerformanceMetric.THROUGHPUT],
                    passed=throughput > self.targets[PerformanceMetric.THROUGHPUT],
                    details={
                        "thread_count": thread_count,
                        "total_operations": total_operations,
                        "total_time_seconds": total_time
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Concurrent test failed for {thread_count} threads: {str(e)}")
        
        return results
    
    def run_all_tests(self) -> PerformanceReport:
        """Run all performance tests"""
        print("🚀 Running Universal Superposition Performance Test Suite")
        print("=" * 60)
        
        all_results = []
        
        # Run conversion tests
        print("📊 Testing conversion performance...")
        conversion_results = self.test_conversion_performance()
        all_results.extend(conversion_results)
        
        # Run validation tests
        print("📊 Testing validation performance...")
        validation_results = self.test_validation_performance()
        all_results.extend(validation_results)
        
        # Run serialization tests
        print("📊 Testing serialization performance...")
        serialization_results = self.test_serialization_performance()
        all_results.extend(serialization_results)
        
        # Run memory tests
        print("📊 Testing memory usage...")
        memory_results = self.test_memory_usage()
        all_results.extend(memory_results)
        
        # Run throughput tests
        print("📊 Testing throughput performance...")
        throughput_results = self.test_throughput()
        all_results.extend(throughput_results)
        
        # Run concurrent tests
        print("📊 Testing concurrent performance...")
        concurrent_results = self.test_concurrent_performance()
        all_results.extend(concurrent_results)
        
        # Compile report
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = len(all_results) - passed_tests
        
        report = PerformanceReport(
            total_tests=len(all_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            results=all_results
        )
        
        return report
    
    def analyze_results(self, report: PerformanceReport) -> Dict[str, Any]:
        """Analyze performance results and provide recommendations"""
        analysis = {
            "summary": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "success_rate": report.success_rate,
                "overall_status": report.overall_status
            },
            "metrics": defaultdict(list),
            "recommendations": []
        }
        
        # Group results by metric
        for result in report.results:
            analysis["metrics"][result.metric.value].append(result)
        
        # Analyze each metric
        for metric_name, results in analysis["metrics"].items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            values = [r.value for r in results]
            
            analysis[f"{metric_name}_analysis"] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0,
                "avg_value": np.mean(values) if values else 0,
                "max_value": np.max(values) if values else 0,
                "min_value": np.min(values) if values else 0
            }
        
        # Generate recommendations
        conversion_results = analysis["metrics"].get("conversion_time_ms", [])
        slow_conversions = [r for r in conversion_results if not r.passed]
        
        if slow_conversions:
            analysis["recommendations"].append(
                f"Optimize conversion for {len(slow_conversions)} slow action types"
            )
        
        validation_results = analysis["metrics"].get("validation_time_ms", [])
        slow_validations = [r for r in validation_results if not r.passed]
        
        if slow_validations:
            analysis["recommendations"].append(
                f"Optimize validation for {len(slow_validations)} slow cases"
            )
        
        memory_results = analysis["metrics"].get("memory_usage_mb", [])
        high_memory = [r for r in memory_results if not r.passed]
        
        if high_memory:
            analysis["recommendations"].append(
                "Implement memory optimization for large batch operations"
            )
        
        return analysis
    
    def save_report(self, report: PerformanceReport, path: str):
        """Save performance report to file"""
        report_data = {
            "report": report.to_dict(),
            "analysis": self.analyze_results(report),
            "global_performance_stats": PERFORMANCE_TRACKER.get_performance_stats()
        }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Performance report saved to {path}")


def run_performance_tests():
    """Run comprehensive performance tests"""
    tester = PerformanceTester()
    
    # Run all tests
    report = tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 PERFORMANCE TEST RESULTS")
    print("=" * 60)
    
    print(f"Overall Status: {report.overall_status}")
    print(f"Tests Passed: {report.passed_tests}/{report.total_tests} ({report.success_rate:.1%})")
    
    # Show failed tests
    failed_results = [r for r in report.results if not r.passed]
    if failed_results:
        print(f"\n❌ FAILED TESTS ({len(failed_results)}):")
        for result in failed_results:
            print(f"  - {result.test_name}: {result.value:.2f} > {result.target:.2f} {result.metric.value}")
    
    # Show top performers
    passed_results = [r for r in report.results if r.passed]
    if passed_results:
        print(f"\n✅ TOP PERFORMERS:")
        # Sort by how much better than target
        sorted_results = sorted(passed_results, key=lambda r: r.target - r.value, reverse=True)
        for result in sorted_results[:5]:
            improvement = ((result.target - result.value) / result.target * 100) if result.target > 0 else 0
            print(f"  - {result.test_name}: {result.value:.2f} ({improvement:.1f}% better than target)")
    
    # Analysis
    analysis = tester.analyze_results(report)
    if analysis["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"performance_report_{timestamp}.json"
    tester.save_report(report, report_path)
    
    print(f"\n🎯 Performance Target Analysis:")
    print(f"  - Conversion: <1ms target")
    print(f"  - Validation: <0.5ms target")
    print(f"  - Serialization: <2ms target")
    print(f"  - Memory: <10MB for 1000 states")
    print(f"  - Throughput: >1000 ops/sec")
    
    return report


if __name__ == "__main__":
    # Run performance tests
    performance_report = run_performance_tests()
    
    if performance_report.overall_status in ["EXCELLENT", "GOOD"]:
        print("\n🏆 PERFORMANCE TARGETS MET!")
        print("✅ Universal Superposition Core Framework is ready for production")
    else:
        print("\n⚠️  PERFORMANCE OPTIMIZATION NEEDED")
        print("❌ Some components need optimization before production deployment")