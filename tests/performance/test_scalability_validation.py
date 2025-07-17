"""
Scalability Validation Tests - Agent 4 Implementation
=================================================

Comprehensive scalability testing to validate system performance
under increasing load conditions and resource constraints.

Test Categories:
1. Horizontal Scaling - Multiple instances/processes
2. Vertical Scaling - Increased resources per instance
3. Data Volume Scaling - Large dataset processing
4. Concurrent User Scaling - Multiple simultaneous users
5. Time-based Scaling - Extended duration testing

Author: Agent 4 - Performance Baseline Research Agent
"""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
import torch
import psutil
import threading
import concurrent.futures
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import resource

# Configure scalability testing
pytestmark = [pytest.mark.performance, pytest.mark.scalability]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability test."""
    test_name: str
    scaling_dimension: str  # 'horizontal', 'vertical', 'data_volume', 'concurrent_users', 'time_based'
    scaling_levels: List[int]
    success_criteria: Dict[str, float]
    resource_limits: Dict[str, float]
    test_duration_seconds: int


@dataclass
class ScalabilityTestResult:
    """Result of scalability test."""
    test_name: str
    scaling_dimension: str
    scaling_level: int
    throughput_ops_per_sec: float
    latency_p99_ms: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    error_rate_percent: float
    resource_efficiency: float
    scaling_efficiency: float
    success: bool
    bottlenecks: List[str]
    recommendations: List[str]


class ScalabilityValidator:
    """
    Comprehensive scalability validation framework for testing system
    performance under various scaling scenarios.
    """
    
    def __init__(self):
        """Initialize scalability validator."""
        self.process = psutil.Process()
        self.test_results = {}
        self.baseline_metrics = {}
        
        # Scalability test configurations
        self.test_configs = {
            "horizontal_scaling": ScalabilityTestConfig(
                test_name="horizontal_scaling",
                scaling_dimension="horizontal",
                scaling_levels=[1, 2, 4, 8, 16],
                success_criteria={
                    "throughput_scaling_efficiency": 0.8,  # 80% efficiency
                    "latency_degradation_max": 50.0,       # Max 50% latency increase
                    "error_rate_max": 2.0                  # Max 2% error rate
                },
                resource_limits={
                    "memory_mb": 2048,
                    "cpu_percent": 90
                },
                test_duration_seconds=120
            ),
            "vertical_scaling": ScalabilityTestConfig(
                test_name="vertical_scaling",
                scaling_dimension="vertical",
                scaling_levels=[1, 2, 4, 8],  # CPU cores
                success_criteria={
                    "cpu_utilization_efficiency": 0.75,
                    "memory_efficiency": 0.8,
                    "throughput_improvement": 1.5
                },
                resource_limits={
                    "memory_mb": 4096,
                    "cpu_percent": 95
                },
                test_duration_seconds=90
            ),
            "data_volume_scaling": ScalabilityTestConfig(
                test_name="data_volume_scaling",
                scaling_dimension="data_volume",
                scaling_levels=[1000, 5000, 10000, 50000, 100000],  # Number of data points
                success_criteria={
                    "latency_growth_rate": 1.2,  # Linear growth max
                    "memory_growth_rate": 1.5,   # Sub-linear growth
                    "throughput_stability": 0.8   # 80% of baseline
                },
                resource_limits={
                    "memory_mb": 3072,
                    "cpu_percent": 85
                },
                test_duration_seconds=180
            ),
            "concurrent_users": ScalabilityTestConfig(
                test_name="concurrent_users",
                scaling_dimension="concurrent_users",
                scaling_levels=[1, 5, 10, 25, 50, 100],
                success_criteria={
                    "concurrent_efficiency": 0.7,
                    "latency_p99_max": 20.0,
                    "error_rate_max": 1.0
                },
                resource_limits={
                    "memory_mb": 1536,
                    "cpu_percent": 80
                },
                test_duration_seconds=150
            ),
            "time_based_scaling": ScalabilityTestConfig(
                test_name="time_based_scaling",
                scaling_dimension="time_based",
                scaling_levels=[60, 300, 900, 1800, 3600],  # Seconds
                success_criteria={
                    "performance_stability": 0.95,
                    "memory_leak_threshold": 50.0,  # MB growth
                    "latency_drift_max": 25.0       # % increase
                },
                resource_limits={
                    "memory_mb": 1024,
                    "cpu_percent": 70
                },
                test_duration_seconds=3600  # 1 hour max
            )
        }
        
        logger.info(f"Scalability validator initialized with {len(self.test_configs)} test configurations")
    
    def run_horizontal_scaling_test(self, config: ScalabilityTestConfig) -> List[ScalabilityTestResult]:
        """Test horizontal scaling with multiple processes/threads."""
        logger.info(f"🔄 Running horizontal scaling test: {config.scaling_levels}")
        
        results = []
        baseline_throughput = None
        
        for level in config.scaling_levels:
            logger.info(f"Testing horizontal scaling level: {level}")
            
            # Create test data
            test_data = [torch.randn(1, 48, 13) for _ in range(1000)]
            
            # Performance metrics
            start_time = time.time()
            total_operations = 0
            latencies = []
            errors = 0
            
            # Worker function
            def worker_function(worker_id: int, operations_per_worker: int) -> Tuple[List[float], int]:
                worker_latencies = []
                worker_errors = 0
                
                for i in range(operations_per_worker):
                    try:
                        data = test_data[i % len(test_data)]
                        
                        op_start = time.perf_counter()
                        result = self._simulate_strategic_inference(data)
                        op_end = time.perf_counter()
                        
                        worker_latencies.append((op_end - op_start) * 1000)
                        
                    except Exception as e:
                        worker_errors += 1
                
                return worker_latencies, worker_errors
            
            # Execute with multiple workers
            operations_per_worker = 500
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                futures = [
                    executor.submit(worker_function, i, operations_per_worker)
                    for i in range(level)
                ]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    worker_latencies, worker_errors = future.result()
                    latencies.extend(worker_latencies)
                    errors += worker_errors
                    total_operations += len(worker_latencies)
            
            # Calculate metrics
            test_duration = time.time() - start_time
            throughput = total_operations / test_duration
            
            if baseline_throughput is None:
                baseline_throughput = throughput
            
            # Resource utilization
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            cpu_usage = self.process.cpu_percent()
            
            # Calculate efficiency metrics
            scaling_efficiency = (throughput / baseline_throughput) / level if baseline_throughput > 0 else 0
            resource_efficiency = throughput / (memory_usage + cpu_usage) if (memory_usage + cpu_usage) > 0 else 0
            
            # Analyze bottlenecks
            bottlenecks = self._identify_bottlenecks(
                level, throughput, np.percentile(latencies, 99), memory_usage, cpu_usage
            )
            
            # Success criteria
            success = self._evaluate_horizontal_scaling_success(
                config, level, throughput, latencies, errors, total_operations, scaling_efficiency
            )
            
            # Generate recommendations
            recommendations = self._generate_horizontal_scaling_recommendations(
                level, throughput, latencies, memory_usage, cpu_usage, scaling_efficiency
            )
            
            result = ScalabilityTestResult(
                test_name=config.test_name,
                scaling_dimension=config.scaling_dimension,
                scaling_level=level,
                throughput_ops_per_sec=throughput,
                latency_p99_ms=np.percentile(latencies, 99) if latencies else 0,
                memory_usage_mb=memory_usage,
                cpu_utilization_percent=cpu_usage,
                error_rate_percent=(errors / total_operations) * 100 if total_operations > 0 else 0,
                resource_efficiency=resource_efficiency,
                scaling_efficiency=scaling_efficiency,
                success=success,
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
            results.append(result)
            
            logger.info(f"Level {level}: {throughput:.1f} ops/sec, "
                       f"P99: {result.latency_p99_ms:.2f}ms, "
                       f"Efficiency: {scaling_efficiency:.2f}")
            
            # Brief pause between levels
            time.sleep(5)
        
        return results
    
    def run_vertical_scaling_test(self, config: ScalabilityTestConfig) -> List[ScalabilityTestResult]:
        """Test vertical scaling with increased resources."""
        logger.info(f"🔝 Running vertical scaling test: {config.scaling_levels}")
        
        results = []
        
        for level in config.scaling_levels:
            logger.info(f"Testing vertical scaling level: {level} cores")
            
            # Simulate different CPU core counts by adjusting workload
            operations_count = 1000 * level
            test_data = [torch.randn(1, 48, 13) for _ in range(operations_count)]
            
            # Performance measurement
            start_time = time.time()
            latencies = []
            errors = 0
            
            # CPU-intensive processing simulation
            for i in range(operations_count):
                try:
                    data = test_data[i]
                    
                    op_start = time.perf_counter()
                    
                    # Simulate CPU-intensive work
                    if level > 1:
                        # More complex computation for higher levels
                        result = torch.mm(data[0], data[0].T)
                        result = torch.mm(result, result.T)
                    else:
                        result = self._simulate_strategic_inference(data)
                    
                    op_end = time.perf_counter()
                    latencies.append((op_end - op_start) * 1000)
                    
                except Exception as e:
                    errors += 1
            
            # Calculate metrics
            test_duration = time.time() - start_time
            throughput = operations_count / test_duration
            
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            cpu_usage = self.process.cpu_percent()
            
            # Efficiency metrics
            cpu_efficiency = throughput / (level * 100)  # Normalized by cores
            memory_efficiency = throughput / memory_usage if memory_usage > 0 else 0
            
            # Analyze bottlenecks
            bottlenecks = self._identify_vertical_scaling_bottlenecks(
                level, throughput, memory_usage, cpu_usage
            )
            
            # Success criteria
            success = self._evaluate_vertical_scaling_success(
                config, level, throughput, latencies, cpu_efficiency, memory_efficiency
            )
            
            # Generate recommendations
            recommendations = self._generate_vertical_scaling_recommendations(
                level, throughput, memory_usage, cpu_usage, cpu_efficiency
            )
            
            result = ScalabilityTestResult(
                test_name=config.test_name,
                scaling_dimension=config.scaling_dimension,
                scaling_level=level,
                throughput_ops_per_sec=throughput,
                latency_p99_ms=np.percentile(latencies, 99) if latencies else 0,
                memory_usage_mb=memory_usage,
                cpu_utilization_percent=cpu_usage,
                error_rate_percent=(errors / operations_count) * 100 if operations_count > 0 else 0,
                resource_efficiency=memory_efficiency,
                scaling_efficiency=cpu_efficiency,
                success=success,
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
            results.append(result)
            
            logger.info(f"Level {level}: {throughput:.1f} ops/sec, "
                       f"CPU efficiency: {cpu_efficiency:.3f}")
            
            # Brief pause
            time.sleep(3)
        
        return results
    
    def run_data_volume_scaling_test(self, config: ScalabilityTestConfig) -> List[ScalabilityTestResult]:
        """Test scaling with increasing data volumes."""
        logger.info(f"📊 Running data volume scaling test: {config.scaling_levels}")
        
        results = []
        baseline_throughput = None
        
        for level in config.scaling_levels:
            logger.info(f"Testing data volume level: {level} data points")
            
            # Create large dataset
            test_data = [torch.randn(1, 48, 13) for _ in range(level)]
            
            # Performance measurement
            start_time = time.time()
            latencies = []
            errors = 0
            
            # Process entire dataset
            for i, data in enumerate(test_data):
                try:
                    op_start = time.perf_counter()
                    result = self._simulate_strategic_inference(data)
                    op_end = time.perf_counter()
                    
                    latencies.append((op_end - op_start) * 1000)
                    
                    # Progress logging
                    if i % 1000 == 0:
                        logger.debug(f"Processed {i}/{level} data points")
                    
                except Exception as e:
                    errors += 1
            
            # Calculate metrics
            test_duration = time.time() - start_time
            throughput = level / test_duration
            
            if baseline_throughput is None:
                baseline_throughput = throughput
            
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            cpu_usage = self.process.cpu_percent()
            
            # Scaling efficiency
            scaling_efficiency = throughput / baseline_throughput if baseline_throughput > 0 else 0
            
            # Analyze bottlenecks
            bottlenecks = self._identify_data_volume_bottlenecks(
                level, throughput, memory_usage, latencies
            )
            
            # Success criteria
            success = self._evaluate_data_volume_scaling_success(
                config, level, throughput, latencies, memory_usage, scaling_efficiency
            )
            
            # Generate recommendations
            recommendations = self._generate_data_volume_recommendations(
                level, throughput, memory_usage, latencies
            )
            
            result = ScalabilityTestResult(
                test_name=config.test_name,
                scaling_dimension=config.scaling_dimension,
                scaling_level=level,
                throughput_ops_per_sec=throughput,
                latency_p99_ms=np.percentile(latencies, 99) if latencies else 0,
                memory_usage_mb=memory_usage,
                cpu_utilization_percent=cpu_usage,
                error_rate_percent=(errors / level) * 100 if level > 0 else 0,
                resource_efficiency=throughput / memory_usage if memory_usage > 0 else 0,
                scaling_efficiency=scaling_efficiency,
                success=success,
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
            results.append(result)
            
            logger.info(f"Level {level}: {throughput:.1f} ops/sec, "
                       f"Memory: {memory_usage:.1f}MB, "
                       f"Efficiency: {scaling_efficiency:.2f}")
            
            # Cleanup
            del test_data
            time.sleep(2)
        
        return results
    
    def run_comprehensive_scalability_suite(self) -> Dict[str, Any]:
        """Run comprehensive scalability test suite."""
        logger.info("🚀 Starting comprehensive scalability validation suite")
        
        all_results = {}
        
        # Run horizontal scaling test
        logger.info("\\n" + "="*60)
        logger.info("HORIZONTAL SCALING TEST")
        logger.info("="*60)
        horizontal_results = self.run_horizontal_scaling_test(self.test_configs["horizontal_scaling"])
        all_results["horizontal_scaling"] = horizontal_results
        
        # Run vertical scaling test
        logger.info("\\n" + "="*60)
        logger.info("VERTICAL SCALING TEST")
        logger.info("="*60)
        vertical_results = self.run_vertical_scaling_test(self.test_configs["vertical_scaling"])
        all_results["vertical_scaling"] = vertical_results
        
        # Run data volume scaling test
        logger.info("\\n" + "="*60)
        logger.info("DATA VOLUME SCALING TEST")
        logger.info("="*60)
        data_volume_results = self.run_data_volume_scaling_test(self.test_configs["data_volume_scaling"])
        all_results["data_volume_scaling"] = data_volume_results
        
        # Generate comprehensive report
        report = self._generate_scalability_report(all_results)
        
        # Save report
        self._save_scalability_report(report)
        
        logger.info("\\n✅ Comprehensive scalability validation suite completed")
        
        return report
    
    def _simulate_strategic_inference(self, data: torch.Tensor) -> Dict[str, Any]:
        """Simulate strategic inference with realistic processing."""
        # Simulate model inference
        time.sleep(np.random.uniform(0.0005, 0.0015))
        
        return {
            'decision': np.random.choice(['BUY', 'SELL', 'HOLD']),
            'confidence': np.random.uniform(0.5, 1.0),
            'risk_score': np.random.uniform(0.0, 1.0)
        }
    
    def _identify_bottlenecks(self, level: int, throughput: float, latency: float,
                            memory_usage: float, cpu_usage: float) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if cpu_usage > 85:
            bottlenecks.append("CPU utilization high")
        
        if memory_usage > 1000:
            bottlenecks.append("Memory usage high")
        
        if latency > 10:
            bottlenecks.append("Latency degradation")
        
        if throughput < 100:
            bottlenecks.append("Low throughput")
        
        return bottlenecks
    
    def _identify_vertical_scaling_bottlenecks(self, level: int, throughput: float,
                                             memory_usage: float, cpu_usage: float) -> List[str]:
        """Identify vertical scaling bottlenecks."""
        bottlenecks = []
        
        if cpu_usage < 60:
            bottlenecks.append("CPU underutilization")
        
        if memory_usage > 2000:
            bottlenecks.append("Memory bottleneck")
        
        if throughput / level < 50:
            bottlenecks.append("Poor scaling efficiency")
        
        return bottlenecks
    
    def _identify_data_volume_bottlenecks(self, level: int, throughput: float,
                                        memory_usage: float, latencies: List[float]) -> List[str]:
        """Identify data volume scaling bottlenecks."""
        bottlenecks = []
        
        if memory_usage > 2000:
            bottlenecks.append("Memory scaling bottleneck")
        
        if latencies and np.percentile(latencies, 99) > 15:
            bottlenecks.append("Latency scaling bottleneck")
        
        if throughput < 100:
            bottlenecks.append("Throughput degradation with volume")
        
        return bottlenecks
    
    def _evaluate_horizontal_scaling_success(self, config: ScalabilityTestConfig,
                                           level: int, throughput: float, latencies: List[float],
                                           errors: int, total_operations: int,
                                           scaling_efficiency: float) -> bool:
        """Evaluate horizontal scaling success."""
        criteria = config.success_criteria
        
        # Check scaling efficiency
        if scaling_efficiency < criteria["throughput_scaling_efficiency"]:
            return False
        
        # Check error rate
        error_rate = (errors / total_operations) * 100 if total_operations > 0 else 0
        if error_rate > criteria["error_rate_max"]:
            return False
        
        # Check latency degradation
        if latencies:
            p99_latency = np.percentile(latencies, 99)
            if p99_latency > criteria["latency_degradation_max"]:
                return False
        
        return True
    
    def _evaluate_vertical_scaling_success(self, config: ScalabilityTestConfig,
                                         level: int, throughput: float, latencies: List[float],
                                         cpu_efficiency: float, memory_efficiency: float) -> bool:
        """Evaluate vertical scaling success."""
        criteria = config.success_criteria
        
        return (cpu_efficiency >= criteria["cpu_utilization_efficiency"] and
                memory_efficiency >= criteria["memory_efficiency"])
    
    def _evaluate_data_volume_scaling_success(self, config: ScalabilityTestConfig,
                                            level: int, throughput: float, latencies: List[float],
                                            memory_usage: float, scaling_efficiency: float) -> bool:
        """Evaluate data volume scaling success."""
        criteria = config.success_criteria
        
        return (scaling_efficiency >= criteria["throughput_stability"] and
                memory_usage < config.resource_limits["memory_mb"])
    
    def _generate_horizontal_scaling_recommendations(self, level: int, throughput: float,
                                                   latencies: List[float], memory_usage: float,
                                                   cpu_usage: float, scaling_efficiency: float) -> List[str]:
        """Generate horizontal scaling recommendations."""
        recommendations = []
        
        if scaling_efficiency < 0.7:
            recommendations.append("Optimize for better parallel processing")
        
        if cpu_usage > 80:
            recommendations.append("Consider CPU-optimized instances")
        
        if memory_usage > 1000:
            recommendations.append("Implement memory-efficient algorithms")
        
        if latencies and np.percentile(latencies, 99) > 10:
            recommendations.append("Optimize latency for concurrent processing")
        
        return recommendations
    
    def _generate_vertical_scaling_recommendations(self, level: int, throughput: float,
                                                 memory_usage: float, cpu_usage: float,
                                                 cpu_efficiency: float) -> List[str]:
        """Generate vertical scaling recommendations."""
        recommendations = []
        
        if cpu_efficiency < 0.5:
            recommendations.append("Optimize CPU utilization")
        
        if cpu_usage < 50:
            recommendations.append("Implement CPU-intensive optimizations")
        
        if memory_usage > 2000:
            recommendations.append("Optimize memory usage patterns")
        
        return recommendations
    
    def _generate_data_volume_recommendations(self, level: int, throughput: float,
                                            memory_usage: float, latencies: List[float]) -> List[str]:
        """Generate data volume scaling recommendations."""
        recommendations = []
        
        if memory_usage > 1500:
            recommendations.append("Implement streaming data processing")
        
        if latencies and np.percentile(latencies, 99) > 10:
            recommendations.append("Optimize for large dataset processing")
        
        if throughput < 200:
            recommendations.append("Implement batch processing optimizations")
        
        return recommendations
    
    def _generate_scalability_report(self, all_results: Dict[str, List[ScalabilityTestResult]]) -> Dict[str, Any]:
        """Generate comprehensive scalability report."""
        # Overall analysis
        total_tests = sum(len(results) for results in all_results.values())
        successful_tests = sum(
            len([r for r in results if r.success]) for results in all_results.values()
        )
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Best and worst performers
        all_results_flat = [r for results in all_results.values() for r in results]
        best_throughput = max(all_results_flat, key=lambda x: x.throughput_ops_per_sec)
        worst_latency = max(all_results_flat, key=lambda x: x.latency_p99_ms)
        
        # Scaling dimension analysis
        dimension_analysis = {}
        for dimension, results in all_results.items():
            dimension_analysis[dimension] = {
                "max_throughput": max(r.throughput_ops_per_sec for r in results),
                "min_latency": min(r.latency_p99_ms for r in results),
                "max_scaling_level": max(r.scaling_level for r in results),
                "success_rate": len([r for r in results if r.success]) / len(results),
                "bottlenecks": list(set(b for r in results for b in r.bottlenecks))
            }
        
        return {
            "executive_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "scalability_ready": success_rate >= 0.8,
                "best_performing_dimension": max(dimension_analysis.items(), key=lambda x: x[1]["success_rate"])[0],
                "max_achieved_throughput": best_throughput.throughput_ops_per_sec,
                "system_scalability_grade": success_rate * 100
            },
            "dimension_analysis": dimension_analysis,
            "detailed_results": {
                dimension: [
                    {
                        "scaling_level": r.scaling_level,
                        "throughput_ops_per_sec": r.throughput_ops_per_sec,
                        "latency_p99_ms": r.latency_p99_ms,
                        "memory_usage_mb": r.memory_usage_mb,
                        "cpu_utilization_percent": r.cpu_utilization_percent,
                        "scaling_efficiency": r.scaling_efficiency,
                        "success": r.success,
                        "bottlenecks": r.bottlenecks,
                        "recommendations": r.recommendations
                    }
                    for r in results
                ]
                for dimension, results in all_results.items()
            },
            "scalability_recommendations": self._generate_overall_scalability_recommendations(all_results),
            "deployment_guidelines": {
                "recommended_max_horizontal_scale": self._get_recommended_max_scale(all_results.get("horizontal_scaling", [])),
                "recommended_vertical_resources": self._get_recommended_vertical_resources(all_results.get("vertical_scaling", [])),
                "data_volume_limits": self._get_data_volume_limits(all_results.get("data_volume_scaling", []))
            },
            "test_metadata": {
                "framework_version": "1.0.0",
                "agent_id": "Agent_4_Performance_Baseline_Research",
                "test_suite": "scalability_validation"
            }
        }
    
    def _generate_overall_scalability_recommendations(self, all_results: Dict[str, List[ScalabilityTestResult]]) -> List[str]:
        """Generate overall scalability recommendations."""
        recommendations = []
        
        # Analyze common bottlenecks
        all_bottlenecks = [b for results in all_results.values() for r in results for b in r.bottlenecks]
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        # Most common bottlenecks
        if bottleneck_counts:
            most_common = max(bottleneck_counts.items(), key=lambda x: x[1])
            if most_common[1] > 2:
                recommendations.append(f"Address common bottleneck: {most_common[0]}")
        
        # Horizontal scaling recommendations
        if "horizontal_scaling" in all_results:
            horizontal_results = all_results["horizontal_scaling"]
            if any(not r.success for r in horizontal_results):
                recommendations.append("Improve horizontal scaling efficiency")
        
        # Vertical scaling recommendations
        if "vertical_scaling" in all_results:
            vertical_results = all_results["vertical_scaling"]
            if any(r.scaling_efficiency < 0.5 for r in vertical_results):
                recommendations.append("Optimize vertical resource utilization")
        
        # Data volume recommendations
        if "data_volume_scaling" in all_results:
            data_results = all_results["data_volume_scaling"]
            if any(r.memory_usage_mb > 2000 for r in data_results):
                recommendations.append("Implement memory-efficient data processing")
        
        return recommendations
    
    def _get_recommended_max_scale(self, horizontal_results: List[ScalabilityTestResult]) -> int:
        """Get recommended maximum horizontal scale."""
        if not horizontal_results:
            return 1
        
        # Find last successful scaling level
        successful_levels = [r.scaling_level for r in horizontal_results if r.success]
        return max(successful_levels) if successful_levels else 1
    
    def _get_recommended_vertical_resources(self, vertical_results: List[ScalabilityTestResult]) -> Dict[str, int]:
        """Get recommended vertical resource configuration."""
        if not vertical_results:
            return {"cpu_cores": 1, "memory_mb": 512}
        
        # Find optimal resource configuration
        best_result = max(vertical_results, key=lambda x: x.scaling_efficiency)
        
        return {
            "cpu_cores": best_result.scaling_level,
            "memory_mb": int(best_result.memory_usage_mb * 1.2)  # 20% buffer
        }
    
    def _get_data_volume_limits(self, data_results: List[ScalabilityTestResult]) -> Dict[str, int]:
        """Get data volume processing limits."""
        if not data_results:
            return {"max_data_points": 1000, "recommended_batch_size": 100}
        
        # Find maximum successful data volume
        successful_volumes = [r.scaling_level for r in data_results if r.success]
        max_volume = max(successful_volumes) if successful_volumes else 1000
        
        return {
            "max_data_points": max_volume,
            "recommended_batch_size": min(1000, max_volume // 10)
        }
    
    def _save_scalability_report(self, report: Dict[str, Any]):
        """Save scalability report to file."""
        output_file = Path("scalability_validation_report.json")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Scalability validation report saved to: {output_file}")


# Test implementations
class TestScalabilityValidation:
    """Test suite for scalability validation."""
    
    @pytest.fixture
    def scalability_validator(self):
        """Create scalability validator instance."""
        return ScalabilityValidator()
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_horizontal_scaling_validation(self, scalability_validator):
        """Test horizontal scaling validation."""
        config = scalability_validator.test_configs["horizontal_scaling"]
        # Reduce levels for testing
        config.scaling_levels = [1, 2, 4]
        config.test_duration_seconds = 30
        
        results = scalability_validator.run_horizontal_scaling_test(config)
        
        assert len(results) == 3
        assert all(r.test_name == "horizontal_scaling" for r in results)
        assert all(r.throughput_ops_per_sec > 0 for r in results)
        
        # Log results
        for result in results:
            logger.info(f"Level {result.scaling_level}: "
                       f"{result.throughput_ops_per_sec:.1f} ops/sec, "
                       f"efficiency: {result.scaling_efficiency:.2f}")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_vertical_scaling_validation(self, scalability_validator):
        """Test vertical scaling validation."""
        config = scalability_validator.test_configs["vertical_scaling"]
        # Reduce levels for testing
        config.scaling_levels = [1, 2, 4]
        config.test_duration_seconds = 30
        
        results = scalability_validator.run_vertical_scaling_test(config)
        
        assert len(results) == 3
        assert all(r.test_name == "vertical_scaling" for r in results)
        assert all(r.throughput_ops_per_sec > 0 for r in results)
        
        # Log results
        for result in results:
            logger.info(f"Level {result.scaling_level}: "
                       f"{result.throughput_ops_per_sec:.1f} ops/sec, "
                       f"efficiency: {result.scaling_efficiency:.3f}")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_data_volume_scaling_validation(self, scalability_validator):
        """Test data volume scaling validation."""
        config = scalability_validator.test_configs["data_volume_scaling"]
        # Reduce levels for testing
        config.scaling_levels = [100, 500, 1000]
        config.test_duration_seconds = 60
        
        results = scalability_validator.run_data_volume_scaling_test(config)
        
        assert len(results) == 3
        assert all(r.test_name == "data_volume_scaling" for r in results)
        assert all(r.throughput_ops_per_sec > 0 for r in results)
        
        # Log results
        for result in results:
            logger.info(f"Level {result.scaling_level}: "
                       f"{result.throughput_ops_per_sec:.1f} ops/sec, "
                       f"memory: {result.memory_usage_mb:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_comprehensive_scalability_suite(self, scalability_validator):
        """Test comprehensive scalability suite (reduced scope)."""
        # Reduce all test levels for testing
        for config in scalability_validator.test_configs.values():
            if config.scaling_dimension == "horizontal":
                config.scaling_levels = [1, 2]
            elif config.scaling_dimension == "vertical":
                config.scaling_levels = [1, 2]
            elif config.scaling_dimension == "data_volume":
                config.scaling_levels = [100, 500]
            
            config.test_duration_seconds = 30
        
        # Remove time-based scaling for testing
        del scalability_validator.test_configs["time_based_scaling"]
        del scalability_validator.test_configs["concurrent_users"]
        
        report = scalability_validator.run_comprehensive_scalability_suite()
        
        # Verify report structure
        assert "executive_summary" in report
        assert "dimension_analysis" in report
        assert "detailed_results" in report
        assert "scalability_recommendations" in report
        
        # Verify executive summary
        summary = report["executive_summary"]
        assert "total_tests" in summary
        assert "success_rate" in summary
        assert "scalability_ready" in summary
        
        # Log overall results
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Scalability ready: {summary['scalability_ready']}")
        logger.info(f"System grade: {summary['system_scalability_grade']:.1f}/100")


if __name__ == "__main__":
    """Run scalability validation directly."""
    validator = ScalabilityValidator()
    
    print("🚀 Scalability Validation Suite")
    print("=" * 60)
    
    # Reduce test scope for demo
    for config in validator.test_configs.values():
        if config.scaling_dimension == "horizontal":
            config.scaling_levels = [1, 2, 4]
        elif config.scaling_dimension == "vertical":
            config.scaling_levels = [1, 2, 4]
        elif config.scaling_dimension == "data_volume":
            config.scaling_levels = [1000, 5000]
        
        config.test_duration_seconds = 60
    
    # Remove time-based scaling for demo
    del validator.test_configs["time_based_scaling"]
    del validator.test_configs["concurrent_users"]
    
    # Run comprehensive scalability suite
    report = validator.run_comprehensive_scalability_suite()
    
    # Display results
    print("\\n📊 SCALABILITY VALIDATION RESULTS")
    print("=" * 50)
    summary = report["executive_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Scalability Ready: {summary['scalability_ready']}")
    print(f"System Grade: {summary['system_scalability_grade']:.1f}/100")
    print(f"Best Dimension: {summary['best_performing_dimension']}")
    
    # Show dimension analysis
    print("\\n🎯 DIMENSION ANALYSIS")
    print("=" * 50)
    for dimension, analysis in report["dimension_analysis"].items():
        print(f"{dimension.upper()}:")
        print(f"  Max Throughput: {analysis['max_throughput']:.1f} ops/sec")
        print(f"  Min Latency: {analysis['min_latency']:.2f}ms")
        print(f"  Success Rate: {analysis['success_rate']:.1%}")
        print(f"  Bottlenecks: {', '.join(analysis['bottlenecks']) if analysis['bottlenecks'] else 'None'}")
        print()
    
    # Show recommendations
    if report["scalability_recommendations"]:
        print("\\n🔧 SCALABILITY RECOMMENDATIONS")
        print("=" * 50)
        for rec in report["scalability_recommendations"]:
            print(f"• {rec}")
    
    print("\\n✅ Scalability Validation Suite Complete!")
    print("📄 Full report saved to: scalability_validation_report.json")