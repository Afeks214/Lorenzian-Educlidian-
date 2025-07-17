"""
Comprehensive Performance Validation Framework

This module provides end-to-end performance validation with continuous benchmarking,
regression detection, load testing, and automated reporting for the GrandModel system.

Key Features:
- End-to-end performance validation
- Continuous benchmarking pipeline
- Performance regression detection
- Load testing scenarios
- SLA monitoring and alerting
- Automated performance reporting
- Strategic Inference Latency validation (<50ms)
- Database RTO validation (<30s)
- Trading Engine RTO validation (<5s)

Author: Performance Validation Agent
"""

import asyncio
import time
import json
import sqlite3
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog
from contextlib import contextmanager
import statistics
import gc
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Performance metrics collection
import pymongo
import redis
import aioredis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = structlog.get_logger()

@dataclass
class PerformanceTarget:
    """Performance target specification"""
    name: str
    max_latency_ms: float
    min_throughput_ops_per_sec: float
    max_memory_mb: float
    max_cpu_percent: float
    availability_percent: float = 99.9
    description: str = ""

@dataclass
class PerformanceMetric:
    """Performance metric measurement"""
    timestamp: datetime
    test_name: str
    metric_type: str  # latency, throughput, memory, cpu, availability
    value: float
    unit: str
    target_value: float
    passed: bool
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadTestResult:
    """Load test execution result"""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    resource_utilization: Dict[str, float]
    timestamp: datetime

@dataclass
class SLAViolation:
    """SLA violation record"""
    timestamp: datetime
    sla_name: str
    target_value: float
    actual_value: float
    violation_severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    recommendation: str

class PerformanceValidationFramework:
    """
    Comprehensive performance validation framework for continuous monitoring
    and validation of system performance across all components.
    """

    def __init__(self, db_path: str = "performance_validation.db"):
        self.db_path = db_path
        self.process = psutil.Process()
        self.metrics_history = deque(maxlen=10000)
        self.active_tests = {}
        self.sla_violations = deque(maxlen=1000)
        
        # Performance targets
        self.performance_targets = {
            "strategic_inference": PerformanceTarget(
                name="strategic_inference",
                max_latency_ms=50.0,
                min_throughput_ops_per_sec=100.0,
                max_memory_mb=500.0,
                max_cpu_percent=80.0,
                description="Strategic MARL inference latency target"
            ),
            "tactical_inference": PerformanceTarget(
                name="tactical_inference", 
                max_latency_ms=20.0,
                min_throughput_ops_per_sec=200.0,
                max_memory_mb=300.0,
                max_cpu_percent=70.0,
                description="Tactical MARL inference latency target"
            ),
            "database_rto": PerformanceTarget(
                name="database_rto",
                max_latency_ms=30000.0,  # 30 seconds
                min_throughput_ops_per_sec=50.0,
                max_memory_mb=1000.0,
                max_cpu_percent=90.0,
                description="Database Recovery Time Objective"
            ),
            "trading_engine_rto": PerformanceTarget(
                name="trading_engine_rto",
                max_latency_ms=5000.0,  # 5 seconds
                min_throughput_ops_per_sec=1000.0,
                max_memory_mb=800.0,
                max_cpu_percent=85.0,
                description="Trading Engine Recovery Time Objective"
            ),
            "end_to_end_pipeline": PerformanceTarget(
                name="end_to_end_pipeline",
                max_latency_ms=100.0,
                min_throughput_ops_per_sec=50.0,
                max_memory_mb=1500.0,
                max_cpu_percent=90.0,
                description="Complete end-to-end pipeline performance"
            )
        }
        
        # Initialize monitoring
        self._init_database()
        self._init_prometheus_metrics()
        
        # Background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance validation framework initialized", 
                   targets=len(self.performance_targets))

    def _init_database(self):
        """Initialize performance validation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                target_value REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                additional_data TEXT
            )
        """)
        
        # Load test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS load_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                total_requests INTEGER NOT NULL,
                successful_requests INTEGER NOT NULL,
                failed_requests INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                p95_latency_ms REAL NOT NULL,
                p99_latency_ms REAL NOT NULL,
                max_latency_ms REAL NOT NULL,
                throughput_ops_per_sec REAL NOT NULL,
                error_rate_percent REAL NOT NULL,
                resource_utilization TEXT
            )
        """)
        
        # SLA violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sla_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sla_name TEXT NOT NULL,
                target_value REAL NOT NULL,
                actual_value REAL NOT NULL,
                violation_severity TEXT NOT NULL,
                description TEXT NOT NULL,
                recommendation TEXT NOT NULL
            )
        """)
        
        # Performance baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                test_name TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                baseline_std REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            self.prometheus_metrics = {
                'latency_histogram': Histogram(
                    'performance_latency_seconds',
                    'Performance latency in seconds',
                    ['test_name', 'metric_type']
                ),
                'throughput_gauge': Gauge(
                    'performance_throughput_ops_per_sec',
                    'Performance throughput in operations per second',
                    ['test_name']
                ),
                'memory_gauge': Gauge(
                    'performance_memory_mb',
                    'Performance memory usage in MB',
                    ['test_name']
                ),
                'cpu_gauge': Gauge(
                    'performance_cpu_percent',
                    'Performance CPU usage percentage',
                    ['test_name']
                ),
                'sla_violations_counter': Counter(
                    'performance_sla_violations_total',
                    'Total SLA violations',
                    ['sla_name', 'severity']
                ),
                'test_results_counter': Counter(
                    'performance_test_results_total',
                    'Total performance test results',
                    ['test_name', 'result']
                )
            }
            
            # Start Prometheus metrics server
            start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
            
        except Exception as e:
            logger.warning("Failed to initialize Prometheus metrics", error=str(e))
            self.prometheus_metrics = {}

    def _background_monitoring(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor system resources
                self._monitor_system_resources()
                
                # Check for SLA violations
                self._check_sla_violations()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error("Error in background monitoring", error=str(e))
                time.sleep(10)

    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Record system metrics
            timestamp = datetime.now()
            
            system_metrics = [
                PerformanceMetric(
                    timestamp=timestamp,
                    test_name="system_monitoring",
                    metric_type="cpu",
                    value=cpu_percent,
                    unit="percent",
                    target_value=90.0,
                    passed=cpu_percent <= 90.0
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    test_name="system_monitoring",
                    metric_type="memory",
                    value=memory_mb,
                    unit="MB",
                    target_value=8000.0,
                    passed=memory_mb <= 8000.0
                )
            ]
            
            for metric in system_metrics:
                self.metrics_history.append(metric)
                
        except Exception as e:
            logger.error("Error monitoring system resources", error=str(e))

    @contextmanager
    def performance_test(self, test_name: str, target_name: str = None):
        """Context manager for performance testing"""
        if target_name is None:
            target_name = test_name
            
        target = self.performance_targets.get(target_name)
        if not target:
            raise ValueError(f"Unknown performance target: {target_name}")
        
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        self.active_tests[test_name] = {
            'start_time': start_time,
            'start_memory': start_memory,
            'target': target
        }
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            memory_mb = end_memory
            cpu_percent = end_cpu
            
            # Check against targets
            latency_passed = latency_ms <= target.max_latency_ms
            memory_passed = memory_mb <= target.max_memory_mb
            cpu_passed = cpu_percent <= target.max_cpu_percent
            
            # Record metrics
            timestamp = datetime.now()
            
            metrics = [
                PerformanceMetric(
                    timestamp=timestamp,
                    test_name=test_name,
                    metric_type="latency",
                    value=latency_ms,
                    unit="ms",
                    target_value=target.max_latency_ms,
                    passed=latency_passed
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    test_name=test_name,
                    metric_type="memory",
                    value=memory_mb,
                    unit="MB",
                    target_value=target.max_memory_mb,
                    passed=memory_passed
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    test_name=test_name,
                    metric_type="cpu",
                    value=cpu_percent,
                    unit="percent",
                    target_value=target.max_cpu_percent,
                    passed=cpu_passed
                )
            ]
            
            for metric in metrics:
                self.record_metric(metric)
            
            # Clean up
            if test_name in self.active_tests:
                del self.active_tests[test_name]

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        # Add to history
        self.metrics_history.append(metric)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics 
            (timestamp, test_name, metric_type, value, unit, target_value, passed, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.timestamp.isoformat(),
            metric.test_name,
            metric.metric_type,
            metric.value,
            metric.unit,
            metric.target_value,
            metric.passed,
            json.dumps(metric.additional_data)
        ))
        
        conn.commit()
        conn.close()
        
        # Update Prometheus metrics
        if self.prometheus_metrics:
            try:
                if metric.metric_type == "latency":
                    self.prometheus_metrics['latency_histogram'].labels(
                        test_name=metric.test_name,
                        metric_type=metric.metric_type
                    ).observe(metric.value / 1000)  # Convert to seconds
                elif metric.metric_type == "memory":
                    self.prometheus_metrics['memory_gauge'].labels(
                        test_name=metric.test_name
                    ).set(metric.value)
                elif metric.metric_type == "cpu":
                    self.prometheus_metrics['cpu_gauge'].labels(
                        test_name=metric.test_name
                    ).set(metric.value)
                
                # Record test result
                self.prometheus_metrics['test_results_counter'].labels(
                    test_name=metric.test_name,
                    result="pass" if metric.passed else "fail"
                ).inc()
                
            except Exception as e:
                logger.warning("Failed to update Prometheus metrics", error=str(e))
        
        # Check for SLA violations
        if not metric.passed:
            self._record_sla_violation(metric)

    def _record_sla_violation(self, metric: PerformanceMetric):
        """Record SLA violation"""
        # Determine severity
        violation_percent = ((metric.value - metric.target_value) / metric.target_value) * 100
        
        if violation_percent > 100:
            severity = "CRITICAL"
        elif violation_percent > 50:
            severity = "HIGH"
        elif violation_percent > 25:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        violation = SLAViolation(
            timestamp=metric.timestamp,
            sla_name=f"{metric.test_name}_{metric.metric_type}",
            target_value=metric.target_value,
            actual_value=metric.value,
            violation_severity=severity,
            description=f"{metric.test_name} {metric.metric_type} exceeded target by {violation_percent:.1f}%",
            recommendation=self._generate_violation_recommendation(metric, violation_percent)
        )
        
        self.sla_violations.append(violation)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sla_violations 
            (timestamp, sla_name, target_value, actual_value, violation_severity, description, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            violation.timestamp.isoformat(),
            violation.sla_name,
            violation.target_value,
            violation.actual_value,
            violation.violation_severity,
            violation.description,
            violation.recommendation
        ))
        
        conn.commit()
        conn.close()
        
        # Update Prometheus counter
        if self.prometheus_metrics:
            try:
                self.prometheus_metrics['sla_violations_counter'].labels(
                    sla_name=violation.sla_name,
                    severity=violation.violation_severity
                ).inc()
            except Exception as e:
                logger.warning("Failed to update SLA violation counter", error=str(e))
        
        logger.warning("SLA violation recorded",
                      sla_name=violation.sla_name,
                      severity=violation.violation_severity,
                      target_value=violation.target_value,
                      actual_value=violation.actual_value)

    def _generate_violation_recommendation(self, metric: PerformanceMetric, violation_percent: float) -> str:
        """Generate recommendation for SLA violation"""
        recommendations = []
        
        if metric.metric_type == "latency":
            if violation_percent > 100:
                recommendations.append("CRITICAL: Investigate algorithmic performance bottlenecks")
            elif violation_percent > 50:
                recommendations.append("HIGH: Review recent changes and optimize critical paths")
            else:
                recommendations.append("Monitor latency trends and consider optimization")
        
        elif metric.metric_type == "memory":
            if violation_percent > 50:
                recommendations.append("Investigate memory leaks and optimize allocation patterns")
            else:
                recommendations.append("Monitor memory usage and consider garbage collection tuning")
        
        elif metric.metric_type == "cpu":
            if violation_percent > 50:
                recommendations.append("Investigate CPU-intensive operations and optimize algorithms")
            else:
                recommendations.append("Monitor CPU usage patterns and consider load balancing")
        
        return "; ".join(recommendations) if recommendations else "Monitor performance trends"

    async def run_load_test(self, test_name: str, target_function: Callable,
                          duration_seconds: int = 60, concurrent_users: int = 10,
                          requests_per_second: int = 100) -> LoadTestResult:
        """Run load test scenario"""
        logger.info("Starting load test",
                   test_name=test_name,
                   duration_seconds=duration_seconds,
                   concurrent_users=concurrent_users,
                   requests_per_second=requests_per_second)
        
        # Test metrics
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Resource monitoring
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # Semaphore to control concurrent users
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def worker():
            async with semaphore:
                while time.time() < end_time:
                    try:
                        request_start = time.perf_counter()
                        
                        # Execute target function
                        if asyncio.iscoroutinefunction(target_function):
                            await target_function()
                        else:
                            target_function()
                        
                        request_end = time.perf_counter()
                        latency_ms = (request_end - request_start) * 1000
                        
                        latencies.append(latency_ms)
                        nonlocal successful_requests
                        successful_requests += 1
                        
                    except Exception as e:
                        logger.debug("Load test request failed", error=str(e))
                        nonlocal failed_requests
                        failed_requests += 1
                    
                    nonlocal total_requests
                    total_requests += 1
                    
                    # Control request rate
                    await asyncio.sleep(1.0 / requests_per_second)
        
        # Run workers
        tasks = [worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        if latencies:
            avg_latency_ms = statistics.mean(latencies)
            p95_latency_ms = np.percentile(latencies, 95)
            p99_latency_ms = np.percentile(latencies, 99)
            max_latency_ms = max(latencies)
        else:
            avg_latency_ms = p95_latency_ms = p99_latency_ms = max_latency_ms = 0
        
        throughput_ops_per_sec = total_requests / actual_duration
        error_rate_percent = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        resource_utilization = {
            'memory_delta_mb': end_memory - start_memory,
            'cpu_avg_percent': (start_cpu + end_cpu) / 2,
            'memory_peak_mb': end_memory
        }
        
        result = LoadTestResult(
            test_name=test_name,
            duration_seconds=actual_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            max_latency_ms=max_latency_ms,
            throughput_ops_per_sec=throughput_ops_per_sec,
            error_rate_percent=error_rate_percent,
            resource_utilization=resource_utilization,
            timestamp=datetime.now()
        )
        
        # Store result
        self._store_load_test_result(result)
        
        logger.info("Load test completed",
                   test_name=test_name,
                   total_requests=total_requests,
                   successful_requests=successful_requests,
                   throughput_ops_per_sec=throughput_ops_per_sec,
                   p99_latency_ms=p99_latency_ms,
                   error_rate_percent=error_rate_percent)
        
        return result

    def _store_load_test_result(self, result: LoadTestResult):
        """Store load test result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO load_test_results 
            (test_name, timestamp, duration_seconds, total_requests, successful_requests,
             failed_requests, avg_latency_ms, p95_latency_ms, p99_latency_ms, max_latency_ms,
             throughput_ops_per_sec, error_rate_percent, resource_utilization)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_name,
            result.timestamp.isoformat(),
            result.duration_seconds,
            result.total_requests,
            result.successful_requests,
            result.failed_requests,
            result.avg_latency_ms,
            result.p95_latency_ms,
            result.p99_latency_ms,
            result.max_latency_ms,
            result.throughput_ops_per_sec,
            result.error_rate_percent,
            json.dumps(result.resource_utilization)
        ))
        
        conn.commit()
        conn.close()

    def validate_strategic_inference_latency(self, test_function: Callable, 
                                           iterations: int = 1000) -> Dict[str, Any]:
        """Validate Strategic Inference Latency <50ms target"""
        logger.info("Validating Strategic Inference Latency", iterations=iterations)
        
        latencies = []
        
        # Warm-up
        for _ in range(10):
            test_function()
        
        # Main test
        for _ in range(iterations):
            with self.performance_test("strategic_inference_validation", "strategic_inference"):
                test_function()
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.metrics_history 
            if m.test_name == "strategic_inference_validation" and m.metric_type == "latency"
        ][-iterations:]
        
        if recent_metrics:
            latencies = [m.value for m in recent_metrics]
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            max_latency = max(latencies)
            avg_latency = statistics.mean(latencies)
            
            target_met = p99 <= 50.0
            
            return {
                'test_name': 'strategic_inference_latency',
                'target_ms': 50.0,
                'iterations': iterations,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50,
                'p95_latency_ms': p95,
                'p99_latency_ms': p99,
                'max_latency_ms': max_latency,
                'target_met': target_met,
                'success_rate': len([l for l in latencies if l <= 50.0]) / len(latencies),
                'timestamp': datetime.now().isoformat()
            }
        
        return {'error': 'No metrics collected'}

    def validate_database_rto(self, recovery_function: Callable) -> Dict[str, Any]:
        """Validate Database RTO <30s target"""
        logger.info("Validating Database RTO")
        
        with self.performance_test("database_rto_validation", "database_rto"):
            recovery_function()
        
        # Get the most recent metric
        recent_metrics = [
            m for m in self.metrics_history 
            if m.test_name == "database_rto_validation" and m.metric_type == "latency"
        ]
        
        if recent_metrics:
            latest_metric = recent_metrics[-1]
            rto_ms = latest_metric.value
            target_met = rto_ms <= 30000.0
            
            return {
                'test_name': 'database_rto',
                'target_ms': 30000.0,
                'actual_rto_ms': rto_ms,
                'target_met': target_met,
                'timestamp': datetime.now().isoformat()
            }
        
        return {'error': 'No RTO metrics collected'}

    def validate_trading_engine_rto(self, recovery_function: Callable) -> Dict[str, Any]:
        """Validate Trading Engine RTO <5s target"""
        logger.info("Validating Trading Engine RTO")
        
        with self.performance_test("trading_engine_rto_validation", "trading_engine_rto"):
            recovery_function()
        
        # Get the most recent metric
        recent_metrics = [
            m for m in self.metrics_history 
            if m.test_name == "trading_engine_rto_validation" and m.metric_type == "latency"
        ]
        
        if recent_metrics:
            latest_metric = recent_metrics[-1]
            rto_ms = latest_metric.value
            target_met = rto_ms <= 5000.0
            
            return {
                'test_name': 'trading_engine_rto',
                'target_ms': 5000.0,
                'actual_rto_ms': rto_ms,
                'target_met': target_met,
                'timestamp': datetime.now().isoformat()
            }
        
        return {'error': 'No RTO metrics collected'}

    def _check_sla_violations(self):
        """Check for recent SLA violations"""
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        recent_violations = [
            v for v in self.sla_violations
            if v.timestamp >= cutoff_time
        ]
        
        if recent_violations:
            critical_violations = [v for v in recent_violations if v.violation_severity == "CRITICAL"]
            high_violations = [v for v in recent_violations if v.violation_severity == "HIGH"]
            
            if critical_violations:
                logger.critical("Critical SLA violations detected",
                              count=len(critical_violations),
                              violations=[v.sla_name for v in critical_violations])
            
            if high_violations:
                logger.warning("High priority SLA violations detected",
                             count=len(high_violations),
                             violations=[v.sla_name for v in high_violations])

    def _update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        if not self.prometheus_metrics:
            return
        
        try:
            # Update throughput metrics from active tests
            for test_name, test_info in self.active_tests.items():
                if 'throughput' in test_info:
                    self.prometheus_metrics['throughput_gauge'].labels(
                        test_name=test_name
                    ).set(test_info['throughput'])
            
        except Exception as e:
            logger.debug("Error updating Prometheus metrics", error=str(e))

    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # Group by test name and metric type
        metrics_by_test = defaultdict(lambda: defaultdict(list))
        for metric in recent_metrics:
            metrics_by_test[metric.test_name][metric.metric_type].append(metric)
        
        # Calculate summary statistics
        test_summaries = {}
        for test_name, metric_types in metrics_by_test.items():
            summary = {}
            for metric_type, metrics in metric_types.items():
                values = [m.value for m in metrics]
                if values:
                    summary[metric_type] = {
                        'count': len(values),
                        'avg': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99),
                        'target_met_rate': len([m for m in metrics if m.passed]) / len(metrics)
                    }
            test_summaries[test_name] = summary
        
        # Recent SLA violations
        recent_violations = [
            v for v in self.sla_violations
            if v.timestamp >= cutoff_time
        ]
        
        # System health score
        overall_health = self._calculate_system_health_score(recent_metrics)
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'system_health_score': overall_health,
            'summary': {
                'total_tests': len(test_summaries),
                'total_metrics': len(recent_metrics),
                'total_violations': len(recent_violations),
                'critical_violations': len([v for v in recent_violations if v.violation_severity == "CRITICAL"])
            },
            'test_summaries': test_summaries,
            'recent_violations': [
                {
                    'timestamp': v.timestamp.isoformat(),
                    'sla_name': v.sla_name,
                    'severity': v.violation_severity,
                    'target_value': v.target_value,
                    'actual_value': v.actual_value,
                    'description': v.description,
                    'recommendation': v.recommendation
                }
                for v in recent_violations[-20:]  # Latest 20 violations
            ],
            'performance_targets': {
                name: {
                    'max_latency_ms': target.max_latency_ms,
                    'min_throughput_ops_per_sec': target.min_throughput_ops_per_sec,
                    'max_memory_mb': target.max_memory_mb,
                    'max_cpu_percent': target.max_cpu_percent,
                    'description': target.description
                }
                for name, target in self.performance_targets.items()
            }
        }

    def _calculate_system_health_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall system health score (0-100)"""
        if not metrics:
            return 100.0
        
        total_metrics = len(metrics)
        passed_metrics = len([m for m in metrics if m.passed])
        
        base_score = (passed_metrics / total_metrics) * 100
        
        # Penalty for recent violations
        recent_violations = len([v for v in self.sla_violations if v.timestamp >= datetime.now() - timedelta(hours=1)])
        violation_penalty = min(recent_violations * 5, 30)  # Max 30 point penalty
        
        return max(0, base_score - violation_penalty)

    def cleanup(self):
        """Cleanup resources"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance validation framework cleanup completed")

    def __del__(self):
        """Destructor"""
        self.cleanup()


# Global instance
performance_validator = PerformanceValidationFramework()