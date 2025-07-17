"""
Automated Performance Regression Testing Framework
===============================================

This module provides comprehensive automated performance regression testing
with baseline management, trend analysis, and automated alerting.

Key Features:
- Automated performance baseline management
- Regression detection and alerting
- Performance trend analysis
- Load testing and stress testing
- Memory and CPU profiling
- Latency and throughput monitoring
- Automated performance reporting

Performance Targets:
- Latency: <50ms p95 for critical operations
- Throughput: >1000 RPS for trading operations
- Memory: <2GB baseline usage
- CPU: <70% average utilization
"""

import asyncio
import time
import logging
import json
import traceback
import threading
import statistics
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
import asyncpg
import aioredis
import httpx
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import cProfile
import pstats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"


class PerformanceTestType(Enum):
    """Types of performance tests."""
    BASELINE = "baseline"
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    ENDURANCE_TEST = "endurance_test"
    SPIKE_TEST = "spike_test"
    REGRESSION_TEST = "regression_test"
    PROFILING = "profiling"


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: PerformanceMetricType
    value: float
    unit: str
    timestamp: datetime
    test_id: str
    component: str
    
    # Statistical metrics
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    std_dev: Optional[float] = None
    
    # Context
    load_level: Optional[str] = None
    test_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    component: str
    metric_type: PerformanceMetricType
    baseline_value: float
    baseline_timestamp: datetime
    
    # Thresholds
    warning_threshold: float = 0.1  # 10% degradation
    critical_threshold: float = 0.25  # 25% degradation
    
    # Statistical data
    historical_values: List[float] = field(default_factory=list)
    trend_direction: str = "stable"  # stable, improving, degrading
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def is_regression(self, current_value: float) -> Tuple[bool, RegressionSeverity]:
        """Check if current value represents a regression."""
        if self.baseline_value == 0:
            return False, RegressionSeverity.INFO
        
        degradation = (current_value - self.baseline_value) / self.baseline_value
        
        if degradation > self.critical_threshold:
            return True, RegressionSeverity.CRITICAL
        elif degradation > self.warning_threshold:
            return True, RegressionSeverity.HIGH
        elif degradation > 0.05:  # 5% degradation
            return True, RegressionSeverity.MEDIUM
        else:
            return False, RegressionSeverity.INFO


@dataclass
class PerformanceTestConfig:
    """Configuration for performance tests."""
    test_id: str
    test_type: PerformanceTestType
    
    # Test parameters
    duration_seconds: int = 300
    concurrent_users: int = 10
    requests_per_second: int = 100
    ramp_up_duration: int = 30
    
    # Target endpoints
    endpoints: List[str] = field(default_factory=list)
    
    # Database configuration
    database_url: str = "postgresql://admin:admin@localhost:5432/trading_db"
    redis_url: str = "redis://localhost:6379"
    
    # Thresholds
    latency_threshold_ms: float = 100.0
    throughput_threshold_rps: float = 1000.0
    memory_threshold_mb: float = 2000.0
    cpu_threshold_percent: float = 70.0
    error_rate_threshold_percent: float = 1.0
    
    # Baseline configuration
    baseline_database_path: str = "performance_baselines.db"
    enable_baseline_comparison: bool = True
    update_baseline_on_success: bool = True
    
    # Profiling configuration
    enable_profiling: bool = True
    profiling_interval: int = 10
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    
    # Reporting configuration
    enable_reporting: bool = True
    report_output_dir: str = "performance_reports"
    enable_charts: bool = True


@dataclass
class PerformanceTestResult:
    """Result of performance test execution."""
    test_id: str
    test_type: PerformanceTestType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    # Test status
    status: str = "running"
    success: bool = False
    
    # Metrics
    metrics: List[PerformanceMetric] = field(default_factory=list)
    
    # Aggregated results
    latency_stats: Dict[str, float] = field(default_factory=dict)
    throughput_stats: Dict[str, float] = field(default_factory=dict)
    resource_stats: Dict[str, float] = field(default_factory=dict)
    
    # Regression analysis
    regressions_detected: List[Dict[str, Any]] = field(default_factory=list)
    baseline_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    
    # Profiling results
    profiling_results: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and recommendations
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric."""
        self.metrics.append(metric)
    
    def add_regression(self, component: str, metric_type: PerformanceMetricType, 
                      baseline_value: float, current_value: float, severity: RegressionSeverity):
        """Add a regression detection."""
        self.regressions_detected.append({
            "component": component,
            "metric_type": metric_type.value,
            "baseline_value": baseline_value,
            "current_value": current_value,
            "degradation_percent": ((current_value - baseline_value) / baseline_value) * 100,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat()
        })


class PerformanceMetricsCollector:
    """Collector for performance metrics."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.metrics_buffer: List[PerformanceMetric] = []
        self.metrics_lock = threading.Lock()
        
    async def start_collection(self, test_id: str, component: str):
        """Start metrics collection."""
        self.collection_active = True
        self.collection_task = asyncio.create_task(
            self._collection_loop(test_id, component)
        )
        logger.info(f"Started metrics collection for {component}")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self, test_id: str, component: str):
        """Main metrics collection loop."""
        while self.collection_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics(test_id, component)
                
                # Collect application metrics
                await self._collect_application_metrics(test_id, component)
                
                # Collect database metrics
                await self._collect_database_metrics(test_id, component)
                
                # Collect Redis metrics
                await self._collect_redis_metrics(test_id, component)
                
                await asyncio.sleep(self.config.profiling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self, test_id: str, component: str):
        """Collect system-level metrics."""
        try:
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(PerformanceMetric(
                metric_type=PerformanceMetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                test_id=test_id,
                component=component
            ))
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            self._add_metric(PerformanceMetric(
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                value=memory_info.used / (1024 * 1024),  # Convert to MB
                unit="MB",
                timestamp=timestamp,
                test_id=test_id,
                component=component
            ))
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._add_metric(PerformanceMetric(
                    metric_type=PerformanceMetricType.DISK_IO,
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    unit="bytes",
                    timestamp=timestamp,
                    test_id=test_id,
                    component=component
                ))
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io:
                self._add_metric(PerformanceMetric(
                    metric_type=PerformanceMetricType.NETWORK_IO,
                    value=network_io.bytes_sent + network_io.bytes_recv,
                    unit="bytes",
                    timestamp=timestamp,
                    test_id=test_id,
                    component=component
                ))
                
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_application_metrics(self, test_id: str, component: str):
        """Collect application-specific metrics."""
        try:
            timestamp = datetime.now()
            
            # Test application endpoints
            for endpoint in self.config.endpoints:
                try:
                    start_time = time.time()
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{endpoint}/metrics", timeout=5.0)
                        latency = time.time() - start_time
                        
                        # Record latency
                        self._add_metric(PerformanceMetric(
                            metric_type=PerformanceMetricType.LATENCY,
                            value=latency * 1000,  # Convert to ms
                            unit="ms",
                            timestamp=timestamp,
                            test_id=test_id,
                            component=f"{component}_{endpoint.split('/')[-1]}"
                        ))
                        
                        # Record availability
                        self._add_metric(PerformanceMetric(
                            metric_type=PerformanceMetricType.AVAILABILITY,
                            value=1.0 if response.status_code == 200 else 0.0,
                            unit="percent",
                            timestamp=timestamp,
                            test_id=test_id,
                            component=f"{component}_{endpoint.split('/')[-1]}"
                        ))
                        
                except Exception as e:
                    # Record error
                    self._add_metric(PerformanceMetric(
                        metric_type=PerformanceMetricType.ERROR_RATE,
                        value=1.0,
                        unit="errors",
                        timestamp=timestamp,
                        test_id=test_id,
                        component=f"{component}_{endpoint.split('/')[-1]}"
                    ))
                    
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")
    
    async def _collect_database_metrics(self, test_id: str, component: str):
        """Collect database performance metrics."""
        try:
            timestamp = datetime.now()
            
            # Test database performance
            start_time = time.time()
            conn = await asyncpg.connect(self.config.database_url)
            
            # Simple query latency
            await conn.fetchval("SELECT 1")
            query_latency = time.time() - start_time
            
            # Get database stats
            stats = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT COUNT(*) FROM pg_stat_activity) as total_connections
            """)
            
            await conn.close()
            
            # Record metrics
            self._add_metric(PerformanceMetric(
                metric_type=PerformanceMetricType.LATENCY,
                value=query_latency * 1000,  # Convert to ms
                unit="ms",
                timestamp=timestamp,
                test_id=test_id,
                component=f"{component}_database"
            ))
            
        except Exception as e:
            logger.error(f"Database metrics collection failed: {e}")
    
    async def _collect_redis_metrics(self, test_id: str, component: str):
        """Collect Redis performance metrics."""
        try:
            timestamp = datetime.now()
            
            # Test Redis performance
            start_time = time.time()
            redis_client = aioredis.from_url(self.config.redis_url)
            
            # Simple ping latency
            await redis_client.ping()
            ping_latency = time.time() - start_time
            
            # Get Redis info
            info = await redis_client.info()
            
            await redis_client.close()
            
            # Record metrics
            self._add_metric(PerformanceMetric(
                metric_type=PerformanceMetricType.LATENCY,
                value=ping_latency * 1000,  # Convert to ms
                unit="ms",
                timestamp=timestamp,
                test_id=test_id,
                component=f"{component}_redis"
            ))
            
            # Record memory usage
            used_memory = info.get('used_memory', 0)
            self._add_metric(PerformanceMetric(
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                value=used_memory / (1024 * 1024),  # Convert to MB
                unit="MB",
                timestamp=timestamp,
                test_id=test_id,
                component=f"{component}_redis"
            ))
            
        except Exception as e:
            logger.error(f"Redis metrics collection failed: {e}")
    
    def _add_metric(self, metric: PerformanceMetric):
        """Add metric to buffer."""
        with self.metrics_lock:
            self.metrics_buffer.append(metric)
    
    def get_collected_metrics(self) -> List[PerformanceMetric]:
        """Get all collected metrics."""
        with self.metrics_lock:
            return self.metrics_buffer.copy()
    
    def clear_metrics(self):
        """Clear metrics buffer."""
        with self.metrics_lock:
            self.metrics_buffer.clear()


class PerformanceBaselineManager:
    """Manager for performance baselines."""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize baseline database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    baseline_timestamp TEXT NOT NULL,
                    warning_threshold REAL NOT NULL,
                    critical_threshold REAL NOT NULL,
                    historical_values TEXT,
                    trend_direction TEXT,
                    confidence_interval TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_baseline_component_metric 
                ON performance_baselines(component, metric_type)
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize baseline database: {e}")
    
    def load_baselines(self):
        """Load baselines from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM performance_baselines")
            rows = cursor.fetchall()
            
            for row in rows:
                component = row[1]
                metric_type = PerformanceMetricType(row[2])
                baseline_value = row[3]
                baseline_timestamp = datetime.fromisoformat(row[4])
                warning_threshold = row[5]
                critical_threshold = row[6]
                historical_values = json.loads(row[7]) if row[7] else []
                trend_direction = row[8] or "stable"
                confidence_interval = tuple(json.loads(row[9])) if row[9] else (0.0, 0.0)
                
                baseline = PerformanceBaseline(
                    component=component,
                    metric_type=metric_type,
                    baseline_value=baseline_value,
                    baseline_timestamp=baseline_timestamp,
                    warning_threshold=warning_threshold,
                    critical_threshold=critical_threshold,
                    historical_values=historical_values,
                    trend_direction=trend_direction,
                    confidence_interval=confidence_interval
                )
                
                self.baselines[f"{component}_{metric_type.value}"] = baseline
            
            conn.close()
            logger.info(f"Loaded {len(self.baselines)} baselines from database")
            
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")
    
    def save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO performance_baselines 
                (component, metric_type, baseline_value, baseline_timestamp, 
                 warning_threshold, critical_threshold, historical_values, 
                 trend_direction, confidence_interval, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.component,
                baseline.metric_type.value,
                baseline.baseline_value,
                baseline.baseline_timestamp.isoformat(),
                baseline.warning_threshold,
                baseline.critical_threshold,
                json.dumps(baseline.historical_values),
                baseline.trend_direction,
                json.dumps(baseline.confidence_interval),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update in-memory cache
            self.baselines[f"{baseline.component}_{baseline.metric_type.value}"] = baseline
            
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def get_baseline(self, component: str, metric_type: PerformanceMetricType) -> Optional[PerformanceBaseline]:
        """Get baseline for component and metric type."""
        return self.baselines.get(f"{component}_{metric_type.value}")
    
    def update_baseline(self, component: str, metric_type: PerformanceMetricType, 
                       new_value: float, historical_values: List[float]):
        """Update baseline with new data."""
        baseline = self.get_baseline(component, metric_type)
        
        if baseline:
            # Update existing baseline
            baseline.baseline_value = new_value
            baseline.baseline_timestamp = datetime.now()
            baseline.historical_values = historical_values[-100:]  # Keep last 100 values
            
            # Update trend direction
            if len(historical_values) >= 10:
                recent_values = historical_values[-10:]
                if len(set(recent_values)) > 1:
                    trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    if trend_slope > 0.01:
                        baseline.trend_direction = "degrading"
                    elif trend_slope < -0.01:
                        baseline.trend_direction = "improving"
                    else:
                        baseline.trend_direction = "stable"
            
            # Update confidence interval
            if len(historical_values) >= 5:
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                baseline.confidence_interval = (mean_val - 2*std_val, mean_val + 2*std_val)
            
        else:
            # Create new baseline
            baseline = PerformanceBaseline(
                component=component,
                metric_type=metric_type,
                baseline_value=new_value,
                baseline_timestamp=datetime.now(),
                historical_values=historical_values[-100:]
            )
        
        self.save_baseline(baseline)
    
    def check_for_regressions(self, metrics: List[PerformanceMetric]) -> List[Tuple[PerformanceMetric, PerformanceBaseline, RegressionSeverity]]:
        """Check metrics against baselines for regressions."""
        regressions = []
        
        for metric in metrics:
            baseline = self.get_baseline(metric.component, metric.metric_type)
            if baseline:
                is_regression, severity = baseline.is_regression(metric.value)
                if is_regression:
                    regressions.append((metric, baseline, severity))
        
        return regressions


class LoadTestExecutor:
    """Executor for load testing."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        
    async def execute_load_test(self, test_id: str) -> Dict[str, Any]:
        """Execute load test."""
        try:
            results = {
                "test_id": test_id,
                "test_type": "load_test",
                "start_time": time.time(),
                "concurrent_users": self.config.concurrent_users,
                "duration": self.config.duration_seconds,
                "requests_completed": 0,
                "requests_failed": 0,
                "response_times": [],
                "error_details": []
            }
            
            # Create tasks for concurrent users
            tasks = []
            for user_id in range(self.config.concurrent_users):
                task = asyncio.create_task(self._simulate_user_load(user_id, test_id, results))
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            results["end_time"] = time.time()
            results["total_duration"] = results["end_time"] - results["start_time"]
            
            # Calculate statistics
            if results["response_times"]:
                results["avg_response_time"] = statistics.mean(results["response_times"])
                results["min_response_time"] = min(results["response_times"])
                results["max_response_time"] = max(results["response_times"])
                results["p95_response_time"] = np.percentile(results["response_times"], 95)
                results["p99_response_time"] = np.percentile(results["response_times"], 99)
            
            # Calculate throughput
            total_requests = results["requests_completed"] + results["requests_failed"]
            if results["total_duration"] > 0:
                results["throughput_rps"] = total_requests / results["total_duration"]
            
            # Calculate error rate
            if total_requests > 0:
                results["error_rate"] = (results["requests_failed"] / total_requests) * 100
            
            return results
            
        except Exception as e:
            logger.error(f"Load test execution failed: {e}")
            return {"error": str(e)}
    
    async def _simulate_user_load(self, user_id: int, test_id: str, results: Dict[str, Any]):
        """Simulate load for a single user."""
        start_time = time.time()
        
        while time.time() - start_time < self.config.duration_seconds:
            try:
                # Choose random endpoint
                endpoint = random.choice(self.config.endpoints) if self.config.endpoints else "http://localhost:8001"
                
                # Send request
                request_start = time.time()
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{endpoint}/ping", timeout=10.0)
                    response_time = time.time() - request_start
                    
                    # Record results
                    results["response_times"].append(response_time)
                    
                    if response.status_code == 200:
                        results["requests_completed"] += 1
                    else:
                        results["requests_failed"] += 1
                        results["error_details"].append({
                            "user_id": user_id,
                            "status_code": response.status_code,
                            "timestamp": time.time()
                        })
                
                # Wait between requests
                await asyncio.sleep(1.0 / self.config.requests_per_second)
                
            except Exception as e:
                results["requests_failed"] += 1
                results["error_details"].append({
                    "user_id": user_id,
                    "error": str(e),
                    "timestamp": time.time()
                })
                
                await asyncio.sleep(1.0)


class PerformanceProfiler:
    """Profiler for performance analysis."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.profiler = None
        
    def start_profiling(self, test_id: str):
        """Start performance profiling."""
        if self.config.enable_cpu_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            logger.info(f"Started CPU profiling for test {test_id}")
    
    def stop_profiling(self, test_id: str) -> Dict[str, Any]:
        """Stop profiling and return results."""
        results = {}
        
        if self.profiler:
            self.profiler.disable()
            
            # Save profile results
            profile_path = f"{self.config.report_output_dir}/profile_{test_id}.prof"
            Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
            self.profiler.dump_stats(profile_path)
            
            # Generate profile statistics
            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            
            # Get top functions
            top_functions = []
            for func_info in stats.get_stats().items():
                func_name = func_info[0]
                func_stats = func_info[1]
                
                top_functions.append({
                    "function": f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                    "calls": func_stats[0],
                    "total_time": func_stats[2],
                    "cumulative_time": func_stats[3]
                })
            
            # Sort by cumulative time and take top 20
            top_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
            
            results["cpu_profiling"] = {
                "profile_path": profile_path,
                "top_functions": top_functions[:20],
                "total_calls": sum(f["calls"] for f in top_functions),
                "total_time": max(f["total_time"] for f in top_functions) if top_functions else 0
            }
            
            logger.info(f"Stopped CPU profiling for test {test_id}")
        
        return results
    
    @profile
    def memory_intensive_function(self):
        """Example function for memory profiling."""
        # This is a placeholder for memory-intensive operations
        data = []
        for i in range(1000):
            data.append([random.random() for _ in range(1000)])
        return data


class PerformanceReporter:
    """Reporter for performance test results."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.output_dir = Path(config.report_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, test_result: PerformanceTestResult) -> str:
        """Generate comprehensive performance report."""
        try:
            report_path = self.output_dir / f"performance_report_{test_result.test_id}.html"
            
            # Generate HTML report
            html_content = self._generate_html_report(test_result)
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            # Generate charts if enabled
            if self.config.enable_charts:
                self._generate_charts(test_result)
            
            # Generate JSON summary
            self._generate_json_summary(test_result)
            
            logger.info(f"Generated performance report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return ""
    
    def _generate_html_report(self, test_result: PerformanceTestResult) -> str:
        """Generate HTML report."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report - {test_result.test_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .metric-box {{ border: 1px solid #ddd; padding: 15px; margin: 10px; display: inline-block; }}
                .regression {{ background-color: #ffebee; border-color: #f44336; }}
                .success {{ background-color: #e8f5e8; border-color: #4caf50; }}
                .warning {{ background-color: #fff3e0; border-color: #ff9800; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <p><strong>Test ID:</strong> {test_result.test_id}</p>
                <p><strong>Test Type:</strong> {test_result.test_type.value}</p>
                <p><strong>Duration:</strong> {test_result.duration:.2f} seconds</p>
                <p><strong>Status:</strong> {test_result.status}</p>
                <p><strong>Success:</strong> {test_result.success}</p>
            </div>
            
            <div class="metrics-summary">
                <h2>Metrics Summary</h2>
                {self._generate_metrics_summary_html(test_result)}
            </div>
            
            <div class="regressions">
                <h2>Regression Analysis</h2>
                {self._generate_regressions_html(test_result)}
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(test_result)}
            </div>
            
            <div class="detailed-metrics">
                <h2>Detailed Metrics</h2>
                {self._generate_detailed_metrics_html(test_result)}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_metrics_summary_html(self, test_result: PerformanceTestResult) -> str:
        """Generate metrics summary HTML."""
        html = ""
        
        # Latency metrics
        if test_result.latency_stats:
            css_class = "success" if test_result.latency_stats.get("avg", 0) < 100 else "warning"
            html += f"""
            <div class="metric-box {css_class}">
                <h3>Latency</h3>
                <p>Average: {test_result.latency_stats.get('avg', 0):.2f}ms</p>
                <p>P95: {test_result.latency_stats.get('p95', 0):.2f}ms</p>
                <p>P99: {test_result.latency_stats.get('p99', 0):.2f}ms</p>
            </div>
            """
        
        # Throughput metrics
        if test_result.throughput_stats:
            css_class = "success" if test_result.throughput_stats.get('rps', 0) > 1000 else "warning"
            html += f"""
            <div class="metric-box {css_class}">
                <h3>Throughput</h3>
                <p>RPS: {test_result.throughput_stats.get('rps', 0):.2f}</p>
                <p>Total Requests: {test_result.throughput_stats.get('total', 0)}</p>
                <p>Error Rate: {test_result.throughput_stats.get('error_rate', 0):.2f}%</p>
            </div>
            """
        
        # Resource metrics
        if test_result.resource_stats:
            html += f"""
            <div class="metric-box">
                <h3>Resource Usage</h3>
                <p>Memory: {test_result.resource_stats.get('memory_mb', 0):.2f}MB</p>
                <p>CPU: {test_result.resource_stats.get('cpu_percent', 0):.2f}%</p>
            </div>
            """
        
        return html
    
    def _generate_regressions_html(self, test_result: PerformanceTestResult) -> str:
        """Generate regressions HTML."""
        if not test_result.regressions_detected:
            return "<p>No regressions detected.</p>"
        
        html = "<table><tr><th>Component</th><th>Metric</th><th>Baseline</th><th>Current</th><th>Degradation</th><th>Severity</th></tr>"
        
        for regression in test_result.regressions_detected:
            html += f"""
            <tr>
                <td>{regression['component']}</td>
                <td>{regression['metric_type']}</td>
                <td>{regression['baseline_value']:.2f}</td>
                <td>{regression['current_value']:.2f}</td>
                <td>{regression['degradation_percent']:.1f}%</td>
                <td>{regression['severity']}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, test_result: PerformanceTestResult) -> str:
        """Generate recommendations HTML."""
        if not test_result.recommendations:
            return "<p>No specific recommendations.</p>"
        
        html = "<ul>"
        for recommendation in test_result.recommendations:
            html += f"<li>{recommendation}</li>"
        html += "</ul>"
        
        return html
    
    def _generate_detailed_metrics_html(self, test_result: PerformanceTestResult) -> str:
        """Generate detailed metrics HTML."""
        html = "<table><tr><th>Timestamp</th><th>Component</th><th>Metric Type</th><th>Value</th><th>Unit</th></tr>"
        
        for metric in test_result.metrics[-50:]:  # Show last 50 metrics
            html += f"""
            <tr>
                <td>{metric.timestamp.strftime('%H:%M:%S')}</td>
                <td>{metric.component}</td>
                <td>{metric.metric_type.value}</td>
                <td>{metric.value:.2f}</td>
                <td>{metric.unit}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_charts(self, test_result: PerformanceTestResult):
        """Generate performance charts."""
        try:
            # Group metrics by type
            metrics_by_type = {}
            for metric in test_result.metrics:
                if metric.metric_type not in metrics_by_type:
                    metrics_by_type[metric.metric_type] = []
                metrics_by_type[metric.metric_type].append(metric)
            
            # Create charts for each metric type
            for metric_type, metrics in metrics_by_type.items():
                if len(metrics) > 1:
                    self._create_time_series_chart(metric_type, metrics, test_result.test_id)
                    
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
    
    def _create_time_series_chart(self, metric_type: PerformanceMetricType, 
                                 metrics: List[PerformanceMetric], test_id: str):
        """Create time series chart for metrics."""
        try:
            # Prepare data
            df = pd.DataFrame([
                {
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'component': metric.component
                }
                for metric in metrics
            ])
            
            # Create chart
            plt.figure(figsize=(12, 6))
            
            # Plot each component separately
            for component in df['component'].unique():
                component_data = df[df['component'] == component]
                plt.plot(component_data['timestamp'], component_data['value'], 
                        label=component, marker='o', markersize=2)
            
            plt.title(f'{metric_type.value.title()} Over Time - Test {test_id}')
            plt.xlabel('Time')
            plt.ylabel(f'{metric_type.value.title()} ({metrics[0].unit})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"chart_{test_id}_{metric_type.value}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create chart for {metric_type.value}: {e}")
    
    def _generate_json_summary(self, test_result: PerformanceTestResult):
        """Generate JSON summary."""
        try:
            summary = {
                "test_id": test_result.test_id,
                "test_type": test_result.test_type.value,
                "start_time": test_result.start_time.isoformat(),
                "end_time": test_result.end_time.isoformat() if test_result.end_time else None,
                "duration": test_result.duration,
                "status": test_result.status,
                "success": test_result.success,
                "latency_stats": test_result.latency_stats,
                "throughput_stats": test_result.throughput_stats,
                "resource_stats": test_result.resource_stats,
                "regressions_count": len(test_result.regressions_detected),
                "recommendations_count": len(test_result.recommendations),
                "metrics_count": len(test_result.metrics)
            }
            
            json_path = self.output_dir / f"summary_{test_result.test_id}.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to generate JSON summary: {e}")


class PerformanceRegressionTester:
    """Main class for performance regression testing."""
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.metrics_collector = PerformanceMetricsCollector(config)
        self.baseline_manager = PerformanceBaselineManager(config.baseline_database_path)
        self.load_executor = LoadTestExecutor(config)
        self.profiler = PerformanceProfiler(config)
        self.reporter = PerformanceReporter(config)
        
        # Load existing baselines
        self.baseline_manager.load_baselines()
    
    async def run_performance_test(self, test_type: PerformanceTestType) -> PerformanceTestResult:
        """Run comprehensive performance test."""
        test_id = f"{test_type.value}_{int(time.time())}"
        
        result = PerformanceTestResult(
            test_id=test_id,
            test_type=test_type,
            start_time=datetime.now(),
            status="running"
        )
        
        logger.info(f"Starting performance test: {test_id}")
        
        try:
            # Start metrics collection
            await self.metrics_collector.start_collection(test_id, "system")
            
            # Start profiling
            if self.config.enable_profiling:
                self.profiler.start_profiling(test_id)
            
            # Execute test based on type
            if test_type == PerformanceTestType.BASELINE:
                await self._run_baseline_test(result)
            elif test_type == PerformanceTestType.LOAD_TEST:
                await self._run_load_test(result)
            elif test_type == PerformanceTestType.STRESS_TEST:
                await self._run_stress_test(result)
            elif test_type == PerformanceTestType.REGRESSION_TEST:
                await self._run_regression_test(result)
            else:
                await self._run_generic_test(result)
            
            # Stop profiling
            if self.config.enable_profiling:
                profiling_results = self.profiler.stop_profiling(test_id)
                result.profiling_results = profiling_results
            
            # Stop metrics collection
            await self.metrics_collector.stop_collection()
            
            # Collect all metrics
            collected_metrics = self.metrics_collector.get_collected_metrics()
            for metric in collected_metrics:
                result.add_metric(metric)
            
            # Analyze results
            await self._analyze_results(result)
            
            # Check for regressions
            await self._check_regressions(result)
            
            # Generate recommendations
            await self._generate_recommendations(result)
            
            # Update baselines if successful
            if result.success and self.config.update_baseline_on_success:
                await self._update_baselines(result)
            
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.status = "completed"
            
            # Generate report
            if self.config.enable_reporting:
                self.reporter.generate_report(result)
            
            logger.info(f"Performance test completed: {test_id}")
            logger.info(f"Test success: {result.success}")
            
            return result
            
        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.issues.append({
                "severity": "critical",
                "message": f"Test execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.error(f"Performance test failed: {test_id} - {str(e)}")
            logger.error(traceback.format_exc())
            
            return result
        
        finally:
            # Cleanup
            try:
                await self.metrics_collector.stop_collection()
                self.metrics_collector.clear_metrics()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    
    async def _run_baseline_test(self, result: PerformanceTestResult):
        """Run baseline establishment test."""
        logger.info("Running baseline test")
        
        # Run for shorter duration to establish baseline
        await asyncio.sleep(60)  # 1 minute baseline collection
        
        result.status = "baseline_established"
    
    async def _run_load_test(self, result: PerformanceTestResult):
        """Run load test."""
        logger.info("Running load test")
        
        # Execute load test
        load_results = await self.load_executor.execute_load_test(result.test_id)
        
        # Store load test results
        result.throughput_stats = {
            "rps": load_results.get("throughput_rps", 0),
            "total": load_results.get("requests_completed", 0) + load_results.get("requests_failed", 0),
            "error_rate": load_results.get("error_rate", 0)
        }
        
        result.latency_stats = {
            "avg": load_results.get("avg_response_time", 0) * 1000,  # Convert to ms
            "p95": load_results.get("p95_response_time", 0) * 1000,
            "p99": load_results.get("p99_response_time", 0) * 1000
        }
    
    async def _run_stress_test(self, result: PerformanceTestResult):
        """Run stress test."""
        logger.info("Running stress test")
        
        # Increase load parameters for stress test
        original_concurrent_users = self.config.concurrent_users
        original_rps = self.config.requests_per_second
        
        # Increase by 3x for stress test
        self.config.concurrent_users *= 3
        self.config.requests_per_second *= 3
        
        try:
            # Execute stress test
            stress_results = await self.load_executor.execute_load_test(result.test_id)
            
            # Store results
            result.throughput_stats = {
                "rps": stress_results.get("throughput_rps", 0),
                "total": stress_results.get("requests_completed", 0) + stress_results.get("requests_failed", 0),
                "error_rate": stress_results.get("error_rate", 0)
            }
            
        finally:
            # Restore original parameters
            self.config.concurrent_users = original_concurrent_users
            self.config.requests_per_second = original_rps
    
    async def _run_regression_test(self, result: PerformanceTestResult):
        """Run regression test."""
        logger.info("Running regression test")
        
        # Run standard load test for regression comparison
        await self._run_load_test(result)
    
    async def _run_generic_test(self, result: PerformanceTestResult):
        """Run generic performance test."""
        logger.info("Running generic performance test")
        
        # Wait for metrics collection
        await asyncio.sleep(self.config.duration_seconds)
    
    async def _analyze_results(self, result: PerformanceTestResult):
        """Analyze test results."""
        try:
            # Group metrics by type and component
            metrics_by_type = {}
            for metric in result.metrics:
                key = f"{metric.component}_{metric.metric_type.value}"
                if key not in metrics_by_type:
                    metrics_by_type[key] = []
                metrics_by_type[key].append(metric.value)
            
            # Calculate statistics for each metric type
            for key, values in metrics_by_type.items():
                if len(values) > 1:
                    component, metric_type = key.rsplit('_', 1)
                    
                    # Calculate basic statistics
                    avg_value = statistics.mean(values)
                    min_value = min(values)
                    max_value = max(values)
                    
                    if len(values) > 1:
                        std_dev = statistics.stdev(values)
                        p95_value = np.percentile(values, 95)
                        p99_value = np.percentile(values, 99)
                    else:
                        std_dev = 0
                        p95_value = avg_value
                        p99_value = avg_value
                    
                    # Store in appropriate stats
                    if metric_type == "latency":
                        result.latency_stats.update({
                            "avg": avg_value,
                            "min": min_value,
                            "max": max_value,
                            "p95": p95_value,
                            "p99": p99_value,
                            "std_dev": std_dev
                        })
                    elif metric_type == "memory_usage":
                        result.resource_stats.update({
                            "memory_mb": avg_value,
                            "memory_peak": max_value
                        })
                    elif metric_type == "cpu_usage":
                        result.resource_stats.update({
                            "cpu_percent": avg_value,
                            "cpu_peak": max_value
                        })
                        
        except Exception as e:
            logger.error(f"Result analysis failed: {e}")
    
    async def _check_regressions(self, result: PerformanceTestResult):
        """Check for performance regressions."""
        try:
            if not self.config.enable_baseline_comparison:
                return
            
            # Check metrics against baselines
            regressions = self.baseline_manager.check_for_regressions(result.metrics)
            
            for metric, baseline, severity in regressions:
                result.add_regression(
                    metric.component,
                    metric.metric_type,
                    baseline.baseline_value,
                    metric.value,
                    severity
                )
                
                # Add to baseline comparisons
                result.baseline_comparisons.append({
                    "component": metric.component,
                    "metric_type": metric.metric_type.value,
                    "baseline_value": baseline.baseline_value,
                    "current_value": metric.value,
                    "degradation_percent": ((metric.value - baseline.baseline_value) / baseline.baseline_value) * 100,
                    "severity": severity.value,
                    "baseline_timestamp": baseline.baseline_timestamp.isoformat()
                })
                
        except Exception as e:
            logger.error(f"Regression check failed: {e}")
    
    async def _generate_recommendations(self, result: PerformanceTestResult):
        """Generate performance recommendations."""
        try:
            recommendations = []
            
            # Latency recommendations
            if result.latency_stats.get("avg", 0) > self.config.latency_threshold_ms:
                recommendations.append(
                    f"Average latency {result.latency_stats['avg']:.2f}ms exceeds threshold {self.config.latency_threshold_ms}ms"
                )
            
            # Throughput recommendations
            if result.throughput_stats.get("rps", 0) < self.config.throughput_threshold_rps:
                recommendations.append(
                    f"Throughput {result.throughput_stats['rps']:.2f} RPS below threshold {self.config.throughput_threshold_rps} RPS"
                )
            
            # Memory recommendations
            if result.resource_stats.get("memory_mb", 0) > self.config.memory_threshold_mb:
                recommendations.append(
                    f"Memory usage {result.resource_stats['memory_mb']:.2f}MB exceeds threshold {self.config.memory_threshold_mb}MB"
                )
            
            # CPU recommendations
            if result.resource_stats.get("cpu_percent", 0) > self.config.cpu_threshold_percent:
                recommendations.append(
                    f"CPU usage {result.resource_stats['cpu_percent']:.2f}% exceeds threshold {self.config.cpu_threshold_percent}%"
                )
            
            # Regression recommendations
            if result.regressions_detected:
                critical_regressions = [r for r in result.regressions_detected if r["severity"] == "critical"]
                if critical_regressions:
                    recommendations.append("Critical performance regressions detected - immediate attention required")
            
            # Error rate recommendations
            if result.throughput_stats.get("error_rate", 0) > self.config.error_rate_threshold_percent:
                recommendations.append(
                    f"Error rate {result.throughput_stats['error_rate']:.2f}% exceeds threshold {self.config.error_rate_threshold_percent}%"
                )
            
            result.recommendations = recommendations
            
            # Determine overall success
            result.success = (
                len(result.regressions_detected) == 0 and
                result.latency_stats.get("avg", 0) <= self.config.latency_threshold_ms and
                result.throughput_stats.get("rps", 0) >= self.config.throughput_threshold_rps and
                result.resource_stats.get("memory_mb", 0) <= self.config.memory_threshold_mb and
                result.resource_stats.get("cpu_percent", 0) <= self.config.cpu_threshold_percent
            )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
    
    async def _update_baselines(self, result: PerformanceTestResult):
        """Update performance baselines."""
        try:
            # Group metrics by component and type
            metrics_by_key = {}
            for metric in result.metrics:
                key = f"{metric.component}_{metric.metric_type.value}"
                if key not in metrics_by_key:
                    metrics_by_key[key] = []
                metrics_by_key[key].append(metric.value)
            
            # Update baselines
            for key, values in metrics_by_key.items():
                if len(values) > 5:  # Only update if we have enough data points
                    component, metric_type_str = key.rsplit('_', 1)
                    metric_type = PerformanceMetricType(metric_type_str)
                    
                    # Calculate new baseline value (median of recent values)
                    new_baseline = statistics.median(values)
                    
                    # Update baseline
                    self.baseline_manager.update_baseline(
                        component,
                        metric_type,
                        new_baseline,
                        values
                    )
                    
            logger.info("Updated performance baselines")
            
        except Exception as e:
            logger.error(f"Baseline update failed: {e}")


# Example usage
async def main():
    """Demonstrate performance regression testing."""
    config = PerformanceTestConfig(
        test_id="performance_regression_test_001",
        test_type=PerformanceTestType.REGRESSION_TEST,
        duration_seconds=300,
        concurrent_users=10,
        endpoints=["http://localhost:8001", "http://localhost:8002"],
        enable_profiling=True,
        enable_reporting=True
    )
    
    tester = PerformanceRegressionTester(config)
    
    # Run regression test
    result = await tester.run_performance_test(PerformanceTestType.REGRESSION_TEST)
    
    print(f"Test ID: {result.test_id}")
    print(f"Status: {result.status}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Regressions: {len(result.regressions_detected)}")
    print(f"Recommendations: {len(result.recommendations)}")
    
    if result.latency_stats:
        print(f"Average Latency: {result.latency_stats['avg']:.2f}ms")
        print(f"P95 Latency: {result.latency_stats['p95']:.2f}ms")
    
    if result.throughput_stats:
        print(f"Throughput: {result.throughput_stats['rps']:.2f} RPS")
        print(f"Error Rate: {result.throughput_stats['error_rate']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())