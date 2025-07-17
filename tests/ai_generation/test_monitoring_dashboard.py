"""
Real-time Test Execution Monitoring Dashboard
Agent 5 Mission: Real-time Test Monitoring & Analytics
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import sqlite3
import pickle
import numpy as np
from scipy import stats
import structlog

from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class TestMetrics:
    """Comprehensive test execution metrics"""
    test_id: str
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, passed, failed, skipped
    duration_ms: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_bytes: float = 0.0
    error_message: Optional[str] = None
    failure_count: int = 0
    retry_count: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TestSuiteMetrics:
    """Test suite execution metrics"""
    suite_id: str
    suite_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration_ms: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}


@dataclass
class TestFailurePattern:
    """Test failure pattern analysis"""
    pattern_id: str
    pattern_type: str  # error_message, resource_exhaustion, timeout, etc.
    pattern_signature: str
    occurrences: int = 1
    first_seen: datetime = None
    last_seen: datetime = None
    affected_tests: List[str] = None
    severity: str = "medium"  # low, medium, high, critical
    suggested_fix: Optional[str] = None
    
    def __post_init__(self):
        if self.affected_tests is None:
            self.affected_tests = []
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_seen is None:
            self.last_seen = datetime.now()


class ResourceMonitor:
    """Real-time resource usage monitoring"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.process = psutil.Process()
        self.start_stats = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_stats = {
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'disk_io': self.process.io_counters() if hasattr(self.process, 'io_counters') else None,
            'network_io': psutil.net_io_counters()
        }
        
        def monitor_loop():
            while self.monitoring:
                try:
                    cpu_percent = self.process.cpu_percent()
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    disk_io = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
                    network_io = psutil.net_io_counters()
                    
                    metrics = {
                        'timestamp': datetime.now(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb,
                        'disk_read_mb': disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                        'disk_write_mb': disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                        'network_bytes': network_io.bytes_sent + network_io.bytes_recv
                    }
                    
                    self.metrics_history.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error monitoring resources: {e}")
                
                time.sleep(self.sample_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics"""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
    
    def get_average_metrics(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get average metrics over time window"""
        if not self.metrics_history:
            return {}
        
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=window_seconds)
        
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > cutoff_time]
        if not recent_metrics:
            return {}
        
        return {
            'cpu_percent': np.mean([m['cpu_percent'] for m in recent_metrics]),
            'memory_mb': np.mean([m['memory_mb'] for m in recent_metrics]),
            'disk_read_mb': np.mean([m['disk_read_mb'] for m in recent_metrics]),
            'disk_write_mb': np.mean([m['disk_write_mb'] for m in recent_metrics]),
            'network_bytes': np.mean([m['network_bytes'] for m in recent_metrics])
        }


class TestProgressPredictor:
    """ML-based test execution time prediction"""
    
    def __init__(self, history_file: str = "test_history.db"):
        self.history_file = history_file
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for test history"""
        conn = sqlite3.connect(self.history_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                duration_ms REAL,
                cpu_usage REAL,
                memory_usage_mb REAL,
                timestamp DATETIME,
                status TEXT,
                tags TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_test_completion(self, metrics: TestMetrics):
        """Record completed test for future predictions"""
        conn = sqlite3.connect(self.history_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_history 
            (test_name, duration_ms, cpu_usage, memory_usage_mb, timestamp, status, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.test_name,
            metrics.duration_ms,
            metrics.cpu_usage,
            metrics.memory_usage_mb,
            metrics.start_time,
            metrics.status,
            json.dumps(metrics.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def predict_test_duration(self, test_name: str, tags: List[str] = None) -> float:
        """Predict test duration based on historical data"""
        conn = sqlite3.connect(self.history_file)
        cursor = conn.cursor()
        
        # Get historical data for similar tests
        cursor.execute('''
            SELECT duration_ms FROM test_history 
            WHERE test_name = ? AND status = 'passed'
            ORDER BY timestamp DESC LIMIT 10
        ''', (test_name,))
        
        durations = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not durations:
            # No historical data, use default estimate
            return 5000.0  # 5 seconds default
        
        # Use exponential smoothing for prediction
        if len(durations) == 1:
            return durations[0]
        
        alpha = 0.3
        prediction = durations[0]
        for duration in durations[1:]:
            prediction = alpha * duration + (1 - alpha) * prediction
        
        return prediction
    
    def estimate_suite_completion(self, suite_metrics: TestSuiteMetrics, 
                                 remaining_tests: List[str]) -> datetime:
        """Estimate suite completion time"""
        if not remaining_tests:
            return datetime.now()
        
        total_estimated_duration = 0
        for test_name in remaining_tests:
            estimated_duration = self.predict_test_duration(test_name)
            total_estimated_duration += estimated_duration
        
        # Add buffer for overhead (10%)
        total_estimated_duration *= 1.1
        
        return datetime.now() + timedelta(milliseconds=total_estimated_duration)


class TestFailureAnalyzer:
    """Test failure pattern analysis and prediction"""
    
    def __init__(self):
        self.failure_patterns = {}
        self.failure_history = deque(maxlen=1000)
        
    def analyze_failure(self, test_metrics: TestMetrics) -> TestFailurePattern:
        """Analyze test failure and identify patterns"""
        if test_metrics.status != "failed":
            return None
        
        # Extract failure signature
        signature = self._extract_failure_signature(test_metrics)
        pattern_id = f"pattern_{hash(signature)}"
        
        if pattern_id in self.failure_patterns:
            pattern = self.failure_patterns[pattern_id]
            pattern.occurrences += 1
            pattern.last_seen = datetime.now()
            pattern.affected_tests.append(test_metrics.test_name)
        else:
            pattern = TestFailurePattern(
                pattern_id=pattern_id,
                pattern_type=self._classify_failure_type(test_metrics),
                pattern_signature=signature,
                affected_tests=[test_metrics.test_name]
            )
            self.failure_patterns[pattern_id] = pattern
        
        # Update severity based on frequency
        pattern.severity = self._calculate_severity(pattern)
        pattern.suggested_fix = self._suggest_fix(pattern)
        
        self.failure_history.append(test_metrics)
        return pattern
    
    def _extract_failure_signature(self, test_metrics: TestMetrics) -> str:
        """Extract failure signature from test metrics"""
        error_msg = test_metrics.error_message or ""
        
        # Normalize error message for pattern matching
        signature_parts = []
        
        # Extract key error indicators
        if "timeout" in error_msg.lower():
            signature_parts.append("timeout")
        elif "memory" in error_msg.lower() or "oom" in error_msg.lower():
            signature_parts.append("memory_error")
        elif "assertion" in error_msg.lower():
            signature_parts.append("assertion_error")
        elif "connection" in error_msg.lower():
            signature_parts.append("connection_error")
        elif "import" in error_msg.lower() or "module" in error_msg.lower():
            signature_parts.append("import_error")
        else:
            signature_parts.append("unknown_error")
        
        # Add resource-based indicators
        if test_metrics.memory_usage_mb > 1000:  # High memory usage
            signature_parts.append("high_memory")
        if test_metrics.cpu_usage > 90:  # High CPU usage
            signature_parts.append("high_cpu")
        
        return "|".join(signature_parts)
    
    def _classify_failure_type(self, test_metrics: TestMetrics) -> str:
        """Classify failure type"""
        error_msg = test_metrics.error_message or ""
        
        if "timeout" in error_msg.lower():
            return "timeout"
        elif "memory" in error_msg.lower() or "oom" in error_msg.lower():
            return "resource_exhaustion"
        elif "assertion" in error_msg.lower():
            return "assertion_failure"
        elif "connection" in error_msg.lower():
            return "network_error"
        elif "import" in error_msg.lower() or "module" in error_msg.lower():
            return "dependency_error"
        else:
            return "unknown"
    
    def _calculate_severity(self, pattern: TestFailurePattern) -> str:
        """Calculate failure pattern severity"""
        if pattern.occurrences >= 10:
            return "critical"
        elif pattern.occurrences >= 5:
            return "high"
        elif pattern.occurrences >= 3:
            return "medium"
        else:
            return "low"
    
    def _suggest_fix(self, pattern: TestFailurePattern) -> str:
        """Suggest fix based on failure pattern"""
        fixes = {
            "timeout": "Consider increasing timeout values or optimizing test performance",
            "resource_exhaustion": "Optimize memory usage or increase resource limits",
            "assertion_failure": "Review test assertions and expected behavior",
            "network_error": "Check network connectivity and retry logic",
            "dependency_error": "Verify all dependencies are installed and up to date"
        }
        return fixes.get(pattern.pattern_type, "Manual investigation required")


class TestHealthScorer:
    """Test suite health scoring and recommendations"""
    
    def __init__(self):
        self.weight_config = {
            'pass_rate': 0.4,
            'performance': 0.3,
            'stability': 0.2,
            'resource_efficiency': 0.1
        }
    
    def calculate_health_score(self, suite_metrics: TestSuiteMetrics, 
                             test_metrics: List[TestMetrics]) -> Dict[str, Any]:
        """Calculate comprehensive health score"""
        if not test_metrics:
            return {'overall_score': 0, 'components': {}, 'recommendations': []}
        
        components = {}
        
        # Pass rate score
        total_tests = len(test_metrics)
        passed_tests = sum(1 for t in test_metrics if t.status == 'passed')
        components['pass_rate'] = (passed_tests / total_tests) * 100
        
        # Performance score
        avg_duration = np.mean([t.duration_ms for t in test_metrics if t.duration_ms])
        performance_score = max(0, 100 - (avg_duration / 1000))  # Penalize slow tests
        components['performance'] = min(100, performance_score)
        
        # Stability score (based on failure patterns)
        failed_tests = [t for t in test_metrics if t.status == 'failed']
        unique_failures = len(set(t.error_message for t in failed_tests))
        stability_score = 100 - (unique_failures * 5)  # Penalize diverse failures
        components['stability'] = max(0, stability_score)
        
        # Resource efficiency score
        avg_memory = np.mean([t.memory_usage_mb for t in test_metrics])
        avg_cpu = np.mean([t.cpu_usage for t in test_metrics])
        resource_score = 100 - (avg_memory / 10) - (avg_cpu / 10)  # Penalize high usage
        components['resource_efficiency'] = max(0, resource_score)
        
        # Calculate overall score
        overall_score = sum(
            components[component] * self.weight_config[component]
            for component in components
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(components, test_metrics)
        
        return {
            'overall_score': overall_score,
            'components': components,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    def _generate_recommendations(self, components: Dict[str, float], 
                                test_metrics: List[TestMetrics]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if components['pass_rate'] < 95:
            recommendations.append("Improve test pass rate by fixing failing tests")
        
        if components['performance'] < 70:
            recommendations.append("Optimize slow tests to improve performance")
        
        if components['stability'] < 80:
            recommendations.append("Address flaky tests to improve stability")
        
        if components['resource_efficiency'] < 70:
            recommendations.append("Optimize resource usage in tests")
        
        # Specific recommendations based on patterns
        flaky_tests = [t for t in test_metrics if t.retry_count > 0]
        if len(flaky_tests) > len(test_metrics) * 0.1:  # More than 10% flaky
            recommendations.append("High flaky test rate detected - investigate root causes")
        
        slow_tests = [t for t in test_metrics if t.duration_ms and t.duration_ms > 10000]
        if len(slow_tests) > len(test_metrics) * 0.05:  # More than 5% slow
            recommendations.append("Multiple slow tests detected - consider parallelization")
        
        return recommendations


class RealTimeTestMonitor:
    """Main real-time test monitoring dashboard"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.resource_monitor = ResourceMonitor()
        self.progress_predictor = TestProgressPredictor()
        self.failure_analyzer = TestFailureAnalyzer()
        self.health_scorer = TestHealthScorer()
        
        # Active monitoring data
        self.active_tests = {}
        self.active_suites = {}
        self.completed_tests = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # Configuration
        self.alert_thresholds = {
            'memory_mb': 1000,
            'cpu_percent': 90,
            'duration_ms': 30000,
            'failure_rate': 0.1
        }
        
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Setup event handlers for test monitoring"""
        # We'll use custom event types for test monitoring
        pass
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.resource_monitor.start_monitoring()
        logger.info("Real-time test monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.resource_monitor.stop_monitoring()
        logger.info("Real-time test monitoring stopped")
    
    def start_test(self, test_name: str, tags: List[str] = None) -> str:
        """Start monitoring a test"""
        test_id = f"test_{int(time.time() * 1000)}"
        
        metrics = TestMetrics(
            test_id=test_id,
            test_name=test_name,
            start_time=datetime.now(),
            tags=tags or []
        )
        
        self.active_tests[test_id] = metrics
        
        # Predict duration
        predicted_duration = self.progress_predictor.predict_test_duration(test_name, tags)
        
        logger.info(f"Test started: {test_name} (predicted: {predicted_duration}ms)")
        return test_id
    
    def finish_test(self, test_id: str, status: str, error_message: str = None):
        """Finish monitoring a test"""
        if test_id not in self.active_tests:
            return
        
        metrics = self.active_tests[test_id]
        metrics.end_time = datetime.now()
        metrics.status = status
        metrics.error_message = error_message
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        
        # Add resource usage
        current_resources = self.resource_monitor.get_current_metrics()
        if current_resources:
            metrics.cpu_usage = current_resources.get('cpu_percent', 0)
            metrics.memory_usage_mb = current_resources.get('memory_mb', 0)
            metrics.disk_io_read_mb = current_resources.get('disk_read_mb', 0)
            metrics.disk_io_write_mb = current_resources.get('disk_write_mb', 0)
            metrics.network_io_bytes = current_resources.get('network_bytes', 0)
        
        # Move to completed tests
        self.completed_tests.append(metrics)
        del self.active_tests[test_id]
        
        # Record for predictions
        self.progress_predictor.record_test_completion(metrics)
        
        # Analyze failures
        if status == "failed":
            failure_pattern = self.failure_analyzer.analyze_failure(metrics)
            if failure_pattern:
                self._handle_failure_pattern(failure_pattern)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        logger.info(f"Test completed: {metrics.test_name} ({status}) - {metrics.duration_ms}ms")
    
    def start_test_suite(self, suite_name: str, total_tests: int) -> str:
        """Start monitoring a test suite"""
        suite_id = f"suite_{int(time.time() * 1000)}"
        
        suite_metrics = TestSuiteMetrics(
            suite_id=suite_id,
            suite_name=suite_name,
            start_time=datetime.now(),
            total_tests=total_tests
        )
        
        self.active_suites[suite_id] = suite_metrics
        
        logger.info(f"Test suite started: {suite_name} ({total_tests} tests)")
        return suite_id
    
    def update_suite_progress(self, suite_id: str, completed_tests: int, 
                            remaining_tests: List[str] = None):
        """Update test suite progress"""
        if suite_id not in self.active_suites:
            return
        
        suite_metrics = self.active_suites[suite_id]
        suite_metrics.progress_percentage = (completed_tests / suite_metrics.total_tests) * 100
        
        if remaining_tests:
            suite_metrics.estimated_completion_time = self.progress_predictor.estimate_suite_completion(
                suite_metrics, remaining_tests
            )
        
        logger.debug(f"Suite progress updated: {suite_metrics.suite_name} "
                    f"({suite_metrics.progress_percentage:.1f}%)")
    
    def finish_test_suite(self, suite_id: str):
        """Finish monitoring a test suite"""
        if suite_id not in self.active_suites:
            return
        
        suite_metrics = self.active_suites[suite_id]
        suite_metrics.end_time = datetime.now()
        
        # Calculate final statistics
        suite_tests = [t for t in self.completed_tests 
                      if t.start_time >= suite_metrics.start_time]
        
        suite_metrics.passed_tests = sum(1 for t in suite_tests if t.status == 'passed')
        suite_metrics.failed_tests = sum(1 for t in suite_tests if t.status == 'failed')
        suite_metrics.skipped_tests = sum(1 for t in suite_tests if t.status == 'skipped')
        suite_metrics.total_duration_ms = sum(t.duration_ms for t in suite_tests if t.duration_ms)
        
        # Calculate health score
        health_score = self.health_scorer.calculate_health_score(suite_metrics, suite_tests)
        
        del self.active_suites[suite_id]
        
        logger.info(f"Test suite completed: {suite_metrics.suite_name} "
                   f"(Health Score: {health_score['overall_score']:.1f})")
        
        return health_score
    
    def _handle_failure_pattern(self, pattern: TestFailurePattern):
        """Handle identified failure pattern"""
        alert = {
            'type': 'failure_pattern',
            'severity': pattern.severity,
            'message': f"Failure pattern detected: {pattern.pattern_type}",
            'details': {
                'pattern_id': pattern.pattern_id,
                'occurrences': pattern.occurrences,
                'affected_tests': pattern.affected_tests,
                'suggested_fix': pattern.suggested_fix
            },
            'timestamp': datetime.now()
        }
        
        self.alerts.append(alert)
        logger.warning(f"Failure pattern detected: {pattern.pattern_type} "
                      f"({pattern.occurrences} occurrences)")
    
    def _check_alerts(self, metrics: TestMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # High memory usage
        if metrics.memory_usage_mb > self.alert_thresholds['memory_mb']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                'test_name': metrics.test_name
            })
        
        # High CPU usage
        if metrics.cpu_usage > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_usage:.1f}%",
                'test_name': metrics.test_name
            })
        
        # Long duration
        if metrics.duration_ms and metrics.duration_ms > self.alert_thresholds['duration_ms']:
            alerts.append({
                'type': 'slow_test',
                'severity': 'info',
                'message': f"Slow test execution: {metrics.duration_ms:.0f}ms",
                'test_name': metrics.test_name
            })
        
        for alert in alerts:
            alert['timestamp'] = datetime.now()
            self.alerts.append(alert)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        current_resources = self.resource_monitor.get_current_metrics()
        
        return {
            'active_tests': len(self.active_tests),
            'active_suites': len(self.active_suites),
            'completed_tests': len(self.completed_tests),
            'recent_alerts': list(self.alerts)[-10:],
            'current_resources': current_resources,
            'average_resources': self.resource_monitor.get_average_metrics(300),  # 5 min
            'active_test_details': [asdict(t) for t in self.active_tests.values()],
            'active_suite_details': [asdict(s) for s in self.active_suites.values()],
            'failure_patterns': [asdict(p) for p in self.failure_analyzer.failure_patterns.values()],
            'timestamp': datetime.now()
        }
    
    def get_analytics_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate analytics report"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_tests = [t for t in self.completed_tests if t.start_time > cutoff_time]
        
        if not recent_tests:
            return {'message': 'No test data available for the specified period'}
        
        # Basic statistics
        total_tests = len(recent_tests)
        passed_tests = sum(1 for t in recent_tests if t.status == 'passed')
        failed_tests = sum(1 for t in recent_tests if t.status == 'failed')
        
        # Performance metrics
        durations = [t.duration_ms for t in recent_tests if t.duration_ms]
        avg_duration = np.mean(durations) if durations else 0
        p95_duration = np.percentile(durations, 95) if durations else 0
        
        # Resource usage
        memory_usage = [t.memory_usage_mb for t in recent_tests if t.memory_usage_mb]
        cpu_usage = [t.cpu_usage for t in recent_tests if t.cpu_usage]
        
        # Trends
        daily_stats = self._calculate_daily_trends(recent_tests)
        
        return {
            'period_days': days,
            'total_tests': total_tests,
            'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'performance': {
                'avg_duration_ms': avg_duration,
                'p95_duration_ms': p95_duration,
                'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                'avg_cpu_percent': np.mean(cpu_usage) if cpu_usage else 0
            },
            'daily_trends': daily_stats,
            'top_failure_patterns': self._get_top_failure_patterns(),
            'recommendations': self._generate_analytics_recommendations(recent_tests),
            'timestamp': datetime.now()
        }
    
    def _calculate_daily_trends(self, tests: List[TestMetrics]) -> List[Dict[str, Any]]:
        """Calculate daily trends"""
        daily_data = defaultdict(lambda: {'tests': [], 'date': None})
        
        for test in tests:
            date_key = test.start_time.date()
            daily_data[date_key]['tests'].append(test)
            daily_data[date_key]['date'] = date_key
        
        trends = []
        for date_key in sorted(daily_data.keys()):
            day_tests = daily_data[date_key]['tests']
            
            trends.append({
                'date': date_key.isoformat(),
                'total_tests': len(day_tests),
                'passed_tests': sum(1 for t in day_tests if t.status == 'passed'),
                'failed_tests': sum(1 for t in day_tests if t.status == 'failed'),
                'avg_duration_ms': np.mean([t.duration_ms for t in day_tests if t.duration_ms]),
                'avg_memory_mb': np.mean([t.memory_usage_mb for t in day_tests if t.memory_usage_mb])
            })
        
        return trends
    
    def _get_top_failure_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top failure patterns"""
        patterns = sorted(
            self.failure_analyzer.failure_patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True
        )
        
        return [asdict(p) for p in patterns[:limit]]
    
    def _generate_analytics_recommendations(self, tests: List[TestMetrics]) -> List[str]:
        """Generate analytics-based recommendations"""
        recommendations = []
        
        # Analyze test performance
        slow_tests = [t for t in tests if t.duration_ms and t.duration_ms > 10000]
        if len(slow_tests) > len(tests) * 0.1:
            recommendations.append("Consider optimizing slow tests or running them in parallel")
        
        # Analyze failure patterns
        failed_tests = [t for t in tests if t.status == 'failed']
        if len(failed_tests) > len(tests) * 0.05:
            recommendations.append("High failure rate detected - investigate root causes")
        
        # Analyze resource usage
        high_memory_tests = [t for t in tests if t.memory_usage_mb > 500]
        if len(high_memory_tests) > len(tests) * 0.2:
            recommendations.append("High memory usage detected - optimize test resource usage")
        
        return recommendations