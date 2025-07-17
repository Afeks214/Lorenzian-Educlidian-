"""
Test Suite Health Reporting System
Agent 5 Mission: Real-time Test Monitoring & Analytics
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import structlog
from enum import Enum

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class TestSuiteHealthMetrics:
    """Comprehensive test suite health metrics"""
    suite_name: str
    timestamp: datetime
    overall_health_score: float  # 0-100
    health_status: HealthStatus
    
    # Performance metrics
    avg_execution_time_ms: float
    p95_execution_time_ms: float
    total_execution_time_ms: float
    performance_score: float
    
    # Reliability metrics
    pass_rate: float
    flakiness_score: float
    stability_score: float
    
    # Resource efficiency
    avg_memory_usage_mb: float
    avg_cpu_usage_percent: float
    resource_efficiency_score: float
    
    # Maintainability
    code_coverage: float
    test_complexity_score: float
    maintainability_score: float
    
    # Trend analysis
    performance_trend: str  # improving, stable, degrading
    reliability_trend: str
    resource_trend: str
    
    # Test counts
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    flaky_tests: int
    slow_tests: int
    
    # Recommendations
    critical_issues: List[str]
    recommendations: List[str]
    action_items: List[str]


@dataclass
class TestHealthIndicator:
    """Individual test health indicator"""
    test_name: str
    health_score: float
    status: HealthStatus
    execution_time_ms: float
    pass_rate: float
    flakiness_score: float
    memory_usage_mb: float
    last_failure_reason: Optional[str]
    recommendations: List[str]


class TestSuiteHealthAnalyzer:
    """Analyzes test suite health metrics"""
    
    def __init__(self, db_path: str = "test_health.db"):
        self.db_path = db_path
        self.health_history = deque(maxlen=1000)
        self.thresholds = {
            'performance': {
                'excellent': 1000,  # < 1s
                'good': 5000,       # < 5s
                'fair': 15000,      # < 15s
                'poor': 30000,      # < 30s
                'critical': 60000   # < 60s
            },
            'pass_rate': {
                'excellent': 0.99,
                'good': 0.95,
                'fair': 0.90,
                'poor': 0.80,
                'critical': 0.70
            },
            'flakiness': {
                'excellent': 0.01,
                'good': 0.05,
                'fair': 0.10,
                'poor': 0.20,
                'critical': 0.30
            },
            'memory': {
                'excellent': 100,   # < 100MB
                'good': 500,        # < 500MB
                'fair': 1000,       # < 1GB
                'poor': 2000,       # < 2GB
                'critical': 4000    # < 4GB
            }
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Setup database for health tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suite_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_name TEXT,
                timestamp DATETIME,
                overall_health_score REAL,
                health_status TEXT,
                performance_score REAL,
                reliability_score REAL,
                resource_efficiency_score REAL,
                maintainability_score REAL,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                metrics_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                suite_name TEXT,
                timestamp DATETIME,
                health_score REAL,
                status TEXT,
                execution_time_ms REAL,
                pass_rate REAL,
                flakiness_score REAL,
                memory_usage_mb REAL,
                recommendations TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_suite_health(self, suite_data: Dict[str, Any]) -> TestSuiteHealthMetrics:
        """Analyze comprehensive test suite health"""
        suite_name = suite_data['suite_name']
        tests = suite_data['tests']
        
        # Calculate individual test health
        test_health_indicators = []
        for test_data in tests:
            indicator = self._analyze_test_health(test_data)
            test_health_indicators.append(indicator)
        
        # Calculate suite-level metrics
        total_tests = len(tests)
        passed_tests = sum(1 for t in tests if t.get('status') == 'passed')
        failed_tests = sum(1 for t in tests if t.get('status') == 'failed')
        skipped_tests = sum(1 for t in tests if t.get('status') == 'skipped')
        
        # Performance metrics
        execution_times = [t.get('duration_ms', 0) for t in tests if t.get('duration_ms')]
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        p95_execution_time = np.percentile(execution_times, 95) if execution_times else 0
        total_execution_time = sum(execution_times)
        
        # Reliability metrics
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        flaky_tests = [t for t in test_health_indicators if t.flakiness_score > 0.1]
        flakiness_score = np.mean([t.flakiness_score for t in test_health_indicators])
        
        # Resource metrics
        memory_usage = [t.get('memory_usage_mb', 0) for t in tests if t.get('memory_usage_mb')]
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        
        cpu_usage = [t.get('cpu_usage', 0) for t in tests if t.get('cpu_usage')]
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        
        # Calculate component scores
        performance_score = self._calculate_performance_score(avg_execution_time, p95_execution_time)
        reliability_score = self._calculate_reliability_score(pass_rate, flakiness_score)
        resource_efficiency_score = self._calculate_resource_score(avg_memory_usage, avg_cpu_usage)
        maintainability_score = self._calculate_maintainability_score(tests)
        
        # Calculate overall health score
        overall_health_score = self._calculate_overall_health_score(
            performance_score, reliability_score, resource_efficiency_score, maintainability_score
        )
        
        # Determine health status
        health_status = self._determine_health_status(overall_health_score)
        
        # Analyze trends
        performance_trend = self._analyze_performance_trend(suite_name)
        reliability_trend = self._analyze_reliability_trend(suite_name)
        resource_trend = self._analyze_resource_trend(suite_name)
        
        # Generate recommendations
        critical_issues = self._identify_critical_issues(test_health_indicators, suite_data)
        recommendations = self._generate_recommendations(test_health_indicators, suite_data)
        action_items = self._generate_action_items(critical_issues, recommendations)
        
        # Create health metrics
        health_metrics = TestSuiteHealthMetrics(
            suite_name=suite_name,
            timestamp=datetime.now(),
            overall_health_score=overall_health_score,
            health_status=health_status,
            avg_execution_time_ms=avg_execution_time,
            p95_execution_time_ms=p95_execution_time,
            total_execution_time_ms=total_execution_time,
            performance_score=performance_score,
            pass_rate=pass_rate,
            flakiness_score=flakiness_score,
            stability_score=reliability_score,
            avg_memory_usage_mb=avg_memory_usage,
            avg_cpu_usage_percent=avg_cpu_usage,
            resource_efficiency_score=resource_efficiency_score,
            code_coverage=suite_data.get('code_coverage', 0),
            test_complexity_score=self._calculate_complexity_score(tests),
            maintainability_score=maintainability_score,
            performance_trend=performance_trend,
            reliability_trend=reliability_trend,
            resource_trend=resource_trend,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            flaky_tests=len(flaky_tests),
            slow_tests=len([t for t in tests if t.get('duration_ms', 0) > 10000]),
            critical_issues=critical_issues,
            recommendations=recommendations,
            action_items=action_items
        )
        
        # Store in database
        self._store_health_metrics(health_metrics)
        
        return health_metrics
    
    def _analyze_test_health(self, test_data: Dict[str, Any]) -> TestHealthIndicator:
        """Analyze individual test health"""
        test_name = test_data['test_name']
        execution_time = test_data.get('duration_ms', 0)
        
        # Calculate metrics from test history
        test_history = test_data.get('test_history', [])
        
        if test_history:
            pass_rate = sum(1 for r in test_history if r.get('status') == 'passed') / len(test_history)
            flakiness_score = self._calculate_flakiness_from_history(test_history)
        else:
            pass_rate = 1.0 if test_data.get('status') == 'passed' else 0.0
            flakiness_score = 0.0
        
        memory_usage = test_data.get('memory_usage_mb', 0)
        
        # Calculate individual health score
        performance_component = self._score_performance(execution_time)
        reliability_component = self._score_reliability(pass_rate, flakiness_score)
        resource_component = self._score_resource_usage(memory_usage)
        
        health_score = (performance_component + reliability_component + resource_component) / 3
        
        # Determine status
        if health_score >= 90:
            status = HealthStatus.EXCELLENT
        elif health_score >= 75:
            status = HealthStatus.GOOD
        elif health_score >= 60:
            status = HealthStatus.FAIR
        elif health_score >= 40:
            status = HealthStatus.POOR
        else:
            status = HealthStatus.CRITICAL
        
        # Generate test-specific recommendations
        recommendations = self._generate_test_recommendations(test_data, health_score)
        
        return TestHealthIndicator(
            test_name=test_name,
            health_score=health_score,
            status=status,
            execution_time_ms=execution_time,
            pass_rate=pass_rate,
            flakiness_score=flakiness_score,
            memory_usage_mb=memory_usage,
            last_failure_reason=test_data.get('last_failure_reason'),
            recommendations=recommendations
        )
    
    def _calculate_performance_score(self, avg_time: float, p95_time: float) -> float:
        """Calculate performance score"""
        # Base score on average execution time
        avg_score = self._score_performance(avg_time)
        
        # Penalty for high P95 (indicates inconsistent performance)
        if p95_time > avg_time * 2:
            consistency_penalty = 10
        else:
            consistency_penalty = 0
        
        return max(0, avg_score - consistency_penalty)
    
    def _calculate_reliability_score(self, pass_rate: float, flakiness_score: float) -> float:
        """Calculate reliability score"""
        pass_rate_score = pass_rate * 100
        flakiness_penalty = flakiness_score * 50  # High flakiness severely impacts score
        
        return max(0, pass_rate_score - flakiness_penalty)
    
    def _calculate_resource_score(self, avg_memory: float, avg_cpu: float) -> float:
        """Calculate resource efficiency score"""
        memory_score = self._score_resource_usage(avg_memory)
        cpu_score = max(0, 100 - avg_cpu)  # Lower CPU usage = higher score
        
        return (memory_score + cpu_score) / 2
    
    def _calculate_maintainability_score(self, tests: List[Dict[str, Any]]) -> float:
        """Calculate maintainability score"""
        if not tests:
            return 0
        
        # Factors affecting maintainability
        avg_complexity = np.mean([self._calculate_test_complexity(t) for t in tests])
        duplicate_code_penalty = self._calculate_duplicate_code_penalty(tests)
        
        base_score = 100
        complexity_penalty = min(50, avg_complexity * 2)  # Complex tests are harder to maintain
        
        return max(0, base_score - complexity_penalty - duplicate_code_penalty)
    
    def _calculate_complexity_score(self, tests: List[Dict[str, Any]]) -> float:
        """Calculate average test complexity"""
        if not tests:
            return 0
        
        complexities = [self._calculate_test_complexity(t) for t in tests]
        return np.mean(complexities)
    
    def _calculate_test_complexity(self, test_data: Dict[str, Any]) -> float:
        """Calculate individual test complexity"""
        complexity = 0
        
        # Lines of code
        loc = test_data.get('lines_of_code', 0)
        complexity += loc * 0.1
        
        # Number of assertions
        assertions = test_data.get('assertions', 0)
        complexity += assertions * 0.5
        
        # Dependencies
        dependencies = test_data.get('dependencies', [])
        complexity += len(dependencies) * 1.0
        
        # Mock usage
        mocks = test_data.get('mocks', [])
        complexity += len(mocks) * 0.5
        
        return complexity
    
    def _calculate_duplicate_code_penalty(self, tests: List[Dict[str, Any]]) -> float:
        """Calculate penalty for duplicate code"""
        # Simplified duplicate detection - in real implementation would use AST analysis
        test_patterns = defaultdict(int)
        
        for test in tests:
            # Simple pattern based on test structure
            pattern = f"{test.get('setup_type', '')}-{test.get('test_type', '')}"
            test_patterns[pattern] += 1
        
        # Calculate penalty based on duplicates
        penalty = 0
        for pattern, count in test_patterns.items():
            if count > 3:  # More than 3 similar tests
                penalty += (count - 3) * 2
        
        return min(30, penalty)  # Cap at 30 points
    
    def _calculate_flakiness_from_history(self, test_history: List[Dict[str, Any]]) -> float:
        """Calculate flakiness score from test history"""
        if len(test_history) < 10:
            return 0.0  # Not enough data
        
        # Look for patterns of intermittent failures
        results = [r.get('status') for r in test_history]
        
        # Count transitions from pass to fail
        transitions = 0
        for i in range(1, len(results)):
            if results[i] != results[i-1]:
                transitions += 1
        
        # Flakiness score based on transition frequency
        flakiness_score = transitions / len(results)
        return min(1.0, flakiness_score)
    
    def _score_performance(self, execution_time: float) -> float:
        """Score performance based on execution time"""
        thresholds = self.thresholds['performance']
        
        if execution_time <= thresholds['excellent']:
            return 100
        elif execution_time <= thresholds['good']:
            return 85
        elif execution_time <= thresholds['fair']:
            return 70
        elif execution_time <= thresholds['poor']:
            return 50
        else:
            return 25
    
    def _score_reliability(self, pass_rate: float, flakiness_score: float) -> float:
        """Score reliability based on pass rate and flakiness"""
        base_score = pass_rate * 100
        flakiness_penalty = flakiness_score * 30
        
        return max(0, base_score - flakiness_penalty)
    
    def _score_resource_usage(self, memory_usage: float) -> float:
        """Score resource usage"""
        thresholds = self.thresholds['memory']
        
        if memory_usage <= thresholds['excellent']:
            return 100
        elif memory_usage <= thresholds['good']:
            return 85
        elif memory_usage <= thresholds['fair']:
            return 70
        elif memory_usage <= thresholds['poor']:
            return 50
        else:
            return 25
    
    def _calculate_overall_health_score(self, performance: float, reliability: float, 
                                      resource: float, maintainability: float) -> float:
        """Calculate overall health score"""
        weights = {
            'performance': 0.3,
            'reliability': 0.4,
            'resource': 0.2,
            'maintainability': 0.1
        }
        
        overall_score = (
            performance * weights['performance'] +
            reliability * weights['reliability'] +
            resource * weights['resource'] +
            maintainability * weights['maintainability']
        )
        
        return overall_score
    
    def _determine_health_status(self, score: float) -> HealthStatus:
        """Determine health status from score"""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 75:
            return HealthStatus.GOOD
        elif score >= 60:
            return HealthStatus.FAIR
        elif score >= 40:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _analyze_performance_trend(self, suite_name: str) -> str:
        """Analyze performance trend"""
        # Get historical performance data
        recent_scores = self._get_recent_performance_scores(suite_name, days=30)
        
        if len(recent_scores) < 3:
            return "stable"
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        if slope > 2:
            return "improving"
        elif slope < -2:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_reliability_trend(self, suite_name: str) -> str:
        """Analyze reliability trend"""
        recent_scores = self._get_recent_reliability_scores(suite_name, days=30)
        
        if len(recent_scores) < 3:
            return "stable"
        
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        if slope > 1:
            return "improving"
        elif slope < -1:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_resource_trend(self, suite_name: str) -> str:
        """Analyze resource usage trend"""
        recent_scores = self._get_recent_resource_scores(suite_name, days=30)
        
        if len(recent_scores) < 3:
            return "stable"
        
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        if slope > 1:
            return "improving"
        elif slope < -1:
            return "degrading"
        else:
            return "stable"
    
    def _get_recent_performance_scores(self, suite_name: str, days: int) -> List[float]:
        """Get recent performance scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT performance_score FROM suite_health
            WHERE suite_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (suite_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]
    
    def _get_recent_reliability_scores(self, suite_name: str, days: int) -> List[float]:
        """Get recent reliability scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT reliability_score FROM suite_health
            WHERE suite_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (suite_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]
    
    def _get_recent_resource_scores(self, suite_name: str, days: int) -> List[float]:
        """Get recent resource efficiency scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT resource_efficiency_score FROM suite_health
            WHERE suite_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (suite_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]
    
    def _identify_critical_issues(self, test_indicators: List[TestHealthIndicator], 
                                 suite_data: Dict[str, Any]) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        critical_issues = []
        
        # Critical test failures
        critical_tests = [t for t in test_indicators if t.status == HealthStatus.CRITICAL]
        if critical_tests:
            critical_issues.append(f"{len(critical_tests)} tests in critical state")
        
        # High flakiness
        flaky_tests = [t for t in test_indicators if t.flakiness_score > 0.3]
        if len(flaky_tests) > 5:
            critical_issues.append(f"{len(flaky_tests)} highly flaky tests detected")
        
        # Performance issues
        slow_tests = [t for t in test_indicators if t.execution_time_ms > 30000]
        if len(slow_tests) > 3:
            critical_issues.append(f"{len(slow_tests)} tests taking >30s to execute")
        
        # Low pass rate
        pass_rate = sum(1 for t in test_indicators if t.pass_rate > 0.9) / len(test_indicators)
        if pass_rate < 0.8:
            critical_issues.append(f"Overall pass rate below 80%: {pass_rate:.1%}")
        
        return critical_issues
    
    def _generate_recommendations(self, test_indicators: List[TestHealthIndicator], 
                                 suite_data: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Performance recommendations
        slow_tests = [t for t in test_indicators if t.execution_time_ms > 10000]
        if len(slow_tests) > 5:
            recommendations.append("Consider parallelizing or optimizing slow tests")
        
        # Reliability recommendations
        unreliable_tests = [t for t in test_indicators if t.pass_rate < 0.95]
        if len(unreliable_tests) > 3:
            recommendations.append("Investigate and fix unreliable tests")
        
        # Resource recommendations
        memory_heavy_tests = [t for t in test_indicators if t.memory_usage_mb > 500]
        if len(memory_heavy_tests) > 5:
            recommendations.append("Optimize memory usage in resource-intensive tests")
        
        # Maintenance recommendations
        if suite_data.get('code_coverage', 0) < 0.8:
            recommendations.append("Increase test coverage to improve quality")
        
        return recommendations
    
    def _generate_action_items(self, critical_issues: List[str], 
                              recommendations: List[str]) -> List[str]:
        """Generate specific action items"""
        action_items = []
        
        # Critical issues become high-priority actions
        for issue in critical_issues:
            action_items.append(f"URGENT: Address {issue}")
        
        # Convert recommendations to actions
        for rec in recommendations:
            action_items.append(f"TODO: {rec}")
        
        return action_items
    
    def _generate_test_recommendations(self, test_data: Dict[str, Any], 
                                     health_score: float) -> List[str]:
        """Generate test-specific recommendations"""
        recommendations = []
        
        if health_score < 40:
            recommendations.append("Critical: Immediate attention required")
        
        if test_data.get('duration_ms', 0) > 30000:
            recommendations.append("Optimize test performance")
        
        if test_data.get('memory_usage_mb', 0) > 1000:
            recommendations.append("Reduce memory usage")
        
        test_history = test_data.get('test_history', [])
        if test_history:
            failures = [r for r in test_history if r.get('status') == 'failed']
            if len(failures) > len(test_history) * 0.2:
                recommendations.append("Investigate frequent failures")
        
        return recommendations
    
    def _store_health_metrics(self, metrics: TestSuiteHealthMetrics):
        """Store health metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO suite_health 
            (suite_name, timestamp, overall_health_score, health_status,
             performance_score, reliability_score, resource_efficiency_score,
             maintainability_score, total_tests, passed_tests, failed_tests, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.suite_name,
            metrics.timestamp,
            metrics.overall_health_score,
            metrics.health_status.value,
            metrics.performance_score,
            metrics.stability_score,
            metrics.resource_efficiency_score,
            metrics.maintainability_score,
            metrics.total_tests,
            metrics.passed_tests,
            metrics.failed_tests,
            json.dumps(asdict(metrics), default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def generate_health_report(self, metrics: TestSuiteHealthMetrics) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        return {
            'executive_summary': {
                'suite_name': metrics.suite_name,
                'overall_health': metrics.health_status.value,
                'health_score': metrics.overall_health_score,
                'total_tests': metrics.total_tests,
                'pass_rate': metrics.pass_rate,
                'critical_issues_count': len(metrics.critical_issues),
                'timestamp': metrics.timestamp
            },
            'performance_analysis': {
                'avg_execution_time_ms': metrics.avg_execution_time_ms,
                'p95_execution_time_ms': metrics.p95_execution_time_ms,
                'performance_score': metrics.performance_score,
                'performance_trend': metrics.performance_trend,
                'slow_tests_count': metrics.slow_tests
            },
            'reliability_analysis': {
                'pass_rate': metrics.pass_rate,
                'flakiness_score': metrics.flakiness_score,
                'stability_score': metrics.stability_score,
                'reliability_trend': metrics.reliability_trend,
                'flaky_tests_count': metrics.flaky_tests
            },
            'resource_analysis': {
                'avg_memory_usage_mb': metrics.avg_memory_usage_mb,
                'avg_cpu_usage_percent': metrics.avg_cpu_usage_percent,
                'resource_efficiency_score': metrics.resource_efficiency_score,
                'resource_trend': metrics.resource_trend
            },
            'maintainability_analysis': {
                'code_coverage': metrics.code_coverage,
                'test_complexity_score': metrics.test_complexity_score,
                'maintainability_score': metrics.maintainability_score
            },
            'issues_and_recommendations': {
                'critical_issues': metrics.critical_issues,
                'recommendations': metrics.recommendations,
                'action_items': metrics.action_items
            },
            'trends': {
                'performance_trend': metrics.performance_trend,
                'reliability_trend': metrics.reliability_trend,
                'resource_trend': metrics.resource_trend
            }
        }
    
    def get_health_history(self, suite_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get health history for a test suite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT timestamp, overall_health_score, health_status,
                   performance_score, reliability_score, resource_efficiency_score
            FROM suite_health
            WHERE suite_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (suite_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': r[0],
                'overall_health_score': r[1],
                'health_status': r[2],
                'performance_score': r[3],
                'reliability_score': r[4],
                'resource_efficiency_score': r[5]
            }
            for r in results
        ]