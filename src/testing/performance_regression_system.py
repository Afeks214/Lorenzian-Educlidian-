"""
Performance Regression Detection System - Agent 3

This module provides comprehensive performance regression detection and monitoring
using pytest-benchmark with advanced statistical analysis and automated alerting.

Features:
- pytest-benchmark integration with baseline tracking
- Performance regression detection with statistical significance testing
- Performance trend analysis and prediction
- Automated performance reporting and alerting
- CI/CD performance gates
- Performance budget management
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import pytest
import structlog
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result"""
    test_name: str
    timestamp: datetime
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    stddev_time: float
    rounds: int
    iterations: int
    git_commit: Optional[str] = None
    branch: Optional[str] = None
    environment: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionResult:
    """Performance regression analysis result"""
    test_name: str
    current_performance: float
    baseline_performance: float
    regression_detected: bool
    regression_severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # IMPROVING, STABLE, DEGRADING
    recommendation: str
    timestamp: datetime

@dataclass
class PerformanceBudget:
    """Performance budget configuration"""
    test_name: str
    max_time_ms: float
    max_regression_percent: float
    min_samples_for_baseline: int = 10
    significance_threshold: float = 0.05
    enabled: bool = True

class PerformanceRegressionDetector:
    """
    Advanced performance regression detection system with statistical analysis
    """
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = db_path
        self.baseline_cache = {}
        self.performance_budgets = {}
        self._init_database()
        
        # Statistical analysis parameters
        self.min_samples_for_analysis = 5
        self.significance_threshold = 0.05
        self.trend_analysis_window = 20
        
        logger.info("PerformanceRegressionDetector initialized", db_path=db_path)
    
    def _init_database(self):
        """Initialize SQLite database for performance history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                min_time REAL NOT NULL,
                max_time REAL NOT NULL,
                mean_time REAL NOT NULL,
                median_time REAL NOT NULL,
                stddev_time REAL NOT NULL,
                rounds INTEGER NOT NULL,
                iterations INTEGER NOT NULL,
                git_commit TEXT,
                branch TEXT,
                environment TEXT,
                additional_metrics TEXT
            )
        """)
        
        # Create baseline table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                test_name TEXT PRIMARY KEY,
                baseline_mean REAL NOT NULL,
                baseline_stddev REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                git_commit TEXT,
                branch TEXT
            )
        """)
        
        # Create regression alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_performance REAL NOT NULL,
                baseline_performance REAL NOT NULL,
                regression_severity TEXT NOT NULL,
                statistical_significance REAL NOT NULL,
                recommendation TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create performance budgets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_budgets (
                test_name TEXT PRIMARY KEY,
                max_time_ms REAL NOT NULL,
                max_regression_percent REAL NOT NULL,
                min_samples_for_baseline INTEGER NOT NULL,
                significance_threshold REAL NOT NULL,
                enabled BOOLEAN NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_benchmark(self, benchmark: PerformanceBenchmark):
        """Record a benchmark result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_history 
            (test_name, timestamp, min_time, max_time, mean_time, median_time, 
             stddev_time, rounds, iterations, git_commit, branch, environment, 
             additional_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            benchmark.test_name,
            benchmark.timestamp.isoformat(),
            benchmark.min_time,
            benchmark.max_time,
            benchmark.mean_time,
            benchmark.median_time,
            benchmark.stddev_time,
            benchmark.rounds,
            benchmark.iterations,
            benchmark.git_commit,
            benchmark.branch,
            benchmark.environment,
            json.dumps(benchmark.additional_metrics)
        ))
        
        conn.commit()
        conn.close()
        
        # Update baseline if needed
        self._update_baseline(benchmark)
        
        logger.info("Benchmark recorded", 
                   test_name=benchmark.test_name,
                   mean_time=benchmark.mean_time)
    
    def _update_baseline(self, benchmark: PerformanceBenchmark):
        """Update baseline performance if this is a stable/production run"""
        if benchmark.branch not in ['main', 'master', 'production']:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent performance data for this test
        cursor.execute("""
            SELECT mean_time FROM performance_history 
            WHERE test_name = ? AND branch IN ('main', 'master', 'production')
            ORDER BY timestamp DESC 
            LIMIT 20
        """, (benchmark.test_name,))
        
        recent_times = [row[0] for row in cursor.fetchall()]
        
        if len(recent_times) >= 5:
            baseline_mean = np.mean(recent_times)
            baseline_stddev = np.std(recent_times)
            
            # Update or insert baseline
            cursor.execute("""
                INSERT OR REPLACE INTO performance_baselines 
                (test_name, baseline_mean, baseline_stddev, sample_count, 
                 last_updated, git_commit, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark.test_name,
                baseline_mean,
                baseline_stddev,
                len(recent_times),
                datetime.now().isoformat(),
                benchmark.git_commit,
                benchmark.branch
            ))
            
            # Update cache
            self.baseline_cache[benchmark.test_name] = {
                'mean': baseline_mean,
                'stddev': baseline_stddev,
                'sample_count': len(recent_times)
            }
            
            logger.info("Baseline updated", 
                       test_name=benchmark.test_name,
                       baseline_mean=baseline_mean,
                       sample_count=len(recent_times))
        
        conn.commit()
        conn.close()
    
    def detect_regression(self, benchmark: PerformanceBenchmark) -> Optional[RegressionResult]:
        """Detect performance regression using statistical analysis"""
        baseline = self._get_baseline(benchmark.test_name)
        if not baseline:
            return None
        
        # Get performance budget
        budget = self.performance_budgets.get(benchmark.test_name)
        if budget and not budget.enabled:
            return None
        
        # Statistical significance test
        current_mean = benchmark.mean_time
        baseline_mean = baseline['mean']
        baseline_stddev = baseline['stddev']
        sample_count = baseline['sample_count']
        
        # Calculate z-score
        standard_error = baseline_stddev / np.sqrt(sample_count)
        z_score = (current_mean - baseline_mean) / standard_error
        p_value = 1 - stats.norm.cdf(abs(z_score))
        
        # Calculate confidence interval
        confidence_level = 0.95
        z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin_of_error = z_critical * standard_error
        confidence_interval = (
            baseline_mean - margin_of_error,
            baseline_mean + margin_of_error
        )
        
        # Determine regression severity
        regression_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        regression_detected = False
        severity = "NONE"
        
        if p_value < self.significance_threshold and current_mean > baseline_mean:
            regression_detected = True
            
            if budget:
                if regression_percent > budget.max_regression_percent:
                    severity = "CRITICAL"
                elif regression_percent > budget.max_regression_percent * 0.75:
                    severity = "HIGH"
                elif regression_percent > budget.max_regression_percent * 0.5:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
            else:
                # Default thresholds
                if regression_percent > 50:
                    severity = "CRITICAL"
                elif regression_percent > 25:
                    severity = "HIGH"
                elif regression_percent > 10:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
        
        # Trend analysis
        trend_direction = self._analyze_trend(benchmark.test_name)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            benchmark.test_name, regression_percent, severity, trend_direction
        )
        
        result = RegressionResult(
            test_name=benchmark.test_name,
            current_performance=current_mean,
            baseline_performance=baseline_mean,
            regression_detected=regression_detected,
            regression_severity=severity,
            statistical_significance=p_value,
            confidence_interval=confidence_interval,
            trend_direction=trend_direction,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        
        if regression_detected:
            self._record_regression_alert(result)
        
        return result
    
    def _get_baseline(self, test_name: str) -> Optional[Dict]:
        """Get baseline performance for a test"""
        # Check cache first
        if test_name in self.baseline_cache:
            return self.baseline_cache[test_name]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT baseline_mean, baseline_stddev, sample_count 
            FROM performance_baselines 
            WHERE test_name = ?
        """, (test_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            baseline = {
                'mean': result[0],
                'stddev': result[1],
                'sample_count': result[2]
            }
            self.baseline_cache[test_name] = baseline
            return baseline
        
        return None
    
    def _analyze_trend(self, test_name: str) -> str:
        """Analyze performance trend over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, mean_time FROM performance_history 
            WHERE test_name = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (test_name, self.trend_analysis_window))
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < 3:
            return "STABLE"
        
        # Convert to arrays for trend analysis
        times = []
        values = []
        
        for timestamp_str, mean_time in results:
            timestamp = datetime.fromisoformat(timestamp_str)
            times.append(timestamp.timestamp())
            values.append(mean_time)
        
        # Reverse to get chronological order
        times.reverse()
        values.reverse()
        
        # Linear regression for trend
        X = np.array(times).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        
        # Determine trend direction
        if abs(slope) < 1e-6:  # Very small slope
            return "STABLE"
        elif slope > 0:
            return "DEGRADING"
        else:
            return "IMPROVING"
    
    def _generate_recommendation(self, test_name: str, regression_percent: float, 
                               severity: str, trend_direction: str) -> str:
        """Generate optimization recommendation"""
        recommendations = []
        
        if severity == "CRITICAL":
            recommendations.append("IMMEDIATE ACTION REQUIRED: Performance has degraded significantly")
        elif severity == "HIGH":
            recommendations.append("High priority: Performance regression detected")
        elif severity == "MEDIUM":
            recommendations.append("Medium priority: Monitor performance closely")
        
        if trend_direction == "DEGRADING":
            recommendations.append("Performance is consistently degrading over time")
        elif trend_direction == "IMPROVING":
            recommendations.append("Performance is improving over time")
        
        if regression_percent > 100:
            recommendations.append("Consider algorithmic optimization - performance is >100% slower")
        elif regression_percent > 50:
            recommendations.append("Review recent changes and optimize critical paths")
        elif regression_percent > 25:
            recommendations.append("Investigate recent changes that may impact performance")
        
        return "; ".join(recommendations) if recommendations else "Monitor performance"
    
    def _record_regression_alert(self, result: RegressionResult):
        """Record regression alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO regression_alerts 
            (test_name, timestamp, current_performance, baseline_performance, 
             regression_severity, statistical_significance, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_name,
            result.timestamp.isoformat(),
            result.current_performance,
            result.baseline_performance,
            result.regression_severity,
            result.statistical_significance,
            result.recommendation
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning("Performance regression detected",
                      test_name=result.test_name,
                      severity=result.regression_severity,
                      current_performance=result.current_performance,
                      baseline_performance=result.baseline_performance)
    
    def set_performance_budget(self, budget: PerformanceBudget):
        """Set performance budget for a test"""
        self.performance_budgets[budget.test_name] = budget
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO performance_budgets 
            (test_name, max_time_ms, max_regression_percent, min_samples_for_baseline,
             significance_threshold, enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            budget.test_name,
            budget.max_time_ms,
            budget.max_regression_percent,
            budget.min_samples_for_baseline,
            budget.significance_threshold,
            budget.enabled
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("Performance budget set", 
                   test_name=budget.test_name,
                   max_time_ms=budget.max_time_ms,
                   max_regression_percent=budget.max_regression_percent)
    
    def get_performance_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent benchmarks
        cursor.execute("""
            SELECT test_name, timestamp, mean_time, branch 
            FROM performance_history 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        """, (cutoff_time.isoformat(),))
        
        recent_benchmarks = cursor.fetchall()
        
        # Get recent alerts
        cursor.execute("""
            SELECT test_name, timestamp, regression_severity, current_performance, 
                   baseline_performance, recommendation 
            FROM regression_alerts 
            WHERE timestamp >= ? AND resolved = FALSE
            ORDER BY timestamp DESC
        """, (cutoff_time.isoformat(),))
        
        recent_alerts = cursor.fetchall()
        
        # Get performance trends
        cursor.execute("""
            SELECT test_name, COUNT(*) as run_count, AVG(mean_time) as avg_time,
                   MIN(mean_time) as min_time, MAX(mean_time) as max_time
            FROM performance_history 
            WHERE timestamp >= ?
            GROUP BY test_name
        """, (cutoff_time.isoformat(),))
        
        performance_summary = cursor.fetchall()
        
        conn.close()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'summary': {
                'total_benchmark_runs': len(recent_benchmarks),
                'total_alerts': len(recent_alerts),
                'tests_analyzed': len(performance_summary)
            },
            'recent_benchmarks': [
                {
                    'test_name': row[0],
                    'timestamp': row[1],
                    'mean_time': row[2],
                    'branch': row[3]
                }
                for row in recent_benchmarks
            ],
            'active_alerts': [
                {
                    'test_name': row[0],
                    'timestamp': row[1],
                    'severity': row[2],
                    'current_performance': row[3],
                    'baseline_performance': row[4],
                    'recommendation': row[5]
                }
                for row in recent_alerts
            ],
            'performance_summary': [
                {
                    'test_name': row[0],
                    'run_count': row[1],
                    'avg_time': row[2],
                    'min_time': row[3],
                    'max_time': row[4]
                }
                for row in performance_summary
            ]
        }
    
    def predict_performance_trend(self, test_name: str, days_ahead: int = 7) -> Dict:
        """Predict future performance trend using linear regression"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get historical data
        cursor.execute("""
            SELECT timestamp, mean_time FROM performance_history 
            WHERE test_name = ? 
            ORDER BY timestamp ASC 
            LIMIT 100
        """, (test_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < 10:
            return {"error": "Insufficient data for trend prediction"}
        
        # Prepare data
        times = []
        values = []
        
        for timestamp_str, mean_time in results:
            timestamp = datetime.fromisoformat(timestamp_str)
            times.append(timestamp.timestamp())
            values.append(mean_time)
        
        # Train model
        X = np.array(times).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        future_times = []
        current_time = datetime.now().timestamp()
        
        for day in range(1, days_ahead + 1):
            future_time = current_time + (day * 24 * 60 * 60)
            future_times.append(future_time)
        
        future_X = np.array(future_times).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Calculate prediction confidence
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'test_name': test_name,
            'prediction_period_days': days_ahead,
            'current_trend': "IMPROVING" if model.coef_[0] < 0 else "DEGRADING",
            'trend_strength': abs(model.coef_[0]),
            'predictions': [
                {
                    'date': datetime.fromtimestamp(future_times[i]).isoformat(),
                    'predicted_time': float(predictions[i]),
                    'confidence_interval': [
                        float(predictions[i] - rmse * 1.96),
                        float(predictions[i] + rmse * 1.96)
                    ]
                }
                for i in range(len(predictions))
            ],
            'model_accuracy': {
                'rmse': float(rmse),
                'r_squared': float(model.score(X, y))
            }
        }


# Global instance
performance_detector = PerformanceRegressionDetector()