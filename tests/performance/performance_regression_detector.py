"""
Performance Regression Detection Framework - Agent 4 Implementation
==============================================================

Continuous performance monitoring and regression detection system for the GrandModel.
Automatically detects performance degradation and provides alerts for production systems.

Features:
- Historical performance baseline tracking
- Statistical regression detection
- Automated alerting system
- Performance trend analysis
- Git commit correlation
- Automated performance reports

Author: Agent 4 - Performance Baseline Research Agent
"""

import json
import time
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import statistics
import subprocess
import hashlib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    test_context: str
    commit_hash: str
    environment: str
    tags: Dict[str, str]


@dataclass
class RegressionAlert:
    """Performance regression alert."""
    alert_id: str
    metric_name: str
    severity: str  # 'warning', 'critical', 'severe'
    current_value: float
    baseline_value: float
    regression_percent: float
    confidence_level: float
    timestamp: datetime
    context: Dict[str, Any]
    recommendations: List[str]


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    baseline_std: float
    sample_size: int
    last_updated: datetime
    baseline_window_days: int
    confidence_interval: Tuple[float, float]


class PerformanceRegressionDetector:
    """
    Performance regression detection system with statistical analysis
    and automated alerting capabilities.
    """
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        """Initialize performance regression detector."""
        self.db_path = Path(db_path)
        self.db_connection = None
        self.baselines = {}
        self.alerts = []
        
        # Configuration
        self.config = {
            "baseline_window_days": 30,
            "min_samples_for_baseline": 10,
            "regression_threshold_warning": 15.0,  # 15% degradation
            "regression_threshold_critical": 25.0,  # 25% degradation
            "regression_threshold_severe": 40.0,   # 40% degradation
            "confidence_level": 0.95,
            "trend_analysis_window": 7,  # days
            "alert_cooldown_minutes": 60
        }
        
        # Initialize database
        self._initialize_database()
        
        # Load existing baselines
        self._load_baselines()
        
        logger.info("Performance regression detector initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for performance metrics."""
        self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.db_connection.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                test_context TEXT,
                commit_hash TEXT,
                environment TEXT,
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT UNIQUE NOT NULL,
                baseline_value REAL NOT NULL,
                baseline_std REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                last_updated DATETIME NOT NULL,
                baseline_window_days INTEGER NOT NULL,
                confidence_interval_lower REAL NOT NULL,
                confidence_interval_upper REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                metric_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                regression_percent REAL NOT NULL,
                confidence_level REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                context TEXT,
                recommendations TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON performance_metrics(metric_name, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_commit ON performance_metrics(commit_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON regression_alerts(timestamp)')
        
        self.db_connection.commit()
        logger.info("Database initialized successfully")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric measurement."""
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (metric_name, value, unit, timestamp, test_context, commit_hash, environment, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_name,
            metric.value,
            metric.unit,
            metric.timestamp.isoformat(),
            metric.test_context,
            metric.commit_hash,
            metric.environment,
            json.dumps(metric.tags)
        ))
        
        self.db_connection.commit()
        
        # Update baseline if needed
        self._update_baseline(metric.metric_name)
        
        # Check for regressions
        self._check_for_regression(metric)
        
        logger.debug(f"Recorded metric: {metric.metric_name} = {metric.value} {metric.unit}")
    
    def _update_baseline(self, metric_name: str):
        """Update baseline for a metric based on recent data."""
        cursor = self.db_connection.cursor()
        
        # Get recent data for baseline calculation
        cutoff_date = datetime.now() - timedelta(days=self.config["baseline_window_days"])
        
        cursor.execute('''
            SELECT value FROM performance_metrics
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (metric_name, cutoff_date.isoformat()))
        
        recent_values = [row[0] for row in cursor.fetchall()]
        
        if len(recent_values) >= self.config["min_samples_for_baseline"]:
            # Calculate baseline statistics
            baseline_value = np.mean(recent_values)
            baseline_std = np.std(recent_values)
            sample_size = len(recent_values)
            
            # Calculate confidence interval
            confidence_level = self.config["confidence_level"]
            se = baseline_std / np.sqrt(sample_size)
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * se
            
            ci_lower = baseline_value - margin_of_error
            ci_upper = baseline_value + margin_of_error
            
            # Update baseline in database
            cursor.execute('''
                INSERT OR REPLACE INTO performance_baselines
                (metric_name, baseline_value, baseline_std, sample_size, last_updated, 
                 baseline_window_days, confidence_interval_lower, confidence_interval_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric_name,
                baseline_value,
                baseline_std,
                sample_size,
                datetime.now().isoformat(),
                self.config["baseline_window_days"],
                ci_lower,
                ci_upper
            ))
            
            self.db_connection.commit()
            
            # Update in-memory baseline
            self.baselines[metric_name] = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=baseline_value,
                baseline_std=baseline_std,
                sample_size=sample_size,
                last_updated=datetime.now(),
                baseline_window_days=self.config["baseline_window_days"],
                confidence_interval=(ci_lower, ci_upper)
            )
            
            logger.debug(f"Updated baseline for {metric_name}: {baseline_value:.2f} ± {baseline_std:.2f}")
    
    def _load_baselines(self):
        """Load existing baselines from database."""
        cursor = self.db_connection.cursor()
        
        cursor.execute('SELECT * FROM performance_baselines')
        
        for row in cursor.fetchall():
            baseline = PerformanceBaseline(
                metric_name=row[1],
                baseline_value=row[2],
                baseline_std=row[3],
                sample_size=row[4],
                last_updated=datetime.fromisoformat(row[5]),
                baseline_window_days=row[6],
                confidence_interval=(row[7], row[8])
            )
            self.baselines[baseline.metric_name] = baseline
        
        logger.info(f"Loaded {len(self.baselines)} performance baselines")
    
    def _check_for_regression(self, metric: PerformanceMetric):
        """Check if a metric measurement indicates a performance regression."""
        if metric.metric_name not in self.baselines:
            return  # No baseline available yet
        
        baseline = self.baselines[metric.metric_name]
        
        # Calculate regression percentage
        if baseline.baseline_value == 0:
            return  # Avoid division by zero
        
        # For latency metrics, higher values are worse
        # For throughput metrics, lower values are worse
        if "latency" in metric.metric_name.lower() or "time" in metric.metric_name.lower():
            regression_percent = ((metric.value - baseline.baseline_value) / baseline.baseline_value) * 100
        else:  # throughput, ops/sec, etc.
            regression_percent = ((baseline.baseline_value - metric.value) / baseline.baseline_value) * 100
        
        # Determine severity
        severity = None
        if regression_percent >= self.config["regression_threshold_severe"]:
            severity = "severe"
        elif regression_percent >= self.config["regression_threshold_critical"]:
            severity = "critical"
        elif regression_percent >= self.config["regression_threshold_warning"]:
            severity = "warning"
        
        if severity:
            # Calculate statistical confidence
            z_score = (metric.value - baseline.baseline_value) / baseline.baseline_std
            confidence_level = stats.norm.cdf(abs(z_score))
            
            # Check alert cooldown
            if self._is_alert_cooldown_active(metric.metric_name):
                return
            
            # Generate alert
            alert = RegressionAlert(
                alert_id=self._generate_alert_id(metric.metric_name, metric.timestamp),
                metric_name=metric.metric_name,
                severity=severity,
                current_value=metric.value,
                baseline_value=baseline.baseline_value,
                regression_percent=regression_percent,
                confidence_level=confidence_level,
                timestamp=metric.timestamp,
                context={
                    "test_context": metric.test_context,
                    "commit_hash": metric.commit_hash,
                    "environment": metric.environment,
                    "tags": metric.tags,
                    "baseline_sample_size": baseline.sample_size,
                    "baseline_window_days": baseline.baseline_window_days
                },
                recommendations=self._generate_regression_recommendations(metric, baseline, regression_percent)
            )
            
            self._process_alert(alert)
    
    def _is_alert_cooldown_active(self, metric_name: str) -> bool:
        """Check if alert cooldown is active for a metric."""
        cursor = self.db_connection.cursor()
        
        cooldown_time = datetime.now() - timedelta(minutes=self.config["alert_cooldown_minutes"])
        
        cursor.execute('''
            SELECT COUNT(*) FROM regression_alerts
            WHERE metric_name = ? AND timestamp > ? AND resolved = FALSE
        ''', (metric_name, cooldown_time.isoformat()))
        
        return cursor.fetchone()[0] > 0
    
    def _generate_alert_id(self, metric_name: str, timestamp: datetime) -> str:
        """Generate unique alert ID."""
        content = f"{metric_name}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _generate_regression_recommendations(self, metric: PerformanceMetric, 
                                           baseline: PerformanceBaseline,
                                           regression_percent: float) -> List[str]:
        """Generate recommendations for addressing performance regression."""
        recommendations = []
        
        # General recommendations
        if regression_percent > 30:
            recommendations.append("Investigate recent code changes for performance impact")
            recommendations.append("Review system resource utilization")
        
        # Metric-specific recommendations
        if "latency" in metric.metric_name.lower():
            recommendations.append("Profile application for bottlenecks")
            recommendations.append("Check for increased I/O operations")
            recommendations.append("Review database query performance")
        
        elif "throughput" in metric.metric_name.lower():
            recommendations.append("Analyze system capacity limits")
            recommendations.append("Check for resource contention")
            recommendations.append("Review concurrent processing efficiency")
        
        elif "memory" in metric.metric_name.lower():
            recommendations.append("Check for memory leaks")
            recommendations.append("Review garbage collection patterns")
            recommendations.append("Analyze memory allocation patterns")
        
        # Environment-specific recommendations
        if metric.environment == "production":
            recommendations.append("Consider immediate rollback if critical")
            recommendations.append("Enable enhanced monitoring")
        
        # Commit-specific recommendations
        if metric.commit_hash:
            recommendations.append(f"Review changes in commit {metric.commit_hash[:7]}")
        
        return recommendations
    
    def _process_alert(self, alert: RegressionAlert):
        """Process and store performance regression alert."""
        cursor = self.db_connection.cursor()
        
        # Store alert in database
        cursor.execute('''
            INSERT INTO regression_alerts
            (alert_id, metric_name, severity, current_value, baseline_value, 
             regression_percent, confidence_level, timestamp, context, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.metric_name,
            alert.severity,
            alert.current_value,
            alert.baseline_value,
            alert.regression_percent,
            alert.confidence_level,
            alert.timestamp.isoformat(),
            json.dumps(alert.context),
            json.dumps(alert.recommendations)
        ))
        
        self.db_connection.commit()
        
        # Store in memory
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"PERFORMANCE REGRESSION DETECTED: {alert.metric_name}")
        logger.warning(f"  Severity: {alert.severity.upper()}")
        logger.warning(f"  Current: {alert.current_value:.2f}")
        logger.warning(f"  Baseline: {alert.baseline_value:.2f}")
        logger.warning(f"  Regression: {alert.regression_percent:.1f}%")
        logger.warning(f"  Confidence: {alert.confidence_level:.1%}")
        
        # Send notifications
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: RegressionAlert):
        """Send alert notification (placeholder for actual notification system)."""
        # In a real implementation, this would send emails, Slack messages, etc.
        logger.info(f"Alert notification sent for {alert.metric_name} ({alert.severity})")
        
        # Create alert report
        alert_report = {
            "alert_id": alert.alert_id,
            "metric_name": alert.metric_name,
            "severity": alert.severity,
            "current_value": alert.current_value,
            "baseline_value": alert.baseline_value,
            "regression_percent": alert.regression_percent,
            "confidence_level": alert.confidence_level,
            "timestamp": alert.timestamp.isoformat(),
            "context": alert.context,
            "recommendations": alert.recommendations
        }
        
        # Save alert report
        alert_file = Path(f"alert_{alert.alert_id}.json")
        with open(alert_file, 'w') as f:
            json.dump(alert_report, f, indent=2)
        
        logger.info(f"Alert report saved to: {alert_file}")
    
    def get_performance_trend(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance trend analysis for a metric."""
        cursor = self.db_connection.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT value, timestamp FROM performance_metrics
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (metric_name, start_date.isoformat()))
        
        data = cursor.fetchall()
        
        if len(data) < 2:
            return {"trend": "insufficient_data", "data_points": len(data)}
        
        values = [row[0] for row in data]
        timestamps = [datetime.fromisoformat(row[1]) for row in data]
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend = "increasing" if "throughput" in metric_name.lower() else "degrading"
            else:
                trend = "decreasing" if "throughput" in metric_name.lower() else "improving"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "data_points": len(values),
            "recent_average": np.mean(values[-5:]) if len(values) >= 5 else np.mean(values),
            "trend_strength": abs(r_value),
            "statistical_significance": p_value < 0.05
        }
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cursor = self.db_connection.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all metrics in the time period
        cursor.execute('''
            SELECT DISTINCT metric_name FROM performance_metrics
            WHERE timestamp > ?
        ''', (start_date.isoformat(),))
        
        metrics = [row[0] for row in cursor.fetchall()]
        
        # Get alerts in the time period
        cursor.execute('''
            SELECT * FROM regression_alerts
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (start_date.isoformat(),))
        
        alerts = cursor.fetchall()
        
        # Analyze each metric
        metric_analysis = {}
        for metric_name in metrics:
            trend = self.get_performance_trend(metric_name, days)
            
            # Get recent values
            cursor.execute('''
                SELECT value, timestamp FROM performance_metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (metric_name, start_date.isoformat()))
            
            recent_data = cursor.fetchall()
            recent_values = [row[0] for row in recent_data]
            
            if recent_values:
                metric_analysis[metric_name] = {
                    "trend": trend,
                    "recent_average": np.mean(recent_values),
                    "recent_std": np.std(recent_values),
                    "min_value": min(recent_values),
                    "max_value": max(recent_values),
                    "data_points": len(recent_values),
                    "baseline": self.baselines.get(metric_name)
                }
        
        # Alert analysis
        alert_analysis = {
            "total_alerts": len(alerts),
            "alerts_by_severity": {
                "warning": len([a for a in alerts if a[3] == "warning"]),
                "critical": len([a for a in alerts if a[3] == "critical"]),
                "severe": len([a for a in alerts if a[3] == "severe"])
            },
            "most_problematic_metrics": self._get_most_problematic_metrics(alerts),
            "recent_alerts": [
                {
                    "alert_id": alert[1],
                    "metric_name": alert[2],
                    "severity": alert[3],
                    "regression_percent": alert[6],
                    "timestamp": alert[8]
                }
                for alert in alerts[:5]  # Last 5 alerts
            ]
        }
        
        # Overall system health
        health_score = self._calculate_health_score(metric_analysis, alert_analysis)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "system_health": {
                "health_score": health_score,
                "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical",
                "total_metrics": len(metrics),
                "metrics_with_trends": len([m for m in metric_analysis.values() if m["trend"]["trend"] != "stable"])
            },
            "metric_analysis": metric_analysis,
            "alert_analysis": alert_analysis,
            "recommendations": self._generate_system_recommendations(metric_analysis, alert_analysis),
            "generated_at": datetime.now().isoformat()
        }
    
    def _get_most_problematic_metrics(self, alerts: List[Tuple]) -> List[str]:
        """Get metrics with the most alerts."""
        metric_alert_counts = {}
        
        for alert in alerts:
            metric_name = alert[2]
            if metric_name not in metric_alert_counts:
                metric_alert_counts[metric_name] = 0
            metric_alert_counts[metric_name] += 1
        
        # Sort by alert count
        sorted_metrics = sorted(metric_alert_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [metric for metric, count in sorted_metrics[:5]]
    
    def _calculate_health_score(self, metric_analysis: Dict[str, Any], 
                              alert_analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        base_score = 100.0
        
        # Deduct points for alerts
        alert_penalties = {
            "warning": 5,
            "critical": 15,
            "severe": 25
        }
        
        for severity, count in alert_analysis["alerts_by_severity"].items():
            base_score -= count * alert_penalties[severity]
        
        # Deduct points for degrading trends
        degrading_trends = 0
        for metric_name, analysis in metric_analysis.items():
            if analysis["trend"]["trend"] in ["degrading", "decreasing"]:
                degrading_trends += 1
        
        base_score -= degrading_trends * 3
        
        # Bonus for stable metrics
        stable_metrics = 0
        for metric_name, analysis in metric_analysis.items():
            if analysis["trend"]["trend"] == "stable":
                stable_metrics += 1
        
        base_score += stable_metrics * 1
        
        return max(0, min(100, base_score))
    
    def _generate_system_recommendations(self, metric_analysis: Dict[str, Any],
                                       alert_analysis: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []
        
        # Alert-based recommendations
        if alert_analysis["total_alerts"] > 10:
            recommendations.append("High number of performance alerts - investigate system stability")
        
        if alert_analysis["alerts_by_severity"]["severe"] > 0:
            recommendations.append("Severe performance regressions detected - immediate action required")
        
        # Trend-based recommendations
        degrading_metrics = [
            name for name, analysis in metric_analysis.items()
            if analysis["trend"]["trend"] in ["degrading", "decreasing"]
        ]
        
        if len(degrading_metrics) > len(metric_analysis) * 0.3:
            recommendations.append("Multiple metrics showing degrading trends - system health declining")
        
        # Specific metric recommendations
        if any("latency" in name for name in degrading_metrics):
            recommendations.append("Latency metrics degrading - optimize response times")
        
        if any("throughput" in name for name in degrading_metrics):
            recommendations.append("Throughput metrics degrading - scale system capacity")
        
        if any("memory" in name for name in degrading_metrics):
            recommendations.append("Memory metrics degrading - investigate memory leaks")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save performance report to file."""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_file = Path(filename)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {report_file}")
        return report_file
    
    def close(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = PerformanceRegressionDetector()
    
    # Simulate some performance metrics
    print("🚀 Performance Regression Detection Framework Demo")
    print("=" * 60)
    
    # Simulate normal performance
    current_time = datetime.now()
    commit_hash = "abc123ef"
    
    # Strategic MARL metrics
    for i in range(50):
        metric = PerformanceMetric(
            metric_name="strategic_marl_p99_latency_ms",
            value=1.5 + np.random.normal(0, 0.2),  # Normal around 1.5ms
            unit="milliseconds",
            timestamp=current_time - timedelta(minutes=i * 5),
            test_context="strategic_marl_benchmark",
            commit_hash=commit_hash,
            environment="production",
            tags={"component": "strategic_marl", "test_type": "benchmark"}
        )
        detector.record_metric(metric)
    
    # Simulate performance regression
    print("\\n⚠️  Simulating performance regression...")
    
    for i in range(5):
        metric = PerformanceMetric(
            metric_name="strategic_marl_p99_latency_ms",
            value=2.5 + np.random.normal(0, 0.1),  # Degraded to 2.5ms
            unit="milliseconds",
            timestamp=current_time + timedelta(minutes=i * 2),
            test_context="strategic_marl_benchmark",
            commit_hash="def456gh",
            environment="production",
            tags={"component": "strategic_marl", "test_type": "benchmark"}
        )
        detector.record_metric(metric)
    
    # Generate performance report
    print("\\n📊 Generating performance report...")
    report = detector.generate_performance_report(days=1)
    
    # Display key findings
    print("\\n🎯 PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"System Health Score: {report['system_health']['health_score']:.1f}/100")
    print(f"System Status: {report['system_health']['status'].upper()}")
    print(f"Total Alerts: {report['alert_analysis']['total_alerts']}")
    print(f"Critical Alerts: {report['alert_analysis']['alerts_by_severity']['critical']}")
    print(f"Metrics Monitored: {report['system_health']['total_metrics']}")
    
    # Show recent alerts
    if report['alert_analysis']['recent_alerts']:
        print("\\n🚨 RECENT ALERTS")
        print("=" * 50)
        for alert in report['alert_analysis']['recent_alerts']:
            print(f"• {alert['metric_name']} - {alert['severity'].upper()}")
            print(f"  Regression: {alert['regression_percent']:.1f}%")
            print(f"  Time: {alert['timestamp']}")
            print()
    
    # Show recommendations
    if report['recommendations']:
        print("\\n🔧 SYSTEM RECOMMENDATIONS")
        print("=" * 50)
        for rec in report['recommendations']:
            print(f"• {rec}")
    
    # Save report
    report_file = detector.save_report(report)
    print(f"\\n📄 Full report saved to: {report_file}")
    
    # Close detector
    detector.close()
    
    print("\\n✅ Performance Regression Detection Framework Demo Complete!")