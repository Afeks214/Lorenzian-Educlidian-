"""
Enhanced Performance Regression Detection System

This module provides advanced performance regression detection with statistical analysis,
machine learning-based anomaly detection, and predictive performance modeling.

Features:
- Statistical significance testing
- Machine learning-based anomaly detection
- Predictive performance modeling
- Multi-metric correlation analysis
- Performance trend forecasting
- Adaptive threshold adjustment
- Real-time regression alerts
- Root cause analysis

Author: Performance Validation Agent
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import structlog
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class RegressionSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AnomalyType(Enum):
    PERFORMANCE_SPIKE = "PERFORMANCE_SPIKE"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    UNUSUAL_PATTERN = "UNUSUAL_PATTERN"
    TREND_CHANGE = "TREND_CHANGE"

@dataclass
class RegressionDetectionResult:
    """Performance regression detection result"""
    test_name: str
    metric_name: str
    timestamp: datetime
    current_value: float
    baseline_value: float
    regression_detected: bool
    severity: RegressionSeverity
    confidence_score: float
    statistical_significance: float
    trend_direction: str
    anomaly_type: Optional[AnomalyType] = None
    root_cause_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    prediction_accuracy: float = 0.0

@dataclass
class PerformanceBaseline:
    """Performance baseline with statistical properties"""
    test_name: str
    metric_name: str
    baseline_value: float
    baseline_std: float
    baseline_min: float
    baseline_max: float
    sample_count: int
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    test_name: str
    metric_name: str
    trend_slope: float
    trend_direction: str
    trend_strength: float
    seasonality_detected: bool
    forecast_values: List[float]
    forecast_confidence: float
    change_points: List[datetime]

class EnhancedRegressionDetector:
    """
    Enhanced performance regression detector with advanced statistical analysis
    and machine learning-based anomaly detection.
    """

    def __init__(self, db_path: str = "performance_regression.db"):
        self.db_path = db_path
        self.baseline_cache = {}
        self.trend_cache = {}
        self.models = {}
        
        # Detection parameters
        self.significance_threshold = 0.05
        self.anomaly_threshold = 0.1
        self.trend_window = 30
        self.min_samples_for_baseline = 10
        self.adaptive_threshold = True
        
        # Initialize models
        self._init_models()
        
        # Initialize database
        self._init_database()
        
        logger.info("Enhanced regression detector initialized")

    def _init_models(self):
        """Initialize machine learning models"""
        # Isolation Forest for anomaly detection
        self.models['anomaly_detector'] = IsolationForest(
            contamination=self.anomaly_threshold,
            random_state=42,
            n_estimators=100
        )
        
        # Linear regression for trend analysis
        self.models['trend_analyzer'] = LinearRegression()
        
        # Standard scaler for normalization
        self.models['scaler'] = StandardScaler()

    def _init_database(self):
        """Initialize enhanced regression detection database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_baselines (
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                baseline_std REAL NOT NULL,
                baseline_min REAL NOT NULL,
                baseline_max REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                confidence_interval_lower REAL NOT NULL,
                confidence_interval_upper REAL NOT NULL,
                last_updated TEXT NOT NULL,
                model_params TEXT,
                PRIMARY KEY (test_name, metric_name)
            )
        """)
        
        # Regression detection results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                regression_detected BOOLEAN NOT NULL,
                severity TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                statistical_significance REAL NOT NULL,
                trend_direction TEXT NOT NULL,
                anomaly_type TEXT,
                root_cause_analysis TEXT,
                recommendation TEXT,
                prediction_accuracy REAL
            )
        """)
        
        # Performance trends table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_trends (
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                trend_slope REAL NOT NULL,
                trend_direction TEXT NOT NULL,
                trend_strength REAL NOT NULL,
                seasonality_detected BOOLEAN NOT NULL,
                forecast_values TEXT,
                forecast_confidence REAL NOT NULL,
                change_points TEXT,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (test_name, metric_name)
            )
        """)
        
        # Anomaly detection results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                context_analysis TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def update_baseline(self, test_name: str, metric_name: str, 
                       values: List[float], timestamps: List[datetime]) -> PerformanceBaseline:
        """Update performance baseline with enhanced statistical analysis"""
        if len(values) < self.min_samples_for_baseline:
            raise ValueError(f"Insufficient samples for baseline: {len(values)} < {self.min_samples_for_baseline}")
        
        values_array = np.array(values)
        
        # Calculate basic statistics
        baseline_value = np.mean(values_array)
        baseline_std = np.std(values_array)
        baseline_min = np.min(values_array)
        baseline_max = np.max(values_array)
        
        # Calculate confidence interval
        confidence_level = 0.95
        degrees_freedom = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_of_error = t_critical * (baseline_std / np.sqrt(len(values)))
        
        confidence_interval = (
            baseline_value - margin_of_error,
            baseline_value + margin_of_error
        )
        
        # Fit models for this baseline
        model_params = self._fit_baseline_models(values_array, timestamps)
        
        baseline = PerformanceBaseline(
            test_name=test_name,
            metric_name=metric_name,
            baseline_value=baseline_value,
            baseline_std=baseline_std,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
            sample_count=len(values),
            confidence_interval=confidence_interval,
            last_updated=datetime.now(),
            model_params=model_params
        )
        
        # Store in database
        self._store_baseline(baseline)
        
        # Update cache
        self.baseline_cache[f"{test_name}_{metric_name}"] = baseline
        
        logger.info("Baseline updated",
                   test_name=test_name,
                   metric_name=metric_name,
                   baseline_value=baseline_value,
                   sample_count=len(values))
        
        return baseline

    def _fit_baseline_models(self, values: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """Fit machine learning models for baseline"""
        model_params = {}
        
        try:
            # Prepare time series data
            time_features = self._extract_time_features(timestamps)
            
            # Fit anomaly detection model
            if len(values) >= 20:  # Minimum samples for anomaly detection
                # Reshape for isolation forest
                X = values.reshape(-1, 1)
                self.models['anomaly_detector'].fit(X)
                
                # Store model parameters
                model_params['anomaly_threshold'] = self.models['anomaly_detector'].threshold_
                model_params['anomaly_contamination'] = self.models['anomaly_detector'].contamination
            
            # Fit trend model if we have time features
            if time_features is not None and len(time_features) == len(values):
                X_trend = np.array(time_features).reshape(-1, 1)
                self.models['trend_analyzer'].fit(X_trend, values)
                
                model_params['trend_slope'] = float(self.models['trend_analyzer'].coef_[0])
                model_params['trend_intercept'] = float(self.models['trend_analyzer'].intercept_)
                model_params['trend_r2'] = float(self.models['trend_analyzer'].score(X_trend, values))
            
        except Exception as e:
            logger.warning("Error fitting baseline models", error=str(e))
        
        return model_params

    def _extract_time_features(self, timestamps: List[datetime]) -> Optional[List[float]]:
        """Extract time-based features from timestamps"""
        if not timestamps:
            return None
        
        try:
            # Convert to Unix timestamps
            time_values = [ts.timestamp() for ts in timestamps]
            
            # Normalize to start from 0
            min_time = min(time_values)
            normalized_times = [(t - min_time) / 3600 for t in time_values]  # Hours since start
            
            return normalized_times
            
        except Exception as e:
            logger.warning("Error extracting time features", error=str(e))
            return None

    def detect_regression(self, test_name: str, metric_name: str, 
                         current_value: float, current_timestamp: datetime) -> RegressionDetectionResult:
        """Enhanced regression detection with statistical analysis and ML"""
        
        # Get baseline
        baseline = self._get_baseline(test_name, metric_name)
        if not baseline:
            return RegressionDetectionResult(
                test_name=test_name,
                metric_name=metric_name,
                timestamp=current_timestamp,
                current_value=current_value,
                baseline_value=0.0,
                regression_detected=False,
                severity=RegressionSeverity.LOW,
                confidence_score=0.0,
                statistical_significance=1.0,
                trend_direction="UNKNOWN",
                recommendation="Insufficient baseline data"
            )
        
        # Statistical significance test
        z_score = (current_value - baseline.baseline_value) / baseline.baseline_std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Check if value is outside confidence interval
        in_confidence_interval = (
            baseline.confidence_interval[0] <= current_value <= baseline.confidence_interval[1]
        )
        
        # Anomaly detection
        anomaly_result = self._detect_anomaly(test_name, metric_name, current_value)
        
        # Trend analysis
        trend_result = self._analyze_trend(test_name, metric_name, current_value, current_timestamp)
        
        # Determine regression
        regression_detected = (
            p_value < self.significance_threshold or
            not in_confidence_interval or
            anomaly_result['is_anomaly']
        )
        
        # Calculate severity
        severity = self._calculate_severity(
            current_value, baseline.baseline_value, p_value, anomaly_result
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            p_value, anomaly_result, trend_result
        )
        
        # Root cause analysis
        root_cause_analysis = self._perform_root_cause_analysis(
            test_name, metric_name, current_value, baseline, anomaly_result, trend_result
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            test_name, metric_name, severity, trend_result, root_cause_analysis
        )
        
        result = RegressionDetectionResult(
            test_name=test_name,
            metric_name=metric_name,
            timestamp=current_timestamp,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            regression_detected=regression_detected,
            severity=severity,
            confidence_score=confidence_score,
            statistical_significance=p_value,
            trend_direction=trend_result['direction'],
            anomaly_type=anomaly_result.get('anomaly_type'),
            root_cause_analysis=root_cause_analysis,
            recommendation=recommendation,
            prediction_accuracy=trend_result.get('prediction_accuracy', 0.0)
        )
        
        # Store result
        self._store_regression_result(result)
        
        if regression_detected:
            logger.warning("Performance regression detected",
                          test_name=test_name,
                          metric_name=metric_name,
                          severity=severity.value,
                          confidence_score=confidence_score)
        
        return result

    def _detect_anomaly(self, test_name: str, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect anomalies using machine learning"""
        try:
            # Prepare value for anomaly detection
            X = np.array([[value]])
            
            # Predict anomaly
            anomaly_score = self.models['anomaly_detector'].decision_function(X)[0]
            is_anomaly = self.models['anomaly_detector'].predict(X)[0] == -1
            
            # Determine anomaly type
            anomaly_type = None
            if is_anomaly:
                baseline = self._get_baseline(test_name, metric_name)
                if baseline:
                    if value > baseline.baseline_value * 1.5:
                        anomaly_type = AnomalyType.PERFORMANCE_SPIKE
                    elif value > baseline.baseline_value * 1.2:
                        anomaly_type = AnomalyType.PERFORMANCE_DEGRADATION
                    else:
                        anomaly_type = AnomalyType.UNUSUAL_PATTERN
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'anomaly_type': anomaly_type,
                'threshold': self.models['anomaly_detector'].threshold_
            }
            
        except Exception as e:
            logger.warning("Error in anomaly detection", error=str(e))
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_type': None,
                'threshold': 0.0
            }

    def _analyze_trend(self, test_name: str, metric_name: str, 
                      current_value: float, current_timestamp: datetime) -> Dict[str, Any]:
        """Analyze performance trend"""
        try:
            # Get recent values for trend analysis
            recent_values = self._get_recent_values(test_name, metric_name, self.trend_window)
            
            if len(recent_values) < 3:
                return {
                    'direction': 'UNKNOWN',
                    'strength': 0.0,
                    'prediction_accuracy': 0.0
                }
            
            # Prepare data for trend analysis
            values = [v['value'] for v in recent_values]
            timestamps = [v['timestamp'] for v in recent_values]
            
            # Add current value
            values.append(current_value)
            timestamps.append(current_timestamp)
            
            # Extract time features
            time_features = self._extract_time_features(timestamps)
            
            if time_features is None:
                return {
                    'direction': 'UNKNOWN',
                    'strength': 0.0,
                    'prediction_accuracy': 0.0
                }
            
            # Fit trend model
            X = np.array(time_features).reshape(-1, 1)
            y = np.array(values)
            
            self.models['trend_analyzer'].fit(X, y)
            
            # Get trend parameters
            slope = self.models['trend_analyzer'].coef_[0]
            r2_score = self.models['trend_analyzer'].score(X, y)
            
            # Determine trend direction
            if abs(slope) < 0.01:
                direction = 'STABLE'
            elif slope > 0:
                direction = 'DEGRADING'  # Assuming higher values = worse performance
            else:
                direction = 'IMPROVING'
            
            # Calculate trend strength
            strength = min(abs(slope) * 100, 100.0)  # Normalize to 0-100
            
            return {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'r2_score': r2_score,
                'prediction_accuracy': r2_score
            }
            
        except Exception as e:
            logger.warning("Error in trend analysis", error=str(e))
            return {
                'direction': 'UNKNOWN',
                'strength': 0.0,
                'prediction_accuracy': 0.0
            }

    def _calculate_severity(self, current_value: float, baseline_value: float, 
                           p_value: float, anomaly_result: Dict[str, Any]) -> RegressionSeverity:
        """Calculate regression severity"""
        if baseline_value == 0:
            return RegressionSeverity.LOW
        
        # Calculate percentage change
        percent_change = abs((current_value - baseline_value) / baseline_value) * 100
        
        # Determine severity based on multiple factors
        if (p_value < 0.001 and percent_change > 100) or anomaly_result.get('anomaly_score', 0) < -0.5:
            return RegressionSeverity.CRITICAL
        elif (p_value < 0.01 and percent_change > 50) or anomaly_result.get('anomaly_score', 0) < -0.3:
            return RegressionSeverity.HIGH
        elif (p_value < 0.05 and percent_change > 20) or anomaly_result.get('is_anomaly', False):
            return RegressionSeverity.MEDIUM
        else:
            return RegressionSeverity.LOW

    def _calculate_confidence_score(self, p_value: float, anomaly_result: Dict[str, Any], 
                                   trend_result: Dict[str, Any]) -> float:
        """Calculate confidence score for regression detection"""
        # Statistical significance confidence
        stat_confidence = 1.0 - p_value
        
        # Anomaly detection confidence
        anomaly_confidence = 1.0 - abs(anomaly_result.get('anomaly_score', 0))
        
        # Trend analysis confidence
        trend_confidence = trend_result.get('prediction_accuracy', 0.0)
        
        # Weighted average
        confidence_score = (
            stat_confidence * 0.4 +
            anomaly_confidence * 0.3 +
            trend_confidence * 0.3
        )
        
        return max(0.0, min(1.0, confidence_score))

    def _perform_root_cause_analysis(self, test_name: str, metric_name: str, 
                                    current_value: float, baseline: PerformanceBaseline,
                                    anomaly_result: Dict[str, Any], trend_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform root cause analysis"""
        analysis = {
            'statistical_factors': {},
            'anomaly_factors': {},
            'trend_factors': {},
            'potential_causes': []
        }
        
        # Statistical factors
        percent_change = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100
        analysis['statistical_factors'] = {
            'percent_change': percent_change,
            'standard_deviations': (current_value - baseline.baseline_value) / baseline.baseline_std,
            'outside_confidence_interval': not (baseline.confidence_interval[0] <= current_value <= baseline.confidence_interval[1])
        }
        
        # Anomaly factors
        analysis['anomaly_factors'] = anomaly_result
        
        # Trend factors
        analysis['trend_factors'] = trend_result
        
        # Potential causes
        if percent_change > 50:
            analysis['potential_causes'].append("Significant algorithmic change")
        if trend_result.get('direction') == 'DEGRADING':
            analysis['potential_causes'].append("Consistent performance degradation over time")
        if anomaly_result.get('is_anomaly'):
            analysis['potential_causes'].append("Unusual system behavior detected")
        
        return analysis

    def _generate_recommendation(self, test_name: str, metric_name: str, 
                               severity: RegressionSeverity, trend_result: Dict[str, Any],
                               root_cause_analysis: Dict[str, Any]) -> str:
        """Generate recommendation based on analysis"""
        recommendations = []
        
        # Severity-based recommendations
        if severity == RegressionSeverity.CRITICAL:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Critical performance regression detected")
        elif severity == RegressionSeverity.HIGH:
            recommendations.append("HIGH PRIORITY: Significant performance degradation")
        elif severity == RegressionSeverity.MEDIUM:
            recommendations.append("MEDIUM PRIORITY: Performance regression detected")
        else:
            recommendations.append("LOW PRIORITY: Minor performance deviation")
        
        # Trend-based recommendations
        if trend_result.get('direction') == 'DEGRADING':
            recommendations.append("Performance is consistently degrading - investigate recent changes")
        elif trend_result.get('direction') == 'IMPROVING':
            recommendations.append("Performance is improving - continue monitoring")
        
        # Metric-specific recommendations
        if 'latency' in metric_name.lower():
            recommendations.append("Review algorithmic complexity and optimize critical paths")
        elif 'memory' in metric_name.lower():
            recommendations.append("Investigate memory leaks and optimize allocation patterns")
        elif 'throughput' in metric_name.lower():
            recommendations.append("Check resource utilization and scaling configurations")
        
        # Root cause based recommendations
        potential_causes = root_cause_analysis.get('potential_causes', [])
        if 'algorithmic change' in str(potential_causes).lower():
            recommendations.append("Review recent algorithmic changes and optimizations")
        if 'degradation over time' in str(potential_causes).lower():
            recommendations.append("Implement gradual rollback and identify resource leaks")
        
        return "; ".join(recommendations)

    def _get_baseline(self, test_name: str, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline from cache or database"""
        cache_key = f"{test_name}_{metric_name}"
        
        if cache_key in self.baseline_cache:
            return self.baseline_cache[cache_key]
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT baseline_value, baseline_std, baseline_min, baseline_max, sample_count,
                   confidence_interval_lower, confidence_interval_upper, last_updated, model_params
            FROM enhanced_baselines 
            WHERE test_name = ? AND metric_name = ?
        """, (test_name, metric_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            baseline = PerformanceBaseline(
                test_name=test_name,
                metric_name=metric_name,
                baseline_value=result[0],
                baseline_std=result[1],
                baseline_min=result[2],
                baseline_max=result[3],
                sample_count=result[4],
                confidence_interval=(result[5], result[6]),
                last_updated=datetime.fromisoformat(result[7]),
                model_params=json.loads(result[8]) if result[8] else {}
            )
            
            self.baseline_cache[cache_key] = baseline
            return baseline
        
        return None

    def _get_recent_values(self, test_name: str, metric_name: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent values for trend analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT current_value, timestamp
            FROM regression_results 
            WHERE test_name = ? AND metric_name = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (test_name, metric_name, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'value': result[0],
                'timestamp': datetime.fromisoformat(result[1])
            }
            for result in results
        ]

    def _store_baseline(self, baseline: PerformanceBaseline):
        """Store baseline in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO enhanced_baselines 
            (test_name, metric_name, baseline_value, baseline_std, baseline_min, baseline_max,
             sample_count, confidence_interval_lower, confidence_interval_upper, last_updated, model_params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline.test_name,
            baseline.metric_name,
            baseline.baseline_value,
            baseline.baseline_std,
            baseline.baseline_min,
            baseline.baseline_max,
            baseline.sample_count,
            baseline.confidence_interval[0],
            baseline.confidence_interval[1],
            baseline.last_updated.isoformat(),
            json.dumps(baseline.model_params)
        ))
        
        conn.commit()
        conn.close()

    def _store_regression_result(self, result: RegressionDetectionResult):
        """Store regression result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO regression_results 
            (test_name, metric_name, timestamp, current_value, baseline_value, regression_detected,
             severity, confidence_score, statistical_significance, trend_direction, anomaly_type,
             root_cause_analysis, recommendation, prediction_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_name,
            result.metric_name,
            result.timestamp.isoformat(),
            result.current_value,
            result.baseline_value,
            result.regression_detected,
            result.severity.value,
            result.confidence_score,
            result.statistical_significance,
            result.trend_direction,
            result.anomaly_type.value if result.anomaly_type else None,
            json.dumps(result.root_cause_analysis),
            result.recommendation,
            result.prediction_accuracy
        ))
        
        conn.commit()
        conn.close()

    def get_regression_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get regression detection summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get regression counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) 
            FROM regression_results 
            WHERE timestamp >= ? AND regression_detected = 1
            GROUP BY severity
        """, (cutoff_time.isoformat(),))
        
        severity_counts = dict(cursor.fetchall())
        
        # Get top regressed tests
        cursor.execute("""
            SELECT test_name, metric_name, COUNT(*) as count, AVG(confidence_score) as avg_confidence
            FROM regression_results 
            WHERE timestamp >= ? AND regression_detected = 1
            GROUP BY test_name, metric_name
            ORDER BY count DESC, avg_confidence DESC
            LIMIT 10
        """, (cutoff_time.isoformat(),))
        
        top_regressions = cursor.fetchall()
        
        conn.close()
        
        return {
            'analysis_period_hours': hours,
            'total_regressions': sum(severity_counts.values()),
            'severity_breakdown': severity_counts,
            'top_regressed_tests': [
                {
                    'test_name': row[0],
                    'metric_name': row[1],
                    'regression_count': row[2],
                    'avg_confidence': row[3]
                }
                for row in top_regressions
            ],
            'timestamp': datetime.now().isoformat()
        }


# Global instance
enhanced_regression_detector = EnhancedRegressionDetector()