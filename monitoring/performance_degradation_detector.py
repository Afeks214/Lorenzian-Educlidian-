#!/usr/bin/env python3
"""
Performance Degradation Detection and Automated Response System
Comprehensive performance monitoring with statistical detection algorithms,
automated model retraining triggers, and intelligent response mechanisms.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import sqlite3
import redis
from pathlib import Path

# Statistical analysis
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Monitoring imports
from .alerting_system import AlertManager, Alert, AlertType, AlertSeverity, AlertStatus
from .adaptive_threshold_monitor import AdaptiveThresholdMonitor
from .health_check_system import ComprehensiveHealthCheckSystem

# Metrics
from prometheus_client import Counter, Histogram, Gauge, Summary
DEGRADATION_DETECTIONS = Counter('performance_degradation_detections_total', 'Total degradation detections', ['component', 'metric', 'severity'])
RETRAINING_TRIGGERS = Counter('model_retraining_triggers_total', 'Total model retraining triggers', ['model_type', 'trigger_reason'])
ROLLBACK_EXECUTIONS = Counter('model_rollback_executions_total', 'Total model rollbacks', ['model_type', 'rollback_reason'])
PERFORMANCE_TREND_SCORE = Gauge('performance_trend_score', 'Performance trend score', ['component', 'metric'])
DEGRADATION_RESPONSE_TIME = Histogram('degradation_response_time_seconds', 'Response time for degradation handling', ['response_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DegradationSeverity(Enum):
    """Degradation severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

class DegradationType(Enum):
    """Types of performance degradation."""
    GRADUAL_DECLINE = "gradual_decline"
    SUDDEN_DROP = "sudden_drop"
    VARIANCE_INCREASE = "variance_increase"
    TREND_REVERSAL = "trend_reversal"
    OUTLIER_SURGE = "outlier_surge"
    CYCLICAL_DEGRADATION = "cyclical_degradation"

class ResponseAction(Enum):
    """Automated response actions."""
    ALERT_ONLY = "alert_only"
    INCREASE_MONITORING = "increase_monitoring"
    TRIGGER_RETRAINING = "trigger_retraining"
    ROLLBACK_MODEL = "rollback_model"
    SCALE_RESOURCES = "scale_resources"
    RESTART_SERVICE = "restart_service"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'metadata': self.metadata or {}
        }

@dataclass
class DegradationEvent:
    """Performance degradation event."""
    event_id: str
    degradation_type: DegradationType
    severity: DegradationSeverity
    affected_metrics: List[str]
    affected_components: List[str]
    detection_time: datetime
    statistical_evidence: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    confidence_score: float
    suggested_actions: List[ResponseAction]
    auto_response_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'degradation_type': self.degradation_type.value,
            'severity': self.severity.value,
            'affected_metrics': self.affected_metrics,
            'affected_components': self.affected_components,
            'detection_time': self.detection_time.isoformat(),
            'statistical_evidence': self.statistical_evidence,
            'trend_analysis': self.trend_analysis,
            'confidence_score': self.confidence_score,
            'suggested_actions': [action.value for action in self.suggested_actions],
            'auto_response_enabled': self.auto_response_enabled
        }

class StatisticalDegradationDetector:
    """Statistical algorithms for performance degradation detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_samples = config.get('min_samples', 20)
        self.window_size = config.get('window_size', 50)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.trend_sensitivity = config.get('trend_sensitivity', 0.05)
        self.variance_threshold = config.get('variance_threshold', 2.0)
        
    def detect_degradation(self, metrics: List[PerformanceMetric]) -> List[DegradationEvent]:
        """Detect performance degradation using statistical methods."""
        events = []
        
        if len(metrics) < self.min_samples:
            return events
        
        try:
            # Convert to time series
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Group by component and metric
            grouped = df.groupby(['component', 'name'])
            
            for (component, metric_name), group in grouped:
                if len(group) < self.min_samples:
                    continue
                
                # Extract values and timestamps
                values = group['value'].values
                timestamps = group['timestamp'].values
                
                # Run detection algorithms
                degradation_events = []
                
                # 1. Trend analysis
                trend_event = self._detect_trend_degradation(
                    values, timestamps, component, metric_name
                )
                if trend_event:
                    degradation_events.append(trend_event)
                
                # 2. Sudden drop detection
                sudden_drop_event = self._detect_sudden_drop(
                    values, timestamps, component, metric_name
                )
                if sudden_drop_event:
                    degradation_events.append(sudden_drop_event)
                
                # 3. Variance increase detection
                variance_event = self._detect_variance_increase(
                    values, timestamps, component, metric_name
                )
                if variance_event:
                    degradation_events.append(variance_event)
                
                # 4. Outlier surge detection
                outlier_event = self._detect_outlier_surge(
                    values, timestamps, component, metric_name
                )
                if outlier_event:
                    degradation_events.append(outlier_event)
                
                # 5. Cyclical degradation detection
                cyclical_event = self._detect_cyclical_degradation(
                    values, timestamps, component, metric_name
                )
                if cyclical_event:
                    degradation_events.append(cyclical_event)
                
                events.extend(degradation_events)
                
        except Exception as e:
            logger.error(f"Error in degradation detection: {e}")
        
        return events
    
    def _detect_trend_degradation(self, values: np.ndarray, timestamps: np.ndarray, 
                                component: str, metric_name: str) -> Optional[DegradationEvent]:
        """Detect gradual trend degradation."""
        try:
            # Use recent window for trend analysis
            window = min(self.window_size, len(values))
            recent_values = values[-window:]
            recent_timestamps = timestamps[-window:]
            
            # Calculate trend using linear regression
            x = np.arange(len(recent_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
            
            # Check if trend is significantly negative (degradation)
            if p_value < 0.05 and slope < -self.trend_sensitivity:
                # Calculate confidence based on r-squared and p-value
                confidence = abs(r_value) * (1 - p_value)
                
                if confidence > self.confidence_threshold:
                    # Determine severity based on slope magnitude
                    if abs(slope) > 0.2:
                        severity = DegradationSeverity.CRITICAL
                    elif abs(slope) > 0.1:
                        severity = DegradationSeverity.MAJOR
                    elif abs(slope) > 0.05:
                        severity = DegradationSeverity.MODERATE
                    else:
                        severity = DegradationSeverity.MINOR
                    
                    return DegradationEvent(
                        event_id=f"trend_{component}_{metric_name}_{int(time.time())}",
                        degradation_type=DegradationType.GRADUAL_DECLINE,
                        severity=severity,
                        affected_metrics=[metric_name],
                        affected_components=[component],
                        detection_time=datetime.utcnow(),
                        statistical_evidence={
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'std_error': std_err,
                            'window_size': window
                        },
                        trend_analysis={
                            'trend_direction': 'declining',
                            'trend_strength': abs(slope),
                            'linear_fit_quality': r_value**2
                        },
                        confidence_score=confidence,
                        suggested_actions=self._get_suggested_actions(severity),
                        auto_response_enabled=self.config.get('auto_response', False)
                    )
            
        except Exception as e:
            logger.error(f"Error in trend degradation detection: {e}")
        
        return None
    
    def _detect_sudden_drop(self, values: np.ndarray, timestamps: np.ndarray,
                           component: str, metric_name: str) -> Optional[DegradationEvent]:
        """Detect sudden performance drops."""
        try:
            if len(values) < 10:
                return None
            
            # Calculate rolling mean and std
            window = min(10, len(values) // 2)
            rolling_mean = pd.Series(values).rolling(window=window).mean()
            rolling_std = pd.Series(values).rolling(window=window).std()
            
            # Look for sudden drops (values significantly below recent mean)
            recent_mean = rolling_mean.iloc[-5:].mean()
            recent_std = rolling_std.iloc[-5:].mean()
            
            # Check last few values for sudden drop
            last_values = values[-3:]
            drop_threshold = recent_mean - 2 * recent_std
            
            sudden_drops = np.sum(last_values < drop_threshold)
            
            if sudden_drops >= 2:  # At least 2 of last 3 values are drops
                # Calculate drop magnitude
                drop_magnitude = (recent_mean - np.mean(last_values)) / recent_std
                
                if drop_magnitude > 2.0:  # Significant drop
                    # Determine severity
                    if drop_magnitude > 4.0:
                        severity = DegradationSeverity.CRITICAL
                    elif drop_magnitude > 3.0:
                        severity = DegradationSeverity.MAJOR
                    elif drop_magnitude > 2.5:
                        severity = DegradationSeverity.MODERATE
                    else:
                        severity = DegradationSeverity.MINOR
                    
                    confidence = min(0.95, drop_magnitude / 4.0)
                    
                    return DegradationEvent(
                        event_id=f"sudden_drop_{component}_{metric_name}_{int(time.time())}",
                        degradation_type=DegradationType.SUDDEN_DROP,
                        severity=severity,
                        affected_metrics=[metric_name],
                        affected_components=[component],
                        detection_time=datetime.utcnow(),
                        statistical_evidence={
                            'drop_magnitude': drop_magnitude,
                            'recent_mean': recent_mean,
                            'recent_std': recent_std,
                            'affected_values': sudden_drops,
                            'threshold': drop_threshold
                        },
                        trend_analysis={
                            'change_type': 'sudden_drop',
                            'magnitude': drop_magnitude,
                            'affected_points': sudden_drops
                        },
                        confidence_score=confidence,
                        suggested_actions=self._get_suggested_actions(severity),
                        auto_response_enabled=self.config.get('auto_response', False)
                    )
            
        except Exception as e:
            logger.error(f"Error in sudden drop detection: {e}")
        
        return None
    
    def _detect_variance_increase(self, values: np.ndarray, timestamps: np.ndarray,
                                component: str, metric_name: str) -> Optional[DegradationEvent]:
        """Detect increased variance indicating instability."""
        try:
            if len(values) < 20:
                return None
            
            # Split into two halves
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            # Calculate variance for each half
            var1 = np.var(first_half)
            var2 = np.var(second_half)
            
            # Check if variance increased significantly
            if var1 > 0:  # Avoid division by zero
                variance_ratio = var2 / var1
                
                if variance_ratio > self.variance_threshold:
                    # Perform F-test for variance difference
                    f_stat = var2 / var1 if var2 > var1 else var1 / var2
                    df1 = len(second_half) - 1
                    df2 = len(first_half) - 1
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                    
                    if p_value < 0.05:  # Significant variance increase
                        # Determine severity
                        if variance_ratio > 5.0:
                            severity = DegradationSeverity.CRITICAL
                        elif variance_ratio > 3.0:
                            severity = DegradationSeverity.MAJOR
                        elif variance_ratio > 2.5:
                            severity = DegradationSeverity.MODERATE
                        else:
                            severity = DegradationSeverity.MINOR
                        
                        confidence = min(0.95, (1 - p_value) * 0.8)
                        
                        return DegradationEvent(
                            event_id=f"variance_{component}_{metric_name}_{int(time.time())}",
                            degradation_type=DegradationType.VARIANCE_INCREASE,
                            severity=severity,
                            affected_metrics=[metric_name],
                            affected_components=[component],
                            detection_time=datetime.utcnow(),
                            statistical_evidence={
                                'variance_ratio': variance_ratio,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'first_half_var': var1,
                                'second_half_var': var2
                            },
                            trend_analysis={
                                'stability_change': 'increased_variance',
                                'variance_increase_factor': variance_ratio,
                                'statistical_significance': p_value < 0.05
                            },
                            confidence_score=confidence,
                            suggested_actions=self._get_suggested_actions(severity),
                            auto_response_enabled=self.config.get('auto_response', False)
                        )
            
        except Exception as e:
            logger.error(f"Error in variance increase detection: {e}")
        
        return None
    
    def _detect_outlier_surge(self, values: np.ndarray, timestamps: np.ndarray,
                            component: str, metric_name: str) -> Optional[DegradationEvent]:
        """Detect surge in outliers indicating system instability."""
        try:
            if len(values) < 20:
                return None
            
            # Calculate IQR-based outliers
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = (values < lower_bound) | (values > upper_bound)
            
            # Check recent outlier rate
            recent_window = min(20, len(values) // 2)
            recent_outliers = outliers[-recent_window:]
            historical_outliers = outliers[:-recent_window]
            
            recent_outlier_rate = np.mean(recent_outliers)
            historical_outlier_rate = np.mean(historical_outliers) if len(historical_outliers) > 0 else 0
            
            # Check if outlier rate has increased significantly
            if recent_outlier_rate > 0.15 and recent_outlier_rate > historical_outlier_rate * 2:
                # Determine severity based on outlier rate
                if recent_outlier_rate > 0.4:
                    severity = DegradationSeverity.CRITICAL
                elif recent_outlier_rate > 0.3:
                    severity = DegradationSeverity.MAJOR
                elif recent_outlier_rate > 0.2:
                    severity = DegradationSeverity.MODERATE
                else:
                    severity = DegradationSeverity.MINOR
                
                confidence = min(0.9, recent_outlier_rate * 2)
                
                return DegradationEvent(
                    event_id=f"outlier_surge_{component}_{metric_name}_{int(time.time())}",
                    degradation_type=DegradationType.OUTLIER_SURGE,
                    severity=severity,
                    affected_metrics=[metric_name],
                    affected_components=[component],
                    detection_time=datetime.utcnow(),
                    statistical_evidence={
                        'recent_outlier_rate': recent_outlier_rate,
                        'historical_outlier_rate': historical_outlier_rate,
                        'outlier_increase_factor': recent_outlier_rate / max(historical_outlier_rate, 0.01),
                        'iqr_bounds': [lower_bound, upper_bound],
                        'recent_outlier_count': np.sum(recent_outliers)
                    },
                    trend_analysis={
                        'instability_type': 'outlier_surge',
                        'outlier_rate_change': recent_outlier_rate - historical_outlier_rate,
                        'stability_degradation': True
                    },
                    confidence_score=confidence,
                    suggested_actions=self._get_suggested_actions(severity),
                    auto_response_enabled=self.config.get('auto_response', False)
                )
            
        except Exception as e:
            logger.error(f"Error in outlier surge detection: {e}")
        
        return None
    
    def _detect_cyclical_degradation(self, values: np.ndarray, timestamps: np.ndarray,
                                   component: str, metric_name: str) -> Optional[DegradationEvent]:
        """Detect cyclical performance degradation patterns."""
        try:
            if len(values) < 30:
                return None
            
            # Smooth the data to identify cycles
            if len(values) > 10:
                smoothed = savgol_filter(values, min(11, len(values) // 3), 3)
            else:
                smoothed = values
            
            # Find peaks and troughs
            peaks, _ = find_peaks(smoothed, height=np.mean(smoothed))
            troughs, _ = find_peaks(-smoothed, height=-np.mean(smoothed))
            
            # Check if there's a degrading cyclical pattern
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Check if peaks are generally declining
                peak_values = smoothed[peaks]
                peak_trend = np.polyfit(range(len(peak_values)), peak_values, 1)[0]
                
                # Check if troughs are generally declining
                trough_values = smoothed[troughs]
                trough_trend = np.polyfit(range(len(trough_values)), trough_values, 1)[0]
                
                # Both peaks and troughs should be declining for cyclical degradation
                if peak_trend < -0.01 and trough_trend < -0.01:
                    # Calculate cycle degradation rate
                    cycle_degradation = abs(peak_trend) + abs(trough_trend)
                    
                    if cycle_degradation > 0.05:
                        # Determine severity
                        if cycle_degradation > 0.2:
                            severity = DegradationSeverity.CRITICAL
                        elif cycle_degradation > 0.15:
                            severity = DegradationSeverity.MAJOR
                        elif cycle_degradation > 0.1:
                            severity = DegradationSeverity.MODERATE
                        else:
                            severity = DegradationSeverity.MINOR
                        
                        confidence = min(0.8, cycle_degradation / 0.2)
                        
                        return DegradationEvent(
                            event_id=f"cyclical_{component}_{metric_name}_{int(time.time())}",
                            degradation_type=DegradationType.CYCLICAL_DEGRADATION,
                            severity=severity,
                            affected_metrics=[metric_name],
                            affected_components=[component],
                            detection_time=datetime.utcnow(),
                            statistical_evidence={
                                'peak_trend': peak_trend,
                                'trough_trend': trough_trend,
                                'cycle_degradation_rate': cycle_degradation,
                                'peak_count': len(peaks),
                                'trough_count': len(troughs)
                            },
                            trend_analysis={
                                'pattern_type': 'cyclical_degradation',
                                'peak_degradation_rate': peak_trend,
                                'trough_degradation_rate': trough_trend,
                                'cycle_health': 'degrading'
                            },
                            confidence_score=confidence,
                            suggested_actions=self._get_suggested_actions(severity),
                            auto_response_enabled=self.config.get('auto_response', False)
                        )
            
        except Exception as e:
            logger.error(f"Error in cyclical degradation detection: {e}")
        
        return None
    
    def _get_suggested_actions(self, severity: DegradationSeverity) -> List[ResponseAction]:
        """Get suggested actions based on severity."""
        if severity == DegradationSeverity.CRITICAL:
            return [
                ResponseAction.EMERGENCY_STOP,
                ResponseAction.ROLLBACK_MODEL,
                ResponseAction.SCALE_RESOURCES,
                ResponseAction.TRIGGER_RETRAINING
            ]
        elif severity == DegradationSeverity.MAJOR:
            return [
                ResponseAction.ROLLBACK_MODEL,
                ResponseAction.TRIGGER_RETRAINING,
                ResponseAction.SCALE_RESOURCES,
                ResponseAction.INCREASE_MONITORING
            ]
        elif severity == DegradationSeverity.MODERATE:
            return [
                ResponseAction.TRIGGER_RETRAINING,
                ResponseAction.INCREASE_MONITORING,
                ResponseAction.SCALE_RESOURCES
            ]
        else:
            return [
                ResponseAction.INCREASE_MONITORING,
                ResponseAction.ALERT_ONLY
            ]

class PerformanceTrendAnalyzer:
    """Advanced trend analysis for performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_window = config.get('lookback_window', 100)
        self.trend_segments = config.get('trend_segments', 5)
        
    def analyze_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends."""
        try:
            if len(metrics) < 10:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            trend_analysis = {}
            
            # Group by component and metric
            grouped = df.groupby(['component', 'name'])
            
            for (component, metric_name), group in grouped:
                if len(group) < 10:
                    continue
                
                values = group['value'].values
                timestamps = group['timestamp'].values
                
                # Perform trend analysis
                trend_info = self._analyze_metric_trend(values, timestamps, component, metric_name)
                trend_analysis[f"{component}_{metric_name}"] = trend_info
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {}
    
    def _analyze_metric_trend(self, values: np.ndarray, timestamps: np.ndarray, 
                            component: str, metric_name: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric."""
        try:
            # Basic trend analysis
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Segment analysis
            segment_trends = self._analyze_trend_segments(values)
            
            # Seasonal analysis
            seasonal_info = self._analyze_seasonality(values, timestamps)
            
            # Volatility analysis
            volatility_info = self._analyze_volatility(values)
            
            # Forecast short-term trend
            forecast = self._forecast_trend(values)
            
            return {
                'overall_trend': {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'strength': abs(slope),
                    'significance': p_value < 0.05
                },
                'segment_analysis': segment_trends,
                'seasonal_patterns': seasonal_info,
                'volatility_analysis': volatility_info,
                'forecast': forecast,
                'trend_score': self._calculate_trend_score(slope, r_value, p_value),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing metric trend: {e}")
            return {}
    
    def _analyze_trend_segments(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in different segments."""
        try:
            segment_size = len(values) // self.trend_segments
            segments = []
            
            for i in range(self.trend_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < self.trend_segments - 1 else len(values)
                
                segment_values = values[start_idx:end_idx]
                
                if len(segment_values) > 2:
                    x = np.arange(len(segment_values))
                    slope, _, r_value, p_value, _ = stats.linregress(x, segment_values)
                    
                    segments.append({
                        'segment': i + 1,
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'mean_value': np.mean(segment_values),
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
            
            # Analyze trend changes between segments
            trend_changes = []
            for i in range(1, len(segments)):
                prev_slope = segments[i-1]['slope']
                curr_slope = segments[i]['slope']
                
                if prev_slope * curr_slope < 0:  # Sign change
                    trend_changes.append({
                        'segment_transition': f"{i} -> {i+1}",
                        'change_type': 'trend_reversal',
                        'magnitude': abs(curr_slope - prev_slope)
                    })
                elif abs(curr_slope - prev_slope) > 0.1:  # Significant change
                    trend_changes.append({
                        'segment_transition': f"{i} -> {i+1}",
                        'change_type': 'trend_acceleration' if abs(curr_slope) > abs(prev_slope) else 'trend_deceleration',
                        'magnitude': abs(curr_slope - prev_slope)
                    })
            
            return {
                'segments': segments,
                'trend_changes': trend_changes,
                'segment_consistency': self._calculate_segment_consistency(segments)
            }
            
        except Exception as e:
            logger.error(f"Error in segment analysis: {e}")
            return {}
    
    def _analyze_seasonality(self, values: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        try:
            # This is a simplified seasonal analysis
            # In practice, you might want to use more sophisticated methods
            
            seasonal_info = {
                'has_seasonality': False,
                'seasonal_strength': 0.0,
                'seasonal_period': None
            }
            
            if len(values) > 20:
                # Simple autocorrelation analysis
                autocorr = np.correlate(values, values, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Look for peaks in autocorrelation (indicating periodicity)
                peaks, _ = find_peaks(autocorr[1:], height=0.5 * np.max(autocorr))
                
                if len(peaks) > 0:
                    seasonal_info['has_seasonality'] = True
                    seasonal_info['seasonal_period'] = peaks[0] + 1
                    seasonal_info['seasonal_strength'] = autocorr[peaks[0] + 1] / np.max(autocorr)
            
            return seasonal_info
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            return {}
    
    def _analyze_volatility(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        try:
            # Calculate rolling volatility
            window = min(10, len(values) // 3)
            rolling_std = pd.Series(values).rolling(window=window).std()
            
            # Volatility metrics
            volatility_info = {
                'overall_volatility': np.std(values),
                'recent_volatility': rolling_std.iloc[-5:].mean(),
                'volatility_trend': 'increasing' if rolling_std.iloc[-1] > rolling_std.iloc[-10] else 'decreasing',
                'volatility_percentile': stats.percentileofscore(rolling_std.dropna(), rolling_std.iloc[-1])
            }
            
            # Volatility clustering analysis
            vol_changes = np.diff(rolling_std.dropna())
            volatility_info['volatility_clustering'] = np.sum(vol_changes[:-1] * vol_changes[1:] > 0) / len(vol_changes)
            
            return volatility_info
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {}
    
    def _forecast_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Simple trend forecasting."""
        try:
            # Use linear regression for simple forecasting
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Forecast next 5 points
            forecast_horizon = 5
            forecast_x = np.arange(len(values), len(values) + forecast_horizon)
            forecast_values = slope * forecast_x + intercept
            
            # Calculate confidence intervals
            confidence_interval = 1.96 * std_err * np.sqrt(1 + 1/len(values) + (forecast_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            
            return {
                'forecast_values': forecast_values.tolist(),
                'confidence_intervals': {
                    'lower': (forecast_values - confidence_interval).tolist(),
                    'upper': (forecast_values + confidence_interval).tolist()
                },
                'forecast_horizon': forecast_horizon,
                'model_quality': r_value**2
            }
            
        except Exception as e:
            logger.error(f"Error in trend forecasting: {e}")
            return {}
    
    def _calculate_segment_consistency(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate consistency across segments."""
        try:
            if len(segments) < 2:
                return 1.0
            
            slopes = [s['slope'] for s in segments]
            slope_std = np.std(slopes)
            slope_mean = np.mean(np.abs(slopes))
            
            # Lower standard deviation relative to mean indicates higher consistency
            consistency = 1.0 / (1.0 + slope_std / max(slope_mean, 0.001))
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating segment consistency: {e}")
            return 0.0
    
    def _calculate_trend_score(self, slope: float, r_value: float, p_value: float) -> float:
        """Calculate overall trend score."""
        try:
            # Combine slope strength, fit quality, and significance
            slope_component = min(1.0, abs(slope) / 0.1)  # Normalize slope
            fit_component = r_value**2  # R-squared
            significance_component = 1.0 - p_value if p_value < 0.05 else 0.0
            
            trend_score = (slope_component * 0.4 + fit_component * 0.4 + significance_component * 0.2)
            
            return trend_score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0

class PerformanceDegradationDetector:
    """Main performance degradation detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.statistical_detector = StatisticalDegradationDetector(config.get('statistical', {}))
        self.trend_analyzer = PerformanceTrendAnalyzer(config.get('trend_analysis', {}))
        
        # Data storage
        self.metrics_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.degradation_events = deque(maxlen=config.get('events_buffer_size', 100))
        
        # Monitoring state
        self.monitoring_active = False
        self.last_analysis_time = datetime.utcnow()
        self.analysis_interval = config.get('analysis_interval', 60)  # seconds
        
        # External dependencies
        self.alert_manager = None
        self.model_manager = None
        self.threshold_monitor = None
        
        # Performance tracking
        self.detection_stats = {
            'total_analyses': 0,
            'degradations_detected': 0,
            'false_positives': 0,
            'response_times': []
        }
    
    def set_dependencies(self, alert_manager=None, model_manager=None, threshold_monitor=None):
        """Set external dependencies."""
        self.alert_manager = alert_manager
        self.model_manager = model_manager
        self.threshold_monitor = threshold_monitor
    
    async def start_monitoring(self):
        """Start performance degradation monitoring."""
        self.monitoring_active = True
        logger.info("Starting performance degradation monitoring")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._maintenance_loop())
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Stopping performance degradation monitoring")
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric for monitoring."""
        self.metrics_buffer.append(metric)
        
        # Update Prometheus metrics
        PERFORMANCE_TREND_SCORE.labels(
            component=metric.component,
            metric=metric.name
        ).set(metric.value)
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check if it's time for analysis
                if self._should_run_analysis():
                    await self._run_degradation_analysis()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _analysis_loop(self):
        """Performance analysis loop."""
        while self.monitoring_active:
            try:
                # Run trend analysis
                if len(self.metrics_buffer) > 10:
                    trend_analysis = self.trend_analyzer.analyze_trends(list(self.metrics_buffer))
                    
                    # Update trend scores
                    for metric_key, trend_info in trend_analysis.items():
                        trend_score = trend_info.get('trend_score', 0.0)
                        component, metric_name = metric_key.split('_', 1)
                        
                        PERFORMANCE_TREND_SCORE.labels(
                            component=component,
                            metric=metric_name
                        ).set(trend_score)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(300)
    
    async def _maintenance_loop(self):
        """Maintenance and cleanup loop."""
        while self.monitoring_active:
            try:
                # Clean up old events
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=24)
                
                # Remove old degradation events
                self.degradation_events = deque(
                    [event for event in self.degradation_events 
                     if event.detection_time > cutoff_time],
                    maxlen=self.config.get('events_buffer_size', 100)
                )
                
                # Update statistics
                self._update_detection_statistics()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)
    
    def _should_run_analysis(self) -> bool:
        """Check if analysis should be run."""
        time_since_last = (datetime.utcnow() - self.last_analysis_time).total_seconds()
        return time_since_last >= self.analysis_interval and len(self.metrics_buffer) > 10
    
    async def _run_degradation_analysis(self):
        """Run degradation detection analysis."""
        start_time = time.time()
        
        try:
            # Get recent metrics
            recent_metrics = list(self.metrics_buffer)[-500:]  # Last 500 metrics
            
            # Run statistical detection
            detected_events = self.statistical_detector.detect_degradation(recent_metrics)
            
            # Process detected events
            for event in detected_events:
                await self._process_degradation_event(event)
            
            # Update last analysis time
            self.last_analysis_time = datetime.utcnow()
            
            # Update statistics
            self.detection_stats['total_analyses'] += 1
            self.detection_stats['degradations_detected'] += len(detected_events)
            
            # Record response time
            response_time = time.time() - start_time
            self.detection_stats['response_times'].append(response_time)
            DEGRADATION_RESPONSE_TIME.labels(response_type='analysis').observe(response_time)
            
            if detected_events:
                logger.info(f"Detected {len(detected_events)} performance degradation events")
            
        except Exception as e:
            logger.error(f"Error in degradation analysis: {e}")
    
    async def _process_degradation_event(self, event: DegradationEvent):
        """Process a detected degradation event."""
        try:
            # Add to events buffer
            self.degradation_events.append(event)
            
            # Update Prometheus metrics
            DEGRADATION_DETECTIONS.labels(
                component=event.affected_components[0] if event.affected_components else 'unknown',
                metric=event.affected_metrics[0] if event.affected_metrics else 'unknown',
                severity=event.severity.value
            ).inc()
            
            # Generate alert
            if self.alert_manager:
                await self._generate_alert(event)
            
            # Execute automated response if enabled
            if event.auto_response_enabled:
                await self._execute_automated_response(event)
            
            logger.warning(f"Performance degradation detected: {event.degradation_type.value} "
                         f"- {event.severity.value} - {event.affected_components}")
            
        except Exception as e:
            logger.error(f"Error processing degradation event: {e}")
    
    async def _generate_alert(self, event: DegradationEvent):
        """Generate alert for degradation event."""
        try:
            # Map severity to alert severity
            severity_map = {
                DegradationSeverity.MINOR: AlertSeverity.LOW,
                DegradationSeverity.MODERATE: AlertSeverity.MEDIUM,
                DegradationSeverity.MAJOR: AlertSeverity.HIGH,
                DegradationSeverity.CRITICAL: AlertSeverity.CRITICAL
            }
            
            alert = Alert(
                alert_id=event.event_id,
                alert_type=AlertType.SYSTEM_PERFORMANCE,
                severity=severity_map[event.severity],
                title=f"Performance Degradation: {event.degradation_type.value}",
                description=f"Performance degradation detected in {', '.join(event.affected_components)} "
                           f"for metrics: {', '.join(event.affected_metrics)}",
                timestamp=event.detection_time,
                source='performance_degradation_detector',
                status=AlertStatus.ACTIVE,
                metadata={
                    'degradation_type': event.degradation_type.value,
                    'confidence_score': event.confidence_score,
                    'statistical_evidence': event.statistical_evidence,
                    'trend_analysis': event.trend_analysis,
                    'suggested_actions': [action.value for action in event.suggested_actions]
                },
                tags=event.affected_components + event.affected_metrics
            )
            
            await self.alert_manager.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    async def _execute_automated_response(self, event: DegradationEvent):
        """Execute automated response to degradation."""
        try:
            for action in event.suggested_actions:
                await self._execute_response_action(action, event)
            
        except Exception as e:
            logger.error(f"Error executing automated response: {e}")
    
    async def _execute_response_action(self, action: ResponseAction, event: DegradationEvent):
        """Execute a specific response action."""
        start_time = time.time()
        
        try:
            if action == ResponseAction.TRIGGER_RETRAINING:
                await self._trigger_model_retraining(event)
            
            elif action == ResponseAction.ROLLBACK_MODEL:
                await self._rollback_model(event)
            
            elif action == ResponseAction.SCALE_RESOURCES:
                await self._scale_resources(event)
            
            elif action == ResponseAction.INCREASE_MONITORING:
                await self._increase_monitoring(event)
            
            elif action == ResponseAction.RESTART_SERVICE:
                await self._restart_service(event)
            
            elif action == ResponseAction.EMERGENCY_STOP:
                await self._emergency_stop(event)
            
            # Record response time
            response_time = time.time() - start_time
            DEGRADATION_RESPONSE_TIME.labels(response_type=action.value).observe(response_time)
            
            logger.info(f"Executed response action: {action.value} for event {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error executing response action {action.value}: {e}")
    
    async def _trigger_model_retraining(self, event: DegradationEvent):
        """Trigger model retraining."""
        try:
            if self.model_manager:
                # Determine which models need retraining based on affected components
                models_to_retrain = []
                
                for component in event.affected_components:
                    if 'tactical' in component.lower():
                        models_to_retrain.append('tactical_model')
                    elif 'strategic' in component.lower():
                        models_to_retrain.append('strategic_model')
                    elif 'risk' in component.lower():
                        models_to_retrain.append('risk_model')
                    else:
                        models_to_retrain.append('default_model')
                
                for model_type in models_to_retrain:
                    await self.model_manager.trigger_retraining(
                        model_type=model_type,
                        reason=f"Performance degradation: {event.degradation_type.value}",
                        priority='high' if event.severity in [DegradationSeverity.MAJOR, DegradationSeverity.CRITICAL] else 'medium'
                    )
                    
                    RETRAINING_TRIGGERS.labels(
                        model_type=model_type,
                        trigger_reason=event.degradation_type.value
                    ).inc()
                
                logger.info(f"Triggered retraining for models: {models_to_retrain}")
            
        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")
    
    async def _rollback_model(self, event: DegradationEvent):
        """Rollback model to previous version."""
        try:
            if self.model_manager:
                # Determine which models need rollback
                models_to_rollback = []
                
                for component in event.affected_components:
                    if 'tactical' in component.lower():
                        models_to_rollback.append('tactical_model')
                    elif 'strategic' in component.lower():
                        models_to_rollback.append('strategic_model')
                    elif 'risk' in component.lower():
                        models_to_rollback.append('risk_model')
                
                for model_type in models_to_rollback:
                    await self.model_manager.rollback_model(
                        model_type=model_type,
                        reason=f"Performance degradation: {event.degradation_type.value}",
                        emergency=event.severity == DegradationSeverity.CRITICAL
                    )
                    
                    ROLLBACK_EXECUTIONS.labels(
                        model_type=model_type,
                        rollback_reason=event.degradation_type.value
                    ).inc()
                
                logger.info(f"Rolled back models: {models_to_rollback}")
            
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
    
    async def _scale_resources(self, event: DegradationEvent):
        """Scale system resources."""
        try:
            # This is a placeholder for resource scaling logic
            # In practice, this would integrate with container orchestration
            # or cloud auto-scaling services
            
            logger.info(f"Resource scaling triggered for event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error scaling resources: {e}")
    
    async def _increase_monitoring(self, event: DegradationEvent):
        """Increase monitoring frequency."""
        try:
            if self.threshold_monitor:
                # Temporarily reduce monitoring intervals
                for component in event.affected_components:
                    await self.threshold_monitor.set_monitoring_frequency(
                        component=component,
                        frequency_multiplier=2.0,
                        duration_minutes=60
                    )
                
                logger.info(f"Increased monitoring for components: {event.affected_components}")
            
        except Exception as e:
            logger.error(f"Error increasing monitoring: {e}")
    
    async def _restart_service(self, event: DegradationEvent):
        """Restart affected service."""
        try:
            # This is a placeholder for service restart logic
            # In practice, this would integrate with service management
            
            logger.info(f"Service restart triggered for event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error restarting service: {e}")
    
    async def _emergency_stop(self, event: DegradationEvent):
        """Execute emergency stop procedures."""
        try:
            # This is a placeholder for emergency stop logic
            # In practice, this would trigger circuit breakers,
            # stop trading, and activate failover procedures
            
            logger.critical(f"Emergency stop triggered for event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
    
    def _update_detection_statistics(self):
        """Update detection statistics."""
        try:
            # Keep only recent response times
            if len(self.detection_stats['response_times']) > 100:
                self.detection_stats['response_times'] = self.detection_stats['response_times'][-100:]
            
            # Calculate average response time
            if self.detection_stats['response_times']:
                avg_response_time = np.mean(self.detection_stats['response_times'])
                logger.info(f"Average degradation detection response time: {avg_response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error updating detection statistics: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.monitoring_active,
            'metrics_buffer_size': len(self.metrics_buffer),
            'degradation_events_count': len(self.degradation_events),
            'last_analysis_time': self.last_analysis_time.isoformat(),
            'detection_statistics': self.detection_stats.copy(),
            'recent_events': [event.to_dict() for event in list(self.degradation_events)[-10:]],
            'configuration': {
                'analysis_interval': self.analysis_interval,
                'buffer_size': self.config.get('buffer_size', 1000),
                'auto_response_enabled': self.config.get('auto_response', False)
            }
        }

# Factory function
def create_performance_degradation_detector(config: Dict[str, Any]) -> PerformanceDegradationDetector:
    """Create performance degradation detector instance."""
    return PerformanceDegradationDetector(config)

# Example configuration
EXAMPLE_CONFIG = {
    'analysis_interval': 60,  # seconds
    'buffer_size': 1000,
    'events_buffer_size': 100,
    'auto_response': False,
    'statistical': {
        'min_samples': 20,
        'window_size': 50,
        'confidence_threshold': 0.8,
        'trend_sensitivity': 0.05,
        'variance_threshold': 2.0
    },
    'trend_analysis': {
        'lookback_window': 100,
        'trend_segments': 5
    }
}

# Example usage
async def main():
    """Example usage of performance degradation detector."""
    config = EXAMPLE_CONFIG
    detector = create_performance_degradation_detector(config)
    
    # Start monitoring
    monitoring_task = asyncio.create_task(detector.start_monitoring())
    
    # Simulate adding metrics
    async def simulate_metrics():
        for i in range(100):
            # Simulate normal performance
            base_value = 100 + 10 * np.sin(i / 10)
            
            # Add some degradation after point 50
            if i > 50:
                base_value *= (1 - (i - 50) * 0.01)  # Gradual degradation
            
            # Add noise
            value = base_value + np.random.normal(0, 5)
            
            metric = PerformanceMetric(
                name='response_time',
                value=value,
                timestamp=datetime.utcnow(),
                component='web_server',
                metadata={'endpoint': '/api/test'}
            )
            
            detector.add_metric(metric)
            await asyncio.sleep(1)
    
    # Run simulation
    await asyncio.gather(monitoring_task, simulate_metrics())

if __name__ == "__main__":
    asyncio.run(main())