"""
Quality Assurance Monitor - Intelligence Integration Layer

Real-time validation and health monitoring system for intelligence coordination.
Ensures system reliability, detects anomalies, and maintains quality standards.

Features:
- Real-time validation of intelligence component outputs
- Anomaly detection for intelligence system behavior
- Automatic recalibration when performance degrades
- Health monitoring for all intelligence components
- Emergency shutdown procedures for faulty intelligence
- Quality metrics tracking and alerting

Architecture:
- Health Monitor: Component status and performance tracking
- Anomaly Detector: Statistical and ML-based anomaly detection
- Quality Validator: Output validation and consistency checking
- Alert System: Real-time alerting and escalation
- Recovery Engine: Automatic recovery and recalibration
"""

import asyncio
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque
import json
import math
from abc import ABC, abstractmethod
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    LATENCY_SPIKE = "latency_spike"
    ACCURACY_DROP = "accuracy_drop"
    CONFIDENCE_ANOMALY = "confidence_anomaly"
    OUTPUT_INCONSISTENCY = "output_inconsistency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMMUNICATION_FAILURE = "communication_failure"
    BEHAVIORAL_DRIFT = "behavioral_drift"


class QualityMetric(Enum):
    """Quality metrics to monitor"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    CONFIDENCE_CONSISTENCY = "confidence_consistency"
    OUTPUT_VALIDITY = "output_validity"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"


@dataclass
class HealthCheck:
    """Health check result"""
    component_name: str
    check_type: str
    status: HealthStatus
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    details: str = ""
    recovery_suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityAlert:
    """Quality assurance alert"""
    alert_id: str
    component_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    anomaly_type: Optional[AnomalyType] = None
    threshold_violated: Optional[str] = None
    recommended_actions: List[str] = field(default_factory=list)
    auto_resolved: bool = False


@dataclass
class QualityThreshold:
    """Quality threshold definition"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "above"  # "above" or "below"
    window_size: int = 10
    consecutive_violations: int = 3


@dataclass
class ComponentMetrics:
    """Component performance metrics"""
    component_name: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    confidence_values: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.now)


class AnomalyDetector:
    """Statistical and ML-based anomaly detection"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for outlier detection
        self.baseline_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # ML-based anomaly detection
        self.ml_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_matrices: Dict[str, np.ndarray] = {}
        self.ml_training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.ml_models_trained: Dict[str, bool] = defaultdict(bool)
        self.ml_model_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Advanced detection parameters
        self.min_samples_for_ml = 50
        self.retrain_interval = 100  # Retrain every N samples
        self.contamination_rate = 0.1  # Expected anomaly rate
        self.ensemble_voting_threshold = 0.5  # For ensemble methods
        
    def detect_anomalies(self, 
                        component_name: str,
                        metric_name: str,
                        value: float,
                        timestamp: datetime,
                        additional_features: Optional[Dict[str, float]] = None) -> List[AnomalyType]:
        """Detect anomalies in metric values using multiple detection methods"""
        anomalies = []
        
        # Store value in baseline window
        key = f"{component_name}_{metric_name}"
        self.baseline_windows[key].append(value)
        
        # Store ML training data with additional features
        feature_vector = [value, timestamp.timestamp()]
        if additional_features:
            feature_vector.extend(additional_features.values())
        
        self.ml_training_data[key].append({
            'timestamp': timestamp,
            'value': value,
            'features': feature_vector
        })
        
        # Need sufficient data for baseline
        if len(self.baseline_windows[key]) < 10:
            return anomalies
        
        baseline_values = list(self.baseline_windows[key])
        
        # 1. Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(
            metric_name, value, baseline_values
        )
        anomalies.extend(statistical_anomalies)
        
        # 2. ML-based anomaly detection
        if len(self.ml_training_data[key]) >= self.min_samples_for_ml:
            ml_anomalies = self._detect_ml_anomalies(
                key, value, feature_vector, timestamp
            )
            anomalies.extend(ml_anomalies)
        
        # 3. Behavioral drift detection
        if len(baseline_values) >= 30:
            drift_anomaly = self._detect_behavioral_drift(
                metric_name, baseline_values
            )
            if drift_anomaly:
                anomalies.append(drift_anomaly)
        
        # 4. Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(
            key, baseline_values, value
        )
        anomalies.extend(pattern_anomalies)
        
        # 5. Ensemble voting for final decision
        final_anomalies = self._ensemble_anomaly_voting(
            anomalies, statistical_anomalies, ml_anomalies if len(self.ml_training_data[key]) >= self.min_samples_for_ml else []
        )
        
        # Record anomalies
        for anomaly in final_anomalies:
            self.anomaly_history.append({
                'component': component_name,
                'metric': metric_name,
                'anomaly_type': anomaly,
                'value': value,
                'timestamp': timestamp,
                'detection_methods': self._get_detection_methods(anomaly, anomalies)
            })
        
        return final_anomalies
    
    def _detect_statistical_anomalies(self, 
                                   metric_name: str,
                                   value: float,
                                   baseline: List[float]) -> List[AnomalyType]:
        """Detect statistical anomalies using z-score and IQR"""
        anomalies = []
        
        if len(baseline) < 5:
            return anomalies
        
        # Z-score based detection
        mean_val = np.mean(baseline)
        std_val = np.std(baseline)
        
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            
            if z_score > self.sensitivity:
                if metric_name in ["response_time", "latency"]:
                    anomalies.append(AnomalyType.LATENCY_SPIKE)
                elif metric_name in ["accuracy", "precision", "recall"]:
                    anomalies.append(AnomalyType.ACCURACY_DROP)
                elif metric_name in ["confidence"]:
                    anomalies.append(AnomalyType.CONFIDENCE_ANOMALY)
        
        # IQR based detection for outliers
        q1 = np.percentile(baseline, 25)
        q3 = np.percentile(baseline, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if value < lower_bound or value > upper_bound:
            anomalies.append(AnomalyType.OUTPUT_INCONSISTENCY)
        
        return anomalies
    
    def _detect_behavioral_drift(self, 
                               metric_name: str,
                               values: List[float]) -> Optional[AnomalyType]:
        """Detect behavioral drift using trend analysis"""
        if len(values) < 30:
            return None
        
        # Split into early and recent periods
        split_point = len(values) // 2
        early_values = values[:split_point]
        recent_values = values[split_point:]
        
        # Statistical test for distribution shift
        try:
            statistic, p_value = stats.ks_2samp(early_values, recent_values)
            
            # Significant drift detected
            if p_value < 0.05:
                return AnomalyType.BEHAVIORAL_DRIFT
        except Exception:
            pass
        
        return None
    
    def _detect_ml_anomalies(self,
                           key: str,
                           value: float,
                           feature_vector: List[float],
                           timestamp: datetime) -> List[AnomalyType]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        try:
            # Train or retrain models if needed
            if not self.ml_models_trained[key] or len(self.ml_training_data[key]) % self.retrain_interval == 0:
                self._train_ml_models(key)
            
            if not self.ml_models_trained[key]:
                return anomalies
            
            # Prepare feature vector
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            if key in self.scalers:
                feature_array = self.scalers[key].transform(feature_array)
            
            # Isolation Forest detection
            if f"{key}_isolation_forest" in self.ml_models:
                isolation_score = self.ml_models[f"{key}_isolation_forest"].decision_function(feature_array)[0]
                if isolation_score < -0.5:  # Threshold for anomaly
                    anomalies.append(AnomalyType.BEHAVIORAL_DRIFT)
            
            # DBSCAN clustering detection
            if f"{key}_dbscan" in self.ml_models:
                cluster_label = self.ml_models[f"{key}_dbscan"].fit_predict(feature_array)[0]
                if cluster_label == -1:  # Outlier cluster
                    anomalies.append(AnomalyType.OUTPUT_INCONSISTENCY)
            
            # PCA-based reconstruction error
            if f"{key}_pca" in self.ml_models:
                pca_model = self.ml_models[f"{key}_pca"]
                transformed = pca_model.transform(feature_array)
                reconstructed = pca_model.inverse_transform(transformed)
                reconstruction_error = np.mean((feature_array - reconstructed) ** 2)
                
                # Store reconstruction error for threshold calculation
                self.ml_model_scores[f"{key}_pca"].append(reconstruction_error)
                
                if len(self.ml_model_scores[f"{key}_pca"]) > 10:
                    threshold = np.percentile(self.ml_model_scores[f"{key}_pca"], 95)
                    if reconstruction_error > threshold:
                        anomalies.append(AnomalyType.CONFIDENCE_ANOMALY)
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection for {key}: {e}")
        
        return anomalies
    
    def _train_ml_models(self, key: str):
        """Train ML models for anomaly detection"""
        try:
            if len(self.ml_training_data[key]) < self.min_samples_for_ml:
                return
            
            # Prepare training data
            training_data = list(self.ml_training_data[key])
            features = np.array([item['features'] for item in training_data])
            
            # Feature scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[key] = scaler
            
            # Train Isolation Forest
            isolation_forest = IsolationForest(
                contamination=self.contamination_rate,
                random_state=42,
                n_estimators=100
            )
            isolation_forest.fit(features_scaled)
            self.ml_models[f"{key}_isolation_forest"] = isolation_forest
            
            # Train DBSCAN (for clustering-based anomaly detection)
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(features_scaled)
            self.ml_models[f"{key}_dbscan"] = dbscan
            
            # Train PCA for reconstruction error
            pca = PCA(n_components=min(features_scaled.shape[1], 3))
            pca.fit(features_scaled)
            self.ml_models[f"{key}_pca"] = pca
            
            # Mark as trained
            self.ml_models_trained[key] = True
            
            logger.debug(f"ML models trained for {key}")
            
        except Exception as e:
            logger.error(f"Error training ML models for {key}: {e}")
    
    def _detect_pattern_anomalies(self,
                                key: str,
                                baseline_values: List[float],
                                current_value: float) -> List[AnomalyType]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        try:
            if len(baseline_values) < 20:
                return anomalies
            
            # Convert to numpy array for analysis
            values = np.array(baseline_values)
            
            # 1. Detect sudden spikes or drops
            recent_values = values[-5:]  # Last 5 values
            historical_mean = np.mean(values[:-5])
            recent_mean = np.mean(recent_values)
            
            if abs(recent_mean - historical_mean) > 2 * np.std(values):
                if 'response_time' in key or 'latency' in key:
                    anomalies.append(AnomalyType.LATENCY_SPIKE)
                else:
                    anomalies.append(AnomalyType.OUTPUT_INCONSISTENCY)
            
            # 2. Detect trend changes
            if len(values) >= 30:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                # Calculate trends
                x = np.arange(len(first_half))
                first_trend = np.polyfit(x, first_half, 1)[0]
                
                x = np.arange(len(second_half))
                second_trend = np.polyfit(x, second_half, 1)[0]
                
                # Significant trend change
                if abs(first_trend - second_trend) > np.std(values) * 0.1:
                    anomalies.append(AnomalyType.BEHAVIORAL_DRIFT)
            
            # 3. Detect oscillations/cyclical patterns
            if len(values) >= 20:
                # Use autocorrelation to detect patterns
                normalized_values = (values - np.mean(values)) / np.std(values)
                autocorr = np.correlate(normalized_values, normalized_values, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Look for significant autocorrelation at lag > 1
                if len(autocorr) > 5:
                    max_autocorr = np.max(autocorr[2:min(10, len(autocorr))])
                    if max_autocorr > 0.7:  # Strong correlation indicating pattern
                        anomalies.append(AnomalyType.OUTPUT_INCONSISTENCY)
            
            # 4. Detect value stagnation
            if len(set(values[-10:])) == 1:  # All same values in last 10 samples
                anomalies.append(AnomalyType.COMMUNICATION_FAILURE)
            
        except Exception as e:
            logger.error(f"Error in pattern anomaly detection for {key}: {e}")
        
        return anomalies
    
    def _ensemble_anomaly_voting(self,
                               all_anomalies: List[AnomalyType],
                               statistical_anomalies: List[AnomalyType],
                               ml_anomalies: List[AnomalyType]) -> List[AnomalyType]:
        """Use ensemble voting to determine final anomalies"""
        # Count votes for each anomaly type
        anomaly_votes = defaultdict(int)
        
        for anomaly in all_anomalies:
            anomaly_votes[anomaly] += 1
        
        # Weight ML anomalies higher if models are well-trained
        for anomaly in ml_anomalies:
            anomaly_votes[anomaly] += 0.5  # Additional weight for ML detection
        
        # Determine final anomalies based on voting threshold
        final_anomalies = []
        for anomaly, votes in anomaly_votes.items():
            if votes >= self.ensemble_voting_threshold:
                final_anomalies.append(anomaly)
        
        return final_anomalies
    
    def _get_detection_methods(self, anomaly: AnomalyType, all_anomalies: List[AnomalyType]) -> List[str]:
        """Get list of detection methods that identified the anomaly"""
        methods = []
        
        if anomaly in all_anomalies:
            methods.append("statistical")
        
        # Could add more specific method tracking here
        return methods
    
    def get_ml_model_performance(self, key: str) -> Dict[str, Any]:
        """Get performance metrics for ML models"""
        performance = {
            'models_trained': self.ml_models_trained.get(key, False),
            'training_samples': len(self.ml_training_data.get(key, [])),
            'model_types': []
        }
        
        # Check which models are available
        for model_type in ['isolation_forest', 'dbscan', 'pca']:
            model_key = f"{key}_{model_type}"
            if model_key in self.ml_models:
                performance['model_types'].append(model_type)
        
        # Get recent scores if available
        if f"{key}_pca" in self.ml_model_scores:
            scores = list(self.ml_model_scores[f"{key}_pca"])
            if scores:
                performance['pca_reconstruction_error'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'latest': scores[-1] if scores else None
                }
        
        return performance
    
    def retrain_models(self, key: Optional[str] = None):
        """Manually trigger model retraining"""
        if key:
            self._train_ml_models(key)
        else:
            # Retrain all models
            for model_key in self.ml_models_trained.keys():
                self._train_ml_models(model_key)
    
    def get_anomaly_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent anomalies"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_anomalies = [
            a for a in self.anomaly_history 
            if a['timestamp'] >= cutoff_time
        ]
        
        # Count by type
        type_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for anomaly in recent_anomalies:
            type_counts[anomaly['anomaly_type'].value] += 1
            component_counts[anomaly['component']] += 1
        
        return {
            'total_anomalies': len(recent_anomalies),
            'anomaly_types': dict(type_counts),
            'components_affected': dict(component_counts),
            'time_window_hours': hours_back
        }


class HealthMonitor:
    """Component health monitoring system"""
    
    def __init__(self):
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.quality_thresholds: Dict[str, QualityThreshold] = {}
        self.consecutive_violations: Dict[str, int] = defaultdict(int)
        
    def register_component(self, component_name: str):
        """Register component for health monitoring"""
        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = ComponentMetrics(
                component_name=component_name
            )
            logger.debug("Component registered for health monitoring", 
                        component=component_name)
    
    def set_quality_threshold(self, threshold: QualityThreshold):
        """Set quality threshold for monitoring"""
        key = f"{threshold.metric_name}"
        self.quality_thresholds[key] = threshold
        logger.debug("Quality threshold set",
                    metric=threshold.metric_name,
                    warning=threshold.warning_threshold,
                    critical=threshold.critical_threshold)
    
    def record_response_time(self, component_name: str, response_time_ms: float):
        """Record component response time"""
        if component_name not in self.component_metrics:
            self.register_component(component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.response_times.append(response_time_ms)
        
        # Check thresholds
        self._check_threshold_violations(component_name, "response_time", response_time_ms)
    
    def record_accuracy(self, component_name: str, accuracy: float):
        """Record component accuracy"""
        if component_name not in self.component_metrics:
            self.register_component(component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.accuracy_scores.append(accuracy)
        
        self._check_threshold_violations(component_name, "accuracy", accuracy)
    
    def record_confidence(self, component_name: str, confidence: float):
        """Record component confidence"""
        if component_name not in self.component_metrics:
            self.register_component(component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.confidence_values.append(confidence)
        
        self._check_threshold_violations(component_name, "confidence", confidence)
    
    def record_success(self, component_name: str):
        """Record successful operation"""
        if component_name not in self.component_metrics:
            self.register_component(component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.success_count += 1
        metrics.last_success = datetime.now()
        
        # Reset consecutive violations on success
        violation_keys = [k for k in self.consecutive_violations.keys() 
                         if k.startswith(component_name)]
        for key in violation_keys:
            self.consecutive_violations[key] = 0
    
    def record_failure(self, component_name: str, error_details: str = ""):
        """Record failed operation"""
        if component_name not in self.component_metrics:
            self.register_component(component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.error_count += 1
        metrics.last_failure = datetime.now()
        
        logger.warning("Component failure recorded",
                      component=component_name,
                      error_details=error_details)
    
    def _check_threshold_violations(self, 
                                  component_name: str,
                                  metric_name: str,
                                  value: float) -> List[str]:
        """Check for threshold violations"""
        violations = []
        threshold_key = metric_name
        
        if threshold_key not in self.quality_thresholds:
            return violations
        
        threshold = self.quality_thresholds[threshold_key]
        violation_key = f"{component_name}_{metric_name}"
        
        # Check violation based on direction
        violation_detected = False
        
        if threshold.direction == "above":
            if value > threshold.critical_threshold:
                violations.append("critical")
                violation_detected = True
            elif value > threshold.warning_threshold:
                violations.append("warning")
                violation_detected = True
        else:  # "below"
            if value < threshold.critical_threshold:
                violations.append("critical")
                violation_detected = True
            elif value < threshold.warning_threshold:
                violations.append("warning")
                violation_detected = True
        
        # Track consecutive violations
        if violation_detected:
            self.consecutive_violations[violation_key] += 1
        else:
            self.consecutive_violations[violation_key] = 0
        
        return violations
    
    def get_component_health(self, component_name: str) -> HealthCheck:
        """Get current health status for component"""
        if component_name not in self.component_metrics:
            return HealthCheck(
                component_name=component_name,
                check_type="general",
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                details="Component not registered"
            )
        
        metrics = self.component_metrics[component_name]
        status = HealthStatus.HEALTHY
        details = []
        recovery_suggestions = []
        health_metrics = {}
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - metrics.uptime_start).total_seconds()
        health_metrics['uptime_hours'] = uptime_seconds / 3600
        
        # Calculate error rate
        total_operations = metrics.success_count + metrics.error_count
        if total_operations > 0:
            error_rate = metrics.error_count / total_operations
            health_metrics['error_rate'] = error_rate
            
            if error_rate > 0.1:  # 10% error rate
                status = HealthStatus.DEGRADED
                details.append(f"High error rate: {error_rate:.1%}")
                recovery_suggestions.append("Check component logs for errors")
        
        # Check response time
        if metrics.response_times:
            avg_response_time = np.mean(metrics.response_times)
            p95_response_time = np.percentile(metrics.response_times, 95)
            health_metrics['avg_response_time_ms'] = avg_response_time
            health_metrics['p95_response_time_ms'] = p95_response_time
            
            if avg_response_time > 50:  # 50ms threshold
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                details.append(f"Slow response time: {avg_response_time:.1f}ms")
                recovery_suggestions.append("Check component performance and resources")
        
        # Check accuracy
        if metrics.accuracy_scores:
            avg_accuracy = np.mean(metrics.accuracy_scores)
            health_metrics['avg_accuracy'] = avg_accuracy
            
            if avg_accuracy < 0.7:  # 70% accuracy threshold
                status = HealthStatus.CRITICAL
                details.append(f"Low accuracy: {avg_accuracy:.1%}")
                recovery_suggestions.append("Recalibrate component or check input data")
        
        # Check recent failures
        if metrics.last_failure:
            time_since_failure = datetime.now() - metrics.last_failure
            if time_since_failure.total_seconds() < 300:  # 5 minutes
                if status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
                    status = HealthStatus.DEGRADED
                details.append("Recent failure detected")
                recovery_suggestions.append("Investigate recent failure cause")
        
        # Check availability
        if metrics.last_success:
            time_since_success = datetime.now() - metrics.last_success
            if time_since_success.total_seconds() > 600:  # 10 minutes
                status = HealthStatus.FAILED
                details.append("No successful operations recently")
                recovery_suggestions.append("Check component connectivity and status")
        
        return HealthCheck(
            component_name=component_name,
            check_type="comprehensive",
            status=status,
            timestamp=datetime.now(),
            metrics=health_metrics,
            details="; ".join(details) if details else "All metrics within normal ranges",
            recovery_suggestions=recovery_suggestions
        )
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        component_statuses = {}
        status_counts = defaultdict(int)
        
        for component_name in self.component_metrics.keys():
            health_check = self.get_component_health(component_name)
            component_statuses[component_name] = health_check.status.value
            status_counts[health_check.status.value] += 1
        
        # Overall system status
        if status_counts[HealthStatus.FAILED.value] > 0:
            overall_status = HealthStatus.FAILED
        elif status_counts[HealthStatus.CRITICAL.value] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.WARNING.value] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'overall_status': overall_status.value,
            'component_count': len(self.component_metrics),
            'status_distribution': dict(status_counts),
            'component_statuses': component_statuses
        }


class QualityAssuranceMonitor:
    """
    Quality Assurance Monitor for Intelligence Coordination
    
    Provides comprehensive monitoring, validation, and health tracking for
    all intelligence components with real-time alerting and recovery.
    """
    
    def __init__(self):
        # Core monitoring components
        self.health_monitor = HealthMonitor()
        self.anomaly_detector = AnomalyDetector()
        
        # Alert management
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        # Quality tracking
        self.quality_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.validation_results: deque = deque(maxlen=500)
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.alert_suppression: Dict[str, datetime] = {}
        self.suppression_window_minutes = 5
        
        # Background monitoring
        self.monitoring_thread = None
        self.running = False
        
        # Recovery engine
        self.auto_recovery_enabled = True
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.max_recovery_attempts = 3
        
        # Quality standards
        self._initialize_quality_standards()
        
        logger.info("Quality assurance monitor initialized")
    
    def _initialize_quality_standards(self):
        """Initialize quality standards and thresholds"""
        # Response time thresholds
        self.health_monitor.set_quality_threshold(QualityThreshold(
            metric_name="response_time",
            warning_threshold=10.0,   # 10ms warning
            critical_threshold=25.0,  # 25ms critical
            direction="above",
            consecutive_violations=3
        ))
        
        # Accuracy thresholds
        self.health_monitor.set_quality_threshold(QualityThreshold(
            metric_name="accuracy",
            warning_threshold=0.8,    # 80% warning
            critical_threshold=0.7,   # 70% critical
            direction="below",
            consecutive_violations=5
        ))
        
        # Confidence thresholds
        self.health_monitor.set_quality_threshold(QualityThreshold(
            metric_name="confidence",
            warning_threshold=0.6,    # 60% warning
            critical_threshold=0.4,   # 40% critical
            direction="below",
            consecutive_violations=3
        ))
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="quality_monitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Periodic health checks
                self._perform_health_checks()
                
                # Check for alert escalations
                self._check_alert_escalations()
                
                # Cleanup old alerts
                self._cleanup_expired_alerts()
                
                # Auto-recovery attempts
                if self.auto_recovery_enabled:
                    self._attempt_auto_recovery()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(5)
    
    def register_component(self, component_name: str):
        """Register component for monitoring"""
        self.health_monitor.register_component(component_name)
        logger.debug("Component registered for QA monitoring", component=component_name)
    
    def validate_component_output(self, 
                                component_name: str,
                                output_data: Any,
                                expected_type: type,
                                confidence: Optional[float] = None) -> bool:
        """Validate component output quality"""
        validation_start = datetime.now()
        
        try:
            # Type validation
            if not isinstance(output_data, expected_type):
                self._create_alert(
                    component_name,
                    AlertSeverity.ERROR,
                    f"Invalid output type: expected {expected_type.__name__}, got {type(output_data).__name__}",
                    AnomalyType.OUTPUT_INCONSISTENCY
                )
                return False
            
            # Range validation for numerical outputs
            if isinstance(output_data, (int, float)):
                if not np.isfinite(output_data):
                    self._create_alert(
                        component_name,
                        AlertSeverity.ERROR,
                        f"Invalid numerical output: {output_data}",
                        AnomalyType.OUTPUT_INCONSISTENCY
                    )
                    return False
            
            # Confidence validation
            if confidence is not None:
                if not (0.0 <= confidence <= 1.0):
                    self._create_alert(
                        component_name,
                        AlertSeverity.WARNING,
                        f"Confidence out of range: {confidence}",
                        AnomalyType.CONFIDENCE_ANOMALY
                    )
                
                # Record confidence for monitoring
                self.health_monitor.record_confidence(component_name, confidence)
            
            # Record successful validation
            self.health_monitor.record_success(component_name)
            
            # Store validation result
            validation_time_ms = (datetime.now() - validation_start).total_seconds() * 1000
            self.validation_results.append({
                'component': component_name,
                'timestamp': datetime.now(),
                'success': True,
                'validation_time_ms': validation_time_ms
            })
            
            return True
            
        except Exception as e:
            self.health_monitor.record_failure(component_name, str(e))
            self._create_alert(
                component_name,
                AlertSeverity.ERROR,
                f"Validation error: {str(e)}",
                AnomalyType.OUTPUT_INCONSISTENCY
            )
            return False
    
    def record_performance_metric(self, 
                                component_name: str,
                                metric_type: QualityMetric,
                                value: float):
        """Record performance metric and check for anomalies"""
        timestamp = datetime.now()
        
        # Record in health monitor
        if metric_type == QualityMetric.RESPONSE_TIME:
            self.health_monitor.record_response_time(component_name, value)
        elif metric_type == QualityMetric.ACCURACY:
            self.health_monitor.record_accuracy(component_name, value)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            component_name, metric_type.value, value, timestamp
        )
        
        # Create alerts for anomalies
        for anomaly in anomalies:
            severity = self._get_anomaly_severity(anomaly, value)
            self._create_alert(
                component_name,
                severity,
                f"{anomaly.value} detected: {metric_type.value} = {value}",
                anomaly
            )
        
        # Store quality score
        self.quality_scores[component_name].append(value)
    
    def _get_anomaly_severity(self, anomaly: AnomalyType, value: float) -> AlertSeverity:
        """Determine alert severity based on anomaly type"""
        if anomaly in [AnomalyType.COMMUNICATION_FAILURE, AnomalyType.RESOURCE_EXHAUSTION]:
            return AlertSeverity.CRITICAL
        elif anomaly in [AnomalyType.LATENCY_SPIKE, AnomalyType.ACCURACY_DROP]:
            return AlertSeverity.ERROR
        elif anomaly in [AnomalyType.CONFIDENCE_ANOMALY, AnomalyType.OUTPUT_INCONSISTENCY]:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _create_alert(self, 
                     component_name: str,
                     severity: AlertSeverity,
                     message: str,
                     anomaly_type: Optional[AnomalyType] = None):
        """Create and manage quality alert"""
        # Check alert suppression
        suppression_key = f"{component_name}_{severity.value}_{anomaly_type.value if anomaly_type else 'general'}"
        
        if suppression_key in self.alert_suppression:
            last_alert_time = self.alert_suppression[suppression_key]
            time_since_last = datetime.now() - last_alert_time
            
            if time_since_last.total_seconds() < self.suppression_window_minutes * 60:
                return  # Suppress duplicate alert
        
        # Create alert
        alert_id = f"qa_{int(time.time())}_{component_name}_{severity.value}"
        
        alert = QualityAlert(
            alert_id=alert_id,
            component_name=component_name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            recommended_actions=self._get_recommended_actions(component_name, severity, anomaly_type)
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.alert_suppression[suppression_key] = datetime.now()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Error in alert callback", error=str(e))
        
        # Log alert
        logger.log(
            self._get_log_level(severity),
            "Quality alert created",
            alert_id=alert_id,
            component=component_name,
            severity=severity.value,
            message=message
        )
    
    def _get_log_level(self, severity: AlertSeverity) -> str:
        """Get log level for alert severity"""
        level_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.EMERGENCY: "critical"
        }
        return level_map.get(severity, "info")
    
    def _get_recommended_actions(self, 
                               component_name: str,
                               severity: AlertSeverity,
                               anomaly_type: Optional[AnomalyType]) -> List[str]:
        """Get recommended actions for alert"""
        actions = []
        
        if anomaly_type == AnomalyType.LATENCY_SPIKE:
            actions.extend([
                "Check system resource usage",
                "Review component performance logs",
                "Consider scaling resources"
            ])
        elif anomaly_type == AnomalyType.ACCURACY_DROP:
            actions.extend([
                "Recalibrate component parameters",
                "Check input data quality",
                "Review model performance"
            ])
        elif anomaly_type == AnomalyType.CONFIDENCE_ANOMALY:
            actions.extend([
                "Review confidence calculation logic",
                "Check input uncertainty",
                "Validate model outputs"
            ])
        elif anomaly_type == AnomalyType.OUTPUT_INCONSISTENCY:
            actions.extend([
                "Validate output format",
                "Check component logic",
                "Review input processing"
            ])
        
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            actions.append("Consider emergency shutdown if issues persist")
        
        return actions
    
    def _perform_health_checks(self):
        """Perform periodic health checks on all components"""
        for component_name in self.health_monitor.component_metrics.keys():
            health_check = self.health_monitor.get_component_health(component_name)
            
            # Create alerts for unhealthy components
            if health_check.status == HealthStatus.CRITICAL:
                self._create_alert(
                    component_name,
                    AlertSeverity.CRITICAL,
                    f"Component health critical: {health_check.details}"
                )
            elif health_check.status == HealthStatus.DEGRADED:
                self._create_alert(
                    component_name,
                    AlertSeverity.WARNING,
                    f"Component health degraded: {health_check.details}"
                )
            elif health_check.status == HealthStatus.FAILED:
                self._create_alert(
                    component_name,
                    AlertSeverity.EMERGENCY,
                    f"Component failed: {health_check.details}"
                )
    
    def _check_alert_escalations(self):
        """Check for alert escalations"""
        for alert in list(self.active_alerts.values()):
            age_minutes = (datetime.now() - alert.timestamp).total_seconds() / 60
            
            # Escalate unresolved critical alerts after 15 minutes
            if (alert.severity == AlertSeverity.CRITICAL and 
                age_minutes > 15 and 
                not alert.auto_resolved):
                
                self._create_alert(
                    alert.component_name,
                    AlertSeverity.EMERGENCY,
                    f"Escalated: Unresolved critical alert - {alert.message}"
                )
    
    def _cleanup_expired_alerts(self):
        """Cleanup expired alerts"""
        current_time = datetime.now()
        expired_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve old info/warning alerts after 1 hour
            if alert.severity in [AlertSeverity.INFO, AlertSeverity.WARNING]:
                age_hours = (current_time - alert.timestamp).total_seconds() / 3600
                if age_hours > 1:
                    alert.auto_resolved = True
                    expired_alerts.append(alert_id)
        
        # Remove expired alerts
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
    
    def _attempt_auto_recovery(self):
        """Attempt automatic recovery for failed components"""
        for component_name in self.health_monitor.component_metrics.keys():
            health_check = self.health_monitor.get_component_health(component_name)
            
            if health_check.status in [HealthStatus.FAILED, HealthStatus.CRITICAL]:
                attempts = self.recovery_attempts[component_name]
                
                if attempts < self.max_recovery_attempts:
                    logger.info("Attempting auto-recovery",
                               component=component_name,
                               attempt=attempts + 1)
                    
                    # Placeholder for actual recovery logic
                    # This would restart, recalibrate, or reset the component
                    self.recovery_attempts[component_name] += 1
                    
                    # Simulate recovery success (would be actual recovery logic)
                    if attempts == 0:  # First attempt often succeeds
                        self.health_monitor.record_success(component_name)
                        logger.info("Auto-recovery successful", component=component_name)
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Manually resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.auto_resolved = False  # Mark as manually resolved
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved",
                       alert_id=alert_id,
                       resolution_note=resolution_note)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report"""
        # System health summary
        health_summary = self.health_monitor.get_system_health_summary()
        
        # Alert summary
        active_alert_count = len(self.active_alerts)
        alert_severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            alert_severity_counts[alert.severity.value] += 1
        
        # Anomaly summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        
        # Validation metrics
        recent_validations = list(self.validation_results)[-100:]  # Last 100
        validation_success_rate = (
            sum(1 for v in recent_validations if v['success']) / len(recent_validations)
            if recent_validations else 0.0
        )
        
        # Quality scores
        component_quality = {}
        for component, scores in self.quality_scores.items():
            if scores:
                component_quality[component] = {
                    'avg_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'score_count': len(scores)
                }
        
        return {
            'timestamp': datetime.now(),
            'monitoring_enabled': self.monitoring_enabled,
            'health_summary': health_summary,
            'active_alerts': {
                'count': active_alert_count,
                'severity_distribution': dict(alert_severity_counts)
            },
            'anomaly_summary': anomaly_summary,
            'validation_metrics': {
                'success_rate': validation_success_rate,
                'total_validations': len(self.validation_results)
            },
            'component_quality_scores': component_quality,
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'recovery_attempts': dict(self.recovery_attempts)
        }
    
    def emergency_shutdown_component(self, component_name: str, reason: str):
        """Emergency shutdown of component"""
        logger.critical("EMERGENCY COMPONENT SHUTDOWN",
                       component=component_name,
                       reason=reason)
        
        # Create emergency alert
        self._create_alert(
            component_name,
            AlertSeverity.EMERGENCY,
            f"Emergency shutdown: {reason}"
        )
        
        # Mark component as failed
        if component_name in self.health_monitor.component_metrics:
            self.health_monitor.record_failure(component_name, reason)
        
        # Disable auto-recovery for this component
        self.recovery_attempts[component_name] = self.max_recovery_attempts
    
    def enable_monitoring(self):
        """Enable quality monitoring"""
        self.monitoring_enabled = True
        logger.info("Quality monitoring enabled")
    
    def disable_monitoring(self):
        """Disable quality monitoring"""
        self.monitoring_enabled = False
        logger.info("Quality monitoring disabled")
    
    def reset_component_metrics(self, component_name: str):
        """Reset metrics for specific component"""
        if component_name in self.health_monitor.component_metrics:
            # Reset metrics
            metrics = self.health_monitor.component_metrics[component_name]
            metrics.response_times.clear()
            metrics.accuracy_scores.clear()
            metrics.confidence_values.clear()
            metrics.error_count = 0
            metrics.success_count = 0
            metrics.uptime_start = datetime.now()
            
            # Reset recovery attempts
            self.recovery_attempts[component_name] = 0
            
            logger.info("Component metrics reset", component=component_name)