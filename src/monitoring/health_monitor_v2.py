"""
Enhanced Health Monitoring System with Predictive Analytics
===========================================================

Advanced health monitoring system that combines traditional health checks with
machine learning-based predictive failure detection to enable proactive self-healing.

Features:
- Multi-layered health assessment (shallow, deep, predictive)
- Machine learning-based failure prediction
- Anomaly detection for performance metrics
- Predictive alerting before failures occur
- Integration with automated recovery systems
- Performance trend analysis
- Resource utilization prediction
"""

import asyncio
import logging
import time
import json
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import pickle
import threading
from pathlib import Path

# Machine learning imports
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available. Predictive features disabled.")

from ..core.event_bus import EventBus
from ..core.events import Event

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Enhanced health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    PREDICTING_FAILURE = "predicting_failure"
    UNKNOWN = "unknown"


class PredictionType(Enum):
    """Types of failure predictions."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SERVICE_FAILURE = "service_failure"
    NETWORK_ISSUES = "network_issues"
    OVERLOAD_CONDITION = "overload_condition"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics."""
    timestamp: datetime
    
    # System metrics
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    disk_percent: float
    network_io_read: float
    network_io_write: float
    
    # Application metrics
    response_time_ms: float
    error_rate: float
    request_rate: float
    active_connections: int
    queue_size: int
    
    # Service-specific metrics
    service_name: str
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of failure prediction."""
    prediction_type: PredictionType
    probability: float
    confidence: float
    time_to_failure_seconds: Optional[float]
    risk_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthAssessment:
    """Comprehensive health assessment."""
    service_name: str
    overall_status: HealthStatus
    health_score: float  # 0-100
    
    # Traditional health checks
    shallow_check_passed: bool
    deep_check_passed: bool
    
    # Predictive analysis
    failure_predictions: List[PredictionResult]
    anomalies_detected: List[str]
    
    # Performance metrics
    current_metrics: HealthMetrics
    trend_analysis: Dict[str, Any]
    
    # Recommendations
    immediate_actions: List[str]
    preventive_actions: List[str]
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PredictiveFailureDetector:
    """Machine learning-based failure prediction system."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "/tmp/failure_prediction_model.pkl"
        self.models = {}
        self.scalers = {}
        self.feature_history = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_history = deque(maxlen=500)
        self.training_data = []
        self.is_trained = False
        self._lock = threading.Lock()
        
        if ML_AVAILABLE:
            self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models for different prediction types."""
        # Performance degradation predictor
        self.models['performance'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Resource exhaustion predictor
        self.models['resources'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Service failure predictor
        self.models['service'] = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
        
        # Initialize scalers
        for model_type in self.models.keys():
            self.scalers[model_type] = StandardScaler()
    
    def extract_features(self, metrics: HealthMetrics) -> np.ndarray:
        """Extract feature vector from health metrics."""
        features = [
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.disk_percent,
            metrics.response_time_ms,
            metrics.error_rate,
            metrics.request_rate,
            metrics.active_connections,
            metrics.queue_size,
            metrics.network_io_read,
            metrics.network_io_write
        ]
        
        # Add custom metrics
        for value in metrics.custom_metrics.values():
            features.append(value)
        
        return np.array(features).reshape(1, -1)
    
    def add_training_data(self, metrics: HealthMetrics, failure_occurred: bool):
        """Add training data point."""
        if not ML_AVAILABLE:
            return
            
        with self._lock:
            features = self.extract_features(metrics)
            self.training_data.append({
                'features': features.flatten(),
                'target': failure_occurred,
                'timestamp': metrics.timestamp
            })
    
    def train_models(self) -> bool:
        """Train prediction models with accumulated data."""
        if not ML_AVAILABLE or len(self.training_data) < 50:
            return False
        
        with self._lock:
            try:
                # Prepare training data
                X = np.array([item['features'] for item in self.training_data])
                y = np.array([item['target'] for item in self.training_data])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train each model
                for model_type, model in self.models.items():
                    # Fit scaler
                    X_train_scaled = self.scalers[model_type].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_type].transform(X_test)
                    
                    if model_type == 'performance':
                        # Unsupervised anomaly detection
                        model.fit(X_train_scaled[y_train == 0])  # Train on normal data
                    else:
                        # Supervised classification
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate model
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        
                        logger.info(f"Model {model_type} - Accuracy: {accuracy:.3f}, "
                                  f"Precision: {precision:.3f}, Recall: {recall:.3f}")
                
                self.is_trained = True
                self._save_models()
                return True
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                return False
    
    def predict_failures(self, metrics: HealthMetrics) -> List[PredictionResult]:
        """Predict potential failures based on current metrics."""
        if not ML_AVAILABLE or not self.is_trained:
            return []
        
        predictions = []
        
        with self._lock:
            try:
                features = self.extract_features(metrics)
                
                # Performance degradation prediction
                perf_features = self.scalers['performance'].transform(features)
                perf_anomaly_score = self.models['performance'].decision_function(perf_features)[0]
                
                if perf_anomaly_score < -0.1:  # Threshold for anomaly
                    predictions.append(PredictionResult(
                        prediction_type=PredictionType.PERFORMANCE_DEGRADATION,
                        probability=min(abs(perf_anomaly_score), 1.0),
                        confidence=0.8,
                        time_to_failure_seconds=self._estimate_time_to_failure(
                            metrics, PredictionType.PERFORMANCE_DEGRADATION
                        ),
                        risk_factors=self._identify_performance_risk_factors(metrics),
                        recommended_actions=self._get_performance_recommendations(metrics)
                    ))
                
                # Resource exhaustion prediction
                resource_features = self.scalers['resources'].transform(features)
                if hasattr(self.models['resources'], 'predict_proba'):
                    resource_prob = self.models['resources'].predict_proba(resource_features)[0][1]
                    
                    if resource_prob > 0.7:
                        predictions.append(PredictionResult(
                            prediction_type=PredictionType.RESOURCE_EXHAUSTION,
                            probability=resource_prob,
                            confidence=0.85,
                            time_to_failure_seconds=self._estimate_time_to_failure(
                                metrics, PredictionType.RESOURCE_EXHAUSTION
                            ),
                            risk_factors=self._identify_resource_risk_factors(metrics),
                            recommended_actions=self._get_resource_recommendations(metrics)
                        ))
                
                # Service failure prediction
                service_features = self.scalers['service'].transform(features)
                if hasattr(self.models['service'], 'predict_proba'):
                    service_prob = self.models['service'].predict_proba(service_features)[0][1]
                    
                    if service_prob > 0.6:
                        predictions.append(PredictionResult(
                            prediction_type=PredictionType.SERVICE_FAILURE,
                            probability=service_prob,
                            confidence=0.75,
                            time_to_failure_seconds=self._estimate_time_to_failure(
                                metrics, PredictionType.SERVICE_FAILURE
                            ),
                            risk_factors=self._identify_service_risk_factors(metrics),
                            recommended_actions=self._get_service_recommendations(metrics)
                        ))
                
                # Store prediction history
                for prediction in predictions:
                    self.prediction_history.append({
                        'timestamp': datetime.utcnow(),
                        'service': metrics.service_name,
                        'prediction': prediction
                    })
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
        
        return predictions
    
    def _estimate_time_to_failure(self, metrics: HealthMetrics, prediction_type: PredictionType) -> Optional[float]:
        """Estimate time until failure based on current trends."""
        service_history = self.feature_history[metrics.service_name]
        
        if len(service_history) < 10:
            return None
        
        # Simple trend analysis
        recent_metrics = list(service_history)[-10:]
        
        if prediction_type == PredictionType.RESOURCE_EXHAUSTION:
            if metrics.memory_percent > 80:
                # Estimate based on memory growth rate
                memory_trend = np.polyfit(range(len(recent_metrics)), 
                                        [m.memory_percent for m in recent_metrics], 1)
                if memory_trend[0] > 0:  # Increasing trend
                    remaining_capacity = 100 - metrics.memory_percent
                    time_to_failure = remaining_capacity / memory_trend[0] * 60  # Convert to seconds
                    return max(60, min(time_to_failure, 3600))  # Between 1 min and 1 hour
        
        elif prediction_type == PredictionType.PERFORMANCE_DEGRADATION:
            if metrics.response_time_ms > 1000:
                # Estimate based on response time growth
                response_times = [m.response_time_ms for m in recent_metrics]
                if len(set(response_times)) > 1:  # Check if there's variation
                    trend = np.polyfit(range(len(response_times)), response_times, 1)
                    if trend[0] > 0:  # Increasing trend
                        time_to_failure = (5000 - metrics.response_time_ms) / trend[0] * 60
                        return max(300, min(time_to_failure, 1800))  # Between 5 min and 30 min
        
        return 1800  # Default to 30 minutes
    
    def _identify_performance_risk_factors(self, metrics: HealthMetrics) -> List[str]:
        """Identify performance-related risk factors."""
        risk_factors = []
        
        if metrics.response_time_ms > 1000:
            risk_factors.append("High response time")
        if metrics.error_rate > 0.05:
            risk_factors.append("Elevated error rate")
        if metrics.queue_size > 100:
            risk_factors.append("Large queue size")
        if metrics.active_connections > 500:
            risk_factors.append("High connection count")
        
        return risk_factors
    
    def _identify_resource_risk_factors(self, metrics: HealthMetrics) -> List[str]:
        """Identify resource-related risk factors."""
        risk_factors = []
        
        if metrics.cpu_percent > 80:
            risk_factors.append("High CPU usage")
        if metrics.memory_percent > 85:
            risk_factors.append("High memory usage")
        if metrics.disk_percent > 90:
            risk_factors.append("High disk usage")
        
        return risk_factors
    
    def _identify_service_risk_factors(self, metrics: HealthMetrics) -> List[str]:
        """Identify service-related risk factors."""
        risk_factors = []
        
        if metrics.error_rate > 0.1:
            risk_factors.append("High error rate")
        if metrics.response_time_ms > 2000:
            risk_factors.append("Very slow responses")
        if metrics.request_rate < 1 and metrics.queue_size > 0:
            risk_factors.append("Processing bottleneck")
        
        return risk_factors
    
    def _get_performance_recommendations(self, metrics: HealthMetrics) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        if metrics.response_time_ms > 1000:
            recommendations.append("Scale horizontally or optimize processing")
        if metrics.queue_size > 100:
            recommendations.append("Increase worker capacity")
        if metrics.error_rate > 0.05:
            recommendations.append("Investigate and fix error sources")
        
        return recommendations
    
    def _get_resource_recommendations(self, metrics: HealthMetrics) -> List[str]:
        """Get resource management recommendations."""
        recommendations = []
        
        if metrics.memory_percent > 85:
            recommendations.append("Increase memory allocation or optimize usage")
        if metrics.cpu_percent > 80:
            recommendations.append("Scale out or optimize CPU-intensive operations")
        if metrics.disk_percent > 90:
            recommendations.append("Clean up disk space or add storage")
        
        return recommendations
    
    def _get_service_recommendations(self, metrics: HealthMetrics) -> List[str]:
        """Get service-specific recommendations."""
        recommendations = []
        
        if metrics.error_rate > 0.1:
            recommendations.append("Enable circuit breaker or fallback mechanisms")
        if metrics.response_time_ms > 2000:
            recommendations.append("Consider graceful degradation")
        
        return recommendations
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'is_trained': self.is_trained
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.is_trained = model_data['is_trained']
                return True
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
        
        return False


class AnomalyDetector:
    """Statistical anomaly detection for performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.thresholds = {}
    
    def add_metric(self, metric_name: str, value: float):
        """Add metric value for anomaly detection."""
        self.metric_history[metric_name].append(value)
        self._update_baseline(metric_name)
    
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for metric."""
        values = list(self.metric_history[metric_name])
        
        if len(values) >= 10:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            self.baselines[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'median': statistics.median(values),
                'percentile_95': np.percentile(values, 95) if len(values) >= 20 else mean_val
            }
            
            # Set anomaly threshold (3 standard deviations)
            self.thresholds[metric_name] = mean_val + 3 * std_val
    
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[str]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        for metric_name, value in current_metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                baseline = self.baselines[metric_name]
                
                # Check for anomalies
                if value > threshold or value < baseline['mean'] - 3 * baseline['std']:
                    anomalies.append(f"{metric_name}: {value:.2f} (baseline: {baseline['mean']:.2f})")
        
        return anomalies


class EnhancedHealthMonitor:
    """Enhanced health monitoring system with predictive capabilities."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.services = {}
        self.health_check_functions = {}
        self.failure_detector = PredictiveFailureDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Configuration
        self.check_interval = 30.0
        self.prediction_interval = 60.0
        
        # Background tasks
        self.monitoring_task = None
        self.prediction_task = None
        self.running = False
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 1000.0,
            'error_rate': 0.05
        }
        
        logger.info("Enhanced health monitor initialized")
    
    async def initialize(self):
        """Initialize the enhanced health monitor."""
        # Load pre-trained models if available
        if ML_AVAILABLE:
            self.failure_detector._load_models()
        
        # Start monitoring tasks
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        
        logger.info("Enhanced health monitor started")
    
    async def close(self):
        """Close the health monitor."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.prediction_task:
            self.prediction_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.monitoring_task, self.prediction_task, return_exceptions=True
        )
        
        logger.info("Enhanced health monitor closed")
    
    def register_service(
        self,
        service_name: str,
        health_check_function: Optional[Callable] = None,
        deep_check_function: Optional[Callable] = None,
        metrics_collector: Optional[Callable] = None
    ):
        """Register a service for enhanced monitoring."""
        self.services[service_name] = {
            'health_check': health_check_function,
            'deep_check': deep_check_function,
            'metrics_collector': metrics_collector,
            'last_assessment': None,
            'assessment_history': deque(maxlen=100)
        }
        
        logger.info(f"Service registered for enhanced monitoring: {service_name}")
    
    async def assess_service_health(self, service_name: str) -> HealthAssessment:
        """Perform comprehensive health assessment for a service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_info = self.services[service_name]
        
        # Collect current metrics
        current_metrics = await self._collect_metrics(service_name)
        
        # Perform traditional health checks
        shallow_check_passed = await self._perform_shallow_check(service_name)
        deep_check_passed = await self._perform_deep_check(service_name)
        
        # Add metrics to anomaly detection
        metric_dict = {
            'cpu_percent': current_metrics.cpu_percent,
            'memory_percent': current_metrics.memory_percent,
            'response_time_ms': current_metrics.response_time_ms,
            'error_rate': current_metrics.error_rate
        }
        
        for metric_name, value in metric_dict.items():
            self.anomaly_detector.add_metric(f"{service_name}_{metric_name}", value)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(metric_dict)
        
        # Predict failures
        failure_predictions = self.failure_detector.predict_failures(current_metrics)
        
        # Calculate health score
        health_score = self._calculate_health_score(
            current_metrics, shallow_check_passed, deep_check_passed, anomalies, failure_predictions
        )
        
        # Determine overall status
        overall_status = self._determine_health_status(
            health_score, shallow_check_passed, deep_check_passed, failure_predictions
        )
        
        # Generate recommendations
        immediate_actions, preventive_actions = self._generate_recommendations(
            current_metrics, failure_predictions, anomalies
        )
        
        # Create assessment
        assessment = HealthAssessment(
            service_name=service_name,
            overall_status=overall_status,
            health_score=health_score,
            shallow_check_passed=shallow_check_passed,
            deep_check_passed=deep_check_passed,
            failure_predictions=failure_predictions,
            anomalies_detected=anomalies,
            current_metrics=current_metrics,
            trend_analysis=self._analyze_trends(service_name),
            immediate_actions=immediate_actions,
            preventive_actions=preventive_actions
        )
        
        # Store assessment
        service_info['last_assessment'] = assessment
        service_info['assessment_history'].append(assessment)
        
        # Send events for status changes
        await self._handle_status_change(service_name, assessment)
        
        return assessment
    
    async def _collect_metrics(self, service_name: str) -> HealthMetrics:
        """Collect comprehensive metrics for a service."""
        service_info = self.services[service_name]
        
        # Default metrics
        metrics = HealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=0.0,
            memory_used_mb=0.0,
            memory_percent=0.0,
            disk_percent=0.0,
            network_io_read=0.0,
            network_io_write=0.0,
            response_time_ms=0.0,
            error_rate=0.0,
            request_rate=0.0,
            active_connections=0,
            queue_size=0,
            service_name=service_name
        )
        
        # Use custom metrics collector if available
        if service_info['metrics_collector']:
            try:
                custom_metrics = await service_info['metrics_collector']()
                if isinstance(custom_metrics, dict):
                    for key, value in custom_metrics.items():
                        if hasattr(metrics, key):
                            setattr(metrics, key, value)
                        else:
                            metrics.custom_metrics[key] = value
            except Exception as e:
                logger.error(f"Failed to collect custom metrics for {service_name}: {e}")
        
        # Collect system metrics if no custom collector
        else:
            try:
                import psutil
                
                metrics.cpu_percent = psutil.cpu_percent(interval=1)
                
                memory = psutil.virtual_memory()
                metrics.memory_used_mb = memory.used / (1024 * 1024)
                metrics.memory_percent = memory.percent
                
                disk = psutil.disk_usage('/')
                metrics.disk_percent = disk.percent
                
                net_io = psutil.net_io_counters()
                metrics.network_io_read = net_io.bytes_recv
                metrics.network_io_write = net_io.bytes_sent
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    async def _perform_shallow_check(self, service_name: str) -> bool:
        """Perform shallow health check."""
        service_info = self.services[service_name]
        
        if service_info['health_check']:
            try:
                result = await service_info['health_check']()
                return bool(result)
            except Exception as e:
                logger.error(f"Shallow health check failed for {service_name}: {e}")
                return False
        
        return True  # Default to healthy if no check defined
    
    async def _perform_deep_check(self, service_name: str) -> bool:
        """Perform deep health check."""
        service_info = self.services[service_name]
        
        if service_info['deep_check']:
            try:
                result = await service_info['deep_check']()
                return bool(result)
            except Exception as e:
                logger.error(f"Deep health check failed for {service_name}: {e}")
                return False
        
        return True  # Default to healthy if no check defined
    
    def _calculate_health_score(
        self,
        metrics: HealthMetrics,
        shallow_check: bool,
        deep_check: bool,
        anomalies: List[str],
        predictions: List[PredictionResult]
    ) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Health check penalties
        if not shallow_check:
            score -= 30
        if not deep_check:
            score -= 20
        
        # Metric-based penalties
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            score -= (metrics.cpu_percent - self.thresholds['cpu_percent']) * 0.5
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            score -= (metrics.memory_percent - self.thresholds['memory_percent']) * 0.5
        
        if metrics.response_time_ms > self.thresholds['response_time_ms']:
            score -= min(20, (metrics.response_time_ms - self.thresholds['response_time_ms']) / 100)
        
        if metrics.error_rate > self.thresholds['error_rate']:
            score -= min(25, (metrics.error_rate - self.thresholds['error_rate']) * 500)
        
        # Anomaly penalties
        score -= len(anomalies) * 5
        
        # Prediction penalties
        for prediction in predictions:
            if prediction.probability > 0.8:
                score -= 15
            elif prediction.probability > 0.6:
                score -= 10
            elif prediction.probability > 0.4:
                score -= 5
        
        return max(0, min(100, score))
    
    def _determine_health_status(
        self,
        health_score: float,
        shallow_check: bool,
        deep_check: bool,
        predictions: List[PredictionResult]
    ) -> HealthStatus:
        """Determine overall health status."""
        # Check for critical conditions
        if not shallow_check and not deep_check:
            return HealthStatus.CRITICAL
        
        # Check for predicted failures
        high_risk_predictions = [p for p in predictions if p.probability > 0.8]
        if high_risk_predictions:
            return HealthStatus.PREDICTING_FAILURE
        
        # Score-based status
        if health_score >= 90:
            return HealthStatus.HEALTHY
        elif health_score >= 75:
            return HealthStatus.WARNING
        elif health_score >= 50:
            return HealthStatus.DEGRADED
        elif health_score >= 25:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _generate_recommendations(
        self,
        metrics: HealthMetrics,
        predictions: List[PredictionResult],
        anomalies: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Generate immediate and preventive action recommendations."""
        immediate_actions = []
        preventive_actions = []
        
        # Immediate actions based on current state
        if metrics.memory_percent > 90:
            immediate_actions.append("Restart service or increase memory allocation")
        
        if metrics.cpu_percent > 95:
            immediate_actions.append("Scale out immediately or reduce load")
        
        if metrics.error_rate > 0.1:
            immediate_actions.append("Enable circuit breaker and investigate errors")
        
        if metrics.response_time_ms > 5000:
            immediate_actions.append("Enable graceful degradation mode")
        
        # Preventive actions based on predictions
        for prediction in predictions:
            immediate_actions.extend(prediction.recommended_actions)
            
            if prediction.time_to_failure_seconds and prediction.time_to_failure_seconds < 600:
                immediate_actions.append(f"Take action within {prediction.time_to_failure_seconds/60:.1f} minutes")
        
        # Preventive actions based on trends
        if metrics.memory_percent > 80:
            preventive_actions.append("Monitor memory usage and plan capacity increase")
        
        if metrics.cpu_percent > 70:
            preventive_actions.append("Consider horizontal scaling")
        
        if anomalies:
            preventive_actions.append("Investigate detected anomalies")
        
        return immediate_actions, preventive_actions
    
    def _analyze_trends(self, service_name: str) -> Dict[str, Any]:
        """Analyze historical trends for the service."""
        service_info = self.services[service_name]
        history = list(service_info['assessment_history'])
        
        if len(history) < 5:
            return {'trend': 'insufficient_data'}
        
        # Analyze health score trend
        scores = [assessment.health_score for assessment in history[-10:]]
        
        if len(scores) >= 2:
            trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
            
            if trend_slope > 2:
                trend = 'improving'
            elif trend_slope < -2:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_scores': scores,
            'average_score': statistics.mean(scores),
            'score_variance': statistics.variance(scores) if len(scores) > 1 else 0
        }
    
    async def _handle_status_change(self, service_name: str, assessment: HealthAssessment):
        """Handle health status changes and send events."""
        service_info = self.services[service_name]
        previous_assessment = service_info.get('last_assessment')
        
        # Check for status change
        if previous_assessment and previous_assessment.overall_status != assessment.overall_status:
            logger.info(f"Health status changed for {service_name}: "
                       f"{previous_assessment.overall_status.value} -> {assessment.overall_status.value}")
            
            # Send event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type="health_status_changed",
                    data={
                        'service_name': service_name,
                        'previous_status': previous_assessment.overall_status.value,
                        'new_status': assessment.overall_status.value,
                        'health_score': assessment.health_score,
                        'predictions': [p.__dict__ for p in assessment.failure_predictions],
                        'immediate_actions': assessment.immediate_actions,
                        'timestamp': assessment.timestamp.isoformat()
                    }
                ))
        
        # Send prediction alerts
        for prediction in assessment.failure_predictions:
            if prediction.probability > 0.7:
                if self.event_bus:
                    await self.event_bus.publish(Event(
                        type="failure_prediction",
                        data={
                            'service_name': service_name,
                            'prediction_type': prediction.prediction_type.value,
                            'probability': prediction.probability,
                            'time_to_failure': prediction.time_to_failure_seconds,
                            'risk_factors': prediction.risk_factors,
                            'recommended_actions': prediction.recommended_actions,
                            'timestamp': prediction.timestamp.isoformat()
                        }
                    ))
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Assess all registered services
                for service_name in self.services.keys():
                    await self.assess_service_health(service_name)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _prediction_loop(self):
        """Prediction and model training loop."""
        while self.running:
            try:
                # Train models periodically if enough data
                if ML_AVAILABLE and len(self.failure_detector.training_data) >= 100:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.failure_detector.train_models
                    )
                
                await asyncio.sleep(self.prediction_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(self.prediction_interval)
    
    def get_service_health(self, service_name: str) -> Optional[HealthAssessment]:
        """Get latest health assessment for a service."""
        if service_name in self.services:
            return self.services[service_name]['last_assessment']
        return None
    
    def get_all_services_health(self) -> Dict[str, HealthAssessment]:
        """Get health assessments for all services."""
        result = {}
        for service_name, service_info in self.services.items():
            if service_info['last_assessment']:
                result[service_name] = service_info['last_assessment']
        return result
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction system statistics."""
        return {
            'model_trained': self.failure_detector.is_trained,
            'training_data_points': len(self.failure_detector.training_data),
            'recent_predictions': len(self.failure_detector.prediction_history),
            'ml_available': ML_AVAILABLE
        }
    
    def add_training_data(self, service_name: str, failure_occurred: bool):
        """Add training data for a service."""
        if service_name in self.services:
            last_assessment = self.services[service_name]['last_assessment']
            if last_assessment:
                self.failure_detector.add_training_data(
                    last_assessment.current_metrics, failure_occurred
                )


# Global enhanced health monitor instance
enhanced_health_monitor = None


def get_enhanced_health_monitor() -> EnhancedHealthMonitor:
    """Get global enhanced health monitor instance."""
    global enhanced_health_monitor
    
    if enhanced_health_monitor is None:
        enhanced_health_monitor = EnhancedHealthMonitor()
    
    return enhanced_health_monitor