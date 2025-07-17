"""
Adaptive Circuit Breaker with Machine Learning
==============================================

Advanced circuit breaker that uses machine learning to adapt its behavior
based on service patterns, failure types, and environmental conditions.

Features:
- ML-based failure prediction
- Adaptive threshold adjustment
- Pattern recognition for different failure modes
- Self-tuning parameters based on historical data
- Integration with time-series forecasting
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, FailureType
from ..event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive circuit breaker."""
    # ML model parameters
    model_update_interval: int = 300  # 5 minutes
    feature_window_size: int = 100
    prediction_confidence_threshold: float = 0.7
    
    # Adaptation parameters
    adaptation_rate: float = 0.1
    min_adaptation_samples: int = 50
    max_threshold_adjustment: float = 0.5
    
    # Time-based adaptation
    time_of_day_adjustment: bool = True
    seasonal_adjustment: bool = True
    load_based_adjustment: bool = True
    
    # Performance monitoring
    track_prediction_accuracy: bool = True
    accuracy_threshold: float = 0.8
    
    # Advanced features
    enable_anomaly_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_cascade_failure_detection: bool = True


class AdaptiveCircuitBreaker(CircuitBreaker):
    """
    Adaptive circuit breaker with machine learning capabilities.
    
    This circuit breaker learns from historical data to:
    1. Predict failures before they occur
    2. Adapt thresholds based on service patterns
    3. Recognize different failure modes
    4. Adjust behavior based on time and load
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        adaptive_config: AdaptiveConfig,
        event_bus: Optional[EventBus] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """Initialize adaptive circuit breaker."""
        super().__init__(config, event_bus, redis_client)
        self.adaptive_config = adaptive_config
        
        # ML components
        self.ml_model: Optional[RandomForestClassifier] = None
        self.feature_scaler = StandardScaler()
        self.feature_buffer: List[Dict[str, Any]] = []
        self.prediction_buffer: List[Dict[str, Any]] = []
        
        # Adaptation tracking
        self.original_thresholds = {
            'failure_threshold': config.failure_threshold,
            'timeout_seconds': config.timeout_seconds,
            'success_threshold': config.success_threshold
        }
        
        # Pattern recognition
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.load_patterns: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.prediction_accuracy_history: List[float] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self.ml_update_task: Optional[asyncio.Task] = None
        self.pattern_analysis_task: Optional[asyncio.Task] = None
        
        logger.info(f"Adaptive circuit breaker initialized for {config.service_name}")
    
    async def initialize(self):
        """Initialize adaptive circuit breaker."""
        await super().initialize()
        
        # Load existing ML model if available
        await self._load_ml_model()
        
        # Start adaptive tasks
        self.ml_update_task = asyncio.create_task(self._ml_update_loop())
        self.pattern_analysis_task = asyncio.create_task(self._pattern_analysis_loop())
        
        logger.info(f"Adaptive circuit breaker fully initialized for {self.config.service_name}")
    
    async def close(self):
        """Close adaptive circuit breaker."""
        # Cancel adaptive tasks
        if self.ml_update_task:
            self.ml_update_task.cancel()
        if self.pattern_analysis_task:
            self.pattern_analysis_task.cancel()
        
        # Save ML model
        await self._save_ml_model()
        
        await super().close()
    
    async def _record_success(self, execution_time: float):
        """Record successful execution with ML feature extraction."""
        await super()._record_success(execution_time)
        
        # Extract features for ML
        features = await self._extract_features(success=True, execution_time=execution_time)
        self.feature_buffer.append(features)
        
        # Limit buffer size
        if len(self.feature_buffer) > self.adaptive_config.feature_window_size:
            self.feature_buffer.pop(0)
        
        # Update load patterns
        await self._update_load_patterns(execution_time)
    
    async def _record_failure(self, exception: Exception, execution_time: float):
        """Record failed execution with ML feature extraction."""
        await super()._record_failure(exception, execution_time)
        
        # Extract features for ML
        features = await self._extract_features(success=False, execution_time=execution_time, exception=exception)
        self.feature_buffer.append(features)
        
        # Limit buffer size
        if len(self.feature_buffer) > self.adaptive_config.feature_window_size:
            self.feature_buffer.pop(0)
        
        # Update failure patterns
        failure_type = self._classify_failure(exception)
        await self._update_failure_patterns(failure_type, features)
        
        # Trigger adaptive adjustment
        await self._trigger_adaptive_adjustment(failure_type)
    
    async def _extract_features(
        self, 
        success: bool, 
        execution_time: float, 
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Extract features for ML model."""
        current_time = time.time()
        
        # Basic features
        features = {
            'success': success,
            'execution_time': execution_time,
            'timestamp': current_time,
            'hour_of_day': datetime.fromtimestamp(current_time).hour,
            'day_of_week': datetime.fromtimestamp(current_time).weekday(),
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'current_state': self.state.value,
            'failure_rate': self.metrics.failure_rate,
            'average_response_time': self.metrics.average_response_time,
            'time_since_last_failure': current_time - self.last_failure_time if self.last_failure_time > 0 else 0,
            'time_since_last_success': current_time - self.last_success_time if self.last_success_time > 0 else 0,
            'request_count_last_minute': len([
                req for req in self.request_history 
                if req['timestamp'] > current_time - 60
            ])
        }
        
        # Failure-specific features
        if not success and exception:
            failure_type = self._classify_failure(exception)
            features['failure_type'] = failure_type.value
            features['exception_length'] = len(str(exception))
            features['is_timeout'] = failure_type == FailureType.TIMEOUT
            features['is_connection_error'] = failure_type == FailureType.CONNECTION_ERROR
            features['is_critical'] = failure_type == FailureType.CRITICAL_ERROR
        
        # Performance features
        if len(self.performance_history) > 0:
            features['performance_trend'] = self._calculate_performance_trend()
            features['performance_variance'] = np.var(self.performance_history[-20:])
        
        # Load-based features
        features['current_load'] = await self._calculate_current_load()
        features['load_trend'] = await self._calculate_load_trend()
        
        return features
    
    async def _calculate_current_load(self) -> float:
        """Calculate current system load."""
        # Count requests in last minute
        current_time = time.time()
        recent_requests = [
            req for req in self.request_history 
            if req['timestamp'] > current_time - 60
        ]
        
        return len(recent_requests) / 60.0  # requests per second
    
    async def _calculate_load_trend(self) -> float:
        """Calculate load trend over time."""
        if len(self.load_patterns) < 2:
            return 0.0
        
        recent_loads = [pattern['load'] for pattern in self.load_patterns[-10:]]
        if len(recent_loads) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_loads))
        y = np.array(recent_loads)
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend."""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_times = self.performance_history[-10:]
        x = np.arange(len(recent_times))
        y = np.array(recent_times)
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    async def _update_load_patterns(self, execution_time: float):
        """Update load patterns for analysis."""
        current_time = time.time()
        current_load = await self._calculate_current_load()
        
        self.load_patterns.append({
            'timestamp': current_time,
            'load': current_load,
            'execution_time': execution_time,
            'hour': datetime.fromtimestamp(current_time).hour,
            'day_of_week': datetime.fromtimestamp(current_time).weekday()
        })
        
        # Limit pattern history
        if len(self.load_patterns) > 1000:
            self.load_patterns.pop(0)
    
    async def _update_failure_patterns(self, failure_type: FailureType, features: Dict[str, Any]):
        """Update failure patterns for analysis."""
        pattern_key = failure_type.value
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []
        
        self.failure_patterns[pattern_key].append(features)
        
        # Limit pattern history
        if len(self.failure_patterns[pattern_key]) > 100:
            self.failure_patterns[pattern_key].pop(0)
    
    async def _trigger_adaptive_adjustment(self, failure_type: FailureType):
        """Trigger adaptive threshold adjustment based on failure patterns."""
        if len(self.feature_buffer) < self.adaptive_config.min_adaptation_samples:
            return
        
        # Analyze recent failure patterns
        recent_failures = [
            f for f in self.feature_buffer[-20:] 
            if not f['success']
        ]
        
        if len(recent_failures) < 3:
            return
        
        # Calculate adaptation recommendations
        adjustments = await self._calculate_adaptive_adjustments(failure_type, recent_failures)
        
        # Apply adjustments
        await self._apply_adaptive_adjustments(adjustments)
    
    async def _calculate_adaptive_adjustments(
        self, 
        failure_type: FailureType, 
        recent_failures: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate adaptive adjustments based on failure patterns."""
        adjustments = {}
        
        # Analyze failure frequency
        failure_rate = len(recent_failures) / len(self.feature_buffer[-20:])
        
        if failure_rate > 0.3:  # High failure rate
            # Decrease failure threshold to open circuit sooner
            adjustment = -self.adaptive_config.adaptation_rate * failure_rate
            adjustments['failure_threshold'] = max(
                adjustment, 
                -self.adaptive_config.max_threshold_adjustment
            )
        
        # Analyze execution time patterns
        if failure_type == FailureType.TIMEOUT:
            avg_execution_time = np.mean([f['execution_time'] for f in recent_failures])
            if avg_execution_time > self.config.timeout_seconds * 0.8:
                # Decrease timeout threshold
                adjustment = -self.adaptive_config.adaptation_rate
                adjustments['timeout_seconds'] = max(
                    adjustment * self.config.timeout_seconds,
                    -self.adaptive_config.max_threshold_adjustment * self.config.timeout_seconds
                )
        
        # Time-based adjustments
        if self.adaptive_config.time_of_day_adjustment:
            hour = datetime.now().hour
            if 9 <= hour <= 17:  # Business hours - be more conservative
                adjustments['failure_threshold'] = adjustments.get('failure_threshold', 0) - 0.1
        
        # Load-based adjustments
        if self.adaptive_config.load_based_adjustment:
            current_load = await self._calculate_current_load()
            if current_load > 10:  # High load - be more conservative
                adjustments['failure_threshold'] = adjustments.get('failure_threshold', 0) - 0.2
        
        return adjustments
    
    async def _apply_adaptive_adjustments(self, adjustments: Dict[str, float]):
        """Apply adaptive adjustments to circuit breaker configuration."""
        for parameter, adjustment in adjustments.items():
            if parameter == 'failure_threshold':
                new_value = max(1, int(self.config.failure_threshold + adjustment))
                if new_value != self.config.failure_threshold:
                    logger.info(f"Adapting failure_threshold: {self.config.failure_threshold} -> {new_value}")
                    self.config.failure_threshold = new_value
            
            elif parameter == 'timeout_seconds':
                new_value = max(5.0, self.config.timeout_seconds + adjustment)
                if abs(new_value - self.config.timeout_seconds) > 0.1:
                    logger.info(f"Adapting timeout_seconds: {self.config.timeout_seconds} -> {new_value}")
                    self.config.timeout_seconds = new_value
            
            elif parameter == 'success_threshold':
                new_value = max(1, int(self.config.success_threshold + adjustment))
                if new_value != self.config.success_threshold:
                    logger.info(f"Adapting success_threshold: {self.config.success_threshold} -> {new_value}")
                    self.config.success_threshold = new_value
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': time.time(),
            'adjustments': adjustments,
            'config_state': {
                'failure_threshold': self.config.failure_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'success_threshold': self.config.success_threshold
            }
        })
        
        # Limit adaptation history
        if len(self.adaptation_history) > 100:
            self.adaptation_history.pop(0)
    
    async def _ml_update_loop(self):
        """Background task to update ML model."""
        while True:
            try:
                await asyncio.sleep(self.adaptive_config.model_update_interval)
                await self._update_ml_model()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ML update error: {e}")
    
    async def _pattern_analysis_loop(self):
        """Background task for pattern analysis."""
        while True:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                await self._analyze_patterns()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern analysis error: {e}")
    
    async def _update_ml_model(self):
        """Update ML model with recent data."""
        if len(self.feature_buffer) < self.adaptive_config.min_adaptation_samples:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 10:
                return
            
            # Train or update model
            if self.ml_model is None:
                self.ml_model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
            
            # Fit scaler and transform features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.ml_model.fit(X_scaled, y)
            
            # Evaluate model accuracy
            accuracy = self.ml_model.score(X_scaled, y)
            self.prediction_accuracy_history.append(accuracy)
            
            # Limit accuracy history
            if len(self.prediction_accuracy_history) > 100:
                self.prediction_accuracy_history.pop(0)
            
            logger.info(f"ML model updated for {self.config.service_name}, accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"ML model update failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model."""
        features = []
        labels = []
        
        # Create features and labels from buffer
        for i, sample in enumerate(self.feature_buffer[:-1]):  # Exclude last sample
            # Features
            feature_vector = [
                sample['execution_time'],
                sample['hour_of_day'],
                sample['day_of_week'],
                sample['failure_count'],
                sample['success_count'],
                sample['failure_rate'],
                sample['average_response_time'],
                sample['time_since_last_failure'],
                sample['time_since_last_success'],
                sample['request_count_last_minute'],
                sample.get('performance_trend', 0),
                sample.get('performance_variance', 0),
                sample.get('current_load', 0),
                sample.get('load_trend', 0)
            ]
            
            # Look ahead to determine if next request will fail
            next_sample = self.feature_buffer[i + 1]
            label = 1 if not next_sample['success'] else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    async def _analyze_patterns(self):
        """Analyze patterns in failure and load data."""
        # Analyze failure patterns
        if self.adaptive_config.enable_pattern_recognition:
            await self._analyze_failure_patterns()
        
        # Analyze load patterns
        await self._analyze_load_patterns()
        
        # Detect anomalies
        if self.adaptive_config.enable_anomaly_detection:
            await self._detect_anomalies()
    
    async def _analyze_failure_patterns(self):
        """Analyze failure patterns for insights."""
        for failure_type, patterns in self.failure_patterns.items():
            if len(patterns) < 10:
                continue
            
            # Analyze time-based patterns
            hours = [p['hour_of_day'] for p in patterns]
            most_common_hour = max(set(hours), key=hours.count)
            
            if hours.count(most_common_hour) > len(hours) * 0.3:
                logger.info(f"Pattern detected: {failure_type} failures common at hour {most_common_hour}")
            
            # Analyze load-based patterns
            loads = [p.get('current_load', 0) for p in patterns]
            avg_load = np.mean(loads)
            
            if avg_load > 5:
                logger.info(f"Pattern detected: {failure_type} failures correlate with high load ({avg_load:.2f})")
    
    async def _analyze_load_patterns(self):
        """Analyze load patterns for capacity planning."""
        if len(self.load_patterns) < 50:
            return
        
        # Analyze daily patterns
        hourly_loads = {}
        for pattern in self.load_patterns[-200:]:  # Last 200 samples
            hour = pattern['hour']
            if hour not in hourly_loads:
                hourly_loads[hour] = []
            hourly_loads[hour].append(pattern['load'])
        
        # Find peak hours
        peak_hours = []
        for hour, loads in hourly_loads.items():
            if len(loads) > 5 and np.mean(loads) > 5:
                peak_hours.append(hour)
        
        if peak_hours:
            logger.info(f"Peak load hours detected: {peak_hours}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in system behavior."""
        if len(self.feature_buffer) < 50:
            return
        
        # Analyze execution time anomalies
        recent_times = [f['execution_time'] for f in self.feature_buffer[-50:]]
        mean_time = np.mean(recent_times)
        std_time = np.std(recent_times)
        
        # Check for outliers
        outliers = [t for t in recent_times if abs(t - mean_time) > 3 * std_time]
        
        if len(outliers) > 5:
            logger.warning(f"Anomaly detected: {len(outliers)} execution time outliers")
        
        # Analyze failure rate anomalies
        recent_failures = [f['success'] for f in self.feature_buffer[-50:]]
        failure_rate = (len(recent_failures) - sum(recent_failures)) / len(recent_failures)
        
        if failure_rate > 0.2:  # 20% failure rate
            logger.warning(f"Anomaly detected: High failure rate {failure_rate:.2f}")
    
    async def _get_ml_prediction(self) -> float:
        """Get ML-based failure prediction."""
        if self.ml_model is None or len(self.feature_buffer) == 0:
            return 0.0
        
        try:
            # Get latest features
            latest_features = self.feature_buffer[-1]
            
            # Prepare feature vector
            feature_vector = np.array([[
                latest_features['execution_time'],
                latest_features['hour_of_day'],
                latest_features['day_of_week'],
                latest_features['failure_count'],
                latest_features['success_count'],
                latest_features['failure_rate'],
                latest_features['average_response_time'],
                latest_features['time_since_last_failure'],
                latest_features['time_since_last_success'],
                latest_features['request_count_last_minute'],
                latest_features.get('performance_trend', 0),
                latest_features.get('performance_variance', 0),
                latest_features.get('current_load', 0),
                latest_features.get('load_trend', 0)
            ]])
            
            # Scale features
            feature_vector_scaled = self.feature_scaler.transform(feature_vector)
            
            # Get prediction probability
            prediction_proba = self.ml_model.predict_proba(feature_vector_scaled)[0]
            
            # Return probability of failure (class 1)
            return prediction_proba[1] if len(prediction_proba) > 1 else 0.0
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 0.0
    
    async def _load_ml_model(self):
        """Load existing ML model from storage."""
        try:
            if self.redis_client:
                model_key = f"ml_model:{self.config.service_name}"
                model_data = await self.redis_client.get(model_key)
                
                if model_data:
                    # In production, you would deserialize the actual model
                    # For now, we'll initialize a new model
                    self.ml_model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=10,
                        random_state=42
                    )
                    logger.info(f"ML model loaded for {self.config.service_name}")
                
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
    
    async def _save_ml_model(self):
        """Save ML model to storage."""
        try:
            if self.ml_model and self.redis_client:
                model_key = f"ml_model:{self.config.service_name}"
                
                # In production, you would serialize the actual model
                # For now, we'll save metadata
                model_metadata = {
                    'accuracy_history': self.prediction_accuracy_history,
                    'adaptation_history': self.adaptation_history,
                    'timestamp': time.time()
                }
                
                await self.redis_client.set(
                    model_key,
                    json.dumps(model_metadata),
                    ex=86400  # 24 hours
                )
                
                logger.info(f"ML model saved for {self.config.service_name}")
                
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get adaptive circuit breaker status."""
        base_status = self.get_status()
        
        # Add adaptive-specific information
        base_status.update({
            'adaptive_features': {
                'ml_model_trained': self.ml_model is not None,
                'prediction_accuracy': (
                    np.mean(self.prediction_accuracy_history) 
                    if self.prediction_accuracy_history else 0.0
                ),
                'feature_buffer_size': len(self.feature_buffer),
                'adaptation_count': len(self.adaptation_history),
                'original_thresholds': self.original_thresholds,
                'threshold_adjustments': {
                    'failure_threshold': self.config.failure_threshold - self.original_thresholds['failure_threshold'],
                    'timeout_seconds': self.config.timeout_seconds - self.original_thresholds['timeout_seconds'],
                    'success_threshold': self.config.success_threshold - self.original_thresholds['success_threshold']
                }
            },
            'pattern_insights': {
                'failure_patterns': {
                    pattern_type: len(patterns) 
                    for pattern_type, patterns in self.failure_patterns.items()
                },
                'load_patterns_count': len(self.load_patterns),
                'anomaly_detection_enabled': self.adaptive_config.enable_anomaly_detection
            }
        })
        
        return base_status