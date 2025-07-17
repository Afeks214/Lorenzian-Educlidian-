#!/usr/bin/env python3
"""
Adaptive Threshold Monitoring System
Dynamic threshold adjustment based on system behavior and performance patterns
"""

import asyncio
import numpy as np
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import pickle
import redis
from concurrent.futures import ThreadPoolExecutor

# Statistical analysis
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Monitoring imports
from .superposition_monitoring import SuperpositionMeasurement
from .alerting_system import Alert, AlertType, AlertSeverity

# Metrics
from prometheus_client import Counter, Histogram, Gauge
THRESHOLD_ADAPTATIONS = Counter('threshold_adaptations_total', 'Total threshold adaptations', ['metric_name', 'direction'])
THRESHOLD_ACCURACY = Gauge('threshold_accuracy', 'Threshold accuracy score', ['metric_name'])
ADAPTIVE_THRESHOLD_VALUE = Gauge('adaptive_threshold_value', 'Current adaptive threshold value', ['metric_name', 'threshold_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdType(Enum):
    """Types of thresholds."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"

class AdaptationStrategy(Enum):
    """Threshold adaptation strategies."""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"

class ThresholdDirection(Enum):
    """Threshold direction."""
    UPPER = "upper"
    LOWER = "lower"
    BIDIRECTIONAL = "bidirectional"

@dataclass
class ThresholdConfiguration:
    """Threshold configuration."""
    metric_name: str
    threshold_type: ThresholdType
    adaptation_strategy: AdaptationStrategy
    direction: ThresholdDirection
    initial_value: float
    min_value: float
    max_value: float
    sensitivity: float
    adaptation_rate: float
    lookback_window: int
    confidence_level: float
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ThresholdState:
    """Current threshold state."""
    config: ThresholdConfiguration
    current_value: float
    last_adaptation: datetime
    adaptation_count: int
    accuracy_score: float
    recent_values: deque
    statistical_properties: Dict[str, float]
    ml_model: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'current_value': self.current_value,
            'last_adaptation': self.last_adaptation.isoformat(),
            'adaptation_count': self.adaptation_count,
            'accuracy_score': self.accuracy_score,
            'recent_values': list(self.recent_values),
            'statistical_properties': self.statistical_properties
        }

class StatisticalThresholdAdapter:
    """Statistical threshold adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.stability_threshold = config.get('stability_threshold', 0.1)
        
    def adapt_threshold(self, threshold_state: ThresholdState, recent_data: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Adapt threshold using statistical methods."""
        if len(recent_data) < 10:
            return threshold_state.current_value, {'reason': 'insufficient_data'}
        
        try:
            # Calculate statistical properties
            mean = np.mean(recent_data)
            std = np.std(recent_data)
            median = np.median(recent_data)
            
            # Calculate percentiles
            p25, p75 = np.percentile(recent_data, [25, 75])
            iqr = p75 - p25
            
            # Detect outliers using IQR method
            outlier_lower = p25 - 1.5 * iqr
            outlier_upper = p75 + 1.5 * iqr
            
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(recent_data))
            outlier_count = np.sum(z_scores > self.outlier_threshold)
            
            # Determine new threshold based on direction
            config = threshold_state.config
            new_threshold = threshold_state.current_value
            adaptation_info = {'method': 'statistical'}
            
            if config.direction == ThresholdDirection.UPPER:
                # Upper threshold adaptation
                if config.confidence_level:
                    # Use confidence interval
                    confidence_interval = stats.t.interval(
                        config.confidence_level,
                        len(recent_data) - 1,
                        loc=mean,
                        scale=stats.sem(recent_data)
                    )
                    new_threshold = confidence_interval[1]
                else:
                    # Use percentile-based approach
                    new_threshold = np.percentile(recent_data, 95)
                
            elif config.direction == ThresholdDirection.LOWER:
                # Lower threshold adaptation
                if config.confidence_level:
                    confidence_interval = stats.t.interval(
                        config.confidence_level,
                        len(recent_data) - 1,
                        loc=mean,
                        scale=stats.sem(recent_data)
                    )
                    new_threshold = confidence_interval[0]
                else:
                    new_threshold = np.percentile(recent_data, 5)
            
            elif config.direction == ThresholdDirection.BIDIRECTIONAL:
                # Bidirectional threshold (anomaly detection)
                new_threshold = mean + config.sensitivity * std
            
            # Apply adaptation rate
            if config.adaptation_rate < 1.0:
                new_threshold = (
                    threshold_state.current_value * (1 - config.adaptation_rate) +
                    new_threshold * config.adaptation_rate
                )
            
            # Ensure bounds
            new_threshold = max(config.min_value, min(config.max_value, new_threshold))
            
            # Update statistical properties
            adaptation_info.update({
                'mean': mean,
                'std': std,
                'median': median,
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_count / len(recent_data),
                'confidence_interval': confidence_interval if config.confidence_level else None
            })
            
            return new_threshold, adaptation_info
            
        except Exception as e:
            logger.error(f"Error in statistical threshold adaptation: {e}")
            return threshold_state.current_value, {'error': str(e)}

class MachineLearningThresholdAdapter:
    """Machine learning based threshold adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('model_type', 'isolation_forest')
        self.contamination = config.get('contamination', 0.1)
        self.min_samples = config.get('min_samples', 50)
        
    def adapt_threshold(self, threshold_state: ThresholdState, recent_data: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Adapt threshold using machine learning methods."""
        if len(recent_data) < self.min_samples:
            return threshold_state.current_value, {'reason': 'insufficient_data'}
        
        try:
            # Prepare data
            X = np.array(recent_data).reshape(-1, 1)
            
            # Feature engineering
            features = self._engineer_features(recent_data)
            
            # Train or update model
            if self.model_type == 'isolation_forest':
                model = IsolationForest(contamination=self.contamination, random_state=42)
                model.fit(features)
                
                # Get anomaly scores
                anomaly_scores = model.decision_function(features)
                
                # Calculate threshold based on anomaly scores
                threshold_percentile = (1 - self.contamination) * 100
                new_threshold = np.percentile(anomaly_scores, threshold_percentile)
                
            elif self.model_type == 'statistical_clustering':
                # Use clustering to identify normal vs anomalous regions
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Perform clustering
                n_clusters = min(5, len(recent_data) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Calculate distances to cluster centers
                distances = []
                for i, label in enumerate(cluster_labels):
                    center = kmeans.cluster_centers_[label]
                    distance = np.linalg.norm(scaled_features[i] - center)
                    distances.append(distance)
                
                # Set threshold based on distance distribution
                new_threshold = np.percentile(distances, 95)
                
            else:
                # Fallback to statistical method
                new_threshold = np.percentile(recent_data, 95)
            
            # Apply adaptation rate
            config = threshold_state.config
            if config.adaptation_rate < 1.0:
                new_threshold = (
                    threshold_state.current_value * (1 - config.adaptation_rate) +
                    new_threshold * config.adaptation_rate
                )
            
            # Ensure bounds
            new_threshold = max(config.min_value, min(config.max_value, new_threshold))
            
            # Store model
            threshold_state.ml_model = model if self.model_type == 'isolation_forest' else kmeans
            
            adaptation_info = {
                'method': 'machine_learning',
                'model_type': self.model_type,
                'contamination': self.contamination,
                'feature_count': features.shape[1],
                'anomaly_score_mean': np.mean(anomaly_scores) if self.model_type == 'isolation_forest' else None
            }
            
            return new_threshold, adaptation_info
            
        except Exception as e:
            logger.error(f"Error in ML threshold adaptation: {e}")
            return threshold_state.current_value, {'error': str(e)}
    
    def _engineer_features(self, data: List[float]) -> np.ndarray:
        """Engineer features for ML model."""
        try:
            data_array = np.array(data)
            
            # Basic statistical features
            features = []
            window_size = min(10, len(data) // 5)
            
            for i in range(window_size, len(data)):
                window = data_array[i-window_size:i]
                
                feature_vector = [
                    np.mean(window),
                    np.std(window),
                    np.min(window),
                    np.max(window),
                    np.median(window),
                    stats.skew(window),
                    stats.kurtosis(window)
                ]
                
                # Trend features
                if len(window) > 2:
                    slope, _ = np.polyfit(range(len(window)), window, 1)
                    feature_vector.append(slope)
                else:
                    feature_vector.append(0)
                
                # Volatility features
                if len(window) > 1:
                    volatility = np.std(np.diff(window))
                    feature_vector.append(volatility)
                else:
                    feature_vector.append(0)
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return np.array(data).reshape(-1, 1)

class RuleBasedThresholdAdapter:
    """Rule-based threshold adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = config.get('rules', [])
        
    def adapt_threshold(self, threshold_state: ThresholdState, recent_data: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Adapt threshold using rule-based methods."""
        if len(recent_data) < 5:
            return threshold_state.current_value, {'reason': 'insufficient_data'}
        
        try:
            config = threshold_state.config
            current_threshold = threshold_state.current_value
            applied_rules = []
            
            # Calculate recent statistics
            recent_mean = np.mean(recent_data[-10:])
            recent_std = np.std(recent_data[-10:])
            recent_trend = self._calculate_trend(recent_data[-10:])
            
            # Apply rules
            for rule in self.rules:
                rule_type = rule.get('type')
                condition = rule.get('condition')
                action = rule.get('action')
                
                if rule_type == 'trend_based':
                    if self._evaluate_trend_condition(recent_trend, condition):
                        current_threshold = self._apply_action(current_threshold, action, recent_data)
                        applied_rules.append(rule_type)
                
                elif rule_type == 'volatility_based':
                    volatility = recent_std / recent_mean if recent_mean > 0 else 0
                    if self._evaluate_volatility_condition(volatility, condition):
                        current_threshold = self._apply_action(current_threshold, action, recent_data)
                        applied_rules.append(rule_type)
                
                elif rule_type == 'violation_based':
                    violations = sum(1 for x in recent_data[-10:] if x > current_threshold)
                    violation_rate = violations / len(recent_data[-10:])
                    if self._evaluate_violation_condition(violation_rate, condition):
                        current_threshold = self._apply_action(current_threshold, action, recent_data)
                        applied_rules.append(rule_type)
                
                elif rule_type == 'time_based':
                    time_since_adaptation = (datetime.utcnow() - threshold_state.last_adaptation).total_seconds()
                    if self._evaluate_time_condition(time_since_adaptation, condition):
                        current_threshold = self._apply_action(current_threshold, action, recent_data)
                        applied_rules.append(rule_type)
            
            # Ensure bounds
            current_threshold = max(config.min_value, min(config.max_value, current_threshold))
            
            adaptation_info = {
                'method': 'rule_based',
                'applied_rules': applied_rules,
                'recent_mean': recent_mean,
                'recent_std': recent_std,
                'recent_trend': recent_trend
            }
            
            return current_threshold, adaptation_info
            
        except Exception as e:
            logger.error(f"Error in rule-based threshold adaptation: {e}")
            return threshold_state.current_value, {'error': str(e)}
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend slope."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _evaluate_trend_condition(self, trend: float, condition: Dict[str, Any]) -> bool:
        """Evaluate trend condition."""
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)
        
        if operator == '>':
            return trend > value
        elif operator == '<':
            return trend < value
        elif operator == '>=':
            return trend >= value
        elif operator == '<=':
            return trend <= value
        elif operator == '==':
            return abs(trend - value) < 1e-6
        
        return False
    
    def _evaluate_volatility_condition(self, volatility: float, condition: Dict[str, Any]) -> bool:
        """Evaluate volatility condition."""
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)
        
        if operator == '>':
            return volatility > value
        elif operator == '<':
            return volatility < value
        elif operator == '>=':
            return volatility >= value
        elif operator == '<=':
            return volatility <= value
        
        return False
    
    def _evaluate_violation_condition(self, violation_rate: float, condition: Dict[str, Any]) -> bool:
        """Evaluate violation condition."""
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)
        
        if operator == '>':
            return violation_rate > value
        elif operator == '<':
            return violation_rate < value
        elif operator == '>=':
            return violation_rate >= value
        elif operator == '<=':
            return violation_rate <= value
        
        return False
    
    def _evaluate_time_condition(self, time_elapsed: float, condition: Dict[str, Any]) -> bool:
        """Evaluate time condition."""
        operator = condition.get('operator', '>')
        value = condition.get('value', 0)  # seconds
        
        if operator == '>':
            return time_elapsed > value
        elif operator == '<':
            return time_elapsed < value
        elif operator == '>=':
            return time_elapsed >= value
        elif operator == '<=':
            return time_elapsed <= value
        
        return False
    
    def _apply_action(self, current_threshold: float, action: Dict[str, Any], recent_data: List[float]) -> float:
        """Apply threshold adjustment action."""
        action_type = action.get('type')
        
        if action_type == 'increase':
            factor = action.get('factor', 1.1)
            return current_threshold * factor
        
        elif action_type == 'decrease':
            factor = action.get('factor', 0.9)
            return current_threshold * factor
        
        elif action_type == 'set_percentile':
            percentile = action.get('percentile', 95)
            return np.percentile(recent_data, percentile)
        
        elif action_type == 'set_mean_plus_std':
            multiplier = action.get('multiplier', 2)
            return np.mean(recent_data) + multiplier * np.std(recent_data)
        
        elif action_type == 'set_value':
            return action.get('value', current_threshold)
        
        return current_threshold

class AdaptiveThresholdMonitor:
    """Adaptive threshold monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(**config.get('redis', {}))
        self.threshold_states = {}
        self.adapters = {}
        self.monitoring_active = False
        
        # Initialize adapters
        self._initialize_adapters()
        
        # Load threshold configurations
        self._load_threshold_configurations()
        
        # Performance tracking
        self.adaptation_history = deque(maxlen=1000)
        self.accuracy_tracking = {}
        
    def _initialize_adapters(self):
        """Initialize threshold adapters."""
        adapter_config = self.config.get('adapters', {})
        
        if 'statistical' in adapter_config:
            self.adapters[AdaptationStrategy.STATISTICAL] = StatisticalThresholdAdapter(
                adapter_config['statistical']
            )
        
        if 'machine_learning' in adapter_config:
            self.adapters[AdaptationStrategy.MACHINE_LEARNING] = MachineLearningThresholdAdapter(
                adapter_config['machine_learning']
            )
        
        if 'rule_based' in adapter_config:
            self.adapters[AdaptationStrategy.RULE_BASED] = RuleBasedThresholdAdapter(
                adapter_config['rule_based']
            )
    
    def _load_threshold_configurations(self):
        """Load threshold configurations."""
        threshold_configs = self.config.get('thresholds', [])
        
        for config_dict in threshold_configs:
            config = ThresholdConfiguration(
                metric_name=config_dict['metric_name'],
                threshold_type=ThresholdType(config_dict.get('threshold_type', 'adaptive')),
                adaptation_strategy=AdaptationStrategy(config_dict.get('adaptation_strategy', 'statistical')),
                direction=ThresholdDirection(config_dict.get('direction', 'upper')),
                initial_value=config_dict['initial_value'],
                min_value=config_dict.get('min_value', 0),
                max_value=config_dict.get('max_value', float('inf')),
                sensitivity=config_dict.get('sensitivity', 1.0),
                adaptation_rate=config_dict.get('adaptation_rate', 0.1),
                lookback_window=config_dict.get('lookback_window', 100),
                confidence_level=config_dict.get('confidence_level', 0.95),
                enabled=config_dict.get('enabled', True)
            )
            
            # Create threshold state
            threshold_state = ThresholdState(
                config=config,
                current_value=config.initial_value,
                last_adaptation=datetime.utcnow(),
                adaptation_count=0,
                accuracy_score=0.0,
                recent_values=deque(maxlen=config.lookback_window),
                statistical_properties={}
            )
            
            self.threshold_states[config.metric_name] = threshold_state
    
    async def start_monitoring(self):
        """Start adaptive threshold monitoring."""
        self.monitoring_active = True
        logger.info("Starting adaptive threshold monitoring")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._threshold_adaptation_loop()),
            asyncio.create_task(self._accuracy_tracking_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Stopping adaptive threshold monitoring")
    
    async def _threshold_adaptation_loop(self):
        """Main threshold adaptation loop."""
        while self.monitoring_active:
            try:
                for metric_name, threshold_state in self.threshold_states.items():
                    if not threshold_state.config.enabled:
                        continue
                    
                    # Check if adaptation is needed
                    if await self._should_adapt_threshold(threshold_state):
                        await self._adapt_threshold(threshold_state)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in threshold adaptation loop: {e}")
                await asyncio.sleep(60)
    
    async def _accuracy_tracking_loop(self):
        """Track threshold accuracy."""
        while self.monitoring_active:
            try:
                for metric_name, threshold_state in self.threshold_states.items():
                    accuracy = await self._calculate_threshold_accuracy(threshold_state)
                    threshold_state.accuracy_score = accuracy
                    
                    # Update Prometheus metrics
                    THRESHOLD_ACCURACY.labels(metric_name=metric_name).set(accuracy)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in accuracy tracking loop: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_loop(self):
        """Cleanup old data."""
        while self.monitoring_active:
            try:
                # Clean up old adaptation history
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=24)
                
                # Remove old entries
                while (self.adaptation_history and 
                       self.adaptation_history[0].get('timestamp', current_time) < cutoff_time):
                    self.adaptation_history.popleft()
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _should_adapt_threshold(self, threshold_state: ThresholdState) -> bool:
        """Check if threshold should be adapted."""
        try:
            # Check if enough time has passed since last adaptation
            time_since_adaptation = (datetime.utcnow() - threshold_state.last_adaptation).total_seconds()
            min_adaptation_interval = 300  # 5 minutes
            
            if time_since_adaptation < min_adaptation_interval:
                return False
            
            # Check if we have enough data
            if len(threshold_state.recent_values) < 10:
                return False
            
            # Check if there's significant change in data distribution
            if len(threshold_state.recent_values) >= 20:
                recent_half = list(threshold_state.recent_values)[-10:]
                earlier_half = list(threshold_state.recent_values)[-20:-10]
                
                # Perform statistical test
                statistic, p_value = stats.ks_2samp(recent_half, earlier_half)
                
                # Adapt if distributions are significantly different
                if p_value < 0.05:  # 5% significance level
                    return True
            
            # Check violation rate
            violations = sum(1 for x in threshold_state.recent_values 
                           if x > threshold_state.current_value)
            violation_rate = violations / len(threshold_state.recent_values)
            
            # Adapt if violation rate is too high or too low
            if violation_rate > 0.15 or violation_rate < 0.02:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if threshold should adapt: {e}")
            return False
    
    async def _adapt_threshold(self, threshold_state: ThresholdState):
        """Adapt threshold based on strategy."""
        try:
            config = threshold_state.config
            recent_data = list(threshold_state.recent_values)
            
            # Get appropriate adapter
            adapter = self.adapters.get(config.adaptation_strategy)
            if not adapter:
                logger.warning(f"No adapter found for strategy: {config.adaptation_strategy}")
                return
            
            # Adapt threshold
            old_value = threshold_state.current_value
            new_value, adaptation_info = adapter.adapt_threshold(threshold_state, recent_data)
            
            # Update threshold state
            threshold_state.current_value = new_value
            threshold_state.last_adaptation = datetime.utcnow()
            threshold_state.adaptation_count += 1
            
            # Record adaptation
            adaptation_record = {
                'timestamp': datetime.utcnow(),
                'metric_name': config.metric_name,
                'old_value': old_value,
                'new_value': new_value,
                'adaptation_info': adaptation_info,
                'strategy': config.adaptation_strategy.value
            }
            
            self.adaptation_history.append(adaptation_record)
            
            # Update Prometheus metrics
            direction = 'increase' if new_value > old_value else 'decrease'
            THRESHOLD_ADAPTATIONS.labels(
                metric_name=config.metric_name,
                direction=direction
            ).inc()
            
            ADAPTIVE_THRESHOLD_VALUE.labels(
                metric_name=config.metric_name,
                threshold_type=config.threshold_type.value
            ).set(new_value)
            
            # Store in Redis
            await self._store_threshold_state(threshold_state)
            
            logger.info(f"Adapted threshold for {config.metric_name}: {old_value:.4f} -> {new_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error adapting threshold: {e}")
    
    async def _calculate_threshold_accuracy(self, threshold_state: ThresholdState) -> float:
        """Calculate threshold accuracy."""
        try:
            if len(threshold_state.recent_values) < 10:
                return 0.0
            
            # Calculate expected violation rate based on threshold type
            config = threshold_state.config
            recent_data = list(threshold_state.recent_values)
            current_threshold = threshold_state.current_value
            
            violations = sum(1 for x in recent_data if x > current_threshold)
            violation_rate = violations / len(recent_data)
            
            # Expected violation rate (target)
            if config.direction == ThresholdDirection.UPPER:
                target_violation_rate = 0.05  # 5% expected violations
            elif config.direction == ThresholdDirection.LOWER:
                target_violation_rate = 0.05  # 5% expected violations
            else:
                target_violation_rate = 0.10  # 10% expected violations for bidirectional
            
            # Calculate accuracy as inverse of deviation from target
            deviation = abs(violation_rate - target_violation_rate)
            accuracy = max(0.0, 1.0 - (deviation / target_violation_rate))
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating threshold accuracy: {e}")
            return 0.0
    
    async def _store_threshold_state(self, threshold_state: ThresholdState):
        """Store threshold state in Redis."""
        try:
            key = f"adaptive_threshold:{threshold_state.config.metric_name}"
            data = json.dumps(threshold_state.to_dict(), default=str)
            await self.redis_client.setex(key, 3600, data)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error storing threshold state: {e}")
    
    async def update_metric_value(self, metric_name: str, value: float):
        """Update metric value for threshold monitoring."""
        try:
            if metric_name in self.threshold_states:
                threshold_state = self.threshold_states[metric_name]
                threshold_state.recent_values.append(value)
                
                # Update statistical properties
                if len(threshold_state.recent_values) >= 10:
                    recent_data = list(threshold_state.recent_values)
                    threshold_state.statistical_properties = {
                        'mean': np.mean(recent_data),
                        'std': np.std(recent_data),
                        'min': np.min(recent_data),
                        'max': np.max(recent_data),
                        'median': np.median(recent_data)
                    }
                
        except Exception as e:
            logger.error(f"Error updating metric value: {e}")
    
    def get_threshold_value(self, metric_name: str) -> Optional[float]:
        """Get current threshold value for metric."""
        if metric_name in self.threshold_states:
            return self.threshold_states[metric_name].current_value
        return None
    
    def get_threshold_status(self) -> Dict[str, Any]:
        """Get status of all thresholds."""
        status = {
            'monitoring_active': self.monitoring_active,
            'thresholds': {},
            'adaptation_history_count': len(self.adaptation_history),
            'last_update': datetime.utcnow().isoformat()
        }
        
        for metric_name, threshold_state in self.threshold_states.items():
            status['thresholds'][metric_name] = {
                'current_value': threshold_state.current_value,
                'last_adaptation': threshold_state.last_adaptation.isoformat(),
                'adaptation_count': threshold_state.adaptation_count,
                'accuracy_score': threshold_state.accuracy_score,
                'recent_values_count': len(threshold_state.recent_values),
                'enabled': threshold_state.config.enabled
            }
        
        return status

# Factory function
def create_adaptive_threshold_monitor(config: Dict[str, Any]) -> AdaptiveThresholdMonitor:
    """Create adaptive threshold monitor instance."""
    return AdaptiveThresholdMonitor(config)

# Example configuration
EXAMPLE_CONFIG = {
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    },
    'adapters': {
        'statistical': {
            'outlier_threshold': 3.0,
            'stability_threshold': 0.1
        },
        'machine_learning': {
            'model_type': 'isolation_forest',
            'contamination': 0.1,
            'min_samples': 50
        },
        'rule_based': {
            'rules': [
                {
                    'type': 'trend_based',
                    'condition': {'operator': '>', 'value': 0.01},
                    'action': {'type': 'increase', 'factor': 1.1}
                },
                {
                    'type': 'violation_based',
                    'condition': {'operator': '>', 'value': 0.15},
                    'action': {'type': 'set_percentile', 'percentile': 98}
                }
            ]
        }
    },
    'thresholds': [
        {
            'metric_name': 'superposition_coherence',
            'threshold_type': 'adaptive',
            'adaptation_strategy': 'statistical',
            'direction': 'lower',
            'initial_value': 0.7,
            'min_value': 0.5,
            'max_value': 0.95,
            'sensitivity': 2.0,
            'adaptation_rate': 0.1,
            'lookback_window': 100,
            'confidence_level': 0.95
        },
        {
            'metric_name': 'system_cpu_usage',
            'threshold_type': 'adaptive',
            'adaptation_strategy': 'machine_learning',
            'direction': 'upper',
            'initial_value': 80,
            'min_value': 50,
            'max_value': 95,
            'sensitivity': 1.0,
            'adaptation_rate': 0.2,
            'lookback_window': 50,
            'confidence_level': 0.90
        }
    ]
}

# Example usage
async def main():
    """Example usage of adaptive threshold monitor."""
    config = EXAMPLE_CONFIG
    monitor = create_adaptive_threshold_monitor(config)
    
    # Start monitoring
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    # Simulate metric updates
    async def simulate_metrics():
        for i in range(100):
            # Simulate coherence values
            coherence = 0.8 + 0.2 * np.random.normal() + 0.1 * np.sin(i / 10)
            await monitor.update_metric_value('superposition_coherence', coherence)
            
            # Simulate CPU usage
            cpu_usage = 70 + 20 * np.random.normal() + 5 * np.sin(i / 20)
            await monitor.update_metric_value('system_cpu_usage', cpu_usage)
            
            await asyncio.sleep(1)
    
    # Run simulation
    await asyncio.gather(monitoring_task, simulate_metrics())

if __name__ == "__main__":
    asyncio.run(main())