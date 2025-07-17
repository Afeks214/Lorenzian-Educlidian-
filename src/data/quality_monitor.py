"""
Comprehensive Data Quality Monitor
Agent Delta: Data Pipeline Transformation Specialist

Real-time data quality monitoring system with advanced statistical process control,
predictive quality scoring, and automated quality reporting. Integrates with the
enhanced QualityAssuranceMonitor to provide comprehensive data pipeline oversight.

Key Features:
- Statistical Process Control (SPC) charts with advanced algorithms
- Predictive quality scoring using machine learning
- Real-time quality metrics and alerts
- Automated quality reporting and recommendations
- Integration with data lineage tracking
- Performance-optimized continuous monitoring
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammainc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import structlog
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class QualityMetricType(str, Enum):
    """Types of quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"
    CONFORMITY = "conformity"

class SPCRuleType(str, Enum):
    """Statistical Process Control rule types"""
    WESTERN_ELECTRIC_1 = "western_electric_1"  # Point beyond 3 sigma
    WESTERN_ELECTRIC_2 = "western_electric_2"  # 2 of 3 points beyond 2 sigma
    WESTERN_ELECTRIC_3 = "western_electric_3"  # 4 of 5 points beyond 1 sigma
    WESTERN_ELECTRIC_4 = "western_electric_4"  # 8 consecutive points on one side
    NELSON_1 = "nelson_1"  # Point beyond 3 sigma
    NELSON_2 = "nelson_2"  # 9 consecutive points on one side
    NELSON_3 = "nelson_3"  # 6 consecutive increasing/decreasing
    NELSON_4 = "nelson_4"  # 14 consecutive alternating
    NELSON_5 = "nelson_5"  # 2 of 3 points beyond 2 sigma
    NELSON_6 = "nelson_6"  # 4 of 5 points beyond 1 sigma
    NELSON_7 = "nelson_7"  # 15 consecutive points within 1 sigma
    NELSON_8 = "nelson_8"  # 8 consecutive points beyond 1 sigma

class QualityTrendType(str, Enum):
    """Quality trend types"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QualityMetric:
    """Quality metric measurement"""
    metric_id: str
    metric_type: QualityMetricType
    value: float
    timestamp: datetime
    component: str
    data_source: str
    
    # Statistical properties
    mean: Optional[float] = None
    std: Optional[float] = None
    percentile_95: Optional[float] = None
    
    # Quality assessment
    quality_score: float = 0.0
    quality_grade: str = "UNKNOWN"
    
    # Metadata
    sample_size: int = 1
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SPCChart:
    """Statistical Process Control chart"""
    chart_id: str
    metric_type: QualityMetricType
    component: str
    
    # Chart data
    data_points: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Control limits
    center_line: float = 0.0
    upper_control_limit: float = 0.0
    lower_control_limit: float = 0.0
    upper_warning_limit: float = 0.0
    lower_warning_limit: float = 0.0
    
    # Chart parameters
    sigma_multiplier: float = 3.0
    subgroup_size: int = 1
    
    # Rule violations
    rule_violations: List[Dict[str, Any]] = field(default_factory=list)
    out_of_control_points: List[int] = field(default_factory=list)
    
    # Chart statistics
    process_capability: Optional[float] = None
    process_performance: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityPrediction:
    """Quality prediction result"""
    prediction_id: str
    component: str
    metric_type: QualityMetricType
    
    # Prediction details
    current_quality: float
    predicted_quality: float
    prediction_horizon_minutes: int
    confidence_score: float
    
    # Confidence intervals
    lower_bound: float
    upper_bound: float
    
    # Contributing factors
    risk_factors: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    
    # Model information
    model_type: str = "unknown"
    model_accuracy: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityAlert:
    """Quality alert"""
    alert_id: str
    alert_level: AlertLevel
    component: str
    metric_type: QualityMetricType
    
    # Alert details
    title: str
    description: str
    trigger_value: float
    threshold_value: float
    
    # Context
    affected_systems: List[str] = field(default_factory=list)
    impact_assessment: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    escalation_level: int = 0

# =============================================================================
# STATISTICAL PROCESS CONTROL ENGINE
# =============================================================================

class SPCEngine:
    """Statistical Process Control engine"""
    
    def __init__(self):
        self.charts: Dict[str, SPCChart] = {}
        self.rule_processors: Dict[SPCRuleType, Callable] = {}
        self._initialize_rule_processors()
        
    def _initialize_rule_processors(self):
        """Initialize SPC rule processors"""
        self.rule_processors = {
            SPCRuleType.WESTERN_ELECTRIC_1: self._check_western_electric_1,
            SPCRuleType.WESTERN_ELECTRIC_2: self._check_western_electric_2,
            SPCRuleType.WESTERN_ELECTRIC_3: self._check_western_electric_3,
            SPCRuleType.WESTERN_ELECTRIC_4: self._check_western_electric_4,
            SPCRuleType.NELSON_1: self._check_nelson_1,
            SPCRuleType.NELSON_2: self._check_nelson_2,
            SPCRuleType.NELSON_3: self._check_nelson_3,
            SPCRuleType.NELSON_4: self._check_nelson_4,
            SPCRuleType.NELSON_5: self._check_nelson_5,
            SPCRuleType.NELSON_6: self._check_nelson_6,
            SPCRuleType.NELSON_7: self._check_nelson_7,
            SPCRuleType.NELSON_8: self._check_nelson_8,
        }
    
    def create_chart(self, 
                    component: str, 
                    metric_type: QualityMetricType,
                    sigma_multiplier: float = 3.0,
                    subgroup_size: int = 1) -> str:
        """Create a new SPC chart"""
        
        chart_id = f"{component}_{metric_type.value}_{uuid.uuid4().hex[:8]}"
        
        chart = SPCChart(
            chart_id=chart_id,
            metric_type=metric_type,
            component=component,
            sigma_multiplier=sigma_multiplier,
            subgroup_size=subgroup_size
        )
        
        self.charts[chart_id] = chart
        
        logger.debug(f"Created SPC chart: {chart_id}")
        return chart_id
    
    def add_data_point(self, chart_id: str, value: float, timestamp: datetime) -> Dict[str, Any]:
        """Add data point to SPC chart"""
        
        if chart_id not in self.charts:
            raise ValueError(f"Chart {chart_id} not found")
        
        chart = self.charts[chart_id]
        
        # Add data point
        chart.data_points.append(value)
        chart.timestamps.append(timestamp)
        
        # Update control limits if we have enough data
        if len(chart.data_points) >= 20:
            self._update_control_limits(chart)
        
        # Check for rule violations
        violations = self._check_all_rules(chart)
        
        # Update chart metadata
        chart.last_updated = datetime.utcnow()
        
        return {
            'chart_id': chart_id,
            'value': value,
            'timestamp': timestamp,
            'violations': violations,
            'control_limits': {
                'center_line': chart.center_line,
                'upper_control_limit': chart.upper_control_limit,
                'lower_control_limit': chart.lower_control_limit,
                'upper_warning_limit': chart.upper_warning_limit,
                'lower_warning_limit': chart.lower_warning_limit
            }
        }
    
    def _update_control_limits(self, chart: SPCChart):
        """Update control limits for the chart"""
        
        data = list(chart.data_points)
        
        if len(data) < 20:
            return
        
        # Calculate center line (mean)
        chart.center_line = np.mean(data)
        
        # Calculate standard deviation
        sigma = np.std(data, ddof=1)
        
        # Calculate control limits
        chart.upper_control_limit = chart.center_line + (chart.sigma_multiplier * sigma)
        chart.lower_control_limit = chart.center_line - (chart.sigma_multiplier * sigma)
        
        # Calculate warning limits (2 sigma)
        chart.upper_warning_limit = chart.center_line + (2 * sigma)
        chart.lower_warning_limit = chart.center_line - (2 * sigma)
        
        # Calculate process capability if specification limits are available
        # This would require specification limits which aren't available here
        # chart.process_capability = self._calculate_process_capability(chart)
    
    def _check_all_rules(self, chart: SPCChart) -> List[Dict[str, Any]]:
        """Check all SPC rules for violations"""
        
        violations = []
        
        if len(chart.data_points) < 10:
            return violations
        
        # Check each rule
        for rule_type, rule_processor in self.rule_processors.items():
            try:
                violation = rule_processor(chart)
                if violation:
                    violations.append({
                        'rule_type': rule_type.value,
                        'description': violation['description'],
                        'points': violation['points'],
                        'timestamp': datetime.utcnow()
                    })
            except Exception as e:
                logger.error(f"Error checking rule {rule_type}: {e}")
        
        # Store violations in chart
        chart.rule_violations.extend(violations)
        
        return violations
    
    def _check_western_electric_1(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Western Electric Rule 1: Point beyond 3 sigma"""
        
        if len(chart.data_points) < 1:
            return None
        
        latest_point = chart.data_points[-1]
        
        if (latest_point > chart.upper_control_limit or 
            latest_point < chart.lower_control_limit):
            return {
                'description': 'Point beyond 3 sigma control limits',
                'points': [len(chart.data_points) - 1]
            }
        
        return None
    
    def _check_western_electric_2(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Western Electric Rule 2: 2 of 3 points beyond 2 sigma"""
        
        if len(chart.data_points) < 3:
            return None
        
        recent_points = list(chart.data_points)[-3:]
        
        beyond_2_sigma = 0
        violation_indices = []
        
        for i, point in enumerate(recent_points):
            if (point > chart.upper_warning_limit or 
                point < chart.lower_warning_limit):
                beyond_2_sigma += 1
                violation_indices.append(len(chart.data_points) - 3 + i)
        
        if beyond_2_sigma >= 2:
            return {
                'description': '2 of 3 points beyond 2 sigma limits',
                'points': violation_indices
            }
        
        return None
    
    def _check_western_electric_3(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Western Electric Rule 3: 4 of 5 points beyond 1 sigma"""
        
        if len(chart.data_points) < 5:
            return None
        
        recent_points = list(chart.data_points)[-5:]
        sigma = (chart.upper_control_limit - chart.center_line) / 3
        
        beyond_1_sigma = 0
        violation_indices = []
        
        for i, point in enumerate(recent_points):
            if (point > chart.center_line + sigma or 
                point < chart.center_line - sigma):
                beyond_1_sigma += 1
                violation_indices.append(len(chart.data_points) - 5 + i)
        
        if beyond_1_sigma >= 4:
            return {
                'description': '4 of 5 points beyond 1 sigma',
                'points': violation_indices
            }
        
        return None
    
    def _check_western_electric_4(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Western Electric Rule 4: 8 consecutive points on one side"""
        
        if len(chart.data_points) < 8:
            return None
        
        recent_points = list(chart.data_points)[-8:]
        
        # Check if all points are on one side of center line
        above_center = all(point > chart.center_line for point in recent_points)
        below_center = all(point < chart.center_line for point in recent_points)
        
        if above_center or below_center:
            return {
                'description': '8 consecutive points on one side of center line',
                'points': list(range(len(chart.data_points) - 8, len(chart.data_points)))
            }
        
        return None
    
    def _check_nelson_1(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 1: Point beyond 3 sigma"""
        return self._check_western_electric_1(chart)
    
    def _check_nelson_2(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 2: 9 consecutive points on one side"""
        
        if len(chart.data_points) < 9:
            return None
        
        recent_points = list(chart.data_points)[-9:]
        
        # Check if all points are on one side of center line
        above_center = all(point > chart.center_line for point in recent_points)
        below_center = all(point < chart.center_line for point in recent_points)
        
        if above_center or below_center:
            return {
                'description': '9 consecutive points on one side of center line',
                'points': list(range(len(chart.data_points) - 9, len(chart.data_points)))
            }
        
        return None
    
    def _check_nelson_3(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 3: 6 consecutive increasing or decreasing points"""
        
        if len(chart.data_points) < 6:
            return None
        
        recent_points = list(chart.data_points)[-6:]
        
        # Check for increasing trend
        increasing = all(recent_points[i] < recent_points[i+1] for i in range(len(recent_points)-1))
        
        # Check for decreasing trend
        decreasing = all(recent_points[i] > recent_points[i+1] for i in range(len(recent_points)-1))
        
        if increasing or decreasing:
            return {
                'description': '6 consecutive increasing or decreasing points',
                'points': list(range(len(chart.data_points) - 6, len(chart.data_points)))
            }
        
        return None
    
    def _check_nelson_4(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 4: 14 consecutive alternating points"""
        
        if len(chart.data_points) < 14:
            return None
        
        recent_points = list(chart.data_points)[-14:]
        
        # Check for alternating pattern
        alternating = True
        for i in range(len(recent_points) - 2):
            if not ((recent_points[i] > chart.center_line and 
                    recent_points[i+1] < chart.center_line and 
                    recent_points[i+2] > chart.center_line) or
                   (recent_points[i] < chart.center_line and 
                    recent_points[i+1] > chart.center_line and 
                    recent_points[i+2] < chart.center_line)):
                alternating = False
                break
        
        if alternating:
            return {
                'description': '14 consecutive alternating points',
                'points': list(range(len(chart.data_points) - 14, len(chart.data_points)))
            }
        
        return None
    
    def _check_nelson_5(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 5: 2 of 3 points beyond 2 sigma"""
        return self._check_western_electric_2(chart)
    
    def _check_nelson_6(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 6: 4 of 5 points beyond 1 sigma"""
        return self._check_western_electric_3(chart)
    
    def _check_nelson_7(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 7: 15 consecutive points within 1 sigma"""
        
        if len(chart.data_points) < 15:
            return None
        
        recent_points = list(chart.data_points)[-15:]
        sigma = (chart.upper_control_limit - chart.center_line) / 3
        
        # Check if all points are within 1 sigma
        within_1_sigma = all(
            chart.center_line - sigma <= point <= chart.center_line + sigma
            for point in recent_points
        )
        
        if within_1_sigma:
            return {
                'description': '15 consecutive points within 1 sigma (may indicate stratification)',
                'points': list(range(len(chart.data_points) - 15, len(chart.data_points)))
            }
        
        return None
    
    def _check_nelson_8(self, chart: SPCChart) -> Optional[Dict[str, Any]]:
        """Check Nelson Rule 8: 8 consecutive points beyond 1 sigma"""
        
        if len(chart.data_points) < 8:
            return None
        
        recent_points = list(chart.data_points)[-8:]
        sigma = (chart.upper_control_limit - chart.center_line) / 3
        
        # Check if all points are beyond 1 sigma
        beyond_1_sigma = all(
            point > chart.center_line + sigma or point < chart.center_line - sigma
            for point in recent_points
        )
        
        if beyond_1_sigma:
            return {
                'description': '8 consecutive points beyond 1 sigma',
                'points': list(range(len(chart.data_points) - 8, len(chart.data_points)))
            }
        
        return None
    
    def get_chart_summary(self, chart_id: str) -> Dict[str, Any]:
        """Get summary statistics for a chart"""
        
        if chart_id not in self.charts:
            return {'error': f'Chart {chart_id} not found'}
        
        chart = self.charts[chart_id]
        data = list(chart.data_points)
        
        if not data:
            return {'error': 'No data points in chart'}
        
        return {
            'chart_id': chart_id,
            'component': chart.component,
            'metric_type': chart.metric_type.value,
            'data_points_count': len(data),
            'statistics': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data),
                'q1': np.percentile(data, 25),
                'q3': np.percentile(data, 75)
            },
            'control_limits': {
                'center_line': chart.center_line,
                'upper_control_limit': chart.upper_control_limit,
                'lower_control_limit': chart.lower_control_limit,
                'upper_warning_limit': chart.upper_warning_limit,
                'lower_warning_limit': chart.lower_warning_limit
            },
            'violations_count': len(chart.rule_violations),
            'recent_violations': chart.rule_violations[-5:] if chart.rule_violations else [],
            'process_capability': chart.process_capability,
            'created_at': chart.created_at,
            'last_updated': chart.last_updated
        }

# =============================================================================
# PREDICTIVE QUALITY SCORING ENGINE
# =============================================================================

class PredictiveQualityEngine:
    """Predictive quality scoring using machine learning"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
        # Model configuration
        self.min_training_samples = 100
        self.retrain_interval = 50
        self.prediction_horizons = [5, 15, 30, 60]  # minutes
        
        # Supported models
        self.model_types = {
            'random_forest': RandomForestRegressor,
            'gradient_boost': GradientBoostingRegressor,
            'linear': LinearRegression
        }
    
    def add_training_data(self, 
                         component: str, 
                         metric_type: QualityMetricType,
                         features: Dict[str, float],
                         quality_score: float,
                         timestamp: datetime):
        """Add training data for predictive models"""
        
        key = f"{component}_{metric_type.value}"
        
        # Create feature vector
        feature_vector = [
            features.get('current_value', 0),
            features.get('mean_5min', 0),
            features.get('std_5min', 0),
            features.get('trend_slope', 0),
            features.get('volatility', 0),
            features.get('error_rate', 0),
            features.get('throughput', 0),
            timestamp.timestamp() % 86400,  # Time of day
            timestamp.weekday(),  # Day of week
        ]
        
        self.training_data[key].append({
            'features': feature_vector,
            'quality_score': quality_score,
            'timestamp': timestamp
        })
        
        # Retrain model if needed
        if (len(self.training_data[key]) >= self.min_training_samples and
            len(self.training_data[key]) % self.retrain_interval == 0):
            self._train_models(key)
    
    def _train_models(self, key: str):
        """Train predictive models for a component/metric combination"""
        
        try:
            data = list(self.training_data[key])
            
            if len(data) < self.min_training_samples:
                return
            
            # Prepare training data
            X = np.array([item['features'] for item in data])
            y = np.array([item['quality_score'] for item in data])
            
            # Split into train/validation
            split_idx = int(len(data) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            self.scalers[key] = scaler
            
            # Train multiple models
            models = {}
            performance = {}
            
            for model_name, model_class in self.model_types.items():
                try:
                    # Configure model
                    if model_name == 'random_forest':
                        model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
                    elif model_name == 'gradient_boost':
                        model = model_class(n_estimators=100, random_state=42)
                    else:
                        model = model_class()
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_val_scaled)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    models[model_name] = model
                    performance[model_name] = {
                        'mse': mse,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    # Store feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[f"{key}_{model_name}"] = {
                            f'feature_{i}': importance 
                            for i, importance in enumerate(model.feature_importances_)
                        }
                
                except Exception as e:
                    logger.error(f"Error training {model_name} for {key}: {e}")
            
            # Store models and performance
            self.models[key] = models
            self.model_performance[key] = performance
            
            logger.debug(f"Trained {len(models)} models for {key}")
            
        except Exception as e:
            logger.error(f"Error training models for {key}: {e}")
    
    def predict_quality(self, 
                       component: str,
                       metric_type: QualityMetricType,
                       features: Dict[str, float],
                       horizon_minutes: int = 15) -> Optional[QualityPrediction]:
        """Predict quality score for given horizon"""
        
        key = f"{component}_{metric_type.value}"
        
        if key not in self.models or not self.models[key]:
            return None
        
        try:
            # Prepare feature vector
            feature_vector = np.array([
                features.get('current_value', 0),
                features.get('mean_5min', 0),
                features.get('std_5min', 0),
                features.get('trend_slope', 0),
                features.get('volatility', 0),
                features.get('error_rate', 0),
                features.get('throughput', 0),
                (datetime.utcnow().timestamp() + horizon_minutes * 60) % 86400,
                datetime.utcnow().weekday(),
            ]).reshape(1, -1)
            
            # Scale features
            if key in self.scalers:
                feature_vector = self.scalers[key].transform(feature_vector)
            
            # Get predictions from all models
            predictions = []
            model_accuracies = []
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(feature_vector)[0]
                    predictions.append(pred)
                    
                    # Get model accuracy
                    if key in self.model_performance:
                        accuracy = self.model_performance[key].get(model_name, {}).get('r2', 0)
                        model_accuracies.append(accuracy)
                    else:
                        model_accuracies.append(0.5)  # Default
                        
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
            
            if not predictions:
                return None
            
            # Ensemble prediction (weighted by model accuracy)
            if model_accuracies:
                weights = np.array(model_accuracies)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
                predicted_quality = np.average(predictions, weights=weights)
                confidence_score = np.mean(model_accuracies)
            else:
                predicted_quality = np.mean(predictions)
                confidence_score = 0.5
            
            # Calculate confidence interval
            pred_std = np.std(predictions) if len(predictions) > 1 else 0.1
            lower_bound = max(0, predicted_quality - 1.96 * pred_std)
            upper_bound = min(1, predicted_quality + 1.96 * pred_std)
            
            # Identify risk factors and opportunities
            risk_factors = self._identify_risk_factors(features, predicted_quality)
            improvement_opportunities = self._identify_improvements(features, predicted_quality)
            
            # Get best model and its importance
            best_model = max(self.models[key].items(), key=lambda x: model_accuracies[list(self.models[key].keys()).index(x[0])])
            model_type = best_model[0]
            model_accuracy = model_accuracies[list(self.models[key].keys()).index(model_type)]
            
            feature_importance = self.feature_importance.get(f"{key}_{model_type}", {})
            
            return QualityPrediction(
                prediction_id=str(uuid.uuid4()),
                component=component,
                metric_type=metric_type,
                current_quality=features.get('current_quality', 0),
                predicted_quality=predicted_quality,
                prediction_horizon_minutes=horizon_minutes,
                confidence_score=confidence_score,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                risk_factors=risk_factors,
                improvement_opportunities=improvement_opportunities,
                model_type=model_type,
                model_accuracy=model_accuracy,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error predicting quality for {key}: {e}")
            return None
    
    def _identify_risk_factors(self, features: Dict[str, float], predicted_quality: float) -> List[str]:
        """Identify risk factors based on features and prediction"""
        
        risk_factors = []
        
        # Check various risk indicators
        if features.get('error_rate', 0) > 0.05:
            risk_factors.append("High error rate detected")
        
        if features.get('volatility', 0) > 0.3:
            risk_factors.append("High volatility in metrics")
        
        if features.get('trend_slope', 0) < -0.1:
            risk_factors.append("Negative trend in quality metrics")
        
        if features.get('throughput', 0) < 0.5:
            risk_factors.append("Low throughput affecting quality")
        
        if predicted_quality < 0.6:
            risk_factors.append("Predicted quality below acceptable threshold")
        
        return risk_factors
    
    def _identify_improvements(self, features: Dict[str, float], predicted_quality: float) -> List[str]:
        """Identify improvement opportunities"""
        
        improvements = []
        
        # Suggest improvements based on features
        if features.get('error_rate', 0) > 0.02:
            improvements.append("Reduce error rate through better validation")
        
        if features.get('volatility', 0) > 0.2:
            improvements.append("Stabilize metrics through process control")
        
        if features.get('throughput', 0) < 0.8:
            improvements.append("Optimize throughput for better quality")
        
        if predicted_quality < 0.8:
            improvements.append("Implement proactive quality measures")
        
        return improvements
    
    def get_model_performance(self, component: str, metric_type: QualityMetricType) -> Dict[str, Any]:
        """Get performance metrics for trained models"""
        
        key = f"{component}_{metric_type.value}"
        
        if key not in self.model_performance:
            return {'error': f'No models trained for {key}'}
        
        return {
            'component': component,
            'metric_type': metric_type.value,
            'models': self.model_performance[key],
            'training_samples': len(self.training_data[key]),
            'feature_importance': {
                model: self.feature_importance.get(f"{key}_{model}", {})
                for model in self.models.get(key, {}).keys()
            }
        }

# =============================================================================
# COMPREHENSIVE QUALITY MONITOR
# =============================================================================

class ComprehensiveQualityMonitor:
    """Comprehensive data quality monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Core engines
        self.spc_engine = SPCEngine()
        self.prediction_engine = PredictiveQualityEngine()
        
        # Data storage
        self.quality_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.quality_alerts: Dict[str, QualityAlert] = {}
        self.quality_trends: Dict[str, QualityTrendType] = {}
        
        # Monitoring state
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Background processing
        self.running = False
        self.monitor_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'metrics_processed': 0,
            'alerts_generated': 0,
            'predictions_made': 0,
            'charts_created': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Comprehensive Quality Monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'enable_spc': True,
            'enable_prediction': True,
            'enable_automated_alerts': True,
            'alert_cooldown_minutes': 15,
            'trend_analysis_window': 50,
            'quality_thresholds': {
                'warning': 0.7,
                'error': 0.5,
                'critical': 0.3
            },
            'prediction_horizons': [5, 15, 30, 60],
            'max_alerts_per_component': 10,
            'cleanup_interval_hours': 24
        }
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="quality_monitor"
            )
            self.monitor_thread.start()
            logger.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("Quality monitoring stopped")
    
    def register_component(self, 
                          component: str, 
                          metric_types: List[QualityMetricType],
                          thresholds: Optional[Dict[str, float]] = None):
        """Register component for quality monitoring"""
        
        with self.lock:
            self.active_monitors[component] = {
                'metric_types': metric_types,
                'registered_at': datetime.utcnow(),
                'spc_charts': {},
                'last_quality_check': None
            }
            
            # Set thresholds
            if thresholds:
                self.alert_thresholds[component] = thresholds
            else:
                self.alert_thresholds[component] = self.config['quality_thresholds'].copy()
            
            # Create SPC charts if enabled
            if self.config['enable_spc']:
                for metric_type in metric_types:
                    chart_id = self.spc_engine.create_chart(component, metric_type)
                    self.active_monitors[component]['spc_charts'][metric_type] = chart_id
                    self.stats['charts_created'] += 1
            
            logger.info(f"Registered component {component} with {len(metric_types)} metrics")
    
    def record_quality_metric(self, 
                            component: str,
                            metric_type: QualityMetricType,
                            value: float,
                            additional_features: Optional[Dict[str, float]] = None,
                            timestamp: Optional[datetime] = None):
        """Record quality metric"""
        
        timestamp = timestamp or datetime.utcnow()
        
        with self.lock:
            # Create quality metric
            metric = QualityMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                component=component,
                data_source="quality_monitor",
                quality_score=self._calculate_quality_score(value, metric_type),
                quality_grade=self._get_quality_grade(value, metric_type),
                tags=additional_features or {}
            )
            
            # Store metric
            key = f"{component}_{metric_type.value}"
            self.quality_metrics[key].append(metric)
            
            # Update SPC chart
            if (self.config['enable_spc'] and 
                component in self.active_monitors and 
                metric_type in self.active_monitors[component]['spc_charts']):
                
                chart_id = self.active_monitors[component]['spc_charts'][metric_type]
                spc_result = self.spc_engine.add_data_point(chart_id, value, timestamp)
                
                # Check for SPC violations
                if spc_result['violations']:
                    self._handle_spc_violations(component, metric_type, spc_result['violations'])
            
            # Add to prediction engine
            if self.config['enable_prediction'] and additional_features:
                self.prediction_engine.add_training_data(
                    component, metric_type, additional_features, 
                    metric.quality_score, timestamp
                )
            
            # Update quality trend
            self._update_quality_trend(component, metric_type)
            
            # Check for alerts
            if self.config['enable_automated_alerts']:
                self._check_quality_alerts(component, metric_type, metric)
            
            self.stats['metrics_processed'] += 1
            
            # Update last quality check
            if component in self.active_monitors:
                self.active_monitors[component]['last_quality_check'] = timestamp
    
    def _calculate_quality_score(self, value: float, metric_type: QualityMetricType) -> float:
        """Calculate quality score (0-1) based on metric value and type"""
        
        # Different scoring logic based on metric type
        if metric_type in [QualityMetricType.COMPLETENESS, QualityMetricType.ACCURACY, 
                          QualityMetricType.VALIDITY, QualityMetricType.INTEGRITY]:
            # Higher is better
            return max(0, min(1, value))
        
        elif metric_type == QualityMetricType.TIMELINESS:
            # Lower latency is better (assuming value is latency in seconds)
            if value <= 1.0:
                return 1.0
            elif value <= 5.0:
                return 0.8
            elif value <= 10.0:
                return 0.6
            else:
                return max(0, 1.0 - (value - 10.0) / 90.0)
        
        else:
            # Default: assume higher is better
            return max(0, min(1, value))
    
    def _get_quality_grade(self, value: float, metric_type: QualityMetricType) -> str:
        """Get quality grade (A-F) based on score"""
        
        score = self._calculate_quality_score(value, metric_type)
        
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _handle_spc_violations(self, component: str, metric_type: QualityMetricType, violations: List[Dict[str, Any]]):
        """Handle SPC rule violations"""
        
        for violation in violations:
            alert = QualityAlert(
                alert_id=str(uuid.uuid4()),
                alert_level=AlertLevel.WARNING,
                component=component,
                metric_type=metric_type,
                title=f"SPC Violation: {violation['rule_type']}",
                description=violation['description'],
                trigger_value=0.0,  # SPC violations don't have single trigger values
                threshold_value=0.0,
                recommended_actions=[
                    "Investigate process changes",
                    "Review recent data for patterns",
                    "Check for external factors"
                ]
            )
            
            self.quality_alerts[alert.alert_id] = alert
            self.stats['alerts_generated'] += 1
            
            logger.warning(f"SPC violation in {component}/{metric_type.value}: {violation['description']}")
    
    def _update_quality_trend(self, component: str, metric_type: QualityMetricType):
        """Update quality trend for component/metric"""
        
        key = f"{component}_{metric_type.value}"
        
        if key not in self.quality_metrics or len(self.quality_metrics[key]) < 10:
            self.quality_trends[key] = QualityTrendType.UNKNOWN
            return
        
        # Get recent quality scores
        recent_metrics = list(self.quality_metrics[key])[-self.config['trend_analysis_window']:]
        quality_scores = [m.quality_score for m in recent_metrics]
        
        # Calculate trend
        if len(quality_scores) >= 5:
            # Use linear regression to detect trend
            x = np.arange(len(quality_scores))
            slope, _, r_value, _, _ = stats.linregress(x, quality_scores)
            
            # Classify trend
            if abs(r_value) < 0.3:
                trend = QualityTrendType.STABLE
            elif slope > 0.01:
                trend = QualityTrendType.IMPROVING
            elif slope < -0.01:
                trend = QualityTrendType.DEGRADING
            else:
                # Check volatility
                volatility = np.std(quality_scores)
                if volatility > 0.1:
                    trend = QualityTrendType.VOLATILE
                else:
                    trend = QualityTrendType.STABLE
            
            self.quality_trends[key] = trend
    
    def _check_quality_alerts(self, component: str, metric_type: QualityMetricType, metric: QualityMetric):
        """Check if quality alerts should be generated"""
        
        thresholds = self.alert_thresholds.get(component, self.config['quality_thresholds'])
        
        # Check thresholds
        alert_level = None
        if metric.quality_score < thresholds['critical']:
            alert_level = AlertLevel.CRITICAL
        elif metric.quality_score < thresholds['error']:
            alert_level = AlertLevel.ERROR
        elif metric.quality_score < thresholds['warning']:
            alert_level = AlertLevel.WARNING
        
        if alert_level:
            # Check alert cooldown
            recent_alerts = [
                alert for alert in self.quality_alerts.values()
                if (alert.component == component and 
                    alert.metric_type == metric_type and
                    alert.alert_level == alert_level and
                    (datetime.utcnow() - alert.created_at).total_seconds() < 
                    self.config['alert_cooldown_minutes'] * 60)
            ]
            
            if not recent_alerts:
                alert = QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_level=alert_level,
                    component=component,
                    metric_type=metric_type,
                    title=f"Quality {alert_level.value.title()}: {metric_type.value}",
                    description=f"Quality score {metric.quality_score:.3f} below {alert_level.value} threshold",
                    trigger_value=metric.quality_score,
                    threshold_value=thresholds[alert_level.value],
                    recommended_actions=self._get_recommended_actions(component, metric_type, alert_level)
                )
                
                self.quality_alerts[alert.alert_id] = alert
                self.stats['alerts_generated'] += 1
                
                logger.warning(f"Quality alert generated for {component}/{metric_type.value}: {alert.description}")
    
    def _get_recommended_actions(self, component: str, metric_type: QualityMetricType, alert_level: AlertLevel) -> List[str]:
        """Get recommended actions for quality alerts"""
        
        actions = []
        
        # Generic actions based on metric type
        if metric_type == QualityMetricType.COMPLETENESS:
            actions.extend([
                "Check data source connectivity",
                "Review data collection processes",
                "Verify data ingestion pipelines"
            ])
        elif metric_type == QualityMetricType.ACCURACY:
            actions.extend([
                "Validate data transformation logic",
                "Check reference data quality",
                "Review calculation algorithms"
            ])
        elif metric_type == QualityMetricType.TIMELINESS:
            actions.extend([
                "Optimize data processing pipelines",
                "Check network connectivity",
                "Review system resource usage"
            ])
        
        # Actions based on alert level
        if alert_level == AlertLevel.CRITICAL:
            actions.extend([
                "Escalate to on-call team",
                "Consider emergency procedures",
                "Review recent system changes"
            ])
        
        return actions
    
    def get_quality_prediction(self, 
                             component: str, 
                             metric_type: QualityMetricType,
                             horizon_minutes: int = 15) -> Optional[QualityPrediction]:
        """Get quality prediction for component/metric"""
        
        if not self.config['enable_prediction']:
            return None
        
        # Get recent metrics to build features
        key = f"{component}_{metric_type.value}"
        if key not in self.quality_metrics or len(self.quality_metrics[key]) < 10:
            return None
        
        recent_metrics = list(self.quality_metrics[key])[-20:]
        values = [m.value for m in recent_metrics]
        quality_scores = [m.quality_score for m in recent_metrics]
        
        # Calculate features
        features = {
            'current_value': values[-1],
            'current_quality': quality_scores[-1],
            'mean_5min': np.mean(values[-5:]),
            'std_5min': np.std(values[-5:]),
            'trend_slope': self._calculate_trend_slope(values),
            'volatility': np.std(values),
            'error_rate': sum(1 for q in quality_scores if q < 0.5) / len(quality_scores),
            'throughput': len(values) / 20  # Normalized throughput
        }
        
        prediction = self.prediction_engine.predict_quality(
            component, metric_type, features, horizon_minutes
        )
        
        if prediction:
            self.stats['predictions_made'] += 1
        
        return prediction
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope for values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Cleanup old alerts
                self._cleanup_old_alerts()
                
                # Generate quality reports
                self._generate_periodic_reports()
                
                # Update trends
                self._update_all_trends()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config['cleanup_interval_hours'])
        
        with self.lock:
            old_alerts = [
                alert_id for alert_id, alert in self.quality_alerts.items()
                if alert.created_at < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.quality_alerts[alert_id]
            
            if old_alerts:
                logger.debug(f"Cleaned up {len(old_alerts)} old alerts")
    
    def _generate_periodic_reports(self):
        """Generate periodic quality reports"""
        # This would generate summary reports
        # Implementation depends on reporting requirements
        pass
    
    def _update_all_trends(self):
        """Update quality trends for all components"""
        with self.lock:
            for component, monitor_info in self.active_monitors.items():
                for metric_type in monitor_info['metric_types']:
                    self._update_quality_trend(component, metric_type)
    
    def get_component_summary(self, component: str) -> Dict[str, Any]:
        """Get comprehensive summary for component"""
        
        if component not in self.active_monitors:
            return {'error': f'Component {component} not registered'}
        
        monitor_info = self.active_monitors[component]
        summary = {
            'component': component,
            'registered_at': monitor_info['registered_at'],
            'last_quality_check': monitor_info['last_quality_check'],
            'metrics': []
        }
        
        # Get metrics summary
        for metric_type in monitor_info['metric_types']:
            key = f"{component}_{metric_type.value}"
            
            # Get recent metrics
            recent_metrics = list(self.quality_metrics[key])[-20:] if key in self.quality_metrics else []
            
            if recent_metrics:
                quality_scores = [m.quality_score for m in recent_metrics]
                metric_summary = {
                    'metric_type': metric_type.value,
                    'current_quality': quality_scores[-1],
                    'average_quality': np.mean(quality_scores),
                    'trend': self.quality_trends.get(key, QualityTrendType.UNKNOWN).value,
                    'measurements_count': len(recent_metrics),
                    'quality_grade': recent_metrics[-1].quality_grade
                }
                
                # Add SPC chart summary
                if metric_type in monitor_info['spc_charts']:
                    chart_id = monitor_info['spc_charts'][metric_type]
                    chart_summary = self.spc_engine.get_chart_summary(chart_id)
                    if 'error' not in chart_summary:
                        metric_summary['spc_violations'] = chart_summary['violations_count']
                
                # Add prediction if available
                prediction = self.get_quality_prediction(component, metric_type)
                if prediction:
                    metric_summary['prediction'] = {
                        'predicted_quality': prediction.predicted_quality,
                        'confidence': prediction.confidence_score,
                        'horizon_minutes': prediction.prediction_horizon_minutes
                    }
                
                summary['metrics'].append(metric_summary)
        
        # Get active alerts
        active_alerts = [
            {
                'alert_id': alert.alert_id,
                'level': alert.alert_level.value,
                'title': alert.title,
                'created_at': alert.created_at
            }
            for alert in self.quality_alerts.values()
            if alert.component == component and not alert.resolved_at
        ]
        
        summary['active_alerts'] = active_alerts
        summary['alert_count'] = len(active_alerts)
        
        return summary
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide quality overview"""
        
        overview = {
            'timestamp': datetime.utcnow(),
            'components_monitored': len(self.active_monitors),
            'total_metrics': sum(len(info['metric_types']) for info in self.active_monitors.values()),
            'statistics': self.stats.copy(),
            'quality_distribution': {},
            'trend_distribution': {},
            'alert_distribution': {}
        }
        
        # Calculate quality distribution
        all_quality_scores = []
        for metrics in self.quality_metrics.values():
            if metrics:
                all_quality_scores.extend([m.quality_score for m in list(metrics)[-10:]])
        
        if all_quality_scores:
            overview['quality_distribution'] = {
                'excellent': sum(1 for q in all_quality_scores if q >= 0.9) / len(all_quality_scores),
                'good': sum(1 for q in all_quality_scores if 0.7 <= q < 0.9) / len(all_quality_scores),
                'acceptable': sum(1 for q in all_quality_scores if 0.5 <= q < 0.7) / len(all_quality_scores),
                'poor': sum(1 for q in all_quality_scores if q < 0.5) / len(all_quality_scores)
            }
        
        # Calculate trend distribution
        trend_counts = defaultdict(int)
        for trend in self.quality_trends.values():
            trend_counts[trend.value] += 1
        
        if trend_counts:
            total_trends = sum(trend_counts.values())
            overview['trend_distribution'] = {
                trend: count / total_trends for trend, count in trend_counts.items()
            }
        
        # Calculate alert distribution
        alert_counts = defaultdict(int)
        for alert in self.quality_alerts.values():
            if not alert.resolved_at:
                alert_counts[alert.alert_level.value] += 1
        
        overview['alert_distribution'] = dict(alert_counts)
        
        return overview

# Export key components
__all__ = [
    'QualityMetricType',
    'SPCRuleType',
    'QualityTrendType',
    'AlertLevel',
    'QualityMetric',
    'SPCChart',
    'QualityPrediction',
    'QualityAlert',
    'SPCEngine',
    'PredictiveQualityEngine',
    'ComprehensiveQualityMonitor'
]