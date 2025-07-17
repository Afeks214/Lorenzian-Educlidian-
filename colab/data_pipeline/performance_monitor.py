"""
Performance Monitoring System for NQ Data Pipeline

Provides unified performance metrics, data loading benchmarks,
and real-time monitoring dashboards for the data pipeline.
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import psutil
import torch
import warnings
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy import stats
from scipy.signal import find_peaks
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import pickle
import uuid
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import asyncio
from queue import Queue, PriorityQueue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import websockets
from threading import Event, Lock

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class PredictionType(Enum):
    """Types of predictions"""
    ANOMALY = "anomaly"
    TREND = "trend"
    CAPACITY = "capacity"
    FAILURE = "failure"

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AnomalyType(Enum):
    """Types of anomalies"""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    ML_BASED = "ml_based"
    MULTI_DIMENSIONAL = "multi_dimensional"
    THRESHOLD = "threshold"

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    threshold: float
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class QualityScore:
    """Data quality score"""
    overall_score: float  # 0-100
    completeness: float
    consistency: float
    accuracy: float
    timeliness: float
    validity: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
    """Single performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    anomaly_flags: List[AnomalyResult] = field(default_factory=list)

@dataclass
class PredictionResult:
    """Result of predictive analysis"""
    metric_name: str
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    timestamp: float
    horizon_minutes: int
    risk_score: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert object"""
    id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: float
    timestamp: float
    prediction_based: bool = False
    correlation_id: Optional[str] = None
    escalation_level: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None

@dataclass
class CapacityPrediction:
    """Capacity planning prediction"""
    resource_type: str
    current_usage: float
    predicted_usage: float
    capacity_limit: float
    time_to_limit: Optional[float]
    recommendation: str
    confidence: float
    timestamp: float

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    test_name: str
    duration_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class AnomalyDetector(ABC):
    """Base class for anomaly detectors"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Train the anomaly detector"""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> List[AnomalyResult]:
        """Detect anomalies in data"""
        pass
    
    @abstractmethod
    def get_severity(self, score: float) -> AnomalySeverity:
        """Determine anomaly severity from score"""
        pass

class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detection using Z-score, IQR, and Moving Average"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("statistical", config)
        self.z_threshold = self.config.get('z_threshold', 3.0)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.window_size = self.config.get('window_size', 50)
        self.min_samples = self.config.get('min_samples', 30)
        
        # Statistical parameters
        self.mean = 0
        self.std = 0
        self.q1 = 0
        self.q3 = 0
        self.iqr = 0
        self.moving_average = deque(maxlen=self.window_size)
        
    def fit(self, data: np.ndarray) -> None:
        """Train statistical models on historical data"""
        if len(data) < self.min_samples:
            self.logger.warning(f"Insufficient data for training: {len(data)} < {self.min_samples}")
            return
        
        # Calculate statistical parameters
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1
        
        # Initialize moving average
        self.moving_average.extend(data[-self.window_size:])
        
        self.is_trained = True
        self.logger.info(f"Statistical detector trained on {len(data)} samples")
    
    def predict(self, data: np.ndarray) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods"""
        if not self.is_trained:
            self.logger.warning("Detector not trained, skipping prediction")
            return []
        
        anomalies = []
        current_time = time.time()
        
        for i, value in enumerate(data):
            timestamp = current_time - (len(data) - i - 1)
            
            # Z-score anomaly detection
            z_score = abs(value - self.mean) / self.std if self.std > 0 else 0
            if z_score > self.z_threshold:
                anomalies.append(AnomalyResult(
                    metric_name="unknown",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=self.get_severity(z_score),
                    score=z_score,
                    threshold=self.z_threshold,
                    value=value,
                    timestamp=timestamp,
                    context={'method': 'z_score', 'mean': self.mean, 'std': self.std},
                    description=f"Z-score anomaly: {z_score:.2f} > {self.z_threshold}"
                ))
            
            # IQR anomaly detection
            iqr_lower = self.q1 - self.iqr_multiplier * self.iqr
            iqr_upper = self.q3 + self.iqr_multiplier * self.iqr
            
            if value < iqr_lower or value > iqr_upper:
                iqr_score = max(abs(value - iqr_lower), abs(value - iqr_upper)) / self.iqr if self.iqr > 0 else 0
                anomalies.append(AnomalyResult(
                    metric_name="unknown",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=self.get_severity(iqr_score),
                    score=iqr_score,
                    threshold=self.iqr_multiplier,
                    value=value,
                    timestamp=timestamp,
                    context={'method': 'iqr', 'q1': self.q1, 'q3': self.q3, 'iqr': self.iqr},
                    description=f"IQR anomaly: value {value:.2f} outside [{iqr_lower:.2f}, {iqr_upper:.2f}]"
                ))
            
            # Moving average anomaly detection
            if len(self.moving_average) >= self.window_size:
                ma_value = np.mean(self.moving_average)
                ma_std = np.std(self.moving_average)
                
                if ma_std > 0:
                    ma_score = abs(value - ma_value) / ma_std
                    if ma_score > self.z_threshold:
                        anomalies.append(AnomalyResult(
                            metric_name="unknown",
                            anomaly_type=AnomalyType.STATISTICAL,
                            severity=self.get_severity(ma_score),
                            score=ma_score,
                            threshold=self.z_threshold,
                            value=value,
                            timestamp=timestamp,
                            context={'method': 'moving_average', 'window_size': self.window_size, 'ma_value': ma_value},
                            description=f"Moving average anomaly: {ma_score:.2f} > {self.z_threshold}"
                        ))
            
            # Update moving average
            self.moving_average.append(value)
        
        return anomalies
    
    def get_severity(self, score: float) -> AnomalySeverity:
        """Determine severity based on statistical score"""
        if score < 2.0:
            return AnomalySeverity.LOW
        elif score < 3.0:
            return AnomalySeverity.MEDIUM
        elif score < 4.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

class MLAnomalyDetector(AnomalyDetector):
    """Machine Learning-based anomaly detection using Isolation Forest and One-Class SVM"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ml_based", config)
        self.method = self.config.get('method', 'isolation_forest')  # 'isolation_forest' or 'one_class_svm'
        self.contamination = self.config.get('contamination', 0.1)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize models
        self.scaler = StandardScaler()
        self.model = self._create_model()
        
    def _create_model(self):
        """Create the ML model based on configuration"""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        elif self.method == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown ML method: {self.method}")
    
    def fit(self, data: np.ndarray) -> None:
        """Train ML model on historical data"""
        if len(data) < 50:
            self.logger.warning(f"Insufficient data for ML training: {len(data)} < 50")
            return
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.is_trained = True
        
        self.logger.info(f"ML detector ({self.method}) trained on {len(data)} samples")
    
    def predict(self, data: np.ndarray) -> List[AnomalyResult]:
        """Detect anomalies using ML model"""
        if not self.is_trained:
            self.logger.warning("ML detector not trained, skipping prediction")
            return []
        
        # Prepare features
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        predictions = self.model.predict(features_scaled)
        
        # Get anomaly scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features_scaled)
        else:
            scores = self.model.score_samples(features_scaled)
        
        # Convert to anomaly results
        anomalies = []
        current_time = time.time()
        
        for i, (prediction, score) in enumerate(zip(predictions, scores)):
            if prediction == -1:  # Anomaly detected
                timestamp = current_time - (len(data) - i - 1)
                
                # Normalize score to positive value for severity calculation
                abs_score = abs(score)
                
                anomalies.append(AnomalyResult(
                    metric_name="unknown",
                    anomaly_type=AnomalyType.ML_BASED,
                    severity=self.get_severity(abs_score),
                    score=abs_score,
                    threshold=0.0,  # ML models use decision functions
                    value=data[i],
                    timestamp=timestamp,
                    context={'method': self.method, 'raw_score': score, 'contamination': self.contamination},
                    description=f"ML anomaly detected using {self.method} (score: {score:.3f})"
                ))
        
        return anomalies
    
    def _prepare_features(self, data: np.ndarray) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        for i in range(len(data)):
            feature_vector = [data[i]]  # Current value
            
            # Add windowed features
            window_size = min(10, i + 1)
            if i >= window_size - 1:
                window = data[i-window_size+1:i+1]
                feature_vector.extend([
                    np.mean(window),
                    np.std(window),
                    np.min(window),
                    np.max(window),
                    np.median(window)
                ])
            else:
                feature_vector.extend([data[i]] * 5)  # Pad with current value
            
            # Add trend features
            if i >= 2:
                trend = np.polyfit(range(3), data[i-2:i+1], 1)[0]
                feature_vector.append(trend)
            else:
                feature_vector.append(0.0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_severity(self, score: float) -> AnomalySeverity:
        """Determine severity based on ML score"""
        # Normalize score based on typical ranges
        if score < 0.5:
            return AnomalySeverity.LOW
        elif score < 1.0:
            return AnomalySeverity.MEDIUM
        elif score < 2.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

class PatternAnomalyDetector(AnomalyDetector):
    """Pattern-based anomaly detection for time series data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("pattern", config)
        self.window_size = self.config.get('window_size', 50)
        self.min_pattern_length = self.config.get('min_pattern_length', 5)
        self.max_pattern_length = self.config.get('max_pattern_length', 20)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
        self.peak_threshold = self.config.get('peak_threshold', 2.0)
        
        # Pattern storage
        self.normal_patterns = []
        self.seasonal_patterns = []
        self.trend_patterns = []
        
    def fit(self, data: np.ndarray) -> None:
        """Learn patterns from historical data"""
        if len(data) < self.window_size:
            self.logger.warning(f"Insufficient data for pattern learning: {len(data)} < {self.window_size}")
            return
        
        # Extract normal patterns
        self.normal_patterns = self._extract_patterns(data)
        
        # Detect seasonal patterns
        self.seasonal_patterns = self._detect_seasonal_patterns(data)
        
        # Detect trend patterns
        self.trend_patterns = self._detect_trend_patterns(data)
        
        self.is_trained = True
        self.logger.info(f"Pattern detector trained: {len(self.normal_patterns)} normal patterns, "
                        f"{len(self.seasonal_patterns)} seasonal patterns, "
                        f"{len(self.trend_patterns)} trend patterns")
    
    def predict(self, data: np.ndarray) -> List[AnomalyResult]:
        """Detect pattern-based anomalies"""
        if not self.is_trained:
            self.logger.warning("Pattern detector not trained, skipping prediction")
            return []
        
        anomalies = []
        current_time = time.time()
        
        # Check for pattern violations
        for i in range(len(data) - self.min_pattern_length + 1):
            timestamp = current_time - (len(data) - i - 1)
            
            # Extract current pattern
            pattern_length = min(self.max_pattern_length, len(data) - i)
            current_pattern = data[i:i + pattern_length]
            
            # Check against normal patterns
            pattern_score = self._calculate_pattern_score(current_pattern, self.normal_patterns)
            
            if pattern_score > self.correlation_threshold:
                anomalies.append(AnomalyResult(
                    metric_name="unknown",
                    anomaly_type=AnomalyType.PATTERN,
                    severity=self.get_severity(pattern_score),
                    score=pattern_score,
                    threshold=self.correlation_threshold,
                    value=data[i],
                    timestamp=timestamp,
                    context={'method': 'pattern_violation', 'pattern_length': pattern_length},
                    description=f"Pattern anomaly: unusual pattern detected (score: {pattern_score:.3f})"
                ))
        
        # Check for peak anomalies
        peaks, _ = find_peaks(data, height=np.mean(data) + self.peak_threshold * np.std(data))
        for peak_idx in peaks:
            if peak_idx < len(data):
                timestamp = current_time - (len(data) - peak_idx - 1)
                peak_value = data[peak_idx]
                
                anomalies.append(AnomalyResult(
                    metric_name="unknown",
                    anomaly_type=AnomalyType.PATTERN,
                    severity=self.get_severity(peak_value / np.mean(data)),
                    score=peak_value / np.mean(data),
                    threshold=self.peak_threshold,
                    value=peak_value,
                    timestamp=timestamp,
                    context={'method': 'peak_detection', 'peak_index': peak_idx},
                    description=f"Peak anomaly: unusual peak detected at value {peak_value:.3f}"
                ))
        
        # Check for trend anomalies
        trend_anomalies = self._detect_trend_anomalies(data)
        for trend_anomaly in trend_anomalies:
            trend_anomaly.timestamp = current_time - (len(data) - trend_anomaly.timestamp - 1)
            anomalies.append(trend_anomaly)
        
        return anomalies
    
    def _extract_patterns(self, data: np.ndarray) -> List[np.ndarray]:
        """Extract normal patterns from data"""
        patterns = []
        
        for length in range(self.min_pattern_length, min(self.max_pattern_length, len(data) // 2)):
            for i in range(len(data) - length + 1):
                pattern = data[i:i + length]
                patterns.append(pattern)
        
        return patterns
    
    def _detect_seasonal_patterns(self, data: np.ndarray) -> List[np.ndarray]:
        """Detect seasonal patterns in data"""
        seasonal_patterns = []
        
        # Check for different seasonal periods
        for period in [24, 168, 720]:  # Daily, weekly, monthly (in hours)
            if len(data) >= period * 2:
                # Extract seasonal components
                seasonal_data = []
                for i in range(0, len(data), period):
                    if i + period <= len(data):
                        seasonal_data.append(data[i:i + period])
                
                if len(seasonal_data) >= 2:
                    # Average seasonal pattern
                    avg_seasonal = np.mean(seasonal_data, axis=0)
                    seasonal_patterns.append(avg_seasonal)
        
        return seasonal_patterns
    
    def _detect_trend_patterns(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect trend patterns in data"""
        trend_patterns = []
        
        # Calculate moving trends
        window_size = min(20, len(data) // 4)
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            
            # Linear regression for trend
            x = np.arange(len(window))
            slope, intercept = np.polyfit(x, window, 1)
            
            trend_patterns.append({
                'slope': slope,
                'intercept': intercept,
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'r_squared': np.corrcoef(x, window)[0, 1] ** 2
            })
        
        return trend_patterns
    
    def _calculate_pattern_score(self, pattern: np.ndarray, reference_patterns: List[np.ndarray]) -> float:
        """Calculate how anomalous a pattern is compared to reference patterns"""
        if not reference_patterns:
            return 0.0
        
        min_correlation = float('inf')
        
        for ref_pattern in reference_patterns:
            if len(ref_pattern) == len(pattern):
                correlation = np.corrcoef(pattern, ref_pattern)[0, 1]
                if not np.isnan(correlation):
                    min_correlation = min(min_correlation, abs(correlation))
        
        return 1.0 - min_correlation if min_correlation != float('inf') else 1.0
    
    def _detect_trend_anomalies(self, data: np.ndarray) -> List[AnomalyResult]:
        """Detect trend-based anomalies"""
        anomalies = []
        
        # Calculate current trend
        if len(data) >= 10:
            x = np.arange(len(data))
            current_slope, _ = np.polyfit(x, data, 1)
            
            # Compare with historical trends
            historical_slopes = [tp['slope'] for tp in self.trend_patterns]
            if historical_slopes:
                slope_mean = np.mean(historical_slopes)
                slope_std = np.std(historical_slopes)
                
                if slope_std > 0:
                    z_score = abs(current_slope - slope_mean) / slope_std
                    
                    if z_score > 2.0:  # Significant trend change
                        anomalies.append(AnomalyResult(
                            metric_name="unknown",
                            anomaly_type=AnomalyType.PATTERN,
                            severity=self.get_severity(z_score),
                            score=z_score,
                            threshold=2.0,
                            value=current_slope,
                            timestamp=len(data) - 1,  # Will be adjusted by caller
                            context={'method': 'trend_change', 'current_slope': current_slope, 'historical_mean': slope_mean},
                            description=f"Trend anomaly: unusual trend change detected (slope: {current_slope:.3f})"
                        ))
        
        return anomalies
    
    def get_severity(self, score: float) -> AnomalySeverity:
        """Determine severity based on pattern score"""
        if score < 0.5:
            return AnomalySeverity.LOW
        elif score < 0.7:
            return AnomalySeverity.MEDIUM
        elif score < 0.9:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

class DataQualityAnalyzer:
    """Comprehensive data quality scoring and analysis framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.completeness_threshold = self.config.get('completeness_threshold', 0.95)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.90)
        self.accuracy_threshold = self.config.get('accuracy_threshold', 0.95)
        self.timeliness_threshold = self.config.get('timeliness_threshold', 0.90)
        self.validity_threshold = self.config.get('validity_threshold', 0.95)
        
        # Historical quality scores
        self.quality_history = deque(maxlen=1000)
        
    def calculate_quality_score(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> QualityScore:
        """Calculate comprehensive quality score for data"""
        metadata = metadata or {}
        
        # Calculate individual quality dimensions
        completeness = self._calculate_completeness(data, metadata)
        consistency = self._calculate_consistency(data, metadata)
        accuracy = self._calculate_accuracy(data, metadata)
        timeliness = self._calculate_timeliness(data, metadata)
        validity = self._calculate_validity(data, metadata)
        
        # Calculate overall score (weighted average)
        weights = self.config.get('quality_weights', {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.25,
            'timeliness': 0.15,
            'validity': 0.15
        })
        
        overall_score = (
            completeness * weights['completeness'] +
            consistency * weights['consistency'] +
            accuracy * weights['accuracy'] +
            timeliness * weights['timeliness'] +
            validity * weights['validity']
        ) * 100
        
        quality_score = QualityScore(
            overall_score=overall_score,
            completeness=completeness * 100,
            consistency=consistency * 100,
            accuracy=accuracy * 100,
            timeliness=timeliness * 100,
            validity=validity * 100,
            timestamp=time.time(),
            details={
                'data_points': len(data),
                'weights': weights,
                'metadata': metadata
            }
        )
        
        # Store in history
        self.quality_history.append(quality_score)
        
        return quality_score
    
    def _calculate_completeness(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate data completeness (0-1 scale)"""
        if len(data) == 0:
            return 0.0
        
        # Check for missing values (NaN, None, etc.)
        missing_count = np.sum(np.isnan(data)) if data.dtype.kind in 'fc' else 0
        
        # Check for expected data points
        expected_count = metadata.get('expected_data_points', len(data))
        
        completeness = (len(data) - missing_count) / expected_count
        return max(0.0, min(1.0, completeness))
    
    def _calculate_consistency(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate data consistency (0-1 scale)"""
        if len(data) < 2:
            return 1.0
        
        # Check for outliers using IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return 1.0
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.sum((data < lower_bound) | (data > upper_bound))
        consistency = (len(data) - outliers) / len(data)
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_accuracy(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate data accuracy (0-1 scale)"""
        # Check against expected ranges
        min_expected = metadata.get('min_value', np.min(data))
        max_expected = metadata.get('max_value', np.max(data))
        
        if min_expected == max_expected:
            return 1.0
        
        # Count values within expected range
        within_range = np.sum((data >= min_expected) & (data <= max_expected))
        accuracy = within_range / len(data)
        
        return max(0.0, min(1.0, accuracy))
    
    def _calculate_timeliness(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate data timeliness (0-1 scale)"""
        current_time = time.time()
        data_timestamp = metadata.get('timestamp', current_time)
        
        # Check if data is recent enough
        max_age = metadata.get('max_age_seconds', 3600)  # 1 hour default
        age = current_time - data_timestamp
        
        if age <= 0:
            return 1.0
        
        timeliness = max(0.0, 1.0 - (age / max_age))
        return min(1.0, timeliness)
    
    def _calculate_validity(self, data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate data validity (0-1 scale)"""
        # Check for valid data types and formats
        invalid_count = 0
        
        # Check for infinite values
        if data.dtype.kind in 'fc':
            invalid_count += np.sum(np.isinf(data))
        
        # Check for negative values if not allowed
        if not metadata.get('allow_negative', True):
            invalid_count += np.sum(data < 0)
        
        # Check for zero values if not allowed
        if not metadata.get('allow_zero', True):
            invalid_count += np.sum(data == 0)
        
        validity = (len(data) - invalid_count) / len(data) if len(data) > 0 else 1.0
        return max(0.0, min(1.0, validity))
    
    def get_quality_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality trend analysis for specified time period"""
        if not self.quality_history:
            return {'status': 'no_data'}
        
        # Filter by time
        cutoff_time = time.time() - (hours * 3600)
        recent_scores = [qs for qs in self.quality_history if qs.timestamp >= cutoff_time]
        
        if not recent_scores:
            return {'status': 'no_recent_data'}
        
        # Calculate trends
        overall_scores = [qs.overall_score for qs in recent_scores]
        timestamps = [qs.timestamp for qs in recent_scores]
        
        # Linear regression for trend
        if len(overall_scores) >= 2:
            x = np.array(timestamps)
            y = np.array(overall_scores)
            trend_slope, _ = np.polyfit(x, y, 1)
        else:
            trend_slope = 0.0
        
        return {
            'count': len(recent_scores),
            'current_score': recent_scores[-1].overall_score,
            'average_score': np.mean(overall_scores),
            'min_score': np.min(overall_scores),
            'max_score': np.max(overall_scores),
            'trend_slope': trend_slope,
            'trend_direction': 'improving' if trend_slope > 0 else 'degrading' if trend_slope < 0 else 'stable',
            'scores_by_dimension': {
                'completeness': [qs.completeness for qs in recent_scores],
                'consistency': [qs.consistency for qs in recent_scores],
                'accuracy': [qs.accuracy for qs in recent_scores],
                'timeliness': [qs.timeliness for qs in recent_scores],
                'validity': [qs.validity for qs in recent_scores]
            }
        }
    
    def predict_quality_score(self, steps_ahead: int = 10) -> Dict[str, Any]:
        """Predict future quality scores based on historical data"""
        if len(self.quality_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Extract time series data
        recent_scores = list(self.quality_history)[-100:]  # Last 100 scores
        overall_scores = [qs.overall_score for qs in recent_scores]
        
        # Simple linear regression for prediction
        x = np.arange(len(overall_scores))
        slope, intercept = np.polyfit(x, overall_scores, 1)
        
        # Predict future scores
        future_x = np.arange(len(overall_scores), len(overall_scores) + steps_ahead)
        predicted_scores = slope * future_x + intercept
        
        # Calculate confidence intervals (simplified)
        residuals = overall_scores - (slope * x + intercept)
        mse = np.mean(residuals ** 2)
        std_error = np.sqrt(mse)
        
        return {
            'predicted_scores': predicted_scores.tolist(),
            'confidence_interval': (std_error * 1.96),  # 95% confidence
            'trend_slope': slope,
            'prediction_quality': 'good' if abs(slope) < 0.1 else 'moderate' if abs(slope) < 0.5 else 'poor',
            'steps_ahead': steps_ahead
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        if not self.quality_history:
            return {'status': 'no_data'}
        
        latest_score = self.quality_history[-1]
        trend_analysis = self.get_quality_trend()
        prediction = self.predict_quality_score()
        
        # Quality alerts
        alerts = []
        if latest_score.overall_score < 70:
            alerts.append({'level': 'high', 'message': 'Overall quality score below 70%'})
        if latest_score.completeness < 90:
            alerts.append({'level': 'medium', 'message': 'Data completeness below 90%'})
        if latest_score.consistency < 80:
            alerts.append({'level': 'medium', 'message': 'Data consistency below 80%'})
        
        return {
            'current_quality': {
                'overall_score': latest_score.overall_score,
                'completeness': latest_score.completeness,
                'consistency': latest_score.consistency,
                'accuracy': latest_score.accuracy,
                'timeliness': latest_score.timeliness,
                'validity': latest_score.validity
            },
            'trend_analysis': trend_analysis,
            'prediction': prediction,
            'alerts': alerts,
            'recommendations': self._generate_recommendations(latest_score, trend_analysis),
            'timestamp': time.time()
        }
    
    def _generate_recommendations(self, latest_score: QualityScore, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if latest_score.completeness < 90:
            recommendations.append("Improve data collection processes to reduce missing values")
        
        if latest_score.consistency < 80:
            recommendations.append("Implement data validation rules to catch outliers")
        
        if latest_score.accuracy < 90:
            recommendations.append("Review data sources and collection methods for accuracy")
        
        if latest_score.timeliness < 80:
            recommendations.append("Optimize data pipeline for faster processing")
        
        if latest_score.validity < 90:
            recommendations.append("Implement stricter data validation and type checking")
        
        if trend_analysis.get('trend_direction') == 'degrading':
            recommendations.append("Investigate recent changes that may be affecting quality")
        
        return recommendations

class RealTimeAnomalyDetector:
    """Real-time streaming anomaly detection system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.detectors = {
            'statistical': StatisticalAnomalyDetector(self.config.get('statistical', {})),
            'ml': MLAnomalyDetector(self.config.get('ml', {})),
            'pattern': PatternAnomalyDetector(self.config.get('pattern', {}))
        }
        
        # Real-time processing
        self.processing_queue = Queue()
        self.anomaly_queue = Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        self.threshold_adaptation_rate = self.config.get('threshold_adaptation_rate', 0.1)
        
        # Anomaly aggregation
        self.anomaly_buffer = deque(maxlen=1000)
        self.anomaly_summary = defaultdict(list)
        
        # Alert suppression
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # 5 minutes
        self.last_alert_time = {}
        
    def start(self):
        """Start real-time anomaly detection"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Real-time anomaly detection started")
    
    def stop(self):
        """Stop real-time anomaly detection"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("Real-time anomaly detection stopped")
    
    def train_detectors(self, historical_data: Dict[str, np.ndarray]):
        """Train all detectors with historical data"""
        for metric_name, data in historical_data.items():
            if len(data) > 0:
                for detector in self.detectors.values():
                    detector.fit(data)
                
                # Initialize adaptive thresholds
                self.adaptive_thresholds[metric_name] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'upper_bound': np.mean(data) + 2 * np.std(data),
                    'lower_bound': np.mean(data) - 2 * np.std(data)
                }
        
        self.logger.info(f"Trained detectors for {len(historical_data)} metrics")
    
    def process_metric(self, metric_name: str, value: float, timestamp: float = None):
        """Process a single metric value for anomaly detection"""
        if timestamp is None:
            timestamp = time.time()
        
        # Add to processing queue
        self.processing_queue.put({
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp
        })
    
    def _processing_loop(self):
        """Main processing loop for real-time anomaly detection"""
        while self.is_running:
            try:
                # Get metric from queue
                if not self.processing_queue.empty():
                    metric_data = self.processing_queue.get(timeout=1)
                    self._detect_anomalies(metric_data)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _detect_anomalies(self, metric_data: Dict[str, Any]):
        """Detect anomalies for a single metric"""
        metric_name = metric_data['metric_name']
        value = metric_data['value']
        timestamp = metric_data['timestamp']
        
        anomalies = []
        
        # Run all detectors
        for detector_name, detector in self.detectors.items():
            try:
                if detector.is_trained:
                    # Prepare data for detector
                    data = np.array([value])
                    detector_anomalies = detector.predict(data)
                    
                    # Update metric name in results
                    for anomaly in detector_anomalies:
                        anomaly.metric_name = metric_name
                        anomaly.timestamp = timestamp
                        anomalies.append(anomaly)
                        
            except Exception as e:
                self.logger.error(f"Error in {detector_name} detector: {e}")
        
        # Check adaptive thresholds
        if metric_name in self.adaptive_thresholds:
            threshold_anomaly = self._check_adaptive_threshold(metric_name, value, timestamp)
            if threshold_anomaly:
                anomalies.append(threshold_anomaly)
        
        # Process anomalies
        if anomalies:
            self._process_anomalies(anomalies)
            
        # Update adaptive thresholds
        self._update_adaptive_thresholds(metric_name, value)
    
    def _check_adaptive_threshold(self, metric_name: str, value: float, timestamp: float) -> Optional[AnomalyResult]:
        """Check value against adaptive thresholds"""
        thresholds = self.adaptive_thresholds[metric_name]
        
        if value > thresholds['upper_bound'] or value < thresholds['lower_bound']:
            # Calculate severity based on how far from bounds
            upper_distance = abs(value - thresholds['upper_bound'])
            lower_distance = abs(value - thresholds['lower_bound'])
            distance = min(upper_distance, lower_distance)
            
            score = distance / thresholds['std'] if thresholds['std'] > 0 else 0
            
            severity = AnomalySeverity.LOW
            if score > 3:
                severity = AnomalySeverity.CRITICAL
            elif score > 2:
                severity = AnomalySeverity.HIGH
            elif score > 1:
                severity = AnomalySeverity.MEDIUM
            
            return AnomalyResult(
                metric_name=metric_name,
                anomaly_type=AnomalyType.THRESHOLD,
                severity=severity,
                score=score,
                threshold=thresholds['upper_bound'] if value > thresholds['upper_bound'] else thresholds['lower_bound'],
                value=value,
                timestamp=timestamp,
                context={'adaptive_threshold': True, 'bounds': thresholds},
                description=f"Adaptive threshold anomaly: {value:.3f} outside bounds [{thresholds['lower_bound']:.3f}, {thresholds['upper_bound']:.3f}]"
            )
        
        return None
    
    def _update_adaptive_thresholds(self, metric_name: str, value: float):
        """Update adaptive thresholds based on new data"""
        if metric_name not in self.adaptive_thresholds:
            return
        
        thresholds = self.adaptive_thresholds[metric_name]
        
        # Exponential moving average
        alpha = self.threshold_adaptation_rate
        thresholds['mean'] = (1 - alpha) * thresholds['mean'] + alpha * value
        
        # Update standard deviation estimate
        variance = (value - thresholds['mean']) ** 2
        thresholds['std'] = np.sqrt((1 - alpha) * thresholds['std'] ** 2 + alpha * variance)
        
        # Update bounds
        thresholds['upper_bound'] = thresholds['mean'] + 2 * thresholds['std']
        thresholds['lower_bound'] = thresholds['mean'] - 2 * thresholds['std']
    
    def _process_anomalies(self, anomalies: List[AnomalyResult]):
        """Process detected anomalies"""
        for anomaly in anomalies:
            # Add to buffer
            self.anomaly_buffer.append(anomaly)
            
            # Add to summary
            self.anomaly_summary[anomaly.metric_name].append({
                'timestamp': anomaly.timestamp,
                'severity': anomaly.severity,
                'score': anomaly.score,
                'type': anomaly.anomaly_type
            })
            
            # Add to anomaly queue for alerting
            self.anomaly_queue.put(anomaly)
    
    def get_recent_anomalies(self, hours: int = 1) -> List[AnomalyResult]:
        """Get recent anomalies"""
        cutoff_time = time.time() - (hours * 3600)
        return [a for a in self.anomaly_buffer if a.timestamp >= cutoff_time]
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly statistics"""
        recent_anomalies = self.get_recent_anomalies(24)  # Last 24 hours
        
        if not recent_anomalies:
            return {'status': 'no_anomalies'}
        
        # Group by metric
        by_metric = defaultdict(list)
        for anomaly in recent_anomalies:
            by_metric[anomaly.metric_name].append(anomaly)
        
        # Group by severity
        by_severity = defaultdict(int)
        for anomaly in recent_anomalies:
            by_severity[anomaly.severity.name] += 1
        
        # Group by type
        by_type = defaultdict(int)
        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type.value] += 1
        
        return {
            'total_anomalies': len(recent_anomalies),
            'anomalies_by_metric': {k: len(v) for k, v in by_metric.items()},
            'anomalies_by_severity': dict(by_severity),
            'anomalies_by_type': dict(by_type),
            'most_problematic_metrics': sorted(by_metric.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        }

class AlertManager:
    """Multi-channel alert management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Alert channels
        self.channels = {
            'email': self.config.get('email', {}),
            'webhook': self.config.get('webhook', {}),
            'log': self.config.get('log', {'enabled': True})
        }
        
        # Alert rules
        self.alert_rules = self.config.get('alert_rules', {})
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        
        # Alert suppression
        self.suppression_rules = self.config.get('suppression_rules', {})
        self.suppressed_alerts = {}
        
    def send_alert(self, anomaly: AnomalyResult):
        """Send alert through configured channels"""
        # Check if alert should be suppressed
        if self._should_suppress_alert(anomaly):
            return
        
        # Create alert message
        alert_message = self._create_alert_message(anomaly)
        
        # Send through all enabled channels
        for channel_name, channel_config in self.channels.items():
            if channel_config.get('enabled', True):
                try:
                    if channel_name == 'email':
                        self._send_email_alert(alert_message, channel_config)
                    elif channel_name == 'webhook':
                        self._send_webhook_alert(alert_message, channel_config)
                    elif channel_name == 'log':
                        self._send_log_alert(alert_message, channel_config)
                except Exception as e:
                    self.logger.error(f"Error sending alert via {channel_name}: {e}")
        
        # Record alert
        self.alert_history.append({
            'timestamp': time.time(),
            'anomaly': anomaly,
            'message': alert_message,
            'channels_sent': list(self.channels.keys())
        })
    
    def _should_suppress_alert(self, anomaly: AnomalyResult) -> bool:
        """Check if alert should be suppressed"""
        alert_key = f"{anomaly.metric_name}_{anomaly.anomaly_type.value}"
        
        # Check cooldown
        if alert_key in self.suppressed_alerts:
            last_alert_time = self.suppressed_alerts[alert_key]
            cooldown_period = self.suppression_rules.get('cooldown_seconds', 300)
            
            if time.time() - last_alert_time < cooldown_period:
                return True
        
        # Check severity threshold
        min_severity = self.suppression_rules.get('min_severity', AnomalySeverity.LOW)
        if anomaly.severity.value < min_severity.value:
            return True
        
        return False
    
    def _create_alert_message(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Create alert message from anomaly"""
        return {
            'title': f"Anomaly Alert: {anomaly.metric_name}",
            'severity': anomaly.severity.name,
            'metric': anomaly.metric_name,
            'value': anomaly.value,
            'score': anomaly.score,
            'threshold': anomaly.threshold,
            'type': anomaly.anomaly_type.value,
            'description': anomaly.description,
            'timestamp': datetime.fromtimestamp(anomaly.timestamp).isoformat(),
            'context': anomaly.context
        }
    
    def _send_email_alert(self, message: Dict[str, Any], config: Dict[str, Any]):
        """Send email alert"""
        if not config.get('smtp_server'):
            return
        
        self.logger.info(f"Email alert would be sent: {message['title']}")
        # Email sending implementation would go here
    
    def _send_webhook_alert(self, message: Dict[str, Any], config: Dict[str, Any]):
        """Send webhook alert"""
        webhook_url = config.get('url')
        if not webhook_url:
            return
        
        self.logger.info(f"Webhook alert would be sent to {webhook_url}: {message['title']}")
        # Webhook sending implementation would go here
    
    def _send_log_alert(self, message: Dict[str, Any], config: Dict[str, Any]):
        """Send log alert"""
        log_level = config.get('level', 'ERROR')
        
        log_message = f"ANOMALY ALERT [{message['severity']}] {message['metric']}: {message['description']}"
        
        if log_level == 'ERROR':
            self.logger.error(log_message)
        elif log_level == 'WARNING':
            self.logger.warning(log_message)
        elif log_level == 'INFO':
            self.logger.info(log_message)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        if not self.alert_history:
            return {'status': 'no_alerts'}
        
        recent_alerts = [a for a in self.alert_history if a['timestamp'] >= time.time() - 86400]  # Last 24 hours
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'total_alerts_all_time': len(self.alert_history),
            'alerts_by_severity': self._group_alerts_by_severity(recent_alerts),
            'alerts_by_metric': self._group_alerts_by_metric(recent_alerts),
            'most_recent_alert': self.alert_history[-1]['timestamp'] if self.alert_history else None
        }
    
    def _group_alerts_by_severity(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by severity"""
        severity_count = defaultdict(int)
        for alert in alerts:
            severity_count[alert['anomaly'].severity.name] += 1
        return dict(severity_count)
    
    def _group_alerts_by_metric(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by metric"""
        metric_count = defaultdict(int)
        for alert in alerts:
            metric_count[alert['anomaly'].metric_name] += 1
        return dict(metric_count)

class PredictiveFailureDetector:
    """Predictive failure detection using machine learning"""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.feature_history = defaultdict(lambda: deque(maxlen=window_size))
        self.failure_patterns = defaultdict(list)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Model training parameters
        self.retrain_interval = 3600  # 1 hour
        self.last_retrain = {}
        self.prediction_cache = {}
        
    def add_metric_data(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Add metric data for analysis"""
        with self.lock:
            timestamp = time.time()
            
            # Create feature vector
            features = self._extract_features(metric_name, value, metadata or {})
            
            # Add to history
            self.feature_history[metric_name].append({
                'timestamp': timestamp,
                'value': value,
                'features': features
            })
            
            # Train model if needed
            if self._should_retrain(metric_name):
                self._train_model(metric_name)
    
    def _extract_features(self, metric_name: str, value: float, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features from metric data"""
        history = self.feature_history[metric_name]
        
        # Basic features
        features = [value]
        
        if len(history) > 0:
            # Recent values
            recent_values = [h['value'] for h in list(history)[-10:]]
            features.extend([
                np.mean(recent_values),
                np.std(recent_values),
                np.median(recent_values),
                max(recent_values) - min(recent_values),  # Range
                recent_values[-1] - recent_values[0] if len(recent_values) > 1 else 0  # Trend
            ])
            
            # Time-based features
            if len(history) > 1:
                time_diffs = [history[i]['timestamp'] - history[i-1]['timestamp'] 
                             for i in range(1, len(history))]
                features.extend([
                    np.mean(time_diffs),
                    np.std(time_diffs)
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # System state features
        features.extend([
            psutil.cpu_percent(),
            psutil.virtual_memory().percent,
            psutil.disk_usage('/').percent
        ])
        
        # Metadata features
        features.append(metadata.get('load_factor', 0))
        features.append(metadata.get('concurrent_requests', 0))
        
        return np.array(features)
    
    def _should_retrain(self, metric_name: str) -> bool:
        """Check if model should be retrained"""
        if metric_name not in self.models:
            return len(self.feature_history[metric_name]) >= self.window_size // 2
        
        last_train = self.last_retrain.get(metric_name, 0)
        return (time.time() - last_train) > self.retrain_interval
    
    def _train_model(self, metric_name: str):
        """Train anomaly detection model"""
        with self.lock:
            history = self.feature_history[metric_name]
            
            if len(history) < 10:
                return
            
            # Prepare training data
            X = []
            for record in history:
                X.append(record['features'])
            
            X = np.array(X)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            self.last_retrain[metric_name] = time.time()
            
            self.logger.info(f"Trained failure detection model for {metric_name}")
    
    def predict_failure(self, metric_name: str, value: float, metadata: Dict[str, Any] = None) -> PredictionResult:
        """Predict potential failure"""
        with self.lock:
            if metric_name not in self.models:
                # Return neutral prediction
                return PredictionResult(
                    metric_name=metric_name,
                    prediction_type=PredictionType.FAILURE,
                    predicted_value=0.0,
                    confidence=0.0,
                    timestamp=time.time(),
                    horizon_minutes=0,
                    risk_score=0.0,
                    details={'status': 'model_not_trained'}
                )
            
            # Extract features
            features = self._extract_features(metric_name, value, metadata or {})
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scalers[metric_name].transform([features])
            
            # Predict anomaly
            anomaly_score = self.models[metric_name].decision_function(X_scaled)[0]
            is_anomaly = self.models[metric_name].predict(X_scaled)[0] == -1
            
            # Calculate risk score (0-1)
            risk_score = max(0, (0.5 - anomaly_score) * 2)
            
            # Estimate failure probability
            failure_probability = risk_score if is_anomaly else 0.0
            
            # Estimate time to failure
            horizon_minutes = self._estimate_failure_horizon(metric_name, risk_score)
            
            return PredictionResult(
                metric_name=metric_name,
                prediction_type=PredictionType.FAILURE,
                predicted_value=failure_probability,
                confidence=min(1.0, abs(anomaly_score) * 2),
                timestamp=time.time(),
                horizon_minutes=horizon_minutes,
                risk_score=risk_score,
                details={
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                    'features_used': len(features)
                }
            )
    
    def _estimate_failure_horizon(self, metric_name: str, risk_score: float) -> int:
        """Estimate time to failure in minutes"""
        if risk_score < 0.3:
            return 0
        elif risk_score < 0.6:
            return 60  # 1 hour
        elif risk_score < 0.8:
            return 30  # 30 minutes
        else:
            return 15  # 15 minutes
    
    def detect_anomalies(self, metric_name: str, lookback_hours: int = 1) -> List[Dict[str, Any]]:
        """Detect anomalies in historical data"""
        with self.lock:
            if metric_name not in self.models:
                return []
            
            history = self.feature_history[metric_name]
            cutoff_time = time.time() - (lookback_hours * 3600)
            
            anomalies = []
            for record in history:
                if record['timestamp'] < cutoff_time:
                    continue
                
                features = np.nan_to_num(record['features'], nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = self.scalers[metric_name].transform([features])
                
                if self.models[metric_name].predict(X_scaled)[0] == -1:
                    anomalies.append({
                        'timestamp': record['timestamp'],
                        'value': record['value'],
                        'anomaly_score': self.models[metric_name].decision_function(X_scaled)[0]
                    })
            
            return anomalies
    
    def get_model_info(self, metric_name: str) -> Dict[str, Any]:
        """Get information about trained model"""
        with self.lock:
            if metric_name not in self.models:
                return {'status': 'no_model'}
            
            return {
                'model_type': 'IsolationForest',
                'contamination': self.contamination,
                'training_samples': len(self.feature_history[metric_name]),
                'last_retrain': self.last_retrain.get(metric_name, 0),
                'features_count': len(self.feature_history[metric_name][0]['features']) if self.feature_history[metric_name] else 0
            }

class CapacityPlanner:
    """Capacity planning and resource forecasting"""
    
    def __init__(self, forecasting_window: int = 24):
        self.forecasting_window = forecasting_window  # hours
        self.models = {}
        self.resource_limits = {}
        self.growth_patterns = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Default resource limits
        self.default_limits = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'gpu_memory_usage': 85.0,
            'data_throughput': 1000.0,
            'concurrent_requests': 100
        }
        
        # Initialize with defaults
        self.resource_limits.update(self.default_limits)
    
    def set_resource_limit(self, resource_type: str, limit: float):
        """Set resource capacity limit"""
        with self.lock:
            self.resource_limits[resource_type] = limit
            self.logger.info(f"Set {resource_type} limit to {limit}")
    
    def analyze_growth_pattern(self, metric_name: str, history: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze growth patterns in metrics"""
        if len(history) < 10:
            return {'status': 'insufficient_data'}
        
        # Extract time series data
        timestamps = [m.timestamp for m in history]
        values = [m.value for m in history]
        
        # Convert to relative time (hours)
        base_time = min(timestamps)
        x = np.array([(t - base_time) / 3600 for t in timestamps]).reshape(-1, 1)
        y = np.array(values)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Calculate growth rate
        growth_rate = model.coef_[0]  # units per hour
        
        # Calculate R-squared
        y_pred = model.predict(x)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Detect trend
        if growth_rate > 0.01:
            trend = 'increasing'
        elif growth_rate < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate volatility
        volatility = np.std(y - y_pred)
        
        return {
            'trend': trend,
            'growth_rate': growth_rate,
            'r_squared': r_squared,
            'volatility': volatility,
            'model': model,
            'current_value': values[-1],
            'predicted_next_hour': model.predict([[x[-1][0] + 1]])[0]
        }
    
    def predict_capacity_need(self, metric_name: str, history: List[PerformanceMetric]) -> CapacityPrediction:
        """Predict when capacity limit will be reached"""
        with self.lock:
            current_time = time.time()
            
            # Analyze growth pattern
            growth_analysis = self.analyze_growth_pattern(metric_name, history)
            
            if growth_analysis['status'] == 'insufficient_data':
                return CapacityPrediction(
                    resource_type=metric_name,
                    current_usage=0,
                    predicted_usage=0,
                    capacity_limit=self.resource_limits.get(metric_name, 100),
                    time_to_limit=None,
                    recommendation="Insufficient data for prediction",
                    confidence=0.0,
                    timestamp=current_time
                )
            
            current_usage = growth_analysis['current_value']
            growth_rate = growth_analysis['growth_rate']
            capacity_limit = self.resource_limits.get(metric_name, 100)
            
            # Predict usage for next forecasting window
            predicted_usage = growth_analysis['predicted_next_hour']
            
            # Calculate time to reach limit
            time_to_limit = None
            if growth_rate > 0 and current_usage < capacity_limit:
                hours_to_limit = (capacity_limit - current_usage) / growth_rate
                time_to_limit = current_time + (hours_to_limit * 3600)
            
            # Generate recommendation
            recommendation = self._generate_capacity_recommendation(
                metric_name, current_usage, predicted_usage, capacity_limit, 
                time_to_limit, growth_analysis
            )
            
            # Calculate confidence based on R-squared and data quality
            confidence = min(1.0, growth_analysis['r_squared'] * 
                           (len(history) / 100) * 
                           (1 - growth_analysis['volatility'] / max(1, current_usage)))
            
            return CapacityPrediction(
                resource_type=metric_name,
                current_usage=current_usage,
                predicted_usage=predicted_usage,
                capacity_limit=capacity_limit,
                time_to_limit=time_to_limit,
                recommendation=recommendation,
                confidence=confidence,
                timestamp=current_time
            )
    
    def _generate_capacity_recommendation(self, metric_name: str, current: float, 
                                        predicted: float, limit: float, 
                                        time_to_limit: Optional[float], 
                                        growth_analysis: Dict[str, Any]) -> str:
        """Generate capacity planning recommendation"""
        utilization = current / limit
        predicted_utilization = predicted / limit
        
        if time_to_limit and time_to_limit - time.time() < 3600:  # Less than 1 hour
            return f"URGENT: {metric_name} will reach capacity limit in less than 1 hour. Immediate action required."
        elif time_to_limit and time_to_limit - time.time() < 86400:  # Less than 24 hours
            hours = (time_to_limit - time.time()) / 3600
            return f"WARNING: {metric_name} will reach capacity limit in {hours:.1f} hours. Plan scaling action."
        elif predicted_utilization > 0.8:
            return f"CAUTION: {metric_name} predicted to reach 80% capacity utilization. Monitor closely."
        elif growth_analysis['trend'] == 'increasing':
            return f"INFO: {metric_name} showing increasing trend. Consider capacity planning."
        else:
            return f"OK: {metric_name} capacity utilization is within normal range."
    
    def forecast_resource_usage(self, metric_name: str, history: List[PerformanceMetric], 
                              forecast_hours: int = 24) -> Dict[str, Any]:
        """Forecast resource usage for specified time horizon"""
        with self.lock:
            growth_analysis = self.analyze_growth_pattern(metric_name, history)
            
            if growth_analysis['status'] == 'insufficient_data':
                return {'status': 'insufficient_data'}
            
            model = growth_analysis['model']
            current_time = time.time()
            
            # Generate forecast points
            forecast_points = []
            for hour in range(1, forecast_hours + 1):
                # Predict usage
                predicted_value = model.predict([[hour]])[0]
                
                # Add some uncertainty
                uncertainty = growth_analysis['volatility'] * np.sqrt(hour)
                
                forecast_points.append({
                    'hour': hour,
                    'timestamp': current_time + (hour * 3600),
                    'predicted_value': predicted_value,
                    'upper_bound': predicted_value + uncertainty,
                    'lower_bound': max(0, predicted_value - uncertainty),
                    'confidence': max(0, 1 - (hour / forecast_hours) * 0.5)
                })
            
            # Identify critical points
            capacity_limit = self.resource_limits.get(metric_name, 100)
            critical_points = []
            
            for point in forecast_points:
                if point['predicted_value'] > capacity_limit * 0.8:
                    critical_points.append(point)
            
            return {
                'forecast_points': forecast_points,
                'critical_points': critical_points,
                'capacity_limit': capacity_limit,
                'current_trend': growth_analysis['trend'],
                'growth_rate': growth_analysis['growth_rate'],
                'confidence': growth_analysis['r_squared']
            }
    
    def get_capacity_recommendations(self, metrics_history: Dict[str, List[PerformanceMetric]]) -> List[Dict[str, Any]]:
        """Get capacity recommendations for all monitored resources"""
        recommendations = []
        
        for metric_name, history in metrics_history.items():
            if metric_name in self.resource_limits:
                prediction = self.predict_capacity_need(metric_name, history)
                
                recommendations.append({
                    'metric': metric_name,
                    'prediction': prediction,
                    'priority': self._get_recommendation_priority(prediction),
                    'timestamp': time.time()
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _get_recommendation_priority(self, prediction: CapacityPrediction) -> int:
        """Get priority score for recommendation"""
        if prediction.time_to_limit:
            hours_remaining = (prediction.time_to_limit - time.time()) / 3600
            if hours_remaining < 1:
                return 100  # Critical
            elif hours_remaining < 24:
                return 80   # High
            elif hours_remaining < 168:  # 1 week
                return 60   # Medium
            else:
                return 40   # Low
        
        utilization = prediction.current_usage / prediction.capacity_limit
        if utilization > 0.9:
            return 90
        elif utilization > 0.8:
            return 70
        elif utilization > 0.7:
            return 50
        else:
            return 30
    
    def simulate_scaling_scenarios(self, metric_name: str, history: List[PerformanceMetric], 
                                 scaling_factors: List[float]) -> Dict[str, Any]:
        """Simulate different scaling scenarios"""
        scenarios = {}
        
        for factor in scaling_factors:
            new_limit = self.resource_limits.get(metric_name, 100) * factor
            
            # Temporarily update limit
            old_limit = self.resource_limits.get(metric_name, 100)
            self.resource_limits[metric_name] = new_limit
            
            # Get prediction with new limit
            prediction = self.predict_capacity_need(metric_name, history)
            
            # Calculate cost-benefit
            cost_increase = factor - 1.0
            benefit_hours = 0
            
            if prediction.time_to_limit:
                benefit_hours = (prediction.time_to_limit - time.time()) / 3600
            
            scenarios[f"scale_{factor}x"] = {
                'scaling_factor': factor,
                'new_limit': new_limit,
                'prediction': prediction,
                'cost_increase': cost_increase,
                'benefit_hours': benefit_hours,
                'cost_benefit_ratio': benefit_hours / cost_increase if cost_increase > 0 else float('inf')
            }
            
            # Restore old limit
            self.resource_limits[metric_name] = old_limit
        
        return scenarios

class TrendForecaster:
    """Advanced trend forecasting for performance metrics"""
    
    def __init__(self, seasonal_periods: int = 24):
        self.seasonal_periods = seasonal_periods  # Hours for seasonal patterns
        self.models = {}
        self.trend_cache = {}
        self.seasonal_patterns = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def analyze_seasonality(self, metric_name: str, history: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze seasonal patterns in metrics"""
        if len(history) < self.seasonal_periods * 2:
            return {'status': 'insufficient_data'}
        
        # Extract hourly data
        hourly_data = defaultdict(list)
        for metric in history:
            hour = datetime.fromtimestamp(metric.timestamp).hour
            hourly_data[hour].append(metric.value)
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour in range(24):
            if hour in hourly_data:
                hourly_averages[hour] = np.mean(hourly_data[hour])
            else:
                hourly_averages[hour] = 0
        
        # Detect seasonal strength
        overall_mean = np.mean(list(hourly_averages.values()))
        seasonal_variance = np.var(list(hourly_averages.values()))
        seasonal_strength = seasonal_variance / overall_mean if overall_mean > 0 else 0
        
        # Find peak hours
        peak_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)[:3]
        low_hours = sorted(hourly_averages.items(), key=lambda x: x[1])[:3]
        
        return {
            'seasonal_strength': seasonal_strength,
            'hourly_averages': hourly_averages,
            'peak_hours': [h[0] for h in peak_hours],
            'low_hours': [h[0] for h in low_hours],
            'overall_mean': overall_mean,
            'seasonal_variance': seasonal_variance
        }
    
    def decompose_trend(self, metric_name: str, history: List[PerformanceMetric]) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        if len(history) < 50:
            return {'status': 'insufficient_data'}
        
        # Create time series
        timestamps = [m.timestamp for m in history]
        values = [m.value for m in history]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data)
        
        # Convert to pandas series with datetime index
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        ts = pd.Series(values, index=dates)
        
        # Resample to hourly data
        ts_hourly = ts.resample('H').mean().fillna(method='forward')
        
        if len(ts_hourly) < 24:
            return {'status': 'insufficient_data'}
        
        # Simple trend decomposition
        # Calculate trend using rolling mean
        trend = ts_hourly.rolling(window=min(24, len(ts_hourly)//2), center=True).mean()
        
        # Calculate seasonal component
        seasonal = ts_hourly.groupby(ts_hourly.index.hour).transform('mean')
        
        # Calculate residual
        residual = ts_hourly - trend - seasonal + ts_hourly.mean()
        
        return {
            'original': ts_hourly.values,
            'trend': trend.values,
            'seasonal': seasonal.values,
            'residual': residual.values,
            'timestamps': [ts.timestamp() for ts in ts_hourly.index],
            'trend_slope': self._calculate_trend_slope(trend.dropna()),
            'seasonal_amplitude': seasonal.std(),
            'noise_level': residual.std()
        }
    
    def _calculate_trend_slope(self, trend_series: pd.Series) -> float:
        """Calculate trend slope"""
        if len(trend_series) < 2:
            return 0.0
        
        x = np.arange(len(trend_series))
        y = trend_series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Linear regression
        slope, _ = np.polyfit(x_clean, y_clean, 1)
        return slope
    
    def forecast_trend(self, metric_name: str, history: List[PerformanceMetric], 
                      forecast_hours: int = 24) -> Dict[str, Any]:
        """Forecast trend using decomposition and extrapolation"""
        decomposition = self.decompose_trend(metric_name, history)
        
        if decomposition['status'] == 'insufficient_data':
            return {'status': 'insufficient_data'}
        
        seasonality = self.analyze_seasonality(metric_name, history)
        
        # Current time
        current_time = time.time()
        
        # Generate forecast
        forecast_points = []
        
        for hour in range(1, forecast_hours + 1):
            future_time = current_time + (hour * 3600)
            future_hour = datetime.fromtimestamp(future_time).hour
            
            # Trend component
            trend_value = decomposition['trend'][-1] if decomposition['trend'] is not None else 0
            trend_contribution = trend_value + (decomposition['trend_slope'] * hour)
            
            # Seasonal component
            seasonal_contribution = seasonality['hourly_averages'].get(future_hour, 0)
            
            # Combine components
            predicted_value = trend_contribution + seasonal_contribution - seasonality['overall_mean']
            
            # Add uncertainty
            uncertainty = decomposition['noise_level'] * np.sqrt(hour)
            
            forecast_points.append({
                'hour': hour,
                'timestamp': future_time,
                'predicted_value': max(0, predicted_value),
                'trend_component': trend_contribution,
                'seasonal_component': seasonal_contribution,
                'upper_bound': predicted_value + uncertainty,
                'lower_bound': max(0, predicted_value - uncertainty),
                'confidence': max(0, 1 - (hour / forecast_hours) * 0.3)
            })
        
        return {
            'forecast_points': forecast_points,
            'decomposition': decomposition,
            'seasonality': seasonality,
            'forecast_summary': {
                'trend_direction': 'increasing' if decomposition['trend_slope'] > 0 else 'decreasing' if decomposition['trend_slope'] < 0 else 'stable',
                'trend_strength': abs(decomposition['trend_slope']),
                'seasonal_strength': seasonality['seasonal_strength'],
                'predictability': 1 - (decomposition['noise_level'] / max(1, np.mean(decomposition['original'])))
            }
        }
    
    def detect_trend_changes(self, metric_name: str, history: List[PerformanceMetric], 
                           window_size: int = 20) -> List[Dict[str, Any]]:
        """Detect significant trend changes"""
        if len(history) < window_size * 2:
            return []
        
        changes = []
        values = [m.value for m in history]
        timestamps = [m.timestamp for m in history]
        
        # Sliding window analysis
        for i in range(window_size, len(values) - window_size):
            # Calculate trends before and after
            before_values = values[i-window_size:i]
            after_values = values[i:i+window_size]
            
            # Linear regression for each window
            x_before = np.arange(len(before_values))
            x_after = np.arange(len(after_values))
            
            slope_before = np.polyfit(x_before, before_values, 1)[0]
            slope_after = np.polyfit(x_after, after_values, 1)[0]
            
            # Check for significant change
            slope_change = slope_after - slope_before
            
            # Statistical significance test
            if abs(slope_change) > np.std(values) * 0.1:
                changes.append({
                    'timestamp': timestamps[i],
                    'change_point': i,
                    'slope_before': slope_before,
                    'slope_after': slope_after,
                    'slope_change': slope_change,
                    'significance': abs(slope_change) / np.std(values),
                    'change_type': 'acceleration' if slope_change > 0 else 'deceleration'
                })
        
        return changes
    
    def predict_anomalous_periods(self, metric_name: str, history: List[PerformanceMetric], 
                                forecast_hours: int = 24) -> List[Dict[str, Any]]:
        """Predict periods when metrics might behave anomalously"""
        forecast = self.forecast_trend(metric_name, history, forecast_hours)
        
        if forecast['status'] == 'insufficient_data':
            return []
        
        anomalous_periods = []
        
        # Calculate thresholds based on historical data
        historical_values = [m.value for m in history]
        mean_value = np.mean(historical_values)
        std_value = np.std(historical_values)
        
        upper_threshold = mean_value + (2 * std_value)
        lower_threshold = max(0, mean_value - (2 * std_value))
        
        # Check forecast points
        for point in forecast['forecast_points']:
            if (point['predicted_value'] > upper_threshold or 
                point['predicted_value'] < lower_threshold or
                point['confidence'] < 0.5):
                
                anomalous_periods.append({
                    'timestamp': point['timestamp'],
                    'hour': point['hour'],
                    'predicted_value': point['predicted_value'],
                    'threshold_exceeded': point['predicted_value'] > upper_threshold,
                    'threshold_undershot': point['predicted_value'] < lower_threshold,
                    'low_confidence': point['confidence'] < 0.5,
                    'risk_level': 'high' if point['confidence'] < 0.3 else 'medium'
                })
        
        return anomalous_periods
    
    def get_trend_summary(self, metric_name: str, history: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get comprehensive trend summary"""
        if len(history) < 10:
            return {'status': 'insufficient_data'}
        
        decomposition = self.decompose_trend(metric_name, history)
        seasonality = self.analyze_seasonality(metric_name, history)
        trend_changes = self.detect_trend_changes(metric_name, history)
        
        # Calculate trend statistics
        values = [m.value for m in history]
        recent_values = values[-10:]
        
        summary = {
            'current_value': values[-1],
            'mean_value': np.mean(values),
            'trend_direction': 'increasing' if decomposition['trend_slope'] > 0 else 'decreasing' if decomposition['trend_slope'] < 0 else 'stable',
            'trend_strength': abs(decomposition['trend_slope']),
            'seasonal_pattern': seasonality['seasonal_strength'] > 0.1,
            'volatility': np.std(values),
            'recent_trend': np.mean(recent_values[-5:]) - np.mean(recent_values[:5]) if len(recent_values) >= 5 else 0,
            'trend_changes_count': len(trend_changes),
            'predictability_score': 1 - (decomposition['noise_level'] / max(1, np.mean(values))),
            'peak_hours': seasonality['peak_hours'],
            'low_hours': seasonality['low_hours'],
            'last_updated': time.time()
        }
        
        return summary

class ProactiveAlertSystem:
    """Proactive alerting system with machine learning"""
    
    def __init__(self, failure_detector: PredictiveFailureDetector, capacity_planner: CapacityPlanner):
        self.failure_detector = failure_detector
        self.capacity_planner = capacity_planner
        self.alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.suppression_rules = {}
        self.escalation_policies = {}
        self.notification_channels = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds
        self.alert_thresholds = {
            'failure_risk': 0.7,
            'capacity_warning': 0.8,
            'capacity_critical': 0.9,
            'anomaly_score': 0.6
        }
        
        # Alert cooldown periods (seconds)
        self.cooldown_periods = {
            AlertSeverity.LOW: 300,      # 5 minutes
            AlertSeverity.MEDIUM: 600,   # 10 minutes
            AlertSeverity.HIGH: 1800,    # 30 minutes
            AlertSeverity.CRITICAL: 3600  # 1 hour
        }
    
    def add_notification_channel(self, channel_name: str, channel_type: str, config: Dict[str, Any]):
        """Add notification channel"""
        with self.lock:
            self.notification_channels[channel_name] = {
                'type': channel_type,
                'config': config,
                'enabled': True,
                'last_used': 0
            }
    
    def set_escalation_policy(self, policy_name: str, levels: List[Dict[str, Any]]):
        """Set escalation policy"""
        with self.lock:
            self.escalation_policies[policy_name] = levels
    
    def analyze_metric_for_alerts(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Analyze metric and generate proactive alerts"""
        with self.lock:
            current_time = time.time()
            alerts_generated = []
            
            # Add to failure detector
            self.failure_detector.add_metric_data(metric_name, value, metadata)
            
            # Check for failure prediction
            failure_prediction = self.failure_detector.predict_failure(metric_name, value, metadata)
            
            if failure_prediction.risk_score > self.alert_thresholds['failure_risk']:
                alert = self._create_alert(
                    metric_name=metric_name,
                    severity=self._get_severity_from_risk(failure_prediction.risk_score),
                    message=f"Predicted failure risk for {metric_name}: {failure_prediction.risk_score:.2%}",
                    value=value,
                    threshold=self.alert_thresholds['failure_risk'],
                    prediction_based=True,
                    details={
                        'prediction': failure_prediction.__dict__,
                        'risk_score': failure_prediction.risk_score,
                        'horizon_minutes': failure_prediction.horizon_minutes
                    }
                )
                alerts_generated.append(alert)
            
            # Check capacity thresholds
            if metric_name in self.capacity_planner.resource_limits:
                capacity_limit = self.capacity_planner.resource_limits[metric_name]
                utilization = value / capacity_limit
                
                if utilization > self.alert_thresholds['capacity_critical']:
                    alert = self._create_alert(
                        metric_name=metric_name,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Critical capacity utilization for {metric_name}: {utilization:.1%}",
                        value=value,
                        threshold=capacity_limit * self.alert_thresholds['capacity_critical'],
                        prediction_based=False,
                        details={
                            'utilization': utilization,
                            'capacity_limit': capacity_limit
                        }
                    )
                    alerts_generated.append(alert)
                
                elif utilization > self.alert_thresholds['capacity_warning']:
                    alert = self._create_alert(
                        metric_name=metric_name,
                        severity=AlertSeverity.HIGH,
                        message=f"High capacity utilization for {metric_name}: {utilization:.1%}",
                        value=value,
                        threshold=capacity_limit * self.alert_thresholds['capacity_warning'],
                        prediction_based=False,
                        details={
                            'utilization': utilization,
                            'capacity_limit': capacity_limit
                        }
                    )
                    alerts_generated.append(alert)
            
            # Process generated alerts
            for alert in alerts_generated:
                if self._should_suppress_alert(alert):
                    continue
                
                self._process_alert(alert)
            
            return alerts_generated
    
    def _create_alert(self, metric_name: str, severity: AlertSeverity, message: str, 
                     value: float, threshold: float, prediction_based: bool = False,
                     details: Dict[str, Any] = None) -> Alert:
        """Create new alert"""
        alert_id = str(uuid.uuid4())
        
        return Alert(
            id=alert_id,
            metric_name=metric_name,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=time.time(),
            prediction_based=prediction_based,
            details=details or {}
        )
    
    def _get_severity_from_risk(self, risk_score: float) -> AlertSeverity:
        """Get alert severity from risk score"""
        if risk_score >= 0.9:
            return AlertSeverity.CRITICAL
        elif risk_score >= 0.8:
            return AlertSeverity.HIGH
        elif risk_score >= 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        current_time = time.time()
        
        # Check cooldown period
        similar_alerts = [a for a in self.alert_history 
                         if a.metric_name == alert.metric_name and 
                         a.severity == alert.severity and
                         current_time - a.timestamp < self.cooldown_periods[alert.severity]]
        
        if similar_alerts:
            return True
        
        # Check suppression rules
        for rule_name, rule in self.suppression_rules.items():
            if self._matches_suppression_rule(alert, rule):
                return True
        
        return False
    
    def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule"""
        # Simple rule matching
        if 'metric_name' in rule and rule['metric_name'] != alert.metric_name:
            return False
        
        if 'severity' in rule and rule['severity'] != alert.severity:
            return False
        
        if 'time_window' in rule:
            current_time = time.time()
            if current_time - alert.timestamp > rule['time_window']:
                return False
        
        return True
    
    def _process_alert(self, alert: Alert):
        """Process and store alert"""
        with self.lock:
            # Store alert
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            # Log alert
            self.logger.warning(f"Alert generated: {alert.message}")
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for channel_name, channel in self.notification_channels.items():
            if not channel['enabled']:
                continue
            
            try:
                if channel['type'] == 'email':
                    self._send_email_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_notification(alert, channel['config'])
                elif channel['type'] == 'slack':
                    self._send_slack_notification(alert, channel['config'])
                
                channel['last_used'] = time.time()
                
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"Alert: {alert.severity.value.upper()} - {alert.metric_name}"
        
        body = f"""
        Alert ID: {alert.id}
        Metric: {alert.metric_name}
        Severity: {alert.severity.value.upper()}
        Message: {alert.message}
        Value: {alert.value}
        Threshold: {alert.threshold}
        Timestamp: {datetime.fromtimestamp(alert.timestamp)}
        
        Details: {json.dumps(alert.details, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config.get('use_tls'):
            server.starttls()
        if config.get('username'):
            server.login(config['username'], config['password'])
        
        server.send_message(msg)
        server.quit()
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        payload = {
            'alert_id': alert.id,
            'metric_name': alert.metric_name,
            'severity': alert.severity.value,
            'message': alert.message,
            'value': alert.value,
            'threshold': alert.threshold,
            'timestamp': alert.timestamp,
            'prediction_based': alert.prediction_based,
            'details': alert.details
        }
        
        response = requests.post(config['url'], json=payload, timeout=30)
        response.raise_for_status()
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        color = {
            AlertSeverity.LOW: 'good',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.HIGH: 'danger',
            AlertSeverity.CRITICAL: 'danger'
        }
        
        payload = {
            'channel': config['channel'],
            'username': config.get('username', 'PerformanceMonitor'),
            'attachments': [{
                'color': color[alert.severity],
                'title': f"Alert: {alert.metric_name}",
                'text': alert.message,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                    {'title': 'Value', 'value': str(alert.value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                    {'title': 'Prediction Based', 'value': str(alert.prediction_based), 'short': True}
                ],
                'timestamp': int(alert.timestamp)
            }]
        }
        
        response = requests.post(config['webhook_url'], json=payload, timeout=30)
        response.raise_for_status()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = time.time()
                
                self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                
                self.logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self.lock:
            total_alerts = len(self.alert_history)
            if total_alerts == 0:
                return {'total_alerts': 0}
            
            severity_counts = defaultdict(int)
            status_counts = defaultdict(int)
            prediction_based_count = 0
            
            for alert in self.alert_history:
                severity_counts[alert.severity.value] += 1
                status_counts[alert.status.value] += 1
                if alert.prediction_based:
                    prediction_based_count += 1
            
            return {
                'total_alerts': total_alerts,
                'severity_distribution': dict(severity_counts),
                'status_distribution': dict(status_counts),
                'prediction_based_alerts': prediction_based_count,
                'prediction_based_percentage': (prediction_based_count / total_alerts) * 100
            }

class MetricsCollector:
    """Enhanced metrics collector with anomaly detection and quality monitoring"""
    
    def __init__(self, max_history: int = 10000, anomaly_config: Dict[str, Any] = None):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Metric metadata
        self.metric_definitions = {}
        
        # Aggregation settings
        self.aggregation_window = 60  # 1 minute
        self.aggregated_metrics = {}
        
        # Alert thresholds
        self.alert_thresholds = {}
        self.alert_callbacks = []
        
        # Enhanced anomaly detection and quality monitoring
        self.anomaly_detector = RealTimeAnomalyDetector(anomaly_config)
        self.quality_analyzer = DataQualityAnalyzer(anomaly_config)
        self.alert_manager = AlertManager(anomaly_config)
        
        # Start anomaly detection
        self.anomaly_detector.start()
        
        # Monitor anomaly queue
        self.anomaly_monitor_thread = threading.Thread(target=self._monitor_anomalies)
        self.anomaly_monitor_thread.daemon = True
        self.anomaly_monitor_thread.start()
    
    def register_metric(self, 
                       name: str,
                       unit: str,
                       category: str,
                       description: str,
                       alert_threshold: Optional[float] = None):
        """Register a new metric type"""
        with self.lock:
            self.metric_definitions[name] = {
                'unit': unit,
                'category': category,
                'description': description,
                'alert_threshold': alert_threshold
            }
            
            if alert_threshold is not None:
                self.alert_thresholds[name] = alert_threshold
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value with enhanced anomaly detection and quality monitoring"""
        with self.lock:
            metric_def = self.metric_definitions.get(name)
            if not metric_def:
                self.logger.warning(f"Unknown metric: {name}")
                return
            
            timestamp = time.time()
            metadata = metadata or {}
            
            # Calculate quality score
            historical_values = np.array([m.value for m in self.metrics[name]])
            if len(historical_values) > 0:
                combined_data = np.append(historical_values, value)
                quality_score = self.quality_analyzer.calculate_quality_score(
                    combined_data, 
                    {**metadata, 'timestamp': timestamp}
                )
            else:
                quality_score = None
            
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=metric_def['unit'],
                timestamp=timestamp,
                category=metric_def['category'],
                metadata=metadata,
                quality_score=quality_score
            )
            
            self.metrics[name].append(metric)
            
            # Process with anomaly detector
            self.anomaly_detector.process_metric(name, value, timestamp)
            
            # Check alert threshold (legacy)
            if name in self.alert_thresholds and value > self.alert_thresholds[name]:
                self._trigger_alert(name, value, self.alert_thresholds[name])
    
    def _monitor_anomalies(self):
        """Monitor anomaly queue and send alerts"""
        while True:
            try:
                if not self.anomaly_detector.anomaly_queue.empty():
                    anomaly = self.anomaly_detector.anomaly_queue.get(timeout=1)
                    
                    # Update metric with anomaly flag
                    self._update_metric_with_anomaly(anomaly)
                    
                    # Send alert
                    self.alert_manager.send_alert(anomaly)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error monitoring anomalies: {e}")
                time.sleep(1)
    
    def _update_metric_with_anomaly(self, anomaly: AnomalyResult):
        """Update metric with anomaly flag"""
        with self.lock:
            metric_name = anomaly.metric_name
            if metric_name in self.metrics and self.metrics[metric_name]:
                # Find the metric with matching timestamp
                for metric in reversed(self.metrics[metric_name]):
                    if abs(metric.timestamp - anomaly.timestamp) < 1.0:  # Within 1 second
                        metric.anomaly_flags.append(anomaly)
                        break
    
    def train_anomaly_detectors(self):
        """Train anomaly detectors with historical data"""
        with self.lock:
            historical_data = {}
            for name, metrics in self.metrics.items():
                if len(metrics) > 30:  # Minimum data for training
                    historical_data[name] = np.array([m.value for m in metrics])
            
            if historical_data:
                self.anomaly_detector.train_detectors(historical_data)
                self.logger.info(f"Trained anomaly detectors for {len(historical_data)} metrics")
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[PerformanceMetric]:
        """Get metric history for specified time period"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistical summary of metric"""
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {'status': 'no_data'}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'latest': values[-1],
            'time_range': {
                'start': history[0].timestamp,
                'end': history[-1].timestamp
            }
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get enhanced summary of all metrics with quality and anomaly information"""
        with self.lock:
            summary = {}
            
            for name in self.metrics.keys():
                base_stats = self.get_metric_statistics(name)
                
                # Add quality information
                quality_trend = self.quality_analyzer.get_quality_trend(24)
                
                # Add anomaly information
                recent_anomalies = self.anomaly_detector.get_recent_anomalies(24)
                metric_anomalies = [a for a in recent_anomalies if a.metric_name == name]
                
                summary[name] = {
                    **base_stats,
                    'quality_trend': quality_trend,
                    'anomaly_count_24h': len(metric_anomalies),
                    'recent_anomalies': [
                        {
                            'severity': a.severity.name,
                            'score': a.score,
                            'type': a.anomaly_type.value,
                            'timestamp': a.timestamp
                        } for a in metric_anomalies[-5:]  # Last 5 anomalies
                    ]
                }
            
            return summary
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report"""
        return self.quality_analyzer.generate_quality_report()
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly statistics"""
        return self.anomaly_detector.get_anomaly_statistics()
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return self.alert_manager.get_alert_statistics()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        return {
            'metrics_summary': self.get_all_metrics_summary(),
            'quality_report': self.get_quality_report(),
            'anomaly_statistics': self.get_anomaly_statistics(),
            'alert_statistics': self.get_alert_statistics(),
            'system_status': self._get_system_status()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        quality_report = self.get_quality_report()
        anomaly_stats = self.get_anomaly_statistics()
        alert_stats = self.get_alert_statistics()
        
        # Determine overall health
        overall_health = "HEALTHY"
        
        if quality_report.get('current_quality', {}).get('overall_score', 100) < 70:
            overall_health = "DEGRADED"
        
        if anomaly_stats.get('total_anomalies', 0) > 10:
            overall_health = "CRITICAL" if overall_health == "DEGRADED" else "WARNING"
        
        if alert_stats.get('total_alerts_24h', 0) > 50:
            overall_health = "CRITICAL"
        
        return {
            'overall_health': overall_health,
            'metrics_count': len(self.metrics),
            'active_detectors': len([d for d in self.anomaly_detector.detectors.values() if d.is_trained]),
            'quality_score': quality_report.get('current_quality', {}).get('overall_score', 0),
            'anomaly_count_24h': anomaly_stats.get('total_anomalies', 0),
            'alert_count_24h': alert_stats.get('total_alerts_24h', 0),
            'timestamp': time.time()
        }
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add callback for metric alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger alert for metric threshold violation"""
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        with self.lock:
            if format == 'json':
                data = {}
                for name, metrics in self.metrics.items():
                    data[name] = [
                        {
                            'value': m.value,
                            'timestamp': m.timestamp,
                            'metadata': m.metadata
                        }
                        for m in metrics
                    ]
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'csv':
                # Export as CSV
                rows = []
                for name, metrics in self.metrics.items():
                    for m in metrics:
                        rows.append({
                            'metric_name': name,
                            'value': m.value,
                            'unit': m.unit,
                            'category': m.category,
                            'timestamp': m.timestamp,
                            'datetime': pd.to_datetime(m.timestamp, unit='s')
                        })
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'anomaly_detector'):
            self.anomaly_detector.stop()
        self.logger.info("MetricsCollector cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

class DataLoadingBenchmark:
    """Benchmark data loading performance"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.results = []
        self.logger = logging.getLogger(__name__)
    
    def benchmark_loading_performance(self, 
                                    timeframes: List[str],
                                    iterations: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark data loading performance across timeframes"""
        results = {}
        
        for timeframe in timeframes:
            self.logger.info(f"Benchmarking {timeframe} loading...")
            timeframe_results = []
            
            for i in range(iterations):
                # Clear cache for accurate measurement
                self.data_loader.clear_cache()
                
                # Measure loading time
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                try:
                    data = self.data_loader.load_data(timeframe)
                    success = True
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    duration = end_time - start_time
                    memory_usage = end_memory - start_memory
                    throughput = len(data) / duration if duration > 0 else 0
                    
                    result = BenchmarkResult(
                        test_name=f"{timeframe}_load_iteration_{i+1}",
                        duration_seconds=duration,
                        throughput_items_per_second=throughput,
                        memory_usage_mb=memory_usage,
                        success=success,
                        details={
                            'rows_loaded': len(data),
                            'columns': len(data.columns),
                            'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                        }
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=f"{timeframe}_load_iteration_{i+1}",
                        duration_seconds=0,
                        throughput_items_per_second=0,
                        memory_usage_mb=0,
                        success=False,
                        details={'error': str(e)}
                    )
                
                timeframe_results.append(result)
                self.results.append(result)
            
            results[timeframe] = timeframe_results
        
        return results
    
    def benchmark_chunked_loading(self, 
                                 timeframe: str,
                                 chunk_sizes: List[int]) -> Dict[int, BenchmarkResult]:
        """Benchmark chunked loading performance"""
        results = {}
        
        for chunk_size in chunk_sizes:
            self.logger.info(f"Benchmarking chunked loading with chunk size {chunk_size}...")
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                total_rows = 0
                chunks_processed = 0
                
                for chunk in self.data_loader.load_chunked_data(timeframe, chunk_size):
                    total_rows += len(chunk)
                    chunks_processed += 1
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_usage = end_memory - start_memory
                throughput = total_rows / duration if duration > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"chunked_load_{chunk_size}",
                    duration_seconds=duration,
                    throughput_items_per_second=throughput,
                    memory_usage_mb=memory_usage,
                    success=True,
                    details={
                        'chunk_size': chunk_size,
                        'total_rows': total_rows,
                        'chunks_processed': chunks_processed,
                        'avg_chunk_processing_time': duration / chunks_processed if chunks_processed > 0 else 0
                    }
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"chunked_load_{chunk_size}",
                    duration_seconds=0,
                    throughput_items_per_second=0,
                    memory_usage_mb=0,
                    success=False,
                    details={'error': str(e)}
                )
            
            results[chunk_size] = result
            self.results.append(result)
        
        return results
    
    def benchmark_caching_performance(self, timeframe: str) -> Dict[str, BenchmarkResult]:
        """Benchmark caching performance"""
        results = {}
        
        # First load (cache miss)
        self.data_loader.clear_cache()
        
        start_time = time.time()
        data = self.data_loader.load_data(timeframe)
        end_time = time.time()
        
        cache_miss_result = BenchmarkResult(
            test_name="cache_miss",
            duration_seconds=end_time - start_time,
            throughput_items_per_second=len(data) / (end_time - start_time),
            memory_usage_mb=0,
            success=True,
            details={'cache_status': 'miss', 'rows': len(data)}
        )
        
        results['cache_miss'] = cache_miss_result
        
        # Second load (cache hit)
        start_time = time.time()
        data = self.data_loader.load_data(timeframe)
        end_time = time.time()
        
        cache_hit_result = BenchmarkResult(
            test_name="cache_hit",
            duration_seconds=end_time - start_time,
            throughput_items_per_second=len(data) / (end_time - start_time),
            memory_usage_mb=0,
            success=True,
            details={'cache_status': 'hit', 'rows': len(data)}
        )
        
        results['cache_hit'] = cache_hit_result
        
        # Calculate speedup
        speedup = cache_miss_result.duration_seconds / cache_hit_result.duration_seconds if cache_hit_result.duration_seconds > 0 else 0
        
        results['speedup'] = BenchmarkResult(
            test_name="cache_speedup",
            duration_seconds=0,
            throughput_items_per_second=speedup,
            memory_usage_mb=0,
            success=True,
            details={'speedup_factor': speedup}
        )
        
        return results
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        if not self.results:
            return {'status': 'no_benchmarks_run'}
        
        # Group by test type
        by_test_type = defaultdict(list)
        for result in self.results:
            test_type = result.test_name.split('_')[0]
            by_test_type[test_type].append(result)
        
        summary = {}
        for test_type, results in by_test_type.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                durations = [r.duration_seconds for r in successful_results]
                throughputs = [r.throughput_items_per_second for r in successful_results]
                memory_usages = [r.memory_usage_mb for r in successful_results]
                
                summary[test_type] = {
                    'total_tests': len(results),
                    'successful_tests': len(successful_results),
                    'success_rate': len(successful_results) / len(results),
                    'avg_duration': np.mean(durations),
                    'avg_throughput': np.mean(throughputs),
                    'avg_memory_usage': np.mean(memory_usages),
                    'best_duration': min(durations),
                    'best_throughput': max(throughputs),
                    'min_memory_usage': min(memory_usages)
                }
        
        return summary

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.update_interval = 5  # seconds
        self.dashboard_thread = None
        self.running = False
        
        # Chart settings
        plt.style.use('seaborn-v0_8')
        self.figure_size = (15, 10)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def start_dashboard(self):
        """Start real-time dashboard"""
        self.running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        self.logger.info("Performance dashboard started")
    
    def stop_dashboard(self):
        """Stop dashboard"""
        self.running = False
        if self.dashboard_thread:
            self.dashboard_thread.join()
        self.logger.info("Performance dashboard stopped")
    
    def _dashboard_loop(self):
        """Main dashboard update loop"""
        while self.running:
            try:
                self.update_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                time.sleep(self.update_interval)
    
    def update_dashboard(self):
        """Update dashboard with latest metrics"""
        # Get current metrics with enhanced information
        metrics_summary = self.metrics_collector.get_system_health_summary()
        
        if not metrics_summary.get('metrics_summary'):
            return
        
        # Create enhanced dashboard plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🚀 Enhanced NQ Data Pipeline Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: System Health Overview
        self._plot_system_health_overview(axes[0, 0], metrics_summary)
        
        # Plot 2: Quality Trends
        self._plot_quality_trends(axes[0, 1], metrics_summary)
        
        # Plot 3: Anomaly Statistics
        self._plot_anomaly_statistics(axes[0, 2], metrics_summary)
        
        # Plot 4: Loading Performance
        self._plot_loading_performance(axes[1, 0], metrics_summary['metrics_summary'])
        
        # Plot 5: Alert Summary
        self._plot_alert_summary(axes[1, 1], metrics_summary)
        
        # Plot 6: Throughput Trends
        self._plot_throughput_trends(axes[1, 2], metrics_summary['metrics_summary'])
        
        plt.tight_layout()
        plt.savefig('/tmp/enhanced_performance_dashboard.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_system_health_overview(self, ax, metrics_summary):
        """Plot system health overview"""
        ax.set_title('System Health Overview')
        
        system_status = metrics_summary.get('system_status', {})
        health_status = system_status.get('overall_health', 'UNKNOWN')
        
        # Health indicator
        colors = {'HEALTHY': 'green', 'WARNING': 'yellow', 'DEGRADED': 'orange', 'CRITICAL': 'red'}
        color = colors.get(health_status, 'gray')
        
        # Create status pie chart
        ax.pie([1], labels=[f'{health_status}\nSystem'], colors=[color], autopct='')
        
        # Add key metrics as text
        metrics_text = f"""
        Quality Score: {system_status.get('quality_score', 0):.1f}%
        Anomalies (24h): {system_status.get('anomaly_count_24h', 0)}
        Alerts (24h): {system_status.get('alert_count_24h', 0)}
        Active Detectors: {system_status.get('active_detectors', 0)}
        """
        ax.text(1.2, 0.5, metrics_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    def _plot_quality_trends(self, ax, metrics_summary):
        """Plot quality trends"""
        ax.set_title('Data Quality Trends')
        
        quality_report = metrics_summary.get('quality_report', {})
        current_quality = quality_report.get('current_quality', {})
        
        if current_quality:
            dimensions = ['completeness', 'consistency', 'accuracy', 'timeliness', 'validity']
            values = [current_quality.get(dim, 0) for dim in dimensions]
            
            bars = ax.bar(dimensions, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_ylabel('Quality Score (%)')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_anomaly_statistics(self, ax, metrics_summary):
        """Plot anomaly statistics"""
        ax.set_title('Anomaly Statistics (24h)')
        
        anomaly_stats = metrics_summary.get('anomaly_statistics', {})
        
        if anomaly_stats.get('status') != 'no_anomalies':
            by_severity = anomaly_stats.get('anomalies_by_severity', {})
            by_type = anomaly_stats.get('anomalies_by_type', {})
            
            if by_severity:
                # Create stacked bar chart
                severities = list(by_severity.keys())
                counts = list(by_severity.values())
                
                bars = ax.bar(severities, counts, color=['green', 'yellow', 'orange', 'red'])
                ax.set_ylabel('Count')
                ax.set_xlabel('Severity')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_alert_summary(self, ax, metrics_summary):
        """Plot alert summary"""
        ax.set_title('Alert Summary')
        
        alert_stats = metrics_summary.get('alert_statistics', {})
        
        if alert_stats.get('status') != 'no_alerts':
            alerts_24h = alert_stats.get('total_alerts_24h', 0)
            alerts_total = alert_stats.get('total_alerts_all_time', 0)
            
            categories = ['24h Alerts', 'Total Alerts']
            values = [alerts_24h, alerts_total]
            
            bars = ax.bar(categories, values, color=['orange', 'blue'])
            ax.set_ylabel('Count')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                       str(value), ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No alerts generated', ha='center', va='center', transform=ax.transAxes)

class PerformanceMonitor:
    """Enhanced performance monitoring system with advanced anomaly detection"""
    
    def __init__(self, enable_dashboard: bool = True, anomaly_config: Dict[str, Any] = None):
        self.metrics_collector = MetricsCollector(anomaly_config=anomaly_config)
        self.dashboard = PerformanceDashboard(self.metrics_collector) if enable_dashboard else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
        
        # Start dashboard if enabled
        if self.dashboard:
            self.dashboard.start_dashboard()
        
        # Auto-train detectors after initialization
        self._auto_train_detectors()
    
    def _initialize_standard_metrics(self):
        """Initialize standard performance metrics"""
        standard_metrics = [
            ('data_load_time', 'seconds', 'loading', 'Time to load data from disk'),
            ('chunk_load_time', 'seconds', 'loading', 'Time to load data chunks'),
            ('cache_load_time', 'seconds', 'loading', 'Time to load data from cache'),
            ('data_throughput', 'items/second', 'throughput', 'Data processing throughput'),
            ('memory_usage', 'MB', 'memory', 'Memory usage'),
            ('shared_pool_usage', 'MB', 'memory', 'Shared memory pool usage'),
            ('gpu_memory_usage', 'MB', 'memory', 'GPU memory usage'),
            ('validation_time', 'seconds', 'processing', 'Data validation time'),
            ('preprocessing_time', 'seconds', 'processing', 'Data preprocessing time'),
            ('stream_latency', 'milliseconds', 'streaming', 'Data stream latency'),
            ('concurrent_processing_time', 'seconds', 'processing', 'Concurrent processing time')
        ]
        
        for name, unit, category, description in standard_metrics:
            self.metrics_collector.register_metric(name, unit, category, description)
    
    def _auto_train_detectors(self):
        """Auto-train anomaly detectors with available historical data"""
        # Delay training to allow some metrics to be collected
        def delayed_training():
            time.sleep(30)  # Wait 30 seconds
            self.metrics_collector.train_anomaly_detectors()
        
        training_thread = threading.Thread(target=delayed_training)
        training_thread.daemon = True
        training_thread.start()
    
    def create_benchmark_suite(self, data_loader) -> DataLoadingBenchmark:
        """Create benchmark suite for data loader"""
        return DataLoadingBenchmark(data_loader)
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        self.metrics_collector.record_metric(name, value, metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return self.metrics_collector.get_system_health_summary()
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality report"""
        return self.metrics_collector.get_quality_report()
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly statistics"""
        return self.metrics_collector.get_anomaly_statistics()
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return self.metrics_collector.get_alert_statistics()
    
    def train_anomaly_detectors(self):
        """Manually train anomaly detectors"""
        self.metrics_collector.train_anomaly_detectors()
    
    def add_alert(self, metric_name: str, threshold: float, callback: Callable[[str, float, float], None]):
        """Add performance alert"""
        self.metrics_collector.alert_thresholds[metric_name] = threshold
        self.metrics_collector.add_alert_callback(callback)
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics"""
        self.metrics_collector.export_metrics(filepath, format)
    
    def generate_report(self, output_file: str = 'enhanced_performance_report.html'):
        """Generate enhanced performance report"""
        if self.dashboard:
            self.dashboard.generate_enhanced_performance_report(output_file)
        else:
            self.logger.warning("Dashboard not enabled, cannot generate report")
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        if self.dashboard:
            self.dashboard.stop_dashboard()
        if self.metrics_collector:
            self.metrics_collector.cleanup()
        self.logger.info("Performance monitor cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Context manager for performance measurement
class PerformanceTimer:
    """Context manager for measuring performance"""
    
    def __init__(self, performance_monitor: PerformanceMonitor, metric_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.performance_monitor = performance_monitor
        self.metric_name = metric_name
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.performance_monitor.record_metric(self.metric_name, duration, self.metadata)