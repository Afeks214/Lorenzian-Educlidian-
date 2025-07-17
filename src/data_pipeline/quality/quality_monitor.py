"""
Data quality monitoring and anomaly detection for real-time data pipelines

This module implements comprehensive data quality monitoring with real-time
anomaly detection, statistical analysis, and automated alerting.
"""

import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import statistics
import math
from concurrent.futures import ThreadPoolExecutor
import queue
import json
from pathlib import Path
import sqlite3
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_BREAK = "trend_break"
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    INVALID_FORMAT = "invalid_format"
    STALE_DATA = "stale_data"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_BURST = "volatility_burst"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Data quality metric"""
    name: str
    value: float
    threshold: float
    status: str  # 'good', 'warning', 'error'
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    anomaly_type: AnomalyType
    severity: AlertSeverity
    message: str
    value: Any
    expected_range: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    timestamp: float
    overall_score: float  # 0-100
    total_records: int
    valid_records: int
    invalid_records: int
    metrics: List[QualityMetric]
    anomalies: List[AnomalyAlert]
    statistics: Dict[str, Any]
    recommendations: List[str]

class StatisticalAnomalyDetector:
    """Statistical anomaly detection using multiple methods"""
    
    def __init__(self, window_size: int = 1000, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.data_buffer = deque(maxlen=window_size)
        self.lock = threading.RLock()
        
        # Statistical thresholds
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.mad_threshold = 3.0
        
        # Models for ML-based detection
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.model_trained = False
    
    def add_data_point(self, value: float, timestamp: Optional[float] = None):
        """Add data point to buffer"""
        with self.lock:
            self.data_buffer.append({
                'value': value,
                'timestamp': timestamp or time.time()
            })
    
    def detect_z_score_anomalies(self, value: float) -> Optional[AnomalyAlert]:
        """Detect anomalies using Z-score method"""
        with self.lock:
            if len(self.data_buffer) < 30:  # Need minimum data
                return None
            
            values = [point['value'] for point in self.data_buffer]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            if std_val == 0:
                return None
            
            z_score = abs(value - mean_val) / std_val
            
            if z_score > self.z_score_threshold:
                return AnomalyAlert(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.WARNING if z_score < 4 else AlertSeverity.ERROR,
                    message=f"Z-score anomaly detected: {z_score:.2f} (threshold: {self.z_score_threshold})",
                    value=value,
                    expected_range=(mean_val - 3*std_val, mean_val + 3*std_val),
                    confidence=min(1.0, z_score / 10.0),
                    metadata={'z_score': z_score, 'mean': mean_val, 'std': std_val}
                )
        
        return None
    
    def detect_iqr_anomalies(self, value: float) -> Optional[AnomalyAlert]:
        """Detect anomalies using Interquartile Range method"""
        with self.lock:
            if len(self.data_buffer) < 30:
                return None
            
            values = [point['value'] for point in self.data_buffer]
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            if value < lower_bound or value > upper_bound:
                distance = min(abs(value - lower_bound), abs(value - upper_bound))
                normalized_distance = distance / (iqr if iqr > 0 else 1)
                
                return AnomalyAlert(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.WARNING if normalized_distance < 2 else AlertSeverity.ERROR,
                    message=f"IQR anomaly detected: {value:.2f} outside range [{lower_bound:.2f}, {upper_bound:.2f}]",
                    value=value,
                    expected_range=(lower_bound, upper_bound),
                    confidence=min(1.0, normalized_distance / 5.0),
                    metadata={'q1': q1, 'q3': q3, 'iqr': iqr, 'distance': distance}
                )
        
        return None
    
    def detect_mad_anomalies(self, value: float) -> Optional[AnomalyAlert]:
        """Detect anomalies using Median Absolute Deviation method"""
        with self.lock:
            if len(self.data_buffer) < 30:
                return None
            
            values = [point['value'] for point in self.data_buffer]
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            
            if mad == 0:
                return None
            
            # Modified Z-score using MAD
            mad_score = 0.6745 * (value - median) / mad
            
            if abs(mad_score) > self.mad_threshold:
                return AnomalyAlert(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.WARNING if abs(mad_score) < 4 else AlertSeverity.ERROR,
                    message=f"MAD anomaly detected: MAD score {mad_score:.2f} (threshold: {self.mad_threshold})",
                    value=value,
                    expected_range=(median - 3*mad, median + 3*mad),
                    confidence=min(1.0, abs(mad_score) / 10.0),
                    metadata={'mad_score': mad_score, 'median': median, 'mad': mad}
                )
        
        return None
    
    def detect_isolation_forest_anomalies(self, value: float) -> Optional[AnomalyAlert]:
        """Detect anomalies using Isolation Forest"""
        with self.lock:
            if len(self.data_buffer) < 100:  # Need more data for ML
                return None
            
            try:
                # Prepare data
                values = np.array([point['value'] for point in self.data_buffer]).reshape(-1, 1)
                
                # Train model if not already trained
                if not self.model_trained:
                    self.isolation_forest = IsolationForest(
                        contamination=self.sensitivity,
                        random_state=42,
                        n_estimators=100
                    )
                    self.scaler.fit(values)
                    values_scaled = self.scaler.transform(values)
                    self.isolation_forest.fit(values_scaled)
                    self.model_trained = True
                
                # Predict anomaly
                value_scaled = self.scaler.transform([[value]])
                prediction = self.isolation_forest.predict(value_scaled)[0]
                anomaly_score = self.isolation_forest.score_samples(value_scaled)[0]
                
                if prediction == -1:  # Anomaly detected
                    return AnomalyAlert(
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=AlertSeverity.WARNING if anomaly_score > -0.5 else AlertSeverity.ERROR,
                        message=f"Isolation Forest anomaly detected: score {anomaly_score:.3f}",
                        value=value,
                        confidence=min(1.0, abs(anomaly_score)),
                        metadata={'anomaly_score': anomaly_score, 'prediction': prediction}
                    )
            
            except Exception as e:
                logger.error(f"Error in isolation forest detection: {str(e)}")
        
        return None
    
    def detect_all_anomalies(self, value: float) -> List[AnomalyAlert]:
        """Run all anomaly detection methods"""
        anomalies = []
        
        # Statistical methods
        z_score_anomaly = self.detect_z_score_anomalies(value)
        if z_score_anomaly:
            anomalies.append(z_score_anomaly)
        
        iqr_anomaly = self.detect_iqr_anomalies(value)
        if iqr_anomaly:
            anomalies.append(iqr_anomaly)
        
        mad_anomaly = self.detect_mad_anomalies(value)
        if mad_anomaly:
            anomalies.append(mad_anomaly)
        
        # ML-based method
        if_anomaly = self.detect_isolation_forest_anomalies(value)
        if if_anomaly:
            anomalies.append(if_anomaly)
        
        return anomalies
    
    def reset(self):
        """Reset detector state"""
        with self.lock:
            self.data_buffer.clear()
            self.model_trained = False
            self.isolation_forest = None

class DataQualityMonitor:
    """Comprehensive data quality monitoring system"""
    
    def __init__(self, 
                 enable_persistence: bool = True,
                 persistence_path: Optional[str] = None,
                 alert_callbacks: Optional[List[Callable]] = None):
        
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path) if persistence_path else Path("/tmp/quality_monitor")
        self.alert_callbacks = alert_callbacks or []
        
        # Create storage directory
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.db_path = self.persistence_path / "quality_monitor.db"
            self._init_database()
        
        # Anomaly detectors
        self.anomaly_detectors = {
            'price': StatisticalAnomalyDetector(window_size=1000, sensitivity=0.05),
            'volume': StatisticalAnomalyDetector(window_size=1000, sensitivity=0.05),
            'generic': StatisticalAnomalyDetector(window_size=1000, sensitivity=0.05)
        }
        
        # Quality metrics tracking
        self.quality_metrics = deque(maxlen=10000)
        self.quality_lock = threading.RLock()
        
        # Alert history
        self.alert_history = deque(maxlen=10000)
        self.alert_lock = threading.RLock()
        
        # Data statistics
        self.data_stats = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'sum_squared': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'recent_values': deque(maxlen=1000)
        })
        self.stats_lock = threading.RLock()
        
        # Performance metrics
        self.total_records_processed = 0
        self.total_anomalies_detected = 0
        self.processing_times = deque(maxlen=1000)
        
        # Background workers
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.alert_queue = queue.Queue()
        
        # Start alert processing worker
        self._start_alert_worker()
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("DataQualityMonitor cleanup completed")
    
    def _init_database(self):
        """Initialize database for persistence"""
        if not self.enable_persistence:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expected_range TEXT,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
    
    def _start_alert_worker(self):
        """Start background worker for alert processing"""
        def alert_worker():
            while True:
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                    
                    # Process alert
                    self._process_alert(alert)
                    
                    # Save to database
                    if self.enable_persistence:
                        self._save_alert_to_db(alert)
                    
                    # Notify callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {str(e)}")
                    
                    self.alert_queue.task_done()
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in alert worker: {str(e)}")
        
        alert_thread = threading.Thread(target=alert_worker, daemon=True)
        alert_thread.start()
    
    def _process_alert(self, alert: AnomalyAlert):
        """Process alert and update history"""
        with self.alert_lock:
            self.alert_history.append(alert)
            self.total_anomalies_detected += 1
        
        # Log alert
        logger.warning(f"Quality Alert [{alert.severity.value.upper()}]: {alert.message}")
    
    def _save_alert_to_db(self, alert: AnomalyAlert):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT INTO anomaly_alerts (anomaly_type, severity, message, value, expected_range, confidence, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (alert.anomaly_type.value, alert.severity.value, alert.message, 
                     str(alert.value), str(alert.expected_range), alert.confidence, 
                     alert.timestamp, json.dumps(alert.metadata))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving alert to database: {str(e)}")
    
    def monitor_data_point(self, 
                          field_name: str, 
                          value: Any, 
                          data_type: str = 'numeric',
                          timestamp: Optional[float] = None) -> List[AnomalyAlert]:
        """Monitor a single data point for quality issues"""
        start_time = time.time_ns()
        alerts = []
        
        try:
            # Update statistics
            self._update_statistics(field_name, value, data_type)
            
            # Increment processing counter
            self.total_records_processed += 1
            
            # Check for basic data quality issues
            basic_alerts = self._check_basic_quality(field_name, value, data_type)
            alerts.extend(basic_alerts)
            
            # Run anomaly detection for numeric data
            if data_type == 'numeric' and isinstance(value, (int, float)) and not math.isnan(value):
                detector_name = 'price' if 'price' in field_name.lower() else 'volume' if 'volume' in field_name.lower() else 'generic'
                detector = self.anomaly_detectors[detector_name]
                
                # Add to detector buffer
                detector.add_data_point(value, timestamp)
                
                # Run anomaly detection
                anomalies = detector.detect_all_anomalies(value)
                alerts.extend(anomalies)
            
            # Queue alerts for processing
            for alert in alerts:
                try:
                    self.alert_queue.put(alert, timeout=0.1)
                except queue.Full:
                    logger.warning("Alert queue full, dropping alert")
            
            # Record processing time
            end_time = time.time_ns()
            processing_time_us = (end_time - start_time) / 1000
            self.processing_times.append(processing_time_us)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring data point {field_name}: {str(e)}")
            return []
    
    def _check_basic_quality(self, field_name: str, value: Any, data_type: str) -> List[AnomalyAlert]:
        """Check basic data quality issues"""
        alerts = []
        
        # Check for null/None values
        if value is None:
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.MISSING_DATA,
                severity=AlertSeverity.ERROR,
                message=f"Null value detected in field {field_name}",
                value=value
            ))
        
        # Check for NaN values in numeric data
        elif data_type == 'numeric' and isinstance(value, float) and math.isnan(value):
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.INVALID_FORMAT,
                severity=AlertSeverity.ERROR,
                message=f"NaN value detected in numeric field {field_name}",
                value=value
            ))
        
        # Check for infinite values
        elif data_type == 'numeric' and isinstance(value, float) and math.isinf(value):
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.INVALID_FORMAT,
                severity=AlertSeverity.ERROR,
                message=f"Infinite value detected in numeric field {field_name}",
                value=value
            ))
        
        # Check for negative values in fields that shouldn't be negative
        elif (data_type == 'numeric' and isinstance(value, (int, float)) and 
              value < 0 and ('volume' in field_name.lower() or 'count' in field_name.lower())):
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.INVALID_FORMAT,
                severity=AlertSeverity.WARNING,
                message=f"Negative value detected in field {field_name}: {value}",
                value=value
            ))
        
        return alerts
    
    def _update_statistics(self, field_name: str, value: Any, data_type: str):
        """Update field statistics"""
        with self.stats_lock:
            stats = self.data_stats[field_name]
            stats['count'] += 1
            stats['recent_values'].append(value)
            
            if data_type == 'numeric' and isinstance(value, (int, float)) and not math.isnan(value):
                stats['sum'] += value
                stats['sum_squared'] += value * value
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
    
    def monitor_dataframe(self, df: pd.DataFrame, timestamp_col: Optional[str] = None) -> DataQualityReport:
        """Monitor entire DataFrame for quality issues"""
        start_time = time.time()
        all_alerts = []
        all_metrics = []
        
        # Basic DataFrame checks
        total_records = len(df)
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            all_alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.DUPLICATE_DATA,
                severity=AlertSeverity.WARNING,
                message=f"Found {duplicate_count} duplicate rows",
                value=duplicate_count,
                metadata={'duplicate_percentage': (duplicate_count / total_records) * 100}
            ))
        
        # Check each column
        for col in df.columns:
            if col == timestamp_col:
                continue
            
            # Determine data type
            if df[col].dtype in ['int64', 'float64']:
                data_type = 'numeric'
            else:
                data_type = 'categorical'
            
            # Calculate column metrics
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / total_records) * 100
            
            # Add null percentage metric
            all_metrics.append(QualityMetric(
                name=f"{col}_null_percentage",
                value=null_percentage,
                threshold=5.0,  # 5% threshold
                status='good' if null_percentage < 5 else 'warning' if null_percentage < 20 else 'error'
            ))
            
            # Check for high null percentage
            if null_percentage > 20:
                all_alerts.append(AnomalyAlert(
                    anomaly_type=AnomalyType.MISSING_DATA,
                    severity=AlertSeverity.ERROR,
                    message=f"High null percentage in column {col}: {null_percentage:.1f}%",
                    value=null_percentage,
                    metadata={'null_count': null_count, 'total_records': total_records}
                ))
            
            # Sample data points for anomaly detection
            sample_size = min(1000, total_records)
            sample_data = df[col].dropna().sample(n=min(sample_size, len(df[col].dropna())))
            
            for value in sample_data:
                alerts = self.monitor_data_point(col, value, data_type)
                all_alerts.extend(alerts)
        
        # Calculate overall quality score
        valid_records = total_records - sum(1 for alert in all_alerts if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL])
        overall_score = (valid_records / total_records) * 100 if total_records > 0 else 0
        
        # Generate statistics
        statistics_dict = {
            'processing_time_seconds': time.time() - start_time,
            'columns_analyzed': len(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': duplicate_count
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_alerts, all_metrics, df)
        
        # Create report
        report = DataQualityReport(
            timestamp=time.time(),
            overall_score=overall_score,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            metrics=all_metrics,
            anomalies=all_alerts,
            statistics=statistics_dict,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, alerts: List[AnomalyAlert], 
                                metrics: List[QualityMetric], 
                                df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        # Check for missing data issues
        missing_data_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.MISSING_DATA]
        if missing_data_alerts:
            recommendations.append("Consider implementing data imputation strategies for missing values")
        
        # Check for duplicate data
        duplicate_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.DUPLICATE_DATA]
        if duplicate_alerts:
            recommendations.append("Implement deduplication process in data ingestion pipeline")
        
        # Check for statistical outliers
        outlier_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.STATISTICAL_OUTLIER]
        if len(outlier_alerts) > len(df) * 0.1:  # More than 10% outliers
            recommendations.append("High number of statistical outliers detected. Review data source quality")
        
        # Check for data format issues
        format_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.INVALID_FORMAT]
        if format_alerts:
            recommendations.append("Implement stronger data validation at ingestion point")
        
        # Performance recommendations
        if len(df) > 100000:
            recommendations.append("Consider implementing data sampling for large datasets")
        
        return recommendations
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality monitoring summary"""
        with self.alert_lock:
            recent_alerts = list(self.alert_history)[-100:]  # Last 100 alerts
        
        with self.stats_lock:
            field_stats = dict(self.data_stats)
        
        # Calculate alert statistics
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity.value] += 1
        
        # Performance statistics
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        throughput = 1000000 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            'total_records_processed': self.total_records_processed,
            'total_anomalies_detected': self.total_anomalies_detected,
            'anomaly_rate': (self.total_anomalies_detected / self.total_records_processed) * 100 if self.total_records_processed > 0 else 0,
            'alert_counts': dict(alert_counts),
            'avg_processing_time_us': avg_processing_time,
            'throughput_records_per_sec': throughput,
            'monitored_fields': list(field_stats.keys()),
            'field_statistics': field_stats
        }
    
    def get_recent_alerts(self, limit: int = 100) -> List[AnomalyAlert]:
        """Get recent alerts"""
        with self.alert_lock:
            return list(self.alert_history)[-limit:]
    
    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def reset_statistics(self):
        """Reset all statistics"""
        with self.stats_lock:
            self.data_stats.clear()
        
        with self.alert_lock:
            self.alert_history.clear()
        
        # Reset anomaly detectors
        for detector in self.anomaly_detectors.values():
            detector.reset()
        
        self.total_records_processed = 0
        self.total_anomalies_detected = 0
        self.processing_times.clear()

# Utility functions
def create_quality_monitor(enable_persistence: bool = True) -> DataQualityMonitor:
    """Create data quality monitor with default settings"""
    return DataQualityMonitor(enable_persistence=enable_persistence)

def create_simple_quality_monitor() -> DataQualityMonitor:
    """Create simple quality monitor without persistence"""
    return DataQualityMonitor(enable_persistence=False)

def demo_quality_monitoring():
    """Demonstration of quality monitoring capabilities"""
    # Create monitor
    monitor = create_quality_monitor()
    
    # Generate sample data with anomalies
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 1000)
    
    # Add some anomalies
    anomalous_data = np.concatenate([
        normal_data,
        [200, 300, -50, np.nan, np.inf]  # Outliers and invalid values
    ])
    
    # Monitor data points
    all_alerts = []
    for i, value in enumerate(anomalous_data):
        alerts = monitor.monitor_data_point('test_field', value, 'numeric')
        all_alerts.extend(alerts)
    
    # Get summary
    summary = monitor.get_quality_summary()
    
    print(f"Quality Monitoring Demo Results:")
    print(f"Total records processed: {summary['total_records_processed']}")
    print(f"Total anomalies detected: {summary['total_anomalies_detected']}")
    print(f"Anomaly rate: {summary['anomaly_rate']:.2f}%")
    print(f"Alert counts: {summary['alert_counts']}")
    print(f"Average processing time: {summary['avg_processing_time_us']:.2f}us")
    
    return monitor, all_alerts

if __name__ == "__main__":
    demo_quality_monitoring()
