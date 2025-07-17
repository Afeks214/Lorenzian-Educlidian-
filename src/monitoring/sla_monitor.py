#!/usr/bin/env python3
"""
AGENT 13: SLA Monitoring and Performance Regression Detection
Comprehensive SLA monitoring with automated performance regression detection
"""

import time
import asyncio
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import numpy as np
from scipy import stats
import redis
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SLA Metrics
SLA_VIOLATIONS = Counter(
    'sla_violations_total',
    'Total number of SLA violations',
    ['service', 'sla_type', 'severity']
)

SLA_COMPLIANCE_RATE = Gauge(
    'sla_compliance_rate_percent',
    'SLA compliance rate percentage',
    ['service', 'sla_type']
)

PERFORMANCE_REGRESSION_DETECTED = Counter(
    'performance_regression_detected_total',
    'Total number of performance regressions detected',
    ['service', 'metric_type', 'severity']
)

SLA_RESPONSE_TIME_PERCENTILES = Gauge(
    'sla_response_time_percentiles_ms',
    'SLA response time percentiles in milliseconds',
    ['service', 'percentile']
)

SLA_AVAILABILITY_SCORE = Gauge(
    'sla_availability_score',
    'SLA availability score (0-1)',
    ['service']
)

class SLAType(Enum):
    """SLA types for monitoring."""
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    ACCURACY = "accuracy"

class SeverityLevel(Enum):
    """Severity levels for SLA violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SLATarget:
    """SLA target definition."""
    service: str
    sla_type: SLAType
    target_value: float
    threshold_warning: float
    threshold_critical: float
    measurement_window: int = 300  # 5 minutes
    evaluation_interval: int = 60  # 1 minute
    unit: str = "ms"
    description: str = ""

@dataclass
class SLAViolation:
    """SLA violation event."""
    service: str
    sla_type: SLAType
    severity: SeverityLevel
    timestamp: datetime
    actual_value: float
    target_value: float
    duration_seconds: float
    description: str
    remediation_actions: List[str] = field(default_factory=list)

@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    service: str
    metric_name: str
    baseline_value: float
    baseline_stddev: float
    measurement_count: int
    last_updated: datetime
    confidence_interval: Tuple[float, float]

class PerformanceRegressionDetector:
    """Statistical performance regression detector."""
    
    def __init__(self, sensitivity: float = 2.0, min_samples: int = 30):
        self.sensitivity = sensitivity  # Standard deviations for detection
        self.min_samples = min_samples
        self.baselines = {}
        self.recent_measurements = defaultdict(lambda: deque(maxlen=100))
        
    def update_baseline(self, service: str, metric_name: str, values: List[float]):
        """Update performance baseline with new measurements."""
        if len(values) < self.min_samples:
            logger.warning(f"Insufficient samples for baseline: {len(values)} < {self.min_samples}")
            return
            
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values)
        
        # Calculate confidence interval (95%)
        confidence_interval = stats.t.interval(
            0.95, len(values) - 1,
            loc=mean_value,
            scale=std_value / np.sqrt(len(values))
        )
        
        baseline = PerformanceBaseline(
            service=service,
            metric_name=metric_name,
            baseline_value=mean_value,
            baseline_stddev=std_value,
            measurement_count=len(values),
            last_updated=datetime.utcnow(),
            confidence_interval=confidence_interval
        )
        
        self.baselines[f"{service}:{metric_name}"] = baseline
        logger.info(f"Updated baseline for {service}:{metric_name} - {mean_value:.2f} ± {std_value:.2f}")
        
    def detect_regression(self, service: str, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Detect if current value represents a performance regression."""
        baseline_key = f"{service}:{metric_name}"
        
        if baseline_key not in self.baselines:
            # Store measurement for future baseline
            self.recent_measurements[baseline_key].append(current_value)
            
            # Try to create baseline if we have enough samples
            if len(self.recent_measurements[baseline_key]) >= self.min_samples:
                self.update_baseline(service, metric_name, list(self.recent_measurements[baseline_key]))
            return None
            
        baseline = self.baselines[baseline_key]
        
        # Calculate z-score
        z_score = (current_value - baseline.baseline_value) / baseline.baseline_stddev
        
        # Store current measurement
        self.recent_measurements[baseline_key].append(current_value)
        
        # Check for regression (performance degradation)
        if abs(z_score) > self.sensitivity:
            severity = SeverityLevel.CRITICAL if abs(z_score) > 3.0 else SeverityLevel.HIGH
            
            # Determine if it's a regression or improvement
            is_regression = (
                (metric_name.endswith('_time') or metric_name.endswith('_latency')) and z_score > 0
            ) or (
                (metric_name.endswith('_rate') or metric_name.endswith('_throughput')) and z_score < 0
            )
            
            if is_regression:
                PERFORMANCE_REGRESSION_DETECTED.labels(
                    service=service,
                    metric_type=metric_name,
                    severity=severity.value
                ).inc()
                
                return {
                    'service': service,
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'baseline_value': baseline.baseline_value,
                    'z_score': z_score,
                    'severity': severity,
                    'confidence_interval': baseline.confidence_interval,
                    'regression_detected': True,
                    'timestamp': datetime.utcnow(),
                    'description': f"Performance regression detected: {metric_name} deviated {z_score:.2f}σ from baseline"
                }
                
        return None
        
    def get_baseline_info(self, service: str, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline information for a specific metric."""
        baseline_key = f"{service}:{metric_name}"
        return self.baselines.get(baseline_key)

class SLAMonitor:
    """Comprehensive SLA monitoring system."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.sla_targets = {}
        self.violations = []
        self.measurements = defaultdict(lambda: deque(maxlen=1000))
        self.regression_detector = PerformanceRegressionDetector()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def add_sla_target(self, target: SLATarget):
        """Add SLA target for monitoring."""
        key = f"{target.service}:{target.sla_type.value}"
        self.sla_targets[key] = target
        logger.info(f"Added SLA target: {target.service} {target.sla_type.value} = {target.target_value}{target.unit}")
        
    def record_measurement(self, service: str, sla_type: SLAType, value: float, timestamp: Optional[datetime] = None):
        """Record a measurement for SLA monitoring."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        key = f"{service}:{sla_type.value}"
        
        # Store measurement
        measurement = {
            'value': value,
            'timestamp': timestamp,
            'service': service,
            'sla_type': sla_type.value
        }
        
        self.measurements[key].append(measurement)
        
        # Check for SLA violation
        if key in self.sla_targets:
            self._check_sla_violation(service, sla_type, value, timestamp)
            
        # Check for performance regression
        regression = self.regression_detector.detect_regression(service, sla_type.value, value)
        if regression:
            self._handle_performance_regression(regression)
            
    def _check_sla_violation(self, service: str, sla_type: SLAType, value: float, timestamp: datetime):
        """Check if measurement violates SLA."""
        key = f"{service}:{sla_type.value}"
        target = self.sla_targets[key]
        
        violation = None
        
        # Check violation based on SLA type
        if sla_type in [SLAType.RESPONSE_TIME]:
            # Lower values are better
            if value > target.threshold_critical:
                violation = SeverityLevel.CRITICAL
            elif value > target.threshold_warning:
                violation = SeverityLevel.HIGH
            elif value > target.target_value:
                violation = SeverityLevel.MEDIUM
                
        elif sla_type in [SLAType.AVAILABILITY, SLAType.THROUGHPUT, SLAType.ACCURACY]:
            # Higher values are better
            if value < target.threshold_critical:
                violation = SeverityLevel.CRITICAL
            elif value < target.threshold_warning:
                violation = SeverityLevel.HIGH
            elif value < target.target_value:
                violation = SeverityLevel.MEDIUM
                
        elif sla_type == SLAType.ERROR_RATE:
            # Lower values are better
            if value > target.threshold_critical:
                violation = SeverityLevel.CRITICAL
            elif value > target.threshold_warning:
                violation = SeverityLevel.HIGH
            elif value > target.target_value:
                violation = SeverityLevel.MEDIUM
                
        if violation:
            self._record_sla_violation(service, sla_type, violation, value, target.target_value, timestamp)
            
    def _record_sla_violation(self, service: str, sla_type: SLAType, severity: SeverityLevel, 
                            actual_value: float, target_value: float, timestamp: datetime):
        """Record SLA violation."""
        violation = SLAViolation(
            service=service,
            sla_type=sla_type,
            severity=severity,
            timestamp=timestamp,
            actual_value=actual_value,
            target_value=target_value,
            duration_seconds=0,  # Will be calculated later
            description=f"SLA violation: {sla_type.value} {actual_value} exceeds target {target_value}",
            remediation_actions=self._get_remediation_actions(service, sla_type, severity)
        )
        
        self.violations.append(violation)
        
        # Record metrics
        SLA_VIOLATIONS.labels(
            service=service,
            sla_type=sla_type.value,
            severity=severity.value
        ).inc()
        
        # Store in Redis for alerting
        violation_data = {
            'service': service,
            'sla_type': sla_type.value,
            'severity': severity.value,
            'actual_value': actual_value,
            'target_value': target_value,
            'timestamp': timestamp.isoformat(),
            'description': violation.description,
            'remediation_actions': violation.remediation_actions
        }
        
        self.redis_client.lpush(
            f"sla_violations:{service}",
            json.dumps(violation_data)
        )
        
        logger.warning(f"SLA violation detected: {violation.description}")
        
    def _get_remediation_actions(self, service: str, sla_type: SLAType, severity: SeverityLevel) -> List[str]:
        """Get remediation actions for SLA violation."""
        actions = []
        
        if sla_type == SLAType.RESPONSE_TIME:
            actions.extend([
                "Check system resource utilization",
                "Review database query performance",
                "Analyze network latency",
                "Consider scaling up resources"
            ])
            
        elif sla_type == SLAType.AVAILABILITY:
            actions.extend([
                "Check service health endpoints",
                "Review error logs",
                "Verify network connectivity",
                "Consider failover to backup systems"
            ])
            
        elif sla_type == SLAType.THROUGHPUT:
            actions.extend([
                "Check for bottlenecks in pipeline",
                "Review resource allocation",
                "Analyze queue depths",
                "Consider horizontal scaling"
            ])
            
        elif sla_type == SLAType.ERROR_RATE:
            actions.extend([
                "Investigate error patterns",
                "Review recent code deployments",
                "Check input data quality",
                "Analyze error logs for root cause"
            ])
            
        if severity == SeverityLevel.CRITICAL:
            actions.insert(0, "URGENT: Escalate to on-call engineer")
            actions.insert(1, "Consider emergency rollback procedures")
            
        return actions
        
    def _handle_performance_regression(self, regression: Dict[str, Any]):
        """Handle detected performance regression."""
        logger.warning(f"Performance regression detected: {regression['description']}")
        
        # Store regression info in Redis
        regression_data = {
            'service': regression['service'],
            'metric_name': regression['metric_name'],
            'current_value': regression['current_value'],
            'baseline_value': regression['baseline_value'],
            'z_score': regression['z_score'],
            'severity': regression['severity'].value,
            'timestamp': regression['timestamp'].isoformat(),
            'description': regression['description']
        }
        
        self.redis_client.lpush(
            f"performance_regressions:{regression['service']}",
            json.dumps(regression_data)
        )
        
    def calculate_sla_compliance(self, service: str, sla_type: SLAType, window_minutes: int = 60) -> float:
        """Calculate SLA compliance rate for a service."""
        key = f"{service}:{sla_type.value}"
        
        if key not in self.sla_targets:
            return 0.0
            
        target = self.sla_targets[key]
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        # Get measurements in window
        measurements = [
            m for m in self.measurements[key]
            if m['timestamp'] > cutoff_time
        ]
        
        if not measurements:
            return 0.0
            
        # Calculate compliance
        compliant_count = 0
        for measurement in measurements:
            value = measurement['value']
            
            if sla_type in [SLAType.RESPONSE_TIME, SLAType.ERROR_RATE]:
                # Lower is better
                if value <= target.target_value:
                    compliant_count += 1
            else:
                # Higher is better
                if value >= target.target_value:
                    compliant_count += 1
                    
        compliance_rate = (compliant_count / len(measurements)) * 100
        
        # Update metrics
        SLA_COMPLIANCE_RATE.labels(
            service=service,
            sla_type=sla_type.value
        ).set(compliance_rate)
        
        return compliance_rate
        
    def calculate_percentiles(self, service: str, sla_type: SLAType, window_minutes: int = 60) -> Dict[str, float]:
        """Calculate percentiles for SLA metrics."""
        key = f"{service}:{sla_type.value}"
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        # Get measurements in window
        measurements = [
            m['value'] for m in self.measurements[key]
            if m['timestamp'] > cutoff_time
        ]
        
        if not measurements:
            return {}
            
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            percentile_value = np.percentile(measurements, p)
            percentiles[f'p{p}'] = percentile_value
            
            # Update metrics
            SLA_RESPONSE_TIME_PERCENTILES.labels(
                service=service,
                percentile=f'p{p}'
            ).set(percentile_value)
            
        return percentiles
        
    def get_sla_status(self, service: str) -> Dict[str, Any]:
        """Get current SLA status for a service."""
        status = {
            'service': service,
            'timestamp': datetime.utcnow().isoformat(),
            'sla_targets': {},
            'compliance_rates': {},
            'percentiles': {},
            'recent_violations': [],
            'performance_baselines': {}
        }
        
        # Get SLA targets and compliance
        for key, target in self.sla_targets.items():
            if target.service == service:
                sla_type = target.sla_type
                
                status['sla_targets'][sla_type.value] = {
                    'target_value': target.target_value,
                    'threshold_warning': target.threshold_warning,
                    'threshold_critical': target.threshold_critical,
                    'unit': target.unit
                }
                
                # Calculate compliance
                compliance_rate = self.calculate_sla_compliance(service, sla_type)
                status['compliance_rates'][sla_type.value] = compliance_rate
                
                # Calculate percentiles
                percentiles = self.calculate_percentiles(service, sla_type)
                status['percentiles'][sla_type.value] = percentiles
                
        # Get recent violations
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_violations = [
            {
                'sla_type': v.sla_type.value,
                'severity': v.severity.value,
                'timestamp': v.timestamp.isoformat(),
                'actual_value': v.actual_value,
                'target_value': v.target_value,
                'description': v.description
            }
            for v in self.violations
            if v.service == service and v.timestamp > cutoff_time
        ]
        status['recent_violations'] = recent_violations
        
        # Get performance baselines
        for key, baseline in self.regression_detector.baselines.items():
            baseline_service, metric_name = key.split(':', 1)
            if baseline_service == service:
                status['performance_baselines'][metric_name] = {
                    'baseline_value': baseline.baseline_value,
                    'baseline_stddev': baseline.baseline_stddev,
                    'measurement_count': baseline.measurement_count,
                    'last_updated': baseline.last_updated.isoformat(),
                    'confidence_interval': baseline.confidence_interval
                }
                
        return status
        
    def start_monitoring(self):
        """Start SLA monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("SLA monitoring started")
            
    def stop_monitoring(self):
        """Stop SLA monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            logger.info("SLA monitoring stopped")
            
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Update compliance rates and percentiles
                for key, target in self.sla_targets.items():
                    self.calculate_sla_compliance(target.service, target.sla_type)
                    self.calculate_percentiles(target.service, target.sla_type)
                    
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                time.sleep(5)

# Predefined SLA targets for trading system
DEFAULT_TRADING_SLA_TARGETS = [
    SLATarget(
        service="trading_engine",
        sla_type=SLAType.RESPONSE_TIME,
        target_value=10.0,  # 10ms
        threshold_warning=15.0,
        threshold_critical=25.0,
        unit="ms",
        description="Trading engine response time"
    ),
    SLATarget(
        service="trading_engine",
        sla_type=SLAType.AVAILABILITY,
        target_value=99.9,  # 99.9%
        threshold_warning=99.5,
        threshold_critical=99.0,
        unit="%",
        description="Trading engine availability"
    ),
    SLATarget(
        service="marl_agents",
        sla_type=SLAType.RESPONSE_TIME,
        target_value=8.0,  # 8ms
        threshold_warning=12.0,
        threshold_critical=20.0,
        unit="ms",
        description="MARL agent inference time"
    ),
    SLATarget(
        service="risk_management",
        sla_type=SLAType.RESPONSE_TIME,
        target_value=5.0,  # 5ms
        threshold_warning=8.0,
        threshold_critical=15.0,
        unit="ms",
        description="Risk management response time"
    ),
    SLATarget(
        service="data_pipeline",
        sla_type=SLAType.THROUGHPUT,
        target_value=1000.0,  # 1000 msgs/sec
        threshold_warning=800.0,
        threshold_critical=500.0,
        unit="msgs/sec",
        description="Data pipeline throughput"
    ),
    SLATarget(
        service="order_execution",
        sla_type=SLAType.ERROR_RATE,
        target_value=0.01,  # 1%
        threshold_warning=0.05,
        threshold_critical=0.1,
        unit="%",
        description="Order execution error rate"
    )
]

# Factory function
def create_sla_monitor(redis_client: redis.Redis, include_default_targets: bool = True) -> SLAMonitor:
    """Create SLA monitor with optional default targets."""
    monitor = SLAMonitor(redis_client)
    
    if include_default_targets:
        for target in DEFAULT_TRADING_SLA_TARGETS:
            monitor.add_sla_target(target)
            
    return monitor

# Example usage
if __name__ == "__main__":
    # Example of how to use the SLA monitor
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    monitor = create_sla_monitor(redis_client)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Example of recording measurements
    monitor.record_measurement("trading_engine", SLAType.RESPONSE_TIME, 12.5)
    monitor.record_measurement("trading_engine", SLAType.AVAILABILITY, 99.8)
    monitor.record_measurement("marl_agents", SLAType.RESPONSE_TIME, 6.2)
    monitor.record_measurement("risk_management", SLAType.RESPONSE_TIME, 4.1)
    
    # Get SLA status
    status = monitor.get_sla_status("trading_engine")
    print(json.dumps(status, indent=2))
    
    logger.info("SLA monitoring example running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logger.info("SLA monitoring stopped")
