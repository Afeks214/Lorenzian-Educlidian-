"""
Temporal Bias Detection and Prevention System
Agent 5: Data Quality & Bias Elimination

This module implements comprehensive temporal bias detection and prevention
mechanisms to ensure backtesting accuracy and eliminate look-ahead bias.

Key Features:
- Strict temporal boundary enforcement
- Look-ahead bias detection and prevention
- Data availability time-stamping
- Multi-timeframe temporal consistency validation
- Causal relationship enforcement
- Future data leak detection
"""

import asyncio
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class BiasType(str, Enum):
    """Types of temporal bias"""
    LOOK_AHEAD_BIAS = "look_ahead_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    SELECTION_BIAS = "selection_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    TEMPORAL_LEAK = "temporal_leak"
    FUTURE_INFORMATION = "future_information"

class SeverityLevel(str, Enum):
    """Bias severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataAvailabilityStatus(str, Enum):
    """Data availability status"""
    AVAILABLE = "available"
    DELAYED = "delayed"
    UNAVAILABLE = "unavailable"
    SYNTHETIC = "synthetic"

class TemporalBoundaryType(str, Enum):
    """Types of temporal boundaries"""
    HARD_BOUNDARY = "hard_boundary"  # Strict enforcement
    SOFT_BOUNDARY = "soft_boundary"  # Warning only
    FLEXIBLE_BOUNDARY = "flexible_boundary"  # Context-dependent

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TemporalBoundary:
    """Temporal boundary definition"""
    boundary_id: str
    timestamp: datetime
    boundary_type: TemporalBoundaryType
    enforcement_level: SeverityLevel
    
    # Boundary constraints
    max_lookback_hours: int = 24
    max_latency_seconds: int = 30
    allow_synthetic_data: bool = True
    
    # Context
    component: str = "unknown"
    data_source: str = "unknown"
    
    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DataAvailabilityRecord:
    """Data availability record with temporal constraints"""
    record_id: str
    data_identifier: str
    timestamp: datetime
    availability_status: DataAvailabilityStatus
    
    # Temporal constraints
    earliest_available_time: datetime
    latest_available_time: datetime
    data_source_latency: timedelta
    
    # Data lineage
    source_system: str
    transformation_chain: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Quality metrics
    quality_score: float = 1.0
    confidence_level: float = 1.0
    
    # Validation metadata
    validated_at: datetime = field(default_factory=datetime.utcnow)
    validator_id: str = "system"
    
    # Tags and attributes
    tags: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""
    detection_id: str
    bias_type: BiasType
    severity_level: SeverityLevel
    
    # Detection details
    detected_at: datetime
    affected_timerange: Tuple[datetime, datetime]
    affected_components: List[str]
    
    # Bias description
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    
    # Impact assessment
    impact_assessment: str = ""
    potential_consequences: List[str] = field(default_factory=list)
    
    # Remediation
    recommended_actions: List[str] = field(default_factory=list)
    auto_remediation_possible: bool = False
    
    # Metadata
    detector_id: str = "system"
    detection_method: str = "unknown"
    
    # Status tracking
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class TemporalConsistencyCheck:
    """Temporal consistency check configuration"""
    check_id: str
    check_name: str
    check_type: str
    
    # Check parameters
    timeframe_minutes: int
    max_time_drift_seconds: int = 5
    enforce_sequential_order: bool = True
    allow_duplicate_timestamps: bool = False
    
    # Data requirements
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    # Validation thresholds
    min_data_points: int = 10
    max_gap_minutes: int = 30
    
    # Consistency rules
    consistency_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# TEMPORAL BIAS DETECTION ENGINE
# =============================================================================

class TemporalBiasDetector:
    """Comprehensive temporal bias detection engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Detection engines
        self.detection_engines = {
            BiasType.LOOK_AHEAD_BIAS: self._detect_look_ahead_bias,
            BiasType.TEMPORAL_LEAK: self._detect_temporal_leak,
            BiasType.FUTURE_INFORMATION: self._detect_future_information,
            BiasType.SURVIVORSHIP_BIAS: self._detect_survivorship_bias,
            BiasType.SELECTION_BIAS: self._detect_selection_bias,
            BiasType.CONFIRMATION_BIAS: self._detect_confirmation_bias,
        }
        
        # Data tracking
        self.data_availability_records: Dict[str, DataAvailabilityRecord] = {}
        self.temporal_boundaries: Dict[str, TemporalBoundary] = {}
        self.bias_detection_history: deque = deque(maxlen=1000)
        
        # Active monitoring
        self.active_checks: Dict[str, TemporalConsistencyCheck] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'biases_detected': 0,
            'critical_biases': 0,
            'auto_remediated': 0,
            'data_points_validated': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Temporal bias detection engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for bias detection"""
        return {
            'enable_continuous_monitoring': True,
            'detection_interval_seconds': 60,
            'max_lookback_hours': 24,
            'strict_temporal_enforcement': True,
            'auto_remediation_enabled': True,
            'bias_severity_threshold': SeverityLevel.MEDIUM,
            'data_availability_timeout_seconds': 300,
            'temporal_consistency_checks': {
                'timestamp_ordering': True,
                'data_availability_validation': True,
                'cross_timeframe_consistency': True,
                'causal_relationship_enforcement': True
            }
        }
    
    def start_monitoring(self):
        """Start continuous bias monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="temporal_bias_monitor"
            )
            self.monitor_thread.start()
            logger.info("Temporal bias monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous bias monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Temporal bias monitoring stopped")
    
    def register_temporal_boundary(self, boundary: TemporalBoundary):
        """Register a temporal boundary for enforcement"""
        with self.lock:
            self.temporal_boundaries[boundary.boundary_id] = boundary
            logger.debug(f"Registered temporal boundary: {boundary.boundary_id}")
    
    def register_data_availability(self, record: DataAvailabilityRecord):
        """Register data availability record"""
        with self.lock:
            self.data_availability_records[record.record_id] = record
            logger.debug(f"Registered data availability: {record.record_id}")
    
    def validate_temporal_access(self, 
                                data_identifier: str,
                                access_time: datetime,
                                requested_data_time: datetime) -> Tuple[bool, Optional[str]]:
        """Validate temporal access to data"""
        
        # Check if data is available at access time
        if data_identifier in self.data_availability_records:
            availability_record = self.data_availability_records[data_identifier]
            
            # Check if data was available at access time
            if access_time < availability_record.earliest_available_time:
                return False, f"Data {data_identifier} not available at {access_time}"
            
            # Check for look-ahead bias
            if requested_data_time > access_time:
                return False, f"Look-ahead bias detected: accessing future data {requested_data_time} at {access_time}"
            
            # Check data source latency
            expected_availability = requested_data_time + availability_record.data_source_latency
            if access_time < expected_availability:
                return False, f"Data not yet available due to source latency"
        
        return True, None
    
    def detect_bias(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect temporal bias in data"""
        
        with self.lock:
            self.stats['total_checks'] += 1
            
            detected_biases = []
            
            # Run all detection engines
            for bias_type, detection_engine in self.detection_engines.items():
                try:
                    bias_results = detection_engine(data, context)
                    if bias_results:
                        detected_biases.extend(bias_results)
                        
                except Exception as e:
                    logger.error(f"Error in {bias_type.value} detection: {e}")
            
            # Update statistics
            self.stats['biases_detected'] += len(detected_biases)
            self.stats['critical_biases'] += sum(
                1 for bias in detected_biases 
                if bias.severity_level == SeverityLevel.CRITICAL
            )
            
            # Store detection history
            for bias in detected_biases:
                self.bias_detection_history.append(bias)
            
            # Auto-remediation if enabled
            if self.config.get('auto_remediation_enabled', True):
                self._attempt_auto_remediation(detected_biases)
            
            logger.debug(f"Bias detection completed: {len(detected_biases)} biases found")
            
            return detected_biases
    
    def _detect_look_ahead_bias(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect look-ahead bias in data usage"""
        
        biases = []
        current_time = context.get('current_time', datetime.utcnow())
        
        # Check for future data usage
        if 'timestamps' in data and 'values' in data:
            timestamps = data['timestamps']
            
            # Find future timestamps
            future_timestamps = [ts for ts in timestamps if ts > current_time]
            
            if future_timestamps:
                bias = BiasDetectionResult(
                    detection_id=str(uuid.uuid4()),
                    bias_type=BiasType.LOOK_AHEAD_BIAS,
                    severity_level=SeverityLevel.CRITICAL,
                    detected_at=datetime.utcnow(),
                    affected_timerange=(min(future_timestamps), max(future_timestamps)),
                    affected_components=[context.get('component', 'unknown')],
                    description=f"Look-ahead bias detected: {len(future_timestamps)} future data points accessed",
                    evidence={
                        'future_timestamps': [ts.isoformat() for ts in future_timestamps],
                        'current_time': current_time.isoformat(),
                        'time_violations': len(future_timestamps)
                    },
                    confidence_score=0.95,
                    recommended_actions=[
                        "Remove future data points from analysis",
                        "Implement strict temporal boundary enforcement",
                        "Review data access patterns"
                    ],
                    auto_remediation_possible=True
                )
                
                biases.append(bias)
        
        # Check data availability constraints
        if 'data_requests' in data:
            for request in data['data_requests']:
                data_id = request.get('data_identifier')
                access_time = request.get('access_time')
                
                if data_id in self.data_availability_records:
                    record = self.data_availability_records[data_id]
                    
                    if access_time < record.earliest_available_time:
                        bias = BiasDetectionResult(
                            detection_id=str(uuid.uuid4()),
                            bias_type=BiasType.LOOK_AHEAD_BIAS,
                            severity_level=SeverityLevel.HIGH,
                            detected_at=datetime.utcnow(),
                            affected_timerange=(access_time, record.earliest_available_time),
                            affected_components=[context.get('component', 'unknown')],
                            description=f"Data accessed before availability: {data_id}",
                            evidence={
                                'data_identifier': data_id,
                                'access_time': access_time.isoformat(),
                                'earliest_available': record.earliest_available_time.isoformat()
                            },
                            confidence_score=0.90,
                            recommended_actions=[
                                "Adjust data access timing",
                                "Implement data availability checks",
                                "Use alternative data sources"
                            ]
                        )
                        
                        biases.append(bias)
        
        return biases
    
    def _detect_temporal_leak(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect temporal information leakage"""
        
        biases = []
        
        # Check for temporal ordering violations
        if 'events' in data:
            events = data['events']
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.get('timestamp', datetime.min))
            
            # Check for dependency violations
            for i, event in enumerate(sorted_events):
                dependencies = event.get('dependencies', [])
                
                for dep_id in dependencies:
                    # Find dependency event
                    dep_event = next((e for e in sorted_events if e.get('id') == dep_id), None)
                    
                    if dep_event and dep_event.get('timestamp') > event.get('timestamp'):
                        bias = BiasDetectionResult(
                            detection_id=str(uuid.uuid4()),
                            bias_type=BiasType.TEMPORAL_LEAK,
                            severity_level=SeverityLevel.HIGH,
                            detected_at=datetime.utcnow(),
                            affected_timerange=(event.get('timestamp'), dep_event.get('timestamp')),
                            affected_components=[context.get('component', 'unknown')],
                            description=f"Temporal leak: dependency {dep_id} occurs after dependent event",
                            evidence={
                                'event_id': event.get('id'),
                                'dependency_id': dep_id,
                                'event_time': event.get('timestamp').isoformat(),
                                'dependency_time': dep_event.get('timestamp').isoformat()
                            },
                            confidence_score=0.85,
                            recommended_actions=[
                                "Reorder event processing",
                                "Implement causal consistency checks",
                                "Review event dependency graph"
                            ]
                        )
                        
                        biases.append(bias)
        
        return biases
    
    def _detect_future_information(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect use of future information"""
        
        biases = []
        analysis_time = context.get('analysis_time', datetime.utcnow())
        
        # Check for future-dated features
        if 'features' in data:
            features = data['features']
            
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, dict) and 'timestamp' in feature_data:
                    feature_time = feature_data['timestamp']
                    
                    if feature_time > analysis_time:
                        bias = BiasDetectionResult(
                            detection_id=str(uuid.uuid4()),
                            bias_type=BiasType.FUTURE_INFORMATION,
                            severity_level=SeverityLevel.CRITICAL,
                            detected_at=datetime.utcnow(),
                            affected_timerange=(analysis_time, feature_time),
                            affected_components=[context.get('component', 'unknown')],
                            description=f"Future information used in feature: {feature_name}",
                            evidence={
                                'feature_name': feature_name,
                                'feature_time': feature_time.isoformat(),
                                'analysis_time': analysis_time.isoformat(),
                                'time_violation': (feature_time - analysis_time).total_seconds()
                            },
                            confidence_score=0.95,
                            recommended_actions=[
                                "Remove future-dated features",
                                "Implement feature timestamp validation",
                                "Review feature engineering pipeline"
                            ],
                            auto_remediation_possible=True
                        )
                        
                        biases.append(bias)
        
        return biases
    
    def _detect_survivorship_bias(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect survivorship bias in data selection"""
        
        biases = []
        
        # Check for missing delisted/failed entities
        if 'entities' in data and 'historical_entities' in context:
            current_entities = set(data['entities'])
            historical_entities = set(context['historical_entities'])
            
            missing_entities = historical_entities - current_entities
            
            if missing_entities and len(missing_entities) > 0.1 * len(historical_entities):
                bias = BiasDetectionResult(
                    detection_id=str(uuid.uuid4()),
                    bias_type=BiasType.SURVIVORSHIP_BIAS,
                    severity_level=SeverityLevel.MEDIUM,
                    detected_at=datetime.utcnow(),
                    affected_timerange=(context.get('start_time'), context.get('end_time')),
                    affected_components=[context.get('component', 'unknown')],
                    description=f"Survivorship bias detected: {len(missing_entities)} entities missing from analysis",
                    evidence={
                        'missing_entities': list(missing_entities),
                        'current_count': len(current_entities),
                        'historical_count': len(historical_entities),
                        'missing_percentage': len(missing_entities) / len(historical_entities) * 100
                    },
                    confidence_score=0.75,
                    recommended_actions=[
                        "Include delisted/failed entities in analysis",
                        "Implement entity lifecycle tracking",
                        "Review data selection criteria"
                    ]
                )
                
                biases.append(bias)
        
        return biases
    
    def _detect_selection_bias(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect selection bias in data sampling"""
        
        biases = []
        
        # Check for biased sampling
        if 'sample_criteria' in data:
            criteria = data['sample_criteria']
            
            # Check for time-based selection bias
            if 'time_filters' in criteria:
                time_filters = criteria['time_filters']
                
                # Check for cherry-picking of time periods
                if len(time_filters) > 1:
                    # Analyze gaps between selected periods
                    sorted_periods = sorted(time_filters, key=lambda x: x.get('start'))
                    
                    total_selected = sum(
                        (period['end'] - period['start']).total_seconds() 
                        for period in sorted_periods
                    )
                    
                    total_span = (
                        sorted_periods[-1]['end'] - sorted_periods[0]['start']
                    ).total_seconds()
                    
                    selection_ratio = total_selected / total_span
                    
                    if selection_ratio < 0.5:  # Less than 50% of time selected
                        bias = BiasDetectionResult(
                            detection_id=str(uuid.uuid4()),
                            bias_type=BiasType.SELECTION_BIAS,
                            severity_level=SeverityLevel.MEDIUM,
                            detected_at=datetime.utcnow(),
                            affected_timerange=(sorted_periods[0]['start'], sorted_periods[-1]['end']),
                            affected_components=[context.get('component', 'unknown')],
                            description=f"Selection bias detected: only {selection_ratio:.1%} of time period selected",
                            evidence={
                                'selected_periods': len(time_filters),
                                'selection_ratio': selection_ratio,
                                'total_selected_seconds': total_selected,
                                'total_span_seconds': total_span
                            },
                            confidence_score=0.70,
                            recommended_actions=[
                                "Use continuous time periods",
                                "Justify time period selection",
                                "Test robustness across different periods"
                            ]
                        )
                        
                        biases.append(bias)
        
        return biases
    
    def _detect_confirmation_bias(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """Detect confirmation bias in analysis"""
        
        biases = []
        
        # Check for parameter optimization bias
        if 'optimization_results' in data:
            results = data['optimization_results']
            
            # Check for excessive parameter tuning
            if 'parameter_combinations_tested' in results:
                combinations_tested = results['parameter_combinations_tested']
                
                if combinations_tested > 1000:  # Arbitrary threshold
                    bias = BiasDetectionResult(
                        detection_id=str(uuid.uuid4()),
                        bias_type=BiasType.CONFIRMATION_BIAS,
                        severity_level=SeverityLevel.LOW,
                        detected_at=datetime.utcnow(),
                        affected_timerange=(context.get('start_time'), context.get('end_time')),
                        affected_components=[context.get('component', 'unknown')],
                        description=f"Potential confirmation bias: {combinations_tested} parameter combinations tested",
                        evidence={
                            'combinations_tested': combinations_tested,
                            'best_performance': results.get('best_performance'),
                            'parameter_space_size': results.get('parameter_space_size')
                        },
                        confidence_score=0.60,
                        recommended_actions=[
                            "Implement out-of-sample testing",
                            "Use cross-validation",
                            "Limit parameter optimization scope"
                        ]
                    )
                    
                    biases.append(bias)
        
        return biases
    
    def _attempt_auto_remediation(self, biases: List[BiasDetectionResult]):
        """Attempt automatic remediation of detected biases"""
        
        for bias in biases:
            if bias.auto_remediation_possible:
                try:
                    if bias.bias_type == BiasType.LOOK_AHEAD_BIAS:
                        self._remediate_look_ahead_bias(bias)
                    elif bias.bias_type == BiasType.FUTURE_INFORMATION:
                        self._remediate_future_information(bias)
                    
                    bias.resolved = True
                    bias.resolved_at = datetime.utcnow()
                    
                    self.stats['auto_remediated'] += 1
                    
                    logger.info(f"Auto-remediated bias: {bias.detection_id}")
                    
                except Exception as e:
                    logger.error(f"Auto-remediation failed for bias {bias.detection_id}: {e}")
    
    def _remediate_look_ahead_bias(self, bias: BiasDetectionResult):
        """Remediate look-ahead bias"""
        # Implementation would depend on specific data structures
        logger.debug(f"Remediating look-ahead bias: {bias.detection_id}")
    
    def _remediate_future_information(self, bias: BiasDetectionResult):
        """Remediate future information bias"""
        # Implementation would depend on specific data structures
        logger.debug(f"Remediating future information bias: {bias.detection_id}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Run periodic bias detection
                self._run_periodic_checks()
                
                # Clean up old records
                self._cleanup_old_records()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep until next check
                time.sleep(self.config.get('detection_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _run_periodic_checks(self):
        """Run periodic bias detection checks"""
        # Check temporal boundaries
        current_time = datetime.utcnow()
        
        with self.lock:
            for boundary_id, boundary in self.temporal_boundaries.items():
                if boundary.boundary_type == TemporalBoundaryType.HARD_BOUNDARY:
                    # Check if boundary is being enforced
                    if (current_time - boundary.last_updated).total_seconds() > 300:
                        logger.warning(f"Temporal boundary {boundary_id} not updated recently")
    
    def _cleanup_old_records(self):
        """Clean up old data availability records"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self.lock:
            old_records = [
                record_id for record_id, record in self.data_availability_records.items()
                if record.validated_at < cutoff_time
            ]
            
            for record_id in old_records:
                del self.data_availability_records[record_id]
            
            if old_records:
                logger.debug(f"Cleaned up {len(old_records)} old availability records")
    
    def _update_statistics(self):
        """Update detection statistics"""
        with self.lock:
            self.stats['data_points_validated'] = len(self.data_availability_records)
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get bias detection summary"""
        
        with self.lock:
            recent_biases = list(self.bias_detection_history)[-20:]
            
            bias_by_type = defaultdict(int)
            bias_by_severity = defaultdict(int)
            
            for bias in recent_biases:
                bias_by_type[bias.bias_type.value] += 1
                bias_by_severity[bias.severity_level.value] += 1
            
            return {
                'statistics': self.stats.copy(),
                'active_boundaries': len(self.temporal_boundaries),
                'active_availability_records': len(self.data_availability_records),
                'recent_biases': {
                    'total': len(recent_biases),
                    'by_type': dict(bias_by_type),
                    'by_severity': dict(bias_by_severity)
                },
                'monitoring_active': self.monitoring_active,
                'last_updated': datetime.utcnow()
            }

# =============================================================================
# MULTI-TIMEFRAME SYNCHRONIZATION ENGINE
# =============================================================================

class MultiTimeframeSynchronizer:
    """Synchronizes data across multiple timeframes"""
    
    def __init__(self, timeframes: List[int]):
        self.timeframes = sorted(timeframes)
        self.data_buffers = {tf: deque(maxlen=1000) for tf in timeframes}
        self.sync_points = {}
        self.sync_lock = threading.RLock()
        
        logger.info(f"Initialized multi-timeframe synchronizer for {timeframes}")
    
    def add_data_point(self, timeframe: int, timestamp: datetime, data: Any):
        """Add data point for specific timeframe"""
        
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        with self.sync_lock:
            self.data_buffers[timeframe].append({
                'timestamp': timestamp,
                'data': data
            })
            
            # Update sync point
            self._update_sync_point(timeframe, timestamp)
    
    def _update_sync_point(self, timeframe: int, timestamp: datetime):
        """Update synchronization point for timeframe"""
        
        # Find latest common timestamp across all timeframes
        latest_timestamps = {}
        
        for tf in self.timeframes:
            buffer = self.data_buffers[tf]
            if buffer:
                latest_timestamps[tf] = buffer[-1]['timestamp']
        
        # Find minimum timestamp (latest common point)
        if len(latest_timestamps) == len(self.timeframes):
            sync_point = min(latest_timestamps.values())
            self.sync_points[timeframe] = sync_point
    
    def get_synchronized_data(self, target_timestamp: datetime) -> Dict[int, Any]:
        """Get synchronized data for target timestamp"""
        
        with self.sync_lock:
            synchronized_data = {}
            
            for tf in self.timeframes:
                buffer = self.data_buffers[tf]
                
                # Find closest data point at or before target timestamp
                closest_data = None
                min_time_diff = float('inf')
                
                for data_point in buffer:
                    time_diff = (target_timestamp - data_point['timestamp']).total_seconds()
                    
                    if 0 <= time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_data = data_point
                
                if closest_data:
                    synchronized_data[tf] = closest_data
            
            return synchronized_data
    
    def validate_synchronization(self) -> Tuple[bool, List[str]]:
        """Validate synchronization across timeframes"""
        
        errors = []
        
        with self.sync_lock:
            # Check if all timeframes have data
            for tf in self.timeframes:
                if not self.data_buffers[tf]:
                    errors.append(f"No data available for timeframe {tf}")
            
            # Check temporal consistency
            for tf in self.timeframes:
                buffer = self.data_buffers[tf]
                if len(buffer) > 1:
                    # Check timestamp ordering
                    timestamps = [dp['timestamp'] for dp in buffer]
                    if timestamps != sorted(timestamps):
                        errors.append(f"Timestamp ordering violation in timeframe {tf}")
            
            # Check cross-timeframe consistency
            if len(self.sync_points) > 1:
                sync_times = list(self.sync_points.values())
                max_drift = max(sync_times) - min(sync_times)
                
                if max_drift.total_seconds() > 300:  # 5 minutes
                    errors.append(f"Excessive drift between timeframes: {max_drift}")
        
        return len(errors) == 0, errors

# Global instances
temporal_bias_detector = TemporalBiasDetector()

# Export key components
__all__ = [
    'BiasType',
    'SeverityLevel',
    'DataAvailabilityStatus',
    'TemporalBoundaryType',
    'TemporalBoundary',
    'DataAvailabilityRecord',
    'BiasDetectionResult',
    'TemporalConsistencyCheck',
    'TemporalBiasDetector',
    'MultiTimeframeSynchronizer',
    'temporal_bias_detector'
]