"""
Multi-Timeframe Data Synchronization System
Agent 5: Data Quality & Bias Elimination

Advanced multi-timeframe synchronization system that ensures temporal
consistency across different timeframes while preventing bias and
maintaining data quality.

Key Features:
- Multi-timeframe data synchronization
- Temporal alignment and consistency checks
- Cross-timeframe bias detection
- Synchronized data delivery
- Timeframe-specific quality validation
- Hierarchical timeframe relationships
- Real-time synchronization monitoring
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import structlog

from .temporal_bias_detector import TemporalBiasDetector, BiasDetectionResult
from .temporal_boundary_enforcer import TemporalBoundaryEnforcer, DataAccessRequest, AccessType
from .data_handler import TickData
from .bar_generator import BarData

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class TimeframeType(str, Enum):
    """Types of timeframes"""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

class SynchronizationMode(str, Enum):
    """Synchronization modes"""
    STRICT = "strict"           # Perfect alignment required
    TOLERANT = "tolerant"       # Small misalignments allowed
    FLEXIBLE = "flexible"       # Adaptive synchronization
    OPPORTUNISTIC = "opportunistic"  # Best effort synchronization

class DataAvailabilityMode(str, Enum):
    """Data availability modes"""
    WAIT_FOR_ALL = "wait_for_all"
    BEST_EFFORT = "best_effort"
    MINIMUM_REQUIRED = "minimum_required"
    PRIORITIZED = "prioritized"

class AlignmentStrategy(str, Enum):
    """Alignment strategies for timeframes"""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    WEIGHTED_AVERAGE = "weighted_average"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimeframeDefinition:
    """Definition of a timeframe"""
    timeframe_id: str
    timeframe_type: TimeframeType
    duration_seconds: int
    
    # Hierarchy
    parent_timeframe: Optional[str] = None
    child_timeframes: List[str] = field(default_factory=list)
    
    # Synchronization settings
    sync_mode: SynchronizationMode = SynchronizationMode.STRICT
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.FORWARD_FILL
    max_delay_seconds: int = 60
    
    # Data requirements
    min_data_points: int = 1
    max_gap_seconds: int = 300
    
    # Quality thresholds
    min_quality_score: float = 0.7
    quality_weight: float = 1.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

@dataclass
class TimeframeDataPoint:
    """Data point with timeframe information"""
    data_id: str
    timeframe_id: str
    timestamp: datetime
    data: Any
    
    # Quality metrics
    quality_score: float = 1.0
    confidence_level: float = 1.0
    
    # Temporal information
    data_latency: timedelta = field(default_factory=lambda: timedelta(0))
    source_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Synchronization metadata
    sync_group_id: Optional[str] = None
    alignment_applied: bool = False
    interpolated: bool = False
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    contributes_to: List[str] = field(default_factory=list)
    
    # Validation
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class SynchronizationPoint:
    """Synchronization point across timeframes"""
    sync_id: str
    reference_timestamp: datetime
    timeframe_data: Dict[str, TimeframeDataPoint] = field(default_factory=dict)
    
    # Synchronization status
    is_synchronized: bool = False
    sync_quality: float = 0.0
    sync_latency: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Alignment information
    aligned_timeframes: Set[str] = field(default_factory=set)
    missing_timeframes: Set[str] = field(default_factory=set)
    interpolated_timeframes: Set[str] = field(default_factory=set)
    
    # Quality metrics
    overall_quality: float = 0.0
    quality_by_timeframe: Dict[str, float] = field(default_factory=dict)
    
    # Bias detection
    bias_check_performed: bool = False
    bias_issues: List[BiasDetectionResult] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

@dataclass
class SynchronizationConfiguration:
    """Configuration for synchronization"""
    config_id: str
    timeframes: List[str]
    
    # Synchronization parameters
    sync_mode: SynchronizationMode = SynchronizationMode.STRICT
    availability_mode: DataAvailabilityMode = DataAvailabilityMode.WAIT_FOR_ALL
    max_sync_delay_seconds: int = 300
    
    # Quality requirements
    min_overall_quality: float = 0.7
    quality_weights: Dict[str, float] = field(default_factory=dict)
    
    # Bias detection
    enable_bias_detection: bool = True
    bias_detection_threshold: float = 0.8
    
    # Alignment settings
    default_alignment_strategy: AlignmentStrategy = AlignmentStrategy.FORWARD_FILL
    alignment_strategies: Dict[str, AlignmentStrategy] = field(default_factory=dict)
    
    # Tolerances
    time_tolerance_seconds: int = 5
    quality_tolerance: float = 0.05
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

# =============================================================================
# MULTI-TIMEFRAME SYNCHRONIZER
# =============================================================================

class MultiTimeframeSynchronizer:
    """Advanced multi-timeframe synchronization system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Timeframe management
        self.timeframes: Dict[str, TimeframeDefinition] = {}
        self.sync_configurations: Dict[str, SynchronizationConfiguration] = {}
        
        # Data buffers
        self.data_buffers: Dict[str, deque] = {}
        self.sync_points: Dict[str, SynchronizationPoint] = {}
        
        # Synchronization state
        self.active_syncs: Dict[str, asyncio.Task] = {}
        self.sync_history: deque = deque(maxlen=1000)
        
        # External dependencies
        self.bias_detector = TemporalBiasDetector()
        self.boundary_enforcer = TemporalBoundaryEnforcer()
        
        # Processing
        self.sync_executor = ThreadPoolExecutor(max_workers=4)
        self.processing_active = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'total_sync_points': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'bias_issues_detected': 0,
            'alignment_operations': 0,
            'quality_violations': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize built-in timeframes
        self._initialize_standard_timeframes()
        
        logger.info("Multi-timeframe synchronizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'enable_continuous_processing': True,
            'processing_interval_seconds': 1,
            'max_buffer_size': 10000,
            'default_sync_mode': SynchronizationMode.STRICT,
            'enable_bias_detection': True,
            'enable_quality_monitoring': True,
            'max_sync_delay_seconds': 300,
            'alignment_tolerance_seconds': 5,
            'quality_threshold': 0.7,
            'auto_create_sync_points': True,
            'cleanup_interval_seconds': 300
        }
    
    def start_processing(self):
        """Start continuous synchronization processing"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="timeframe_sync_processor"
            )
            self.processing_thread.start()
            logger.info("Multi-timeframe synchronization processing started")
    
    def stop_processing(self):
        """Stop continuous synchronization processing"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # Cancel active sync tasks
        for task in self.active_syncs.values():
            task.cancel()
        
        self.sync_executor.shutdown(wait=True)
        logger.info("Multi-timeframe synchronization processing stopped")
    
    def register_timeframe(self, timeframe_def: TimeframeDefinition):
        """Register a timeframe definition"""
        with self.lock:
            self.timeframes[timeframe_def.timeframe_id] = timeframe_def
            
            # Create data buffer
            self.data_buffers[timeframe_def.timeframe_id] = deque(
                maxlen=self.config.get('max_buffer_size', 10000)
            )
            
            logger.debug(f"Registered timeframe: {timeframe_def.timeframe_id}")
    
    def register_sync_configuration(self, sync_config: SynchronizationConfiguration):
        """Register a synchronization configuration"""
        with self.lock:
            self.sync_configurations[sync_config.config_id] = sync_config
            logger.debug(f"Registered sync configuration: {sync_config.config_id}")
    
    async def add_data_point(self, timeframe_id: str, data_point: TimeframeDataPoint) -> bool:
        """Add data point to timeframe buffer"""
        
        if timeframe_id not in self.timeframes:
            logger.error(f"Unknown timeframe: {timeframe_id}")
            return False
        
        with self.lock:
            # Add to buffer
            self.data_buffers[timeframe_id].append(data_point)
            
            # Validate temporal consistency
            if not await self._validate_temporal_consistency(timeframe_id, data_point):
                logger.warning(f"Temporal consistency violation in {timeframe_id}")
                return False
            
            # Check if synchronization is needed
            if self.config.get('auto_create_sync_points', True):
                await self._check_sync_triggers(timeframe_id, data_point)
            
            logger.debug(f"Added data point to {timeframe_id}: {data_point.data_id}")
            return True
    
    async def create_synchronization_point(self, 
                                         config_id: str, 
                                         reference_timestamp: datetime) -> Optional[SynchronizationPoint]:
        """Create a synchronization point"""
        
        if config_id not in self.sync_configurations:
            logger.error(f"Unknown sync configuration: {config_id}")
            return None
        
        sync_config = self.sync_configurations[config_id]
        sync_id = str(uuid.uuid4())
        
        # Create synchronization point
        sync_point = SynchronizationPoint(
            sync_id=sync_id,
            reference_timestamp=reference_timestamp
        )
        
        with self.lock:
            self.sync_points[sync_id] = sync_point
            
            # Start synchronization task
            sync_task = asyncio.create_task(
                self._perform_synchronization(sync_id, sync_config)
            )
            self.active_syncs[sync_id] = sync_task
            
            logger.info(f"Created synchronization point: {sync_id}")
            return sync_point
    
    async def _perform_synchronization(self, sync_id: str, sync_config: SynchronizationConfiguration):
        """Perform synchronization for a sync point"""
        
        try:
            sync_point = self.sync_points[sync_id]
            
            # Collect data from all timeframes
            await self._collect_timeframe_data(sync_point, sync_config)
            
            # Validate data availability
            if not await self._validate_data_availability(sync_point, sync_config):
                logger.warning(f"Data availability validation failed for sync {sync_id}")
                self.stats['failed_syncs'] += 1
                return
            
            # Perform alignment
            await self._perform_alignment(sync_point, sync_config)
            
            # Detect bias
            if sync_config.enable_bias_detection:
                await self._detect_cross_timeframe_bias(sync_point, sync_config)
            
            # Calculate quality metrics
            await self._calculate_sync_quality(sync_point, sync_config)
            
            # Validate overall quality
            if sync_point.overall_quality < sync_config.min_overall_quality:
                logger.warning(f"Sync quality below threshold: {sync_point.overall_quality}")
                self.stats['quality_violations'] += 1
                return
            
            # Mark as synchronized
            sync_point.is_synchronized = True
            sync_point.completed_at = datetime.utcnow()
            sync_point.sync_latency = sync_point.completed_at - sync_point.created_at
            
            # Update statistics
            self.stats['successful_syncs'] += 1
            self.stats['total_sync_points'] += 1
            
            # Add to history
            self.sync_history.append(sync_point)
            
            logger.info(f"Synchronization completed: {sync_id}, quality: {sync_point.overall_quality:.3f}")
            
        except Exception as e:
            logger.error(f"Synchronization failed for {sync_id}: {e}")
            self.stats['failed_syncs'] += 1
            
        finally:
            # Clean up
            with self.lock:
                if sync_id in self.active_syncs:
                    del self.active_syncs[sync_id]
    
    async def _collect_timeframe_data(self, sync_point: SynchronizationPoint, sync_config: SynchronizationConfiguration):
        """Collect data from all timeframes for synchronization"""
        
        reference_time = sync_point.reference_timestamp
        
        for timeframe_id in sync_config.timeframes:
            if timeframe_id not in self.data_buffers:
                sync_point.missing_timeframes.add(timeframe_id)
                continue
            
            # Find closest data point
            closest_data = await self._find_closest_data_point(
                timeframe_id, reference_time, sync_config.time_tolerance_seconds
            )
            
            if closest_data:
                sync_point.timeframe_data[timeframe_id] = closest_data
                sync_point.aligned_timeframes.add(timeframe_id)
            else:
                sync_point.missing_timeframes.add(timeframe_id)
    
    async def _find_closest_data_point(self, 
                                     timeframe_id: str, 
                                     reference_time: datetime, 
                                     tolerance_seconds: int) -> Optional[TimeframeDataPoint]:
        """Find closest data point to reference time"""
        
        buffer = self.data_buffers[timeframe_id]
        closest_data = None
        min_time_diff = float('inf')
        
        for data_point in buffer:
            time_diff = abs((data_point.timestamp - reference_time).total_seconds())
            
            if time_diff <= tolerance_seconds and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_data = data_point
        
        return closest_data
    
    async def _validate_data_availability(self, sync_point: SynchronizationPoint, sync_config: SynchronizationConfiguration) -> bool:
        """Validate data availability according to configuration"""
        
        if sync_config.availability_mode == DataAvailabilityMode.WAIT_FOR_ALL:
            return len(sync_point.missing_timeframes) == 0
        
        elif sync_config.availability_mode == DataAvailabilityMode.MINIMUM_REQUIRED:
            # At least 50% of timeframes must have data
            required_count = len(sync_config.timeframes) // 2
            return len(sync_point.aligned_timeframes) >= required_count
        
        elif sync_config.availability_mode == DataAvailabilityMode.BEST_EFFORT:
            # Any data is acceptable
            return len(sync_point.aligned_timeframes) > 0
        
        elif sync_config.availability_mode == DataAvailabilityMode.PRIORITIZED:
            # Check if high-priority timeframes have data
            # This would require priority information in timeframe definitions
            return len(sync_point.aligned_timeframes) > 0
        
        return True
    
    async def _perform_alignment(self, sync_point: SynchronizationPoint, sync_config: SynchronizationConfiguration):
        """Perform alignment of timeframe data"""
        
        # Handle missing timeframes
        for timeframe_id in sync_point.missing_timeframes:
            if timeframe_id in sync_config.alignment_strategies:
                strategy = sync_config.alignment_strategies[timeframe_id]
            else:
                strategy = sync_config.default_alignment_strategy
            
            # Attempt to create aligned data
            aligned_data = await self._create_aligned_data(
                timeframe_id, sync_point.reference_timestamp, strategy
            )
            
            if aligned_data:
                sync_point.timeframe_data[timeframe_id] = aligned_data
                sync_point.aligned_timeframes.add(timeframe_id)
                sync_point.interpolated_timeframes.add(timeframe_id)
                
                self.stats['alignment_operations'] += 1
                
                logger.debug(f"Aligned data for {timeframe_id} using {strategy.value}")
    
    async def _create_aligned_data(self, 
                                 timeframe_id: str, 
                                 reference_time: datetime, 
                                 strategy: AlignmentStrategy) -> Optional[TimeframeDataPoint]:
        """Create aligned data using specified strategy"""
        
        buffer = self.data_buffers[timeframe_id]
        
        if strategy == AlignmentStrategy.FORWARD_FILL:
            # Find latest data before reference time
            latest_data = None
            for data_point in reversed(buffer):
                if data_point.timestamp <= reference_time:
                    latest_data = data_point
                    break
            
            if latest_data:
                aligned_data = TimeframeDataPoint(
                    data_id=str(uuid.uuid4()),
                    timeframe_id=timeframe_id,
                    timestamp=reference_time,
                    data=latest_data.data,
                    quality_score=latest_data.quality_score * 0.9,  # Reduced quality
                    alignment_applied=True,
                    interpolated=True
                )
                return aligned_data
        
        elif strategy == AlignmentStrategy.BACKWARD_FILL:
            # Find earliest data after reference time
            earliest_data = None
            for data_point in buffer:
                if data_point.timestamp >= reference_time:
                    earliest_data = data_point
                    break
            
            if earliest_data:
                aligned_data = TimeframeDataPoint(
                    data_id=str(uuid.uuid4()),
                    timeframe_id=timeframe_id,
                    timestamp=reference_time,
                    data=earliest_data.data,
                    quality_score=earliest_data.quality_score * 0.9,
                    alignment_applied=True,
                    interpolated=True
                )
                return aligned_data
        
        elif strategy == AlignmentStrategy.NEAREST_NEIGHBOR:
            # Find nearest data point
            nearest_data = None
            min_time_diff = float('inf')
            
            for data_point in buffer:
                time_diff = abs((data_point.timestamp - reference_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    nearest_data = data_point
            
            if nearest_data:
                aligned_data = TimeframeDataPoint(
                    data_id=str(uuid.uuid4()),
                    timeframe_id=timeframe_id,
                    timestamp=reference_time,
                    data=nearest_data.data,
                    quality_score=nearest_data.quality_score * 0.95,
                    alignment_applied=True,
                    interpolated=True
                )
                return aligned_data
        
        elif strategy == AlignmentStrategy.INTERPOLATE:
            # Perform interpolation between surrounding points
            return await self._interpolate_data_point(timeframe_id, reference_time)
        
        return None
    
    async def _interpolate_data_point(self, timeframe_id: str, reference_time: datetime) -> Optional[TimeframeDataPoint]:
        """Interpolate data point between surrounding points"""
        
        buffer = self.data_buffers[timeframe_id]
        
        # Find surrounding points
        before_point = None
        after_point = None
        
        for data_point in buffer:
            if data_point.timestamp <= reference_time:
                if before_point is None or data_point.timestamp > before_point.timestamp:
                    before_point = data_point
            elif data_point.timestamp > reference_time:
                if after_point is None or data_point.timestamp < after_point.timestamp:
                    after_point = data_point
        
        if before_point and after_point:
            # Simple linear interpolation
            time_diff = (after_point.timestamp - before_point.timestamp).total_seconds()
            weight = (reference_time - before_point.timestamp).total_seconds() / time_diff
            
            # Interpolate data (this is simplified - real implementation would handle different data types)
            interpolated_value = self._interpolate_values(before_point.data, after_point.data, weight)
            
            interpolated_data = TimeframeDataPoint(
                data_id=str(uuid.uuid4()),
                timeframe_id=timeframe_id,
                timestamp=reference_time,
                data=interpolated_value,
                quality_score=min(before_point.quality_score, after_point.quality_score) * 0.8,
                alignment_applied=True,
                interpolated=True
            )
            
            return interpolated_data
        
        return None
    
    def _interpolate_values(self, value1: Any, value2: Any, weight: float) -> Any:
        """Interpolate between two values"""
        
        # Simple numeric interpolation
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return value1 + (value2 - value1) * weight
        
        # For other types, use nearest neighbor
        return value1 if weight < 0.5 else value2
    
    async def _detect_cross_timeframe_bias(self, sync_point: SynchronizationPoint, sync_config: SynchronizationConfiguration):
        """Detect bias across timeframes"""
        
        # Prepare data for bias detection
        bias_data = {
            'timeframes': list(sync_point.timeframe_data.keys()),
            'reference_timestamp': sync_point.reference_timestamp,
            'data_points': {
                tf_id: dp.data for tf_id, dp in sync_point.timeframe_data.items()
            }
        }
        
        bias_context = {
            'sync_id': sync_point.sync_id,
            'component': 'multi_timeframe_synchronizer',
            'sync_mode': sync_config.sync_mode.value
        }
        
        # Detect bias
        bias_results = self.bias_detector.detect_bias(bias_data, bias_context)
        
        if bias_results:
            sync_point.bias_issues = bias_results
            self.stats['bias_issues_detected'] += len(bias_results)
            
            logger.warning(f"Bias detected in sync {sync_point.sync_id}: {len(bias_results)} issues")
        
        sync_point.bias_check_performed = True
    
    async def _calculate_sync_quality(self, sync_point: SynchronizationPoint, sync_config: SynchronizationConfiguration):
        """Calculate synchronization quality metrics"""
        
        if not sync_point.timeframe_data:
            sync_point.overall_quality = 0.0
            return
        
        # Calculate quality for each timeframe
        for timeframe_id, data_point in sync_point.timeframe_data.items():
            quality = data_point.quality_score
            
            # Apply penalties for interpolation
            if data_point.interpolated:
                quality *= 0.8
            
            # Apply penalties for alignment
            if data_point.alignment_applied:
                quality *= 0.9
            
            sync_point.quality_by_timeframe[timeframe_id] = quality
        
        # Calculate overall quality
        qualities = list(sync_point.quality_by_timeframe.values())
        weights = [sync_config.quality_weights.get(tf_id, 1.0) for tf_id in sync_point.quality_by_timeframe.keys()]
        
        if qualities:
            sync_point.overall_quality = np.average(qualities, weights=weights)
        else:
            sync_point.overall_quality = 0.0
        
        # Apply bias penalty
        if sync_point.bias_issues:
            critical_biases = sum(1 for bias in sync_point.bias_issues if bias.severity_level.value == 'critical')
            bias_penalty = critical_biases * 0.2
            sync_point.overall_quality = max(0.0, sync_point.overall_quality - bias_penalty)
    
    async def _validate_temporal_consistency(self, timeframe_id: str, data_point: TimeframeDataPoint) -> bool:
        """Validate temporal consistency of data point"""
        
        # Check with boundary enforcer
        access_request = DataAccessRequest(
            request_id=str(uuid.uuid4()),
            requester_id="multi_timeframe_synchronizer",
            access_type=AccessType.READ,
            request_time=datetime.utcnow(),
            data_timestamp=data_point.timestamp,
            latest_allowable_time=datetime.utcnow(),
            data_identifier=data_point.data_id,
            data_type=timeframe_id,
            component="synchronizer"
        )
        
        is_valid, violation_reason, _ = await self.boundary_enforcer.request_data_access(access_request)
        
        if not is_valid:
            logger.warning(f"Temporal consistency violation: {violation_reason}")
            return False
        
        return True
    
    async def _check_sync_triggers(self, timeframe_id: str, data_point: TimeframeDataPoint):
        """Check if synchronization should be triggered"""
        
        # Find applicable sync configurations
        for config_id, sync_config in self.sync_configurations.items():
            if timeframe_id in sync_config.timeframes and sync_config.active:
                # Check if sync is needed
                if await self._should_create_sync_point(sync_config, data_point):
                    await self.create_synchronization_point(config_id, data_point.timestamp)
    
    async def _should_create_sync_point(self, sync_config: SynchronizationConfiguration, data_point: TimeframeDataPoint) -> bool:
        """Determine if sync point should be created"""
        
        # Check if enough time has passed since last sync
        if self.sync_history:
            last_sync = self.sync_history[-1]
            time_since_last = (data_point.timestamp - last_sync.reference_timestamp).total_seconds()
            
            if time_since_last < sync_config.max_sync_delay_seconds:
                return False
        
        # Check if all timeframes have recent data
        for timeframe_id in sync_config.timeframes:
            if timeframe_id not in self.data_buffers:
                continue
            
            buffer = self.data_buffers[timeframe_id]
            if not buffer:
                continue
            
            # Check if latest data is recent enough
            latest_data = buffer[-1]
            age = (data_point.timestamp - latest_data.timestamp).total_seconds()
            
            if age > sync_config.max_sync_delay_seconds:
                return False
        
        return True
    
    def _initialize_standard_timeframes(self):
        """Initialize standard timeframes"""
        
        # 1-second timeframe
        second_tf = TimeframeDefinition(
            timeframe_id="1s",
            timeframe_type=TimeframeType.SECOND,
            duration_seconds=1,
            sync_mode=SynchronizationMode.STRICT
        )
        self.register_timeframe(second_tf)
        
        # 1-minute timeframe
        minute_tf = TimeframeDefinition(
            timeframe_id="1m",
            timeframe_type=TimeframeType.MINUTE,
            duration_seconds=60,
            parent_timeframe="1s",
            sync_mode=SynchronizationMode.STRICT
        )
        self.register_timeframe(minute_tf)
        
        # 5-minute timeframe
        five_min_tf = TimeframeDefinition(
            timeframe_id="5m",
            timeframe_type=TimeframeType.MINUTE,
            duration_seconds=300,
            parent_timeframe="1m",
            sync_mode=SynchronizationMode.TOLERANT
        )
        self.register_timeframe(five_min_tf)
        
        # 30-minute timeframe
        thirty_min_tf = TimeframeDefinition(
            timeframe_id="30m",
            timeframe_type=TimeframeType.MINUTE,
            duration_seconds=1800,
            parent_timeframe="5m",
            sync_mode=SynchronizationMode.TOLERANT
        )
        self.register_timeframe(thirty_min_tf)
        
        # Update parent-child relationships
        second_tf.child_timeframes = ["1m"]
        minute_tf.child_timeframes = ["5m"]
        five_min_tf.child_timeframes = ["30m"]
        
        logger.debug("Standard timeframes initialized")
    
    def _processing_loop(self):
        """Background processing loop"""
        
        while self.processing_active:
            try:
                # Clean up completed sync points
                self._cleanup_sync_points()
                
                # Monitor sync performance
                self._monitor_sync_performance()
                
                # Update statistics
                self._update_statistics()
                
                time.sleep(self.config.get('processing_interval_seconds', 1))
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    
    def _cleanup_sync_points(self):
        """Clean up old sync points"""
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.get('cleanup_interval_seconds', 300))
        
        with self.lock:
            old_sync_ids = [
                sync_id for sync_id, sync_point in self.sync_points.items()
                if sync_point.created_at < cutoff_time and sync_point.is_synchronized
            ]
            
            for sync_id in old_sync_ids:
                del self.sync_points[sync_id]
            
            if old_sync_ids:
                logger.debug(f"Cleaned up {len(old_sync_ids)} old sync points")
    
    def _monitor_sync_performance(self):
        """Monitor synchronization performance"""
        
        if not self.sync_history:
            return
        
        # Calculate average sync latency
        recent_syncs = [s for s in self.sync_history if s.completed_at and 
                       (datetime.utcnow() - s.completed_at).total_seconds() < 3600]
        
        if recent_syncs:
            avg_latency = np.mean([s.sync_latency.total_seconds() for s in recent_syncs])
            avg_quality = np.mean([s.overall_quality for s in recent_syncs])
            
            if avg_latency > 60:  # 1 minute threshold
                logger.warning(f"High sync latency detected: {avg_latency:.1f}s")
            
            if avg_quality < 0.8:
                logger.warning(f"Low sync quality detected: {avg_quality:.3f}")
    
    def _update_statistics(self):
        """Update synchronization statistics"""
        
        with self.lock:
            # Update current statistics
            self.stats['active_syncs'] = len(self.active_syncs)
            self.stats['pending_sync_points'] = len(self.sync_points)
            self.stats['registered_timeframes'] = len(self.timeframes)
            self.stats['sync_configurations'] = len(self.sync_configurations)
    
    def get_synchronization_summary(self) -> Dict[str, Any]:
        """Get synchronization summary"""
        
        with self.lock:
            recent_syncs = [s for s in self.sync_history if 
                          (datetime.utcnow() - s.created_at).total_seconds() < 3600]
            
            return {
                'statistics': self.stats.copy(),
                'registered_timeframes': list(self.timeframes.keys()),
                'active_configurations': len(self.sync_configurations),
                'recent_performance': {
                    'total_syncs': len(recent_syncs),
                    'average_quality': np.mean([s.overall_quality for s in recent_syncs]) if recent_syncs else 0.0,
                    'success_rate': (len([s for s in recent_syncs if s.is_synchronized]) / len(recent_syncs)) if recent_syncs else 0.0
                },
                'processing_active': self.processing_active,
                'last_updated': datetime.utcnow()
            }
    
    async def get_sync_point_status(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific sync point"""
        
        if sync_id not in self.sync_points:
            return None
        
        sync_point = self.sync_points[sync_id]
        
        return {
            'sync_id': sync_id,
            'reference_timestamp': sync_point.reference_timestamp,
            'is_synchronized': sync_point.is_synchronized,
            'overall_quality': sync_point.overall_quality,
            'aligned_timeframes': list(sync_point.aligned_timeframes),
            'missing_timeframes': list(sync_point.missing_timeframes),
            'interpolated_timeframes': list(sync_point.interpolated_timeframes),
            'bias_issues': len(sync_point.bias_issues),
            'sync_latency': sync_point.sync_latency.total_seconds() if sync_point.sync_latency else 0.0,
            'created_at': sync_point.created_at,
            'completed_at': sync_point.completed_at
        }

# Global instance
multi_timeframe_synchronizer = MultiTimeframeSynchronizer()

# Export key components
__all__ = [
    'TimeframeType',
    'SynchronizationMode',
    'DataAvailabilityMode',
    'AlignmentStrategy',
    'TimeframeDefinition',
    'TimeframeDataPoint',
    'SynchronizationPoint',
    'SynchronizationConfiguration',
    'MultiTimeframeSynchronizer',
    'multi_timeframe_synchronizer'
]