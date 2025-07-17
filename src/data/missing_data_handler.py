"""
Missing Data Handler and Interpolation System
Agent 5: Data Quality & Bias Elimination

Advanced missing data detection, handling, and interpolation system that
maintains temporal consistency while providing high-quality data imputation
for trading systems.

Key Features:
- Comprehensive missing data detection
- Multiple interpolation methods
- Temporal consistency preservation
- Quality-aware interpolation
- Bias-free data imputation
- Real-time missing data handling
- Multi-timeframe interpolation
- Confidence scoring for interpolated data
"""

import asyncio
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import structlog

from .data_handler import TickData
from .bar_generator import BarData
from .temporal_bias_detector import TemporalBiasDetector
from .temporal_boundary_enforcer import TemporalBoundaryEnforcer

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class MissingDataType(str, Enum):
    """Types of missing data patterns"""
    RANDOM = "random"                    # Missing completely at random
    SYSTEMATIC = "systematic"            # Missing systematically
    TEMPORAL = "temporal"               # Missing in time periods
    STRUCTURAL = "structural"           # Missing due to structure
    INTERMITTENT = "intermittent"       # Occasional missing values
    BURST = "burst"                     # Multiple consecutive missing

class InterpolationMethod(str, Enum):
    """Interpolation methods"""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MEAN_FILL = "mean_fill"
    MEDIAN_FILL = "median_fill"
    MODE_FILL = "mode_fill"
    REGRESSION = "regression"
    RANDOM_FOREST = "random_forest"
    KALMAN_FILTER = "kalman_filter"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ARIMA = "arima"

class DataQualityLevel(str, Enum):
    """Quality levels for interpolated data"""
    HIGH = "high"           # > 0.9
    MEDIUM = "medium"       # 0.7 - 0.9
    LOW = "low"            # 0.5 - 0.7
    POOR = "poor"          # < 0.5

class GapSeverity(str, Enum):
    """Gap severity levels"""
    MINOR = "minor"         # < 5 minutes
    MODERATE = "moderate"   # 5-30 minutes
    MAJOR = "major"         # 30-120 minutes
    SEVERE = "severe"       # > 120 minutes

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MissingDataPoint:
    """Missing data point identification"""
    gap_id: str
    start_time: datetime
    end_time: datetime
    gap_duration: timedelta
    
    # Data context
    data_type: str
    component: str
    timeframe: str
    
    # Gap characteristics
    gap_severity: GapSeverity
    missing_data_type: MissingDataType
    expected_data_points: int
    
    # Surrounding data
    before_data: Optional[Any] = None
    after_data: Optional[Any] = None
    
    # Context data
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detection_method: str = "automatic"

@dataclass
class InterpolationResult:
    """Result of data interpolation"""
    result_id: str
    gap_id: str
    method_used: InterpolationMethod
    
    # Interpolated data
    interpolated_data: List[Any]
    interpolated_timestamps: List[datetime]
    
    # Quality metrics
    quality_score: float
    confidence_score: float
    quality_level: DataQualityLevel
    
    # Validation metrics
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    
    # Temporal consistency
    temporal_consistency: bool = True
    bias_introduced: bool = False
    
    # Method performance
    execution_time: float = 0.0
    memory_usage: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterpolationConfiguration:
    """Configuration for interpolation"""
    config_id: str
    data_type: str
    timeframe: str
    
    # Method selection
    primary_method: InterpolationMethod
    fallback_methods: List[InterpolationMethod] = field(default_factory=list)
    
    # Quality requirements
    min_quality_score: float = 0.7
    max_gap_duration_minutes: int = 30
    
    # Temporal constraints
    max_lookback_minutes: int = 60
    max_lookahead_minutes: int = 0  # No lookahead by default
    
    # Method parameters
    method_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Context requirements
    required_context_data: List[str] = field(default_factory=list)
    
    # Validation settings
    enable_cross_validation: bool = True
    validation_split: float = 0.2
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

@dataclass
class DataIntegrityCheck:
    """Data integrity check result"""
    check_id: str
    data_identifier: str
    check_timestamp: datetime
    
    # Integrity results
    has_missing_data: bool
    missing_data_percentage: float
    consecutive_gaps: int
    max_gap_duration: timedelta
    
    # Pattern analysis
    missing_pattern: MissingDataType
    pattern_confidence: float
    
    # Recommendations
    recommended_action: str
    interpolation_feasible: bool
    
    # Quality assessment
    data_quality_impact: float
    urgency_level: int  # 1-5 scale
    
    # Metadata
    analysis_duration: timedelta = field(default_factory=lambda: timedelta(0))
    data_points_analyzed: int = 0

# =============================================================================
# MISSING DATA HANDLER
# =============================================================================

class MissingDataHandler:
    """Comprehensive missing data detection and handling system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Data tracking
        self.missing_data_points: Dict[str, MissingDataPoint] = {}
        self.interpolation_results: Dict[str, InterpolationResult] = {}
        self.interpolation_configs: Dict[str, InterpolationConfiguration] = {}
        
        # Data buffers
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.gap_detection_history: deque = deque(maxlen=1000)
        
        # Interpolation engines
        self.interpolation_methods = {
            InterpolationMethod.LINEAR: self._linear_interpolation,
            InterpolationMethod.CUBIC: self._cubic_interpolation,
            InterpolationMethod.SPLINE: self._spline_interpolation,
            InterpolationMethod.POLYNOMIAL: self._polynomial_interpolation,
            InterpolationMethod.FORWARD_FILL: self._forward_fill_interpolation,
            InterpolationMethod.BACKWARD_FILL: self._backward_fill_interpolation,
            InterpolationMethod.MEAN_FILL: self._mean_fill_interpolation,
            InterpolationMethod.MEDIAN_FILL: self._median_fill_interpolation,
            InterpolationMethod.REGRESSION: self._regression_interpolation,
            InterpolationMethod.RANDOM_FOREST: self._random_forest_interpolation,
            InterpolationMethod.KALMAN_FILTER: self._kalman_filter_interpolation,
        }
        
        # External dependencies
        self.bias_detector = TemporalBiasDetector()
        self.boundary_enforcer = TemporalBoundaryEnforcer()
        
        # Processing
        self.processing_executor = ThreadPoolExecutor(max_workers=4)
        self.processing_active = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'total_gaps_detected': 0,
            'gaps_interpolated': 0,
            'interpolation_failures': 0,
            'average_quality_score': 0.0,
            'total_data_points_created': 0,
            'bias_violations': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default configurations
        self._initialize_default_configurations()
        
        logger.info("Missing data handler initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'enable_continuous_monitoring': True,
            'monitoring_interval_seconds': 30,
            'max_gap_duration_minutes': 120,
            'min_quality_threshold': 0.7,
            'enable_bias_detection': True,
            'enable_temporal_validation': True,
            'auto_interpolation_enabled': True,
            'max_interpolation_attempts': 3,
            'quality_validation_enabled': True,
            'cleanup_interval_seconds': 300,
            'default_interpolation_method': InterpolationMethod.LINEAR,
            'confidence_threshold': 0.8
        }
    
    def start_processing(self):
        """Start missing data processing"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="missing_data_processor"
            )
            self.processing_thread.start()
            logger.info("Missing data processing started")
    
    def stop_processing(self):
        """Stop missing data processing"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.processing_executor.shutdown(wait=True)
        logger.info("Missing data processing stopped")
    
    def register_interpolation_config(self, config: InterpolationConfiguration):
        """Register interpolation configuration"""
        with self.lock:
            self.interpolation_configs[config.config_id] = config
            logger.debug(f"Registered interpolation config: {config.config_id}")
    
    async def add_data_point(self, 
                           data_type: str, 
                           timeframe: str, 
                           timestamp: datetime, 
                           data: Any,
                           component: str = "unknown") -> bool:
        """Add data point and check for gaps"""
        
        buffer_key = f"{data_type}_{timeframe}_{component}"
        
        with self.lock:
            # Add to buffer
            self.data_buffers[buffer_key].append({
                'timestamp': timestamp,
                'data': data,
                'component': component
            })
            
            # Check for gaps
            await self._check_for_gaps(buffer_key, data_type, timeframe, component)
            
            return True
    
    async def _check_for_gaps(self, buffer_key: str, data_type: str, timeframe: str, component: str):
        """Check for gaps in data"""
        
        buffer = self.data_buffers[buffer_key]
        
        if len(buffer) < 2:
            return
        
        # Get last two data points
        current_point = buffer[-1]
        previous_point = buffer[-2]
        
        # Calculate expected interval
        expected_interval = self._get_expected_interval(timeframe)
        
        # Check for gap
        actual_interval = (current_point['timestamp'] - previous_point['timestamp']).total_seconds()
        
        if actual_interval > expected_interval * 2:  # Gap detected
            gap_id = str(uuid.uuid4())
            
            # Determine gap severity
            gap_severity = self._classify_gap_severity(actual_interval)
            
            # Create missing data point
            missing_data = MissingDataPoint(
                gap_id=gap_id,
                start_time=previous_point['timestamp'],
                end_time=current_point['timestamp'],
                gap_duration=timedelta(seconds=actual_interval),
                data_type=data_type,
                component=component,
                timeframe=timeframe,
                gap_severity=gap_severity,
                missing_data_type=MissingDataType.TEMPORAL,
                expected_data_points=int(actual_interval / expected_interval) - 1,
                before_data=previous_point['data'],
                after_data=current_point['data']
            )
            
            # Store missing data point
            self.missing_data_points[gap_id] = missing_data
            self.stats['total_gaps_detected'] += 1
            
            # Add to detection history
            self.gap_detection_history.append(missing_data)
            
            # Attempt interpolation if enabled
            if self.config.get('auto_interpolation_enabled', True):
                await self._attempt_interpolation(gap_id)
            
            logger.info(f"Gap detected: {gap_id}, duration: {actual_interval:.0f}s, severity: {gap_severity.value}")
    
    def _get_expected_interval(self, timeframe: str) -> float:
        """Get expected interval for timeframe"""
        
        timeframe_intervals = {
            'tick': 1,
            '1s': 1,
            '5s': 5,
            '10s': 10,
            '30s': 30,
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        return timeframe_intervals.get(timeframe, 60)  # Default to 1 minute
    
    def _classify_gap_severity(self, gap_seconds: float) -> GapSeverity:
        """Classify gap severity"""
        
        if gap_seconds < 300:  # 5 minutes
            return GapSeverity.MINOR
        elif gap_seconds < 1800:  # 30 minutes
            return GapSeverity.MODERATE
        elif gap_seconds < 7200:  # 2 hours
            return GapSeverity.MAJOR
        else:
            return GapSeverity.SEVERE
    
    async def _attempt_interpolation(self, gap_id: str):
        """Attempt to interpolate missing data"""
        
        if gap_id not in self.missing_data_points:
            return
        
        missing_data = self.missing_data_points[gap_id]
        
        # Find appropriate configuration
        config = self._find_interpolation_config(
            missing_data.data_type, 
            missing_data.timeframe, 
            missing_data.component
        )
        
        if not config:
            logger.warning(f"No interpolation config found for {gap_id}")
            return
        
        # Check if gap is too large
        if missing_data.gap_duration.total_seconds() > config.max_gap_duration_minutes * 60:
            logger.warning(f"Gap too large for interpolation: {gap_id}")
            return
        
        # Attempt interpolation with primary method
        result = await self._interpolate_gap(missing_data, config, config.primary_method)
        
        # Try fallback methods if primary fails
        if not result or result.quality_score < config.min_quality_score:
            for fallback_method in config.fallback_methods:
                result = await self._interpolate_gap(missing_data, config, fallback_method)
                if result and result.quality_score >= config.min_quality_score:
                    break
        
        # Store result
        if result:
            self.interpolation_results[gap_id] = result
            
            if result.quality_score >= config.min_quality_score:
                self.stats['gaps_interpolated'] += 1
                self.stats['total_data_points_created'] += len(result.interpolated_data)
                
                # Validate temporal consistency
                if config.enable_cross_validation:
                    await self._validate_interpolation_consistency(result)
                
                logger.info(f"Interpolation successful: {gap_id}, quality: {result.quality_score:.3f}")
            else:
                self.stats['interpolation_failures'] += 1
                logger.warning(f"Interpolation quality too low: {gap_id}, quality: {result.quality_score:.3f}")
        else:
            self.stats['interpolation_failures'] += 1
            logger.error(f"Interpolation failed: {gap_id}")
    
    def _find_interpolation_config(self, data_type: str, timeframe: str, component: str) -> Optional[InterpolationConfiguration]:
        """Find appropriate interpolation configuration"""
        
        # Try exact match first
        for config in self.interpolation_configs.values():
            if (config.data_type == data_type and 
                config.timeframe == timeframe and 
                config.active):
                return config
        
        # Try data type and timeframe match
        for config in self.interpolation_configs.values():
            if (config.data_type == data_type and 
                config.timeframe == timeframe and 
                config.active):
                return config
        
        # Try data type match
        for config in self.interpolation_configs.values():
            if config.data_type == data_type and config.active:
                return config
        
        # Use default configuration
        return self.interpolation_configs.get('default')
    
    async def _interpolate_gap(self, 
                             missing_data: MissingDataPoint, 
                             config: InterpolationConfiguration, 
                             method: InterpolationMethod) -> Optional[InterpolationResult]:
        """Interpolate missing data using specified method"""
        
        start_time = time.time()
        
        try:
            # Prepare data
            buffer_key = f"{missing_data.data_type}_{missing_data.timeframe}_{missing_data.component}"
            buffer = self.data_buffers[buffer_key]
            
            # Get surrounding data
            surrounding_data = self._get_surrounding_data(buffer, missing_data, config)
            
            if not surrounding_data:
                return None
            
            # Run interpolation method
            interpolation_func = self.interpolation_methods.get(method)
            if not interpolation_func:
                logger.error(f"Unknown interpolation method: {method}")
                return None
            
            interpolated_data, timestamps = await asyncio.get_event_loop().run_in_executor(
                self.processing_executor,
                interpolation_func,
                surrounding_data,
                missing_data,
                config.method_parameters.get(method.value, {})
            )
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(
                interpolated_data, surrounding_data, missing_data
            )
            
            confidence_score = self._calculate_confidence_score(
                interpolated_data, surrounding_data, method
            )
            
            quality_level = self._determine_quality_level(quality_score)
            
            # Validate temporal consistency
            temporal_consistency = await self._validate_temporal_consistency(
                timestamps, missing_data
            )
            
            # Check for bias introduction
            bias_introduced = await self._check_bias_introduction(
                interpolated_data, timestamps, missing_data
            )
            
            # Create result
            result = InterpolationResult(
                result_id=str(uuid.uuid4()),
                gap_id=missing_data.gap_id,
                method_used=method,
                interpolated_data=interpolated_data,
                interpolated_timestamps=timestamps,
                quality_score=quality_score,
                confidence_score=confidence_score,
                quality_level=quality_level,
                temporal_consistency=temporal_consistency,
                bias_introduced=bias_introduced,
                execution_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Interpolation error for {missing_data.gap_id}: {e}")
            return None
    
    def _get_surrounding_data(self, 
                            buffer: deque, 
                            missing_data: MissingDataPoint, 
                            config: InterpolationConfiguration) -> List[Dict[str, Any]]:
        """Get surrounding data for interpolation"""
        
        surrounding_data = []
        
        # Get data before gap
        for data_point in buffer:
            if data_point['timestamp'] < missing_data.start_time:
                # Check if within lookback window
                time_diff = (missing_data.start_time - data_point['timestamp']).total_seconds()
                if time_diff <= config.max_lookback_minutes * 60:
                    surrounding_data.append(data_point)
        
        # Get data after gap
        for data_point in buffer:
            if data_point['timestamp'] > missing_data.end_time:
                # Check if within lookahead window
                time_diff = (data_point['timestamp'] - missing_data.end_time).total_seconds()
                if time_diff <= config.max_lookahead_minutes * 60:
                    surrounding_data.append(data_point)
        
        # Sort by timestamp
        surrounding_data.sort(key=lambda x: x['timestamp'])
        
        return surrounding_data
    
    # =============================================================================
    # INTERPOLATION METHODS
    # =============================================================================
    
    def _linear_interpolation(self, 
                            surrounding_data: List[Dict[str, Any]], 
                            missing_data: MissingDataPoint,
                            parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Linear interpolation"""
        
        if len(surrounding_data) < 2:
            raise ValueError("Insufficient data for linear interpolation")
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Create interpolation function
        interp_func = interpolate.interp1d(
            numeric_timestamps, 
            values, 
            kind='linear', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        # Create timestamps for missing data points
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Interpolate values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            interpolated_value = float(interp_func(numeric_ts))
            interpolated_values.append(interpolated_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _cubic_interpolation(self, 
                           surrounding_data: List[Dict[str, Any]], 
                           missing_data: MissingDataPoint,
                           parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Cubic interpolation"""
        
        if len(surrounding_data) < 4:
            # Fall back to linear if insufficient data
            return self._linear_interpolation(surrounding_data, missing_data, parameters)
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Create cubic interpolation function
        interp_func = interpolate.interp1d(
            numeric_timestamps, 
            values, 
            kind='cubic', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Interpolate values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            interpolated_value = float(interp_func(numeric_ts))
            interpolated_values.append(interpolated_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _spline_interpolation(self, 
                            surrounding_data: List[Dict[str, Any]], 
                            missing_data: MissingDataPoint,
                            parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Spline interpolation"""
        
        if len(surrounding_data) < 3:
            return self._linear_interpolation(surrounding_data, missing_data, parameters)
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Create spline interpolation
        degree = min(3, len(values) - 1)
        tck = interpolate.splrep(numeric_timestamps, values, s=0, k=degree)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Interpolate values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            interpolated_value = float(interpolate.splev(numeric_ts, tck))
            interpolated_values.append(interpolated_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _polynomial_interpolation(self, 
                                surrounding_data: List[Dict[str, Any]], 
                                missing_data: MissingDataPoint,
                                parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Polynomial interpolation"""
        
        degree = parameters.get('degree', 2)
        
        if len(surrounding_data) < degree + 1:
            return self._linear_interpolation(surrounding_data, missing_data, parameters)
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Fit polynomial
        coeffs = np.polyfit(numeric_timestamps, values, degree)
        poly_func = np.poly1d(coeffs)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Interpolate values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            interpolated_value = float(poly_func(numeric_ts))
            interpolated_values.append(interpolated_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _forward_fill_interpolation(self, 
                                  surrounding_data: List[Dict[str, Any]], 
                                  missing_data: MissingDataPoint,
                                  parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Forward fill interpolation"""
        
        # Find last value before gap
        last_value = None
        for data_point in surrounding_data:
            if data_point['timestamp'] < missing_data.start_time:
                last_value = self._extract_numeric_value(data_point['data'])
        
        if last_value is None:
            raise ValueError("No data point before gap for forward fill")
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Fill with last value
        interpolated_values = [last_value] * len(interpolated_timestamps)
        
        return interpolated_values, interpolated_timestamps
    
    def _backward_fill_interpolation(self, 
                                   surrounding_data: List[Dict[str, Any]], 
                                   missing_data: MissingDataPoint,
                                   parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Backward fill interpolation"""
        
        # Find first value after gap
        next_value = None
        for data_point in surrounding_data:
            if data_point['timestamp'] > missing_data.end_time:
                next_value = self._extract_numeric_value(data_point['data'])
                break
        
        if next_value is None:
            raise ValueError("No data point after gap for backward fill")
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Fill with next value
        interpolated_values = [next_value] * len(interpolated_timestamps)
        
        return interpolated_values, interpolated_timestamps
    
    def _mean_fill_interpolation(self, 
                               surrounding_data: List[Dict[str, Any]], 
                               missing_data: MissingDataPoint,
                               parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Mean fill interpolation"""
        
        if not surrounding_data:
            raise ValueError("No surrounding data for mean fill")
        
        # Calculate mean of surrounding data
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        mean_value = np.mean(values)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Fill with mean value
        interpolated_values = [mean_value] * len(interpolated_timestamps)
        
        return interpolated_values, interpolated_timestamps
    
    def _median_fill_interpolation(self, 
                                 surrounding_data: List[Dict[str, Any]], 
                                 missing_data: MissingDataPoint,
                                 parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Median fill interpolation"""
        
        if not surrounding_data:
            raise ValueError("No surrounding data for median fill")
        
        # Calculate median of surrounding data
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        median_value = np.median(values)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Fill with median value
        interpolated_values = [median_value] * len(interpolated_timestamps)
        
        return interpolated_values, interpolated_timestamps
    
    def _regression_interpolation(self, 
                                surrounding_data: List[Dict[str, Any]], 
                                missing_data: MissingDataPoint,
                                parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Regression-based interpolation"""
        
        if len(surrounding_data) < 5:
            return self._linear_interpolation(surrounding_data, missing_data, parameters)
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric features
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Prepare features (time and time-based features)
        X = []
        for ts in numeric_timestamps:
            X.append([ts, ts**2, np.sin(ts/3600), np.cos(ts/3600)])  # Time features
        
        X = np.array(X)
        y = np.array(values)
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Predict values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            features = np.array([[numeric_ts, numeric_ts**2, np.sin(numeric_ts/3600), np.cos(numeric_ts/3600)]])
            predicted_value = model.predict(features)[0]
            interpolated_values.append(predicted_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _random_forest_interpolation(self, 
                                   surrounding_data: List[Dict[str, Any]], 
                                   missing_data: MissingDataPoint,
                                   parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Random Forest interpolation"""
        
        if len(surrounding_data) < 10:
            return self._regression_interpolation(surrounding_data, missing_data, parameters)
        
        # Extract timestamps and values
        timestamps = [d['timestamp'] for d in surrounding_data]
        values = [self._extract_numeric_value(d['data']) for d in surrounding_data]
        
        # Convert timestamps to numeric features
        start_timestamp = min(timestamps)
        numeric_timestamps = [(ts - start_timestamp).total_seconds() for ts in timestamps]
        
        # Prepare features
        X = []
        for i, ts in enumerate(numeric_timestamps):
            features = [ts, ts**2, np.sin(ts/3600), np.cos(ts/3600)]
            
            # Add lagged features if available
            if i > 0:
                features.append(values[i-1])
            else:
                features.append(values[0])
            
            if i > 1:
                features.append(values[i-2])
            else:
                features.append(values[0])
            
            X.append(features)
        
        X = np.array(X)
        y = np.array(values)
        
        # Fit Random Forest model
        model = RandomForestRegressor(
            n_estimators=parameters.get('n_estimators', 100),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Generate interpolated timestamps
        start_time = missing_data.start_time
        end_time = missing_data.end_time
        
        expected_interval = self._get_expected_interval(missing_data.timeframe)
        interpolated_timestamps = []
        
        current_time = start_time + timedelta(seconds=expected_interval)
        while current_time < end_time:
            interpolated_timestamps.append(current_time)
            current_time += timedelta(seconds=expected_interval)
        
        # Predict values
        interpolated_values = []
        for ts in interpolated_timestamps:
            numeric_ts = (ts - start_timestamp).total_seconds()
            
            # Use last known values for lagged features
            last_value = missing_data.before_data if missing_data.before_data else values[-1]
            second_last_value = values[-2] if len(values) > 1 else last_value
            
            features = np.array([[
                numeric_ts, 
                numeric_ts**2, 
                np.sin(numeric_ts/3600), 
                np.cos(numeric_ts/3600),
                last_value,
                second_last_value
            ]])
            
            predicted_value = model.predict(features)[0]
            interpolated_values.append(predicted_value)
        
        return interpolated_values, interpolated_timestamps
    
    def _kalman_filter_interpolation(self, 
                                   surrounding_data: List[Dict[str, Any]], 
                                   missing_data: MissingDataPoint,
                                   parameters: Dict[str, Any]) -> Tuple[List[Any], List[datetime]]:
        """Kalman filter interpolation (simplified)"""
        
        # For now, fall back to linear interpolation
        # A full Kalman filter implementation would be more complex
        return self._linear_interpolation(surrounding_data, missing_data, parameters)
    
    def _extract_numeric_value(self, data: Any) -> float:
        """Extract numeric value from data"""
        
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, TickData):
            return float(data.price)
        elif isinstance(data, BarData):
            return float(data.close)
        elif hasattr(data, 'value'):
            return float(data.value)
        else:
            return 0.0
    
    def _calculate_quality_score(self, 
                               interpolated_data: List[Any], 
                               surrounding_data: List[Dict[str, Any]], 
                               missing_data: MissingDataPoint) -> float:
        """Calculate quality score for interpolated data"""
        
        # Base quality score
        base_score = 0.8
        
        # Penalize for large gaps
        gap_penalty = min(0.3, missing_data.gap_duration.total_seconds() / 3600 * 0.1)
        
        # Penalize for data scarcity
        data_penalty = max(0, 0.2 - len(surrounding_data) * 0.02)
        
        # Bonus for data consistency
        consistency_bonus = 0.1 if len(surrounding_data) > 5 else 0
        
        quality_score = base_score - gap_penalty - data_penalty + consistency_bonus
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_confidence_score(self, 
                                  interpolated_data: List[Any], 
                                  surrounding_data: List[Dict[str, Any]], 
                                  method: InterpolationMethod) -> float:
        """Calculate confidence score for interpolation"""
        
        # Method-based confidence
        method_confidence = {
            InterpolationMethod.LINEAR: 0.8,
            InterpolationMethod.CUBIC: 0.85,
            InterpolationMethod.SPLINE: 0.9,
            InterpolationMethod.REGRESSION: 0.75,
            InterpolationMethod.RANDOM_FOREST: 0.8,
            InterpolationMethod.FORWARD_FILL: 0.6,
            InterpolationMethod.BACKWARD_FILL: 0.6,
            InterpolationMethod.MEAN_FILL: 0.5,
            InterpolationMethod.MEDIAN_FILL: 0.5,
        }.get(method, 0.7)
        
        # Adjust based on data availability
        data_factor = min(1.0, len(surrounding_data) / 10)
        
        return method_confidence * data_factor
    
    def _determine_quality_level(self, quality_score: float) -> DataQualityLevel:
        """Determine quality level from score"""
        
        if quality_score >= 0.9:
            return DataQualityLevel.HIGH
        elif quality_score >= 0.7:
            return DataQualityLevel.MEDIUM
        elif quality_score >= 0.5:
            return DataQualityLevel.LOW
        else:
            return DataQualityLevel.POOR
    
    async def _validate_temporal_consistency(self, 
                                           timestamps: List[datetime], 
                                           missing_data: MissingDataPoint) -> bool:
        """Validate temporal consistency of interpolated data"""
        
        # Check timestamp ordering
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                return False
        
        # Check that timestamps are within gap
        for ts in timestamps:
            if not (missing_data.start_time < ts < missing_data.end_time):
                return False
        
        return True
    
    async def _check_bias_introduction(self, 
                                     interpolated_data: List[Any], 
                                     timestamps: List[datetime], 
                                     missing_data: MissingDataPoint) -> bool:
        """Check if interpolation introduces bias"""
        
        if not self.config.get('enable_bias_detection', True):
            return False
        
        # Prepare data for bias detection
        bias_data = {
            'timestamps': timestamps,
            'values': interpolated_data,
            'interpolated': True
        }
        
        bias_context = {
            'gap_id': missing_data.gap_id,
            'component': missing_data.component,
            'interpolation_method': 'unknown'
        }
        
        # Check for bias
        bias_results = self.bias_detector.detect_bias(bias_data, bias_context)
        
        if bias_results:
            self.stats['bias_violations'] += 1
            return True
        
        return False
    
    async def _validate_interpolation_consistency(self, result: InterpolationResult):
        """Validate interpolation consistency"""
        
        # Cross-validation would be implemented here
        # For now, just log the validation
        logger.debug(f"Validating interpolation consistency: {result.result_id}")
    
    def _initialize_default_configurations(self):
        """Initialize default interpolation configurations"""
        
        # Default configuration
        default_config = InterpolationConfiguration(
            config_id="default",
            data_type="all",
            timeframe="all",
            primary_method=InterpolationMethod.LINEAR,
            fallback_methods=[
                InterpolationMethod.FORWARD_FILL,
                InterpolationMethod.MEAN_FILL
            ],
            min_quality_score=0.7,
            max_gap_duration_minutes=30
        )
        
        self.register_interpolation_config(default_config)
        
        # High-frequency data configuration
        hf_config = InterpolationConfiguration(
            config_id="high_frequency",
            data_type="tick",
            timeframe="1s",
            primary_method=InterpolationMethod.LINEAR,
            fallback_methods=[InterpolationMethod.FORWARD_FILL],
            min_quality_score=0.8,
            max_gap_duration_minutes=5
        )
        
        self.register_interpolation_config(hf_config)
        
        # Bar data configuration
        bar_config = InterpolationConfiguration(
            config_id="bar_data",
            data_type="bar",
            timeframe="5m",
            primary_method=InterpolationMethod.CUBIC,
            fallback_methods=[
                InterpolationMethod.LINEAR,
                InterpolationMethod.FORWARD_FILL
            ],
            min_quality_score=0.7,
            max_gap_duration_minutes=60
        )
        
        self.register_interpolation_config(bar_config)
    
    def _processing_loop(self):
        """Background processing loop"""
        
        while self.processing_active:
            try:
                # Process pending interpolations
                self._process_pending_interpolations()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Update statistics
                self._update_statistics()
                
                time.sleep(self.config.get('monitoring_interval_seconds', 30))
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    
    def _process_pending_interpolations(self):
        """Process pending interpolations"""
        
        # Check for gaps that need interpolation
        for gap_id, missing_data in self.missing_data_points.items():
            if gap_id not in self.interpolation_results:
                # Attempt interpolation
                asyncio.create_task(self._attempt_interpolation(gap_id))
    
    def _cleanup_old_data(self):
        """Clean up old data"""
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.get('cleanup_interval_seconds', 300))
        
        # Clean up old missing data points
        old_gaps = [
            gap_id for gap_id, missing_data in self.missing_data_points.items()
            if missing_data.detected_at < cutoff_time
        ]
        
        for gap_id in old_gaps:
            if gap_id in self.missing_data_points:
                del self.missing_data_points[gap_id]
            if gap_id in self.interpolation_results:
                del self.interpolation_results[gap_id]
        
        if old_gaps:
            logger.debug(f"Cleaned up {len(old_gaps)} old gaps")
    
    def _update_statistics(self):
        """Update processing statistics"""
        
        # Calculate average quality score
        if self.interpolation_results:
            quality_scores = [r.quality_score for r in self.interpolation_results.values()]
            self.stats['average_quality_score'] = np.mean(quality_scores)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get missing data processing summary"""
        
        return {
            'statistics': self.stats.copy(),
            'active_gaps': len(self.missing_data_points),
            'completed_interpolations': len(self.interpolation_results),
            'registered_configs': len(self.interpolation_configs),
            'processing_active': self.processing_active,
            'recent_gaps': [
                {
                    'gap_id': gap.gap_id,
                    'duration_seconds': gap.gap_duration.total_seconds(),
                    'severity': gap.gap_severity.value,
                    'detected_at': gap.detected_at
                }
                for gap in list(self.gap_detection_history)[-10:]
            ],
            'last_updated': datetime.utcnow()
        }

# Global instance
missing_data_handler = MissingDataHandler()

# Export key components
__all__ = [
    'MissingDataType',
    'InterpolationMethod',
    'DataQualityLevel',
    'GapSeverity',
    'MissingDataPoint',
    'InterpolationResult',
    'InterpolationConfiguration',
    'DataIntegrityCheck',
    'MissingDataHandler',
    'missing_data_handler'
]