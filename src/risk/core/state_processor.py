"""
Risk State Vector Processor for MARL System

Handles processing, normalization, and validation of the 10-dimensional 
risk state vector as specified in the PRD.

Features:
- Real-time state vector processing
- Z-score normalization with rolling statistics
- State validation and sanitization
- Performance monitoring <5ms processing time
- Event-driven state updates
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class NormalizationMethod(Enum):
    """Normalization methods for risk state components"""
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    ROBUST = "robust"  # Median-based normalization
    PERCENTILE = "percentile"


@dataclass
class RiskStateStatistics:
    """Rolling statistics for risk state normalization"""
    
    # Statistics for each of the 10 risk dimensions
    means: np.ndarray = field(default_factory=lambda: np.zeros(10))
    stds: np.ndarray = field(default_factory=lambda: np.ones(10))
    mins: np.ndarray = field(default_factory=lambda: np.full(10, np.inf))
    maxs: np.ndarray = field(default_factory=lambda: np.full(10, -np.inf))
    medians: np.ndarray = field(default_factory=lambda: np.zeros(10))
    percentile_25: np.ndarray = field(default_factory=lambda: np.zeros(10))
    percentile_75: np.ndarray = field(default_factory=lambda: np.zeros(10))
    
    # Update tracking
    update_count: int = 0
    last_update: Optional[datetime] = None
    
    def update_statistics(self, state_vector: np.ndarray, decay_factor: float = 0.95):
        """Update rolling statistics with exponential decay"""
        if len(state_vector) != 10:
            raise ValueError(f"Expected 10-dimensional vector, got {len(state_vector)}")
        
        if self.update_count == 0:
            # Initialize on first update
            self.means = state_vector.copy()
            self.stds = np.ones(10)
            self.mins = state_vector.copy()
            self.maxs = state_vector.copy()
            self.medians = state_vector.copy()
            self.percentile_25 = state_vector.copy()
            self.percentile_75 = state_vector.copy()
        else:
            # Exponentially weighted moving average
            self.means = decay_factor * self.means + (1 - decay_factor) * state_vector
            
            # Update min/max
            self.mins = np.minimum(self.mins, state_vector)
            self.maxs = np.maximum(self.maxs, state_vector)
            
            # Exponentially weighted standard deviation estimate
            squared_diff = (state_vector - self.means) ** 2
            if self.update_count == 1:
                self.stds = np.sqrt(squared_diff)
            else:
                variance = decay_factor * (self.stds ** 2) + (1 - decay_factor) * squared_diff
                self.stds = np.sqrt(variance)
                
            # Update medians and percentiles (simplified exponential weighting)
            self.medians = decay_factor * self.medians + (1 - decay_factor) * state_vector
            self.percentile_25 = decay_factor * self.percentile_25 + (1 - decay_factor) * state_vector * 0.8
            self.percentile_75 = decay_factor * self.percentile_75 + (1 - decay_factor) * state_vector * 1.2
        
        self.update_count += 1
        self.last_update = datetime.now()
    
    def is_valid(self) -> bool:
        """Check if statistics are valid for normalization"""
        return (self.update_count > 0 and 
                not np.any(np.isnan(self.means)) and
                not np.any(np.isnan(self.stds)) and
                np.all(self.stds > 1e-8))  # Avoid division by zero


@dataclass
class StateProcessingConfig:
    """Configuration for state processing"""
    
    # Normalization settings
    normalization_method: NormalizationMethod = NormalizationMethod.Z_SCORE
    decay_factor: float = 0.95
    min_samples_for_normalization: int = 10
    
    # Validation settings
    enable_outlier_detection: bool = True
    outlier_threshold_sigma: float = 4.0
    enable_range_clipping: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    
    # Performance settings
    max_processing_time_ms: float = 5.0
    enable_parallel_processing: bool = True
    
    # State vector dimension names for logging
    dimension_names: List[str] = field(default_factory=lambda: [
        'account_equity_normalized',
        'open_positions_count', 
        'volatility_regime',
        'correlation_risk',
        'var_estimate_5pct',
        'current_drawdown_pct',
        'margin_usage_pct',
        'time_of_day_risk',
        'market_stress_level',
        'liquidity_conditions'
    ])


class RiskStateProcessor:
    """
    Processes and normalizes 10-dimensional risk state vectors for MARL agents
    
    Provides:
    - Real-time state normalization with rolling statistics
    - State validation and outlier detection
    - Performance monitoring <5ms processing time
    - Thread-safe operations for concurrent agent access
    """
    
    def __init__(self, 
                 config: StateProcessingConfig,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize risk state processor
        
        Args:
            config: Processing configuration
            event_bus: Event bus for state update notifications
        """
        self.config = config
        self.event_bus = event_bus
        
        # Rolling statistics
        self.statistics = RiskStateStatistics()
        self.raw_state_history = deque(maxlen=1000)  # Keep last 1000 states
        
        # Processing performance tracking
        self.processing_times = deque(maxlen=100)
        self.total_processed = 0
        self.outliers_detected = 0
        self.validation_failures = 0
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2) if config.enable_parallel_processing else None
        
        logger.info("Risk state processor initialized",
                   normalization_method=config.normalization_method.value,
                   decay_factor=config.decay_factor)
    
    def process_state(self, raw_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process raw risk state vector into normalized form
        
        Args:
            raw_state: Raw 10-dimensional risk state vector
            
        Returns:
            Tuple of (normalized_state, processing_metadata)
        """
        start_time = datetime.now()
        metadata = {
            'processing_start': start_time,
            'outlier_detected': False,
            'validation_passed': True,
            'normalization_applied': False,
            'clipping_applied': False
        }
        
        try:
            with self.lock:
                # Validate input
                if not self._validate_raw_state(raw_state):
                    self.validation_failures += 1
                    metadata['validation_passed'] = False
                    # Return safe default state
                    normalized_state = np.zeros(10)
                    logger.warning("State validation failed, using default state")
                    return normalized_state, metadata
                
                # Store raw state for statistics
                self.raw_state_history.append(raw_state.copy())
                
                # Detect outliers
                if self.config.enable_outlier_detection and self.statistics.is_valid():
                    is_outlier = self._detect_outliers(raw_state)
                    if is_outlier:
                        self.outliers_detected += 1
                        metadata['outlier_detected'] = True
                        logger.warning("Outlier detected in risk state",
                                     state=raw_state.tolist(),
                                     dimension_names=self.config.dimension_names)
                
                # Update statistics
                self.statistics.update_statistics(raw_state, self.config.decay_factor)
                
                # Normalize state
                normalized_state = self._normalize_state(raw_state)
                metadata['normalization_applied'] = True
                
                # Apply range clipping if enabled
                if self.config.enable_range_clipping:
                    original_state = normalized_state.copy()
                    normalized_state = np.clip(
                        normalized_state,
                        self.config.clip_range[0],
                        self.config.clip_range[1]
                    )
                    if not np.array_equal(original_state, normalized_state):
                        metadata['clipping_applied'] = True
                
                # Track performance
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.processing_times.append(processing_time)
                self.total_processed += 1
                
                metadata['processing_time_ms'] = processing_time
                metadata['processing_end'] = datetime.now()
                
                # Check performance target
                if processing_time > self.config.max_processing_time_ms:
                    logger.warning("State processing exceeded target time",
                                 processing_time=processing_time,
                                 target=self.config.max_processing_time_ms)
                
                # Publish state update event
                if self.event_bus:
                    self._publish_state_update(raw_state, normalized_state, metadata)
                
                return normalized_state, metadata
                
        except Exception as e:
            logger.error("Error in state processing", error=str(e))
            metadata['error'] = str(e)
            return np.zeros(10), metadata
    
    def _validate_raw_state(self, state: np.ndarray) -> bool:
        """Validate raw state vector"""
        
        # Check dimensions
        if state.shape != (10,):
            logger.error("Invalid state dimensions", shape=state.shape, expected=(10,))
            return False
        
        # Check for NaN/Inf values
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            logger.error("State contains NaN or Inf values", state=state.tolist())
            return False
        
        # Check reasonable ranges for each dimension
        dimension_checks = [
            (0, 0.0, 100.0),    # account_equity_normalized (0-100x)
            (1, 0, 1000),       # open_positions_count (0-1000)
            (2, 0.0, 1.0),      # volatility_regime (0-1)
            (3, -1.0, 1.0),     # correlation_risk (-1 to 1)
            (4, 0.0, 1.0),      # var_estimate_5pct (0-100%)
            (5, 0.0, 1.0),      # current_drawdown_pct (0-100%)
            (6, 0.0, 1.0),      # margin_usage_pct (0-100%)
            (7, 0.0, 1.0),      # time_of_day_risk (0-1)
            (8, 0.0, 1.0),      # market_stress_level (0-1)
            (9, 0.0, 1.0),      # liquidity_conditions (0-1)
        ]
        
        for dim_idx, min_val, max_val in dimension_checks:
            if not (min_val <= state[dim_idx] <= max_val):
                logger.warning("State dimension out of expected range",
                             dimension=self.config.dimension_names[dim_idx],
                             value=state[dim_idx],
                             expected_range=(min_val, max_val))
                # Don't fail validation for range issues, just warn
        
        return True
    
    def _detect_outliers(self, state: np.ndarray) -> bool:
        """Detect outliers using statistical methods"""
        if not self.statistics.is_valid():
            return False
        
        # Z-score based outlier detection
        z_scores = np.abs((state - self.statistics.means) / (self.statistics.stds + 1e-8))
        outlier_mask = z_scores > self.config.outlier_threshold_sigma
        
        return np.any(outlier_mask)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state vector using configured method"""
        
        if not self.statistics.is_valid():
            # Not enough data for normalization yet
            return state.copy()
        
        if self.config.normalization_method == NormalizationMethod.Z_SCORE:
            return (state - self.statistics.means) / (self.statistics.stds + 1e-8)
        
        elif self.config.normalization_method == NormalizationMethod.MIN_MAX:
            ranges = self.statistics.maxs - self.statistics.mins
            ranges[ranges < 1e-8] = 1.0  # Avoid division by zero
            return (state - self.statistics.mins) / ranges
        
        elif self.config.normalization_method == NormalizationMethod.ROBUST:
            # Median-based robust normalization
            mad = np.abs(state - self.statistics.medians)  # Median absolute deviation
            return (state - self.statistics.medians) / (mad + 1e-8)
        
        elif self.config.normalization_method == NormalizationMethod.PERCENTILE:
            # Percentile-based normalization
            iqr = self.statistics.percentile_75 - self.statistics.percentile_25
            iqr[iqr < 1e-8] = 1.0
            return (state - self.statistics.medians) / iqr
        
        else:
            logger.warning("Unknown normalization method", method=self.config.normalization_method)
            return state.copy()
    
    def _publish_state_update(self, raw_state: np.ndarray, normalized_state: np.ndarray, metadata: Dict[str, Any]):
        """Publish state update event via event bus"""
        if not self.event_bus:
            return
        
        event_data = {
            'raw_state': raw_state.tolist(),
            'normalized_state': normalized_state.tolist(),
            'processing_metadata': metadata,
            'statistics_summary': {
                'update_count': self.statistics.update_count,
                'last_update': self.statistics.last_update.isoformat() if self.statistics.last_update else None
            },
            'performance_metrics': self.get_performance_metrics()
        }
        
        event = self.event_bus.create_event(
            EventType.STATE_UPDATE,
            event_data,
            "risk_state_processor"
        )
        self.event_bus.publish(event)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get state processing performance metrics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        max_processing_time = np.max(self.processing_times) if self.processing_times else 0.0
        
        return {
            'total_processed': self.total_processed,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'outliers_detected': self.outliers_detected,
            'validation_failures': self.validation_failures,
            'outlier_rate': self.outliers_detected / max(1, self.total_processed),
            'validation_failure_rate': self.validation_failures / max(1, self.total_processed),
            'statistics_valid': self.statistics.is_valid(),
            'update_count': self.statistics.update_count
        }
    
    def get_state_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of current state statistics"""
        if not self.statistics.is_valid():
            return {'status': 'insufficient_data'}
        
        return {
            'status': 'valid',
            'update_count': self.statistics.update_count,
            'last_update': self.statistics.last_update.isoformat() if self.statistics.last_update else None,
            'means': self.statistics.means.tolist(),
            'stds': self.statistics.stds.tolist(),
            'mins': self.statistics.mins.tolist(),
            'maxs': self.statistics.maxs.tolist(),
            'dimension_names': self.config.dimension_names
        }
    
    def reset_statistics(self):
        """Reset all accumulated statistics"""
        with self.lock:
            self.statistics = RiskStateStatistics()
            self.raw_state_history.clear()
            self.processing_times.clear()
            self.total_processed = 0
            self.outliers_detected = 0
            self.validation_failures = 0
            logger.info("Risk state processor statistics reset")
    
    def set_statistics(self, means: np.ndarray, stds: np.ndarray):
        """Manually set statistics (for loading pre-trained models)"""
        with self.lock:
            if len(means) != 10 or len(stds) != 10:
                raise ValueError("Statistics must be 10-dimensional")
            
            self.statistics.means = means.copy()
            self.statistics.stds = stds.copy()
            self.statistics.update_count = 1
            self.statistics.last_update = datetime.now()
            
            logger.info("Risk state processor statistics manually set")
    
    def shutdown(self):
        """Shutdown processor and cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Risk state processor shutdown")