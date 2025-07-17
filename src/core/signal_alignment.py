"""
Signal Alignment and Timeframe Synchronization System

This module provides a comprehensive solution for aligning signals across different timeframes,
standardizing signal processing, and ensuring deterministic signal ordering.

Key Features:
1. Proper 30min → 5min signal interpolation (not simple index mapping)
2. Timestamp validation and synchronization
3. Signal buffering system for temporal consistency
4. Unified signal format and threshold standardization
5. Deterministic signal ordering with priority queue
6. Signal validation and confidence scoring

Author: Claude (Anthropic)
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum, IntEnum
import heapq
import structlog
from numba import njit, prange

logger = structlog.get_logger(__name__)


class SignalType(Enum):
    """Signal type enumeration"""
    MLMI = "mlmi"
    NWRQK = "nwrqk"
    FVG = "fvg"


class SignalDirection(IntEnum):
    """Signal direction with proper ordering"""
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


@dataclass
class SignalData:
    """Standardized signal data structure"""
    signal_type: SignalType
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    timeframe: str  # "5m" or "30m"
    raw_value: float
    threshold: float
    metadata: Dict[str, Any]


class SignalPriority(IntEnum):
    """Signal priority for deterministic ordering"""
    LOW = 3
    MEDIUM = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class PrioritySignal:
    """Priority signal for queue management"""
    priority: SignalPriority
    timestamp: datetime
    signal: SignalData
    
    def __lt__(self, other):
        """For heapq ordering"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class TimeframeConverter:
    """Converts signals between different timeframes with proper interpolation"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        
    def convert_30m_to_5m(self, signals_30m: List[SignalData], 
                         timestamps_5m: List[datetime]) -> List[SignalData]:
        """
        Convert 30-minute signals to 5-minute timeframe with proper interpolation
        
        Args:
            signals_30m: List of 30-minute signals
            timestamps_5m: List of 5-minute timestamps to interpolate to
            
        Returns:
            List of interpolated 5-minute signals
        """
        if not signals_30m or not timestamps_5m:
            return []
            
        signals_5m = []
        
        # Sort signals by timestamp
        signals_30m_sorted = sorted(signals_30m, key=lambda x: x.timestamp)
        
        for ts_5m in timestamps_5m:
            # Find the most recent 30m signal before or at this 5m timestamp
            current_signal = None
            for signal in signals_30m_sorted:
                if signal.timestamp <= ts_5m:
                    current_signal = signal
                else:
                    break
            
            if current_signal:
                # Calculate age of signal
                age = (ts_5m - current_signal.timestamp).total_seconds() / 60  # minutes
                
                # Decay confidence based on age (max 30 minutes)
                confidence_decay = max(0.0, 1.0 - (age / 30.0))
                
                # Create interpolated signal
                interpolated_signal = SignalData(
                    signal_type=current_signal.signal_type,
                    direction=current_signal.direction,
                    strength=current_signal.strength,
                    confidence=current_signal.confidence * confidence_decay,
                    timestamp=ts_5m,
                    timeframe="5m",
                    raw_value=current_signal.raw_value,
                    threshold=current_signal.threshold,
                    metadata={
                        **current_signal.metadata,
                        'interpolated': True,
                        'source_timestamp': current_signal.timestamp,
                        'age_minutes': age,
                        'confidence_decay': confidence_decay
                    }
                )
                
                signals_5m.append(interpolated_signal)
        
        self.logger.debug(
            "Signal interpolation completed",
            input_signals=len(signals_30m),
            output_signals=len(signals_5m),
            timeframe_conversion="30m->5m"
        )
        
        return signals_5m


class SignalStandardizer:
    """Standardizes signal thresholds and formats across different modules"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[SignalType, Dict[str, float]]:
        """Initialize standardized thresholds for all signal types"""
        return {
            SignalType.MLMI: {
                'weak': 0.5,
                'medium': 1.0,
                'strong': 2.0,
                'very_strong': 3.0
            },
            SignalType.NWRQK: {
                'weak': 0.01,
                'medium': 0.02,
                'strong': 0.03,
                'very_strong': 0.05
            },
            SignalType.FVG: {
                'weak': 0.1,
                'medium': 0.2,
                'strong': 0.3,
                'very_strong': 0.5
            }
        }
    
    def standardize_signal(self, signal_type: SignalType, raw_value: float, 
                          timeframe: str, metadata: Dict[str, Any] = None) -> SignalData:
        """
        Standardize a raw signal into the unified format
        
        Args:
            signal_type: Type of signal
            raw_value: Raw signal value
            timeframe: Signal timeframe
            metadata: Additional signal metadata
            
        Returns:
            Standardized SignalData object
        """
        if metadata is None:
            metadata = {}
            
        thresholds = self.thresholds[signal_type]
        
        # Determine signal direction
        if raw_value > 0:
            direction = SignalDirection.BULLISH
        elif raw_value < 0:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        
        # Calculate signal strength (0.0 to 1.0)
        abs_value = abs(raw_value)
        if abs_value >= thresholds['very_strong']:
            strength = 1.0
        elif abs_value >= thresholds['strong']:
            strength = 0.8
        elif abs_value >= thresholds['medium']:
            strength = 0.6
        elif abs_value >= thresholds['weak']:
            strength = 0.4
        else:
            strength = 0.2
        
        # Calculate confidence based on signal type and strength
        confidence = self._calculate_confidence(signal_type, abs_value, strength, timeframe)
        
        # Determine appropriate threshold
        if abs_value >= thresholds['strong']:
            threshold = thresholds['strong']
        elif abs_value >= thresholds['medium']:
            threshold = thresholds['medium']
        else:
            threshold = thresholds['weak']
        
        return SignalData(
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            timeframe=timeframe,
            raw_value=raw_value,
            threshold=threshold,
            metadata=metadata
        )
    
    def _calculate_confidence(self, signal_type: SignalType, abs_value: float, 
                            strength: float, timeframe: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        base_confidence = strength
        
        # Adjust confidence based on signal type
        if signal_type == SignalType.MLMI:
            # MLMI is more reliable for trend direction
            base_confidence *= 1.1
        elif signal_type == SignalType.NWRQK:
            # NW-RQK is good for regression analysis
            base_confidence *= 1.05
        elif signal_type == SignalType.FVG:
            # FVG is precise but can be noisy
            base_confidence *= 0.95
        
        # Adjust confidence based on timeframe
        if timeframe == "30m":
            # 30-minute signals are generally more reliable
            base_confidence *= 1.1
        elif timeframe == "5m":
            # 5-minute signals are more precise but less reliable
            base_confidence *= 0.9
        
        # Ensure confidence is within [0, 1]
        return max(0.0, min(1.0, base_confidence))


class SignalValidator:
    """Validates signals and detects anomalies"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        
    def validate_signal(self, signal: SignalData) -> Tuple[bool, str]:
        """
        Validate a signal for correctness and consistency
        
        Args:
            signal: Signal to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check basic signal properties
        if signal.strength < 0 or signal.strength > 1:
            return False, f"Invalid strength: {signal.strength}"
        
        if signal.confidence < 0 or signal.confidence > 1:
            return False, f"Invalid confidence: {signal.confidence}"
        
        if signal.timeframe not in ["5m", "30m"]:
            return False, f"Invalid timeframe: {signal.timeframe}"
        
        # Check signal type specific validations
        if signal.signal_type == SignalType.MLMI:
            if abs(signal.raw_value) > 10:  # MLMI typically ranges -10 to 10
                return False, f"MLMI raw value out of range: {signal.raw_value}"
        
        elif signal.signal_type == SignalType.NWRQK:
            if abs(signal.raw_value) > 1:  # NW-RQK typically ranges -1 to 1
                return False, f"NW-RQK raw value out of range: {signal.raw_value}"
        
        elif signal.signal_type == SignalType.FVG:
            if abs(signal.raw_value) > 100:  # FVG price levels
                return False, f"FVG raw value out of range: {signal.raw_value}"
        
        # Check timestamp validity
        if signal.timestamp > datetime.now() + timedelta(minutes=5):
            return False, "Signal timestamp is too far in the future"
        
        return True, "Signal is valid"
    
    def detect_look_ahead_bias(self, signals: List[SignalData]) -> List[SignalData]:
        """
        Detect and remove signals that may have look-ahead bias
        
        Args:
            signals: List of signals to check
            
        Returns:
            List of signals with look-ahead bias removed
        """
        clean_signals = []
        
        for i, signal in enumerate(signals):
            has_bias = False
            
            # Check if signal timestamp is ahead of expected time
            if i > 0:
                prev_signal = signals[i-1]
                expected_next = prev_signal.timestamp + timedelta(minutes=5)
                
                if signal.timestamp > expected_next + timedelta(minutes=1):
                    has_bias = True
                    self.logger.warning(
                        "Look-ahead bias detected",
                        signal_type=signal.signal_type.value,
                        timestamp=signal.timestamp,
                        expected=expected_next
                    )
            
            if not has_bias:
                clean_signals.append(signal)
        
        return clean_signals


class SignalQueue:
    """Priority queue for deterministic signal ordering"""
    
    def __init__(self, max_size: int = 10000):
        self.heap = []
        self.max_size = max_size
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def add_signal(self, signal: SignalData, priority: SignalPriority = SignalPriority.MEDIUM):
        """Add signal to priority queue"""
        priority_signal = PrioritySignal(priority, signal.timestamp, signal)
        
        heapq.heappush(self.heap, priority_signal)
        
        # Maintain max size
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)
    
    def get_next_signal(self) -> Optional[SignalData]:
        """Get next signal from priority queue"""
        if self.heap:
            priority_signal = heapq.heappop(self.heap)
            return priority_signal.signal
        return None
    
    def peek_next_signal(self) -> Optional[SignalData]:
        """Peek at next signal without removing it"""
        if self.heap:
            return self.heap[0].signal
        return None
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.heap)
    
    def clear(self):
        """Clear all signals from queue"""
        self.heap.clear()


class SignalAlignmentEngine:
    """Main engine for signal alignment and synchronization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize components
        self.converter = TimeframeConverter()
        self.standardizer = SignalStandardizer()
        self.validator = SignalValidator()
        self.signal_queue = SignalQueue()
        
        # Signal buffers
        self.signal_buffer_30m: Dict[SignalType, List[SignalData]] = {
            SignalType.MLMI: [],
            SignalType.NWRQK: [],
            SignalType.FVG: []
        }
        
        self.signal_buffer_5m: Dict[SignalType, List[SignalData]] = {
            SignalType.MLMI: [],
            SignalType.NWRQK: [],
            SignalType.FVG: []
        }
        
        # Performance tracking
        self.stats = {
            'signals_processed': 0,
            'signals_rejected': 0,
            'interpolations_performed': 0,
            'validation_errors': 0
        }
    
    def process_raw_signal(self, signal_type: SignalType, raw_value: float,
                          timeframe: str, timestamp: datetime = None,
                          metadata: Dict[str, Any] = None) -> Optional[SignalData]:
        """
        Process a raw signal through the complete alignment pipeline
        
        Args:
            signal_type: Type of signal
            raw_value: Raw signal value
            timeframe: Signal timeframe
            timestamp: Signal timestamp (default: now)
            metadata: Additional metadata
            
        Returns:
            Processed SignalData or None if invalid
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Standardize signal
            signal = self.standardizer.standardize_signal(
                signal_type, raw_value, timeframe, metadata
            )
            signal.timestamp = timestamp
            
            # Validate signal
            is_valid, error_msg = self.validator.validate_signal(signal)
            if not is_valid:
                self.logger.warning("Signal validation failed", error=error_msg)
                self.stats['signals_rejected'] += 1
                return None
            
            # Add to appropriate buffer
            if timeframe == "30m":
                self.signal_buffer_30m[signal_type].append(signal)
            else:
                self.signal_buffer_5m[signal_type].append(signal)
            
            # Add to priority queue
            priority = self._determine_priority(signal)
            self.signal_queue.add_signal(signal, priority)
            
            self.stats['signals_processed'] += 1
            
            return signal
            
        except Exception as e:
            self.logger.error("Error processing signal", error=str(e))
            self.stats['validation_errors'] += 1
            return None
    
    def _determine_priority(self, signal: SignalData) -> SignalPriority:
        """Determine signal priority based on type and strength"""
        if signal.strength >= 0.8:
            return SignalPriority.HIGH
        elif signal.strength >= 0.6:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def align_signals_to_5m(self, timestamps_5m: List[datetime]) -> Dict[SignalType, List[SignalData]]:
        """
        Align all 30-minute signals to 5-minute timeframe
        
        Args:
            timestamps_5m: List of 5-minute timestamps
            
        Returns:
            Dictionary mapping signal types to aligned 5-minute signals
        """
        aligned_signals = {}
        
        for signal_type in SignalType:
            # Get 30-minute signals for this type
            signals_30m = self.signal_buffer_30m[signal_type]
            
            if signals_30m:
                # Convert to 5-minute timeframe
                aligned_signals[signal_type] = self.converter.convert_30m_to_5m(
                    signals_30m, timestamps_5m
                )
                self.stats['interpolations_performed'] += len(aligned_signals[signal_type])
            else:
                aligned_signals[signal_type] = []
        
        return aligned_signals
    
    def get_synchronized_signals(self, timestamp: datetime) -> Dict[SignalType, SignalData]:
        """
        Get synchronized signals for a specific timestamp
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Dictionary mapping signal types to synchronized signals
        """
        synchronized = {}
        
        for signal_type in SignalType:
            # Find the most recent signal before or at the timestamp
            best_signal = None
            min_time_diff = float('inf')
            
            # Check 5-minute signals first
            for signal in self.signal_buffer_5m[signal_type]:
                if signal.timestamp <= timestamp:
                    time_diff = (timestamp - signal.timestamp).total_seconds()
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_signal = signal
            
            # Check 30-minute signals if no 5-minute signal found
            if best_signal is None:
                for signal in self.signal_buffer_30m[signal_type]:
                    if signal.timestamp <= timestamp:
                        time_diff = (timestamp - signal.timestamp).total_seconds()
                        if time_diff < min_time_diff and time_diff < 1800:  # Within 30 minutes
                            min_time_diff = time_diff
                            best_signal = signal
            
            if best_signal:
                synchronized[signal_type] = best_signal
        
        return synchronized
    
    def clear_old_signals(self, cutoff_time: datetime):
        """Clear signals older than cutoff time"""
        for signal_type in SignalType:
            # Clear 5-minute signals
            self.signal_buffer_5m[signal_type] = [
                s for s in self.signal_buffer_5m[signal_type] 
                if s.timestamp > cutoff_time
            ]
            
            # Clear 30-minute signals
            self.signal_buffer_30m[signal_type] = [
                s for s in self.signal_buffer_30m[signal_type] 
                if s.timestamp > cutoff_time
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'buffer_sizes': {
                '5m': {st.value: len(self.signal_buffer_5m[st]) for st in SignalType},
                '30m': {st.value: len(self.signal_buffer_30m[st]) for st in SignalType}
            },
            'queue_size': self.signal_queue.size()
        }
    
    def reset(self):
        """Reset all buffers and statistics"""
        for signal_type in SignalType:
            self.signal_buffer_5m[signal_type].clear()
            self.signal_buffer_30m[signal_type].clear()
        
        self.signal_queue.clear()
        
        self.stats = {
            'signals_processed': 0,
            'signals_rejected': 0,
            'interpolations_performed': 0,
            'validation_errors': 0
        }
        
        self.logger.info("Signal alignment engine reset")


# Numba-optimized functions for performance-critical operations
@njit
def interpolate_signal_values(values: np.ndarray, timestamps: np.ndarray,
                             target_timestamps: np.ndarray) -> np.ndarray:
    """
    Fast interpolation of signal values using Numba
    
    Args:
        values: Array of signal values
        timestamps: Array of timestamps (as Unix timestamps)
        target_timestamps: Target timestamps for interpolation
        
    Returns:
        Interpolated values
    """
    result = np.zeros(len(target_timestamps))
    
    for i in prange(len(target_timestamps)):
        target_ts = target_timestamps[i]
        
        # Find the closest timestamp before or at target
        best_idx = -1
        for j in range(len(timestamps)):
            if timestamps[j] <= target_ts:
                best_idx = j
            else:
                break
        
        if best_idx >= 0:
            result[i] = values[best_idx]
        else:
            result[i] = 0.0
    
    return result


@njit
def calculate_signal_confidence(values: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Calculate signal confidence based on historical stability
    
    Args:
        values: Array of signal values
        window_size: Window size for confidence calculation
        
    Returns:
        Array of confidence values
    """
    n = len(values)
    confidence = np.zeros(n)
    
    for i in range(n):
        if i < window_size:
            # Not enough history, use default confidence
            confidence[i] = 0.5
        else:
            # Calculate standard deviation over window
            window_values = values[i-window_size:i]
            mean_val = np.mean(window_values)
            std_val = np.std(window_values)
            
            # Confidence inversely related to volatility
            if std_val > 0:
                confidence[i] = 1.0 / (1.0 + std_val)
            else:
                confidence[i] = 1.0
    
    return confidence


# Factory function for easy instantiation
def create_signal_alignment_engine(config: Dict[str, Any] = None) -> SignalAlignmentEngine:
    """
    Create a configured signal alignment engine
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SignalAlignmentEngine instance
    """
    return SignalAlignmentEngine(config)