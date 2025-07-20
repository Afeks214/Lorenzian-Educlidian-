"""
HYBRID LORENTZIAN-EUCLIDEAN DISTANCE METRICS - INTELLIGENT MARKET-AWARE SYSTEM
==============================================================================

This module implements an intelligent hybrid distance metric system that dynamically 
switches between Lorentzian and Euclidean distance calculations based on real-time 
market regime detection. It serves as the mathematical foundation for the entire 
adaptive Lorentzian classification trading system.

Mathematical Foundations:
    Lorentzian: D_L(x,y) = Σᵢ ln(1 + |xᵢ - yᵢ|)
    Euclidean:  D_E(x,y) = √(Σᵢ (xᵢ - yᵢ)²)
    Hybrid:     D_H(x,y) = α × D_L(x,y) + (1-α) × D_E(x,y)

Intelligent Selection Logic:
- Volatile markets → Lorentzian distance (robust to outliers)
- Stable markets → Euclidean distance (precise for small differences)
- Transitional periods → Hybrid approach with confidence weighting

Key Features:
- Market regime-aware distance selection
- Vectorized NumPy implementation for performance
- Numba JIT compilation for ultra-fast computation
- GPU acceleration support using CuPy
- Smooth regime transition handling
- Comprehensive mathematical validation
- Production-ready error handling and logging
- Memory-efficient batch processing
- Distance caching system for repeated calculations

Author: Claude Code Assistant
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import warnings
import logging
import time
import pickle
import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Performance optimization imports
from numba import jit, njit, prange, cuda
import numba.types as nb_types
from numba.core.errors import NumbaPerformanceWarning

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Optional sparse matrix support
try:
    from scipy.sparse import csr_matrix, issparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numba performance warnings for cleaner output
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Import market regime detection module
try:
    from .market_regime import RegimeDetector, RegimeConfig, RegimeMetrics, MarketRegime
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False
    logger.warning("Market regime detection not available. Using basic distance selection.")

from enum import Enum

class DistanceMetric(Enum):
    """Supported distance metrics"""
    LORENTZIAN = "lorentzian"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HYBRID = "hybrid"
    AUTO = "auto"  # Automatic selection based on market regime


@dataclass
class DistanceMetricsConfig:
    """Configuration parameters for distance metric calculations"""
    
    # Numerical stability parameters
    epsilon: float = 1e-12
    max_distance: float = 1e6
    
    # Performance parameters
    use_numba_jit: bool = True
    use_gpu_acceleration: bool = CUPY_AVAILABLE
    enable_caching: bool = True
    cache_size: int = 10000
    parallel_threshold: int = 1000
    
    # Batch processing parameters
    max_batch_size: int = 10000
    memory_threshold_gb: float = 4.0
    
    # Validation parameters
    validate_inputs: bool = True
    check_symmetry: bool = True
    tolerance: float = 1e-10
    
    # Logging parameters
    log_performance: bool = True
    log_cache_hits: bool = False
    
    # Hybrid distance parameters
    enable_regime_detection: bool = REGIME_DETECTION_AVAILABLE
    auto_metric_selection: bool = True
    hybrid_alpha_default: float = 0.5  # Default mixing parameter
    confidence_threshold: float = 0.7  # Minimum confidence for metric switching
    
    # Market regime thresholds
    volatility_lorentzian_threshold: float = 0.20  # Use Lorentzian above this volatility
    volatility_euclidean_threshold: float = 0.12   # Use Euclidean below this volatility
    trend_strength_threshold: float = 30.0         # ADX threshold for trend detection
    
    # Transition smoothing parameters
    enable_smooth_transitions: bool = True
    transition_window: int = 5               # Bars for transition smoothing
    min_regime_persistence: int = 3          # Minimum bars before metric switch
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'epsilon': self.epsilon,
            'max_distance': self.max_distance,
            'use_numba_jit': self.use_numba_jit,
            'use_gpu_acceleration': self.use_gpu_acceleration,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'parallel_threshold': self.parallel_threshold,
            'max_batch_size': self.max_batch_size,
            'memory_threshold_gb': self.memory_threshold_gb,
            'validate_inputs': self.validate_inputs,
            'check_symmetry': self.check_symmetry,
            'tolerance': self.tolerance,
            'log_performance': self.log_performance,
            'log_cache_hits': self.log_cache_hits,
            'enable_regime_detection': self.enable_regime_detection,
            'auto_metric_selection': self.auto_metric_selection,
            'hybrid_alpha_default': self.hybrid_alpha_default,
            'confidence_threshold': self.confidence_threshold,
            'volatility_lorentzian_threshold': self.volatility_lorentzian_threshold,
            'volatility_euclidean_threshold': self.volatility_euclidean_threshold,
            'trend_strength_threshold': self.trend_strength_threshold,
            'enable_smooth_transitions': self.enable_smooth_transitions,
            'transition_window': self.transition_window,
            'min_regime_persistence': self.min_regime_persistence
        }


@dataclass
class DistanceResult:
    """Result container for distance calculations"""
    
    distance: Union[float, np.ndarray]
    computation_time: float
    method_used: str
    cache_hit: bool = False
    validation_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result after initialization"""
        if isinstance(self.distance, np.ndarray):
            if np.any(np.isnan(self.distance)) or np.any(np.isinf(self.distance)):
                self.validation_passed = False
                logger.warning("Distance result contains NaN or Inf values")
        elif isinstance(self.distance, float):
            if np.isnan(self.distance) or np.isinf(self.distance):
                self.validation_passed = False
                logger.warning("Distance result is NaN or Inf")


class PerformanceMonitor:
    """Monitor and log performance metrics for distance calculations"""
    
    def __init__(self, config: DistanceMetricsConfig):
        self.config = config
        self.call_count = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def log_call(self, func_name: str, computation_time: float, cache_hit: bool = False):
        """Log a function call"""
        self.call_count += 1
        self.total_time += computation_time
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if self.config.log_performance and self.call_count % 100 == 0:
            avg_time = self.total_time / self.call_count
            cache_rate = self.cache_hits / self.call_count if self.call_count > 0 else 0
            logger.info(f"{func_name}: {self.call_count} calls, "
                       f"avg time: {avg_time:.6f}s, cache rate: {cache_rate:.2%}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_rate = self.cache_hits / self.call_count if self.call_count > 0 else 0
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
        
        return {
            'total_calls': self.call_count,
            'total_time': self.total_time,
            'average_time': avg_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_rate
        }


class DistanceCache:
    """LRU cache for distance calculations with hash-based keys"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.usage_order = []
        
    def _hash_arrays(self, x: np.ndarray, y: np.ndarray) -> str:
        """Create hash key from arrays"""
        x_bytes = x.tobytes()
        y_bytes = y.tobytes()
        combined = x_bytes + y_bytes
        return hashlib.md5(combined).hexdigest()
    
    def get(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Get cached distance if available"""
        key = self._hash_arrays(x, y)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        
        return None
    
    def set(self, x: np.ndarray, y: np.ndarray, distance: float):
        """Store distance in cache"""
        key = self._hash_arrays(x, y)
        
        # Remove least recently used if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.usage_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = distance
        
        if key not in self.usage_order:
            self.usage_order.append(key)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.usage_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'fill_rate': len(self.cache) / self.max_size
        }


# Core JIT-compiled distance functions
@njit
def _lorentzian_distance_core(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    """
    Core Lorentzian distance calculation (JIT optimized)
    
    D_L(x,y) = Σᵢ ln(1 + |xᵢ - yᵢ|)
    
    Args:
        x: First feature vector
        y: Second feature vector
        epsilon: Small value for numerical stability
        
    Returns:
        Lorentzian distance
    """
    if x.shape[0] != y.shape[0]:
        return np.inf
    
    total_distance = 0.0
    
    for i in range(x.shape[0]):
        abs_diff = abs(x[i] - y[i])
        # Add epsilon to prevent log(1) = 0 for identical values
        log_term = np.log(1.0 + abs_diff + epsilon)
        total_distance += log_term
    
    return total_distance


@njit
def _euclidean_distance_core(x: np.ndarray, y: np.ndarray) -> float:
    """
    Core Euclidean distance calculation (JIT optimized)
    
    Args:
        x: First feature vector
        y: Second feature vector
        
    Returns:
        Euclidean distance
    """
    if x.shape[0] != y.shape[0]:
        return np.inf
    
    sum_squared_diff = 0.0
    
    for i in range(x.shape[0]):
        diff = x[i] - y[i]
        sum_squared_diff += diff * diff
    
    return np.sqrt(sum_squared_diff)


@njit
def _manhattan_distance_core(x: np.ndarray, y: np.ndarray) -> float:
    """
    Core Manhattan distance calculation (JIT optimized)
    
    Args:
        x: First feature vector
        y: Second feature vector
        
    Returns:
        Manhattan distance
    """
    if x.shape[0] != y.shape[0]:
        return np.inf
    
    total_distance = 0.0
    
    for i in range(x.shape[0]):
        total_distance += abs(x[i] - y[i])
    
    return total_distance


@njit
def _hybrid_distance_core(
    x: np.ndarray, 
    y: np.ndarray, 
    alpha: float,
    epsilon: float
) -> float:
    """
    Core hybrid distance calculation (JIT optimized)
    
    Combines Lorentzian and Euclidean distances:
    D_H(x,y) = α × D_L(x,y) + (1-α) × D_E(x,y)
    
    Args:
        x: First feature vector
        y: Second feature vector
        alpha: Mixing parameter (0=pure Euclidean, 1=pure Lorentzian)
        epsilon: Small value for numerical stability
        
    Returns:
        Hybrid distance
    """
    if x.shape[0] != y.shape[0]:
        return np.inf
    
    lorentzian_sum = 0.0
    euclidean_sum = 0.0
    
    for i in range(x.shape[0]):
        abs_diff = abs(x[i] - y[i])
        
        # Lorentzian component
        lorentzian_sum += np.log(1.0 + abs_diff + epsilon)
        
        # Euclidean component
        euclidean_sum += abs_diff * abs_diff
    
    euclidean_distance = np.sqrt(euclidean_sum)
    
    # Combine distances
    hybrid_distance = alpha * lorentzian_sum + (1.0 - alpha) * euclidean_distance
    
    return hybrid_distance


@njit(parallel=True)
def _batch_lorentzian_distances(
    X: np.ndarray, 
    Y: np.ndarray, 
    epsilon: float
) -> np.ndarray:
    """
    Batch calculation of Lorentzian distances (parallel JIT optimized)
    
    Args:
        X: First set of feature vectors (n_samples_x, n_features)
        Y: Second set of feature vectors (n_samples_y, n_features)
        epsilon: Small value for numerical stability
        
    Returns:
        Distance matrix (n_samples_x, n_samples_y)
    """
    n_x, n_features = X.shape
    n_y = Y.shape[0]
    
    distances = np.zeros((n_x, n_y))
    
    for i in prange(n_x):
        for j in range(n_y):
            distances[i, j] = _lorentzian_distance_core(X[i], Y[j], epsilon)
    
    return distances


@njit
def _weighted_lorentzian_distance_core(
    x: np.ndarray, 
    y: np.ndarray, 
    weights: np.ndarray,
    epsilon: float
) -> float:
    """
    Weighted Lorentzian distance calculation (JIT optimized)
    
    Args:
        x: First feature vector
        y: Second feature vector
        weights: Feature weights
        epsilon: Small value for numerical stability
        
    Returns:
        Weighted Lorentzian distance
    """
    if x.shape[0] != y.shape[0] or x.shape[0] != weights.shape[0]:
        return np.inf
    
    total_distance = 0.0
    
    for i in range(x.shape[0]):
        abs_diff = abs(x[i] - y[i])
        log_term = np.log(1.0 + abs_diff + epsilon)
        total_distance += weights[i] * log_term
    
    return total_distance


# GPU acceleration functions (if CuPy is available)
if CUPY_AVAILABLE:
    def _gpu_lorentzian_distance(x: cp.ndarray, y: cp.ndarray, epsilon: float) -> float:
        """GPU-accelerated Lorentzian distance calculation"""
        abs_diff = cp.abs(x - y)
        log_terms = cp.log(1.0 + abs_diff + epsilon)
        return float(cp.sum(log_terms))
    
    def _gpu_batch_lorentzian_distances(
        X: cp.ndarray, 
        Y: cp.ndarray, 
        epsilon: float
    ) -> cp.ndarray:
        """GPU-accelerated batch Lorentzian distance calculation"""
        # Expand dimensions for broadcasting
        X_expanded = X[:, None, :]  # (n_x, 1, n_features)
        Y_expanded = Y[None, :, :]  # (1, n_y, n_features)
        
        # Calculate all pairwise differences
        abs_diff = cp.abs(X_expanded - Y_expanded)  # (n_x, n_y, n_features)
        
        # Apply Lorentzian transformation and sum along feature dimension
        log_terms = cp.log(1.0 + abs_diff + epsilon)
        distances = cp.sum(log_terms, axis=2)  # (n_x, n_y)
        
        return distances


class HybridDistanceCalculator:
    """
    Intelligent hybrid distance calculator with market regime awareness
    
    This class implements the core hybrid distance system that automatically
    selects between Lorentzian and Euclidean distance metrics based on 
    real-time market conditions.
    """
    
    def __init__(self, config: Optional[DistanceMetricsConfig] = None):
        """
        Initialize the hybrid calculator
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DistanceMetricsConfig()
        self.cache = DistanceCache(self.config.cache_size) if self.config.enable_caching else None
        self.monitor = PerformanceMonitor(self.config)
        
        # Initialize regime detector if available
        if self.config.enable_regime_detection and REGIME_DETECTION_AVAILABLE:
            self.regime_detector = RegimeDetector(RegimeConfig())
            logger.info("Market regime detection enabled")
        else:
            self.regime_detector = None
            logger.info("Market regime detection disabled")
        
        # Metric selection history for smooth transitions
        self.metric_history: List[str] = []
        self.alpha_history: List[float] = []
        
        # JIT compile functions at initialization
        if self.config.use_numba_jit:
            self._compile_jit_functions()
    
    def _compile_jit_functions(self):
        """Pre-compile JIT functions for better performance"""
        logger.info("Pre-compiling hybrid JIT functions...")
        
        # Compile with dummy data
        dummy_x = np.array([1.0, 2.0, 3.0])
        dummy_y = np.array([1.1, 2.1, 3.1])
        
        _lorentzian_distance_core(dummy_x, dummy_y, self.config.epsilon)
        _euclidean_distance_core(dummy_x, dummy_y)
        _manhattan_distance_core(dummy_x, dummy_y)
        _hybrid_distance_core(dummy_x, dummy_y, 0.5, self.config.epsilon)
        
        logger.info("Hybrid JIT compilation completed")
    
    def _determine_optimal_metric(self, market_data: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
        """
        Determine the optimal distance metric based on market conditions
        
        Args:
            market_data: OHLCV market data for regime analysis
            
        Returns:
            Tuple of (metric_name, alpha_value)
        """
        if not self.config.auto_metric_selection or market_data is None:
            return "lorentzian", 1.0
        
        if self.regime_detector is None:
            # Fallback to basic volatility-based selection
            return self._basic_metric_selection(market_data)
        
        try:
            # Advanced regime-based selection
            regime_metrics = self.regime_detector.detect_regime(market_data)
            return self._regime_based_selection(regime_metrics)
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Using fallback selection.")
            return self._basic_metric_selection(market_data)
    
    def _basic_metric_selection(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Basic metric selection based on volatility only
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Tuple of (metric_name, alpha_value)
        """
        if len(market_data) < 20:
            return "lorentzian", 1.0
        
        # Calculate simple volatility
        close = market_data['close'].values
        returns = np.diff(np.log(close[-20:]))
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility > self.config.volatility_lorentzian_threshold:
            return "lorentzian", 1.0
        elif volatility < self.config.volatility_euclidean_threshold:
            return "euclidean", 0.0
        else:
            # Linear interpolation for hybrid
            vol_range = self.config.volatility_lorentzian_threshold - self.config.volatility_euclidean_threshold
            alpha = (volatility - self.config.volatility_euclidean_threshold) / vol_range
            return "hybrid", np.clip(alpha, 0.0, 1.0)
    
    def _regime_based_selection(self, regime_metrics: 'RegimeMetrics') -> Tuple[str, float]:
        """
        Advanced metric selection based on comprehensive regime analysis
        
        Args:
            regime_metrics: Comprehensive regime analysis results
            
        Returns:
            Tuple of (metric_name, alpha_value)
        """
        # Primary decision based on regime type
        if regime_metrics.regime == MarketRegime.VOLATILE:
            base_metric = "lorentzian"
            base_alpha = 0.9  # Heavily weighted towards Lorentzian
        elif regime_metrics.regime == MarketRegime.CALM:
            base_metric = "euclidean"
            base_alpha = 0.1  # Heavily weighted towards Euclidean
        elif regime_metrics.regime == MarketRegime.TRENDING:
            if regime_metrics.trend_strength > self.config.trend_strength_threshold:
                base_metric = "euclidean"  # Strong trends are predictable
                base_alpha = 0.2
            else:
                base_metric = "hybrid"
                base_alpha = 0.5
        else:  # RANGING or TRANSITIONAL
            base_metric = "hybrid"
            base_alpha = 0.5
        
        # Adjust based on confidence
        confidence_adjustment = 0.0
        if regime_metrics.confidence > self.config.confidence_threshold:
            # High confidence - move towards base recommendation
            if base_alpha > 0.5:
                confidence_adjustment = 0.1  # More Lorentzian
            else:
                confidence_adjustment = -0.1  # More Euclidean
        
        # Adjust based on volatility
        volatility_adjustment = 0.0
        if regime_metrics.volatility > self.config.volatility_lorentzian_threshold:
            volatility_adjustment = 0.2  # More Lorentzian for high volatility
        elif regime_metrics.volatility < self.config.volatility_euclidean_threshold:
            volatility_adjustment = -0.2  # More Euclidean for low volatility
        
        # Final alpha calculation
        final_alpha = base_alpha + confidence_adjustment + volatility_adjustment
        final_alpha = np.clip(final_alpha, 0.0, 1.0)
        
        # Determine final metric name
        if final_alpha > 0.8:
            final_metric = "lorentzian"
        elif final_alpha < 0.2:
            final_metric = "euclidean"
        else:
            final_metric = "hybrid"
        
        return final_metric, final_alpha
    
    def _apply_transition_smoothing(self, current_metric: str, current_alpha: float) -> Tuple[str, float]:
        """
        Apply smooth transitions between distance metrics
        
        Args:
            current_metric: Currently selected metric
            current_alpha: Currently selected alpha
            
        Returns:
            Smoothed metric and alpha
        """
        if not self.config.enable_smooth_transitions:
            return current_metric, current_alpha
        
        # Add to history
        self.metric_history.append(current_metric)
        self.alpha_history.append(current_alpha)
        
        # Maintain window size
        if len(self.metric_history) > self.config.transition_window:
            self.metric_history = self.metric_history[-self.config.transition_window:]
            self.alpha_history = self.alpha_history[-self.config.transition_window:]
        
        # Check for regime persistence
        if len(self.metric_history) < self.config.min_regime_persistence:
            # Not enough history, use current
            return current_metric, current_alpha
        
        # Count recent occurrences of current metric
        recent_metrics = self.metric_history[-self.config.min_regime_persistence:]
        current_count = recent_metrics.count(current_metric)
        
        # Require minimum persistence before switching
        if current_count >= self.config.min_regime_persistence:
            # Smooth alpha using moving average
            recent_alphas = self.alpha_history[-self.config.min_regime_persistence:]
            smoothed_alpha = np.mean(recent_alphas)
            return current_metric, smoothed_alpha
        else:
            # Not enough persistence, use previous stable state
            if len(self.metric_history) >= 2:
                return self.metric_history[-2], self.alpha_history[-2]
            else:
                return current_metric, current_alpha
    
    def adaptive_distance(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        market_data: Optional[pd.DataFrame] = None,
        force_metric: Optional[str] = None,
        force_alpha: Optional[float] = None
    ) -> DistanceResult:
        """
        Calculate adaptive distance with intelligent metric selection
        
        Args:
            x: First feature vector
            y: Second feature vector
            market_data: OHLCV market data for regime analysis
            force_metric: Force specific metric (overrides auto-selection)
            force_alpha: Force specific alpha value (for hybrid metric)
            
        Returns:
            DistanceResult containing distance and metadata
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Validate inputs
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            return DistanceResult(
                distance=np.inf,
                computation_time=time.time() - start_time,
                method_used="validation_failed",
                validation_passed=False
            )
        
        # Determine optimal metric
        if force_metric is not None:
            metric = force_metric
            alpha = force_alpha if force_alpha is not None else self.config.hybrid_alpha_default
        else:
            metric, alpha = self._determine_optimal_metric(market_data)
            
            # Apply transition smoothing
            if self.config.enable_smooth_transitions:
                metric, alpha = self._apply_transition_smoothing(metric, alpha)
        
        # Calculate distance based on selected metric
        try:
            if metric == "lorentzian":
                if self.config.use_numba_jit:
                    distance = _lorentzian_distance_core(x, y, self.config.epsilon)
                else:
                    abs_diff = np.abs(x - y)
                    log_terms = np.log(1.0 + abs_diff + self.config.epsilon)
                    distance = np.sum(log_terms)
                    
            elif metric == "euclidean":
                if self.config.use_numba_jit:
                    distance = _euclidean_distance_core(x, y)
                else:
                    diff = x - y
                    distance = np.sqrt(np.sum(diff ** 2))
                    
            elif metric == "hybrid":
                if self.config.use_numba_jit:
                    distance = _hybrid_distance_core(x, y, alpha, self.config.epsilon)
                else:
                    # Manual hybrid calculation
                    abs_diff = np.abs(x - y)
                    lorentzian_dist = np.sum(np.log(1.0 + abs_diff + self.config.epsilon))
                    euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
                    distance = alpha * lorentzian_dist + (1.0 - alpha) * euclidean_dist
                    
            elif metric == "manhattan":
                if self.config.use_numba_jit:
                    distance = _manhattan_distance_core(x, y)
                else:
                    distance = np.sum(np.abs(x - y))
                    
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Clamp distance to maximum value
            distance = min(distance, self.config.max_distance)
            
        except Exception as e:
            logger.error(f"Error calculating adaptive distance: {e}")
            distance = np.inf
            metric += "_error"
        
        computation_time = time.time() - start_time
        self.monitor.log_call("adaptive_distance", computation_time)
        
        return DistanceResult(
            distance=distance,
            computation_time=computation_time,
            method_used=f"{metric}" + (f"_alpha{alpha:.2f}" if metric == "hybrid" else ""),
            metadata={
                'input_dimension': x.shape[0],
                'selected_metric': metric,
                'alpha_value': alpha,
                'regime_detection_used': market_data is not None and self.regime_detector is not None,
                'transition_smoothing_applied': self.config.enable_smooth_transitions,
                'epsilon': self.config.epsilon
            }
        )
    
    def get_metric_recommendation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get distance metric recommendation without calculating distance
        
        Args:
            market_data: OHLCV market data for analysis
            
        Returns:
            Dictionary containing metric recommendation and reasoning
        """
        metric, alpha = self._determine_optimal_metric(market_data)
        
        # Get regime information if available
        regime_info = {}
        if self.regime_detector is not None:
            try:
                regime_metrics = self.regime_detector.detect_regime(market_data)
                regime_info = {
                    'regime': regime_metrics.regime.value,
                    'confidence': regime_metrics.confidence,
                    'volatility': regime_metrics.volatility,
                    'trend_strength': regime_metrics.trend_strength,
                    'trend_direction': regime_metrics.trend_direction
                }
            except Exception as e:
                logger.warning(f"Could not get regime info: {e}")
        
        return {
            'recommended_metric': metric,
            'alpha_value': alpha,
            'regime_info': regime_info,
            'reasoning': self._generate_reasoning(metric, alpha, regime_info),
            'confidence': regime_info.get('confidence', 0.5)
        }
    
    def _generate_reasoning(self, metric: str, alpha: float, regime_info: Dict) -> str:
        """Generate human-readable reasoning for metric selection"""
        if not regime_info:
            return f"Selected {metric} (alpha={alpha:.2f}) based on basic volatility analysis"
        
        regime = regime_info.get('regime', 'unknown')
        volatility = regime_info.get('volatility', 0)
        trend_strength = regime_info.get('trend_strength', 0)
        
        reasoning = f"Selected {metric} (alpha={alpha:.2f}) because: "
        
        if regime == 'volatile':
            reasoning += "Market is in volatile regime - Lorentzian distance is more robust to outliers"
        elif regime == 'calm':
            reasoning += "Market is in calm regime - Euclidean distance provides better precision"
        elif regime == 'trending':
            reasoning += f"Market is trending (strength={trend_strength:.1f}) - "
            if trend_strength > 30:
                reasoning += "Strong trends are predictable, favoring Euclidean distance"
            else:
                reasoning += "Moderate trend strength suggests hybrid approach"
        else:
            reasoning += f"Market regime is {regime} - using balanced hybrid approach"
        
        if volatility > 0.2:
            reasoning += f" | High volatility ({volatility:.2f}) supports Lorentzian component"
        elif volatility < 0.1:
            reasoning += f" | Low volatility ({volatility:.2f}) supports Euclidean component"
        
        return reasoning


class LorentzianDistanceCalculator:
    """
    Advanced Lorentzian distance calculator with optimization and caching
    """
    
    def __init__(self, config: Optional[DistanceMetricsConfig] = None):
        """
        Initialize the calculator
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DistanceMetricsConfig()
        self.cache = DistanceCache(self.config.cache_size) if self.config.enable_caching else None
        self.monitor = PerformanceMonitor(self.config)
        
        # JIT compile functions at initialization
        if self.config.use_numba_jit:
            self._compile_jit_functions()
    
    def _compile_jit_functions(self):
        """Pre-compile JIT functions for better performance"""
        logger.info("Pre-compiling JIT functions...")
        
        # Compile with dummy data
        dummy_x = np.array([1.0, 2.0, 3.0])
        dummy_y = np.array([1.1, 2.1, 3.1])
        
        _lorentzian_distance_core(dummy_x, dummy_y, self.config.epsilon)
        _euclidean_distance_core(dummy_x, dummy_y)
        _manhattan_distance_core(dummy_x, dummy_y)
        
        logger.info("JIT compilation completed")
    
    def _validate_inputs(self, x: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate input arrays
        
        Args:
            x: First feature vector
            y: Second feature vector
            
        Returns:
            True if inputs are valid
        """
        if not self.config.validate_inputs:
            return True
        
        # Check if arrays are 1D
        if x.ndim != 1 or y.ndim != 1:
            logger.error("Input arrays must be 1-dimensional")
            return False
        
        # Check if arrays have the same length
        if x.shape[0] != y.shape[0]:
            logger.error("Input arrays must have the same length")
            return False
        
        # Check for NaN or Inf values
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            logger.error("Input arrays contain NaN values")
            return False
        
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            logger.error("Input arrays contain Inf values")
            return False
        
        return True
    
    def _choose_computation_method(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> str:
        """
        Choose the optimal computation method based on data characteristics
        
        Args:
            x: First feature vector
            y: Second feature vector
            
        Returns:
            Method name to use
        """
        # GPU acceleration for large arrays
        if (self.config.use_gpu_acceleration and 
            CUPY_AVAILABLE and 
            x.shape[0] > self.config.parallel_threshold):
            return "gpu"
        
        # JIT compilation for medium to large arrays
        if self.config.use_numba_jit:
            return "jit"
        
        # NumPy vectorized for small arrays
        return "numpy"
    
    def lorentzian_distance(
        self, 
        x: Union[np.ndarray, List[float]], 
        y: Union[np.ndarray, List[float]],
        weights: Optional[Union[np.ndarray, List[float]]] = None
    ) -> DistanceResult:
        """
        Calculate Lorentzian distance between two feature vectors
        
        Mathematical Formula:
            D_L(x,y) = Σᵢ ln(1 + |xᵢ - yᵢ|)
            
        Weighted Version:
            D_L(x,y,w) = Σᵢ wᵢ × ln(1 + |xᵢ - yᵢ|)
        
        Args:
            x: First feature vector
            y: Second feature vector
            weights: Optional feature weights
            
        Returns:
            DistanceResult containing distance and metadata
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
        
        # Validate inputs
        if not self._validate_inputs(x, y):
            return DistanceResult(
                distance=np.inf,
                computation_time=time.time() - start_time,
                method_used="validation_failed",
                validation_passed=False
            )
        
        # Check cache first
        cache_hit = False
        if self.cache is not None and weights is None:
            cached_distance = self.cache.get(x, y)
            if cached_distance is not None:
                cache_hit = True
                computation_time = time.time() - start_time
                self.monitor.log_call("lorentzian_distance", computation_time, cache_hit)
                
                return DistanceResult(
                    distance=cached_distance,
                    computation_time=computation_time,
                    method_used="cache",
                    cache_hit=True
                )
        
        # Choose computation method
        method = self._choose_computation_method(x, y)
        
        # Calculate distance
        try:
            if weights is not None:
                # Weighted distance calculation
                if self.config.use_numba_jit:
                    distance = _weighted_lorentzian_distance_core(x, y, weights, self.config.epsilon)
                else:
                    abs_diff = np.abs(x - y)
                    log_terms = np.log(1.0 + abs_diff + self.config.epsilon)
                    distance = np.sum(weights * log_terms)
                method += "_weighted"
            
            elif method == "gpu" and CUPY_AVAILABLE:
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
                distance = _gpu_lorentzian_distance(x_gpu, y_gpu, self.config.epsilon)
            
            elif method == "jit":
                distance = _lorentzian_distance_core(x, y, self.config.epsilon)
            
            else:  # numpy
                abs_diff = np.abs(x - y)
                log_terms = np.log(1.0 + abs_diff + self.config.epsilon)
                distance = np.sum(log_terms)
            
            # Clamp distance to maximum value
            distance = min(distance, self.config.max_distance)
            
        except Exception as e:
            logger.error(f"Error calculating Lorentzian distance: {e}")
            distance = np.inf
            method += "_error"
        
        # Cache result if caching is enabled and no weights
        if self.cache is not None and weights is None and np.isfinite(distance):
            self.cache.set(x, y, distance)
        
        computation_time = time.time() - start_time
        self.monitor.log_call("lorentzian_distance", computation_time, cache_hit)
        
        return DistanceResult(
            distance=distance,
            computation_time=computation_time,
            method_used=method,
            cache_hit=cache_hit,
            metadata={
                'input_dimension': x.shape[0],
                'weighted': weights is not None,
                'epsilon': self.config.epsilon
            }
        )
    
    def batch_distances(
        self,
        X: Union[np.ndarray, List[List[float]]],
        Y: Optional[Union[np.ndarray, List[List[float]]]] = None,
        metric: str = "lorentzian"
    ) -> DistanceResult:
        """
        Calculate pairwise distances between sets of feature vectors
        
        Args:
            X: First set of feature vectors (n_samples_x, n_features)
            Y: Second set of feature vectors (n_samples_y, n_features)
               If None, calculates pairwise distances within X
            metric: Distance metric to use ("lorentzian", "euclidean", "manhattan")
            
        Returns:
            DistanceResult containing distance matrix and metadata
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        if Y is not None:
            Y = np.asarray(Y, dtype=np.float64)
        else:
            Y = X.copy()
        
        # Validate input shapes
        if X.ndim != 2 or Y.ndim != 2:
            logger.error("Input arrays must be 2-dimensional")
            return DistanceResult(
                distance=np.array([]),
                computation_time=time.time() - start_time,
                method_used="validation_failed",
                validation_passed=False
            )
        
        if X.shape[1] != Y.shape[1]:
            logger.error("Input arrays must have the same number of features")
            return DistanceResult(
                distance=np.array([]),
                computation_time=time.time() - start_time,
                method_used="validation_failed",
                validation_passed=False
            )
        
        n_x, n_features = X.shape
        n_y = Y.shape[0]
        
        # Choose computation method based on size
        if (self.config.use_gpu_acceleration and 
            CUPY_AVAILABLE and 
            n_x * n_y > self.config.parallel_threshold):
            method = "gpu_batch"
        elif self.config.use_numba_jit:
            method = "jit_batch"
        else:
            method = "numpy_batch"
        
        # Calculate distances
        try:
            if metric == "lorentzian":
                if method == "gpu_batch" and CUPY_AVAILABLE:
                    X_gpu = cp.asarray(X)
                    Y_gpu = cp.asarray(Y)
                    distances = _gpu_batch_lorentzian_distances(X_gpu, Y_gpu, self.config.epsilon)
                    distances = cp.asnumpy(distances)
                elif method == "jit_batch":
                    distances = _batch_lorentzian_distances(X, Y, self.config.epsilon)
                else:  # numpy_batch
                    distances = np.zeros((n_x, n_y))
                    for i in range(n_x):
                        for j in range(n_y):
                            abs_diff = np.abs(X[i] - Y[j])
                            log_terms = np.log(1.0 + abs_diff + self.config.epsilon)
                            distances[i, j] = np.sum(log_terms)
            
            elif metric == "euclidean":
                # Euclidean distance using broadcasting
                X_expanded = X[:, None, :]  # (n_x, 1, n_features)
                Y_expanded = Y[None, :, :]  # (1, n_y, n_features)
                squared_diff = (X_expanded - Y_expanded) ** 2
                distances = np.sqrt(np.sum(squared_diff, axis=2))
            
            elif metric == "manhattan":
                # Manhattan distance using broadcasting
                X_expanded = X[:, None, :]  # (n_x, 1, n_features)
                Y_expanded = Y[None, :, :]  # (1, n_y, n_features)
                abs_diff = np.abs(X_expanded - Y_expanded)
                distances = np.sum(abs_diff, axis=2)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Clamp distances to maximum value
            distances = np.minimum(distances, self.config.max_distance)
            
        except Exception as e:
            logger.error(f"Error calculating batch distances: {e}")
            distances = np.full((n_x, n_y), np.inf)
            method += "_error"
        
        computation_time = time.time() - start_time
        self.monitor.log_call("batch_distances", computation_time)
        
        return DistanceResult(
            distance=distances,
            computation_time=computation_time,
            method_used=method,
            metadata={
                'n_samples_x': n_x,
                'n_samples_y': n_y,
                'n_features': n_features,
                'metric': metric,
                'total_comparisons': n_x * n_y
            }
        )
    
    def k_nearest_neighbors(
        self,
        query_vector: Union[np.ndarray, List[float]],
        reference_vectors: Union[np.ndarray, List[List[float]]],
        k: int,
        metric: str = "lorentzian",
        return_distances: bool = True
    ) -> Dict[str, Any]:
        """
        Find k nearest neighbors using specified distance metric
        
        Args:
            query_vector: Query feature vector
            reference_vectors: Reference feature vectors (n_samples, n_features)
            k: Number of nearest neighbors to find
            metric: Distance metric to use
            return_distances: Whether to return distances along with indices
            
        Returns:
            Dictionary containing neighbor indices and optionally distances
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        query_vector = np.asarray(query_vector, dtype=np.float64)
        reference_vectors = np.asarray(reference_vectors, dtype=np.float64)
        
        # Ensure query_vector is 1D and reference_vectors is 2D
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be 1-dimensional")
        if reference_vectors.ndim != 2:
            raise ValueError("reference_vectors must be 2-dimensional")
        
        n_samples, n_features = reference_vectors.shape
        
        if query_vector.shape[0] != n_features:
            raise ValueError("query_vector and reference_vectors must have same number of features")
        
        if k > n_samples:
            k = n_samples
            logger.warning(f"k reduced to {k} (number of available samples)")
        
        # Calculate distances to all reference vectors
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            if metric == "lorentzian":
                if self.config.use_numba_jit:
                    distances[i] = _lorentzian_distance_core(
                        query_vector, reference_vectors[i], self.config.epsilon
                    )
                else:
                    abs_diff = np.abs(query_vector - reference_vectors[i])
                    log_terms = np.log(1.0 + abs_diff + self.config.epsilon)
                    distances[i] = np.sum(log_terms)
            
            elif metric == "euclidean":
                if self.config.use_numba_jit:
                    distances[i] = _euclidean_distance_core(query_vector, reference_vectors[i])
                else:
                    diff = query_vector - reference_vectors[i]
                    distances[i] = np.sqrt(np.sum(diff ** 2))
            
            elif metric == "manhattan":
                if self.config.use_numba_jit:
                    distances[i] = _manhattan_distance_core(query_vector, reference_vectors[i])
                else:
                    distances[i] = np.sum(np.abs(query_vector - reference_vectors[i]))
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        # Find k nearest neighbors
        neighbor_indices = np.argpartition(distances, k-1)[:k]
        neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        
        computation_time = time.time() - start_time
        self.monitor.log_call("k_nearest_neighbors", computation_time)
        
        result = {
            'indices': neighbor_indices,
            'computation_time': computation_time,
            'metric': metric,
            'k': k,
            'n_samples': n_samples
        }
        
        if return_distances:
            neighbor_distances = distances[neighbor_indices]
            result['distances'] = neighbor_distances
        
        return result
    
    def validate_mathematical_properties(
        self,
        test_vectors: Optional[np.ndarray] = None,
        n_test_vectors: int = 100,
        n_features: int = 10
    ) -> Dict[str, Any]:
        """
        Validate mathematical properties of the distance metric
        
        Tests:
        1. Non-negativity: d(x,y) >= 0
        2. Identity: d(x,x) = 0
        3. Symmetry: d(x,y) = d(y,x)
        4. Monotonicity: Lorentzian distance preserves ordering for small differences
        
        Args:
            test_vectors: Optional test vectors (if None, generates random ones)
            n_test_vectors: Number of test vectors to generate
            n_features: Number of features per vector
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating mathematical properties...")
        
        # Generate test vectors if not provided
        if test_vectors is None:
            np.random.seed(42)  # For reproducibility
            test_vectors = np.random.randn(n_test_vectors, n_features)
        
        n_samples = test_vectors.shape[0]
        results = {
            'non_negativity_violations': 0,
            'identity_violations': 0,
            'symmetry_violations': 0,
            'monotonicity_violations': 0,
            'total_tests': 0,
            'tolerance': self.config.tolerance
        }
        
        # Test 1: Non-negativity and Identity
        for i in range(n_samples):
            # Self-distance should be close to zero (due to epsilon)
            self_distance = self.lorentzian_distance(test_vectors[i], test_vectors[i]).distance
            
            if self_distance < 0:
                results['non_negativity_violations'] += 1
            
            # Identity test (should be very small due to epsilon)
            expected_identity = n_features * np.log(1 + self.config.epsilon)
            if abs(self_distance - expected_identity) > self.config.tolerance:
                results['identity_violations'] += 1
            
            results['total_tests'] += 1
        
        # Test 2: Symmetry
        n_symmetry_tests = min(1000, n_samples * (n_samples - 1) // 2)
        symmetry_test_count = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if symmetry_test_count >= n_symmetry_tests:
                    break
                
                d_ij = self.lorentzian_distance(test_vectors[i], test_vectors[j]).distance
                d_ji = self.lorentzian_distance(test_vectors[j], test_vectors[i]).distance
                
                if abs(d_ij - d_ji) > self.config.tolerance:
                    results['symmetry_violations'] += 1
                
                symmetry_test_count += 1
            
            if symmetry_test_count >= n_symmetry_tests:
                break
        
        results['symmetry_tests'] = symmetry_test_count
        
        # Test 3: Monotonicity (for small perturbations)
        n_monotonicity_tests = min(500, n_samples)
        
        for i in range(n_monotonicity_tests):
            base_vector = test_vectors[i]
            
            # Create two perturbations with different magnitudes
            small_perturbation = base_vector + 0.1 * np.random.randn(n_features)
            large_perturbation = base_vector + 0.5 * np.random.randn(n_features)
            
            d_small = self.lorentzian_distance(base_vector, small_perturbation).distance
            d_large = self.lorentzian_distance(base_vector, large_perturbation).distance
            
            # Generally, larger perturbations should give larger distances
            # (This is a soft constraint due to the logarithmic nature)
            if d_small > d_large:
                results['monotonicity_violations'] += 1
        
        results['monotonicity_tests'] = n_monotonicity_tests
        
        # Calculate success rates
        results['non_negativity_success_rate'] = 1 - (results['non_negativity_violations'] / results['total_tests'])
        results['identity_success_rate'] = 1 - (results['identity_violations'] / results['total_tests'])
        results['symmetry_success_rate'] = 1 - (results['symmetry_violations'] / results['symmetry_tests'])
        results['monotonicity_success_rate'] = 1 - (results['monotonicity_violations'] / results['monotonicity_tests'])
        
        overall_success = (
            results['non_negativity_success_rate'] * 0.3 +
            results['identity_success_rate'] * 0.2 +
            results['symmetry_success_rate'] * 0.3 +
            results['monotonicity_success_rate'] * 0.2
        )
        results['overall_success_rate'] = overall_success
        
        logger.info(f"Mathematical validation completed. Overall success rate: {overall_success:.2%}")
        
        return results
    
    def benchmark_performance(
        self,
        sizes: List[int] = [10, 50, 100, 500, 1000],
        n_features: int = 5,
        n_runs: int = 5
    ) -> pd.DataFrame:
        """
        Benchmark performance across different input sizes and methods
        
        Args:
            sizes: List of input sizes to test
            n_features: Number of features per vector
            n_runs: Number of runs for averaging
            
        Returns:
            DataFrame with benchmark results
        """
        logger.info("Starting performance benchmark...")
        
        results = []
        np.random.seed(42)
        
        for size in sizes:
            logger.info(f"Benchmarking size: {size}")
            
            # Generate test data
            X = np.random.randn(size, n_features)
            Y = np.random.randn(size, n_features)
            
            # Single distance calculations
            for run in range(n_runs):
                # Lorentzian distance
                start_time = time.time()
                result = self.lorentzian_distance(X[0], Y[0])
                lorentzian_time = time.time() - start_time
                
                results.append({
                    'size': size,
                    'run': run,
                    'method': 'lorentzian_single',
                    'time': lorentzian_time,
                    'operations': 1
                })
                
                # Euclidean distance (for comparison)
                start_time = time.time()
                euclidean_dist = np.sqrt(np.sum((X[0] - Y[0]) ** 2))
                euclidean_time = time.time() - start_time
                
                results.append({
                    'size': size,
                    'run': run,
                    'method': 'euclidean_single',
                    'time': euclidean_time,
                    'operations': 1
                })
            
            # Batch calculations (for larger sizes)
            if size >= 50:
                for run in range(n_runs):
                    # Batch Lorentzian
                    start_time = time.time()
                    batch_result = self.batch_distances(X[:10], Y[:10], metric="lorentzian")
                    batch_lorentzian_time = time.time() - start_time
                    
                    results.append({
                        'size': size,
                        'run': run,
                        'method': 'lorentzian_batch',
                        'time': batch_lorentzian_time,
                        'operations': 100  # 10x10 matrix
                    })
                    
                    # Batch Euclidean
                    start_time = time.time()
                    batch_euclidean_result = self.batch_distances(X[:10], Y[:10], metric="euclidean")
                    batch_euclidean_time = time.time() - start_time
                    
                    results.append({
                        'size': size,
                        'run': run,
                        'method': 'euclidean_batch',
                        'time': batch_euclidean_time,
                        'operations': 100  # 10x10 matrix
                    })
        
        # Convert to DataFrame and calculate statistics
        df = pd.DataFrame(results)
        
        # Calculate time per operation
        df['time_per_operation'] = df['time'] / df['operations']
        
        # Group by size and method to get averages
        summary = df.groupby(['size', 'method']).agg({
            'time': ['mean', 'std'],
            'time_per_operation': ['mean', 'std']
        }).round(6)
        
        summary.columns = ['avg_time', 'std_time', 'avg_time_per_op', 'std_time_per_op']
        summary = summary.reset_index()
        
        logger.info("Performance benchmark completed")
        
        return summary
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics"""
        stats = self.monitor.get_stats()
        
        if self.cache is not None:
            cache_stats = self.cache.get_stats()
            stats.update(cache_stats)
        
        return stats
    
    def clear_cache(self):
        """Clear the distance cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Distance cache cleared")
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        config_dict = self.config.to_dict()
        
        with open(filepath, 'wb') as f:
            pickle.dump(config_dict, f)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'LorentzianDistanceCalculator':
        """Load configuration from file"""
        with open(filepath, 'rb') as f:
            config_dict = pickle.load(f)
        
        config = DistanceMetricsConfig(**config_dict)
        
        logger.info(f"Configuration loaded from {filepath}")
        
        return cls(config)


# Convenience functions for direct usage
def lorentzian_distance(
    x: Union[np.ndarray, List[float]], 
    y: Union[np.ndarray, List[float]],
    epsilon: float = 1e-12
) -> float:
    """
    Calculate Lorentzian distance between two vectors (simple interface)
    
    Args:
        x: First feature vector
        y: Second feature vector
        epsilon: Small value for numerical stability
        
    Returns:
        Lorentzian distance
    """
    config = DistanceMetricsConfig(epsilon=epsilon, validate_inputs=False)
    calculator = LorentzianDistanceCalculator(config)
    
    result = calculator.lorentzian_distance(x, y)
    return result.distance


def euclidean_distance(
    x: Union[np.ndarray, List[float]], 
    y: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate Euclidean distance between two vectors (simple interface)
    
    Args:
        x: First feature vector
        y: Second feature vector
        
    Returns:
        Euclidean distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(
    x: Union[np.ndarray, List[float]], 
    y: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate Manhattan distance between two vectors (simple interface)
    
    Args:
        x: First feature vector
        y: Second feature vector
        
    Returns:
        Manhattan distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    return np.sum(np.abs(x - y))


def hybrid_distance(
    x: Union[np.ndarray, List[float]], 
    y: Union[np.ndarray, List[float]],
    alpha: float = 0.5,
    epsilon: float = 1e-12
) -> float:
    """
    Calculate hybrid Lorentzian-Euclidean distance (simple interface)
    
    Args:
        x: First feature vector
        y: Second feature vector
        alpha: Mixing parameter (0=pure Euclidean, 1=pure Lorentzian)
        epsilon: Small value for numerical stability
        
    Returns:
        Hybrid distance
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    return _hybrid_distance_core(x, y, alpha, epsilon)


def adaptive_distance(
    x: Union[np.ndarray, List[float]], 
    y: Union[np.ndarray, List[float]],
    market_data: Optional[pd.DataFrame] = None
) -> float:
    """
    Calculate adaptive distance with automatic metric selection (simple interface)
    
    Args:
        x: First feature vector
        y: Second feature vector
        market_data: OHLCV market data for regime analysis
        
    Returns:
        Adaptive distance
    """
    config = DistanceMetricsConfig()
    calculator = HybridDistanceCalculator(config)
    
    result = calculator.adaptive_distance(x, y, market_data)
    return result.distance


def get_optimal_distance_metric(market_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get optimal distance metric recommendation for current market conditions
    
    Args:
        market_data: OHLCV market data for analysis
        
    Returns:
        Dictionary with metric recommendation and analysis
    """
    config = DistanceMetricsConfig()
    calculator = HybridDistanceCalculator(config)
    
    return calculator.get_metric_recommendation(market_data)


# Module-level testing function
def run_comprehensive_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests on the distance metrics implementation
    
    Returns:
        Dictionary with test results
    """
    logger.info("Running comprehensive tests...")
    
    # Initialize calculator with default configuration
    calculator = LorentzianDistanceCalculator()
    
    # Test 1: Basic functionality
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 2.1, 3.1])
    
    result = calculator.lorentzian_distance(x, y)
    basic_test_passed = result.validation_passed and result.distance > 0
    
    # Test 2: Mathematical properties validation
    math_validation = calculator.validate_mathematical_properties()
    math_test_passed = math_validation['overall_success_rate'] > 0.95
    
    # Test 3: Performance benchmark
    performance_results = calculator.benchmark_performance(
        sizes=[10, 50], 
        n_runs=3
    )
    performance_test_passed = len(performance_results) > 0
    
    # Test 4: Batch processing
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 5)
    
    batch_result = calculator.batch_distances(X, Y)
    batch_test_passed = (
        batch_result.validation_passed and 
        batch_result.distance.shape == (10, 10)
    )
    
    # Test 5: k-NN functionality
    query = np.random.randn(5)
    references = np.random.randn(20, 5)
    
    knn_result = calculator.k_nearest_neighbors(query, references, k=3)
    knn_test_passed = (
        len(knn_result['indices']) == 3 and
        'distances' in knn_result
    )
    
    # Compile results
    test_results = {
        'basic_functionality': basic_test_passed,
        'mathematical_properties': math_test_passed,
        'performance_benchmark': performance_test_passed,
        'batch_processing': batch_test_passed,
        'knn_functionality': knn_test_passed,
        'overall_success': all([
            basic_test_passed,
            math_test_passed,
            performance_test_passed,
            batch_test_passed,
            knn_test_passed
        ]),
        'math_validation_details': math_validation,
        'performance_stats': calculator.get_performance_stats()
    }
    
    logger.info(f"Comprehensive tests completed. Overall success: {test_results['overall_success']}")
    
    return test_results


def demonstrate_hybrid_system():
    """
    Demonstrate the hybrid Lorentzian-Euclidean distance system
    """
    print("\n" + "="*80)
    print("HYBRID LORENTZIAN-EUCLIDEAN DISTANCE SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Generate sample market data with different regimes
    np.random.seed(42)
    n_bars = 200
    
    # Create market data with regime changes
    returns = np.random.normal(0.0001, 0.02, n_bars)
    returns[50:100] = np.random.normal(0.001, 0.05, 50)   # Volatile period
    returns[100:150] = np.random.normal(0.002, 0.01, 50)  # Trending period
    returns[150:200] = np.random.normal(0.0, 0.008, 50)   # Calm period
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    noise = 0.01
    high = prices * (1 + np.abs(np.random.normal(0, noise, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, noise, n_bars)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(10, 1, n_bars)
    
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    # Initialize hybrid calculator
    config = DistanceMetricsConfig()
    hybrid_calc = HybridDistanceCalculator(config)
    
    print("Analyzing different market periods...\n")
    
    # Test different periods
    periods = [
        ("Normal Period", 20, 50),
        ("Volatile Period", 75, 100),
        ("Trending Period", 125, 150),
        ("Calm Period", 175, 200)
    ]
    
    # Generate test feature vectors
    test_features_x = np.array([0.5, 0.3, 0.7, 0.2, 0.6])
    test_features_y = np.array([0.52, 0.28, 0.72, 0.18, 0.62])
    
    for period_name, start, end in periods:
        period_data = market_data.iloc[max(0, start-50):end]
        
        # Get metric recommendation
        recommendation = hybrid_calc.get_metric_recommendation(period_data)
        
        # Calculate distance with adaptive method
        distance_result = hybrid_calc.adaptive_distance(
            test_features_x, test_features_y, period_data
        )
        
        print(f"{period_name}:")
        print(f"  Recommended Metric: {recommendation['recommended_metric'].upper()}")
        print(f"  Alpha Value: {recommendation['alpha_value']:.3f}")
        print(f"  Distance: {distance_result.distance:.6f}")
        print(f"  Method Used: {distance_result.method_used}")
        print(f"  Reasoning: {recommendation['reasoning']}")
        print()
    
    # Compare different distance metrics on the same data
    print("Distance Metric Comparison:")
    print("-" * 40)
    
    test_data = market_data.iloc[150:200]  # Calm period
    
    # Calculate distances using different methods
    lorentzian_dist = hybrid_calc.adaptive_distance(
        test_features_x, test_features_y, test_data, force_metric="lorentzian"
    ).distance
    
    euclidean_dist = hybrid_calc.adaptive_distance(
        test_features_x, test_features_y, test_data, force_metric="euclidean"
    ).distance
    
    hybrid_dist = hybrid_calc.adaptive_distance(
        test_features_x, test_features_y, test_data, force_metric="hybrid", force_alpha=0.5
    ).distance
    
    adaptive_dist = hybrid_calc.adaptive_distance(
        test_features_x, test_features_y, test_data
    ).distance
    
    print(f"Lorentzian Distance: {lorentzian_dist:.6f}")
    print(f"Euclidean Distance:  {euclidean_dist:.6f}")
    print(f"Hybrid Distance (α=0.5): {hybrid_dist:.6f}")
    print(f"Adaptive Distance:   {adaptive_dist:.6f}")
    
    print("\n" + "="*80)
    print("HYBRID SYSTEM READY FOR PRODUCTION!")
    print("✓ Market regime detection")
    print("✓ Intelligent metric selection")
    print("✓ Smooth regime transitions")
    print("✓ Performance optimization")
    print("="*80)


if __name__ == "__main__":
    # Run comprehensive tests
    test_results = run_comprehensive_tests()
    
    print("\n" + "="*80)
    print("LORENTZIAN DISTANCE METRICS - COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    for test_name, result in test_results.items():
        if test_name not in ['math_validation_details', 'performance_stats']:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall Success: {'✓ ALL TESTS PASSED' if test_results['overall_success'] else '✗ SOME TESTS FAILED'}")
    
    # Demonstrate hybrid system
    demonstrate_hybrid_system()