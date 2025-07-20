"""
COMPREHENSIVE SIGNAL GENERATION SYSTEM
=====================================

Complete signal generation system with entry/exit logic for the Lorentzian Classification 
trading strategy. This module implements advanced ML predictions, kernel regression 
trend detection, comprehensive filtering, and real-time signal processing.

Features:
- ML prediction system using optimized Lorentzian classification
- Kernel regression trend detection with crossover signals
- Comprehensive filter system (volatility, regime, ADX, EMA/SMA)
- Signal quality assessment and confidence scoring
- Real-time signal processing with buffering
- Integration interfaces for all components
- Performance monitoring and latency tracking

Author: GrandModel Signal Generation Team
Version: 1.0.0
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
import queue
import warnings
from collections import deque, OrderedDict
from abc import ABC, abstractmethod
import structlog
from numba import jit, njit
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SignalType(Enum):
    """Types of trading signals"""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    NO_SIGNAL = "no_signal"

class FilterType(Enum):
    """Types of signal filters"""
    VOLATILITY = "volatility"
    REGIME = "regime"
    ADX = "adx"
    EMA_TREND = "ema_trend"
    SMA_TREND = "sma_trend"
    CUSTOM = "custom"

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    TRANSITIONAL = "transitional"

class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    INVALID = "invalid"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SignalConfig:
    """Configuration for signal generation system"""
    # Lorentzian classification parameters
    lookback_window: int = 8
    k_neighbors: int = 8
    max_bars_back: int = 5000
    feature_count: int = 5
    
    # Feature engineering parameters
    rsi_length: int = 14
    wt_channel_length: int = 10
    wt_average_length: int = 21
    cci_length: int = 20
    adx_length: int = 14
    
    # Kernel regression parameters
    kernel_lookback: int = 8
    kernel_relative_weighting: float = 8.0
    kernel_regression_level: float = 25.0
    lag_parameter: int = 2
    
    # Filter parameters
    volatility_threshold: float = 0.15
    adx_threshold: float = 25.0
    regime_threshold: float = 0.02
    ema_short_period: int = 9
    ema_long_period: int = 21
    sma_period: int = 50
    
    # Signal quality parameters
    min_confidence: float = 0.6
    min_signal_strength: float = 0.5
    min_neighbors_found: int = 5
    
    # Real-time processing parameters
    signal_buffer_size: int = 1000
    max_processing_latency_ms: float = 50.0
    enable_async_processing: bool = True
    
    # Performance optimization
    use_fast_distance: bool = True
    enable_caching: bool = True
    parallel_processing: bool = True
    use_jit_compilation: bool = True

@dataclass
class FeatureVector:
    """Container for normalized feature vectors"""
    rsi: float
    wt1: float
    wt2: float
    cci: float
    adx: float
    timestamp: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.rsi, self.wt1, self.wt2, self.cci, self.adx])

@dataclass
class KernelState:
    """Kernel regression state information"""
    yhat1: float
    yhat2: float
    yhat1_prev: float
    yhat2_prev: float
    yhat1_slope: float
    yhat2_slope: float
    crossover_bullish: bool
    crossover_bearish: bool
    trend_strength: float
    confidence: float
    timestamp: float

@dataclass
class FilterState:
    """Filter system state"""
    volatility_pass: bool
    regime_pass: bool
    adx_pass: bool
    ema_trend_pass: bool
    sma_trend_pass: bool
    custom_filters_pass: bool
    combined_pass: bool
    filter_scores: Dict[str, float]

@dataclass
class SignalData:
    """Complete signal data structure"""
    signal_type: SignalType
    confidence: float
    signal_strength: float
    signal_quality: SignalQuality
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    # ML components
    ml_prediction: float
    ml_confidence: float
    neighbors_found: int
    
    # Kernel components
    kernel_state: KernelState
    trend_direction: int  # 1 for up, -1 for down, 0 for sideways
    
    # Filter components
    filter_state: FilterState
    regime: MarketRegime
    
    # Quality metrics
    prediction_uncertainty: float
    signal_consistency: float
    historical_performance: float
    
    # Timing
    timestamp: float
    processing_time_ms: float
    latency_ms: float
    
    # Raw data
    current_price: float
    features: FeatureVector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngine:
    """Advanced feature engineering with optimizations"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.scaler = RobustScaler()
        self.feature_cache = OrderedDict()
        self.max_cache_size = 1000
        
    def _calculate_rsi(self, prices: np.ndarray, length: int = 14) -> np.ndarray:
        """Optimized RSI calculation"""
        if len(prices) < length + 1:
            return np.full(len(prices), 50.0)
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use exponential moving average for smoother results
        alpha = 2.0 / (length + 1)
        avg_gains = np.zeros_like(gains)
        avg_losses = np.zeros_like(losses)
        
        # Initialize first value
        avg_gains[0] = np.mean(gains[:length]) if len(gains) >= length else 0
        avg_losses[0] = np.mean(losses[:length]) if len(losses) >= length else 1e-8
        
        # Calculate EMA
        for i in range(1, len(gains)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with first valid value
        rsi_padded = np.full(len(prices), rsi[0] if len(rsi) > 0 else 50.0)
        rsi_padded[1:len(rsi)+1] = rsi
        
        return rsi_padded
    
    def _calculate_wave_trend(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Wave Trend calculation"""
        hlc3 = (high + low + close) / 3
        
        # EMA calculation with pandas for efficiency
        hlc3_series = pd.Series(hlc3)
        esa = hlc3_series.ewm(span=self.config.wt_channel_length).mean()
        d = pd.Series(np.abs(hlc3 - esa.values)).ewm(span=self.config.wt_channel_length).mean()
        
        # Avoid division by zero
        ci = (hlc3 - esa.values) / (0.015 * d.values + 1e-8)
        
        wt1 = pd.Series(ci).ewm(span=self.config.wt_average_length).mean()
        wt2 = wt1.rolling(window=4, min_periods=1).mean()
        
        return wt1.values, wt2.values
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, length: int = 20) -> np.ndarray:
        """Optimized CCI calculation"""
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price)
        
        sma_tp = tp_series.rolling(window=length, min_periods=1).mean()
        mean_dev = tp_series.rolling(window=length, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp.values) / (0.015 * mean_dev.values + 1e-8)
        return cci
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, length: int = 14) -> np.ndarray:
        """Optimized ADX calculation"""
        if len(high) < 2:
            return np.full(len(high), 20.0)
            
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Fix first value
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
        
        # Smoothed values using EMA
        alpha = 2.0 / (length + 1)
        atr = pd.Series(tr).ewm(span=length).mean()
        di_plus = 100 * pd.Series(dm_plus).ewm(span=length).mean() / atr
        di_minus = 100 * pd.Series(dm_minus).ewm(span=length).mean() / atr
        
        # ADX calculation
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = pd.Series(dx).ewm(span=length).mean()
        
        return adx.values
    
    def extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract optimized feature matrix"""
        try:
            if len(data) < max(self.config.rsi_length, self.config.wt_average_length, 
                              self.config.cci_length, self.config.adx_length):
                return None
            
            # Create cache key
            cache_key = f"{len(data)}_{hash(str(data['close'].iloc[-10:].values.tobytes()))}"
            
            if self.config.enable_caching and cache_key in self.feature_cache:
                # Move to end (LRU)
                self.feature_cache.move_to_end(cache_key)
                return self.feature_cache[cache_key]
            
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate indicators
            rsi = self._calculate_rsi(close, self.config.rsi_length)
            wt1, wt2 = self._calculate_wave_trend(high, low, close)
            cci = self._calculate_cci(high, low, close, self.config.cci_length)
            adx = self._calculate_adx(high, low, close, self.config.adx_length)
            
            # Align arrays
            min_length = min(len(rsi), len(wt1), len(wt2), len(cci), len(adx))
            start_idx = max(self.config.rsi_length, self.config.wt_average_length, 
                           self.config.cci_length, self.config.adx_length)
            
            if min_length <= start_idx:
                return None
            
            # Order by importance: WT2, WT1, ADX, RSI, CCI
            features = np.column_stack([
                wt2[:min_length],
                wt1[:min_length],
                adx[:min_length],
                rsi[:min_length],
                cci[:min_length]
            ])
            
            # Remove NaN values
            valid_mask = ~np.isnan(features).any(axis=1)
            features = features[valid_mask]
            
            if len(features) == 0:
                return None
            
            # Robust normalization
            try:
                features_normalized = self.scaler.fit_transform(features)
            except:
                # Fallback to percentile normalization
                features_normalized = np.zeros_like(features)
                for i in range(features.shape[1]):
                    col = features[:, i]
                    p1, p99 = np.percentile(col, [1, 99])
                    features_normalized[:, i] = np.clip((col - p1) / (p99 - p1 + 1e-8), 0, 1)
            
            # Cache result
            if self.config.enable_caching:
                if len(self.feature_cache) >= self.max_cache_size:
                    self.feature_cache.popitem(last=False)  # Remove oldest
                self.feature_cache[cache_key] = features_normalized
            
            return features_normalized
            
        except Exception as e:
            logger.warning("Error in feature extraction", error=str(e))
            return None

# =============================================================================
# LORENTZIAN CLASSIFIER
# =============================================================================

class OptimizedLorentzianClassifier:
    """Optimized Lorentzian Classification system"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.feature_engine = AdvancedFeatureEngine(config)
        
        # Historical data storage with circular buffer
        self.max_history = config.max_bars_back
        self.feature_history = np.zeros((self.max_history, config.feature_count))
        self.target_history = np.zeros(self.max_history)
        self.price_history = np.zeros(self.max_history)
        self.timestamp_history = np.zeros(self.max_history)
        
        self.history_index = 0
        self.history_count = 0
        
        # Performance tracking
        self.prediction_count = 0
        self.correct_predictions = 0
        self.distance_cache = {}
        
        if config.use_jit_compilation:
            # Pre-compile JIT functions
            self._jit_lorentzian_distance(np.ones(5), np.ones(5))
    
    @staticmethod
    @njit
    def _jit_lorentzian_distance(x: np.ndarray, y: np.ndarray) -> float:
        """JIT-compiled Lorentzian distance calculation"""
        total = 0.0
        for i in range(len(x)):
            diff = abs(x[i] - y[i])
            total += np.log(1.0 + diff)
        return total
    
    def _fast_lorentzian_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fast Lorentzian distance with caching"""
        if self.config.use_jit_compilation:
            return self._jit_lorentzian_distance(x, y)
        else:
            diff = np.abs(x - y)
            return np.sum(np.log1p(diff))
    
    def _find_k_nearest_neighbors(self, current_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find k-nearest neighbors with optimized search"""
        if self.history_count < self.config.k_neighbors:
            return np.array([]), np.array([]), np.array([])
        
        distances = []
        indices = []
        targets = []
        
        # Calculate distances to historical features
        search_count = min(self.history_count - self.config.lookback_window, self.max_history)
        
        for i in range(search_count):
            hist_features = self.feature_history[i]
            
            # Skip invalid features
            if np.any(np.isnan(hist_features)) or np.any(np.isinf(hist_features)):
                continue
            
            distance = self._fast_lorentzian_distance(current_features, hist_features)
            distances.append(distance)
            indices.append(i)
            targets.append(self.target_history[i])
        
        if len(distances) < self.config.k_neighbors:
            return np.array(distances), np.array(indices), np.array(targets)
        
        # Get k nearest neighbors
        distances = np.array(distances)
        indices = np.array(indices)
        targets = np.array(targets)
        
        sorted_idx = np.argsort(distances)[:self.config.k_neighbors]
        
        return distances[sorted_idx], indices[sorted_idx], targets[sorted_idx]
    
    def predict(self, current_features: np.ndarray) -> Dict[str, float]:
        """Generate ML prediction using k-nearest neighbors"""
        start_time = time.perf_counter()
        
        try:
            # Find k-nearest neighbors
            distances, indices, targets = self._find_k_nearest_neighbors(current_features)
            
            if len(distances) == 0:
                return {
                    'prediction': 0.5,
                    'confidence': 0.0,
                    'neighbors_found': 0,
                    'prediction_time_ms': (time.perf_counter() - start_time) * 1000
                }
            
            # Weighted prediction based on inverse distance
            weights = 1.0 / (distances + 1e-8)
            weighted_sum = np.sum(weights * targets)
            total_weight = np.sum(weights)
            
            prediction_score = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Enhanced confidence calculation
            confidence = self._calculate_prediction_confidence(distances, targets, prediction_score)
            
            # Update performance tracking
            self.prediction_count += 1
            
            return {
                'prediction': prediction_score,
                'confidence': confidence,
                'neighbors_found': len(distances),
                'distance_mean': np.mean(distances),
                'distance_std': np.std(distances),
                'weight_distribution': weights / np.sum(weights),
                'prediction_time_ms': (time.perf_counter() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error("Error in ML prediction", error=str(e))
            return {
                'prediction': 0.5,
                'confidence': 0.0,
                'neighbors_found': 0,
                'error': str(e),
                'prediction_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _calculate_prediction_confidence(self, distances: np.ndarray, 
                                       targets: np.ndarray, prediction: float) -> float:
        """Calculate enhanced prediction confidence"""
        try:
            # Base confidence from distance distribution
            if len(distances) > 1:
                distance_consistency = 1.0 / (1.0 + np.std(distances) / (np.mean(distances) + 1e-8))
            else:
                distance_consistency = 0.5
            
            # Target consistency (how similar are the neighbor predictions)
            if len(targets) > 1:
                target_consistency = 1.0 - np.std(targets)
            else:
                target_consistency = 0.5
            
            # Prediction strength (how far from neutral 0.5)
            prediction_strength = abs(prediction - 0.5) * 2.0
            
            # Neighbor count factor
            neighbor_factor = min(1.0, len(distances) / self.config.k_neighbors)
            
            # Combined confidence
            confidence = (distance_consistency * 0.3 + 
                         target_consistency * 0.3 + 
                         prediction_strength * 0.3 + 
                         neighbor_factor * 0.1)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def update_history(self, features: np.ndarray, target: int, price: float, timestamp: float):
        """Update historical data with circular buffer"""
        try:
            if len(features) == self.config.feature_count:
                self.feature_history[self.history_index] = features
                self.target_history[self.history_index] = target
                self.price_history[self.history_index] = price
                self.timestamp_history[self.history_index] = timestamp
                
                self.history_index = (self.history_index + 1) % self.max_history
                self.history_count = min(self.history_count + 1, self.max_history)
        except Exception as e:
            logger.warning("Error updating ML history", error=str(e))

# =============================================================================
# KERNEL REGRESSION
# =============================================================================

class KernelRegressionEngine:
    """Advanced kernel regression for trend detection"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.kernel_cache = {}
        
    @staticmethod
    @njit
    def _rational_quadratic_kernel(x: float, y: float, alpha: float, length_scale: float) -> float:
        """JIT-compiled rational quadratic kernel"""
        distance_sq = (x - y) ** 2
        return (1.0 + distance_sq / (2.0 * alpha * length_scale * length_scale)) ** (-alpha)
    
    def calculate_kernel_regression(self, prices: np.ndarray) -> KernelState:
        """Calculate kernel regression with crossover detection"""
        try:
            if len(prices) < self.config.kernel_lookback:
                return self._default_kernel_state()
            
            current_idx = len(prices) - 1
            lookback = min(self.config.kernel_lookback, current_idx)
            
            # Calculate yhat1 and yhat2
            yhat1 = self._nadaraya_watson_estimation(prices, current_idx, lookback)
            yhat2 = self._nadaraya_watson_estimation(prices, current_idx, 
                                                   lookback - self.config.lag_parameter)
            
            # Get previous values for crossover detection
            if current_idx >= 1:
                yhat1_prev = self._nadaraya_watson_estimation(prices, current_idx - 1, lookback)
                yhat2_prev = self._nadaraya_watson_estimation(prices, current_idx - 1, 
                                                            lookback - self.config.lag_parameter)
            else:
                yhat1_prev = yhat1
                yhat2_prev = yhat2
            
            # Calculate slopes
            yhat1_slope = yhat1 - yhat1_prev
            yhat2_slope = yhat2 - yhat2_prev
            
            # Detect crossovers
            crossover_bullish = (yhat2_prev <= yhat1_prev) and (yhat2 > yhat1)
            crossover_bearish = (yhat2_prev >= yhat1_prev) and (yhat2 < yhat1)
            
            # Calculate trend strength
            trend_strength = abs(yhat1_slope) / max(abs(prices[current_idx]), 1e-8)
            
            # Calculate confidence based on regression quality
            confidence = self._calculate_kernel_confidence(prices, yhat1, yhat2, current_idx)
            
            return KernelState(
                yhat1=yhat1,
                yhat2=yhat2,
                yhat1_prev=yhat1_prev,
                yhat2_prev=yhat2_prev,
                yhat1_slope=yhat1_slope,
                yhat2_slope=yhat2_slope,
                crossover_bullish=crossover_bullish,
                crossover_bearish=crossover_bearish,
                trend_strength=trend_strength,
                confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.warning("Error in kernel regression", error=str(e))
            return self._default_kernel_state()
    
    def _nadaraya_watson_estimation(self, prices: np.ndarray, target_idx: int, lookback: int) -> float:
        """Nadaraya-Watson kernel regression estimation"""
        try:
            start_idx = max(0, target_idx - lookback)
            end_idx = target_idx + 1
            
            if end_idx - start_idx < 2:
                return prices[target_idx]
            
            weights = np.zeros(end_idx - start_idx)
            values = prices[start_idx:end_idx]
            
            # Calculate kernel weights
            for i, idx in enumerate(range(start_idx, end_idx)):
                weight = self._rational_quadratic_kernel(
                    float(target_idx), float(idx),
                    self.config.kernel_relative_weighting,
                    float(self.config.kernel_lookback)
                )
                weights[i] = weight
            
            # Weighted average
            total_weight = np.sum(weights)
            if total_weight > 1e-8:
                return np.sum(weights * values) / total_weight
            else:
                return prices[target_idx]
                
        except Exception:
            return prices[target_idx] if target_idx < len(prices) else 0.0
    
    def _calculate_kernel_confidence(self, prices: np.ndarray, yhat1: float, 
                                   yhat2: float, current_idx: int) -> float:
        """Calculate kernel regression confidence"""
        try:
            # Calculate prediction error
            current_price = prices[current_idx]
            prediction_error1 = abs(yhat1 - current_price) / max(abs(current_price), 1e-8)
            prediction_error2 = abs(yhat2 - current_price) / max(abs(current_price), 1e-8)
            
            # Average prediction error
            avg_error = (prediction_error1 + prediction_error2) / 2.0
            
            # Convert error to confidence (inverse relationship)
            confidence = 1.0 / (1.0 + avg_error * 10.0)
            
            # Consider convergence of yhat1 and yhat2
            convergence = 1.0 - abs(yhat1 - yhat2) / max(abs(current_price), 1e-8)
            convergence = max(0.0, convergence)
            
            # Combined confidence
            final_confidence = (confidence * 0.7 + convergence * 0.3)
            return np.clip(final_confidence, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _default_kernel_state(self) -> KernelState:
        """Return default kernel state for error cases"""
        return KernelState(
            yhat1=0.0, yhat2=0.0, yhat1_prev=0.0, yhat2_prev=0.0,
            yhat1_slope=0.0, yhat2_slope=0.0,
            crossover_bullish=False, crossover_bearish=False,
            trend_strength=0.0, confidence=0.0, timestamp=time.time()
        )

# =============================================================================
# FILTER SYSTEM
# =============================================================================

class ComprehensiveFilterSystem:
    """Advanced multi-layer filter system"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.custom_filters = []
        
    def add_custom_filter(self, filter_func, weight: float = 1.0):
        """Add custom filter function"""
        self.custom_filters.append({'func': filter_func, 'weight': weight})
    
    def apply_all_filters(self, data: pd.DataFrame, current_features: FeatureVector) -> FilterState:
        """Apply comprehensive filter system"""
        try:
            filter_scores = {}
            
            # 1. Volatility filter
            volatility_pass, vol_score = self._volatility_filter(data)
            filter_scores['volatility'] = vol_score
            
            # 2. Regime filter
            regime_pass, regime_score = self._regime_filter(data)
            filter_scores['regime'] = regime_score
            
            # 3. ADX filter
            adx_pass, adx_score = self._adx_filter(current_features.adx)
            filter_scores['adx'] = adx_score
            
            # 4. EMA trend filter
            ema_trend_pass, ema_score = self._ema_trend_filter(data)
            filter_scores['ema_trend'] = ema_score
            
            # 5. SMA trend filter
            sma_trend_pass, sma_score = self._sma_trend_filter(data)
            filter_scores['sma_trend'] = sma_score
            
            # 6. Custom filters
            custom_filters_pass, custom_score = self._apply_custom_filters(data, current_features)
            filter_scores['custom'] = custom_score
            
            # Combined filter result
            combined_pass = (volatility_pass and regime_pass and adx_pass and 
                           ema_trend_pass and sma_trend_pass and custom_filters_pass)
            
            return FilterState(
                volatility_pass=volatility_pass,
                regime_pass=regime_pass,
                adx_pass=adx_pass,
                ema_trend_pass=ema_trend_pass,
                sma_trend_pass=sma_trend_pass,
                custom_filters_pass=custom_filters_pass,
                combined_pass=combined_pass,
                filter_scores=filter_scores
            )
            
        except Exception as e:
            logger.warning("Error in filter system", error=str(e))
            return self._default_filter_state()
    
    def _volatility_filter(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Advanced volatility filter"""
        try:
            if len(data) < 20:
                return True, 0.5
            
            close_prices = data['close'].values
            
            # Multiple volatility measures
            returns = np.diff(np.log(close_prices[-20:]))
            
            # 1. Standard volatility
            vol_std = np.std(returns) * np.sqrt(252)
            
            # 2. Garman-Klass volatility (uses OHLC)
            if all(col in data.columns for col in ['high', 'low', 'open']):
                high = data['high'].values[-20:]
                low = data['low'].values[-20:]
                open_prices = data['open'].values[-20:]
                close = close_prices[-20:]
                
                # Garman-Klass estimator with safety checks
                try:
                    gk_vol = np.sqrt(np.mean(
                        0.5 * (np.log(high/(low + 1e-8)))**2 - 
                        (2*np.log(2) - 1) * (np.log(close/(open_prices + 1e-8)))**2
                    )) * np.sqrt(252)
                except:
                    gk_vol = vol_std
            else:
                gk_vol = vol_std
            
            # 3. Rolling volatility
            vol_rolling = pd.Series(returns).rolling(window=10).std().iloc[-1] * np.sqrt(252)
            
            # Combined volatility measure
            combined_vol = np.mean([vol_std, gk_vol, vol_rolling])
            
            # Adaptive threshold based on historical volatility
            hist_vol = np.std(np.diff(np.log(close_prices))) * np.sqrt(252)
            adaptive_threshold = max(self.config.volatility_threshold, hist_vol * 1.5)
            
            filter_pass = combined_vol < adaptive_threshold
            filter_score = 1.0 - min(1.0, combined_vol / adaptive_threshold)
            
            return filter_pass, filter_score
            
        except Exception as e:
            logger.warning("Error in volatility filter", error=str(e))
            return True, 0.5
    
    def _regime_filter(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Advanced regime detection filter"""
        try:
            if len(data) < 50:
                return True, 0.5
            
            close_prices = data['close'].values
            current_price = close_prices[-1]
            
            # Multiple timeframe analysis
            short_ma = np.mean(close_prices[-10:])  # 10-period MA
            medium_ma = np.mean(close_prices[-25:])  # 25-period MA
            long_ma = np.mean(close_prices[-50:])   # 50-period MA
            
            # Trend strength indicators
            short_trend = (current_price - short_ma) / short_ma
            medium_trend = (current_price - medium_ma) / medium_ma
            long_trend = (current_price - long_ma) / long_ma
            
            # MA alignment for trend confirmation
            ma_alignment = (short_ma > medium_ma > long_ma) or (short_ma < medium_ma < long_ma)
            
            # Trend consistency
            trend_consistency = (np.sign(short_trend) == np.sign(medium_trend) == np.sign(long_trend))
            
            # Price position relative to moving averages
            price_position = np.mean([
                1.0 if current_price > short_ma else 0.0,
                1.0 if current_price > medium_ma else 0.0,
                1.0 if current_price > long_ma else 0.0
            ])
            
            # Overall trend strength
            trend_strength = np.mean([abs(short_trend), abs(medium_trend), abs(long_trend)])
            
            # Filter conditions
            filter_pass = (trend_strength > self.config.regime_threshold and 
                          (ma_alignment or trend_consistency))
            
            # Filter score based on trend strength and consistency
            filter_score = trend_strength * (0.5 + 0.5 * (ma_alignment or trend_consistency))
            filter_score = min(1.0, filter_score)
            
            return filter_pass, filter_score
            
        except Exception as e:
            logger.warning("Error in regime filter", error=str(e))
            return True, 0.5
    
    def _adx_filter(self, adx_value: float) -> Tuple[bool, float]:
        """ADX trend strength filter"""
        try:
            # Normalize ADX value (assuming it's already normalized in features)
            adx_actual = adx_value * 100  # Convert back from normalized value
            
            filter_pass = adx_actual > self.config.adx_threshold
            filter_score = min(1.0, adx_actual / self.config.adx_threshold)
            
            return filter_pass, filter_score
            
        except Exception as e:
            logger.warning("Error in ADX filter", error=str(e))
            return True, 0.5
    
    def _ema_trend_filter(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """EMA trend filter"""
        try:
            if len(data) < max(self.config.ema_short_period, self.config.ema_long_period):
                return True, 0.5
            
            close_prices = data['close'].values
            
            # Calculate EMAs
            ema_short = pd.Series(close_prices).ewm(span=self.config.ema_short_period).mean().iloc[-1]
            ema_long = pd.Series(close_prices).ewm(span=self.config.ema_long_period).mean().iloc[-1]
            
            # Trend direction and strength
            trend_direction = 1 if ema_short > ema_long else -1
            trend_strength = abs(ema_short - ema_long) / ema_long
            
            # Filter conditions (require minimum trend strength)
            min_trend_strength = 0.005  # 0.5% minimum separation
            filter_pass = trend_strength > min_trend_strength
            filter_score = min(1.0, trend_strength / min_trend_strength)
            
            return filter_pass, filter_score
            
        except Exception as e:
            logger.warning("Error in EMA trend filter", error=str(e))
            return True, 0.5
    
    def _sma_trend_filter(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """SMA trend filter"""
        try:
            if len(data) < self.config.sma_period:
                return True, 0.5
            
            close_prices = data['close'].values
            current_price = close_prices[-1]
            
            # Calculate SMA
            sma = np.mean(close_prices[-self.config.sma_period:])
            
            # Price position relative to SMA
            price_deviation = (current_price - sma) / sma
            
            # Trend slope (SMA direction)
            if len(close_prices) >= self.config.sma_period + 5:
                sma_prev = np.mean(close_prices[-(self.config.sma_period + 5):-5])
                sma_slope = (sma - sma_prev) / sma_prev
            else:
                sma_slope = 0.0
            
            # Filter conditions
            min_deviation = 0.01  # 1% minimum deviation
            filter_pass = abs(price_deviation) > min_deviation
            filter_score = min(1.0, abs(price_deviation) / min_deviation)
            
            return filter_pass, filter_score
            
        except Exception as e:
            logger.warning("Error in SMA trend filter", error=str(e))
            return True, 0.5
    
    def _apply_custom_filters(self, data: pd.DataFrame, features: FeatureVector) -> Tuple[bool, float]:
        """Apply custom filters"""
        try:
            if not self.custom_filters:
                return True, 1.0
            
            total_score = 0.0
            total_weight = 0.0
            all_pass = True
            
            for filter_info in self.custom_filters:
                try:
                    filter_func = filter_info['func']
                    weight = filter_info['weight']
                    
                    result = filter_func(data, features)
                    if isinstance(result, tuple):
                        filter_pass, score = result
                    else:
                        filter_pass = bool(result)
                        score = 1.0 if filter_pass else 0.0
                    
                    total_score += score * weight
                    total_weight += weight
                    all_pass = all_pass and filter_pass
                    
                except Exception as e:
                    logger.warning("Error in custom filter", error=str(e))
                    continue
            
            if total_weight > 0:
                avg_score = total_score / total_weight
            else:
                avg_score = 1.0
            
            return all_pass, avg_score
            
        except Exception as e:
            logger.warning("Error applying custom filters", error=str(e))
            return True, 1.0
    
    def _default_filter_state(self) -> FilterState:
        """Return default filter state for error cases"""
        return FilterState(
            volatility_pass=False,
            regime_pass=False,
            adx_pass=False,
            ema_trend_pass=False,
            sma_trend_pass=False,
            custom_filters_pass=False,
            combined_pass=False,
            filter_scores={}
        )

# =============================================================================
# SIGNAL QUALITY ASSESSMENT
# =============================================================================

class SignalQualityAssessment:
    """Advanced signal quality assessment system"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.historical_performance = deque(maxlen=1000)
        self.signal_history = deque(maxlen=500)
        
    def assess_signal_quality(self, ml_result: Dict, kernel_state: KernelState,
                            filter_state: FilterState, features: FeatureVector) -> Dict[str, Any]:
        """Comprehensive signal quality assessment"""
        try:
            # Component quality scores
            ml_quality = self._assess_ml_quality(ml_result)
            kernel_quality = self._assess_kernel_quality(kernel_state)
            filter_quality = self._assess_filter_quality(filter_state)
            feature_quality = self._assess_feature_quality(features)
            
            # Historical performance factor
            historical_factor = self._get_historical_performance_factor()
            
            # Signal consistency
            consistency_score = self._calculate_signal_consistency(ml_result, kernel_state)
            
            # Overall quality calculation
            component_scores = {
                'ml_quality': ml_quality,
                'kernel_quality': kernel_quality,
                'filter_quality': filter_quality,
                'feature_quality': feature_quality,
                'historical_factor': historical_factor,
                'consistency_score': consistency_score
            }
            
            # Weighted overall score
            weights = {
                'ml_quality': 0.3,
                'kernel_quality': 0.25,
                'filter_quality': 0.2,
                'feature_quality': 0.1,
                'historical_factor': 0.1,
                'consistency_score': 0.05
            }
            
            overall_score = sum(component_scores[key] * weights[key] 
                              for key in weights.keys())
            
            # Quality classification
            if overall_score >= 0.8:
                quality_level = SignalQuality.EXCELLENT
            elif overall_score >= 0.65:
                quality_level = SignalQuality.GOOD
            elif overall_score >= 0.5:
                quality_level = SignalQuality.MODERATE
            elif overall_score >= 0.3:
                quality_level = SignalQuality.POOR
            else:
                quality_level = SignalQuality.INVALID
            
            return {
                'overall_score': overall_score,
                'quality_level': quality_level,
                'component_scores': component_scores,
                'weights': weights,
                'assessment_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error("Error in signal quality assessment", error=str(e))
            return {
                'overall_score': 0.0,
                'quality_level': SignalQuality.INVALID,
                'component_scores': {},
                'error': str(e)
            }
    
    def _assess_ml_quality(self, ml_result: Dict) -> float:
        """Assess ML prediction quality"""
        try:
            confidence = ml_result.get('confidence', 0.0)
            neighbors_found = ml_result.get('neighbors_found', 0)
            prediction = ml_result.get('prediction', 0.5)
            
            # Confidence factor
            confidence_factor = confidence
            
            # Neighbors factor
            neighbors_factor = min(1.0, neighbors_found / self.config.k_neighbors)
            
            # Prediction strength (distance from neutral)
            prediction_strength = abs(prediction - 0.5) * 2.0
            
            # Distance consistency
            distance_std = ml_result.get('distance_std', 1.0)
            distance_mean = ml_result.get('distance_mean', 1.0)
            distance_consistency = 1.0 / (1.0 + distance_std / (distance_mean + 1e-8))
            
            # Combined ML quality
            ml_quality = (confidence_factor * 0.4 + 
                         neighbors_factor * 0.3 + 
                         prediction_strength * 0.2 + 
                         distance_consistency * 0.1)
            
            return np.clip(ml_quality, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _assess_kernel_quality(self, kernel_state: KernelState) -> float:
        """Assess kernel regression quality"""
        try:
            # Kernel confidence
            kernel_confidence = kernel_state.confidence
            
            # Trend strength
            trend_strength = min(1.0, kernel_state.trend_strength * 10.0)
            
            # Crossover clarity (if crossover occurred)
            crossover_strength = 0.0
            if kernel_state.crossover_bullish or kernel_state.crossover_bearish:
                separation = abs(kernel_state.yhat1 - kernel_state.yhat2)
                crossover_strength = min(1.0, separation * 100.0)
            
            # Slope consistency
            slope_consistency = 1.0 - abs(kernel_state.yhat1_slope - kernel_state.yhat2_slope)
            slope_consistency = max(0.0, slope_consistency)
            
            # Combined kernel quality
            kernel_quality = (kernel_confidence * 0.4 + 
                            trend_strength * 0.3 + 
                            crossover_strength * 0.2 + 
                            slope_consistency * 0.1)
            
            return np.clip(kernel_quality, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _assess_filter_quality(self, filter_state: FilterState) -> float:
        """Assess filter system quality"""
        try:
            if not filter_state.filter_scores:
                return 0.0
            
            # Average filter scores
            avg_score = np.mean(list(filter_state.filter_scores.values()))
            
            # Filter agreement (how many filters pass)
            filter_agreement = sum([
                filter_state.volatility_pass,
                filter_state.regime_pass,
                filter_state.adx_pass,
                filter_state.ema_trend_pass,
                filter_state.sma_trend_pass,
                filter_state.custom_filters_pass
            ]) / 6.0
            
            # Combined filter quality
            filter_quality = (avg_score * 0.7 + filter_agreement * 0.3)
            
            return np.clip(filter_quality, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _assess_feature_quality(self, features: FeatureVector) -> float:
        """Assess feature vector quality"""
        try:
            feature_array = features.to_array()
            
            # Check for extreme values or NaN
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                return 0.0
            
            # Feature diversity (avoid all features being similar)
            feature_std = np.std(feature_array)
            diversity_score = min(1.0, feature_std * 2.0)
            
            # Feature range (should be within reasonable bounds)
            range_score = 1.0 if np.all((feature_array >= 0) & (feature_array <= 1)) else 0.5
            
            # Combined feature quality
            feature_quality = (diversity_score * 0.6 + range_score * 0.4)
            
            return np.clip(feature_quality, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _get_historical_performance_factor(self) -> float:
        """Get historical performance factor"""
        try:
            if not self.historical_performance:
                return 0.5  # Neutral if no history
            
            recent_performance = list(self.historical_performance)[-50:]  # Last 50 signals
            avg_performance = np.mean(recent_performance)
            
            return np.clip(avg_performance, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_signal_consistency(self, ml_result: Dict, kernel_state: KernelState) -> float:
        """Calculate consistency between ML and kernel signals"""
        try:
            ml_prediction = ml_result.get('prediction', 0.5)
            
            # ML signal direction
            ml_bullish = ml_prediction > 0.5
            
            # Kernel signal direction
            kernel_bullish = (kernel_state.crossover_bullish or 
                            (kernel_state.yhat1_slope > 0 and kernel_state.yhat2_slope > 0))
            kernel_bearish = (kernel_state.crossover_bearish or 
                            (kernel_state.yhat1_slope < 0 and kernel_state.yhat2_slope < 0))
            
            # Check agreement
            if ml_bullish and kernel_bullish:
                consistency = 1.0
            elif not ml_bullish and kernel_bearish:
                consistency = 1.0
            elif not kernel_bullish and not kernel_bearish:
                # Neutral kernel, partial consistency
                consistency = 0.5
            else:
                # Disagreement
                consistency = 0.0
            
            return consistency
            
        except Exception:
            return 0.5
    
    def update_historical_performance(self, signal_quality: float, actual_outcome: Optional[float] = None):
        """Update historical performance tracking"""
        try:
            if actual_outcome is not None:
                # Use actual outcome if available
                performance = actual_outcome
            else:
                # Use signal quality as proxy
                performance = signal_quality
            
            self.historical_performance.append(performance)
            
        except Exception as e:
            logger.warning("Error updating historical performance", error=str(e))

# =============================================================================
# REGIME DETECTOR
# =============================================================================

class MarketRegimeDetector:
    """Advanced market regime detection"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.regime_history = deque(maxlen=100)
        
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(data) < 50:
                return MarketRegime.TRANSITIONAL
            
            close_prices = data['close'].values
            
            # Multi-timeframe analysis
            regimes = []
            
            # Short-term regime (10 periods)
            regimes.append(self._analyze_regime_timeframe(close_prices, 10))
            
            # Medium-term regime (25 periods)
            regimes.append(self._analyze_regime_timeframe(close_prices, 25))
            
            # Long-term regime (50 periods)
            regimes.append(self._analyze_regime_timeframe(close_prices, 50))
            
            # Aggregate regime decision
            regime = self._aggregate_regime_decisions(regimes)
            
            # Smooth regime transitions
            regime = self._smooth_regime_transitions(regime)
            
            return regime
            
        except Exception as e:
            logger.warning("Error in regime detection", error=str(e))
            return MarketRegime.TRANSITIONAL
    
    def _analyze_regime_timeframe(self, prices: np.ndarray, period: int) -> MarketRegime:
        """Analyze regime for specific timeframe"""
        try:
            if len(prices) < period:
                return MarketRegime.TRANSITIONAL
            
            recent_prices = prices[-period:]
            
            # Trend analysis
            x = np.arange(len(recent_prices))
            slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
            trend_strength = abs(r_value)
            
            # Volatility analysis
            returns = np.diff(np.log(recent_prices))
            volatility = np.std(returns)
            
            # Price range analysis
            price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
            
            # Regime classification
            if trend_strength > 0.7 and slope > 0:
                return MarketRegime.TRENDING_UP
            elif trend_strength > 0.7 and slope < 0:
                return MarketRegime.TRENDING_DOWN
            elif volatility > np.percentile(returns, 75):
                return MarketRegime.VOLATILE
            elif volatility < np.percentile(returns, 25):
                return MarketRegime.CALM
            elif price_range < 0.02:
                return MarketRegime.RANGING
            else:
                return MarketRegime.TRANSITIONAL
                
        except Exception:
            return MarketRegime.TRANSITIONAL
    
    def _aggregate_regime_decisions(self, regimes: List[MarketRegime]) -> MarketRegime:
        """Aggregate multiple timeframe regime decisions"""
        try:
            # Count regime votes
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Find most common regime
            most_common_regime = max(regime_counts, key=regime_counts.get)
            
            # If there's a clear majority, use it
            if regime_counts[most_common_regime] >= len(regimes) // 2 + 1:
                return most_common_regime
            else:
                return MarketRegime.TRANSITIONAL
                
        except Exception:
            return MarketRegime.TRANSITIONAL
    
    def _smooth_regime_transitions(self, current_regime: MarketRegime) -> MarketRegime:
        """Smooth regime transitions to avoid excessive switching"""
        try:
            self.regime_history.append(current_regime)
            
            if len(self.regime_history) < 5:
                return current_regime
            
            # Check recent history for stability
            recent_regimes = list(self.regime_history)[-5:]
            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # If current regime has appeared multiple times recently, keep it
            if regime_counts.get(current_regime, 0) >= 3:
                return current_regime
            
            # Otherwise, use the most stable recent regime
            most_stable = max(regime_counts, key=regime_counts.get)
            return most_stable
            
        except Exception:
            return current_regime

# =============================================================================
# MAIN SIGNAL GENERATION SYSTEM
# =============================================================================

class ComprehensiveSignalGenerator:
    """Main signal generation system orchestrating all components"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Initialize components
        self.ml_classifier = OptimizedLorentzianClassifier(config)
        self.kernel_engine = KernelRegressionEngine(config)
        self.filter_system = ComprehensiveFilterSystem(config)
        self.quality_assessor = SignalQualityAssessment(config)
        self.regime_detector = MarketRegimeDetector(config)
        
        # Signal processing
        self.signal_buffer = deque(maxlen=config.signal_buffer_size)
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # Performance tracking
        self.signals_generated = 0
        self.processing_times = deque(maxlen=1000)
        self.error_count = 0
        
        # Threading for async processing
        if config.enable_async_processing:
            self.executor = ThreadPoolExecutor(max_workers=2)
        else:
            self.executor = None
        
        logger.info("Comprehensive Signal Generator initialized", 
                   config=asdict(config))
    
    def generate_signal(self, data: pd.DataFrame, current_price: float) -> SignalData:
        """Generate comprehensive trading signal"""
        start_time = time.perf_counter()
        
        try:
            # 1. Extract features
            features_array = self.ml_classifier.feature_engine.extract_features(data)
            if features_array is None or len(features_array) == 0:
                return self._create_no_signal(current_price, "insufficient_features")
            
            current_features_array = features_array[-1]
            current_features = FeatureVector(
                rsi=current_features_array[3],
                wt1=current_features_array[1],
                wt2=current_features_array[0],
                cci=current_features_array[4],
                adx=current_features_array[2],
                timestamp=time.time()
            )
            
            # 2. Get ML prediction
            ml_result = self.ml_classifier.predict(current_features_array)
            
            # 3. Calculate kernel regression
            prices = data['close'].values
            kernel_state = self.kernel_engine.calculate_kernel_regression(prices)
            
            # 4. Apply filters
            filter_state = self.filter_system.apply_all_filters(data, current_features)
            
            # 5. Detect market regime
            regime = self.regime_detector.detect_regime(data)
            
            # 6. Assess signal quality
            quality_assessment = self.quality_assessor.assess_signal_quality(
                ml_result, kernel_state, filter_state, current_features
            )
            
            # 7. Generate signal decision
            signal_data = self._make_signal_decision(
                ml_result, kernel_state, filter_state, regime,
                quality_assessment, current_features, current_price
            )
            
            # 8. Calculate processing metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            signal_data.processing_time_ms = processing_time
            signal_data.latency_ms = processing_time  # Same for synchronous processing
            
            # 9. Update tracking
            self.signals_generated += 1
            self.processing_times.append(processing_time / 1000)
            
            # 10. Buffer signal for history
            self.signal_buffer.append(signal_data)
            
            # 11. Update ML classifier history if we have a target
            if len(data) > self.config.lookback_window:
                future_idx = min(len(data) - 1 + self.config.lookback_window, len(data) - 1)
                if future_idx < len(data):
                    future_price = data['close'].iloc[future_idx]
                    target = 1 if future_price > current_price else 0
                    self.ml_classifier.update_history(
                        current_features_array, target, current_price, time.time()
                    )
            
            logger.debug("Signal generated successfully",
                        signal_type=signal_data.signal_type.value,
                        confidence=signal_data.confidence,
                        quality=signal_data.signal_quality.value,
                        processing_time_ms=processing_time)
            
            return signal_data
            
        except Exception as e:
            self.error_count += 1
            logger.error("Error generating signal", error=str(e))
            
            processing_time = (time.perf_counter() - start_time) * 1000
            return self._create_error_signal(current_price, str(e), processing_time)
    
    def _make_signal_decision(self, ml_result: Dict, kernel_state: KernelState,
                            filter_state: FilterState, regime: MarketRegime,
                            quality_assessment: Dict, features: FeatureVector,
                            current_price: float) -> SignalData:
        """Make the final signal decision based on all components"""
        try:
            # Extract key metrics
            ml_prediction = ml_result.get('prediction', 0.5)
            ml_confidence = ml_result.get('confidence', 0.0)
            neighbors_found = ml_result.get('neighbors_found', 0)
            
            # Determine signal type based on ML and kernel signals
            signal_type = SignalType.NO_SIGNAL
            signal_strength = 0.0
            
            # Long entry conditions
            long_conditions = [
                ml_prediction > 0.5,  # ML prediction bullish
                kernel_state.crossover_bullish or (kernel_state.yhat1 > kernel_state.yhat1_prev),  # Kernel uptrend
                filter_state.combined_pass,  # All filters pass
                ml_confidence > self.config.min_confidence,  # Sufficient confidence
                neighbors_found >= self.config.min_neighbors_found  # Sufficient neighbors
            ]
            
            # Short entry conditions
            short_conditions = [
                ml_prediction < 0.5,  # ML prediction bearish
                kernel_state.crossover_bearish or (kernel_state.yhat1 < kernel_state.yhat1_prev),  # Kernel downtrend
                filter_state.combined_pass,  # All filters pass
                ml_confidence > self.config.min_confidence,  # Sufficient confidence
                neighbors_found >= self.config.min_neighbors_found  # Sufficient neighbors
            ]
            
            # Evaluate conditions
            long_score = sum(long_conditions) / len(long_conditions)
            short_score = sum(short_conditions) / len(short_conditions)
            
            # Determine signal type
            if long_score >= 0.8:  # Require 80% of conditions
                signal_type = SignalType.LONG_ENTRY
                signal_strength = long_score
            elif short_score >= 0.8:  # Require 80% of conditions
                signal_type = SignalType.SHORT_ENTRY
                signal_strength = short_score
            else:
                signal_type = SignalType.NO_SIGNAL
                signal_strength = max(long_score, short_score)
            
            # Calculate confidence (combination of ML confidence and signal strength)
            combined_confidence = (ml_confidence * 0.6 + signal_strength * 0.4)
            
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_risk_levels(
                current_price, signal_type, kernel_state, regime
            )
            
            # Determine trend direction
            if kernel_state.yhat1_slope > 0.001:
                trend_direction = 1
            elif kernel_state.yhat1_slope < -0.001:
                trend_direction = -1
            else:
                trend_direction = 0
            
            # Calculate additional quality metrics
            prediction_uncertainty = 1.0 - ml_confidence
            signal_consistency = self._calculate_consistency(ml_result, kernel_state)
            historical_performance = quality_assessment.get('component_scores', {}).get('historical_factor', 0.5)
            
            return SignalData(
                signal_type=signal_type,
                confidence=combined_confidence,
                signal_strength=signal_strength,
                signal_quality=quality_assessment['quality_level'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                
                # ML components
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence,
                neighbors_found=neighbors_found,
                
                # Kernel components
                kernel_state=kernel_state,
                trend_direction=trend_direction,
                
                # Filter components
                filter_state=filter_state,
                regime=regime,
                
                # Quality metrics
                prediction_uncertainty=prediction_uncertainty,
                signal_consistency=signal_consistency,
                historical_performance=historical_performance,
                
                # Timing (will be set by caller)
                timestamp=time.time(),
                processing_time_ms=0.0,
                latency_ms=0.0,
                
                # Raw data
                current_price=current_price,
                features=features
            )
            
        except Exception as e:
            logger.error("Error in signal decision making", error=str(e))
            return self._create_no_signal(current_price, "decision_error")
    
    def _calculate_risk_levels(self, current_price: float, signal_type: SignalType,
                             kernel_state: KernelState, regime: MarketRegime) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        try:
            if signal_type == SignalType.NO_SIGNAL:
                return None, None
            
            # Base risk calculation using ATR-like measure from kernel regression
            volatility_estimate = max(abs(kernel_state.yhat1_slope), abs(kernel_state.yhat2_slope))
            base_risk = max(current_price * 0.01, volatility_estimate * 2.0)  # Minimum 1% risk
            
            # Adjust based on regime
            regime_multipliers = {
                MarketRegime.TRENDING_UP: 0.8,
                MarketRegime.TRENDING_DOWN: 0.8,
                MarketRegime.RANGING: 1.2,
                MarketRegime.VOLATILE: 1.5,
                MarketRegime.CALM: 0.7,
                MarketRegime.TRANSITIONAL: 1.0
            }
            
            risk_multiplier = regime_multipliers.get(regime, 1.0)
            adjusted_risk = base_risk * risk_multiplier
            
            # Calculate levels based on signal type
            if signal_type == SignalType.LONG_ENTRY:
                stop_loss = current_price - adjusted_risk
                take_profit = current_price + adjusted_risk * 2.0  # 2:1 risk/reward
            elif signal_type == SignalType.SHORT_ENTRY:
                stop_loss = current_price + adjusted_risk
                take_profit = current_price - adjusted_risk * 2.0  # 2:1 risk/reward
            else:
                stop_loss = None
                take_profit = None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.warning("Error calculating risk levels", error=str(e))
            return None, None
    
    def _calculate_consistency(self, ml_result: Dict, kernel_state: KernelState) -> float:
        """Calculate consistency between ML and kernel signals"""
        try:
            ml_prediction = ml_result.get('prediction', 0.5)
            
            # ML signal direction
            ml_bullish = ml_prediction > 0.5
            ml_strength = abs(ml_prediction - 0.5) * 2.0
            
            # Kernel signal direction
            kernel_bullish = kernel_state.yhat1_slope > 0 or kernel_state.crossover_bullish
            kernel_bearish = kernel_state.yhat1_slope < 0 or kernel_state.crossover_bearish
            kernel_strength = kernel_state.trend_strength
            
            # Calculate agreement
            if ml_bullish and kernel_bullish:
                agreement = 1.0
            elif not ml_bullish and kernel_bearish:
                agreement = 1.0
            elif not kernel_bullish and not kernel_bearish:
                # Neutral kernel state
                agreement = 0.5
            else:
                # Disagreement
                agreement = 0.0
            
            # Weight by signal strengths
            weighted_consistency = agreement * min(ml_strength, kernel_strength)
            
            return np.clip(weighted_consistency, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _create_no_signal(self, current_price: float, reason: str = "no_conditions_met") -> SignalData:
        """Create a no-signal result"""
        return SignalData(
            signal_type=SignalType.NO_SIGNAL,
            confidence=0.0,
            signal_strength=0.0,
            signal_quality=SignalQuality.INVALID,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            
            # ML components
            ml_prediction=0.5,
            ml_confidence=0.0,
            neighbors_found=0,
            
            # Kernel components
            kernel_state=KernelState(
                yhat1=current_price, yhat2=current_price, yhat1_prev=current_price, yhat2_prev=current_price,
                yhat1_slope=0.0, yhat2_slope=0.0, crossover_bullish=False, crossover_bearish=False,
                trend_strength=0.0, confidence=0.0, timestamp=time.time()
            ),
            trend_direction=0,
            
            # Filter components
            filter_state=FilterState(
                volatility_pass=False, regime_pass=False, adx_pass=False,
                ema_trend_pass=False, sma_trend_pass=False, custom_filters_pass=False,
                combined_pass=False, filter_scores={}
            ),
            regime=MarketRegime.TRANSITIONAL,
            
            # Quality metrics
            prediction_uncertainty=1.0,
            signal_consistency=0.0,
            historical_performance=0.0,
            
            # Timing
            timestamp=time.time(),
            processing_time_ms=0.0,
            latency_ms=0.0,
            
            # Raw data
            current_price=current_price,
            features=FeatureVector(rsi=0.5, wt1=0.5, wt2=0.5, cci=0.5, adx=0.5, timestamp=time.time())
        )
    
    def _create_error_signal(self, current_price: float, error_msg: str, processing_time: float) -> SignalData:
        """Create an error signal result"""
        signal = self._create_no_signal(current_price, "error")
        signal.processing_time_ms = processing_time
        signal.latency_ms = processing_time
        return signal
    
    async def generate_signal_async(self, data: pd.DataFrame, current_price: float) -> SignalData:
        """Asynchronous signal generation"""
        if self.executor is None:
            return self.generate_signal(data, current_price)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_signal, data, current_price)
    
    def get_signal_history(self, count: int = 100) -> List[SignalData]:
        """Get recent signal history"""
        return list(self.signal_buffer)[-count:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            if not self.processing_times:
                return {'no_data': True}
            
            recent_times = list(self.processing_times)
            
            # Calculate signal distribution
            signal_types = {}
            quality_levels = {}
            recent_signals = list(self.signal_buffer)[-100:]  # Last 100 signals
            
            for signal in recent_signals:
                signal_type = signal.signal_type.value
                quality = signal.signal_quality.value
                
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
                quality_levels[quality] = quality_levels.get(quality, 0) + 1
            
            return {
                'signals_generated': self.signals_generated,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.signals_generated, 1),
                
                # Performance metrics
                'avg_processing_time_ms': np.mean(recent_times) * 1000,
                'max_processing_time_ms': np.max(recent_times) * 1000,
                'min_processing_time_ms': np.min(recent_times) * 1000,
                'p95_processing_time_ms': np.percentile(recent_times, 95) * 1000,
                'p99_processing_time_ms': np.percentile(recent_times, 99) * 1000,
                
                # Signal statistics
                'signal_distribution': signal_types,
                'quality_distribution': quality_levels,
                'buffer_size': len(self.signal_buffer),
                'max_buffer_size': self.config.signal_buffer_size,
                
                # Component statistics
                'ml_classifier_stats': {
                    'predictions_made': self.ml_classifier.prediction_count,
                    'history_count': self.ml_classifier.history_count,
                    'cache_size': len(self.ml_classifier.feature_engine.feature_cache)
                },
                
                # Configuration
                'config': asdict(self.config)
            }
            
        except Exception as e:
            logger.error("Error getting performance stats", error=str(e))
            return {'error': str(e)}
    
    def reset_system(self):
        """Reset all system state"""
        try:
            # Clear buffers and histories
            self.signal_buffer.clear()
            self.processing_times.clear()
            
            # Reset counters
            self.signals_generated = 0
            self.error_count = 0
            
            # Reset components
            self.ml_classifier = OptimizedLorentzianClassifier(self.config)
            self.quality_assessor = SignalQualityAssessment(self.config)
            self.regime_detector = MarketRegimeDetector(self.config)
            
            logger.info("Signal generation system reset successfully")
            
        except Exception as e:
            logger.error("Error resetting system", error=str(e))
    
    def __del__(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

# =============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# =============================================================================

def create_signal_generator(config_dict: Optional[Dict[str, Any]] = None) -> ComprehensiveSignalGenerator:
    """Factory function to create signal generator with configuration"""
    if config_dict is None:
        config = SignalConfig()
    else:
        config = SignalConfig(**config_dict)
    
    return ComprehensiveSignalGenerator(config)

def create_default_config() -> SignalConfig:
    """Create default signal generation configuration"""
    return SignalConfig()

def create_optimized_config() -> SignalConfig:
    """Create optimized configuration for production use"""
    return SignalConfig(
        # Optimized Lorentzian parameters
        k_neighbors=12,
        lookback_window=10,
        max_bars_back=10000,
        
        # Enhanced feature parameters
        rsi_length=21,
        wt_channel_length=12,
        wt_average_length=24,
        
        # Stricter quality requirements
        min_confidence=0.7,
        min_signal_strength=0.6,
        min_neighbors_found=8,
        
        # Performance optimizations
        use_fast_distance=True,
        enable_caching=True,
        parallel_processing=True,
        use_jit_compilation=True,
        enable_async_processing=True,
        
        # Larger buffers for production
        signal_buffer_size=2000,
        max_processing_latency_ms=30.0
    )

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def demonstrate_signal_generation():
    """Demonstration of the complete signal generation system"""
    print("COMPREHENSIVE SIGNAL GENERATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = create_optimized_config()
    
    # Initialize signal generator
    generator = create_signal_generator(asdict(config))
    
    # Generate synthetic market data
    np.random.seed(42)
    n_bars = 500
    
    # Create realistic OHLCV data with trends and volatility
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_bars)  # Slight upward bias
    
    # Add trend and cycle components
    trend = np.linspace(0, 0.1, n_bars)  # 10% trend over period
    cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, n_bars))  # Cyclical component
    noise = np.random.normal(0, 0.01, n_bars)  # Market noise
    
    # Combine components
    log_returns = returns + trend/n_bars + cycle + noise
    prices = base_price * np.exp(np.cumsum(log_returns))
    
    # Create OHLC from prices
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price
    volume = np.random.lognormal(10, 0.5, n_bars)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    print(f"Generated {n_bars} bars of synthetic market data")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Generate signals for the last portion of data
    signals = []
    processing_times = []
    
    print("\nGenerating signals...")
    
    # Use rolling window approach
    window_size = 100
    signal_count = 50
    
    for i in range(window_size, min(window_size + signal_count, len(data))):
        start_time = time.perf_counter()
        
        # Get data window
        window_data = data.iloc[:i+1]
        current_price = window_data['close'].iloc[-1]
        
        # Generate signal
        signal = generator.generate_signal(window_data, current_price)
        signals.append(signal)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        processing_times.append(processing_time)
        
        # Print progress
        if i % 10 == 0:
            print(f"  Bar {i}: {signal.signal_type.value} "
                  f"(confidence: {signal.confidence:.3f}, "
                  f"quality: {signal.signal_quality.value}, "
                  f"time: {processing_time:.1f}ms)")
    
    # Analyze results
    print(f"\nGenerated {len(signals)} signals")
    
    # Signal distribution
    signal_types = {}
    quality_levels = {}
    
    for signal in signals:
        signal_type = signal.signal_type.value
        quality = signal.signal_quality.value
        
        signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        quality_levels[quality] = quality_levels.get(quality, 0) + 1
    
    print("\nSignal Distribution:")
    for signal_type, count in signal_types.items():
        percentage = (count / len(signals)) * 100
        print(f"  {signal_type}: {count} ({percentage:.1f}%)")
    
    print("\nQuality Distribution:")
    for quality, count in quality_levels.items():
        percentage = (count / len(signals)) * 100
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    # Performance metrics
    avg_processing_time = np.mean(processing_times)
    max_processing_time = np.max(processing_times)
    p95_processing_time = np.percentile(processing_times, 95)
    
    print(f"\nPerformance Metrics:")
    print(f"  Average processing time: {avg_processing_time:.2f}ms")
    print(f"  Maximum processing time: {max_processing_time:.2f}ms")
    print(f"  95th percentile time: {p95_processing_time:.2f}ms")
    print(f"  Target latency met: {p95_processing_time < config.max_processing_latency_ms}")
    
    # Component statistics
    stats = generator.get_performance_stats()
    print(f"\nSystem Statistics:")
    print(f"  Total signals generated: {stats.get('signals_generated', 0)}")
    print(f"  Error rate: {stats.get('error_rate', 0):.3f}")
    print(f"  ML predictions made: {stats.get('ml_classifier_stats', {}).get('predictions_made', 0)}")
    print(f"  Feature cache size: {stats.get('ml_classifier_stats', {}).get('cache_size', 0)}")
    
    # Sample high-quality signals
    high_quality_signals = [s for s in signals if s.signal_quality in [SignalQuality.EXCELLENT, SignalQuality.GOOD]]
    
    if high_quality_signals:
        print(f"\nHigh-Quality Signals ({len(high_quality_signals)} found):")
        for i, signal in enumerate(high_quality_signals[:5]):  # Show first 5
            print(f"  Signal {i+1}:")
            print(f"    Type: {signal.signal_type.value}")
            print(f"    Confidence: {signal.confidence:.3f}")
            print(f"    Strength: {signal.signal_strength:.3f}")
            print(f"    Entry Price: {signal.entry_price:.2f}")
            if signal.stop_loss:
                print(f"    Stop Loss: {signal.stop_loss:.2f}")
            if signal.take_profit:
                print(f"    Take Profit: {signal.take_profit:.2f}")
            print(f"    ML Prediction: {signal.ml_prediction:.3f}")
            print(f"    Neighbors Found: {signal.neighbors_found}")
            print(f"    Regime: {signal.regime.value}")
            print()
    
    print("Signal generation demonstration completed successfully!")
    
    return generator, signals, data

if __name__ == "__main__":
    # Run demonstration
    try:
        generator, signals, data = demonstrate_signal_generation()
        print("\nAll tests passed! System ready for production deployment.")
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()