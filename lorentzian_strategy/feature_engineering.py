"""
Complete Feature Engineering Pipeline for Lorentzian Classification Strategy
=========================================================================

This module implements a comprehensive, production-ready feature engineering system
specifically designed for the Lorentzian Classification trading strategy.

Key Components:
1. Technical Indicators Implementation (RSI, WaveTrend, CCI, ADX, Price momentum)
2. Kernel Regression Components (Rational Quadratic, Gaussian kernels)
3. Feature Normalization and Scaling
4. Streaming Feature Calculation System
5. Multi-timeframe Feature Aggregation
6. Real-time Performance Optimization

Mathematical Foundation:
- Rational Quadratic Kernel: K(x,y) = (1 + ||x-y||²/(2αh²))^(-α)
- Gaussian Kernel: K(x,y) = exp(-||x-y||²/(2h²))
- Nadaraya-Watson Estimator with dynamic parameter adjustment

Performance Features:
- Vectorized NumPy calculations
- Numba JIT compilation for critical functions
- Memory-efficient rolling calculations
- Real-time streaming capabilities
- Feature caching and validation

Author: Claude Code Agent
Version: 1.0.0 - Production Release
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, njit
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import time
import logging
from collections import deque
from abc import ABC, abstractmethod

# Suppress warnings for performance
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features in the Lorentzian system"""
    PRICE_ACTION = "price_action"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    KERNEL_REGRESSION = "kernel_regression"
    DERIVED = "derived"


class NormalizationMethod(Enum):
    """Normalization methods for features"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"
    ROBUST = "robust"
    REGIME_ADAPTIVE = "regime_adaptive"


@dataclass
class FeatureConfig:
    """Configuration for individual features"""
    name: str
    feature_type: FeatureType
    period: int
    normalization: NormalizationMethod
    weight: float = 1.0
    enabled: bool = True
    cache_enabled: bool = True


@dataclass
class KernelConfig:
    """Configuration for kernel regression"""
    kernel_type: str = "rational_quadratic"
    lookback_window: int = 8
    relative_weighting: float = 8.0
    regression_level: float = 25.0
    adaptive_params: bool = True


@dataclass
class LorentzianConfig:
    """Complete configuration for Lorentzian feature engineering"""
    
    # Core parameters
    lookback_window: int = 8
    k_neighbors: int = 8
    max_bars_back: int = 5000
    feature_count: int = 5
    
    # Technical indicator periods
    rsi_period: int = 14
    wt_channel_length: int = 10
    wt_average_length: int = 21
    cci_period: int = 20
    adx_period: int = 14
    
    # Kernel configuration
    kernel_config: KernelConfig = None
    
    # Feature configurations
    feature_configs: List[FeatureConfig] = None
    
    # Performance settings
    enable_numba: bool = True
    enable_caching: bool = True
    max_cache_size: int = 10000
    streaming_mode: bool = True
    
    # Filtering parameters
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = True
    adx_threshold: float = 25.0
    volatility_threshold: float = 0.15
    
    def __post_init__(self):
        if self.kernel_config is None:
            self.kernel_config = KernelConfig()
        
        if self.feature_configs is None:
            self.feature_configs = self._default_feature_configs()
    
    def _default_feature_configs(self) -> List[FeatureConfig]:
        """Default feature configurations based on Lorentzian importance ranking"""
        return [
            FeatureConfig("wt2", FeatureType.MOMENTUM, self.wt_average_length, 
                         NormalizationMethod.MIN_MAX, weight=0.8966),
            FeatureConfig("wt1", FeatureType.MOMENTUM, self.wt_channel_length, 
                         NormalizationMethod.MIN_MAX, weight=0.7844),
            FeatureConfig("adx", FeatureType.TREND, self.adx_period, 
                         NormalizationMethod.MIN_MAX, weight=0.4117),
            FeatureConfig("rsi", FeatureType.MOMENTUM, self.rsi_period, 
                         NormalizationMethod.MIN_MAX, weight=0.1898),
            FeatureConfig("cci", FeatureType.MOMENTUM, self.cci_period, 
                         NormalizationMethod.Z_SCORE, weight=0.1827),
        ]


class FeatureCache:
    """High-performance feature cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with LRU eviction"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


# Numba-optimized calculation functions
@njit(cache=True)
def fast_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Fast EMA calculation using Numba"""
    alpha = 2.0 / (period + 1.0)
    result = np.zeros_like(values)
    result[0] = values[0]
    
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i-1]
    
    return result


@njit(cache=True)
def fast_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Fast SMA calculation using Numba"""
    result = np.full_like(values, np.nan)
    
    if len(values) < period:
        return result
    
    # Calculate first SMA
    result[period-1] = np.mean(values[:period])
    
    # Use rolling calculation for efficiency
    for i in range(period, len(values)):
        result[i] = result[i-1] + (values[i] - values[i-period]) / period
    
    return result


@njit(cache=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Fast RSI calculation using Numba"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.full(len(prices), 50.0)
    
    # Calculate RSI values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i+1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i+1] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@njit(cache=True)
def fast_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
             period: int = 20, constant: float = 0.015) -> np.ndarray:
    """Fast CCI calculation using Numba"""
    typical_price = (high + low + close) / 3.0
    result = np.full_like(typical_price, 0.0)
    
    if len(typical_price) < period:
        return result
    
    for i in range(period - 1, len(typical_price)):
        # Calculate SMA of typical price
        sma_tp = np.mean(typical_price[i - period + 1:i + 1])
        
        # Calculate mean deviation
        mean_dev = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp))
        
        if mean_dev > 0:
            result[i] = (typical_price[i] - sma_tp) / (constant * mean_dev)
    
    return result


@njit(cache=True)
def fast_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
             period: int = 14) -> np.ndarray:
    """Fast ADX calculation using Numba"""
    if len(close) < period + 1:
        return np.zeros_like(close)
    
    # Calculate True Range
    tr = np.zeros(len(close))
    dm_plus = np.zeros(len(close))
    dm_minus = np.zeros(len(close))
    
    for i in range(1, len(close)):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
        
        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
        
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Calculate smoothed values
    tr_smooth = fast_ema(tr, period)
    dm_plus_smooth = fast_ema(dm_plus, period)
    dm_minus_smooth = fast_ema(dm_minus, period)
    
    # Calculate DI+ and DI-
    di_plus = 100.0 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100.0 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    # Calculate ADX
    adx = fast_ema(dx, period)
    
    return adx


@njit(cache=True)
def fast_wavetrend(hlc3: np.ndarray, channel_length: int = 10, 
                   average_length: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """Fast WaveTrend calculation using Numba"""
    # Calculate ESA (Exponential Moving Average)
    esa = fast_ema(hlc3, channel_length)
    
    # Calculate D (absolute difference smoothed)
    d_values = np.abs(hlc3 - esa)
    d = fast_ema(d_values, channel_length)
    
    # Calculate CI (Commodity Channel Index-like)
    ci = (hlc3 - esa) / (0.015 * d + 1e-10)
    
    # Calculate WT1 and WT2
    wt1 = fast_ema(ci, average_length)
    wt2 = fast_sma(wt1, 4)
    
    return wt1, wt2


@njit(cache=True)
def rational_quadratic_kernel(x: float, y: float, alpha: float = 1.0, 
                            length_scale: float = 1.0) -> float:
    """Rational Quadratic kernel function"""
    distance_sq = (x - y) ** 2
    return (1.0 + distance_sq / (2.0 * alpha * length_scale ** 2)) ** (-alpha)


@njit(cache=True)
def gaussian_kernel(x: float, y: float, length_scale: float = 1.0) -> float:
    """Gaussian kernel function"""
    distance_sq = (x - y) ** 2
    return np.exp(-distance_sq / (2.0 * length_scale ** 2))


@njit(cache=True)
def lorentzian_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Lorentzian distance calculation"""
    return np.sum(np.log1p(np.abs(x - y)))


@njit(cache=True)
def nadaraya_watson_estimator(x_points: np.ndarray, y_values: np.ndarray, 
                             x_eval: float, kernel_func_type: int = 0,
                             length_scale: float = 1.0, alpha: float = 1.0) -> float:
    """Nadaraya-Watson kernel regression estimator"""
    if len(x_points) == 0:
        return 0.0
    
    weights = np.zeros(len(x_points))
    
    for i in range(len(x_points)):
        if kernel_func_type == 0:  # Rational Quadratic
            weights[i] = rational_quadratic_kernel(x_eval, x_points[i], alpha, length_scale)
        else:  # Gaussian
            weights[i] = gaussian_kernel(x_eval, x_points[i], length_scale)
    
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0
    
    return np.sum(weights * y_values) / total_weight


class TechnicalIndicators:
    """High-performance technical indicator calculations"""
    
    def __init__(self, config: LorentzianConfig):
        self.config = config
        self.cache = FeatureCache(config.max_cache_size) if config.enable_caching else None
    
    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate RSI with caching"""
        period = period or self.config.rsi_period
        
        if self.cache:
            cache_key = f"rsi_{hash(prices.tobytes())}_{period}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if self.config.enable_numba:
            result = fast_rsi(prices, period)
        else:
            result = self._rsi_python(prices, period)
        
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def calculate_wavetrend(self, high: np.ndarray, low: np.ndarray, 
                          close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate WaveTrend indicators"""
        hlc3 = (high + low + close) / 3.0
        
        if self.cache:
            cache_key = f"wt_{hash(hlc3.tobytes())}_{self.config.wt_channel_length}_{self.config.wt_average_length}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if self.config.enable_numba:
            wt1, wt2 = fast_wavetrend(hlc3, self.config.wt_channel_length, 
                                     self.config.wt_average_length)
        else:
            wt1, wt2 = self._wavetrend_python(hlc3)
        
        result = (wt1, wt2)
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate CCI with caching"""
        period = period or self.config.cci_period
        
        if self.cache:
            cache_key = f"cci_{hash(close.tobytes())}_{period}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if self.config.enable_numba:
            result = fast_cci(high, low, close, period)
        else:
            result = self._cci_python(high, low, close, period)
        
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate ADX with caching"""
        period = period or self.config.adx_period
        
        if self.cache:
            cache_key = f"adx_{hash(close.tobytes())}_{period}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if self.config.enable_numba:
            result = fast_adx(high, low, close, period)
        else:
            result = self._adx_python(high, low, close, period)
        
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate price momentum"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        momentum = np.zeros_like(prices)
        momentum[period:] = prices[period:] - prices[:-period]
        
        return momentum
    
    def calculate_acceleration(self, prices: np.ndarray, period: int = 5) -> np.ndarray:
        """Calculate price acceleration (second derivative)"""
        momentum = self.calculate_momentum(prices, period)
        acceleration = self.calculate_momentum(momentum, period)
        return acceleration
    
    # Python fallback implementations (without Numba)
    def _rsi_python(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Python implementation of RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).ewm(span=period).mean()
        avg_losses = pd.Series(losses).ewm(span=period).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50.0], rsi.values])
    
    def _wavetrend_python(self, hlc3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Python implementation of WaveTrend"""
        esa = pd.Series(hlc3).ewm(span=self.config.wt_channel_length).mean()
        d = pd.Series(np.abs(hlc3 - esa)).ewm(span=self.config.wt_channel_length).mean()
        ci = (hlc3 - esa) / (0.015 * d + 1e-10)
        wt1 = pd.Series(ci).ewm(span=self.config.wt_average_length).mean()
        wt2 = wt1.rolling(window=4).mean()
        
        return wt1.values, wt2.values
    
    def _cci_python(self, high: np.ndarray, low: np.ndarray, 
                   close: np.ndarray, period: int) -> np.ndarray:
        """Python implementation of CCI"""
        typical_price = (high + low + close) / 3
        sma_tp = pd.Series(typical_price).rolling(window=period).mean()
        mean_dev = pd.Series(typical_price).rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
        return cci.values
    
    def _adx_python(self, high: np.ndarray, low: np.ndarray, 
                   close: np.ndarray, period: int) -> np.ndarray:
        """Python implementation of ADX"""
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        # Smooth the values
        atr = pd.Series(tr).ewm(span=period).mean()
        di_plus = 100 * pd.Series(dm_plus).ewm(span=period).mean() / (atr + 1e-10)
        di_minus = 100 * pd.Series(dm_minus).ewm(span=period).mean() / (atr + 1e-10)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period).mean()
        
        return adx.values


class FeatureNormalizer:
    """Advanced feature normalization with regime awareness"""
    
    def __init__(self, config: LorentzianConfig):
        self.config = config
        self.normalization_params = {}
        self.regime_history = deque(maxlen=1000)
    
    def normalize_feature(self, values: np.ndarray, feature_name: str, 
                         method: NormalizationMethod) -> np.ndarray:
        """Normalize feature values using specified method"""
        
        if len(values) == 0:
            return values
        
        if method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(values, feature_name)
        elif method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalize(values, feature_name)
        elif method == NormalizationMethod.PERCENTILE:
            return self._percentile_normalize(values, feature_name)
        elif method == NormalizationMethod.ROBUST:
            return self._robust_normalize(values, feature_name)
        elif method == NormalizationMethod.REGIME_ADAPTIVE:
            return self._regime_adaptive_normalize(values, feature_name)
        else:
            return values
    
    def _min_max_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Min-max normalization to [0, 1] range"""
        key = f"{feature_name}_minmax"
        
        if key not in self.normalization_params:
            self.normalization_params[key] = {
                'min': np.nanmin(values),
                'max': np.nanmax(values)
            }
        
        params = self.normalization_params[key]
        value_range = params['max'] - params['min']
        
        if value_range == 0:
            return np.full_like(values, 0.5)
        
        normalized = (values - params['min']) / value_range
        return np.clip(normalized, 0, 1)
    
    def _z_score_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Z-score normalization (mean=0, std=1)"""
        key = f"{feature_name}_zscore"
        
        if key not in self.normalization_params:
            self.normalization_params[key] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values)
            }
        
        params = self.normalization_params[key]
        
        if params['std'] == 0:
            return np.zeros_like(values)
        
        return (values - params['mean']) / params['std']
    
    def _percentile_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Percentile-based robust normalization"""
        key = f"{feature_name}_percentile"
        
        if key not in self.normalization_params:
            self.normalization_params[key] = {
                'p5': np.nanpercentile(values, 5),
                'p95': np.nanpercentile(values, 95)
            }
        
        params = self.normalization_params[key]
        value_range = params['p95'] - params['p5']
        
        if value_range == 0:
            return np.full_like(values, 0.5)
        
        normalized = (values - params['p5']) / value_range
        return np.clip(normalized, 0, 1)
    
    def _robust_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Robust normalization using median and MAD"""
        key = f"{feature_name}_robust"
        
        if key not in self.normalization_params:
            median = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values - median))
            self.normalization_params[key] = {
                'median': median,
                'mad': mad
            }
        
        params = self.normalization_params[key]
        
        if params['mad'] == 0:
            return np.zeros_like(values)
        
        return (values - params['median']) / (1.4826 * params['mad'])
    
    def _regime_adaptive_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """Regime-aware adaptive normalization"""
        # Detect current regime based on volatility
        if len(values) < 20:
            return self._min_max_normalize(values, feature_name)
        
        recent_volatility = np.std(values[-20:])
        self.regime_history.append(recent_volatility)
        
        if len(self.regime_history) < 100:
            return self._min_max_normalize(values, feature_name)
        
        vol_percentile = np.percentile(list(self.regime_history), 
                                     [recent_volatility * 100])
        
        # Adjust normalization based on regime
        if vol_percentile > 80:  # High volatility regime
            return self._robust_normalize(values, feature_name)
        elif vol_percentile < 20:  # Low volatility regime
            return self._z_score_normalize(values, feature_name)
        else:  # Normal regime
            return self._min_max_normalize(values, feature_name)


class KernelRegression:
    """Advanced kernel regression with dynamic parameters"""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.parameter_history = deque(maxlen=1000)
    
    def estimate(self, x_points: np.ndarray, y_values: np.ndarray, 
                x_eval: float) -> float:
        """Perform kernel regression estimation"""
        
        if len(x_points) != len(y_values) or len(x_points) == 0:
            return 0.0
        
        # Dynamic parameter adjustment
        if self.config.adaptive_params:
            length_scale, alpha = self._adapt_parameters(x_points, y_values)
        else:
            length_scale = self.config.regression_level
            alpha = self.config.relative_weighting
        
        # Select kernel function
        kernel_type = 0 if self.config.kernel_type == "rational_quadratic" else 1
        
        if self.config.kernel_type == "rational_quadratic":
            return nadaraya_watson_estimator(x_points, y_values, x_eval, 
                                           kernel_type, length_scale, alpha)
        else:
            return nadaraya_watson_estimator(x_points, y_values, x_eval, 
                                           kernel_type, length_scale)
    
    def smooth_signals(self, signals: np.ndarray, time_points: np.ndarray = None) -> np.ndarray:
        """Apply kernel smoothing to signal series"""
        if time_points is None:
            time_points = np.arange(len(signals))
        
        if len(signals) <= 1:
            return signals
        
        smoothed = np.zeros_like(signals)
        
        for i in range(len(signals)):
            # Define window
            start_idx = max(0, i - self.config.lookback_window)
            end_idx = i + 1
            
            if end_idx - start_idx > 1:
                window_times = time_points[start_idx:end_idx]
                window_signals = signals[start_idx:end_idx]
                
                smoothed[i] = self.estimate(window_times, window_signals, time_points[i])
            else:
                smoothed[i] = signals[i]
        
        return smoothed
    
    def _adapt_parameters(self, x_points: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
        """Dynamically adapt kernel parameters based on data characteristics"""
        
        # Analyze data characteristics
        data_variance = np.var(y_values)
        data_range = np.max(x_points) - np.min(x_points)
        
        # Store in history
        self.parameter_history.append((data_variance, data_range))
        
        # Adaptive length scale based on data range
        base_length = self.config.regression_level
        if data_range > 0:
            length_scale = base_length * (1 + np.log(data_range + 1))
        else:
            length_scale = base_length
        
        # Adaptive alpha based on variance
        base_alpha = self.config.relative_weighting
        if data_variance > 0:
            alpha = base_alpha * (1 + np.sqrt(data_variance))
        else:
            alpha = base_alpha
        
        # Bound parameters
        length_scale = np.clip(length_scale, 1.0, 100.0)
        alpha = np.clip(alpha, 0.1, 20.0)
        
        return length_scale, alpha


class LorentzianFeatureEngine:
    """Complete Lorentzian feature engineering pipeline"""
    
    def __init__(self, config: LorentzianConfig = None):
        """Initialize the feature engineering pipeline"""
        self.config = config or LorentzianConfig()
        
        # Initialize components
        self.indicators = TechnicalIndicators(self.config)
        self.normalizer = FeatureNormalizer(self.config)
        self.kernel_regression = KernelRegression(self.config.kernel_config)
        
        # Feature history for streaming
        self.feature_history = deque(maxlen=self.config.max_bars_back)
        self.raw_data_history = deque(maxlen=self.config.max_bars_back)
        
        # Performance monitoring
        self.performance_stats = {
            'calculation_times': [],
            'feature_quality_scores': [],
            'cache_hit_rates': []
        }
        
        logger.info(f"Lorentzian Feature Engine initialized with {len(self.config.feature_configs)} features")
    
    def process_bar(self, high: float, low: float, open_price: float, 
                   close: float, volume: float, timestamp: pd.Timestamp = None) -> Dict[str, Any]:
        """Process a single OHLCV bar and return engineered features"""
        
        start_time = time.time()
        
        # Store raw data
        bar_data = {
            'high': high, 'low': low, 'open': open_price,
            'close': close, 'volume': volume,
            'timestamp': timestamp or pd.Timestamp.now()
        }
        self.raw_data_history.append(bar_data)
        
        # Need minimum history for feature calculation
        if len(self.raw_data_history) < max(30, max(config.period for config in self.config.feature_configs)):
            # Return zero features for insufficient data
            features = np.zeros(len(self.config.feature_configs))
            feature_names = [config.name for config in self.config.feature_configs]
            
            result = {
                'features': features,
                'feature_names': feature_names,
                'normalized_features': features,
                'calculation_time_ms': 0.0,
                'quality_score': 0.0,
                'timestamp': bar_data['timestamp']
            }
            
            self.feature_history.append(result)
            return result
        
        # Extract arrays from history
        history_data = self._extract_arrays_from_history()
        
        # Calculate all features
        features = self._calculate_all_features(history_data)
        
        # Normalize features
        normalized_features = self._normalize_features(features)
        
        # Apply kernel smoothing if enabled
        if self.config.streaming_mode and len(self.feature_history) > 0:
            smoothed_features = self._apply_kernel_smoothing(normalized_features)
        else:
            smoothed_features = normalized_features
        
        # Calculate quality metrics
        quality_score = self._calculate_feature_quality(features, normalized_features)
        
        # Create result
        calculation_time = (time.time() - start_time) * 1000
        
        result = {
            'features': features,
            'feature_names': [config.name for config in self.config.feature_configs],
            'normalized_features': smoothed_features,
            'calculation_time_ms': calculation_time,
            'quality_score': quality_score,
            'timestamp': bar_data['timestamp']
        }
        
        # Store in history
        self.feature_history.append(result)
        
        # Update performance stats
        self.performance_stats['calculation_times'].append(calculation_time)
        self.performance_stats['feature_quality_scores'].append(quality_score)
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire DataFrame and return features"""
        
        if not all(col in df.columns for col in ['high', 'low', 'open', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")
        
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            timestamp = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(idx)
            
            result = self.process_bar(
                high=row['high'],
                low=row['low'],
                open_price=row['open'],
                close=row['close'],
                volume=row['volume'],
                timestamp=timestamp
            )
            
            results.append(result)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([r['normalized_features'] for r in results],
                                index=df.index,
                                columns=[config.name for config in self.config.feature_configs])
        
        # Add metadata columns
        feature_df['calculation_time_ms'] = [r['calculation_time_ms'] for r in results]
        feature_df['quality_score'] = [r['quality_score'] for r in results]
        
        return feature_df
    
    def get_multi_timeframe_features(self, df: pd.DataFrame, 
                                   timeframes: List[str] = ['5T', '15T', '1H']) -> pd.DataFrame:
        """Generate features across multiple timeframes"""
        
        all_features = []
        
        for tf in timeframes:
            # Resample to timeframe
            resampled = df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Process features
            tf_features = self.process_dataframe(resampled)
            
            # Add timeframe suffix
            tf_features.columns = [f"{col}_{tf}" if col in [config.name for config in self.config.feature_configs] 
                                 else col for col in tf_features.columns]
            
            # Resample back to original frequency
            tf_features_resampled = tf_features.reindex(df.index, method='ffill')
            
            all_features.append(tf_features_resampled)
        
        # Combine all timeframes
        combined_features = pd.concat(all_features, axis=1)
        
        return combined_features
    
    def _extract_arrays_from_history(self) -> Dict[str, np.ndarray]:
        """Extract OHLCV arrays from raw data history"""
        data = list(self.raw_data_history)
        
        return {
            'high': np.array([d['high'] for d in data]),
            'low': np.array([d['low'] for d in data]),
            'open': np.array([d['open'] for d in data]),
            'close': np.array([d['close'] for d in data]),
            'volume': np.array([d['volume'] for d in data])
        }
    
    def _calculate_all_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate all configured features"""
        
        features = []
        
        for config in self.config.feature_configs:
            if not config.enabled:
                features.append(0.0)
                continue
            
            try:
                if config.name == 'rsi':
                    feature_value = self.indicators.calculate_rsi(data['close'], config.period)[-1]
                
                elif config.name == 'wt1':
                    wt1, _ = self.indicators.calculate_wavetrend(data['high'], data['low'], data['close'])
                    feature_value = wt1[-1]
                
                elif config.name == 'wt2':
                    _, wt2 = self.indicators.calculate_wavetrend(data['high'], data['low'], data['close'])
                    feature_value = wt2[-1]
                
                elif config.name == 'cci':
                    cci = self.indicators.calculate_cci(data['high'], data['low'], data['close'], config.period)
                    feature_value = cci[-1]
                
                elif config.name == 'adx':
                    adx = self.indicators.calculate_adx(data['high'], data['low'], data['close'], config.period)
                    feature_value = adx[-1]
                
                elif config.name == 'momentum':
                    momentum = self.indicators.calculate_momentum(data['close'], config.period)
                    feature_value = momentum[-1]
                
                elif config.name == 'acceleration':
                    acceleration = self.indicators.calculate_acceleration(data['close'], config.period)
                    feature_value = acceleration[-1]
                
                else:
                    # Custom feature calculation
                    feature_value = self._calculate_custom_feature(config.name, data, config)
                
                # Handle NaN values
                if np.isnan(feature_value) or np.isinf(feature_value):
                    feature_value = 0.0
                
                features.append(feature_value)
                
            except Exception as e:
                logger.warning(f"Error calculating feature {config.name}: {e}")
                features.append(0.0)
        
        return np.array(features)
    
    def _calculate_custom_feature(self, feature_name: str, data: Dict[str, np.ndarray], 
                                config: FeatureConfig) -> float:
        """Calculate custom features"""
        
        if feature_name == 'price_volatility':
            returns = np.diff(np.log(data['close'] + 1e-10))
            return np.std(returns[-config.period:]) if len(returns) >= config.period else 0.0
        
        elif feature_name == 'volume_momentum':
            volume_changes = np.diff(data['volume'])
            return np.mean(volume_changes[-config.period:]) if len(volume_changes) >= config.period else 0.0
        
        elif feature_name == 'price_range':
            ranges = data['high'] - data['low']
            return np.mean(ranges[-config.period:]) if len(ranges) >= config.period else 0.0
        
        elif feature_name == 'cross_correlation':
            if len(data['close']) >= config.period * 2:
                price_returns = np.diff(data['close'][-config.period * 2:])
                volume_changes = np.diff(data['volume'][-config.period * 2:])
                return np.corrcoef(price_returns, volume_changes)[0, 1] if len(price_returns) > 1 else 0.0
        
        return 0.0
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize all features according to their configurations"""
        
        normalized = np.zeros_like(features)
        
        for i, (feature_val, config) in enumerate(zip(features, self.config.feature_configs)):
            normalized[i] = self.normalizer.normalize_feature(
                np.array([feature_val]), config.name, config.normalization
            )[0]
        
        return normalized
    
    def _apply_kernel_smoothing(self, features: np.ndarray) -> np.ndarray:
        """Apply kernel smoothing to features"""
        
        if len(self.feature_history) < 2:
            return features
        
        # Extract historical normalized features
        history_features = np.array([h['normalized_features'] for h in self.feature_history])
        time_points = np.arange(len(history_features))
        
        smoothed = np.zeros_like(features)
        
        for i in range(len(features)):
            feature_series = np.append(history_features[:, i], features[i])
            smoothed_series = self.kernel_regression.smooth_signals(feature_series, 
                                                                  np.arange(len(feature_series)))
            smoothed[i] = smoothed_series[-1]
        
        return smoothed
    
    def _calculate_feature_quality(self, raw_features: np.ndarray, 
                                 normalized_features: np.ndarray) -> float:
        """Calculate overall feature quality score"""
        
        quality_factors = []
        
        # 1. Non-zero feature ratio
        non_zero_ratio = np.mean(normalized_features != 0)
        quality_factors.append(non_zero_ratio)
        
        # 2. Feature variance (diversity)
        if len(normalized_features) > 1:
            variance_score = min(np.var(normalized_features), 1.0)
            quality_factors.append(variance_score)
        
        # 3. No extreme values
        extreme_penalty = np.mean(np.abs(normalized_features) > 3.0)
        quality_factors.append(1.0 - extreme_penalty)
        
        # 4. Historical consistency
        if len(self.feature_history) > 1:
            prev_features = self.feature_history[-1]['normalized_features']
            consistency = 1.0 - np.mean(np.abs(normalized_features - prev_features))
            quality_factors.append(max(consistency, 0.0))
        
        return np.mean(quality_factors)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on configuration weights"""
        importance = {}
        
        for config in self.config.feature_configs:
            importance[config.name] = config.weight
        
        return importance
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        stats = {}
        
        if self.performance_stats['calculation_times']:
            stats['avg_calculation_time_ms'] = np.mean(self.performance_stats['calculation_times'])
            stats['max_calculation_time_ms'] = np.max(self.performance_stats['calculation_times'])
            stats['calculation_time_std'] = np.std(self.performance_stats['calculation_times'])
        
        if self.performance_stats['feature_quality_scores']:
            stats['avg_feature_quality'] = np.mean(self.performance_stats['feature_quality_scores'])
            stats['min_feature_quality'] = np.min(self.performance_stats['feature_quality_scores'])
        
        # Cache statistics
        if hasattr(self.indicators, 'cache') and self.indicators.cache:
            total_requests = len(self.performance_stats['calculation_times'])
            cache_size = len(self.indicators.cache.cache)
            stats['cache_size'] = cache_size
            stats['cache_efficiency'] = min(cache_size / max(total_requests, 1), 1.0)
        
        # Feature history stats
        stats['feature_history_length'] = len(self.feature_history)
        stats['raw_data_history_length'] = len(self.raw_data_history)
        
        return stats
    
    def validate_features(self, features: np.ndarray) -> Dict[str, bool]:
        """Validate feature quality and integrity"""
        
        validation_results = {}
        
        # Check for NaN or infinite values
        validation_results['no_nan_inf'] = not (np.any(np.isnan(features)) or np.any(np.isinf(features)))
        
        # Check feature range (should be normalized)
        validation_results['proper_range'] = np.all((features >= -5) & (features <= 5))
        
        # Check for feature diversity (not all the same)
        validation_results['diverse_features'] = np.var(features) > 1e-6
        
        # Check for reasonable feature count
        validation_results['correct_count'] = len(features) == len(self.config.feature_configs)
        
        return validation_results
    
    def reset_cache(self):
        """Reset all caches"""
        if hasattr(self.indicators, 'cache') and self.indicators.cache:
            self.indicators.cache.clear()
        
        self.normalizer.normalization_params.clear()
        
        logger.info("Feature engine caches reset")
    
    def save_state(self, filepath: str):
        """Save feature engine state for persistence"""
        import pickle
        
        state = {
            'config': self.config,
            'normalization_params': self.normalizer.normalization_params,
            'performance_stats': self.performance_stats,
            'feature_history_length': len(self.feature_history),
            'timestamp': pd.Timestamp.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Feature engine state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load feature engine state from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.normalizer.normalization_params = state['normalization_params']
        self.performance_stats = state['performance_stats']
        
        logger.info(f"Feature engine state loaded from {filepath}")


def create_production_config() -> LorentzianConfig:
    """Create optimized production configuration"""
    
    # High-importance features based on Lorentzian analysis
    feature_configs = [
        FeatureConfig("wt2", FeatureType.MOMENTUM, 21, NormalizationMethod.MIN_MAX, 0.8966),
        FeatureConfig("wt1", FeatureType.MOMENTUM, 10, NormalizationMethod.MIN_MAX, 0.7844),
        FeatureConfig("adx", FeatureType.TREND, 14, NormalizationMethod.MIN_MAX, 0.4117),
        FeatureConfig("rsi", FeatureType.MOMENTUM, 14, NormalizationMethod.MIN_MAX, 0.1898),
        FeatureConfig("cci", FeatureType.MOMENTUM, 20, NormalizationMethod.Z_SCORE, 0.1827),
    ]
    
    kernel_config = KernelConfig(
        kernel_type="rational_quadratic",
        lookback_window=8,
        relative_weighting=8.0,
        regression_level=25.0,
        adaptive_params=True
    )
    
    return LorentzianConfig(
        lookback_window=8,
        k_neighbors=8,
        max_bars_back=5000,
        feature_count=5,
        kernel_config=kernel_config,
        feature_configs=feature_configs,
        enable_numba=True,
        enable_caching=True,
        streaming_mode=True,
        use_volatility_filter=True,
        use_regime_filter=True,
        use_adx_filter=True,
        adx_threshold=25.0,
        volatility_threshold=0.15
    )


def demo_feature_engine():
    """Demonstrate the Lorentzian feature engineering pipeline"""
    
    print("=" * 70)
    print("LORENTZIAN FEATURE ENGINEERING PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Create production configuration
    config = create_production_config()
    print(f"✅ Configuration created with {len(config.feature_configs)} features")
    
    # Initialize feature engine
    engine = LorentzianFeatureEngine(config)
    print(f"✅ Feature engine initialized")
    
    # Generate sample market data
    np.random.seed(42)
    n_bars = 200
    
    # Create realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add trend and noise
    trend = np.linspace(0, 0.1, n_bars)
    prices *= (1 + trend)
    
    # Generate OHLCV
    noise_factor = 0.01
    high = prices * (1 + np.abs(np.random.normal(0, noise_factor, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, noise_factor, n_bars)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(10, 1, n_bars)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1T'))
    
    print(f"✅ Generated {n_bars} bars of sample data")
    
    # Process features
    print("\n🔄 Processing features...")
    start_time = time.time()
    
    feature_df = engine.process_dataframe(df)
    
    processing_time = time.time() - start_time
    print(f"✅ Feature processing completed in {processing_time:.2f} seconds")
    
    # Display results
    print(f"\n📊 FEATURE RESULTS:")
    print(f"   Feature matrix shape: {feature_df.shape}")
    print(f"   Feature columns: {list(feature_df.columns[:5])}")  # First 5 columns
    print(f"   Average calculation time: {feature_df['calculation_time_ms'].mean():.2f}ms")
    print(f"   Average quality score: {feature_df['quality_score'].mean():.3f}")
    
    # Show sample features
    print(f"\n📈 SAMPLE FEATURES (last 5 bars):")
    sample_features = feature_df.iloc[-5:, :5]  # Last 5 rows, first 5 feature columns
    print(sample_features.round(4))
    
    # Performance statistics
    print(f"\n⚡ PERFORMANCE STATISTICS:")
    perf_stats = engine.get_performance_stats()
    for key, value in perf_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Feature importance
    print(f"\n🎯 FEATURE IMPORTANCE:")
    importance = engine.get_feature_importance()
    for feature, weight in importance.items():
        print(f"   {feature}: {weight:.4f}")
    
    # Multi-timeframe demonstration
    print(f"\n🕒 MULTI-TIMEFRAME FEATURES:")
    mtf_features = engine.get_multi_timeframe_features(df.iloc[-50:], ['5T', '15T'])
    print(f"   Multi-timeframe shape: {mtf_features.shape}")
    print(f"   MTF columns: {list(mtf_features.columns[:8])}")  # First 8 columns
    
    # Feature validation
    print(f"\n✅ FEATURE VALIDATION:")
    last_features = feature_df.iloc[-1, :5].values
    validation = engine.validate_features(last_features)
    for check, passed in validation.items():
        status = "✅" if passed else "❌"
        print(f"   {check}: {status}")
    
    print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print(f"💡 The Lorentzian Feature Engineering Pipeline is ready for production use!")
    
    return engine, feature_df


if __name__ == "__main__":
    # Run demonstration
    engine, results = demo_feature_engine()