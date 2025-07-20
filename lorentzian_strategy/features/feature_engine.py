"""
Feature Engineering System for Lorentzian Trading Strategy
Implements rolling windows, technical indicators, and feature buffers.
"""

import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import deque

from ..core.base import BaseFeatureExtractor, ValidationMixin, CacheableMixin
from ..config.config import get_config
from ..utils.logging_config import get_logger, log_function_call


@dataclass
class FeatureBuffer:
    """Rolling buffer for efficient feature calculation"""
    size: int
    data: deque
    feature_name: str
    
    def __post_init__(self):
        self.data = deque(maxlen=self.size)
    
    def add(self, value: float):
        """Add new value to buffer"""
        self.data.append(value)
    
    def get_array(self) -> np.ndarray:
        """Get buffer as numpy array"""
        return np.array(self.data)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.data) == self.size
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical measures from buffer"""
        if not self.data:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        arr = self.get_array()
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }


class BufferManager:
    """Manages multiple feature buffers efficiently"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.buffers: Dict[str, FeatureBuffer] = {}
        self.logger = get_logger("lorentzian_strategy.features.buffers")
    
    def create_buffer(self, name: str, size: int) -> FeatureBuffer:
        """Create a new feature buffer"""
        buffer = FeatureBuffer(size=size, data=deque(), feature_name=name)
        self.buffers[name] = buffer
        self.logger.debug(f"Created buffer '{name}' with size {size}")
        return buffer
    
    def add_value(self, buffer_name: str, value: float):
        """Add value to specific buffer"""
        if buffer_name in self.buffers:
            self.buffers[buffer_name].add(value)
        else:
            raise KeyError(f"Buffer '{buffer_name}' not found")
    
    def get_buffer(self, name: str) -> FeatureBuffer:
        """Get buffer by name"""
        return self.buffers.get(name)
    
    def get_all_ready_buffers(self) -> Dict[str, np.ndarray]:
        """Get all buffers that are full and ready"""
        ready = {}
        for name, buffer in self.buffers.items():
            if buffer.is_full():
                ready[name] = buffer.get_array()
        return ready
    
    def reset_all(self):
        """Reset all buffers"""
        for buffer in self.buffers.values():
            buffer.data.clear()


# Numba-optimized technical indicators
@nb.jit(nopython=True, cache=True)
def rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Numba for speed"""
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    
    rsi = np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    
    # Initial calculation
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Rolling calculation
    for i in range(period + 1, len(prices)):
        gain = gains[i-1] if gains[i-1] > 0 else 0
        loss = losses[i-1] if losses[i-1] > 0 else 0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@nb.jit(nopython=True, cache=True)
def sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using Numba"""
    if len(prices) < period:
        return np.full(len(prices), np.mean(prices))
    
    sma = np.full(len(prices), np.nan)
    
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    
    # Fill initial values with expanding mean
    for i in range(period - 1):
        sma[i] = np.mean(prices[:i + 1])
    
    return sma


@nb.jit(nopython=True, cache=True)
def ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average using Numba"""
    if len(prices) == 0:
        return np.array([])
    
    alpha = 2.0 / (period + 1.0)
    ema = np.full(len(prices), prices[0])
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@nb.jit(nopython=True, cache=True)
def atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range using Numba"""
    if len(high) < 2:
        return np.zeros(len(high))
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return ema_numba(tr, period)


@nb.jit(nopython=True, cache=True)
def wt_oscillator_numba(hlc3: np.ndarray, channel_length: int = 9, avg_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Wave Trend oscillator using Numba"""
    if len(hlc3) < max(channel_length, avg_length):
        wt1 = np.zeros(len(hlc3))
        wt2 = np.zeros(len(hlc3))
        return wt1, wt2
    
    # Calculate EMA of HLC3
    esa = ema_numba(hlc3, channel_length)
    
    # Calculate absolute difference
    abs_diff = np.abs(hlc3 - esa)
    d = ema_numba(abs_diff, channel_length)
    
    # Calculate Wave Trend
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ema_numba(ci, avg_length)
    wt2 = sma_numba(wt1, 4)
    
    return wt1, wt2


class TechnicalIndicators:
    """Collection of technical indicators optimized for Lorentzian features"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.use_numba = self.config.optimization.use_numba
    
    def rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate RSI"""
        period = period or self.config.features.rsi_length
        if self.use_numba:
            return rsi_numba(prices, period)
        else:
            # Pandas fallback
            deltas = pd.Series(prices).diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            avg_gains = gains.rolling(period).mean()
            avg_losses = losses.rolling(period).mean()
            rs = avg_gains / avg_losses
            return (100 - (100 / (1 + rs))).fillna(50).values
    
    def wave_trend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Wave Trend oscillator"""
        hlc3 = (high + low + close) / 3
        if self.use_numba:
            return wt_oscillator_numba(
                hlc3, 
                self.config.features.wt_channel_length,
                self.config.features.wt_average_length
            )
        else:
            # Pandas fallback
            esa = pd.Series(hlc3).ewm(span=self.config.features.wt_channel_length).mean()
            d = pd.Series(np.abs(hlc3 - esa)).ewm(span=self.config.features.wt_channel_length).mean()
            ci = (hlc3 - esa) / (0.015 * d)
            wt1 = pd.Series(ci).ewm(span=self.config.features.wt_average_length).mean()
            wt2 = wt1.rolling(4).mean()
            return wt1.fillna(0).values, wt2.fillna(0).values
    
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate Average True Range"""
        period = period or self.config.features.atr_length
        if self.use_numba:
            return atr_numba(high, low, close, period)
        else:
            # Pandas fallback
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.ewm(span=period).mean().fillna(0).values
    
    def returns(self, prices: np.ndarray, periods: List[int] = None) -> Dict[str, np.ndarray]:
        """Calculate returns for multiple periods"""
        periods = periods or [1, 2, 3, 5, 8, 13, 21]
        results = {}
        
        for period in periods:
            if len(prices) > period:
                ret = np.log(prices[period:] / prices[:-period])
                # Pad with zeros for the initial values
                padded = np.zeros(len(prices))
                padded[period:] = ret
                results[f'return_{period}'] = padded
            else:
                results[f'return_{period}'] = np.zeros(len(prices))
        
        return results


class LorentzianFeatureExtractor(BaseFeatureExtractor, ValidationMixin, CacheableMixin):
    """Feature extractor specifically designed for Lorentzian classification"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.indicators = TechnicalIndicators(config)
        self.buffer_manager = BufferManager(config)
        self._setup_buffers()
        self.feature_names = self._get_feature_names()
    
    def _setup(self):
        """Setup feature extractor"""
        self._setup_buffers()
        self.logger.info("Lorentzian feature extractor initialized")
    
    def _setup_buffers(self):
        """Setup feature buffers"""
        config = self.config.features
        
        # Create buffers for different lookback periods
        self.buffer_manager.create_buffer('rsi', config.long_window)
        self.buffer_manager.create_buffer('wt1', config.long_window)
        self.buffer_manager.create_buffer('wt2', config.long_window)
        self.buffer_manager.create_buffer('atr', config.long_window)
        self.buffer_manager.create_buffer('close', config.long_window)
        self.buffer_manager.create_buffer('volume', config.long_window)
        
        # Return buffers
        for period in [1, 2, 3, 5, 8, 13, 21]:
            self.buffer_manager.create_buffer(f'return_{period}', config.medium_window)
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        base_features = [
            'rsi', 'rsi_normalized',
            'wt1', 'wt2', 'wt_cross',
            'atr_normalized',
            'return_1', 'return_2', 'return_3', 'return_5', 'return_8', 'return_13', 'return_21',
            'volume_ratio',
            'price_position',
            'volatility_regime'
        ]
        return base_features
    
    @log_function_call(logger=logging.getLogger("lorentzian_strategy.features"))
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from OHLCV data"""
        if len(data) < self.config.features.long_window:
            # Not enough data, return zeros
            return np.zeros(len(self.feature_names))
        
        # Convert to numpy arrays for speed
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values
        
        # Calculate basic indicators
        rsi = self.indicators.rsi(close)
        wt1, wt2 = self.indicators.wave_trend(high, low, close)
        atr = self.indicators.atr(high, low, close)
        returns_dict = self.indicators.returns(close)
        
        # Use the latest values for feature vector
        features = []
        
        # RSI features
        features.append(rsi[-1])
        features.append(self._normalize_rsi(rsi[-1]))
        
        # Wave Trend features
        features.append(wt1[-1])
        features.append(wt2[-1])
        features.append(1.0 if wt1[-1] > wt2[-1] else 0.0)  # Cross signal
        
        # ATR normalized by price
        features.append(atr[-1] / close[-1] if close[-1] > 0 else 0.0)
        
        # Return features
        for period in [1, 2, 3, 5, 8, 13, 21]:
            ret_values = returns_dict[f'return_{period}']
            features.append(ret_values[-1] if len(ret_values) > 0 else 0.0)
        
        # Volume ratio (current vs average)
        if len(volume) >= 20:
            vol_avg = np.mean(volume[-20:])
            features.append(volume[-1] / vol_avg if vol_avg > 0 else 1.0)
        else:
            features.append(1.0)
        
        # Price position within recent range
        if len(close) >= 20:
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            if recent_high > recent_low:
                features.append((close[-1] - recent_low) / (recent_high - recent_low))
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Volatility regime
        if len(atr) >= 20:
            atr_avg = np.mean(atr[-20:])
            features.append(1.0 if atr[-1] > atr_avg else 0.0)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float64)
    
    def _normalize_rsi(self, rsi_value: float) -> float:
        """Normalize RSI to [-1, 1] range"""
        return (rsi_value - 50.0) / 50.0
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def extract_feature_matrix(self, data: pd.DataFrame, lookback: int = None) -> Tuple[np.ndarray, List[str]]:
        """Extract feature matrix for multiple time points"""
        lookback = lookback or self.config.lorentzian.max_bars_back
        
        if len(data) < lookback:
            lookback = len(data)
        
        feature_matrix = []
        valid_timestamps = []
        
        min_required = self.config.features.long_window
        
        for i in range(min_required, len(data)):
            # Extract features for this timestamp
            subset = data.iloc[:i+1]
            try:
                features = self.extract_features(subset)
                if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                    feature_matrix.append(features)
                    valid_timestamps.append(data.iloc[i]['Timestamp'])
            except Exception as e:
                self.logger.warning(f"Failed to extract features at index {i}: {e}")
                continue
        
        if not feature_matrix:
            raise ValueError("No valid features could be extracted")
        
        return np.array(feature_matrix), valid_timestamps
    
    def update_buffers(self, new_data: Dict[str, float]):
        """Update feature buffers with new data point"""
        # This method would be used for real-time feature calculation
        for key, value in new_data.items():
            try:
                self.buffer_manager.add_value(key, value)
            except KeyError:
                pass  # Buffer doesn't exist, skip
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder for future ML integration)"""
        # This would be populated by the ML model
        feature_names = self.get_feature_names()
        return {name: 1.0 / len(feature_names) for name in feature_names}


def create_feature_extractor(config=None) -> LorentzianFeatureExtractor:
    """Factory function to create feature extractor"""
    return LorentzianFeatureExtractor(config)