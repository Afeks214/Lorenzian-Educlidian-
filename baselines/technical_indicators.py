"""
Technical Indicators Module

Efficient vectorized technical indicator calculations for baseline trading strategies.
Designed for high-performance real-time trading applications.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from functools import lru_cache
import warnings

# Suppress numpy warnings for performance
warnings.filterwarnings('ignore', category=RuntimeWarning)


class TechnicalIndicators:
    """
    High-performance technical indicators with vectorized calculations
    
    Features:
    - Vectorized NumPy operations for speed
    - Caching for repeated calculations
    - Memory-efficient implementations
    - Robust handling of edge cases
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize technical indicators calculator
        
        Args:
            cache_size: LRU cache size for indicator calculations
        """
        self.cache_size = cache_size
        self._indicator_cache = {}
        
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Simple Moving Average (SMA)
        
        Args:
            prices: Price array
            period: Moving average period
            
        Returns:
            SMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
            
        # Vectorized convolution approach
        cumsum = np.cumsum(np.insert(prices, 0, 0))
        result = (cumsum[period:] - cumsum[:-period]) / period
        
        # Pad with NaN for first period-1 values
        return np.concatenate([np.full(period - 1, np.nan), result])
    
    @staticmethod
    def ema(prices: np.ndarray, period: int, alpha: Optional[float] = None) -> np.ndarray:
        """
        Exponential Moving Average (EMA)
        
        Args:
            prices: Price array
            period: EMA period
            alpha: Smoothing factor (default: 2/(period+1))
            
        Returns:
            EMA values
        """
        if len(prices) == 0:
            return np.array([])
            
        if alpha is None:
            alpha = 2.0 / (period + 1)
            
        # Initialize with first price
        ema_values = np.zeros(len(prices))
        ema_values[0] = prices[0]
        
        # Vectorized EMA calculation
        for i in range(1, len(prices)):
            ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i-1]
            
        return ema_values
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index (RSI)
        
        Args:
            prices: Price array
            period: RSI period
            
        Returns:
            RSI values (0-100)
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses using EMA
        avg_gains = TechnicalIndicators.ema(gains, period)
        avg_losses = TechnicalIndicators.ema(losses, period)
        
        # Calculate RSI
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi_values = 100 - (100 / (1 + rs))
        
        # Pad with NaN for first value
        return np.concatenate([np.array([np.nan]), rsi_values])
    
    @staticmethod
    def macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Price array
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow_period:
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
            
        # Calculate fast and slow EMAs
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD)
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands
        
        Args:
            prices: Price array
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        if len(prices) < period:
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
            
        # Middle band (SMA)
        middle_band = TechnicalIndicators.sma(prices, period)
        
        # Calculate rolling standard deviation
        rolling_std = np.zeros(len(prices))
        for i in range(period - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - period + 1:i + 1])
            
        # Pad with NaN for first period-1 values
        rolling_std[:period - 1] = np.nan
        
        # Upper and lower bands
        upper_band = middle_band + (std_dev * rolling_std)
        lower_band = middle_band - (std_dev * rolling_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(close) < k_period:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array
            
        # Calculate %K
        k_values = np.zeros(len(close))
        
        for i in range(k_period - 1, len(close)):
            period_high = np.max(high[i - k_period + 1:i + 1])
            period_low = np.min(low[i - k_period + 1:i + 1])
            
            if period_high == period_low:
                k_values[i] = 50.0  # Avoid division by zero
            else:
                k_values[i] = 100 * (close[i] - period_low) / (period_high - period_low)
                
        # Pad with NaN for first k_period-1 values
        k_values[:k_period - 1] = np.nan
        
        # Calculate %D (SMA of %K)
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 14) -> np.ndarray:
        """
        Williams %R
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: Williams %R period
            
        Returns:
            Williams %R values (-100 to 0)
        """
        if len(close) < period:
            return np.full(len(close), np.nan)
            
        williams_r = np.zeros(len(close))
        
        for i in range(period - 1, len(close)):
            period_high = np.max(high[i - period + 1:i + 1])
            period_low = np.min(low[i - period + 1:i + 1])
            
            if period_high == period_low:
                williams_r[i] = -50.0  # Avoid division by zero
            else:
                williams_r[i] = -100 * (period_high - close[i]) / (period_high - period_low)
                
        # Pad with NaN for first period-1 values
        williams_r[:period - 1] = np.nan
        
        return williams_r
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            period: int = 14) -> np.ndarray:
        """
        Average True Range (ATR)
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: ATR period
            
        Returns:
            ATR values
        """
        if len(close) < 2:
            return np.full(len(close), np.nan)
            
        # Calculate True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using EMA
        atr_values = TechnicalIndicators.ema(true_range, period)
        
        return atr_values
    
    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Price Momentum
        
        Args:
            prices: Price array
            period: Momentum period
            
        Returns:
            Momentum values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
            
        # Calculate momentum as price change over period
        momentum_values = np.zeros(len(prices))
        momentum_values[:period] = np.nan
        momentum_values[period:] = prices[period:] - prices[:-period]
        
        return momentum_values
    
    @staticmethod
    def rate_of_change(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of Change (ROC)
        
        Args:
            prices: Price array
            period: ROC period
            
        Returns:
            ROC values (percentage)
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
            
        # Calculate ROC
        roc_values = np.zeros(len(prices))
        roc_values[:period] = np.nan
        
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                roc_values[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
            else:
                roc_values[i] = 0
                
        return roc_values
    
    @staticmethod
    def zscore(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Z-Score (standardized price)
        
        Args:
            prices: Price array
            period: Rolling window period
            
        Returns:
            Z-score values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
            
        # Calculate rolling mean and std
        rolling_mean = TechnicalIndicators.sma(prices, period)
        
        rolling_std = np.zeros(len(prices))
        for i in range(period - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - period + 1:i + 1])
            
        rolling_std[:period - 1] = np.nan
        
        # Calculate Z-score
        zscore_values = np.divide(
            prices - rolling_mean, 
            rolling_std,
            out=np.zeros_like(prices),
            where=rolling_std != 0
        )
        
        return zscore_values


class IndicatorSignals:
    """
    Signal generation from technical indicators
    
    Converts raw indicator values into trading signals
    """
    
    @staticmethod
    def rsi_signals(rsi_values: np.ndarray, oversold: float = 30, 
                   overbought: float = 70) -> np.ndarray:
        """
        Generate RSI-based signals
        
        Args:
            rsi_values: RSI values
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Signal array: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = np.zeros(len(rsi_values))
        
        # Buy signal when RSI crosses above oversold
        signals[rsi_values < oversold] = 1
        
        # Sell signal when RSI crosses above overbought
        signals[rsi_values > overbought] = -1
        
        return signals
    
    @staticmethod
    def macd_signals(macd_line: np.ndarray, signal_line: np.ndarray, 
                    histogram: np.ndarray) -> np.ndarray:
        """
        Generate MACD-based signals
        
        Args:
            macd_line: MACD line
            signal_line: Signal line
            histogram: MACD histogram
            
        Returns:
            Signal array: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = np.zeros(len(macd_line))
        
        # Buy signal when MACD crosses above signal line
        crossover = (macd_line > signal_line) & (np.roll(macd_line, 1) <= np.roll(signal_line, 1))
        signals[crossover] = 1
        
        # Sell signal when MACD crosses below signal line
        crossunder = (macd_line < signal_line) & (np.roll(macd_line, 1) >= np.roll(signal_line, 1))
        signals[crossunder] = -1
        
        return signals
    
    @staticmethod
    def bollinger_signals(prices: np.ndarray, upper_band: np.ndarray, 
                         lower_band: np.ndarray, middle_band: np.ndarray) -> np.ndarray:
        """
        Generate Bollinger Band signals
        
        Args:
            prices: Price array
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band
            middle_band: Middle Bollinger Band
            
        Returns:
            Signal array: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = np.zeros(len(prices))
        
        # Buy signal when price touches lower band
        signals[prices <= lower_band] = 1
        
        # Sell signal when price touches upper band
        signals[prices >= upper_band] = -1
        
        return signals
    
    @staticmethod
    def stochastic_signals(k_values: np.ndarray, d_values: np.ndarray,
                          oversold: float = 20, overbought: float = 80) -> np.ndarray:
        """
        Generate Stochastic signals
        
        Args:
            k_values: %K values
            d_values: %D values
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Signal array: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = np.zeros(len(k_values))
        
        # Buy signal when %K crosses above %D in oversold region
        oversold_region = (k_values < oversold) & (d_values < oversold)
        k_cross_d = (k_values > d_values) & (np.roll(k_values, 1) <= np.roll(d_values, 1))
        signals[oversold_region & k_cross_d] = 1
        
        # Sell signal when %K crosses below %D in overbought region
        overbought_region = (k_values > overbought) & (d_values > overbought)
        k_cross_d_down = (k_values < d_values) & (np.roll(k_values, 1) >= np.roll(d_values, 1))
        signals[overbought_region & k_cross_d_down] = -1
        
        return signals
    
    @staticmethod
    def momentum_signals(momentum_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Generate momentum signals
        
        Args:
            momentum_values: Momentum values
            threshold: Signal threshold
            
        Returns:
            Signal array: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = np.zeros(len(momentum_values))
        
        # Buy signal when momentum is positive
        signals[momentum_values > threshold] = 1
        
        # Sell signal when momentum is negative
        signals[momentum_values < -threshold] = -1
        
        return signals
    
    @staticmethod
    def multi_timeframe_signals(short_signal: np.ndarray, long_signal: np.ndarray,
                               short_weight: float = 0.6) -> np.ndarray:
        """
        Combine signals from multiple timeframes
        
        Args:
            short_signal: Short timeframe signal
            long_signal: Long timeframe signal
            short_weight: Weight for short timeframe
            
        Returns:
            Combined signal array
        """
        long_weight = 1.0 - short_weight
        
        # Weighted combination of signals
        combined_signals = short_weight * short_signal + long_weight * long_signal
        
        # Convert to discrete signals
        discrete_signals = np.zeros(len(combined_signals))
        discrete_signals[combined_signals > 0.3] = 1
        discrete_signals[combined_signals < -0.3] = -1
        
        return discrete_signals


class AdvancedTechnicalIndicators(TechnicalIndicators):
    """
    Advanced technical indicators with enhanced features
    """
    
    @staticmethod
    def adaptive_ema(prices: np.ndarray, period: int = 14, 
                    efficiency_ratio_period: int = 10) -> np.ndarray:
        """
        Adaptive Exponential Moving Average (Kaufman's AMA)
        
        Args:
            prices: Price array
            period: Base period for EMA
            efficiency_ratio_period: Period for efficiency ratio calculation
            
        Returns:
            Adaptive EMA values
        """
        if len(prices) < max(period, efficiency_ratio_period):
            return np.full(len(prices), np.nan)
        
        # Calculate efficiency ratio
        direction = np.abs(prices - np.roll(prices, efficiency_ratio_period))
        volatility = np.sum(np.abs(np.diff(prices, axis=0)), axis=0) if prices.ndim > 1 else np.abs(np.diff(prices))
        
        # Avoid division by zero
        efficiency_ratio = np.divide(direction, volatility, out=np.zeros_like(direction), where=volatility!=0)
        
        # Smoothing constants
        fast_sc = 2.0 / (2 + 1)  # Fast smoothing constant
        slow_sc = 2.0 / (30 + 1)  # Slow smoothing constant
        
        # Adaptive smoothing constant
        adaptive_sc = np.power(efficiency_ratio * (fast_sc - slow_sc) + slow_sc, 2)
        
        # Calculate adaptive EMA
        ama = np.zeros_like(prices)
        ama[0] = prices[0]
        
        for i in range(1, len(prices)):
            ama[i] = ama[i-1] + adaptive_sc[i] * (prices[i] - ama[i-1])
        
        return ama
    
    @staticmethod
    def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 20, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keltner Channels
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: Period for calculations
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Upper channel, Middle line, Lower channel)
        """
        if len(close) < period:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
        
        # Middle line (EMA of close)
        middle_line = TechnicalIndicators.ema(close, period)
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # Upper and lower channels
        upper_channel = middle_line + multiplier * atr
        lower_channel = middle_line - multiplier * atr
        
        return upper_channel, middle_line, lower_channel
    
    @staticmethod
    def donchian_channels(high: np.ndarray, low: np.ndarray, 
                         period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Donchian Channels
        
        Args:
            high: High price array
            low: Low price array
            period: Period for calculation
            
        Returns:
            Tuple of (Upper channel, Middle line, Lower channel)
        """
        if len(high) < period:
            nan_array = np.full(len(high), np.nan)
            return nan_array, nan_array, nan_array
        
        upper_channel = np.zeros_like(high)
        lower_channel = np.zeros_like(low)
        
        for i in range(period - 1, len(high)):
            upper_channel[i] = np.max(high[i - period + 1:i + 1])
            lower_channel[i] = np.min(low[i - period + 1:i + 1])
        
        # Set initial values to NaN
        upper_channel[:period - 1] = np.nan
        lower_channel[:period - 1] = np.nan
        
        # Middle line
        middle_line = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_line, lower_channel
    
    @staticmethod
    def commodity_channel_index(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                               period: int = 20, constant: float = 0.015) -> np.ndarray:
        """
        Commodity Channel Index (CCI)
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: Period for calculation
            constant: Constant factor (typically 0.015)
            
        Returns:
            CCI values
        """
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Calculate CCI
        cci = np.zeros_like(close)
        
        for i in range(period - 1, len(close)):
            tp_window = typical_price[i - period + 1:i + 1]
            sma_tp = np.mean(tp_window)
            mean_deviation = np.mean(np.abs(tp_window - sma_tp))
            
            if mean_deviation > 0:
                cci[i] = (typical_price[i] - sma_tp) / (constant * mean_deviation)
            else:
                cci[i] = 0
        
        # Set initial values to NaN
        cci[:period - 1] = np.nan
        
        return cci
    
    @staticmethod
    def parabolic_sar(high: np.ndarray, low: np.ndarray, 
                     initial_af: float = 0.02, max_af: float = 0.2) -> np.ndarray:
        """
        Parabolic SAR
        
        Args:
            high: High price array
            low: Low price array
            initial_af: Initial acceleration factor
            max_af: Maximum acceleration factor
            
        Returns:
            Parabolic SAR values
        """
        if len(high) < 2:
            return np.full(len(high), np.nan)
        
        sar = np.zeros_like(high)
        trend = np.zeros_like(high)  # 1 for uptrend, -1 for downtrend
        af = np.full_like(high, initial_af)
        extreme_point = np.zeros_like(high)
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        extreme_point[0] = high[0]
        
        for i in range(1, len(high)):
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = extreme_point[i-1]
            
            # Calculate SAR
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Check for trend reversal
            if prev_trend == 1:  # Uptrend
                if low[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep
                    extreme_point[i] = low[i]
                    af[i] = initial_af
                else:
                    # Continue uptrend
                    trend[i] = 1
                    if high[i] > prev_ep:
                        extreme_point[i] = high[i]
                        af[i] = min(prev_af + initial_af, max_af)
                    else:
                        extreme_point[i] = prev_ep
                        af[i] = prev_af
            else:  # Downtrend
                if high[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep
                    extreme_point[i] = high[i]
                    af[i] = initial_af
                else:
                    # Continue downtrend
                    trend[i] = -1
                    if low[i] < prev_ep:
                        extreme_point[i] = low[i]
                        af[i] = min(prev_af + initial_af, max_af)
                    else:
                        extreme_point[i] = prev_ep
                        af[i] = prev_af
        
        return sar
    
    @staticmethod
    def ichimoku_cloud(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      conversion_period: int = 9, base_period: int = 26,
                      span_b_period: int = 52, displacement: int = 26) -> Dict[str, np.ndarray]:
        """
        Ichimoku Cloud components
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            conversion_period: Conversion line period
            base_period: Base line period
            span_b_period: Span B period
            displacement: Displacement for cloud
            
        Returns:
            Dictionary with all Ichimoku components
        """
        if len(close) < max(conversion_period, base_period, span_b_period):
            nan_array = np.full(len(close), np.nan)
            return {
                'conversion_line': nan_array,
                'base_line': nan_array,
                'span_a': nan_array,
                'span_b': nan_array,
                'lagging_span': nan_array
            }
        
        # Conversion Line (Tenkan-sen)
        conversion_line = np.zeros_like(close)
        for i in range(conversion_period - 1, len(close)):
            period_high = np.max(high[i - conversion_period + 1:i + 1])
            period_low = np.min(low[i - conversion_period + 1:i + 1])
            conversion_line[i] = (period_high + period_low) / 2
        conversion_line[:conversion_period - 1] = np.nan
        
        # Base Line (Kijun-sen)
        base_line = np.zeros_like(close)
        for i in range(base_period - 1, len(close)):
            period_high = np.max(high[i - base_period + 1:i + 1])
            period_low = np.min(low[i - base_period + 1:i + 1])
            base_line[i] = (period_high + period_low) / 2
        base_line[:base_period - 1] = np.nan
        
        # Span A (Leading Span A)
        span_a = (conversion_line + base_line) / 2
        
        # Span B (Leading Span B)
        span_b = np.zeros_like(close)
        for i in range(span_b_period - 1, len(close)):
            period_high = np.max(high[i - span_b_period + 1:i + 1])
            period_low = np.min(low[i - span_b_period + 1:i + 1])
            span_b[i] = (period_high + period_low) / 2
        span_b[:span_b_period - 1] = np.nan
        
        # Lagging Span (Chikou Span)
        lagging_span = np.roll(close, -displacement)
        lagging_span[-displacement:] = np.nan
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'span_a': span_a,
            'span_b': span_b,
            'lagging_span': lagging_span
        }
    
    @staticmethod
    def vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vortex Indicator
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: Period for calculation
            
        Returns:
            Tuple of (VI+, VI-)
        """
        if len(close) < period + 1:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array
        
        # Calculate True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        
        # Calculate Vortex Movements
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        
        # Set first values to 0
        vm_plus[0] = 0
        vm_minus[0] = 0
        
        # Calculate VI+ and VI-
        vi_plus = np.zeros_like(close)
        vi_minus = np.zeros_like(close)
        
        for i in range(period, len(close)):
            sum_vm_plus = np.sum(vm_plus[i - period + 1:i + 1])
            sum_vm_minus = np.sum(vm_minus[i - period + 1:i + 1])
            sum_tr = np.sum(tr[i - period + 1:i + 1])
            
            if sum_tr > 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr
        
        # Set initial values to NaN
        vi_plus[:period] = np.nan
        vi_minus[:period] = np.nan
        
        return vi_plus, vi_minus


class PerformanceOptimizer:
    """
    Performance optimization utilities for technical indicators
    """
    
    @staticmethod
    def vectorized_indicator_batch(prices_batch: np.ndarray, 
                                  indicator_func: callable, 
                                  **kwargs) -> np.ndarray:
        """
        Apply indicator function to batch of price series
        
        Args:
            prices_batch: Batch of price series (2D array)
            indicator_func: Indicator function to apply
            **kwargs: Additional arguments for indicator function
            
        Returns:
            Batch of indicator values
        """
        if prices_batch.ndim == 1:
            return indicator_func(prices_batch, **kwargs)
            
        results = []
        for prices in prices_batch:
            results.append(indicator_func(prices, **kwargs))
            
        return np.array(results)
    
    @staticmethod
    def parallel_indicator_calculation(prices_list: list, 
                                      indicator_configs: list) -> Dict[str, np.ndarray]:
        """
        Calculate multiple indicators in parallel
        
        Args:
            prices_list: List of price arrays
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary of indicator results
        """
        results = {}
        
        for config in indicator_configs:
            indicator_name = config['name']
            indicator_func = getattr(TechnicalIndicators, config['function'])
            params = config.get('params', {})
            
            # Calculate for all price series
            indicator_results = []
            for prices in prices_list:
                result = indicator_func(prices, **params)
                indicator_results.append(result)
                
            results[indicator_name] = np.array(indicator_results)
            
        return results
    
    @staticmethod
    def cached_indicator_calculation(prices: np.ndarray, 
                                   indicator_func: callable,
                                   cache_dict: Dict[str, Any],
                                   cache_key: str,
                                   **kwargs) -> np.ndarray:
        """
        Calculate indicator with caching
        
        Args:
            prices: Price array
            indicator_func: Indicator function
            cache_dict: Cache dictionary
            cache_key: Key for caching
            **kwargs: Additional arguments
            
        Returns:
            Cached or computed indicator values
        """
        # Create hash key based on prices and parameters
        price_hash = hash(prices.tobytes())
        param_hash = hash(str(sorted(kwargs.items())))
        full_key = f"{cache_key}_{price_hash}_{param_hash}"
        
        # Check cache
        if full_key in cache_dict:
            return cache_dict[full_key]
        
        # Calculate and cache
        result = indicator_func(prices, **kwargs)
        cache_dict[full_key] = result
        
        # Limit cache size
        if len(cache_dict) > 1000:
            # Remove oldest entries
            keys_to_remove = list(cache_dict.keys())[:100]
            for key in keys_to_remove:
                del cache_dict[key]
        
        return result
    
    @staticmethod
    def optimize_indicator_pipeline(price_data: np.ndarray,
                                   indicator_pipeline: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Optimize calculation of multiple indicators
        
        Args:
            price_data: Price data array
            indicator_pipeline: List of indicator configurations
            
        Returns:
            Dictionary of optimized indicator results
        """
        results = {}
        
        # Group indicators by type for batch processing
        indicator_groups = {}
        for config in indicator_pipeline:
            indicator_type = config['type']
            if indicator_type not in indicator_groups:
                indicator_groups[indicator_type] = []
            indicator_groups[indicator_type].append(config)
        
        # Process each group
        for indicator_type, configs in indicator_groups.items():
            if indicator_type == 'moving_average':
                # Batch process moving averages
                for config in configs:
                    name = config['name']
                    period = config['period']
                    ma_type = config.get('ma_type', 'sma')
                    
                    if ma_type == 'sma':
                        results[name] = TechnicalIndicators.sma(price_data, period)
                    elif ma_type == 'ema':
                        results[name] = TechnicalIndicators.ema(price_data, period)
                        
            elif indicator_type == 'momentum':
                # Batch process momentum indicators
                for config in configs:
                    name = config['name']
                    if 'rsi' in name.lower():
                        period = config.get('period', 14)
                        results[name] = TechnicalIndicators.rsi(price_data, period)
                    elif 'momentum' in name.lower():
                        period = config.get('period', 10)
                        results[name] = TechnicalIndicators.momentum(price_data, period)
                        
            elif indicator_type == 'volatility':
                # Batch process volatility indicators
                for config in configs:
                    name = config['name']
                    if 'atr' in name.lower():
                        period = config.get('period', 14)
                        # Assume we have OHLC data
                        if price_data.ndim == 2 and price_data.shape[1] >= 4:
                            high = price_data[:, 1]
                            low = price_data[:, 2]
                            close = price_data[:, 3]
                            results[name] = TechnicalIndicators.atr(high, low, close, period)
                        else:
                            # Use price as proxy for OHLC
                            results[name] = TechnicalIndicators.atr(price_data, price_data, price_data, period)
        
        return results


class IndicatorCombinations:
    """
    Combinations of multiple indicators for enhanced signals
    """
    
    @staticmethod
    def bollinger_rsi_combo(prices: np.ndarray, bb_period: int = 20, 
                           rsi_period: int = 14, bb_std: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Combine Bollinger Bands and RSI for enhanced signals
        
        Args:
            prices: Price array
            bb_period: Bollinger Bands period
            rsi_period: RSI period
            bb_std: Bollinger Bands standard deviation
            
        Returns:
            Dictionary with combined signals
        """
        # Calculate individual indicators
        upper_band, middle_band, lower_band = TechnicalIndicators.bollinger_bands(prices, bb_period, bb_std)
        rsi_values = TechnicalIndicators.rsi(prices, rsi_period)
        
        # Generate combined signals
        combined_signals = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if not (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(rsi_values[i])):
                # Strong buy signal: price at lower band AND RSI oversold
                if prices[i] <= lower_band[i] and rsi_values[i] < 30:
                    combined_signals[i] = 2.0  # Strong buy
                # Regular buy signal: price at lower band OR RSI oversold
                elif prices[i] <= lower_band[i] or rsi_values[i] < 30:
                    combined_signals[i] = 1.0  # Buy
                # Strong sell signal: price at upper band AND RSI overbought
                elif prices[i] >= upper_band[i] and rsi_values[i] > 70:
                    combined_signals[i] = -2.0  # Strong sell
                # Regular sell signal: price at upper band OR RSI overbought
                elif prices[i] >= upper_band[i] or rsi_values[i] > 70:
                    combined_signals[i] = -1.0  # Sell
        
        return {
            'combined_signals': combined_signals,
            'bollinger_upper': upper_band,
            'bollinger_middle': middle_band,
            'bollinger_lower': lower_band,
            'rsi': rsi_values
        }
    
    @staticmethod
    def macd_stochastic_combo(prices: np.ndarray, high: np.ndarray, 
                             low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Combine MACD and Stochastic for momentum confirmation
        
        Args:
            prices: Price array
            high: High price array
            low: Low price array
            close: Close price array
            
        Returns:
            Dictionary with combined momentum signals
        """
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
        
        # Calculate Stochastic
        k_values, d_values = TechnicalIndicators.stochastic(high, low, close)
        
        # Generate combined momentum signals
        momentum_signals = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if not (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
                   np.isnan(k_values[i]) or np.isnan(d_values[i])):
                
                # MACD signals
                macd_bullish = macd_line[i] > signal_line[i] and histogram[i] > 0
                macd_bearish = macd_line[i] < signal_line[i] and histogram[i] < 0
                
                # Stochastic signals
                stoch_bullish = k_values[i] > d_values[i] and k_values[i] < 80
                stoch_bearish = k_values[i] < d_values[i] and k_values[i] > 20
                
                # Combined signals
                if macd_bullish and stoch_bullish:
                    momentum_signals[i] = 2.0  # Strong bullish momentum
                elif macd_bullish or stoch_bullish:
                    momentum_signals[i] = 1.0  # Bullish momentum
                elif macd_bearish and stoch_bearish:
                    momentum_signals[i] = -2.0  # Strong bearish momentum
                elif macd_bearish or stoch_bearish:
                    momentum_signals[i] = -1.0  # Bearish momentum
        
        return {
            'momentum_signals': momentum_signals,
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'stochastic_k': k_values,
            'stochastic_d': d_values
        }
    
    @staticmethod
    def triple_ema_system(prices: np.ndarray, fast_period: int = 8,
                         medium_period: int = 21, slow_period: int = 55) -> Dict[str, np.ndarray]:
        """
        Triple EMA crossover system
        
        Args:
            prices: Price array
            fast_period: Fast EMA period
            medium_period: Medium EMA period
            slow_period: Slow EMA period
            
        Returns:
            Dictionary with trend signals
        """
        # Calculate EMAs
        ema_fast = TechnicalIndicators.ema(prices, fast_period)
        ema_medium = TechnicalIndicators.ema(prices, medium_period)
        ema_slow = TechnicalIndicators.ema(prices, slow_period)
        
        # Generate trend signals
        trend_signals = np.zeros_like(prices)
        trend_strength = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if not (np.isnan(ema_fast[i]) or np.isnan(ema_medium[i]) or np.isnan(ema_slow[i])):
                # Strong uptrend: fast > medium > slow
                if ema_fast[i] > ema_medium[i] > ema_slow[i]:
                    trend_signals[i] = 2.0
                    trend_strength[i] = (ema_fast[i] - ema_slow[i]) / ema_slow[i]
                # Uptrend: fast > medium or medium > slow
                elif ema_fast[i] > ema_medium[i] or ema_medium[i] > ema_slow[i]:
                    trend_signals[i] = 1.0
                    trend_strength[i] = (ema_fast[i] - ema_slow[i]) / ema_slow[i]
                # Strong downtrend: fast < medium < slow
                elif ema_fast[i] < ema_medium[i] < ema_slow[i]:
                    trend_signals[i] = -2.0
                    trend_strength[i] = (ema_slow[i] - ema_fast[i]) / ema_slow[i]
                # Downtrend: fast < medium or medium < slow
                elif ema_fast[i] < ema_medium[i] or ema_medium[i] < ema_slow[i]:
                    trend_signals[i] = -1.0
                    trend_strength[i] = (ema_slow[i] - ema_fast[i]) / ema_slow[i]
        
        return {
            'trend_signals': trend_signals,
            'trend_strength': trend_strength,
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow
        }