#!/usr/bin/env python3
"""
TRADING INDICATORS IMPLEMENTATION
Complete implementation of all trading indicators used in the GrandModel system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLMIIndicator:
    """
    Multi-Layer Market Intelligence (MLMI) Indicator
    Advanced momentum and trend analysis
    """
    
    def __init__(self, k_neighbors: int = 5, trend_length: int = 14, smoothing_factor: float = 0.8):
        self.k_neighbors = k_neighbors
        self.trend_length = trend_length
        self.smoothing_factor = smoothing_factor
        self.scaler = StandardScaler()
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MLMI signals
        """
        if len(data) < self.trend_length + 20:
            return pd.Series(0, index=data.index)
        
        # Calculate features
        features = self._calculate_features(data)
        
        # Apply nearest neighbors analysis
        signals = self._apply_nearest_neighbors(features)
        
        # Apply smoothing
        smoothed_signals = self._apply_smoothing(signals)
        
        return smoothed_signals
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for MLMI analysis
        """
        features = pd.DataFrame(index=data.index)
        
        # Price momentum
        features['price_momentum'] = data['close'].pct_change(self.trend_length)
        
        # Volume momentum
        if 'volume' in data.columns:
            features['volume_momentum'] = data['volume'].pct_change(self.trend_length)
        else:
            features['volume_momentum'] = 0
        
        # RSI
        features['rsi'] = self._calculate_rsi(data['close'])
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(data['close'])
        
        # Bollinger Bands position
        features['bb_position'] = self._calculate_bb_position(data['close'])
        
        # ATR
        features['atr'] = self._calculate_atr(data)
        
        # Stochastic
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(data)
        
        return features.fillna(0)
    
    def _apply_nearest_neighbors(self, features: pd.DataFrame) -> pd.Series:
        """
        Apply nearest neighbors analysis
        """
        signals = pd.Series(0.0, index=features.index)
        
        # Need enough data for training
        if len(features) < 100:
            return signals
        
        # Prepare features
        feature_matrix = features.fillna(0).values
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
        
        # Rolling window analysis
        window_size = 50
        for i in range(window_size, len(features)):
            # Training window
            train_data = feature_matrix[i-window_size:i]
            
            # Fit model
            nn.fit(train_data)
            
            # Current point
            current_point = feature_matrix[i:i+1]
            
            # Find neighbors
            distances, indices = nn.kneighbors(current_point)
            
            # Calculate signal based on neighbor patterns
            neighbor_returns = []
            for idx in indices[0]:
                actual_idx = i - window_size + idx
                if actual_idx < len(features) - 1:
                    neighbor_returns.append(features.iloc[actual_idx+1]['price_momentum'])
            
            if neighbor_returns:
                signals.iloc[i] = np.mean(neighbor_returns)
        
        return signals
    
    def _apply_smoothing(self, signals: pd.Series) -> pd.Series:
        """
        Apply exponential smoothing
        """
        return signals.ewm(alpha=self.smoothing_factor).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize to [-1, 1]
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd / prices, macd_signal / prices  # Normalize
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Bands position"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Position within bands
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return (bb_position - 0.5) * 2  # Normalize to [-1, 1]
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(0, index=data.index)
        
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr / data['close']  # Normalize
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
        
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return (k_percent - 50) / 50, (d_percent - 50) / 50  # Normalize to [-1, 1]

class FVGIndicator:
    """
    Fair Value Gap (FVG) Indicator
    Identifies market inefficiencies and liquidity gaps
    """
    
    def __init__(self, threshold: float = 0.001, max_age: int = 50, gap_size_min: float = 0.0005):
        self.threshold = threshold
        self.max_age = max_age
        self.gap_size_min = gap_size_min
        self.active_gaps = []
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate FVG signals
        """
        if len(data) < 3:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0.0, index=data.index)
        
        # Detect fair value gaps
        for i in range(2, len(data)):
            # Check for bullish FVG
            if self._is_bullish_fvg(data.iloc[i-2:i+1]):
                gap_strength = self._calculate_gap_strength(data.iloc[i-2:i+1], 'bullish')
                signals.iloc[i] = gap_strength
            
            # Check for bearish FVG
            elif self._is_bearish_fvg(data.iloc[i-2:i+1]):
                gap_strength = self._calculate_gap_strength(data.iloc[i-2:i+1], 'bearish')
                signals.iloc[i] = -gap_strength
        
        return signals
    
    def _is_bullish_fvg(self, candles: pd.DataFrame) -> bool:
        """
        Check if there's a bullish fair value gap
        """
        if len(candles) < 3:
            return False
        
        # Bullish FVG: candle 1 low > candle 3 high
        candle1 = candles.iloc[0]
        candle2 = candles.iloc[1]
        candle3 = candles.iloc[2]
        
        # Check for gap
        gap_exists = candle1['low'] > candle3['high']
        
        # Check gap size
        if gap_exists:
            gap_size = (candle1['low'] - candle3['high']) / candle2['close']
            return gap_size > self.gap_size_min
        
        return False
    
    def _is_bearish_fvg(self, candles: pd.DataFrame) -> bool:
        """
        Check if there's a bearish fair value gap
        """
        if len(candles) < 3:
            return False
        
        # Bearish FVG: candle 1 high < candle 3 low
        candle1 = candles.iloc[0]
        candle2 = candles.iloc[1]
        candle3 = candles.iloc[2]
        
        # Check for gap
        gap_exists = candle1['high'] < candle3['low']
        
        # Check gap size
        if gap_exists:
            gap_size = (candle3['low'] - candle1['high']) / candle2['close']
            return gap_size > self.gap_size_min
        
        return False
    
    def _calculate_gap_strength(self, candles: pd.DataFrame, gap_type: str) -> float:
        """
        Calculate the strength of the fair value gap
        """
        if len(candles) < 3:
            return 0.0
        
        candle1 = candles.iloc[0]
        candle2 = candles.iloc[1]
        candle3 = candles.iloc[2]
        
        if gap_type == 'bullish':
            gap_size = (candle1['low'] - candle3['high']) / candle2['close']
            volume_strength = candle2.get('volume', 1) / candles['volume'].mean()
        else:
            gap_size = (candle3['low'] - candle1['high']) / candle2['close']
            volume_strength = candle2.get('volume', 1) / candles['volume'].mean()
        
        # Combine gap size and volume strength
        strength = gap_size * volume_strength
        
        return min(strength, 1.0)  # Cap at 1.0

class NWRQKIndicator:
    """
    Nadaraya-Watson Regression Quantile Kernel (NWRQK) Indicator
    Advanced non-parametric regression for trend analysis
    """
    
    def __init__(self, bandwidth: float = 46, alpha: float = 8, length_scale: float = 1.0):
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.length_scale = length_scale
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate NWRQK signals
        """
        if len(data) < self.bandwidth + 10:
            return pd.Series(0, index=data.index)
        
        prices = data['close']
        signals = pd.Series(0.0, index=data.index)
        
        # Apply Nadaraya-Watson regression
        for i in range(int(self.bandwidth), len(prices)):
            # Get lookback window
            window_prices = prices.iloc[i-int(self.bandwidth):i+1]
            
            # Calculate kernel weights
            weights = self._calculate_kernel_weights(len(window_prices))
            
            # Calculate weighted regression
            regression_value = self._nadaraya_watson_regression(window_prices, weights)
            
            # Calculate signal
            current_price = prices.iloc[i]
            signal = (regression_value - current_price) / current_price
            
            signals.iloc[i] = np.tanh(signal * self.alpha)  # Apply hyperbolic tangent
        
        return signals
    
    def _calculate_kernel_weights(self, window_size: int) -> np.ndarray:
        """
        Calculate Gaussian kernel weights
        """
        x = np.arange(window_size)
        center = window_size - 1  # Most recent point
        
        # Gaussian kernel
        weights = np.exp(-0.5 * ((x - center) / self.length_scale) ** 2)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _nadaraya_watson_regression(self, prices: pd.Series, weights: np.ndarray) -> float:
        """
        Apply Nadaraya-Watson regression
        """
        if len(prices) != len(weights):
            return prices.iloc[-1]
        
        # Weighted regression
        weighted_sum = np.sum(prices.values * weights)
        
        return weighted_sum

class LVNIndicator:
    """
    Liquidity Volume Nodes (LVN) Indicator
    Identifies significant volume levels and liquidity zones
    """
    
    def __init__(self, lookback_periods: int = 20, strength_threshold: float = 0.7):
        self.lookback_periods = lookback_periods
        self.strength_threshold = strength_threshold
        self.volume_nodes = []
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate LVN signals
        """
        if len(data) < self.lookback_periods + 10:
            return pd.Series(0, index=data.index)
        
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0.0, index=data.index)
        
        # Calculate volume-weighted levels
        for i in range(self.lookback_periods, len(data)):
            # Get lookback window
            window_data = data.iloc[i-self.lookback_periods:i+1]
            
            # Calculate volume nodes
            volume_nodes = self._calculate_volume_nodes(window_data)
            
            # Calculate signal based on price proximity to volume nodes
            current_price = data.iloc[i]['close']
            signal = self._calculate_lvn_signal(current_price, volume_nodes)
            
            signals.iloc[i] = signal
        
        return signals
    
    def _calculate_volume_nodes(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Calculate significant volume nodes
        """
        # Create price-volume distribution
        price_levels = np.linspace(data['low'].min(), data['high'].max(), 50)
        volume_at_level = np.zeros(len(price_levels))
        
        for i, row in data.iterrows():
            # Distribute volume across price range
            low_idx = np.searchsorted(price_levels, row['low'])
            high_idx = np.searchsorted(price_levels, row['high'])
            
            if low_idx < high_idx:
                volume_per_level = row['volume'] / (high_idx - low_idx)
                volume_at_level[low_idx:high_idx] += volume_per_level
        
        # Find significant nodes
        mean_volume = np.mean(volume_at_level)
        std_volume = np.std(volume_at_level)
        threshold = mean_volume + std_volume * self.strength_threshold
        
        nodes = []
        for i, volume in enumerate(volume_at_level):
            if volume > threshold:
                nodes.append({
                    'price': price_levels[i],
                    'volume': volume,
                    'strength': volume / mean_volume
                })
        
        return nodes
    
    def _calculate_lvn_signal(self, current_price: float, volume_nodes: List[Dict[str, float]]) -> float:
        """
        Calculate signal based on proximity to volume nodes
        """
        if not volume_nodes:
            return 0.0
        
        # Find closest volume node
        closest_node = min(volume_nodes, key=lambda x: abs(x['price'] - current_price))
        
        # Calculate distance to closest node
        distance = abs(current_price - closest_node['price']) / current_price
        
        # Calculate signal strength
        signal_strength = closest_node['strength'] * (1 - distance)
        
        # Determine signal direction
        if current_price > closest_node['price']:
            return -signal_strength * 0.5  # Resistance
        else:
            return signal_strength * 0.5   # Support

class MMDIndicator:
    """
    Market Microstructure Dynamics (MMD) Indicator
    Analyzes market microstructure for trading opportunities
    """
    
    def __init__(self, window_size: int = 20, sensitivity: float = 0.5):
        self.window_size = window_size
        self.sensitivity = sensitivity
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MMD signals
        """
        if len(data) < self.window_size + 10:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0.0, index=data.index)
        
        # Calculate various microstructure metrics
        for i in range(self.window_size, len(data)):
            window_data = data.iloc[i-self.window_size:i+1]
            
            # Calculate components
            spread_signal = self._calculate_spread_signal(window_data)
            momentum_signal = self._calculate_momentum_signal(window_data)
            volume_signal = self._calculate_volume_signal(window_data)
            
            # Combine signals
            combined_signal = (spread_signal + momentum_signal + volume_signal) / 3
            
            signals.iloc[i] = combined_signal * self.sensitivity
        
        return signals
    
    def _calculate_spread_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate spread-based signal
        """
        # Use high-low spread as proxy for bid-ask spread
        spreads = (data['high'] - data['low']) / data['close']
        
        # Calculate relative spread
        current_spread = spreads.iloc[-1]
        avg_spread = spreads.mean()
        
        if avg_spread > 0:
            spread_ratio = current_spread / avg_spread
            return (1 - spread_ratio) * 0.5  # Lower spread = better conditions
        
        return 0.0
    
    def _calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum-based signal
        """
        # Price momentum
        price_change = data['close'].pct_change()
        momentum = price_change.iloc[-1]
        
        # Normalize momentum
        momentum_std = price_change.std()
        if momentum_std > 0:
            normalized_momentum = momentum / momentum_std
            return np.tanh(normalized_momentum)
        
        return 0.0
    
    def _calculate_volume_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate volume-based signal
        """
        if 'volume' not in data.columns:
            return 0.0
        
        # Volume momentum
        volume_change = data['volume'].pct_change()
        volume_momentum = volume_change.iloc[-1]
        
        # Price-volume correlation
        price_change = data['close'].pct_change()
        correlation = price_change.corr(volume_change)
        
        # Combine volume momentum and correlation
        if not np.isnan(correlation):
            return (volume_momentum * correlation) * 0.1
        
        return 0.0

# Export all indicator classes
__all__ = [
    'MLMIIndicator',
    'FVGIndicator', 
    'NWRQKIndicator',
    'LVNIndicator',
    'MMDIndicator'
]