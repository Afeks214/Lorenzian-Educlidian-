"""
HYBRID LORENTZIAN-EUCLIDEAN DISTANCE SYSTEM - MARKET REGIME DETECTION
====================================================================

An intelligent market regime detection system that dynamically switches between
Lorentzian and Euclidean distance metrics based on real-time market conditions.

This system implements:
1. Market Regime Detection (volatile, trending, ranging, calm)
2. Volatility Analysis with multiple estimators
3. Trend Strength and Persistence Analysis
4. Regime Transition Detection
5. Confidence-based Distance Metric Selection

Author: Claude AI Research Division
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    VOLATILE = "volatile"      # High volatility, unpredictable movements
    TRENDING = "trending"      # Strong directional movement
    RANGING = "ranging"        # Sideways movement, mean-reverting
    CALM = "calm"             # Low volatility, stable conditions
    TRANSITIONAL = "transitional"  # Between regimes

@dataclass
class RegimeConfig:
    """Configuration for market regime detection"""
    
    # Volatility thresholds
    high_volatility_threshold: float = 0.25    # Annualized volatility
    low_volatility_threshold: float = 0.10     # Annualized volatility
    
    # Trend strength thresholds
    strong_trend_threshold: float = 0.7        # ADX threshold
    weak_trend_threshold: float = 25.0         # ADX threshold
    
    # Regime persistence settings
    regime_persistence_window: int = 10        # Bars to confirm regime
    transition_threshold: float = 0.3          # Confidence for regime change
    
    # Analysis windows
    volatility_window: int = 20               # Rolling volatility window
    trend_window: int = 14                    # Trend analysis window
    atr_window: int = 14                      # ATR calculation window
    
    # Distance metric selection
    lorentzian_volatility_threshold: float = 0.20  # Switch to Lorentzian above this
    euclidean_stability_threshold: float = 0.12    # Use Euclidean below this
    confidence_threshold: float = 0.6              # Minimum confidence for switching
    
    # Advanced parameters
    garch_window: int = 50                    # GARCH estimation window
    realized_vol_window: int = 15             # Realized volatility window
    momentum_window: int = 10                 # Momentum calculation window

@dataclass
class RegimeMetrics:
    """Container for regime analysis metrics"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    trend_direction: int = 0  # 1 for up, -1 for down, 0 for neutral
    momentum: float = 0.0
    regime: MarketRegime = MarketRegime.CALM
    confidence: float = 0.0
    distance_metric: str = "euclidean"
    regime_persistence: int = 0
    atr: float = 0.0
    realized_volatility: float = 0.0
    garch_volatility: float = 0.0
    
    # Additional context
    price_change: float = 0.0
    volume_factor: float = 1.0
    recent_volatility: float = 0.0
    trend_persistence: float = 0.0

class VolatilityEstimator:
    """Advanced volatility estimation with multiple methods"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        
    def calculate_realized_volatility(self, prices: np.ndarray, 
                                    window: Optional[int] = None) -> float:
        """
        Calculate realized volatility using high-frequency returns
        
        RV = sqrt(sum(log_returns^2)) * sqrt(252)
        """
        if window is None:
            window = self.config.realized_vol_window
            
        if len(prices) < window:
            return 0.0
            
        log_returns = np.diff(np.log(prices[-window:]))
        realized_var = np.sum(log_returns ** 2)
        return np.sqrt(realized_var * 252)  # Annualized
    
    def calculate_garch_volatility(self, prices: np.ndarray, 
                                 window: Optional[int] = None) -> float:
        """
        Simplified GARCH(1,1) volatility estimation
        
        σ²ₜ = ω + α * ε²ₜ₋₁ + β * σ²ₜ₋₁
        """
        if window is None:
            window = self.config.garch_window
            
        if len(prices) < window + 5:
            return self.calculate_realized_volatility(prices)
            
        returns = np.diff(np.log(prices[-window:]))
        
        # Simple GARCH parameters (could be estimated via MLE)
        omega = 0.01  # Long-term variance component
        alpha = 0.05  # ARCH effect
        beta = 0.90   # GARCH effect
        
        # Initialize
        variance = np.var(returns)
        garch_variances = [variance]
        
        for i in range(1, len(returns)):
            variance = (omega + 
                       alpha * returns[i-1]**2 + 
                       beta * variance)
            garch_variances.append(variance)
        
        return np.sqrt(garch_variances[-1] * 252)  # Annualized
    
    def calculate_atr_volatility(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, window: Optional[int] = None) -> float:
        """
        Calculate ATR-based volatility measure
        """
        if window is None:
            window = self.config.atr_window
            
        if len(high) < window:
            return 0.0
            
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR as exponential moving average
        atr = pd.Series(true_range[-window:]).ewm(span=window).mean().iloc[-1]
        
        # Convert to volatility (ATR as % of price)
        current_price = close[-1]
        return (atr / current_price) * np.sqrt(252) if current_price > 0 else 0.0

class TrendAnalyzer:
    """Advanced trend analysis and strength measurement"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, window: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Calculate ADX with DI+ and DI- components
        
        Returns: (ADX, DI+, DI-)
        """
        if window is None:
            window = self.config.trend_window
            
        if len(high) < window + 1:
            return 0.0, 0.0, 0.0
            
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        # Smoothed values
        atr = pd.Series(tr[-window:]).ewm(span=window).mean().iloc[-1]
        di_plus_raw = pd.Series(dm_plus[-window:]).ewm(span=window).mean().iloc[-1]
        di_minus_raw = pd.Series(dm_minus[-window:]).ewm(span=window).mean().iloc[-1]
        
        # Calculate DI+ and DI-
        di_plus = 100 * (di_plus_raw / atr) if atr > 0 else 0.0
        di_minus = 100 * (di_minus_raw / atr) if atr > 0 else 0.0
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0.0
        
        # For simplicity, using current DX as ADX (should be smoothed over time)
        adx = dx
        
        return adx, di_plus, di_minus
    
    def calculate_trend_persistence(self, prices: np.ndarray, 
                                  window: Optional[int] = None) -> float:
        """
        Calculate trend persistence using directional movement consistency
        """
        if window is None:
            window = self.config.trend_window
            
        if len(prices) < window:
            return 0.0
            
        returns = np.diff(prices[-window:])
        
        # Count consecutive moves in same direction
        positive_moves = np.sum(returns > 0)
        negative_moves = np.sum(returns < 0)
        
        # Persistence as consistency of direction
        total_moves = len(returns)
        if total_moves == 0:
            return 0.0
            
        max_directional = max(positive_moves, negative_moves)
        return max_directional / total_moves
    
    def calculate_momentum_indicators(self, prices: np.ndarray, 
                                    window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate multiple momentum indicators
        """
        if window is None:
            window = self.config.momentum_window
            
        if len(prices) < window + 1:
            return {"roc": 0.0, "momentum": 0.0, "acceleration": 0.0}
            
        current_price = prices[-1]
        past_price = prices[-(window + 1)]
        
        # Rate of Change
        roc = (current_price - past_price) / past_price if past_price > 0 else 0.0
        
        # Price momentum
        momentum = current_price - past_price
        
        # Acceleration (second derivative)
        if len(prices) >= window + 2:
            mid_price = prices[-(window // 2 + 1)]
            old_momentum = mid_price - past_price
            acceleration = momentum - old_momentum
        else:
            acceleration = 0.0
        
        return {
            "roc": roc,
            "momentum": momentum,
            "acceleration": acceleration
        }

class RegimeDetector:
    """Core market regime detection system"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.volatility_estimator = VolatilityEstimator(config)
        self.trend_analyzer = TrendAnalyzer(config)
        
        # Regime history for persistence tracking
        self.regime_history: List[MarketRegime] = []
        self.confidence_history: List[float] = []
        
    def detect_regime(self, data: pd.DataFrame) -> RegimeMetrics:
        """
        Main regime detection function
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            RegimeMetrics containing comprehensive regime analysis
        """
        if len(data) < max(self.config.volatility_window, self.config.trend_window):
            return RegimeMetrics()
            
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate volatility measures
        volatility = self._calculate_volatility_composite(high, low, close)
        realized_vol = self.volatility_estimator.calculate_realized_volatility(close)
        garch_vol = self.volatility_estimator.calculate_garch_volatility(close)
        atr = self.volatility_estimator.calculate_atr_volatility(high, low, close)
        
        # Calculate trend measures
        adx, di_plus, di_minus = self.trend_analyzer.calculate_adx(high, low, close)
        trend_persistence = self.trend_analyzer.calculate_trend_persistence(close)
        momentum_indicators = self.trend_analyzer.calculate_momentum_indicators(close)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(di_plus, di_minus, close)
        
        # Classify regime
        regime, confidence = self._classify_regime(volatility, adx, trend_persistence)
        
        # Update regime history
        self._update_regime_history(regime, confidence)
        
        # Determine optimal distance metric
        distance_metric = self._select_distance_metric(volatility, regime, confidence)
        
        # Calculate additional metrics
        price_change = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0.0
        volume_factor = self._calculate_volume_factor(data)
        recent_volatility = self._calculate_recent_volatility(close)
        
        return RegimeMetrics(
            volatility=volatility,
            trend_strength=adx,
            trend_direction=trend_direction,
            momentum=momentum_indicators["roc"],
            regime=regime,
            confidence=confidence,
            distance_metric=distance_metric,
            regime_persistence=self._get_regime_persistence(),
            atr=atr,
            realized_volatility=realized_vol,
            garch_volatility=garch_vol,
            price_change=price_change,
            volume_factor=volume_factor,
            recent_volatility=recent_volatility,
            trend_persistence=trend_persistence
        )
    
    def _calculate_volatility_composite(self, high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray) -> float:
        """
        Calculate composite volatility using multiple estimators
        """
        # Standard rolling volatility
        returns = np.diff(np.log(close[-self.config.volatility_window:]))
        rolling_vol = np.std(returns) * np.sqrt(252)
        
        # ATR-based volatility
        atr_vol = self.volatility_estimator.calculate_atr_volatility(high, low, close)
        
        # Realized volatility
        realized_vol = self.volatility_estimator.calculate_realized_volatility(close)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.3]  # Adjust based on reliability
        volatilities = [rolling_vol, atr_vol, realized_vol]
        
        composite_vol = np.average(volatilities, weights=weights)
        return composite_vol
    
    def _determine_trend_direction(self, di_plus: float, di_minus: float, 
                                 close: np.ndarray) -> int:
        """
        Determine trend direction using multiple indicators
        """
        # Primary: DI+ vs DI-
        di_signal = 1 if di_plus > di_minus else -1 if di_minus > di_plus else 0
        
        # Secondary: Simple moving average comparison
        if len(close) >= 20:
            sma_short = np.mean(close[-10:])
            sma_long = np.mean(close[-20:])
            sma_signal = 1 if sma_short > sma_long else -1 if sma_short < sma_long else 0
        else:
            sma_signal = 0
        
        # Combine signals
        if di_signal == sma_signal:
            return di_signal
        elif abs(di_plus - di_minus) > 5:  # Strong DI signal
            return di_signal
        else:
            return 0  # Neutral
    
    def _classify_regime(self, volatility: float, adx: float, 
                        trend_persistence: float) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on volatility and trend strength
        """
        confidence_factors = []
        
        # Volatility-based classification
        if volatility > self.config.high_volatility_threshold:
            vol_regime = MarketRegime.VOLATILE
            vol_confidence = min(1.0, (volatility - self.config.high_volatility_threshold) / 0.1)
        elif volatility < self.config.low_volatility_threshold:
            vol_regime = MarketRegime.CALM
            vol_confidence = min(1.0, (self.config.low_volatility_threshold - volatility) / 0.05)
        else:
            vol_regime = MarketRegime.RANGING
            vol_confidence = 0.5
        
        confidence_factors.append(vol_confidence)
        
        # Trend-based classification
        if adx > self.config.strong_trend_threshold:
            trend_regime = MarketRegime.TRENDING
            trend_confidence = min(1.0, (adx - self.config.strong_trend_threshold) / 20.0)
        elif adx < self.config.weak_trend_threshold:
            trend_regime = MarketRegime.RANGING
            trend_confidence = min(1.0, (self.config.weak_trend_threshold - adx) / 10.0)
        else:
            trend_regime = MarketRegime.RANGING
            trend_confidence = 0.5
        
        confidence_factors.append(trend_confidence)
        
        # Persistence-based adjustment
        persistence_confidence = trend_persistence
        confidence_factors.append(persistence_confidence)
        
        # Combined classification logic
        if vol_regime == MarketRegime.VOLATILE:
            final_regime = MarketRegime.VOLATILE
        elif trend_regime == MarketRegime.TRENDING and adx > 30:
            final_regime = MarketRegime.TRENDING
        elif vol_regime == MarketRegime.CALM and trend_regime == MarketRegime.RANGING:
            final_regime = MarketRegime.CALM
        else:
            final_regime = MarketRegime.RANGING
        
        # Calculate composite confidence
        composite_confidence = np.mean(confidence_factors)
        
        return final_regime, composite_confidence
    
    def _select_distance_metric(self, volatility: float, regime: MarketRegime, 
                              confidence: float) -> str:
        """
        Select optimal distance metric based on market conditions
        
        Decision Logic:
        - High volatility (>threshold) OR volatile regime → Lorentzian (robust)
        - Low volatility (<threshold) AND calm/stable → Euclidean (precise)
        - Medium volatility → Hybrid approach based on confidence
        """
        # Primary decision based on volatility
        if volatility > self.config.lorentzian_volatility_threshold:
            return "lorentzian"
        elif volatility < self.config.euclidean_stability_threshold:
            return "euclidean"
        
        # Secondary decision based on regime
        if regime == MarketRegime.VOLATILE:
            return "lorentzian"
        elif regime == MarketRegime.CALM and confidence > self.config.confidence_threshold:
            return "euclidean"
        elif regime == MarketRegime.TRENDING and confidence > 0.7:
            return "euclidean"  # Trending markets can be more predictable
        
        # Default to hybrid approach for uncertain conditions
        return "hybrid"
    
    def _update_regime_history(self, regime: MarketRegime, confidence: float):
        """Update regime history for persistence tracking"""
        self.regime_history.append(regime)
        self.confidence_history.append(confidence)
        
        # Maintain history window
        max_history = self.config.regime_persistence_window * 2
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
    
    def _get_regime_persistence(self) -> int:
        """Calculate how long the current regime has persisted"""
        if not self.regime_history:
            return 0
            
        current_regime = self.regime_history[-1]
        persistence = 1
        
        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i] == current_regime:
                persistence += 1
            else:
                break
                
        return persistence
    
    def _calculate_volume_factor(self, data: pd.DataFrame) -> float:
        """Calculate volume-based market activity factor"""
        if 'volume' not in data.columns:
            return 1.0
            
        volume = data['volume'].values
        if len(volume) < 10:
            return 1.0
            
        recent_vol = np.mean(volume[-5:])
        avg_vol = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        
        return recent_vol / avg_vol if avg_vol > 0 else 1.0
    
    def _calculate_recent_volatility(self, close: np.ndarray) -> float:
        """Calculate very recent volatility (short-term)"""
        if len(close) < 5:
            return 0.0
            
        recent_returns = np.diff(np.log(close[-5:]))
        return np.std(recent_returns) * np.sqrt(252)
    
    def detect_regime_transition(self, current_metrics: RegimeMetrics, 
                               previous_metrics: Optional[RegimeMetrics] = None) -> Dict[str, any]:
        """
        Detect regime transitions and provide transition signals
        """
        if previous_metrics is None or len(self.regime_history) < 2:
            return {
                "transition_detected": False,
                "transition_type": None,
                "transition_confidence": 0.0,
                "recommended_action": "maintain"
            }
        
        # Check for regime change
        regime_changed = current_metrics.regime != previous_metrics.regime
        
        # Check for significant volatility change
        vol_change = abs(current_metrics.volatility - previous_metrics.volatility)
        significant_vol_change = vol_change > 0.05  # 5% volatility change
        
        # Check for trend strength change
        trend_change = abs(current_metrics.trend_strength - previous_metrics.trend_strength)
        significant_trend_change = trend_change > 15.0  # ADX change > 15
        
        # Determine transition type and confidence
        transition_detected = regime_changed or significant_vol_change or significant_trend_change
        
        if not transition_detected:
            return {
                "transition_detected": False,
                "transition_type": None,
                "transition_confidence": 0.0,
                "recommended_action": "maintain"
            }
        
        # Classify transition
        if regime_changed:
            transition_type = f"{previous_metrics.regime.value}_to_{current_metrics.regime.value}"
        elif significant_vol_change:
            transition_type = "volatility_shift"
        else:
            transition_type = "trend_shift"
        
        # Calculate transition confidence
        confidence_factors = [
            current_metrics.confidence,
            1.0 if regime_changed else 0.5,
            min(1.0, vol_change / 0.1) if significant_vol_change else 0.0,
            min(1.0, trend_change / 30.0) if significant_trend_change else 0.0
        ]
        
        transition_confidence = np.mean([f for f in confidence_factors if f > 0])
        
        # Recommend action
        if transition_confidence > 0.7:
            recommended_action = "recalibrate"
        elif transition_confidence > 0.5:
            recommended_action = "monitor"
        else:
            recommended_action = "maintain"
        
        return {
            "transition_detected": transition_detected,
            "transition_type": transition_type,
            "transition_confidence": transition_confidence,
            "recommended_action": recommended_action,
            "details": {
                "regime_changed": regime_changed,
                "volatility_change": vol_change,
                "trend_change": trend_change,
                "previous_regime": previous_metrics.regime.value,
                "current_regime": current_metrics.regime.value
            }
        }

# Convenience functions for easy integration

def create_regime_detector(config: Optional[RegimeConfig] = None) -> RegimeDetector:
    """Create a regime detector with default or custom configuration"""
    if config is None:
        config = RegimeConfig()
    return RegimeDetector(config)

def quick_regime_analysis(data: pd.DataFrame, 
                         config: Optional[RegimeConfig] = None) -> RegimeMetrics:
    """Quick regime analysis for single data frame"""
    detector = create_regime_detector(config)
    return detector.detect_regime(data)

def get_optimal_distance_metric(data: pd.DataFrame, 
                              config: Optional[RegimeConfig] = None) -> str:
    """Get optimal distance metric recommendation for current market conditions"""
    metrics = quick_regime_analysis(data, config)
    return metrics.distance_metric

if __name__ == "__main__":
    # Demonstration of the regime detection system
    print("HYBRID LORENTZIAN-EUCLIDEAN DISTANCE SYSTEM")
    print("Market Regime Detection Module")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_bars = 500
    
    # Create realistic market data with different regimes
    returns = np.random.normal(0.0001, 0.02, n_bars)
    
    # Add regime changes
    returns[100:200] = np.random.normal(0.001, 0.05, 100)  # Volatile period
    returns[200:300] = np.random.normal(0.002, 0.01, 100)  # Trending period
    returns[300:400] = np.random.normal(0.0, 0.008, 100)   # Calm period
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    noise = 0.01
    high = prices * (1 + np.abs(np.random.normal(0, noise, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, noise, n_bars)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(10, 1, n_bars)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    # Initialize regime detector
    config = RegimeConfig()
    detector = RegimeDetector(config)
    
    print(f"Analyzing {len(data)} bars of market data...")
    print()
    
    # Analyze different periods
    periods = [
        ("Early Period (Bars 50-100)", 50, 100),
        ("Volatile Period (Bars 150-200)", 150, 200), 
        ("Trending Period (Bars 250-300)", 250, 300),
        ("Calm Period (Bars 350-400)", 350, 400),
        ("Recent Period (Bars 450-500)", 450, 500)
    ]
    
    previous_metrics = None
    
    for period_name, start, end in periods:
        period_data = data.iloc[max(0, start-50):end]  # Include lookback
        metrics = detector.detect_regime(period_data)
        
        print(f"{period_name}:")
        print(f"  Regime: {metrics.regime.value.upper()}")
        print(f"  Confidence: {metrics.confidence:.3f}")
        print(f"  Volatility: {metrics.volatility:.3f}")
        print(f"  Trend Strength (ADX): {metrics.trend_strength:.1f}")
        print(f"  Trend Direction: {'+' if metrics.trend_direction > 0 else '-' if metrics.trend_direction < 0 else '='}")
        print(f"  Distance Metric: {metrics.distance_metric.upper()}")
        print(f"  Regime Persistence: {metrics.regime_persistence} bars")
        
        # Check for transitions
        if previous_metrics:
            transition_info = detector.detect_regime_transition(metrics, previous_metrics)
            if transition_info["transition_detected"]:
                print(f"  🔄 TRANSITION: {transition_info['transition_type']}")
                print(f"     Confidence: {transition_info['transition_confidence']:.3f}")
                print(f"     Action: {transition_info['recommended_action']}")
        
        print()
        previous_metrics = metrics
    
    print("REGIME DETECTION SUMMARY:")
    print("-" * 40)
    print("✓ Multi-estimator volatility analysis")
    print("✓ Trend strength and direction detection") 
    print("✓ Regime persistence tracking")
    print("✓ Intelligent distance metric selection")
    print("✓ Regime transition detection")
    print()
    print("System ready for integration with Lorentzian Classification!")