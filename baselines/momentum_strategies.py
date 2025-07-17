"""
Momentum Strategy Agents

Collection of momentum-based trading strategies for baseline comparison.
Each agent implements a specific momentum approach with optimized parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .technical_indicators import TechnicalIndicators, IndicatorSignals
from .rule_based_agent import RuleBasedAgent


class MACDCrossoverAgent(RuleBasedAgent):
    """
    MACD Crossover momentum agent
    
    Features:
    - MACD line and signal line crossovers
    - Histogram divergence detection
    - Dynamic position sizing based on MACD strength
    - Multi-timeframe confirmation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # MACD parameters
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
        
        # Signal thresholds
        self.crossover_threshold = self.config.get('crossover_threshold', 0.0001)
        self.histogram_threshold = self.config.get('histogram_threshold', 0.0)
        
        # Position sizing
        self.base_position = self.config.get('base_position', 0.7)
        self.max_position = self.config.get('max_position', 0.9)
        
        # Multi-timeframe
        self.use_multi_timeframe = self.config.get('use_multi_timeframe', True)
        self.timeframe_periods = self.config.get('timeframe_periods', [(6, 13, 4), (12, 26, 9), (24, 52, 18)])
        
        # History tracking
        self.price_history = []
        self.macd_signals = []
        self.max_history = 200
        
    def update_price_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]
    
    def calculate_macd_signals(self) -> Dict[str, float]:
        """Calculate MACD-based signals"""
        if len(self.price_history) < self.slow_period + self.signal_period:
            return {'signal': 0.0, 'strength': 0.0, 'divergence': 0.0}
        
        prices = np.array(self.price_history)
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            prices, self.fast_period, self.slow_period, self.signal_period
        )
        
        if np.isnan(macd_line[-1]) or np.isnan(signal_line[-1]):
            return {'signal': 0.0, 'strength': 0.0, 'divergence': 0.0}
        
        # Current values
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        # Previous values for crossover detection
        prev_macd = macd_line[-2] if len(macd_line) > 1 else current_macd
        prev_signal = signal_line[-2] if len(signal_line) > 1 else current_signal
        
        # Crossover detection
        crossover_signal = 0.0
        if current_macd > current_signal and prev_macd <= prev_signal:
            crossover_signal = 1.0  # Bullish crossover
        elif current_macd < current_signal and prev_macd >= prev_signal:
            crossover_signal = -1.0  # Bearish crossover
        
        # Signal strength based on MACD line magnitude
        signal_strength = min(abs(current_macd) * 1000, 2.0)  # Scale for typical price ranges
        
        # Histogram divergence (momentum change)
        histogram_divergence = 0.0
        if len(histogram) > 5:
            recent_histogram = histogram[-5:]
            if not np.any(np.isnan(recent_histogram)):
                # Look for divergence pattern
                if current_histogram > 0 and np.mean(recent_histogram) > 0:
                    histogram_divergence = 1.0
                elif current_histogram < 0 and np.mean(recent_histogram) < 0:
                    histogram_divergence = -1.0
        
        return {
            'signal': crossover_signal,
            'strength': signal_strength,
            'divergence': histogram_divergence,
            'macd_value': current_macd,
            'signal_value': current_signal
        }
    
    def calculate_multi_timeframe_signals(self) -> Dict[str, float]:
        """Calculate multi-timeframe MACD signals"""
        if not self.use_multi_timeframe or len(self.price_history) < 60:
            return {'multi_signal': 0.0, 'multi_strength': 0.0}
        
        prices = np.array(self.price_history)
        timeframe_signals = []
        
        for fast, slow, signal in self.timeframe_periods:
            if len(prices) >= slow + signal:
                macd_line, signal_line, _ = TechnicalIndicators.macd(prices, fast, slow, signal)
                
                if not np.isnan(macd_line[-1]) and not np.isnan(signal_line[-1]):
                    # Simple signal based on MACD vs signal line
                    if macd_line[-1] > signal_line[-1]:
                        timeframe_signals.append(1.0)
                    elif macd_line[-1] < signal_line[-1]:
                        timeframe_signals.append(-1.0)
                    else:
                        timeframe_signals.append(0.0)
        
        if not timeframe_signals:
            return {'multi_signal': 0.0, 'multi_strength': 0.0}
        
        # Consensus signal
        multi_signal = np.mean(timeframe_signals)
        multi_strength = 1.0 - np.std(timeframe_signals)  # Higher when signals agree
        
        return {
            'multi_signal': multi_signal,
            'multi_strength': multi_strength
        }
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on MACD analysis"""
        # Update price history
        self.update_price_history(observation)
        
        # Get base action
        base_action = super().get_action(observation)
        
        # Calculate MACD signals
        macd_signals = self.calculate_macd_signals()
        multi_signals = self.calculate_multi_timeframe_signals()
        
        # Combine signals
        primary_signal = macd_signals['signal']
        confirmation_signal = multi_signals['multi_signal']
        
        combined_signal = (
            0.7 * primary_signal +
            0.3 * confirmation_signal
        )
        
        # Calculate position size
        signal_strength = macd_signals['strength']
        multi_strength = multi_signals['multi_strength']
        
        position_size = self.base_position * (1.0 + 0.5 * signal_strength * multi_strength)
        position_size = min(position_size, self.max_position)
        
        # Generate MACD action
        if abs(combined_signal) > 0.3:
            if combined_signal > 0:  # Bullish
                macd_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                macd_action = np.array([position_size, 0.1, 0.1])
        else:
            macd_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        macd_action = macd_action / macd_action.sum()
        
        # Combine with base action
        final_action = 0.2 * base_action + 0.8 * macd_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        self.macd_signals.append({
            'macd': macd_signals,
            'multi_timeframe': multi_signals,
            'combined_signal': combined_signal,
            'position_size': position_size
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.macd_signals = []


class RSIAgent(RuleBasedAgent):
    """
    RSI-based momentum agent
    
    Features:
    - RSI overbought/oversold signals
    - RSI divergence detection
    - Multi-period RSI analysis
    - Dynamic thresholds based on volatility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # RSI parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.oversold_threshold = self.config.get('oversold_threshold', 30)
        self.overbought_threshold = self.config.get('overbought_threshold', 70)
        
        # Multi-period RSI
        self.rsi_periods = self.config.get('rsi_periods', [9, 14, 21])
        
        # Dynamic thresholds
        self.use_dynamic_thresholds = self.config.get('use_dynamic_thresholds', True)
        self.volatility_adjustment = self.config.get('volatility_adjustment', 0.3)
        
        # Position sizing
        self.base_position = self.config.get('base_position', 0.7)
        self.max_position = self.config.get('max_position', 0.9)
        
        # History tracking
        self.price_history = []
        self.rsi_signals = []
        self.max_history = 150
        
    def update_price_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]
    
    def calculate_dynamic_thresholds(self, rsi_values: np.ndarray) -> Tuple[float, float]:
        """Calculate dynamic RSI thresholds based on recent volatility"""
        if not self.use_dynamic_thresholds or len(rsi_values) < 20:
            return self.oversold_threshold, self.overbought_threshold
        
        # Calculate RSI volatility
        recent_rsi = rsi_values[-20:]
        rsi_volatility = np.std(recent_rsi)
        
        # Adjust thresholds based on volatility
        adjustment = self.volatility_adjustment * rsi_volatility
        
        dynamic_oversold = max(20, self.oversold_threshold - adjustment)
        dynamic_overbought = min(80, self.overbought_threshold + adjustment)
        
        return dynamic_oversold, dynamic_overbought
    
    def calculate_rsi_signals(self) -> Dict[str, float]:
        """Calculate RSI-based signals"""
        if len(self.price_history) < max(self.rsi_periods) + 1:
            return {'signal': 0.0, 'strength': 0.0, 'divergence': 0.0}
        
        prices = np.array(self.price_history)
        
        # Calculate multi-period RSI
        rsi_values_dict = {}
        for period in self.rsi_periods:
            rsi_values = TechnicalIndicators.rsi(prices, period)
            if not np.isnan(rsi_values[-1]):
                rsi_values_dict[period] = rsi_values
        
        if not rsi_values_dict:
            return {'signal': 0.0, 'strength': 0.0, 'divergence': 0.0}
        
        # Use primary RSI for main signal
        primary_rsi = rsi_values_dict[self.rsi_period]
        current_rsi = primary_rsi[-1]
        
        # Calculate dynamic thresholds
        oversold, overbought = self.calculate_dynamic_thresholds(primary_rsi)
        
        # Generate RSI signal
        rsi_signal = 0.0
        if current_rsi < oversold:
            rsi_signal = 1.0  # Oversold -> Buy
        elif current_rsi > overbought:
            rsi_signal = -1.0  # Overbought -> Sell
        
        # Signal strength based on distance from thresholds
        if current_rsi < oversold:
            signal_strength = (oversold - current_rsi) / oversold
        elif current_rsi > overbought:
            signal_strength = (current_rsi - overbought) / (100 - overbought)
        else:
            signal_strength = 0.0
        
        # RSI divergence detection
        divergence = self.calculate_rsi_divergence(prices, primary_rsi)
        
        # Multi-period consensus
        multi_period_signals = []
        for period, rsi_vals in rsi_values_dict.items():
            if rsi_vals[-1] < oversold:
                multi_period_signals.append(1.0)
            elif rsi_vals[-1] > overbought:
                multi_period_signals.append(-1.0)
            else:
                multi_period_signals.append(0.0)
        
        consensus_signal = np.mean(multi_period_signals)
        
        return {
            'signal': rsi_signal,
            'strength': signal_strength,
            'divergence': divergence,
            'consensus': consensus_signal,
            'current_rsi': current_rsi
        }
    
    def calculate_rsi_divergence(self, prices: np.ndarray, rsi_values: np.ndarray) -> float:
        """Calculate RSI divergence with price"""
        if len(prices) < 10 or len(rsi_values) < 10:
            return 0.0
        
        # Look for divergence in last 10 periods
        recent_prices = prices[-10:]
        recent_rsi = rsi_values[-10:]
        
        if np.any(np.isnan(recent_rsi)):
            return 0.0
        
        # Price trend
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # RSI trend
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        
        # Divergence occurs when price and RSI move in opposite directions
        if price_trend > 0 and rsi_trend < -5:  # Price up, RSI down
            return -1.0  # Bearish divergence
        elif price_trend < 0 and rsi_trend > 5:  # Price down, RSI up
            return 1.0   # Bullish divergence
        else:
            return 0.0
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on RSI analysis"""
        # Update price history
        self.update_price_history(observation)
        
        # Get base action
        base_action = super().get_action(observation)
        
        # Calculate RSI signals
        rsi_signals = self.calculate_rsi_signals()
        
        # Combine signals
        combined_signal = (
            0.5 * rsi_signals['signal'] +
            0.3 * rsi_signals.get('consensus', 0.0) +
            0.2 * rsi_signals['divergence']
        )
        
        # Calculate position size
        signal_strength = rsi_signals['strength']
        position_size = self.base_position * (1.0 + signal_strength)
        position_size = min(position_size, self.max_position)
        
        # Generate RSI action
        if abs(combined_signal) > 0.3:
            if combined_signal > 0:  # Bullish
                rsi_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                rsi_action = np.array([position_size, 0.1, 0.1])
        else:
            rsi_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        rsi_action = rsi_action / rsi_action.sum()
        
        # Combine with base action
        final_action = 0.2 * base_action + 0.8 * rsi_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        self.rsi_signals.append({
            'rsi': rsi_signals,
            'combined_signal': combined_signal,
            'position_size': position_size
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.rsi_signals = []


class DualMomentumAgent(RuleBasedAgent):
    """
    Dual Momentum agent combining absolute and relative momentum
    
    Features:
    - Absolute momentum (time series momentum)
    - Relative momentum (cross-sectional momentum)
    - Risk-adjusted momentum scoring
    - Dynamic lookback periods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Momentum parameters
        self.absolute_lookback = self.config.get('absolute_lookback', 20)
        self.relative_lookback = self.config.get('relative_lookback', 10)
        self.risk_lookback = self.config.get('risk_lookback', 30)
        
        # Multiple timeframes
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20])
        
        # Risk adjustment
        self.use_risk_adjustment = self.config.get('use_risk_adjustment', True)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        # Position sizing
        self.base_position = self.config.get('base_position', 0.7)
        self.max_position = self.config.get('max_position', 0.9)
        
        # History tracking
        self.price_history = []
        self.momentum_signals = []
        self.max_history = 100
        
    def update_price_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]
    
    def calculate_absolute_momentum(self) -> Dict[str, float]:
        """Calculate absolute momentum signals"""
        if len(self.price_history) < max(self.lookback_periods):
            return {'momentum': 0.0, 'strength': 0.0}
        
        prices = np.array(self.price_history)
        momentum_scores = []
        
        for period in self.lookback_periods:
            if len(prices) >= period:
                # Calculate momentum as price change
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                momentum_scores.append(momentum)
        
        if not momentum_scores:
            return {'momentum': 0.0, 'strength': 0.0}
        
        # Average momentum across timeframes
        avg_momentum = np.mean(momentum_scores)
        
        # Momentum strength (consistency across timeframes)
        momentum_strength = 1.0 - np.std(momentum_scores)
        momentum_strength = max(0.0, momentum_strength)
        
        return {
            'momentum': avg_momentum,
            'strength': momentum_strength
        }
    
    def calculate_risk_adjusted_momentum(self) -> Dict[str, float]:
        """Calculate risk-adjusted momentum"""
        if not self.use_risk_adjustment or len(self.price_history) < self.risk_lookback:
            return {'risk_adjusted': 0.0, 'sharpe_ratio': 0.0}
        
        prices = np.array(self.price_history)
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.risk_lookback:
            return {'risk_adjusted': 0.0, 'sharpe_ratio': 0.0}
        
        # Recent returns for risk calculation
        recent_returns = returns[-self.risk_lookback:]
        
        # Calculate risk metrics
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        if volatility > 0:
            # Annualized Sharpe ratio
            sharpe_ratio = (avg_return * 252 - self.risk_free_rate) / (volatility * np.sqrt(252))
            
            # Risk-adjusted momentum
            risk_adjusted_momentum = avg_return / volatility
        else:
            sharpe_ratio = 0.0
            risk_adjusted_momentum = 0.0
        
        return {
            'risk_adjusted': risk_adjusted_momentum,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_momentum_persistence(self) -> float:
        """Calculate momentum persistence score"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(self.price_history)
        
        # Calculate momentum over different periods
        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        momentum_20 = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Check for consistent momentum direction
        momentums = [momentum_5, momentum_10, momentum_20]
        
        # Filter out zero momentums
        valid_momentums = [m for m in momentums if m != 0]
        
        if not valid_momentums:
            return 0.0
        
        # Calculate persistence as consistency of signs
        signs = [1 if m > 0 else -1 for m in valid_momentums]
        
        if all(s == signs[0] for s in signs):
            # All signs are the same - strong persistence
            persistence = 1.0 * signs[0]
        else:
            # Mixed signs - weak persistence
            persistence = 0.0
        
        return persistence
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on dual momentum analysis"""
        # Update price history
        self.update_price_history(observation)
        
        # Get base action
        base_action = super().get_action(observation)
        
        # Calculate momentum signals
        absolute_momentum = self.calculate_absolute_momentum()
        risk_adjusted = self.calculate_risk_adjusted_momentum()
        persistence = self.calculate_momentum_persistence()
        
        # Combine signals
        combined_signal = (
            0.4 * absolute_momentum['momentum'] +
            0.3 * risk_adjusted['risk_adjusted'] +
            0.3 * persistence
        )
        
        # Scale by momentum strength
        combined_signal *= absolute_momentum['strength']
        
        # Calculate position size
        momentum_strength = absolute_momentum['strength']
        sharpe_ratio = risk_adjusted['sharpe_ratio']
        
        position_size = self.base_position * (1.0 + 0.3 * momentum_strength + 0.2 * min(sharpe_ratio, 2.0))
        position_size = min(position_size, self.max_position)
        
        # Generate momentum action
        if abs(combined_signal) > 0.02:  # Lower threshold for momentum
            if combined_signal > 0:  # Bullish
                momentum_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                momentum_action = np.array([position_size, 0.1, 0.1])
        else:
            momentum_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        momentum_action = momentum_action / momentum_action.sum()
        
        # Combine with base action
        final_action = 0.2 * base_action + 0.8 * momentum_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        self.momentum_signals.append({
            'absolute': absolute_momentum,
            'risk_adjusted': risk_adjusted,
            'persistence': persistence,
            'combined_signal': combined_signal,
            'position_size': position_size
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.momentum_signals = []


class BreakoutAgent(RuleBasedAgent):
    """
    Breakout strategy agent
    
    Features:
    - Support/resistance level detection
    - Volume-confirmed breakouts
    - False breakout filtering
    - Multi-timeframe breakout analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Breakout parameters
        self.lookback_window = self.config.get('lookback_window', 20)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.02)
        self.volume_confirmation = self.config.get('volume_confirmation', True)
        
        # Multi-timeframe
        self.timeframe_windows = self.config.get('timeframe_windows', [10, 20, 50])
        
        # False breakout filtering
        self.use_false_breakout_filter = self.config.get('use_false_breakout_filter', True)
        self.confirmation_periods = self.config.get('confirmation_periods', 3)
        
        # Position sizing
        self.base_position = self.config.get('base_position', 0.8)
        self.max_position = self.config.get('max_position', 0.95)
        
        # History tracking
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.volume_history = []
        self.breakout_signals = []
        self.max_history = 150
        
    def update_history(self, observation: Dict[str, Any]):
        """Update OHLCV history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            # Extract OHLCV if available
            if len(features) >= 4:
                self.high_history.append(features[1])
                self.low_history.append(features[2])
                volume = features[4] if len(features) > 4 else 1.0
                self.volume_history.append(volume)
            else:
                # Use price as high/low if OHLC not available
                self.high_history.append(price)
                self.low_history.append(price)
                self.volume_history.append(1.0)
            
            # Maintain history limits
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]
                self.high_history = self.high_history[-self.max_history:]
                self.low_history = self.low_history[-self.max_history:]
                self.volume_history = self.volume_history[-self.max_history:]
    
    def calculate_support_resistance(self, window: int) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        if len(self.price_history) < window:
            return {'support': 0.0, 'resistance': 0.0}
        
        highs = np.array(self.high_history[-window:])
        lows = np.array(self.low_history[-window:])
        
        # Support and resistance as max/min of recent period
        resistance = np.max(highs)
        support = np.min(lows)
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    def calculate_breakout_signals(self) -> Dict[str, float]:
        """Calculate breakout signals"""
        if len(self.price_history) < self.lookback_window:
            return {'breakout': 0.0, 'strength': 0.0, 'volume_confirmed': False}
        
        current_price = self.price_history[-1]
        
        # Calculate support/resistance levels
        levels = self.calculate_support_resistance(self.lookback_window)
        support = levels['support']
        resistance = levels['resistance']
        
        if resistance <= support:
            return {'breakout': 0.0, 'strength': 0.0, 'volume_confirmed': False}
        
        # Calculate breakout signal
        breakout_signal = 0.0
        breakout_strength = 0.0
        
        # Upward breakout
        if current_price > resistance:
            breakout_signal = 1.0
            breakout_strength = (current_price - resistance) / resistance
        
        # Downward breakout
        elif current_price < support:
            breakout_signal = -1.0
            breakout_strength = (support - current_price) / support
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_confirmation and len(self.volume_history) > 5:
            recent_volume = np.mean(self.volume_history[-3:])
            avg_volume = np.mean(self.volume_history[-20:]) if len(self.volume_history) >= 20 else recent_volume
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                volume_confirmed = volume_ratio > 1.2  # 20% above average
        
        return {
            'breakout': breakout_signal,
            'strength': min(breakout_strength, 2.0),
            'volume_confirmed': volume_confirmed
        }
    
    def calculate_multi_timeframe_breakouts(self) -> Dict[str, float]:
        """Calculate breakout signals across multiple timeframes"""
        if len(self.price_history) < max(self.timeframe_windows):
            return {'multi_breakout': 0.0, 'consensus': 0.0}
        
        timeframe_signals = []
        
        for window in self.timeframe_windows:
            if len(self.price_history) >= window:
                levels = self.calculate_support_resistance(window)
                current_price = self.price_history[-1]
                
                # Simple breakout check
                if current_price > levels['resistance']:
                    timeframe_signals.append(1.0)
                elif current_price < levels['support']:
                    timeframe_signals.append(-1.0)
                else:
                    timeframe_signals.append(0.0)
        
        if not timeframe_signals:
            return {'multi_breakout': 0.0, 'consensus': 0.0}
        
        # Calculate consensus
        multi_breakout = np.mean(timeframe_signals)
        consensus = 1.0 - np.std(timeframe_signals)  # Higher when signals agree
        
        return {
            'multi_breakout': multi_breakout,
            'consensus': max(0.0, consensus)
        }
    
    def filter_false_breakouts(self, breakout_signal: float) -> float:
        """Filter out potential false breakouts"""
        if not self.use_false_breakout_filter or len(self.breakout_signals) < self.confirmation_periods:
            return breakout_signal
        
        # Check recent breakout signals
        recent_signals = [s.get('breakout', 0.0) for s in self.breakout_signals[-self.confirmation_periods:]]
        
        # If recent signals are inconsistent, reduce current signal
        if len(set(np.sign(recent_signals))) > 1:  # Mixed signals
            return breakout_signal * 0.5
        
        return breakout_signal
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on breakout analysis"""
        # Update history
        self.update_history(observation)
        
        # Get base action
        base_action = super().get_action(observation)
        
        # Calculate breakout signals
        breakout_signals = self.calculate_breakout_signals()
        multi_timeframe = self.calculate_multi_timeframe_breakouts()
        
        # Apply false breakout filter
        filtered_breakout = self.filter_false_breakouts(breakout_signals['breakout'])
        
        # Combine signals
        combined_signal = (
            0.6 * filtered_breakout +
            0.4 * multi_timeframe['multi_breakout']
        )
        
        # Apply volume confirmation
        if self.volume_confirmation and not breakout_signals['volume_confirmed']:
            combined_signal *= 0.7
        
        # Calculate position size
        breakout_strength = breakout_signals['strength']
        consensus = multi_timeframe['consensus']
        
        position_size = self.base_position * (1.0 + 0.3 * breakout_strength * consensus)
        position_size = min(position_size, self.max_position)
        
        # Generate breakout action
        if abs(combined_signal) > 0.3:
            if combined_signal > 0:  # Bullish breakout
                breakout_action = np.array([0.05, 0.05, position_size])
            else:  # Bearish breakout
                breakout_action = np.array([position_size, 0.05, 0.05])
        else:
            breakout_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        breakout_action = breakout_action / breakout_action.sum()
        
        # Combine with base action
        final_action = 0.1 * base_action + 0.9 * breakout_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        signal_data = {
            'breakout': breakout_signals,
            'multi_timeframe': multi_timeframe,
            'combined_signal': combined_signal,
            'position_size': position_size
        }
        self.breakout_signals.append(signal_data)
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.volume_history = []
        self.breakout_signals = []