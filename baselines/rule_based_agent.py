"""
Rule-Based Baseline Agent

Simple non-ML agent that follows fixed rules based on synergy detection.
This represents the original, deterministic trading strategy before MARL enhancement.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from .technical_indicators import TechnicalIndicators, IndicatorSignals


class RuleBasedAgent:
    """
    Rule-based agent following simple synergy-based logic
    
    Trading Rules:
    1. If bullish synergy detected -> Strong bullish action [0.8, 0.1, 0.1]
    2. If bearish synergy detected -> Strong bearish action [0.1, 0.1, 0.8]
    3. If no synergy -> Neutral/hold position [0.1, 0.8, 0.1]
    4. Confidence affects action strength
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rule-based agent
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Action strength parameters
        self.strong_action_weight = self.config.get('strong_action_weight', 0.8)
        self.neutral_action_weight = self.config.get('neutral_action_weight', 0.8)
        self.min_confidence_threshold = self.config.get('min_confidence', 0.6)
        
        # Track agent statistics
        self.action_history = []
        self.synergy_history = []
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Get action based on rules
        
        Args:
            observation: Dictionary containing:
                - features: Agent-specific features
                - shared_context: Market context
                - synergy_active: Whether synergy is detected
                - synergy_type: Type of synergy (0-4)
                - synergy_info: Optional detailed synergy information
                
        Returns:
            Action probabilities [bearish, neutral, bullish]
        """
        # Extract synergy information
        synergy_active = observation.get('synergy_active', 0)
        synergy_info = observation.get('synergy_info', {})
        
        # Default neutral action
        action = np.array([0.1, 0.8, 0.1])
        
        if synergy_active and synergy_info:
            direction = synergy_info.get('direction', 0)
            confidence = synergy_info.get('confidence', 0.5)
            synergy_type = synergy_info.get('type', 'UNKNOWN')
            
            # Only act if confidence exceeds threshold
            if confidence >= self.min_confidence_threshold:
                # Scale action strength by confidence
                strength = self.strong_action_weight * (confidence / 1.0)
                weak_weight = (1.0 - strength) / 2.0
                
                if direction > 0:  # Bullish
                    action = np.array([weak_weight, weak_weight, strength])
                elif direction < 0:  # Bearish
                    action = np.array([strength, weak_weight, weak_weight])
                else:  # Neutral but with synergy
                    # Slightly reduce neutral position
                    action = np.array([0.2, 0.6, 0.2])
                    
                # Record synergy for analysis
                self.synergy_history.append({
                    'type': synergy_type,
                    'direction': direction,
                    'confidence': confidence,
                    'action': action.copy()
                })
        
        # Ensure valid probability distribution
        action = action / action.sum()
        
        # Record action
        self.action_history.append(action)
        
        return action
        
    def get_action_with_features(
        self, 
        features: np.ndarray,
        synergy_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Alternative interface using raw features
        
        Args:
            features: Feature array
            synergy_info: Optional synergy information
            
        Returns:
            Action probabilities
        """
        observation = {
            'features': features,
            'synergy_active': 1 if synergy_info else 0,
            'synergy_info': synergy_info or {}
        }
        
        return self.get_action(observation)
        
    def reset(self):
        """Reset agent state"""
        self.action_history.clear()
        self.synergy_history.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if not self.action_history:
            return {
                'total_actions': 0,
                'synergy_actions': 0,
                'action_distribution': {'bearish': 0, 'neutral': 0, 'bullish': 0}
            }
            
        actions = np.array(self.action_history)
        
        # Calculate action distribution
        avg_action = actions.mean(axis=0)
        
        # Classify actions
        action_types = []
        for action in actions:
            if action[0] > 0.5:
                action_types.append('bearish')
            elif action[2] > 0.5:
                action_types.append('bullish')
            else:
                action_types.append('neutral')
                
        action_counts = {
            'bearish': action_types.count('bearish'),
            'neutral': action_types.count('neutral'),
            'bullish': action_types.count('bullish')
        }
        
        return {
            'total_actions': len(self.action_history),
            'synergy_actions': len(self.synergy_history),
            'avg_action': avg_action.tolist(),
            'action_distribution': action_counts,
            'synergy_types': [s['type'] for s in self.synergy_history]
        }
        

class TechnicalRuleBasedAgent(RuleBasedAgent):
    """
    Technical analysis-based rule agent with momentum and mean reversion strategies
    
    Features:
    - MACD crossover signals
    - RSI overbought/oversold signals
    - Bollinger Band mean reversion
    - Stochastic momentum
    - Volatility-based position sizing
    - Multi-timeframe signal integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Technical indicator parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        self.stoch_k = self.config.get('stoch_k', 14)
        self.stoch_d = self.config.get('stoch_d', 3)
        
        # Strategy weights
        self.momentum_weight = self.config.get('momentum_weight', 0.4)
        self.mean_reversion_weight = self.config.get('mean_reversion_weight', 0.3)
        self.volatility_weight = self.config.get('volatility_weight', 0.2)
        self.trend_weight = self.config.get('trend_weight', 0.1)
        
        # Position sizing parameters
        self.base_position_size = self.config.get('base_position_size', 0.8)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.vol_target = self.config.get('vol_target', 0.15)
        
        # Signal confirmation
        self.min_signal_strength = self.config.get('min_signal_strength', 0.3)
        self.signal_decay = self.config.get('signal_decay', 0.95)
        
        # Price history for technical analysis
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.volume_history = []
        self.max_history = self.config.get('max_history', 100)
        
        # Signal tracking
        self.signal_history = []
        self.last_signals = {}
        
    def update_price_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        # Extract price data from observation
        features = observation.get('features', np.array([]))
        shared_context = observation.get('shared_context', np.array([]))
        
        if len(features) > 0:
            # Assume first feature is price
            price = features[0]
            self.price_history.append(price)
            
            # For OHLC data, extract if available
            if len(features) >= 4:
                self.high_history.append(features[1])
                self.low_history.append(features[2])
                # Volume might be in features[4] if available
                if len(features) > 4:
                    self.volume_history.append(features[4])
            else:
                # Use price as high/low if OHLC not available
                self.high_history.append(price)
                self.low_history.append(price)
                self.volume_history.append(1.0)  # Default volume
        else:
            # Use shared context if features not available
            if len(shared_context) > 0:
                price = shared_context[0]
                self.price_history.append(price)
                self.high_history.append(price)
                self.low_history.append(price)
                self.volume_history.append(1.0)
        
        # Maintain max history length
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]
            self.high_history = self.high_history[-self.max_history:]
            self.low_history = self.low_history[-self.max_history:]
            self.volume_history = self.volume_history[-self.max_history:]
    
    def calculate_technical_signals(self) -> Dict[str, float]:
        """Calculate all technical indicator signals"""
        if len(self.price_history) < 30:  # Need minimum history
            return {'momentum': 0.0, 'mean_reversion': 0.0, 'volatility': 0.0, 'trend': 0.0}
        
        prices = np.array(self.price_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        
        signals = {}
        
        # 1. Momentum Signals
        momentum_signals = []
        
        # RSI Signal
        rsi_values = TechnicalIndicators.rsi(prices, self.rsi_period)
        rsi_signal = IndicatorSignals.rsi_signals(rsi_values, self.rsi_oversold, self.rsi_overbought)
        if not np.isnan(rsi_signal[-1]):
            momentum_signals.append(rsi_signal[-1])
        
        # MACD Signal
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            prices, self.macd_fast, self.macd_slow, self.macd_signal
        )
        macd_signal = IndicatorSignals.macd_signals(macd_line, signal_line, histogram)
        if not np.isnan(macd_signal[-1]):
            momentum_signals.append(macd_signal[-1])
        
        # Stochastic Signal
        k_values, d_values = TechnicalIndicators.stochastic(highs, lows, prices, self.stoch_k, self.stoch_d)
        stoch_signal = IndicatorSignals.stochastic_signals(k_values, d_values)
        if not np.isnan(stoch_signal[-1]):
            momentum_signals.append(stoch_signal[-1])
        
        # Momentum Signal
        momentum_values = TechnicalIndicators.momentum(prices, 10)
        momentum_signal = IndicatorSignals.momentum_signals(momentum_values)
        if not np.isnan(momentum_signal[-1]):
            momentum_signals.append(momentum_signal[-1])
        
        signals['momentum'] = np.mean(momentum_signals) if momentum_signals else 0.0
        
        # 2. Mean Reversion Signals
        mean_reversion_signals = []
        
        # Bollinger Bands Signal
        upper_band, middle_band, lower_band = TechnicalIndicators.bollinger_bands(
            prices, self.bb_period, self.bb_std
        )
        bb_signal = IndicatorSignals.bollinger_signals(prices, upper_band, lower_band, middle_band)
        if not np.isnan(bb_signal[-1]):
            mean_reversion_signals.append(bb_signal[-1])
        
        # Z-Score Signal
        zscore_values = TechnicalIndicators.zscore(prices, 20)
        if not np.isnan(zscore_values[-1]):
            # Z-score mean reversion: buy when oversold, sell when overbought
            if zscore_values[-1] < -2.0:
                mean_reversion_signals.append(1.0)  # Buy
            elif zscore_values[-1] > 2.0:
                mean_reversion_signals.append(-1.0)  # Sell
            else:
                mean_reversion_signals.append(0.0)  # Hold
        
        signals['mean_reversion'] = np.mean(mean_reversion_signals) if mean_reversion_signals else 0.0
        
        # 3. Volatility Signal
        atr_values = TechnicalIndicators.atr(highs, lows, prices, 14)
        if not np.isnan(atr_values[-1]) and len(atr_values) > 20:
            current_vol = atr_values[-1]
            avg_vol = np.mean(atr_values[-20:])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Reduce position size in high volatility
            signals['volatility'] = max(0.3, 1.0 / vol_ratio)
        else:
            signals['volatility'] = 1.0
        
        # 4. Trend Signal
        trend_signals = []
        
        # Simple trend using moving averages
        if len(prices) >= 20:
            sma_fast = TechnicalIndicators.sma(prices, 10)
            sma_slow = TechnicalIndicators.sma(prices, 20)
            
            if not np.isnan(sma_fast[-1]) and not np.isnan(sma_slow[-1]):
                if sma_fast[-1] > sma_slow[-1]:
                    trend_signals.append(1.0)  # Uptrend
                else:
                    trend_signals.append(-1.0)  # Downtrend
        
        signals['trend'] = np.mean(trend_signals) if trend_signals else 0.0
        
        return signals
    
    def calculate_position_size(self, base_signal: float, volatility_signal: float) -> float:
        """Calculate position size based on volatility"""
        # Volatility-based position sizing
        vol_adjusted_size = self.base_position_size * volatility_signal
        
        # Apply signal strength
        signal_strength = abs(base_signal)
        size_multiplier = 0.5 + 0.5 * signal_strength  # Range: 0.5 to 1.0
        
        return vol_adjusted_size * size_multiplier
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Enhanced action selection with technical analysis"""
        # Update price history
        self.update_price_history(observation)
        
        # Get base synergy action
        base_action = super().get_action(observation)
        
        # Calculate technical signals
        tech_signals = self.calculate_technical_signals()
        
        # Combine signals
        combined_signal = (
            self.momentum_weight * tech_signals['momentum'] +
            self.mean_reversion_weight * tech_signals['mean_reversion'] +
            self.trend_weight * tech_signals['trend']
        )
        
        # Calculate position size
        position_size = self.calculate_position_size(combined_signal, tech_signals['volatility'])
        
        # Generate technical action
        if abs(combined_signal) >= self.min_signal_strength:
            if combined_signal > 0:  # Bullish
                tech_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                tech_action = np.array([position_size, 0.1, 0.1])
        else:
            # Neutral with reduced position
            tech_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        tech_action = tech_action / tech_action.sum()
        
        # Combine with base action (weight technical analysis higher)
        synergy_weight = 0.3
        technical_weight = 0.7
        
        final_action = synergy_weight * base_action + technical_weight * tech_action
        final_action = final_action / final_action.sum()
        
        # Store signal for analysis
        self.signal_history.append({
            'signals': tech_signals,
            'combined_signal': combined_signal,
            'position_size': position_size,
            'action': final_action.copy()
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.volume_history = []
        self.signal_history = []
        self.last_signals = {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        base_stats = super().get_statistics()
        
        if not self.signal_history:
            return base_stats
        
        # Technical analysis statistics
        signals_df = self.signal_history
        
        # Average signals
        avg_momentum = np.mean([s['signals']['momentum'] for s in signals_df])
        avg_mean_reversion = np.mean([s['signals']['mean_reversion'] for s in signals_df])
        avg_volatility = np.mean([s['signals']['volatility'] for s in signals_df])
        avg_trend = np.mean([s['signals']['trend'] for s in signals_df])
        
        # Signal strength distribution
        combined_signals = [s['combined_signal'] for s in signals_df]
        signal_strength_dist = {
            'weak': len([s for s in combined_signals if abs(s) < 0.3]),
            'moderate': len([s for s in combined_signals if 0.3 <= abs(s) < 0.7]),
            'strong': len([s for s in combined_signals if abs(s) >= 0.7])
        }
        
        # Position size statistics
        position_sizes = [s['position_size'] for s in signals_df]
        avg_position_size = np.mean(position_sizes)
        
        base_stats.update({
            'technical_analysis': {
                'avg_momentum_signal': avg_momentum,
                'avg_mean_reversion_signal': avg_mean_reversion,
                'avg_volatility_signal': avg_volatility,
                'avg_trend_signal': avg_trend,
                'signal_strength_distribution': signal_strength_dist,
                'avg_position_size': avg_position_size,
                'total_technical_signals': len(signals_df)
            }
        })
        
        return base_stats


class EnhancedRuleBasedAgent(RuleBasedAgent):
    """
    Enhanced rule-based agent with additional logic
    
    Adds:
    - Synergy type-specific rules
    - Market regime awareness
    - Risk management rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Enhanced parameters
        self.type_weights = {
            'TYPE_1': 1.0,   # Strongest signal
            'TYPE_2': 0.8,   # Strong signal
            'TYPE_3': 0.6,   # Moderate signal
            'TYPE_4': 0.7    # Good signal
        }
        
        self.volatility_threshold = self.config.get('volatility_threshold', 1.5)
        self.reduce_position_in_high_vol = self.config.get('reduce_in_high_vol', True)
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Enhanced action selection with additional rules"""
        # Get base action
        action = super().get_action(observation)
        
        # Apply enhancements
        synergy_info = observation.get('synergy_info', {})
        shared_context = observation.get('shared_context', np.zeros(6))
        
        if synergy_info:
            # Adjust by synergy type
            synergy_type = synergy_info.get('type', 'UNKNOWN')
            if synergy_type in self.type_weights:
                type_mult = self.type_weights[synergy_type]
                
                # Amplify non-neutral components
                if action[0] > action[2]:  # Bearish
                    action[0] = min(0.9, action[0] * type_mult)
                elif action[2] > action[0]:  # Bullish
                    action[2] = min(0.9, action[2] * type_mult)
                    
        # Risk management based on volatility
        if len(shared_context) > 2:
            volatility = np.exp(shared_context[2])  # Assuming log(volatility) in position 2
            
            if self.reduce_position_in_high_vol and volatility > self.volatility_threshold:
                # Reduce position size in high volatility
                reduction_factor = self.volatility_threshold / volatility
                
                # Move weight toward neutral
                neutral_weight = action[1]
                action[0] *= reduction_factor
                action[2] *= reduction_factor
                action[1] = 1.0 - action[0] - action[2]
                
        # Ensure valid distribution
        action = action / action.sum()
        
        return action


class AdvancedMomentumAgent(RuleBasedAgent):
    """
    Advanced momentum-based agent with multiple momentum strategies
    
    Features:
    - Multi-timeframe momentum analysis
    - Momentum strength filtering
    - Trend persistence detection
    - Breakout pattern recognition
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Momentum parameters
        self.momentum_periods = self.config.get('momentum_periods', [5, 10, 20])
        self.momentum_threshold = self.config.get('momentum_threshold', 0.02)
        self.trend_persistence_period = self.config.get('trend_persistence', 10)
        
        # Breakout parameters
        self.breakout_window = self.config.get('breakout_window', 20)
        self.breakout_threshold = self.config.get('breakout_threshold', 2.0)
        
        # Position sizing
        self.momentum_position_scale = self.config.get('momentum_position_scale', 0.8)
        self.breakout_position_scale = self.config.get('breakout_position_scale', 1.0)
        
        # Price history tracking
        self.price_history = []
        self.momentum_signals = []
        self.breakout_signals = []
        
    def update_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            # Maintain rolling window
            max_window = max(self.momentum_periods + [self.breakout_window, self.trend_persistence_period])
            if len(self.price_history) > max_window * 2:
                self.price_history = self.price_history[-max_window * 2:]
    
    def calculate_momentum_signals(self) -> Dict[str, float]:
        """Calculate momentum signals across multiple timeframes"""
        if len(self.price_history) < max(self.momentum_periods):
            return {'momentum_strength': 0.0, 'momentum_direction': 0.0}
        
        prices = np.array(self.price_history)
        momentum_signals = []
        
        for period in self.momentum_periods:
            if len(prices) >= period:
                # Calculate momentum as price change
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                momentum_signals.append(momentum)
        
        if not momentum_signals:
            return {'momentum_strength': 0.0, 'momentum_direction': 0.0}
        
        # Calculate average momentum and strength
        avg_momentum = np.mean(momentum_signals)
        momentum_strength = np.std(momentum_signals)  # Consistency across timeframes
        
        # Direction based on sign
        momentum_direction = 1.0 if avg_momentum > self.momentum_threshold else -1.0 if avg_momentum < -self.momentum_threshold else 0.0
        
        return {
            'momentum_strength': momentum_strength,
            'momentum_direction': momentum_direction,
            'avg_momentum': avg_momentum
        }
    
    def calculate_breakout_signals(self) -> Dict[str, float]:
        """Calculate breakout signals"""
        if len(self.price_history) < self.breakout_window:
            return {'breakout_strength': 0.0, 'breakout_direction': 0.0}
        
        prices = np.array(self.price_history)
        recent_prices = prices[-self.breakout_window:]
        
        # Calculate price range
        price_high = np.max(recent_prices[:-1])  # Exclude current price
        price_low = np.min(recent_prices[:-1])
        current_price = prices[-1]
        
        # Calculate breakout strength
        if price_high > price_low:
            upper_breakout = (current_price - price_high) / (price_high - price_low)
            lower_breakout = (price_low - current_price) / (price_high - price_low)
            
            if upper_breakout > 0.1:  # Upward breakout
                breakout_strength = min(upper_breakout, 2.0)
                breakout_direction = 1.0
            elif lower_breakout > 0.1:  # Downward breakout
                breakout_strength = min(lower_breakout, 2.0)
                breakout_direction = -1.0
            else:
                breakout_strength = 0.0
                breakout_direction = 0.0
        else:
            breakout_strength = 0.0
            breakout_direction = 0.0
        
        return {
            'breakout_strength': breakout_strength,
            'breakout_direction': breakout_direction
        }
    
    def calculate_trend_persistence(self) -> float:
        """Calculate trend persistence score"""
        if len(self.price_history) < self.trend_persistence_period:
            return 0.0
        
        prices = np.array(self.price_history)
        recent_prices = prices[-self.trend_persistence_period:]
        
        # Calculate trend direction for each period
        trend_directions = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                trend_directions.append(1)
            elif recent_prices[i] < recent_prices[i-1]:
                trend_directions.append(-1)
            else:
                trend_directions.append(0)
        
        if not trend_directions:
            return 0.0
        
        # Calculate persistence as consistency of direction
        trend_array = np.array(trend_directions)
        positive_count = np.sum(trend_array == 1)
        negative_count = np.sum(trend_array == -1)
        
        # Persistence score
        persistence = abs(positive_count - negative_count) / len(trend_directions)
        direction = 1.0 if positive_count > negative_count else -1.0
        
        return persistence * direction
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on momentum analysis"""
        # Update price history
        self.update_history(observation)
        
        # Get base synergy action
        base_action = super().get_action(observation)
        
        # Calculate momentum signals
        momentum_signals = self.calculate_momentum_signals()
        breakout_signals = self.calculate_breakout_signals()
        trend_persistence = self.calculate_trend_persistence()
        
        # Combine signals
        combined_signal = (
            0.4 * momentum_signals['momentum_direction'] +
            0.4 * breakout_signals['breakout_direction'] +
            0.2 * trend_persistence
        )
        
        # Calculate position size based on signal strength
        momentum_strength = momentum_signals['momentum_strength']
        breakout_strength = breakout_signals['breakout_strength']
        
        position_size = min(
            self.momentum_position_scale * (1.0 + momentum_strength),
            self.breakout_position_scale * (1.0 + breakout_strength)
        )
        position_size = min(position_size, 0.9)  # Cap at 90%
        
        # Generate momentum action
        if abs(combined_signal) > 0.3:
            if combined_signal > 0:  # Bullish
                momentum_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                momentum_action = np.array([position_size, 0.1, 0.1])
        else:
            momentum_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        momentum_action = momentum_action / momentum_action.sum()
        
        # Combine with base action
        final_action = 0.3 * base_action + 0.7 * momentum_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        self.momentum_signals.append({
            'momentum': momentum_signals,
            'breakout': breakout_signals,
            'trend_persistence': trend_persistence,
            'combined_signal': combined_signal,
            'position_size': position_size
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.momentum_signals = []
        self.breakout_signals = []


class AdvancedMeanReversionAgent(RuleBasedAgent):
    """
    Advanced mean reversion agent with multiple strategies
    
    Features:
    - Z-score based mean reversion
    - Bollinger Band squeeze detection
    - Statistical arbitrage signals
    - Volatility regime filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Mean reversion parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)
        self.bb_squeeze_threshold = self.config.get('bb_squeeze_threshold', 0.1)
        
        # Volatility filtering
        self.vol_regime_period = self.config.get('vol_regime_period', 30)
        self.high_vol_threshold = self.config.get('high_vol_threshold', 1.5)
        
        # Position sizing
        self.mean_rev_position_scale = self.config.get('mean_rev_position_scale', 0.8)
        self.squeeze_position_scale = self.config.get('squeeze_position_scale', 1.0)
        
        # Price history tracking
        self.price_history = []
        self.mean_rev_signals = []
        
    def update_history(self, observation: Dict[str, Any]):
        """Update price history from observation"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0:
            price = features[0]
            self.price_history.append(price)
            
            # Maintain rolling window
            max_window = max(self.lookback_period, self.vol_regime_period) * 2
            if len(self.price_history) > max_window:
                self.price_history = self.price_history[-max_window:]
    
    def calculate_zscore_signals(self) -> Dict[str, float]:
        """Calculate Z-score based mean reversion signals"""
        if len(self.price_history) < self.lookback_period:
            return {'zscore': 0.0, 'zscore_signal': 0.0}
        
        prices = np.array(self.price_history)
        recent_prices = prices[-self.lookback_period:]
        
        # Calculate Z-score
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price > 0:
            zscore = (prices[-1] - mean_price) / std_price
        else:
            zscore = 0.0
        
        # Generate signal
        if zscore > self.zscore_threshold:
            zscore_signal = -1.0  # Sell (mean reversion)
        elif zscore < -self.zscore_threshold:
            zscore_signal = 1.0   # Buy (mean reversion)
        else:
            zscore_signal = 0.0
        
        return {
            'zscore': zscore,
            'zscore_signal': zscore_signal
        }
    
    def calculate_bollinger_squeeze(self) -> Dict[str, float]:
        """Calculate Bollinger Band squeeze signals"""
        if len(self.price_history) < self.lookback_period:
            return {'squeeze_strength': 0.0, 'squeeze_direction': 0.0}
        
        prices = np.array(self.price_history)
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = TechnicalIndicators.bollinger_bands(
            prices, self.lookback_period, 2.0
        )
        
        if np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]):
            return {'squeeze_strength': 0.0, 'squeeze_direction': 0.0}
        
        # Calculate band width
        band_width = (upper_band[-1] - lower_band[-1]) / middle_band[-1]
        
        # Historical band width for comparison
        if len(upper_band) > self.lookback_period:
            historical_widths = []
            for i in range(self.lookback_period, len(upper_band)):
                if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                    width = (upper_band[i] - lower_band[i]) / middle_band[i]
                    historical_widths.append(width)
            
            if historical_widths:
                avg_width = np.mean(historical_widths)
                squeeze_strength = max(0.0, (avg_width - band_width) / avg_width)
            else:
                squeeze_strength = 0.0
        else:
            squeeze_strength = 0.0
        
        # Direction based on price position in bands
        current_price = prices[-1]
        if current_price > middle_band[-1]:
            squeeze_direction = 1.0
        elif current_price < middle_band[-1]:
            squeeze_direction = -1.0
        else:
            squeeze_direction = 0.0
        
        return {
            'squeeze_strength': squeeze_strength,
            'squeeze_direction': squeeze_direction
        }
    
    def calculate_volatility_regime(self) -> str:
        """Determine current volatility regime"""
        if len(self.price_history) < self.vol_regime_period:
            return 'normal'
        
        prices = np.array(self.price_history)
        recent_prices = prices[-self.vol_regime_period:]
        
        # Calculate rolling volatility
        returns = np.diff(np.log(recent_prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Historical volatility for comparison
        if len(prices) > self.vol_regime_period * 2:
            historical_prices = prices[-self.vol_regime_period * 2:-self.vol_regime_period]
            historical_returns = np.diff(np.log(historical_prices))
            historical_vol = np.std(historical_returns) * np.sqrt(252)
            
            vol_ratio = volatility / historical_vol if historical_vol > 0 else 1.0
            
            if vol_ratio > self.high_vol_threshold:
                return 'high'
            elif vol_ratio < 1.0 / self.high_vol_threshold:
                return 'low'
            else:
                return 'normal'
        else:
            return 'normal'
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate action based on mean reversion analysis"""
        # Update price history
        self.update_history(observation)
        
        # Get base synergy action
        base_action = super().get_action(observation)
        
        # Calculate mean reversion signals
        zscore_signals = self.calculate_zscore_signals()
        squeeze_signals = self.calculate_bollinger_squeeze()
        vol_regime = self.calculate_volatility_regime()
        
        # Combine signals
        combined_signal = (
            0.6 * zscore_signals['zscore_signal'] +
            0.4 * squeeze_signals['squeeze_direction'] * squeeze_signals['squeeze_strength']
        )
        
        # Adjust for volatility regime
        if vol_regime == 'high':
            combined_signal *= 0.5  # Reduce position in high volatility
        elif vol_regime == 'low':
            combined_signal *= 1.2  # Increase position in low volatility
        
        # Calculate position size
        position_size = min(
            self.mean_rev_position_scale * (1.0 + abs(zscore_signals['zscore']) / 4.0),
            self.squeeze_position_scale * (1.0 + squeeze_signals['squeeze_strength'])
        )
        position_size = min(position_size, 0.9)  # Cap at 90%
        
        # Generate mean reversion action
        if abs(combined_signal) > 0.3:
            if combined_signal > 0:  # Bullish
                mean_rev_action = np.array([0.1, 0.1, position_size])
            else:  # Bearish
                mean_rev_action = np.array([position_size, 0.1, 0.1])
        else:
            mean_rev_action = np.array([0.2, 0.6, 0.2])
        
        # Normalize
        mean_rev_action = mean_rev_action / mean_rev_action.sum()
        
        # Combine with base action
        final_action = 0.3 * base_action + 0.7 * mean_rev_action
        final_action = final_action / final_action.sum()
        
        # Store signals
        self.mean_rev_signals.append({
            'zscore': zscore_signals,
            'squeeze': squeeze_signals,
            'vol_regime': vol_regime,
            'combined_signal': combined_signal,
            'position_size': position_size
        })
        
        return final_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.price_history = []
        self.mean_rev_signals = []