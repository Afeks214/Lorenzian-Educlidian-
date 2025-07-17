"""
Regime-Aware Reward Function for Contextual Learning.

Adapts reward calculations based on detected market regime to provide 
contextually appropriate feedback for agent learning. Optimized for <0.2ms performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import logging
from .regime_detector import MarketRegime, RegimeAnalysis

class RegimeAwareRewardFunction:
    """
    Regime-aware reward function that adapts reward calculations based on
    current market regime to provide contextually appropriate learning signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance optimization settings
        self.fast_mode = config.get('fast_mode', True)
        self.cache_rewards = config.get('cache_rewards', True)
        
        # Regime-specific reward parameters
        self.regime_parameters = {
            MarketRegime.BULL_TREND: {
                'trend_weight': 1.5,        # Reward trend following
                'volatility_penalty': 0.5,  # Lower volatility penalty
                'momentum_bonus': 1.2,      # Bonus for momentum alignment
                'risk_tolerance': 1.3       # Higher risk tolerance
            },
            MarketRegime.BEAR_TREND: {
                'trend_weight': 1.5,        # Reward trend following (shorts)
                'volatility_penalty': 0.7,  # Moderate volatility penalty
                'momentum_bonus': 1.2,      # Bonus for momentum alignment
                'risk_tolerance': 0.8       # Lower risk tolerance
            },
            MarketRegime.SIDEWAYS: {
                'trend_weight': 0.3,        # Low trend following reward
                'volatility_penalty': 0.3,  # Low volatility penalty
                'momentum_bonus': 0.5,      # Low momentum bonus
                'risk_tolerance': 0.9       # Moderate risk tolerance
            },
            MarketRegime.CRISIS: {
                'trend_weight': 0.1,        # Very low trend following
                'volatility_penalty': 2.0,  # High volatility penalty
                'momentum_bonus': 0.2,      # Very low momentum bonus
                'risk_tolerance': 0.3       # Very low risk tolerance
            },
            MarketRegime.RECOVERY: {
                'trend_weight': 1.0,        # Moderate trend following
                'volatility_penalty': 0.8,  # Moderate volatility penalty
                'momentum_bonus': 0.8,      # Moderate momentum bonus
                'risk_tolerance': 1.1       # Moderate risk tolerance
            }
        }
        
        # Base reward components weights
        self.base_weights = {
            'pnl_weight': config.get('pnl_weight', 1.0),
            'risk_weight': config.get('risk_weight', 0.5),
            'consistency_weight': config.get('consistency_weight', 0.3),
            'regime_alignment_weight': config.get('regime_alignment_weight', 0.4)
        }
        
        # Performance tracking
        self.reward_history = []
        self.regime_performance = {regime: [] for regime in MarketRegime}
        
        # Reward normalization parameters
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.normalization_window = config.get('normalization_window', 100)
        
        # Caching for performance
        self.reward_cache = {}
        self.cache_ttl = config.get('cache_ttl_seconds', 1)
        
        self.logger.info("Regime-aware reward function initialized")
    
    def compute_reward(
        self,
        action: int,
        action_probabilities: np.ndarray,
        market_context: Dict[str, Any],
        regime_analysis: RegimeAnalysis,
        pnl: float,
        portfolio_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute regime-aware reward for the given action and context.
        
        Args:
            action: Taken action (0=buy, 1=hold, 2=sell)
            action_probabilities: Agent's action probabilities
            market_context: Current market context
            regime_analysis: Current regime analysis
            pnl: Realized P&L from the action
            portfolio_metrics: Portfolio risk metrics
            
        Returns:
            Dictionary with reward components and total reward
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            if self.cache_rewards:
                cache_key = self._create_reward_cache_key(
                    action, regime_analysis.regime, pnl, portfolio_metrics
                )
                cached_reward = self._get_cached_reward(cache_key)
                if cached_reward is not None:
                    return cached_reward
            
            # Get regime-specific parameters
            regime_params = self.regime_parameters[regime_analysis.regime]
            
            # Compute reward components
            reward_components = self._compute_reward_components(
                action, action_probabilities, market_context, 
                regime_analysis, pnl, portfolio_metrics, regime_params
            )
            
            # Combine components with regime-aware weighting
            total_reward = self._combine_reward_components(
                reward_components, regime_params
            )
            
            # Apply normalization
            normalized_reward = self._normalize_reward(total_reward)
            
            # Create result
            reward_result = {
                'total_reward': normalized_reward,
                'raw_reward': total_reward,
                'regime': regime_analysis.regime.value,
                'regime_confidence': regime_analysis.confidence,
                'computation_time_ms': (time.perf_counter() - start_time) * 1000,
                **reward_components
            }
            
            # Cache result
            if self.cache_rewards:
                self._cache_reward(cache_key, reward_result)
            
            # Update tracking
            self._update_reward_tracking(reward_result, regime_analysis.regime)
            
            return reward_result
            
        except Exception as e:
            self.logger.error(f"Error computing regime-aware reward: {e}")
            
            # Return fallback reward
            return {
                'total_reward': 0.0,
                'raw_reward': 0.0,
                'regime': regime_analysis.regime.value,
                'regime_confidence': regime_analysis.confidence,
                'computation_time_ms': (time.perf_counter() - start_time) * 1000,
                'pnl_reward': 0.0,
                'risk_penalty': 0.0,
                'regime_alignment_bonus': 0.0,
                'consistency_bonus': 0.0
            }
    
    def _compute_reward_components(
        self,
        action: int,
        action_probabilities: np.ndarray,
        market_context: Dict[str, Any],
        regime_analysis: RegimeAnalysis,
        pnl: float,
        portfolio_metrics: Dict[str, float],
        regime_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute individual reward components."""
        
        components = {}
        
        # 1. P&L-based reward (core component)
        components['pnl_reward'] = self._compute_pnl_reward(pnl, regime_params)
        
        # 2. Risk penalty (regime-adjusted)
        components['risk_penalty'] = self._compute_risk_penalty(
            portfolio_metrics, regime_params
        )
        
        # 3. Regime alignment bonus
        components['regime_alignment_bonus'] = self._compute_regime_alignment_bonus(
            action, regime_analysis, market_context
        )
        
        # 4. Consistency bonus (stable decision making)
        components['consistency_bonus'] = self._compute_consistency_bonus(
            action_probabilities, regime_analysis
        )
        
        # 5. Momentum alignment (regime-specific)
        components['momentum_alignment'] = self._compute_momentum_alignment(
            action, market_context, regime_params
        )
        
        return components
    
    def _compute_pnl_reward(self, pnl: float, regime_params: Dict[str, float]) -> float:
        """Compute P&L-based reward with regime adjustments."""
        
        # Base P&L reward
        pnl_reward = np.tanh(pnl * 10)  # Bounded between -1 and 1
        
        # Apply regime-specific risk tolerance
        risk_adjusted_reward = pnl_reward * regime_params['risk_tolerance']
        
        return float(risk_adjusted_reward)
    
    def _compute_risk_penalty(
        self, 
        portfolio_metrics: Dict[str, float], 
        regime_params: Dict[str, float]
    ) -> float:
        """Compute risk penalty with regime adjustments."""
        
        # Extract risk metrics
        volatility = portfolio_metrics.get('volatility', 0.1)
        drawdown = portfolio_metrics.get('max_drawdown', 0.0)
        leverage = portfolio_metrics.get('leverage', 1.0)
        
        # Base risk penalty
        risk_penalty = (
            volatility * 0.5 +
            abs(drawdown) * 1.0 +
            max(0, leverage - 1.0) * 0.3
        )
        
        # Apply regime-specific volatility penalty
        regime_adjusted_penalty = risk_penalty * regime_params['volatility_penalty']
        
        return -float(regime_adjusted_penalty)  # Negative because it's a penalty
    
    def _compute_regime_alignment_bonus(
        self,
        action: int,
        regime_analysis: RegimeAnalysis,
        market_context: Dict[str, Any]
    ) -> float:
        """Compute bonus for regime-appropriate actions."""
        
        regime = regime_analysis.regime
        confidence = regime_analysis.confidence
        
        # Define regime-appropriate actions
        if regime == MarketRegime.BULL_TREND:
            # Reward long positions
            if action == 0:  # Buy
                alignment_score = 1.0
            elif action == 1:  # Hold
                alignment_score = 0.3
            else:  # Sell
                alignment_score = -0.5
                
        elif regime == MarketRegime.BEAR_TREND:
            # Reward short positions
            if action == 2:  # Sell
                alignment_score = 1.0
            elif action == 1:  # Hold
                alignment_score = 0.3
            else:  # Buy
                alignment_score = -0.5
                
        elif regime == MarketRegime.CRISIS:
            # Reward defensive positioning
            if action == 2:  # Sell (defensive)
                alignment_score = 1.0
            elif action == 1:  # Hold
                alignment_score = 0.5
            else:  # Buy
                alignment_score = -1.0
                
        else:  # SIDEWAYS or RECOVERY
            # Neutral positioning preferred
            if action == 1:  # Hold
                alignment_score = 1.0
            else:
                alignment_score = 0.0
        
        # Weight by regime confidence
        bonus = alignment_score * confidence * 0.5
        
        return float(bonus)
    
    def _compute_consistency_bonus(
        self,
        action_probabilities: np.ndarray,
        regime_analysis: RegimeAnalysis
    ) -> float:
        """Compute bonus for consistent decision making."""
        
        # Calculate decision confidence (entropy-based)
        entropy = -np.sum(action_probabilities * np.log(action_probabilities + 1e-8))
        max_entropy = np.log(len(action_probabilities))
        
        # Lower entropy = higher consistency
        consistency_score = 1.0 - (entropy / max_entropy)
        
        # Weight by regime confidence
        bonus = consistency_score * regime_analysis.confidence * 0.3
        
        return float(bonus)
    
    def _compute_momentum_alignment(
        self,
        action: int,
        market_context: Dict[str, Any],
        regime_params: Dict[str, float]
    ) -> float:
        """Compute bonus for momentum alignment."""
        
        momentum_20 = market_context.get('momentum_20', 0.0)
        momentum_50 = market_context.get('momentum_50', 0.0)
        
        # Combined momentum signal
        momentum_signal = momentum_20 * 0.7 + momentum_50 * 0.3
        
        # Check action-momentum alignment
        if momentum_signal > 0.01:  # Positive momentum
            alignment = 1.0 if action == 0 else (0.0 if action == 1 else -1.0)
        elif momentum_signal < -0.01:  # Negative momentum
            alignment = 1.0 if action == 2 else (0.0 if action == 1 else -1.0)
        else:  # Neutral momentum
            alignment = 1.0 if action == 1 else 0.0
        
        # Apply regime-specific momentum bonus
        bonus = alignment * abs(momentum_signal) * regime_params['momentum_bonus'] * 0.2
        
        return float(bonus)
    
    def _combine_reward_components(
        self,
        components: Dict[str, float],
        regime_params: Dict[str, float]
    ) -> float:
        """Combine reward components with regime-aware weighting."""
        
        total_reward = (
            components['pnl_reward'] * self.base_weights['pnl_weight'] +
            components['risk_penalty'] * self.base_weights['risk_weight'] +
            components['regime_alignment_bonus'] * self.base_weights['regime_alignment_weight'] +
            components['consistency_bonus'] * self.base_weights['consistency_weight'] +
            components['momentum_alignment'] * regime_params['momentum_bonus']
        )
        
        return float(total_reward)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        
        # Update running statistics
        self.reward_history.append(reward)
        if len(self.reward_history) > self.normalization_window:
            self.reward_history.pop(0)
        
        if len(self.reward_history) >= 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8
        
        # Normalize
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        
        # Clip to reasonable bounds
        normalized_reward = np.clip(normalized_reward, -3.0, 3.0)
        
        return float(normalized_reward)
    
    def _create_reward_cache_key(
        self,
        action: int,
        regime: MarketRegime,
        pnl: float,
        portfolio_metrics: Dict[str, float]
    ) -> str:
        """Create cache key for reward computation."""
        
        key_values = [
            action,
            regime.value,
            round(pnl, 4),
            round(portfolio_metrics.get('volatility', 0), 3),
            round(portfolio_metrics.get('leverage', 1), 2)
        ]
        
        return f"reward_{hash(tuple(key_values))}"
    
    def _get_cached_reward(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Get cached reward result."""
        if cache_key in self.reward_cache:
            result, timestamp = self.reward_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.reward_cache[cache_key]
        return None
    
    def _cache_reward(self, cache_key: str, reward_result: Dict[str, float]):
        """Cache reward result."""
        self.reward_cache[cache_key] = (reward_result, time.time())
        
        # Limit cache size
        if len(self.reward_cache) > 200:
            oldest_key = min(self.reward_cache.keys(),
                           key=lambda k: self.reward_cache[k][1])
            del self.reward_cache[oldest_key]
    
    def _update_reward_tracking(self, reward_result: Dict[str, float], regime: MarketRegime):
        """Update reward tracking statistics."""
        
        self.regime_performance[regime].append(reward_result['total_reward'])
        
        # Limit history per regime
        if len(self.regime_performance[regime]) > 50:
            self.regime_performance[regime].pop(0)
    
    def get_regime_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics by regime."""
        
        stats = {}
        
        for regime, rewards in self.regime_performance.items():
            if rewards:
                stats[regime.value] = {
                    'mean_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'num_samples': len(rewards),
                    'best_reward': float(np.max(rewards)),
                    'worst_reward': float(np.min(rewards))
                }
            else:
                stats[regime.value] = {'num_samples': 0}
        
        return stats
    
    def reset_tracking(self):
        """Reset all tracking statistics."""
        self.reward_history.clear()
        for regime in self.regime_performance:
            self.regime_performance[regime].clear()
        self.reward_cache.clear()
        
        self.logger.info("Reward tracking reset")