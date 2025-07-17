"""
Regime-Aware Reward Function - The Contextual Judge

A sophisticated reward function that adapts to market regimes and provides contextually
intelligent rewards that encourage appropriate behavior for each market condition.

Key Features:
- Regime-specific reward multipliers and penalties
- Context-aware behavior incentives
- Conservative bias during crisis periods
- Trend-following rewards during trending markets
- Risk-adjusted reward scaling based on market volatility

Reward Philosophy:
- Crisis: Heavily reward conservative actions and risk management
- High Vol: Reward profitable trades more, penalize large losses more severely
- Low Vol: Slightly penalize inaction to encourage strategic waiting
- Trending: Reward trend-following behavior and momentum alignment
- Recovery: Reward gradual position building and patient capital deployment

Author: Agent Gamma - The Contextual Judge
Version: 1.0 - Regime-Aware Intelligence
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time
from dataclasses import dataclass, asdict

# Import regime detection
from ..intelligence.regime_detector import RegimeDetector, RegimeAnalysis, MarketRegime, create_regime_detector

logger = logging.getLogger(__name__)

@dataclass
class RewardAnalysis:
    """Detailed breakdown of regime-aware reward calculation."""
    base_reward: float
    regime_adjusted_reward: float
    final_reward: float
    regime: str
    regime_confidence: float
    multipliers_applied: Dict[str, float]
    bonuses_applied: Dict[str, float]
    regime_rationale: str
    timestamp: float

class RegimeAwareRewardFunction:
    """
    Sophisticated reward function that adapts to market regimes.
    
    Rewards appropriate behavior for each market condition:
    - Crisis: Heavily reward conservative actions and risk management
    - High Vol: Reward profitable trades more, penalize large losses more
    - Low Vol: Slightly penalize to encourage waiting for better opportunities
    - Trending: Reward trend-following behavior
    - Recovery: Reward gradual position building
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime detector
        self.regime_detector = create_regime_detector(config.get('regime_detection', {}))
        
        # Regime-specific reward multipliers
        self.regime_multipliers = {
            MarketRegime.CRISIS: {
                'profit_multiplier': 2.0,      # Double reward for profits in crisis
                'loss_multiplier': 0.5,        # Reduce penalty for small losses
                'conservative_bonus': 0.3,     # Bonus for not trading in crisis
                'risk_penalty_factor': 3.0     # Triple penalty for high-risk actions
            },
            MarketRegime.HIGH_VOLATILITY: {
                'profit_multiplier': 1.5,      # Higher reward for profitable trades
                'loss_multiplier': 1.2,        # Slightly higher penalty for losses
                'conservative_bonus': 0.1,     # Small bonus for conservative actions
                'risk_penalty_factor': 2.0     # Double penalty for excessive risk
            },
            MarketRegime.LOW_VOLATILITY: {
                'profit_multiplier': 0.9,      # Slightly lower profit rewards
                'loss_multiplier': 1.0,        # Normal loss penalties
                'conservative_bonus': -0.1,    # Small penalty to encourage action
                'risk_penalty_factor': 1.0     # Normal risk penalties
            },
            MarketRegime.BULL_TREND: {
                'profit_multiplier': 1.3,      # Reward trend-following profits
                'loss_multiplier': 0.8,        # Reduce penalty for trend losses
                'conservative_bonus': -0.2,    # Penalize missing trend opportunities
                'risk_penalty_factor': 1.5     # Moderate risk penalty
            },
            MarketRegime.BEAR_TREND: {
                'profit_multiplier': 1.4,      # Reward shorting profits
                'loss_multiplier': 0.9,        # Moderate loss penalties
                'conservative_bonus': 0.1,     # Small bonus for defensive actions
                'risk_penalty_factor': 1.8     # Higher risk penalty in bear market
            },
            MarketRegime.RECOVERY: {
                'profit_multiplier': 1.2,      # Moderate profit rewards
                'loss_multiplier': 0.7,        # Lower loss penalties during recovery
                'conservative_bonus': 0.05,    # Small conservative bonus
                'risk_penalty_factor': 1.2     # Slightly higher risk penalty
            },
            MarketRegime.SIDEWAYS: {
                'profit_multiplier': 1.0,      # Baseline rewards
                'loss_multiplier': 1.0,        # Baseline penalties
                'conservative_bonus': 0.0,     # No bonus/penalty
                'risk_penalty_factor': 1.0     # Baseline risk penalty
            }
        }
        
        # Performance tracking
        self.regime_performance_history = {regime: [] for regime in MarketRegime}
        self.reward_history = []
        self.regime_transition_rewards = []
        
        # Configuration
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.max_reward_scale = config.get('max_reward_scale', 5.0)
        self.min_reward_scale = config.get('min_reward_scale', -5.0)
        
        self.logger.info("Regime-Aware Reward Function initialized with contextual intelligence")
        
    def compute_reward(
        self, 
        trade_outcome: Dict[str, Any], 
        market_context: Dict[str, Any],
        agent_action: Dict[str, Any]
    ) -> float:
        """
        Compute regime-aware reward for agent action.
        
        Args:
            trade_outcome: Dictionary with trade results (pnl, risk_metrics, etc.)
            market_context: Current market context for regime detection
            agent_action: Action taken by agent (for risk assessment)
            
        Returns:
            Regime-adjusted reward value
        """
        try:
            # Detect current market regime
            regime_analysis = self.regime_detector.detect_regime(market_context)
            regime = regime_analysis.regime
            confidence = regime_analysis.confidence
            
            # Calculate base reward (traditional P&L based)
            base_reward = self._calculate_base_reward(trade_outcome)
            
            # Get regime-specific multipliers
            multipliers = self.regime_multipliers[regime]
            
            # Apply regime-aware adjustments
            regime_adjusted_reward = self._apply_regime_adjustments(
                base_reward, trade_outcome, agent_action, multipliers, confidence
            )
            
            # Add regime-specific bonuses/penalties
            final_reward = self._add_regime_bonuses(
                regime_adjusted_reward, regime, agent_action, multipliers, confidence
            )
            
            # Apply risk scaling based on market volatility
            final_reward = self._apply_volatility_scaling(
                final_reward, regime_analysis.volatility, regime
            )
            
            # Ensure reward bounds
            final_reward = np.clip(final_reward, self.min_reward_scale, self.max_reward_scale)
            
            # Create detailed analysis
            analysis = RewardAnalysis(
                base_reward=base_reward,
                regime_adjusted_reward=regime_adjusted_reward,
                final_reward=final_reward,
                regime=regime.value,
                regime_confidence=confidence,
                multipliers_applied={
                    'profit_multiplier': multipliers['profit_multiplier'],
                    'loss_multiplier': multipliers['loss_multiplier'],
                    'risk_penalty_factor': multipliers['risk_penalty_factor']
                },
                bonuses_applied=self._get_applied_bonuses(regime, agent_action, multipliers, confidence),
                regime_rationale=self._get_regime_rationale(regime, regime_analysis),
                timestamp=time.time()
            )
            
            # Log reward calculation for analysis
            self._log_reward_calculation(analysis, market_context, agent_action)
            
            # Update performance tracking
            self._update_performance_tracking(regime, final_reward, analysis)
            
            return final_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating regime-aware reward: {e}")
            # Fallback to base reward
            return self._calculate_base_reward(trade_outcome)
    
    def _calculate_base_reward(self, trade_outcome: Dict[str, Any]) -> float:
        """Calculate traditional P&L-based reward."""
        pnl = trade_outcome.get('pnl', 0.0)
        risk_penalty = trade_outcome.get('risk_penalty', 0.0)
        
        # Basic reward = PnL - risk penalty
        base_reward = pnl - risk_penalty
        
        # Normalize to reasonable range using tanh
        return np.tanh(base_reward / 1000.0)  # Normalize around $1000 trades
    
    def _apply_regime_adjustments(
        self, 
        base_reward: float, 
        trade_outcome: Dict[str, Any],
        agent_action: Dict[str, Any],
        multipliers: Dict[str, float],
        confidence: float
    ) -> float:
        """Apply regime-specific reward adjustments."""
        
        adjusted_reward = base_reward
        
        # Apply profit/loss multipliers based on regime
        if base_reward > 0:  # Profitable trade
            adjusted_reward *= multipliers['profit_multiplier']
        else:  # Loss trade
            adjusted_reward *= multipliers['loss_multiplier']
        
        # Apply risk penalty based on regime
        risk_level = self._assess_action_risk(agent_action, trade_outcome)
        if risk_level > 0.5:  # High-risk action
            risk_penalty = risk_level * multipliers['risk_penalty_factor'] * 0.1
            adjusted_reward -= risk_penalty
        
        # Weight adjustment by regime detection confidence
        confidence_weight = 0.7 + 0.3 * confidence  # Range: [0.7, 1.0]
        adjusted_reward = base_reward + confidence_weight * (adjusted_reward - base_reward)
        
        return adjusted_reward
    
    def _add_regime_bonuses(
        self,
        reward: float,
        regime: MarketRegime, 
        agent_action: Dict[str, Any],
        multipliers: Dict[str, float],
        confidence: float
    ) -> float:
        """Add regime-specific bonuses for appropriate behavior."""
        
        final_reward = reward
        
        # Conservative action bonus (relevant for crisis/high volatility)
        action_type = agent_action.get('action', 'hold')
        position_size = agent_action.get('position_size', 0.5)
        
        if action_type == 'hold' or position_size < 0.3:
            conservative_bonus = multipliers['conservative_bonus'] * confidence
            final_reward += conservative_bonus
        
        # Regime-specific behavioral rewards
        if regime == MarketRegime.CRISIS:
            # Extra reward for risk management in crisis
            if agent_action.get('stop_loss_used', False):
                final_reward += 0.2 * confidence
            
            # Penalty for aggressive trading in crisis
            if position_size > 0.7:
                final_reward -= 0.3 * confidence
                
        elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # Reward trend-following behavior
            trend_alignment = self._calculate_trend_alignment(agent_action, regime)
            final_reward += trend_alignment * 0.15 * confidence
            
        elif regime == MarketRegime.RECOVERY:
            # Reward gradual position building
            if 0.2 < position_size < 0.6:  # Moderate position sizing
                final_reward += 0.1 * confidence
                
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Encourage strategic patience but penalize complete inaction
            if action_type == 'hold':
                final_reward -= 0.05  # Small penalty for inaction
            else:
                final_reward += 0.05  # Small bonus for taking action
        
        return final_reward
    
    def _apply_volatility_scaling(self, reward: float, volatility: float, regime: MarketRegime) -> float:
        """Apply volatility-based reward scaling."""
        # In high volatility, scale rewards more conservatively
        if volatility > 3.0:  # Very high volatility
            scaling_factor = 0.8
        elif volatility > 2.0:  # High volatility
            scaling_factor = 0.9
        elif volatility < 0.5:  # Very low volatility
            scaling_factor = 1.1  # Encourage action in low vol
        else:
            scaling_factor = 1.0
        
        # Additional scaling for crisis regime
        if regime == MarketRegime.CRISIS:
            scaling_factor *= 0.7  # Very conservative in crisis
        
        return reward * scaling_factor
    
    def _assess_action_risk(self, agent_action: Dict[str, Any], trade_outcome: Dict[str, Any]) -> float:
        """Assess risk level of agent action."""
        position_size = agent_action.get('position_size', 0.5)
        leverage = agent_action.get('leverage', 1.0)
        stop_loss_distance = agent_action.get('stop_loss_distance', 0.02)
        
        # Higher position size and leverage = higher risk
        risk_score = (position_size * leverage) / 2.0
        
        # Lack of stop loss increases risk
        if not agent_action.get('stop_loss_used', False):
            risk_score += 0.3
        elif stop_loss_distance > 0.05:  # Very wide stop loss
            risk_score += 0.2
        
        # Consider actual trade outcome volatility
        if trade_outcome:
            actual_volatility = trade_outcome.get('volatility', 0.01)
            if actual_volatility > 0.03:  # High realized volatility
                risk_score += 0.2
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _calculate_trend_alignment(self, agent_action: Dict[str, Any], regime: MarketRegime) -> float:
        """Calculate how well action aligns with trend."""
        action_type = agent_action.get('action', 'hold')
        position_size = agent_action.get('position_size', 0.5)
        
        if regime == MarketRegime.BULL_TREND:
            if action_type == 'buy':
                return position_size  # Reward based on position size
            elif action_type == 'sell':
                return -0.5 * position_size  # Penalty for counter-trend
            else:
                return -0.1  # Small penalty for inaction in trend
                
        elif regime == MarketRegime.BEAR_TREND:
            if action_type == 'sell':
                return position_size  # Reward based on position size
            elif action_type == 'buy':
                return -0.5 * position_size  # Penalty for counter-trend
            else:
                return -0.1  # Small penalty for inaction in trend
        else:
            return 0.0
    
    def _get_applied_bonuses(self, regime: MarketRegime, agent_action: Dict[str, Any], 
                           multipliers: Dict[str, float], confidence: float) -> Dict[str, float]:
        """Get dictionary of bonuses that were applied."""
        bonuses = {}
        
        action_type = agent_action.get('action', 'hold')
        position_size = agent_action.get('position_size', 0.5)
        
        # Conservative bonus
        if action_type == 'hold' or position_size < 0.3:
            bonuses['conservative_bonus'] = multipliers['conservative_bonus'] * confidence
        
        # Regime-specific bonuses
        if regime == MarketRegime.CRISIS and agent_action.get('stop_loss_used', False):
            bonuses['crisis_risk_management'] = 0.2 * confidence
            
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            trend_alignment = self._calculate_trend_alignment(agent_action, regime)
            bonuses['trend_alignment'] = trend_alignment * 0.15 * confidence
            
        if regime == MarketRegime.RECOVERY and 0.2 < position_size < 0.6:
            bonuses['recovery_positioning'] = 0.1 * confidence
        
        return bonuses
    
    def _get_regime_rationale(self, regime: MarketRegime, regime_analysis: RegimeAnalysis) -> str:
        """Get human-readable rationale for regime classification."""
        rationales = {
            MarketRegime.CRISIS: f"Crisis detected due to high volatility ({regime_analysis.volatility:.2f}) and extreme market conditions",
            MarketRegime.HIGH_VOLATILITY: f"High volatility regime ({regime_analysis.volatility:.2f}) with elevated uncertainty",
            MarketRegime.LOW_VOLATILITY: f"Low volatility regime ({regime_analysis.volatility:.2f}) indicating quiet market conditions",
            MarketRegime.BULL_TREND: f"Bull trend identified with positive momentum ({regime_analysis.momentum:.3f})",
            MarketRegime.BEAR_TREND: f"Bear trend identified with negative momentum ({regime_analysis.momentum:.3f})",
            MarketRegime.RECOVERY: f"Recovery phase with stabilizing volatility ({regime_analysis.volatility:.2f})",
            MarketRegime.SIDEWAYS: f"Sideways market with neutral momentum ({regime_analysis.momentum:.3f})"
        }
        return rationales.get(regime, "Unknown regime")
    
    def _log_reward_calculation(self, analysis: RewardAnalysis, market_context: Dict[str, Any], 
                              agent_action: Dict[str, Any]) -> None:
        """Log detailed reward calculation for analysis."""
        self.logger.debug(
            f"Regime reward: {analysis.regime} (conf: {analysis.regime_confidence:.3f}) | "
            f"Base: {analysis.base_reward:.3f} -> Final: {analysis.final_reward:.3f} | "
            f"Action: {agent_action.get('action', 'unknown')} | "
            f"Vol: {market_context.get('volatility_30', 0):.3f}"
        )
    
    def _update_performance_tracking(self, regime: MarketRegime, reward: float, 
                                   analysis: RewardAnalysis) -> None:
        """Update performance tracking for regime-specific analysis."""
        self.regime_performance_history[regime].append({
            'reward': reward,
            'timestamp': analysis.timestamp,
            'confidence': analysis.regime_confidence
        })
        
        # Keep history manageable
        if len(self.regime_performance_history[regime]) > 1000:
            self.regime_performance_history[regime].pop(0)
        
        # Store overall reward history
        self.reward_history.append(analysis)
        if len(self.reward_history) > 10000:
            self.reward_history.pop(0)
    
    def get_regime_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary by regime."""
        summary = {}
        for regime, rewards in self.regime_performance_history.items():
            if rewards:
                reward_values = [r['reward'] for r in rewards]
                confidences = [r['confidence'] for r in rewards]
                
                summary[regime.value] = {
                    'mean_reward': np.mean(reward_values),
                    'std_reward': np.std(reward_values),
                    'total_trades': len(rewards),
                    'win_rate': sum(1 for r in reward_values if r > 0) / len(reward_values),
                    'avg_confidence': np.mean(confidences),
                    'sharpe_ratio': np.mean(reward_values) / max(np.std(reward_values), 0.001)
                }
        return summary
    
    def get_recent_analysis(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent reward analysis history."""
        recent = self.reward_history[-limit:]
        return [asdict(analysis) for analysis in recent]
    
    def reset_performance_history(self) -> None:
        """Reset performance tracking history."""
        for regime in MarketRegime:
            self.regime_performance_history[regime].clear()
        self.reward_history.clear()
        self.regime_transition_rewards.clear()
        
        self.logger.info("Regime-aware reward performance history reset")
    
    def get_regime_detector_stats(self) -> Dict[str, Any]:
        """Get regime detector performance statistics."""
        return self.regime_detector.get_regime_statistics()

def create_regime_aware_reward_function(config: Optional[Dict[str, Any]] = None) -> RegimeAwareRewardFunction:
    """
    Factory function to create a RegimeAwareRewardFunction with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RegimeAwareRewardFunction instance
    """
    default_config = {
        'risk_free_rate': 0.02,
        'max_reward_scale': 5.0,
        'min_reward_scale': -5.0,
        'regime_detection': {
            'crisis_volatility': 3.5,
            'high_volatility': 2.5,
            'low_volatility': 0.8,
            'strong_momentum': 0.05,
            'weak_momentum': 0.02
        }
    }
    
    if config:
        # Deep merge configurations
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(default_config, config)
    
    return RegimeAwareRewardFunction(default_config)