"""
Regime-Aware Reward System Integration

This module integrates the regime-aware reward function with the existing
training reward system, providing contextual intelligence that adapts to
market conditions while maintaining compatibility with the current training pipeline.

Key Features:
- Seamless integration with existing RewardSystem
- Regime-aware reward adjustments
- Backward compatibility for existing training code
- Enhanced performance tracking and analysis
- Market context-aware training adaptation

Author: Agent Gamma - The Contextual Judge
Version: 1.0 - Training Integration
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, asdict
import time

# Import existing components
from .reward_system import RewardSystem, RewardComponents

# Import regime-aware components
from src.agents.regime_aware_reward import (
    RegimeAwareRewardFunction, 
    create_regime_aware_reward_function,
    RewardAnalysis
)
from src.intelligence.regime_detector import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRewardComponents(RewardComponents):
    """Enhanced reward components with regime information."""
    # Existing components from RewardComponents
    # pnl: float
    # synergy: float
    # risk: float  
    # exploration: float
    # total: float
    
    # New regime-aware components
    regime: str = "unknown"
    regime_confidence: float = 0.5
    regime_adjusted_total: float = 0.0
    base_vs_regime_diff: float = 0.0
    regime_rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including regime components."""
        base_dict = super().to_dict()
        base_dict.update({
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'regime_adjusted_total': self.regime_adjusted_total,
            'base_vs_regime_diff': self.base_vs_regime_diff,
            'regime_rationale': self.regime_rationale
        })
        return base_dict

class RegimeAwareRewardSystem(RewardSystem):
    """
    Enhanced reward system with regime awareness.
    
    Extends the existing RewardSystem to incorporate market regime detection
    and context-aware reward adjustments while maintaining full backward
    compatibility with existing training code.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None, 
                 enable_regime_awareness: bool = True):
        """
        Initialize Enhanced Regime-Aware Reward System.
        
        Args:
            config_path: Path to configuration YAML file
            config: Direct configuration dictionary (overrides config_path)
            enable_regime_awareness: Whether to enable regime-aware adjustments
        """
        # Initialize parent RewardSystem
        super().__init__(config_path, config)
        
        self.enable_regime_awareness = enable_regime_awareness
        
        if self.enable_regime_awareness:
            # Initialize regime-aware reward function
            regime_config = self._extract_regime_config()
            self.regime_aware_function = create_regime_aware_reward_function(regime_config)
            
            # Additional tracking for regime analysis
            self.regime_performance_tracker = {}
            self.regime_transition_history = []
            self.training_adaptation_history = []
            
            logger.info("Regime-Aware Reward System initialized with contextual intelligence")
        else:
            self.regime_aware_function = None
            logger.info("Regime-Aware Reward System initialized in legacy mode (regime awareness disabled)")
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Union[RewardComponents, EnhancedRewardComponents]:
        """
        Calculate multi-objective reward with optional regime awareness.
        
        Args:
            state: Current environment state
            action: Agent action (probability distribution)
            next_state: Next environment state
            info: Additional information (synergy, market context, etc.)
            
        Returns:
            EnhancedRewardComponents with regime adjustments (if enabled)
            or standard RewardComponents (if regime awareness disabled)
        """
        # Calculate base rewards using parent implementation
        base_rewards = super().calculate_reward(state, action, next_state, info)
        
        if not self.enable_regime_awareness:
            return base_rewards
        
        try:
            # Extract market context for regime detection
            market_context = self._extract_market_context(state, info)
            
            # Create agent action context for regime-aware analysis
            agent_action = self._convert_action_to_agent_format(action, info)
            
            # Create trade outcome context
            trade_outcome = self._extract_trade_outcome(state, next_state, info, base_rewards)
            
            # Calculate regime-aware reward adjustment
            regime_adjusted_reward = self.regime_aware_function.compute_reward(
                trade_outcome=trade_outcome,
                market_context=market_context,
                agent_action=agent_action
            )
            
            # Get regime analysis for detailed tracking
            regime_analysis = self.regime_aware_function.regime_detector.detect_regime(market_context)
            
            # Create enhanced reward components
            enhanced_rewards = EnhancedRewardComponents(
                pnl=base_rewards.pnl,
                synergy=base_rewards.synergy,
                risk=base_rewards.risk,
                exploration=base_rewards.exploration,
                total=base_rewards.total,  # Keep original total as baseline
                regime=regime_analysis.regime.value,
                regime_confidence=regime_analysis.confidence,
                regime_adjusted_total=regime_adjusted_reward,
                base_vs_regime_diff=regime_adjusted_reward - base_rewards.total,
                regime_rationale=self._get_regime_rationale(regime_analysis, trade_outcome, agent_action)
            )
            
            # Update regime-specific tracking
            self._update_regime_tracking(enhanced_rewards, market_context, action)
            
            # Log significant regime adjustments
            if abs(enhanced_rewards.base_vs_regime_diff) > 0.1:
                logger.debug(
                    f"Significant regime adjustment: {regime_analysis.regime.value} "
                    f"(conf: {regime_analysis.confidence:.3f}) | "
                    f"Base: {base_rewards.total:.3f} -> Regime: {regime_adjusted_reward:.3f} "
                    f"(diff: {enhanced_rewards.base_vs_regime_diff:+.3f})"
                )
            
            return enhanced_rewards
            
        except Exception as e:
            logger.error(f"Error in regime-aware reward calculation: {e}")
            # Fallback to base rewards
            return base_rewards
    
    def _extract_regime_config(self) -> Dict[str, Any]:
        """Extract regime-specific configuration from main config."""
        regime_config = {
            'risk_free_rate': 0.02,
            'max_reward_scale': 5.0,
            'min_reward_scale': -5.0
        }
        
        # Extract regime detection parameters if available
        if 'regime_detection' in self.config:
            regime_config['regime_detection'] = self.config['regime_detection']
        
        # Extract regime-specific multipliers if available
        if 'regime_multipliers' in self.config:
            regime_config['regime_multipliers'] = self.config['regime_multipliers']
        
        return regime_config
    
    def _extract_market_context(self, state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context for regime detection."""
        market_context = {}
        
        # Extract volatility measures
        market_context['volatility_30'] = info.get('volatility_30', 
                                                 state.get('volatility_30', 1.0))
        
        # Extract momentum measures
        market_context['momentum_20'] = info.get('momentum_20',
                                               state.get('momentum_20', 0.0))
        market_context['momentum_50'] = info.get('momentum_50',
                                               state.get('momentum_50', 0.0))
        
        # Extract volume measures
        market_context['volume_ratio'] = info.get('volume_ratio',
                                                state.get('volume_ratio', 1.0))
        
        # Extract MMD score
        market_context['mmd_score'] = info.get('mmd_score',
                                             state.get('mmd_score', 0.0))
        
        # Extract any additional market features
        for key in ['price_momentum_5', 'volume_profile_skew', 'market_stress']:
            if key in info:
                market_context[key] = info[key]
            elif key in state:
                market_context[key] = state[key]
        
        return market_context
    
    def _convert_action_to_agent_format(self, action: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MARL action to agent action format."""
        # action is typically [p_bearish, p_neutral, p_bullish]
        
        # Determine primary action
        action_idx = np.argmax(action)
        action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        primary_action = action_map[action_idx]
        
        # Estimate position size from action confidence
        action_confidence = np.max(action)
        position_size = action_confidence * 0.5  # Conservative scaling
        
        # Extract additional action context from info
        agent_action = {
            'action': primary_action,
            'position_size': position_size,
            'leverage': info.get('leverage', 1.0),
            'stop_loss_used': info.get('stop_loss_used', True),  # Assume good practices
            'stop_loss_distance': info.get('stop_loss_distance', 0.02)
        }
        
        return agent_action
    
    def _extract_trade_outcome(self, state: Dict[str, Any], next_state: Dict[str, Any], 
                             info: Dict[str, Any], base_rewards: RewardComponents) -> Dict[str, Any]:
        """Extract trade outcome for regime-aware analysis."""
        
        # Use PnL from info if available, otherwise estimate from rewards
        pnl = info.get('pnl', base_rewards.pnl * self.pnl_normalizer)
        
        trade_outcome = {
            'pnl': pnl,
            'risk_penalty': abs(base_rewards.risk) * 100,  # Convert back to dollar terms
            'volatility': info.get('realized_volatility', 0.02),
            'drawdown': info.get('drawdown', 0.0),
            'slippage': info.get('slippage', 0.001)
        }
        
        return trade_outcome
    
    def _get_regime_rationale(self, regime_analysis, trade_outcome: Dict[str, Any], 
                            agent_action: Dict[str, Any]) -> str:
        """Generate human-readable rationale for regime-based adjustment."""
        regime = regime_analysis.regime
        confidence = regime_analysis.confidence
        pnl = trade_outcome['pnl']
        action = agent_action['action']
        
        if regime == MarketRegime.CRISIS:
            if action == 'hold' and pnl >= 0:
                return f"Crisis regime (conf: {confidence:.2f}): Conservative action rewarded"
            elif action != 'hold' and pnl > 0:
                return f"Crisis regime (conf: {confidence:.2f}): Profitable trade heavily rewarded"
            else:
                return f"Crisis regime (conf: {confidence:.2f}): Risk management emphasized"
                
        elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            trend_dir = "upward" if regime == MarketRegime.BULL_TREND else "downward"
            expected_action = "buy" if regime == MarketRegime.BULL_TREND else "sell"
            
            if action == expected_action:
                return f"{trend_dir.title()} trend (conf: {confidence:.2f}): Trend-following rewarded"
            else:
                return f"{trend_dir.title()} trend (conf: {confidence:.2f}): Counter-trend penalty applied"
                
        elif regime == MarketRegime.LOW_VOLATILITY:
            return f"Low volatility (conf: {confidence:.2f}): Action encouraged over inaction"
            
        else:
            return f"{regime.value} regime (conf: {confidence:.2f}): Standard reward adjustment"
    
    def _update_regime_tracking(self, enhanced_rewards: EnhancedRewardComponents, 
                              market_context: Dict[str, Any], action: np.ndarray):
        """Update regime-specific performance tracking."""
        regime = enhanced_rewards.regime
        
        if regime not in self.regime_performance_tracker:
            self.regime_performance_tracker[regime] = {
                'rewards': [],
                'confidences': [],
                'adjustments': [],
                'actions': [],
                'market_contexts': []
            }
        
        tracker = self.regime_performance_tracker[regime]
        tracker['rewards'].append(enhanced_rewards.regime_adjusted_total)
        tracker['confidences'].append(enhanced_rewards.regime_confidence)
        tracker['adjustments'].append(enhanced_rewards.base_vs_regime_diff)
        tracker['actions'].append(action.tolist())
        tracker['market_contexts'].append(market_context.copy())
        
        # Keep only recent data
        max_history = 1000
        if len(tracker['rewards']) > max_history:
            for key in tracker:
                tracker[key] = tracker[key][-max_history:]
    
    def get_regime_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive regime-specific performance summary."""
        if not self.enable_regime_awareness:
            return {"regime_awareness": "disabled"}
        
        summary = {}
        
        for regime, tracker in self.regime_performance_tracker.items():
            if not tracker['rewards']:
                continue
                
            rewards = np.array(tracker['rewards'])
            confidences = np.array(tracker['confidences'])
            adjustments = np.array(tracker['adjustments'])
            
            summary[regime] = {
                'total_episodes': len(rewards),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'mean_confidence': float(np.mean(confidences)),
                'mean_adjustment': float(np.mean(adjustments)),
                'positive_adjustment_rate': float(np.mean(adjustments > 0)),
                'significant_adjustment_rate': float(np.mean(np.abs(adjustments) > 0.1)),
                'reward_improvement': float(np.mean(adjustments)),  # Average improvement
                'sharpe_ratio': float(np.mean(rewards) / max(np.std(rewards), 0.001))
            }
        
        # Add regime-aware function performance if available
        if hasattr(self.regime_aware_function, 'get_regime_performance_summary'):
            regime_function_summary = self.regime_aware_function.get_regime_performance_summary()
            summary['regime_function_performance'] = regime_function_summary
        
        return summary
    
    def get_training_adaptation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for training adaptation based on regime performance."""
        if not self.enable_regime_awareness:
            return {"regime_awareness": "disabled"}
        
        recommendations = {
            'timestamp': time.time(),
            'regime_distribution': {},
            'adaptation_suggestions': [],
            'performance_alerts': []
        }
        
        # Analyze regime distribution
        total_episodes = sum(len(tracker['rewards']) for tracker in self.regime_performance_tracker.values())
        
        if total_episodes == 0:
            return recommendations
        
        for regime, tracker in self.regime_performance_tracker.items():
            regime_episodes = len(tracker['rewards'])
            if regime_episodes > 0:
                recommendations['regime_distribution'][regime] = {
                    'frequency': regime_episodes / total_episodes,
                    'episodes': regime_episodes
                }
        
        # Generate adaptation suggestions
        for regime, tracker in self.regime_performance_tracker.items():
            if len(tracker['rewards']) < 10:  # Need minimum data
                continue
                
            rewards = np.array(tracker['rewards'])
            adjustments = np.array(tracker['adjustments'])
            confidences = np.array(tracker['confidences'])
            
            # Check for consistent negative adjustments
            if np.mean(adjustments) < -0.1:
                recommendations['adaptation_suggestions'].append(
                    f"Consider reducing base reward weights for {regime} regime - "
                    f"consistently negative adjustments ({np.mean(adjustments):.3f})"
                )
            
            # Check for low confidence
            if np.mean(confidences) < 0.6:
                recommendations['performance_alerts'].append(
                    f"Low regime detection confidence for {regime} ({np.mean(confidences):.3f}) - "
                    f"consider tuning detection thresholds"
                )
            
            # Check for high variance in rewards
            if np.std(rewards) > 1.0:
                recommendations['performance_alerts'].append(
                    f"High reward variance in {regime} regime ({np.std(rewards):.3f}) - "
                    f"consider reward scaling adjustments"
                )
        
        return recommendations
    
    def enable_regime_awareness_mode(self, enable: bool = True) -> None:
        """Enable or disable regime awareness mode."""
        if enable and not self.enable_regime_awareness:
            # Initialize regime components
            regime_config = self._extract_regime_config()
            self.regime_aware_function = create_regime_aware_reward_function(regime_config)
            self.regime_performance_tracker = {}
            self.enable_regime_awareness = True
            logger.info("Regime awareness enabled")
            
        elif not enable and self.enable_regime_awareness:
            # Disable regime components
            self.regime_aware_function = None
            self.enable_regime_awareness = False
            logger.info("Regime awareness disabled")
    
    def reset_regime_tracking(self) -> None:
        """Reset regime-specific tracking data."""
        if self.enable_regime_awareness:
            self.regime_performance_tracker.clear()
            self.regime_transition_history.clear()
            self.training_adaptation_history.clear()
            
            if self.regime_aware_function:
                self.regime_aware_function.reset_performance_history()
            
            logger.info("Regime tracking data reset")


# Convenience factory function
def create_regime_aware_reward_system(config_path: Optional[str] = None, 
                                     config: Optional[Dict] = None,
                                     enable_regime_awareness: bool = True) -> RegimeAwareRewardSystem:
    """
    Factory function to create a RegimeAwareRewardSystem.
    
    Args:
        config_path: Path to configuration YAML file
        config: Direct configuration dictionary (overrides config_path)
        enable_regime_awareness: Whether to enable regime-aware adjustments
        
    Returns:
        Configured RegimeAwareRewardSystem instance
    """
    return RegimeAwareRewardSystem(config_path, config, enable_regime_awareness)


# Backward compatibility function
def calculate_enhanced_reward(
    state: Dict[str, Any],
    action: np.ndarray,
    next_state: Dict[str, Any],
    info: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    enable_regime_awareness: bool = True
) -> Dict[str, Any]:
    """
    Calculate enhanced reward with optional regime awareness.
    
    Args:
        state: Current environment state
        action: Agent action (probability distribution)
        next_state: Next environment state  
        info: Additional information
        config: Optional configuration override
        enable_regime_awareness: Whether to enable regime adjustments
        
    Returns:
        Dictionary of enhanced reward components
    """
    reward_system = create_regime_aware_reward_system(
        config=config, 
        enable_regime_awareness=enable_regime_awareness
    )
    rewards = reward_system.calculate_reward(state, action, next_state, info)
    return rewards.to_dict()