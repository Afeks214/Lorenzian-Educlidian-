"""
MAPPO Training Components for Strategic MARL System

Enhanced with Regime-Aware Contextual Intelligence by Agent Gamma
"""

# from .mappo_trainer import MAPPOTrainer  # Currently missing
from .losses import (
    PPOLoss, GAEAdvantageEstimator, MultiAgentPPOLoss, AdaptiveKLPenalty
)
from .schedulers import (
    LinearDecayScheduler, CosineAnnealingWarmRestarts,
    AdaptiveScheduler, CyclicScheduler, SchedulerManager
)
from .reward_system import RewardSystem, RewardComponents, calculate_reward
from .regime_aware_reward_integration import (
    RegimeAwareRewardSystem, 
    EnhancedRewardComponents,
    create_regime_aware_reward_system,
    calculate_enhanced_reward
)

__all__ = [
    # 'MAPPOTrainer',  # Currently missing
    'PPOLoss',
    'GAEAdvantageEstimator',
    'MultiAgentPPOLoss',
    'AdaptiveKLPenalty',
    'LinearDecayScheduler',
    'CosineAnnealingWarmRestarts',
    'AdaptiveScheduler',
    'CyclicScheduler',
    'SchedulerManager',
    # Reward Systems
    'RewardSystem',
    'RewardComponents', 
    'calculate_reward',
    # Regime-Aware Enhancements
    'RegimeAwareRewardSystem',
    'EnhancedRewardComponents',
    'create_regime_aware_reward_system',
    'calculate_enhanced_reward'
]