"""
Algorithms Module

This module contains state-of-the-art algorithmic implementations for
GrandModel system optimization:

- consensus_optimizer: Hierarchical PBFT consensus with O(n log n) complexity
- copula_models: Dynamic copula modeling for enhanced VaR calculation
- adaptive_weights: Adaptive weight learning for MARL coordination

Author: Agent Gamma - Algorithmic Excellence Implementation Specialist
"""

from .consensus_optimizer import (
    HierarchicalConsensusOptimizer,
    MessageBatchOptimizer,
    AdaptiveViewChangeOptimizer,
    create_consensus_optimizer
)

from .copula_models import (
    CopulaVaRCalculator,
    DynamicCopulaSelector,
    CopulaType,
    MarketRegime,
    create_copula_var_calculator,
    compare_copula_models
)

from .adaptive_weights import (
    create_adaptive_weight_learner,
    AdaptationStrategy,
    HybridAdaptiveWeightLearner,
    benchmark_adaptation_strategies
)

__all__ = [
    # Consensus optimization
    'HierarchicalConsensusOptimizer',
    'MessageBatchOptimizer', 
    'AdaptiveViewChangeOptimizer',
    'create_consensus_optimizer',
    
    # Copula modeling
    'CopulaVaRCalculator',
    'DynamicCopulaSelector',
    'CopulaType',
    'MarketRegime',
    'create_copula_var_calculator',
    'compare_copula_models',
    
    # Adaptive weights
    'create_adaptive_weight_learner',
    'AdaptationStrategy',
    'HybridAdaptiveWeightLearner',
    'benchmark_adaptation_strategies'
]