"""Baseline agents for performance comparison"""

# Core baseline agents
from .rule_based_agent import (
    RuleBasedAgent, 
    TechnicalRuleBasedAgent,
    EnhancedRuleBasedAgent,
    AdvancedMomentumAgent,
    AdvancedMeanReversionAgent
)

from .random_agent import (
    RandomAgent, 
    BiasedRandomAgent, 
    ContextualRandomAgent
)

# Momentum strategy agents
from .momentum_strategies import (
    MACDCrossoverAgent,
    RSIAgent,
    DualMomentumAgent,
    BreakoutAgent
)

# Benchmark agents
from .benchmark_agents import (
    BuyAndHoldAgent,
    EqualWeightAgent,
    MarketCapWeightedAgent,
    SectorRotationAgent,
    RiskParityAgent
)

# Technical indicators
from .technical_indicators import (
    TechnicalIndicators,
    AdvancedTechnicalIndicators,
    IndicatorSignals,
    IndicatorCombinations,
    PerformanceOptimizer
)

# Performance benchmarking
from .performance_benchmark import PerformanceBenchmark

__all__ = [
    # Core agents
    'RuleBasedAgent', 
    'TechnicalRuleBasedAgent',
    'EnhancedRuleBasedAgent',
    'AdvancedMomentumAgent',
    'AdvancedMeanReversionAgent',
    'RandomAgent',
    'BiasedRandomAgent',
    'ContextualRandomAgent',
    
    # Momentum strategies
    'MACDCrossoverAgent',
    'RSIAgent',
    'DualMomentumAgent',
    'BreakoutAgent',
    
    # Benchmark agents
    'BuyAndHoldAgent',
    'EqualWeightAgent',
    'MarketCapWeightedAgent',
    'SectorRotationAgent',
    'RiskParityAgent',
    
    # Technical indicators
    'TechnicalIndicators',
    'AdvancedTechnicalIndicators',
    'IndicatorSignals',
    'IndicatorCombinations',
    'PerformanceOptimizer',
    
    # Performance benchmarking
    'PerformanceBenchmark'
]