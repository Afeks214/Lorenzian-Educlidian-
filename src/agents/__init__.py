"""
Strategic Agents Module

This module contains all strategic agents for the 30-minute MARL trading system.
Each agent specializes in specific aspects of market analysis and strategic decision making.
"""

# Import the new strategic MARL components
try:
    from .strategic_marl_component import StrategicMARLComponent, StrategicDecision
    strategic_marl_available = True
except ImportError:
    strategic_marl_available = False

try:
    from .strategic_agent_base import (
        StrategicAgentBase, 
        MLMIStrategicAgent, 
        NWRQKStrategicAgent, 
        RegimeDetectionAgent,
        AgentPrediction
    )
    strategic_agents_available = True
except ImportError:
    strategic_agents_available = False

try:
    from .mathematical_validator import MathematicalValidator, ValidationResult
    validator_available = True
except ImportError:
    validator_available = False

# Legacy imports (if they exist)
try:
    from .base_strategic_agent import BaseStrategicAgent, StrategicAction, MarketRegime
    from .nwrqk_strategic_agent import SupportResistanceLevel
    from .regime_detection_agent import RegimeTransition
    legacy_available = True
except ImportError:
    legacy_available = False

__all__ = []

# Add available components to __all__
if strategic_marl_available:
    __all__.extend(['StrategicMARLComponent', 'StrategicDecision'])

if strategic_agents_available:
    __all__.extend(['StrategicAgentBase', 'MLMIStrategicAgent', 'NWRQKStrategicAgent', 'RegimeDetectionAgent', 'AgentPrediction'])

if validator_available:
    __all__.extend(['MathematicalValidator', 'ValidationResult'])

# Add legacy exports if available
if legacy_available:
    __all__.extend([
        'BaseStrategicAgent',
        'StrategicAction',
        'MarketRegime',
        'SupportResistanceLevel',
        'RegimeTransition'
    ])