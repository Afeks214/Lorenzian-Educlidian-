"""
Intelligence Module for Advanced Market Analysis and Contextual Decision Making

This module provides sophisticated market intelligence capabilities including:
- Advanced regime detection and classification
- Contextual reward systems that adapt to market conditions
- Market microstructure analysis and pattern recognition
- Intelligent decision support systems

Components:
- RegimeDetector: Advanced market regime classification system
- RegimeAwareRewardFunction: Context-aware reward adaptation
- Market intelligence utilities and analysis tools

Author: Agent Gamma - The Contextual Judge
"""

from .regime_detector import (
    RegimeDetector, 
    MarketRegime, 
    RegimeAnalysis, 
    create_regime_detector
)

__all__ = [
    'RegimeDetector',
    'MarketRegime', 
    'RegimeAnalysis',
    'create_regime_detector'
]

__version__ = "1.0.0"
__author__ = "Agent Gamma - The Contextual Judge"