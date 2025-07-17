"""
AI Ethics and Bias Detection Framework
Agent Epsilon: Advanced XAI Implementation Specialist

This module implements comprehensive AI ethics monitoring and bias detection
for trading systems, ensuring fair and responsible AI deployment.
"""

from .bias_detector import BiasDetector, BiasType, BiasMetric, BiasResult
from .ethics_engine import EthicsEngine, EthicsViolation, EthicsRule, EthicsAssessment
from .fairness_monitor import FairnessMonitor, FairnessMetric, FairnessAlert

__all__ = [
    'BiasDetector',
    'BiasType',
    'BiasMetric', 
    'BiasResult',
    'EthicsEngine',
    'EthicsViolation',
    'EthicsRule',
    'EthicsAssessment',
    'FairnessMonitor',
    'FairnessMetric',
    'FairnessAlert'
]