"""
Risk Management Multi-Agent Reinforcement Learning (MARL) System

This module provides the MARL framework for coordinated risk management
across 4 specialized risk agents as specified in the PRD.

Components:
- CentralizedCritic: Global portfolio risk evaluation V(s)
- RiskEnvironment: Multi-agent simulation environment
- AgentCoordinator: Agent consensus and emergency protocols
- StateProcessor: 10-dimensional risk vector processing
"""

from .centralized_critic import CentralizedCritic
from .risk_environment import RiskEnvironment
from .agent_coordinator import AgentCoordinator

__all__ = [
    'CentralizedCritic',
    'RiskEnvironment', 
    'AgentCoordinator'
]