"""Human Interface Module for Expert Trading Feedback

This module provides a comprehensive system for capturing expert trader preferences
and aligning MARL execution intelligence with human trading intuition through
Reinforcement Learning from Human Feedback (RLHF).
"""

__version__ = "1.0.0"
__author__ = "GrandModel MARL Team"

from .feedback_api import FeedbackAPI, DecisionPoint, ExpertChoice
from .choice_generator import ChoiceGenerator, TradingStrategy
from .rlhf_trainer import RLHFTrainer, PreferenceDatabase

__all__ = [
    "FeedbackAPI",
    "DecisionPoint", 
    "ExpertChoice",
    "ChoiceGenerator",
    "TradingStrategy",
    "RLHFTrainer",
    "PreferenceDatabase"
]