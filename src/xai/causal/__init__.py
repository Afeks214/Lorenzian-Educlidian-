"""
Causal Inference Engine for Advanced XAI
Agent Epsilon: Advanced XAI Implementation Specialist

This module implements industry-first causal inference capabilities for trading explainability,
providing counterfactual analysis and causal graph construction for perfect 10/10 XAI scores.
"""

from .do_calculus_engine import DoCalculusEngine, CausalQuery, CausalGraph, CausalEvidence
from .counterfactual_engine import CounterfactualEngine, CounterfactualQuery, CounterfactualResult
from .causal_narrative_generator import CausalNarrativeGenerator, CausalStory, CausalExplanation

__all__ = [
    'DoCalculusEngine',
    'CausalQuery', 
    'CausalGraph',
    'CausalEvidence',
    'CounterfactualEngine',
    'CounterfactualQuery',
    'CounterfactualResult',
    'CausalNarrativeGenerator',
    'CausalStory',
    'CausalExplanation'
]