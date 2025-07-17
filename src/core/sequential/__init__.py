"""
Sequential Agent Architecture Components

This module provides the Universal Observation Enrichment System
that enables every agent to receive context from all preceding agents
and upstream MARL systems through sophisticated attention mechanisms.

Key Features:
- Multi-layered context enrichment (intra-MARL, inter-MARL, temporal, attention-weighted)
- Dynamic attention mechanisms for relevance weighting
- High-performance caching with <5ms enrichment time
- Correlation tracking between all agents and MARL systems
- Scalable architecture for large numbers of agents

Usage:
    from src.core.sequential import UniversalObservationEnricher
    
    # Initialize with event bus and configuration
    enricher = UniversalObservationEnricher(event_bus, config)
    
    # Register agents and MARL systems
    enricher.register_agent("agent_1", "strategic_marl", {"agent_type": "entry_agent"})
    enricher.register_marl_system("strategic_marl", {"system_type": "strategic"})
    
    # Enrich observations
    enriched_obs = enricher.enrich_observation("agent_1", base_observation)
"""

from .universal_observation_enricher import (
    UniversalObservationEnricher,
    EnrichedObservation,
    AgentObservation,
    ContextLayer
)
from .dynamic_attention_mechanism import (
    DynamicAttentionMechanism,
    AttentionType,
    AttentionWeight
)
from .observation_cache import (
    ObservationCache,
    CacheEntry,
    CacheEvictionPolicy
)
from .correlation_tracker import (
    CorrelationTracker,
    CorrelationType,
    CorrelationMetric,
    SuperpositionState
)

__all__ = [
    # Core classes
    "UniversalObservationEnricher",
    "DynamicAttentionMechanism", 
    "ObservationCache",
    "CorrelationTracker",
    
    # Data structures
    "EnrichedObservation",
    "AgentObservation",
    "AttentionWeight",
    "CacheEntry",
    "CorrelationMetric",
    "SuperpositionState",
    
    # Enums
    "ContextLayer",
    "AttentionType",
    "CacheEvictionPolicy",
    "CorrelationType"
]

# Version information
__version__ = "1.0.0"
__author__ = "GrandModel Agent 3"
__description__ = "Universal Observation Enrichment System for Sequential Agent Architecture"