"""
Universal Observation Enrichment System

This module provides sophisticated observation enrichment that enables every agent
to receive context from all preceding agents and upstream MARL systems through
multi-layered context enrichment and dynamic attention mechanisms.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import structlog
from enum import Enum

from ..events import EventBus, EventType, Event
from .observation_cache import ObservationCache
from .dynamic_attention_mechanism import DynamicAttentionMechanism
from .correlation_tracker import CorrelationTracker

logger = structlog.get_logger(__name__)


class ContextLayer(Enum):
    """Different layers of context enrichment"""
    INTRA_MARL = "intra_marl"          # Within same MARL system
    INTER_MARL = "inter_marl"          # Between different MARL systems
    TEMPORAL = "temporal"              # Historical context
    ATTENTION_WEIGHTED = "attention"    # Attention-weighted context


@dataclass
class AgentObservation:
    """Structured agent observation with metadata"""
    agent_id: str
    marl_system: str
    observation: Dict[str, Any]
    timestamp: datetime
    confidence: float
    context_layer: ContextLayer
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedObservation:
    """Enriched observation with context from all layers"""
    base_observation: Dict[str, Any]
    intra_marl_context: Dict[str, Any]
    inter_marl_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    attention_weights: Dict[str, float]
    superposition_relevance: Dict[str, float]
    enrichment_time_ms: float
    total_context_sources: int


class UniversalObservationEnricher:
    """
    Universal Observation Enrichment System
    
    Provides sophisticated multi-layered context enrichment that enables every agent
    to receive optimal context from all preceding agents and upstream MARL systems.
    """
    
    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        """
        Initialize the Universal Observation Enricher
        
        Args:
            event_bus: System event bus for communication
            config: Configuration dictionary with enrichment settings
        """
        self.event_bus = event_bus
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Core components
        self.cache = ObservationCache(config.get("cache", {}))
        self.attention_mechanism = DynamicAttentionMechanism(config.get("attention", {}))
        self.correlation_tracker = CorrelationTracker(config.get("correlation", {}))
        
        # Agent and MARL system tracking
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.marl_system_registry: Dict[str, Dict[str, Any]] = {}
        self.observation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance monitoring
        self.performance_metrics = {
            "enrichment_times": deque(maxlen=1000),
            "cache_hit_rate": 0.0,
            "attention_computation_time": deque(maxlen=1000),
            "correlation_computation_time": deque(maxlen=1000)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Configuration parameters
        self.max_enrichment_time_ms = config.get("max_enrichment_time_ms", 5.0)
        self.enable_temporal_context = config.get("enable_temporal_context", True)
        self.temporal_window_size = config.get("temporal_window_size", 20)
        self.enable_correlation_tracking = config.get("enable_correlation_tracking", True)
        
        # Initialize event subscriptions
        self._setup_event_subscriptions()
        
        self.logger.info(
            "Universal Observation Enricher initialized",
            max_enrichment_time_ms=self.max_enrichment_time_ms,
            temporal_window_size=self.temporal_window_size
        )
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for observation tracking"""
        
        # Subscribe to agent decision events
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._on_strategic_decision)
        self.event_bus.subscribe(EventType.TACTICAL_DECISION, self._on_tactical_decision)
        self.event_bus.subscribe(EventType.SYNERGY_DETECTED, self._on_synergy_detected)
        
        # Subscribe to system events
        self.event_bus.subscribe(EventType.INDICATORS_READY, self._on_indicators_ready)
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._on_risk_update)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._on_var_update)
        
        self.logger.info("Event subscriptions configured")
    
    def register_agent(self, agent_id: str, marl_system: str, 
                      agent_metadata: Dict[str, Any]) -> None:
        """
        Register an agent in the enrichment system
        
        Args:
            agent_id: Unique agent identifier
            marl_system: MARL system the agent belongs to
            agent_metadata: Agent metadata and capabilities
        """
        with self.lock:
            self.agent_registry[agent_id] = {
                "marl_system": marl_system,
                "metadata": agent_metadata,
                "registration_time": datetime.now(),
                "last_observation": None
            }
            
            # Initialize observation history
            self.observation_history[agent_id] = deque(maxlen=self.temporal_window_size)
            
            self.logger.info(
                "Agent registered",
                agent_id=agent_id,
                marl_system=marl_system
            )
    
    def register_marl_system(self, system_id: str, system_metadata: Dict[str, Any]) -> None:
        """
        Register a MARL system in the enrichment system
        
        Args:
            system_id: Unique MARL system identifier
            system_metadata: System metadata and capabilities
        """
        with self.lock:
            self.marl_system_registry[system_id] = {
                "metadata": system_metadata,
                "registration_time": datetime.now(),
                "agents": [],
                "last_activity": datetime.now()
            }
            
            self.logger.info(
                "MARL system registered",
                system_id=system_id
            )
    
    def enrich_observation(self, agent_id: str, base_observation: Dict[str, Any], 
                          context_requirements: Optional[Dict[str, Any]] = None) -> EnrichedObservation:
        """
        Enrich an agent's observation with multi-layered context
        
        Args:
            agent_id: Agent requesting enrichment
            base_observation: Base observation to enrich
            context_requirements: Specific context requirements
            
        Returns:
            EnrichedObservation with all context layers
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Validate agent registration
                if agent_id not in self.agent_registry:
                    raise ValueError(f"Agent {agent_id} not registered")
                
                agent_info = self.agent_registry[agent_id]
                marl_system = agent_info["marl_system"]
                
                # Check cache first
                cache_key = self._generate_cache_key(agent_id, base_observation)
                cached_result = self.cache.get(cache_key)
                
                if cached_result:
                    self.performance_metrics["cache_hit_rate"] += 1
                    return cached_result
                
                # Build enriched observation
                enriched_obs = self._build_enriched_observation(
                    agent_id, marl_system, base_observation, context_requirements
                )
                
                # Cache the result
                self.cache.put(cache_key, enriched_obs)
                
                # Update performance metrics
                enrichment_time_ms = (time.time() - start_time) * 1000
                self.performance_metrics["enrichment_times"].append(enrichment_time_ms)
                
                # Performance warning if too slow
                if enrichment_time_ms > self.max_enrichment_time_ms:
                    self.logger.warning(
                        "Enrichment time exceeded target",
                        enrichment_time_ms=enrichment_time_ms,
                        target_ms=self.max_enrichment_time_ms,
                        agent_id=agent_id
                    )
                
                return enriched_obs
                
        except Exception as e:
            self.logger.error(
                "Observation enrichment failed",
                agent_id=agent_id,
                error=str(e)
            )
            # Return minimal enriched observation on error
            return EnrichedObservation(
                base_observation=base_observation,
                intra_marl_context={},
                inter_marl_context={},
                temporal_context={},
                attention_weights={},
                superposition_relevance={},
                enrichment_time_ms=(time.time() - start_time) * 1000,
                total_context_sources=0
            )
    
    def _build_enriched_observation(self, agent_id: str, marl_system: str,
                                   base_observation: Dict[str, Any],
                                   context_requirements: Optional[Dict[str, Any]]) -> EnrichedObservation:
        """Build enriched observation with all context layers"""
        
        # 1. Intra-MARL context (agents within same MARL system)
        intra_marl_context = self._build_intra_marl_context(agent_id, marl_system)
        
        # 2. Inter-MARL context (information from other MARL systems)
        inter_marl_context = self._build_inter_marl_context(agent_id, marl_system)
        
        # 3. Temporal context (historical information)
        temporal_context = self._build_temporal_context(agent_id) if self.enable_temporal_context else {}
        
        # 4. Compute attention weights for relevance
        attention_weights = self._compute_attention_weights(
            agent_id, base_observation, intra_marl_context, 
            inter_marl_context, temporal_context
        )
        
        # 5. Compute superposition relevance
        superposition_relevance = self._compute_superposition_relevance(
            agent_id, base_observation, attention_weights
        )
        
        # 6. Update correlation tracking
        if self.enable_correlation_tracking:
            self._update_correlation_tracking(agent_id, base_observation, intra_marl_context)
        
        # 7. Store observation in history
        self._store_observation_history(agent_id, base_observation)
        
        return EnrichedObservation(
            base_observation=base_observation,
            intra_marl_context=intra_marl_context,
            inter_marl_context=inter_marl_context,
            temporal_context=temporal_context,
            attention_weights=attention_weights,
            superposition_relevance=superposition_relevance,
            enrichment_time_ms=0.0,  # Will be set by caller
            total_context_sources=len(intra_marl_context) + len(inter_marl_context) + len(temporal_context)
        )
    
    def _build_intra_marl_context(self, agent_id: str, marl_system: str) -> Dict[str, Any]:
        """Build context from agents within the same MARL system"""
        context = {}
        
        # Find other agents in the same MARL system
        for other_agent_id, agent_info in self.agent_registry.items():
            if other_agent_id != agent_id and agent_info["marl_system"] == marl_system:
                last_obs = agent_info.get("last_observation")
                if last_obs:
                    context[other_agent_id] = {
                        "observation": last_obs,
                        "timestamp": agent_info.get("last_observation_time", datetime.now()),
                        "agent_type": agent_info["metadata"].get("agent_type", "unknown")
                    }
        
        return context
    
    def _build_inter_marl_context(self, agent_id: str, current_marl_system: str) -> Dict[str, Any]:
        """Build context from other MARL systems"""
        context = {}
        
        # Aggregate information from other MARL systems
        for system_id, system_info in self.marl_system_registry.items():
            if system_id != current_marl_system:
                # Get representative agents from this system
                system_agents = [
                    aid for aid, ainfo in self.agent_registry.items()
                    if ainfo["marl_system"] == system_id
                ]
                
                if system_agents:
                    # Get most recent and relevant observations
                    recent_observations = []
                    for sys_agent_id in system_agents[:3]:  # Limit to 3 most relevant
                        agent_info = self.agent_registry.get(sys_agent_id)
                        if agent_info and agent_info.get("last_observation"):
                            recent_observations.append({
                                "agent_id": sys_agent_id,
                                "observation": agent_info["last_observation"],
                                "timestamp": agent_info.get("last_observation_time")
                            })
                    
                    if recent_observations:
                        context[system_id] = {
                            "system_metadata": system_info["metadata"],
                            "recent_observations": recent_observations,
                            "last_activity": system_info["last_activity"]
                        }
        
        return context
    
    def _build_temporal_context(self, agent_id: str) -> Dict[str, Any]:
        """Build temporal context from historical observations"""
        context = {}
        
        history = self.observation_history.get(agent_id, deque())
        if len(history) > 0:
            # Recent history (last 5 observations)
            recent_history = list(history)[-5:]
            
            # Trend analysis
            if len(recent_history) >= 2:
                context["recent_trend"] = self._analyze_observation_trend(recent_history)
            
            # Pattern recognition
            if len(history) >= 10:
                context["patterns"] = self._detect_observation_patterns(list(history))
            
            # Statistical summaries
            context["statistics"] = self._compute_historical_statistics(list(history))
        
        return context
    
    def _compute_attention_weights(self, agent_id: str, base_observation: Dict[str, Any],
                                  intra_marl_context: Dict[str, Any],
                                  inter_marl_context: Dict[str, Any],
                                  temporal_context: Dict[str, Any]) -> Dict[str, float]:
        """Compute attention weights for different context sources"""
        
        start_time = time.time()
        
        # Use dynamic attention mechanism
        attention_weights = self.attention_mechanism.compute_attention_weights(
            agent_id=agent_id,
            base_observation=base_observation,
            context_sources={
                "intra_marl": intra_marl_context,
                "inter_marl": inter_marl_context,
                "temporal": temporal_context
            }
        )
        
        # Record computation time
        computation_time = (time.time() - start_time) * 1000
        self.performance_metrics["attention_computation_time"].append(computation_time)
        
        return attention_weights
    
    def _compute_superposition_relevance(self, agent_id: str, base_observation: Dict[str, Any],
                                        attention_weights: Dict[str, float]) -> Dict[str, float]:
        """Compute relevance scores for superposition states"""
        
        # Use correlation tracker to compute relevance
        if self.enable_correlation_tracking:
            return self.correlation_tracker.compute_superposition_relevance(
                agent_id, base_observation, attention_weights
            )
        
        return {}
    
    def _update_correlation_tracking(self, agent_id: str, base_observation: Dict[str, Any],
                                    intra_marl_context: Dict[str, Any]) -> None:
        """Update correlation tracking with new observation"""
        
        start_time = time.time()
        
        self.correlation_tracker.update_correlations(
            agent_id, base_observation, intra_marl_context
        )
        
        # Record computation time
        computation_time = (time.time() - start_time) * 1000
        self.performance_metrics["correlation_computation_time"].append(computation_time)
    
    def _store_observation_history(self, agent_id: str, observation: Dict[str, Any]) -> None:
        """Store observation in agent's history"""
        
        self.observation_history[agent_id].append({
            "observation": observation,
            "timestamp": datetime.now()
        })
        
        # Update agent registry
        self.agent_registry[agent_id]["last_observation"] = observation
        self.agent_registry[agent_id]["last_observation_time"] = datetime.now()
    
    def _analyze_observation_trend(self, recent_observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in recent observations"""
        
        if len(recent_observations) < 2:
            return {}
        
        # Simple trend analysis - can be extended with more sophisticated methods
        trend_analysis = {
            "direction": "stable",
            "magnitude": 0.0,
            "confidence": 0.0
        }
        
        # Example: analyze numerical values if present
        numeric_keys = []
        for obs in recent_observations:
            for key, value in obs["observation"].items():
                if isinstance(value, (int, float)) and key not in numeric_keys:
                    numeric_keys.append(key)
        
        # Compute trends for numeric values
        if numeric_keys:
            key_trends = {}
            for key in numeric_keys:
                values = [obs["observation"].get(key, 0) for obs in recent_observations]
                if len(values) >= 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    key_trends[key] = trend
            
            trend_analysis["key_trends"] = key_trends
        
        return trend_analysis
    
    def _detect_observation_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in observation history"""
        
        patterns = {
            "cyclical": False,
            "seasonal": False,
            "recurring_events": []
        }
        
        # Simple pattern detection - can be extended with ML models
        # For now, just detect basic recurring patterns
        
        return patterns
    
    def _compute_historical_statistics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistical summaries of historical observations"""
        
        if not history:
            return {}
        
        stats = {
            "total_observations": len(history),
            "time_span_minutes": 0.0,
            "observation_frequency": 0.0
        }
        
        # Compute time span
        if len(history) >= 2:
            first_time = history[0]["timestamp"]
            last_time = history[-1]["timestamp"]
            time_span = (last_time - first_time).total_seconds() / 60.0
            stats["time_span_minutes"] = time_span
            stats["observation_frequency"] = len(history) / max(time_span, 1.0)
        
        return stats
    
    def _generate_cache_key(self, agent_id: str, observation: Dict[str, Any]) -> str:
        """Generate cache key for observation"""
        
        # Simple hash-based key generation
        obs_str = str(sorted(observation.items()))
        return f"{agent_id}_{hash(obs_str)}"
    
    def _on_strategic_decision(self, event: Event) -> None:
        """Handle strategic decision events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "unknown")
        
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["last_observation"] = payload
            self.agent_registry[agent_id]["last_observation_time"] = event.timestamp
    
    def _on_tactical_decision(self, event: Event) -> None:
        """Handle tactical decision events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "unknown")
        
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["last_observation"] = payload
            self.agent_registry[agent_id]["last_observation_time"] = event.timestamp
    
    def _on_synergy_detected(self, event: Event) -> None:
        """Handle synergy detection events"""
        
        # Update system-wide context
        payload = event.payload
        
        # Notify all agents of synergy event
        for agent_id in self.agent_registry:
            if agent_id not in self.observation_history:
                self.observation_history[agent_id] = deque(maxlen=self.temporal_window_size)
            
            self.observation_history[agent_id].append({
                "observation": {"synergy_event": payload},
                "timestamp": event.timestamp
            })
    
    def _on_indicators_ready(self, event: Event) -> None:
        """Handle indicator ready events"""
        
        # Update global context with new indicators
        payload = event.payload
        
        # Store as system-wide context
        self.marl_system_registry.setdefault("indicators", {})["last_update"] = {
            "indicators": payload,
            "timestamp": event.timestamp
        }
    
    def _on_risk_update(self, event: Event) -> None:
        """Handle risk update events"""
        
        payload = event.payload
        
        # Update risk context
        self.marl_system_registry.setdefault("risk_management", {})["last_update"] = {
            "risk_metrics": payload,
            "timestamp": event.timestamp
        }
    
    def _on_var_update(self, event: Event) -> None:
        """Handle VaR update events"""
        
        payload = event.payload
        
        # Update VaR context
        self.marl_system_registry.setdefault("risk_management", {}).setdefault("var_context", {})["last_update"] = {
            "var_metrics": payload,
            "timestamp": event.timestamp
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        with self.lock:
            metrics = {
                "registered_agents": len(self.agent_registry),
                "registered_marl_systems": len(self.marl_system_registry),
                "cache_performance": self.cache.get_performance_metrics(),
                "attention_performance": self.attention_mechanism.get_performance_metrics(),
                "correlation_performance": self.correlation_tracker.get_performance_metrics()
            }
            
            # Compute average enrichment time
            if self.performance_metrics["enrichment_times"]:
                metrics["avg_enrichment_time_ms"] = np.mean(self.performance_metrics["enrichment_times"])
                metrics["max_enrichment_time_ms"] = np.max(self.performance_metrics["enrichment_times"])
                metrics["enrichment_time_p95_ms"] = np.percentile(self.performance_metrics["enrichment_times"], 95)
            
            # Compute average attention computation time
            if self.performance_metrics["attention_computation_time"]:
                metrics["avg_attention_time_ms"] = np.mean(self.performance_metrics["attention_computation_time"])
            
            # Compute average correlation computation time
            if self.performance_metrics["correlation_computation_time"]:
                metrics["avg_correlation_time_ms"] = np.mean(self.performance_metrics["correlation_computation_time"])
            
            return metrics
    
    def get_agent_context_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get context summary for a specific agent"""
        
        with self.lock:
            if agent_id not in self.agent_registry:
                return {}
            
            agent_info = self.agent_registry[agent_id]
            history = self.observation_history.get(agent_id, deque())
            
            return {
                "agent_id": agent_id,
                "marl_system": agent_info["marl_system"],
                "registration_time": agent_info["registration_time"],
                "last_observation_time": agent_info.get("last_observation_time"),
                "observation_history_size": len(history),
                "available_context_sources": {
                    "intra_marl": len([a for a in self.agent_registry.values() 
                                     if a["marl_system"] == agent_info["marl_system"]]) - 1,
                    "inter_marl": len([s for s in self.marl_system_registry.keys() 
                                     if s != agent_info["marl_system"]]),
                    "temporal": len(history)
                }
            }