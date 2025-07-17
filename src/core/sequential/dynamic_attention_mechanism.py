"""
Dynamic Attention Mechanism for Sequential Observation Enrichment

This module provides sophisticated attention mechanisms that focus on the most
relevant information superpositions and context sources for each agent.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)


class AttentionType(Enum):
    """Different types of attention mechanisms"""
    CONTENT_BASED = "content_based"
    TEMPORAL = "temporal"
    AGENT_BASED = "agent_based"
    SYSTEM_BASED = "system_based"
    CORRELATION_BASED = "correlation_based"


@dataclass
class AttentionWeight:
    """Attention weight with metadata"""
    source_id: str
    weight: float
    attention_type: AttentionType
    confidence: float
    computation_time_ms: float
    metadata: Dict[str, Any]


class DynamicAttentionMechanism:
    """
    Dynamic Attention Mechanism for Context Relevance Weighting
    
    Provides sophisticated attention mechanisms that dynamically focus on the most
    relevant information superpositions from preceding agents and MARL systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Dynamic Attention Mechanism
        
        Args:
            config: Configuration dictionary with attention settings
        """
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Configuration parameters
        self.attention_temperature = config.get("attention_temperature", 1.0)
        self.min_attention_threshold = config.get("min_attention_threshold", 0.01)
        self.max_context_sources = config.get("max_context_sources", 50)
        self.enable_adaptive_temperature = config.get("enable_adaptive_temperature", True)
        self.temporal_decay_factor = config.get("temporal_decay_factor", 0.95)
        
        # Attention learning parameters
        self.learning_rate = config.get("learning_rate", 0.01)
        self.enable_attention_learning = config.get("enable_attention_learning", True)
        
        # Agent-specific attention patterns
        self.agent_attention_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_attention_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance monitoring
        self.performance_metrics = {
            "attention_computations": 0,
            "computation_times": deque(maxlen=1000),
            "adaptive_temperature_adjustments": 0,
            "attention_pattern_updates": 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize attention mechanisms
        self._initialize_attention_mechanisms()
        
        self.logger.info(
            "Dynamic Attention Mechanism initialized",
            attention_temperature=self.attention_temperature,
            max_context_sources=self.max_context_sources,
            enable_adaptive_temperature=self.enable_adaptive_temperature
        )
    
    def _initialize_attention_mechanisms(self) -> None:
        """Initialize different attention mechanisms"""
        
        # Content-based attention weights
        self.content_attention_weights = {
            "price_data": 0.3,
            "volume_data": 0.2,
            "indicator_data": 0.25,
            "risk_metrics": 0.15,
            "execution_data": 0.1
        }
        
        # Temporal attention decay
        self.temporal_attention_decay = np.exp(-np.arange(20) * 0.1)
        
        # Agent type relevance matrix
        self.agent_relevance_matrix = {
            "entry_agent": {
                "momentum_agent": 0.8,
                "fvg_agent": 0.7,
                "mlmi_agent": 0.6,
                "risk_agent": 0.9
            },
            "momentum_agent": {
                "entry_agent": 0.6,
                "fvg_agent": 0.5,
                "mlmi_agent": 0.7,
                "risk_agent": 0.8
            },
            "fvg_agent": {
                "entry_agent": 0.7,
                "momentum_agent": 0.5,
                "mlmi_agent": 0.6,
                "risk_agent": 0.8
            },
            "mlmi_agent": {
                "entry_agent": 0.6,
                "momentum_agent": 0.7,
                "fvg_agent": 0.6,
                "risk_agent": 0.9
            }
        }
        
        self.logger.info("Attention mechanisms initialized")
    
    def compute_attention_weights(self, agent_id: str, base_observation: Dict[str, Any],
                                 context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute attention weights for different context sources
        
        Args:
            agent_id: Agent requesting attention weights
            base_observation: Base observation for context
            context_sources: Available context sources
            
        Returns:
            Dictionary of attention weights for each context source
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Initialize attention weights
                attention_weights = {}
                
                # 1. Content-based attention
                content_weights = self._compute_content_based_attention(
                    agent_id, base_observation, context_sources
                )
                
                # 2. Temporal attention
                temporal_weights = self._compute_temporal_attention(
                    agent_id, context_sources
                )
                
                # 3. Agent-based attention
                agent_weights = self._compute_agent_based_attention(
                    agent_id, context_sources
                )
                
                # 4. System-based attention
                system_weights = self._compute_system_based_attention(
                    agent_id, context_sources
                )
                
                # 5. Correlation-based attention
                correlation_weights = self._compute_correlation_based_attention(
                    agent_id, base_observation, context_sources
                )
                
                # 6. Combine all attention mechanisms
                combined_weights = self._combine_attention_weights(
                    content_weights, temporal_weights, agent_weights,
                    system_weights, correlation_weights
                )
                
                # 7. Apply adaptive temperature
                if self.enable_adaptive_temperature:
                    combined_weights = self._apply_adaptive_temperature(
                        agent_id, combined_weights, base_observation
                    )
                
                # 8. Normalize and threshold
                final_weights = self._normalize_and_threshold_weights(combined_weights)
                
                # 9. Update agent attention patterns
                if self.enable_attention_learning:
                    self._update_attention_patterns(agent_id, final_weights)
                
                # Record performance metrics
                computation_time = (time.time() - start_time) * 1000
                self.performance_metrics["attention_computations"] += 1
                self.performance_metrics["computation_times"].append(computation_time)
                
                return final_weights
                
        except Exception as e:
            self.logger.error(
                "Attention weight computation failed",
                agent_id=agent_id,
                error=str(e)
            )
            return {}
    
    def _compute_content_based_attention(self, agent_id: str, base_observation: Dict[str, Any],
                                        context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute content-based attention weights"""
        
        weights = {}
        
        # Analyze content similarity between base observation and context sources
        for source_id, context_data in context_sources.items():
            if isinstance(context_data, dict):
                # Compute content similarity
                similarity_score = self._compute_content_similarity(
                    base_observation, context_data
                )
                
                # Apply content-based weights
                content_weight = similarity_score * self.content_attention_weights.get(
                    self._classify_content_type(context_data), 0.1
                )
                
                weights[source_id] = content_weight
        
        return weights
    
    def _compute_temporal_attention(self, agent_id: str, 
                                   context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute temporal attention weights based on recency"""
        
        weights = {}
        current_time = datetime.now()
        
        for source_id, context_data in context_sources.items():
            if isinstance(context_data, dict):
                # Extract timestamp information
                timestamp = self._extract_timestamp(context_data)
                
                if timestamp:
                    # Compute temporal distance
                    time_diff = (current_time - timestamp).total_seconds()
                    
                    # Apply temporal decay
                    temporal_weight = np.exp(-time_diff * self.temporal_decay_factor / 3600)  # hourly decay
                    weights[source_id] = temporal_weight
                else:
                    # Default weight for missing timestamp
                    weights[source_id] = 0.5
        
        return weights
    
    def _compute_agent_based_attention(self, agent_id: str,
                                      context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute agent-based attention weights"""
        
        weights = {}
        
        # Get agent type for relevance matrix lookup
        agent_type = self._get_agent_type(agent_id)
        
        for source_id, context_data in context_sources.items():
            if isinstance(context_data, dict):
                # Extract agent information from context
                source_agent_type = self._extract_agent_type(context_data)
                
                if source_agent_type and agent_type:
                    # Look up relevance in matrix
                    relevance = self.agent_relevance_matrix.get(agent_type, {}).get(
                        source_agent_type, 0.5
                    )
                    weights[source_id] = relevance
                else:
                    # Default weight
                    weights[source_id] = 0.5
        
        return weights
    
    def _compute_system_based_attention(self, agent_id: str,
                                       context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute system-based attention weights"""
        
        weights = {}
        
        # System priority weights
        system_priorities = {
            "strategic_marl": 0.8,
            "tactical_marl": 0.7,
            "risk_management": 0.9,
            "execution_marl": 0.6,
            "indicators": 0.5
        }
        
        for source_id, context_data in context_sources.items():
            if isinstance(context_data, dict):
                # Identify system type
                system_type = self._identify_system_type(context_data)
                
                # Apply system priority
                weights[source_id] = system_priorities.get(system_type, 0.4)
        
        return weights
    
    def _compute_correlation_based_attention(self, agent_id: str, base_observation: Dict[str, Any],
                                           context_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute correlation-based attention weights"""
        
        weights = {}
        
        # Use historical correlations to weight attention
        agent_patterns = self.agent_attention_patterns.get(agent_id, {})
        
        for source_id, context_data in context_sources.items():
            if isinstance(context_data, dict):
                # Get historical correlation with this source
                historical_correlation = agent_patterns.get(source_id, 0.5)
                
                # Compute current correlation
                current_correlation = self._compute_current_correlation(
                    base_observation, context_data
                )
                
                # Combine historical and current correlation
                combined_correlation = (0.7 * historical_correlation + 
                                      0.3 * current_correlation)
                
                weights[source_id] = combined_correlation
        
        return weights
    
    def _combine_attention_weights(self, content_weights: Dict[str, float],
                                  temporal_weights: Dict[str, float],
                                  agent_weights: Dict[str, float],
                                  system_weights: Dict[str, float],
                                  correlation_weights: Dict[str, float]) -> Dict[str, float]:
        """Combine different attention weight types"""
        
        combined_weights = {}
        
        # Get all unique source IDs
        all_sources = set()
        for weights in [content_weights, temporal_weights, agent_weights, 
                       system_weights, correlation_weights]:
            all_sources.update(weights.keys())
        
        # Combine weights for each source
        for source_id in all_sources:
            # Get weights from each mechanism (default to 0.5 if missing)
            content_w = content_weights.get(source_id, 0.5)
            temporal_w = temporal_weights.get(source_id, 0.5)
            agent_w = agent_weights.get(source_id, 0.5)
            system_w = system_weights.get(source_id, 0.5)
            correlation_w = correlation_weights.get(source_id, 0.5)
            
            # Weighted combination
            combined_weight = (
                0.25 * content_w +
                0.20 * temporal_w +
                0.20 * agent_w +
                0.15 * system_w +
                0.20 * correlation_w
            )
            
            combined_weights[source_id] = combined_weight
        
        return combined_weights
    
    def _apply_adaptive_temperature(self, agent_id: str, weights: Dict[str, float],
                                   base_observation: Dict[str, Any]) -> Dict[str, float]:
        """Apply adaptive temperature scaling"""
        
        # Compute dynamic temperature based on observation uncertainty
        observation_uncertainty = self._compute_observation_uncertainty(base_observation)
        
        # Adjust temperature based on uncertainty
        if observation_uncertainty > 0.7:
            # High uncertainty - use lower temperature (sharper attention)
            temperature = self.attention_temperature * 0.7
        elif observation_uncertainty < 0.3:
            # Low uncertainty - use higher temperature (smoother attention)
            temperature = self.attention_temperature * 1.3
        else:
            temperature = self.attention_temperature
        
        # Apply temperature scaling
        scaled_weights = {}
        for source_id, weight in weights.items():
            scaled_weights[source_id] = weight ** (1.0 / temperature)
        
        self.performance_metrics["adaptive_temperature_adjustments"] += 1
        
        return scaled_weights
    
    def _normalize_and_threshold_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1 and apply threshold"""
        
        if not weights:
            return {}
        
        # Apply minimum threshold
        thresholded_weights = {
            k: max(v, self.min_attention_threshold) 
            for k, v in weights.items()
        }
        
        # Normalize to sum to 1
        total_weight = sum(thresholded_weights.values())
        if total_weight > 0:
            normalized_weights = {
                k: v / total_weight for k, v in thresholded_weights.items()
            }
        else:
            # Equal weights if all are zero
            normalized_weights = {
                k: 1.0 / len(thresholded_weights) 
                for k in thresholded_weights.keys()
            }
        
        # Limit to max context sources
        if len(normalized_weights) > self.max_context_sources:
            # Keep top sources by weight
            sorted_weights = sorted(normalized_weights.items(), 
                                  key=lambda x: x[1], reverse=True)
            normalized_weights = dict(sorted_weights[:self.max_context_sources])
            
            # Renormalize
            total_weight = sum(normalized_weights.values())
            normalized_weights = {
                k: v / total_weight for k, v in normalized_weights.items()
            }
        
        return normalized_weights
    
    def _update_attention_patterns(self, agent_id: str, weights: Dict[str, float]) -> None:
        """Update agent attention patterns for learning"""
        
        # Update patterns with exponential moving average
        current_patterns = self.agent_attention_patterns[agent_id]
        
        for source_id, weight in weights.items():
            if source_id in current_patterns:
                # Update existing pattern
                current_patterns[source_id] = (
                    (1 - self.learning_rate) * current_patterns[source_id] +
                    self.learning_rate * weight
                )
            else:
                # Initialize new pattern
                current_patterns[source_id] = weight
        
        # Store in history
        self.agent_attention_history[agent_id].append({
            "timestamp": datetime.now(),
            "weights": weights.copy()
        })
        
        self.performance_metrics["attention_pattern_updates"] += 1
    
    def _compute_content_similarity(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> float:
        """Compute content similarity between two observations"""
        
        # Simple cosine similarity for numerical features
        try:
            # Extract numerical features
            features1 = self._extract_numerical_features(obs1)
            features2 = self._extract_numerical_features(obs2)
            
            if len(features1) == 0 or len(features2) == 0:
                return 0.5  # Default similarity
            
            # Compute cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.5
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception:
            return 0.5  # Default similarity on error
    
    def _extract_numerical_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from observation"""
        
        features = []
        
        def extract_recursively(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_recursively(value, f"{prefix}{key}_")
            elif isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    extract_recursively(value, f"{prefix}{i}_")
            elif isinstance(obj, (int, float)):
                features.append(float(obj))
        
        extract_recursively(obs)
        return np.array(features) if features else np.array([0.0])
    
    def _classify_content_type(self, context_data: Dict[str, Any]) -> str:
        """Classify content type for weighting"""
        
        # Simple heuristic classification
        if "price" in str(context_data).lower() or "open" in str(context_data).lower():
            return "price_data"
        elif "volume" in str(context_data).lower():
            return "volume_data"
        elif "indicator" in str(context_data).lower() or "signal" in str(context_data).lower():
            return "indicator_data"
        elif "risk" in str(context_data).lower() or "var" in str(context_data).lower():
            return "risk_metrics"
        elif "execution" in str(context_data).lower() or "trade" in str(context_data).lower():
            return "execution_data"
        else:
            return "unknown"
    
    def _extract_timestamp(self, context_data: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from context data"""
        
        # Look for common timestamp fields
        timestamp_fields = ["timestamp", "time", "datetime", "last_update"]
        
        for field in timestamp_fields:
            if field in context_data:
                timestamp = context_data[field]
                if isinstance(timestamp, datetime):
                    return timestamp
                elif isinstance(timestamp, str):
                    try:
                        return datetime.fromisoformat(timestamp)
                    except ValueError:
                        continue
        
        return None
    
    def _get_agent_type(self, agent_id: str) -> Optional[str]:
        """Get agent type from agent ID"""
        
        # Extract agent type from ID (simple heuristic)
        if "entry" in agent_id.lower():
            return "entry_agent"
        elif "momentum" in agent_id.lower():
            return "momentum_agent"
        elif "fvg" in agent_id.lower():
            return "fvg_agent"
        elif "mlmi" in agent_id.lower():
            return "mlmi_agent"
        elif "risk" in agent_id.lower():
            return "risk_agent"
        else:
            return None
    
    def _extract_agent_type(self, context_data: Dict[str, Any]) -> Optional[str]:
        """Extract agent type from context data"""
        
        # Look for agent type information
        if "agent_type" in context_data:
            return context_data["agent_type"]
        elif "agent_id" in context_data:
            return self._get_agent_type(context_data["agent_id"])
        else:
            return None
    
    def _identify_system_type(self, context_data: Dict[str, Any]) -> str:
        """Identify system type from context data"""
        
        context_str = str(context_data).lower()
        
        if "strategic" in context_str:
            return "strategic_marl"
        elif "tactical" in context_str:
            return "tactical_marl"
        elif "risk" in context_str:
            return "risk_management"
        elif "execution" in context_str:
            return "execution_marl"
        elif "indicator" in context_str:
            return "indicators"
        else:
            return "unknown"
    
    def _compute_current_correlation(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> float:
        """Compute current correlation between observations"""
        
        # Simple correlation based on feature similarity
        return self._compute_content_similarity(obs1, obs2)
    
    def _compute_observation_uncertainty(self, observation: Dict[str, Any]) -> float:
        """Compute uncertainty in observation"""
        
        # Simple heuristic for uncertainty
        uncertainty_indicators = ["confidence", "uncertainty", "variance", "std"]
        
        uncertainty_values = []
        for key, value in observation.items():
            if any(indicator in key.lower() for indicator in uncertainty_indicators):
                if isinstance(value, (int, float)):
                    uncertainty_values.append(value)
        
        if uncertainty_values:
            return min(1.0, max(0.0, np.mean(uncertainty_values)))
        else:
            return 0.5  # Default uncertainty
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        with self.lock:
            metrics = {
                "total_computations": self.performance_metrics["attention_computations"],
                "adaptive_temperature_adjustments": self.performance_metrics["adaptive_temperature_adjustments"],
                "attention_pattern_updates": self.performance_metrics["attention_pattern_updates"],
                "registered_agents": len(self.agent_attention_patterns)
            }
            
            # Compute average computation time
            if self.performance_metrics["computation_times"]:
                metrics["avg_computation_time_ms"] = np.mean(self.performance_metrics["computation_times"])
                metrics["max_computation_time_ms"] = np.max(self.performance_metrics["computation_times"])
            
            return metrics
    
    def get_agent_attention_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get attention summary for specific agent"""
        
        with self.lock:
            summary = {
                "agent_id": agent_id,
                "attention_patterns": self.agent_attention_patterns.get(agent_id, {}),
                "attention_history_size": len(self.agent_attention_history.get(agent_id, [])),
                "top_attention_sources": {}
            }
            
            # Get top attention sources
            patterns = self.agent_attention_patterns.get(agent_id, {})
            if patterns:
                sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
                summary["top_attention_sources"] = dict(sorted_patterns[:10])
            
            return summary