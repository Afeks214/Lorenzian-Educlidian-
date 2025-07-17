"""
Correlation Tracker for Inter-Agent and Inter-MARL Analysis

This module provides comprehensive correlation tracking between all agents and MARL systems
to enable intelligent context enrichment and superposition relevance computation.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import structlog
from enum import Enum
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine

logger = structlog.get_logger(__name__)


class CorrelationType(Enum):
    """Different types of correlations to track"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    COSINE_SIMILARITY = "cosine"
    MUTUAL_INFORMATION = "mutual_info"
    DYNAMIC_TIME_WARPING = "dtw"


@dataclass
class CorrelationMetric:
    """Correlation metric with metadata"""
    source_id: str
    target_id: str
    correlation_type: CorrelationType
    value: float
    confidence: float
    timestamp: datetime
    window_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuperpositionState:
    """Superposition state for correlation analysis"""
    agent_id: str
    state_vector: np.ndarray
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationTracker:
    """
    Comprehensive Correlation Tracker
    
    Tracks correlations between all agents and MARL systems to enable
    intelligent context enrichment and superposition relevance computation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Correlation Tracker
        
        Args:
            config: Configuration dictionary with correlation settings
        """
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Configuration parameters
        self.correlation_window_size = config.get("correlation_window_size", 100)
        self.min_correlation_samples = config.get("min_correlation_samples", 10)
        self.correlation_update_interval = config.get("correlation_update_interval", 5.0)
        self.enable_adaptive_windows = config.get("enable_adaptive_windows", True)
        self.correlation_threshold = config.get("correlation_threshold", 0.1)
        
        # Correlation types to compute
        self.correlation_types = [
            CorrelationType.PEARSON,
            CorrelationType.SPEARMAN,
            CorrelationType.COSINE_SIMILARITY
        ]
        
        # Data storage
        self.agent_observations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.correlation_window_size)
        )
        self.marl_system_states: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.correlation_window_size)
        )
        
        # Correlation matrices
        self.correlation_matrices: Dict[CorrelationType, Dict[str, Dict[str, float]]] = {
            corr_type: defaultdict(lambda: defaultdict(float))
            for corr_type in self.correlation_types
        }
        
        # Superposition tracking
        self.superposition_states: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self.superposition_correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance monitoring
        self.performance_metrics = {
            "correlation_computations": 0,
            "computation_times": deque(maxlen=1000),
            "matrix_updates": 0,
            "superposition_updates": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background update thread
        self.update_thread = None
        self.shutdown_event = threading.Event()
        
        # Correlation cache for performance
        self.correlation_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = timedelta(seconds=30)
        
        # Initialize correlation tracking
        self._initialize_correlation_tracking()
        
        self.logger.info(
            "Correlation Tracker initialized",
            correlation_window_size=self.correlation_window_size,
            correlation_types=[ct.value for ct in self.correlation_types],
            enable_adaptive_windows=self.enable_adaptive_windows
        )
    
    def _initialize_correlation_tracking(self) -> None:
        """Initialize correlation tracking components"""
        
        # Start background correlation update thread
        if self.correlation_update_interval > 0:
            self.update_thread = threading.Thread(
                target=self._background_correlation_update,
                daemon=True
            )
            self.update_thread.start()
    
    def update_correlations(self, agent_id: str, observation: Dict[str, Any],
                           context: Dict[str, Any]) -> None:
        """
        Update correlations with new observation
        
        Args:
            agent_id: Agent identifier
            observation: New observation
            context: Context information
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Convert observation to numerical vector
                obs_vector = self._extract_numerical_vector(observation)
                
                if len(obs_vector) == 0:
                    return
                
                # Store observation
                self.agent_observations[agent_id].append({
                    "vector": obs_vector,
                    "timestamp": datetime.now(),
                    "raw_observation": observation
                })
                
                # Update superposition state
                self._update_superposition_state(agent_id, obs_vector, observation)
                
                # Update correlation matrices (if we have enough samples)
                if len(self.agent_observations[agent_id]) >= self.min_correlation_samples:
                    self._update_correlation_matrices(agent_id)
                
                # Update performance metrics
                computation_time = (time.time() - start_time) * 1000
                self.performance_metrics["correlation_computations"] += 1
                self.performance_metrics["computation_times"].append(computation_time)
                
        except Exception as e:
            self.logger.error(
                "Correlation update failed",
                agent_id=agent_id,
                error=str(e)
            )
    
    def compute_superposition_relevance(self, agent_id: str, base_observation: Dict[str, Any],
                                       attention_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Compute superposition relevance scores
        
        Args:
            agent_id: Agent identifier
            base_observation: Base observation
            attention_weights: Attention weights for context sources
            
        Returns:
            Dictionary of superposition relevance scores
        """
        start_time = time.time()
        
        try:
            with self.lock:
                relevance_scores = {}
                
                # Extract base vector
                base_vector = self._extract_numerical_vector(base_observation)
                
                if len(base_vector) == 0:
                    return {}
                
                # Compute relevance for each superposition state
                for state_id, states in self.superposition_states.items():
                    if state_id != agent_id and len(states) > 0:
                        # Get most recent state
                        recent_state = states[-1]
                        
                        # Compute correlation with base observation
                        correlation = self._compute_vector_correlation(
                            base_vector, recent_state.state_vector
                        )
                        
                        # Weight by attention if available
                        attention_weight = attention_weights.get(state_id, 1.0)
                        
                        # Compute final relevance score
                        relevance = correlation * attention_weight * recent_state.confidence
                        relevance_scores[state_id] = max(0.0, min(1.0, relevance))
                
                # Update performance metrics
                computation_time = (time.time() - start_time) * 1000
                self.performance_metrics["computation_times"].append(computation_time)
                
                return relevance_scores
                
        except Exception as e:
            self.logger.error(
                "Superposition relevance computation failed",
                agent_id=agent_id,
                error=str(e)
            )
            return {}
    
    def get_correlation_matrix(self, correlation_type: CorrelationType = CorrelationType.PEARSON) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix for specified correlation type
        
        Args:
            correlation_type: Type of correlation to retrieve
            
        Returns:
            Correlation matrix
        """
        with self.lock:
            return dict(self.correlation_matrices[correlation_type])
    
    def get_agent_correlations(self, agent_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top correlations for a specific agent
        
        Args:
            agent_id: Agent identifier
            top_k: Number of top correlations to return
            
        Returns:
            List of (other_agent_id, correlation_value) tuples
        """
        with self.lock:
            correlations = []
            
            # Get correlations from all types
            for corr_type in self.correlation_types:
                agent_correlations = self.correlation_matrices[corr_type].get(agent_id, {})
                
                for other_agent_id, correlation in agent_correlations.items():
                    if other_agent_id != agent_id:
                        correlations.append((other_agent_id, correlation))
            
            # Sort by correlation value and return top K
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            return correlations[:top_k]
    
    def _extract_numerical_vector(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract numerical vector from observation"""
        
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
            elif isinstance(obj, bool):
                features.append(float(obj))
            elif isinstance(obj, str):
                # Simple string hashing for categorical features
                features.append(float(hash(obj) % 1000) / 1000.0)
        
        extract_recursively(observation)
        return np.array(features) if features else np.array([])
    
    def _update_superposition_state(self, agent_id: str, obs_vector: np.ndarray,
                                   observation: Dict[str, Any]) -> None:
        """Update superposition state for agent"""
        
        # Compute confidence based on vector stability
        confidence = self._compute_state_confidence(agent_id, obs_vector)
        
        # Create superposition state
        state = SuperpositionState(
            agent_id=agent_id,
            state_vector=obs_vector,
            timestamp=datetime.now(),
            confidence=confidence,
            metadata={"observation_keys": list(observation.keys())}
        )
        
        # Store superposition state
        self.superposition_states[agent_id].append(state)
        
        self.performance_metrics["superposition_updates"] += 1
    
    def _compute_state_confidence(self, agent_id: str, obs_vector: np.ndarray) -> float:
        """Compute confidence in superposition state"""
        
        # Get recent states
        recent_states = list(self.superposition_states[agent_id])[-5:]
        
        if len(recent_states) < 2:
            return 0.8  # Default confidence
        
        # Compute stability as inverse of variance
        recent_vectors = [state.state_vector for state in recent_states]
        
        if len(recent_vectors) >= 2:
            # Compute vector similarities
            similarities = []
            for i in range(len(recent_vectors) - 1):
                similarity = self._compute_vector_correlation(
                    recent_vectors[i], recent_vectors[i + 1]
                )
                similarities.append(similarity)
            
            # Confidence is average similarity
            confidence = np.mean(similarities) if similarities else 0.8
            return max(0.1, min(1.0, confidence))
        
        return 0.8
    
    def _update_correlation_matrices(self, agent_id: str) -> None:
        """Update correlation matrices for agent"""
        
        # Get this agent's observations
        agent_obs = list(self.agent_observations[agent_id])
        
        if len(agent_obs) < self.min_correlation_samples:
            return
        
        # Extract vectors
        agent_vectors = [obs["vector"] for obs in agent_obs]
        
        # Compute correlations with all other agents
        for other_agent_id, other_obs in self.agent_observations.items():
            if other_agent_id == agent_id or len(other_obs) < self.min_correlation_samples:
                continue
            
            # Check cache first
            cache_key = f"{agent_id}_{other_agent_id}"
            if cache_key in self.correlation_cache:
                cached_corr, cached_time = self.correlation_cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    self.performance_metrics["cache_hits"] += 1
                    continue
            
            self.performance_metrics["cache_misses"] += 1
            
            # Get other agent's vectors
            other_vectors = [obs["vector"] for obs in other_obs]
            
            # Align vectors (use minimum length)
            min_len = min(len(agent_vectors), len(other_vectors))
            agent_aligned = agent_vectors[-min_len:]
            other_aligned = other_vectors[-min_len:]
            
            # Compute correlations for each type
            for corr_type in self.correlation_types:
                correlation = self._compute_correlation(
                    agent_aligned, other_aligned, corr_type
                )
                
                if not np.isnan(correlation):
                    self.correlation_matrices[corr_type][agent_id][other_agent_id] = correlation
                    self.correlation_matrices[corr_type][other_agent_id][agent_id] = correlation
                    
                    # Cache result
                    self.correlation_cache[cache_key] = (correlation, datetime.now())
        
        self.performance_metrics["matrix_updates"] += 1
    
    def _compute_correlation(self, vectors1: List[np.ndarray], vectors2: List[np.ndarray],
                            corr_type: CorrelationType) -> float:
        """Compute correlation between two vector sequences"""
        
        try:
            if len(vectors1) != len(vectors2) or len(vectors1) < 2:
                return 0.0
            
            # Flatten vectors into sequences
            seq1 = np.concatenate(vectors1)
            seq2 = np.concatenate(vectors2)
            
            # Align sequences to same length
            min_len = min(len(seq1), len(seq2))
            seq1 = seq1[:min_len]
            seq2 = seq2[:min_len]
            
            if len(seq1) == 0 or len(seq2) == 0:
                return 0.0
            
            # Compute correlation based on type
            if corr_type == CorrelationType.PEARSON:
                if np.std(seq1) == 0 or np.std(seq2) == 0:
                    return 0.0
                corr, _ = pearsonr(seq1, seq2)
                return corr if not np.isnan(corr) else 0.0
            
            elif corr_type == CorrelationType.SPEARMAN:
                if len(np.unique(seq1)) == 1 or len(np.unique(seq2)) == 1:
                    return 0.0
                corr, _ = spearmanr(seq1, seq2)
                return corr if not np.isnan(corr) else 0.0
            
            elif corr_type == CorrelationType.COSINE_SIMILARITY:
                if np.linalg.norm(seq1) == 0 or np.linalg.norm(seq2) == 0:
                    return 0.0
                similarity = 1 - cosine(seq1, seq2)
                return similarity if not np.isnan(similarity) else 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Correlation computation failed: {e}")
            return 0.0
    
    def _compute_vector_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute correlation between two vectors"""
        
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Align vectors to same length
        min_len = min(len(vec1), len(vec2))
        vec1_aligned = vec1[:min_len]
        vec2_aligned = vec2[:min_len]
        
        try:
            # Use cosine similarity for single vectors
            if np.linalg.norm(vec1_aligned) == 0 or np.linalg.norm(vec2_aligned) == 0:
                return 0.0
            
            similarity = 1 - cosine(vec1_aligned, vec2_aligned)
            return similarity if not np.isnan(similarity) else 0.0
            
        except Exception:
            return 0.0
    
    def _background_correlation_update(self) -> None:
        """Background correlation update thread"""
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for update interval
                if self.shutdown_event.wait(self.correlation_update_interval):
                    break
                
                # Update correlations for all agents
                self._update_all_correlations()
                
                # Cleanup old correlations
                self._cleanup_old_correlations()
                
            except Exception as e:
                self.logger.error(f"Background correlation update error: {e}")
    
    def _update_all_correlations(self) -> None:
        """Update correlations for all agents"""
        
        with self.lock:
            for agent_id in list(self.agent_observations.keys()):
                if len(self.agent_observations[agent_id]) >= self.min_correlation_samples:
                    self._update_correlation_matrices(agent_id)
    
    def _cleanup_old_correlations(self) -> None:
        """Clean up old correlation cache entries"""
        
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for cache_key, (_, timestamp) in self.correlation_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.correlation_cache[key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        with self.lock:
            metrics = {
                "correlation_computations": self.performance_metrics["correlation_computations"],
                "matrix_updates": self.performance_metrics["matrix_updates"],
                "superposition_updates": self.performance_metrics["superposition_updates"],
                "cache_hit_rate": self.performance_metrics["cache_hits"] / max(1, 
                    self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]),
                "tracked_agents": len(self.agent_observations),
                "superposition_states": len(self.superposition_states),
                "correlation_cache_size": len(self.correlation_cache)
            }
            
            # Average computation time
            if self.performance_metrics["computation_times"]:
                metrics["avg_computation_time_ms"] = np.mean(self.performance_metrics["computation_times"])
                metrics["max_computation_time_ms"] = np.max(self.performance_metrics["computation_times"])
            
            return metrics
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get correlation summary"""
        
        with self.lock:
            summary = {
                "correlation_matrices": {},
                "top_correlated_pairs": [],
                "correlation_statistics": {}
            }
            
            # Get correlation matrices
            for corr_type in self.correlation_types:
                matrix = self.correlation_matrices[corr_type]
                summary["correlation_matrices"][corr_type.value] = dict(matrix)
            
            # Find top correlated pairs
            all_correlations = []
            for corr_type in self.correlation_types:
                matrix = self.correlation_matrices[corr_type]
                for agent1, correlations in matrix.items():
                    for agent2, correlation in correlations.items():
                        if agent1 < agent2:  # Avoid duplicates
                            all_correlations.append((agent1, agent2, correlation, corr_type.value))
            
            # Sort by absolute correlation
            all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            summary["top_correlated_pairs"] = all_correlations[:20]
            
            # Compute statistics
            if all_correlations:
                correlations_values = [corr[2] for corr in all_correlations]
                summary["correlation_statistics"] = {
                    "mean": np.mean(correlations_values),
                    "std": np.std(correlations_values),
                    "min": np.min(correlations_values),
                    "max": np.max(correlations_values),
                    "median": np.median(correlations_values)
                }
            
            return summary
    
    def get_agent_correlation_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get correlation profile for specific agent"""
        
        with self.lock:
            profile = {
                "agent_id": agent_id,
                "observation_count": len(self.agent_observations.get(agent_id, [])),
                "superposition_states": len(self.superposition_states.get(agent_id, [])),
                "correlations": {},
                "top_correlations": []
            }
            
            # Get correlations for this agent
            for corr_type in self.correlation_types:
                matrix = self.correlation_matrices[corr_type]
                if agent_id in matrix:
                    profile["correlations"][corr_type.value] = dict(matrix[agent_id])
            
            # Get top correlations
            profile["top_correlations"] = self.get_agent_correlations(agent_id)
            
            return profile
    
    def shutdown(self) -> None:
        """Shutdown correlation tracker"""
        
        self.shutdown_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        self.logger.info("Correlation Tracker shutdown complete")