"""
Universal Base Superposition Class for MARL Agent State Management.

This module provides the foundational superposition class that all agent-specific
superposition classes inherit from, implementing quantum-inspired state representation
and processing capabilities for financial trading agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog
from copy import deepcopy

logger = structlog.get_logger()


class SuperpositionState(Enum):
    """Superposition states for quantum-inspired processing"""
    COLLAPSED = "collapsed"
    SUPERPOSED = "superposed"
    ENTANGLED = "entangled"
    DECOHERENT = "decoherent"


@dataclass
class QuantumState:
    """Quantum state representation for superposition"""
    amplitude: complex
    phase: float
    probability: float
    coherence: float
    
    def __post_init__(self):
        """Validate quantum state properties"""
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be in [0,1], got {self.probability}")
        if not 0 <= self.coherence <= 1:
            raise ValueError(f"Coherence must be in [0,1], got {self.coherence}")


@dataclass
class SuperpositionMetadata:
    """Metadata for superposition state tracking"""
    creation_time: datetime
    last_update: datetime
    update_count: int
    coherence_decay_rate: float
    entanglement_strength: float
    measurement_history: List[Dict[str, Any]] = field(default_factory=list)


class UniversalSuperposition(ABC):
    """
    Universal base class for all agent-specific superposition implementations.
    
    This class provides the foundational quantum-inspired state representation
    capabilities that all trading agents can leverage for enhanced decision-making
    and state management.
    """
    
    def __init__(self, 
                 agent_id: str,
                 config: Dict[str, Any],
                 initial_state: Optional[np.ndarray] = None):
        """
        Initialize universal superposition
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
            initial_state: Initial state vector (optional)
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = logger.bind(agent_id=agent_id)
        
        # Core superposition properties
        self.state = SuperpositionState.COLLAPSED
        self.quantum_states: Dict[str, QuantumState] = {}
        self.classical_state: Optional[np.ndarray] = initial_state
        
        # Metadata and tracking
        self.metadata = SuperpositionMetadata(
            creation_time=datetime.now(),
            last_update=datetime.now(),
            update_count=0,
            coherence_decay_rate=config.get('coherence_decay_rate', 0.001),
            entanglement_strength=config.get('entanglement_strength', 0.5)
        )
        
        # Domain-specific components (to be implemented by subclasses)
        self.domain_features: Dict[str, Any] = {}
        self.attention_weights: Dict[str, float] = {}
        self.reasoning_chain: List[str] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            'coherence_stability': 1.0,
            'entanglement_efficiency': 0.0,
            'measurement_accuracy': 0.0,
            'computational_cost': 0.0
        }
        
        # Initialize domain-specific components
        self._initialize_domain_features()
        
    @abstractmethod
    def _initialize_domain_features(self) -> None:
        """Initialize domain-specific features - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the agent type identifier"""
        pass
    
    @abstractmethod
    def get_state_dimension(self) -> int:
        """Return the expected state vector dimension"""
        pass
    
    def create_superposition(self, 
                           states: List[np.ndarray],
                           weights: Optional[List[float]] = None) -> 'UniversalSuperposition':
        """
        Create a superposition of multiple states
        
        Args:
            states: List of state vectors to superpose
            weights: Optional weights for each state (normalized if not provided)
            
        Returns:
            Updated superposition instance
        """
        if not states:
            raise ValueError("At least one state required for superposition")
            
        if weights is None:
            weights = [1.0 / len(states)] * len(states)
        elif len(weights) != len(states):
            raise ValueError("Number of weights must match number of states")
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        weights = [w / total_weight for w in weights]
        
        # Create quantum states
        self.quantum_states.clear()
        for i, (state, weight) in enumerate(zip(states, weights)):
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            amplitude = np.sqrt(weight) * np.exp(1j * phase)
            coherence = 1.0 - (i * 0.1)  # Decreasing coherence
            
            self.quantum_states[f"state_{i}"] = QuantumState(
                amplitude=amplitude,
                phase=phase,
                probability=weight,
                coherence=max(0.0, coherence)
            )
        
        self.state = SuperpositionState.SUPERPOSED
        self._update_metadata()
        
        return self
    
    def collapse_superposition(self, 
                             measurement_operator: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Collapse superposition to classical state
        
        Args:
            measurement_operator: Optional measurement operator
            
        Returns:
            Collapsed classical state
        """
        if self.state != SuperpositionState.SUPERPOSED:
            if self.classical_state is not None:
                return self.classical_state
            else:
                raise ValueError("No superposition to collapse")
        
        # Probabilistic collapse based on quantum state probabilities
        probabilities = [qs.probability for qs in self.quantum_states.values()]
        chosen_idx = np.random.choice(len(probabilities), p=probabilities)
        
        # Extract the chosen state (simplified - would need proper state reconstruction)
        chosen_key = list(self.quantum_states.keys())[chosen_idx]
        chosen_quantum_state = self.quantum_states[chosen_key]
        
        # Apply measurement operator if provided
        if measurement_operator is not None:
            # Simplified measurement - in real implementation would apply proper operator
            collapsed_state = measurement_operator @ self.classical_state
        else:
            # Default collapse to expected value
            collapsed_state = self._compute_expected_state()
        
        self.classical_state = collapsed_state
        self.state = SuperpositionState.COLLAPSED
        
        # Record measurement
        self.metadata.measurement_history.append({
            'timestamp': datetime.now(),
            'chosen_state': chosen_key,
            'probability': chosen_quantum_state.probability,
            'coherence': chosen_quantum_state.coherence
        })
        
        self._update_metadata()
        return collapsed_state
    
    def entangle_with(self, other: 'UniversalSuperposition') -> None:
        """
        Create entanglement with another superposition
        
        Args:
            other: Another superposition instance to entangle with
        """
        if self.state != SuperpositionState.SUPERPOSED or other.state != SuperpositionState.SUPERPOSED:
            raise ValueError("Both superpositions must be in superposed state for entanglement")
        
        # Create entanglement by correlating quantum states
        entanglement_strength = min(self.metadata.entanglement_strength, 
                                   other.metadata.entanglement_strength)
        
        # Modify quantum state phases to create correlation
        for key in self.quantum_states:
            if key in other.quantum_states:
                # Correlate phases
                phase_correlation = entanglement_strength * np.pi
                self.quantum_states[key].phase += phase_correlation
                other.quantum_states[key].phase -= phase_correlation
                
                # Update amplitudes
                self.quantum_states[key].amplitude *= np.exp(1j * phase_correlation)
                other.quantum_states[key].amplitude *= np.exp(-1j * phase_correlation)
        
        self.state = SuperpositionState.ENTANGLED
        other.state = SuperpositionState.ENTANGLED
        
        self._update_metadata()
        other._update_metadata()
    
    def evolve(self, time_step: float) -> None:
        """
        Evolve superposition state over time
        
        Args:
            time_step: Time step for evolution
        """
        if self.state not in [SuperpositionState.SUPERPOSED, SuperpositionState.ENTANGLED]:
            return
        
        # Apply coherence decay
        decay_factor = np.exp(-self.metadata.coherence_decay_rate * time_step)
        
        for quantum_state in self.quantum_states.values():
            quantum_state.coherence *= decay_factor
            
            # Check for decoherence
            if quantum_state.coherence < 0.1:
                self.state = SuperpositionState.DECOHERENT
                break
        
        # Update performance metrics
        self.performance_metrics['coherence_stability'] = np.mean([
            qs.coherence for qs in self.quantum_states.values()
        ])
        
        self._update_metadata()
    
    def add_attention_weight(self, feature: str, weight: float) -> None:
        """Add attention weight for a feature"""
        self.attention_weights[feature] = np.clip(weight, 0.0, 1.0)
    
    def update_reasoning_chain(self, step: str) -> None:
        """Add step to reasoning chain"""
        self.reasoning_chain.append(f"{datetime.now()}: {step}")
        
        # Keep only last 100 steps
        if len(self.reasoning_chain) > 100:
            self.reasoning_chain = self.reasoning_chain[-100:]
    
    def get_domain_feature(self, feature_name: str) -> Any:
        """Get domain-specific feature value"""
        return self.domain_features.get(feature_name)
    
    def set_domain_feature(self, feature_name: str, value: Any) -> None:
        """Set domain-specific feature value"""
        self.domain_features[feature_name] = value
    
    def _compute_expected_state(self) -> np.ndarray:
        """Compute expected state from quantum superposition"""
        if not self.quantum_states:
            return np.zeros(self.get_state_dimension())
        
        # Simplified expected value computation
        expected_state = np.zeros(self.get_state_dimension())
        for quantum_state in self.quantum_states.values():
            # In real implementation, would need proper state reconstruction
            contribution = quantum_state.probability * np.ones(self.get_state_dimension())
            expected_state += contribution
        
        return expected_state
    
    def _update_metadata(self) -> None:
        """Update metadata after state changes"""
        self.metadata.last_update = datetime.now()
        self.metadata.update_count += 1
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize superposition state"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.get_agent_type(),
            'state': self.state.value,
            'classical_state': self.classical_state.tolist() if self.classical_state is not None else None,
            'quantum_states': {
                key: {
                    'amplitude': complex(qs.amplitude),
                    'phase': qs.phase,
                    'probability': qs.probability,
                    'coherence': qs.coherence
                } for key, qs in self.quantum_states.items()
            },
            'domain_features': self.domain_features,
            'attention_weights': self.attention_weights,
            'reasoning_chain': self.reasoning_chain[-10:],  # Last 10 steps
            'performance_metrics': self.performance_metrics,
            'metadata': {
                'creation_time': self.metadata.creation_time.isoformat(),
                'last_update': self.metadata.last_update.isoformat(),
                'update_count': self.metadata.update_count,
                'coherence_decay_rate': self.metadata.coherence_decay_rate,
                'entanglement_strength': self.metadata.entanglement_strength
            }
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'UniversalSuperposition':
        """Deserialize superposition state"""
        # This would need to be implemented by subclasses
        raise NotImplementedError("Deserialization must be implemented by subclasses")
    
    def validate(self) -> bool:
        """Validate superposition state consistency"""
        try:
            # Check state consistency
            if self.state == SuperpositionState.SUPERPOSED and not self.quantum_states:
                return False
            
            # Check probability normalization
            if self.quantum_states:
                total_prob = sum(qs.probability for qs in self.quantum_states.values())
                if not np.isclose(total_prob, 1.0, atol=1e-6):
                    return False
            
            # Check coherence bounds
            for qs in self.quantum_states.values():
                if not 0 <= qs.coherence <= 1:
                    return False
            
            # Check attention weights
            for weight in self.attention_weights.values():
                if not 0 <= weight <= 1:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Validation failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id}, state={self.state.value})"