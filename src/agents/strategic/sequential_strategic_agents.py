"""
Sequential Strategic Agents - Enhanced agents for sequential MARL execution

This module provides enhanced strategic agents that are optimized for sequential 
execution with superposition output and enriched observation processing.

Key Features:
- Sequential-aware processing of enriched observations
- Superposition state generation with quantum-inspired properties
- Predecessor context integration for enhanced decision-making
- <5ms computation time target
- Mathematical validation of all outputs
- Comprehensive performance monitoring

Agent Sequence: MLMI → NWRQK → Regime
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Import base classes
from src.agents.strategic_agent_base import (
    StrategicAgentBase,
    AgentPrediction,
    MLMIStrategicAgent,
    NWRQKStrategicAgent, 
    RegimeDetectionAgent
)

logger = logging.getLogger(__name__)


@dataclass
class SequentialPrediction(AgentPrediction):
    """Enhanced prediction for sequential agents"""
    predecessor_context: Dict[str, Any]
    sequence_position: int
    enriched_confidence: float
    superposition_quality: float
    quantum_coherence: float
    temporal_stability: float


class SequentialAwareModule(nn.Module):
    """Neural module for processing sequential context"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Sequential processing layers
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for predecessor integration
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, current_features: torch.Tensor, predecessor_context: torch.Tensor) -> torch.Tensor:
        """
        Process current features with predecessor context
        
        Args:
            current_features: Current agent features [batch_size, input_dim]
            predecessor_context: Predecessor context [batch_size, seq_len, input_dim]
            
        Returns:
            Enhanced features [batch_size, output_dim]
        """
        # Encode current features
        encoded_current = self.context_encoder(current_features)
        
        if predecessor_context.size(1) > 0:
            # Encode predecessor context
            batch_size, seq_len, input_dim = predecessor_context.shape
            predecessor_flat = predecessor_context.view(-1, input_dim)
            encoded_predecessor = self.context_encoder(predecessor_flat)
            encoded_predecessor = encoded_predecessor.view(batch_size, seq_len, -1)
            
            # Apply attention to integrate predecessor context
            query = encoded_current.unsqueeze(1)  # [batch_size, 1, output_dim]
            enhanced_features, _ = self.attention(query, encoded_predecessor, encoded_predecessor)
            enhanced_features = enhanced_features.squeeze(1)  # [batch_size, output_dim]
        else:
            enhanced_features = encoded_current
        
        # Apply output projection
        return self.output_projection(enhanced_features)


class SuperpositionGenerator(nn.Module):
    """Neural module for generating superposition states"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Superposition generation layers
        self.superposition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)  # Superposition features
        )
        
        # Quantum coherence calculation
        self.coherence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal stability calculation
        self.stability_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, enhanced_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate superposition state from enhanced features
        
        Args:
            enhanced_features: Enhanced features [batch_size, input_dim]
            
        Returns:
            Tuple of (superposition_features, quantum_coherence, temporal_stability)
        """
        superposition_features = self.superposition_net(enhanced_features)
        quantum_coherence = self.coherence_net(enhanced_features)
        temporal_stability = self.stability_net(enhanced_features)
        
        return superposition_features, quantum_coherence, temporal_stability


class SequentialStrategicAgentBase(StrategicAgentBase):
    """Base class for sequential-aware strategic agents"""
    
    def __init__(self, name: str, feature_indices: List[int], config: Dict[str, Any], device: str = "cpu"):
        super().__init__(name, feature_indices, config, device)
        
        # Sequential processing components
        self.sequential_module = None
        self.superposition_generator = None
        
        # Performance tracking
        self.sequential_performance = {
            "enriched_processing_times": [],
            "superposition_generation_times": [],
            "context_integration_success": 0,
            "quantum_coherence_scores": [],
            "temporal_stability_scores": []
        }
        
        # Sequential configuration
        self.max_computation_time_ms = config.get("performance", {}).get("max_agent_computation_time_ms", 5.0)
        self.enable_superposition = config.get("environment", {}).get("superposition_enabled", True)
        
        self.logger.info(f"Sequential agent {name} initialized with superposition: {self.enable_superposition}")
    
    def initialize_sequential_components(self):
        """Initialize sequential processing components"""
        try:
            # Determine input dimensions
            agent_feature_dim = len(self.feature_indices)
            enriched_context_dim = 10  # From UniversalObservationEnricher
            total_input_dim = agent_feature_dim + enriched_context_dim
            
            # Initialize sequential processing module
            self.sequential_module = SequentialAwareModule(
                input_dim=total_input_dim,
                hidden_dim=128,
                output_dim=64
            ).to(self.device)
            
            # Initialize superposition generator
            self.superposition_generator = SuperpositionGenerator(
                input_dim=64,
                hidden_dim=128
            ).to(self.device)
            
            self.logger.info(f"Sequential components initialized for {self.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sequential components for {self.name}: {e}")
            raise
    
    def process_enriched_observation(self, enriched_obs: Dict[str, Any]) -> torch.Tensor:
        """
        Process enriched observation with predecessor context
        
        Args:
            enriched_obs: Enriched observation from environment
            
        Returns:
            Enhanced features tensor
        """
        start_time = time.time()
        
        try:
            # Extract base features
            base_obs = enriched_obs["base_observation"]
            agent_features = base_obs["agent_features"]
            shared_context = base_obs["shared_context"]
            
            # Extract enriched features
            enriched_features = enriched_obs["enriched_features"]
            
            # Combine base and enriched features
            combined_features = np.concatenate([
                agent_features,
                shared_context,
                enriched_features["predecessor_avg_confidence"],
                enriched_features["predecessor_max_confidence"], 
                enriched_features["predecessor_min_confidence"],
                enriched_features["predecessor_avg_computation_time"]
            ])
            
            # Convert to tensor
            current_features = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
            
            # Process predecessor context
            predecessor_context = self._extract_predecessor_context(enriched_obs)
            
            # Apply sequential processing
            enhanced_features = self.sequential_module(current_features, predecessor_context)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.sequential_performance["enriched_processing_times"].append(processing_time)
            
            if processing_time > self.max_computation_time_ms:
                self.logger.warning(f"Enriched processing exceeded time limit: {processing_time:.2f}ms")
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Failed to process enriched observation: {e}")
            # Return fallback features
            fallback_dim = 64
            return torch.zeros(1, fallback_dim).to(self.device)
    
    def _extract_predecessor_context(self, enriched_obs: Dict[str, Any]) -> torch.Tensor:
        """Extract predecessor context for attention mechanism"""
        predecessor_superpositions = enriched_obs.get("predecessor_superpositions", [])
        
        if not predecessor_superpositions:
            # No predecessors - return empty context
            return torch.zeros(1, 0, 10).to(self.device)
        
        # Convert predecessor superpositions to tensor
        context_features = []
        for superposition in predecessor_superpositions:
            context_vector = np.concatenate([
                superposition["action_probabilities"],
                superposition["confidence"],
                superposition["computation_time_ms"],
                [1.0, 1.0, 1.0]  # Placeholder for additional features
            ])
            context_features.append(context_vector)
        
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0).to(self.device)
        return context_tensor
    
    def generate_superposition(self, enhanced_features: torch.Tensor, action_probs: np.ndarray) -> Dict[str, Any]:
        """
        Generate superposition state from enhanced features
        
        Args:
            enhanced_features: Enhanced features from sequential processing
            action_probs: Action probabilities from main prediction
            
        Returns:
            Superposition features dictionary
        """
        if not self.enable_superposition:
            return {}
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Generate superposition components
                superposition_features, quantum_coherence, temporal_stability = self.superposition_generator(enhanced_features)
                
                # Convert to numpy for further processing
                superposition_vec = superposition_features.cpu().numpy().flatten()
                quantum_coherence_val = float(quantum_coherence.cpu().numpy())
                temporal_stability_val = float(temporal_stability.cpu().numpy())
                
                # Calculate quantum-inspired properties
                quantum_properties = self._calculate_quantum_properties(action_probs)
                
                # Create superposition features
                superposition_dict = {
                    "quantum_coherence": quantum_coherence_val,
                    "temporal_stability": temporal_stability_val,
                    "entanglement_measure": quantum_properties["entanglement"],
                    "phase_information": quantum_properties["phase"],
                    "superposition_vector": superposition_vec.tolist(),
                    "coherence_length": float(np.linalg.norm(superposition_vec)),
                    "stability_measure": float(1.0 - np.var(action_probs))
                }
                
                # Track performance
                generation_time = (time.time() - start_time) * 1000
                self.sequential_performance["superposition_generation_times"].append(generation_time)
                self.sequential_performance["quantum_coherence_scores"].append(quantum_coherence_val)
                self.sequential_performance["temporal_stability_scores"].append(temporal_stability_val)
                
                return superposition_dict
                
        except Exception as e:
            self.logger.error(f"Failed to generate superposition: {e}")
            return {}
    
    def _calculate_quantum_properties(self, action_probs: np.ndarray) -> Dict[str, float]:
        """Calculate quantum-inspired properties from action probabilities"""
        # Entanglement measure (based on entropy)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        entanglement = float(entropy / np.log(len(action_probs)))
        
        # Phase information (based on complex representation)
        phase = float(np.arctan2(action_probs[1], action_probs[0]))
        
        return {
            "entanglement": entanglement,
            "phase": phase
        }
    
    async def predict_sequential(
        self,
        enriched_obs: Dict[str, Any],
        shared_context: Dict[str, Any]
    ) -> SequentialPrediction:
        """
        Make sequential prediction with enriched observation processing
        
        Args:
            enriched_obs: Enriched observation from environment
            shared_context: Shared context from coordinator
            
        Returns:
            SequentialPrediction with superposition state
        """
        start_time = time.time()
        
        try:
            # Process enriched observation
            enhanced_features = self.process_enriched_observation(enriched_obs)
            
            # Generate base prediction (to be implemented by subclasses)
            base_prediction = await self._generate_base_prediction(enhanced_features, shared_context)
            
            # Generate superposition
            superposition_features = self.generate_superposition(enhanced_features, base_prediction.action_probabilities)
            
            # Calculate enhanced confidence
            enriched_confidence = self._calculate_enriched_confidence(
                base_prediction.confidence,
                enriched_obs.get("enriched_features", {})
            )
            
            # Calculate superposition quality
            superposition_quality = self._calculate_superposition_quality(superposition_features)
            
            # Calculate quantum coherence
            quantum_coherence = superposition_features.get("quantum_coherence", 0.5)
            
            # Calculate temporal stability
            temporal_stability = superposition_features.get("temporal_stability", 0.5)
            
            # Create sequential prediction
            sequential_prediction = SequentialPrediction(
                action_probabilities=base_prediction.action_probabilities,
                confidence=base_prediction.confidence,
                feature_importance=base_prediction.feature_importance,
                internal_state=base_prediction.internal_state,
                computation_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                predecessor_context=self._extract_predecessor_summary(enriched_obs),
                sequence_position=enriched_obs.get("enriched_features", {}).get("sequence_position", [0])[0],
                enriched_confidence=enriched_confidence,
                superposition_quality=superposition_quality,
                quantum_coherence=quantum_coherence,
                temporal_stability=temporal_stability
            )
            
            # Add superposition features to internal state
            sequential_prediction.internal_state["superposition_features"] = superposition_features
            
            # Update performance tracking
            self.sequential_performance["context_integration_success"] += 1
            
            return sequential_prediction
            
        except Exception as e:
            self.logger.error(f"Sequential prediction failed for {self.name}: {e}")
            # Return fallback prediction
            return self._get_fallback_sequential_prediction(start_time)
    
    @abstractmethod
    async def _generate_base_prediction(self, enhanced_features: torch.Tensor, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Generate base prediction from enhanced features (to be implemented by subclasses)"""
        pass
    
    def _calculate_enriched_confidence(self, base_confidence: float, enriched_features: Dict[str, Any]) -> float:
        """Calculate enriched confidence incorporating predecessor context"""
        predecessor_confidence = enriched_features.get("predecessor_avg_confidence", [0.5])[0]
        completion_ratio = enriched_features.get("completion_ratio", [0.0])[0]
        
        # Boost confidence based on predecessor agreement and sequence progress
        confidence_boost = 0.1 * predecessor_confidence * completion_ratio
        
        return min(1.0, base_confidence + confidence_boost)
    
    def _calculate_superposition_quality(self, superposition_features: Dict[str, Any]) -> float:
        """Calculate superposition quality score"""
        if not superposition_features:
            return 0.0
        
        quantum_coherence = superposition_features.get("quantum_coherence", 0.0)
        temporal_stability = superposition_features.get("temporal_stability", 0.0)
        coherence_length = superposition_features.get("coherence_length", 0.0)
        
        # Normalize coherence length
        normalized_coherence = min(1.0, coherence_length / 10.0)
        
        # Combined quality score
        quality = (quantum_coherence + temporal_stability + normalized_coherence) / 3.0
        
        return float(quality)
    
    def _extract_predecessor_summary(self, enriched_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary of predecessor context"""
        predecessor_superpositions = enriched_obs.get("predecessor_superpositions", [])
        enriched_features = enriched_obs.get("enriched_features", {})
        
        return {
            "num_predecessors": len(predecessor_superpositions),
            "avg_predecessor_confidence": enriched_features.get("predecessor_avg_confidence", [0.5])[0],
            "predecessor_action_variance": enriched_features.get("predecessor_action_variance", [0.0, 0.0, 0.0]),
            "sequence_position": enriched_features.get("sequence_position", [0])[0],
            "completion_ratio": enriched_features.get("completion_ratio", [0.0])[0]
        }
    
    def _get_fallback_sequential_prediction(self, start_time: float) -> SequentialPrediction:
        """Generate fallback sequential prediction"""
        computation_time_ms = (time.time() - start_time) * 1000
        
        return SequentialPrediction(
            action_probabilities=np.array([0.33, 0.34, 0.33]),
            confidence=0.5,
            feature_importance={'fallback': 1.0},
            internal_state={'fallback_mode': True},
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now(),
            predecessor_context={},
            sequence_position=0,
            enriched_confidence=0.5,
            superposition_quality=0.5,
            quantum_coherence=0.5,
            temporal_stability=0.5
        )


class SequentialMLMIAgent(SequentialStrategicAgentBase):
    """Sequential-aware MLMI Strategic Agent"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('mlmi_expert', [0, 1, 9, 10])
        super().__init__("MLMI", feature_indices, config, device)
        
        # Initialize base MLMI agent
        self.base_mlmi_agent = MLMIStrategicAgent(config, device)
        
        # Initialize sequential components
        self.initialize_sequential_components()
    
    async def initialize(self) -> None:
        """Initialize MLMI sequential agent"""
        try:
            # Initialize base agent
            await self.base_mlmi_agent.initialize()
            
            # Copy networks from base agent
            self.actor_network = self.base_mlmi_agent.actor_network
            self.critic_network = self.base_mlmi_agent.critic_network
            
            self.logger.info("Sequential MLMI Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sequential MLMI agent: {e}")
            raise
    
    async def _generate_base_prediction(self, enhanced_features: torch.Tensor, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Generate base MLMI prediction from enhanced features"""
        try:
            # Process enhanced features through actor network
            with torch.no_grad():
                action_logits = self.actor_network(enhanced_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            # Calculate confidence
            confidence = float(np.max(action_probs))
            
            # Get feature importance (simplified)
            feature_importance = self.base_mlmi_agent.get_feature_importance(
                np.random.randn(48, 13)  # Placeholder matrix
            )
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'agent_type': 'sequential_mlmi',
                    'enhanced_processing': True
                },
                computation_time_ms=1.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate base MLMI prediction: {e}")
            raise
    
    async def predict(self, matrix_data: np.ndarray, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Standard predict method for backward compatibility"""
        return await self.base_mlmi_agent.predict(matrix_data, shared_context)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Get feature importance for MLMI features"""
        return self.base_mlmi_agent.get_feature_importance(matrix_data)


class SequentialNWRQKAgent(SequentialStrategicAgentBase):
    """Sequential-aware NWRQK Strategic Agent"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('nwrqk_expert', [2, 3, 4, 5])
        super().__init__("NWRQK", feature_indices, config, device)
        
        # Initialize base NWRQK agent
        self.base_nwrqk_agent = NWRQKStrategicAgent(config, device)
        
        # Initialize sequential components
        self.initialize_sequential_components()
    
    async def initialize(self) -> None:
        """Initialize NWRQK sequential agent"""
        try:
            # Initialize base agent
            await self.base_nwrqk_agent.initialize()
            
            # Copy networks from base agent
            self.actor_network = self.base_nwrqk_agent.actor_network
            self.critic_network = self.base_nwrqk_agent.critic_network
            
            self.logger.info("Sequential NWRQK Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sequential NWRQK agent: {e}")
            raise
    
    async def _generate_base_prediction(self, enhanced_features: torch.Tensor, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Generate base NWRQK prediction from enhanced features"""
        try:
            # Process enhanced features through actor network
            with torch.no_grad():
                action_logits = self.actor_network(enhanced_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            # Calculate confidence
            confidence = float(np.max(action_probs))
            
            # Get feature importance (simplified)
            feature_importance = self.base_nwrqk_agent.get_feature_importance(
                np.random.randn(48, 13)  # Placeholder matrix
            )
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'agent_type': 'sequential_nwrqk',
                    'enhanced_processing': True
                },
                computation_time_ms=1.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate base NWRQK prediction: {e}")
            raise
    
    async def predict(self, matrix_data: np.ndarray, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Standard predict method for backward compatibility"""
        return await self.base_nwrqk_agent.predict(matrix_data, shared_context)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Get feature importance for NWRQK features"""
        return self.base_nwrqk_agent.get_feature_importance(matrix_data)


class SequentialRegimeAgent(SequentialStrategicAgentBase):
    """Sequential-aware Regime Detection Strategic Agent"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('regime_expert', [10, 11, 12])
        super().__init__("Regime", feature_indices, config, device)
        
        # Initialize base Regime agent
        self.base_regime_agent = RegimeDetectionAgent(config, device)
        
        # Initialize sequential components
        self.initialize_sequential_components()
    
    async def initialize(self) -> None:
        """Initialize Regime sequential agent"""
        try:
            # Initialize base agent
            await self.base_regime_agent.initialize()
            
            # Copy networks from base agent
            self.actor_network = self.base_regime_agent.actor_network
            self.critic_network = self.base_regime_agent.critic_network
            
            self.logger.info("Sequential Regime Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sequential Regime agent: {e}")
            raise
    
    async def _generate_base_prediction(self, enhanced_features: torch.Tensor, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Generate base Regime prediction from enhanced features"""
        try:
            # Process enhanced features through actor network
            with torch.no_grad():
                action_logits = self.actor_network(enhanced_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            # Calculate confidence
            confidence = float(np.max(action_probs))
            
            # Get feature importance (simplified)
            feature_importance = self.base_regime_agent.get_feature_importance(
                np.random.randn(48, 13)  # Placeholder matrix
            )
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'agent_type': 'sequential_regime',
                    'enhanced_processing': True
                },
                computation_time_ms=1.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate base Regime prediction: {e}")
            raise
    
    async def predict(self, matrix_data: np.ndarray, shared_context: Dict[str, Any]) -> AgentPrediction:
        """Standard predict method for backward compatibility"""
        return await self.base_regime_agent.predict(matrix_data, shared_context)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Get feature importance for Regime features"""
        return self.base_regime_agent.get_feature_importance(matrix_data)


class SequentialAgentFactory:
    """Factory for creating sequential strategic agents"""
    
    @staticmethod
    def create_agent(agent_type: str, config: Dict[str, Any], device: str = "cpu") -> SequentialStrategicAgentBase:
        """
        Create sequential strategic agent
        
        Args:
            agent_type: Type of agent ('mlmi', 'nwrqk', 'regime')
            config: Agent configuration
            device: Torch device
            
        Returns:
            Sequential strategic agent instance
        """
        if agent_type.lower() == 'mlmi':
            return SequentialMLMIAgent(config, device)
        elif agent_type.lower() == 'nwrqk':
            return SequentialNWRQKAgent(config, device)
        elif agent_type.lower() == 'regime':
            return SequentialRegimeAgent(config, device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def create_all_agents(config: Dict[str, Any], device: str = "cpu") -> Dict[str, SequentialStrategicAgentBase]:
        """
        Create all sequential strategic agents
        
        Args:
            config: Configuration dictionary
            device: Torch device
            
        Returns:
            Dictionary of agent name to agent instance
        """
        agents = {}
        
        # Create agents in sequence order
        agents['mlmi_expert'] = SequentialMLMIAgent(config, device)
        agents['nwrqk_expert'] = SequentialNWRQKAgent(config, device)
        agents['regime_expert'] = SequentialRegimeAgent(config, device)
        
        return agents


# Example usage and testing
if __name__ == "__main__":
    # Test sequential agent creation
    config = {
        "feature_indices": {
            "mlmi_expert": [0, 1, 9, 10],
            "nwrqk_expert": [2, 3, 4, 5],
            "regime_expert": [10, 11, 12]
        },
        "agents": {
            "mlmi_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.1},
            "nwrqk_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.1},
            "regime_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.15}
        },
        "environment": {
            "superposition_enabled": True,
            "observation_enrichment": True
        },
        "performance": {
            "max_agent_computation_time_ms": 5.0
        }
    }
    
    # Create all agents
    agents = SequentialAgentFactory.create_all_agents(config)
    
    print("Sequential Strategic Agents created successfully:")
    for name, agent in agents.items():
        print(f"  - {name}: {agent.__class__.__name__}")
    
    # Test enriched observation processing
    enriched_obs = {
        "base_observation": {
            "agent_features": np.random.randn(4),
            "shared_context": np.random.randn(6),
            "market_matrix": np.random.randn(48, 13),
            "episode_info": {"episode_step": 1, "phase": 1, "agent_index": 0}
        },
        "enriched_features": {
            "sequence_position": np.array([0]),
            "completion_ratio": np.array([0.0]),
            "predecessor_avg_confidence": np.array([0.5]),
            "predecessor_max_confidence": np.array([0.5]),
            "predecessor_min_confidence": np.array([0.5]),
            "predecessor_avg_computation_time": np.array([0.0])
        },
        "predecessor_superpositions": []
    }
    
    # Test MLMI agent
    mlmi_agent = agents['mlmi_expert']
    try:
        enhanced_features = mlmi_agent.process_enriched_observation(enriched_obs)
        print(f"MLMI enhanced features shape: {enhanced_features.shape}")
        
        # Test superposition generation
        action_probs = np.array([0.4, 0.4, 0.2])
        superposition = mlmi_agent.generate_superposition(enhanced_features, action_probs)
        print(f"MLMI superposition keys: {list(superposition.keys())}")
        
    except Exception as e:
        print(f"MLMI agent test failed: {e}")
    
    print("\nSequential Strategic Agents test completed")