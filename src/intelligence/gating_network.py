"""
Intelligent Gating Network for Dynamic Agent Coordination.

Optimized gating network that dynamically weights agent contributions based on
market context and agent expertise. Targets <0.3ms performance for real-time operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import logging

class GatingNetwork(nn.Module):
    """
    Intelligent gating network for dynamic agent coordination.
    
    Determines optimal weighting of different agents based on current market
    conditions and agent expertise areas. Optimized for minimal latency.
    """
    
    def __init__(
        self,
        shared_context_dim: int = 6,
        n_agents: int = 3,
        hidden_dim: int = 32,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
        min_weight: float = 0.01,
        max_weight: float = 0.99
    ):
        super().__init__()
        
        self.shared_context_dim = shared_context_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Gating network architecture (kept simple for speed)
        self.context_encoder = nn.Sequential(
            nn.Linear(shared_context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_agents)
        )
        
        # Agent expertise biases (learnable specialization)
        self.agent_expertise = nn.Parameter(torch.zeros(n_agents, shared_context_dim))
        
        # Context normalization for stability
        self.context_norm = nn.BatchNorm1d(shared_context_dim)
        
        # Performance tracking
        self.gating_history = []
        self.max_history = 100
        
        # Initialize weights optimally
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize network weights for optimal performance."""
        # Initialize context encoder
        for layer in self.context_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize agent expertise based on known specializations
        # Agent 0: MLMI (focus on volatility, mmd_score)
        # Agent 1: NWRQK (focus on momentum features)  
        # Agent 2: Regime (focus on regime indicators)
        
        expertise_init = torch.tensor([
            [0.3, 0.2, 0.3, 0.1, 0.05, 0.05],  # MLMI: volatility, mmd focused
            [0.1, 0.1, 0.1, 0.3, 0.3, 0.1],    # NWRQK: momentum focused
            [0.2, 0.3, 0.2, 0.1, 0.1, 0.1]     # Regime: volatility, mmd focused
        ])
        
        self.agent_expertise.data = expertise_init
        
        self.logger.info("Gating network initialized with agent specializations")
    
    def forward(
        self, 
        shared_context: torch.Tensor,
        agent_confidences: Optional[torch.Tensor] = None,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute dynamic gating weights.
        
        Args:
            shared_context: Market context tensor (batch_size, context_dim)
            agent_confidences: Optional confidence scores (batch_size, n_agents)
            temperature_override: Optional temperature for softmax
            
        Returns:
            Dictionary with gating weights and additional info
        """
        batch_size = shared_context.shape[0]
        
        # Normalize context for stability
        if batch_size > 1:
            context_normalized = self.context_norm(shared_context)
        else:
            # Handle single sample case
            context_normalized = shared_context
        
        # Compute agent-context compatibility scores
        expertise_scores = torch.matmul(context_normalized, self.agent_expertise.T)
        
        # Compute context-based gating scores
        context_scores = self.context_encoder(context_normalized)
        
        # Combine expertise and context scores
        combined_scores = expertise_scores + context_scores
        
        # Apply agent confidences if provided
        if agent_confidences is not None:
            combined_scores = combined_scores * agent_confidences
        
        # Apply temperature scaling
        temp = temperature_override if temperature_override is not None else self.temperature
        scaled_scores = combined_scores / temp
        
        # Compute gating weights with softmax
        gating_weights = F.softmax(scaled_scores, dim=-1)
        
        # Enforce weight bounds for stability
        gating_weights = torch.clamp(gating_weights, self.min_weight, self.max_weight)
        
        # Renormalize after clamping
        gating_weights = gating_weights / gating_weights.sum(dim=-1, keepdim=True)
        
        # Track gating patterns for analysis
        if self.training and len(self.gating_history) < self.max_history:
            self.gating_history.append(gating_weights.detach().cpu().mean(0))
        elif self.training:
            # Rolling window
            self.gating_history.pop(0)
            self.gating_history.append(gating_weights.detach().cpu().mean(0))
        
        return {
            'gating_weights': gating_weights,
            'expertise_scores': expertise_scores,
            'context_scores': context_scores,
            'combined_scores': combined_scores
        }
    
    def get_agent_specialization_info(self) -> Dict[str, Any]:
        """Get information about agent specializations."""
        expertise = self.agent_expertise.detach().cpu().numpy()
        
        agent_info = {}
        agent_names = ['MLMI', 'NWRQK', 'Regime']
        context_features = ['volatility_30', 'mmd_score', 'momentum_20', 
                           'momentum_50', 'volume_ratio', 'price_trend']
        
        for i, agent_name in enumerate(agent_names):
            agent_info[agent_name] = {
                'expertise_weights': expertise[i].tolist(),
                'primary_features': [
                    context_features[j] for j in np.argsort(expertise[i])[::-1][:2]
                ],
                'specialization_strength': float(np.max(expertise[i]))
            }
        
        return agent_info
    
    def get_gating_statistics(self) -> Dict[str, Any]:
        """Get statistics about gating patterns."""
        if not self.gating_history:
            return {'status': 'no_data'}
        
        gating_stack = torch.stack(self.gating_history)
        
        return {
            'mean_weights': gating_stack.mean(0).tolist(),
            'std_weights': gating_stack.std(0).tolist(),
            'min_weights': gating_stack.min(0)[0].tolist(),
            'max_weights': gating_stack.max(0)[0].tolist(),
            'weight_stability': float(gating_stack.std().item()),
            'num_samples': len(self.gating_history)
        }
    
    def update_agent_expertise(self, performance_feedback: Dict[str, float]):
        """Update agent expertise based on performance feedback."""
        try:
            # Simple adaptive update based on performance
            learning_rate = 0.01
            
            for agent_name, performance in performance_feedback.items():
                agent_idx = {'MLMI': 0, 'NWRQK': 1, 'Regime': 2}.get(agent_name)
                if agent_idx is not None:
                    # Adjust expertise weights based on performance
                    adjustment = (performance - 0.5) * learning_rate  # Center around 0.5
                    self.agent_expertise.data[agent_idx] += adjustment
                    
                    # Ensure weights remain positive and normalized
                    self.agent_expertise.data[agent_idx] = torch.clamp(
                        self.agent_expertise.data[agent_idx], min=0.01
                    )
                    self.agent_expertise.data[agent_idx] /= self.agent_expertise.data[agent_idx].sum()
            
            self.logger.info("Agent expertise updated based on performance feedback")
            
        except Exception as e:
            self.logger.error(f"Error updating agent expertise: {e}")
    
    def reset_gating_history(self):
        """Reset gating pattern history."""
        self.gating_history.clear()
        self.logger.info("Gating history reset")

class FastGatingNetwork(nn.Module):
    """
    Ultra-fast gating network optimized for minimal latency.
    
    Simplified version of GatingNetwork with pre-computed lookup tables
    and minimal operations for production use.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pre-computed weight matrices for speed
        self.register_buffer('weight_matrix', torch.randn(3, 6) * 0.1)
        self.register_buffer('bias_vector', torch.zeros(3))
        
        # Bounds for stability
        self.min_weight = config.get('min_weight', 0.01)
        self.max_weight = config.get('max_weight', 0.99)
        
        # Initialize with agent specializations
        self._initialize_fast_weights()
    
    def _initialize_fast_weights(self):
        """Initialize pre-computed weights for fast inference."""
        # Hardcoded optimal weights based on agent specializations
        fast_weights = torch.tensor([
            [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],  # MLMI weights
            [0.1, 0.1, 0.4, 0.3, 0.05, 0.05],  # NWRQK weights  
            [0.3, 0.4, 0.1, 0.1, 0.1, 0.0]     # Regime weights
        ])
        
        self.weight_matrix.copy_(fast_weights)
        self.logger.info("Fast gating network weights initialized")
    
    def forward(self, shared_context: torch.Tensor) -> torch.Tensor:
        """Ultra-fast forward pass."""
        # Single matrix multiplication
        scores = torch.matmul(shared_context, self.weight_matrix.T) + self.bias_vector
        
        # Fast softmax
        weights = F.softmax(scores, dim=-1)
        
        # Clamp and renormalize
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights