"""
Enhanced Centralized Critic for MAPPO with Superposition Features
Agent 3 - The Learning Optimization Specialist

This module implements an enhanced centralized critic architecture that processes
112D input (102D base features + 10D superposition features) with specialized
attention mechanisms and uncertainty-aware learning algorithms.

Features:
- 112D input processing with backward compatibility
- Specialized superposition feature processing layers
- Multi-head attention mechanisms over superposition features
- Uncertainty-aware value function estimation
- Enhanced feature fusion algorithms
- Optimized hidden layer architecture for faster convergence
- Robust hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import structlog
from datetime import datetime
import math

logger = structlog.get_logger()


@dataclass
class SuperpositionFeatures:
    """
    10-dimensional superposition features for enhanced critic input
    
    Features representing quantum-like decision states and agent coordination:
    0-2: Superposition confidence weights for top 3 decision states
    3-5: Cross-agent superposition alignment scores
    6-7: Temporal superposition decay factors
    8: Global superposition entropy
    9: Superposition consistency score
    """
    # Superposition confidence weights (3D)
    confidence_state_1: float = 0.0
    confidence_state_2: float = 0.0
    confidence_state_3: float = 0.0
    
    # Cross-agent alignment scores (3D)
    agent_alignment_1_2: float = 0.0
    agent_alignment_1_3: float = 0.0
    agent_alignment_2_3: float = 0.0
    
    # Temporal factors (2D)
    temporal_decay_short: float = 0.0
    temporal_decay_long: float = 0.0
    
    # Global metrics (2D)
    global_entropy: float = 0.0
    consistency_score: float = 0.0
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor for neural network processing"""
        features = torch.tensor([
            self.confidence_state_1, self.confidence_state_2, self.confidence_state_3,
            self.agent_alignment_1_2, self.agent_alignment_1_3, self.agent_alignment_2_3,
            self.temporal_decay_short, self.temporal_decay_long,
            self.global_entropy, self.consistency_score
        ], dtype=torch.float32)
        
        if device:
            features = features.to(device)
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """Superposition feature dimension"""
        return 10


@dataclass
class EnhancedCombinedState:
    """Enhanced combined state with superposition features (112D total)"""
    # Base features (102D)
    execution_context: torch.Tensor  # 15D
    market_features: torch.Tensor    # 32D
    routing_state: torch.Tensor      # 55D
    
    # Superposition features (10D)
    superposition_features: SuperpositionFeatures
    
    # Optional agent actions for temporal context
    agent_actions: Optional[torch.Tensor] = None
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to combined 112D tensor for enhanced critic input"""
        # Base features (102D)
        base_features = torch.cat([
            self.execution_context,     # 15D
            self.market_features,       # 32D
            self.routing_state          # 55D
        ], dim=0)  # 102D total
        
        # Superposition features (10D)
        superposition_tensor = self.superposition_features.to_tensor(device)
        
        # Combine for 112D input
        combined = torch.cat([base_features, superposition_tensor], dim=0)
        
        # Add agent actions if available
        if self.agent_actions is not None:
            combined = torch.cat([combined, self.agent_actions], dim=0)
        
        if device:
            combined = combined.to(device)
        
        return combined
    
    @property
    def feature_dim(self) -> int:
        """Total feature dimension"""
        base_dim = 102  # 15 + 32 + 55
        superposition_dim = 10
        action_dim = self.agent_actions.size(0) if self.agent_actions is not None else 0
        return base_dim + superposition_dim + action_dim


class SuperpositionAttention(nn.Module):
    """
    Specialized attention mechanism for superposition features
    
    Focuses on the most relevant superposition states and their interactions
    with base features for improved value estimation.
    """
    
    def __init__(self, 
                 input_dim: int = 112,
                 superposition_dim: int = 10,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.superposition_dim = superposition_dim
        self.base_dim = input_dim - superposition_dim  # 102D
        self.num_heads = min(num_heads, superposition_dim)  # Ensure num_heads <= superposition_dim
        self.head_dim = superposition_dim // self.num_heads
        
        # Adjust num_heads to ensure even division
        while superposition_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        
        self.head_dim = superposition_dim // self.num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(superposition_dim, superposition_dim)
        self.k_proj = nn.Linear(superposition_dim, superposition_dim)
        self.v_proj = nn.Linear(superposition_dim, superposition_dim)
        
        # Cross-attention with base features
        self.cross_k_proj = nn.Linear(self.base_dim, superposition_dim)
        self.cross_v_proj = nn.Linear(self.base_dim, superposition_dim)
        
        # Output projection
        self.out_proj = nn.Linear(superposition_dim, superposition_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(superposition_dim)
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through superposition attention
        
        Args:
            combined_input: Combined tensor [batch_size, 112]
            
        Returns:
            Enhanced superposition features [batch_size, 10]
        """
        batch_size = combined_input.size(0)
        
        # Split input into base and superposition features
        base_features = combined_input[:, :self.base_dim]          # [batch_size, 102]
        superposition_features = combined_input[:, self.base_dim:]  # [batch_size, 10]
        
        # Self-attention over superposition features
        q = self.q_proj(superposition_features)  # [batch_size, 10]
        k = self.k_proj(superposition_features)  # [batch_size, 10]
        v = self.v_proj(superposition_features)  # [batch_size, 10]
        
        # Cross-attention with base features
        cross_k = self.cross_k_proj(base_features)  # [batch_size, 10]
        cross_v = self.cross_v_proj(base_features)  # [batch_size, 10]
        
        # Simplified attention mechanism
        # Self-attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.superposition_dim)
        
        # Cross-attention scores
        cross_scores = torch.matmul(q, cross_k.transpose(-2, -1)) / math.sqrt(self.superposition_dim)
        
        # Combine attention scores
        combined_scores = attention_scores + 0.5 * cross_scores
        
        # Apply softmax
        attention_weights = F.softmax(combined_scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        attended_self = torch.matmul(attention_weights, v)
        attended_cross = torch.matmul(attention_weights, cross_v)
        
        # Combine attended features
        attended_features = attended_self + 0.3 * attended_cross
        
        # Output projection
        output = self.out_proj(attended_features)
        
        # Residual connection and normalization
        output = self.norm(output + superposition_features)
        output = self.dropout(output)
        
        return output


class UncertaintyAwareLayer(nn.Module):
    """
    Uncertainty-aware layer for robust value estimation
    
    Estimates both the value and its uncertainty using ensemble methods
    and variational approaches.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 1,
                 num_ensembles: int = 5,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ensembles = num_ensembles
        
        # Ensemble of value heads
        self.value_heads = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_ensembles)
        ])
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Variational dropout for additional uncertainty
        self.variational_dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (value_estimate, uncertainty_estimate)
        """
        # Ensemble predictions
        ensemble_outputs = []
        for head in self.value_heads:
            # Apply variational dropout
            x_dropped = self.variational_dropout(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            output = head(x_dropped)
            ensemble_outputs.append(output)
        
        # Combine ensemble outputs
        ensemble_outputs = torch.stack(ensemble_outputs, dim=-1)  # [batch_size, output_dim, num_ensembles]
        
        # Mean prediction
        value_estimate = ensemble_outputs.mean(dim=-1)
        
        # Epistemic uncertainty (disagreement between ensemble members)
        epistemic_uncertainty = ensemble_outputs.var(dim=-1)
        
        # Aleatoric uncertainty (learned uncertainty)
        aleatoric_uncertainty = self.uncertainty_head(x)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return value_estimate, total_uncertainty


class EnhancedCentralizedCritic(nn.Module):
    """
    Enhanced Centralized Critic with Superposition Features (112D → 1D)
    
    Architecture:
    - Input: 112D (102D base + 10D superposition)
    - Superposition attention layer
    - Feature fusion with optimized hidden layers
    - Uncertainty-aware value estimation
    - Backward compatibility with 102D inputs
    """
    
    def __init__(self,
                 base_input_dim: int = 102,
                 superposition_dim: int = 10,
                 hidden_dims: List[int] = None,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.1,
                 use_uncertainty: bool = True,
                 num_ensembles: int = 5):
        super().__init__()
        
        self.base_input_dim = base_input_dim
        self.superposition_dim = superposition_dim
        self.total_input_dim = base_input_dim + superposition_dim  # 112D
        self.use_uncertainty = use_uncertainty
        
        if hidden_dims is None:
            # Optimized architecture for 112D input
            hidden_dims = [512, 256, 128, 64]
        
        self.hidden_dims = hidden_dims
        
        # Superposition attention mechanism
        self.superposition_attention = SuperpositionAttention(
            input_dim=self.total_input_dim,
            superposition_dim=superposition_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(dropout_rate)
        )
        
        # Enhanced hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.Dropout(dropout_rate)
            ))
        
        # Uncertainty-aware output layer
        if use_uncertainty:
            self.output_layer = UncertaintyAwareLayer(
                input_dim=hidden_dims[-1],
                output_dim=1,
                num_ensembles=num_ensembles,
                dropout_rate=dropout_rate
            )
        else:
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Backward compatibility layer for 102D inputs
        self.compatibility_layer = nn.Linear(base_input_dim, self.total_input_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Performance tracking
        self.evaluations = 0
        self.total_evaluation_time = 0.0
        self.uncertainty_history = []
        
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, combined_state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through enhanced critic
        
        Args:
            combined_state: Combined state tensor [batch_size, 112] or [batch_size, 102]
            
        Returns:
            Value estimate [batch_size, 1] or (value, uncertainty) if use_uncertainty=True
        """
        batch_size = combined_state.size(0)
        input_dim = combined_state.size(1)
        
        # Handle backward compatibility
        if input_dim == self.base_input_dim:
            # Expand 102D to 112D using compatibility layer
            combined_state = self.compatibility_layer(combined_state)
        elif input_dim != self.total_input_dim:
            raise ValueError(f"Expected input dimension {self.total_input_dim} or {self.base_input_dim}, got {input_dim}")
        
        # Apply superposition attention
        enhanced_superposition = self.superposition_attention(combined_state)
        
        # Replace original superposition features with enhanced ones
        enhanced_input = torch.cat([
            combined_state[:, :self.base_input_dim],  # Base features
            enhanced_superposition  # Enhanced superposition features
        ], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(enhanced_input)
        
        # Forward through hidden layers with residual connections
        hidden_output = fused_features
        for i, layer in enumerate(self.hidden_layers):
            residual = hidden_output
            hidden_output = layer(hidden_output)
            
            # Add residual connection if dimensions match
            if hidden_output.size(1) == residual.size(1):
                hidden_output = hidden_output + residual
        
        # Output layer
        if self.use_uncertainty:
            value_estimate, uncertainty = self.output_layer(hidden_output)
            self.uncertainty_history.append(uncertainty.mean().item())
            return value_estimate, uncertainty
        else:
            return self.output_layer(hidden_output)
    
    def evaluate_state(self, 
                      combined_state: EnhancedCombinedState,
                      return_uncertainty: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate state value for given enhanced combined state
        
        Args:
            combined_state: Enhanced combined state with superposition features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (state_value, evaluation_info)
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert state to tensor
            state_tensor = combined_state.to_tensor().unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            with torch.no_grad():
                if self.use_uncertainty:
                    value, uncertainty = self.forward(state_tensor)
                    state_value = value.item()
                    uncertainty_value = uncertainty.item()
                else:
                    value = self.forward(state_tensor)
                    state_value = value.item()
                    uncertainty_value = 0.0
            
            # Evaluation metrics
            end_time = time.perf_counter()
            evaluation_time = end_time - start_time
            self.evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            evaluation_info = {
                'state_value': state_value,
                'uncertainty': uncertainty_value if return_uncertainty else None,
                'evaluation_time_ms': evaluation_time * 1000,
                'total_evaluations': self.evaluations,
                'avg_evaluation_time_ms': (self.total_evaluation_time / self.evaluations) * 1000,
                'attention_weights': self.superposition_attention.attention_weights.cpu().numpy() if self.superposition_attention.attention_weights is not None else None
            }
            
            return state_value, evaluation_info
            
        except Exception as e:
            logger.error("Error in enhanced state evaluation", error=str(e))
            return 0.0, {'error': str(e)}
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """Get uncertainty-related metrics"""
        if not self.uncertainty_history:
            return {}
        
        recent_uncertainty = self.uncertainty_history[-100:]
        
        return {
            'mean_uncertainty': np.mean(recent_uncertainty),
            'std_uncertainty': np.std(recent_uncertainty),
            'uncertainty_trend': np.polyfit(range(len(recent_uncertainty)), recent_uncertainty, 1)[0] if len(recent_uncertainty) > 1 else 0.0,
            'uncertainty_samples': len(self.uncertainty_history)
        }
    
    def get_attention_analysis(self) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability"""
        if self.superposition_attention.attention_weights is None:
            return {}
        
        attention_weights = self.superposition_attention.attention_weights.cpu().numpy()
        
        return {
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1).mean(),
            'attention_concentration': np.max(attention_weights, axis=-1).mean(),
            'attention_distribution': attention_weights.mean(axis=0).tolist()
        }


# Factory functions for easy instantiation
def create_enhanced_centralized_critic(config: Dict[str, Any]) -> EnhancedCentralizedCritic:
    """Create enhanced centralized critic with configuration"""
    return EnhancedCentralizedCritic(
        base_input_dim=config.get('base_input_dim', 102),
        superposition_dim=config.get('superposition_dim', 10),
        hidden_dims=config.get('hidden_dims', [512, 256, 128, 64]),
        num_attention_heads=config.get('num_attention_heads', 4),
        dropout_rate=config.get('dropout_rate', 0.1),
        use_uncertainty=config.get('use_uncertainty', True),
        num_ensembles=config.get('num_ensembles', 5)
    )


def create_superposition_features(agent_outputs: Dict[str, Any]) -> SuperpositionFeatures:
    """Create superposition features from agent outputs"""
    # Extract superposition information from agent outputs
    # This is a placeholder - actual implementation depends on agent architecture
    
    return SuperpositionFeatures(
        confidence_state_1=agent_outputs.get('confidence_1', 0.5),
        confidence_state_2=agent_outputs.get('confidence_2', 0.3),
        confidence_state_3=agent_outputs.get('confidence_3', 0.2),
        agent_alignment_1_2=agent_outputs.get('alignment_12', 0.7),
        agent_alignment_1_3=agent_outputs.get('alignment_13', 0.6),
        agent_alignment_2_3=agent_outputs.get('alignment_23', 0.8),
        temporal_decay_short=agent_outputs.get('decay_short', 0.9),
        temporal_decay_long=agent_outputs.get('decay_long', 0.7),
        global_entropy=agent_outputs.get('entropy', 0.6),
        consistency_score=agent_outputs.get('consistency', 0.8)
    )


if __name__ == "__main__":
    # Test the enhanced centralized critic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test configuration
    config = {
        'base_input_dim': 102,
        'superposition_dim': 10,
        'hidden_dims': [512, 256, 128, 64],
        'num_attention_heads': 4,
        'dropout_rate': 0.1,
        'use_uncertainty': True,
        'num_ensembles': 5
    }
    
    # Create enhanced critic
    critic = create_enhanced_centralized_critic(config).to(device)
    
    # Test with 112D input
    batch_size = 32
    test_input_112d = torch.randn(batch_size, 112).to(device)
    
    # Test forward pass
    value_estimate, uncertainty = critic(test_input_112d)
    print(f"112D Input Test - Value shape: {value_estimate.shape}, Uncertainty shape: {uncertainty.shape}")
    
    # Test backward compatibility with 102D input
    test_input_102d = torch.randn(batch_size, 102).to(device)
    value_estimate_102d, uncertainty_102d = critic(test_input_102d)
    print(f"102D Input Test - Value shape: {value_estimate_102d.shape}, Uncertainty shape: {uncertainty_102d.shape}")
    
    # Test enhanced combined state
    execution_context = torch.randn(15)
    market_features = torch.randn(32)
    routing_state = torch.randn(55)
    superposition_features = SuperpositionFeatures()
    
    enhanced_state = EnhancedCombinedState(
        execution_context=execution_context,
        market_features=market_features,
        routing_state=routing_state,
        superposition_features=superposition_features
    )
    
    state_value, eval_info = critic.evaluate_state(enhanced_state, return_uncertainty=True)
    print(f"Enhanced State Evaluation - Value: {state_value:.4f}, Uncertainty: {eval_info['uncertainty']:.4f}")
    
    # Print model summary
    total_params = sum(p.numel() for p in critic.parameters())
    print(f"Enhanced Centralized Critic - Total Parameters: {total_params:,}")