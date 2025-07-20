"""
Enhanced Centralized Critic for MAPPO with MC Dropout Integration
AGENT 3 - Maximum Velocity Deployment

This module implements an enhanced centralized critic that receives and processes
MC dropout sample statistics from the single execution layer MC dropout engine.

Features:
- 127D input processing (112D base + 15D MC dropout features)
- Specialized MC dropout attention mechanism
- Integration with 1000-sample MC dropout statistics
- Uncertainty-aware value function estimation
- Feedback loop for MAPPO learning from MC dropout decisions
- Real-time MC dropout pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import math

# Import MC dropout integration
from ..execution.mc_dropout_execution_integration import (
    BinaryExecutionResult, 
    UncertaintyMetrics, 
    SampleStatistics
)

logger = logging.getLogger(__name__)


@dataclass
class MCDropoutFeatures:
    """
    15-dimensional MC dropout features for enhanced critic input
    
    Features extracted from 1000-sample MC dropout analysis:
    0: Mean prediction confidence
    1: Sample variance
    2: Epistemic uncertainty
    3: Aleatoric uncertainty
    4: Total uncertainty
    5: Confidence score
    6: Convergence quality
    7: Decision boundary distance
    8: Samples above threshold ratio
    9: Outlier count ratio
    10: Processing time (normalized)
    11: GPU utilization
    12: MC dropout approval (binary)
    13: MC dropout confidence
    14: Sample agreement score
    """
    
    # Core MC dropout statistics
    mean_prediction: float = 0.5
    sample_variance: float = 0.0
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    total_uncertainty: float = 0.0
    confidence_score: float = 0.5
    convergence_quality: float = 1.0
    decision_boundary_distance: float = 0.0
    
    # Sample analysis
    samples_above_threshold_ratio: float = 0.5
    outlier_count_ratio: float = 0.0
    
    # Performance metrics
    processing_time_normalized: float = 0.5  # Normalized to [0,1]
    gpu_utilization: float = 0.0
    
    # MC dropout decision
    mc_dropout_approval: float = 0.0  # 0 or 1
    mc_dropout_confidence: float = 0.5
    sample_agreement: float = 0.5
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor for neural network processing"""
        features = torch.tensor([
            self.mean_prediction,
            self.sample_variance,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
            self.total_uncertainty,
            self.confidence_score,
            self.convergence_quality,
            self.decision_boundary_distance,
            self.samples_above_threshold_ratio,
            self.outlier_count_ratio,
            self.processing_time_normalized,
            self.gpu_utilization,
            self.mc_dropout_approval,
            self.mc_dropout_confidence,
            self.sample_agreement
        ], dtype=torch.float32)
        
        if device:
            features = features.to(device)
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """MC dropout feature dimension"""
        return 15
    
    @classmethod
    def from_mc_result(cls, mc_result: BinaryExecutionResult, target_latency_us: float = 500.0) -> 'MCDropoutFeatures':
        """Create MC dropout features from binary execution result."""
        stats = mc_result.sample_statistics
        uncertainty = mc_result.uncertainty_metrics
        
        return cls(
            mean_prediction=stats.mean_prediction,
            sample_variance=uncertainty.sample_variance,
            epistemic_uncertainty=uncertainty.epistemic_uncertainty,
            aleatoric_uncertainty=uncertainty.aleatoric_uncertainty,
            total_uncertainty=uncertainty.total_uncertainty,
            confidence_score=uncertainty.confidence_score,
            convergence_quality=uncertainty.convergence_quality,
            decision_boundary_distance=uncertainty.decision_boundary_distance,
            samples_above_threshold_ratio=stats.samples_above_threshold / 1000.0,
            outlier_count_ratio=stats.outlier_count / 1000.0,
            processing_time_normalized=min(mc_result.processing_time_us / target_latency_us, 2.0) / 2.0,
            gpu_utilization=mc_result.gpu_utilization,
            mc_dropout_approval=float(mc_result.execute_trade),
            mc_dropout_confidence=mc_result.confidence,
            sample_agreement=stats.samples_above_threshold / 1000.0 if mc_result.execute_trade else 
                            (1000 - stats.samples_above_threshold) / 1000.0
        )


@dataclass
class EnhancedCombinedStateWithMC:
    """Enhanced combined state with MC dropout features (127D total)"""
    # Base features (112D) - from original system
    base_features: torch.Tensor  # 112D from existing MAPPO system
    
    # MC dropout features (15D) - from 1000-sample analysis
    mc_dropout_features: MCDropoutFeatures
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to combined 127D tensor for enhanced critic input"""
        # MC dropout features (15D)
        mc_tensor = self.mc_dropout_features.to_tensor(device)
        
        # Combine for 127D input
        combined = torch.cat([self.base_features, mc_tensor], dim=0)
        
        if device:
            combined = combined.to(device)
        
        return combined
    
    @property
    def feature_dim(self) -> int:
        """Total feature dimension"""
        return 112 + 15  # Base + MC dropout features


class MCDropoutAttention(nn.Module):
    """
    Specialized attention mechanism for MC dropout features
    
    Focuses on the most relevant MC dropout patterns and their interactions
    with base MAPPO features for improved value estimation.
    """
    
    def __init__(self, 
                 base_dim: int = 112,
                 mc_dropout_dim: int = 15,
                 num_heads: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.base_dim = base_dim
        self.mc_dropout_dim = mc_dropout_dim
        self.num_heads = num_heads
        self.head_dim = mc_dropout_dim // num_heads
        
        # Ensure even division for multi-head attention
        while mc_dropout_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = mc_dropout_dim // self.num_heads
        
        # Query, Key, Value projections for MC dropout features
        self.q_proj = nn.Linear(mc_dropout_dim, mc_dropout_dim)
        self.k_proj = nn.Linear(mc_dropout_dim, mc_dropout_dim)
        self.v_proj = nn.Linear(mc_dropout_dim, mc_dropout_dim)
        
        # Cross-attention with base MAPPO features
        self.cross_k_proj = nn.Linear(base_dim, mc_dropout_dim)
        self.cross_v_proj = nn.Linear(base_dim, mc_dropout_dim)
        
        # Output projection
        self.out_proj = nn.Linear(mc_dropout_dim, mc_dropout_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(mc_dropout_dim)
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MC dropout attention
        
        Args:
            combined_input: Combined tensor [batch_size, 127]
            
        Returns:
            Enhanced MC dropout features [batch_size, 15]
        """
        batch_size = combined_input.size(0)
        
        # Split input into base and MC dropout features
        base_features = combined_input[:, :self.base_dim]          # [batch_size, 112]
        mc_dropout_features = combined_input[:, self.base_dim:]    # [batch_size, 15]
        
        # Self-attention over MC dropout features
        q = self.q_proj(mc_dropout_features)  # [batch_size, 15]
        k = self.k_proj(mc_dropout_features)  # [batch_size, 15]
        v = self.v_proj(mc_dropout_features)  # [batch_size, 15]
        
        # Cross-attention with base MAPPO features
        cross_k = self.cross_k_proj(base_features)  # [batch_size, 15]
        cross_v = self.cross_v_proj(base_features)  # [batch_size, 15]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.mc_dropout_dim)
        cross_scores = torch.matmul(q, cross_k.transpose(-2, -1)) / math.sqrt(self.mc_dropout_dim)
        
        # Combine attention scores (weight cross-attention higher for learning)
        combined_scores = attention_scores + 0.7 * cross_scores
        
        # Apply softmax
        attention_weights = F.softmax(combined_scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        attended_self = torch.matmul(attention_weights, v)
        attended_cross = torch.matmul(attention_weights, cross_v)
        
        # Combine attended features (emphasis on cross-attention for learning)
        attended_features = attended_self + 0.5 * attended_cross
        
        # Output projection
        output = self.out_proj(attended_features)
        
        # Residual connection and normalization
        output = self.norm(output + mc_dropout_features)
        output = self.dropout(output)
        
        return output


class MCDropoutLearningLayer(nn.Module):
    """
    Learning layer that helps MAPPO understand MC dropout patterns
    
    This layer learns to predict MC dropout decisions and outcomes,
    enabling the critic to provide better value estimates.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Predict MC dropout approval probability
        self.approval_predictor = nn.Linear(hidden_dim, 1)
        
        # Predict MC dropout confidence
        self.confidence_predictor = nn.Linear(hidden_dim, 1)
        
        # Predict execution success probability
        self.success_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self, enhanced_mc_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for MC dropout learning."""
        pattern_features = self.pattern_analyzer(enhanced_mc_features)
        
        # Predictions for learning
        approval_prob = torch.sigmoid(self.approval_predictor(pattern_features))
        confidence_pred = torch.sigmoid(self.confidence_predictor(pattern_features))
        success_prob = torch.sigmoid(self.success_predictor(pattern_features))
        
        return {
            'mc_approval_prediction': approval_prob,
            'mc_confidence_prediction': confidence_pred,
            'execution_success_prediction': success_prob,
            'pattern_features': pattern_features
        }


class EnhancedCentralizedCriticWithMC(nn.Module):
    """
    Enhanced Centralized Critic with MC Dropout Integration (127D → 1D)
    
    Architecture:
    - Input: 127D (112D base + 15D MC dropout)
    - MC dropout attention layer
    - MC dropout learning layer
    - Enhanced value estimation with uncertainty
    - Feedback mechanism for MAPPO improvement
    """
    
    def __init__(self,
                 base_input_dim: int = 112,
                 mc_dropout_dim: int = 15,
                 hidden_dims: List[int] = None,
                 num_attention_heads: int = 3,
                 dropout_rate: float = 0.1,
                 use_uncertainty: bool = True,
                 learning_rate: float = 3e-4):
        super().__init__()
        
        self.base_input_dim = base_input_dim
        self.mc_dropout_dim = mc_dropout_dim
        self.total_input_dim = base_input_dim + mc_dropout_dim  # 127D
        self.use_uncertainty = use_uncertainty
        
        if hidden_dims is None:
            # Optimized architecture for 127D input
            hidden_dims = [512, 256, 128, 64]
        
        self.hidden_dims = hidden_dims
        
        # MC dropout attention mechanism
        self.mc_dropout_attention = MCDropoutAttention(
            base_dim=base_input_dim,
            mc_dropout_dim=mc_dropout_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # MC dropout learning layer
        self.mc_learning_layer = MCDropoutLearningLayer(
            input_dim=mc_dropout_dim,
            hidden_dim=64
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
        
        # Value output head
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        # Uncertainty estimation (if enabled)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )
        
        # Backward compatibility layer for 112D inputs
        self.compatibility_layer = nn.Linear(base_input_dim, self.total_input_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Performance tracking
        self.evaluations = 0
        self.mc_feedback_history = []
        self.learning_metrics = {
            'mc_approval_accuracy': 0.0,
            'mc_confidence_mae': 0.0,
            'execution_success_accuracy': 0.0,
            'total_feedback_samples': 0
        }
        
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
    
    def forward(self, combined_state: torch.Tensor, return_learning_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through enhanced critic with MC dropout integration
        
        Args:
            combined_state: Combined state tensor [batch_size, 127] or [batch_size, 112]
            return_learning_outputs: Whether to return MC dropout learning outputs
            
        Returns:
            Value estimate [batch_size, 1] or (value, learning_outputs, uncertainty) if requested
        """
        batch_size = combined_state.size(0)
        input_dim = combined_state.size(1)
        
        # Handle backward compatibility
        if input_dim == self.base_input_dim:
            # Expand 112D to 127D using compatibility layer (zeros for MC dropout features)
            expanded_input = self.compatibility_layer(combined_state)
        elif input_dim != self.total_input_dim:
            raise ValueError(f"Expected input dimension {self.total_input_dim} or {self.base_input_dim}, got {input_dim}")
        else:
            expanded_input = combined_state
        
        # Apply MC dropout attention
        enhanced_mc_features = self.mc_dropout_attention(expanded_input)
        
        # MC dropout learning outputs
        learning_outputs = self.mc_learning_layer(enhanced_mc_features)
        
        # Replace original MC dropout features with enhanced ones
        enhanced_input = torch.cat([
            expanded_input[:, :self.base_input_dim],  # Base features (112D)
            enhanced_mc_features  # Enhanced MC dropout features (15D)
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
        
        # Value estimation
        value_estimate = self.value_head(hidden_output)
        
        # Uncertainty estimation (if enabled)
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(hidden_output)
        
        self.evaluations += 1
        
        if return_learning_outputs:
            return value_estimate, learning_outputs, uncertainty
        else:
            if uncertainty is not None:
                return value_estimate, uncertainty
            else:
                return value_estimate
    
    def update_mc_feedback(self, 
                          mc_result: BinaryExecutionResult, 
                          actual_outcome: Dict[str, Any]):
        """
        Update critic with MC dropout feedback for learning
        
        Args:
            mc_result: MC dropout execution result
            actual_outcome: Actual execution outcome (success, PnL, etc.)
        """
        feedback_sample = {
            'timestamp': datetime.now(),
            'mc_approved': mc_result.execute_trade,
            'mc_confidence': mc_result.confidence,
            'actual_success': actual_outcome.get('success', False),
            'actual_pnl': actual_outcome.get('pnl', 0.0),
            'uncertainty_metrics': mc_result.uncertainty_metrics,
            'processing_time': mc_result.processing_time_us
        }
        
        self.mc_feedback_history.append(feedback_sample)
        
        # Keep limited history
        if len(self.mc_feedback_history) > 1000:
            self.mc_feedback_history = self.mc_feedback_history[-500:]
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def _update_learning_metrics(self):
        """Update learning metrics based on MC dropout feedback."""
        if len(self.mc_feedback_history) < 10:
            return
        
        recent_samples = self.mc_feedback_history[-100:]
        
        # MC approval accuracy
        correct_approvals = sum(
            1 for sample in recent_samples 
            if sample['mc_approved'] == sample['actual_success']
        )
        self.learning_metrics['mc_approval_accuracy'] = correct_approvals / len(recent_samples)
        
        # MC confidence MAE
        confidence_errors = [
            abs(sample['mc_confidence'] - (1.0 if sample['actual_success'] else 0.0))
            for sample in recent_samples
        ]
        self.learning_metrics['mc_confidence_mae'] = sum(confidence_errors) / len(confidence_errors)
        
        # Update total samples
        self.learning_metrics['total_feedback_samples'] = len(self.mc_feedback_history)
    
    def get_mc_learning_metrics(self) -> Dict[str, Any]:
        """Get MC dropout learning metrics."""
        metrics = self.learning_metrics.copy()
        
        # Add attention analysis
        if self.mc_dropout_attention.attention_weights is not None:
            attention_weights = self.mc_dropout_attention.attention_weights.cpu().numpy()
            metrics['attention_entropy'] = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1).mean()
            metrics['attention_concentration'] = np.max(attention_weights, axis=-1).mean()
        
        return metrics
    
    def compute_mc_learning_loss(self, 
                                learning_outputs: Dict[str, torch.Tensor],
                                mc_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute learning loss for MC dropout pattern recognition
        
        Args:
            learning_outputs: Outputs from MC learning layer
            mc_targets: Target values for MC dropout predictions
            
        Returns:
            Combined learning loss
        """
        losses = []
        
        # MC approval prediction loss
        if 'mc_approval_target' in mc_targets:
            approval_loss = F.binary_cross_entropy(
                learning_outputs['mc_approval_prediction'],
                mc_targets['mc_approval_target']
            )
            losses.append(approval_loss)
        
        # MC confidence prediction loss
        if 'mc_confidence_target' in mc_targets:
            confidence_loss = F.mse_loss(
                learning_outputs['mc_confidence_prediction'],
                mc_targets['mc_confidence_target']
            )
            losses.append(confidence_loss)
        
        # Execution success prediction loss
        if 'execution_success_target' in mc_targets:
            success_loss = F.binary_cross_entropy(
                learning_outputs['execution_success_prediction'],
                mc_targets['execution_success_target']
            )
            losses.append(success_loss)
        
        # Combine losses
        if losses:
            return sum(losses) / len(losses)
        else:
            return torch.tensor(0.0, device=learning_outputs['mc_approval_prediction'].device)


# Factory function
def create_enhanced_critic_with_mc(config: Dict[str, Any]) -> EnhancedCentralizedCriticWithMC:
    """Create enhanced centralized critic with MC dropout integration"""
    return EnhancedCentralizedCriticWithMC(
        base_input_dim=config.get('base_input_dim', 112),
        mc_dropout_dim=config.get('mc_dropout_dim', 15),
        hidden_dims=config.get('hidden_dims', [512, 256, 128, 64]),
        num_attention_heads=config.get('num_attention_heads', 3),
        dropout_rate=config.get('dropout_rate', 0.1),
        use_uncertainty=config.get('use_uncertainty', True),
        learning_rate=config.get('learning_rate', 3e-4)
    )


# Global instance for integration
_global_enhanced_critic = None

def get_enhanced_critic_with_mc() -> EnhancedCentralizedCriticWithMC:
    """Get the global enhanced critic instance."""
    global _global_enhanced_critic
    if _global_enhanced_critic is None:
        config = {
            'base_input_dim': 112,
            'mc_dropout_dim': 15,
            'hidden_dims': [512, 256, 128, 64],
            'use_uncertainty': True
        }
        _global_enhanced_critic = create_enhanced_critic_with_mc(config)
    return _global_enhanced_critic