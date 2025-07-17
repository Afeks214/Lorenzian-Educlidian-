"""
Intelligent Gating Network for Dynamic Agent Coordination.

This module implements a sophisticated gating network that acts as an intelligent coordinator,
deciding which of the three strategic agents (MLMI, NWRQK, Regime) should be trusted most
for the current market context. It replaces static ensemble weights with dynamic, 
context-aware expert selection.

Key Features:
- Dynamic weight generation based on market context
- Context analysis and interpretation
- Performance tracking and feedback learning
- Confidence-based gating decisions
- Real-time adaptation to market conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime


class GatingNetwork(nn.Module):
    """
    Intelligent Gating Network for Dynamic Agent Coordination.
    
    This network analyzes market context and decides which of the three strategic agents
    (MLMI, NWRQK, Regime) should be trusted most for the current market state.
    
    The gating network transforms the system from fixed expert weighting to intelligent
    expert selection based on real-time market conditions.
    """
    
    def __init__(self, shared_context_dim: int, n_agents: int = 3, hidden_dim: int = 64):
        """
        Initialize the Intelligent Gating Network.
        
        Args:
            shared_context_dim: Dimension of shared market context vector
            n_agents: Number of agents to coordinate (default: 3)
            hidden_dim: Hidden layer dimension for neural networks
        """
        super(GatingNetwork, self).__init__()
        
        self.shared_context_dim = shared_context_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        
        # Ultra-minimal architecture for maximum performance
        self.gate = nn.Linear(shared_context_dim, n_agents)
        
        # Regime-specific bias buffers (non-trainable for performance) 
        # Enhanced for better differentiation across market conditions with balanced specialization
        self.register_buffer('volatility_bias', torch.tensor([-1.0, -1.0, 3.0]))  # Strong Regime bias for volatility
        self.register_buffer('momentum_bias', torch.tensor([4.0, -1.0, -1.0]))  # Stronger MLMI bias for momentum
        self.register_buffer('volume_bias', torch.tensor([-1.0, 2.5, -1.0]))    # Moderate NWRQK bias for volume
        self.register_buffer('trend_bias', torch.tensor([3.0, -1.0, -1.0]))      # Strong MLMI bias for trend strength
        self.register_buffer('mmd_bias', torch.tensor([-1.0, -1.0, 4.0]))       # Strong Regime bias for MMD/regime detection
        
        # Minimal context analysis for maximum performance
        self.context_analyzer = nn.Identity()  # No-op for performance
        
        # Performance tracking for agents
        self.agent_performance_history = torch.zeros(n_agents)
        self.context_history = []
        self.gating_decision_history = []
        
        # Initialize weights with Xavier initialization
        self._initialize_weights()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"GatingNetwork initialized: context_dim={shared_context_dim}, "
            f"n_agents={n_agents}, hidden_dim={hidden_dim}"
        )
        
    def _initialize_weights(self):
        """Initialize network weights for enhanced sensitivity."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use larger initialization for better sensitivity
                nn.init.xavier_uniform_(module.weight, gain=2.0)
                if module.bias is not None:
                    # Initialize with small random bias for asymmetry
                    nn.init.uniform_(module.bias, -0.1, 0.1)
    
    def forward(self, shared_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate dynamic gating weights based on market context.
        
        Args:
            shared_context: Market context tensor with features like volatility, 
                          volume, momentum, regime indicators
            
        Returns:
            Dictionary containing:
            - gating_weights: Dynamic weights for each agent [n_agents]
            - context_features: Analyzed context features
            - confidence: Confidence in gating decision [0, 1]
            - weight_entropy: Entropy of weight distribution (uncertainty measure)
        """
        # Ensure proper tensor shape and handle NaN/inf values
        if shared_context.dim() == 1:
            shared_context = shared_context.unsqueeze(0)
        
        # Clean context data - replace NaN/inf with zeros
        shared_context = torch.where(
            torch.isfinite(shared_context), 
            shared_context, 
            torch.zeros_like(shared_context)
        )
        
        # Ultra-fast processing: direct linear transformation
        base_logits = self.gate(shared_context)
        
        # Add regime-specific biases based on context characteristics
        if shared_context.shape[-1] >= 6:
            volatility = torch.abs(shared_context[:, 0])  # volatility_30
            volume = shared_context[:, 1]  # volume_ratio (not absolute)
            momentum = torch.abs(shared_context[:, 2]) + torch.abs(shared_context[:, 3])  # momentum signals
            mmd_score = torch.abs(shared_context[:, 4])  # mmd_score
            trend_strength = torch.abs(shared_context[:, 5])  # price_trend
            
            # Calculate context-dependent biases with enhanced specialization
            vol_bias = volatility.unsqueeze(-1) * self.volatility_bias
            mom_bias = momentum.unsqueeze(-1) * self.momentum_bias
            vol_bias_scaled = volume.unsqueeze(-1) * self.volume_bias
            trend_bias = trend_strength.unsqueeze(-1) * self.trend_bias
            mmd_bias_scaled = mmd_score.unsqueeze(-1) * self.mmd_bias
            
            # Apply biases to create regime-specific preferences
            enhanced_logits = base_logits + vol_bias + mom_bias + vol_bias_scaled + trend_bias + mmd_bias_scaled
        else:
            enhanced_logits = base_logits
        
        # Generate final gating weights
        gating_weights = torch.softmax(enhanced_logits, dim=-1)
        
        # Minimal context analysis for interpretability (performance optimized)
        context_features = shared_context  # Direct passthrough for performance
        
        # Calculate confidence in gating decision
        # Low entropy = high confidence (concentrated weights)
        # High entropy = low confidence (distributed weights)
        weight_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1)
        confidence = 1.0 - (weight_entropy / np.log(self.n_agents))  # Normalized confidence
        
        # Store decision for history tracking only when training
        if self.training and len(self.gating_decision_history) % 20 == 0:  # Store every 20th decision during training only
            self.gating_decision_history.append({
                'weights': gating_weights.detach().clone(),
                'context': shared_context.detach().clone(),
                'confidence': confidence.detach().clone(),
                'timestamp': datetime.now()
            })
            
            # Limit history size
            if len(self.gating_decision_history) > 50:  # Smaller history for performance
                self.gating_decision_history = self.gating_decision_history[-25:]
        
        return {
            'gating_weights': gating_weights,
            'context_features': context_features,
            'confidence': confidence,
            'weight_entropy': weight_entropy
        }
        
    def update_performance_history(self, agent_idx: int, performance_score: float):
        """
        Update historical performance tracking for agents.
        
        Args:
            agent_idx: Index of the agent (0=MLMI, 1=NWRQK, 2=Regime)
            performance_score: Performance score for the agent [0, 1]
        """
        if 0 <= agent_idx < self.n_agents:
            # Exponential moving average of agent performance
            alpha = 0.1
            self.agent_performance_history[agent_idx] = (
                alpha * performance_score + 
                (1 - alpha) * self.agent_performance_history[agent_idx]
            )
            
            self.logger.debug(
                f"Updated agent {agent_idx} performance: {performance_score:.3f}, "
                f"EMA: {self.agent_performance_history[agent_idx]:.3f}"
            )
        else:
            self.logger.warning(f"Invalid agent index: {agent_idx}")
            
    def get_agent_specialization(self, shared_context: torch.Tensor) -> Dict[str, str]:
        """
        Analyze which agent should specialize in current context.
        
        Args:
            shared_context: Market context tensor
            
        Returns:
            Dictionary mapping agent names to specialization descriptions
        """
        with torch.no_grad():
            context_analysis = self.context_analyzer(shared_context)
            
            # Extract context characteristics
            volatility = shared_context[0].item() if len(shared_context) > 0 else 0.0
            volume_ratio = shared_context[1].item() if len(shared_context) > 1 else 1.0
            momentum = shared_context[2].item() if len(shared_context) > 2 else 0.0
            regime_score = shared_context[5].item() if len(shared_context) > 5 else 0.0
            
            # Generate specialization descriptions
            specialization = {
                'MLMI': f"Market impact analysis (vol: {volatility:.3f}, volume: {volume_ratio:.3f})",
                'NWRQK': f"Risk-quality assessment (vol: {volatility:.3f}, momentum: {momentum:.3f})", 
                'Regime': f"Regime detection (momentum: {momentum:.3f}, regime: {regime_score:.3f})"
            }
            
            return specialization
    
    def get_gating_rationale(self, shared_context: torch.Tensor, gating_weights: torch.Tensor) -> str:
        """
        Generate human-readable rationale for gating decisions.
        
        Args:
            shared_context: Market context that influenced the decision
            gating_weights: The gating weights produced
            
        Returns:
            Human-readable explanation of the gating decision
        """
        agent_names = ['MLMI', 'NWRQK', 'Regime']
        
        # Find dominant agent
        dominant_idx = torch.argmax(gating_weights).item()
        dominant_weight = gating_weights[dominant_idx].item()
        dominant_agent = agent_names[dominant_idx]
        
        # Extract key context features
        volatility = shared_context[0].item() if len(shared_context) > 0 else 0.0
        volume_ratio = shared_context[1].item() if len(shared_context) > 1 else 1.0
        momentum = shared_context[2].item() if len(shared_context) > 2 else 0.0
        
        # Generate rationale based on context
        if volatility > 0.02:  # High volatility
            market_state = "high volatility"
        elif abs(momentum) > 0.01:  # Strong momentum
            market_state = "trending"
        elif volume_ratio > 1.5:  # High volume
            market_state = "high volume"
        else:
            market_state = "normal"
            
        rationale = (
            f"In {market_state} conditions (vol={volatility:.3f}, "
            f"momentum={momentum:.3f}), {dominant_agent} agent "
            f"receives highest weight ({dominant_weight:.3f}) for optimal "
            f"decision coordination."
        )
        
        return rationale
    
    def analyze_weight_distribution(self, gating_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the distribution characteristics of gating weights.
        
        Args:
            gating_weights: Tensor of gating weights
            
        Returns:
            Dictionary with distribution analysis
        """
        weights_np = gating_weights.detach().cpu().numpy().flatten()
        
        return {
            'max_weight': float(np.max(weights_np)),
            'min_weight': float(np.min(weights_np)),
            'weight_std': float(np.std(weights_np)),
            'entropy': float(-np.sum(weights_np * np.log(weights_np + 1e-8))),
            'dominant_agent': int(np.argmax(weights_np)),
            'concentration_ratio': float(np.max(weights_np) / np.mean(weights_np))
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the gating network.
        
        Returns:
            Dictionary containing performance statistics
        """
        if len(self.gating_decision_history) == 0:
            return {'decision_count': 0}
            
        # Calculate recent performance statistics
        recent_decisions = self.gating_decision_history[-100:]  # Last 100 decisions
        
        confidences = [d['confidence'].item() for d in recent_decisions]
        weights_history = [d['weights'].detach().cpu().numpy() for d in recent_decisions]
        
        # Weight stability (variance of weights over time)
        weight_matrix = np.array(weights_history)
        weight_stability = {
            f'agent_{i}_stability': float(1.0 / (1.0 + np.var(weight_matrix[:, i])))
            for i in range(self.n_agents)
        }
        
        return {
            'decision_count': len(self.gating_decision_history),
            'avg_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'agent_performance_ema': self.agent_performance_history.tolist(),
            'weight_stability': weight_stability,
            'recent_avg_weights': np.mean(weight_matrix, axis=0).tolist()
        }
    
    def reset_history(self):
        """Reset decision and performance history."""
        self.gating_decision_history.clear()
        self.agent_performance_history.zero_()
        self.logger.info("Gating network history reset")
        
    def save_checkpoint(self, filepath: str):
        """
        Save gating network checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'performance_history': self.agent_performance_history,
            'config': {
                'shared_context_dim': self.shared_context_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Gating network checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        Load gating network checkpoint.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            self.agent_performance_history = checkpoint['performance_history']
            
            self.logger.info(f"Gating network checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


class GatingNetworkTrainer:
    """
    Training utilities for the Gating Network.
    
    Provides methods to train the gating network based on agent performance
    feedback and market outcomes.
    """
    
    def __init__(self, gating_network: GatingNetwork, learning_rate: float = 1e-4):
        """
        Initialize the gating network trainer.
        
        Args:
            gating_network: The gating network to train
            learning_rate: Learning rate for optimization
        """
        self.gating_network = gating_network
        self.optimizer = torch.optim.Adam(
            gating_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.training_history = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def train_step(
        self, 
        context_batch: torch.Tensor,
        agent_performance_batch: torch.Tensor,
        target_weights: torch.Tensor
    ) -> float:
        """
        Perform a single training step on the gating network.
        
        Args:
            context_batch: Batch of market contexts [batch_size, context_dim]
            agent_performance_batch: Historical agent performance [batch_size, n_agents]
            target_weights: Optimal weights based on performance [batch_size, n_agents]
        
        Returns:
            Training loss value
        """
        self.gating_network.train()
        
        # Forward pass
        gating_result = self.gating_network(context_batch)
        predicted_weights = gating_result['gating_weights']
        
        # Calculate loss (KL divergence between predicted and optimal weights)
        loss = F.kl_div(
            torch.log(predicted_weights + 1e-8),
            target_weights,
            reduction='batchmean'
        )
        
        # Add regularization terms
        l2_reg = sum(p.pow(2).sum() for p in self.gating_network.parameters())
        
        # Entropy regularization to prevent overconfident predictions
        entropy_reg = -torch.mean(torch.sum(predicted_weights * torch.log(predicted_weights + 1e-8), dim=-1))
        
        total_loss = loss + 1e-5 * l2_reg + 0.01 * entropy_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.gating_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track training progress
        self.training_history.append({
            'loss': total_loss.item(),
            'kl_loss': loss.item(),
            'l2_reg': l2_reg.item(),
            'entropy_reg': entropy_reg.item(),
            'timestamp': datetime.now()
        })
        
        self.logger.debug(f"Training step - Total loss: {total_loss.item():.6f}")
        
        return total_loss.item()
    
    def generate_training_targets(
        self,
        agent_performance_scores: np.ndarray,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        Generate target weights based on agent performance scores.
        
        Args:
            agent_performance_scores: Performance scores for each agent [n_agents]
            temperature: Softmax temperature for target generation
        
        Returns:
            Target weight distribution
        """
        # Apply softmax with temperature to performance scores
        scores_tensor = torch.FloatTensor(agent_performance_scores) / temperature
        target_weights = F.softmax(scores_tensor, dim=-1)
        
        return target_weights
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        if not self.training_history:
            return {'steps': 0}
            
        recent_history = self.training_history[-100:]
        
        return {
            'total_steps': len(self.training_history),
            'recent_avg_loss': float(np.mean([h['loss'] for h in recent_history])),
            'recent_avg_kl_loss': float(np.mean([h['kl_loss'] for h in recent_history])),
            'loss_trend': 'improving' if len(recent_history) > 10 and 
                         recent_history[-1]['loss'] < recent_history[-10]['loss'] else 'stable'
        }