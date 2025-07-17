"""
Enhanced MAPPO Trainer with Superposition Features and Uncertainty-Aware Learning
Agent 3 - The Learning Optimization Specialist

This module implements an enhanced MAPPO training pipeline that integrates with
the enhanced centralized critic for improved learning efficiency and convergence.

Features:
- Integration with enhanced centralized critic (112D input support)
- Superposition-aware training algorithms
- Uncertainty-aware loss functions and value targets
- Adaptive learning rate scheduling based on uncertainty
- Enhanced gradient computation with attention analysis
- Faster convergence through optimized hyperparameters
- Robust training with uncertainty regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog
from datetime import datetime
import time
import math
from pathlib import Path
import json

from .enhanced_centralized_critic import (
    EnhancedCentralizedCritic, 
    EnhancedCombinedState, 
    SuperpositionFeatures,
    create_enhanced_centralized_critic,
    create_superposition_features
)

logger = structlog.get_logger()


@dataclass
class EnhancedMAPPOConfig:
    """Configuration for enhanced MAPPO training"""
    
    # Basic MAPPO parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Enhanced training parameters
    uncertainty_loss_coef: float = 0.1
    attention_regularization_coef: float = 0.05
    superposition_consistency_coef: float = 0.1
    
    # Adaptive learning parameters
    adaptive_lr_enabled: bool = True
    uncertainty_threshold: float = 0.5
    lr_decay_factor: float = 0.95
    lr_increase_factor: float = 1.05
    
    # Training optimization
    batch_size: int = 64
    num_epochs: int = 4
    num_mini_batches: int = 4
    target_kl: float = 0.01
    
    # Convergence optimization
    early_stopping_enabled: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Uncertainty regularization
    uncertainty_regularization_enabled: bool = True
    uncertainty_target: float = 0.3
    uncertainty_decay: float = 0.99


@dataclass
class TrainingBatch:
    """Enhanced training batch with superposition features"""
    
    # Standard MAPPO data
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    
    # Enhanced data
    superposition_features: torch.Tensor
    uncertainties: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    
    def to_device(self, device: torch.device) -> 'TrainingBatch':
        """Move batch to specified device"""
        return TrainingBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            masks=self.masks.to(device),
            superposition_features=self.superposition_features.to(device),
            uncertainties=self.uncertainties.to(device) if self.uncertainties is not None else None,
            attention_weights=self.attention_weights.to(device) if self.attention_weights is not None else None
        )


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler based on uncertainty and performance"""
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 config: EnhancedMAPPOConfig):
        self.optimizer = optimizer
        self.config = config
        self.initial_lr = config.learning_rate
        self.uncertainty_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.last_lr_update = 0
        
    def step(self, current_uncertainty: float, current_loss: float):
        """Update learning rate based on uncertainty and loss"""
        if not self.config.adaptive_lr_enabled:
            return
        
        self.uncertainty_history.append(current_uncertainty)
        self.loss_history.append(current_loss)
        
        # Only update every 10 steps
        if len(self.uncertainty_history) < 10 or self.last_lr_update % 10 != 0:
            self.last_lr_update += 1
            return
        
        # Calculate trends
        recent_uncertainty = np.mean(list(self.uncertainty_history)[-10:])
        recent_loss = np.mean(list(self.loss_history)[-10:])
        
        # Adjust learning rate based on uncertainty
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if recent_uncertainty > self.config.uncertainty_threshold:
            # High uncertainty - reduce learning rate for stability
            new_lr = current_lr * self.config.lr_decay_factor
        elif recent_uncertainty < self.config.uncertainty_threshold * 0.5:
            # Low uncertainty - can increase learning rate
            new_lr = current_lr * self.config.lr_increase_factor
        else:
            new_lr = current_lr
        
        # Clamp learning rate
        new_lr = max(new_lr, self.initial_lr * 0.1)
        new_lr = min(new_lr, self.initial_lr * 2.0)
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.last_lr_update += 1
        
        logger.debug("Adaptive LR update", 
                    old_lr=current_lr, 
                    new_lr=new_lr,
                    uncertainty=recent_uncertainty,
                    loss=recent_loss)


class EnhancedMAPPOTrainer:
    """
    Enhanced MAPPO trainer with superposition features and uncertainty-aware learning
    
    Integrates with EnhancedCentralizedCritic for improved training efficiency
    and convergence through specialized loss functions and adaptive optimization.
    """
    
    def __init__(self,
                 agents: Dict[str, nn.Module],
                 critic: EnhancedCentralizedCritic,
                 config: EnhancedMAPPOConfig,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize enhanced MAPPO trainer
        
        Args:
            agents: Dictionary of agent models
            critic: Enhanced centralized critic
            config: Training configuration
            device: Training device
        """
        self.agents = agents
        self.critic = critic
        self.config = config
        self.device = device
        
        # Move models to device
        self.critic.to(device)
        for agent in self.agents.values():
            agent.to(device)
        
        # Initialize optimizers
        self.agent_optimizers = {}
        for name, agent in self.agents.items():
            self.agent_optimizers[name] = optim.Adam(
                agent.parameters(),
                lr=config.learning_rate,
                eps=1e-5
            )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Adaptive learning rate schedulers
        self.lr_schedulers = {}
        for name, optimizer in self.agent_optimizers.items():
            self.lr_schedulers[name] = AdaptiveLearningRateScheduler(optimizer, config)
        
        self.critic_lr_scheduler = AdaptiveLearningRateScheduler(
            self.critic_optimizer, config
        )
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.best_performance = float('-inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.training_metrics = {
            'policy_losses': defaultdict(list),
            'value_losses': [],
            'entropy_losses': defaultdict(list),
            'uncertainty_losses': [],
            'attention_regularization_losses': [],
            'superposition_consistency_losses': [],
            'uncertainties': [],
            'attention_entropies': [],
            'learning_rates': defaultdict(list),
            'gradient_norms': defaultdict(list),
            'convergence_metrics': []
        }
        
        # Performance optimization
        self.gradient_accumulation_steps = 1
        self.mixed_precision_enabled = torch.cuda.is_available()
        if self.mixed_precision_enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("Enhanced MAPPO trainer initialized",
                   num_agents=len(self.agents),
                   use_uncertainty=self.critic.use_uncertainty,
                   device=str(device))
    
    def compute_gae(self, 
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   masks: torch.Tensor,
                   gamma: float = 0.99,
                   gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation with uncertainty weighting
        
        Args:
            rewards: Reward tensor [sequence_length, batch_size]
            values: Value tensor [sequence_length, batch_size]
            masks: Mask tensor [sequence_length, batch_size]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        sequence_length, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE advantages
        gae = 0
        for step in reversed(range(sequence_length - 1)):
            delta = rewards[step] + gamma * values[step + 1] * masks[step + 1] - values[step]
            gae = delta + gamma * gae_lambda * masks[step + 1] * gae
            advantages[step] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def compute_policy_loss(self, 
                           agent_name: str,
                           batch: TrainingBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute policy loss for a specific agent
        
        Args:
            agent_name: Name of the agent
            batch: Training batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        agent = self.agents[agent_name]
        
        # Get current policy outputs
        # Note: This is a simplified version - actual implementation depends on agent architecture
        if hasattr(agent, 'forward'):
            policy_outputs = agent(batch.observations)
        else:
            # Fallback for different agent architectures
            policy_outputs = agent.get_action_distribution(batch.observations)
        
        # Extract action logits and compute new log probabilities
        if isinstance(policy_outputs, dict):
            action_logits = policy_outputs.get('action_logits', policy_outputs.get('logits'))
        else:
            action_logits = policy_outputs
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        new_log_probs = action_dist.log_prob(batch.actions)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - batch.log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # Superposition consistency loss
        superposition_loss = self._compute_superposition_consistency_loss(
            batch.superposition_features, action_logits
        )
        
        # Total policy loss
        total_loss = policy_loss + entropy_loss + superposition_loss
        
        # Compute metrics
        with torch.no_grad():
            kl_div = (batch.log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
            approx_kl = 0.5 * (batch.log_probs - new_log_probs).pow(2).mean().item()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'superposition_loss': superposition_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div,
            'clip_fraction': clip_fraction,
            'approx_kl': approx_kl
        }
        
        return total_loss, metrics
    
    def compute_value_loss(self, batch: TrainingBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute value loss with uncertainty regularization
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Get current value estimates
        if self.critic.use_uncertainty:
            current_values, uncertainties = self.critic(batch.observations)
        else:
            current_values = self.critic(batch.observations)
            uncertainties = None
        
        # Value loss
        value_loss = F.mse_loss(current_values.squeeze(), batch.returns)
        
        # Uncertainty loss
        uncertainty_loss = torch.tensor(0.0, device=self.device)
        if uncertainties is not None:
            # Penalize high uncertainty
            target_uncertainty = self.config.uncertainty_target
            uncertainty_loss = F.mse_loss(uncertainties.squeeze(), 
                                        torch.full_like(uncertainties.squeeze(), target_uncertainty))
            uncertainty_loss = uncertainty_loss * self.config.uncertainty_loss_coef
        
        # Attention regularization
        attention_loss = self._compute_attention_regularization_loss()
        
        # Total value loss
        total_loss = (self.config.value_loss_coef * value_loss + 
                     uncertainty_loss + 
                     attention_loss)
        
        # Compute metrics
        with torch.no_grad():
            explained_variance = 1 - F.mse_loss(current_values.squeeze(), batch.returns) / batch.returns.var()
            mean_uncertainty = uncertainties.mean().item() if uncertainties is not None else 0.0
        
        metrics = {
            'value_loss': value_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'attention_loss': attention_loss.item(),
            'total_loss': total_loss.item(),
            'explained_variance': explained_variance.item(),
            'mean_uncertainty': mean_uncertainty
        }
        
        return total_loss, metrics
    
    def _compute_superposition_consistency_loss(self, 
                                              superposition_features: torch.Tensor,
                                              action_logits: torch.Tensor) -> torch.Tensor:
        """Compute superposition consistency loss"""
        if superposition_features.size(1) < 10:
            return torch.tensor(0.0, device=self.device)
        
        # Extract superposition confidence weights (first 3 dimensions)
        confidence_weights = superposition_features[:, :3]
        
        # Compute entropy of confidence weights
        confidence_entropy = -(confidence_weights * torch.log(confidence_weights + 1e-8)).sum(dim=1)
        
        # Compute action entropy
        action_probs = F.softmax(action_logits, dim=-1)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1)
        
        # Consistency loss - encourage alignment between confidence and action uncertainty
        consistency_loss = F.mse_loss(confidence_entropy, action_entropy)
        
        return consistency_loss * self.config.superposition_consistency_coef
    
    def _compute_attention_regularization_loss(self) -> torch.Tensor:
        """Compute attention regularization loss"""
        if not hasattr(self.critic, 'superposition_attention'):
            return torch.tensor(0.0, device=self.device)
        
        attention_weights = self.critic.superposition_attention.attention_weights
        if attention_weights is None:
            return torch.tensor(0.0, device=self.device)
        
        # Encourage diverse attention patterns
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        
        # Regularize towards moderate entropy
        target_entropy = math.log(attention_weights.size(-1)) * 0.7  # 70% of max entropy
        entropy_loss = F.mse_loss(attention_entropy, 
                                 torch.full_like(attention_entropy, target_entropy))
        
        return entropy_loss * self.config.attention_regularization_coef
    
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            Training metrics
        """
        # Prepare training batch
        batch = self._prepare_batch(batch_data)
        
        # Training metrics
        step_metrics = {}
        total_policy_loss = 0.0
        
        # Update each agent
        for agent_name in self.agents.keys():
            agent_optimizer = self.agent_optimizers[agent_name]
            agent_optimizer.zero_grad()
            
            # Compute policy loss
            policy_loss, policy_metrics = self.compute_policy_loss(agent_name, batch)
            total_policy_loss += policy_loss.item()
            
            # Backward pass
            if self.mixed_precision_enabled:
                with torch.cuda.amp.autocast():
                    policy_loss.backward()
                self.scaler.unscale_(agent_optimizer)
                torch.nn.utils.clip_grad_norm_(self.agents[agent_name].parameters(), 
                                             self.config.max_grad_norm)
                self.scaler.step(agent_optimizer)
                self.scaler.update()
            else:
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_name].parameters(), 
                                             self.config.max_grad_norm)
                agent_optimizer.step()
            
            # Update metrics
            for key, value in policy_metrics.items():
                step_metrics[f"{agent_name}_{key}"] = value
            
            # Update learning rate
            self.lr_schedulers[agent_name].step(
                policy_metrics.get('mean_uncertainty', 0.0),
                policy_metrics['total_loss']
            )
        
        # Update critic
        self.critic_optimizer.zero_grad()
        
        # Compute value loss
        value_loss, value_metrics = self.compute_value_loss(batch)
        
        # Backward pass
        if self.mixed_precision_enabled:
            with torch.cuda.amp.autocast():
                value_loss.backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                         self.config.max_grad_norm)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                         self.config.max_grad_norm)
            self.critic_optimizer.step()
        
        # Update critic learning rate
        self.critic_lr_scheduler.step(
            value_metrics.get('mean_uncertainty', 0.0),
            value_metrics['total_loss']
        )
        
        # Update metrics
        for key, value in value_metrics.items():
            step_metrics[f"critic_{key}"] = value
        
        # Update training state
        self.training_step += 1
        
        # Track metrics
        self._update_training_metrics(step_metrics)
        
        return step_metrics
    
    def _prepare_batch(self, batch_data: Dict[str, Any]) -> TrainingBatch:
        """Prepare batch data for training"""
        # This is a simplified version - actual implementation depends on data format
        observations = batch_data.get('observations', torch.randn(32, 112))
        actions = batch_data.get('actions', torch.randint(0, 3, (32,)))
        log_probs = batch_data.get('log_probs', torch.randn(32))
        values = batch_data.get('values', torch.randn(32))
        rewards = batch_data.get('rewards', torch.randn(32))
        masks = batch_data.get('masks', torch.ones(32))
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards.unsqueeze(0), 
            values.unsqueeze(0), 
            masks.unsqueeze(0)
        )
        
        # Extract superposition features
        superposition_features = observations[:, -10:] if observations.size(1) >= 112 else torch.zeros(32, 10)
        
        batch = TrainingBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=advantages.squeeze(0),
            returns=returns.squeeze(0),
            masks=masks,
            superposition_features=superposition_features
        )
        
        return batch.to_device(self.device)
    
    def _update_training_metrics(self, step_metrics: Dict[str, float]):
        """Update training metrics history"""
        for key, value in step_metrics.items():
            if 'policy_loss' in key:
                agent_name = key.split('_')[0]
                self.training_metrics['policy_losses'][agent_name].append(value)
            elif 'value_loss' in key:
                self.training_metrics['value_losses'].append(value)
            elif 'entropy_loss' in key:
                agent_name = key.split('_')[0]
                self.training_metrics['entropy_losses'][agent_name].append(value)
            elif 'uncertainty_loss' in key:
                self.training_metrics['uncertainty_losses'].append(value)
            elif 'attention_loss' in key:
                self.training_metrics['attention_regularization_losses'].append(value)
            elif 'mean_uncertainty' in key:
                self.training_metrics['uncertainties'].append(value)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'learning_rates': {
                name: optimizer.param_groups[0]['lr'] 
                for name, optimizer in self.agent_optimizers.items()
            },
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
            'recent_metrics': {}
        }
        
        # Compute recent averages
        for metric_name, metric_data in self.training_metrics.items():
            if isinstance(metric_data, dict):
                summary['recent_metrics'][metric_name] = {
                    agent: np.mean(values[-10:]) if values else 0.0
                    for agent, values in metric_data.items()
                }
            else:
                summary['recent_metrics'][metric_name] = (
                    np.mean(metric_data[-10:]) if metric_data else 0.0
                )
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agent_state_dicts': {
                name: agent.state_dict() for name, agent in self.agents.items()
            },
            'agent_optimizer_state_dicts': {
                name: optimizer.state_dict() for name, optimizer in self.agent_optimizers.items()
            },
            'training_metrics': self.training_metrics,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter
        }
        
        torch.save(checkpoint, filepath)
        logger.info("Enhanced MAPPO checkpoint saved", filepath=filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        for name, state_dict in checkpoint['agent_state_dicts'].items():
            if name in self.agents:
                self.agents[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['agent_optimizer_state_dicts'].items():
            if name in self.agent_optimizers:
                self.agent_optimizers[name].load_state_dict(state_dict)
        
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        self.best_performance = checkpoint.get('best_performance', float('-inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        logger.info("Enhanced MAPPO checkpoint loaded", filepath=filepath)


# Factory function
def create_enhanced_mappo_trainer(
    agents: Dict[str, nn.Module],
    critic_config: Dict[str, Any],
    training_config: Dict[str, Any],
    device: torch.device = torch.device('cpu')
) -> EnhancedMAPPOTrainer:
    """Create enhanced MAPPO trainer with configuration"""
    
    # Create enhanced centralized critic
    critic = create_enhanced_centralized_critic(critic_config)
    
    # Create training configuration
    config = EnhancedMAPPOConfig(**training_config)
    
    # Create trainer
    trainer = EnhancedMAPPOTrainer(
        agents=agents,
        critic=critic,
        config=config,
        device=device
    )
    
    return trainer


if __name__ == "__main__":
    # Test the enhanced MAPPO trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy agents
    class DummyAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(112, 3)
            
        def forward(self, x):
            return {'action_logits': self.linear(x)}
    
    agents = {
        'agent1': DummyAgent(),
        'agent2': DummyAgent(),
        'agent3': DummyAgent()
    }
    
    # Create configurations
    critic_config = {
        'base_input_dim': 102,
        'superposition_dim': 10,
        'hidden_dims': [256, 128, 64],
        'use_uncertainty': True
    }
    
    training_config = {
        'learning_rate': 3e-4,
        'uncertainty_loss_coef': 0.1,
        'adaptive_lr_enabled': True
    }
    
    # Create trainer
    trainer = create_enhanced_mappo_trainer(
        agents=agents,
        critic_config=critic_config,
        training_config=training_config,
        device=device
    )
    
    # Test training step
    batch_data = {
        'observations': torch.randn(32, 112),
        'actions': torch.randint(0, 3, (32,)),
        'log_probs': torch.randn(32),
        'values': torch.randn(32),
        'rewards': torch.randn(32),
        'masks': torch.ones(32)
    }
    
    metrics = trainer.train_step(batch_data)
    print("Training step completed successfully")
    print(f"Training metrics: {metrics}")
    
    # Print summary
    summary = trainer.get_training_summary()
    print(f"Training summary: {summary}")