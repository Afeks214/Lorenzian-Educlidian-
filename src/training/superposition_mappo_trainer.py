"""
Superposition-Aware MAPPO Trainer for Sequential Agent Coordination

This module implements a specialized MAPPO trainer that can handle superposition outputs
from agents, enabling training systems where agents produce multiple probabilistic
decisions simultaneously while maintaining sequential coordination.

Key Features:
- Superposition-aware policy networks that output multiple decision states
- Sequential coordination training with proper temporal dependencies
- Confidence-weighted superposition loss functions
- Entropy regularization for maintaining superposition diversity
- Consistency penalties for conflicting superposition states
- Cascade training support for progressive agent introduction

Author: AGENT 9 - Superposition-Aware Training Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from pathlib import Path
import json
import time
import math
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class SuperpositionConfig:
    """Configuration for superposition-aware training"""
    # Superposition parameters
    superposition_dim: int = 8  # Number of simultaneous decision states
    confidence_threshold: float = 0.7  # Minimum confidence for decision execution
    entropy_weight: float = 0.1  # Weight for entropy regularization
    consistency_weight: float = 0.2  # Weight for consistency penalties
    
    # Sequential coordination
    sequence_length: int = 4  # Number of agents in sequence
    temporal_discount: float = 0.95  # Discount for temporal dependencies
    coordination_bonus: float = 0.1  # Bonus for good coordination
    
    # Training parameters
    superposition_lr: float = 3e-4  # Learning rate for superposition networks
    confidence_lr: float = 1e-4  # Learning rate for confidence networks
    warmup_episodes: int = 100  # Episodes before full superposition training
    
    # Quality metrics
    min_entropy: float = 0.5  # Minimum entropy for healthy superposition
    max_confidence_variance: float = 0.3  # Maximum variance in confidence
    consistency_tolerance: float = 0.1  # Tolerance for consistency violations


@dataclass
class SuperpositionOutput:
    """Container for superposition network outputs"""
    decision_states: torch.Tensor  # [batch_size, superposition_dim, action_dim]
    confidence_weights: torch.Tensor  # [batch_size, superposition_dim]
    entropy: torch.Tensor  # [batch_size]
    consistency_score: torch.Tensor  # [batch_size]
    
    def collapse_to_action(self, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collapse superposition to single action"""
        # Weight decision states by confidence
        weighted_logits = (self.decision_states * self.confidence_weights.unsqueeze(-1)).sum(dim=1)
        
        # Apply temperature scaling
        scaled_logits = weighted_logits / temperature
        
        # Sample action
        action_dist = Categorical(logits=scaled_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob


class SuperpositionPolicyNetwork(nn.Module):
    """Policy network that outputs superposition states"""
    
    def __init__(self, obs_dim: int, action_dim: int, superposition_dim: int,
                 hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.superposition_dim = superposition_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Superposition decision heads
        self.decision_heads = nn.ModuleList([
            nn.Linear(prev_dim, action_dim) for _ in range(superposition_dim)
        ])
        
        # Confidence network
        self.confidence_network = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, superposition_dim),
            nn.Softmax(dim=-1)
        )
        
        # Consistency network
        self.consistency_network = nn.Sequential(
            nn.Linear(prev_dim + superposition_dim * action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, obs: torch.Tensor) -> SuperpositionOutput:
        """Forward pass through superposition network"""
        batch_size = obs.size(0)
        
        # Extract features
        features = self.feature_extractor(obs)
        
        # Generate decision states
        decision_states = []
        for head in self.decision_heads:
            decision_states.append(head(features))
        
        decision_states = torch.stack(decision_states, dim=1)  # [batch, superposition_dim, action_dim]
        
        # Generate confidence weights
        confidence_weights = self.confidence_network(features)
        
        # Calculate entropy
        entropy = self._calculate_entropy(decision_states, confidence_weights)
        
        # Calculate consistency score
        consistency_input = torch.cat([
            features,
            decision_states.view(batch_size, -1)
        ], dim=1)
        consistency_score = self.consistency_network(consistency_input).squeeze(-1)
        
        return SuperpositionOutput(
            decision_states=decision_states,
            confidence_weights=confidence_weights,
            entropy=entropy,
            consistency_score=consistency_score
        )
    
    def _calculate_entropy(self, decision_states: torch.Tensor, 
                          confidence_weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of superposition states"""
        # Entropy of each decision state
        decision_entropy = []
        for i in range(decision_states.size(1)):
            probs = F.softmax(decision_states[:, i], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            decision_entropy.append(entropy)
        
        decision_entropy = torch.stack(decision_entropy, dim=1)  # [batch, superposition_dim]
        
        # Weighted entropy
        weighted_entropy = (decision_entropy * confidence_weights).sum(dim=1)
        
        # Confidence entropy
        confidence_entropy = -(confidence_weights * torch.log(confidence_weights + 1e-8)).sum(dim=1)
        
        return weighted_entropy + 0.1 * confidence_entropy


class SuperpositionValueNetwork(nn.Module):
    """Value network for superposition states"""
    
    def __init__(self, obs_dim: int, superposition_dim: int,
                 hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.superposition_dim = superposition_dim
        
        # Value network
        layers = []
        prev_dim = obs_dim + superposition_dim  # obs + confidence weights
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.value_network = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor, confidence_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        value_input = torch.cat([obs, confidence_weights], dim=1)
        return self.value_network(value_input).squeeze(-1)


class SuperpositionMAPPOTrainer:
    """MAPPO trainer specialized for superposition outputs and sequential coordination"""
    
    def __init__(self, config: SuperpositionConfig, agent_names: List[str],
                 obs_dims: Dict[str, int], action_dims: Dict[str, int],
                 device: torch.device = torch.device('cpu')):
        self.config = config
        self.agent_names = agent_names
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        
        # Initialize networks
        self.policy_networks = {}
        self.value_networks = {}
        self.optimizers = {}
        
        for agent_name in agent_names:
            obs_dim = obs_dims[agent_name]
            action_dim = action_dims[agent_name]
            
            # Policy network
            policy_net = SuperpositionPolicyNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                superposition_dim=config.superposition_dim
            ).to(device)
            
            # Value network
            value_net = SuperpositionValueNetwork(
                obs_dim=obs_dim,
                superposition_dim=config.superposition_dim
            ).to(device)
            
            self.policy_networks[agent_name] = policy_net
            self.value_networks[agent_name] = value_net
            
            # Optimizers
            self.optimizers[agent_name] = {
                'policy': optim.Adam(policy_net.parameters(), lr=config.superposition_lr),
                'value': optim.Adam(value_net.parameters(), lr=config.superposition_lr)
            }
        
        # Training state
        self.episode_count = 0
        self.training_step = 0
        self.superposition_enabled = False
        
        # Metrics tracking
        self.metrics = {
            'superposition_quality': defaultdict(list),
            'coordination_scores': [],
            'entropy_scores': defaultdict(list),
            'consistency_scores': defaultdict(list),
            'confidence_variance': defaultdict(list),
            'policy_losses': defaultdict(list),
            'value_losses': defaultdict(list),
            'sequence_rewards': []
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized SuperpositionMAPPOTrainer for {len(agent_names)} agents")
    
    def get_superposition_actions(self, observations: Dict[str, torch.Tensor],
                                 temperature: float = 1.0) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get superposition actions for all agents"""
        agent_outputs = {}
        
        for agent_name in self.agent_names:
            obs = observations[agent_name]
            
            # Get superposition output
            superposition_output = self.policy_networks[agent_name](obs)
            
            # Collapse to action if superposition is enabled
            if self.superposition_enabled:
                action, log_prob = superposition_output.collapse_to_action(temperature)
                
                # Get value estimate
                value = self.value_networks[agent_name](obs, superposition_output.confidence_weights)
                
                agent_outputs[agent_name] = {
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'superposition_output': superposition_output
                }
            else:
                # Standard action selection during warmup
                decision_states = superposition_output.decision_states
                confidence_weights = superposition_output.confidence_weights
                
                # Use highest confidence decision
                best_decision_idx = torch.argmax(confidence_weights, dim=1)
                batch_size = obs.size(0)
                
                # Extract best decision logits
                best_logits = decision_states[torch.arange(batch_size), best_decision_idx]
                
                # Sample action
                action_dist = Categorical(logits=best_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Get value estimate
                value = self.value_networks[agent_name](obs, confidence_weights)
                
                agent_outputs[agent_name] = {
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'superposition_output': superposition_output
                }
        
        return agent_outputs
    
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Execute training step with superposition-aware loss"""
        if self.episode_count < self.config.warmup_episodes:
            return self._warmup_training_step(batch_data)
        else:
            self.superposition_enabled = True
            return self._superposition_training_step(batch_data)
    
    def _warmup_training_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Warmup training step with standard PPO"""
        losses = {}
        
        for agent_name in self.agent_names:
            agent_batch = batch_data[agent_name]
            
            # Get network outputs
            obs = agent_batch['observations']
            actions = agent_batch['actions']
            old_log_probs = agent_batch['log_probs']
            advantages = agent_batch['advantages']
            returns = agent_batch['returns']
            
            # Forward pass
            superposition_output = self.policy_networks[agent_name](obs)
            
            # Use highest confidence decision head
            confidence_weights = superposition_output.confidence_weights
            best_decision_idx = torch.argmax(confidence_weights, dim=1)
            batch_size = obs.size(0)
            
            # Get logits from best decision head
            best_logits = superposition_output.decision_states[torch.arange(batch_size), best_decision_idx]
            
            # Calculate policy loss
            action_dist = Categorical(logits=best_logits)
            new_log_probs = action_dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value_networks[agent_name](obs, confidence_weights)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_bonus = action_dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            
            # Update networks
            self.optimizers[agent_name]['policy'].zero_grad()
            self.optimizers[agent_name]['value'].zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_networks[agent_name].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_networks[agent_name].parameters(), 0.5)
            
            self.optimizers[agent_name]['policy'].step()
            self.optimizers[agent_name]['value'].step()
            
            losses[f'{agent_name}_policy_loss'] = policy_loss.item()
            losses[f'{agent_name}_value_loss'] = value_loss.item()
            losses[f'{agent_name}_entropy'] = entropy_bonus.item()
        
        return losses
    
    def _superposition_training_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Superposition-aware training step"""
        losses = {}
        
        for agent_name in self.agent_names:
            agent_batch = batch_data[agent_name]
            
            # Get network outputs
            obs = agent_batch['observations']
            actions = agent_batch['actions']
            old_log_probs = agent_batch['log_probs']
            advantages = agent_batch['advantages']
            returns = agent_batch['returns']
            
            # Forward pass
            superposition_output = self.policy_networks[agent_name](obs)
            
            # Superposition policy loss
            policy_loss = self._calculate_superposition_policy_loss(
                superposition_output, actions, old_log_probs, advantages
            )
            
            # Value loss
            values = self.value_networks[agent_name](obs, superposition_output.confidence_weights)
            value_loss = F.mse_loss(values, returns)
            
            # Superposition quality losses
            entropy_loss = self._calculate_entropy_loss(superposition_output)
            consistency_loss = self._calculate_consistency_loss(superposition_output)
            
            # Sequential coordination loss
            coordination_loss = self._calculate_coordination_loss(
                agent_name, superposition_output, batch_data
            )
            
            # Total loss
            total_loss = (
                policy_loss + 
                0.5 * value_loss +
                self.config.entropy_weight * entropy_loss +
                self.config.consistency_weight * consistency_loss +
                coordination_loss
            )
            
            # Update networks
            self.optimizers[agent_name]['policy'].zero_grad()
            self.optimizers[agent_name]['value'].zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_networks[agent_name].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_networks[agent_name].parameters(), 0.5)
            
            self.optimizers[agent_name]['policy'].step()
            self.optimizers[agent_name]['value'].step()
            
            # Track metrics
            losses[f'{agent_name}_policy_loss'] = policy_loss.item()
            losses[f'{agent_name}_value_loss'] = value_loss.item()
            losses[f'{agent_name}_entropy_loss'] = entropy_loss.item()
            losses[f'{agent_name}_consistency_loss'] = consistency_loss.item()
            losses[f'{agent_name}_coordination_loss'] = coordination_loss.item()
            
            # Update metrics
            self.metrics['entropy_scores'][agent_name].append(superposition_output.entropy.mean().item())
            self.metrics['consistency_scores'][agent_name].append(superposition_output.consistency_score.mean().item())
            self.metrics['confidence_variance'][agent_name].append(superposition_output.confidence_weights.var().item())
        
        # Update global metrics
        self._update_global_metrics(batch_data)
        
        return losses
    
    def _calculate_superposition_policy_loss(self, superposition_output: SuperpositionOutput,
                                           actions: torch.Tensor, old_log_probs: torch.Tensor,
                                           advantages: torch.Tensor) -> torch.Tensor:
        """Calculate policy loss for superposition outputs"""
        decision_states = superposition_output.decision_states
        confidence_weights = superposition_output.confidence_weights
        
        # Calculate new log probabilities for all superposition states
        batch_size = decision_states.size(0)
        superposition_dim = decision_states.size(1)
        
        new_log_probs = torch.zeros(batch_size, device=self.device)
        
        for i in range(superposition_dim):
            # Get action distribution for this superposition state
            action_dist = Categorical(logits=decision_states[:, i])
            log_prob = action_dist.log_prob(actions)
            
            # Weight by confidence
            weighted_log_prob = log_prob * confidence_weights[:, i]
            new_log_probs += weighted_log_prob
        
        # PPO clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def _calculate_entropy_loss(self, superposition_output: SuperpositionOutput) -> torch.Tensor:
        """Calculate entropy regularization loss"""
        target_entropy = self.config.min_entropy
        current_entropy = superposition_output.entropy.mean()
        
        # Penalty for entropy below threshold
        entropy_penalty = torch.relu(target_entropy - current_entropy)
        
        return entropy_penalty
    
    def _calculate_consistency_loss(self, superposition_output: SuperpositionOutput) -> torch.Tensor:
        """Calculate consistency loss for superposition states"""
        decision_states = superposition_output.decision_states
        
        # Calculate pairwise KL divergence between decision states
        consistency_loss = 0.0
        n_states = decision_states.size(1)
        
        for i in range(n_states):
            for j in range(i + 1, n_states):
                # Convert to probabilities
                probs_i = F.softmax(decision_states[:, i], dim=-1)
                probs_j = F.softmax(decision_states[:, j], dim=-1)
                
                # KL divergence
                kl_div = F.kl_div(
                    torch.log(probs_i + 1e-8),
                    probs_j,
                    reduction='batchmean'
                )
                
                consistency_loss += kl_div
        
        # Normalize by number of pairs
        consistency_loss /= (n_states * (n_states - 1) / 2)
        
        # Penalize high inconsistency
        consistency_penalty = torch.relu(consistency_loss - self.config.consistency_tolerance)
        
        return consistency_penalty
    
    def _calculate_coordination_loss(self, agent_name: str, superposition_output: SuperpositionOutput,
                                   batch_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate sequential coordination loss"""
        # Get agent position in sequence
        agent_idx = self.agent_names.index(agent_name)
        
        if agent_idx == 0:
            # First agent has no coordination loss
            return torch.tensor(0.0, device=self.device)
        
        # Get previous agent's superposition output
        prev_agent_name = self.agent_names[agent_idx - 1]
        prev_batch = batch_data.get(prev_agent_name)
        
        if prev_batch is None or 'superposition_output' not in prev_batch:
            return torch.tensor(0.0, device=self.device)
        
        prev_superposition_output = prev_batch['superposition_output']
        
        # Calculate coordination alignment
        current_confidence = superposition_output.confidence_weights
        prev_confidence = prev_superposition_output.confidence_weights
        
        # Alignment loss - encourage similar confidence patterns
        alignment_loss = F.mse_loss(current_confidence, prev_confidence)
        
        # Temporal discount
        discount_factor = self.config.temporal_discount ** agent_idx
        
        return alignment_loss * discount_factor
    
    def _update_global_metrics(self, batch_data: Dict[str, Any]):
        """Update global coordination metrics"""
        # Calculate sequence coordination score
        if len(self.agent_names) > 1:
            sequence_reward = 0.0
            
            for i in range(len(self.agent_names) - 1):
                agent_name = self.agent_names[i]
                next_agent_name = self.agent_names[i + 1]
                
                if agent_name in batch_data and next_agent_name in batch_data:
                    # Get rewards
                    agent_reward = batch_data[agent_name].get('rewards', torch.zeros(1)).mean()
                    next_reward = batch_data[next_agent_name].get('rewards', torch.zeros(1)).mean()
                    
                    # Coordination bonus for aligned positive rewards
                    if agent_reward > 0 and next_reward > 0:
                        sequence_reward += self.config.coordination_bonus
            
            self.metrics['sequence_rewards'].append(sequence_reward)
    
    def evaluate_superposition_quality(self, observations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate quality of superposition outputs"""
        quality_metrics = {}
        
        with torch.no_grad():
            for agent_name in self.agent_names:
                obs = observations[agent_name]
                superposition_output = self.policy_networks[agent_name](obs)
                
                # Quality metrics
                entropy = superposition_output.entropy.mean().item()
                consistency = superposition_output.consistency_score.mean().item()
                confidence_var = superposition_output.confidence_weights.var().item()
                
                # Overall quality score
                quality_score = (
                    0.4 * min(entropy / self.config.min_entropy, 1.0) +
                    0.3 * consistency +
                    0.3 * (1.0 - min(confidence_var / self.config.max_confidence_variance, 1.0))
                )
                
                quality_metrics[f'{agent_name}_entropy'] = entropy
                quality_metrics[f'{agent_name}_consistency'] = consistency
                quality_metrics[f'{agent_name}_confidence_var'] = confidence_var
                quality_metrics[f'{agent_name}_quality_score'] = quality_score
        
        return quality_metrics
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'superposition_enabled': self.superposition_enabled,
            'config': self.config,
            'metrics': self.metrics,
            'policy_networks': {
                name: net.state_dict() for name, net in self.policy_networks.items()
            },
            'value_networks': {
                name: net.state_dict() for name, net in self.value_networks.items()
            },
            'optimizers': {
                name: {
                    'policy': opt['policy'].state_dict(),
                    'value': opt['value'].state_dict()
                } for name, opt in self.optimizers.items()
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']
        self.superposition_enabled = checkpoint['superposition_enabled']
        self.metrics = checkpoint['metrics']
        
        # Load networks
        for name, state_dict in checkpoint['policy_networks'].items():
            if name in self.policy_networks:
                self.policy_networks[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['value_networks'].items():
            if name in self.value_networks:
                self.value_networks[name].load_state_dict(state_dict)
        
        # Load optimizers
        for name, opt_states in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name]['policy'].load_state_dict(opt_states['policy'])
                self.optimizers[name]['value'].load_state_dict(opt_states['value'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'superposition_enabled': self.superposition_enabled,
            'config': self.config,
            'metrics_summary': {}
        }
        
        # Summarize metrics
        for metric_name, metric_data in self.metrics.items():
            if isinstance(metric_data, dict):
                summary['metrics_summary'][metric_name] = {
                    agent: {
                        'mean': np.mean(values[-100:]) if values else 0,
                        'std': np.std(values[-100:]) if values else 0,
                        'latest': values[-1] if values else 0
                    }
                    for agent, values in metric_data.items()
                }
            else:
                summary['metrics_summary'][metric_name] = {
                    'mean': np.mean(metric_data[-100:]) if metric_data else 0,
                    'std': np.std(metric_data[-100:]) if metric_data else 0,
                    'latest': metric_data[-1] if metric_data else 0
                }
        
        return summary