"""
Specialized Loss Functions for MAPPO Training
Implements PPO clipped objective, value function loss, and entropy regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization loss with clipping.
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: Optional[float] = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        normalize_advantages: bool = True
    ):
        """
        Initialize PPO loss.
        
        Args:
            clip_epsilon: PPO clipping parameter for policy
            value_clip_epsilon: Optional clipping for value function
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            normalize_advantages: Whether to normalize advantages
        """
        super().__init__()
        
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.normalize_advantages = normalize_advantages
        
    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components.
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
            values: Current value estimates
            old_values: Old value estimates
            returns: Target returns
            entropy: Action distribution entropy
            importance_weights: Optional importance sampling weights
            
        Returns:
            Dictionary with total_loss and individual components
        """
        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # Value function loss with optional clipping
        if self.value_clip_epsilon is not None:
            # Clipped value loss
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.value_clip_epsilon,
                self.value_clip_epsilon
            )
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            # Standard MSE value loss
            value_loss = (values - returns) ** 2
        
        # Entropy bonus (negative because we minimize loss)
        entropy_loss = -entropy
        
        # Apply importance weights if provided
        if importance_weights is not None:
            policy_loss = policy_loss * importance_weights
            value_loss = value_loss * importance_weights
            entropy_loss = entropy_loss * importance_weights
        
        # Total loss
        total_loss = (
            policy_loss.mean() +
            self.value_loss_coef * value_loss.mean() +
            self.entropy_coef * entropy_loss.mean()
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss.mean(),
            'value_loss': value_loss.mean(),
            'entropy_loss': entropy_loss.mean(),
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean(),
            'clip_fraction': ((ratio - 1).abs() > self.clip_epsilon).float().mean()
        }


class GAEAdvantageEstimator:
    """
    Generalized Advantage Estimation (GAE) calculator.
    """
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize GAE estimator.
        
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter (bias-variance tradeoff)
        """
        self.gamma = gamma
        self.lam = lam
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Reward tensor (batch_size, seq_len)
            values: Value estimates (batch_size, seq_len)
            next_values: Next value estimates (batch_size, seq_len)
            dones: Done flags (batch_size, seq_len)
            
        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        
        # Initialize advantages
        advantages = torch.zeros_like(rewards)
        
        # Compute GAE backwards through time
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]
            
            # TD error
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            
            # GAE
            gae = delta + self.gamma * self.lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def compute_td_errors(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD errors for prioritized replay.
        
        Args:
            rewards: Rewards
            values: Current value estimates
            next_values: Next value estimates
            dones: Done flags
            
        Returns:
            TD error tensor
        """
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        return td_errors


class MultiAgentPPOLoss(nn.Module):
    """
    Multi-agent PPO loss that handles multiple agents with shared critic.
    """
    
    def __init__(
        self,
        n_agents: int,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: Optional[float] = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        agent_loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-agent PPO loss.
        
        Args:
            n_agents: Number of agents
            clip_epsilon: PPO clipping parameter
            value_clip_epsilon: Value function clipping
            entropy_coef: Entropy coefficient
            value_loss_coef: Value loss coefficient  
            agent_loss_weights: Optional per-agent loss weights
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.agent_loss_weights = agent_loss_weights or {}
        
        # Single-agent PPO loss for each agent
        self.ppo_loss = PPOLoss(
            clip_epsilon=clip_epsilon,
            value_clip_epsilon=value_clip_epsilon,
            entropy_coef=entropy_coef,
            value_loss_coef=0.0,  # Handle value loss separately
            normalize_advantages=True
        )
        
    def forward(
        self,
        agent_log_probs: Dict[str, torch.Tensor],
        agent_old_log_probs: Dict[str, torch.Tensor],
        agent_advantages: Dict[str, torch.Tensor],
        agent_entropies: Dict[str, torch.Tensor],
        centralized_values: torch.Tensor,
        centralized_old_values: torch.Tensor,
        centralized_returns: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-agent PPO loss.
        
        Args:
            agent_log_probs: Per-agent current log probabilities
            agent_old_log_probs: Per-agent old log probabilities
            agent_advantages: Per-agent advantages
            agent_entropies: Per-agent entropies
            centralized_values: Centralized critic values
            centralized_old_values: Old centralized values
            centralized_returns: Target returns
            importance_weights: Optional IS weights
            
        Returns:
            Loss dictionary
        """
        total_policy_loss = 0
        total_entropy_loss = 0
        agent_losses = {}
        
        # Compute per-agent losses
        for agent_name in agent_log_probs.keys():
            weight = self.agent_loss_weights.get(agent_name, 1.0)
            
            # Get agent-specific PPO loss
            agent_loss = self.ppo_loss(
                log_probs=agent_log_probs[agent_name],
                old_log_probs=agent_old_log_probs[agent_name],
                advantages=agent_advantages[agent_name],
                values=torch.zeros_like(agent_advantages[agent_name]),  # Dummy
                old_values=torch.zeros_like(agent_advantages[agent_name]),  # Dummy
                returns=torch.zeros_like(agent_advantages[agent_name]),  # Dummy
                entropy=agent_entropies[agent_name],
                importance_weights=importance_weights
            )
            
            total_policy_loss += weight * agent_loss['policy_loss']
            total_entropy_loss += weight * agent_loss['entropy_loss']
            agent_losses[f'{agent_name}_policy_loss'] = agent_loss['policy_loss']
            agent_losses[f'{agent_name}_entropy'] = -agent_loss['entropy_loss']
            agent_losses[f'{agent_name}_kl'] = agent_loss['approx_kl']
        
        # Centralized value loss
        if self.value_clip_epsilon is not None:
            value_pred_clipped = centralized_old_values + torch.clamp(
                centralized_values - centralized_old_values,
                -self.value_clip_epsilon,
                self.value_clip_epsilon
            )
            value_losses = (centralized_values - centralized_returns) ** 2
            value_losses_clipped = (value_pred_clipped - centralized_returns) ** 2
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = F.mse_loss(centralized_values, centralized_returns, reduction='none')
        
        if importance_weights is not None:
            value_loss = value_loss * importance_weights
            
        value_loss = value_loss.mean()
        
        # Total loss
        total_loss = (
            total_policy_loss / self.n_agents +
            self.value_loss_coef * value_loss +
            self.entropy_coef * total_entropy_loss / self.n_agents
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': total_policy_loss / self.n_agents,
            'value_loss': value_loss,
            'entropy_loss': total_entropy_loss / self.n_agents,
            **agent_losses
        }


class AdaptiveKLPenalty:
    """
    Adaptive KL penalty for PPO (alternative to clipping).
    """
    
    def __init__(
        self,
        target_kl: float = 0.01,
        penalty_init: float = 1.0,
        penalty_lr: float = 1.5
    ):
        """
        Initialize adaptive KL penalty.
        
        Args:
            target_kl: Target KL divergence
            penalty_init: Initial penalty coefficient
            penalty_lr: Penalty learning rate
        """
        self.target_kl = target_kl
        self.penalty = penalty_init
        self.penalty_lr = penalty_lr
        
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute policy loss with adaptive KL penalty.
        
        Args:
            log_probs: Current log probabilities
            old_log_probs: Old log probabilities
            advantages: Advantages
            
        Returns:
            Tuple of (loss, current_kl)
        """
        # Compute KL divergence
        log_ratio = log_probs - old_log_probs
        kl = (torch.exp(old_log_probs) * log_ratio).mean()
        
        # Policy gradient loss
        pg_loss = -(log_probs * advantages).mean()
        
        # Total loss with KL penalty
        loss = pg_loss + self.penalty * kl
        
        # Update penalty based on KL
        if kl < self.target_kl / self.penalty_lr:
            self.penalty /= self.penalty_lr
        elif kl > self.target_kl * self.penalty_lr:
            self.penalty *= self.penalty_lr
            
        return loss, kl.item()