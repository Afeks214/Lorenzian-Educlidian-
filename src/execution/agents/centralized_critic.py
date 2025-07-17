"""
Centralized Critic for Execution Engine MARL System

Implements centralized critic architecture for Multi-Agent PPO (MAPPO) training
of the execution engine agents. Provides shared value function estimation for
coordinated learning across all 5 agents:
- Position Sizing Agent (π₁)
- Stop/Target Agent (π₂) 
- Risk Monitor Agent (π₃)
- Portfolio Optimizer Agent (π₄)
- Routing Agent (π₅) - The Arbitrageur

Architecture: Combined observation processing → Value estimation
Input: Execution context (15D) + market features (32D) + routing state (55D) = 102D total
Output: State value V(s) for coordinated 5-agent policy learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import structlog
from dataclasses import dataclass

from .position_sizing_agent import ExecutionContext

logger = structlog.get_logger()


@dataclass
class MarketFeatures:
    """
    32-dimensional market features for centralized critic
    
    Extended market context beyond the 15D execution context:
    0-7: Order flow features (buy_volume, sell_volume, order_flow_imbalance, etc.)
    8-15: Price action features (price_momentum, support_resistance, etc.)
    16-23: Volatility surface features (term_structure, skew, etc.)
    24-31: Cross-asset features (correlation_shifts, regime_indicators, etc.)
    """
    # Order flow features (8D)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    order_flow_imbalance: float = 0.0
    large_order_flow: float = 0.0
    retail_flow: float = 0.0
    institutional_flow: float = 0.0
    flow_toxicity: float = 0.0
    flow_persistence: float = 0.0
    
    # Price action features (8D)
    price_momentum_1m: float = 0.0
    price_momentum_5m: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    trend_strength: float = 0.0
    mean_reversion_signal: float = 0.0
    breakout_probability: float = 0.0
    reversal_probability: float = 0.0
    
    # Volatility surface features (8D)
    atm_vol: float = 0.0
    vol_skew: float = 0.0
    vol_term_structure: float = 0.0
    vol_smile_curvature: float = 0.0
    realized_garch: float = 0.0
    vol_risk_premium: float = 0.0
    vol_persistence: float = 0.0
    vol_clustering: float = 0.0
    
    # Cross-asset features (8D)
    correlation_spy: float = 0.0
    correlation_vix: float = 0.0
    correlation_bonds: float = 0.0
    correlation_dollar: float = 0.0
    regime_equity: float = 0.0
    regime_volatility: float = 0.0
    regime_interest_rate: float = 0.0
    regime_risk_off: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        return torch.tensor([
            # Order flow features
            self.buy_volume, self.sell_volume, self.order_flow_imbalance, self.large_order_flow,
            self.retail_flow, self.institutional_flow, self.flow_toxicity, self.flow_persistence,
            # Price action features
            self.price_momentum_1m, self.price_momentum_5m, self.support_level, self.resistance_level,
            self.trend_strength, self.mean_reversion_signal, self.breakout_probability, self.reversal_probability,
            # Volatility surface features
            self.atm_vol, self.vol_skew, self.vol_term_structure, self.vol_smile_curvature,
            self.realized_garch, self.vol_risk_premium, self.vol_persistence, self.vol_clustering,
            # Cross-asset features
            self.correlation_spy, self.correlation_vix, self.correlation_bonds, self.correlation_dollar,
            self.regime_equity, self.regime_volatility, self.regime_interest_rate, self.regime_risk_off
        ], dtype=torch.float32)


@dataclass
class CombinedState:
    """Combined state for centralized critic (now includes routing state)"""
    execution_context: ExecutionContext
    market_features: MarketFeatures
    routing_state: Optional[torch.Tensor] = None  # 55D routing state vector
    agent_actions: Optional[torch.Tensor] = None  # Previous actions from all 5 agents
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to combined tensor for critic input"""
        context_tensor = self.execution_context.to_tensor()  # 15D
        market_tensor = self.market_features.to_tensor()     # 32D
        
        combined = torch.cat([context_tensor, market_tensor], dim=0)  # 47D
        
        # Add routing state if available
        if self.routing_state is not None:
            combined = torch.cat([combined, self.routing_state], dim=0)  # 47D + 55D = 102D
        
        if self.agent_actions is not None:
            combined = torch.cat([combined, self.agent_actions], dim=0)
            
        return combined


class ExecutionCentralizedCritic(nn.Module):
    """
    Centralized Critic for Execution Engine MARL System
    
    Processes combined execution context, market features, and routing state to provide
    shared value function estimation for coordinated 5-agent learning.
    
    Architecture: 102D input → 256→128→64→1 output (state value)
    Input breakdown: Execution context (15D) + Market features (32D) + Routing state (55D)
    """
    
    def __init__(self, 
                 context_dim: int = 15,
                 market_features_dim: int = 32,
                 routing_state_dim: int = 55,
                 num_agents: int = 5,
                 hidden_dims: List[int] = None):
        super().__init__()
        
        self.context_dim = context_dim
        self.market_features_dim = market_features_dim
        self.routing_state_dim = routing_state_dim
        self.num_agents = num_agents
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Combined input dimension (includes routing state)
        self.combined_input_dim = context_dim + market_features_dim + routing_state_dim  # 15 + 32 + 55 = 102
        
        # Build network layers
        layers = []
        prev_dim = self.combined_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Track evaluation metrics
        self.evaluations = 0
        self.total_evaluation_time = 0.0
        
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, combined_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state value
        
        Args:
            combined_state: Combined execution context + market features + routing state [batch_size, 102]
            
        Returns:
            State value estimate [batch_size, 1]
        """
        return self.network(combined_state)
    
    def evaluate_state(self, combined_state: CombinedState) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate state value for given combined state
        
        Args:
            combined_state: Combined execution and market state
            
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
                value = self.forward(state_tensor)
                
            # Extract scalar value
            state_value = value.item()
            
            # Evaluation metrics
            end_time = time.perf_counter()
            evaluation_time = end_time - start_time
            self.evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            evaluation_info = {
                'state_value': state_value,
                'evaluation_time_ms': evaluation_time * 1000,
                'total_evaluations': self.evaluations,
                'avg_evaluation_time_ms': (self.total_evaluation_time / self.evaluations) * 1000
            }
            
            return state_value, evaluation_info
            
        except Exception as e:
            logger.error("Error in state evaluation", error=str(e))
            return 0.0, {'error': str(e)}
    
    def batch_evaluate(self, combined_states: List[CombinedState]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Batch evaluate multiple states for efficient training
        
        Args:
            combined_states: List of combined states
            
        Returns:
            Tuple of (state_values, batch_info)
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert all states to tensors and stack
            state_tensors = [state.to_tensor() for state in combined_states]
            batch_tensor = torch.stack(state_tensors, dim=0)
            
            # Forward pass
            with torch.no_grad():
                values = self.forward(batch_tensor)
                
            # Batch metrics
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            batch_size = len(combined_states)
            
            batch_info = {
                'batch_size': batch_size,
                'batch_time_ms': batch_time * 1000,
                'time_per_evaluation_ms': (batch_time / batch_size) * 1000,
                'total_evaluations': self.evaluations + batch_size
            }
            
            self.evaluations += batch_size
            self.total_evaluation_time += batch_time
            
            return values.squeeze(-1), batch_info  # Remove last dimension
            
        except Exception as e:
            logger.error("Error in batch evaluation", error=str(e))
            return torch.zeros(len(combined_states)), {'error': str(e)}


class MAPPOTrainer:
    """
    Multi-Agent PPO Trainer for Execution Engine
    
    Coordinates training of Position Sizing (π₁), Execution Timing (π₂),
    and Risk Management (π₃) agents using centralized critic.
    """
    
    def __init__(self,
                 critic: ExecutionCentralizedCritic,
                 agents: List[Any],  # List of agent networks
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.critic = critic
        self.agents = agents
        self.num_agents = len(agents)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )
        
        self.agent_optimizers = [
            torch.optim.Adam(agent.parameters(), lr=learning_rate)
            for agent in self.agents
        ]
        
        # Training metrics
        self.training_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        logger.info("MAPPO Trainer initialized",
                   num_agents=self.num_agents,
                   learning_rate=learning_rate,
                   gamma=gamma,
                   clip_epsilon=clip_epsilon)
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Rewards tensor [batch_size, sequence_length]
            values: State values [batch_size, sequence_length]
            dones: Done flags [batch_size, sequence_length]
            
        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_length = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae_lambda = 0.95  # GAE parameter
        
        # Compute advantages using GAE
        for t in reversed(range(seq_length - 1)):
            delta = rewards[:, t] + self.gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
            advantages[:, t] = delta + self.gamma * gae_lambda * (1 - dones[:, t]) * advantages[:, t + 1]
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update_agents(self,
                     combined_states: List[CombinedState],
                     actions: List[torch.Tensor],  # Actions for each agent
                     old_log_probs: List[torch.Tensor],  # Old log probs for each agent
                     rewards: torch.Tensor,
                     dones: torch.Tensor) -> Dict[str, float]:
        """
        Update all agents using MAPPO algorithm
        
        Args:
            combined_states: List of combined states
            actions: List of action tensors for each agent
            old_log_probs: List of old log probability tensors for each agent
            rewards: Reward tensor
            dones: Done flags tensor
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.perf_counter()
        
        # Convert states to batch tensor
        state_tensors = [state.to_tensor() for state in combined_states]
        batch_states = torch.stack(state_tensors, dim=0)
        
        # Get current state values from critic
        current_values = self.critic(batch_states).squeeze(-1)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, current_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        value_loss = F.mse_loss(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Update each agent
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        
        for agent_idx, (agent, agent_actions, agent_old_log_probs) in enumerate(
            zip(self.agents, actions, old_log_probs)
        ):
            # Get current policy distribution
            if hasattr(agent, 'fast_inference'):
                action_probs = agent.fast_inference(batch_states)
            else:
                action_probs = agent(batch_states)
            
            # Compute log probabilities
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(agent_actions)
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - agent_old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute entropy loss
            entropy = action_dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Total loss for this agent
            agent_loss = policy_loss + entropy_loss
            
            # Update agent
            self.agent_optimizers[agent_idx].zero_grad()
            agent_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            self.agent_optimizers[agent_idx].step()
            
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Update training metrics
        self.training_steps += 1
        self.policy_losses.append(total_policy_loss / self.num_agents)
        self.value_losses.append(value_loss.item())
        self.entropy_losses.append(total_entropy_loss / self.num_agents)
        
        # Training time
        end_time = time.perf_counter()
        training_time = end_time - start_time
        
        return {
            'training_step': self.training_steps,
            'policy_loss': total_policy_loss / self.num_agents,
            'value_loss': value_loss.item(),
            'entropy_loss': total_entropy_loss / self.num_agents,
            'training_time_ms': training_time * 1000,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': returns.mean().item()
        }
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        if not self.policy_losses:
            return {}
        
        recent_steps = min(100, len(self.policy_losses))
        
        return {
            'total_training_steps': self.training_steps,
            'avg_policy_loss': np.mean(self.policy_losses[-recent_steps:]),
            'avg_value_loss': np.mean(self.value_losses[-recent_steps:]),
            'avg_entropy_loss': np.mean(self.entropy_losses[-recent_steps:]),
            'policy_loss_trend': np.polyfit(range(recent_steps), self.policy_losses[-recent_steps:], 1)[0],
            'value_loss_trend': np.polyfit(range(recent_steps), self.value_losses[-recent_steps:], 1)[0],
            'critic_evaluations': self.critic.evaluations,
            'critic_avg_eval_time_ms': (self.critic.total_evaluation_time / max(1, self.critic.evaluations)) * 1000
        }


# Factory functions
def create_centralized_critic(config: Dict[str, Any]) -> ExecutionCentralizedCritic:
    """Create and initialize centralized critic for 5-agent system"""
    return ExecutionCentralizedCritic(
        context_dim=config.get('context_dim', 15),
        market_features_dim=config.get('market_features_dim', 32),
        routing_state_dim=config.get('routing_state_dim', 55),
        num_agents=config.get('num_agents', 5),  # Now 5 agents
        hidden_dims=config.get('critic_hidden_dims', [256, 128, 64])
    )


def create_mappo_trainer(critic: ExecutionCentralizedCritic,
                        agents: List[Any],
                        config: Dict[str, Any]) -> MAPPOTrainer:
    """Create and initialize MAPPO trainer"""
    return MAPPOTrainer(
        critic=critic,
        agents=agents,
        learning_rate=config.get('learning_rate', 1e-4),
        gamma=config.get('gamma', 0.99),
        clip_epsilon=config.get('clip_epsilon', 0.2),
        value_loss_coef=config.get('value_loss_coef', 0.5),
        entropy_coef=config.get('entropy_coef', 0.01)
    )