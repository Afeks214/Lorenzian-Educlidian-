"""
Enhanced Tactical MAPPO Trainer with Value Clipping and Adaptive Entropy
Implements production-grade training for the 5-minute tactical MARL system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
from tqdm import tqdm
import time
from dataclasses import dataclass

from models.tactical_architectures import TacticalMARLSystem
from utils.tactical_replay_buffer import TacticalExperienceBuffer, TacticalBatch
from utils.tactical_checkpoint_manager import TacticalModelManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode_reward: float
    episode_length: int
    actor_losses: Dict[str, float]
    critic_loss: float
    entropy_values: Dict[str, float]
    value_loss: float
    policy_loss: float
    advantage_mean: float
    advantage_std: float
    learning_rates: Dict[str, float]
    grad_norms: Dict[str, float]
    temperature_values: Dict[str, float]
    kl_divergence: float
    explained_variance: float


class AdaptiveEntropyScheduler:
    """
    Adaptive entropy scheduler that decays entropy bonus for fine-tuning.
    
    Starts high for exploration, gradually decays as policy converges.
    """
    
    def __init__(
        self,
        initial_entropy: float = 0.01,
        min_entropy: float = 0.001,
        decay_rate: float = 0.995,
        warmup_episodes: int = 100
    ):
        """
        Initialize adaptive entropy scheduler.
        
        Args:
            initial_entropy: Starting entropy coefficient
            min_entropy: Minimum entropy coefficient
            decay_rate: Exponential decay rate per episode
            warmup_episodes: Episodes to keep initial entropy
        """
        self.initial_entropy = initial_entropy
        self.min_entropy = min_entropy
        self.decay_rate = decay_rate
        self.warmup_episodes = warmup_episodes
        
        self.current_entropy = initial_entropy
        self.episode_count = 0
    
    def step(self, episode: int) -> float:
        """
        Update entropy coefficient based on episode count.
        
        Args:
            episode: Current episode number
            
        Returns:
            Current entropy coefficient
        """
        self.episode_count = episode
        
        if episode < self.warmup_episodes:
            # Warmup phase - keep initial entropy
            self.current_entropy = self.initial_entropy
        else:
            # Decay phase
            decay_episodes = episode - self.warmup_episodes
            self.current_entropy = max(
                self.min_entropy,
                self.initial_entropy * (self.decay_rate ** decay_episodes)
            )
        
        return self.current_entropy
    
    def get_entropy(self) -> float:
        """Get current entropy coefficient."""
        return self.current_entropy


class EnhancedGAE:
    """
    Enhanced Generalized Advantage Estimation with normalization.
    
    Implements exact GAE formulation with advantage normalization
    and running statistics for stability.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        clip_advantages: bool = True,
        advantage_clip_range: float = 10.0
    ):
        """
        Initialize Enhanced GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_advantages: Whether to normalize advantages
            clip_advantages: Whether to clip extreme advantages
            advantage_clip_range: Range for advantage clipping
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_advantages = clip_advantages
        self.advantage_clip_range = advantage_clip_range
        
        # Running statistics for advantage normalization
        self.advantage_mean = 0.0
        self.advantage_std = 1.0
        self.advantage_count = 0
        
    def compute_gae_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Rewards tensor (batch_size, sequence_length)
            values: Value estimates (batch_size, sequence_length)
            dones: Done flags (batch_size, sequence_length)
            next_values: Next state values (batch_size, sequence_length)
            
        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, sequence_length = rewards.shape
        
        # Prepare next values
        if next_values is None:
            next_values = torch.zeros_like(values)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]
            
            # TD error: δt = rt + γ * V(s_{t+1}) * (1 - done) - V(st)
            td_error = (
                rewards[:, t] + 
                self.gamma * next_value * (1 - dones[:, t]) - 
                values[:, t]
            )
            
            # GAE: At = δt + γλ * (1 - done) * A_{t+1}
            advantages[:, t] = (
                td_error + 
                self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_advantage
            )
            last_advantage = advantages[:, t]
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = self._normalize_advantages(advantages)
        
        # Clip advantages if requested
        if self.clip_advantages:
            advantages = torch.clamp(
                advantages, 
                -self.advantage_clip_range, 
                self.advantage_clip_range
            )
        
        return advantages, returns
    
    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages using running statistics.
        
        Args:
            advantages: Advantage tensor
            
        Returns:
            Normalized advantages
        """
        # Flatten for statistics
        flat_advantages = advantages.view(-1)
        
        # Update running statistics
        batch_mean = flat_advantages.mean().item()
        batch_std = flat_advantages.std().item() + 1e-8
        
        # Update running statistics with exponential moving average
        alpha = 0.01  # Learning rate for statistics
        self.advantage_mean = (1 - alpha) * self.advantage_mean + alpha * batch_mean
        self.advantage_std = (1 - alpha) * self.advantage_std + alpha * batch_std
        self.advantage_count += 1
        
        # Normalize
        normalized_advantages = (advantages - self.advantage_mean) / (self.advantage_std + 1e-8)
        
        return normalized_advantages


class TacticalMAPPOTrainer:
    """
    Enhanced Tactical MAPPO Trainer with production-grade features.
    
    Features:
    - Value function clipping for stability
    - Adaptive entropy scheduling
    - Enhanced GAE with normalization
    - Comprehensive metric tracking
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Tactical MAPPO Trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Training hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_clip_epsilon = config.get('value_clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.kl_target = config.get('kl_target', 0.01)
        self.kl_tolerance = config.get('kl_tolerance', 1.5)
        
        # Initialize components
        self._initialize_model()
        self._initialize_optimizers()
        self._initialize_schedulers()
        self._initialize_buffer()
        self._initialize_utils()
        
        # Training state
        self.update_count = 0
        self.episode_count = 0
        self.total_steps = 0
        self.training_metrics = defaultdict(list)
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=1000)
        
        logger.info(f"TacticalMAPPOTrainer initialized on {self.device}")
        logger.info(f"Model parameters: {self.model.get_model_info()}")
    
    def _initialize_model(self):
        """Initialize the tactical MARL system."""
        model_config = self.config.get('model', {})
        
        self.model = TacticalMARLSystem(
            input_shape=model_config.get('input_shape', (60, 7)),
            action_dim=model_config.get('action_dim', 3),
            hidden_dim=model_config.get('hidden_dim', 256),
            critic_hidden_dims=model_config.get('critic_hidden_dims', [512, 256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            temperature_init=model_config.get('temperature_init', 1.0)
        ).to(self.device)
    
    def _initialize_optimizers(self):
        """Initialize optimizers for agents and critic."""
        self.optimizers = {}
        
        # Individual optimizers for each agent
        for agent_name in ['fvg', 'momentum', 'entry']:
            self.optimizers[f'{agent_name}_actor'] = optim.Adam(
                self.model.agents[agent_name].parameters(),
                lr=self.learning_rate,
                eps=1e-5
            )
        
        # Critic optimizer
        self.optimizers['critic'] = optim.Adam(
            self.model.critic.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
    
    def _initialize_schedulers(self):
        """Initialize learning rate and entropy schedulers."""
        # Adaptive entropy scheduler
        entropy_config = self.config.get('entropy_scheduler', {})
        self.entropy_scheduler = AdaptiveEntropyScheduler(
            initial_entropy=entropy_config.get('initial_entropy', 0.01),
            min_entropy=entropy_config.get('min_entropy', 0.001),
            decay_rate=entropy_config.get('decay_rate', 0.995),
            warmup_episodes=entropy_config.get('warmup_episodes', 100)
        )
        
        # Learning rate schedulers
        self.lr_schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.lr_schedulers[name] = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1000,
                gamma=0.95
            )
    
    def _initialize_buffer(self):
        """Initialize experience buffer."""
        buffer_config = self.config.get('buffer', {})
        self.buffer = TacticalExperienceBuffer(
            capacity=buffer_config.get('capacity', 10000),
            alpha=buffer_config.get('alpha', 0.6),
            beta=buffer_config.get('beta', 0.4),
            beta_increment=buffer_config.get('beta_increment', 0.001)
        )
    
    def _initialize_utils(self):
        """Initialize utility components."""
        # Enhanced GAE
        self.gae = EnhancedGAE(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantages=True,
            clip_advantages=True,
            advantage_clip_range=10.0
        )
        
        # Model manager
        checkpoint_config = self.config.get('checkpoint', {})
        self.model_manager = TacticalModelManager(
            model_dir=checkpoint_config.get('model_dir', 'models/tactical'),
            checkpoint_interval=checkpoint_config.get('checkpoint_interval', 1000),
            max_checkpoints=checkpoint_config.get('max_checkpoints', 10)
        )
    
    def collect_experience(
        self,
        env,
        n_steps: int = 2048,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Collect experience from environment.
        
        Args:
            env: Environment instance
            n_steps: Number of steps to collect
            render: Whether to render environment
            
        Returns:
            Dictionary with collected experience statistics
        """
        experience_stats = {
            'total_steps': 0,
            'episodes_completed': 0,
            'total_reward': 0.0,
            'average_episode_length': 0.0,
            'agent_actions': defaultdict(int)
        }
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(n_steps):
                # Get current state
                state = env.get_state()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get actions from all agents
                start_time = time.time()
                model_output = self.model(state_tensor, deterministic=False)
                inference_time = (time.time() - start_time) * 1000
                self.inference_times.append(inference_time)
                
                # Extract actions
                actions = {}
                log_probs = {}
                for agent_name in ['fvg', 'momentum', 'entry']:
                    agent_output = model_output['agents'][agent_name]
                    actions[agent_name] = agent_output['action'].item()
                    log_probs[agent_name] = agent_output['log_prob'].item()
                
                # Get value estimate
                value = model_output['critic']['value'].item()
                
                # Take action in environment
                next_state, rewards, done, info = env.step(actions)
                
                # Store experience
                self.buffer.add_experience(
                    state=state,
                    actions=actions,
                    rewards=rewards,
                    next_state=next_state,
                    done=done,
                    log_probs=log_probs,
                    value=value
                )
                
                # Update statistics
                experience_stats['total_steps'] += 1
                experience_stats['total_reward'] += sum(rewards.values())
                
                for agent_name, action in actions.items():
                    experience_stats['agent_actions'][f'{agent_name}_action_{action}'] += 1
                
                if done:
                    experience_stats['episodes_completed'] += 1
                    self.episode_count += 1
                    env.reset()
                
                if render:
                    env.render()
        
        # Calculate averages
        if experience_stats['episodes_completed'] > 0:
            experience_stats['average_episode_length'] = (
                experience_stats['total_steps'] / experience_stats['episodes_completed']
            )
        
        self.total_steps += experience_stats['total_steps']
        return experience_stats
    
    def update_agents(self, batch_size: Optional[int] = None) -> TrainingMetrics:
        """
        Update all agents using collected experience.
        
        Args:
            batch_size: Batch size for training (uses default if None)
            
        Returns:
            Training metrics
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        start_time = time.time()
        
        # Sample batch from buffer
        batch = self.buffer.sample(batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor(batch.states).to(self.device)
        actions = {name: torch.LongTensor(batch.actions[name]).to(self.device) 
                  for name in ['fvg', 'momentum', 'entry']}
        rewards = {name: torch.FloatTensor(batch.rewards[name]).to(self.device) 
                  for name in ['fvg', 'momentum', 'entry']}
        old_log_probs = {name: torch.FloatTensor(batch.log_probs[name]).to(self.device) 
                        for name in ['fvg', 'momentum', 'entry']}
        values = torch.FloatTensor(batch.values).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device)
        
        # Compute advantages and returns
        # Average rewards across agents and reshape for GAE
        avg_rewards = torch.stack(list(rewards.values())).mean(0)  # Average rewards
        avg_rewards = avg_rewards.unsqueeze(1)  # Add sequence dimension
        values_reshaped = values.unsqueeze(1)  # Add sequence dimension
        dones_reshaped = dones.unsqueeze(1)  # Add sequence dimension
        
        advantages, returns = self.gae.compute_gae_advantages(
            rewards=avg_rewards,
            values=values_reshaped,
            dones=dones_reshaped
        )
        
        # Flatten back to 1D
        advantages = advantages.squeeze(1)
        returns = returns.squeeze(1)
        
        # Update entropy coefficient
        current_entropy = self.entropy_scheduler.step(self.episode_count)
        
        # Initialize metrics
        metrics = {
            'actor_losses': {},
            'entropy_values': {},
            'learning_rates': {},
            'grad_norms': {},
            'temperature_values': {},
            'kl_divergences': {}
        }
        
        # Training mode
        self.model.train()
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            # Update each agent
            for agent_name in ['fvg', 'momentum', 'entry']:
                optimizer = self.optimizers[f'{agent_name}_actor']
                optimizer.zero_grad()
                
                # Get current policy
                agent_output = self.model.agents[agent_name](states)
                
                # Compute policy loss
                log_probs = agent_output['log_prob']
                ratio = torch.exp(log_probs - old_log_probs[agent_name])
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy loss
                entropy = torch.distributions.Categorical(agent_output['action_probs']).entropy()
                entropy_loss = -current_entropy * entropy.mean()
                
                # Total actor loss
                actor_loss = policy_loss + entropy_loss
                
                # Backward pass
                actor_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.agents[agent_name].parameters(),
                    self.max_grad_norm
                )
                
                optimizer.step()
                
                # Store metrics
                metrics['actor_losses'][agent_name] = actor_loss.item()
                metrics['entropy_values'][agent_name] = entropy.mean().item()
                metrics['grad_norms'][agent_name] = grad_norm.item()
                metrics['temperature_values'][agent_name] = agent_output['temperature']
                
                # KL divergence
                kl_div = (old_log_probs[agent_name] - log_probs).mean().item()
                metrics['kl_divergences'][agent_name] = kl_div
            
            # Update critic
            optimizer = self.optimizers['critic']
            optimizer.zero_grad()
            
            # Get current values
            critic_output = self.model.critic(states.view(batch_size, -1).repeat(1, 3))
            current_values = critic_output['value']
            
            # Value loss with clipping
            value_pred_clipped = values + torch.clamp(
                current_values - values,
                -self.value_clip_epsilon,
                self.value_clip_epsilon
            )
            
            value_loss_1 = F.mse_loss(current_values, returns)
            value_loss_2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss_1, value_loss_2)
            
            # Backward pass
            value_loss.backward()
            
            # Gradient clipping
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(),
                self.max_grad_norm
            )
            
            optimizer.step()
        
        # Update learning rates
        for scheduler in self.lr_schedulers.values():
            scheduler.step()
        
        # Record training time
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        
        # Create training metrics
        training_metrics = TrainingMetrics(
            episode_reward=0.0,  # Will be filled by caller
            episode_length=0,    # Will be filled by caller
            actor_losses=metrics['actor_losses'],
            critic_loss=value_loss.item(),
            entropy_values=metrics['entropy_values'],
            value_loss=value_loss.item(),
            policy_loss=sum(metrics['actor_losses'].values()) / len(metrics['actor_losses']),
            advantage_mean=advantages.mean().item(),
            advantage_std=advantages.std().item(),
            learning_rates={name: opt.param_groups[0]['lr'] for name, opt in self.optimizers.items()},
            grad_norms=metrics['grad_norms'],
            temperature_values=metrics['temperature_values'],
            kl_divergence=sum(metrics['kl_divergences'].values()) / len(metrics['kl_divergences']),
            explained_variance=self._explained_variance(values, returns)
        )
        
        self.update_count += 1
        return training_metrics
    
    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Calculate explained variance."""
        var_y = torch.var(y_true)
        return (1 - torch.var(y_true - y_pred) / var_y).item() if var_y > 0 else 0.0
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            metrics: Performance metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        return self.model_manager.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.lr_schedulers,
            update_count=self.update_count,
            episode_count=self.episode_count,
            metrics=metrics,
            buffer_state=self.buffer.get_state(),
            additional_info={
                'entropy_scheduler_state': {
                    'current_entropy': self.entropy_scheduler.current_entropy,
                    'episode_count': self.entropy_scheduler.episode_count
                },
                'gae_state': {
                    'advantage_mean': self.gae.advantage_mean,
                    'advantage_std': self.gae.advantage_std,
                    'advantage_count': self.gae.advantage_count
                }
            },
            is_best=is_best
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint metadata
        """
        return self.model_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.lr_schedulers
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'update_times': {
                'mean': np.mean(self.update_times) if self.update_times else 0,
                'std': np.std(self.update_times) if self.update_times else 0,
                'min': np.min(self.update_times) if self.update_times else 0,
                'max': np.max(self.update_times) if self.update_times else 0
            },
            'inference_times': {
                'mean': np.mean(self.inference_times) if self.inference_times else 0,
                'p95': np.percentile(self.inference_times, 95) if self.inference_times else 0,
                'p99': np.percentile(self.inference_times, 99) if self.inference_times else 0
            },
            'model_info': self.model.get_model_info(),
            'training_state': {
                'update_count': self.update_count,
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'current_entropy': self.entropy_scheduler.get_entropy()
            }
        }