"""
PettingZoo-Compatible MAPPO Trainer for MARL System

This module implements a Multi-Agent PPO trainer specifically designed to work
with PettingZoo environments, handling the turn-based execution model and
proper agent lifecycle management.

Key Features:
- Full PettingZoo AEC environment compatibility
- Turn-based agent execution with proper state management
- Centralized training with decentralized execution (CTDE)
- Advanced replay buffer management for multi-agent systems
- Gradient synchronization across agents
- Performance monitoring and metrics collection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from pathlib import Path
import logging
from datetime import datetime
import json
import time
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor
import copy

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PettingZoo MAPPO training"""
    # Environment configuration
    env_factory: Optional[callable] = None
    env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training settings
    num_episodes: int = 1000
    max_episode_steps: int = 1000
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 1
    
    # Multi-agent settings
    shared_policy: bool = True
    independent_critics: bool = True
    communication_enabled: bool = False
    
    # Logging and checkpointing
    log_dir: str = "logs/pettingzoo_mappo"
    save_frequency: int = 100
    eval_frequency: int = 50
    render_frequency: int = 0
    
    # Device configuration
    device: str = "auto"
    num_threads: int = 4
    
    # Performance settings
    early_stopping_patience: int = 100
    target_reward: float = float('inf')
    convergence_threshold: float = 0.01


@dataclass
class AgentExperience:
    """Container for agent experience data"""
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    def __post_init__(self):
        self.length = len(self.observations)
    
    def clear(self):
        """Clear all experience data"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.infos.clear()
        self.length = 0


class PettingZooReplayBuffer:
    """Replay buffer designed for PettingZoo environments"""
    
    def __init__(self, capacity: int, num_agents: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        
        # Initialize storage
        self.buffer = {
            'observations': np.zeros((capacity, *obs_shape), dtype=np.float32),
            'actions': np.zeros(capacity, dtype=np.int64),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'values': np.zeros(capacity, dtype=np.float32),
            'log_probs': np.zeros(capacity, dtype=np.float32),
            'dones': np.zeros(capacity, dtype=np.bool_),
            'agents': np.zeros(capacity, dtype=np.int32),
            'episode_ids': np.zeros(capacity, dtype=np.int32),
            'step_ids': np.zeros(capacity, dtype=np.int32)
        }
        
        self.position = 0
        self.size = 0
        self.episode_id = 0
        
    def add(self, obs: np.ndarray, action: int, reward: float, value: float,
            log_prob: float, done: bool, agent_id: int, step_id: int):
        """Add experience to buffer"""
        self.buffer['observations'][self.position] = obs
        self.buffer['actions'][self.position] = action
        self.buffer['rewards'][self.position] = reward
        self.buffer['values'][self.position] = value
        self.buffer['log_probs'][self.position] = log_prob
        self.buffer['dones'][self.position] = done
        self.buffer['agents'][self.position] = agent_id
        self.buffer['episode_ids'][self.position] = self.episode_id
        self.buffer['step_ids'][self.position] = step_id
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        if done:
            self.episode_id += 1
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {}
        for key, data in self.buffer.items():
            if key in ['observations']:
                batch[key] = torch.FloatTensor(data[indices])
            elif key in ['actions', 'agents', 'episode_ids', 'step_ids']:
                batch[key] = torch.LongTensor(data[indices])
            elif key in ['dones']:
                batch[key] = torch.BoolTensor(data[indices])
            else:
                batch[key] = torch.FloatTensor(data[indices])
        
        return batch
    
    def get_agent_experiences(self, agent_id: int) -> Dict[str, np.ndarray]:
        """Get all experiences for a specific agent"""
        if self.size == 0:
            return {}
        
        mask = self.buffer['agents'][:self.size] == agent_id
        
        experiences = {}
        for key, data in self.buffer.items():
            experiences[key] = data[:self.size][mask]
        
        return experiences
    
    def clear(self):
        """Clear the buffer"""
        self.position = 0
        self.size = 0
        self.episode_id = 0


class PettingZooMAPPOTrainer:
    """
    MAPPO Trainer specifically designed for PettingZoo environments
    
    This trainer handles the turn-based nature of PettingZoo environments
    while implementing centralized training with decentralized execution.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize environment
        self.env = self._create_environment()
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        self.agent_to_id = {agent: i for i, agent in enumerate(self.agents)}
        
        # Get environment spaces
        self.obs_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])
        
        # Initialize networks
        self.networks = self._initialize_networks()
        self.optimizers = self._initialize_optimizers()
        
        # Initialize replay buffer
        obs_shape = self.obs_space.shape
        self.replay_buffer = PettingZooReplayBuffer(
            capacity=config.buffer_size,
            num_agents=self.num_agents,
            obs_shape=obs_shape
        )
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.total_timesteps = 0
        self.best_reward = -float('inf')
        self.recent_rewards = deque(maxlen=100)
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'grad_norms': [],
            'learning_rates': [],
            'agent_rewards': {agent: [] for agent in self.agents},
            'convergence_metrics': [],
            'training_time': []
        }
        
        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        
        logger.info(f"PettingZoo MAPPO Trainer initialized with {self.num_agents} agents")
        logger.info(f"Environment: {type(self.env).__name__}")
        logger.info(f"Observation space: {self.obs_space}")
        logger.info(f"Action space: {self.action_space}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _create_environment(self) -> AECEnv:
        """Create PettingZoo environment"""
        if self.config.env_factory is None:
            raise ValueError("Environment factory must be provided in config")
        
        env = self.config.env_factory(**self.config.env_config)
        
        # Validate that it's a proper PettingZoo environment
        if not hasattr(env, 'agent_selection') or not hasattr(env, 'step'):
            raise ValueError("Environment must be a valid PettingZoo AEC environment")
        
        return env
    
    def _initialize_networks(self) -> Dict[str, nn.Module]:
        """Initialize policy and value networks"""
        networks = {}
        
        # Get observation shape
        if hasattr(self.obs_space, 'shape'):
            obs_dim = np.prod(self.obs_space.shape)
        else:
            obs_dim = self.obs_space.n
        
        # Get action dimension
        if hasattr(self.action_space, 'n'):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space.shape[0]
        
        if self.config.shared_policy:
            # Shared policy for all agents
            networks['shared_policy'] = self._create_policy_network(obs_dim, action_dim)
            
            if self.config.independent_critics:
                # Independent critics for each agent
                for agent in self.agents:
                    networks[f'critic_{agent}'] = self._create_value_network(obs_dim)
            else:
                # Shared critic
                networks['shared_critic'] = self._create_value_network(obs_dim)
        else:
            # Independent policies for each agent
            for agent in self.agents:
                networks[f'policy_{agent}'] = self._create_policy_network(obs_dim, action_dim)
                networks[f'critic_{agent}'] = self._create_value_network(obs_dim)
        
        # Move networks to device
        for network in networks.values():
            network.to(self.device)
        
        return networks
    
    def _create_policy_network(self, obs_dim: int, action_dim: int) -> nn.Module:
        """Create policy network"""
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def _create_value_network(self, obs_dim: int) -> nn.Module:
        """Create value network"""
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _initialize_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for networks"""
        optimizers = {}
        
        for name, network in self.networks.items():
            optimizers[name] = optim.Adam(
                network.parameters(),
                lr=self.config.learning_rate,
                eps=1e-5
            )
        
        return optimizers
    
    def get_action_and_value(self, obs: np.ndarray, agent: str) -> Tuple[int, float, float]:
        """Get action and value for agent"""
        # Prepare observation
        if len(obs.shape) == 1:
            obs_tensor = torch.FloatTensor(obs).flatten().unsqueeze(0).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(obs).flatten().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get policy
            if self.config.shared_policy:
                policy_logits = self.networks['shared_policy'](obs_tensor)
            else:
                policy_logits = self.networks[f'policy_{agent}'](obs_tensor)
            
            # Get value
            if self.config.independent_critics:
                value = self.networks[f'critic_{agent}'](obs_tensor)
            else:
                value = self.networks['shared_critic'](obs_tensor)
            
            # Sample action
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_episodes} episodes")
        
        training_start_time = time.time()
        
        try:
            for episode in range(self.config.num_episodes):
                episode_start_time = time.time()
                
                # Run episode
                episode_result = self._run_episode()
                
                # Update metrics
                episode_time = time.time() - episode_start_time
                self._update_metrics(episode_result, episode_time)
                
                # Training update
                if (episode + 1) % self.config.update_frequency == 0:
                    self._update_networks()
                
                # Logging
                if (episode + 1) % 10 == 0:
                    self._log_progress(episode)
                
                # Evaluation
                if (episode + 1) % self.config.eval_frequency == 0:
                    self._evaluate()
                
                # Save checkpoint
                if (episode + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Convergence achieved at episode {episode}")
                    break
                
                # Render if requested
                if self.config.render_frequency > 0 and (episode + 1) % self.config.render_frequency == 0:
                    self._render_episode()
            
            training_time = time.time() - training_start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return self._get_training_results(training_time)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._get_training_results(time.time() - training_start_time)
        
        finally:
            self._cleanup()
    
    def _run_episode(self) -> Dict[str, Any]:
        """Run single episode with PettingZoo environment"""
        # Reset environment
        self.env.reset()
        
        episode_rewards = {agent: 0.0 for agent in self.agents}
        episode_length = 0
        agent_step_counts = {agent: 0 for agent in self.agents}
        
        # Episode loop
        while self.env.agents:
            # Get current agent
            current_agent = self.env.agent_selection
            
            # Observe
            observation = self.env.observe(current_agent)
            
            # Get action and value
            action, log_prob, value = self.get_action_and_value(observation, current_agent)
            
            # Store experience before step
            agent_id = self.agent_to_id[current_agent]
            
            # Step environment
            self.env.step(action)
            
            # Get reward and done status
            reward = self.env.rewards.get(current_agent, 0.0)
            done = self.env.dones.get(current_agent, False)
            truncated = self.env.truncations.get(current_agent, False)
            
            # Store experience
            self.replay_buffer.add(
                obs=observation.flatten() if len(observation.shape) > 1 else observation,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done or truncated,
                agent_id=agent_id,
                step_id=episode_length
            )
            
            # Update metrics
            episode_rewards[current_agent] += reward
            agent_step_counts[current_agent] += 1
            episode_length += 1
            self.step_count += 1
            
            # Check episode termination
            if done or truncated or episode_length >= self.config.max_episode_steps:
                break
        
        self.episode_count += 1
        
        return {
            'episode_rewards': episode_rewards,
            'total_reward': sum(episode_rewards.values()),
            'episode_length': episode_length,
            'agent_step_counts': agent_step_counts
        }
    
    def _update_networks(self):
        """Update policy and value networks using PPO"""
        if self.replay_buffer.size < self.config.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(min(self.config.batch_size, self.replay_buffer.size))
        
        # Convert to tensors
        obs_batch = batch['observations'].to(self.device)
        action_batch = batch['actions'].to(self.device)
        reward_batch = batch['rewards'].to(self.device)
        value_batch = batch['values'].to(self.device)
        old_log_prob_batch = batch['log_probs'].to(self.device)
        done_batch = batch['dones'].to(self.device)
        agent_batch = batch['agents'].to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(reward_batch, value_batch, done_batch)
        
        # PPO update
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            batch_size = len(obs_batch)
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.config.batch_size):
                end = min(start + self.config.batch_size, batch_size)
                mini_batch_indices = indices[start:end]
                
                # Get mini-batch
                mini_obs = obs_batch[mini_batch_indices]
                mini_actions = action_batch[mini_batch_indices]
                mini_old_log_probs = old_log_prob_batch[mini_batch_indices]
                mini_advantages = advantages[mini_batch_indices]
                mini_returns = returns[mini_batch_indices]
                mini_agents = agent_batch[mini_batch_indices]
                
                # Compute new action probabilities and values
                new_log_probs, new_values, entropy = self._evaluate_actions(
                    mini_obs, mini_actions, mini_agents
                )
                
                # PPO loss
                policy_loss, value_loss = self._compute_ppo_loss(
                    new_log_probs, mini_old_log_probs, mini_advantages,
                    new_values, mini_returns, entropy
                )
                
                total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy.mean()
                
                # Update networks
                self._update_parameters(total_loss)
                
                # Store metrics
                self.metrics['policy_losses'].append(policy_loss.item())
                self.metrics['value_losses'].append(value_loss.item())
                self.metrics['entropy_losses'].append(entropy.mean().item())
    
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                           dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, 
                         agents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for current policies"""
        batch_size = len(observations)
        log_probs = torch.zeros(batch_size, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        entropies = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            obs = observations[i].unsqueeze(0)
            action = actions[i]
            agent_id = agents[i].item()
            agent_name = self.agents[agent_id]
            
            # Get policy logits
            if self.config.shared_policy:
                policy_logits = self.networks['shared_policy'](obs)
            else:
                policy_logits = self.networks[f'policy_{agent_name}'](obs)
            
            # Get value
            if self.config.independent_critics:
                value = self.networks[f'critic_{agent_name}'](obs)
            else:
                value = self.networks['shared_critic'](obs)
            
            # Compute log probability and entropy
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            log_probs[i] = log_prob
            values[i] = value.squeeze()
            entropies[i] = entropy
        
        return log_probs, values, entropies
    
    def _compute_ppo_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                         advantages: torch.Tensor, new_values: torch.Tensor,
                         returns: torch.Tensor, entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PPO policy and value losses"""
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        return policy_loss, value_loss
    
    def _update_parameters(self, loss: torch.Tensor):
        """Update network parameters"""
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        total_norm = 0
        for network in self.networks.values():
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
            total_norm += norm
        
        # Update parameters
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        # Store gradient norm
        self.metrics['grad_norms'].append(total_norm)
    
    def _update_metrics(self, episode_result: Dict[str, Any], episode_time: float):
        """Update training metrics"""
        total_reward = episode_result['total_reward']
        episode_length = episode_result['episode_length']
        
        # Update metrics
        self.metrics['episode_rewards'].append(total_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['training_time'].append(episode_time)
        
        # Update agent-specific rewards
        for agent, reward in episode_result['episode_rewards'].items():
            self.metrics['agent_rewards'][agent].append(reward)
        
        # Update best reward
        if total_reward > self.best_reward:
            self.best_reward = total_reward
        
        # Update recent rewards for convergence checking
        self.recent_rewards.append(total_reward)
        
        # Compute convergence metric
        if len(self.recent_rewards) >= 50:
            convergence_metric = np.std(list(self.recent_rewards)[-50:])
            self.metrics['convergence_metrics'].append(convergence_metric)
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        if not self.metrics['episode_rewards']:
            return
        
        recent_rewards = self.metrics['episode_rewards'][-10:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        logger.info(f"Episode {episode}: "
                   f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}, "
                   f"Best: {self.best_reward:.3f}, "
                   f"Steps: {self.step_count}")
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Training/EpisodeReward', mean_reward, episode)
            self.writer.add_scalar('Training/BestReward', self.best_reward, episode)
            self.writer.add_scalar('Training/EpisodeLength', 
                                 np.mean(self.metrics['episode_lengths'][-10:]), episode)
            
            # Agent-specific rewards
            for agent in self.agents:
                if self.metrics['agent_rewards'][agent]:
                    agent_reward = np.mean(self.metrics['agent_rewards'][agent][-10:])
                    self.writer.add_scalar(f'Agent/{agent}_Reward', agent_reward, episode)
    
    def _evaluate(self):
        """Evaluate current policy"""
        logger.info("Running evaluation...")
        
        eval_rewards = []
        eval_lengths = []
        
        # Run evaluation episodes
        for _ in range(10):
            self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while self.env.agents:
                current_agent = self.env.agent_selection
                observation = self.env.observe(current_agent)
                
                # Get action (no exploration)
                action, _, _ = self.get_action_and_value(observation, current_agent)
                
                self.env.step(action)
                
                reward = self.env.rewards.get(current_agent, 0.0)
                episode_reward += reward
                episode_length += 1
                
                if self.env.dones.get(current_agent, False) or episode_length >= self.config.max_episode_steps:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        mean_eval_reward = np.mean(eval_rewards)
        mean_eval_length = np.mean(eval_lengths)
        
        logger.info(f"Evaluation - Mean Reward: {mean_eval_reward:.3f}, "
                   f"Mean Length: {mean_eval_length:.1f}")
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Evaluation/MeanReward', mean_eval_reward, self.episode_count)
            self.writer.add_scalar('Evaluation/MeanLength', mean_eval_length, self.episode_count)
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.recent_rewards) < self.config.early_stopping_patience:
            return False
        
        # Check if recent performance is stable and above target
        recent_rewards = list(self.recent_rewards)[-self.config.early_stopping_patience:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        # Convergence criteria
        reward_threshold_met = mean_reward >= self.config.target_reward
        stability_achieved = std_reward < self.config.convergence_threshold
        
        return reward_threshold_met and stability_achieved
    
    def _render_episode(self):
        """Render a single episode"""
        logger.info("Rendering episode...")
        
        self.env.reset()
        
        while self.env.agents:
            current_agent = self.env.agent_selection
            observation = self.env.observe(current_agent)
            
            action, _, _ = self.get_action_and_value(observation, current_agent)
            
            self.env.step(action)
            self.env.render()
            
            if self.env.dones.get(current_agent, False):
                break
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'networks': {name: net.state_dict() for name, net in self.networks.items()},
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'metrics': self.metrics,
            'config': self.config,
            'best_reward': self.best_reward,
            'step_count': self.step_count
        }
        
        checkpoint_path = self.log_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _get_training_results(self, training_time: float) -> Dict[str, Any]:
        """Get comprehensive training results"""
        return {
            'training_time': training_time,
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'best_reward': self.best_reward,
            'final_reward': np.mean(self.metrics['episode_rewards'][-10:]) if self.metrics['episode_rewards'] else 0,
            'metrics': self.metrics,
            'convergence_achieved': self._check_convergence(),
            'config': self.config
        }
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up trainer resources...")
        
        # Close environment
        if hasattr(self.env, 'close'):
            self.env.close()
        
        # Close writer
        if self.writer:
            self.writer.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load network states
        for name, state_dict in checkpoint['networks'].items():
            if name in self.networks:
                self.networks[name].load_state_dict(state_dict)
        
        # Load optimizer states
        for name, state_dict in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)
        
        # Load training state
        self.episode_count = checkpoint['episode']
        self.step_count = checkpoint['step_count']
        self.best_reward = checkpoint['best_reward']
        self.metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def create_trainer_config(**kwargs) -> TrainingConfig:
    """Create trainer configuration with defaults"""
    return TrainingConfig(**kwargs)


def create_pettingzoo_trainer(env_factory: callable, config: Optional[TrainingConfig] = None) -> PettingZooMAPPOTrainer:
    """Create PettingZoo MAPPO trainer"""
    if config is None:
        config = TrainingConfig()
    
    config.env_factory = env_factory
    return PettingZooMAPPOTrainer(config)


# Example usage
if __name__ == "__main__":
    # This would be used with a PettingZoo environment
    def dummy_env_factory():
        # This would return a real PettingZoo environment
        pass
    
    # Create trainer
    config = create_trainer_config(
        num_episodes=1000,
        learning_rate=3e-4,
        log_dir="logs/pettingzoo_example"
    )
    
    trainer = create_pettingzoo_trainer(dummy_env_factory, config)
    
    # Start training
    results = trainer.train()
    
    print(f"Training completed! Best reward: {results['best_reward']:.3f}")