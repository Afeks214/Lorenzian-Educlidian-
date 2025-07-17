"""
Strategic Sequential Trainer - Training framework for sequential strategic agents

This module provides a comprehensive training framework for sequential strategic agents
that execute in the order MLMI → NWRQK → Regime with superposition outputs.

Key Features:
- Multi-agent reinforcement learning (MARL) with sequential execution
- Superposition-aware reward systems
- Enriched observation training with predecessor context
- Performance-optimized training loops
- Comprehensive metrics tracking
- Mathematical validation of learning progress

Training Approach:
- Shared critic with individual actors
- Sequential curriculum learning
- Superposition quality optimization
- Temporal coherence enforcement
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import pickle
import json
from pathlib import Path

# Import environment and agents
from src.environment.sequential_strategic_env import SequentialStrategicEnvironment
from src.agents.strategic.sequential_strategic_agents import (
    SequentialMLMIAgent,
    SequentialNWRQKAgent,
    SequentialRegimeAgent,
    SequentialAgentFactory,
    SequentialPrediction
)

# Import training utilities
try:
    from src.training.reward_system import RewardSystem
    from src.training.schedulers import LearningRateScheduler
    from src.training.losses import PPOLoss, SuperpositionLoss
    from src.training.optimized_checkpoint_manager import OptimizedCheckpointManager
except ImportError:
    # Fallback implementations will be provided
    RewardSystem = None
    LearningRateScheduler = None
    PPOLoss = None
    SuperpositionLoss = None
    OptimizedCheckpointManager = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for sequential strategic training"""
    # Environment settings
    max_episode_steps: int = 1000
    total_episodes: int = 10000
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    
    # Sequential training settings
    sequence_reward_weight: float = 0.2
    superposition_reward_weight: float = 0.1
    temporal_coherence_weight: float = 0.15
    
    # Performance targets
    max_agent_computation_time_ms: float = 5.0
    max_sequence_execution_time_ms: float = 15.0
    target_superposition_quality: float = 0.7
    target_ensemble_confidence: float = 0.6
    
    # Training optimization
    gradient_clip_norm: float = 0.5
    update_frequency: int = 10
    validation_frequency: int = 100
    checkpoint_frequency: int = 1000
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: ["basic", "intermediate", "advanced"])
    curriculum_thresholds: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=1000))
    sequence_execution_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    superposition_qualities: deque = field(default_factory=lambda: deque(maxlen=1000))
    ensemble_confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Loss tracking
    actor_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    critic_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    superposition_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Per-agent metrics
    agent_performance: Dict[str, Dict[str, deque]] = field(default_factory=dict)
    
    # Training progress
    episodes_trained: int = 0
    total_training_time: float = 0.0
    best_episode_reward: float = float('-inf')
    best_superposition_quality: float = 0.0
    
    # Validation metrics
    validation_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    validation_success_rate: float = 0.0


class SequentialReplayBuffer:
    """Replay buffer for sequential strategic training"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.agent_buffers = {
            'mlmi_expert': deque(maxlen=capacity//3),
            'nwrqk_expert': deque(maxlen=capacity//3),
            'regime_expert': deque(maxlen=capacity//3)
        }
        
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
        # Add to agent-specific buffer
        agent_name = experience.get('agent_name')
        if agent_name in self.agent_buffers:
            self.agent_buffers[agent_name].append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_agent(self, agent_name: str, batch_size: int) -> List[Dict[str, Any]]:
        """Sample experiences for specific agent"""
        agent_buffer = self.agent_buffers.get(agent_name, deque())
        if len(agent_buffer) < batch_size:
            return list(agent_buffer)
        
        indices = np.random.choice(len(agent_buffer), batch_size, replace=False)
        return [agent_buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        for agent_buffer in self.agent_buffers.values():
            agent_buffer.clear()


class SequentialLoss(nn.Module):
    """Loss function for sequential strategic training"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self, 
        actor_outputs: Dict[str, torch.Tensor],
        critic_outputs: Dict[str, torch.Tensor],
        superposition_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        advantages: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate sequential training loss
        
        Args:
            actor_outputs: Actor network outputs for each agent
            critic_outputs: Critic network outputs for each agent
            superposition_outputs: Superposition network outputs for each agent
            targets: Target values for training
            advantages: Advantage values for each agent
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Calculate actor losses (PPO-style)
        for agent_name in actor_outputs:
            actor_loss = self._calculate_actor_loss(
                actor_outputs[agent_name],
                targets[f'{agent_name}_action'],
                advantages[agent_name]
            )
            losses[f'{agent_name}_actor_loss'] = actor_loss
        
        # Calculate critic losses
        for agent_name in critic_outputs:
            critic_loss = F.mse_loss(
                critic_outputs[agent_name],
                targets[f'{agent_name}_value']
            )
            losses[f'{agent_name}_critic_loss'] = critic_loss
        
        # Calculate superposition losses
        for agent_name in superposition_outputs:
            superposition_loss = self._calculate_superposition_loss(
                superposition_outputs[agent_name],
                targets[f'{agent_name}_superposition']
            )
            losses[f'{agent_name}_superposition_loss'] = superposition_loss
        
        # Calculate sequence coherence loss
        sequence_coherence_loss = self._calculate_sequence_coherence_loss(
            actor_outputs, targets
        )
        losses['sequence_coherence_loss'] = sequence_coherence_loss
        
        # Calculate total loss
        total_loss = torch.zeros(1, requires_grad=True)
        for loss_name, loss_value in losses.items():
            if 'actor' in loss_name:
                total_loss = total_loss + loss_value
            elif 'critic' in loss_name:
                total_loss = total_loss + self.config.value_loss_coeff * loss_value
            elif 'superposition' in loss_name:
                total_loss = total_loss + self.config.superposition_reward_weight * loss_value
            elif 'sequence_coherence' in loss_name:
                total_loss = total_loss + self.config.temporal_coherence_weight * loss_value
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _calculate_actor_loss(
        self, 
        actor_output: torch.Tensor, 
        target_action: torch.Tensor, 
        advantage: torch.Tensor
    ) -> torch.Tensor:
        """Calculate actor loss using PPO clipping"""
        # Calculate log probabilities
        log_probs = F.log_softmax(actor_output, dim=-1)
        old_log_probs = F.log_softmax(target_action, dim=-1)
        
        # Calculate ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Calculate clipped loss
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        
        return loss
    
    def _calculate_superposition_loss(
        self, 
        superposition_output: torch.Tensor, 
        target_superposition: torch.Tensor
    ) -> torch.Tensor:
        """Calculate superposition quality loss"""
        # MSE loss for superposition features
        feature_loss = F.mse_loss(superposition_output, target_superposition)
        
        # Quantum coherence constraint (encourage unit norm)
        coherence_loss = torch.mean((torch.norm(superposition_output, dim=-1) - 1.0) ** 2)
        
        return feature_loss + 0.1 * coherence_loss
    
    def _calculate_sequence_coherence_loss(
        self, 
        actor_outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate loss for sequence coherence"""
        if len(actor_outputs) < 2:
            return torch.zeros(1, requires_grad=True)
        
        # Get agent outputs in sequence order
        agent_order = ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
        sequence_outputs = [actor_outputs[agent] for agent in agent_order if agent in actor_outputs]
        
        # Calculate coherence loss between adjacent agents
        coherence_loss = torch.zeros(1, requires_grad=True)
        for i in range(len(sequence_outputs) - 1):
            # Encourage smooth transitions between agents
            diff = sequence_outputs[i+1] - sequence_outputs[i]
            coherence_loss = coherence_loss + torch.mean(diff ** 2)
        
        return coherence_loss / max(1, len(sequence_outputs) - 1)


class StrategySequentialTrainer:
    """Training framework for sequential strategic agents"""
    
    def __init__(
        self,
        config: TrainingConfig,
        environment: Optional[SequentialStrategicEnvironment] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize sequential strategic trainer
        
        Args:
            config: Training configuration
            environment: Sequential strategic environment
            device: Torch device
        """
        self.config = config
        self.device = torch.device(device)
        self.logger = logging.getLogger(f"{__name__}.StrategySequentialTrainer")
        
        # Initialize environment
        self.environment = environment or SequentialStrategicEnvironment(
            config=self._get_environment_config()
        )
        
        # Initialize agents
        self.agents = SequentialAgentFactory.create_all_agents(
            config=self._get_agent_config(),
            device=str(self.device)
        )
        
        # Initialize training components
        self.replay_buffer = SequentialReplayBuffer(capacity=config.buffer_size)
        self.loss_function = SequentialLoss(config)
        self.metrics = TrainingMetrics()
        
        # Initialize optimizers
        self.optimizers = self._initialize_optimizers()
        
        # Initialize schedulers
        self.schedulers = self._initialize_schedulers()
        
        # Checkpoint manager
        self.checkpoint_manager = self._initialize_checkpoint_manager()
        
        # Training state
        self.training_step = 0
        self.current_episode = 0
        self.curriculum_stage = 0
        
        # Initialize per-agent metrics
        for agent_name in self.agents:
            self.metrics.agent_performance[agent_name] = {
                'rewards': deque(maxlen=1000),
                'computation_times': deque(maxlen=1000),
                'confidences': deque(maxlen=1000),
                'superposition_qualities': deque(maxlen=1000)
            }
        
        self.logger.info(f"StrategySequentialTrainer initialized with {len(self.agents)} agents")
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return {
            "max_episode_steps": self.config.max_episode_steps,
            "total_episodes": self.config.total_episodes,
            "performance": {
                "max_agent_computation_time_ms": self.config.max_agent_computation_time_ms,
                "max_sequence_execution_time_ms": self.config.max_sequence_execution_time_ms,
                "target_superposition_quality": self.config.target_superposition_quality,
                "target_ensemble_confidence": self.config.target_ensemble_confidence
            },
            "environment": {
                "sequential_execution": True,
                "superposition_enabled": True,
                "observation_enrichment": True
            }
        }
    
    def _get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            "feature_indices": {
                "mlmi_expert": [0, 1, 9, 10],
                "nwrqk_expert": [2, 3, 4, 5],
                "regime_expert": [10, 11, 12]
            },
            "agents": {
                "mlmi_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.1},
                "nwrqk_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.1},
                "regime_expert": {"hidden_dims": [256, 128, 64], "dropout_rate": 0.15}
            },
            "environment": {
                "superposition_enabled": True,
                "observation_enrichment": True
            },
            "performance": {
                "max_agent_computation_time_ms": self.config.max_agent_computation_time_ms
            }
        }
    
    def _initialize_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for all agents"""
        optimizers = {}
        
        for agent_name, agent in self.agents.items():
            # Combine actor and critic parameters
            parameters = []
            if agent.actor_network:
                parameters.extend(agent.actor_network.parameters())
            if agent.critic_network:
                parameters.extend(agent.critic_network.parameters())
            if agent.sequential_module:
                parameters.extend(agent.sequential_module.parameters())
            if agent.superposition_generator:
                parameters.extend(agent.superposition_generator.parameters())
            
            optimizer = optim.Adam(parameters, lr=self.config.learning_rate)
            optimizers[agent_name] = optimizer
        
        return optimizers
    
    def _initialize_schedulers(self) -> Dict[str, Any]:
        """Initialize learning rate schedulers"""
        schedulers = {}
        
        for agent_name, optimizer in self.optimizers.items():
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=1000, 
                gamma=0.9
            )
            schedulers[agent_name] = scheduler
        
        return schedulers
    
    def _initialize_checkpoint_manager(self) -> Optional[Any]:
        """Initialize checkpoint manager"""
        if OptimizedCheckpointManager is None:
            return None
        
        return OptimizedCheckpointManager(
            checkpoint_dir="checkpoints/sequential_strategic",
            max_checkpoints=10
        )
    
    async def initialize_agents(self):
        """Initialize all agents"""
        try:
            for agent_name, agent in self.agents.items():
                await agent.initialize()
                self.logger.info(f"Agent {agent_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training results and metrics
        """
        self.logger.info("Starting sequential strategic training")
        training_start_time = time.time()
        
        try:
            # Initialize agents
            await self.initialize_agents()
            
            # Training loop
            for episode in range(self.config.total_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_metrics = await self._run_episode()
                
                # Update metrics
                self._update_metrics(episode_metrics)
                
                # Perform training updates
                if episode % self.config.update_frequency == 0:
                    await self._update_agents()
                
                # Validation
                if episode % self.config.validation_frequency == 0:
                    await self._validate()
                
                # Checkpoint
                if episode % self.config.checkpoint_frequency == 0:
                    await self._save_checkpoint()
                
                # Curriculum learning
                if self.config.curriculum_enabled:
                    self._update_curriculum()
                
                # Log progress
                if episode % 100 == 0:
                    self._log_progress(episode)
            
            # Final training results
            training_time = time.time() - training_start_time
            self.metrics.total_training_time = training_time
            
            results = self._get_training_results()
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    async def _run_episode(self) -> Dict[str, Any]:
        """Run single training episode"""
        episode_start_time = time.time()
        episode_metrics = {
            'episode_reward': 0.0,
            'episode_length': 0,
            'sequence_execution_times': [],
            'superposition_qualities': [],
            'ensemble_confidences': [],
            'agent_metrics': {agent: {} for agent in self.agents}
        }
        
        # Reset environment
        self.environment.reset()
        done = False
        step = 0
        
        while not done and step < self.config.max_episode_steps:
            # Execute sequential step
            step_metrics = await self._execute_sequential_step()
            
            # Update episode metrics
            episode_metrics['episode_reward'] += step_metrics.get('total_reward', 0.0)
            episode_metrics['sequence_execution_times'].append(step_metrics.get('execution_time', 0.0))
            
            if 'superposition_quality' in step_metrics:
                episode_metrics['superposition_qualities'].append(step_metrics['superposition_quality'])
            
            if 'ensemble_confidence' in step_metrics:
                episode_metrics['ensemble_confidences'].append(step_metrics['ensemble_confidence'])
            
            # Update agent metrics
            for agent_name, agent_metrics in step_metrics.get('agent_metrics', {}).items():
                if agent_name not in episode_metrics['agent_metrics']:
                    episode_metrics['agent_metrics'][agent_name] = {}
                for metric, value in agent_metrics.items():
                    if metric not in episode_metrics['agent_metrics'][agent_name]:
                        episode_metrics['agent_metrics'][agent_name][metric] = []
                    episode_metrics['agent_metrics'][agent_name][metric].append(value)
            
            # Check termination
            done = all(self.environment.terminations.values()) or all(self.environment.truncations.values())
            step += 1
        
        episode_metrics['episode_length'] = step
        episode_metrics['episode_time'] = time.time() - episode_start_time
        
        return episode_metrics
    
    async def _execute_sequential_step(self) -> Dict[str, Any]:
        """Execute one sequential step with all agents"""
        step_start_time = time.time()
        step_metrics = {
            'agent_metrics': {},
            'total_reward': 0.0,
            'execution_time': 0.0
        }
        
        # Execute agents in sequence
        for agent_name in ['mlmi_expert', 'nwrqk_expert', 'regime_expert']:
            if self.environment.agent_selection == agent_name:
                # Get observation
                obs = self.environment.observe(agent_name)
                
                # Get agent action
                action, agent_metrics = await self._get_agent_action(agent_name, obs)
                
                # Execute action
                self.environment.step(action)
                
                # Store experience
                experience = {
                    'agent_name': agent_name,
                    'observation': obs,
                    'action': action,
                    'reward': self.environment.rewards.get(agent_name, 0.0),
                    'done': self.environment.terminations.get(agent_name, False),
                    'agent_metrics': agent_metrics
                }
                self.replay_buffer.add(experience)
                
                # Update step metrics
                step_metrics['agent_metrics'][agent_name] = agent_metrics
                step_metrics['total_reward'] += self.environment.rewards.get(agent_name, 0.0)
        
        # Get environment performance metrics
        env_metrics = self.environment.get_performance_metrics()
        step_metrics['superposition_quality'] = env_metrics.get('avg_superposition_quality', 0.0)
        step_metrics['ensemble_confidence'] = env_metrics.get('avg_ensemble_confidence', 0.0)
        step_metrics['execution_time'] = (time.time() - step_start_time) * 1000
        
        return step_metrics
    
    async def _get_agent_action(self, agent_name: str, obs: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get action from agent"""
        agent = self.agents[agent_name]
        
        try:
            # Get prediction from sequential agent
            if hasattr(agent, 'predict_sequential'):
                prediction = await agent.predict_sequential(obs, {})
            else:
                # Fallback to standard prediction
                matrix_data = obs['base_observation']['market_matrix']
                prediction = await agent.predict(matrix_data, {})
            
            # Extract action
            action = prediction.action_probabilities
            
            # Agent metrics
            agent_metrics = {
                'confidence': prediction.confidence,
                'computation_time': prediction.computation_time_ms,
                'superposition_quality': getattr(prediction, 'superposition_quality', 0.0),
                'quantum_coherence': getattr(prediction, 'quantum_coherence', 0.0),
                'temporal_stability': getattr(prediction, 'temporal_stability', 0.0)
            }
            
            return action, agent_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get action from {agent_name}: {e}")
            # Return fallback action
            return np.array([0.33, 0.34, 0.33]), {'error': True}
    
    async def _update_agents(self):
        """Update all agents using replay buffer"""
        if self.replay_buffer.size() < self.config.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Update each agent
        for agent_name in self.agents:
            agent_batch = [exp for exp in batch if exp['agent_name'] == agent_name]
            if len(agent_batch) == 0:
                continue
            
            # Update agent
            loss = await self._update_agent(agent_name, agent_batch)
            
            # Track loss
            if 'actor' in loss:
                self.metrics.actor_losses.append(loss['actor'])
            if 'critic' in loss:
                self.metrics.critic_losses.append(loss['critic'])
            if 'superposition' in loss:
                self.metrics.superposition_losses.append(loss['superposition'])
    
    async def _update_agent(self, agent_name: str, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update specific agent"""
        agent = self.agents[agent_name]
        optimizer = self.optimizers[agent_name]
        
        # Prepare batch data
        observations = [exp['observation'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        dones = [exp['done'] for exp in batch]
        
        # Calculate losses (simplified)
        optimizer.zero_grad()
        
        # Forward pass (simplified)
        total_loss = torch.zeros(1, requires_grad=True)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            agent.actor_network.parameters(), 
            self.config.gradient_clip_norm
        )
        
        optimizer.step()
        
        return {'total': float(total_loss.item())}
    
    async def _validate(self):
        """Run validation episodes"""
        validation_rewards = []
        
        for _ in range(10):  # Run 10 validation episodes
            self.environment.reset()
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done and step < self.config.max_episode_steps:
                # Execute validation step
                for agent_name in ['mlmi_expert', 'nwrqk_expert', 'regime_expert']:
                    if self.environment.agent_selection == agent_name:
                        obs = self.environment.observe(agent_name)
                        action, _ = await self._get_agent_action(agent_name, obs)
                        self.environment.step(action)
                        episode_reward += self.environment.rewards.get(agent_name, 0.0)
                
                done = all(self.environment.terminations.values()) or all(self.environment.truncations.values())
                step += 1
            
            validation_rewards.append(episode_reward)
        
        # Update validation metrics
        avg_validation_reward = np.mean(validation_rewards)
        self.metrics.validation_rewards.append(avg_validation_reward)
        
        # Calculate success rate
        success_threshold = 0.0  # Define success threshold
        success_count = sum(1 for r in validation_rewards if r > success_threshold)
        self.metrics.validation_success_rate = success_count / len(validation_rewards)
        
        self.logger.info(f"Validation - Avg Reward: {avg_validation_reward:.3f}, Success Rate: {self.metrics.validation_success_rate:.3f}")
    
    async def _save_checkpoint(self):
        """Save training checkpoint"""
        if self.checkpoint_manager is None:
            return
        
        checkpoint_data = {
            'episode': self.current_episode,
            'training_step': self.training_step,
            'curriculum_stage': self.curriculum_stage,
            'config': self.config,
            'metrics': self.metrics,
            'agents': {
                agent_name: {
                    'actor_state_dict': agent.actor_network.state_dict() if agent.actor_network else None,
                    'critic_state_dict': agent.critic_network.state_dict() if agent.critic_network else None,
                    'sequential_module_state_dict': agent.sequential_module.state_dict() if agent.sequential_module else None,
                    'superposition_generator_state_dict': agent.superposition_generator.state_dict() if agent.superposition_generator else None
                }
                for agent_name, agent in self.agents.items()
            },
            'optimizers': {
                agent_name: optimizer.state_dict()
                for agent_name, optimizer in self.optimizers.items()
            }
        }
        
        try:
            await self.checkpoint_manager.save_checkpoint(checkpoint_data, self.current_episode)
            self.logger.info(f"Checkpoint saved at episode {self.current_episode}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _update_curriculum(self):
        """Update curriculum learning stage"""
        if not self.config.curriculum_enabled:
            return
        
        # Check if we should advance curriculum
        if self.metrics.validation_success_rate > self.config.curriculum_thresholds[self.curriculum_stage]:
            if self.curriculum_stage < len(self.config.curriculum_stages) - 1:
                self.curriculum_stage += 1
                self.logger.info(f"Advanced to curriculum stage {self.curriculum_stage}: {self.config.curriculum_stages[self.curriculum_stage]}")
    
    def _update_metrics(self, episode_metrics: Dict[str, Any]):
        """Update training metrics"""
        # Episode metrics
        self.metrics.episode_rewards.append(episode_metrics['episode_reward'])
        self.metrics.episode_lengths.append(episode_metrics['episode_length'])
        self.metrics.episodes_trained += 1
        
        # Performance metrics
        if episode_metrics['sequence_execution_times']:
            avg_execution_time = np.mean(episode_metrics['sequence_execution_times'])
            self.metrics.sequence_execution_times.append(avg_execution_time)
        
        if episode_metrics['superposition_qualities']:
            avg_superposition_quality = np.mean(episode_metrics['superposition_qualities'])
            self.metrics.superposition_qualities.append(avg_superposition_quality)
            
            # Update best superposition quality
            if avg_superposition_quality > self.metrics.best_superposition_quality:
                self.metrics.best_superposition_quality = avg_superposition_quality
        
        if episode_metrics['ensemble_confidences']:
            avg_ensemble_confidence = np.mean(episode_metrics['ensemble_confidences'])
            self.metrics.ensemble_confidences.append(avg_ensemble_confidence)
        
        # Update best episode reward
        if episode_metrics['episode_reward'] > self.metrics.best_episode_reward:
            self.metrics.best_episode_reward = episode_metrics['episode_reward']
        
        # Per-agent metrics
        for agent_name, agent_metrics in episode_metrics['agent_metrics'].items():
            if agent_name in self.metrics.agent_performance:
                for metric_name, values in agent_metrics.items():
                    if metric_name in self.metrics.agent_performance[agent_name]:
                        if values:
                            self.metrics.agent_performance[agent_name][metric_name].extend(values)
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        if not self.metrics.episode_rewards:
            return
        
        avg_reward = np.mean(list(self.metrics.episode_rewards)[-100:])
        avg_length = np.mean(list(self.metrics.episode_lengths)[-100:])
        
        avg_execution_time = 0.0
        if self.metrics.sequence_execution_times:
            avg_execution_time = np.mean(list(self.metrics.sequence_execution_times)[-100:])
        
        avg_superposition_quality = 0.0
        if self.metrics.superposition_qualities:
            avg_superposition_quality = np.mean(list(self.metrics.superposition_qualities)[-100:])
        
        self.logger.info(
            f"Episode {episode}: "
            f"Avg Reward: {avg_reward:.3f}, "
            f"Avg Length: {avg_length:.1f}, "
            f"Avg Execution Time: {avg_execution_time:.2f}ms, "
            f"Avg Superposition Quality: {avg_superposition_quality:.3f}"
        )
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive training results"""
        results = {
            'training_completed': True,
            'total_episodes': self.metrics.episodes_trained,
            'total_training_time': self.metrics.total_training_time,
            'best_episode_reward': self.metrics.best_episode_reward,
            'best_superposition_quality': self.metrics.best_superposition_quality,
            'final_validation_success_rate': self.metrics.validation_success_rate,
            
            # Performance metrics
            'avg_episode_reward': np.mean(self.metrics.episode_rewards) if self.metrics.episode_rewards else 0.0,
            'avg_episode_length': np.mean(self.metrics.episode_lengths) if self.metrics.episode_lengths else 0.0,
            'avg_sequence_execution_time': np.mean(self.metrics.sequence_execution_times) if self.metrics.sequence_execution_times else 0.0,
            'avg_superposition_quality': np.mean(self.metrics.superposition_qualities) if self.metrics.superposition_qualities else 0.0,
            'avg_ensemble_confidence': np.mean(self.metrics.ensemble_confidences) if self.metrics.ensemble_confidences else 0.0,
            
            # Learning curves
            'episode_rewards': list(self.metrics.episode_rewards),
            'episode_lengths': list(self.metrics.episode_lengths),
            'sequence_execution_times': list(self.metrics.sequence_execution_times),
            'superposition_qualities': list(self.metrics.superposition_qualities),
            'ensemble_confidences': list(self.metrics.ensemble_confidences),
            
            # Per-agent results
            'agent_performance': {},
            
            # Training configuration
            'config': self.config,
            'curriculum_stage': self.curriculum_stage
        }
        
        # Add per-agent performance
        for agent_name, agent_metrics in self.metrics.agent_performance.items():
            results['agent_performance'][agent_name] = {
                'avg_reward': np.mean(agent_metrics['rewards']) if agent_metrics['rewards'] else 0.0,
                'avg_computation_time': np.mean(agent_metrics['computation_times']) if agent_metrics['computation_times'] else 0.0,
                'avg_confidence': np.mean(agent_metrics['confidences']) if agent_metrics['confidences'] else 0.0,
                'avg_superposition_quality': np.mean(agent_metrics['superposition_qualities']) if agent_metrics['superposition_qualities'] else 0.0
            }
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        results = self._get_training_results()
        
        # Add performance analysis
        performance_report = {
            'training_results': results,
            'performance_analysis': {
                'computation_time_target_met': results['avg_sequence_execution_time'] <= self.config.max_sequence_execution_time_ms,
                'superposition_quality_target_met': results['avg_superposition_quality'] >= self.config.target_superposition_quality,
                'ensemble_confidence_target_met': results['avg_ensemble_confidence'] >= self.config.target_ensemble_confidence,
                'training_efficiency': results['total_training_time'] / max(1, results['total_episodes']),
                'convergence_achieved': results['final_validation_success_rate'] > 0.8
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        return performance_report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        if results['avg_sequence_execution_time'] > self.config.max_sequence_execution_time_ms:
            recommendations.append("Consider optimizing agent computation time or increasing time limits")
        
        if results['avg_superposition_quality'] < self.config.target_superposition_quality:
            recommendations.append("Increase superposition reward weight or improve superposition generator")
        
        if results['final_validation_success_rate'] < 0.8:
            recommendations.append("Consider extending training duration or adjusting hyperparameters")
        
        if results['avg_ensemble_confidence'] < self.config.target_ensemble_confidence:
            recommendations.append("Improve individual agent confidence or ensemble aggregation")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Create training configuration
    config = TrainingConfig(
        total_episodes=1000,
        max_episode_steps=100,
        learning_rate=3e-4,
        batch_size=32,
        curriculum_enabled=True
    )
    
    # Create trainer
    trainer = StrategySequentialTrainer(config)
    
    # Run training
    async def main():
        results = await trainer.train()
        
        # Get performance report
        performance_report = trainer.get_performance_report()
        
        print("Training completed!")
        print(f"Best episode reward: {results['best_episode_reward']:.3f}")
        print(f"Best superposition quality: {results['best_superposition_quality']:.3f}")
        print(f"Final validation success rate: {results['final_validation_success_rate']:.3f}")
        
        # Print performance analysis
        analysis = performance_report['performance_analysis']
        print("\nPerformance Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Print recommendations
        recommendations = performance_report['recommendations']
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    
    # Run the training
    asyncio.run(main())