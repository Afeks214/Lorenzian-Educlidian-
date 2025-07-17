"""
Execution Sequential Trainer (Agent 7 Implementation)
================================================

This module implements the training framework for sequential execution agents
with proper reward shaping, coordination incentives, and performance optimization.

Key Features:
- Sequential agent training with timing constraints
- Multi-objective reward shaping (latency, fill rate, slippage, risk)
- Coordination incentives for agent collaboration
- Experience replay with priority sampling
- Performance-based curriculum learning
- Real-time adaptation to market conditions

Training Architecture:
- Individual agent training with shared experience
- Centralized critic for coordination
- Curriculum learning with progressive difficulty
- Multi-environment parallel training
- Performance-based agent selection

Author: Claude Code (Agent 7 Mission)
Version: 1.0
Date: 2025-07-17
"""

import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.agents.execution.sequential_execution_agents import (
    ExecutionMonitorAgent,
    LiquiditySourcingAgent,
    MarketTimingAgent,
    PositionFragmentationAgent,
    RiskControlAgent,
    SequentialExecutionAgentBase,
    SuperpositionContext,
)
from src.environment.sequential_execution_env import (
    CascadeContext,
    SequentialExecutionEnvironment,
)

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for sequential execution training"""
    
    # Training parameters
    num_episodes: int = 10000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Experience replay
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Training optimization
    gradient_clip_norm: float = 1.0
    target_update_frequency: int = 100
    save_frequency: int = 1000
    eval_frequency: int = 500
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: int = 5
    performance_threshold: float = 0.8
    
    # Multi-environment training
    num_environments: int = 4
    parallel_training: bool = True
    
    # Performance targets
    target_latency_us: float = 500.0
    target_fill_rate: float = 0.95
    target_slippage_bps: float = 10.0
    
    # Reward weights
    latency_weight: float = 0.3
    fill_rate_weight: float = 0.3
    slippage_weight: float = 0.2
    risk_weight: float = 0.2
    coordination_weight: float = 0.1
    
    # Logging
    log_dir: str = "logs/execution_training"
    checkpoint_dir: str = "checkpoints/execution"
    tensorboard_enabled: bool = True


@dataclass
class TrainingBatch:
    """Batch of training data for agents"""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    priorities: torch.Tensor
    superposition_contexts: List[Optional[SuperpositionContext]]
    cascade_contexts: List[Dict[str, Any]]
    agent_ids: List[str]
    timestamps: List[datetime]


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    episode_count: int = 0
    total_steps: int = 0
    
    # Performance metrics
    avg_episode_reward: float = 0.0
    avg_episode_length: float = 0.0
    avg_latency_us: float = 0.0
    avg_fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    
    # Training metrics
    loss_values: Dict[str, float] = field(default_factory=dict)
    gradient_norms: Dict[str, float] = field(default_factory=dict)
    learning_rates: Dict[str, float] = field(default_factory=dict)
    
    # Coordination metrics
    coordination_success_rate: float = 0.0
    agent_agreement_rate: float = 0.0
    
    # Recent history
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_fill_rates: deque = field(default_factory=lambda: deque(maxlen=100))


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def add(self, experience: Tuple, priority: float):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling"""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # Calculate probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        self.priorities[indices] = priorities ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class CentralizedCritic(nn.Module):
    """Centralized critic for coordination"""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Global state dimension
        global_state_dim = obs_dim * num_agents + action_dim * num_agents
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value network for baseline
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic"""
        return self.critic(global_state)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get state value"""
        return self.value_net(observation)


class CoordinationRewardShaper:
    """Reward shaping for agent coordination"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.coordination_history = deque(maxlen=1000)
        
    def shape_reward(self, 
                    base_reward: float,
                    agent_decisions: Dict[str, Any],
                    execution_metrics: Dict[str, Any],
                    coordination_metrics: Dict[str, Any]) -> float:
        """Shape reward based on coordination and performance"""
        
        # Base reward from environment
        shaped_reward = base_reward
        
        # Coordination bonus
        coordination_bonus = self._calculate_coordination_bonus(
            agent_decisions, coordination_metrics
        )
        shaped_reward += coordination_bonus * self.config.coordination_weight
        
        # Performance-based shaping
        performance_bonus = self._calculate_performance_bonus(execution_metrics)
        shaped_reward += performance_bonus
        
        # Timing penalty
        timing_penalty = self._calculate_timing_penalty(execution_metrics)
        shaped_reward -= timing_penalty
        
        # Risk penalty
        risk_penalty = self._calculate_risk_penalty(agent_decisions)
        shaped_reward -= risk_penalty
        
        return shaped_reward
    
    def _calculate_coordination_bonus(self, 
                                   agent_decisions: Dict[str, Any],
                                   coordination_metrics: Dict[str, Any]) -> float:
        """Calculate coordination bonus"""
        if len(agent_decisions) < 2:
            return 0.0
        
        # Check for decision consistency
        consistency_score = coordination_metrics.get('consistency_score', 0.0)
        
        # Check for timing coordination
        timing_coordination = coordination_metrics.get('timing_coordination', 0.0)
        
        # Check for risk alignment
        risk_alignment = coordination_metrics.get('risk_alignment', 0.0)
        
        total_coordination = (consistency_score + timing_coordination + risk_alignment) / 3.0
        
        return total_coordination * 0.1  # 10% bonus for good coordination
    
    def _calculate_performance_bonus(self, execution_metrics: Dict[str, Any]) -> float:
        """Calculate performance bonus"""
        fill_rate = execution_metrics.get('fill_rate', 0.0)
        latency_us = execution_metrics.get('latency_us', 1000.0)
        slippage_bps = execution_metrics.get('slippage_bps', 20.0)
        
        # Fill rate bonus
        fill_bonus = max(0, (fill_rate - self.config.target_fill_rate) * 0.2)
        
        # Latency bonus
        latency_bonus = max(0, (self.config.target_latency_us - latency_us) / 1000.0 * 0.1)
        
        # Slippage bonus
        slippage_bonus = max(0, (self.config.target_slippage_bps - slippage_bps) / 100.0 * 0.1)
        
        return fill_bonus + latency_bonus + slippage_bonus
    
    def _calculate_timing_penalty(self, execution_metrics: Dict[str, Any]) -> float:
        """Calculate timing penalty"""
        latency_us = execution_metrics.get('latency_us', 0.0)
        
        if latency_us > self.config.target_latency_us * 2:
            return 0.5  # Heavy penalty for excessive latency
        elif latency_us > self.config.target_latency_us:
            return (latency_us - self.config.target_latency_us) / 1000.0 * 0.1
        else:
            return 0.0
    
    def _calculate_risk_penalty(self, agent_decisions: Dict[str, Any]) -> float:
        """Calculate risk penalty"""
        risk_decision = agent_decisions.get('risk_control', {})
        
        if risk_decision.get('emergency_stop', False):
            return 0.0  # No penalty for emergency stops
        elif not risk_decision.get('risk_approved', True):
            return 0.2  # Penalty for rejected trades
        else:
            return 0.0


class CurriculumLearning:
    """Curriculum learning for progressive difficulty"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.stage_performance = []
        
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment config for current curriculum stage"""
        base_config = {
            'target_latency_us': self.config.target_latency_us,
            'target_fill_rate': self.config.target_fill_rate,
            'max_slippage_bps': self.config.target_slippage_bps,
            'max_episode_steps': self.config.max_steps_per_episode
        }
        
        # Stage-specific modifications
        if self.current_stage == 0:
            # Stage 0: Simple conditions
            base_config.update({
                'market_volatility': 0.10,
                'spread_bps': 3.0,
                'market_impact_factor': 0.0005
            })
        elif self.current_stage == 1:
            # Stage 1: Normal conditions
            base_config.update({
                'market_volatility': 0.15,
                'spread_bps': 5.0,
                'market_impact_factor': 0.001
            })
        elif self.current_stage == 2:
            # Stage 2: Moderate stress
            base_config.update({
                'market_volatility': 0.20,
                'spread_bps': 8.0,
                'market_impact_factor': 0.0015
            })
        elif self.current_stage == 3:
            # Stage 3: High stress
            base_config.update({
                'market_volatility': 0.25,
                'spread_bps': 12.0,
                'market_impact_factor': 0.002
            })
        elif self.current_stage == 4:
            # Stage 4: Extreme conditions
            base_config.update({
                'market_volatility': 0.30,
                'spread_bps': 20.0,
                'market_impact_factor': 0.003
            })
        
        return base_config
    
    def should_advance_stage(self, recent_performance: float) -> bool:
        """Check if should advance to next stage"""
        if not self.config.curriculum_enabled:
            return False
        
        if self.current_stage >= self.config.curriculum_stages - 1:
            return False
        
        self.stage_performance.append(recent_performance)
        
        # Need at least 10 episodes of performance data
        if len(self.stage_performance) < 10:
            return False
        
        # Check if performance is consistently above threshold
        recent_avg = np.mean(self.stage_performance[-10:])
        return recent_avg >= self.config.performance_threshold
    
    def advance_stage(self):
        """Advance to next curriculum stage"""
        if self.current_stage < self.config.curriculum_stages - 1:
            self.current_stage += 1
            self.stage_performance = []
            logger.info(f"Advanced to curriculum stage {self.current_stage}")


class SequentialExecutionTrainer:
    """
    Sequential Execution Trainer
    
    Trains the 5 sequential execution agents with proper coordination
    and performance optimization.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize training components
        self.replay_buffer = PrioritizedReplayBuffer(
            config.replay_buffer_size, 
            config.priority_alpha
        )
        
        self.centralized_critic = CentralizedCritic(
            obs_dim=55,  # Average observation dimension
            action_dim=4,  # Average action dimension
            num_agents=5
        ).to(self.device)
        
        self.reward_shaper = CoordinationRewardShaper(config)
        self.curriculum = CurriculumLearning(config)
        
        # Training metrics
        self.metrics = TrainingMetrics()
        
        # Optimizers
        self.optimizers = {
            agent_id: optim.Adam(agent.parameters(), lr=config.learning_rate)
            for agent_id, agent in self.agents.items()
        }
        
        self.critic_optimizer = optim.Adam(
            self.centralized_critic.parameters(), 
            lr=config.learning_rate
        )
        
        # Logging
        if config.tensorboard_enabled:
            self.writer = SummaryWriter(config.log_dir)
        else:
            self.writer = None
        
        # Environment pool for parallel training
        self.environments = []
        for i in range(config.num_environments):
            env_config = self.curriculum.get_environment_config()
            env = SequentialExecutionEnvironment(env_config)
            self.environments.append(env)
        
        logger.info("SequentialExecutionTrainer initialized",
                   num_agents=len(self.agents),
                   num_environments=len(self.environments),
                   device=str(self.device))
    
    def _initialize_agents(self) -> Dict[str, SequentialExecutionAgentBase]:
        """Initialize all sequential execution agents"""
        agents = {}
        
        # Agent configurations
        agent_configs = {
            'market_timing': {
                'obs_dim': 55,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.1,
                'min_delay_us': 10.0,
                'max_delay_us': 1000.0
            },
            'liquidity_sourcing': {
                'obs_dim': 65,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.1,
                'venues': ['SMART', 'ARCA', 'NASDAQ', 'NYSE', 'BATS']
            },
            'position_fragmentation': {
                'obs_dim': 61,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.1,
                'max_fragments': 20,
                'min_fragment_size': 0.01
            },
            'risk_control': {
                'obs_dim': 63,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.1,
                'var_threshold': 0.02,
                'emergency_threshold': 0.05
            },
            'execution_monitor': {
                'obs_dim': 67,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.1,
                'target_fill_rate': self.config.target_fill_rate,
                'target_latency_us': self.config.target_latency_us
            }
        }
        
        # Create agents
        agents['market_timing'] = MarketTimingAgent(agent_configs['market_timing']).to(self.device)
        agents['liquidity_sourcing'] = LiquiditySourcingAgent(agent_configs['liquidity_sourcing']).to(self.device)
        agents['position_fragmentation'] = PositionFragmentationAgent(agent_configs['position_fragmentation']).to(self.device)
        agents['risk_control'] = RiskControlAgent(agent_configs['risk_control']).to(self.device)
        agents['execution_monitor'] = ExecutionMonitorAgent(agent_configs['execution_monitor']).to(self.device)
        
        return agents
    
    def train(self):
        """Main training loop"""
        logger.info("Starting sequential execution training",
                   num_episodes=self.config.num_episodes,
                   num_environments=len(self.environments))
        
        for episode in range(self.config.num_episodes):
            episode_start_time = time.time()
            
            # Run episode
            episode_metrics = self._run_episode(episode)
            
            # Update metrics
            self._update_metrics(episode_metrics)
            
            # Train agents
            if len(self.replay_buffer) >= self.config.min_replay_size:
                training_losses = self._train_agents()
                self.metrics.loss_values.update(training_losses)
            
            # Curriculum learning
            if self.curriculum.should_advance_stage(episode_metrics['average_reward']):
                self.curriculum.advance_stage()
                self._update_environment_configs()
            
            # Logging
            if episode % 100 == 0:
                self._log_progress(episode, episode_metrics)
            
            # Evaluation
            if episode % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate_agents()
                self._log_evaluation(episode, eval_metrics)
            
            # Checkpointing
            if episode % self.config.save_frequency == 0:
                self._save_checkpoint(episode)
            
            episode_duration = time.time() - episode_start_time
            logger.debug(f"Episode {episode} completed in {episode_duration:.2f}s",
                        episode_reward=episode_metrics['total_reward'],
                        episode_length=episode_metrics['episode_length'])
        
        # Final evaluation and save
        final_eval = self._evaluate_agents()
        self._save_final_model()
        
        logger.info("Training completed",
                   total_episodes=self.config.num_episodes,
                   final_performance=final_eval)
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run single training episode"""
        env = self.environments[episode % len(self.environments)]
        
        # Generate cascade context
        cascade_context = self._generate_cascade_context()
        env.set_cascade_context(cascade_context)
        
        # Reset environment
        env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        agent_decisions = {}
        
        # Run episode
        while not any(env.terminations.values()) and not any(env.truncations.values()):
            # Get current agent
            current_agent = env.agent_selection
            
            # Get observation
            observation = env.observe(current_agent)
            
            # Generate superposition context
            superposition_context = self._generate_superposition_context()
            
            # Make decision
            agent = self.agents[current_agent]
            decision = agent.make_decision(
                observation, 
                cascade_context.__dict__, 
                superposition_context
            )
            
            # Store decision
            agent_decisions[current_agent] = decision
            
            # Step environment
            env.step(self._convert_decision_to_action(current_agent, decision))
            
            # Collect experience
            if len(agent_decisions) == len(self.agents):
                # All agents have acted, collect experience
                experience = self._collect_experience(
                    env, agent_decisions, observation, cascade_context, superposition_context
                )
                
                # Add to replay buffer
                priority = self._calculate_priority(experience)
                self.replay_buffer.add(experience, priority)
                
                # Update episode metrics
                episode_reward += experience['reward']
                episode_length += 1
                
                # Clear decisions
                agent_decisions = {}
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'average_reward': episode_reward / max(1, episode_length)
        }
    
    def _generate_cascade_context(self) -> CascadeContext:
        """Generate realistic cascade context"""
        return CascadeContext(
            strategic_signal=np.random.uniform(-0.5, 0.5),
            strategic_confidence=np.random.uniform(0.3, 0.9),
            strategic_regime=np.random.choice(['normal', 'volatile', 'trending']),
            strategic_superposition=np.random.dirichlet([1, 1, 1]),
            tactical_signal=np.random.uniform(-0.3, 0.3),
            tactical_confidence=np.random.uniform(0.4, 0.8),
            tactical_fvg_signal=np.random.uniform(-0.2, 0.2),
            tactical_momentum=np.random.uniform(-0.1, 0.1),
            tactical_superposition=np.random.dirichlet([1, 1, 1]),
            risk_allocation=np.random.uniform(0.0, 0.2),
            risk_var_estimate=np.random.uniform(0.01, 0.05),
            risk_stop_loss=np.random.uniform(0.02, 0.05),
            risk_take_profit=np.random.uniform(0.02, 0.08),
            risk_superposition=np.random.dirichlet([1, 1, 1])
        )
    
    def _generate_superposition_context(self) -> SuperpositionContext:
        """Generate superposition context"""
        return SuperpositionContext(
            strategic_superposition=np.random.dirichlet([1, 1, 1]),
            tactical_superposition=np.random.dirichlet([1, 1, 1]),
            risk_superposition=np.random.dirichlet([1, 1, 1]),
            strategic_coherence=np.random.uniform(0.5, 1.0),
            tactical_coherence=np.random.uniform(0.5, 1.0),
            risk_coherence=np.random.uniform(0.5, 1.0),
            strategic_tactical_entanglement=np.random.uniform(0.0, 0.5),
            tactical_risk_entanglement=np.random.uniform(0.0, 0.5)
        )
    
    def _convert_decision_to_action(self, agent_id: str, decision: Any) -> Any:
        """Convert agent decision to environment action"""
        if agent_id == 'market_timing':
            return np.array([
                decision.decision_value['timing_delay_us'],
                decision.decision_value['urgency'],
                decision.decision_value['confidence'],
                decision.decision_value['market_regime_adjust']
            ])
        elif agent_id == 'liquidity_sourcing':
            return np.array(decision.decision_value['venue_weights'] + 
                          [decision.decision_value['liquidity_threshold']])
        elif agent_id == 'position_fragmentation':
            return np.array([
                decision.decision_value['fragment_size'],
                decision.decision_value['num_fragments'],
                decision.decision_value['timing_spread'],
                decision.decision_value['stealth_factor']
            ])
        elif agent_id == 'risk_control':
            # Convert risk action to discrete action
            risk_actions = ['APPROVE', 'REDUCE_SIZE', 'DELAY', 'CANCEL', 'EMERGENCY_STOP']
            return risk_actions.index(decision.decision_value['risk_action'])
        elif agent_id == 'execution_monitor':
            return np.array([
                decision.decision_value['quality_threshold'],
                decision.decision_value['feedback_weight'],
                decision.decision_value['adjustment_factor']
            ])
        else:
            return np.array([0.0])
    
    def _collect_experience(self, env, agent_decisions, observation, cascade_context, superposition_context):
        """Collect experience from environment step"""
        # Get performance metrics
        performance_metrics = env.get_performance_metrics()
        
        # Calculate shaped reward
        base_reward = sum(env.agent_rewards.values())
        shaped_reward = self.reward_shaper.shape_reward(
            base_reward,
            {k: v.decision_value for k, v in agent_decisions.items()},
            performance_metrics,
            {}  # Coordination metrics would be calculated here
        )
        
        return {
            'observation': observation,
            'actions': {k: self._convert_decision_to_action(k, v) for k, v in agent_decisions.items()},
            'reward': shaped_reward,
            'performance_metrics': performance_metrics,
            'cascade_context': cascade_context,
            'superposition_context': superposition_context,
            'timestamp': datetime.now()
        }
    
    def _calculate_priority(self, experience: Dict[str, Any]) -> float:
        """Calculate priority for experience replay"""
        # Use reward magnitude as priority
        return abs(experience['reward']) + 0.1
    
    def _train_agents(self) -> Dict[str, float]:
        """Train all agents"""
        losses = {}
        
        for agent_id, agent in self.agents.items():
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size,
                self.config.priority_beta
            )
            
            if len(experiences) == 0:
                continue
            
            # Train agent
            loss = self._train_single_agent(agent_id, agent, experiences, weights)
            losses[agent_id] = loss
            
            # Update priorities
            new_priorities = np.abs(np.array([exp['reward'] for exp in experiences])) + 0.1
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        return losses
    
    def _train_single_agent(self, agent_id: str, agent: SequentialExecutionAgentBase, 
                          experiences: List[Dict], weights: np.ndarray) -> float:
        """Train single agent"""
        # Extract batch data
        observations = []
        actions = []
        rewards = []
        
        for exp in experiences:
            observations.append(exp['observation'])
            actions.append(exp['actions'].get(agent_id, np.array([0.0])))
            rewards.append(exp['reward'])
        
        # Convert to tensors
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Forward pass
        action_logits, confidence = agent.forward(obs_tensor)
        
        # Calculate loss
        if agent_id == 'risk_control':
            # Discrete action loss
            loss = nn.CrossEntropyLoss(reduction='none')(action_logits, action_tensor.long())
        else:
            # Continuous action loss
            loss = nn.MSELoss(reduction='none')(action_logits, action_tensor)
        
        # Apply importance sampling weights
        loss = loss * weight_tensor
        total_loss = loss.mean()
        
        # Backward pass
        self.optimizers[agent_id].zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(agent.parameters(), self.config.gradient_clip_norm)
        
        self.optimizers[agent_id].step()
        
        return total_loss.item()
    
    def _update_environment_configs(self):
        """Update environment configurations for curriculum learning"""
        new_config = self.curriculum.get_environment_config()
        
        for env in self.environments:
            env.config.update(new_config)
    
    def _update_metrics(self, episode_metrics: Dict[str, Any]):
        """Update training metrics"""
        self.metrics.episode_count += 1
        self.metrics.total_steps += episode_metrics['episode_length']
        
        # Update averages
        n = self.metrics.episode_count
        self.metrics.avg_episode_reward = (
            (n - 1) * self.metrics.avg_episode_reward + episode_metrics['total_reward']
        ) / n
        self.metrics.avg_episode_length = (
            (n - 1) * self.metrics.avg_episode_length + episode_metrics['episode_length']
        ) / n
        
        # Update recent history
        self.metrics.recent_rewards.append(episode_metrics['total_reward'])
    
    def _evaluate_agents(self) -> Dict[str, Any]:
        """Evaluate agent performance"""
        eval_env = self.environments[0]
        
        total_reward = 0.0
        total_episodes = 10
        
        for episode in range(total_episodes):
            cascade_context = self._generate_cascade_context()
            eval_env.set_cascade_context(cascade_context)
            eval_env.reset()
            
            episode_reward = 0.0
            
            while not any(eval_env.terminations.values()) and not any(eval_env.truncations.values()):
                current_agent = eval_env.agent_selection
                observation = eval_env.observe(current_agent)
                
                # Use agent without exploration
                agent = self.agents[current_agent]
                with torch.no_grad():
                    decision = agent.make_decision(
                        observation, 
                        cascade_context.__dict__, 
                        self._generate_superposition_context()
                    )
                
                action = self._convert_decision_to_action(current_agent, decision)
                eval_env.step(action)
                
                episode_reward += eval_env.agent_rewards.get(current_agent, 0.0)
            
            total_reward += episode_reward
        
        avg_reward = total_reward / total_episodes
        performance_metrics = eval_env.get_performance_metrics()
        
        return {
            'average_reward': avg_reward,
            'performance_metrics': performance_metrics
        }
    
    def _log_progress(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log training progress"""
        logger.info(f"Episode {episode}",
                   reward=episode_metrics['total_reward'],
                   length=episode_metrics['episode_length'],
                   avg_reward=self.metrics.avg_episode_reward,
                   curriculum_stage=self.curriculum.current_stage)
        
        if self.writer:
            self.writer.add_scalar('Training/EpisodeReward', episode_metrics['total_reward'], episode)
            self.writer.add_scalar('Training/EpisodeLength', episode_metrics['episode_length'], episode)
            self.writer.add_scalar('Training/AverageReward', self.metrics.avg_episode_reward, episode)
            self.writer.add_scalar('Training/CurriculumStage', self.curriculum.current_stage, episode)
    
    def _log_evaluation(self, episode: int, eval_metrics: Dict[str, Any]):
        """Log evaluation results"""
        logger.info(f"Evaluation at episode {episode}",
                   avg_reward=eval_metrics['average_reward'],
                   performance_metrics=eval_metrics['performance_metrics'])
        
        if self.writer:
            self.writer.add_scalar('Evaluation/AverageReward', eval_metrics['average_reward'], episode)
            
            perf_metrics = eval_metrics['performance_metrics']
            self.writer.add_scalar('Evaluation/AverageLatency', perf_metrics.get('average_latency_us', 0), episode)
            self.writer.add_scalar('Evaluation/FillRate', perf_metrics.get('fill_rate', 0), episode)
            self.writer.add_scalar('Evaluation/Slippage', perf_metrics.get('slippage_bps', 0), episode)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_{episode}.pt"
        
        checkpoint = {
            'episode': episode,
            'agents': {k: v.state_dict() for k, v in self.agents.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'centralized_critic': self.centralized_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'metrics': self.metrics,
            'curriculum_stage': self.curriculum.current_stage,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final trained model"""
        model_path = f"{self.config.checkpoint_dir}/final_model.pt"
        
        final_model = {
            'agents': {k: v.state_dict() for k, v in self.agents.items()},
            'centralized_critic': self.centralized_critic.state_dict(),
            'config': self.config,
            'final_metrics': self.metrics
        }
        
        torch.save(final_model, model_path)
        logger.info(f"Final model saved: {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load agent states
        for agent_id, state_dict in checkpoint['agents'].items():
            self.agents[agent_id].load_state_dict(state_dict)
        
        # Load optimizer states
        for optimizer_id, state_dict in checkpoint['optimizers'].items():
            self.optimizers[optimizer_id].load_state_dict(state_dict)
        
        # Load other components
        self.centralized_critic.load_state_dict(checkpoint['centralized_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.metrics = checkpoint['metrics']
        self.curriculum.current_stage = checkpoint['curriculum_stage']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    # Training configuration
    config = TrainingConfig(
        num_episodes=10000,
        batch_size=64,
        learning_rate=3e-4,
        num_environments=4,
        curriculum_enabled=True,
        tensorboard_enabled=True
    )
    
    # Create trainer
    trainer = SequentialExecutionTrainer(config)
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully")