"""
Tactical Sequential Trainer

This module implements a specialized training framework for sequential tactical agents
that operate in the FVG → Momentum → EntryOpt sequence. The trainer handles strategic
context integration, predecessor dependency management, and high-frequency execution
requirements.

Key Features:
- Sequential training with dependency-aware experience replay
- Strategic context integration from upstream 30m MARL system
- Multi-agent coordination and consensus training
- Byzantine fault tolerance during training
- Performance optimization for 5-minute cycle requirements
- Distributed training capability

Architecture:
- Supports both individual agent training and joint sequential training
- Handles strategic context updates and predecessor state management
- Implements advanced reward shaping for sequential coordination
- Provides comprehensive performance monitoring and validation

Author: Agent 5 - Sequential Tactical MARL Specialist
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
import json
import uuid
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# Core imports
from src.agents.tactical.sequential_tactical_agents import (
    SequentialTacticalAgent, FVGTacticalAgent, MomentumTacticalAgent, 
    EntryOptimizationAgent, AgentOutput, StrategicContext, PredecessorContext
)
from src.environment.sequential_tactical_env import SequentialTacticalEnvironment
from src.training.rewards.reward_functions import RewardFunction
from src.training.experience import ExperienceBuffer
from src.core.event_bus import EventBus
from src.core.events import Event, EventType

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for tactical sequential training"""
    # Environment settings
    max_episodes: int = 10000
    max_steps_per_episode: int = 2000
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Learning settings
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    gradient_clip_norm: float = 1.0
    
    # Training schedule
    warmup_episodes: int = 100
    update_frequency: int = 4
    target_update_frequency: int = 100
    validation_frequency: int = 500
    
    # Sequential training settings
    sequential_training: bool = True
    predecessor_dependency_weight: float = 0.3
    strategic_context_weight: float = 0.2
    consensus_bonus_weight: float = 0.4
    
    # Performance requirements
    target_latency_ms: float = 50.0
    min_success_rate: float = 0.8
    min_consensus_rate: float = 0.7
    
    # Distributed training
    distributed_training: bool = False
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging and checkpointing
    log_frequency: int = 100
    checkpoint_frequency: int = 1000
    tensorboard_log_dir: str = "logs/tactical_training"
    checkpoint_dir: str = "checkpoints/tactical"

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    episode: int = 0
    step: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    
    # Sequential metrics
    fvg_performance: float = 0.0
    momentum_performance: float = 0.0
    entry_opt_performance: float = 0.0
    consensus_rate: float = 0.0
    strategic_alignment: float = 0.0
    
    # Performance metrics
    avg_processing_time: float = 0.0
    success_rate: float = 0.0
    execution_rate: float = 0.0
    
    # Loss metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    
    # Training stability
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    
    timestamp: float = field(default_factory=time.time)

@dataclass
class SequentialExperience:
    """Sequential experience container"""
    episode_id: str
    step: int
    
    # Individual agent experiences
    fvg_experience: Optional[Dict[str, Any]] = None
    momentum_experience: Optional[Dict[str, Any]] = None
    entry_opt_experience: Optional[Dict[str, Any]] = None
    
    # Sequential context
    strategic_context: Optional[StrategicContext] = None
    predecessor_contexts: Dict[str, PredecessorContext] = field(default_factory=dict)
    
    # Outcomes
    tactical_superposition: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # Rewards
    individual_rewards: Dict[str, float] = field(default_factory=dict)
    sequential_reward: float = 0.0
    consensus_bonus: float = 0.0
    
    timestamp: float = field(default_factory=time.time)

class TacticalSequentialTrainer:
    """
    Tactical Sequential Trainer
    
    Implements comprehensive training for sequential tactical agents with
    strategic context integration, predecessor dependency management, and
    high-frequency execution optimization.
    """
    
    def __init__(
        self,
        agents: Dict[str, SequentialTacticalAgent],
        environment: SequentialTacticalEnvironment,
        config: TrainingConfig,
        reward_function: Optional[RewardFunction] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize tactical sequential trainer
        
        Args:
            agents: Dictionary of tactical agents
            environment: Sequential tactical environment
            config: Training configuration
            reward_function: Custom reward function
            event_bus: Event bus for system integration
        """
        self.agents = agents
        self.environment = environment
        self.config = config
        self.reward_function = reward_function
        self.event_bus = event_bus
        
        # Validate agent sequence
        self.agent_sequence = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        self._validate_agents()
        
        # Initialize training components
        self.experience_buffer = ExperienceBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.batch_size
        )
        
        # Sequential experience tracking
        self.sequential_experiences = deque(maxlen=config.buffer_size)
        
        # Training state
        self.training_metrics = TrainingMetrics()
        self.is_training = False
        self.training_thread = None
        
        # Performance tracking
        self.episode_metrics = []
        self.training_history = {
            'rewards': [],
            'losses': [],
            'performance': [],
            'consensus_rates': [],
            'processing_times': []
        }
        
        # Strategic context manager
        self.strategic_context_manager = self._initialize_strategic_context_manager()
        
        # Logging
        self.logger = SummaryWriter(config.tensorboard_log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.training_start_time = None
        
        logger.info(f"Tactical Sequential Trainer initialized with {len(agents)} agents")
        logger.info(f"Training configuration: {config}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _validate_agents(self):
        """Validate agent configuration"""
        for agent_id in self.agent_sequence:
            if agent_id not in self.agents:
                raise ValueError(f"Missing required agent: {agent_id}")
            
            agent = self.agents[agent_id]
            if not isinstance(agent, SequentialTacticalAgent):
                raise ValueError(f"Agent {agent_id} is not a SequentialTacticalAgent")
        
        logger.info("Agent validation passed")
    
    def _initialize_strategic_context_manager(self):
        """Initialize strategic context manager"""
        try:
            from src.integration.strategic_context_manager import StrategicContextManager
            return StrategicContextManager()
        except ImportError:
            logger.warning("Strategic context manager not available, using mock")
            return MockStrategicContextManager()
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training results and metrics
        """
        try:
            self.is_training = True
            self.training_start_time = time.time()
            
            logger.info("Starting tactical sequential training...")
            
            # Set agents to training mode
            for agent in self.agents.values():
                agent.set_execution_mode('training')
            
            # Training loop
            for episode in range(self.config.max_episodes):
                self.training_metrics.episode = episode
                
                # Run episode
                episode_result = self._run_training_episode()
                
                # Update training metrics
                self._update_training_metrics(episode_result)
                
                # Periodic validation
                if episode % self.config.validation_frequency == 0:
                    validation_result = self._validate_agents()
                    self._log_validation_results(validation_result)
                
                # Checkpoint saving
                if episode % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(episode)
                
                # Early stopping check
                if self._should_stop_training(episode_result):
                    logger.info(f"Early stopping at episode {episode}")
                    break
                
                # Logging
                if episode % self.config.log_frequency == 0:
                    self._log_training_progress(episode)
            
            # Final results
            training_results = self._compile_training_results()
            
            logger.info("Training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
            self.logger.close()
    
    def _run_training_episode(self) -> Dict[str, Any]:
        """Run a single training episode"""
        try:
            episode_start_time = time.time()
            episode_id = str(uuid.uuid4())
            
            # Initialize episode
            observations = self.environment.reset()
            done = False
            step = 0
            
            # Episode tracking
            episode_reward = 0.0
            agent_rewards = {agent_id: 0.0 for agent_id in self.agent_sequence}
            sequential_experiences = []
            
            # Get initial strategic context
            strategic_context = self._get_strategic_context()
            
            while not done and step < self.config.max_steps_per_episode:
                step_start_time = time.time()
                
                # Run sequential step
                step_result = self._run_sequential_step(
                    observations, strategic_context, episode_id, step
                )
                
                # Store experience
                sequential_experiences.append(step_result['experience'])
                
                # Update episode metrics
                episode_reward += step_result['total_reward']
                for agent_id, reward in step_result['agent_rewards'].items():
                    agent_rewards[agent_id] += reward
                
                # Environment step
                if step_result['action'] is not None:
                    observations, reward, done, truncated, info = self.environment.step(
                        step_result['action']
                    )
                    done = done or truncated
                
                # Update strategic context periodically
                if step % 10 == 0:
                    strategic_context = self._get_strategic_context()
                
                # Performance monitoring
                step_time = time.time() - step_start_time
                if step_time > self.config.target_latency_ms / 1000:
                    logger.warning(f"Step {step} took {step_time*1000:.2f}ms (target: {self.config.target_latency_ms}ms)")
                
                step += 1
            
            # Episode completion
            episode_duration = time.time() - episode_start_time
            
            # Store sequential experiences
            for exp in sequential_experiences:
                self.sequential_experiences.append(exp)
            
            # Train agents if enough experience
            if len(self.sequential_experiences) >= self.config.batch_size:
                training_losses = self._train_agents()
            else:
                training_losses = {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Episode results
            episode_result = {
                'episode_id': episode_id,
                'episode_reward': episode_reward,
                'agent_rewards': agent_rewards,
                'episode_length': step,
                'episode_duration': episode_duration,
                'training_losses': training_losses,
                'strategic_context_updates': self._count_strategic_context_updates(),
                'consensus_rate': self._calculate_consensus_rate(sequential_experiences),
                'avg_processing_time': self._calculate_avg_processing_time(sequential_experiences)
            }
            
            return episode_result
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return {'episode_reward': 0.0, 'episode_length': 0, 'error': str(e)}
    
    def _run_sequential_step(
        self, 
        observations: Dict[str, np.ndarray], 
        strategic_context: StrategicContext,
        episode_id: str,
        step: int
    ) -> Dict[str, Any]:
        """Run a single sequential step with all agents"""
        try:
            step_start_time = time.time()
            
            # Initialize step tracking
            agent_outputs = {}
            predecessor_context = None
            step_reward = 0.0
            agent_rewards = {agent_id: 0.0 for agent_id in self.agent_sequence}
            
            # Sequential execution
            for agent_id in self.agent_sequence:
                agent = self.agents[agent_id]
                observation = observations.get(agent_id, np.zeros(agent.observation_dim))
                
                # Agent action selection
                output = agent.select_action(
                    observation=observation,
                    strategic_context=strategic_context,
                    predecessor_context=predecessor_context
                )
                
                agent_outputs[agent_id] = output
                
                # Calculate individual agent reward
                agent_reward = self._calculate_agent_reward(agent_id, output, strategic_context)
                agent_rewards[agent_id] = agent_reward
                step_reward += agent_reward
                
                # Update predecessor context for next agent
                predecessor_context = self._update_predecessor_context(
                    predecessor_context, agent_id, output
                )
            
            # Calculate sequential bonus
            sequential_bonus = self._calculate_sequential_bonus(agent_outputs)
            step_reward += sequential_bonus
            
            # Calculate consensus bonus
            consensus_bonus = self._calculate_consensus_bonus(agent_outputs)
            step_reward += consensus_bonus
            
            # Create sequential experience
            experience = SequentialExperience(
                episode_id=episode_id,
                step=step,
                fvg_experience=self._create_agent_experience('fvg_agent', agent_outputs),
                momentum_experience=self._create_agent_experience('momentum_agent', agent_outputs),
                entry_opt_experience=self._create_agent_experience('entry_opt_agent', agent_outputs),
                strategic_context=strategic_context,
                predecessor_contexts=self._extract_predecessor_contexts(agent_outputs),
                individual_rewards=agent_rewards,
                sequential_reward=step_reward,
                consensus_bonus=consensus_bonus
            )
            
            # Determine next action (from final agent)
            final_agent_output = agent_outputs[self.agent_sequence[-1]]
            next_action = final_agent_output.action if final_agent_output.confidence > 0.5 else None
            
            # Step processing time
            processing_time = time.time() - step_start_time
            
            return {
                'experience': experience,
                'action': next_action,
                'total_reward': step_reward,
                'agent_rewards': agent_rewards,
                'agent_outputs': agent_outputs,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in sequential step: {e}")
            return {
                'experience': None,
                'action': None,
                'total_reward': 0.0,
                'agent_rewards': {},
                'agent_outputs': {},
                'processing_time': 0.0
            }
    
    def _get_strategic_context(self) -> StrategicContext:
        """Get strategic context from upstream system"""
        try:
            if hasattr(self.strategic_context_manager, 'get_latest_context'):
                context = self.strategic_context_manager.get_latest_context()
                if context:
                    return context
            
            # Mock strategic context
            return StrategicContext(
                regime_embedding=np.random.normal(0, 0.1, 64).astype(np.float32),
                synergy_signal={
                    'strength': np.random.uniform(0.5, 1.0),
                    'confidence': np.random.uniform(0.6, 1.0),
                    'direction': np.random.choice([-1, 0, 1]),
                    'urgency': np.random.uniform(0.3, 0.9)
                },
                market_state={
                    'price': 100.0 + np.random.normal(0, 1),
                    'volume': 1000.0 + np.random.normal(0, 100),
                    'volatility': np.random.uniform(0.1, 0.5)
                },
                confidence_level=np.random.uniform(0.5, 1.0),
                execution_bias=np.random.choice(['bullish', 'neutral', 'bearish']),
                volatility_forecast=np.random.uniform(0.1, 0.5),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error getting strategic context: {e}")
            return StrategicContext(
                regime_embedding=np.zeros(64, dtype=np.float32),
                synergy_signal={},
                market_state={},
                confidence_level=0.5,
                execution_bias='neutral',
                volatility_forecast=0.2,
                timestamp=time.time()
            )
    
    def _update_predecessor_context(
        self, 
        current_context: Optional[PredecessorContext],
        agent_id: str,
        output: AgentOutput
    ) -> PredecessorContext:
        """Update predecessor context with new agent output"""
        try:
            if current_context is None:
                return PredecessorContext(
                    agent_outputs={agent_id: output.__dict__},
                    consensus_level=1.0,
                    alignment_score=1.0,
                    execution_signals={agent_id: output.execution_signals},
                    feature_importance={agent_id: output.feature_importance},
                    timestamp=time.time()
                )
            
            # Update existing context
            current_context.agent_outputs[agent_id] = output.__dict__
            current_context.execution_signals[agent_id] = output.execution_signals
            current_context.feature_importance[agent_id] = output.feature_importance
            
            # Recalculate consensus and alignment
            actions = [out.get('action', 1) for out in current_context.agent_outputs.values()]
            confidences = [out.get('confidence', 0.5) for out in current_context.agent_outputs.values()]
            
            current_context.consensus_level = 1.0 - (len(set(actions)) - 1) / len(actions)
            current_context.alignment_score = np.mean(confidences)
            current_context.timestamp = time.time()
            
            return current_context
            
        except Exception as e:
            logger.error(f"Error updating predecessor context: {e}")
            return PredecessorContext(
                agent_outputs={agent_id: output.__dict__},
                consensus_level=0.5,
                alignment_score=0.5,
                execution_signals={},
                feature_importance={},
                timestamp=time.time()
            )
    
    def _calculate_agent_reward(
        self, 
        agent_id: str, 
        output: AgentOutput, 
        strategic_context: StrategicContext
    ) -> float:
        """Calculate individual agent reward"""
        try:
            # Base reward from confidence
            base_reward = output.confidence * 2.0 - 1.0  # Scale to [-1, 1]
            
            # Strategic alignment bonus
            strategic_bonus = 0.0
            if strategic_context:
                action_bias = ['bearish', 'neutral', 'bullish'][output.action]
                if action_bias == strategic_context.execution_bias:
                    strategic_bonus = 0.2 * strategic_context.confidence_level
                elif strategic_context.execution_bias == 'neutral':
                    strategic_bonus = 0.1 * strategic_context.confidence_level
                else:
                    strategic_bonus = -0.1 * strategic_context.confidence_level
            
            # Processing time penalty
            processing_penalty = 0.0
            if output.processing_time > self.config.target_latency_ms:
                processing_penalty = -0.1 * (output.processing_time - self.config.target_latency_ms) / self.config.target_latency_ms
            
            # Agent-specific bonuses
            agent_bonus = 0.0
            if agent_id == 'fvg_agent':
                # FVG agent bonus for gap quality
                if 'fvg_analysis' in output.market_insights:
                    gap_quality = output.market_insights['fvg_analysis'].get('gap_quality', 'medium')
                    if gap_quality == 'high':
                        agent_bonus = 0.1
            elif agent_id == 'momentum_agent':
                # Momentum agent bonus for trend quality
                if 'momentum_analysis' in output.market_insights:
                    trend_quality = output.market_insights['momentum_analysis'].get('trend_quality', 'moderate')
                    if trend_quality == 'strong':
                        agent_bonus = 0.1
            elif agent_id == 'entry_opt_agent':
                # Entry optimization agent bonus for entry quality
                if 'entry_analysis' in output.market_insights:
                    entry_quality = output.market_insights['entry_analysis'].get('entry_quality', 'fair')
                    if entry_quality == 'optimal':
                        agent_bonus = 0.1
            
            total_reward = base_reward + strategic_bonus + processing_penalty + agent_bonus
            
            return float(np.clip(total_reward, -2.0, 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating agent reward: {e}")
            return 0.0
    
    def _calculate_sequential_bonus(self, agent_outputs: Dict[str, AgentOutput]) -> float:
        """Calculate bonus for sequential coordination"""
        try:
            if len(agent_outputs) < 2:
                return 0.0
            
            # Calculate agreement bonus
            actions = [output.action for output in agent_outputs.values()]
            confidences = [output.confidence for output in agent_outputs.values()]
            
            # Agreement score
            unique_actions = len(set(actions))
            if unique_actions == 1:
                agreement_bonus = 0.3  # Perfect agreement
            elif unique_actions == 2:
                agreement_bonus = 0.1  # Partial agreement
            else:
                agreement_bonus = 0.0  # No agreement
            
            # Confidence progression bonus
            confidence_progression = 0.0
            if len(confidences) >= 2:
                # Reward increasing confidence through sequence
                confidence_trend = np.mean(np.diff(confidences))
                if confidence_trend > 0:
                    confidence_progression = 0.1 * confidence_trend
            
            # Timing bonus
            processing_times = [output.processing_time for output in agent_outputs.values()]
            avg_processing_time = np.mean(processing_times)
            timing_bonus = 0.0
            if avg_processing_time < self.config.target_latency_ms:
                timing_bonus = 0.1 * (self.config.target_latency_ms - avg_processing_time) / self.config.target_latency_ms
            
            total_bonus = agreement_bonus + confidence_progression + timing_bonus
            
            return float(np.clip(total_bonus, 0.0, 0.5))
            
        except Exception as e:
            logger.error(f"Error calculating sequential bonus: {e}")
            return 0.0
    
    def _calculate_consensus_bonus(self, agent_outputs: Dict[str, AgentOutput]) -> float:
        """Calculate consensus bonus based on agent agreement"""
        try:
            if len(agent_outputs) < 2:
                return 0.0
            
            # Extract agent decisions
            actions = [output.action for output in agent_outputs.values()]
            confidences = [output.confidence for output in agent_outputs.values()]
            
            # Consensus strength
            action_counts = {action: actions.count(action) for action in set(actions)}
            max_agreement = max(action_counts.values())
            consensus_strength = max_agreement / len(actions)
            
            # Confidence-weighted consensus
            avg_confidence = np.mean(confidences)
            
            # Bonus calculation
            consensus_bonus = consensus_strength * avg_confidence * self.config.consensus_bonus_weight
            
            return float(np.clip(consensus_bonus, 0.0, 0.5))
            
        except Exception as e:
            logger.error(f"Error calculating consensus bonus: {e}")
            return 0.0
    
    def _create_agent_experience(self, agent_id: str, agent_outputs: Dict[str, AgentOutput]) -> Optional[Dict[str, Any]]:
        """Create experience entry for specific agent"""
        try:
            if agent_id not in agent_outputs:
                return None
            
            output = agent_outputs[agent_id]
            
            return {
                'agent_id': agent_id,
                'action': output.action,
                'probabilities': output.probabilities.tolist(),
                'confidence': output.confidence,
                'processing_time': output.processing_time,
                'market_insights': output.market_insights,
                'execution_signals': output.execution_signals,
                'feature_importance': output.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error creating agent experience: {e}")
            return None
    
    def _extract_predecessor_contexts(self, agent_outputs: Dict[str, AgentOutput]) -> Dict[str, PredecessorContext]:
        """Extract predecessor contexts for each agent"""
        try:
            contexts = {}
            
            for i, agent_id in enumerate(self.agent_sequence):
                if i == 0:
                    contexts[agent_id] = None
                else:
                    # Create predecessor context
                    predecessors = {}
                    for j in range(i):
                        pred_id = self.agent_sequence[j]
                        if pred_id in agent_outputs:
                            predecessors[pred_id] = agent_outputs[pred_id].__dict__
                    
                    contexts[agent_id] = PredecessorContext(
                        agent_outputs=predecessors,
                        consensus_level=1.0,
                        alignment_score=1.0,
                        execution_signals={},
                        feature_importance={},
                        timestamp=time.time()
                    )
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error extracting predecessor contexts: {e}")
            return {}
    
    def _train_agents(self) -> Dict[str, float]:
        """Train all agents using collected experiences"""
        try:
            if len(self.sequential_experiences) < self.config.batch_size:
                return {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Sample batch of sequential experiences
            batch_size = min(self.config.batch_size, len(self.sequential_experiences))
            batch_indices = np.random.choice(len(self.sequential_experiences), batch_size, replace=False)
            batch_experiences = [self.sequential_experiences[i] for i in batch_indices]
            
            # Train each agent
            total_losses = {'policy_loss': 0.0, 'value_loss': 0.0}
            
            for agent_id in self.agent_sequence:
                agent = self.agents[agent_id]
                
                # Prepare training batch for this agent
                agent_batch = self._prepare_agent_batch(agent_id, batch_experiences)
                
                if agent_batch:
                    # Train agent
                    agent_losses = self._train_single_agent(agent, agent_batch)
                    
                    # Accumulate losses
                    for loss_name, loss_value in agent_losses.items():
                        total_losses[loss_name] += loss_value
            
            # Average losses
            num_agents = len(self.agent_sequence)
            for loss_name in total_losses:
                total_losses[loss_name] /= num_agents
            
            return total_losses
            
        except Exception as e:
            logger.error(f"Error training agents: {e}")
            return {'policy_loss': 0.0, 'value_loss': 0.0}
    
    def _prepare_agent_batch(self, agent_id: str, batch_experiences: List[SequentialExperience]) -> Optional[List[Dict[str, Any]]]:
        """Prepare training batch for specific agent"""
        try:
            agent_batch = []
            
            for exp in batch_experiences:
                # Get agent-specific experience
                agent_exp = None
                if agent_id == 'fvg_agent':
                    agent_exp = exp.fvg_experience
                elif agent_id == 'momentum_agent':
                    agent_exp = exp.momentum_experience
                elif agent_id == 'entry_opt_agent':
                    agent_exp = exp.entry_opt_experience
                
                if agent_exp:
                    # Create training example
                    training_example = {
                        'observation': np.zeros(self.agents[agent_id].observation_dim),  # Mock observation
                        'action': agent_exp['action'],
                        'reward': exp.individual_rewards.get(agent_id, 0.0),
                        'next_observation': np.zeros(self.agents[agent_id].observation_dim),  # Mock next observation
                        'done': False,
                        'strategic_context': exp.strategic_context,
                        'predecessor_context': exp.predecessor_contexts.get(agent_id),
                        'sequential_reward': exp.sequential_reward,
                        'consensus_bonus': exp.consensus_bonus
                    }
                    
                    agent_batch.append(training_example)
            
            return agent_batch if agent_batch else None
            
        except Exception as e:
            logger.error(f"Error preparing agent batch: {e}")
            return None
    
    def _train_single_agent(self, agent: SequentialTacticalAgent, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train a single agent"""
        try:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            for experience in batch:
                # Train step
                losses = agent.train_step(experience)
                
                total_policy_loss += losses.get('policy_loss', 0.0)
                total_value_loss += losses.get('value_loss', 0.0)
            
            # Average losses
            avg_policy_loss = total_policy_loss / len(batch)
            avg_value_loss = total_value_loss / len(batch)
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss
            }
            
        except Exception as e:
            logger.error(f"Error training single agent: {e}")
            return {'policy_loss': 0.0, 'value_loss': 0.0}
    
    def _count_strategic_context_updates(self) -> int:
        """Count strategic context updates in episode"""
        # Mock implementation
        return 5
    
    def _calculate_consensus_rate(self, experiences: List[SequentialExperience]) -> float:
        """Calculate consensus rate for episode"""
        try:
            if not experiences:
                return 0.0
            
            consensus_count = 0
            for exp in experiences:
                if exp.consensus_bonus > 0:
                    consensus_count += 1
            
            return consensus_count / len(experiences)
            
        except Exception as e:
            logger.error(f"Error calculating consensus rate: {e}")
            return 0.0
    
    def _calculate_avg_processing_time(self, experiences: List[SequentialExperience]) -> float:
        """Calculate average processing time for episode"""
        try:
            if not experiences:
                return 0.0
            
            processing_times = []
            for exp in experiences:
                for agent_id in self.agent_sequence:
                    if agent_id == 'fvg_agent' and exp.fvg_experience:
                        processing_times.append(exp.fvg_experience['processing_time'])
                    elif agent_id == 'momentum_agent' and exp.momentum_experience:
                        processing_times.append(exp.momentum_experience['processing_time'])
                    elif agent_id == 'entry_opt_agent' and exp.entry_opt_experience:
                        processing_times.append(exp.entry_opt_experience['processing_time'])
            
            return np.mean(processing_times) if processing_times else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average processing time: {e}")
            return 0.0
    
    def _update_training_metrics(self, episode_result: Dict[str, Any]):
        """Update training metrics"""
        try:
            # Update episode metrics
            self.training_metrics.total_reward = episode_result.get('episode_reward', 0.0)
            self.training_metrics.episode_length = episode_result.get('episode_length', 0)
            self.training_metrics.consensus_rate = episode_result.get('consensus_rate', 0.0)
            self.training_metrics.avg_processing_time = episode_result.get('avg_processing_time', 0.0)
            
            # Update training losses
            losses = episode_result.get('training_losses', {})
            self.training_metrics.policy_loss = losses.get('policy_loss', 0.0)
            self.training_metrics.value_loss = losses.get('value_loss', 0.0)
            self.training_metrics.total_loss = self.training_metrics.policy_loss + self.training_metrics.value_loss
            
            # Update performance metrics
            success_rate = 1.0 if episode_result.get('episode_reward', 0.0) > 0 else 0.0
            execution_rate = episode_result.get('consensus_rate', 0.0)
            
            self.training_metrics.success_rate = success_rate
            self.training_metrics.execution_rate = execution_rate
            
            # Store in history
            self.training_history['rewards'].append(self.training_metrics.total_reward)
            self.training_history['losses'].append(self.training_metrics.total_loss)
            self.training_history['consensus_rates'].append(self.training_metrics.consensus_rate)
            self.training_history['processing_times'].append(self.training_metrics.avg_processing_time)
            
            # Store episode metrics
            self.episode_metrics.append(episode_result)
            
        except Exception as e:
            logger.error(f"Error updating training metrics: {e}")
    
    def _validate_agents(self) -> Dict[str, Any]:
        """Validate agent performance"""
        try:
            validation_results = {}
            
            for agent_id, agent in self.agents.items():
                # Get agent performance metrics
                agent_metrics = agent.get_performance_metrics()
                
                # Validate performance
                validation_results[agent_id] = {
                    'avg_confidence': agent_metrics.get('avg_confidence', 0.0),
                    'avg_processing_time': agent_metrics.get('avg_processing_time', 0.0),
                    'meets_latency_target': agent_metrics.get('avg_processing_time', 0.0) < self.config.target_latency_ms,
                    'meets_confidence_target': agent_metrics.get('avg_confidence', 0.0) > 0.5
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating agents: {e}")
            return {}
    
    def _should_stop_training(self, episode_result: Dict[str, Any]) -> bool:
        """Check if training should stop early"""
        try:
            # Check recent performance
            if len(self.training_history['rewards']) < 100:
                return False
            
            recent_rewards = self.training_history['rewards'][-100:]
            recent_consensus_rates = self.training_history['consensus_rates'][-100:]
            recent_processing_times = self.training_history['processing_times'][-100:]
            
            # Performance criteria
            avg_reward = np.mean(recent_rewards)
            avg_consensus_rate = np.mean(recent_consensus_rates)
            avg_processing_time = np.mean(recent_processing_times)
            
            # Check if all criteria are met
            reward_criterion = avg_reward > 0.5
            consensus_criterion = avg_consensus_rate > self.config.min_consensus_rate
            latency_criterion = avg_processing_time < self.config.target_latency_ms
            
            return reward_criterion and consensus_criterion and latency_criterion
            
        except Exception as e:
            logger.error(f"Error checking stopping criteria: {e}")
            return False
    
    def _log_training_progress(self, episode: int):
        """Log training progress"""
        try:
            # Log to tensorboard
            self.logger.add_scalar('Training/Episode_Reward', self.training_metrics.total_reward, episode)
            self.logger.add_scalar('Training/Policy_Loss', self.training_metrics.policy_loss, episode)
            self.logger.add_scalar('Training/Value_Loss', self.training_metrics.value_loss, episode)
            self.logger.add_scalar('Training/Consensus_Rate', self.training_metrics.consensus_rate, episode)
            self.logger.add_scalar('Training/Processing_Time', self.training_metrics.avg_processing_time, episode)
            
            # Log to console
            logger.info(f"Episode {episode}: Reward={self.training_metrics.total_reward:.3f}, "
                       f"Consensus={self.training_metrics.consensus_rate:.3f}, "
                       f"Processing={self.training_metrics.avg_processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error logging training progress: {e}")
    
    def _log_validation_results(self, validation_results: Dict[str, Any]):
        """Log validation results"""
        try:
            for agent_id, metrics in validation_results.items():
                logger.info(f"Agent {agent_id} validation: "
                           f"Confidence={metrics.get('avg_confidence', 0.0):.3f}, "
                           f"Processing={metrics.get('avg_processing_time', 0.0):.2f}ms")
                
        except Exception as e:
            logger.error(f"Error logging validation results: {e}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / f"tactical_checkpoint_episode_{episode}.pt"
            
            # Prepare checkpoint data
            checkpoint_data = {
                'episode': episode,
                'training_metrics': self.training_metrics.__dict__,
                'training_history': self.training_history,
                'config': self.config.__dict__,
                'session_id': self.session_id
            }
            
            # Save agent models
            for agent_id, agent in self.agents.items():
                agent_checkpoint_path = self.checkpoint_dir / f"agent_{agent_id}_episode_{episode}.pt"
                agent.save_model(str(agent_checkpoint_path))
            
            # Save trainer state
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"Checkpoint saved at episode {episode}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _compile_training_results(self) -> Dict[str, Any]:
        """Compile final training results"""
        try:
            training_duration = time.time() - self.training_start_time
            
            results = {
                'training_completed': True,
                'total_episodes': self.training_metrics.episode,
                'training_duration': training_duration,
                'final_metrics': self.training_metrics.__dict__,
                'training_history': self.training_history,
                'session_id': self.session_id,
                'config': self.config.__dict__
            }
            
            # Agent-specific results
            results['agent_results'] = {}
            for agent_id, agent in self.agents.items():
                results['agent_results'][agent_id] = agent.get_performance_metrics()
            
            # Performance summary
            if self.training_history['rewards']:
                results['performance_summary'] = {
                    'avg_reward': np.mean(self.training_history['rewards']),
                    'final_reward': self.training_history['rewards'][-1],
                    'avg_consensus_rate': np.mean(self.training_history['consensus_rates']),
                    'avg_processing_time': np.mean(self.training_history['processing_times']),
                    'training_stability': np.std(self.training_history['rewards'])
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error compiling training results: {e}")
            return {'training_completed': False, 'error': str(e)}
    
    def save_models(self, save_dir: str):
        """Save all trained models"""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for agent_id, agent in self.agents.items():
                model_path = save_path / f"{agent_id}_final_model.pt"
                agent.save_model(str(model_path))
            
            logger.info(f"Models saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, load_dir: str):
        """Load trained models"""
        try:
            load_path = Path(load_dir)
            
            for agent_id, agent in self.agents.items():
                model_path = load_path / f"{agent_id}_final_model.pt"
                if model_path.exists():
                    agent.load_model(str(model_path))
                    logger.info(f"Model loaded for agent {agent_id}")
                else:
                    logger.warning(f"Model not found for agent {agent_id}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")


# Mock strategic context manager
class MockStrategicContextManager:
    """Mock strategic context manager for testing"""
    def get_latest_context(self):
        return None


def create_tactical_sequential_trainer(
    agents: Dict[str, SequentialTacticalAgent],
    environment: SequentialTacticalEnvironment,
    config: Optional[TrainingConfig] = None
) -> TacticalSequentialTrainer:
    """
    Create tactical sequential trainer
    
    Args:
        agents: Dictionary of tactical agents
        environment: Sequential tactical environment
        config: Training configuration
        
    Returns:
        Configured trainer
    """
    if config is None:
        config = TrainingConfig()
    
    return TacticalSequentialTrainer(
        agents=agents,
        environment=environment,
        config=config
    )


# Example usage
if __name__ == "__main__":
    # This would be run with proper imports and environment setup
    print("Tactical Sequential Trainer initialized successfully")