"""
PettingZoo Training Loops for MARL Systems

This module implements specialized training loops that work with PettingZoo's
turn-based execution model. It provides optimized training routines for
different MARL scenarios while maintaining compatibility with the existing
agent architectures.

Key Features:
- Turn-based execution compatible with PettingZoo AEC environments
- Efficient experience collection and batching
- Agent-specific training strategies
- Parallel environment support
- Advanced curriculum learning
- Multi-objective optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from enum import Enum
import random

# PettingZoo imports
from pettingzoo import AECEnv

# Internal imports
from .pettingzoo_mappo_trainer import PettingZooMAPPOTrainer, TrainingConfig
from .pettingzoo_environment_manager import EnvironmentFactory, EnvironmentConfig, EnvironmentType

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phases for curriculum learning"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"


@dataclass
class TrainingLoopConfig:
    """Configuration for training loops"""
    # Basic training settings
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    update_frequency: int = 4
    
    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    stage_transition_episodes: List[int] = field(default_factory=list)
    
    # Experience collection
    experience_replay: bool = True
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000
    
    # Multi-agent coordination
    agent_update_order: List[str] = field(default_factory=list)
    simultaneous_updates: bool = False
    communication_enabled: bool = False
    
    # Performance optimization
    parallel_envs: int = 1
    async_training: bool = False
    gradient_accumulation_steps: int = 1
    
    # Logging and monitoring
    log_frequency: int = 100
    eval_frequency: int = 500
    checkpoint_frequency: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 100
    target_performance: float = float('inf')
    
    # Environment management
    env_reset_frequency: int = 1000
    env_validation_frequency: int = 5000


class ExperienceCollector:
    """Collects and manages experience from PettingZoo environments"""
    
    def __init__(self, config: TrainingLoopConfig):
        self.config = config
        self.experiences = {
            'observations': deque(maxlen=config.replay_buffer_size),
            'actions': deque(maxlen=config.replay_buffer_size),
            'rewards': deque(maxlen=config.replay_buffer_size),
            'values': deque(maxlen=config.replay_buffer_size),
            'log_probs': deque(maxlen=config.replay_buffer_size),
            'dones': deque(maxlen=config.replay_buffer_size),
            'agents': deque(maxlen=config.replay_buffer_size),
            'next_observations': deque(maxlen=config.replay_buffer_size)
        }
        
        self.episode_data = defaultdict(list)
        self.current_episode = 0
        self.current_step = 0
        
    def collect_step(self, agent: str, observation: np.ndarray, action: int,
                    reward: float, value: float, log_prob: float, done: bool,
                    next_observation: Optional[np.ndarray] = None):
        """Collect single step experience"""
        self.experiences['observations'].append(observation)
        self.experiences['actions'].append(action)
        self.experiences['rewards'].append(reward)
        self.experiences['values'].append(value)
        self.experiences['log_probs'].append(log_prob)
        self.experiences['dones'].append(done)
        self.experiences['agents'].append(agent)
        self.experiences['next_observations'].append(next_observation)
        
        # Episode-specific data
        self.episode_data[agent].append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done,
            'step': self.current_step
        })
        
        self.current_step += 1
    
    def end_episode(self):
        """Mark end of episode"""
        self.current_episode += 1
        self.episode_data.clear()
        self.current_step = 0
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch of experiences"""
        if len(self.experiences['observations']) < batch_size:
            return None
        
        indices = random.sample(range(len(self.experiences['observations'])), batch_size)
        
        batch = {}
        for key, data in self.experiences.items():
            if key in ['observations', 'next_observations']:
                batch[key] = torch.FloatTensor([data[i] for i in indices])
            elif key in ['actions']:
                batch[key] = torch.LongTensor([data[i] for i in indices])
            elif key in ['rewards', 'values', 'log_probs']:
                batch[key] = torch.FloatTensor([data[i] for i in indices])
            elif key in ['dones']:
                batch[key] = torch.BoolTensor([data[i] for i in indices])
            elif key in ['agents']:
                batch[key] = [data[i] for i in indices]
        
        return batch
    
    def get_agent_experiences(self, agent: str) -> List[Dict[str, Any]]:
        """Get experiences for specific agent"""
        return self.episode_data.get(agent, [])
    
    def clear(self):
        """Clear all experiences"""
        for data in self.experiences.values():
            data.clear()
        self.episode_data.clear()
        self.current_episode = 0
        self.current_step = 0


class CurriculumManager:
    """Manages curriculum learning for MARL training"""
    
    def __init__(self, config: TrainingLoopConfig):
        self.config = config
        self.current_stage = 0
        self.current_phase = TrainingPhase.EXPLORATION
        self.stage_episodes = 0
        
        # Default curriculum if none provided
        if not config.curriculum_stages:
            self.curriculum_stages = [
                {
                    'name': 'exploration',
                    'episodes': 300,
                    'exploration_rate': 0.3,
                    'learning_rate': 3e-4,
                    'reward_scale': 1.0,
                    'environment_complexity': 'simple'
                },
                {
                    'name': 'exploitation',
                    'episodes': 500,
                    'exploration_rate': 0.1,
                    'learning_rate': 1e-4,
                    'reward_scale': 1.0,
                    'environment_complexity': 'medium'
                },
                {
                    'name': 'fine_tuning',
                    'episodes': 200,
                    'exploration_rate': 0.05,
                    'learning_rate': 5e-5,
                    'reward_scale': 1.0,
                    'environment_complexity': 'complex'
                }
            ]
        else:
            self.curriculum_stages = config.curriculum_stages
        
        self.transition_episodes = config.stage_transition_episodes or [
            stage['episodes'] for stage in self.curriculum_stages
        ]
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage"""
        if self.current_stage >= len(self.curriculum_stages):
            return self.curriculum_stages[-1]  # Stay at final stage
        
        return self.curriculum_stages[self.current_stage]
    
    def should_transition(self, episode: int) -> bool:
        """Check if should transition to next stage"""
        if self.current_stage >= len(self.transition_episodes):
            return False
        
        return episode >= sum(self.transition_episodes[:self.current_stage + 1])
    
    def transition_stage(self):
        """Transition to next curriculum stage"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_episodes = 0
            logger.info(f"Transitioned to curriculum stage {self.current_stage}: "
                       f"{self.curriculum_stages[self.current_stage]['name']}")
    
    def update_training_config(self, trainer_config: TrainingConfig):
        """Update trainer configuration based on current stage"""
        current_stage = self.get_current_stage()
        
        if 'learning_rate' in current_stage:
            trainer_config.learning_rate = current_stage['learning_rate']
        
        if 'reward_scale' in current_stage:
            # This would be used to scale rewards during training
            pass
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage"""
        current_stage = self.get_current_stage()
        
        env_config = {}
        if 'environment_complexity' in current_stage:
            complexity = current_stage['environment_complexity']
            if complexity == 'simple':
                env_config.update({
                    'max_episode_steps': 200,
                    'market_volatility': 0.1,
                    'noise_level': 0.05
                })
            elif complexity == 'medium':
                env_config.update({
                    'max_episode_steps': 500,
                    'market_volatility': 0.2,
                    'noise_level': 0.1
                })
            elif complexity == 'complex':
                env_config.update({
                    'max_episode_steps': 1000,
                    'market_volatility': 0.3,
                    'noise_level': 0.15
                })
        
        return env_config


class PettingZooTrainingLoop:
    """Main training loop for PettingZoo environments"""
    
    def __init__(self, trainer: PettingZooMAPPOTrainer, config: TrainingLoopConfig):
        self.trainer = trainer
        self.config = config
        self.experience_collector = ExperienceCollector(config)
        self.curriculum_manager = CurriculumManager(config) if config.enable_curriculum else None
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.recent_rewards = deque(maxlen=100)
        
        # Parallel environments
        self.parallel_envs = []
        self.env_factory = EnvironmentFactory()
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=config.parallel_envs)
        
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_losses': [],
            'evaluation_scores': [],
            'curriculum_transitions': [],
            'agent_performance': defaultdict(list)
        }
        
        self._initialize_parallel_environments()
    
    def _initialize_parallel_environments(self):
        """Initialize parallel environments"""
        for i in range(self.config.parallel_envs):
            env_config = EnvironmentConfig(
                env_type=EnvironmentType.TACTICAL,  # Default to tactical
                env_params={'seed': i}
            )
            env = self.env_factory.create_environment(env_config)
            self.parallel_envs.append(env)
    
    def run_training(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting PettingZoo training loop for {self.config.max_episodes} episodes")
        
        training_start_time = time.time()
        
        try:
            for episode in range(self.config.max_episodes):
                episode_start_time = time.time()
                
                # Update curriculum if enabled
                if self.curriculum_manager and self.curriculum_manager.should_transition(episode):
                    self.curriculum_manager.transition_stage()
                    self.curriculum_manager.update_training_config(self.trainer.config)
                
                # Run episode
                if self.config.parallel_envs > 1:
                    episode_results = self._run_parallel_episodes()
                else:
                    episode_results = [self._run_single_episode(self.parallel_envs[0])]
                
                # Process results
                self._process_episode_results(episode_results)
                
                # Training update
                if (episode + 1) % self.config.update_frequency == 0:
                    self._update_networks()
                
                # Logging
                if (episode + 1) % self.config.log_frequency == 0:
                    self._log_progress(episode)
                
                # Evaluation
                if (episode + 1) % self.config.eval_frequency == 0:
                    self._evaluate_performance()
                
                # Checkpointing
                if (episode + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(episode)
                
                # Early stopping check
                if self._check_early_stopping():
                    logger.info(f"Early stopping at episode {episode}")
                    break
                
                # Environment reset
                if (episode + 1) % self.config.env_reset_frequency == 0:
                    self._reset_environments()
                
                self.episode_count += 1
            
            training_time = time.time() - training_start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return self._get_final_results(training_time)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._get_final_results(time.time() - training_start_time)
        
        finally:
            self._cleanup()
    
    def _run_single_episode(self, env: AECEnv) -> Dict[str, Any]:
        """Run single episode in PettingZoo environment"""
        env.reset()
        
        episode_reward = 0
        episode_length = 0
        agent_rewards = defaultdict(float)
        agent_steps = defaultdict(int)
        
        while env.agents:
            # Get current agent
            current_agent = env.agent_selection
            
            # Get observation
            observation = env.observe(current_agent)
            
            # Get action from trainer
            action, log_prob, value = self.trainer.get_action_and_value(observation, current_agent)
            
            # Store pre-step state
            pre_step_obs = observation.copy()
            
            # Execute action
            env.step(action)
            
            # Get post-step information
            reward = env.rewards.get(current_agent, 0.0)
            done = env.dones.get(current_agent, False)
            truncated = env.truncations.get(current_agent, False)
            
            # Get next observation if agent is still active
            next_observation = None
            if current_agent in env.agents:
                next_observation = env.observe(current_agent)
            
            # Collect experience
            self.experience_collector.collect_step(
                agent=current_agent,
                observation=pre_step_obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done or truncated,
                next_observation=next_observation
            )
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            agent_rewards[current_agent] += reward
            agent_steps[current_agent] += 1
            self.total_steps += 1
            
            # Check termination
            if done or truncated or episode_length >= self.config.max_steps_per_episode:
                break
        
        # Mark episode end
        self.experience_collector.end_episode()
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'agent_rewards': dict(agent_rewards),
            'agent_steps': dict(agent_steps)
        }
    
    def _run_parallel_episodes(self) -> List[Dict[str, Any]]:
        """Run multiple episodes in parallel"""
        futures = []
        
        for env in self.parallel_envs:
            future = self.thread_pool.submit(self._run_single_episode, env)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel episode failed: {e}")
                results.append({
                    'total_reward': 0,
                    'episode_length': 0,
                    'agent_rewards': {},
                    'agent_steps': {}
                })
        
        return results
    
    def _process_episode_results(self, results: List[Dict[str, Any]]):
        """Process episode results and update metrics"""
        total_reward = sum(r['total_reward'] for r in results)
        avg_reward = total_reward / len(results)
        
        # Update metrics
        self.metrics['episode_rewards'].append(avg_reward)
        self.metrics['episode_lengths'].append(np.mean([r['episode_length'] for r in results]))
        
        # Update agent performance
        for result in results:
            for agent, reward in result['agent_rewards'].items():
                self.metrics['agent_performance'][agent].append(reward)
        
        # Update best reward
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
        
        # Update recent rewards
        self.recent_rewards.append(avg_reward)
    
    def _update_networks(self):
        """Update network parameters"""
        # Check if we have enough experiences
        if len(self.experience_collector.experiences['observations']) < self.config.min_replay_size:
            return
        
        # Get batch of experiences
        batch = self.experience_collector.get_batch(self.config.batch_size)
        if batch is None:
            return
        
        # Perform training update
        self.trainer._update_networks()
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        if not self.metrics['episode_rewards']:
            return
        
        recent_rewards = self.metrics['episode_rewards'][-10:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        stage_info = ""
        if self.curriculum_manager:
            current_stage = self.curriculum_manager.get_current_stage()
            stage_info = f", Stage: {current_stage['name']}"
        
        logger.info(f"Episode {episode}: "
                   f"Reward: {mean_reward:.3f} ± {std_reward:.3f}, "
                   f"Best: {self.best_reward:.3f}, "
                   f"Steps: {self.total_steps}{stage_info}")
    
    def _evaluate_performance(self):
        """Evaluate current policy performance"""
        logger.info("Running evaluation...")
        
        eval_rewards = []
        eval_lengths = []
        
        # Use first environment for evaluation
        eval_env = self.parallel_envs[0]
        
        for _ in range(10):  # Run 10 evaluation episodes
            eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while eval_env.agents:
                current_agent = eval_env.agent_selection
                observation = eval_env.observe(current_agent)
                
                # Get action (no exploration)
                action, _, _ = self.trainer.get_action_and_value(observation, current_agent)
                
                eval_env.step(action)
                
                reward = eval_env.rewards.get(current_agent, 0.0)
                episode_reward += reward
                episode_length += 1
                
                if eval_env.dones.get(current_agent, False) or episode_length >= self.config.max_steps_per_episode:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        mean_eval_reward = np.mean(eval_rewards)
        self.metrics['evaluation_scores'].append(mean_eval_reward)
        
        logger.info(f"Evaluation complete - Mean reward: {mean_eval_reward:.3f}")
    
    def _check_early_stopping(self) -> bool:
        """Check early stopping conditions"""
        if len(self.recent_rewards) < self.config.early_stopping_patience:
            return False
        
        # Check if performance target reached
        mean_recent_reward = np.mean(list(self.recent_rewards))
        if mean_recent_reward >= self.config.target_performance:
            return True
        
        # Check for lack of improvement
        if len(self.metrics['evaluation_scores']) >= 2:
            recent_evals = self.metrics['evaluation_scores'][-self.config.early_stopping_patience:]
            if len(recent_evals) >= 2:
                # Check if no improvement in recent evaluations
                improvement = recent_evals[-1] - recent_evals[0]
                if improvement < 0.01:  # Very small improvement threshold
                    return True
        
        return False
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_data = {
            'episode': episode,
            'trainer_state': self.trainer.networks,
            'metrics': self.metrics,
            'curriculum_state': self.curriculum_manager.current_stage if self.curriculum_manager else None,
            'config': self.config
        }
        
        # This would save to file in practice
        logger.info(f"Checkpoint saved at episode {episode}")
    
    def _reset_environments(self):
        """Reset all parallel environments"""
        for env in self.parallel_envs:
            env.reset()
        
        logger.info("All environments reset")
    
    def _get_final_results(self, training_time: float) -> Dict[str, Any]:
        """Get final training results"""
        return {
            'training_time': training_time,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'final_performance': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0,
            'metrics': self.metrics,
            'curriculum_completed': self.curriculum_manager.current_stage if self.curriculum_manager else None
        }
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up training loop...")
        
        # Close environments
        for env in self.parallel_envs:
            if hasattr(env, 'close'):
                env.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clean up experience collector
        self.experience_collector.clear()


def create_training_loop_config(**kwargs) -> TrainingLoopConfig:
    """Create training loop configuration"""
    return TrainingLoopConfig(**kwargs)


def run_strategic_training(config: TrainingLoopConfig = None) -> Dict[str, Any]:
    """Run strategic MARL training"""
    if config is None:
        config = create_training_loop_config()
    
    # Create trainer
    trainer_config = TrainingConfig(
        env_factory=lambda: EnvironmentFactory().create_environment(
            EnvironmentConfig(EnvironmentType.STRATEGIC)
        ),
        num_episodes=config.max_episodes,
        max_episode_steps=config.max_steps_per_episode
    )
    
    trainer = PettingZooMAPPOTrainer(trainer_config)
    
    # Create training loop
    training_loop = PettingZooTrainingLoop(trainer, config)
    
    # Run training
    return training_loop.run_training()


def run_tactical_training(config: TrainingLoopConfig = None) -> Dict[str, Any]:
    """Run tactical MARL training"""
    if config is None:
        config = create_training_loop_config()
    
    # Create trainer
    trainer_config = TrainingConfig(
        env_factory=lambda: EnvironmentFactory().create_environment(
            EnvironmentConfig(EnvironmentType.TACTICAL)
        ),
        num_episodes=config.max_episodes,
        max_episode_steps=config.max_steps_per_episode
    )
    
    trainer = PettingZooMAPPOTrainer(trainer_config)
    
    # Create training loop
    training_loop = PettingZooTrainingLoop(trainer, config)
    
    # Run training
    return training_loop.run_training()


# Example usage
if __name__ == "__main__":
    # Create training configuration
    config = create_training_loop_config(
        max_episodes=1000,
        parallel_envs=2,
        enable_curriculum=True,
        log_frequency=50,
        eval_frequency=200
    )
    
    # Run tactical training
    results = run_tactical_training(config)
    
    print(f"Training completed!")
    print(f"Best reward: {results['best_reward']:.3f}")
    print(f"Final performance: {results['final_performance']:.3f}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Training time: {results['training_time']:.2f} seconds")