"""
Cascade Curriculum Learning for Sequential Agent Training

This module implements a sophisticated curriculum learning system specifically
designed for training cascade agent systems with superposition outputs. It
progressively increases training complexity and introduces agents sequentially
to build stable coordination patterns.

Key Features:
- Progressive agent introduction in cascade sequence
- Complexity-based curriculum progression
- Superposition-aware difficulty scaling
- Coordination-focused learning phases
- Adaptive curriculum based on performance metrics

Author: AGENT 9 - Superposition-Aware Training Framework
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from collections import deque, defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CurriculumPhase(Enum):
    """Phases of cascade curriculum learning"""
    INITIALIZATION = "initialization"
    SINGLE_AGENT_TRAINING = "single_agent_training"
    PAIR_COORDINATION = "pair_coordination"
    SEQUENTIAL_INTRODUCTION = "sequential_introduction"
    FULL_CASCADE_TRAINING = "full_cascade_training"
    SUPERPOSITION_OPTIMIZATION = "superposition_optimization"
    ADVANCED_COORDINATION = "advanced_coordination"


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CurriculumConfig:
    """Configuration for cascade curriculum learning"""
    # Phase progression parameters
    phase_transition_threshold: float = 0.75  # Performance threshold for phase transition
    phase_stability_episodes: int = 50  # Episodes of stable performance before transition
    max_episodes_per_phase: int = 1000  # Maximum episodes per phase
    
    # Agent introduction parameters
    agent_introduction_order: List[str] = field(default_factory=list)
    coordination_threshold: float = 0.6  # Threshold for good coordination
    superposition_complexity_factor: float = 0.1  # Factor for superposition complexity
    
    # Difficulty scaling parameters
    difficulty_progression_rate: float = 0.02  # Rate of difficulty increase
    performance_smoothing_window: int = 20  # Window for performance smoothing
    adaptation_sensitivity: float = 0.1  # Sensitivity to performance changes
    
    # Superposition-specific parameters
    entropy_curriculum_weight: float = 0.3  # Weight for entropy in curriculum
    consistency_curriculum_weight: float = 0.4  # Weight for consistency in curriculum
    coordination_curriculum_weight: float = 0.3  # Weight for coordination in curriculum
    
    # Environment complexity parameters
    env_complexity_factors: Dict[str, float] = field(default_factory=lambda: {
        'market_volatility': 0.2,
        'regime_instability': 0.3,
        'execution_complexity': 0.2,
        'risk_constraints': 0.3
    })


@dataclass
class PhaseMetrics:
    """Metrics for tracking curriculum phase performance"""
    phase: CurriculumPhase
    episode_count: int
    performance_scores: List[float]
    coordination_scores: List[float]
    superposition_quality: List[float]
    stability_metric: float
    transition_ready: bool
    
    def get_average_performance(self) -> float:
        """Get average performance for this phase"""
        return np.mean(self.performance_scores) if self.performance_scores else 0.0
    
    def get_performance_trend(self) -> float:
        """Get performance trend (positive means improving)"""
        if len(self.performance_scores) < 10:
            return 0.0
        
        recent = np.mean(self.performance_scores[-10:])
        older = np.mean(self.performance_scores[-20:-10]) if len(self.performance_scores) >= 20 else recent
        
        return recent - older


@dataclass
class CascadeCurriculumState:
    """State of cascade curriculum learning system"""
    current_phase: CurriculumPhase
    active_agents: List[str]
    difficulty_level: DifficultyLevel
    phase_episode_count: int
    total_episode_count: int
    
    # Performance tracking
    current_performance: float
    performance_history: List[float]
    coordination_history: List[float]
    
    # Curriculum parameters
    current_complexity: float
    superposition_difficulty: float
    environment_difficulty: float
    
    # Phase metrics
    phase_metrics: Dict[CurriculumPhase, PhaseMetrics]


class CascadeCurriculumLearning:
    """Main curriculum learning system for cascade agent training"""
    
    def __init__(self, config: CurriculumConfig, agent_names: List[str],
                 superposition_dim: int):
        self.config = config
        self.agent_names = agent_names
        self.superposition_dim = superposition_dim
        
        # Initialize curriculum state
        self.state = CascadeCurriculumState(
            current_phase=CurriculumPhase.INITIALIZATION,
            active_agents=[],
            difficulty_level=DifficultyLevel.BASIC,
            phase_episode_count=0,
            total_episode_count=0,
            current_performance=0.0,
            performance_history=[],
            coordination_history=[],
            current_complexity=0.0,
            superposition_difficulty=0.0,
            environment_difficulty=0.0,
            phase_metrics={}
        )
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=config.performance_smoothing_window)
        self.coordination_buffer = deque(maxlen=config.performance_smoothing_window)
        self.superposition_quality_buffer = deque(maxlen=config.performance_smoothing_window)
        
        # Phase transition callbacks
        self.phase_callbacks = {
            CurriculumPhase.INITIALIZATION: self._initialize_curriculum,
            CurriculumPhase.SINGLE_AGENT_TRAINING: self._single_agent_training,
            CurriculumPhase.PAIR_COORDINATION: self._pair_coordination,
            CurriculumPhase.SEQUENTIAL_INTRODUCTION: self._sequential_introduction,
            CurriculumPhase.FULL_CASCADE_TRAINING: self._full_cascade_training,
            CurriculumPhase.SUPERPOSITION_OPTIMIZATION: self._superposition_optimization,
            CurriculumPhase.ADVANCED_COORDINATION: self._advanced_coordination
        }
        
        # Curriculum metrics
        self.curriculum_metrics = {
            'phase_transitions': 0,
            'total_training_time': 0.0,
            'successful_transitions': 0,
            'failed_transitions': 0,
            'optimal_performance_episodes': 0
        }
        
        logger.info(f"Initialized CascadeCurriculumLearning for {len(agent_names)} agents")
    
    def get_training_configuration(self) -> Dict[str, Any]:
        """Get current training configuration based on curriculum state"""
        config = {
            'active_agents': self.state.active_agents,
            'difficulty_level': self.state.difficulty_level.value,
            'current_phase': self.state.current_phase.value,
            'complexity_scaling': self._get_complexity_scaling(),
            'superposition_config': self._get_superposition_config(),
            'environment_config': self._get_environment_config(),
            'training_parameters': self._get_training_parameters()
        }
        
        return config
    
    def update_curriculum(self, episode_metrics: Dict[str, Any]) -> bool:
        """
        Update curriculum based on episode metrics
        
        Args:
            episode_metrics: Dictionary containing:
                - agent_rewards: Dict[str, float] - Rewards for each agent
                - coordination_score: float - Overall coordination quality
                - superposition_metrics: Dict[str, float] - Superposition quality metrics
                - episode_length: int - Length of episode
                - success_rate: float - Success rate for episode
        
        Returns:
            bool: True if phase transition occurred
        """
        # Extract metrics
        agent_rewards = episode_metrics.get('agent_rewards', {})
        coordination_score = episode_metrics.get('coordination_score', 0.0)
        superposition_metrics = episode_metrics.get('superposition_metrics', {})
        
        # Calculate overall performance
        performance_score = np.mean(list(agent_rewards.values())) if agent_rewards else 0.0
        superposition_quality = np.mean(list(superposition_metrics.values())) if superposition_metrics else 0.0
        
        # Update buffers
        self.performance_buffer.append(performance_score)
        self.coordination_buffer.append(coordination_score)
        self.superposition_quality_buffer.append(superposition_quality)
        
        # Update state
        self.state.current_performance = np.mean(list(self.performance_buffer))
        self.state.performance_history.append(performance_score)
        self.state.coordination_history.append(coordination_score)
        
        # Update episode counts
        self.state.phase_episode_count += 1
        self.state.total_episode_count += 1
        
        # Update phase metrics
        self._update_phase_metrics(performance_score, coordination_score, superposition_quality)
        
        # Check for phase transition
        transition_occurred = self._check_phase_transition()
        
        # Update curriculum parameters
        self._update_curriculum_parameters()
        
        return transition_occurred
    
    def _get_complexity_scaling(self) -> Dict[str, float]:
        """Get complexity scaling factors for different components"""
        base_complexity = self.state.current_complexity
        
        return {
            'observation_noise': base_complexity * 0.1,
            'action_space_complexity': base_complexity * 0.2,
            'reward_sparsity': base_complexity * 0.3,
            'temporal_dependencies': base_complexity * 0.4,
            'coordination_requirements': base_complexity * 0.5
        }
    
    def _get_superposition_config(self) -> Dict[str, Any]:
        """Get superposition-specific configuration"""
        superposition_difficulty = self.state.superposition_difficulty
        
        return {
            'entropy_target': max(0.5, 2.0 - superposition_difficulty),
            'consistency_threshold': min(0.9, 0.5 + superposition_difficulty * 0.4),
            'confidence_variance_penalty': superposition_difficulty * 0.3,
            'superposition_dropout_rate': max(0.0, 0.3 - superposition_difficulty * 0.2),
            'temperature_scaling': max(0.5, 1.5 - superposition_difficulty * 0.5)
        }
    
    def _get_environment_config(self) -> Dict[str, float]:
        """Get environment configuration based on curriculum"""
        env_difficulty = self.state.environment_difficulty
        
        config = {}
        for factor, weight in self.config.env_complexity_factors.items():
            config[factor] = env_difficulty * weight
        
        return config
    
    def _get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters adjusted for curriculum"""
        difficulty_factor = self.state.current_complexity
        
        return {
            'learning_rate': max(1e-5, 3e-4 * (1.0 - difficulty_factor * 0.5)),
            'batch_size': max(32, int(64 * (1.0 + difficulty_factor * 0.5))),
            'update_frequency': max(1, int(1 + difficulty_factor * 2)),
            'exploration_epsilon': max(0.01, 0.1 * (1.0 - difficulty_factor)),
            'entropy_coefficient': max(0.001, 0.01 * (1.0 + difficulty_factor)),
            'gradient_clip_norm': max(0.1, 0.5 * (1.0 - difficulty_factor * 0.3))
        }
    
    def _update_phase_metrics(self, performance: float, coordination: float, 
                            superposition_quality: float):
        """Update metrics for current phase"""
        current_phase = self.state.current_phase
        
        if current_phase not in self.state.phase_metrics:
            self.state.phase_metrics[current_phase] = PhaseMetrics(
                phase=current_phase,
                episode_count=0,
                performance_scores=[],
                coordination_scores=[],
                superposition_quality=[],
                stability_metric=0.0,
                transition_ready=False
            )
        
        metrics = self.state.phase_metrics[current_phase]
        metrics.episode_count += 1
        metrics.performance_scores.append(performance)
        metrics.coordination_scores.append(coordination)
        metrics.superposition_quality.append(superposition_quality)
        
        # Calculate stability metric
        if len(metrics.performance_scores) >= 10:
            recent_std = np.std(metrics.performance_scores[-10:])
            metrics.stability_metric = 1.0 / (1.0 + recent_std)
        
        # Check if ready for transition
        metrics.transition_ready = self._is_phase_ready_for_transition(metrics)
    
    def _check_phase_transition(self) -> bool:
        """Check if current phase should transition to next phase"""
        current_phase = self.state.current_phase
        
        if current_phase not in self.state.phase_metrics:
            return False
        
        metrics = self.state.phase_metrics[current_phase]
        
        # Check transition conditions
        performance_ready = metrics.get_average_performance() >= self.config.phase_transition_threshold
        stability_ready = metrics.stability_metric >= 0.7
        episode_count_ready = metrics.episode_count >= self.config.phase_stability_episodes
        
        # Phase-specific conditions
        phase_specific_ready = self._check_phase_specific_conditions(current_phase, metrics)
        
        # Force transition if max episodes reached
        force_transition = self.state.phase_episode_count >= self.config.max_episodes_per_phase
        
        if (performance_ready and stability_ready and episode_count_ready and phase_specific_ready) or force_transition:
            return self._transition_to_next_phase()
        
        return False
    
    def _check_phase_specific_conditions(self, phase: CurriculumPhase, 
                                       metrics: PhaseMetrics) -> bool:
        """Check phase-specific transition conditions"""
        if phase == CurriculumPhase.INITIALIZATION:
            return True  # Always ready to start training
        
        elif phase == CurriculumPhase.SINGLE_AGENT_TRAINING:
            # Check if single agent is performing well
            return metrics.get_average_performance() >= 0.6
        
        elif phase == CurriculumPhase.PAIR_COORDINATION:
            # Check if pairs are coordinating well
            coordination_score = np.mean(metrics.coordination_scores[-20:]) if len(metrics.coordination_scores) >= 20 else 0.0
            return coordination_score >= self.config.coordination_threshold
        
        elif phase == CurriculumPhase.SEQUENTIAL_INTRODUCTION:
            # Check if sequential introduction is working
            return len(self.state.active_agents) >= 3
        
        elif phase == CurriculumPhase.FULL_CASCADE_TRAINING:
            # Check if full cascade is stable
            return len(self.state.active_agents) == len(self.agent_names)
        
        elif phase == CurriculumPhase.SUPERPOSITION_OPTIMIZATION:
            # Check if superposition quality is good
            superposition_quality = np.mean(metrics.superposition_quality[-20:]) if len(metrics.superposition_quality) >= 20 else 0.0
            return superposition_quality >= 0.7
        
        elif phase == CurriculumPhase.ADVANCED_COORDINATION:
            # Advanced coordination is the final phase
            return False
        
        return False
    
    def _transition_to_next_phase(self) -> bool:
        """Transition to the next curriculum phase"""
        current_phase = self.state.current_phase
        
        # Determine next phase
        phase_order = [
            CurriculumPhase.INITIALIZATION,
            CurriculumPhase.SINGLE_AGENT_TRAINING,
            CurriculumPhase.PAIR_COORDINATION,
            CurriculumPhase.SEQUENTIAL_INTRODUCTION,
            CurriculumPhase.FULL_CASCADE_TRAINING,
            CurriculumPhase.SUPERPOSITION_OPTIMIZATION,
            CurriculumPhase.ADVANCED_COORDINATION
        ]
        
        try:
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                next_phase = phase_order[current_idx + 1]
                
                # Execute phase transition
                success = self._execute_phase_transition(current_phase, next_phase)
                
                if success:
                    self.state.current_phase = next_phase
                    self.state.phase_episode_count = 0
                    self.curriculum_metrics['phase_transitions'] += 1
                    self.curriculum_metrics['successful_transitions'] += 1
                    
                    logger.info(f"Curriculum phase transition: {current_phase.value} -> {next_phase.value}")
                    return True
                else:
                    self.curriculum_metrics['failed_transitions'] += 1
                    logger.warning(f"Failed to transition from {current_phase.value} to {next_phase.value}")
        
        except ValueError:
            logger.error(f"Invalid current phase: {current_phase}")
        
        return False
    
    def _execute_phase_transition(self, current_phase: CurriculumPhase, 
                                 next_phase: CurriculumPhase) -> bool:
        """Execute the transition between phases"""
        try:
            # Execute current phase cleanup
            if current_phase in self.phase_callbacks:
                self.phase_callbacks[current_phase]()
            
            # Execute next phase initialization
            if next_phase in self.phase_callbacks:
                self.phase_callbacks[next_phase]()
            
            return True
        
        except Exception as e:
            logger.error(f"Error during phase transition: {e}")
            return False
    
    def _initialize_curriculum(self):
        """Initialize curriculum learning system"""
        self.state.active_agents = []
        self.state.difficulty_level = DifficultyLevel.BASIC
        self.state.current_complexity = 0.0
        self.state.superposition_difficulty = 0.0
        self.state.environment_difficulty = 0.0
        
        logger.info("Curriculum initialized")
    
    def _single_agent_training(self):
        """Configure for single agent training phase"""
        # Start with first agent (usually regime detection)
        if self.config.agent_introduction_order:
            first_agent = self.config.agent_introduction_order[0]
        else:
            first_agent = self.agent_names[0]
        
        self.state.active_agents = [first_agent]
        self.state.difficulty_level = DifficultyLevel.BASIC
        self.state.current_complexity = 0.1
        self.state.superposition_difficulty = 0.0
        
        logger.info(f"Single agent training phase: {first_agent}")
    
    def _pair_coordination(self):
        """Configure for pair coordination phase"""
        # Add second agent
        if len(self.config.agent_introduction_order) >= 2:
            second_agent = self.config.agent_introduction_order[1]
        else:
            second_agent = self.agent_names[1] if len(self.agent_names) > 1 else self.agent_names[0]
        
        if second_agent not in self.state.active_agents:
            self.state.active_agents.append(second_agent)
        
        self.state.difficulty_level = DifficultyLevel.BASIC
        self.state.current_complexity = 0.2
        self.state.superposition_difficulty = 0.1
        
        logger.info(f"Pair coordination phase: {self.state.active_agents}")
    
    def _sequential_introduction(self):
        """Configure for sequential agent introduction"""
        # Add agents one by one
        if len(self.state.active_agents) < len(self.agent_names):
            if self.config.agent_introduction_order:
                next_agent_idx = len(self.state.active_agents)
                if next_agent_idx < len(self.config.agent_introduction_order):
                    next_agent = self.config.agent_introduction_order[next_agent_idx]
                    self.state.active_agents.append(next_agent)
            else:
                # Default order
                next_agent_idx = len(self.state.active_agents)
                if next_agent_idx < len(self.agent_names):
                    next_agent = self.agent_names[next_agent_idx]
                    self.state.active_agents.append(next_agent)
        
        self.state.difficulty_level = DifficultyLevel.INTERMEDIATE
        self.state.current_complexity = 0.4
        self.state.superposition_difficulty = 0.2
        
        logger.info(f"Sequential introduction phase: {self.state.active_agents}")
    
    def _full_cascade_training(self):
        """Configure for full cascade training"""
        # Ensure all agents are active
        self.state.active_agents = self.agent_names.copy()
        
        self.state.difficulty_level = DifficultyLevel.INTERMEDIATE
        self.state.current_complexity = 0.6
        self.state.superposition_difficulty = 0.4
        self.state.environment_difficulty = 0.3
        
        logger.info("Full cascade training phase")
    
    def _superposition_optimization(self):
        """Configure for superposition optimization phase"""
        self.state.difficulty_level = DifficultyLevel.ADVANCED
        self.state.current_complexity = 0.8
        self.state.superposition_difficulty = 0.7
        self.state.environment_difficulty = 0.5
        
        logger.info("Superposition optimization phase")
    
    def _advanced_coordination(self):
        """Configure for advanced coordination phase"""
        self.state.difficulty_level = DifficultyLevel.EXPERT
        self.state.current_complexity = 1.0
        self.state.superposition_difficulty = 1.0
        self.state.environment_difficulty = 0.8
        
        logger.info("Advanced coordination phase")
    
    def _update_curriculum_parameters(self):
        """Update curriculum parameters based on performance"""
        # Adaptive complexity adjustment
        if len(self.performance_buffer) >= 10:
            recent_performance = np.mean(list(self.performance_buffer)[-10:])
            
            # Increase complexity if performance is good
            if recent_performance > 0.8:
                self.state.current_complexity = min(1.0, 
                    self.state.current_complexity + self.config.difficulty_progression_rate)
            
            # Decrease complexity if performance is poor
            elif recent_performance < 0.4:
                self.state.current_complexity = max(0.0, 
                    self.state.current_complexity - self.config.difficulty_progression_rate)
        
        # Superposition difficulty adjustment
        if len(self.superposition_quality_buffer) >= 10:
            recent_quality = np.mean(list(self.superposition_quality_buffer)[-10:])
            
            if recent_quality > 0.7:
                self.state.superposition_difficulty = min(1.0,
                    self.state.superposition_difficulty + self.config.difficulty_progression_rate)
            elif recent_quality < 0.5:
                self.state.superposition_difficulty = max(0.0,
                    self.state.superposition_difficulty - self.config.difficulty_progression_rate)
        
        # Environment difficulty adjustment
        if len(self.coordination_buffer) >= 10:
            recent_coordination = np.mean(list(self.coordination_buffer)[-10:])
            
            if recent_coordination > 0.8:
                self.state.environment_difficulty = min(1.0,
                    self.state.environment_difficulty + self.config.difficulty_progression_rate)
            elif recent_coordination < 0.4:
                self.state.environment_difficulty = max(0.0,
                    self.state.environment_difficulty - self.config.difficulty_progression_rate)
    
    def _is_phase_ready_for_transition(self, metrics: PhaseMetrics) -> bool:
        """Check if phase is ready for transition"""
        if metrics.episode_count < self.config.phase_stability_episodes:
            return False
        
        avg_performance = metrics.get_average_performance()
        performance_trend = metrics.get_performance_trend()
        
        return (avg_performance >= self.config.phase_transition_threshold and
                performance_trend >= -0.1 and  # Not declining significantly
                metrics.stability_metric >= 0.7)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status"""
        return {
            'current_phase': self.state.current_phase.value,
            'active_agents': self.state.active_agents,
            'difficulty_level': self.state.difficulty_level.value,
            'phase_episode_count': self.state.phase_episode_count,
            'total_episode_count': self.state.total_episode_count,
            'current_performance': self.state.current_performance,
            'current_complexity': self.state.current_complexity,
            'superposition_difficulty': self.state.superposition_difficulty,
            'environment_difficulty': self.state.environment_difficulty,
            'metrics': self.curriculum_metrics,
            'phase_metrics': {
                phase.value: {
                    'episode_count': metrics.episode_count,
                    'avg_performance': metrics.get_average_performance(),
                    'performance_trend': metrics.get_performance_trend(),
                    'stability_metric': metrics.stability_metric,
                    'transition_ready': metrics.transition_ready
                }
                for phase, metrics in self.state.phase_metrics.items()
            }
        }
    
    def save_curriculum_state(self, filepath: str):
        """Save curriculum state to file"""
        state_dict = {
            'current_phase': self.state.current_phase.value,
            'active_agents': self.state.active_agents,
            'difficulty_level': self.state.difficulty_level.value,
            'phase_episode_count': self.state.phase_episode_count,
            'total_episode_count': self.state.total_episode_count,
            'current_performance': self.state.current_performance,
            'performance_history': self.state.performance_history,
            'coordination_history': self.state.coordination_history,
            'current_complexity': self.state.current_complexity,
            'superposition_difficulty': self.state.superposition_difficulty,
            'environment_difficulty': self.state.environment_difficulty,
            'curriculum_metrics': self.curriculum_metrics,
            'config': {
                'phase_transition_threshold': self.config.phase_transition_threshold,
                'phase_stability_episodes': self.config.phase_stability_episodes,
                'max_episodes_per_phase': self.config.max_episodes_per_phase,
                'agent_introduction_order': self.config.agent_introduction_order,
                'coordination_threshold': self.config.coordination_threshold,
                'difficulty_progression_rate': self.config.difficulty_progression_rate,
                'env_complexity_factors': self.config.env_complexity_factors
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Curriculum state saved to {filepath}")
    
    def load_curriculum_state(self, filepath: str):
        """Load curriculum state from file"""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Restore state
        self.state.current_phase = CurriculumPhase(state_dict['current_phase'])
        self.state.active_agents = state_dict['active_agents']
        self.state.difficulty_level = DifficultyLevel(state_dict['difficulty_level'])
        self.state.phase_episode_count = state_dict['phase_episode_count']
        self.state.total_episode_count = state_dict['total_episode_count']
        self.state.current_performance = state_dict['current_performance']
        self.state.performance_history = state_dict['performance_history']
        self.state.coordination_history = state_dict['coordination_history']
        self.state.current_complexity = state_dict['current_complexity']
        self.state.superposition_difficulty = state_dict['superposition_difficulty']
        self.state.environment_difficulty = state_dict['environment_difficulty']
        self.curriculum_metrics = state_dict['curriculum_metrics']
        
        # Restore buffers
        self.performance_buffer = deque(self.state.performance_history[-self.config.performance_smoothing_window:], 
                                       maxlen=self.config.performance_smoothing_window)
        self.coordination_buffer = deque(self.state.coordination_history[-self.config.performance_smoothing_window:], 
                                        maxlen=self.config.performance_smoothing_window)
        
        logger.info(f"Curriculum state loaded from {filepath}")
    
    def reset_curriculum(self):
        """Reset curriculum to initial state"""
        self.state = CascadeCurriculumState(
            current_phase=CurriculumPhase.INITIALIZATION,
            active_agents=[],
            difficulty_level=DifficultyLevel.BASIC,
            phase_episode_count=0,
            total_episode_count=0,
            current_performance=0.0,
            performance_history=[],
            coordination_history=[],
            current_complexity=0.0,
            superposition_difficulty=0.0,
            environment_difficulty=0.0,
            phase_metrics={}
        )
        
        self.performance_buffer.clear()
        self.coordination_buffer.clear()
        self.superposition_quality_buffer.clear()
        
        self.curriculum_metrics = {
            'phase_transitions': 0,
            'total_training_time': 0.0,
            'successful_transitions': 0,
            'failed_transitions': 0,
            'optimal_performance_episodes': 0
        }
        
        logger.info("Curriculum reset to initial state")


class CurriculumEnvironmentWrapper:
    """Wrapper for environment to apply curriculum learning settings"""
    
    def __init__(self, base_env, curriculum_system: CascadeCurriculumLearning):
        self.base_env = base_env
        self.curriculum_system = curriculum_system
    
    def reset(self):
        """Reset environment with curriculum settings"""
        curriculum_config = self.curriculum_system.get_training_configuration()
        
        # Apply environment configuration
        env_config = curriculum_config.get('environment_config', {})
        self._apply_environment_config(env_config)
        
        # Filter active agents
        active_agents = curriculum_config.get('active_agents', [])
        self._set_active_agents(active_agents)
        
        return self.base_env.reset()
    
    def step(self, actions):
        """Step environment with curriculum filtering"""
        curriculum_config = self.curriculum_system.get_training_configuration()
        active_agents = curriculum_config.get('active_agents', [])
        
        # Filter actions for active agents only
        filtered_actions = {agent: action for agent, action in actions.items() 
                          if agent in active_agents}
        
        return self.base_env.step(filtered_actions)
    
    def _apply_environment_config(self, config: Dict[str, float]):
        """Apply environment configuration"""
        for param, value in config.items():
            if hasattr(self.base_env, param):
                setattr(self.base_env, param, value)
    
    def _set_active_agents(self, active_agents: List[str]):
        """Set active agents in environment"""
        if hasattr(self.base_env, 'set_active_agents'):
            self.base_env.set_active_agents(active_agents)
    
    def __getattr__(self, name):
        """Delegate attribute access to base environment"""
        return getattr(self.base_env, name)