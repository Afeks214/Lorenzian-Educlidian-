"""
Risk Sequential MARL Trainer

This module implements a specialized training framework for sequential risk MARL agents.
The trainer handles the unique challenges of sequential decision making in risk management:

1. Sequential coordination between agents
2. VaR correlation system integration
3. Real-time performance constraints (<5ms)
4. Emergency protocol handling
5. Risk superposition generation

Key Features:
- MAPPO-based training with sequential adaptations
- VaR-aware reward shaping
- Correlation shock simulation
- Emergency protocol training
- Performance-constrained optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from dataclasses import dataclass
import json
import structlog

from src.environment.sequential_risk_env import SequentialRiskEnvironment, create_sequential_risk_environment
from src.agents.risk.sequential_risk_agents import (
    SequentialRiskAgent, create_sequential_risk_agents, SequentialContext
)
from src.training.sequential_experience_buffer import SequentialExperienceBuffer
from src.training.superposition_reward_shaping import SuperpositionRewardShaper
from src.training.superposition_mappo_trainer import SuperpositionMAPPOTrainer
from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker
from src.risk.core.var_calculator import VaRCalculator
from src.safety.trading_system_controller import get_controller

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for sequential risk training"""
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Sequential-specific parameters
    sequence_coordination_weight: float = 0.3
    var_performance_weight: float = 0.2
    emergency_protocol_weight: float = 0.4
    correlation_awareness_weight: float = 0.1
    
    # Performance constraints
    max_var_calculation_time_ms: float = 5.0
    max_training_time_hours: float = 24.0
    performance_evaluation_frequency: int = 100
    
    # Risk scenario training
    risk_scenario_probabilities: Dict[str, float] = None
    correlation_shock_training: bool = True
    black_swan_simulation: bool = True
    
    # Validation parameters
    validation_frequency: int = 50
    validation_episodes: int = 10
    early_stopping_patience: int = 5
    
    def __post_init__(self):
        if self.risk_scenario_probabilities is None:
            self.risk_scenario_probabilities = {
                'normal': 0.6,
                'correlation_spike': 0.15,
                'liquidity_crisis': 0.1,
                'flash_crash': 0.1,
                'black_swan': 0.05
            }


@dataclass
class TrainingMetrics:
    """Training metrics for sequential risk agents"""
    episode: int
    total_reward: float
    individual_rewards: Dict[str, float]
    var_performance_met: bool
    avg_var_calculation_time_ms: float
    emergency_activations: int
    correlation_regime_distribution: Dict[str, int]
    risk_superposition_quality: float
    sequence_coordination_score: float
    constraint_violation_rate: float
    timestamp: datetime


class RiskSequentialTrainer:
    """
    Trainer for sequential risk MARL agents
    
    This trainer specializes in training agents that must coordinate sequentially
    while maintaining real-time performance constraints and handling risk events.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.event_bus = EventBus()
        self._initialize_risk_components()
        self._initialize_environment()
        self._initialize_agents()
        self._initialize_training_infrastructure()
        
        # Training state
        self.current_episode = 0
        self.global_step = 0
        self.training_start_time = datetime.now()
        self.best_validation_score = -np.inf
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []
        self.performance_violations = []
        
        # Risk scenario tracking
        self.scenario_performance = defaultdict(list)
        self.correlation_shock_responses = []
        
        logger.info("Risk Sequential Trainer initialized", 
                   config=config.__dict__, 
                   device=str(self.device))
    
    def _initialize_risk_components(self):
        """Initialize risk management components"""
        self.correlation_tracker = CorrelationTracker(
            event_bus=self.event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.3,
            shock_window_minutes=5,
            performance_target_ms=self.config.max_var_calculation_time_ms
        )
        
        self.var_calculator = VaRCalculator(
            correlation_tracker=self.correlation_tracker,
            event_bus=self.event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 10],
            default_method="parametric"
        )
        
        # Initialize asset universe
        self.asset_universe = ['SPY', 'QQQ', 'IWM', 'VTI', 'TLT', 'GLD', 'VIX', 'UUP', 'EFA', 'EEM']
        self.correlation_tracker.initialize_assets(self.asset_universe)
    
    def _initialize_environment(self):
        """Initialize sequential risk environment"""
        env_config = {
            'initial_capital': 1_000_000.0,
            'max_steps': self.config.max_steps_per_episode,
            'risk_tolerance': 0.05,
            'performance_target_ms': self.config.max_var_calculation_time_ms,
            'asset_universe': self.asset_universe,
            'critic_config': {
                'hidden_dim': 256,
                'num_layers': 4,
                'learning_rate': self.config.learning_rate * 0.5,
                'target_update_freq': 50
            }
        }
        
        self.environment = create_sequential_risk_environment(env_config)
        
        # Setup risk scenario rotation
        self.risk_scenarios = list(self.config.risk_scenario_probabilities.keys())
        self.scenario_probabilities = list(self.config.risk_scenario_probabilities.values())
    
    def _initialize_agents(self):
        """Initialize sequential risk agents"""
        agent_config = {
            'position_sizing': {
                'max_position_size': 0.2,
                'min_position_size': 0.01,
                'use_kelly_criterion': True,
                'kelly_multiplier': 0.25
            },
            'stop_target': {
                'min_stop_multiplier': 0.5,
                'max_stop_multiplier': 3.0,
                'min_target_multiplier': 1.0,
                'max_target_multiplier': 5.0,
                'target_risk_reward_ratio': 2.0
            },
            'risk_monitor': {
                'risk_thresholds': {
                    'var_breach': 0.05,
                    'correlation_spike': 0.8,
                    'drawdown_limit': 0.15,
                    'leverage_limit': 4.0
                },
                'emergency_threshold': 0.9
            },
            'portfolio_optimizer': {
                'optimization_method': 'mean_variance',
                'risk_aversion': 2.0,
                'max_asset_weight': 0.3,
                'min_asset_weight': 0.0
            }
        }
        
        self.agents = create_sequential_risk_agents(
            agent_config, self.event_bus, self.correlation_tracker, self.var_calculator
        )
    
    def _initialize_training_infrastructure(self):
        """Initialize training infrastructure"""
        # Experience buffer
        self.experience_buffer = SequentialExperienceBuffer(
            buffer_size=10000,
            batch_size=self.config.batch_size,
            sequence_length=4  # 4 agents in sequence
        )
        
        # Reward shaper
        self.reward_shaper = SuperpositionRewardShaper(
            sequence_coordination_weight=self.config.sequence_coordination_weight,
            var_performance_weight=self.config.var_performance_weight,
            emergency_protocol_weight=self.config.emergency_protocol_weight,
            correlation_awareness_weight=self.config.correlation_awareness_weight
        )
        
        # MAPPO trainer
        self.mappo_trainer = SuperpositionMAPPOTrainer(
            agents=self.agents,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            device=self.device
        )
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor(
            max_var_time_ms=self.config.max_var_calculation_time_ms,
            event_bus=self.event_bus
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop for sequential risk agents
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting sequential risk training", 
                   num_episodes=self.config.num_episodes,
                   max_steps=self.config.max_steps_per_episode)
        
        try:
            for episode in range(self.config.num_episodes):
                self.current_episode = episode
                
                # Check early stopping
                if self._should_early_stop():
                    logger.info("Early stopping triggered", episode=episode)
                    break
                
                # Check training time limit
                if self._training_time_exceeded():
                    logger.info("Training time limit exceeded", episode=episode)
                    break
                
                # Run training episode
                episode_metrics = self._run_training_episode()
                self.training_metrics.append(episode_metrics)
                
                # Update agents
                if len(self.experience_buffer) >= self.config.batch_size:
                    self._update_agents()
                
                # Validation
                if episode % self.config.validation_frequency == 0:
                    validation_metrics = self._run_validation()
                    self.validation_metrics.append(validation_metrics)
                    
                    # Check for improvement
                    if validation_metrics.total_reward > self.best_validation_score:
                        self.best_validation_score = validation_metrics.total_reward
                        self.early_stopping_counter = 0
                        self._save_best_model()
                    else:
                        self.early_stopping_counter += 1
                
                # Performance evaluation
                if episode % self.config.performance_evaluation_frequency == 0:
                    self._evaluate_performance()
                
                # Log progress
                if episode % 10 == 0:
                    self._log_progress(episode_metrics)
            
            # Final evaluation
            final_metrics = self._run_final_evaluation()
            
            # Save results
            results = self._compile_results(final_metrics)
            self._save_results(results)
            
            return results
        
        except Exception as e:
            logger.error("Training failed", error=str(e), exc_info=True)
            raise
        
        finally:
            self._cleanup()
    
    def _run_training_episode(self) -> TrainingMetrics:
        """Run a single training episode"""
        episode_start_time = datetime.now()
        
        # Select risk scenario
        scenario = np.random.choice(self.risk_scenarios, p=self.scenario_probabilities)
        
        # Reset environment with scenario
        self.environment.reset(options={'scenario': scenario})
        
        # Initialize episode state
        episode_rewards = {agent: 0.0 for agent in self.agents}
        episode_experiences = []
        var_calculation_times = []
        emergency_activations = 0
        correlation_regimes = defaultdict(int)
        
        # Run episode
        step = 0
        while step < self.config.max_steps_per_episode:
            # Get current agent
            current_agent = self.environment.agent_selection
            
            # Check if episode is done
            if self.environment.terminations[current_agent] or self.environment.truncations[current_agent]:
                break
            
            # Get observation
            observation = self.environment.observe(current_agent)
            
            # Get action from agent
            action = self.agents[current_agent].policy.predict(observation)
            
            # Step environment
            self.environment.step(action)
            
            # Collect experience
            reward = self.environment.rewards[current_agent]
            info = self.environment.infos[current_agent]
            
            # Track metrics
            episode_rewards[current_agent] += reward
            
            if 'var_calculation_time_ms' in info:
                var_calculation_times.append(info['var_calculation_time_ms'])
            
            if info.get('emergency_active', False):
                emergency_activations += 1
            
            correlation_regime = info.get('correlation_regime', 'NORMAL')
            correlation_regimes[correlation_regime] += 1
            
            # Store experience
            experience = {
                'agent': current_agent,
                'observation': observation,
                'action': action,
                'reward': reward,
                'info': info,
                'step': step
            }
            episode_experiences.append(experience)
            
            step += 1
        
        # Process experiences
        self._process_episode_experiences(episode_experiences)
        
        # Calculate metrics
        total_reward = sum(episode_rewards.values())
        var_performance_met = (
            np.mean(var_calculation_times) < self.config.max_var_calculation_time_ms
            if var_calculation_times else True
        )
        
        sequence_coordination_score = self._calculate_sequence_coordination_score(episode_experiences)
        risk_superposition_quality = self._calculate_risk_superposition_quality()
        
        # Create metrics
        metrics = TrainingMetrics(
            episode=self.current_episode,
            total_reward=total_reward,
            individual_rewards=episode_rewards,
            var_performance_met=var_performance_met,
            avg_var_calculation_time_ms=np.mean(var_calculation_times) if var_calculation_times else 0,
            emergency_activations=emergency_activations,
            correlation_regime_distribution=dict(correlation_regimes),
            risk_superposition_quality=risk_superposition_quality,
            sequence_coordination_score=sequence_coordination_score,
            constraint_violation_rate=self._calculate_constraint_violation_rate(),
            timestamp=datetime.now()
        )
        
        # Track scenario performance
        self.scenario_performance[scenario].append(total_reward)
        
        return metrics
    
    def _process_episode_experiences(self, experiences: List[Dict[str, Any]]):
        """Process episode experiences and add to buffer"""
        # Group experiences by sequence
        sequences = self._group_experiences_by_sequence(experiences)
        
        for sequence in sequences:
            # Apply reward shaping
            shaped_sequence = self.reward_shaper.shape_sequence_rewards(sequence)
            
            # Add to experience buffer
            self.experience_buffer.add_sequence(shaped_sequence)
    
    def _group_experiences_by_sequence(self, experiences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group experiences by sequential execution"""
        sequences = []
        current_sequence = []
        
        for exp in experiences:
            current_sequence.append(exp)
            
            # Check if sequence is complete (all 4 agents have acted)
            if len(current_sequence) == 4:
                sequences.append(current_sequence)
                current_sequence = []
        
        # Add incomplete sequence if exists
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    def _update_agents(self):
        """Update all agents using collected experiences"""
        try:
            # Sample batch from experience buffer
            batch = self.experience_buffer.sample_batch()
            
            if batch is None:
                return
            
            # Update using MAPPO
            update_metrics = self.mappo_trainer.update(batch)
            
            # Log update metrics
            logger.debug("Agent update completed", 
                        metrics=update_metrics,
                        global_step=self.global_step)
            
            self.global_step += 1
            
        except Exception as e:
            logger.error("Agent update failed", error=str(e), exc_info=True)
    
    def _run_validation(self) -> TrainingMetrics:
        """Run validation episodes"""
        validation_rewards = []
        validation_metrics = []
        
        for val_episode in range(self.config.validation_episodes):
            # Run validation episode (no training)
            with torch.no_grad():
                metrics = self._run_validation_episode()
                validation_rewards.append(metrics.total_reward)
                validation_metrics.append(metrics)
        
        # Calculate average validation metrics
        avg_metrics = self._average_metrics(validation_metrics)
        
        logger.info("Validation completed", 
                   avg_reward=avg_metrics.total_reward,
                   var_performance_met=avg_metrics.var_performance_met,
                   episode=self.current_episode)
        
        return avg_metrics
    
    def _run_validation_episode(self) -> TrainingMetrics:
        """Run a single validation episode"""
        # Use normal scenario for validation
        self.environment.reset(options={'scenario': 'normal'})
        
        episode_rewards = {agent: 0.0 for agent in self.agents}
        var_calculation_times = []
        emergency_activations = 0
        correlation_regimes = defaultdict(int)
        
        step = 0
        while step < self.config.max_steps_per_episode:
            current_agent = self.environment.agent_selection
            
            if self.environment.terminations[current_agent] or self.environment.truncations[current_agent]:
                break
            
            observation = self.environment.observe(current_agent)
            
            # Use agent policy for action (no exploration)
            action = self.agents[current_agent].policy.predict(observation, deterministic=True)
            
            self.environment.step(action)
            
            reward = self.environment.rewards[current_agent]
            info = self.environment.infos[current_agent]
            
            episode_rewards[current_agent] += reward
            
            if 'var_calculation_time_ms' in info:
                var_calculation_times.append(info['var_calculation_time_ms'])
            
            if info.get('emergency_active', False):
                emergency_activations += 1
            
            correlation_regime = info.get('correlation_regime', 'NORMAL')
            correlation_regimes[correlation_regime] += 1
            
            step += 1
        
        # Create validation metrics
        return TrainingMetrics(
            episode=self.current_episode,
            total_reward=sum(episode_rewards.values()),
            individual_rewards=episode_rewards,
            var_performance_met=np.mean(var_calculation_times) < self.config.max_var_calculation_time_ms if var_calculation_times else True,
            avg_var_calculation_time_ms=np.mean(var_calculation_times) if var_calculation_times else 0,
            emergency_activations=emergency_activations,
            correlation_regime_distribution=dict(correlation_regimes),
            risk_superposition_quality=self._calculate_risk_superposition_quality(),
            sequence_coordination_score=0.8,  # Default for validation
            constraint_violation_rate=self._calculate_constraint_violation_rate(),
            timestamp=datetime.now()
        )
    
    def _calculate_sequence_coordination_score(self, experiences: List[Dict[str, Any]]) -> float:
        """Calculate how well agents coordinate in sequence"""
        # Analyze sequential coordination patterns
        coordination_scores = []
        
        sequences = self._group_experiences_by_sequence(experiences)
        for sequence in sequences:
            # Check for smooth transitions between agents
            transition_scores = []
            
            for i in range(len(sequence) - 1):
                current_info = sequence[i]['info']
                next_info = sequence[i + 1]['info']
                
                # Check if context is preserved
                if 'sequential_context' in current_info and 'sequential_context' in next_info:
                    # Simple heuristic: consistent emergency state
                    current_emergency = current_info.get('emergency_active', False)
                    next_emergency = next_info.get('emergency_active', False)
                    
                    if current_emergency == next_emergency:
                        transition_scores.append(1.0)
                    else:
                        transition_scores.append(0.5)
                else:
                    transition_scores.append(0.0)
            
            if transition_scores:
                coordination_scores.append(np.mean(transition_scores))
        
        return np.mean(coordination_scores) if coordination_scores else 0.0
    
    def _calculate_risk_superposition_quality(self) -> float:
        """Calculate quality of risk superposition outputs"""
        # Get latest risk superposition from environment
        superposition = self.environment.get_risk_superposition()
        
        if not superposition:
            return 0.0
        
        quality_scores = []
        
        # Check completeness
        if superposition.position_allocations:
            quality_scores.append(0.25)
        if superposition.stop_loss_orders:
            quality_scores.append(0.25)
        if superposition.target_profit_orders:
            quality_scores.append(0.25)
        if superposition.risk_limits:
            quality_scores.append(0.25)
        
        return sum(quality_scores)
    
    def _calculate_constraint_violation_rate(self) -> float:
        """Calculate constraint violation rate across agents"""
        total_violations = 0
        total_actions = 0
        
        for agent in self.agents.values():
            if hasattr(agent, 'constraint_violations'):
                total_violations += len(agent.constraint_violations)
            if hasattr(agent, 'decision_times'):
                total_actions += len(agent.decision_times)
        
        return total_violations / max(total_actions, 1)
    
    def _evaluate_performance(self):
        """Evaluate performance constraints"""
        # Check VaR calculation performance
        avg_var_times = []
        for metrics in self.training_metrics[-self.config.performance_evaluation_frequency:]:
            avg_var_times.append(metrics.avg_var_calculation_time_ms)
        
        if avg_var_times:
            avg_var_time = np.mean(avg_var_times)
            if avg_var_time > self.config.max_var_calculation_time_ms:
                violation = {
                    'type': 'var_performance',
                    'episode': self.current_episode,
                    'avg_time_ms': avg_var_time,
                    'threshold_ms': self.config.max_var_calculation_time_ms,
                    'timestamp': datetime.now()
                }
                self.performance_violations.append(violation)
                logger.warning("VaR performance violation", **violation)
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered"""
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _training_time_exceeded(self) -> bool:
        """Check if training time limit is exceeded"""
        elapsed_hours = (datetime.now() - self.training_start_time).total_seconds() / 3600
        return elapsed_hours > self.config.max_training_time_hours
    
    def _average_metrics(self, metrics_list: List[TrainingMetrics]) -> TrainingMetrics:
        """Calculate average metrics across episodes"""
        if not metrics_list:
            return TrainingMetrics(0, 0, {}, False, 0, 0, {}, 0, 0, 0, datetime.now())
        
        avg_total_reward = np.mean([m.total_reward for m in metrics_list])
        avg_individual_rewards = {}
        
        # Average individual rewards
        for agent in self.agents:
            agent_rewards = [m.individual_rewards.get(agent, 0) for m in metrics_list]
            avg_individual_rewards[agent] = np.mean(agent_rewards)
        
        avg_var_performance = np.mean([m.var_performance_met for m in metrics_list])
        avg_var_time = np.mean([m.avg_var_calculation_time_ms for m in metrics_list])
        avg_emergency_activations = np.mean([m.emergency_activations for m in metrics_list])
        avg_superposition_quality = np.mean([m.risk_superposition_quality for m in metrics_list])
        avg_coordination_score = np.mean([m.sequence_coordination_score for m in metrics_list])
        avg_constraint_violation_rate = np.mean([m.constraint_violation_rate for m in metrics_list])
        
        return TrainingMetrics(
            episode=self.current_episode,
            total_reward=avg_total_reward,
            individual_rewards=avg_individual_rewards,
            var_performance_met=avg_var_performance > 0.5,
            avg_var_calculation_time_ms=avg_var_time,
            emergency_activations=int(avg_emergency_activations),
            correlation_regime_distribution={},
            risk_superposition_quality=avg_superposition_quality,
            sequence_coordination_score=avg_coordination_score,
            constraint_violation_rate=avg_constraint_violation_rate,
            timestamp=datetime.now()
        )
    
    def _run_final_evaluation(self) -> Dict[str, Any]:
        """Run final evaluation with all risk scenarios"""
        final_results = {}
        
        # Test each risk scenario
        for scenario in self.risk_scenarios:
            scenario_results = []
            
            for _ in range(10):  # 10 episodes per scenario
                self.environment.reset(options={'scenario': scenario})
                
                episode_reward = 0
                var_times = []
                emergency_count = 0
                
                step = 0
                while step < self.config.max_steps_per_episode:
                    current_agent = self.environment.agent_selection
                    
                    if self.environment.terminations[current_agent] or self.environment.truncations[current_agent]:
                        break
                    
                    observation = self.environment.observe(current_agent)
                    action = self.agents[current_agent].policy.predict(observation, deterministic=True)
                    
                    self.environment.step(action)
                    
                    reward = self.environment.rewards[current_agent]
                    info = self.environment.infos[current_agent]
                    
                    episode_reward += reward
                    
                    if 'var_calculation_time_ms' in info:
                        var_times.append(info['var_calculation_time_ms'])
                    
                    if info.get('emergency_active', False):
                        emergency_count += 1
                    
                    step += 1
                
                scenario_results.append({
                    'total_reward': episode_reward,
                    'avg_var_time_ms': np.mean(var_times) if var_times else 0,
                    'emergency_activations': emergency_count,
                    'var_performance_met': np.mean(var_times) < self.config.max_var_calculation_time_ms if var_times else True
                })
            
            final_results[scenario] = {
                'avg_reward': np.mean([r['total_reward'] for r in scenario_results]),
                'avg_var_time_ms': np.mean([r['avg_var_time_ms'] for r in scenario_results]),
                'avg_emergency_activations': np.mean([r['emergency_activations'] for r in scenario_results]),
                'var_performance_rate': np.mean([r['var_performance_met'] for r in scenario_results])
            }
        
        return final_results
    
    def _compile_results(self, final_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Compile complete training results"""
        return {
            'training_config': self.config.__dict__,
            'training_metrics': [m.__dict__ for m in self.training_metrics],
            'validation_metrics': [m.__dict__ for m in self.validation_metrics],
            'performance_violations': self.performance_violations,
            'scenario_performance': dict(self.scenario_performance),
            'final_evaluation': final_evaluation,
            'best_validation_score': self.best_validation_score,
            'total_training_time_hours': (datetime.now() - self.training_start_time).total_seconds() / 3600,
            'agents_performance': {
                agent_id: agent.get_sequential_performance_metrics()
                for agent_id, agent in self.agents.items()
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_sequential_training_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Training results saved", filename=filename)
    
    def _save_best_model(self):
        """Save best model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for agent_id, agent in self.agents.items():
            model_path = f"best_risk_sequential_{agent_id}_model_{timestamp}.pt"
            torch.save(agent.policy.state_dict(), model_path)
        
        logger.info("Best model saved", episode=self.current_episode)
    
    def _log_progress(self, metrics: TrainingMetrics):
        """Log training progress"""
        logger.info("Training progress",
                   episode=metrics.episode,
                   total_reward=f"{metrics.total_reward:.2f}",
                   var_performance_met=metrics.var_performance_met,
                   avg_var_time_ms=f"{metrics.avg_var_calculation_time_ms:.2f}",
                   emergency_activations=metrics.emergency_activations,
                   coordination_score=f"{metrics.sequence_coordination_score:.2f}",
                   superposition_quality=f"{metrics.risk_superposition_quality:.2f}")
    
    def _cleanup(self):
        """Clean up resources"""
        self.environment.close()
        logger.info("Training cleanup completed")


class PerformanceMonitor:
    """Monitor performance constraints during training"""
    
    def __init__(self, max_var_time_ms: float, event_bus: EventBus):
        self.max_var_time_ms = max_var_time_ms
        self.event_bus = event_bus
        self.var_time_violations = []
        
        # Subscribe to performance events
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._monitor_var_performance)
    
    def _monitor_var_performance(self, event: Event):
        """Monitor VaR calculation performance"""
        if hasattr(event.payload, 'performance_ms'):
            var_time = event.payload.performance_ms
            if var_time > self.max_var_time_ms:
                violation = {
                    'timestamp': datetime.now(),
                    'var_time_ms': var_time,
                    'threshold_ms': self.max_var_time_ms,
                    'violation_ratio': var_time / self.max_var_time_ms
                }
                self.var_time_violations.append(violation)
                
                logger.warning("VaR performance violation during training", **violation)
    
    def get_violation_rate(self) -> float:
        """Get current violation rate"""
        if not self.var_time_violations:
            return 0.0
        
        # Calculate violation rate over last 100 measurements
        recent_violations = self.var_time_violations[-100:]
        return len(recent_violations) / 100.0


def create_risk_sequential_trainer(config: Optional[TrainingConfig] = None) -> RiskSequentialTrainer:
    """
    Factory function to create risk sequential trainer
    
    Args:
        config: Optional training configuration
        
    Returns:
        Configured RiskSequentialTrainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    return RiskSequentialTrainer(config)


def main():
    """Main training function"""
    # Create training configuration
    config = TrainingConfig(
        num_episodes=1000,
        max_steps_per_episode=1000,
        batch_size=64,
        learning_rate=3e-4,
        max_var_calculation_time_ms=5.0,
        correlation_shock_training=True,
        black_swan_simulation=True
    )
    
    # Create and run trainer
    trainer = create_risk_sequential_trainer(config)
    results = trainer.train()
    
    print("Training completed successfully!")
    print(f"Best validation score: {results['best_validation_score']:.2f}")
    print(f"Training time: {results['total_training_time_hours']:.2f} hours")
    
    return results


if __name__ == "__main__":
    main()