"""
Unified PettingZoo Training Coordinator

This module provides a unified interface for training MARL systems across
different PettingZoo environments (strategic, tactical, execution, risk).
It integrates all the training components and provides a seamless training
experience with advanced features.

Key Features:
- Unified training interface for all MARL environments
- Integration with existing training optimizations
- Support for mixed training scenarios
- Advanced hyperparameter optimization
- Comprehensive logging and monitoring
- Production-ready deployment pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import yaml
import pickle
from collections import defaultdict, deque

# PettingZoo imports
from pettingzoo import AECEnv

# Internal imports
from .pettingzoo_mappo_trainer import PettingZooMAPPOTrainer, TrainingConfig
from .pettingzoo_environment_manager import (
    EnvironmentFactory, EnvironmentConfig, EnvironmentType,
    MultiEnvironmentManager, EnvironmentValidator
)
from .pettingzoo_training_loops import (
    PettingZooTrainingLoop, TrainingLoopConfig, 
    CurriculumManager, ExperienceCollector
)
from .pettingzoo_reward_system import (
    PettingZooRewardSystem, RewardConfig, create_reward_config
)

# Import existing optimizations
from ..unified_training_system import UnifiedTrainingSystem, UnifiedTrainingConfig
from ..hyperparameter_optimizer import HyperparameterOptimizer
from ..advanced_checkpoint_manager import AdvancedCheckpointManager
from ..performance_analysis_framework import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training modes for different scenarios"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EXECUTION = "execution"
    RISK = "risk"
    UNIFIED = "unified"
    MIXED = "mixed"


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    PARETO_OPTIMIZATION = "pareto_optimization"
    HIERARCHICAL = "hierarchical"


@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for PettingZoo training"""
    # Training mode
    training_mode: TrainingMode = TrainingMode.TACTICAL
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.SINGLE_OBJECTIVE
    
    # Environment configuration
    environment_configs: Dict[str, EnvironmentConfig] = field(default_factory=dict)
    mixed_training_weights: Dict[str, float] = field(default_factory=dict)
    
    # Training parameters
    total_episodes: int = 10000
    max_episode_steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 3e-4
    
    # Advanced features
    enable_curriculum_learning: bool = True
    enable_transfer_learning: bool = False
    enable_meta_learning: bool = False
    enable_continual_learning: bool = False
    
    # Optimization features
    enable_hyperparameter_optimization: bool = True
    enable_neural_architecture_search: bool = False
    enable_automated_ml: bool = False
    
    # Parallel training
    parallel_environments: int = 4
    distributed_training: bool = False
    async_training: bool = False
    
    # Logging and monitoring
    experiment_name: str = "unified_pettingzoo_training"
    log_directory: str = "logs/unified_training"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    
    # Checkpointing
    checkpoint_frequency: int = 1000
    keep_checkpoints: int = 5
    enable_auto_save: bool = True
    
    # Performance optimization
    mixed_precision: bool = True
    gradient_clipping: bool = True
    memory_optimization: bool = True
    
    # Evaluation
    evaluation_frequency: int = 500
    evaluation_episodes: int = 10
    
    # Early stopping
    early_stopping_patience: int = 100
    target_performance: float = 0.9
    
    # Resource management
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 8
    gpu_memory_fraction: float = 0.8


class UnifiedPettingZooTrainer:
    """
    Unified trainer for PettingZoo environments with advanced features
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize components
        self.environment_factory = EnvironmentFactory()
        self.environment_validator = EnvironmentValidator()
        self.reward_systems = {}
        self.trainers = {}
        self.training_loops = {}
        
        # Setup logging
        self.log_dir = Path(config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Initialize optimizers
        self.hyperparameter_optimizer = None
        self.checkpoint_manager = None
        self.performance_analyzer = None
        
        # Training state
        self.training_history = []
        self.best_performance = {}
        self.training_metrics = defaultdict(list)
        
        # Initialize systems
        self._initialize_systems()
        
        logger.info(f"Unified PettingZoo Trainer initialized for {config.training_mode.value} mode")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Set memory fraction if specified
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_systems(self):
        """Initialize all training systems"""
        # Initialize reward systems
        self._initialize_reward_systems()
        
        # Initialize environments
        self._initialize_environments()
        
        # Initialize trainers
        self._initialize_trainers()
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize monitoring
        self._initialize_monitoring()
    
    def _initialize_reward_systems(self):
        """Initialize reward systems for each environment type"""
        for env_type in [TrainingMode.STRATEGIC, TrainingMode.TACTICAL, TrainingMode.EXECUTION, TrainingMode.RISK]:
            reward_config = create_reward_config(
                enable_multi_objective=(self.config.optimization_strategy == OptimizationStrategy.MULTI_OBJECTIVE),
                adaptive_weights=True,
                track_reward_history=True
            )
            self.reward_systems[env_type.value] = PettingZooRewardSystem(reward_config)
    
    def _initialize_environments(self):
        """Initialize environments based on training mode"""
        if self.config.training_mode == TrainingMode.MIXED:
            # Initialize all environment types
            for env_type in [EnvironmentType.STRATEGIC, EnvironmentType.TACTICAL, 
                           EnvironmentType.EXECUTION, EnvironmentType.RISK]:
                env_config = self.config.environment_configs.get(
                    env_type.value, 
                    EnvironmentConfig(env_type=env_type)
                )
                
                # Validate environment
                test_env = self.environment_factory.create_environment(env_config)
                validation_result = self.environment_validator.validate_environment(test_env)
                
                if not validation_result['is_valid']:
                    raise ValueError(f"Environment {env_type.value} validation failed: {validation_result['errors']}")
                
                test_env.close()
                logger.info(f"Environment {env_type.value} validated successfully")
        else:
            # Initialize single environment type
            env_type = EnvironmentType(self.config.training_mode.value)
            env_config = self.config.environment_configs.get(
                env_type.value,
                EnvironmentConfig(env_type=env_type)
            )
            
            # Validate environment
            test_env = self.environment_factory.create_environment(env_config)
            validation_result = self.environment_validator.validate_environment(test_env)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Environment validation failed: {validation_result['errors']}")
            
            test_env.close()
            logger.info(f"Environment {env_type.value} validated successfully")
    
    def _initialize_trainers(self):
        """Initialize MAPPO trainers"""
        if self.config.training_mode == TrainingMode.MIXED:
            # Initialize trainers for each environment type
            for env_type in [EnvironmentType.STRATEGIC, EnvironmentType.TACTICAL, 
                           EnvironmentType.EXECUTION, EnvironmentType.RISK]:
                trainer_config = TrainingConfig(
                    env_factory=lambda et=env_type: self.environment_factory.create_environment(
                        self.config.environment_configs.get(et.value, EnvironmentConfig(env_type=et))
                    ),
                    num_episodes=self.config.total_episodes,
                    max_episode_steps=self.config.max_episode_steps,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    device=str(self.device)
                )
                
                self.trainers[env_type.value] = PettingZooMAPPOTrainer(trainer_config)
        else:
            # Initialize single trainer
            env_type = EnvironmentType(self.config.training_mode.value)
            trainer_config = TrainingConfig(
                env_factory=lambda: self.environment_factory.create_environment(
                    self.config.environment_configs.get(env_type.value, EnvironmentConfig(env_type=env_type))
                ),
                num_episodes=self.config.total_episodes,
                max_episode_steps=self.config.max_episode_steps,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                device=str(self.device)
            )
            
            self.trainers[env_type.value] = PettingZooMAPPOTrainer(trainer_config)
    
    def _initialize_optimizers(self):
        """Initialize advanced optimizers"""
        # Hyperparameter optimizer
        if self.config.enable_hyperparameter_optimization:
            from ..hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterConfig
            
            hyperopt_config = HyperparameterConfig(
                max_trials=50,
                results_dir=str(self.log_dir / "hyperopt"),
                optimization_strategy=self.config.optimization_strategy.value
            )
            
            self.hyperparameter_optimizer = HyperparameterOptimizer(hyperopt_config)
        
        # Checkpoint manager
        from ..advanced_checkpoint_manager import AdvancedCheckpointManager, CheckpointConfig
        
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(self.log_dir / "checkpoints"),
            keep_checkpoints=self.config.keep_checkpoints,
            save_frequency=self.config.checkpoint_frequency
        )
        
        self.checkpoint_manager = AdvancedCheckpointManager(checkpoint_config)
    
    def _initialize_monitoring(self):
        """Initialize monitoring systems"""
        # Performance analyzer
        from ..performance_analysis_framework import PerformanceAnalyzer
        
        self.performance_analyzer = PerformanceAnalyzer(
            config={'log_dir': str(self.log_dir)}
        )
        
        # TensorBoard
        if self.config.enable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(
                log_dir=str(self.log_dir / "tensorboard")
            )
        
        # Weights & Biases
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.init(
                    project="unified-pettingzoo-training",
                    name=self.config.experiment_name,
                    config=self.config
                )
                self.wandb_enabled = True
            except ImportError:
                logger.warning("Weights & Biases not available")
                self.wandb_enabled = False
    
    def train(self) -> Dict[str, Any]:
        """Main training method"""
        logger.info(f"Starting unified PettingZoo training: {self.config.training_mode.value}")
        
        training_start_time = time.time()
        
        try:
            # Hyperparameter optimization phase
            if self.config.enable_hyperparameter_optimization:
                logger.info("Starting hyperparameter optimization...")
                best_params = self._optimize_hyperparameters()
                self._update_configs_with_best_params(best_params)
            
            # Main training phase
            logger.info("Starting main training phase...")
            if self.config.training_mode == TrainingMode.MIXED:
                training_results = self._run_mixed_training()
            else:
                training_results = self._run_single_mode_training()
            
            # Evaluation phase
            logger.info("Starting evaluation phase...")
            evaluation_results = self._run_comprehensive_evaluation()
            
            # Compile final results
            final_results = self._compile_final_results(
                training_results, evaluation_results, time.time() - training_start_time
            )
            
            # Save results
            self._save_final_results(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _optimize_hyperparameters(self) -> Dict[str, Any]:
        """Optimize hyperparameters using advanced optimization"""
        if self.hyperparameter_optimizer is None:
            return {}
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization"""
            # Update configurations with trial parameters
            self._update_configs_with_trial_params(params)
            
            # Run short training session
            if self.config.training_mode == TrainingMode.MIXED:
                results = self._run_mixed_training_trial()
            else:
                results = self._run_single_mode_training_trial()
            
            # Return performance metric
            return results.get('final_performance', 0.0)
        
        # Run optimization
        optimization_results = self.hyperparameter_optimizer.optimize(objective_function)
        
        logger.info(f"Hyperparameter optimization completed: {optimization_results}")
        return optimization_results
    
    def _run_single_mode_training(self) -> Dict[str, Any]:
        """Run training for single environment type"""
        env_type = self.config.training_mode.value
        trainer = self.trainers[env_type]
        
        # Create training loop
        loop_config = TrainingLoopConfig(
            max_episodes=self.config.total_episodes,
            max_steps_per_episode=self.config.max_episode_steps,
            batch_size=self.config.batch_size,
            parallel_envs=self.config.parallel_environments,
            enable_curriculum=self.config.enable_curriculum_learning,
            log_frequency=100,
            eval_frequency=self.config.evaluation_frequency,
            checkpoint_frequency=self.config.checkpoint_frequency
        )
        
        training_loop = PettingZooTrainingLoop(trainer, loop_config)
        
        # Run training
        results = training_loop.run_training()
        
        # Update metrics
        self.training_metrics[env_type] = results['metrics']
        self.best_performance[env_type] = results['best_reward']
        
        return results
    
    def _run_mixed_training(self) -> Dict[str, Any]:
        """Run mixed training across multiple environment types"""
        # Initialize training loops for each environment type
        training_loops = {}
        
        for env_type in [EnvironmentType.STRATEGIC, EnvironmentType.TACTICAL, 
                        EnvironmentType.EXECUTION, EnvironmentType.RISK]:
            trainer = self.trainers[env_type.value]
            
            loop_config = TrainingLoopConfig(
                max_episodes=self.config.total_episodes // 4,  # Distribute episodes
                max_steps_per_episode=self.config.max_episode_steps,
                batch_size=self.config.batch_size,
                parallel_envs=max(1, self.config.parallel_environments // 4),
                enable_curriculum=self.config.enable_curriculum_learning,
                log_frequency=100,
                eval_frequency=self.config.evaluation_frequency,
                checkpoint_frequency=self.config.checkpoint_frequency
            )
            
            training_loops[env_type.value] = PettingZooTrainingLoop(trainer, loop_config)
        
        # Run training loops in parallel or sequentially
        if self.config.async_training:
            results = self._run_async_mixed_training(training_loops)
        else:
            results = self._run_sequential_mixed_training(training_loops)
        
        return results
    
    def _run_async_mixed_training(self, training_loops: Dict[str, PettingZooTrainingLoop]) -> Dict[str, Any]:
        """Run mixed training asynchronously"""
        with ThreadPoolExecutor(max_workers=len(training_loops)) as executor:
            futures = {
                executor.submit(loop.run_training): env_type
                for env_type, loop in training_loops.items()
            }
            
            results = {}
            for future in as_completed(futures):
                env_type = futures[future]
                try:
                    result = future.result()
                    results[env_type] = result
                    self.training_metrics[env_type] = result['metrics']
                    self.best_performance[env_type] = result['best_reward']
                except Exception as e:
                    logger.error(f"Training failed for {env_type}: {e}")
                    results[env_type] = {'error': str(e)}
        
        return results
    
    def _run_sequential_mixed_training(self, training_loops: Dict[str, PettingZooTrainingLoop]) -> Dict[str, Any]:
        """Run mixed training sequentially"""
        results = {}
        
        for env_type, loop in training_loops.items():
            logger.info(f"Starting training for {env_type}")
            
            try:
                result = loop.run_training()
                results[env_type] = result
                self.training_metrics[env_type] = result['metrics']
                self.best_performance[env_type] = result['best_reward']
                
                logger.info(f"Training completed for {env_type}: {result['best_reward']:.3f}")
                
            except Exception as e:
                logger.error(f"Training failed for {env_type}: {e}")
                results[env_type] = {'error': str(e)}
        
        return results
    
    def _run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of trained models"""
        evaluation_results = {}
        
        if self.config.training_mode == TrainingMode.MIXED:
            # Evaluate each environment type
            for env_type in [EnvironmentType.STRATEGIC, EnvironmentType.TACTICAL, 
                           EnvironmentType.EXECUTION, EnvironmentType.RISK]:
                evaluation_results[env_type.value] = self._evaluate_environment_type(env_type.value)
        else:
            # Evaluate single environment type
            env_type = self.config.training_mode.value
            evaluation_results[env_type] = self._evaluate_environment_type(env_type)
        
        return evaluation_results
    
    def _evaluate_environment_type(self, env_type: str) -> Dict[str, Any]:
        """Evaluate specific environment type"""
        trainer = self.trainers[env_type]
        
        # Create evaluation environment
        env_config = self.config.environment_configs.get(
            env_type, 
            EnvironmentConfig(env_type=EnvironmentType(env_type))
        )
        eval_env = self.environment_factory.create_environment(env_config)
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.evaluation_episodes):
            eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while eval_env.agents:
                current_agent = eval_env.agent_selection
                observation = eval_env.observe(current_agent)
                
                # Get action from trained policy
                action, _, _ = trainer.get_action_and_value(observation, current_agent)
                
                eval_env.step(action)
                
                reward = eval_env.rewards.get(current_agent, 0.0)
                episode_reward += reward
                episode_length += 1
                
                if eval_env.dones.get(current_agent, False) or episode_length >= self.config.max_episode_steps:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        eval_env.close()
        
        # Calculate statistics
        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        return evaluation_results
    
    def _update_configs_with_best_params(self, best_params: Dict[str, Any]):
        """Update configurations with best hyperparameters"""
        if 'best_parameters' not in best_params:
            return
        
        params = best_params['best_parameters']
        
        # Update training config
        if 'learning_rate' in params:
            self.config.learning_rate = params['learning_rate']
        
        if 'batch_size' in params:
            self.config.batch_size = params['batch_size']
        
        # Update trainer configs
        for trainer in self.trainers.values():
            if 'learning_rate' in params:
                trainer.config.learning_rate = params['learning_rate']
            if 'batch_size' in params:
                trainer.config.batch_size = params['batch_size']
        
        logger.info(f"Updated configurations with optimized parameters: {params}")
    
    def _update_configs_with_trial_params(self, params: Dict[str, Any]):
        """Update configurations with trial parameters"""
        # This would temporarily update configs for hyperparameter optimization
        pass
    
    def _run_mixed_training_trial(self) -> Dict[str, Any]:
        """Run short mixed training trial for hyperparameter optimization"""
        # This would run a shortened version of mixed training
        return {'final_performance': 0.5}
    
    def _run_single_mode_training_trial(self) -> Dict[str, Any]:
        """Run short single mode training trial for hyperparameter optimization"""
        # This would run a shortened version of single mode training
        return {'final_performance': 0.5}
    
    def _compile_final_results(self, training_results: Dict[str, Any], 
                             evaluation_results: Dict[str, Any], 
                             total_time: float) -> Dict[str, Any]:
        """Compile final training results"""
        final_results = {
            'experiment_name': self.config.experiment_name,
            'training_mode': self.config.training_mode.value,
            'total_training_time': total_time,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'best_performance': self.best_performance,
            'training_metrics': dict(self.training_metrics),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.log_dir / f"final_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_file}")
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up unified trainer resources...")
        
        # Close environments
        for trainer in self.trainers.values():
            if hasattr(trainer, '_cleanup'):
                trainer._cleanup()
        
        # Close tensorboard writer
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
        
        # Finish wandb
        if self.config.enable_wandb and hasattr(self, 'wandb_enabled') and self.wandb_enabled:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, checkpoint_name: str = None):
        """Save training checkpoint"""
        if self.checkpoint_manager is None:
            return
        
        checkpoint_data = {
            'config': self.config,
            'trainers': {name: trainer.networks for name, trainer in self.trainers.items()},
            'best_performance': self.best_performance,
            'training_metrics': dict(self.training_metrics)
        }
        
        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            model=checkpoint_data,
            step=sum(len(metrics.get('episode_rewards', [])) for metrics in self.training_metrics.values()),
            metrics=self.best_performance,
            checkpoint_name=checkpoint_name
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if self.checkpoint_manager is None:
            return
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if checkpoint:
            # Restore trainer states
            for name, trainer in self.trainers.items():
                if name in checkpoint['trainers']:
                    for net_name, net_state in checkpoint['trainers'][name].items():
                        if net_name in trainer.networks:
                            trainer.networks[net_name].load_state_dict(net_state)
            
            # Restore metrics
            self.best_performance = checkpoint.get('best_performance', {})
            self.training_metrics = checkpoint.get('training_metrics', {})
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'config': self.config,
            'best_performance': self.best_performance,
            'training_metrics': dict(self.training_metrics),
            'training_mode': self.config.training_mode.value,
            'total_trainers': len(self.trainers)
        }


def create_unified_config(**kwargs) -> UnifiedTrainingConfig:
    """Create unified training configuration"""
    return UnifiedTrainingConfig(**kwargs)


def create_unified_trainer(config: UnifiedTrainingConfig) -> UnifiedPettingZooTrainer:
    """Create unified PettingZoo trainer"""
    return UnifiedPettingZooTrainer(config)


def run_strategic_training(config: UnifiedTrainingConfig = None) -> Dict[str, Any]:
    """Run strategic MARL training"""
    if config is None:
        config = create_unified_config(training_mode=TrainingMode.STRATEGIC)
    
    trainer = create_unified_trainer(config)
    return trainer.train()


def run_tactical_training(config: UnifiedTrainingConfig = None) -> Dict[str, Any]:
    """Run tactical MARL training"""
    if config is None:
        config = create_unified_config(training_mode=TrainingMode.TACTICAL)
    
    trainer = create_unified_trainer(config)
    return trainer.train()


def run_unified_training(config: UnifiedTrainingConfig = None) -> Dict[str, Any]:
    """Run unified training across all environment types"""
    if config is None:
        config = create_unified_config(training_mode=TrainingMode.UNIFIED)
    
    trainer = create_unified_trainer(config)
    return trainer.train()


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_unified_config(
        training_mode=TrainingMode.TACTICAL,
        total_episodes=1000,
        parallel_environments=2,
        enable_curriculum_learning=True,
        enable_hyperparameter_optimization=True,
        experiment_name="tactical_training_demo"
    )
    
    # Run training
    results = run_tactical_training(config)
    
    print("Training completed!")
    print(f"Best performance: {results['best_performance']}")
    print(f"Training time: {results['total_training_time']:.2f} seconds")