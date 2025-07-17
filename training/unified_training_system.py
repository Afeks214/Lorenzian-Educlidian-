"""
Unified Training System for MARL
Integrates all training optimizations into a cohesive system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all optimization components
from parallel_marl_trainer import ParallelMARLTrainer, ParallelTrainingConfig
from distributed_marl_trainer import DistributedMARLTrainer, DistributedTrainingConfig
from memory_optimized_trainer import MemoryOptimizedTrainer, MemoryOptimizationConfig
from advanced_checkpoint_manager import AdvancedCheckpointManager, CheckpointConfig
from hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterConfig
from advanced_lr_scheduler import AdvancedLRSchedulerManager, SchedulerConfig

logger = logging.getLogger(__name__)

@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for all training optimizations"""
    # Training strategy
    training_strategy: str = "sequential"  # "sequential", "parallel", "distributed"
    
    # Model and data
    model_factory: Optional[Callable] = None
    optimizer_factory: Optional[Callable] = None
    environment_factory: Optional[Callable] = None
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    
    # Agent configuration
    num_agents: int = 4
    agent_types: List[str] = None
    
    # Optimization settings
    enable_parallel_training: bool = True
    enable_distributed_training: bool = False
    enable_memory_optimization: bool = True
    enable_checkpointing: bool = True
    enable_hyperparameter_tuning: bool = True
    enable_advanced_scheduling: bool = True
    
    # Performance settings
    target_performance: float = 0.95
    early_stopping_patience: int = 50
    
    # Resource management
    available_memory_gb: float = 8.0
    max_concurrent_jobs: int = 4
    
    # Directories
    base_dir: str = "unified_training"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # Component configurations
    parallel_config: Optional[ParallelTrainingConfig] = None
    distributed_config: Optional[DistributedTrainingConfig] = None
    memory_config: Optional[MemoryOptimizationConfig] = None
    checkpoint_config: Optional[CheckpointConfig] = None
    hyperopt_config: Optional[HyperparameterConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = ['strategic', 'tactical', 'execution', 'risk']
        
        # Create directories
        for dir_name in [self.base_dir, self.results_dir, self.checkpoints_dir, self.logs_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        # Initialize component configurations if not provided
        if self.parallel_config is None and self.enable_parallel_training:
            self.parallel_config = ParallelTrainingConfig(
                num_agents=self.num_agents,
                agent_types=self.agent_types,
                num_workers=self.max_concurrent_jobs
            )
        
        if self.distributed_config is None and self.enable_distributed_training:
            self.distributed_config = DistributedTrainingConfig(
                num_agents=self.num_agents,
                agent_types=self.agent_types
            )
        
        if self.memory_config is None and self.enable_memory_optimization:
            self.memory_config = MemoryOptimizationConfig(
                max_memory_usage_gb=self.available_memory_gb
            )
        
        if self.checkpoint_config is None and self.enable_checkpointing:
            self.checkpoint_config = CheckpointConfig(
                checkpoint_dir=self.checkpoints_dir
            )
        
        if self.hyperopt_config is None and self.enable_hyperparameter_tuning:
            self.hyperopt_config = HyperparameterConfig(
                max_trials=20,
                results_dir=f"{self.base_dir}/hyperopt"
            )
        
        if self.scheduler_config is None and self.enable_advanced_scheduling:
            self.scheduler_config = SchedulerConfig(
                initial_lr=self.learning_rate
            )


class TrainingMetrics:
    """Unified training metrics tracker"""
    
    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'episode_lengths': [],
            'training_time': [],
            'memory_usage': [],
            'learning_rates': [],
            'convergence_metrics': {},
            'performance_metrics': {},
            'system_metrics': {}
        }
        
        self.start_time = time.time()
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.best_episode = 0
        
    def update(self, episode_data: Dict[str, Any]):
        """Update metrics with episode data"""
        self.episode_count += 1
        
        # Core metrics
        if 'episode_reward' in episode_data:
            reward = episode_data['episode_reward']
            self.metrics['episode_rewards'].append(reward)
            
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_episode = self.episode_count
        
        if 'episode_loss' in episode_data:
            self.metrics['episode_losses'].append(episode_data['episode_loss'])
        
        if 'episode_length' in episode_data:
            self.metrics['episode_lengths'].append(episode_data['episode_length'])
        
        if 'training_time' in episode_data:
            self.metrics['training_time'].append(episode_data['training_time'])
        
        if 'memory_usage' in episode_data:
            self.metrics['memory_usage'].append(episode_data['memory_usage'])
        
        if 'learning_rate' in episode_data:
            self.metrics['learning_rates'].append(episode_data['learning_rate'])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_episodes': self.episode_count,
            'total_time': total_time,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'episodes_per_second': self.episode_count / total_time if total_time > 0 else 0
        }
        
        # Add statistics for each metric
        for metric_name, values in self.metrics.items():
            if values and isinstance(values[0], (int, float)):
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_min'] = np.min(values)
                summary[f'{metric_name}_max'] = np.max(values)
        
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'summary': self.get_summary()
            }, f, indent=2, default=str)


class UnifiedTrainingSystem:
    """Unified training system that orchestrates all optimizations"""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        
        # Initialize components
        self.components = {}
        self._initialize_components()
        
        # Training state
        self.training_state = {
            'current_episode': 0,
            'total_episodes': config.num_episodes,
            'training_active': False,
            'best_model_path': None,
            'convergence_achieved': False
        }
        
        # Results storage
        self.results = {
            'training_config': asdict(config),
            'training_metrics': None,
            'optimization_results': {},
            'component_performance': {}
        }
        
        logger.info("Unified training system initialized")
    
    def _initialize_components(self):
        """Initialize all training components"""
        
        # Checkpoint manager
        if self.config.enable_checkpointing:
            self.components['checkpoint_manager'] = AdvancedCheckpointManager(
                self.config.checkpoint_config
            )
        
        # Hyperparameter optimizer
        if self.config.enable_hyperparameter_tuning:
            self.components['hyperopt'] = HyperparameterOptimizer(
                self.config.hyperopt_config
            )
        
        # Memory-optimized trainer
        if self.config.enable_memory_optimization:
            self.components['memory_trainer'] = None  # Will be initialized with model
        
        # Parallel trainer
        if self.config.enable_parallel_training:
            self.components['parallel_trainer'] = None  # Will be initialized with factories
        
        # Distributed trainer
        if self.config.enable_distributed_training:
            self.components['distributed_trainer'] = None  # Will be initialized with factories
        
        # Learning rate scheduler
        if self.config.enable_advanced_scheduling:
            self.components['lr_scheduler'] = None  # Will be initialized with optimizer
        
        logger.info(f"Initialized {len(self.components)} training components")
    
    def train(self) -> Dict[str, Any]:
        """Run unified training process"""
        logger.info("Starting unified training process")
        
        training_start_time = time.time()
        self.training_state['training_active'] = True
        
        try:
            # Phase 1: Hyperparameter optimization (if enabled)
            if self.config.enable_hyperparameter_tuning:
                logger.info("Phase 1: Hyperparameter optimization")
                best_params = self._optimize_hyperparameters()
                self.results['optimization_results']['hyperparameters'] = best_params
                
                # Update configuration with best parameters
                self._update_config_with_best_params(best_params)
            
            # Phase 2: Main training
            logger.info("Phase 2: Main training")
            training_results = self._run_main_training()
            self.results['optimization_results']['training'] = training_results
            
            # Phase 3: Validation and analysis
            logger.info("Phase 3: Validation and analysis")
            validation_results = self._validate_training_results()
            self.results['optimization_results']['validation'] = validation_results
            
            # Compile final results
            self.results['training_metrics'] = self.metrics.get_summary()
            self.results['total_training_time'] = time.time() - training_start_time
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.training_state['training_active'] = False
            self._cleanup()
    
    def _optimize_hyperparameters(self) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        hyperopt = self.components['hyperopt']
        
        def training_objective(params: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization"""
            # Create model with these parameters
            model = self.config.model_factory()
            optimizer = self.config.optimizer_factory(model.parameters())
            
            # Quick training run
            environment = self.config.environment_factory()
            
            # Simple training loop
            total_reward = 0
            for episode in range(min(50, self.config.num_episodes // 10)):  # Quick evaluation
                state = environment.reset()
                episode_reward = 0
                
                for step in range(self.config.max_steps_per_episode):
                    # Simple forward pass
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action_logits = model(state_tensor)
                        action = torch.argmax(action_logits, dim=-1).item()
                    
                    next_state, reward, done, _ = environment.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                    
                    state = next_state
                
                total_reward += episode_reward
            
            return total_reward / 50  # Average reward
        
        # Run optimization
        results = hyperopt.optimize(training_objective)
        
        logger.info(f"Hyperparameter optimization completed: {results['best_parameters']}")
        return results
    
    def _update_config_with_best_params(self, best_params: Dict[str, Any]):
        """Update configuration with best hyperparameters"""
        if 'best_parameters' in best_params:
            params = best_params['best_parameters']
            
            # Update learning rate
            if 'learning_rate' in params:
                self.config.learning_rate = params['learning_rate']
                if self.config.scheduler_config:
                    self.config.scheduler_config.initial_lr = params['learning_rate']
            
            # Update batch size
            if 'batch_size' in params:
                self.config.batch_size = params['batch_size']
            
            # Update other parameters as needed
            logger.info(f"Updated configuration with optimized parameters")
    
    def _run_main_training(self) -> Dict[str, Any]:
        """Run main training process"""
        
        # Select training strategy
        if self.config.training_strategy == "distributed" and self.config.enable_distributed_training:
            return self._run_distributed_training()
        elif self.config.training_strategy == "parallel" and self.config.enable_parallel_training:
            return self._run_parallel_training()
        else:
            return self._run_sequential_training()
    
    def _run_sequential_training(self) -> Dict[str, Any]:
        """Run sequential training"""
        logger.info("Running sequential training")
        
        # Initialize components
        model = self.config.model_factory()
        optimizer = self.config.optimizer_factory(model.parameters())
        environment = self.config.environment_factory()
        
        # Initialize memory-optimized trainer if enabled
        if self.config.enable_memory_optimization:
            from memory_optimized_trainer import MemoryOptimizedTrainer
            trainer = MemoryOptimizedTrainer(
                self.config.memory_config,
                lambda: model,
                lambda params: optimizer,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            trainer = None
        
        # Initialize learning rate scheduler if enabled
        if self.config.enable_advanced_scheduling:
            from advanced_lr_scheduler import create_scheduler_suite
            scheduler_manager = create_scheduler_suite(optimizer, self.config.scheduler_config)
        else:
            scheduler_manager = None
        
        # Training loop
        for episode in range(self.config.num_episodes):
            episode_start_time = time.time()
            
            # Train episode
            if trainer:
                episode_result = trainer.train_episode(environment)
            else:
                episode_result = self._train_episode_basic(model, optimizer, environment)
            
            # Update metrics
            episode_result['training_time'] = time.time() - episode_start_time
            if scheduler_manager:
                episode_result['learning_rate'] = scheduler_manager.get_lr_history()[-1] if scheduler_manager.get_lr_history() else self.config.learning_rate
                
                # Step scheduler
                scheduler_manager.step({
                    'loss': episode_result.get('episode_loss', 0),
                    'reward': episode_result.get('episode_reward', 0)
                })
            
            self.metrics.update(episode_result)
            
            # Checkpointing
            if self.config.enable_checkpointing and episode % 100 == 0:
                checkpoint_manager = self.components['checkpoint_manager']
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    episode=episode,
                    step=episode * self.config.max_steps_per_episode,
                    metrics=episode_result,
                    is_best=(episode_result.get('episode_reward', 0) == self.metrics.best_reward)
                )
                
                if episode_result.get('episode_reward', 0) == self.metrics.best_reward:
                    self.training_state['best_model_path'] = checkpoint_id
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Convergence achieved at episode {episode}")
                self.training_state['convergence_achieved'] = True
                break
            
            # Log progress
            if episode % 50 == 0:
                self._log_progress(episode)
        
        # Cleanup
        if trainer:
            trainer.cleanup()
        
        return {
            'training_strategy': 'sequential',
            'episodes_completed': self.metrics.episode_count,
            'convergence_achieved': self.training_state['convergence_achieved'],
            'best_reward': self.metrics.best_reward,
            'best_episode': self.metrics.best_episode
        }
    
    def _run_parallel_training(self) -> Dict[str, Any]:
        """Run parallel training"""
        logger.info("Running parallel training")
        
        # Initialize parallel trainer
        parallel_trainer = ParallelMARLTrainer(
            config=self.config.parallel_config,
            model_factory=self.config.model_factory,
            optimizer_factory=self.config.optimizer_factory,
            environment_factory=self.config.environment_factory
        )
        
        # Run training
        results = parallel_trainer.train_parallel(self.config.num_episodes)
        
        # Update metrics from parallel results
        for worker_result in results['worker_results']:
            for episode_data in worker_result['episodes']:
                self.metrics.update(episode_data)
        
        return results
    
    def _run_distributed_training(self) -> Dict[str, Any]:
        """Run distributed training"""
        logger.info("Running distributed training")
        
        # This would typically be run with multiple processes
        # For now, simulate with single process
        distributed_trainer = DistributedMARLTrainer(
            config=self.config.distributed_config,
            model_factory=self.config.model_factory,
            optimizer_factory=self.config.optimizer_factory,
            environment_factory=self.config.environment_factory,
            rank=0,
            world_size=1
        )
        
        # Run training
        results = distributed_trainer.train_distributed(self.config.num_episodes)
        
        return results
    
    def _train_episode_basic(self, model: nn.Module, optimizer: optim.Optimizer, environment) -> Dict[str, Any]:
        """Basic episode training"""
        episode_reward = 0
        episode_loss = 0
        episode_length = 0
        
        state = environment.reset()
        
        for step in range(self.config.max_steps_per_episode):
            # Forward pass
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits = model(state_tensor)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            
            # Take action
            next_state, reward, done, _ = environment.step(action.item())
            
            # Compute loss (simple policy gradient)
            log_prob = action_dist.log_prob(action)
            loss = -log_prob * reward
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_reward += reward
            episode_loss += loss.item()
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        return {
            'episode_reward': episode_reward,
            'episode_loss': episode_loss / episode_length if episode_length > 0 else 0,
            'episode_length': episode_length
        }
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if self.metrics.episode_count < self.config.early_stopping_patience:
            return False
        
        # Check if recent performance is consistently good
        recent_rewards = self.metrics.metrics['episode_rewards'][-self.config.early_stopping_patience:]
        mean_recent_reward = np.mean(recent_rewards)
        
        return mean_recent_reward >= self.config.target_performance
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        if self.metrics.metrics['episode_rewards']:
            recent_rewards = self.metrics.metrics['episode_rewards'][-10:]
            mean_recent_reward = np.mean(recent_rewards)
            
            logger.info(f"Episode {episode}: "
                       f"Recent Reward: {mean_recent_reward:.3f}, "
                       f"Best Reward: {self.metrics.best_reward:.3f}, "
                       f"Best Episode: {self.metrics.best_episode}")
    
    def _validate_training_results(self) -> Dict[str, Any]:
        """Validate training results"""
        validation_results = {
            'final_performance': self.metrics.best_reward,
            'convergence_achieved': self.training_state['convergence_achieved'],
            'training_stability': self._calculate_training_stability(),
            'resource_utilization': self._calculate_resource_utilization()
        }
        
        # Additional validation if best model is available
        if self.training_state['best_model_path'] and self.config.enable_checkpointing:
            validation_results['model_validation'] = self._validate_best_model()
        
        return validation_results
    
    def _calculate_training_stability(self) -> float:
        """Calculate training stability metric"""
        if len(self.metrics.metrics['episode_rewards']) < 10:
            return 0.0
        
        recent_rewards = self.metrics.metrics['episode_rewards'][-100:]
        return 1.0 / (1.0 + np.std(recent_rewards))
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization metrics"""
        utilization = {}
        
        if self.metrics.metrics['memory_usage']:
            utilization['memory_efficiency'] = 1.0 - (np.mean(self.metrics.metrics['memory_usage']) / self.config.available_memory_gb)
        
        if self.metrics.metrics['training_time']:
            utilization['time_efficiency'] = self.metrics.episode_count / sum(self.metrics.metrics['training_time'])
        
        return utilization
    
    def _validate_best_model(self) -> Dict[str, Any]:
        """Validate the best model"""
        checkpoint_manager = self.components['checkpoint_manager']
        
        try:
            checkpoint = checkpoint_manager.load_checkpoint(self.training_state['best_model_path'])
            
            if checkpoint:
                return {
                    'model_loadable': True,
                    'checkpoint_episode': checkpoint['episode'],
                    'checkpoint_metrics': checkpoint['metrics']
                }
            else:
                return {'model_loadable': False}
        
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'model_loadable': False, 'error': str(e)}
    
    def _save_results(self):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.config.results_dir) / f"unified_training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save metrics separately
        metrics_file = Path(self.config.results_dir) / f"training_metrics_{timestamp}.json"
        self.metrics.save_metrics(str(metrics_file))
        
        logger.info(f"Results saved to {results_file}")
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up training system")
        
        # Clean up components
        for component_name, component in self.components.items():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup failed for {component_name}: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'config': asdict(self.config),
            'training_state': self.training_state,
            'metrics_summary': self.metrics.get_summary(),
            'results': self.results
        }


def create_unified_training_config(**kwargs) -> UnifiedTrainingConfig:
    """Create unified training configuration"""
    return UnifiedTrainingConfig(**kwargs)


def run_unified_training_example():
    """Example of unified training system"""
    
    # Define factories
    def create_model():
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def create_optimizer(params):
        return optim.Adam(params, lr=0.001)
    
    def create_environment():
        class DummyEnv:
            def reset(self):
                return np.random.randn(128)
            
            def step(self, action):
                return np.random.randn(128), np.random.randn(), np.random.choice([True, False]), {}
        
        return DummyEnv()
    
    # Create configuration
    config = create_unified_training_config(
        model_factory=create_model,
        optimizer_factory=create_optimizer,
        environment_factory=create_environment,
        num_episodes=200,
        training_strategy="sequential",
        enable_parallel_training=False,
        enable_distributed_training=False,
        enable_memory_optimization=True,
        enable_checkpointing=True,
        enable_hyperparameter_tuning=False,  # Disable for quick example
        enable_advanced_scheduling=True,
        target_performance=50.0,
        early_stopping_patience=20
    )
    
    # Initialize and run training
    training_system = UnifiedTrainingSystem(config)
    results = training_system.train()
    
    # Display results
    print("Unified training completed!")
    print(f"Best reward: {results['training_metrics']['best_reward']:.3f}")
    print(f"Total episodes: {results['training_metrics']['total_episodes']}")
    print(f"Convergence achieved: {results['optimization_results']['training']['convergence_achieved']}")
    print(f"Total training time: {results['total_training_time']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    results = run_unified_training_example()
    
    print("Example completed successfully!")