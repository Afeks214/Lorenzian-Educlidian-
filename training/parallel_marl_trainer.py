"""
Parallel Multi-Agent Training System for MARL
Implements parallel training for multiple agents with optimized resource utilization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import json
import queue
import threading
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from collections import defaultdict
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel MARL training"""
    # Agent configuration
    num_agents: int = 4
    agent_types: List[str] = None  # ['strategic', 'tactical', 'execution', 'risk']
    
    # Parallel processing settings
    num_workers: int = 4
    num_gpus: int = 1
    use_multiprocessing: bool = True
    use_async_training: bool = True
    
    # Training parameters
    episodes_per_worker: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Memory management
    max_memory_per_worker: float = 2.0  # GB
    gradient_accumulation_steps: int = 4
    
    # Synchronization
    sync_frequency: int = 10  # Episodes between synchronization
    use_parameter_server: bool = True
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_compression: bool = True
    use_model_parallelism: bool = False
    
    # Directories
    checkpoint_dir: str = "parallel_checkpoints"
    results_dir: str = "parallel_results"
    
    # Communication
    communication_backend: str = "nccl"  # nccl, gloo, or mpi
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = ['strategic', 'tactical', 'execution', 'risk']


class AgentTrainingWorker:
    """Worker class for training individual agents"""
    
    def __init__(self, 
                 agent_id: int,
                 agent_type: str,
                 config: ParallelTrainingConfig,
                 model_factory: Callable,
                 optimizer_factory: Callable,
                 device: torch.device):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.device = device
        
        # Initialize model and optimizer
        self.model = model_factory(agent_type).to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        
        # Training state
        self.training_stats = {
            'episodes_completed': 0,
            'total_reward': 0.0,
            'loss_history': [],
            'performance_metrics': {}
        }
        
        # Memory management
        self.memory_monitor = MemoryMonitor(config.max_memory_per_worker)
        
        # Mixed precision
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        logger.info(f"Agent {agent_id} ({agent_type}) initialized on {device}")
    
    def train_episode(self, environment, shared_parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Train agent for one episode"""
        
        # Sync with shared parameters if provided
        if shared_parameters is not None:
            self._sync_parameters(shared_parameters)
        
        # Memory check before training
        self.memory_monitor.check_memory()
        
        episode_reward = 0.0
        episode_loss = 0.0
        episode_length = 0
        
        # Reset environment
        state = environment.reset()
        
        # Experience buffer for this episode
        experiences = []
        
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            while True:
                # Get action from model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_logits = self.model(state_tensor)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, info = environment.step(action.item())
                
                # Store experience
                experiences.append({
                    'state': state,
                    'action': action.item(),
                    'reward': reward,
                    'log_prob': log_prob.item(),
                    'done': done
                })
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                state = next_state
        
        # Train on collected experiences
        if len(experiences) > 0:
            loss = self._train_on_experiences(experiences)
            episode_loss = loss
        
        # Update statistics
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_reward'] += episode_reward
        self.training_stats['loss_history'].append(episode_loss)
        
        # Clean up memory
        self.memory_monitor.cleanup()
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'episode_reward': episode_reward,
            'episode_loss': episode_loss,
            'episode_length': episode_length,
            'model_parameters': self.model.state_dict() if self.config.use_parameter_server else None
        }
    
    def _train_on_experiences(self, experiences: List[Dict]) -> float:
        """Train model on collected experiences"""
        
        # Prepare batch
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # Training step
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # Forward pass
            action_logits = self.model(states)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(actions)
            
            # Policy loss
            loss = -torch.mean(log_probs * returns)
            
            # Entropy bonus for exploration
            entropy = action_dist.entropy().mean()
            loss -= 0.01 * entropy
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def _calculate_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def _sync_parameters(self, shared_parameters: Dict):
        """Synchronize parameters with shared state"""
        if self.agent_type in shared_parameters:
            self.model.load_state_dict(shared_parameters[self.agent_type])
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'episodes_completed': self.training_stats['episodes_completed'],
            'average_reward': self.training_stats['total_reward'] / max(1, self.training_stats['episodes_completed']),
            'recent_loss': self.training_stats['loss_history'][-10:] if self.training_stats['loss_history'] else [],
            'memory_usage': self.memory_monitor.get_memory_usage()
        }


class MemoryMonitor:
    """Monitor and manage memory usage during training"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
    def check_memory(self):
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if memory_info.rss > self.max_memory_bytes * 0.9:  # 90% threshold
            logger.warning(f"Memory usage high: {memory_info.rss / 1024**3:.2f} GB")
            self.cleanup()
    
    def cleanup(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_rss_gb': memory_info.rss / 1024**3,
            'memory_vms_gb': memory_info.vms / 1024**3,
            'memory_percent': process.memory_percent()
        }


class ParameterServer:
    """Parameter server for coordinating agent training"""
    
    def __init__(self, config: ParallelTrainingConfig):
        self.config = config
        self.shared_parameters = {}
        self.parameter_lock = threading.Lock()
        self.update_counts = defaultdict(int)
        
        # Initialize shared parameters for each agent type
        for agent_type in config.agent_types:
            self.shared_parameters[agent_type] = None
    
    def update_parameters(self, agent_type: str, parameters: Dict[str, torch.Tensor]):
        """Update shared parameters from agent"""
        with self.parameter_lock:
            if self.shared_parameters[agent_type] is None:
                self.shared_parameters[agent_type] = parameters
            else:
                # Exponential moving average update
                alpha = 0.1
                for key, param in parameters.items():
                    if key in self.shared_parameters[agent_type]:
                        self.shared_parameters[agent_type][key] = (
                            alpha * param + (1 - alpha) * self.shared_parameters[agent_type][key]
                        )
                    else:
                        self.shared_parameters[agent_type][key] = param
            
            self.update_counts[agent_type] += 1
    
    def get_parameters(self, agent_type: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get shared parameters for agent type"""
        with self.parameter_lock:
            return self.shared_parameters.get(agent_type)
    
    def get_update_counts(self) -> Dict[str, int]:
        """Get parameter update counts"""
        return dict(self.update_counts)


class ParallelMARLTrainer:
    """Main parallel MARL trainer"""
    
    def __init__(self, 
                 config: ParallelTrainingConfig,
                 model_factory: Callable,
                 optimizer_factory: Callable,
                 environment_factory: Callable):
        self.config = config
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.environment_factory = environment_factory
        
        # Setup devices
        self.devices = self._setup_devices()
        
        # Initialize parameter server
        if config.use_parameter_server:
            self.parameter_server = ParameterServer(config)
        else:
            self.parameter_server = None
        
        # Initialize workers
        self.workers = []
        self._initialize_workers()
        
        # Training state
        self.training_results = {
            'episode_rewards': defaultdict(list),
            'episode_losses': defaultdict(list),
            'training_stats': defaultdict(dict),
            'synchronization_history': []
        }
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.results_dir).mkdir(exist_ok=True)
        
        logger.info(f"Parallel MARL trainer initialized with {len(self.workers)} workers")
    
    def _setup_devices(self) -> List[torch.device]:
        """Setup compute devices for parallel training"""
        devices = []
        
        if torch.cuda.is_available() and self.config.num_gpus > 0:
            for i in range(min(self.config.num_gpus, torch.cuda.device_count())):
                devices.append(torch.device(f'cuda:{i}'))
        
        # Fill remaining with CPU
        while len(devices) < self.config.num_workers:
            devices.append(torch.device('cpu'))
        
        return devices
    
    def _initialize_workers(self):
        """Initialize training workers"""
        for i in range(self.config.num_agents):
            agent_type = self.config.agent_types[i % len(self.config.agent_types)]
            device = self.devices[i % len(self.devices)]
            
            worker = AgentTrainingWorker(
                agent_id=i,
                agent_type=agent_type,
                config=self.config,
                model_factory=self.model_factory,
                optimizer_factory=self.optimizer_factory,
                device=device
            )
            
            self.workers.append(worker)
    
    def train_parallel(self, num_episodes: int) -> Dict[str, Any]:
        """Run parallel training"""
        logger.info(f"Starting parallel training with {len(self.workers)} workers")
        
        training_start_time = time.time()
        
        if self.config.use_async_training:
            results = self._train_async(num_episodes)
        else:
            results = self._train_sync(num_episodes)
        
        training_time = time.time() - training_start_time
        
        # Compile final results
        final_results = self._compile_results(results, training_time)
        
        # Save results
        self._save_results(final_results)
        
        return final_results
    
    def _train_async(self, num_episodes: int) -> Dict[str, Any]:
        """Asynchronous training with multiple workers"""
        logger.info("Running asynchronous parallel training")
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async training
            results = loop.run_until_complete(
                self._async_training_loop(num_episodes)
            )
        finally:
            loop.close()
        
        return results
    
    async def _async_training_loop(self, num_episodes: int) -> Dict[str, Any]:
        """Async training loop"""
        
        # Create tasks for each worker
        tasks = []
        
        for worker in self.workers:
            task = asyncio.create_task(
                self._async_worker_training(worker, num_episodes)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        return {'worker_results': results}
    
    async def _async_worker_training(self, worker: AgentTrainingWorker, num_episodes: int) -> Dict[str, Any]:
        """Async training for single worker"""
        
        environment = self.environment_factory()
        worker_results = []
        
        for episode in range(num_episodes):
            # Get shared parameters if using parameter server
            shared_params = None
            if self.parameter_server and episode % self.config.sync_frequency == 0:
                shared_params = self.parameter_server.get_parameters(worker.agent_type)
            
            # Train episode
            episode_result = worker.train_episode(environment, shared_params)
            worker_results.append(episode_result)
            
            # Update parameter server
            if self.parameter_server and episode_result['model_parameters']:
                self.parameter_server.update_parameters(
                    worker.agent_type,
                    episode_result['model_parameters']
                )
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Worker {worker.agent_id}: Episode {episode}, Reward: {episode_result['episode_reward']:.3f}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
        
        return {
            'worker_id': worker.agent_id,
            'agent_type': worker.agent_type,
            'episodes': worker_results,
            'final_stats': worker.get_training_stats()
        }
    
    def _train_sync(self, num_episodes: int) -> Dict[str, Any]:
        """Synchronous training with process pool"""
        logger.info("Running synchronous parallel training")
        
        if self.config.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                # Submit training tasks
                futures = []
                for worker in self.workers:
                    future = executor.submit(
                        self._sync_worker_training, 
                        worker, 
                        num_episodes
                    )
                    futures.append(future)
                
                # Collect results
                results = [future.result() for future in futures]
        else:
            # Use threading for lighter workload
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                # Submit training tasks
                futures = []
                for worker in self.workers:
                    future = executor.submit(
                        self._sync_worker_training,
                        worker,
                        num_episodes
                    )
                    futures.append(future)
                
                # Collect results
                results = [future.result() for future in futures]
        
        return {'worker_results': results}
    
    def _sync_worker_training(self, worker: AgentTrainingWorker, num_episodes: int) -> Dict[str, Any]:
        """Synchronous training for single worker"""
        
        environment = self.environment_factory()
        worker_results = []
        
        for episode in range(num_episodes):
            # Get shared parameters if using parameter server
            shared_params = None
            if self.parameter_server and episode % self.config.sync_frequency == 0:
                shared_params = self.parameter_server.get_parameters(worker.agent_type)
            
            # Train episode
            episode_result = worker.train_episode(environment, shared_params)
            worker_results.append(episode_result)
            
            # Update parameter server
            if self.parameter_server and episode_result['model_parameters']:
                self.parameter_server.update_parameters(
                    worker.agent_type,
                    episode_result['model_parameters']
                )
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Worker {worker.agent_id}: Episode {episode}, Reward: {episode_result['episode_reward']:.3f}")
        
        return {
            'worker_id': worker.agent_id,
            'agent_type': worker.agent_type,
            'episodes': worker_results,
            'final_stats': worker.get_training_stats()
        }
    
    def _compile_results(self, results: Dict[str, Any], training_time: float) -> Dict[str, Any]:
        """Compile training results"""
        
        worker_results = results['worker_results']
        
        # Aggregate results by agent type
        agent_type_results = defaultdict(list)
        for worker_result in worker_results:
            agent_type = worker_result['agent_type']
            agent_type_results[agent_type].append(worker_result)
        
        # Calculate summary statistics
        summary_stats = {}
        for agent_type, type_results in agent_type_results.items():
            episode_rewards = []
            episode_losses = []
            
            for worker_result in type_results:
                for episode in worker_result['episodes']:
                    episode_rewards.append(episode['episode_reward'])
                    episode_losses.append(episode['episode_loss'])
            
            summary_stats[agent_type] = {
                'total_episodes': len(episode_rewards),
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_loss': np.mean(episode_losses),
                'std_loss': np.std(episode_losses),
                'best_reward': max(episode_rewards) if episode_rewards else 0,
                'worst_reward': min(episode_rewards) if episode_rewards else 0
            }
        
        # Parameter server stats
        parameter_server_stats = {}
        if self.parameter_server:
            parameter_server_stats = {
                'update_counts': self.parameter_server.get_update_counts(),
                'shared_parameter_types': list(self.parameter_server.shared_parameters.keys())
            }
        
        return {
            'training_config': {
                'num_agents': self.config.num_agents,
                'agent_types': self.config.agent_types,
                'num_workers': self.config.num_workers,
                'use_async_training': self.config.use_async_training,
                'use_parameter_server': self.config.use_parameter_server
            },
            'worker_results': worker_results,
            'summary_stats': summary_stats,
            'parameter_server_stats': parameter_server_stats,
            'training_time': training_time,
            'performance_metrics': {
                'episodes_per_second': sum(len(wr['episodes']) for wr in worker_results) / training_time,
                'average_episode_length': np.mean([
                    episode['episode_length'] 
                    for wr in worker_results 
                    for episode in wr['episodes']
                ])
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        results_file = Path(self.config.results_dir) / f"parallel_training_results_{int(time.time())}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def save_models(self, checkpoint_name: str = "parallel_models"):
        """Save all trained models"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        for worker in self.workers:
            model_path = checkpoint_dir / f"{checkpoint_name}_agent_{worker.agent_id}_{worker.agent_type}.pth"
            torch.save({
                'model_state_dict': worker.model.state_dict(),
                'optimizer_state_dict': worker.optimizer.state_dict(),
                'agent_id': worker.agent_id,
                'agent_type': worker.agent_type,
                'training_stats': worker.get_training_stats()
            }, model_path)
        
        logger.info(f"Models saved to {checkpoint_dir}")
    
    def load_models(self, checkpoint_name: str = "parallel_models"):
        """Load trained models"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        for worker in self.workers:
            model_path = checkpoint_dir / f"{checkpoint_name}_agent_{worker.agent_id}_{worker.agent_type}.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=worker.device)
                worker.model.load_state_dict(checkpoint['model_state_dict'])
                worker.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Loaded model for agent {worker.agent_id}")
            else:
                logger.warning(f"Checkpoint not found: {model_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        summary = {
            'total_workers': len(self.workers),
            'worker_stats': []
        }
        
        for worker in self.workers:
            summary['worker_stats'].append(worker.get_training_stats())
        
        return summary


# Convenience functions
def create_parallel_training_config(**kwargs) -> ParallelTrainingConfig:
    """Create parallel training configuration with defaults"""
    return ParallelTrainingConfig(**kwargs)


def run_parallel_marl_training(
    model_factory: Callable,
    optimizer_factory: Callable,
    environment_factory: Callable,
    num_episodes: int = 1000,
    **config_kwargs
) -> Dict[str, Any]:
    """Run parallel MARL training"""
    
    # Create configuration
    config = create_parallel_training_config(**config_kwargs)
    
    # Initialize trainer
    trainer = ParallelMARLTrainer(
        config=config,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        environment_factory=environment_factory
    )
    
    # Run training
    results = trainer.train_parallel(num_episodes)
    
    # Save models
    trainer.save_models()
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    def create_model(agent_type):
        """Example model factory"""
        if agent_type == 'strategic':
            return nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 4))
        elif agent_type == 'tactical':
            return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 3))
        else:
            return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    
    def create_optimizer(params):
        """Example optimizer factory"""
        return optim.Adam(params, lr=0.001)
    
    def create_environment():
        """Example environment factory"""
        class DummyEnv:
            def reset(self):
                return np.random.randn(32)
            
            def step(self, action):
                return np.random.randn(32), np.random.randn(), np.random.choice([True, False]), {}
        
        return DummyEnv()
    
    # Run parallel training
    results = run_parallel_marl_training(
        model_factory=create_model,
        optimizer_factory=create_optimizer,
        environment_factory=create_environment,
        num_episodes=100,
        num_agents=4,
        num_workers=4,
        use_async_training=True,
        use_parameter_server=True
    )
    
    print("Parallel training completed!")
    print(f"Total training time: {results['training_time']:.2f} seconds")
    print(f"Episodes per second: {results['performance_metrics']['episodes_per_second']:.2f}")