"""
Distributed Multi-Agent Reinforcement Learning Training System
Implements distributed training with multiple nodes and GPUs for MARL systems
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import socket
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed MARL training"""
    # Node configuration
    world_size: int = 1  # Total number of processes
    rank: int = 0  # Current process rank
    local_rank: int = 0  # Local rank on current node
    
    # Network configuration
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # nccl, gloo, or mpi
    
    # Training configuration
    num_agents: int = 4
    agent_types: List[str] = None
    episodes_per_process: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    
    # Distributed settings
    use_data_parallel: bool = True
    use_gradient_compression: bool = True
    gradient_compression_ratio: float = 0.1
    
    # Synchronization
    sync_frequency: int = 10
    use_all_reduce: bool = True
    use_async_updates: bool = False
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_frequency: int = 50
    max_failures: int = 3
    
    # Performance optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    prefetch_factor: int = 2
    
    # Directories
    checkpoint_dir: str = "distributed_checkpoints"
    results_dir: str = "distributed_results"
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = ['strategic', 'tactical', 'execution', 'risk']


class DistributedAgentTrainer:
    """Distributed trainer for individual agents"""
    
    def __init__(self, 
                 agent_id: int,
                 agent_type: str,
                 config: DistributedTrainingConfig,
                 model_factory: Callable,
                 optimizer_factory: Callable,
                 rank: int,
                 world_size: int):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Setup device
        self.device = torch.device(f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = model_factory(agent_type).to(self.device)
        
        # Wrap model with DDP if using data parallel
        if config.use_data_parallel and world_size > 1:
            self.model = DDP(self.model, device_ids=[config.local_rank])
        
        # Initialize optimizer
        self.optimizer = optimizer_factory(self.model.parameters())
        
        # Mixed precision scaler
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.training_stats = {
            'episodes_completed': 0,
            'total_reward': 0.0,
            'loss_history': [],
            'sync_history': [],
            'communication_time': 0.0
        }
        
        # Gradient compression
        if config.use_gradient_compression:
            self.gradient_compressor = GradientCompressor(config.gradient_compression_ratio)
        else:
            self.gradient_compressor = None
        
        logger.info(f"Agent {agent_id} ({agent_type}) initialized on rank {rank}, device {self.device}")
    
    def train_episode(self, environment, episode_idx: int) -> Dict[str, Any]:
        """Train agent for one episode with distributed coordination"""
        
        episode_start_time = time.time()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_length = 0
        
        # Reset environment
        state = environment.reset()
        
        # Experience buffer
        experiences = []
        
        # Episode loop
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            while True:
                # Get action from model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    if self.config.use_data_parallel:
                        action_logits = self.model.module(state_tensor)
                    else:
                        action_logits = self.model(state_tensor)
                    
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                
                # Take action
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
        
        # Train on experiences
        if len(experiences) > 0:
            loss = self._train_on_experiences(experiences)
            episode_loss = loss
        
        # Distributed synchronization
        if episode_idx % self.config.sync_frequency == 0:
            sync_time = self._synchronize_parameters()
            self.training_stats['communication_time'] += sync_time
        
        # Update statistics
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_reward'] += episode_reward
        self.training_stats['loss_history'].append(episode_loss)
        
        episode_time = time.time() - episode_start_time
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'episode_reward': episode_reward,
            'episode_loss': episode_loss,
            'episode_length': episode_length,
            'episode_time': episode_time,
            'rank': self.rank
        }
    
    def _train_on_experiences(self, experiences: List[Dict]) -> float:
        """Train model on experiences with distributed updates"""
        
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
            if self.config.use_data_parallel:
                action_logits = self.model(states)
            else:
                action_logits = self.model(states)
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(actions)
            
            # Policy loss
            loss = -torch.mean(log_probs * returns)
            
            # Add entropy bonus
            entropy = action_dist.entropy().mean()
            loss -= 0.01 * entropy
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Gradient compression
            if self.gradient_compressor is not None:
                self._compress_gradients()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient compression
            if self.gradient_compressor is not None:
                self._compress_gradients()
            
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
    
    def _compress_gradients(self):
        """Compress gradients before communication"""
        if self.gradient_compressor is None:
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = self.gradient_compressor.compress(param.grad.data)
    
    def _synchronize_parameters(self) -> float:
        """Synchronize parameters across distributed processes"""
        if self.world_size <= 1:
            return 0.0
        
        sync_start_time = time.time()
        
        if self.config.use_all_reduce:
            # All-reduce synchronization
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
        else:
            # Parameter server style synchronization
            self._parameter_server_sync()
        
        sync_time = time.time() - sync_start_time
        self.training_stats['sync_history'].append(sync_time)
        
        return sync_time
    
    def _parameter_server_sync(self):
        """Parameter server style synchronization"""
        if self.rank == 0:
            # Aggregate gradients from all workers
            for param in self.model.parameters():
                if param.grad is not None:
                    # Receive gradients from other workers
                    for worker_rank in range(1, self.world_size):
                        worker_grad = torch.zeros_like(param.grad)
                        dist.recv(worker_grad, src=worker_rank)
                        param.grad.data += worker_grad
                    
                    # Average gradients
                    param.grad.data /= self.world_size
            
            # Send updated parameters to workers
            for worker_rank in range(1, self.world_size):
                for param in self.model.parameters():
                    dist.send(param.data, dst=worker_rank)
        else:
            # Send gradients to parameter server
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.send(param.grad.data, dst=0)
            
            # Receive updated parameters
            for param in self.model.parameters():
                dist.recv(param.data, src=0)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'rank': self.rank,
            'episodes_completed': self.training_stats['episodes_completed'],
            'average_reward': self.training_stats['total_reward'] / max(1, self.training_stats['episodes_completed']),
            'recent_loss': self.training_stats['loss_history'][-10:] if self.training_stats['loss_history'] else [],
            'avg_sync_time': np.mean(self.training_stats['sync_history']) if self.training_stats['sync_history'] else 0.0,
            'total_communication_time': self.training_stats['communication_time']
        }


class GradientCompressor:
    """Gradient compression for efficient communication"""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
    
    def compress(self, gradient: torch.Tensor) -> torch.Tensor:
        """Compress gradient using top-k compression"""
        # Flatten gradient
        flat_grad = gradient.flatten()
        
        # Find top-k elements
        k = max(1, int(len(flat_grad) * self.compression_ratio))
        _, indices = torch.topk(flat_grad.abs(), k)
        
        # Create compressed gradient
        compressed = torch.zeros_like(flat_grad)
        compressed[indices] = flat_grad[indices]
        
        return compressed.reshape(gradient.shape)


class FaultTolerantCoordinator:
    """Fault tolerance coordinator for distributed training"""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.failed_nodes = set()
        self.failure_count = 0
        self.last_checkpoint = None
        
    def handle_node_failure(self, failed_rank: int):
        """Handle node failure"""
        self.failed_nodes.add(failed_rank)
        self.failure_count += 1
        
        logger.warning(f"Node {failed_rank} failed. Total failures: {self.failure_count}")
        
        if self.failure_count >= self.config.max_failures:
            logger.error("Maximum failures reached. Stopping training.")
            return False
        
        return True
    
    def can_continue_training(self) -> bool:
        """Check if training can continue"""
        active_nodes = self.config.world_size - len(self.failed_nodes)
        return active_nodes >= 1 and self.failure_count < self.config.max_failures


class DistributedMARLTrainer:
    """Main distributed MARL trainer"""
    
    def __init__(self, 
                 config: DistributedTrainingConfig,
                 model_factory: Callable,
                 optimizer_factory: Callable,
                 environment_factory: Callable,
                 rank: int,
                 world_size: int):
        self.config = config
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.environment_factory = environment_factory
        self.rank = rank
        self.world_size = world_size
        
        # Initialize fault tolerance
        if config.enable_fault_tolerance:
            self.fault_coordinator = FaultTolerantCoordinator(config)
        else:
            self.fault_coordinator = None
        
        # Initialize agents
        self.agents = []
        self._initialize_agents()
        
        # Training state
        self.training_results = {
            'episode_rewards': defaultdict(list),
            'episode_losses': defaultdict(list),
            'communication_stats': defaultdict(list),
            'node_stats': {}
        }
        
        # Create directories
        if rank == 0:
            Path(config.checkpoint_dir).mkdir(exist_ok=True)
            Path(config.results_dir).mkdir(exist_ok=True)
        
        logger.info(f"Distributed trainer initialized on rank {rank}/{world_size}")
    
    def _initialize_agents(self):
        """Initialize agents for this process"""
        agents_per_process = self.config.num_agents // self.world_size
        start_idx = self.rank * agents_per_process
        end_idx = start_idx + agents_per_process
        
        # Handle remainder
        if self.rank == self.world_size - 1:
            end_idx = self.config.num_agents
        
        for i in range(start_idx, end_idx):
            agent_type = self.config.agent_types[i % len(self.config.agent_types)]
            
            agent = DistributedAgentTrainer(
                agent_id=i,
                agent_type=agent_type,
                config=self.config,
                model_factory=self.model_factory,
                optimizer_factory=self.optimizer_factory,
                rank=self.rank,
                world_size=self.world_size
            )
            
            self.agents.append(agent)
    
    def train_distributed(self, num_episodes: int) -> Dict[str, Any]:
        """Run distributed training"""
        logger.info(f"Starting distributed training on rank {self.rank}")
        
        training_start_time = time.time()
        
        try:
            # Create environments for each agent
            environments = [self.environment_factory() for _ in self.agents]
            
            # Training loop
            for episode in range(num_episodes):
                episode_results = []
                
                # Train each agent
                for agent, env in zip(self.agents, environments):
                    try:
                        result = agent.train_episode(env, episode)
                        episode_results.append(result)
                        
                        # Store results
                        self.training_results['episode_rewards'][agent.agent_type].append(result['episode_reward'])
                        self.training_results['episode_losses'][agent.agent_type].append(result['episode_loss'])
                        
                    except Exception as e:
                        logger.error(f"Training failed for agent {agent.agent_id}: {e}")
                        if self.fault_coordinator:
                            if not self.fault_coordinator.handle_node_failure(self.rank):
                                break
                
                # Checkpoint saving
                if episode % self.config.checkpoint_frequency == 0 and self.rank == 0:
                    self._save_checkpoint(episode)
                
                # Log progress
                if episode % 10 == 0:
                    self._log_progress(episode, episode_results)
                
                # Check for early termination
                if self.fault_coordinator and not self.fault_coordinator.can_continue_training():
                    logger.info("Stopping training due to fault tolerance limits")
                    break
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        training_time = time.time() - training_start_time
        
        # Gather results from all processes
        final_results = self._gather_results(training_time)
        
        return final_results
    
    def _log_progress(self, episode: int, episode_results: List[Dict]):
        """Log training progress"""
        if not episode_results:
            return
        
        avg_reward = np.mean([r['episode_reward'] for r in episode_results])
        avg_loss = np.mean([r['episode_loss'] for r in episode_results])
        
        logger.info(f"Rank {self.rank} - Episode {episode}: "
                   f"Avg Reward: {avg_reward:.3f}, Avg Loss: {avg_loss:.6f}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'episode': episode,
            'training_results': self.training_results,
            'config': self.config,
            'world_size': self.world_size
        }
        
        # Save agent models
        for agent in self.agents:
            checkpoint[f'agent_{agent.agent_id}_model'] = agent.model.state_dict()
            checkpoint[f'agent_{agent.agent_id}_optimizer'] = agent.optimizer.state_dict()
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"distributed_checkpoint_episode_{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _gather_results(self, training_time: float) -> Dict[str, Any]:
        """Gather results from all processes"""
        
        # Collect local results
        local_results = {
            'rank': self.rank,
            'training_time': training_time,
            'agent_stats': [agent.get_training_stats() for agent in self.agents],
            'training_results': self.training_results
        }
        
        if self.world_size > 1:
            # Gather results from all processes
            all_results = [None] * self.world_size
            dist.all_gather_object(all_results, local_results)
            
            if self.rank == 0:
                # Compile results from all processes
                compiled_results = self._compile_distributed_results(all_results)
                self._save_results(compiled_results)
                return compiled_results
            else:
                return local_results
        else:
            # Single process
            if self.rank == 0:
                self._save_results(local_results)
            return local_results
    
    def _compile_distributed_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compile results from all distributed processes"""
        
        # Aggregate statistics
        total_agents = sum(len(result['agent_stats']) for result in all_results)
        total_training_time = max(result['training_time'] for result in all_results)
        
        # Compile agent statistics
        agent_stats_by_type = defaultdict(list)
        for result in all_results:
            for agent_stat in result['agent_stats']:
                agent_stats_by_type[agent_stat['agent_type']].append(agent_stat)
        
        # Calculate summary statistics
        summary_stats = {}
        for agent_type, stats in agent_stats_by_type.items():
            episode_counts = [stat['episodes_completed'] for stat in stats]
            avg_rewards = [stat['average_reward'] for stat in stats]
            sync_times = [stat['avg_sync_time'] for stat in stats]
            
            summary_stats[agent_type] = {
                'total_agents': len(stats),
                'total_episodes': sum(episode_counts),
                'mean_reward': np.mean(avg_rewards),
                'std_reward': np.std(avg_rewards),
                'mean_sync_time': np.mean(sync_times),
                'total_communication_time': sum(stat['total_communication_time'] for stat in stats)
            }
        
        # Communication efficiency metrics
        total_communication_time = sum(
            stat['total_communication_time'] 
            for result in all_results 
            for stat in result['agent_stats']
        )
        
        communication_efficiency = (total_training_time - total_communication_time) / total_training_time
        
        return {
            'distributed_config': {
                'world_size': self.world_size,
                'backend': self.config.backend,
                'use_data_parallel': self.config.use_data_parallel,
                'use_gradient_compression': self.config.use_gradient_compression
            },
            'summary_stats': summary_stats,
            'performance_metrics': {
                'total_agents': total_agents,
                'total_training_time': total_training_time,
                'total_communication_time': total_communication_time,
                'communication_efficiency': communication_efficiency,
                'average_episodes_per_second': sum(
                    stat['episodes_completed'] 
                    for result in all_results 
                    for stat in result['agent_stats']
                ) / total_training_time
            },
            'node_results': all_results
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        if self.rank != 0:
            return
        
        results_file = Path(self.config.results_dir) / f"distributed_results_{int(time.time())}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")


def setup_distributed_training(rank: int, world_size: int, config: DistributedTrainingConfig):
    """Setup distributed training environment"""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    logger.info(f"Distributed training setup completed for rank {rank}/{world_size}")


def cleanup_distributed_training():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
    
    logger.info("Distributed training cleanup completed")


def run_distributed_training(
    rank: int,
    world_size: int,
    config: DistributedTrainingConfig,
    model_factory: Callable,
    optimizer_factory: Callable,
    environment_factory: Callable,
    num_episodes: int
) -> Dict[str, Any]:
    """Run distributed training on single process"""
    
    try:
        # Setup distributed environment
        setup_distributed_training(rank, world_size, config)
        
        # Initialize trainer
        trainer = DistributedMARLTrainer(
            config=config,
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            environment_factory=environment_factory,
            rank=rank,
            world_size=world_size
        )
        
        # Run training
        results = trainer.train_distributed(num_episodes)
        
        return results
        
    except Exception as e:
        logger.error(f"Distributed training failed on rank {rank}: {e}")
        raise
    finally:
        cleanup_distributed_training()


def launch_distributed_training(
    world_size: int,
    model_factory: Callable,
    optimizer_factory: Callable,
    environment_factory: Callable,
    num_episodes: int = 1000,
    **config_kwargs
) -> Dict[str, Any]:
    """Launch distributed training with multiple processes"""
    
    # Create configuration
    config = DistributedTrainingConfig(world_size=world_size, **config_kwargs)
    
    # Launch processes
    if world_size > 1:
        mp.spawn(
            run_distributed_training,
            args=(world_size, config, model_factory, optimizer_factory, environment_factory, num_episodes),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process training
        results = run_distributed_training(
            rank=0,
            world_size=1,
            config=config,
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            environment_factory=environment_factory,
            num_episodes=num_episodes
        )
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
    
    # Launch distributed training
    world_size = 2  # Number of processes
    results = launch_distributed_training(
        world_size=world_size,
        model_factory=create_model,
        optimizer_factory=create_optimizer,
        environment_factory=create_environment,
        num_episodes=100,
        num_agents=4,
        backend="gloo",  # Use gloo for CPU or mixed training
        use_data_parallel=True,
        use_gradient_compression=True
    )
    
    if results:
        print("Distributed training completed!")
        print(f"Communication efficiency: {results['performance_metrics']['communication_efficiency']:.3f}")
        print(f"Episodes per second: {results['performance_metrics']['average_episodes_per_second']:.2f}")