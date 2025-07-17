"""
PettingZoo Parallel Training Components

This module provides parallel training capabilities specifically designed for
PettingZoo environments, optimizing for multi-agent turn-based execution
while maintaining training efficiency and scalability.

Key Features:
- Parallel environment execution for PettingZoo AEC environments
- Efficient experience aggregation across parallel workers
- Load balancing and resource management
- Fault tolerance and recovery mechanisms
- Performance monitoring and optimization
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time
import threading
import queue
import pickle
import os
import signal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue, Pipe, Event, Lock, Manager
from contextlib import contextmanager
import psutil

# PettingZoo imports
from pettingzoo import AECEnv

# Internal imports
from .pettingzoo_mappo_trainer import PettingZooMAPPOTrainer, TrainingConfig
from .pettingzoo_environment_manager import EnvironmentFactory, EnvironmentConfig
from .pettingzoo_reward_system import PettingZooRewardSystem

logger = logging.getLogger(__name__)


@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training"""
    # Worker configuration
    num_workers: int = 4
    episodes_per_worker: int = 250
    steps_per_episode: int = 1000
    
    # Parallel execution settings
    use_multiprocessing: bool = True
    use_shared_memory: bool = True
    worker_timeout: float = 300.0  # 5 minutes
    
    # Experience aggregation
    experience_buffer_size: int = 100000
    batch_size: int = 64
    aggregation_frequency: int = 100
    
    # Load balancing
    dynamic_load_balancing: bool = True
    load_balance_frequency: int = 50
    max_worker_imbalance: float = 0.2
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    max_worker_failures: int = 3
    worker_restart_delay: float = 1.0
    
    # Performance optimization
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Resource management
    max_memory_per_worker: float = 2.0  # GB
    cpu_affinity: bool = True
    gpu_memory_fraction: float = 0.1
    
    # Monitoring
    enable_worker_monitoring: bool = True
    monitoring_frequency: float = 10.0  # seconds
    log_worker_stats: bool = True


class WorkerProcess:
    """Individual worker process for parallel training"""
    
    def __init__(self, worker_id: int, config: ParallelTrainingConfig, 
                 env_factory: Callable, trainer_config: TrainingConfig):
        self.worker_id = worker_id
        self.config = config
        self.env_factory = env_factory
        self.trainer_config = trainer_config
        
        # Worker state
        self.is_running = False
        self.episode_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        
        # Communication
        self.result_queue = None
        self.command_queue = None
        self.status_pipe = None
        
        # Resources
        self.environment = None
        self.trainer = None
        self.reward_system = None
        
        # Performance tracking
        self.start_time = None
        self.last_update = None
        self.performance_metrics = {
            'episodes_per_second': 0.0,
            'steps_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_percent': 0.0
        }
    
    def initialize(self):
        """Initialize worker components"""
        try:
            # Set process title for monitoring
            if hasattr(os, 'setproctitle'):
                os.setproctitle(f"pettingzoo_worker_{self.worker_id}")
            
            # Set CPU affinity if enabled
            if self.config.cpu_affinity:
                cpu_count = psutil.cpu_count()
                cpu_id = self.worker_id % cpu_count
                psutil.Process().cpu_affinity([cpu_id])
            
            # Initialize environment
            self.environment = self.env_factory()
            
            # Initialize trainer
            self.trainer = PettingZooMAPPOTrainer(self.trainer_config)
            
            # Initialize reward system
            from .pettingzoo_reward_system import create_reward_config
            reward_config = create_reward_config()
            self.reward_system = PettingZooRewardSystem(reward_config)
            
            # Set memory limits
            if self.config.max_memory_per_worker > 0:
                self._set_memory_limit()
            
            self.start_time = time.time()
            self.last_update = time.time()
            self.is_running = True
            
            logger.info(f"Worker {self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            raise
    
    def run_episode(self) -> Dict[str, Any]:
        """Run single episode in worker"""
        if not self.is_running:
            return {'error': 'Worker not initialized'}
        
        try:
            # Reset environment
            self.environment.reset()
            
            episode_reward = 0.0
            episode_length = 0
            agent_experiences = defaultdict(list)
            
            # Episode loop
            while self.environment.agents:
                # Get current agent
                current_agent = self.environment.agent_selection
                
                # Get observation
                observation = self.environment.observe(current_agent)
                
                # Get action from trainer
                action, log_prob, value = self.trainer.get_action_and_value(
                    observation, current_agent
                )
                
                # Store pre-step state
                pre_step_state = {'observation': observation}
                
                # Execute action
                self.environment.step(action)
                
                # Get reward and done status
                reward = self.environment.rewards.get(current_agent, 0.0)
                done = self.environment.dones.get(current_agent, False)
                truncated = self.environment.truncations.get(current_agent, False)
                
                # Calculate enhanced reward
                enhanced_reward = self.reward_system.calculate_agent_reward(
                    agent=current_agent,
                    env=self.environment,
                    action=action,
                    pre_step_state=pre_step_state,
                    post_step_info={'episode_step': episode_length}
                )
                
                # Store experience
                experience = {
                    'observation': observation,
                    'action': action,
                    'reward': enhanced_reward,
                    'log_prob': log_prob,
                    'value': value,
                    'done': done or truncated,
                    'agent': current_agent
                }
                
                agent_experiences[current_agent].append(experience)
                
                # Update metrics
                episode_reward += enhanced_reward
                episode_length += 1
                self.total_steps += 1
                
                # Check termination
                if done or truncated or episode_length >= self.config.steps_per_episode:
                    break
            
            # Episode completed
            self.episode_count += 1
            self.total_reward += episode_reward
            
            # Update performance metrics
            self._update_performance_metrics()
            
            return {
                'worker_id': self.worker_id,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'agent_experiences': dict(agent_experiences),
                'performance_metrics': self.performance_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} episode failed: {e}")
            return {'error': str(e), 'worker_id': self.worker_id}
    
    def run_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Run multiple episodes in worker"""
        results = []
        
        for episode in range(num_episodes):
            result = self.run_episode()
            results.append(result)
            
            # Check for shutdown signal
            if not self.is_running:
                break
        
        return results
    
    def _update_performance_metrics(self):
        """Update worker performance metrics"""
        current_time = time.time()
        
        # Episodes per second
        if self.episode_count > 0:
            elapsed = current_time - self.start_time
            self.performance_metrics['episodes_per_second'] = self.episode_count / elapsed
        
        # Steps per second
        if self.total_steps > 0:
            elapsed = current_time - self.start_time
            self.performance_metrics['steps_per_second'] = self.total_steps / elapsed
        
        # Memory usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.performance_metrics['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
            self.performance_metrics['cpu_percent'] = process.cpu_percent()
        except:
            pass
        
        self.last_update = current_time
    
    def _set_memory_limit(self):
        """Set memory limit for worker process"""
        try:
            import resource
            memory_bytes = int(self.config.max_memory_per_worker * 1024 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except:
            pass
    
    def shutdown(self):
        """Shutdown worker gracefully"""
        self.is_running = False
        
        if self.environment and hasattr(self.environment, 'close'):
            self.environment.close()
        
        if self.trainer and hasattr(self.trainer, '_cleanup'):
            self.trainer._cleanup()
        
        logger.info(f"Worker {self.worker_id} shutdown complete")


def worker_process_main(worker_id: int, config: ParallelTrainingConfig,
                       env_factory: Callable, trainer_config: TrainingConfig,
                       result_queue: Queue, command_queue: Queue,
                       status_pipe: Pipe, shutdown_event: Event):
    """Main function for worker process"""
    try:
        # Initialize worker
        worker = WorkerProcess(worker_id, config, env_factory, trainer_config)
        worker.result_queue = result_queue
        worker.command_queue = command_queue
        worker.status_pipe = status_pipe
        
        worker.initialize()
        
        # Send ready signal
        status_pipe.send({'status': 'ready', 'worker_id': worker_id})
        
        # Main worker loop
        while not shutdown_event.is_set():
            try:
                # Check for commands
                if not command_queue.empty():
                    command = command_queue.get_nowait()
                    
                    if command['type'] == 'run_episodes':
                        num_episodes = command['num_episodes']
                        results = worker.run_episodes(num_episodes)
                        
                        # Send results
                        for result in results:
                            result_queue.put(result)
                        
                        # Send completion signal
                        status_pipe.send({
                            'status': 'completed',
                            'worker_id': worker_id,
                            'episodes_completed': len(results)
                        })
                    
                    elif command['type'] == 'shutdown':
                        break
                
                # Send periodic status updates
                if time.time() - worker.last_update > config.monitoring_frequency:
                    worker._update_performance_metrics()
                    status_pipe.send({
                        'status': 'running',
                        'worker_id': worker_id,
                        'performance_metrics': worker.performance_metrics
                    })
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                status_pipe.send({
                    'status': 'error',
                    'worker_id': worker_id,
                    'error': str(e)
                })
        
        # Shutdown worker
        worker.shutdown()
        status_pipe.send({'status': 'shutdown', 'worker_id': worker_id})
        
    except Exception as e:
        logger.error(f"Worker process {worker_id} failed: {e}")
        status_pipe.send({
            'status': 'failed',
            'worker_id': worker_id,
            'error': str(e)
        })


class ExperienceAggregator:
    """Aggregates experiences from parallel workers"""
    
    def __init__(self, config: ParallelTrainingConfig):
        self.config = config
        self.experiences = defaultdict(list)
        self.episode_count = 0
        self.total_experiences = 0
        
        # Performance tracking
        self.aggregation_times = deque(maxlen=100)
        self.experience_rates = deque(maxlen=100)
    
    def add_episode_result(self, result: Dict[str, Any]):
        """Add episode result to aggregator"""
        if 'error' in result:
            logger.warning(f"Skipping error result: {result['error']}")
            return
        
        start_time = time.time()
        
        # Extract experiences
        agent_experiences = result.get('agent_experiences', {})
        
        for agent, experiences in agent_experiences.items():
            self.experiences[agent].extend(experiences)
            self.total_experiences += len(experiences)
        
        self.episode_count += 1
        
        # Track performance
        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)
        
        if self.aggregation_times:
            avg_time = np.mean(self.aggregation_times)
            self.experience_rates.append(len(agent_experiences) / avg_time if avg_time > 0 else 0)
    
    def get_batch(self, batch_size: int) -> Optional[Dict[str, Any]]:
        """Get batch of experiences for training"""
        if self.total_experiences < batch_size:
            return None
        
        # Collect experiences from all agents
        all_experiences = []
        for agent_experiences in self.experiences.values():
            all_experiences.extend(agent_experiences)
        
        # Sample batch
        indices = np.random.choice(len(all_experiences), batch_size, replace=False)
        batch_experiences = [all_experiences[i] for i in indices]
        
        # Organize batch by components
        batch = {
            'observations': torch.stack([torch.FloatTensor(exp['observation']) for exp in batch_experiences]),
            'actions': torch.LongTensor([exp['action'] for exp in batch_experiences]),
            'rewards': torch.FloatTensor([exp['reward'] for exp in batch_experiences]),
            'log_probs': torch.FloatTensor([exp['log_prob'] for exp in batch_experiences]),
            'values': torch.FloatTensor([exp['value'] for exp in batch_experiences]),
            'dones': torch.BoolTensor([exp['done'] for exp in batch_experiences]),
            'agents': [exp['agent'] for exp in batch_experiences]
        }
        
        return batch
    
    def clear_old_experiences(self, max_size: int):
        """Clear old experiences to manage memory"""
        if self.total_experiences > max_size:
            # Remove oldest experiences from each agent
            for agent in self.experiences:
                if len(self.experiences[agent]) > max_size // len(self.experiences):
                    remove_count = len(self.experiences[agent]) - max_size // len(self.experiences)
                    self.experiences[agent] = self.experiences[agent][remove_count:]
            
            # Recalculate total
            self.total_experiences = sum(len(exp) for exp in self.experiences.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        return {
            'total_experiences': self.total_experiences,
            'episode_count': self.episode_count,
            'agents': list(self.experiences.keys()),
            'experiences_per_agent': {agent: len(exp) for agent, exp in self.experiences.items()},
            'avg_aggregation_time': np.mean(self.aggregation_times) if self.aggregation_times else 0,
            'avg_experience_rate': np.mean(self.experience_rates) if self.experience_rates else 0
        }


class WorkerManager:
    """Manages parallel workers and their coordination"""
    
    def __init__(self, config: ParallelTrainingConfig, env_factory: Callable, 
                 trainer_config: TrainingConfig):
        self.config = config
        self.env_factory = env_factory
        self.trainer_config = trainer_config
        
        # Workers and processes
        self.workers = {}
        self.processes = {}
        self.worker_queues = {}
        self.status_pipes = {}
        
        # Coordination
        self.result_queue = Queue()
        self.shutdown_event = Event()
        
        # Experience aggregation
        self.experience_aggregator = ExperienceAggregator(config)
        
        # Load balancing
        self.worker_loads = defaultdict(float)
        self.last_load_balance = time.time()
        
        # Fault tolerance
        self.worker_failures = defaultdict(int)
        self.failed_workers = set()
        
        # Performance monitoring
        self.worker_stats = defaultdict(dict)
        self.monitoring_thread = None
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize all worker processes"""
        for worker_id in range(self.config.num_workers):
            self._create_worker(worker_id)
        
        # Start monitoring thread
        if self.config.enable_worker_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitor_workers, daemon=True)
            self.monitoring_thread.start()
        
        logger.info(f"Initialized {len(self.workers)} parallel workers")
    
    def _create_worker(self, worker_id: int):
        """Create individual worker process"""
        try:
            # Create communication channels
            command_queue = Queue()
            status_pipe_parent, status_pipe_child = Pipe()
            
            # Create worker process
            process = Process(
                target=worker_process_main,
                args=(
                    worker_id, self.config, self.env_factory, self.trainer_config,
                    self.result_queue, command_queue, status_pipe_child, self.shutdown_event
                ),
                daemon=True
            )
            
            # Start process
            process.start()
            
            # Store references
            self.processes[worker_id] = process
            self.worker_queues[worker_id] = command_queue
            self.status_pipes[worker_id] = status_pipe_parent
            
            logger.info(f"Worker {worker_id} process started (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to create worker {worker_id}: {e}")
            self.worker_failures[worker_id] += 1
    
    def _monitor_workers(self):
        """Monitor worker processes"""
        while not self.shutdown_event.is_set():
            try:
                # Check worker status
                for worker_id, pipe in self.status_pipes.items():
                    if pipe.poll():
                        status = pipe.recv()
                        self.worker_stats[worker_id] = status
                        
                        # Handle worker failures
                        if status['status'] == 'failed':
                            self._handle_worker_failure(worker_id)
                
                # Check process health
                for worker_id, process in self.processes.items():
                    if not process.is_alive():
                        logger.warning(f"Worker {worker_id} process died")
                        self._handle_worker_failure(worker_id)
                
                # Load balancing
                if (self.config.dynamic_load_balancing and 
                    time.time() - self.last_load_balance > self.config.load_balance_frequency):
                    self._balance_worker_loads()
                
                time.sleep(self.config.monitoring_frequency)
                
            except Exception as e:
                logger.error(f"Worker monitoring error: {e}")
    
    def _handle_worker_failure(self, worker_id: int):
        """Handle worker process failure"""
        self.worker_failures[worker_id] += 1
        
        if self.worker_failures[worker_id] > self.config.max_worker_failures:
            logger.error(f"Worker {worker_id} exceeded max failures, marking as failed")
            self.failed_workers.add(worker_id)
            return
        
        if self.config.enable_fault_tolerance:
            logger.info(f"Restarting failed worker {worker_id}")
            
            # Cleanup old process
            if worker_id in self.processes:
                process = self.processes[worker_id]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            
            # Wait before restart
            time.sleep(self.config.worker_restart_delay)
            
            # Recreate worker
            self._create_worker(worker_id)
    
    def _balance_worker_loads(self):
        """Balance loads across workers"""
        if len(self.worker_stats) < 2:
            return
        
        # Calculate load metrics
        loads = {}
        for worker_id, stats in self.worker_stats.items():
            if 'performance_metrics' in stats:
                metrics = stats['performance_metrics']
                # Simple load metric based on episodes/second
                load = 1.0 / (metrics.get('episodes_per_second', 0.1) + 0.1)
                loads[worker_id] = load
        
        if not loads:
            return
        
        # Check if rebalancing is needed
        max_load = max(loads.values())
        min_load = min(loads.values())
        
        if max_load > min_load * (1 + self.config.max_worker_imbalance):
            logger.info(f"Rebalancing worker loads: max={max_load:.2f}, min={min_load:.2f}")
            # Implementation would redistribute work here
        
        self.last_load_balance = time.time()
    
    def submit_work(self, episodes_per_worker: int):
        """Submit work to all workers"""
        active_workers = [w for w in range(self.config.num_workers) if w not in self.failed_workers]
        
        for worker_id in active_workers:
            if worker_id in self.worker_queues:
                command = {
                    'type': 'run_episodes',
                    'num_episodes': episodes_per_worker
                }
                self.worker_queues[worker_id].put(command)
    
    def collect_results(self, timeout: float = None) -> List[Dict[str, Any]]:
        """Collect results from workers"""
        results = []
        start_time = time.time()
        
        while True:
            try:
                # Check timeout
                if timeout and time.time() - start_time > timeout:
                    logger.warning(f"Result collection timeout after {timeout}s")
                    break
                
                # Get result with timeout
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
                
                # Add to experience aggregator
                self.experience_aggregator.add_episode_result(result)
                
            except queue.Empty:
                # Check if all workers have completed
                if self._all_workers_idle():
                    break
                continue
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
                break
        
        return results
    
    def _all_workers_idle(self) -> bool:
        """Check if all workers are idle"""
        for worker_id, stats in self.worker_stats.items():
            if stats.get('status') == 'running':
                return False
        return True
    
    def get_experience_batch(self, batch_size: int) -> Optional[Dict[str, Any]]:
        """Get batch of experiences for training"""
        return self.experience_aggregator.get_batch(batch_size)
    
    def cleanup_experiences(self):
        """Clean up old experiences"""
        self.experience_aggregator.clear_old_experiences(self.config.experience_buffer_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            'num_workers': self.config.num_workers,
            'active_workers': len([w for w in range(self.config.num_workers) if w not in self.failed_workers]),
            'failed_workers': len(self.failed_workers),
            'worker_failures': dict(self.worker_failures),
            'worker_stats': dict(self.worker_stats),
            'experience_stats': self.experience_aggregator.get_statistics()
        }
    
    def shutdown(self):
        """Shutdown all workers"""
        logger.info("Shutting down parallel workers...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send shutdown commands
        for worker_id, command_queue in self.worker_queues.items():
            try:
                command_queue.put({'type': 'shutdown'})
            except:
                pass
        
        # Wait for processes to finish
        for worker_id, process in self.processes.items():
            if process.is_alive():
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning(f"Force killing worker {worker_id}")
                    process.kill()
        
        # Stop monitoring
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("All workers shut down")


class PettingZooParallelTrainer:
    """
    Parallel trainer for PettingZoo environments
    """
    
    def __init__(self, config: ParallelTrainingConfig, env_factory: Callable,
                 trainer_config: TrainingConfig):
        self.config = config
        self.env_factory = env_factory
        self.trainer_config = trainer_config
        
        # Initialize worker manager
        self.worker_manager = WorkerManager(config, env_factory, trainer_config)
        
        # Initialize main trainer for parameter updates
        self.main_trainer = PettingZooMAPPOTrainer(trainer_config)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_time = 0.0
        
        # Performance tracking
        self.performance_history = []
        self.throughput_history = []
        
        logger.info("PettingZoo parallel trainer initialized")
    
    def train(self, total_episodes: int) -> Dict[str, Any]:
        """Run parallel training"""
        logger.info(f"Starting parallel training with {self.config.num_workers} workers")
        
        training_start_time = time.time()
        
        try:
            episodes_per_batch = self.config.episodes_per_worker * self.config.num_workers
            num_batches = total_episodes // episodes_per_batch
            
            for batch in range(num_batches):
                batch_start_time = time.time()
                
                # Submit work to workers
                self.worker_manager.submit_work(self.config.episodes_per_worker)
                
                # Collect results
                results = self.worker_manager.collect_results(timeout=self.config.worker_timeout)
                
                # Process results
                batch_performance = self._process_batch_results(results)
                
                # Update main trainer
                self._update_main_trainer()
                
                # Update performance tracking
                batch_time = time.time() - batch_start_time
                throughput = len(results) / batch_time
                
                self.performance_history.append(batch_performance)
                self.throughput_history.append(throughput)
                
                # Log progress
                if batch % 10 == 0:
                    self._log_training_progress(batch, batch_performance, throughput)
                
                # Cleanup old experiences
                self.worker_manager.cleanup_experiences()
                
                self.episode_count += len(results)
            
            self.training_time = time.time() - training_start_time
            
            # Get final results
            final_results = self._get_final_results()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Parallel training failed: {e}")
            raise
        finally:
            self.worker_manager.shutdown()
    
    def _process_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch of results"""
        if not results:
            return {'mean_reward': 0.0, 'mean_length': 0.0}
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'mean_reward': 0.0, 'mean_length': 0.0}
        
        # Calculate performance metrics
        rewards = [r['episode_reward'] for r in valid_results]
        lengths = [r['episode_length'] for r in valid_results]
        
        performance = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'num_episodes': len(valid_results),
            'num_errors': len(results) - len(valid_results)
        }
        
        return performance
    
    def _update_main_trainer(self):
        """Update main trainer with aggregated experiences"""
        # Get batch of experiences
        batch = self.worker_manager.get_experience_batch(self.config.batch_size)
        
        if batch is None:
            return
        
        # Update main trainer networks
        self.main_trainer._update_networks()
        
        # Sync parameters to workers (would need additional implementation)
        # This would synchronize the updated parameters back to workers
    
    def _log_training_progress(self, batch: int, performance: Dict[str, Any], throughput: float):
        """Log training progress"""
        logger.info(f"Batch {batch}: "
                   f"Reward: {performance['mean_reward']:.3f} ± {performance['std_reward']:.3f}, "
                   f"Length: {performance['mean_length']:.1f}, "
                   f"Throughput: {throughput:.1f} episodes/sec")
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Get final training results"""
        # Calculate final statistics
        final_performance = self.performance_history[-1] if self.performance_history else {}
        avg_throughput = np.mean(self.throughput_history) if self.throughput_history else 0
        
        # Get worker statistics
        worker_stats = self.worker_manager.get_statistics()
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'training_time': self.training_time,
            'final_performance': final_performance,
            'avg_throughput': avg_throughput,
            'performance_history': self.performance_history,
            'worker_statistics': worker_stats,
            'config': self.config
        }


def create_parallel_config(**kwargs) -> ParallelTrainingConfig:
    """Create parallel training configuration"""
    return ParallelTrainingConfig(**kwargs)


def create_parallel_trainer(config: ParallelTrainingConfig, env_factory: Callable,
                          trainer_config: TrainingConfig) -> PettingZooParallelTrainer:
    """Create parallel trainer"""
    return PettingZooParallelTrainer(config, env_factory, trainer_config)


# Example usage
if __name__ == "__main__":
    # Create configuration
    parallel_config = create_parallel_config(
        num_workers=4,
        episodes_per_worker=100,
        use_multiprocessing=True,
        enable_fault_tolerance=True
    )
    
    # Create environment factory
    def create_env():
        from .pettingzoo_environment_manager import create_tactical_environment
        return create_tactical_environment()
    
    # Create trainer config
    trainer_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=64,
        num_episodes=1000
    )
    
    # Create parallel trainer
    trainer = create_parallel_trainer(parallel_config, create_env, trainer_config)
    
    # Run training
    results = trainer.train(total_episodes=1000)
    
    print(f"Parallel training completed!")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Average throughput: {results['avg_throughput']:.2f} episodes/sec")