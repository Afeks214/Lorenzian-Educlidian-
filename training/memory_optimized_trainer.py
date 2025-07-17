"""
Memory-Optimized MARL Training System
Implements advanced memory optimization techniques for efficient MARL training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import gc
import psutil
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass
from collections import deque, defaultdict
import threading
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization"""
    # Memory limits
    max_memory_usage_gb: float = 8.0
    memory_warning_threshold: float = 0.8  # 80% of max memory
    memory_critical_threshold: float = 0.95  # 95% of max memory
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    checkpoint_segments: int = 4
    
    # Memory pooling
    use_memory_pooling: bool = True
    pool_size_mb: int = 1024
    
    # Tensor optimization
    use_tensor_optimization: bool = True
    use_inplace_operations: bool = True
    
    # Garbage collection
    gc_frequency: int = 10  # Every N episodes
    aggressive_gc: bool = False
    
    # Experience replay optimization
    use_circular_buffer: bool = True
    buffer_compression: bool = True
    
    # Model optimization
    use_model_sharding: bool = False
    use_mixed_precision: bool = True
    
    # Monitoring
    enable_memory_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds


class MemoryMonitor:
    """Real-time memory monitoring and management"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.max_memory_bytes = config.max_memory_usage_gb * 1024 * 1024 * 1024
        self.warning_threshold = self.max_memory_bytes * config.memory_warning_threshold
        self.critical_threshold = self.max_memory_bytes * config.memory_critical_threshold
        
        # Monitoring state
        self.memory_history = deque(maxlen=1000)
        self.alerts_sent = set()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Statistics
        self.peak_memory_usage = 0
        self.oom_events = 0
        self.gc_triggered = 0
        
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_memory_status()
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _check_memory_status(self):
        """Check current memory status"""
        # System memory
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        # GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
        
        total_memory = process_memory + gpu_memory
        
        # Update statistics
        self.peak_memory_usage = max(self.peak_memory_usage, total_memory)
        self.memory_history.append({
            'timestamp': time.time(),
            'system_memory': memory_info.used,
            'process_memory': process_memory,
            'gpu_memory': gpu_memory,
            'total_memory': total_memory
        })
        
        # Check thresholds
        if total_memory > self.critical_threshold:
            self._handle_critical_memory()
        elif total_memory > self.warning_threshold:
            self._handle_warning_memory()
    
    def _handle_warning_memory(self):
        """Handle memory warning"""
        if 'warning' not in self.alerts_sent:
            logger.warning(f"Memory usage warning: {self.get_memory_usage_gb():.2f} GB")
            self.alerts_sent.add('warning')
            
            # Trigger mild cleanup
            self._trigger_cleanup(aggressive=False)
    
    def _handle_critical_memory(self):
        """Handle critical memory situation"""
        if 'critical' not in self.alerts_sent:
            logger.critical(f"Critical memory usage: {self.get_memory_usage_gb():.2f} GB")
            self.alerts_sent.add('critical')
            
            # Trigger aggressive cleanup
            self._trigger_cleanup(aggressive=True)
    
    def _trigger_cleanup(self, aggressive: bool = False):
        """Trigger memory cleanup"""
        self.gc_triggered += 1
        
        # Python garbage collection
        if aggressive:
            # Multiple passes for aggressive cleanup
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()
        
        # PyTorch memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
        
        # Clear alerts after cleanup
        self.alerts_sent.clear()
    
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
        
        return (process_memory + gpu_memory) / (1024 * 1024 * 1024)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        stats = {
            'current_usage_gb': self.get_memory_usage_gb(),
            'peak_usage_gb': self.peak_memory_usage / (1024 * 1024 * 1024),
            'system_memory_percent': system_memory.percent,
            'process_memory_percent': process.memory_percent(),
            'oom_events': self.oom_events,
            'gc_triggered': self.gc_triggered
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 * 1024 * 1024),
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
            })
        
        return stats


class MemoryPool:
    """Memory pool for efficient tensor allocation"""
    
    def __init__(self, pool_size_mb: int = 1024):
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.pool = {}
        self.allocated_size = 0
        self.hit_count = 0
        self.miss_count = 0
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Get tensor from pool or allocate new one"""
        key = (shape, dtype, device)
        
        if key in self.pool and self.pool[key]:
            # Pool hit
            tensor = self.pool[key].pop()
            self.hit_count += 1
            return tensor
        else:
            # Pool miss - allocate new tensor
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self.miss_count += 1
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        key = (tensor.shape, tensor.dtype, tensor.device)
        
        if key not in self.pool:
            self.pool[key] = []
        
        # Check pool size limit
        tensor_size = tensor.numel() * tensor.element_size()
        if self.allocated_size + tensor_size <= self.pool_size_bytes:
            # Clear tensor data and add to pool
            tensor.zero_()
            self.pool[key].append(tensor)
            self.allocated_size += tensor_size
        # Otherwise, let tensor be garbage collected
    
    def clear_pool(self):
        """Clear the entire pool"""
        self.pool.clear()
        self.allocated_size = 0
        gc.collect()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'pool_size_mb': self.pool_size_bytes / (1024 * 1024),
            'allocated_size_mb': self.allocated_size / (1024 * 1024),
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0,
            'total_tensors': sum(len(tensors) for tensors in self.pool.values())
        }


class CircularExperienceBuffer:
    """Memory-efficient circular experience buffer"""
    
    def __init__(self, capacity: int, state_dim: int, use_compression: bool = True):
        self.capacity = capacity
        self.state_dim = state_dim
        self.use_compression = use_compression
        self.size = 0
        self.ptr = 0
        
        # Use memory-mapped arrays for large buffers
        self.states = np.memmap(
            'temp_states.dat',
            dtype=np.float32,
            mode='w+',
            shape=(capacity, state_dim)
        )
        
        self.actions = np.memmap(
            'temp_actions.dat',
            dtype=np.int32,
            mode='w+',
            shape=(capacity,)
        )
        
        self.rewards = np.memmap(
            'temp_rewards.dat',
            dtype=np.float32,
            mode='w+',
            shape=(capacity,)
        )
        
        self.dones = np.memmap(
            'temp_dones.dat',
            dtype=np.bool_,
            mode='w+',
            shape=(capacity,)
        )
        
        # Compression state
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0
        }
    
    def add(self, state: np.ndarray, action: int, reward: float, done: bool):
        """Add experience to buffer"""
        # Compress state if enabled
        if self.use_compression:
            state = self._compress_state(state)
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer"""
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch size {batch_size}")
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices]
        }
        
        # Decompress states if needed
        if self.use_compression:
            batch['states'] = self._decompress_states(batch['states'])
        
        return batch
    
    def _compress_state(self, state: np.ndarray) -> np.ndarray:
        """Compress state using quantization"""
        # Simple quantization compression
        compressed = np.round(state * 127).astype(np.int8)
        return compressed.view(np.float32)
    
    def _decompress_states(self, states: np.ndarray) -> np.ndarray:
        """Decompress states"""
        # Convert back from quantized format
        int8_states = states.view(np.int8)
        return int8_states.astype(np.float32) / 127.0
    
    def clear(self):
        """Clear buffer"""
        self.size = 0
        self.ptr = 0
        # Memory maps are automatically cleared
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get buffer memory usage"""
        state_size = self.states.nbytes
        action_size = self.actions.nbytes
        reward_size = self.rewards.nbytes
        done_size = self.dones.nbytes
        
        total_size = state_size + action_size + reward_size + done_size
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'state_size_mb': state_size / (1024 * 1024),
            'action_size_mb': action_size / (1024 * 1024),
            'reward_size_mb': reward_size / (1024 * 1024),
            'done_size_mb': done_size / (1024 * 1024),
            'compression_ratio': self.compression_stats['compression_ratio']
        }


class GradientCheckpointWrapper(nn.Module):
    """Wrapper for gradient checkpointing"""
    
    def __init__(self, module: nn.Module, segments: int = 4):
        super().__init__()
        self.module = module
        self.segments = segments
        
        # Split module into segments
        self.segment_modules = self._create_segments()
    
    def _create_segments(self) -> nn.ModuleList:
        """Create segments for gradient checkpointing"""
        if hasattr(self.module, 'layers') or hasattr(self.module, 'blocks'):
            # For transformer-like models
            layers = getattr(self.module, 'layers', getattr(self.module, 'blocks', []))
            segment_size = max(1, len(layers) // self.segments)
            
            segments = nn.ModuleList()
            for i in range(0, len(layers), segment_size):
                segment = nn.Sequential(*layers[i:i+segment_size])
                segments.append(segment)
            
            return segments
        else:
            # For sequential models
            if isinstance(self.module, nn.Sequential):
                layers = list(self.module.children())
                segment_size = max(1, len(layers) // self.segments)
                
                segments = nn.ModuleList()
                for i in range(0, len(layers), segment_size):
                    segment = nn.Sequential(*layers[i:i+segment_size])
                    segments.append(segment)
                
                return segments
            else:
                # Can't segment, return as single segment
                return nn.ModuleList([self.module])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing"""
        for segment in self.segment_modules:
            x = checkpoint.checkpoint(segment, x)
        
        return x


class MemoryOptimizedTrainer:
    """Memory-optimized trainer for MARL"""
    
    def __init__(self, 
                 config: MemoryOptimizationConfig,
                 model_factory: Callable,
                 optimizer_factory: Callable,
                 device: torch.device = torch.device('cpu')):
        self.config = config
        self.device = device
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(config)
        if config.enable_memory_monitoring:
            self.memory_monitor.start_monitoring()
        
        # Initialize memory pool
        if config.use_memory_pooling:
            self.memory_pool = MemoryPool(config.pool_size_mb)
        else:
            self.memory_pool = None
        
        # Initialize model
        self.model = model_factory()
        
        # Apply gradient checkpointing
        if config.use_gradient_checkpointing:
            self.model = GradientCheckpointWrapper(self.model, config.checkpoint_segments)
        
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optimizer_factory(self.model.parameters())
        
        # Mixed precision
        if config.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Experience buffer
        self.experience_buffer = CircularExperienceBuffer(
            capacity=10000,
            state_dim=128,  # Adjust based on your state dimension
            use_compression=config.buffer_compression
        )
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'memory_usage': [],
            'gc_events': 0,
            'oom_events': 0
        }
        
        logger.info(f"Memory-optimized trainer initialized with max memory: {config.max_memory_usage_gb} GB")
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory-efficient operations"""
        try:
            # Pre-operation cleanup
            if self.config.aggressive_gc:
                gc.collect()
            
            yield
            
        finally:
            # Post-operation cleanup
            if self.config.aggressive_gc:
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def train_episode(self, environment) -> Dict[str, Any]:
        """Train for one episode with memory optimization"""
        with self.memory_context():
            episode_reward = 0.0
            episode_length = 0
            episode_start_memory = self.memory_monitor.get_memory_usage_gb()
            
            # Reset environment
            state = environment.reset()
            
            # Episode loop
            while True:
                # Get action (memory-efficient)
                with torch.no_grad():
                    if self.memory_pool:
                        state_tensor = self.memory_pool.get_tensor(
                            (1, len(state)), 
                            dtype=torch.float32, 
                            device=self.device
                        )
                        state_tensor[0] = torch.FloatTensor(state)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # Forward pass with mixed precision
                    with autocast(enabled=self.config.use_mixed_precision):
                        action_logits = self.model(state_tensor)
                        action_dist = torch.distributions.Categorical(logits=action_logits)
                        action = action_dist.sample()
                    
                    # Return tensor to pool
                    if self.memory_pool:
                        self.memory_pool.return_tensor(state_tensor)
                
                # Take action
                next_state, reward, done, info = environment.step(action.item())
                
                # Store experience
                self.experience_buffer.add(
                    state=np.array(state, dtype=np.float32),
                    action=action.item(),
                    reward=reward,
                    done=done
                )
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                state = next_state
            
            # Train on experiences
            if self.experience_buffer.size >= self.config.batch_size:
                loss = self._train_on_batch()
            else:
                loss = 0.0
            
            # Update statistics
            self.training_stats['episodes'] += 1
            episode_end_memory = self.memory_monitor.get_memory_usage_gb()
            
            # Periodic cleanup
            if self.training_stats['episodes'] % self.config.gc_frequency == 0:
                self._periodic_cleanup()
            
            return {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'episode_loss': loss,
                'memory_start': episode_start_memory,
                'memory_end': episode_end_memory,
                'memory_delta': episode_end_memory - episode_start_memory
            }
    
    def _train_on_batch(self) -> float:
        """Train on a batch of experiences"""
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        
        # Training step
        self.optimizer.zero_grad()
        
        with autocast(enabled=self.config.use_mixed_precision):
            # Forward pass
            action_logits = self.model(states)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(actions)
            
            # Policy loss
            loss = -torch.mean(log_probs * rewards)
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup"""
        logger.info(f"Performing periodic cleanup (episode {self.training_stats['episodes']})")
        
        # Clear memory pool
        if self.memory_pool:
            self.memory_pool.clear_pool()
        
        # Aggressive garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update statistics
        self.training_stats['gc_events'] += 1
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        report = {
            'memory_monitor_stats': self.memory_monitor.get_memory_stats(),
            'training_stats': self.training_stats,
            'experience_buffer_usage': self.experience_buffer.get_memory_usage()
        }
        
        if self.memory_pool:
            report['memory_pool_stats'] = self.memory_pool.get_pool_stats()
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up memory-optimized trainer")
        
        # Stop monitoring
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
        
        # Clear buffers
        self.experience_buffer.clear()
        
        # Clear memory pool
        if self.memory_pool:
            self.memory_pool.clear_pool()
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_memory_optimized_config(**kwargs) -> MemoryOptimizationConfig:
    """Create memory optimization configuration"""
    return MemoryOptimizationConfig(**kwargs)


def run_memory_optimized_training(
    model_factory: Callable,
    optimizer_factory: Callable,
    environment_factory: Callable,
    num_episodes: int = 1000,
    device: torch.device = torch.device('cpu'),
    **config_kwargs
) -> Dict[str, Any]:
    """Run memory-optimized training"""
    
    # Create configuration
    config = create_memory_optimized_config(**config_kwargs)
    
    # Initialize trainer
    trainer = MemoryOptimizedTrainer(
        config=config,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        device=device
    )
    
    # Create environment
    environment = environment_factory()
    
    # Training loop
    episode_results = []
    
    try:
        for episode in range(num_episodes):
            # Train episode
            result = trainer.train_episode(environment)
            episode_results.append(result)
            
            # Log progress
            if episode % 10 == 0:
                memory_usage = trainer.memory_monitor.get_memory_usage_gb()
                logger.info(f"Episode {episode}: Reward={result['episode_reward']:.3f}, "
                           f"Memory={memory_usage:.2f}GB")
        
        # Get final memory report
        memory_report = trainer.get_memory_report()
        
        return {
            'episode_results': episode_results,
            'memory_report': memory_report,
            'training_summary': {
                'total_episodes': len(episode_results),
                'mean_reward': np.mean([r['episode_reward'] for r in episode_results]),
                'peak_memory_gb': memory_report['memory_monitor_stats']['peak_usage_gb'],
                'gc_events': memory_report['training_stats']['gc_events']
            }
        }
        
    finally:
        # Clean up
        trainer.cleanup()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
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
    
    # Run memory-optimized training
    results = run_memory_optimized_training(
        model_factory=create_model,
        optimizer_factory=create_optimizer,
        environment_factory=create_environment,
        num_episodes=100,
        max_memory_usage_gb=4.0,
        use_gradient_checkpointing=True,
        use_memory_pooling=True,
        use_mixed_precision=True,
        buffer_compression=True
    )
    
    print("Memory-optimized training completed!")
    print(f"Mean reward: {results['training_summary']['mean_reward']:.3f}")
    print(f"Peak memory: {results['training_summary']['peak_memory_gb']:.2f} GB")
    print(f"GC events: {results['training_summary']['gc_events']}")