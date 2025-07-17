"""
Distributed Training Coordinator for Large-Scale Model Training
Supports multi-GPU, multi-node, and parameter server architectures
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import os
import time
import json
import socket
import subprocess
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Distributed training backends"""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class DistributedStrategy(Enum):
    """Distributed training strategies"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Basic distributed settings
    backend: DistributedBackend = DistributedBackend.NCCL
    strategy: DistributedStrategy = DistributedStrategy.DATA_PARALLEL
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Network settings
    master_addr: str = "localhost"
    master_port: str = "12355"
    init_method: str = "env://"
    
    # Training optimization
    gradient_compression: bool = True
    all_reduce_bucket_size: int = 25 * 1024 * 1024  # 25MB
    find_unused_parameters: bool = False
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_frequency: int = 100
    health_check_interval: int = 30
    
    # Performance optimization
    use_mixed_precision: bool = True
    static_graph: bool = True
    bucket_cap_mb: int = 25
    
    # Resource management
    cpu_cores_per_process: int = 4
    memory_fraction: float = 0.9
    
    # Monitoring
    enable_profiling: bool = False
    profile_output_dir: str = "distributed_profiles"


class DistributedHealthMonitor:
    """Monitor health of distributed training processes"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.health_status = {}
        self.last_heartbeat = {}
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        """Start health monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_health)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Distributed health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_health(self):
        """Monitor health of all processes"""
        while not self.stop_monitoring.is_set():
            try:
                # Check process health
                current_time = time.time()
                
                for rank in range(self.config.world_size):
                    if rank == self.config.rank:
                        # Update own heartbeat
                        self.last_heartbeat[rank] = current_time
                        self.health_status[rank] = {
                            'status': 'healthy',
                            'memory_usage': psutil.Process().memory_info().rss / 1024**3,
                            'cpu_usage': psutil.Process().cpu_percent(),
                            'timestamp': current_time
                        }
                    else:
                        # Check if other processes are responding
                        if rank in self.last_heartbeat:
                            time_since_heartbeat = current_time - self.last_heartbeat[rank]
                            if time_since_heartbeat > self.config.health_check_interval * 2:
                                self.health_status[rank] = {
                                    'status': 'unhealthy',
                                    'error': 'heartbeat_timeout',
                                    'timestamp': current_time
                                }
                
                # Sleep for health check interval
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'healthy_processes': sum(1 for s in self.health_status.values() if s.get('status') == 'healthy'),
            'total_processes': len(self.health_status),
            'detailed_status': self.health_status.copy()
        }


class GradientCompression:
    """Gradient compression for distributed training"""
    
    def __init__(self, compression_ratio: float = 0.01):
        self.compression_ratio = compression_ratio
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_time': 0,
            'decompression_time': 0
        }
    
    def compress_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compress gradients for efficient communication"""
        start_time = time.time()
        
        compressed_gradients = []
        original_size = 0
        compressed_size = 0
        
        for grad in gradients:
            if grad is None:
                compressed_gradients.append(None)
                continue
            
            original_size += grad.numel() * grad.element_size()
            
            # Top-k sparsification
            k = max(1, int(grad.numel() * self.compression_ratio))
            
            # Flatten and get top-k
            flat_grad = grad.flatten()
            values, indices = torch.topk(torch.abs(flat_grad), k)
            
            # Create compressed representation
            compressed = {
                'indices': indices,
                'values': flat_grad[indices],
                'shape': grad.shape,
                'numel': grad.numel()
            }
            
            compressed_gradients.append(compressed)
            compressed_size += indices.numel() * indices.element_size()
            compressed_size += values.numel() * values.element_size()
        
        self.compression_stats['original_size'] += original_size
        self.compression_stats['compressed_size'] += compressed_size
        self.compression_stats['compression_time'] += time.time() - start_time
        
        return compressed_gradients
    
    def decompress_gradients(self, compressed_gradients: List[Any]) -> List[torch.Tensor]:
        """Decompress gradients"""
        start_time = time.time()
        
        gradients = []
        for compressed in compressed_gradients:
            if compressed is None:
                gradients.append(None)
                continue
            
            # Reconstruct sparse gradient
            grad = torch.zeros(compressed['numel'], device=compressed['values'].device)
            grad[compressed['indices']] = compressed['values']
            grad = grad.reshape(compressed['shape'])
            
            gradients.append(grad)
        
        self.compression_stats['decompression_time'] += time.time() - start_time
        return gradients


class DistributedTrainingCoordinator:
    """
    Coordinator for distributed training across multiple devices/nodes
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: DistributedConfig,
                 device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device(f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distributed process group
        self._init_distributed()
        
        # Wrap model for distributed training
        self.distributed_model = self._wrap_model()
        
        # Initialize components
        self.health_monitor = DistributedHealthMonitor(config)
        self.gradient_compressor = GradientCompression() if config.gradient_compression else None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Performance monitoring
        self.training_stats = {
            'communication_time': [],
            'computation_time': [],
            'total_time': [],
            'gradient_sync_time': [],
            'throughput': []
        }
        
        # Checkpointing
        self.checkpoint_dir = Path("distributed_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Distributed training coordinator initialized on rank {config.rank}/{config.world_size}")
    
    def _init_distributed(self):
        """Initialize distributed process group"""
        if self.config.world_size > 1:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend.value,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                self.device = torch.device(f'cuda:{self.config.local_rank}')
            
            logger.info(f"Distributed process group initialized: rank {self.config.rank}/{self.config.world_size}")
        else:
            logger.info("Single process training mode")
    
    def _wrap_model(self) -> nn.Module:
        """Wrap model for distributed training"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        if self.config.world_size > 1:
            # Wrap with DistributedDataParallel
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb,
                static_graph=self.config.static_graph
            )
        
        return self.model
    
    def create_distributed_sampler(self, dataset) -> DistributedSampler:
        """Create distributed sampler for dataset"""
        if self.config.world_size > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
        return None
    
    def train_step(self, 
                  batch_data: torch.Tensor,
                  loss_fn: Callable,
                  optimizer: torch.optim.Optimizer,
                  targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform one distributed training step
        """
        step_start_time = time.time()
        
        # Move data to device
        batch_data = batch_data.to(self.device, non_blocking=True)
        if targets is not None:
            targets = targets.to(self.device, non_blocking=True)
        
        # Forward pass
        computation_start = time.time()
        
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.model(batch_data)
                loss = loss_fn(output, targets) if targets is not None else loss_fn(output, batch_data)
        else:
            output = self.model(batch_data)
            loss = loss_fn(output, targets) if targets is not None else loss_fn(output, batch_data)
        
        computation_time = time.time() - computation_start
        
        # Backward pass
        optimizer.zero_grad()
        
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient synchronization
        sync_start = time.time()
        
        if self.config.world_size > 1:
            # DDP automatically handles gradient synchronization
            # Additional custom synchronization can be added here
            pass
        
        sync_time = time.time() - sync_start
        
        # Optimizer step
        if self.config.use_mixed_precision:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        # Calculate metrics
        total_time = time.time() - step_start_time
        batch_size = batch_data.size(0)
        throughput = batch_size / total_time
        
        # Store performance metrics
        self.training_stats['computation_time'].append(computation_time)
        self.training_stats['gradient_sync_time'].append(sync_time)
        self.training_stats['total_time'].append(total_time)
        self.training_stats['throughput'].append(throughput)
        
        return {
            'loss': loss.item(),
            'computation_time': computation_time,
            'sync_time': sync_time,
            'total_time': total_time,
            'throughput': throughput,
            'batch_size': batch_size
        }
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metrics across all processes"""
        if self.config.world_size == 1:
            return metrics
        
        reduced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_metrics[key] = (tensor / self.config.world_size).item()
            else:
                reduced_metrics[key] = value
        
        return reduced_metrics
    
    def save_distributed_checkpoint(self, 
                                  model_state: Dict[str, Any],
                                  optimizer_state: Dict[str, Any],
                                  epoch: int,
                                  step: int) -> str:
        """Save distributed checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}_rank_{self.config.rank}.pt"
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'epoch': epoch,
            'step': step,
            'rank': self.config.rank,
            'world_size': self.config.world_size,
            'config': self.config,
            'training_stats': self.training_stats
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Synchronize checkpoint saving across processes
        if self.config.world_size > 1:
            dist.barrier()
        
        if self.config.rank == 0:
            logger.info(f"Distributed checkpoint saved at epoch {epoch}, step {step}")
        
        return str(checkpoint_path)
    
    def load_distributed_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load distributed checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify checkpoint compatibility
        if checkpoint['world_size'] != self.config.world_size:
            logger.warning(f"Checkpoint world_size {checkpoint['world_size']} != current world_size {self.config.world_size}")
        
        return checkpoint
    
    def synchronize_processes(self):
        """Synchronize all processes"""
        if self.config.world_size > 1:
            dist.barrier()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'rank': self.config.rank,
            'world_size': self.config.world_size,
            'device': str(self.device),
            'performance': {
                'avg_computation_time': np.mean(self.training_stats['computation_time']) if self.training_stats['computation_time'] else 0,
                'avg_sync_time': np.mean(self.training_stats['gradient_sync_time']) if self.training_stats['gradient_sync_time'] else 0,
                'avg_total_time': np.mean(self.training_stats['total_time']) if self.training_stats['total_time'] else 0,
                'avg_throughput': np.mean(self.training_stats['throughput']) if self.training_stats['throughput'] else 0
            }
        }
        
        if self.gradient_compressor:
            stats['gradient_compression'] = self.gradient_compressor.compression_stats
        
        return stats
    
    def cleanup(self):
        """Clean up distributed resources"""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        if self.config.world_size > 1:
            dist.destroy_process_group()
        
        logger.info("Distributed training cleanup completed")


def launch_distributed_training(
    training_fn: Callable,
    config: DistributedConfig,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None
):
    """
    Launch distributed training across multiple processes
    """
    kwargs = kwargs or {}
    
    if config.world_size == 1:
        # Single process training
        training_fn(0, *args, **kwargs)
    else:
        # Multi-process training
        mp.spawn(
            training_fn,
            args=(config.world_size,) + args,
            nprocs=config.world_size,
            join=True
        )


def setup_distributed_environment(local_rank: int, world_size: int) -> DistributedConfig:
    """
    Setup distributed environment from environment variables
    """
    config = DistributedConfig()
    
    # Override with environment variables
    config.local_rank = local_rank
    config.rank = int(os.environ.get('RANK', local_rank))
    config.world_size = world_size
    config.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    config.master_port = os.environ.get('MASTER_PORT', '12355')
    
    # Auto-detect backend
    if torch.cuda.is_available():
        config.backend = DistributedBackend.NCCL
    else:
        config.backend = DistributedBackend.GLOO
    
    return config


def estimate_optimal_world_size(
    model_size_mb: float,
    dataset_size_gb: float,
    available_gpus: int,
    target_batch_size: int
) -> int:
    """
    Estimate optimal world size for distributed training
    """
    # Memory requirements per GPU
    memory_per_gpu_gb = model_size_mb / 1024 + 2  # Model + overhead
    
    # Batch size per GPU
    batch_per_gpu = target_batch_size // available_gpus
    
    # Estimate memory per batch
    memory_per_batch_gb = batch_per_gpu * model_size_mb / 1024 / 1000
    
    # Total memory requirement
    total_memory_gb = memory_per_gpu_gb + memory_per_batch_gb
    
    # Assume 8GB per GPU (conservative estimate)
    gpu_memory_gb = 8
    
    # Calculate optimal world size
    optimal_world_size = min(
        available_gpus,
        max(1, int(total_memory_gb / gpu_memory_gb))
    )
    
    return optimal_world_size


# Example usage function
def create_distributed_config_auto(
    model_size_mb: float,
    dataset_size_gb: float,
    target_batch_size: int = 128
) -> DistributedConfig:
    """
    Create optimized distributed configuration automatically
    """
    # Detect available GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Estimate optimal world size
    world_size = estimate_optimal_world_size(
        model_size_mb, dataset_size_gb, available_gpus, target_batch_size
    )
    
    config = DistributedConfig(
        world_size=world_size,
        backend=DistributedBackend.NCCL if torch.cuda.is_available() else DistributedBackend.GLOO,
        gradient_compression=dataset_size_gb > 10,  # Use compression for large datasets
        use_mixed_precision=True,
        static_graph=True,
        enable_fault_tolerance=world_size > 1
    )
    
    return config