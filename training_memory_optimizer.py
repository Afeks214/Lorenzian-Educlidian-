#!/usr/bin/env python3
"""
Training Pipeline Memory Optimization for GrandModel

This module provides comprehensive memory optimization for training pipelines:
1. Dynamic batch size optimization
2. Gradient accumulation strategies
3. Memory-efficient data loading
4. Distributed training memory optimization
5. Checkpoint memory management
6. Model sharding and parallelization

Author: Claude Code Assistant
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import gc
import psutil
import time
import logging
import json
import os
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import math
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class TrainingMemoryConfig:
    """Configuration for training memory optimization"""
    # Batch size optimization
    initial_batch_size: int = 32
    max_batch_size: int = 512
    min_batch_size: int = 8
    batch_size_growth_factor: float = 1.2
    memory_threshold: float = 0.85
    
    # Gradient accumulation
    max_gradient_accumulation: int = 32
    target_effective_batch_size: int = 128
    
    # Memory management
    max_memory_usage_gb: float = 8.0
    memory_buffer_gb: float = 1.0
    enable_memory_monitoring: bool = True
    
    # Mixed precision
    enable_mixed_precision: bool = True
    loss_scale: float = 2.0 ** 16
    
    # Gradient checkpointing
    enable_gradient_checkpointing: bool = True
    checkpoint_segments: int = 4
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Distributed training
    use_distributed: bool = False
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    
    # Checkpoint management
    checkpoint_memory_limit_gb: float = 2.0
    checkpoint_compression: bool = True
    
    # Model optimization
    enable_model_sharding: bool = False
    shard_size_mb: int = 100
    
    # Advanced optimizations
    enable_activation_checkpointing: bool = True
    enable_parameter_offloading: bool = False
    enable_optimizer_state_sharding: bool = False


class MemoryEfficientDataLoader:
    """Memory-efficient data loading with smart batching"""
    
    def __init__(self, dataset: Dataset, config: TrainingMemoryConfig):
        self.dataset = dataset
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.dataloader = None
        self._create_dataloader()
        
        # Memory monitoring
        self.memory_usage_history = deque(maxlen=100)
        self.oom_count = 0
        self.batch_size_history = deque(maxlen=100)
    
    def _create_dataloader(self):
        """Create DataLoader with current configuration"""
        sampler = None
        if self.config.use_distributed:
            sampler = DistributedSampler(self.dataset)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.current_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True  # For consistent batch sizes
        )
    
    def adjust_batch_size(self, memory_usage: float, success: bool) -> None:
        """Dynamically adjust batch size based on memory usage"""
        if not success:  # OOM occurred
            self.oom_count += 1
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)  # Reduce by 20%
            )
            logger.warning(f"OOM detected, reducing batch size from {self.current_batch_size} to {new_batch_size}")
        
        elif memory_usage < self.config.memory_threshold * 0.8:  # Low memory usage
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.batch_size_growth_factor)
            )
            if new_batch_size > self.current_batch_size:
                logger.info(f"Low memory usage, increasing batch size from {self.current_batch_size} to {new_batch_size}")
        
        else:
            new_batch_size = self.current_batch_size
        
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self.batch_size_history.append({
                'timestamp': time.time(),
                'batch_size': new_batch_size,
                'memory_usage': memory_usage,
                'reason': 'oom' if not success else 'optimization'
            })
            self._create_dataloader()
    
    def get_batch_size_stats(self) -> Dict[str, Any]:
        """Get batch size optimization statistics"""
        if not self.batch_size_history:
            return {}
        
        sizes = [entry['batch_size'] for entry in self.batch_size_history]
        
        return {
            'current_batch_size': self.current_batch_size,
            'min_batch_size': min(sizes),
            'max_batch_size': max(sizes),
            'avg_batch_size': np.mean(sizes),
            'oom_count': self.oom_count,
            'adjustments': len(self.batch_size_history)
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class GradientAccumulationManager:
    """Manage gradient accumulation for memory efficiency"""
    
    def __init__(self, config: TrainingMemoryConfig):
        self.config = config
        self.accumulation_steps = self._calculate_accumulation_steps()
        self.current_step = 0
        self.accumulated_loss = 0.0
        self.gradient_norms = []
        
    def _calculate_accumulation_steps(self) -> int:
        """Calculate optimal gradient accumulation steps"""
        target_batch_size = self.config.target_effective_batch_size
        actual_batch_size = self.config.initial_batch_size
        
        accumulation_steps = max(1, target_batch_size // actual_batch_size)
        return min(accumulation_steps, self.config.max_gradient_accumulation)
    
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated"""
        return self.current_step % self.accumulation_steps != 0
    
    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Accumulate loss for gradient accumulation"""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        self.accumulated_loss += scaled_loss.item()
        return scaled_loss
    
    def step(self, optimizer: optim.Optimizer, scaler: Optional[GradScaler] = None) -> bool:
        """Perform optimizer step if accumulation is complete"""
        self.current_step += 1
        
        if self.should_accumulate():
            return False
        
        # Calculate gradient norm before stepping
        total_norm = 0
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Perform optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Reset accumulated loss
        self.accumulated_loss = 0.0
        
        return True
    
    def get_effective_batch_size(self, batch_size: int) -> int:
        """Get effective batch size with accumulation"""
        return batch_size * self.accumulation_steps
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gradient accumulation statistics"""
        return {
            'accumulation_steps': self.accumulation_steps,
            'current_step': self.current_step,
            'accumulated_loss': self.accumulated_loss,
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'gradient_norm_std': np.std(self.gradient_norms) if self.gradient_norms else 0
        }


class MemoryOptimizedCheckpointManager:
    """Memory-efficient checkpoint management"""
    
    def __init__(self, config: TrainingMemoryConfig, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint memory tracking
        self.checkpoint_memory_usage = deque(maxlen=50)
        self.compression_stats = {
            'total_saved': 0,
            'total_compressed': 0,
            'compression_ratio': 0.0
        }
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        **kwargs
    ) -> str:
        """Save checkpoint with memory optimization"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        # Monitor memory usage during save
        initial_memory = self._get_memory_usage()
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **kwargs
        }
        
        # Compress if enabled
        if self.config.checkpoint_compression:
            checkpoint_data = self._compress_checkpoint(checkpoint_data)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track memory usage
        final_memory = self._get_memory_usage()
        self.checkpoint_memory_usage.append({
            'timestamp': time.time(),
            'memory_used_gb': final_memory - initial_memory,
            'file_size_mb': checkpoint_path.stat().st_size / (1024 ** 2)
        })
        
        # Clean up old checkpoints if memory limit exceeded
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint with memory optimization"""
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Decompress if needed
        if self.config.checkpoint_compression and 'compressed' in checkpoint_data:
            checkpoint_data = self._decompress_checkpoint(checkpoint_data)
        
        return checkpoint_data
    
    def _compress_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress checkpoint data"""
        try:
            import pickle
            import gzip
            
            # Serialize and compress
            serialized = pickle.dumps(checkpoint_data)
            compressed = gzip.compress(serialized)
            
            # Update compression stats
            self.compression_stats['total_saved'] += len(serialized)
            self.compression_stats['total_compressed'] += len(compressed)
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['total_compressed'] / 
                max(self.compression_stats['total_saved'], 1)
            )
            
            return {
                'compressed': True,
                'data': compressed,
                'original_size': len(serialized),
                'compressed_size': len(compressed)
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return checkpoint_data
    
    def _decompress_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress checkpoint data"""
        try:
            import pickle
            import gzip
            
            compressed = checkpoint_data['data']
            decompressed = gzip.decompress(compressed)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return checkpoint_data
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save memory/disk space"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoint_files:
            return
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Calculate total size
        total_size_gb = sum(f.stat().st_size for f in checkpoint_files) / (1024 ** 3)
        
        # Remove old checkpoints if over limit
        while total_size_gb > self.config.checkpoint_memory_limit_gb and len(checkpoint_files) > 1:
            oldest_file = checkpoint_files.pop()
            total_size_gb -= oldest_file.stat().st_size / (1024 ** 3)
            oldest_file.unlink()
            logger.info(f"Removed old checkpoint: {oldest_file}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        return {
            'total_checkpoints': len(list(self.checkpoint_dir.glob("checkpoint_*.pt"))),
            'compression_stats': self.compression_stats,
            'memory_usage_history': list(self.checkpoint_memory_usage)
        }


class ModelShardingManager:
    """Manage model sharding for memory efficiency"""
    
    def __init__(self, config: TrainingMemoryConfig):
        self.config = config
        self.shards = {}
        self.shard_metadata = {}
        
    def shard_model(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Shard model parameters across multiple devices/memory segments"""
        if not self.config.enable_model_sharding:
            return {'full_model': model}
        
        shards = {}
        current_shard = 0
        current_shard_size = 0
        shard_limit = self.config.shard_size_mb * 1024 * 1024  # Convert to bytes
        
        for name, param in model.named_parameters():
            param_size = param.numel() * param.element_size()
            
            if current_shard_size + param_size > shard_limit:
                current_shard += 1
                current_shard_size = 0
            
            shard_name = f"shard_{current_shard}"
            if shard_name not in shards:
                shards[shard_name] = {}
            
            shards[shard_name][name] = param
            current_shard_size += param_size
        
        # Store metadata
        self.shard_metadata = {
            'total_shards': len(shards),
            'shard_sizes': {name: sum(p.numel() * p.element_size() for p in shard.values()) 
                          for name, shard in shards.items()}
        }
        
        logger.info(f"Model sharded into {len(shards)} shards")
        return shards
    
    def get_shard_stats(self) -> Dict[str, Any]:
        """Get sharding statistics"""
        return self.shard_metadata


class DistributedMemoryOptimizer:
    """Optimize memory usage in distributed training"""
    
    def __init__(self, config: TrainingMemoryConfig):
        self.config = config
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        if config.use_distributed:
            self._init_distributed()
    
    def _init_distributed(self):
        """Initialize distributed training"""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training"""
        if not self.config.use_distributed:
            return model
        
        # Move model to device
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb
        )
        
        return model
    
    def synchronize_memory_usage(self) -> Dict[str, float]:
        """Synchronize memory usage across all processes"""
        if not self.config.use_distributed:
            return {'local_memory_gb': self._get_memory_usage()}
        
        local_memory = self._get_memory_usage()
        memory_tensor = torch.tensor(local_memory, device=torch.cuda.current_device())
        
        # Gather memory usage from all processes
        gathered_memory = [torch.zeros_like(memory_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_memory, memory_tensor)
        
        memory_stats = {
            'local_memory_gb': local_memory,
            'total_memory_gb': sum(t.item() for t in gathered_memory),
            'avg_memory_gb': sum(t.item() for t in gathered_memory) / self.world_size,
            'max_memory_gb': max(t.item() for t in gathered_memory),
            'min_memory_gb': min(t.item() for t in gathered_memory)
        }
        
        return memory_stats
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.config.use_distributed and dist.is_initialized():
            dist.destroy_process_group()


class TrainingMemoryOptimizer:
    """Comprehensive training memory optimization system"""
    
    def __init__(self, config: Optional[TrainingMemoryConfig] = None):
        self.config = config or TrainingMemoryConfig()
        
        # Initialize components
        self.checkpoint_manager = MemoryOptimizedCheckpointManager(self.config)
        self.sharding_manager = ModelShardingManager(self.config)
        self.distributed_optimizer = DistributedMemoryOptimizer(self.config)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.memory_usage_history = deque(maxlen=1000)
        self.optimization_events = []
        
        # Performance metrics
        self.training_metrics = {
            'memory_efficiency': [],
            'throughput': [],
            'batch_size_history': [],
            'oom_events': 0
        }
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.enable_mixed_precision else None
        
        logger.info("Training memory optimizer initialized")
    
    def optimize_training_loop(
        self,
        model: nn.Module,
        dataset: Dataset,
        optimizer: optim.Optimizer,
        num_epochs: int,
        loss_fn: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Run optimized training loop"""
        
        # Prepare model and data
        model = self._prepare_model(model)
        dataloader = MemoryEfficientDataLoader(dataset, self.config)
        grad_accumulator = GradientAccumulationManager(self.config)
        
        # Training loop
        training_start = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(
                model, dataloader, optimizer, loss_fn, grad_accumulator, **kwargs
            )
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            # Save checkpoint
            if epoch % 10 == 0:  # Save every 10 epochs
                self.checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, self.current_step, epoch_metrics['avg_loss']
                )
        
        training_time = time.time() - training_start
        
        # Final statistics
        final_stats = {
            'total_training_time': training_time,
            'total_epochs': num_epochs,
            'final_memory_usage': self._get_current_memory_usage(),
            'dataloader_stats': dataloader.get_batch_size_stats(),
            'gradient_accumulation_stats': grad_accumulator.get_stats(),
            'checkpoint_stats': self.checkpoint_manager.get_checkpoint_stats(),
            'distributed_stats': self.distributed_optimizer.synchronize_memory_usage(),
            'training_metrics': self.training_metrics
        }
        
        return final_stats
    
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for optimized training"""
        # Apply gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        # Apply model sharding
        if self.config.enable_model_sharding:
            self.sharding_manager.shard_model(model)
        
        # Wrap for distributed training
        model = self.distributed_optimizer.wrap_model(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model"""
        def checkpoint_wrapper(module):
            def forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
            return forward
        
        # Apply to selected layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.forward = checkpoint_wrapper(module.forward)
        
        return model
    
    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: MemoryEfficientDataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        grad_accumulator: GradientAccumulationManager,
        **kwargs
    ) -> Dict[str, Any]:
        """Train one epoch with memory optimization"""
        
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            self.current_step += 1
            
            # Monitor memory before batch
            memory_before = self._get_current_memory_usage()
            
            try:
                # Training step
                loss = self._training_step(
                    model, batch, optimizer, loss_fn, grad_accumulator
                )
                
                epoch_losses.append(loss)
                
                # Monitor memory after batch
                memory_after = self._get_current_memory_usage()
                memory_usage = memory_after / self.config.max_memory_usage_gb
                
                # Adjust batch size based on memory usage
                dataloader.adjust_batch_size(memory_usage, success=True)
                
                # Record metrics
                self._record_training_metrics(batch_idx, loss, memory_usage)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at step {self.current_step}: {e}")
                    self.training_metrics['oom_events'] += 1
                    
                    # Emergency cleanup
                    self._emergency_memory_cleanup()
                    
                    # Adjust batch size
                    dataloader.adjust_batch_size(1.0, success=False)
                    
                    # Skip this batch
                    continue
                else:
                    raise
        
        epoch_time = time.time() - epoch_start
        
        return {
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'epoch_time': epoch_time,
            'batches_processed': len(epoch_losses),
            'memory_usage': self._get_current_memory_usage()
        }
    
    def _training_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        grad_accumulator: GradientAccumulationManager
    ) -> float:
        """Single training step with memory optimization"""
        
        # Move batch to device
        device = next(model.parameters()).device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
        
        # Forward pass with mixed precision
        if self.config.enable_mixed_precision:
            with autocast():
                output = model(batch)
                loss = loss_fn(output, batch)  # Assuming self-supervised
        else:
            output = model(batch)
            loss = loss_fn(output, batch)
        
        # Backward pass with gradient accumulation
        scaled_loss = grad_accumulator.accumulate_loss(loss)
        
        if self.config.enable_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Optimizer step if accumulation is complete
        if grad_accumulator.step(optimizer, self.scaler):
            # Step completed, can record metrics
            pass
        
        return loss.item()
    
    def _record_training_metrics(self, batch_idx: int, loss: float, memory_usage: float):
        """Record training metrics"""
        self.memory_usage_history.append({
            'step': self.current_step,
            'memory_usage': memory_usage,
            'loss': loss,
            'timestamp': time.time()
        })
        
        # Calculate efficiency metrics
        if len(self.memory_usage_history) > 1:
            recent_memory = [entry['memory_usage'] for entry in list(self.memory_usage_history)[-10:]]
            memory_efficiency = 1.0 - np.std(recent_memory)  # Lower variance = higher efficiency
            self.training_metrics['memory_efficiency'].append(memory_efficiency)
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        logger.warning("Executing emergency memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log cleanup event
        self.optimization_events.append({
            'type': 'emergency_cleanup',
            'timestamp': time.time(),
            'step': self.current_step
        })
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            'config': asdict(self.config),
            'training_progress': {
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'total_oom_events': self.training_metrics['oom_events']
            },
            'memory_usage': {
                'current_usage_gb': self._get_current_memory_usage(),
                'usage_history': list(self.memory_usage_history)[-100:],  # Last 100 entries
                'peak_usage_gb': max((entry['memory_usage'] for entry in self.memory_usage_history), default=0)
            },
            'component_stats': {
                'checkpoint_manager': self.checkpoint_manager.get_checkpoint_stats(),
                'sharding_manager': self.sharding_manager.get_shard_stats(),
                'distributed': self.distributed_optimizer.synchronize_memory_usage()
            },
            'optimization_events': self.optimization_events,
            'training_metrics': self.training_metrics
        }
    
    def export_report(self, filepath: str):
        """Export optimization report to file"""
        report = self.get_optimization_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report exported to {filepath}")
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        self.distributed_optimizer.cleanup()
        logger.info("Training memory optimizer cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Utility functions
def estimate_memory_requirements(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    mixed_precision: bool = False
) -> Dict[str, float]:
    """Estimate memory requirements for training"""
    
    # Calculate model parameters
    model_params = sum(p.numel() for p in model.parameters())
    param_size = 4 if not mixed_precision else 2  # bytes per parameter
    
    # Model memory
    model_memory = model_params * param_size
    
    # Gradient memory (same as model)
    gradient_memory = model_memory
    
    # Optimizer memory (depends on optimizer, assume Adam)
    optimizer_memory = model_memory * 2  # momentum and variance
    
    # Activation memory (rough estimate)
    activation_memory = batch_size * sequence_length * 1024 * param_size  # rough estimate
    
    # Total memory
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model_memory_gb': model_memory / (1024 ** 3),
        'gradient_memory_gb': gradient_memory / (1024 ** 3),
        'optimizer_memory_gb': optimizer_memory / (1024 ** 3),
        'activation_memory_gb': activation_memory / (1024 ** 3),
        'total_memory_gb': total_memory / (1024 ** 3),
        'recommended_batch_size': min(batch_size, int(8 * (1024 ** 3) / total_memory * batch_size))
    }


def create_memory_efficient_config(**kwargs) -> TrainingMemoryConfig:
    """Create memory-efficient training configuration"""
    return TrainingMemoryConfig(**kwargs)


@contextmanager
def memory_efficient_training():
    """Context manager for memory-efficient training"""
    # Pre-training setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn, 'MultiheadAttention'):
        torch.nn.MultiheadAttention._use_memory_efficient_attention = True
    
    try:
        yield
    finally:
        # Post-training cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    config = create_memory_efficient_config(
        initial_batch_size=32,
        max_batch_size=128,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True
    )
    
    with TrainingMemoryOptimizer(config) as optimizer:
        # Training would happen here
        print("Training memory optimizer ready")
        
        # Generate report
        report = optimizer.get_optimization_report()
        print(f"Current memory usage: {report['memory_usage']['current_usage_gb']:.2f} GB")
