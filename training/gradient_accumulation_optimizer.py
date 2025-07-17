"""
Advanced Gradient Accumulation Optimizer for Memory-Efficient Training
Supports dynamic batch sizing, gradient compression, and memory-aware scaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import logging
from collections import deque, defaultdict
import time
import psutil
import gc
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class GradientCompressionMethod(Enum):
    """Gradient compression methods"""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    FEDERATED_AVERAGING = "federated_averaging"


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation"""
    # Basic accumulation settings
    accumulation_steps: int = 8
    effective_batch_size: int = 512
    max_gradient_norm: float = 1.0
    
    # Dynamic batch sizing
    dynamic_batch_sizing: bool = True
    min_batch_size: int = 16
    max_batch_size: int = 128
    batch_size_growth_factor: float = 1.2
    
    # Memory management
    memory_threshold_gb: float = 8.0
    gradient_checkpoint_frequency: int = 4
    clear_cache_frequency: int = 16
    
    # Gradient compression
    compression_method: GradientCompressionMethod = GradientCompressionMethod.NONE
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    
    # Stability and convergence
    gradient_variance_threshold: float = 0.1
    convergence_window: int = 10
    stability_check_frequency: int = 50
    
    # Performance optimization
    async_gradient_reduction: bool = True
    use_mixed_precision: bool = True
    pin_memory: bool = True


class GradientCompressor:
    """Gradient compression utilities"""
    
    def __init__(self, method: GradientCompressionMethod, **kwargs):
        self.method = method
        self.kwargs = kwargs
        
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress gradients to reduce memory usage"""
        if self.method == GradientCompressionMethod.NONE:
            return gradients
        
        compressed = {}
        
        for name, grad in gradients.items():
            if grad is None:
                compressed[name] = None
                continue
                
            if self.method == GradientCompressionMethod.QUANTIZATION:
                compressed[name] = self._quantize_gradient(grad)
            elif self.method == GradientCompressionMethod.SPARSIFICATION:
                compressed[name] = self._sparsify_gradient(grad)
            elif self.method == GradientCompressionMethod.LOW_RANK:
                compressed[name] = self._low_rank_compress(grad)
            else:
                compressed[name] = grad
        
        return compressed
    
    def _quantize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Quantize gradient to reduce precision"""
        bits = self.kwargs.get('quantization_bits', 8)
        
        # Simple uniform quantization
        grad_min = grad.min()
        grad_max = grad.max()
        grad_range = grad_max - grad_min
        
        if grad_range > 0:
            scale = (2 ** bits - 1) / grad_range
            quantized = torch.round((grad - grad_min) * scale)
            dequantized = quantized / scale + grad_min
            return dequantized
        
        return grad
    
    def _sparsify_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Sparsify gradient by keeping only top-k elements"""
        compression_ratio = self.kwargs.get('compression_ratio', 0.1)
        
        # Flatten gradient
        flat_grad = grad.flatten()
        k = int(len(flat_grad) * compression_ratio)
        
        # Find top-k elements
        _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create sparse gradient
        sparse_grad = torch.zeros_like(flat_grad)
        sparse_grad[top_k_indices] = flat_grad[top_k_indices]
        
        return sparse_grad.reshape(grad.shape)
    
    def _low_rank_compress(self, grad: torch.Tensor) -> torch.Tensor:
        """Low-rank compression using SVD"""
        if grad.dim() < 2:
            return grad
        
        compression_ratio = self.kwargs.get('compression_ratio', 0.1)
        
        # Reshape to 2D if needed
        original_shape = grad.shape
        if grad.dim() > 2:
            grad_2d = grad.view(grad.shape[0], -1)
        else:
            grad_2d = grad
        
        # SVD decomposition
        U, S, V = torch.svd(grad_2d)
        
        # Keep top singular values
        rank = int(min(U.shape[1], V.shape[0]) * compression_ratio)
        rank = max(1, rank)
        
        # Reconstruct with reduced rank
        compressed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :]
        
        return compressed.reshape(original_shape)


class DynamicBatchSizer:
    """Dynamic batch size adjustment based on memory usage"""
    
    def __init__(self, config: GradientAccumulationConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.memory_history = deque(maxlen=20)
        self.oom_count = 0
        self.stable_count = 0
        
    def adjust_batch_size(self, memory_usage_gb: float, had_oom: bool = False) -> int:
        """Adjust batch size based on memory usage"""
        self.memory_history.append(memory_usage_gb)
        
        if had_oom:
            self.oom_count += 1
            self.stable_count = 0
            # Reduce batch size aggressively
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.7)
            )
            logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
        
        elif memory_usage_gb > self.config.memory_threshold_gb:
            # Approaching memory limit, reduce batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.9)
            )
            logger.info(f"High memory usage, reducing batch size to {self.current_batch_size}")
        
        elif (memory_usage_gb < self.config.memory_threshold_gb * 0.7 and 
              self.stable_count > 10):
            # Memory usage is low and stable, try to increase batch size
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.batch_size_growth_factor)
            )
            logger.info(f"Low stable memory usage, increasing batch size to {self.current_batch_size}")
            self.stable_count = 0
        
        self.stable_count += 1
        return self.current_batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch sizing statistics"""
        return {
            'current_batch_size': self.current_batch_size,
            'oom_count': self.oom_count,
            'stable_count': self.stable_count,
            'avg_memory_usage': np.mean(self.memory_history) if self.memory_history else 0,
            'max_memory_usage': np.max(self.memory_history) if self.memory_history else 0
        }


class GradientAccumulationOptimizer:
    """
    Advanced gradient accumulation optimizer with memory efficiency
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: GradientAccumulationConfig,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.gradient_compressor = GradientCompressor(
            config.compression_method,
            compression_ratio=config.compression_ratio,
            quantization_bits=config.quantization_bits
        )
        
        self.dynamic_batch_sizer = DynamicBatchSizer(config) if config.dynamic_batch_sizing else None
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Accumulation state
        self.accumulated_gradients = {}
        self.accumulation_count = 0
        self.total_samples = 0
        
        # Performance monitoring
        self.gradient_norms = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.update_times = deque(maxlen=100)
        self.convergence_metrics = deque(maxlen=config.convergence_window)
        
        # Stability tracking
        self.gradient_variance_history = deque(maxlen=20)
        self.unstable_updates = 0
        
        logger.info(f"Gradient Accumulation Optimizer initialized with {config.accumulation_steps} steps")
    
    def _clear_gradients(self):
        """Clear accumulated gradients"""
        self.accumulated_gradients.clear()
        self.accumulation_count = 0
        
        # Clear model gradients
        self.optimizer.zero_grad()
        
        # Memory cleanup
        if self.accumulation_count % self.config.clear_cache_frequency == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _accumulate_gradients(self, loss: torch.Tensor):
        """Accumulate gradients from a batch"""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.config.accumulation_steps
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Store gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                self.accumulated_gradients[name] += param.grad.clone()
        
        self.accumulation_count += 1
    
    def _apply_accumulated_gradients(self) -> Dict[str, float]:
        """Apply accumulated gradients"""
        if self.accumulation_count == 0:
            return {'gradient_norm': 0.0, 'gradient_variance': 0.0}
        
        # Calculate gradient statistics
        total_norm = 0.0
        gradient_values = []
        
        # Apply accumulated gradients
        for name, param in self.model.named_parameters():
            if name in self.accumulated_gradients:
                # Average accumulated gradients
                avg_grad = self.accumulated_gradients[name] / self.accumulation_count
                param.grad = avg_grad
                
                # Calculate norm
                param_norm = avg_grad.norm().item()
                total_norm += param_norm ** 2
                
                # Collect gradient values for variance calculation
                gradient_values.extend(avg_grad.flatten().cpu().numpy())
        
        total_norm = math.sqrt(total_norm)
        
        # Calculate gradient variance
        gradient_variance = np.var(gradient_values) if gradient_values else 0.0
        
        # Gradient clipping
        if total_norm > self.config.max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_gradient_norm
            )
            total_norm = self.config.max_gradient_norm
        
        # Update optimizer
        if self.config.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Store statistics
        self.gradient_norms.append(total_norm)
        self.gradient_variance_history.append(gradient_variance)
        
        # Check for gradient instability
        if gradient_variance > self.config.gradient_variance_threshold:
            self.unstable_updates += 1
            logger.warning(f"Unstable gradients detected: variance={gradient_variance:.6f}")
        
        return {
            'gradient_norm': total_norm,
            'gradient_variance': gradient_variance
        }
    
    def step(self, 
             batch_data: torch.Tensor,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform one optimization step with gradient accumulation
        """
        start_time = time.time()
        
        # Memory monitoring
        memory_before = psutil.Process().memory_info().rss / 1024**3
        
        # Dynamic batch sizing
        if self.dynamic_batch_sizer:
            current_batch_size = self.dynamic_batch_sizer.current_batch_size
            
            # Split batch if it's too large
            if len(batch_data) > current_batch_size:
                batch_data = batch_data[:current_batch_size]
                if target is not None:
                    target = target[:current_batch_size]
        
        try:
            # Forward pass
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(batch_data)
                    loss = loss_fn(output, target) if target is not None else loss_fn(output, batch_data)
            else:
                output = self.model(batch_data)
                loss = loss_fn(output, target) if target is not None else loss_fn(output, batch_data)
            
            # Accumulate gradients
            self._accumulate_gradients(loss)
            
            # Check if we should apply gradients
            should_update = (self.accumulation_count >= self.config.accumulation_steps)
            
            metrics = {'loss': loss.item(), 'batch_size': len(batch_data)}
            
            if should_update:
                # Apply accumulated gradients
                gradient_stats = self._apply_accumulated_gradients()
                metrics.update(gradient_stats)
                
                # Clear accumulated gradients
                self._clear_gradients()
                
                # Update convergence metrics
                self.convergence_metrics.append(loss.item())
            
            # Memory monitoring
            memory_after = psutil.Process().memory_info().rss / 1024**3
            memory_usage = memory_after
            self.memory_usage.append(memory_usage)
            
            # Dynamic batch size adjustment
            if self.dynamic_batch_sizer:
                new_batch_size = self.dynamic_batch_sizer.adjust_batch_size(memory_usage)
                metrics['batch_size'] = new_batch_size
            
            # Performance metrics
            step_time = time.time() - start_time
            self.update_times.append(step_time)
            
            metrics.update({
                'memory_usage_gb': memory_usage,
                'step_time': step_time,
                'accumulation_count': self.accumulation_count,
                'updated': should_update
            })
            
            self.total_samples += len(batch_data)
            
            return metrics
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during gradient accumulation: {e}")
            
            # Clear cache and reduce batch size
            torch.cuda.empty_cache()
            gc.collect()
            
            if self.dynamic_batch_sizer:
                self.dynamic_batch_sizer.adjust_batch_size(memory_usage, had_oom=True)
            
            # Clear accumulated gradients
            self._clear_gradients()
            
            return {
                'loss': float('inf'),
                'error': 'OOM',
                'memory_usage_gb': memory_usage,
                'step_time': time.time() - start_time
            }
    
    def is_converged(self) -> bool:
        """Check if training has converged"""
        if len(self.convergence_metrics) < self.config.convergence_window:
            return False
        
        # Calculate loss variance over convergence window
        recent_losses = list(self.convergence_metrics)
        loss_variance = np.var(recent_losses)
        
        # Check if variance is below threshold
        return loss_variance < 1e-6
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        stats = {
            'total_samples': self.total_samples,
            'accumulation_count': self.accumulation_count,
            'unstable_updates': self.unstable_updates,
            'gradient_norms': {
                'mean': np.mean(self.gradient_norms) if self.gradient_norms else 0,
                'std': np.std(self.gradient_norms) if self.gradient_norms else 0,
                'max': np.max(self.gradient_norms) if self.gradient_norms else 0
            },
            'memory_usage': {
                'mean': np.mean(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0,
                'current': self.memory_usage[-1] if self.memory_usage else 0
            },
            'performance': {
                'avg_step_time': np.mean(self.update_times) if self.update_times else 0,
                'samples_per_second': self.total_samples / np.sum(self.update_times) if self.update_times else 0
            },
            'convergence': {
                'is_converged': self.is_converged(),
                'recent_loss_variance': np.var(self.convergence_metrics) if self.convergence_metrics else 0
            }
        }
        
        # Add dynamic batch sizer stats
        if self.dynamic_batch_sizer:
            stats['dynamic_batch_sizing'] = self.dynamic_batch_sizer.get_stats()
        
        return stats
    
    def save_state(self, filepath: str):
        """Save optimizer state"""
        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'accumulated_gradients': self.accumulated_gradients,
            'accumulation_count': self.accumulation_count,
            'total_samples': self.total_samples,
            'gradient_norms': list(self.gradient_norms),
            'memory_usage': list(self.memory_usage),
            'convergence_metrics': list(self.convergence_metrics),
            'unstable_updates': self.unstable_updates
        }
        
        if self.dynamic_batch_sizer:
            state['dynamic_batch_sizer_state'] = {
                'current_batch_size': self.dynamic_batch_sizer.current_batch_size,
                'oom_count': self.dynamic_batch_sizer.oom_count,
                'stable_count': self.dynamic_batch_sizer.stable_count
            }
        
        torch.save(state, filepath)
        logger.info(f"Gradient accumulation optimizer state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load optimizer state"""
        state = torch.load(filepath, map_location=self.device)
        
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if self.scaler and state.get('scaler_state_dict'):
            self.scaler.load_state_dict(state['scaler_state_dict'])
        
        self.accumulated_gradients = state.get('accumulated_gradients', {})
        self.accumulation_count = state.get('accumulation_count', 0)
        self.total_samples = state.get('total_samples', 0)
        self.gradient_norms = deque(state.get('gradient_norms', []), maxlen=100)
        self.memory_usage = deque(state.get('memory_usage', []), maxlen=100)
        self.convergence_metrics = deque(state.get('convergence_metrics', []), maxlen=self.config.convergence_window)
        self.unstable_updates = state.get('unstable_updates', 0)
        
        if self.dynamic_batch_sizer and 'dynamic_batch_sizer_state' in state:
            bs_state = state['dynamic_batch_sizer_state']
            self.dynamic_batch_sizer.current_batch_size = bs_state['current_batch_size']
            self.dynamic_batch_sizer.oom_count = bs_state['oom_count']
            self.dynamic_batch_sizer.stable_count = bs_state['stable_count']
        
        logger.info(f"Gradient accumulation optimizer state loaded from {filepath}")


def create_gradient_accumulation_config(
    available_memory_gb: float,
    target_batch_size: int,
    model_size_mb: float
) -> GradientAccumulationConfig:
    """
    Create optimized gradient accumulation configuration
    """
    # Calculate optimal accumulation steps
    memory_per_sample_mb = model_size_mb * 4  # Rough estimate
    max_batch_size = int(available_memory_gb * 1024 * 0.5 / memory_per_sample_mb)
    max_batch_size = max(16, min(128, max_batch_size))
    
    accumulation_steps = max(1, target_batch_size // max_batch_size)
    
    # Memory threshold (leave 20% buffer)
    memory_threshold = available_memory_gb * 0.8
    
    return GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        effective_batch_size=target_batch_size,
        max_batch_size=max_batch_size,
        memory_threshold_gb=memory_threshold,
        dynamic_batch_sizing=True,
        use_mixed_precision=True,
        compression_method=GradientCompressionMethod.QUANTIZATION if available_memory_gb < 8 else GradientCompressionMethod.NONE
    )