"""
Online Gradient Accumulation System for Live Trading
Real-time gradient accumulation and streaming updates for continuous learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
import logging
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
import asyncio
import queue
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from enum import Enum
import math
import json
from datetime import datetime, timedelta
import pickle
import zlib
import sys
import traceback
from contextlib import contextmanager
from functools import wraps
import uuid

logger = logging.getLogger(__name__)


class GradientUpdateStrategy(Enum):
    """Strategy for gradient updates"""
    IMMEDIATE = "immediate"
    BATCHED = "batched"
    ADAPTIVE = "adaptive"
    MOMENTUM_BASED = "momentum_based"


class CompressionMethod(Enum):
    """Gradient compression methods"""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    DELTA_COMPRESSION = "delta_compression"
    HUFFMAN = "huffman"


@dataclass
class OnlineGradientConfig:
    """Configuration for online gradient accumulation"""
    # Core settings
    max_accumulation_steps: int = 16
    min_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    
    # Memory management
    max_memory_mb: float = 1024.0
    gradient_buffer_size: int = 1000
    memory_cleanup_threshold: float = 0.8
    
    # Compression
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    
    # Performance
    target_latency_ms: float = 50.0
    max_concurrent_updates: int = 4
    use_jit_compilation: bool = True
    async_processing: bool = True
    
    # Adaptive parameters
    adaptive_batch_sizing: bool = True
    performance_window_size: int = 100
    latency_threshold_ms: float = 100.0
    
    # Convergence monitoring
    convergence_check_interval: int = 50
    convergence_threshold: float = 1e-6
    stability_window: int = 20
    
    # Streaming settings
    stream_buffer_size: int = 500
    stream_chunk_size: int = 32
    stream_timeout_ms: float = 10.0


class GradientBuffer:
    """Thread-safe gradient buffer with compression"""
    
    def __init__(self, config: OnlineGradientConfig):
        self.config = config
        self.buffer = deque(maxlen=config.gradient_buffer_size)
        self.lock = threading.RLock()
        self.compression_stats = {
            'total_compressed': 0,
            'total_uncompressed': 0,
            'compression_ratio': 0.0
        }
        
    def add_gradient(self, gradient_dict: Dict[str, torch.Tensor], 
                    metadata: Dict[str, Any] = None) -> str:
        """Add gradient to buffer with compression"""
        gradient_id = str(uuid.uuid4())
        
        with self.lock:
            # Compress gradient
            compressed_gradient = self._compress_gradient(gradient_dict)
            
            # Create gradient entry
            entry = {
                'id': gradient_id,
                'gradients': compressed_gradient,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'size': self._calculate_size(gradient_dict)
            }
            
            self.buffer.append(entry)
            
            # Update compression stats
            self._update_compression_stats(gradient_dict, compressed_gradient)
            
        return gradient_id
    
    def get_gradients(self, count: int = None) -> List[Dict[str, Any]]:
        """Get gradients from buffer"""
        with self.lock:
            if count is None:
                count = len(self.buffer)
            
            gradients = []
            for _ in range(min(count, len(self.buffer))):
                if self.buffer:
                    entry = self.buffer.popleft()
                    # Decompress gradient
                    entry['gradients'] = self._decompress_gradient(entry['gradients'])
                    gradients.append(entry)
            
            return gradients
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def _compress_gradient(self, gradient_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress gradient dictionary"""
        if self.config.compression_method == CompressionMethod.NONE:
            return gradient_dict
        
        compressed = {}
        
        for name, grad in gradient_dict.items():
            if grad is None:
                compressed[name] = None
                continue
            
            if self.config.compression_method == CompressionMethod.QUANTIZATION:
                compressed[name] = self._quantize_tensor(grad)
            elif self.config.compression_method == CompressionMethod.SPARSIFICATION:
                compressed[name] = self._sparsify_tensor(grad)
            elif self.config.compression_method == CompressionMethod.DELTA_COMPRESSION:
                compressed[name] = self._delta_compress_tensor(grad)
            else:
                compressed[name] = grad
        
        return compressed
    
    def _decompress_gradient(self, compressed_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress gradient dictionary"""
        if self.config.compression_method == CompressionMethod.NONE:
            return compressed_dict
        
        decompressed = {}
        
        for name, compressed_grad in compressed_dict.items():
            if compressed_grad is None:
                decompressed[name] = None
                continue
            
            if self.config.compression_method == CompressionMethod.QUANTIZATION:
                decompressed[name] = self._dequantize_tensor(compressed_grad)
            elif self.config.compression_method == CompressionMethod.SPARSIFICATION:
                decompressed[name] = self._desparsify_tensor(compressed_grad)
            elif self.config.compression_method == CompressionMethod.DELTA_COMPRESSION:
                decompressed[name] = self._delta_decompress_tensor(compressed_grad)
            else:
                decompressed[name] = compressed_grad
        
        return decompressed
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Quantize tensor to reduce size"""
        if tensor.numel() == 0:
            return {'type': 'empty', 'shape': tensor.shape}
        
        # Calculate quantization parameters
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        
        if t_max == t_min:
            return {
                'type': 'constant',
                'value': t_min,
                'shape': tensor.shape
            }
        
        # Quantize
        scale = (t_max - t_min) / (2**self.config.quantization_bits - 1)
        quantized = torch.round((tensor - t_min) / scale).to(torch.uint8)
        
        return {
            'type': 'quantized',
            'data': quantized,
            'scale': scale,
            'offset': t_min,
            'shape': tensor.shape
        }
    
    def _dequantize_tensor(self, quantized_data: Dict[str, Any]) -> torch.Tensor:
        """Dequantize tensor"""
        if quantized_data['type'] == 'empty':
            return torch.empty(quantized_data['shape'])
        elif quantized_data['type'] == 'constant':
            return torch.full(quantized_data['shape'], quantized_data['value'])
        elif quantized_data['type'] == 'quantized':
            quantized = quantized_data['data'].float()
            return quantized * quantized_data['scale'] + quantized_data['offset']
        else:
            return quantized_data['data']
    
    def _sparsify_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Sparsify tensor by keeping only top-k elements"""
        if tensor.numel() == 0:
            return {'type': 'empty', 'shape': tensor.shape}
        
        # Flatten tensor
        flat_tensor = tensor.flatten()
        k = max(1, int(tensor.numel() * self.config.compression_ratio))
        
        # Get top-k indices
        _, top_indices = torch.topk(torch.abs(flat_tensor), k)
        
        # Create sparse representation
        sparse_values = flat_tensor[top_indices]
        
        return {
            'type': 'sparse',
            'indices': top_indices,
            'values': sparse_values,
            'shape': tensor.shape
        }
    
    def _desparsify_tensor(self, sparse_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct tensor from sparse representation"""
        if sparse_data['type'] == 'empty':
            return torch.empty(sparse_data['shape'])
        elif sparse_data['type'] == 'sparse':
            # Reconstruct tensor
            flat_tensor = torch.zeros(torch.prod(torch.tensor(sparse_data['shape'])))
            flat_tensor[sparse_data['indices']] = sparse_data['values']
            return flat_tensor.reshape(sparse_data['shape'])
        else:
            return sparse_data['data']
    
    def _delta_compress_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Delta compression - store differences from previous"""
        # For now, implement simple delta compression
        # In practice, you'd maintain previous state
        return {
            'type': 'delta',
            'data': tensor,
            'shape': tensor.shape
        }
    
    def _delta_decompress_tensor(self, delta_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress delta-compressed tensor"""
        return delta_data['data']
    
    def _calculate_size(self, gradient_dict: Dict[str, torch.Tensor]) -> int:
        """Calculate size of gradient dictionary"""
        total_size = 0
        for grad in gradient_dict.values():
            if grad is not None:
                total_size += grad.numel() * grad.element_size()
        return total_size
    
    def _update_compression_stats(self, original: Dict[str, torch.Tensor], 
                                 compressed: Dict[str, Any]):
        """Update compression statistics"""
        original_size = self._calculate_size(original)
        compressed_size = sys.getsizeof(pickle.dumps(compressed))
        
        self.compression_stats['total_uncompressed'] += original_size
        self.compression_stats['total_compressed'] += compressed_size
        
        if self.compression_stats['total_uncompressed'] > 0:
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['total_compressed'] / 
                self.compression_stats['total_uncompressed']
            )
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.compression_stats.copy()


class PerformanceMonitor:
    """Monitor performance metrics for gradient processing"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'latencies': deque(maxlen=window_size),
            'throughput': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'gradient_norms': deque(maxlen=window_size),
            'compression_ratios': deque(maxlen=window_size)
        }
        self.lock = threading.RLock()
    
    def record_latency(self, latency_ms: float):
        """Record processing latency"""
        with self.lock:
            self.metrics['latencies'].append(latency_ms)
    
    def record_throughput(self, samples_per_second: float):
        """Record throughput"""
        with self.lock:
            self.metrics['throughput'].append(samples_per_second)
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        with self.lock:
            self.metrics['memory_usage'].append(memory_mb)
    
    def record_gradient_norm(self, norm: float):
        """Record gradient norm"""
        with self.lock:
            self.metrics['gradient_norms'].append(norm)
    
    def record_compression_ratio(self, ratio: float):
        """Record compression ratio"""
        with self.lock:
            self.metrics['compression_ratios'].append(ratio)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            for metric_name, values in self.metrics.items():
                if values:
                    stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
                else:
                    stats[metric_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                    }
            return stats
    
    def is_latency_acceptable(self, threshold_ms: float) -> bool:
        """Check if latency is within acceptable range"""
        with self.lock:
            if not self.metrics['latencies']:
                return True
            
            recent_latencies = list(self.metrics['latencies'])[-10:]
            avg_latency = np.mean(recent_latencies)
            return avg_latency <= threshold_ms


class OnlineGradientAccumulator:
    """
    Online gradient accumulation system for real-time learning
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: OnlineGradientConfig,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.gradient_buffer = GradientBuffer(config)
        self.performance_monitor = PerformanceMonitor(config.performance_window_size)
        
        # State tracking
        self.accumulated_steps = 0
        self.total_updates = 0
        self.last_update_time = time.time()
        
        # Adaptive parameters
        self.current_accumulation_steps = config.min_accumulation_steps
        self.adaptive_enabled = config.adaptive_batch_sizing
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.gradient_queue = queue.Queue(maxsize=config.stream_buffer_size)
        
        # Performance tracking
        self.update_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=50)
        
        # Convergence monitoring
        self.convergence_history = deque(maxlen=config.stability_window)
        self.convergence_counter = 0
        
        # JIT compilation
        if config.use_jit_compilation:
            self._compile_critical_paths()
        
        # Start processing thread if async
        if config.async_processing:
            self.start_async_processing()
        
        logger.info(f"Online gradient accumulator initialized with {config.max_accumulation_steps} max steps")
    
    def _compile_critical_paths(self):
        """Compile critical paths with JIT for performance"""
        try:
            # Create dummy inputs for JIT compilation
            dummy_input = torch.randn(1, 128, device=self.device)
            
            # Compile forward pass
            self.model.eval()
            with torch.no_grad():
                torch.jit.trace(self.model, dummy_input)
            
            self.model.train()
            logger.info("JIT compilation completed successfully")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
    
    def start_async_processing(self):
        """Start asynchronous gradient processing"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = threading.Thread(
                target=self._process_gradients_async,
                daemon=True
            )
            self.processing_thread.start()
            logger.info("Async gradient processing started")
    
    def stop_async_processing(self):
        """Stop asynchronous gradient processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=5.0)
            logger.info("Async gradient processing stopped")
    
    def _process_gradients_async(self):
        """Asynchronous gradient processing loop"""
        while not self.stop_event.is_set():
            try:
                # Get gradient from queue with timeout
                gradient_data = self.gradient_queue.get(timeout=0.1)
                
                if gradient_data is None:  # Shutdown signal
                    break
                
                # Process gradient
                self._process_single_gradient(gradient_data)
                
                # Mark task as done
                self.gradient_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in async gradient processing: {e}")
                logger.error(traceback.format_exc())
    
    def accumulate_gradient(self, 
                          loss: torch.Tensor,
                          batch_data: torch.Tensor = None,
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Accumulate gradient from a single batch
        """
        start_time = time.time()
        
        try:
            # Compute gradient
            scaled_loss = loss / self.current_accumulation_steps
            scaled_loss.backward()
            
            # Extract gradients
            gradient_dict = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            
            # Calculate gradient norm
            gradient_norm = self._calculate_gradient_norm(gradient_dict)
            
            # Create gradient data
            gradient_data = {
                'gradients': gradient_dict,
                'loss': loss.item(),
                'gradient_norm': gradient_norm,
                'batch_size': len(batch_data) if batch_data is not None else 1,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Process gradient
            if self.config.async_processing:
                # Add to queue for async processing
                try:
                    self.gradient_queue.put(gradient_data, timeout=0.001)
                except queue.Full:
                    logger.warning("Gradient queue full, processing synchronously")
                    result = self._process_single_gradient(gradient_data)
            else:
                # Process synchronously
                result = self._process_single_gradient(gradient_data)
            
            # Record performance
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.performance_monitor.record_latency(processing_time)
            
            # Memory monitoring
            memory_usage = psutil.Process().memory_info().rss / 1024**2
            self.performance_monitor.record_memory_usage(memory_usage)
            
            # Check for memory pressure
            if memory_usage > self.config.max_memory_mb * self.config.memory_cleanup_threshold:
                self._cleanup_memory()
            
            return {
                'gradient_norm': gradient_norm,
                'processing_time_ms': processing_time,
                'memory_usage_mb': memory_usage,
                'accumulated_steps': self.accumulated_steps,
                'updated': self.accumulated_steps >= self.current_accumulation_steps
            }
            
        except Exception as e:
            logger.error(f"Error in gradient accumulation: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'gradient_norm': 0.0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024**2,
                'accumulated_steps': self.accumulated_steps,
                'updated': False
            }
    
    def _process_single_gradient(self, gradient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single gradient"""
        # Add to buffer
        gradient_id = self.gradient_buffer.add_gradient(
            gradient_data['gradients'],
            gradient_data['metadata']
        )
        
        self.accumulated_steps += 1
        
        # Check if we should update
        should_update = self.accumulated_steps >= self.current_accumulation_steps
        
        if should_update:
            update_result = self._apply_accumulated_gradients()
            
            # Update adaptive parameters
            if self.adaptive_enabled:
                self._update_adaptive_parameters()
            
            # Check convergence
            self._check_convergence(gradient_data['loss'])
            
            return update_result
        
        return {
            'gradient_id': gradient_id,
            'accumulated_steps': self.accumulated_steps,
            'target_steps': self.current_accumulation_steps,
            'updated': False
        }
    
    def _apply_accumulated_gradients(self) -> Dict[str, Any]:
        """Apply accumulated gradients to model"""
        start_time = time.time()
        
        try:
            # Get accumulated gradients
            gradient_entries = self.gradient_buffer.get_gradients(self.accumulated_steps)
            
            if not gradient_entries:
                return {'error': 'No gradients to apply'}
            
            # Average gradients
            averaged_gradients = self._average_gradients(gradient_entries)
            
            # Apply to model
            for name, param in self.model.named_parameters():
                if name in averaged_gradients:
                    param.grad = averaged_gradients[name]
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
            
            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Reset accumulation
            self.accumulated_steps = 0
            self.total_updates += 1
            
            # Record performance
            update_time = (time.time() - start_time) * 1000
            self.update_times.append(update_time)
            
            # Performance monitoring
            self.performance_monitor.record_gradient_norm(total_norm.item())
            
            return {
                'gradient_norm': total_norm.item(),
                'update_time_ms': update_time,
                'total_updates': self.total_updates,
                'gradients_processed': len(gradient_entries),
                'updated': True
            }
            
        except Exception as e:
            logger.error(f"Error applying gradients: {e}")
            return {'error': str(e), 'updated': False}
    
    def _average_gradients(self, gradient_entries: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Average gradients from multiple entries"""
        if not gradient_entries:
            return {}
        
        averaged = {}
        param_counts = {}
        
        for entry in gradient_entries:
            gradients = entry['gradients']
            for name, grad in gradients.items():
                if grad is not None:
                    if name not in averaged:
                        averaged[name] = torch.zeros_like(grad)
                        param_counts[name] = 0
                    
                    averaged[name] += grad
                    param_counts[name] += 1
        
        # Average
        for name in averaged:
            if param_counts[name] > 0:
                averaged[name] /= param_counts[name]
        
        return averaged
    
    def _calculate_gradient_norm(self, gradient_dict: Dict[str, torch.Tensor]) -> float:
        """Calculate gradient norm"""
        total_norm = 0.0
        for grad in gradient_dict.values():
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        return math.sqrt(total_norm)
    
    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on performance"""
        if not self.adaptive_enabled:
            return
        
        # Get recent performance metrics
        stats = self.performance_monitor.get_stats()
        
        # Adjust accumulation steps based on latency
        if stats['latencies']['count'] > 0:
            avg_latency = stats['latencies']['mean']
            
            if avg_latency > self.config.latency_threshold_ms:
                # Reduce accumulation steps to improve latency
                self.current_accumulation_steps = max(
                    self.config.min_accumulation_steps,
                    self.current_accumulation_steps - 1
                )
            elif avg_latency < self.config.target_latency_ms:
                # Increase accumulation steps for better efficiency
                self.current_accumulation_steps = min(
                    self.config.max_accumulation_steps,
                    self.current_accumulation_steps + 1
                )
        
        logger.debug(f"Adaptive accumulation steps: {self.current_accumulation_steps}")
    
    def _check_convergence(self, loss: float):
        """Check for convergence"""
        self.convergence_history.append(loss)
        self.convergence_counter += 1
        
        if (self.convergence_counter % self.config.convergence_check_interval == 0 and
            len(self.convergence_history) >= self.config.stability_window):
            
            # Calculate loss variance
            recent_losses = list(self.convergence_history)
            loss_variance = np.var(recent_losses)
            
            if loss_variance < self.config.convergence_threshold:
                logger.info(f"Convergence detected: loss variance = {loss_variance:.8f}")
    
    def _cleanup_memory(self):
        """Clean up memory"""
        # Clear old gradients
        self.gradient_buffer.clear()
        
        # Clear old metrics
        if len(self.update_times) > 50:
            self.update_times = deque(list(self.update_times)[-50:], maxlen=100)
        
        if len(self.memory_usage) > 25:
            self.memory_usage = deque(list(self.memory_usage)[-25:], maxlen=50)
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_monitor.get_stats()
        
        # Add system-specific stats
        stats['system'] = {
            'accumulated_steps': self.accumulated_steps,
            'total_updates': self.total_updates,
            'current_accumulation_steps': self.current_accumulation_steps,
            'gradient_buffer_size': self.gradient_buffer.size(),
            'compression_stats': self.gradient_buffer.get_compression_stats()
        }
        
        # Add timing stats
        if self.update_times:
            stats['update_timing'] = {
                'avg_update_time_ms': np.mean(self.update_times),
                'max_update_time_ms': np.max(self.update_times),
                'min_update_time_ms': np.min(self.update_times)
            }
        
        return stats
    
    def is_converged(self) -> bool:
        """Check if model has converged"""
        if len(self.convergence_history) < self.config.stability_window:
            return False
        
        recent_losses = list(self.convergence_history)
        loss_variance = np.var(recent_losses)
        return loss_variance < self.config.convergence_threshold
    
    def save_state(self, filepath: str):
        """Save accumulator state"""
        state = {
            'config': self.config.__dict__,
            'accumulated_steps': self.accumulated_steps,
            'total_updates': self.total_updates,
            'current_accumulation_steps': self.current_accumulation_steps,
            'convergence_history': list(self.convergence_history),
            'performance_stats': self.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Accumulator state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load accumulator state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.accumulated_steps = state.get('accumulated_steps', 0)
        self.total_updates = state.get('total_updates', 0)
        self.current_accumulation_steps = state.get('current_accumulation_steps', 
                                                   self.config.min_accumulation_steps)
        
        # Restore convergence history
        convergence_history = state.get('convergence_history', [])
        self.convergence_history = deque(convergence_history, 
                                       maxlen=self.config.stability_window)
        
        logger.info(f"Accumulator state loaded from {filepath}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_async_processing()


def create_online_gradient_config(
    target_latency_ms: float = 50.0,
    max_memory_mb: float = 1024.0,
    compression_enabled: bool = True
) -> OnlineGradientConfig:
    """Create optimized online gradient configuration"""
    
    # Determine accumulation steps based on target latency
    if target_latency_ms <= 20.0:
        max_accumulation_steps = 8
        min_accumulation_steps = 2
    elif target_latency_ms <= 50.0:
        max_accumulation_steps = 16
        min_accumulation_steps = 4
    else:
        max_accumulation_steps = 32
        min_accumulation_steps = 8
    
    # Set compression method
    compression_method = (CompressionMethod.QUANTIZATION if compression_enabled 
                         else CompressionMethod.NONE)
    
    return OnlineGradientConfig(
        max_accumulation_steps=max_accumulation_steps,
        min_accumulation_steps=min_accumulation_steps,
        target_latency_ms=target_latency_ms,
        max_memory_mb=max_memory_mb,
        compression_method=compression_method,
        adaptive_batch_sizing=True,
        async_processing=True,
        use_jit_compilation=True
    )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create configuration
    config = create_online_gradient_config(
        target_latency_ms=30.0,
        max_memory_mb=512.0,
        compression_enabled=True
    )
    
    # Initialize accumulator
    accumulator = OnlineGradientAccumulator(model, optimizer, config)
    
    # Simulate training
    print("Starting online gradient accumulation demo...")
    
    for i in range(100):
        # Simulate batch data
        batch_data = torch.randn(32, 128)
        target = torch.randn(32, 1)
        
        # Forward pass
        output = model(batch_data)
        loss = nn.MSELoss()(output, target)
        
        # Accumulate gradient
        result = accumulator.accumulate_gradient(loss, batch_data)
        
        if i % 10 == 0:
            print(f"Step {i}: Loss={loss.item():.6f}, "
                  f"Gradient Norm={result['gradient_norm']:.6f}, "
                  f"Latency={result['processing_time_ms']:.2f}ms")
    
    # Get final statistics
    stats = accumulator.get_performance_stats()
    print(f"\nFinal Statistics:")
    print(f"Total Updates: {stats['system']['total_updates']}")
    print(f"Average Latency: {stats['latencies']['mean']:.2f}ms")
    print(f"Average Memory Usage: {stats['memory_usage']['mean']:.2f}MB")
    print(f"Compression Ratio: {stats['system']['compression_stats']['compression_ratio']:.3f}")
    
    # Check convergence
    print(f"Converged: {accumulator.is_converged()}")
    
    # Cleanup
    accumulator.stop_async_processing()
    print("Demo completed!")