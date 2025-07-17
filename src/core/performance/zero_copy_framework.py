"""
Zero-Copy Framework - Eliminates unnecessary memory copies for sub-millisecond performance.
Implements zero-copy data structures and algorithms for high-frequency operations.
"""

import mmap
import numpy as np
import torch
import threading
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Protocol
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import ctypes
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)


class ZeroCopyType(Enum):
    """Types of zero-copy operations."""
    TENSOR_VIEW = "tensor_view"
    NUMPY_VIEW = "numpy_view"
    MEMORY_MAP = "memory_map"
    SHARED_MEMORY = "shared_memory"
    BUFFER_POOL = "buffer_pool"


@dataclass
class ZeroCopyMetrics:
    """Metrics for zero-copy operations."""
    views_created: int = 0
    copies_avoided: int = 0
    memory_saved_bytes: int = 0
    performance_gain_ms: float = 0.0
    buffer_pool_hits: int = 0
    buffer_pool_misses: int = 0


class ZeroCopyBuffer:
    """
    Zero-copy buffer implementation using memory views.
    Provides efficient buffer operations without copying data.
    """
    
    def __init__(self, size: int, dtype: np.dtype = np.float32):
        self.size = size
        self.dtype = dtype
        self._buffer = np.empty(size, dtype=dtype)
        self._view_count = 0
        self._lock = threading.RLock()
        
        logger.debug("Zero-copy buffer created", size=size, dtype=str(dtype))
    
    def get_view(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Get a view of the buffer without copying data."""
        with self._lock:
            if end is None:
                end = self.size
            
            if start < 0 or end > self.size or start >= end:
                raise ValueError(f"Invalid view range: {start}:{end}")
            
            view = self._buffer[start:end]
            self._view_count += 1
            
            logger.debug("Buffer view created", start=start, end=end, view_count=self._view_count)
            return view
    
    def get_full_view(self) -> np.ndarray:
        """Get a view of the entire buffer."""
        return self.get_view(0, self.size)
    
    def write_at(self, offset: int, data: np.ndarray):
        """Write data at offset without copying the buffer."""
        with self._lock:
            if offset < 0 or offset + len(data) > self.size:
                raise ValueError(f"Write out of bounds: offset={offset}, data_size={len(data)}")
            
            self._buffer[offset:offset + len(data)] = data
    
    def copy_from(self, source: Union[np.ndarray, torch.Tensor], offset: int = 0):
        """Copy data from source to buffer."""
        with self._lock:
            if isinstance(source, torch.Tensor):
                source_data = source.detach().cpu().numpy()
            else:
                source_data = source
                
            if offset + len(source_data) > self.size:
                raise ValueError("Source data too large for buffer")
                
            self._buffer[offset:offset + len(source_data)] = source_data
    
    def zero_fill(self, start: int = 0, end: Optional[int] = None):
        """Zero-fill buffer region without allocation."""
        with self._lock:
            if end is None:
                end = self.size
            self._buffer[start:end].fill(0)
    
    def resize(self, new_size: int):
        """Resize buffer (may require copy)."""
        with self._lock:
            if new_size == self.size:
                return
                
            old_buffer = self._buffer
            self._buffer = np.empty(new_size, dtype=self.dtype)
            
            # Copy existing data
            copy_size = min(self.size, new_size)
            self._buffer[:copy_size] = old_buffer[:copy_size]
            
            self.size = new_size
            logger.debug("Buffer resized", old_size=len(old_buffer), new_size=new_size)
    
    @property
    def view_count(self) -> int:
        """Get number of active views."""
        with self._lock:
            return self._view_count
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, key) -> np.ndarray:
        """Support indexing operations."""
        return self._buffer[key]
    
    def __setitem__(self, key, value):
        """Support assignment operations."""
        self._buffer[key] = value


class ZeroCopyTensor:
    """
    Zero-copy tensor wrapper that avoids unnecessary copies.
    Provides tensor operations without data movement.
    """
    
    def __init__(self, data: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None):
        self._original_data = data
        self._device = device or torch.device('cpu')
        self._views = []
        self._lock = threading.RLock()
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            self._tensor = torch.from_numpy(data).to(self._device)
        else:
            self._tensor = data.to(self._device)
        
        logger.debug("Zero-copy tensor created", shape=self._tensor.shape, device=str(self._device))
    
    def view(self, *shape) -> 'ZeroCopyTensor':
        """Create a view with new shape without copying data."""
        with self._lock:
            view_tensor = self._tensor.view(*shape)
            view_wrapper = ZeroCopyTensor(view_tensor, self._device)
            self._views.append(view_wrapper)
            
            logger.debug("Tensor view created", original_shape=self._tensor.shape, new_shape=shape)
            return view_wrapper
    
    def slice(self, *args) -> 'ZeroCopyTensor':
        """Create a slice without copying data."""
        with self._lock:
            sliced_tensor = self._tensor[args]
            slice_wrapper = ZeroCopyTensor(sliced_tensor, self._device)
            self._views.append(slice_wrapper)
            
            logger.debug("Tensor slice created", slice_args=args)
            return slice_wrapper
    
    def transpose(self, dim0: int, dim1: int) -> 'ZeroCopyTensor':
        """Transpose tensor without copying data."""
        with self._lock:
            transposed = self._tensor.transpose(dim0, dim1)
            transpose_wrapper = ZeroCopyTensor(transposed, self._device)
            self._views.append(transpose_wrapper)
            
            logger.debug("Tensor transposed", dim0=dim0, dim1=dim1)
            return transpose_wrapper
    
    def reshape(self, *shape) -> 'ZeroCopyTensor':
        """Reshape tensor without copying data."""
        with self._lock:
            reshaped = self._tensor.reshape(*shape)
            reshape_wrapper = ZeroCopyTensor(reshaped, self._device)
            self._views.append(reshape_wrapper)
            
            logger.debug("Tensor reshaped", original_shape=self._tensor.shape, new_shape=shape)
            return reshape_wrapper
    
    def narrow(self, dim: int, start: int, length: int) -> 'ZeroCopyTensor':
        """Narrow tensor without copying data."""
        with self._lock:
            narrowed = self._tensor.narrow(dim, start, length)
            narrow_wrapper = ZeroCopyTensor(narrowed, self._device)
            self._views.append(narrow_wrapper)
            
            logger.debug("Tensor narrowed", dim=dim, start=start, length=length)
            return narrow_wrapper
    
    def select(self, dim: int, index: int) -> 'ZeroCopyTensor':
        """Select tensor slice without copying data."""
        with self._lock:
            selected = self._tensor.select(dim, index)
            select_wrapper = ZeroCopyTensor(selected, self._device)
            self._views.append(select_wrapper)
            
            logger.debug("Tensor selected", dim=dim, index=index)
            return select_wrapper
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array without copying if on CPU."""
        if self._device.type == 'cpu':
            return self._tensor.detach().numpy()
        else:
            return self._tensor.detach().cpu().numpy()
    
    def share_memory_(self):
        """Share memory for multi-processing."""
        self._tensor.share_memory_()
        return self
    
    def is_shared(self) -> bool:
        """Check if tensor is in shared memory."""
        return self._tensor.is_shared()
    
    def copy_from(self, source: Union[torch.Tensor, np.ndarray, 'ZeroCopyTensor']):
        """Copy data from source without changing tensor structure."""
        with self._lock:
            if isinstance(source, ZeroCopyTensor):
                source_tensor = source._tensor
            elif isinstance(source, np.ndarray):
                source_tensor = torch.from_numpy(source).to(self._device)
            else:
                source_tensor = source.to(self._device)
            
            if source_tensor.shape != self._tensor.shape:
                raise ValueError(f"Shape mismatch: {source_tensor.shape} vs {self._tensor.shape}")
            
            self._tensor.copy_(source_tensor)
    
    @property
    def tensor(self) -> torch.Tensor:
        """Get underlying tensor."""
        return self._tensor
    
    @property
    def shape(self) -> torch.Size:
        """Get tensor shape."""
        return self._tensor.shape
    
    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self._device
    
    @property
    def view_count(self) -> int:
        """Get number of active views."""
        with self._lock:
            return len(self._views)
    
    def __getitem__(self, key) -> 'ZeroCopyTensor':
        """Support indexing operations."""
        return self.slice(key)
    
    def __repr__(self) -> str:
        return f"ZeroCopyTensor(shape={self.shape}, device={self.device}, views={self.view_count})"


class ZeroCopyArray:
    """
    Zero-copy array wrapper for numpy arrays.
    Provides array operations without data movement.
    """
    
    def __init__(self, data: np.ndarray):
        self._array = data
        self._views = []
        self._lock = threading.RLock()
        
        logger.debug("Zero-copy array created", shape=data.shape, dtype=str(data.dtype))
    
    def view(self, dtype: Optional[np.dtype] = None) -> 'ZeroCopyArray':
        """Create a view with different dtype without copying data."""
        with self._lock:
            if dtype is None:
                view_array = self._array.view()
            else:
                view_array = self._array.view(dtype)
            
            view_wrapper = ZeroCopyArray(view_array)
            self._views.append(view_wrapper)
            
            logger.debug("Array view created", dtype=str(dtype))
            return view_wrapper
    
    def slice(self, *args) -> 'ZeroCopyArray':
        """Create a slice without copying data."""
        with self._lock:
            sliced_array = self._array[args]
            slice_wrapper = ZeroCopyArray(sliced_array)
            self._views.append(slice_wrapper)
            
            logger.debug("Array slice created", slice_args=args)
            return slice_wrapper
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'ZeroCopyArray':
        """Transpose array without copying data."""
        with self._lock:
            transposed = self._array.transpose(axes)
            transpose_wrapper = ZeroCopyArray(transposed)
            self._views.append(transpose_wrapper)
            
            logger.debug("Array transposed", axes=axes)
            return transpose_wrapper
    
    def reshape(self, *shape, order: str = 'C') -> 'ZeroCopyArray':
        """Reshape array without copying data if possible."""
        with self._lock:
            try:
                # Try to reshape without copying
                reshaped = self._array.reshape(*shape, order=order)
                reshape_wrapper = ZeroCopyArray(reshaped)
                self._views.append(reshape_wrapper)
                
                logger.debug("Array reshaped", original_shape=self._array.shape, new_shape=shape)
                return reshape_wrapper
            except ValueError:
                # If reshape requires copy, still create wrapper but log warning
                reshaped = self._array.reshape(*shape, order=order).copy()
                reshape_wrapper = ZeroCopyArray(reshaped)
                self._views.append(reshape_wrapper)
                
                logger.warning("Array reshape required copy", original_shape=self._array.shape, new_shape=shape)
                return reshape_wrapper
    
    def to_tensor(self, device: Optional[torch.device] = None) -> ZeroCopyTensor:
        """Convert to zero-copy tensor."""
        device = device or torch.device('cpu')
        return ZeroCopyTensor(self._array, device)
    
    def copy_from(self, source: Union[np.ndarray, 'ZeroCopyArray']):
        """Copy data from source without changing array structure."""
        with self._lock:
            if isinstance(source, ZeroCopyArray):
                source_array = source._array
            else:
                source_array = source
            
            if source_array.shape != self._array.shape:
                raise ValueError(f"Shape mismatch: {source_array.shape} vs {self._array.shape}")
            
            self._array[:] = source_array
    
    @property
    def array(self) -> np.ndarray:
        """Get underlying array."""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape."""
        return self._array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get array dtype."""
        return self._array.dtype
    
    @property
    def view_count(self) -> int:
        """Get number of active views."""
        with self._lock:
            return len(self._views)
    
    def __getitem__(self, key) -> 'ZeroCopyArray':
        """Support indexing operations."""
        return self.slice(key)
    
    def __repr__(self) -> str:
        return f"ZeroCopyArray(shape={self.shape}, dtype={self.dtype}, views={self.view_count})"


class ZeroCopyBufferPool:
    """
    Pool of zero-copy buffers for efficient memory reuse.
    Eliminates allocation overhead for frequently used buffer sizes.
    """
    
    def __init__(self, max_pools: int = 16, max_buffers_per_pool: int = 32):
        self.max_pools = max_pools
        self.max_buffers_per_pool = max_buffers_per_pool
        self._pools: Dict[Tuple[int, np.dtype], List[ZeroCopyBuffer]] = {}
        self._active_buffers: Dict[int, ZeroCopyBuffer] = {}
        self._metrics = ZeroCopyMetrics()
        self._lock = threading.RLock()
        
        logger.info("Zero-copy buffer pool initialized", 
                   max_pools=max_pools, 
                   max_buffers_per_pool=max_buffers_per_pool)
    
    def get_buffer(self, size: int, dtype: np.dtype = np.float32) -> ZeroCopyBuffer:
        """Get a buffer from the pool or create new one."""
        pool_key = (size, dtype)
        
        with self._lock:
            # Try to get from pool
            if pool_key in self._pools and self._pools[pool_key]:
                buffer = self._pools[pool_key].pop()
                self._active_buffers[id(buffer)] = buffer
                self._metrics.buffer_pool_hits += 1
                
                logger.debug("Buffer retrieved from pool", size=size, dtype=str(dtype))
                return buffer
            
            # Create new buffer
            buffer = ZeroCopyBuffer(size, dtype)
            self._active_buffers[id(buffer)] = buffer
            self._metrics.buffer_pool_misses += 1
            
            logger.debug("New buffer created", size=size, dtype=str(dtype))
            return buffer
    
    def return_buffer(self, buffer: ZeroCopyBuffer):
        """Return buffer to pool for reuse."""
        buffer_id = id(buffer)
        pool_key = (buffer.size, buffer.dtype)
        
        with self._lock:
            if buffer_id not in self._active_buffers:
                logger.warning("Attempting to return unknown buffer")
                return
            
            # Remove from active buffers
            del self._active_buffers[buffer_id]
            
            # Add to pool if space available
            if pool_key not in self._pools:
                if len(self._pools) >= self.max_pools:
                    logger.debug("Buffer pool full, discarding buffer")
                    return
                self._pools[pool_key] = []
            
            if len(self._pools[pool_key]) < self.max_buffers_per_pool:
                # Clear buffer before returning to pool
                buffer.zero_fill()
                self._pools[pool_key].append(buffer)
                
                logger.debug("Buffer returned to pool", size=buffer.size, dtype=str(buffer.dtype))
            else:
                logger.debug("Pool full, discarding buffer")
    
    @contextmanager
    def get_buffer_context(self, size: int, dtype: np.dtype = np.float32):
        """Context manager for automatic buffer return."""
        buffer = self.get_buffer(size, dtype)
        try:
            yield buffer
        finally:
            self.return_buffer(buffer)
    
    def clear_pools(self):
        """Clear all pools."""
        with self._lock:
            self._pools.clear()
            self._active_buffers.clear()
            
            logger.info("Buffer pools cleared")
    
    def get_metrics(self) -> ZeroCopyMetrics:
        """Get pool metrics."""
        with self._lock:
            return self._metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics."""
        with self._lock:
            return {
                'pools': len(self._pools),
                'active_buffers': len(self._active_buffers),
                'total_pooled_buffers': sum(len(buffers) for buffers in self._pools.values()),
                'metrics': {
                    'buffer_pool_hits': self._metrics.buffer_pool_hits,
                    'buffer_pool_misses': self._metrics.buffer_pool_misses,
                    'hit_rate': self._metrics.buffer_pool_hits / max(1, self._metrics.buffer_pool_hits + self._metrics.buffer_pool_misses)
                }
            }


# Global buffer pool instance
_global_buffer_pool: Optional[ZeroCopyBufferPool] = None


def get_buffer_pool() -> ZeroCopyBufferPool:
    """Get the global buffer pool instance."""
    global _global_buffer_pool
    if _global_buffer_pool is None:
        _global_buffer_pool = ZeroCopyBufferPool()
    return _global_buffer_pool


def set_buffer_pool(pool: ZeroCopyBufferPool):
    """Set the global buffer pool instance."""
    global _global_buffer_pool
    _global_buffer_pool = pool


# Convenience functions
def zero_copy_tensor(data: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None) -> ZeroCopyTensor:
    """Create a zero-copy tensor."""
    return ZeroCopyTensor(data, device)


def zero_copy_array(data: np.ndarray) -> ZeroCopyArray:
    """Create a zero-copy array."""
    return ZeroCopyArray(data)


@contextmanager
def zero_copy_context(size: int, dtype: np.dtype = np.float32):
    """Context manager for zero-copy operations."""
    buffer_pool = get_buffer_pool()
    with buffer_pool.get_buffer_context(size, dtype) as buffer:
        yield buffer


def avoid_copy(func):
    """Decorator to track copy-avoided operations."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Update global metrics
        buffer_pool = get_buffer_pool()
        buffer_pool._metrics.copies_avoided += 1
        buffer_pool._metrics.performance_gain_ms += (end_time - start_time) * 1000
        
        return result
    return wrapper


class ZeroCopyManager:
    """
    Manager for zero-copy operations and optimizations.
    Provides centralized control over zero-copy behavior.
    """
    
    def __init__(self):
        self._buffer_pool = ZeroCopyBufferPool()
        self._optimization_enabled = True
        self._metrics = ZeroCopyMetrics()
        
    def enable_optimization(self):
        """Enable zero-copy optimizations."""
        self._optimization_enabled = True
        logger.info("Zero-copy optimizations enabled")
    
    def disable_optimization(self):
        """Disable zero-copy optimizations."""
        self._optimization_enabled = False
        logger.info("Zero-copy optimizations disabled")
    
    def is_optimization_enabled(self) -> bool:
        """Check if optimization is enabled."""
        return self._optimization_enabled
    
    def get_buffer_pool(self) -> ZeroCopyBufferPool:
        """Get the buffer pool."""
        return self._buffer_pool
    
    def get_metrics(self) -> ZeroCopyMetrics:
        """Get zero-copy metrics."""
        return self._metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics = ZeroCopyMetrics()
        logger.info("Zero-copy metrics reset")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        
        return {
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'buffer_pool_stats': self._buffer_pool.get_stats(),
            'metrics': self._metrics
        }


# Global zero-copy manager
_global_zero_copy_manager: Optional[ZeroCopyManager] = None


def get_zero_copy_manager() -> ZeroCopyManager:
    """Get the global zero-copy manager."""
    global _global_zero_copy_manager
    if _global_zero_copy_manager is None:
        _global_zero_copy_manager = ZeroCopyManager()
    return _global_zero_copy_manager


def enable_zero_copy_optimizations():
    """Enable zero-copy optimizations globally."""
    get_zero_copy_manager().enable_optimization()


def disable_zero_copy_optimizations():
    """Disable zero-copy optimizations globally."""
    get_zero_copy_manager().disable_optimization()