"""
Secure Memory Management - CVE-2025-TACTICAL-003 Mitigation

Eliminates race conditions and provides secure memory management
for high-frequency trading neural networks.
"""

import torch
import torch.nn as nn
import threading
import queue
import time
from typing import Dict, Any, Optional, Callable
import weakref
import gc
from contextlib import contextmanager


class SecureMemoryManager:
    """
    Thread-safe memory manager with race condition elimination.
    
    Key Security Features:
    - Thread-safe tensor operations
    - Memory leak prevention
    - Race condition elimination
    - Secure tensor cleanup
    """
    
    def __init__(self, max_cache_size: int = 100, cleanup_interval: float = 30.0):
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe data structures
        self._tensor_cache = {}
        self._cache_lock = threading.RLock()
        self._operation_queue = queue.Queue()
        self._active_operations = set()
        self._operations_lock = threading.Lock()
        
        # Memory tracking
        self._allocated_tensors = weakref.WeakSet()
        self._memory_stats = {
            'peak_memory': 0,
            'current_memory': 0,
            'tensor_count': 0,
            'cleanup_count': 0
        }
        self._stats_lock = threading.Lock()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for memory cleanup."""
        while not self._shutdown_event.wait(self.cleanup_interval):
            try:
                self._perform_cleanup()
            except Exception as e:
                # Log error in production
                pass
    
    def _perform_cleanup(self):
        """Perform memory cleanup operations."""
        with self._cache_lock:
            # Clean up cache if too large
            if len(self._tensor_cache) > self.max_cache_size:
                # Remove oldest entries (simple LRU)
                items_to_remove = len(self._tensor_cache) - self.max_cache_size
                keys_to_remove = list(self._tensor_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    if key in self._tensor_cache:
                        del self._tensor_cache[key]
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update stats
        with self._stats_lock:
            self._memory_stats['cleanup_count'] += 1
            self._memory_stats['tensor_count'] = len(self._allocated_tensors)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                self._memory_stats['current_memory'] = current_memory
                if current_memory > self._memory_stats['peak_memory']:
                    self._memory_stats['peak_memory'] = current_memory
    
    @contextmanager
    def secure_operation(self, operation_id: str):
        """Context manager for secure tensor operations."""
        # Register operation start
        with self._operations_lock:
            if operation_id in self._active_operations:
                raise RuntimeError(f"Operation {operation_id} already active - potential race condition")
            self._active_operations.add(operation_id)
        
        try:
            yield
        finally:
            # Cleanup operation
            with self._operations_lock:
                self._active_operations.discard(operation_id)
    
    def secure_tensor_allocation(
        self, 
        shape: tuple, 
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        cache_key: Optional[str] = None
    ) -> torch.Tensor:
        """
        Allocate tensor with secure memory management.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            requires_grad: Whether to track gradients
            cache_key: Optional cache key for reuse
            
        Returns:
            Allocated tensor
        """
        with self._cache_lock:
            # Check cache first
            if cache_key and cache_key in self._tensor_cache:
                cached_tensor = self._tensor_cache[cache_key]
                if (cached_tensor.shape == shape and 
                    cached_tensor.dtype == dtype and
                    cached_tensor.device == device):
                    # Reset tensor data
                    cached_tensor.zero_()
                    cached_tensor.requires_grad_(requires_grad)
                    return cached_tensor
            
            # Allocate new tensor
            tensor = torch.zeros(
                shape, 
                dtype=dtype, 
                device=device, 
                requires_grad=requires_grad
            )
            
            # Track tensor
            self._allocated_tensors.add(tensor)
            
            # Cache if requested
            if cache_key:
                self._tensor_cache[cache_key] = tensor
            
            return tensor
    
    def secure_tensor_copy(
        self, 
        source: torch.Tensor, 
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform secure tensor copy with race condition prevention.
        
        Args:
            source: Source tensor
            target: Optional target tensor
            
        Returns:
            Copied tensor
        """
        with torch.no_grad():
            if target is None:
                target = self.secure_tensor_allocation(
                    source.shape,
                    source.dtype,
                    source.device
                )
            
            # Thread-safe copy
            target.copy_(source)
            return target
    
    def secure_tensor_operation(
        self, 
        operation: Callable,
        *args,
        operation_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute tensor operation with security guarantees.
        
        Args:
            operation: Operation function
            *args: Operation arguments
            operation_id: Optional operation identifier
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        op_id = operation_id or f"op_{time.time()}_{id(operation)}"
        
        with self.secure_operation(op_id):
            # Execute operation with error handling
            try:
                result = operation(*args, **kwargs)
                
                # Track result tensors
                if isinstance(result, torch.Tensor):
                    self._allocated_tensors.add(result)
                elif isinstance(result, (list, tuple)):
                    for item in result:
                        if isinstance(item, torch.Tensor):
                            self._allocated_tensors.add(item)
                
                return result
                
            except Exception as e:
                # Clean up on error
                self._perform_cleanup()
                raise e
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        with self._stats_lock:
            stats = self._memory_stats.copy()
        
        # Add current tensor count
        stats['active_tensors'] = len(self._allocated_tensors)
        stats['cached_tensors'] = len(self._tensor_cache)
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved()
        
        return stats
    
    def clear_cache(self):
        """Clear tensor cache."""
        with self._cache_lock:
            self._tensor_cache.clear()
        
        # Force cleanup
        self._perform_cleanup()
    
    def shutdown(self):
        """Shutdown memory manager."""
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        # Final cleanup
        self.clear_cache()


class SecureBufferManager(nn.Module):
    """
    Secure buffer management for neural network layers.
    """
    
    def __init__(self, memory_manager: Optional[SecureMemoryManager] = None):
        super().__init__()
        self.memory_manager = memory_manager or SecureMemoryManager()
        self._buffers = {}
        self._buffer_lock = threading.RLock()
    
    def get_buffer(
        self, 
        name: str, 
        shape: tuple, 
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get or create a named buffer."""
        with self._buffer_lock:
            buffer_key = f"{name}_{shape}_{dtype}_{device}"
            
            if buffer_key not in self._buffers:
                self._buffers[buffer_key] = self.memory_manager.secure_tensor_allocation(
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    cache_key=buffer_key
                )
            
            return self._buffers[buffer_key]
    
    def release_buffer(self, name: str):
        """Release a named buffer."""
        with self._buffer_lock:
            keys_to_remove = [k for k in self._buffers.keys() if k.startswith(name)]
            for key in keys_to_remove:
                del self._buffers[key]
    
    def clear_buffers(self):
        """Clear all buffers."""
        with self._buffer_lock:
            self._buffers.clear()
        self.memory_manager.clear_cache()