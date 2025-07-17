"""
Memory Manager - Aggressive tensor cleanup and memory optimization
Implements memory pools, leak detection, and automatic cleanup for tensor operations.
"""

import logging


import gc
import psutil
import time
import torch
import threading
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)


class MemoryPoolType(Enum):
    """Types of memory pools."""
    TENSOR_CACHE = "tensor_cache"
    GRADIENT_CACHE = "gradient_cache"
    ACTIVATION_CACHE = "activation_cache"
    TEMPORARY_BUFFER = "temporary_buffer"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated_mb: float
    total_cached_mb: float
    total_reserved_mb: float
    gpu_memory_mb: float
    cpu_memory_mb: float
    pool_usage_mb: Dict[str, float]
    active_tensors: int
    leak_candidates: int
    last_cleanup_time: float


@dataclass
class TensorInfo:
    """Information about a tracked tensor."""
    id: int
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    size_bytes: int
    created_at: float
    last_accessed: float
    ref_count: int
    source_location: str


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""
    
    def __init__(self, pool_type: MemoryPoolType, max_size_mb: int = 256):
        self.pool_type = pool_type
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Pool storage
        self._pool: Dict[Tuple[torch.Size, torch.dtype, torch.device], deque] = defaultdict(deque)
        self._current_size = 0
        self._allocations = 0
        self._hits = 0
        self._misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug("Memory pool created", 
                    pool_type=pool_type.value,
                    max_size_mb=max_size_mb)
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        """Get a tensor from the pool or return None if not available."""
        key = (shape, dtype, device)
        
        with self._lock:
            if key in self._pool and self._pool[key]:
                tensor = self._pool[key].popleft()
                self._hits += 1
                
                # Zero out the tensor for safety
                with torch.no_grad():
                    tensor.zero_()
                
                return tensor
            
            self._misses += 1
            return None
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool."""
        if tensor is None:
            return
            
        key = (tensor.shape, tensor.dtype, tensor.device)
        tensor_size = tensor.numel() * tensor.element_size()
        
        with self._lock:
            # Check if we have space
            if self._current_size + tensor_size > self.max_size_bytes:
                self._evict_oldest()
            
            # Add to pool
            self._pool[key].append(tensor.detach())
            self._current_size += tensor_size
            
            logger.debug("Tensor returned to pool", 
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        pool_type=self.pool_type.value)
    
    def _evict_oldest(self):
        """Evict oldest tensors to make space."""
        while self._current_size > self.max_size_bytes * 0.8:  # Target 80% capacity
            evicted = False
            
            for key, tensor_queue in self._pool.items():
                if tensor_queue:
                    tensor = tensor_queue.popleft()
                    tensor_size = tensor.numel() * tensor.element_size()
                    self._current_size -= tensor_size
                    del tensor
                    evicted = True
                    break
            
            if not evicted:
                break
    
    def clear(self):
        """Clear the entire pool."""
        with self._lock:
            self._pool.clear()
            self._current_size = 0
            gc.collect()
            
            logger.info("Memory pool cleared", pool_type=self.pool_type.value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_type': self.pool_type.value,
                'current_size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'utilization': self._current_size / self.max_size_bytes,
                'allocations': self._allocations,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / max(1, self._hits + self._misses),
                'tensor_count': sum(len(queue) for queue in self._pool.values()),
                'unique_shapes': len(self._pool)
            }


class TensorTracker:
    """Tracks tensor allocations and detects memory leaks."""
    
    def __init__(self, max_tracked_tensors: int = 10000):
        self.max_tracked_tensors = max_tracked_tensors
        
        # Tensor tracking
        self._tracked_tensors: Dict[int, TensorInfo] = {}
        self._tensor_refs: Dict[int, weakref.ref] = {}
        self._creation_stack: Dict[int, List[str]] = {}
        
        # Leak detection
        self._potential_leaks: Set[int] = set()
        self._leak_threshold_seconds = 300  # 5 minutes
        
        # Statistics
        self._total_allocated = 0
        self._total_freed = 0
        self._peak_memory = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Tensor tracker initialized", 
                   max_tracked_tensors=max_tracked_tensors)
    
    def track_tensor(self, tensor: torch.Tensor, source_location: str = "unknown"):
        """Track a tensor allocation."""
        if tensor is None:
            return
            
        tensor_id = id(tensor)
        tensor_size = tensor.numel() * tensor.element_size()
        current_time = time.time()
        
        with self._lock:
            # Limit tracking to prevent memory issues
            if len(self._tracked_tensors) >= self.max_tracked_tensors:
                self._cleanup_old_entries()
            
            # Create tensor info
            tensor_info = TensorInfo(
                id=tensor_id,
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                size_bytes=tensor_size,
                created_at=current_time,
                last_accessed=current_time,
                ref_count=1,
                source_location=source_location
            )
            
            self._tracked_tensors[tensor_id] = tensor_info
            
            # Create weak reference for cleanup detection
            self._tensor_refs[tensor_id] = weakref.ref(tensor, self._tensor_cleanup_callback)
            
            # Update statistics
            self._total_allocated += tensor_size
            self._peak_memory = max(self._peak_memory, self._get_current_memory_usage())
    
    def _tensor_cleanup_callback(self, ref):
        """Callback when a tensor is garbage collected."""
        tensor_id = None
        
        with self._lock:
            # Find the tensor ID by reference
            for tid, tensor_ref in self._tensor_refs.items():
                if tensor_ref is ref:
                    tensor_id = tid
                    break
            
            if tensor_id and tensor_id in self._tracked_tensors:
                tensor_info = self._tracked_tensors[tensor_id]
                self._total_freed += tensor_info.size_bytes
                
                # Remove from tracking
                del self._tracked_tensors[tensor_id]
                del self._tensor_refs[tensor_id]
                
                # Remove from potential leaks
                self._potential_leaks.discard(tensor_id)
    
    def update_tensor_access(self, tensor: torch.Tensor):
        """Update last accessed time for a tensor."""
        tensor_id = id(tensor)
        current_time = time.time()
        
        with self._lock:
            if tensor_id in self._tracked_tensors:
                self._tracked_tensors[tensor_id].last_accessed = current_time
    
    def detect_leaks(self) -> List[TensorInfo]:
        """Detect potential memory leaks."""
        current_time = time.time()
        leak_candidates = []
        
        with self._lock:
            for tensor_id, tensor_info in self._tracked_tensors.items():
                # Check if tensor hasn't been accessed recently
                if (current_time - tensor_info.last_accessed) > self._leak_threshold_seconds:
                    leak_candidates.append(tensor_info)
                    self._potential_leaks.add(tensor_id)
        
        if leak_candidates:
            logger.warning("Potential memory leaks detected", 
                          leak_count=len(leak_candidates),
                          total_size_mb=sum(info.size_bytes for info in leak_candidates) / (1024 * 1024))
        
        return leak_candidates
    
    def _cleanup_old_entries(self):
        """Remove old tensor entries to prevent memory issues."""
        current_time = time.time()
        
        # Sort by creation time and remove oldest 10%
        sorted_tensors = sorted(
            self._tracked_tensors.items(),
            key=lambda item: item[1].created_at
        )
        
        cleanup_count = len(sorted_tensors) // 10
        for tensor_id, _ in sorted_tensors[:cleanup_count]:
            if tensor_id in self._tracked_tensors:
                del self._tracked_tensors[tensor_id]
            if tensor_id in self._tensor_refs:
                del self._tensor_refs[tensor_id]
            self._potential_leaks.discard(tensor_id)
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage from tracked tensors."""
        return sum(info.size_bytes for info in self._tracked_tensors.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        with self._lock:
            current_memory = self._get_current_memory_usage()
            
            return {
                'tracked_tensors': len(self._tracked_tensors),
                'potential_leaks': len(self._potential_leaks),
                'total_allocated_mb': self._total_allocated / (1024 * 1024),
                'total_freed_mb': self._total_freed / (1024 * 1024),
                'current_memory_mb': current_memory / (1024 * 1024),
                'peak_memory_mb': self._peak_memory / (1024 * 1024),
                'leak_threshold_seconds': self._leak_threshold_seconds
            }
    
    def clear(self):
        """Clear all tracking data."""
        with self._lock:
            self._tracked_tensors.clear()
            self._tensor_refs.clear()
            self._potential_leaks.clear()
            self._total_allocated = 0
            self._total_freed = 0
            self._peak_memory = 0
            
            logger.info("Tensor tracker cleared")


class MemoryManager:
    """
    Comprehensive memory management system with pools, tracking, and automatic cleanup.
    
    Features:
    - Memory pools for efficient tensor reuse
    - Tensor allocation tracking and leak detection
    - Automatic cleanup and garbage collection
    - GPU memory management
    - Performance monitoring
    """
    
    def __init__(self, 
                 enable_pools: bool = True,
                 enable_tracking: bool = True,
                 pool_sizes_mb: Optional[Dict[str, int]] = None,
                 cleanup_interval_seconds: int = 60):
        
        self.enable_pools = enable_pools
        self.enable_tracking = enable_tracking
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Default pool sizes
        if pool_sizes_mb is None:
            pool_sizes_mb = {
                'tensor_cache': 256,
                'gradient_cache': 128,
                'activation_cache': 256,
                'temporary_buffer': 128
            }
        
        # Initialize memory pools
        self._pools: Dict[MemoryPoolType, MemoryPool] = {}
        if enable_pools:
            for pool_type in MemoryPoolType:
                size_mb = pool_sizes_mb.get(pool_type.value, 128)
                self._pools[pool_type] = MemoryPool(pool_type, size_mb)
        
        # Initialize tensor tracker
        self._tracker = TensorTracker() if enable_tracking else None
        
        # Cleanup scheduling
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_active = False
        
        # Performance monitoring
        self._cleanup_count = 0
        self._last_cleanup_time = 0.0
        self._cleanup_durations = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("Memory manager initialized", 
                   enable_pools=enable_pools,
                   enable_tracking=enable_tracking,
                   pool_count=len(self._pools))
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype, device: torch.device, 
                   pool_type: MemoryPoolType = MemoryPoolType.TENSOR_CACHE) -> torch.Tensor:
        """Get a tensor from pool or create new one."""
        
        # Try to get from pool first
        if self.enable_pools and pool_type in self._pools:
            pooled_tensor = self._pools[pool_type].get_tensor(shape, dtype, device)
            if pooled_tensor is not None:
                if self.enable_tracking:
                    self._tracker.track_tensor(pooled_tensor, "memory_pool")
                return pooled_tensor
        
        # Create new tensor
        tensor = torch.empty(shape, dtype=dtype, device=device)
        
        # Track the tensor
        if self.enable_tracking:
            self._tracker.track_tensor(tensor, "new_allocation")
        
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor, 
                     pool_type: MemoryPoolType = MemoryPoolType.TENSOR_CACHE):
        """Return a tensor to the pool."""
        if not self.enable_pools or pool_type not in self._pools:
            return
            
        self._pools[pool_type].return_tensor(tensor)
    
    @contextmanager
    def temporary_tensor(self, shape: torch.Size, dtype: torch.dtype, device: torch.device):
        """Context manager for temporary tensors that are automatically returned to pool."""
        tensor = self.get_tensor(shape, dtype, device, MemoryPoolType.TEMPORARY_BUFFER)
        try:
            yield tensor
        finally:
            self.return_tensor(tensor, MemoryPoolType.TEMPORARY_BUFFER)
    
    def track_tensor(self, tensor: torch.Tensor, source_location: str = "unknown"):
        """Track a tensor allocation."""
        if self.enable_tracking and self._tracker:
            self._tracker.track_tensor(tensor, source_location)
    
    def update_tensor_access(self, tensor: torch.Tensor):
        """Update tensor access time."""
        if self.enable_tracking and self._tracker:
            self._tracker.update_tensor_access(tensor)
    
    def cleanup_memory(self, force: bool = False):
        """Perform comprehensive memory cleanup."""
        start_time = time.time()
        
        try:
            # Detect and log potential leaks
            if self.enable_tracking and self._tracker:
                leak_candidates = self._tracker.detect_leaks()
                if leak_candidates:
                    total_leaked_mb = sum(info.size_bytes for info in leak_candidates) / (1024 * 1024)
                    logger.warning("Memory leak detection", 
                                  leak_count=len(leak_candidates),
                                  total_leaked_mb=total_leaked_mb)
            
            # Clear empty cache entries in pools
            if self.enable_pools:
                for pool in self._pools.values():
                    if force:
                        pool.clear()
                    else:
                        # Only clear if pool is over 90% capacity
                        stats = pool.get_stats()
                        if stats['utilization'] > 0.9:
                            pool._evict_oldest()
            
            # PyTorch-specific cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Python garbage collection
            collected = gc.collect()
            
            # Update statistics
            cleanup_duration = time.time() - start_time
            self._cleanup_count += 1
            self._last_cleanup_time = start_time
            self._cleanup_durations.append(cleanup_duration)
            
            logger.info("Memory cleanup completed", 
                       cleanup_duration_ms=cleanup_duration * 1000,
                       objects_collected=collected,
                       force_cleanup=force)
            
        except Exception as e:
            logger.error("Error during memory cleanup", error=str(e))
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        
        # PyTorch memory stats
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_cached_mb = torch.cuda.memory_cached() / (1024 * 1024)
        else:
            gpu_memory_mb = gpu_reserved_mb = gpu_cached_mb = 0.0
        
        # System memory stats
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Pool usage stats
        pool_usage_mb = {}
        if self.enable_pools:
            for pool_type, pool in self._pools.items():
                stats = pool.get_stats()
                pool_usage_mb[pool_type.value] = stats['current_size_mb']
        
        # Tracking stats
        active_tensors = 0
        leak_candidates = 0
        if self.enable_tracking and self._tracker:
            tracker_stats = self._tracker.get_stats()
            active_tensors = tracker_stats['tracked_tensors']
            leak_candidates = tracker_stats['potential_leaks']
        
        return MemoryStats(
            total_allocated_mb=gpu_memory_mb + cpu_memory_mb,
            total_cached_mb=gpu_cached_mb,
            total_reserved_mb=gpu_reserved_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_memory_mb=cpu_memory_mb,
            pool_usage_mb=pool_usage_mb,
            active_tensors=active_tensors,
            leak_candidates=leak_candidates,
            last_cleanup_time=self._last_cleanup_time
        )
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        memory_stats = self.get_memory_stats()
        
        # Pool statistics
        pool_stats = {}
        if self.enable_pools:
            for pool_type, pool in self._pools.items():
                pool_stats[pool_type.value] = pool.get_stats()
        
        # Tracker statistics
        tracker_stats = {}
        if self.enable_tracking and self._tracker:
            tracker_stats = self._tracker.get_stats()
        
        # Cleanup statistics
        cleanup_stats = {
            'cleanup_count': self._cleanup_count,
            'last_cleanup_time': self._last_cleanup_time,
            'avg_cleanup_duration_ms': (
                sum(self._cleanup_durations) / len(self._cleanup_durations) * 1000
                if self._cleanup_durations else 0
            ),
            'cleanup_interval_seconds': self.cleanup_interval_seconds
        }
        
        return {
            'memory_stats': memory_stats.__dict__,
            'pool_stats': pool_stats,
            'tracker_stats': tracker_stats,
            'cleanup_stats': cleanup_stats,
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'python_gc_counts': gc.get_count()
            }
        }
    
    def _start_cleanup_thread(self):
        """Start the periodic cleanup thread."""
        if self.cleanup_interval_seconds <= 0:
            return
            
        self._cleanup_active = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("Memory cleanup thread started", 
                   interval_seconds=self.cleanup_interval_seconds)
    
    def _cleanup_loop(self):
        """Periodic cleanup loop."""
        while self._cleanup_active:
            try:
                time.sleep(self.cleanup_interval_seconds)
                if self._cleanup_active:
                    self.cleanup_memory(force=False)
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
    
    def stop(self):
        """Stop the memory manager and cleanup threads."""
        self._cleanup_active = False
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        # Final cleanup
        self.cleanup_memory(force=True)
        
        logger.info("Memory manager stopped")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    
    return _global_memory_manager


def set_memory_manager(manager: MemoryManager):
    """Set the global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.stop()
    
    _global_memory_manager = manager


# Convenience functions
def get_tensor(shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get a tensor from the global memory manager."""
    return get_memory_manager().get_tensor(shape, dtype, device)


def return_tensor(tensor: torch.Tensor):
    """Return a tensor to the global memory manager."""
    get_memory_manager().return_tensor(tensor)


def temporary_tensor(shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """Context manager for temporary tensors."""
    return get_memory_manager().temporary_tensor(shape, dtype, device)


def cleanup_memory(force: bool = False):
    """Perform global memory cleanup."""
    get_memory_manager().cleanup_memory(force)


def get_memory_stats() -> MemoryStats:
    """Get global memory statistics."""
    return get_memory_manager().get_memory_stats()