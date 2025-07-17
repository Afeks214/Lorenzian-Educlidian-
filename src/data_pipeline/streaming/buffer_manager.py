"""
Buffer management for efficient data streaming
"""

import threading
import queue
import time
import weakref
from typing import Any, Dict, List, Optional, Callable, Iterator
from dataclasses import dataclass, field
from collections import deque
import psutil
import gc
import logging

logger = logging.getLogger(__name__)

@dataclass
class BufferConfig:
    """Configuration for buffer management"""
    max_buffer_size: int = 1000
    high_water_mark: float = 0.8  # 80% of buffer size
    low_water_mark: float = 0.2   # 20% of buffer size
    memory_threshold_mb: float = 500.0
    enable_compression: bool = True
    enable_spill_to_disk: bool = True
    spill_directory: str = "/tmp/data_pipeline_spill"
    cleanup_interval: int = 60  # seconds

@dataclass
class BufferItem:
    """Item stored in buffer"""
    data: Any
    timestamp: float
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of data"""
        try:
            if hasattr(self.data, 'memory_usage'):
                return int(self.data.memory_usage(deep=True).sum())
            else:
                import sys
                return sys.getsizeof(self.data)
        except Exception:
            return 1024  # Default size

class BufferManager:
    """
    Advanced buffer manager with memory management and disk spilling
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self._buffer: deque = deque()
        self._buffer_dict: Dict[str, BufferItem] = {}
        self._lock = threading.RLock()
        self._stats = BufferStats()
        
        # Memory monitoring
        self._memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self._memory_monitor.start()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        # State
        self._is_running = True
        self._spill_manager = SpillManager(self.config.spill_directory)
    
    def put(self, key: str, data: Any, priority: int = 0) -> bool:
        """
        Add data to buffer with optional priority
        """
        with self._lock:
            try:
                # Check if buffer is full
                if len(self._buffer) >= self.config.max_buffer_size:
                    if not self._make_space():
                        return False
                
                # Create buffer item
                item = BufferItem(
                    data=data,
                    timestamp=time.time(),
                    size_bytes=0  # Will be calculated in __post_init__
                )
                
                # Add to buffer
                self._buffer.append((key, item, priority))
                self._buffer_dict[key] = item
                
                # Update stats
                self._stats.items_added += 1
                self._stats.total_size_bytes += item.size_bytes
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding item to buffer: {str(e)}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get data from buffer
        """
        with self._lock:
            if key in self._buffer_dict:
                item = self._buffer_dict[key]
                item.access_count += 1
                item.last_access = time.time()
                self._stats.items_retrieved += 1
                return item.data
            
            # Check if data is spilled to disk
            return self._spill_manager.get(key)
    
    def contains(self, key: str) -> bool:
        """Check if key exists in buffer"""
        with self._lock:
            return key in self._buffer_dict or self._spill_manager.contains(key)
    
    def remove(self, key: str) -> bool:
        """Remove item from buffer"""
        with self._lock:
            if key in self._buffer_dict:
                item = self._buffer_dict[key]
                del self._buffer_dict[key]
                
                # Remove from deque (expensive operation)
                self._buffer = deque(
                    (k, i, p) for k, i, p in self._buffer if k != key
                )
                
                # Update stats
                self._stats.items_removed += 1
                self._stats.total_size_bytes -= item.size_bytes
                
                return True
            
            # Try to remove from spill
            return self._spill_manager.remove(key)
    
    def _make_space(self) -> bool:
        """Make space in buffer by removing items"""
        target_size = int(self.config.max_buffer_size * self.config.low_water_mark)
        items_to_remove = []
        
        # Sort by access patterns (LRU with access count)
        sorted_items = sorted(
            self._buffer,
            key=lambda x: (x[1].access_count, x[1].last_access)
        )
        
        current_size = len(self._buffer)
        for key, item, priority in sorted_items:
            if current_size <= target_size:
                break
            
            # Try to spill to disk first
            if self.config.enable_spill_to_disk:
                if self._spill_manager.spill(key, item.data):
                    items_to_remove.append(key)
                    current_size -= 1
                    continue
            
            # Otherwise, just remove
            items_to_remove.append(key)
            current_size -= 1
        
        # Remove items
        for key in items_to_remove:
            if key in self._buffer_dict:
                item = self._buffer_dict[key]
                del self._buffer_dict[key]
                self._stats.total_size_bytes -= item.size_bytes
        
        # Rebuild deque without removed items
        self._buffer = deque(
            (k, i, p) for k, i, p in self._buffer if k not in items_to_remove
        )
        
        self._stats.items_evicted += len(items_to_remove)
        return True
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup"""
        while self._is_running:
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_mb > self.config.memory_threshold_mb:
                    logger.warning(f"Memory usage high: {memory_mb:.2f}MB")
                    self._force_cleanup()
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in memory monitor: {str(e)}")
                time.sleep(10)
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self._is_running:
            try:
                self._cleanup_old_items()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}")
                time.sleep(30)
    
    def _cleanup_old_items(self):
        """Clean up old items from buffer"""
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour
        
        with self._lock:
            keys_to_remove = []
            for key, item in self._buffer_dict.items():
                if current_time - item.last_access > cleanup_threshold:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.remove(key)
    
    def _force_cleanup(self):
        """Force cleanup of buffer"""
        with self._lock:
            # Remove half of the buffer
            target_size = len(self._buffer) // 2
            self._make_space_to_size(target_size)
            
            # Force garbage collection
            gc.collect()
    
    def _make_space_to_size(self, target_size: int):
        """Make space to reach target size"""
        while len(self._buffer) > target_size:
            if not self._make_space():
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                'buffer_size': len(self._buffer),
                'max_buffer_size': self.config.max_buffer_size,
                'buffer_utilization': len(self._buffer) / self.config.max_buffer_size,
                'total_size_bytes': self._stats.total_size_bytes,
                'items_added': self._stats.items_added,
                'items_retrieved': self._stats.items_retrieved,
                'items_removed': self._stats.items_removed,
                'items_evicted': self._stats.items_evicted,
                'hit_rate': self._stats.get_hit_rate(),
                'spill_stats': self._spill_manager.get_stats()
            }
    
    def clear(self):
        """Clear all buffer contents"""
        with self._lock:
            self._buffer.clear()
            self._buffer_dict.clear()
            self._stats.reset()
            self._spill_manager.clear()
    
    def shutdown(self):
        """Shutdown buffer manager"""
        self._is_running = False
        self._spill_manager.shutdown()
        logger.info("Buffer manager shutdown")


class BufferStats:
    """Statistics for buffer operations"""
    
    def __init__(self):
        self.items_added = 0
        self.items_retrieved = 0
        self.items_removed = 0
        self.items_evicted = 0
        self.total_size_bytes = 0
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self) -> float:
        """Calculate hit rate"""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.items_added = 0
        self.items_retrieved = 0
        self.items_removed = 0
        self.items_evicted = 0
        self.total_size_bytes = 0
        self.hits = 0
        self.misses = 0


class SpillManager:
    """Manage spilling data to disk"""
    
    def __init__(self, spill_directory: str):
        self.spill_directory = spill_directory
        self.spilled_items: Dict[str, str] = {}  # key -> file_path
        self._lock = threading.Lock()
        
        # Create spill directory
        import os
        os.makedirs(spill_directory, exist_ok=True)
    
    def spill(self, key: str, data: Any) -> bool:
        """Spill data to disk"""
        try:
            import pickle
            import os
            
            file_path = os.path.join(self.spill_directory, f"{key}.pkl")
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            with self._lock:
                self.spilled_items[key] = file_path
            
            return True
        except Exception as e:
            logger.error(f"Error spilling data to disk: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get spilled data from disk"""
        with self._lock:
            if key not in self.spilled_items:
                return None
            
            file_path = self.spilled_items[key]
        
        try:
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading spilled data: {str(e)}")
            return None
    
    def contains(self, key: str) -> bool:
        """Check if key is spilled"""
        with self._lock:
            return key in self.spilled_items
    
    def remove(self, key: str) -> bool:
        """Remove spilled data"""
        with self._lock:
            if key not in self.spilled_items:
                return False
            
            file_path = self.spilled_items[key]
            del self.spilled_items[key]
        
        try:
            import os
            os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error removing spilled file: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spill statistics"""
        with self._lock:
            return {
                'spilled_items': len(self.spilled_items),
                'spill_directory': self.spill_directory
            }
    
    def clear(self):
        """Clear all spilled data"""
        with self._lock:
            import os
            for file_path in self.spilled_items.values():
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing spilled file {file_path}: {str(e)}")
            
            self.spilled_items.clear()
    
    def shutdown(self):
        """Shutdown spill manager"""
        self.clear()


class CircularBuffer:
    """Circular buffer for fixed-size streaming data"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.head = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.Lock()
    
    def put(self, item: Any) -> bool:
        """Add item to circular buffer"""
        with self._lock:
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.max_size
            
            if self.size < self.max_size:
                self.size += 1
            else:
                # Buffer is full, move head
                self.head = (self.head + 1) % self.max_size
            
            return True
    
    def get(self) -> Optional[Any]:
        """Get item from circular buffer"""
        with self._lock:
            if self.size == 0:
                return None
            
            item = self.buffer[self.head]
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.max_size
            self.size -= 1
            
            return item
    
    def peek(self) -> Optional[Any]:
        """Peek at next item without removing it"""
        with self._lock:
            if self.size == 0:
                return None
            return self.buffer[self.head]
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self.size == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self._lock:
            return self.size == self.max_size
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.buffer = [None] * self.max_size
            self.head = 0
            self.tail = 0
            self.size = 0


class PriorityBuffer:
    """Priority-based buffer for streaming data"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
        self._lock = threading.Lock()
    
    def put(self, item: Any, priority: int) -> bool:
        """Add item with priority"""
        with self._lock:
            import heapq
            
            if len(self.buffer) >= self.max_size:
                # Remove lowest priority item
                heapq.heappop(self.buffer)
            
            heapq.heappush(self.buffer, (priority, time.time(), item))
            return True
    
    def get(self) -> Optional[Any]:
        """Get highest priority item"""
        with self._lock:
            import heapq
            
            if not self.buffer:
                return None
            
            _, _, item = heapq.heappop(self.buffer)
            return item
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return len(self.buffer) == 0
    
    def size(self) -> int:
        """Get buffer size"""
        with self._lock:
            return len(self.buffer)