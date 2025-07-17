"""
Memory Management System for NQ Data Pipeline

Provides shared memory pools, intelligent caching, and memory optimization
for efficient data processing across multiple notebooks.
"""

import os
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
import mmap
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor
import weakref

class MemoryMonitor:
    """Monitor system memory usage and performance"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        self.memory_history = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous memory monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            memory_info = self.get_memory_info()
            self.memory_history.append(memory_info)
            
            # Keep only last 100 measurements
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
            
            # Check thresholds
            if memory_info['usage_percent'] > self.critical_threshold:
                self.logger.critical(f"Critical memory usage: {memory_info['usage_percent']:.1%}")
                # Trigger emergency cleanup
                self._emergency_cleanup()
            elif memory_info['usage_percent'] > self.warning_threshold:
                self.logger.warning(f"High memory usage: {memory_info['usage_percent']:.1%}")
            
            time.sleep(interval)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent / 100,
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'gpu_memory_gb': self._get_gpu_memory() if torch.cuda.is_available() else 0,
            'timestamp': time.time()
        }
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return 0
        
        try:
            return torch.cuda.memory_allocated() / (1024**3)
        except:
            return 0
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        self.logger.warning("Triggering emergency memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear torch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear weak references
        for obj in list(weakref.WeakSet()):
            if hasattr(obj, 'clear_cache'):
                obj.clear_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.memory_history:
            return {}
        
        usage_history = [entry['usage_percent'] for entry in self.memory_history]
        
        return {
            'current_usage': self.memory_history[-1]['usage_percent'],
            'avg_usage': np.mean(usage_history),
            'max_usage': max(usage_history),
            'min_usage': min(usage_history),
            'measurements': len(self.memory_history),
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold
        }

@dataclass
class SharedDataObject:
    """Shared data object with metadata"""
    data: Any
    key: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1

class SharedMemoryPool:
    """Shared memory pool for efficient data sharing between processes"""
    
    def __init__(self, 
                 max_size_gb: float = 4.0,
                 eviction_policy: str = "lru",
                 enable_persistence: bool = True,
                 persistence_dir: Optional[str] = None):
        """
        Initialize shared memory pool
        
        Args:
            max_size_gb: Maximum memory pool size in GB
            eviction_policy: Eviction policy ('lru', 'lfu', 'fifo')
            enable_persistence: Enable disk persistence
            persistence_dir: Directory for persistence files
        """
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.eviction_policy = eviction_policy
        self.enable_persistence = enable_persistence
        
        # Storage
        self.data_objects: Dict[str, SharedDataObject] = {}
        self.current_size_bytes = 0
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Persistence
        if enable_persistence:
            self.persistence_dir = Path(persistence_dir or "/tmp/shared_memory_pool")
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'stores': 0,
            'persistence_loads': 0,
            'persistence_saves': 0
        }
    
    def store(self, key: str, data: Any, force: bool = False) -> bool:
        """Store data in shared memory pool"""
        with self.lock:
            # Calculate data size
            data_size = self._calculate_size(data)
            
            # Check if data already exists
            if key in self.data_objects:
                if not force:
                    return False
                
                # Remove existing data
                self._remove_data(key)
            
            # Ensure enough space
            if not self._ensure_space(data_size):
                self.logger.warning(f"Cannot store {key}: insufficient space")
                return False
            
            # Store data
            shared_obj = SharedDataObject(
                data=data,
                key=key,
                size_bytes=data_size,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1
            )
            
            self.data_objects[key] = shared_obj
            self.current_size_bytes += data_size
            self.stats['stores'] += 1
            
            # Persist if enabled
            if self.enable_persistence:
                self._persist_data(key, data)
            
            self.logger.info(f"Stored {key} ({data_size / 1024**2:.1f} MB)")
            return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from shared memory pool"""
        with self.lock:
            if key in self.data_objects:
                # Cache hit
                shared_obj = self.data_objects[key]
                shared_obj.update_access()
                self.stats['hits'] += 1
                return shared_obj.data
            
            # Cache miss - try to load from persistence
            if self.enable_persistence:
                data = self._load_from_persistence(key)
                if data is not None:
                    # Store in memory
                    self.store(key, data)
                    self.stats['persistence_loads'] += 1
                    return data
            
            self.stats['misses'] += 1
            return None
    
    def exists(self, key: str) -> bool:
        """Check if key exists in pool"""
        with self.lock:
            return key in self.data_objects
    
    def remove(self, key: str) -> bool:
        """Remove data from pool"""
        with self.lock:
            if key in self.data_objects:
                self._remove_data(key)
                return True
            return False
    
    def clear(self):
        """Clear all data from pool"""
        with self.lock:
            self.data_objects.clear()
            self.current_size_bytes = 0
            self.logger.info("Shared memory pool cleared")
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data object"""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        else:
            # Fallback to pickle size
            try:
                return len(pickle.dumps(data))
            except:
                return 1024  # Default 1KB
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure enough space is available"""
        while self.current_size_bytes + required_bytes > self.max_size_bytes:
            if not self._evict_object():
                return False
        return True
    
    def _evict_object(self) -> bool:
        """Evict an object based on policy"""
        if not self.data_objects:
            return False
        
        # Select victim based on policy
        if self.eviction_policy == "lru":
            victim_key = min(self.data_objects.keys(), 
                           key=lambda k: self.data_objects[k].last_accessed)
        elif self.eviction_policy == "lfu":
            victim_key = min(self.data_objects.keys(), 
                           key=lambda k: self.data_objects[k].access_count)
        elif self.eviction_policy == "fifo":
            victim_key = min(self.data_objects.keys(), 
                           key=lambda k: self.data_objects[k].created_at)
        else:
            # Default to LRU
            victim_key = min(self.data_objects.keys(), 
                           key=lambda k: self.data_objects[k].last_accessed)
        
        self._remove_data(victim_key)
        self.stats['evictions'] += 1
        return True
    
    def _remove_data(self, key: str):
        """Remove data object from pool"""
        if key in self.data_objects:
            shared_obj = self.data_objects[key]
            self.current_size_bytes -= shared_obj.size_bytes
            del self.data_objects[key]
            self.logger.debug(f"Removed {key} ({shared_obj.size_bytes / 1024**2:.1f} MB)")
    
    def _persist_data(self, key: str, data: Any):
        """Persist data to disk"""
        if not self.enable_persistence:
            return
        
        try:
            filepath = self.persistence_dir / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.stats['persistence_saves'] += 1
        except Exception as e:
            self.logger.warning(f"Failed to persist {key}: {e}")
    
    def _load_from_persistence(self, key: str) -> Optional[Any]:
        """Load data from persistence"""
        if not self.enable_persistence:
            return None
        
        try:
            filepath = self.persistence_dir / f"{key}.pkl"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load {key} from persistence: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            
            return {
                'objects_count': len(self.data_objects),
                'current_size_mb': self.current_size_bytes / 1024**2,
                'max_size_mb': self.max_size_bytes / 1024**2,
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'stores': self.stats['stores'],
                'persistence_loads': self.stats['persistence_loads'],
                'persistence_saves': self.stats['persistence_saves']
            }

class MemoryMappedDataLoader:
    """Memory-mapped data loader for efficient large file access"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.mmap_file = None
        self.file_handle = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """Context manager entry"""
        self.file_handle = open(self.file_path, 'rb')
        self.mmap_file = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.mmap_file:
            self.mmap_file.close()
        if self.file_handle:
            self.file_handle.close()
    
    def read_chunk(self, offset: int, size: int) -> bytes:
        """Read a chunk of data from memory-mapped file"""
        if not self.mmap_file:
            raise RuntimeError("File not opened. Use as context manager.")
        
        self.mmap_file.seek(offset)
        return self.mmap_file.read(size)
    
    def get_file_size(self) -> int:
        """Get file size in bytes"""
        return self.file_path.stat().st_size

class MemoryManager:
    """Central memory management system"""
    
    def __init__(self, 
                 shared_pool_size_gb: float = 4.0,
                 enable_monitoring: bool = True,
                 monitoring_interval: float = 5.0):
        """
        Initialize memory manager
        
        Args:
            shared_pool_size_gb: Size of shared memory pool in GB
            enable_monitoring: Enable memory monitoring
            monitoring_interval: Monitoring interval in seconds
        """
        self.shared_pool = SharedMemoryPool(max_size_gb=shared_pool_size_gb)
        self.monitor = MemoryMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Memory optimization settings
        self.optimization_enabled = True
        self.auto_cleanup_threshold = 0.8
        
        # Start monitoring if enabled
        if enable_monitoring:
            self.monitor.start_monitoring(monitoring_interval)
        
        self.logger.info("Memory manager initialized")
    
    def store_data(self, key: str, data: Any, force: bool = False) -> bool:
        """Store data in shared pool"""
        return self.shared_pool.store(key, data, force)
    
    def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data from shared pool"""
        return self.shared_pool.retrieve(key)
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if not self.optimization_enabled:
            return
        
        self.logger.info("Starting memory optimization")
        
        # Force garbage collection
        gc.collect()
        
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if cleanup is needed
        memory_info = self.monitor.get_memory_info()
        if memory_info['usage_percent'] > self.auto_cleanup_threshold:
            self.logger.warning("High memory usage detected, triggering cleanup")
            self._aggressive_cleanup()
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Clear some shared pool objects
        pool_stats = self.shared_pool.get_stats()
        if pool_stats['utilization'] > 0.8:
            # Evict half of the objects
            keys_to_remove = list(self.shared_pool.data_objects.keys())[:len(self.shared_pool.data_objects) // 2]
            for key in keys_to_remove:
                self.shared_pool.remove(key)
        
        # Force multiple garbage collections
        for _ in range(3):
            gc.collect()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        memory_info = self.monitor.get_memory_info()
        memory_stats = self.monitor.get_memory_stats()
        pool_stats = self.shared_pool.get_stats()
        
        return {
            'system_memory': memory_info,
            'memory_history': memory_stats,
            'shared_pool': pool_stats,
            'optimization_enabled': self.optimization_enabled,
            'auto_cleanup_threshold': self.auto_cleanup_threshold
        }
    
    def set_optimization_level(self, level: str):
        """Set memory optimization level"""
        if level == "aggressive":
            self.auto_cleanup_threshold = 0.7
            self.optimization_enabled = True
        elif level == "moderate":
            self.auto_cleanup_threshold = 0.8
            self.optimization_enabled = True
        elif level == "conservative":
            self.auto_cleanup_threshold = 0.9
            self.optimization_enabled = True
        elif level == "disabled":
            self.optimization_enabled = False
        else:
            raise ValueError(f"Invalid optimization level: {level}")
        
        self.logger.info(f"Memory optimization level set to: {level}")
    
    def create_memory_mapped_loader(self, file_path: str) -> MemoryMappedDataLoader:
        """Create memory-mapped data loader"""
        return MemoryMappedDataLoader(file_path)
    
    def cleanup(self):
        """Cleanup memory manager"""
        self.monitor.stop_monitoring()
        self.shared_pool.clear()
        self.logger.info("Memory manager cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()