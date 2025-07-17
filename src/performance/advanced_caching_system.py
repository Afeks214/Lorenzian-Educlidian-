#!/usr/bin/env python3
"""
Advanced Caching System for Performance Optimization
Implements multi-level caching with intelligent eviction, memory pooling, and async processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, TypeVar, Generic, Callable
import threading
import asyncio
import hashlib
import pickle
import weakref
import gc
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from functools import lru_cache, wraps
import logging
from pathlib import Path
import json
import sqlite3
from abc import ABC, abstractmethod
import heapq
import zlib
import redis
from enum import Enum
import multiprocessing as mp

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class CacheLevel(Enum):
    """Cache levels for hierarchical caching"""
    L1_MEMORY = "l1_memory"        # Fastest, smallest capacity
    L2_COMPRESSED = "l2_compressed" # Medium speed, medium capacity
    L3_DISK = "l3_disk"            # Slowest, largest capacity
    L4_REDIS = "l4_redis"          # Distributed cache

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    FIFO = "fifo"                  # First In, First Out
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns

@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: float = 0.0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    compression_ratio: float = 0.0
    total_requests: int = 0

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    compressed: bool = False
    ttl: Optional[float] = None
    priority: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access information"""
        self.last_accessed = time.time()
        self.access_count += 1

class TensorMemoryPool:
    """Memory pool for tensor allocation optimization"""
    
    def __init__(self, max_pool_size: int = 1000):
        self.max_pool_size = max_pool_size
        self.tensor_pools = defaultdict(deque)
        self.lock = threading.RLock()
        self.stats = {
            'pool_hits': 0,
            'pool_misses': 0,
            'current_size': 0,
            'memory_saved_mb': 0.0
        }
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   device: str = 'cpu') -> torch.Tensor:
        """Get tensor from pool or create new one"""
        key = (shape, dtype, device)
        
        with self.lock:
            if key in self.tensor_pools and self.tensor_pools[key]:
                tensor = self.tensor_pools[key].popleft()
                tensor.zero_()
                self.stats['pool_hits'] += 1
                return tensor
        
        # Create new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.stats['pool_misses'] += 1
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if tensor.device.type == 'cpu':  # Only pool CPU tensors
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            device = str(tensor.device)
            key = (shape, dtype, device)
            
            with self.lock:
                if len(self.tensor_pools[key]) < self.max_pool_size:
                    self.tensor_pools[key].append(tensor)
                    self.stats['current_size'] += 1
                    # Estimate memory saved
                    tensor_size = tensor.numel() * tensor.element_size()
                    self.stats['memory_saved_mb'] += tensor_size / (1024 * 1024)
    
    def cleanup(self):
        """Clean up memory pools"""
        with self.lock:
            for pool in self.tensor_pools.values():
                pool.clear()
            self.tensor_pools.clear()
            self.stats['current_size'] = 0
        gc.collect()

class CompressedStorage:
    """Compressed storage for large cache entries"""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.compression_stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'avg_compression_ratio': 0.0
        }
    
    def compress(self, data: Any) -> bytes:
        """Compress data using zlib"""
        try:
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            compressed = zlib.compress(serialized, self.compression_level)
            compressed_size = len(compressed)
            
            # Update stats
            self.compression_stats['total_compressed'] += 1
            self.compression_stats['total_original_size'] += original_size
            self.compression_stats['total_compressed_size'] += compressed_size
            self.compression_stats['avg_compression_ratio'] = (
                self.compression_stats['total_original_size'] / 
                self.compression_stats['total_compressed_size']
            )
            
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return pickle.dumps(data)
    
    def decompress(self, data: bytes) -> Any:
        """Decompress data using zlib"""
        try:
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except zlib.error:
            # Fallback to direct pickle loading
            return pickle.loads(data)

class L1MemoryCache:
    """L1 memory cache with LRU eviction"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    del self.cache[key]
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.touch()
                self.stats.hits += 1
                return entry.value
            
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache"""
        with self.lock:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                oldest_key, _ = self.cache.popitem(last=False)
                self.stats.evictions += 1
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=self._estimate_size(value)
            )
            self.cache[key] = entry
            self._update_stats()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except:
            return 0
    
    def _update_stats(self):
        """Update cache statistics"""
        self.stats.total_requests = self.stats.hits + self.stats.misses
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests
        
        total_memory = sum(entry.size_bytes for entry in self.cache.values())
        self.stats.memory_usage = total_memory / (1024 * 1024)  # MB
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()

class L2CompressedCache:
    """L2 compressed cache with LFU eviction"""
    
    def __init__(self, capacity: int = 500, compression_level: int = 6):
        self.capacity = capacity
        self.cache = {}
        self.frequencies = defaultdict(int)
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.compressor = CompressedStorage(compression_level)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    del self.cache[key]
                    del self.frequencies[key]
                    return None
                
                entry.touch()
                self.frequencies[key] += 1
                self.stats.hits += 1
                
                # Decompress if needed
                if entry.compressed:
                    return self.compressor.decompress(entry.value)
                return entry.value
            
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache"""
        with self.lock:
            if len(self.cache) >= self.capacity:
                # Remove least frequently used item
                lfu_key = min(self.frequencies, key=self.frequencies.get)
                del self.cache[lfu_key]
                del self.frequencies[lfu_key]
                self.stats.evictions += 1
            
            # Compress large values
            compressed_value = self.compressor.compress(value)
            
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=len(compressed_value),
                compressed=True
            )
            
            self.cache[key] = entry
            self.frequencies[key] = 1
            self._update_stats()
    
    def _update_stats(self):
        """Update cache statistics"""
        self.stats.total_requests = self.stats.hits + self.stats.misses
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests
        
        total_memory = sum(entry.size_bytes for entry in self.cache.values())
        self.stats.memory_usage = total_memory / (1024 * 1024)  # MB
        self.stats.compression_ratio = self.compressor.compression_stats['avg_compression_ratio']

class L3DiskCache:
    """L3 disk-based cache with SQLite backend"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.db_path = self.cache_dir / "cache.db"
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.compressor = CompressedStorage()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    size_bytes INTEGER,
                    ttl REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT value, timestamp, ttl FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        value_blob, timestamp, ttl = row
                        
                        # Check expiration
                        if ttl and time.time() - timestamp > ttl:
                            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                            return None
                        
                        # Update access information
                        conn.execute("""
                            UPDATE cache_entries 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE key = ?
                        """, (time.time(), key))
                        
                        self.stats.hits += 1
                        return self.compressor.decompress(value_blob)
                    
                    self.stats.misses += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in disk cache"""
        with self.lock:
            try:
                compressed_value = self.compressor.compress(value)
                size_bytes = len(compressed_value)
                
                with sqlite3.connect(str(self.db_path)) as conn:
                    # Check if we need to evict entries
                    self._evict_if_needed(conn, size_bytes)
                    
                    # Insert or replace entry
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, timestamp, access_count, last_accessed, size_bytes, ttl)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, compressed_value, time.time(), 1, time.time(), size_bytes, ttl
                    ))
                    
                    self._update_stats()
                    
            except Exception as e:
                logger.error(f"Disk cache put error: {e}")
    
    def _evict_if_needed(self, conn: sqlite3.Connection, new_size: int):
        """Evict entries if cache is too large"""
        # Check current size
        cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
        current_size = cursor.fetchone()[0] or 0
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if current_size + new_size > max_size_bytes:
            # Evict least recently used entries
            cursor = conn.execute("""
                SELECT key, size_bytes FROM cache_entries 
                ORDER BY last_accessed ASC
            """)
            
            for key, size_bytes in cursor:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                current_size -= size_bytes
                self.stats.evictions += 1
                
                if current_size + new_size <= max_size_bytes:
                    break
    
    def _update_stats(self):
        """Update cache statistics"""
        self.stats.total_requests = self.stats.hits + self.stats.misses
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests

class MultiLevelCache:
    """Multi-level hierarchical cache system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tensor_pool = TensorMemoryPool(config.get('tensor_pool_size', 1000))
        
        # Initialize cache levels
        self.l1_cache = L1MemoryCache(config.get('l1_capacity', 1000))
        self.l2_cache = L2CompressedCache(
            config.get('l2_capacity', 500),
            config.get('compression_level', 6)
        )
        self.l3_cache = L3DiskCache(
            Path(config.get('cache_dir', '/tmp/grandmodel_cache')),
            config.get('l3_max_size_mb', 1000)
        )
        
        # Redis cache (optional)
        self.redis_cache = None
        if config.get('enable_redis', False):
            try:
                import redis
                self.redis_cache = redis.Redis(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    db=config.get('redis_db', 0)
                )
            except ImportError:
                logger.warning("Redis not available, skipping L4 cache")
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        start_time = time.time()
        
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.performance_monitor.record_access('l1', time.time() - start_time)
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value)
            self.performance_monitor.record_access('l2', time.time() - start_time)
            return value
        
        # Try L3 cache
        value = self.l3_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            self.l2_cache.put(key, value)
            self.l1_cache.put(key, value)
            self.performance_monitor.record_access('l3', time.time() - start_time)
            return value
        
        # Try Redis cache
        if self.redis_cache:
            try:
                redis_value = self.redis_cache.get(key)
                if redis_value:
                    value = pickle.loads(redis_value)
                    # Promote through all levels
                    self.l3_cache.put(key, value)
                    self.l2_cache.put(key, value)
                    self.l1_cache.put(key, value)
                    self.performance_monitor.record_access('redis', time.time() - start_time)
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        self.performance_monitor.record_miss(time.time() - start_time)
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache hierarchy"""
        # Always store in L1
        self.l1_cache.put(key, value, ttl)
        
        # Store in L2 for medium-term caching
        self.l2_cache.put(key, value, ttl)
        
        # Store in L3 for long-term caching
        self.l3_cache.put(key, value, ttl)
        
        # Store in Redis if available
        if self.redis_cache:
            try:
                serialized = pickle.dumps(value)
                if ttl:
                    self.redis_cache.setex(key, int(ttl), serialized)
                else:
                    self.redis_cache.set(key, serialized)
            except Exception as e:
                logger.warning(f"Redis cache put error: {e}")
    
    def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        self.l1_cache.cache.pop(key, None)
        self.l2_cache.cache.pop(key, None)
        if self.redis_cache:
            try:
                self.redis_cache.delete(key)
            except Exception as e:
                logger.warning(f"Redis invalidate error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'l1_stats': self.l1_cache.stats.__dict__,
            'l2_stats': self.l2_cache.stats.__dict__,
            'l3_stats': self.l3_cache.stats.__dict__,
            'tensor_pool_stats': self.tensor_pool.stats,
            'performance_stats': self.performance_monitor.get_stats()
        }
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                # Clean up expired entries
                current_time = time.time()
                
                # Clean L1 cache
                with self.l1_cache.lock:
                    expired_keys = [
                        key for key, entry in self.l1_cache.cache.items()
                        if entry.is_expired
                    ]
                    for key in expired_keys:
                        del self.l1_cache.cache[key]
                
                # Clean L2 cache
                with self.l2_cache.lock:
                    expired_keys = [
                        key for key, entry in self.l2_cache.cache.items()
                        if entry.is_expired
                    ]
                    for key in expired_keys:
                        del self.l2_cache.cache[key]
                        del self.l2_cache.frequencies[key]
                
                # Trigger garbage collection if memory usage is high
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 80:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")

class PerformanceMonitor:
    """Monitor cache performance and access patterns"""
    
    def __init__(self):
        self.access_times = defaultdict(deque)
        self.access_counts = defaultdict(int)
        self.miss_count = 0
        self.miss_time = 0.0
        self.lock = threading.RLock()
    
    def record_access(self, level: str, access_time: float):
        """Record cache access"""
        with self.lock:
            self.access_times[level].append(access_time)
            self.access_counts[level] += 1
            
            # Keep only last 1000 measurements
            if len(self.access_times[level]) > 1000:
                self.access_times[level].popleft()
    
    def record_miss(self, miss_time: float):
        """Record cache miss"""
        with self.lock:
            self.miss_count += 1
            self.miss_time += miss_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            
            for level, times in self.access_times.items():
                if times:
                    stats[f'{level}_avg_time'] = sum(times) / len(times)
                    stats[f'{level}_count'] = self.access_counts[level]
                else:
                    stats[f'{level}_avg_time'] = 0.0
                    stats[f'{level}_count'] = 0
            
            total_accesses = sum(self.access_counts.values()) + self.miss_count
            if total_accesses > 0:
                stats['overall_hit_rate'] = (
                    sum(self.access_counts.values()) / total_accesses
                )
            else:
                stats['overall_hit_rate'] = 0.0
                
            stats['miss_count'] = self.miss_count
            stats['avg_miss_time'] = (
                self.miss_time / self.miss_count if self.miss_count > 0 else 0.0
            )
            
            return stats

class CacheKeyGenerator:
    """Generate cache keys for different data types"""
    
    @staticmethod
    def generate_key(data: Any, prefix: str = "") -> str:
        """Generate cache key for arbitrary data"""
        if isinstance(data, torch.Tensor):
            return CacheKeyGenerator.tensor_key(data, prefix)
        elif isinstance(data, np.ndarray):
            return CacheKeyGenerator.array_key(data, prefix)
        elif isinstance(data, dict):
            return CacheKeyGenerator.dict_key(data, prefix)
        else:
            return CacheKeyGenerator.generic_key(data, prefix)
    
    @staticmethod
    def tensor_key(tensor: torch.Tensor, prefix: str = "") -> str:
        """Generate key for tensor"""
        shape_str = "_".join(map(str, tensor.shape))
        dtype_str = str(tensor.dtype).replace("torch.", "")
        device_str = str(tensor.device)
        
        # Use hash of tensor data for uniqueness
        tensor_hash = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()[:8]
        
        return f"{prefix}_tensor_{shape_str}_{dtype_str}_{device_str}_{tensor_hash}"
    
    @staticmethod
    def array_key(array: np.ndarray, prefix: str = "") -> str:
        """Generate key for numpy array"""
        shape_str = "_".join(map(str, array.shape))
        dtype_str = str(array.dtype)
        
        # Use hash of array data
        array_hash = hashlib.md5(array.tobytes()).hexdigest()[:8]
        
        return f"{prefix}_array_{shape_str}_{dtype_str}_{array_hash}"
    
    @staticmethod
    def dict_key(data: dict, prefix: str = "") -> str:
        """Generate key for dictionary"""
        # Sort keys for consistent hashing
        sorted_items = sorted(data.items())
        dict_str = json.dumps(sorted_items, sort_keys=True, default=str)
        dict_hash = hashlib.md5(dict_str.encode()).hexdigest()[:8]
        
        return f"{prefix}_dict_{dict_hash}"
    
    @staticmethod
    def generic_key(data: Any, prefix: str = "") -> str:
        """Generate key for generic data"""
        data_str = str(data)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        
        return f"{prefix}_generic_{data_hash}"

class AsyncCacheManager:
    """Async cache manager for non-blocking operations"""
    
    def __init__(self, cache: MultiLevelCache, max_workers: int = 4):
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_operations = {}
        self.lock = asyncio.Lock()
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Async get from cache"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.cache.get, key)
    
    async def put_async(self, key: str, value: Any, ttl: Optional[float] = None):
        """Async put to cache"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.cache.put, key, value, ttl)
    
    async def get_or_compute_async(self, key: str, compute_fn: Callable[[], Any], 
                                   ttl: Optional[float] = None) -> Any:
        """Get from cache or compute value asynchronously"""
        async with self.lock:
            if key in self.pending_operations:
                # Wait for pending operation
                return await self.pending_operations[key]
            
            # Check cache first
            value = await self.get_async(key)
            if value is not None:
                return value
            
            # Create computation task
            async def compute_and_cache():
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, compute_fn)
                    await self.put_async(key, result, ttl)
                    return result
                finally:
                    # Remove from pending operations
                    self.pending_operations.pop(key, None)
            
            task = asyncio.create_task(compute_and_cache())
            self.pending_operations[key] = task
            return await task

# Decorators for caching functions
def cached(cache: MultiLevelCache, ttl: Optional[float] = None, prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {'args': args, 'kwargs': kwargs, 'func': func.__name__}
            cache_key = CacheKeyGenerator.generate_key(key_data, prefix)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

def tensor_cached(cache: MultiLevelCache, ttl: Optional[float] = None):
    """Decorator for caching tensor operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(tensor: torch.Tensor, *args, **kwargs):
            # Generate cache key for tensor
            cache_key = CacheKeyGenerator.tensor_key(tensor, func.__name__)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(tensor, *args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

# Global cache instance
_global_cache = None

def get_global_cache() -> MultiLevelCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        config = {
            'l1_capacity': 1000,
            'l2_capacity': 500,
            'l3_max_size_mb': 1000,
            'cache_dir': '/tmp/grandmodel_cache',
            'tensor_pool_size': 1000,
            'enable_redis': False
        }
        _global_cache = MultiLevelCache(config)
    return _global_cache

def configure_global_cache(config: Dict[str, Any]):
    """Configure global cache"""
    global _global_cache
    _global_cache = MultiLevelCache(config)

# Context manager for cache sessions
@contextmanager
def cache_session(cache: MultiLevelCache):
    """Context manager for cache session"""
    try:
        yield cache
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    # Example usage
    config = {
        'l1_capacity': 1000,
        'l2_capacity': 500,
        'l3_max_size_mb': 1000,
        'cache_dir': '/tmp/grandmodel_cache',
        'tensor_pool_size': 1000,
        'enable_redis': False
    }
    
    cache = MultiLevelCache(config)
    
    # Test caching
    test_tensor = torch.randn(100, 50)
    cache.put("test_tensor", test_tensor)
    
    retrieved = cache.get("test_tensor")
    print(f"Cache test: {torch.allclose(test_tensor, retrieved)}")
    
    # Print stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")