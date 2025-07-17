"""
Intelligent caching system for high-frequency data pipeline operations

This module implements advanced caching strategies with predictive prefetching,
LRU eviction, and intelligent cache warming for optimal performance.
"""

import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import OrderedDict, defaultdict, deque
import hashlib
import pickle
import weakref
import psutil
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
import sqlite3
import json
import heapq
from functools import wraps
import statistics

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Combination of strategies

class CacheLevel(Enum):
    """Cache levels for hierarchical caching"""
    L1_MEMORY = "l1_memory"  # Fast in-memory cache
    L2_DISK = "l2_disk"  # Disk-based cache
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache

class PrefetchStrategy(Enum):
    """Prefetch strategies for predictive caching"""
    NONE = "none"
    SEQUENTIAL = "sequential"  # Prefetch next items in sequence
    PATTERN_BASED = "pattern_based"  # Based on historical patterns
    FREQUENCY_BASED = "frequency_based"  # Based on access frequency
    ML_PREDICTED = "ml_predicted"  # Machine learning predictions

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    hit_count: int
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access metrics"""
        self.last_accessed = time.time()
        self.access_count += 1
        self.hit_count += 1

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time_us: float = 0.0
    memory_usage_mb: float = 0.0
    storage_efficiency: float = 0.0
    eviction_count: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class AccessPattern:
    """Access pattern for predictive caching"""
    key: str
    access_times: List[float]
    access_intervals: List[float]
    frequency: float
    pattern_score: float
    next_predicted_access: Optional[float] = None

class IntelligentCache:
    """High-performance intelligent caching system"""
    
    def __init__(self,
                 max_size: int = 10000,
                 max_memory_mb: int = 1000,
                 cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 prefetch_strategy: PrefetchStrategy = PrefetchStrategy.PATTERN_BASED,
                 enable_persistence: bool = True,
                 persistence_path: Optional[str] = None,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024,
                 ttl_seconds: Optional[float] = None,
                 enable_statistics: bool = True):
        
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache_strategy = cache_strategy
        self.prefetch_strategy = prefetch_strategy
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path) if persistence_path else Path("/tmp/cache")
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.ttl_seconds = ttl_seconds
        self.enable_statistics = enable_statistics
        
        # Main cache storage
        self.cache = OrderedDict()  # For LRU functionality
        self.cache_lock = threading.RLock()
        
        # Frequency tracking for LFU
        self.access_frequencies = defaultdict(int)
        self.frequency_lock = threading.RLock()
        
        # Access pattern tracking
        self.access_patterns = {}
        self.pattern_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = CacheMetrics()
        self.metrics_lock = threading.Lock()
        
        # Response time tracking
        self.response_times = deque(maxlen=10000)
        
        # Memory usage tracking
        self.current_memory_usage = 0
        self.memory_lock = threading.Lock()
        
        # Prefetch queue and worker
        self.prefetch_queue = queue.Queue(maxsize=1000)
        self.prefetch_callbacks = {}
        self.prefetch_worker = None
        
        # Persistence database
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.db_path = self.persistence_path / "cache.db"
            self._init_persistence()
        
        # Background workers
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics collection
        if enable_statistics:
            self._start_statistics_collection()
        
        # Start prefetch worker
        if prefetch_strategy != PrefetchStrategy.NONE:
            self._start_prefetch_worker()
        
        # Start cleanup worker
        self._start_cleanup_worker()
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.enable_persistence:
            self._save_cache_to_disk()
        
        self.executor.shutdown(wait=True)
        logger.info("IntelligentCache cleanup completed")
    
    def _init_persistence(self):
        """Initialize persistence database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    size_bytes INTEGER,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    hit_count INTEGER,
                    ttl_seconds REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_patterns (
                    key TEXT PRIMARY KEY,
                    access_times TEXT,
                    access_intervals TEXT,
                    frequency REAL,
                    pattern_score REAL,
                    next_predicted_access REAL
                )
            ''')
            
            conn.commit()
        
        # Load existing cache from disk
        self._load_cache_from_disk()
    
    def _load_cache_from_disk(self):
        """Load cache from persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM cache_entries')
                
                for row in cursor.fetchall():
                    key, value_blob, size_bytes, created_at, last_accessed, access_count, hit_count, ttl_seconds, metadata_json = row
                    
                    # Deserialize value
                    try:
                        value = pickle.loads(value_blob)
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        
                        # Create cache entry
                        entry = CacheEntry(
                            key=key,
                            value=value,
                            size_bytes=size_bytes,
                            created_at=created_at,
                            last_accessed=last_accessed,
                            access_count=access_count,
                            hit_count=hit_count,
                            ttl_seconds=ttl_seconds,
                            metadata=metadata
                        )
                        
                        # Check if expired
                        if not entry.is_expired():
                            with self.cache_lock:
                                self.cache[key] = entry
                            
                            with self.memory_lock:
                                self.current_memory_usage += size_bytes
                    
                    except Exception as e:
                        logger.warning(f"Failed to load cache entry {key}: {str(e)}")
                
                # Load access patterns
                cursor = conn.execute('SELECT * FROM access_patterns')
                for row in cursor.fetchall():
                    key, access_times_json, access_intervals_json, frequency, pattern_score, next_predicted_access = row
                    
                    try:
                        access_times = json.loads(access_times_json)
                        access_intervals = json.loads(access_intervals_json)
                        
                        pattern = AccessPattern(
                            key=key,
                            access_times=access_times,
                            access_intervals=access_intervals,
                            frequency=frequency,
                            pattern_score=pattern_score,
                            next_predicted_access=next_predicted_access
                        )
                        
                        with self.pattern_lock:
                            self.access_patterns[key] = pattern
                    
                    except Exception as e:
                        logger.warning(f"Failed to load access pattern {key}: {str(e)}")
            
            logger.info(f"Loaded {len(self.cache)} cache entries from disk")
        
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {str(e)}")
    
    def _save_cache_to_disk(self):
        """Save cache to persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing entries
                conn.execute('DELETE FROM cache_entries')
                conn.execute('DELETE FROM access_patterns')
                
                # Save cache entries
                with self.cache_lock:
                    for key, entry in self.cache.items():
                        try:
                            value_blob = pickle.dumps(entry.value)
                            metadata_json = json.dumps(entry.metadata)
                            
                            conn.execute(
                                'INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                (key, value_blob, entry.size_bytes, entry.created_at, entry.last_accessed,
                                 entry.access_count, entry.hit_count, entry.ttl_seconds, metadata_json)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save cache entry {key}: {str(e)}")
                
                # Save access patterns
                with self.pattern_lock:
                    for key, pattern in self.access_patterns.items():
                        try:
                            access_times_json = json.dumps(pattern.access_times)
                            access_intervals_json = json.dumps(pattern.access_intervals)
                            
                            conn.execute(
                                'INSERT INTO access_patterns VALUES (?, ?, ?, ?, ?, ?)',
                                (key, access_times_json, access_intervals_json, pattern.frequency,
                                 pattern.pattern_score, pattern.next_predicted_access)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save access pattern {key}: {str(e)}")
                
                conn.commit()
            
            logger.info(f"Saved {len(self.cache)} cache entries to disk")
        
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time_ns()
        
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    with self.memory_lock:
                        self.current_memory_usage -= entry.size_bytes
                    self._update_metrics(False, start_time)
                    return None
                
                # Update access metrics
                entry.update_access()
                
                # Move to end for LRU
                self.cache.move_to_end(key)
                
                # Update access patterns
                self._update_access_pattern(key)
                
                # Update metrics
                self._update_metrics(True, start_time)
                
                return entry.value
            
            else:
                self._update_metrics(False, start_time)
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache"""
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default size estimate
        
        # Check memory constraints
        if size_bytes > self.max_memory_mb * 1024 * 1024:
            logger.warning(f"Value too large for cache: {size_bytes} bytes")
            return False
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            hit_count=0,
            ttl_seconds=ttl_seconds or self.ttl_seconds,
            metadata=metadata or {}
        )
        
        with self.cache_lock:
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                with self.memory_lock:
                    self.current_memory_usage -= old_entry.size_bytes
            
            # Ensure we have space
            while (len(self.cache) >= self.max_size or 
                   self.current_memory_usage + size_bytes > self.max_memory_mb * 1024 * 1024):
                if not self._evict_entry():
                    break
            
            # Add to cache
            self.cache[key] = entry
            
            with self.memory_lock:
                self.current_memory_usage += size_bytes
        
        # Update frequency tracking
        with self.frequency_lock:
            self.access_frequencies[key] += 1
        
        # Trigger prefetch if applicable
        if self.prefetch_strategy != PrefetchStrategy.NONE:
            self._trigger_prefetch(key)
        
        return True
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on strategy"""
        if not self.cache:
            return False
        
        key_to_evict = None
        
        if self.cache_strategy == CacheStrategy.LRU:
            key_to_evict = next(iter(self.cache))
        
        elif self.cache_strategy == CacheStrategy.LFU:
            with self.frequency_lock:
                if self.access_frequencies:
                    key_to_evict = min(self.access_frequencies.keys(), 
                                     key=lambda k: self.access_frequencies[k])
        
        elif self.cache_strategy == CacheStrategy.FIFO:
            key_to_evict = next(iter(self.cache))
        
        elif self.cache_strategy == CacheStrategy.TTL:
            # Find expired entries first
            current_time = time.time()
            for key, entry in self.cache.items():
                if entry.is_expired():
                    key_to_evict = key
                    break
            
            if key_to_evict is None:
                key_to_evict = next(iter(self.cache))
        
        elif self.cache_strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction()
        
        else:  # Default to LRU
            key_to_evict = next(iter(self.cache))
        
        if key_to_evict:
            entry = self.cache.pop(key_to_evict)
            with self.memory_lock:
                self.current_memory_usage -= entry.size_bytes
            
            with self.frequency_lock:
                if key_to_evict in self.access_frequencies:
                    del self.access_frequencies[key_to_evict]
            
            with self.metrics_lock:
                self.metrics.eviction_count += 1
            
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns"""
        if not self.cache:
            return None
        
        # Score entries based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Factors: recency, frequency, size, TTL
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count / (current_time - entry.created_at + 1)
            size_score = 1.0 / (entry.size_bytes + 1)
            
            # TTL score
            ttl_score = 1.0
            if entry.ttl_seconds:
                time_left = entry.ttl_seconds - (current_time - entry.created_at)
                ttl_score = max(0, time_left / entry.ttl_seconds)
            
            # Combined score (higher is better)
            scores[key] = (recency_score * 0.4 + frequency_score * 0.3 + 
                          size_score * 0.2 + ttl_score * 0.1)
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for predictive caching"""
        if not self.enable_statistics:
            return
        
        current_time = time.time()
        
        with self.pattern_lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(
                    key=key,
                    access_times=[current_time],
                    access_intervals=[],
                    frequency=1.0,
                    pattern_score=0.0
                )
            else:
                pattern = self.access_patterns[key]
                
                # Update access times
                pattern.access_times.append(current_time)
                
                # Keep only recent accesses (last 100)
                if len(pattern.access_times) > 100:
                    pattern.access_times = pattern.access_times[-100:]
                
                # Calculate intervals
                if len(pattern.access_times) >= 2:
                    intervals = [pattern.access_times[i] - pattern.access_times[i-1] 
                               for i in range(1, len(pattern.access_times))]
                    pattern.access_intervals = intervals
                    
                    # Calculate frequency and pattern score
                    pattern.frequency = len(pattern.access_times) / (current_time - pattern.access_times[0] + 1)
                    
                    # Pattern score based on regularity
                    if len(intervals) >= 3:
                        pattern.pattern_score = 1.0 / (statistics.stdev(intervals) + 1)
                    
                    # Predict next access
                    if len(intervals) >= 2:
                        avg_interval = statistics.mean(intervals)
                        pattern.next_predicted_access = current_time + avg_interval
    
    def _trigger_prefetch(self, key: str):
        """Trigger prefetch based on strategy"""
        if self.prefetch_strategy == PrefetchStrategy.NONE:
            return
        
        if self.prefetch_strategy == PrefetchStrategy.SEQUENTIAL:
            # Try to prefetch next sequential key
            if key.endswith('_0') or key.endswith('_1') or key.endswith('_2'):
                # Extract base and increment
                base = key.rsplit('_', 1)[0]
                try:
                    num = int(key.rsplit('_', 1)[1])
                    next_key = f"{base}_{num + 1}"
                    self._queue_prefetch(next_key)
                except:
                    pass
        
        elif self.prefetch_strategy == PrefetchStrategy.PATTERN_BASED:
            # Prefetch based on access patterns
            with self.pattern_lock:
                if key in self.access_patterns:
                    pattern = self.access_patterns[key]
                    if pattern.pattern_score > 0.5 and pattern.next_predicted_access:
                        time_to_prefetch = pattern.next_predicted_access - time.time()
                        if 0 < time_to_prefetch < 60:  # Prefetch if within next minute
                            self._queue_prefetch(key, delay=time_to_prefetch)
    
    def _queue_prefetch(self, key: str, delay: float = 0):
        """Queue a prefetch operation"""
        if key in self.prefetch_callbacks:
            try:
                self.prefetch_queue.put((key, delay, time.time()), timeout=0.1)
            except queue.Full:
                pass  # Queue is full, skip this prefetch
    
    def _start_prefetch_worker(self):
        """Start prefetch worker thread"""
        def prefetch_worker():
            while True:
                try:
                    key, delay, queued_time = self.prefetch_queue.get(timeout=1.0)
                    
                    # Check if delay is needed
                    if delay > 0:
                        time.sleep(delay)
                    
                    # Check if key is still relevant
                    if key in self.cache:
                        continue  # Already cached
                    
                    # Execute prefetch callback
                    if key in self.prefetch_callbacks:
                        try:
                            callback = self.prefetch_callbacks[key]
                            value = callback(key)
                            if value is not None:
                                self.put(key, value)
                                with self.metrics_lock:
                                    self.metrics.prefetch_hits += 1
                        except Exception as e:
                            logger.warning(f"Prefetch failed for {key}: {str(e)}")
                            with self.metrics_lock:
                                self.metrics.prefetch_misses += 1
                    
                    self.prefetch_queue.task_done()
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in prefetch worker: {str(e)}")
        
        self.prefetch_worker = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_worker.start()
    
    def _start_cleanup_worker(self):
        """Start cleanup worker for expired entries"""
        def cleanup_worker():
            while True:
                time.sleep(60)  # Run every minute
                
                try:
                    expired_keys = []
                    current_time = time.time()
                    
                    with self.cache_lock:
                        for key, entry in self.cache.items():
                            if entry.is_expired():
                                expired_keys.append(key)
                    
                    # Remove expired entries
                    for key in expired_keys:
                        with self.cache_lock:
                            if key in self.cache:
                                entry = self.cache.pop(key)
                                with self.memory_lock:
                                    self.current_memory_usage -= entry.size_bytes
                        
                        with self.frequency_lock:
                            if key in self.access_frequencies:
                                del self.access_frequencies[key]
                    
                    if expired_keys:
                        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _start_statistics_collection(self):
        """Start statistics collection worker"""
        def stats_worker():
            while True:
                time.sleep(10)  # Update stats every 10 seconds
                
                try:
                    with self.metrics_lock:
                        # Update memory usage
                        self.metrics.memory_usage_mb = self.current_memory_usage / (1024 * 1024)
                        
                        # Update hit rate
                        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
                        if total_requests > 0:
                            self.metrics.hit_rate = self.metrics.cache_hits / total_requests
                        
                        # Update average response time
                        if self.response_times:
                            self.metrics.avg_response_time_us = statistics.mean(self.response_times)
                        
                        # Update storage efficiency
                        if self.max_memory_mb > 0:
                            self.metrics.storage_efficiency = (self.current_memory_usage / (1024 * 1024)) / self.max_memory_mb
                        
                        self.metrics.timestamp = time.time()
                
                except Exception as e:
                    logger.error(f"Error in statistics worker: {str(e)}")
        
        stats_thread = threading.Thread(target=stats_worker, daemon=True)
        stats_thread.start()
    
    def _update_metrics(self, cache_hit: bool, start_time: int):
        """Update performance metrics"""
        end_time = time.time_ns()
        response_time_us = (end_time - start_time) / 1000
        
        with self.metrics_lock:
            self.metrics.total_requests += 1
            
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
        
        # Update response times
        self.response_times.append(response_time_us)
    
    def register_prefetch_callback(self, key: str, callback: Callable[[str], Any]):
        """Register callback for prefetching"""
        self.prefetch_callbacks[key] = callback
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self.cache_lock:
            return key in self.cache and not self.cache[key].is_expired()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                with self.memory_lock:
                    self.current_memory_usage -= entry.size_bytes
                return True
        
        with self.frequency_lock:
            if key in self.access_frequencies:
                del self.access_frequencies[key]
        
        return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.cache_lock:
            self.cache.clear()
        
        with self.frequency_lock:
            self.access_frequencies.clear()
        
        with self.pattern_lock:
            self.access_patterns.clear()
        
        with self.memory_lock:
            self.current_memory_usage = 0
        
        with self.metrics_lock:
            self.metrics = CacheMetrics()
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        with self.metrics_lock:
            return CacheMetrics(
                total_requests=self.metrics.total_requests,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                hit_rate=self.metrics.hit_rate,
                avg_response_time_us=self.metrics.avg_response_time_us,
                memory_usage_mb=self.metrics.memory_usage_mb,
                storage_efficiency=self.metrics.storage_efficiency,
                eviction_count=self.metrics.eviction_count,
                prefetch_hits=self.metrics.prefetch_hits,
                prefetch_misses=self.metrics.prefetch_misses,
                timestamp=time.time()
            )
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self.cache_lock:
            cache_size = len(self.cache)
        
        return {
            'cache_size': cache_size,
            'max_size': self.max_size,
            'utilization': cache_size / self.max_size if self.max_size > 0 else 0,
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': (self.current_memory_usage / (1024 * 1024)) / self.max_memory_mb if self.max_memory_mb > 0 else 0,
            'cache_strategy': self.cache_strategy.value,
            'prefetch_strategy': self.prefetch_strategy.value,
            'enable_persistence': self.enable_persistence,
            'enable_compression': self.enable_compression
        }
    
    def get_top_keys(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get top N most frequently accessed keys"""
        with self.frequency_lock:
            return sorted(self.access_frequencies.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def warm_cache(self, keys_and_values: Dict[str, Any]):
        """Warm cache with predefined values"""
        for key, value in keys_and_values.items():
            self.put(key, value)
    
    def size(self) -> int:
        """Get current cache size"""
        with self.cache_lock:
            return len(self.cache)
    
    def memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        with self.memory_lock:
            return self.current_memory_usage

# Decorator for automatic caching
def cached(cache_instance: IntelligentCache, ttl_seconds: Optional[float] = None):
    """Decorator for automatic function result caching"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            result = cache_instance.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_instance.put(key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator

# Utility functions
def create_intelligent_cache(max_size: int = 10000, 
                           max_memory_mb: int = 1000,
                           cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE) -> IntelligentCache:
    """Create intelligent cache with default settings"""
    return IntelligentCache(
        max_size=max_size,
        max_memory_mb=max_memory_mb,
        cache_strategy=cache_strategy,
        prefetch_strategy=PrefetchStrategy.PATTERN_BASED,
        enable_persistence=True,
        enable_compression=True
    )

def create_simple_cache(max_size: int = 1000) -> IntelligentCache:
    """Create simple cache without advanced features"""
    return IntelligentCache(
        max_size=max_size,
        max_memory_mb=100,
        cache_strategy=CacheStrategy.LRU,
        prefetch_strategy=PrefetchStrategy.NONE,
        enable_persistence=False,
        enable_compression=False,
        enable_statistics=False
    )
