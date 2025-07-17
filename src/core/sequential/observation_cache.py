"""
High-Performance Observation Cache for Sequential Architecture

This module provides a high-performance caching system optimized for <5ms enrichment time
with intelligent cache management and memory optimization.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import pickle
import hashlib
import structlog
from enum import Enum
import weakref
import gc

logger = structlog.get_logger(__name__)


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive eviction based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int
    last_access: datetime
    ttl: Optional[timedelta]
    size_bytes: int
    metadata: Dict[str, Any]


class MemoryOptimizedDict:
    """Memory-optimized dictionary with weak references"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._data = OrderedDict()
        self._weak_refs = weakref.WeakValueDictionary()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self._data:
                # Move to end (most recently used)
                self._data.move_to_end(key)
                return self._data[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self.lock:
            if key in self._data:
                # Update existing
                self._data[key] = value
                self._data.move_to_end(key)
            else:
                # Add new entry
                self._data[key] = value
                
                # Evict if over limit
                if len(self._data) > self.max_size:
                    self._data.popitem(last=False)  # Remove oldest
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items"""
        with self.lock:
            self._data.clear()
    
    def keys(self) -> List[str]:
        """Get all keys"""
        with self.lock:
            return list(self._data.keys())
    
    def size(self) -> int:
        """Get number of items"""
        with self.lock:
            return len(self._data)


class ObservationCache:
    """
    High-Performance Observation Cache
    
    Provides ultra-fast caching for observation enrichment with intelligent
    eviction policies and memory optimization to achieve <5ms enrichment time.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Observation Cache
        
        Args:
            config: Configuration dictionary with cache settings
        """
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Configuration parameters
        self.max_size = config.get("max_size", 10000)
        self.max_memory_mb = config.get("max_memory_mb", 512)
        self.default_ttl_seconds = config.get("default_ttl_seconds", 300)  # 5 minutes
        self.eviction_policy = CacheEvictionPolicy(config.get("eviction_policy", "adaptive"))
        self.enable_compression = config.get("enable_compression", False)
        self.enable_persistent_cache = config.get("enable_persistent_cache", False)
        
        # Performance optimization settings
        self.enable_prefetching = config.get("enable_prefetching", True)
        self.prefetch_threshold = config.get("prefetch_threshold", 0.8)
        self.background_cleanup_interval = config.get("background_cleanup_interval", 60)
        
        # Multi-level cache structure
        self.l1_cache = MemoryOptimizedDict(max_size=min(1000, self.max_size // 4))  # Hot cache
        self.l2_cache = MemoryOptimizedDict(max_size=self.max_size)  # Main cache
        self.cache_metadata: Dict[str, CacheEntry] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "evictions": 0,
            "access_times": [],
            "memory_usage_bytes": 0,
            "compression_ratio": 0.0
        }
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.hot_keys: set = set()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize cache
        self._initialize_cache()
        
        self.logger.info(
            "Observation Cache initialized",
            max_size=self.max_size,
            max_memory_mb=self.max_memory_mb,
            eviction_policy=self.eviction_policy.value,
            enable_compression=self.enable_compression
        )
    
    def _initialize_cache(self) -> None:
        """Initialize cache components"""
        
        # Start background cleanup thread
        if self.background_cleanup_interval > 0:
            self.cleanup_thread = threading.Thread(
                target=self._background_cleanup,
                daemon=True
            )
            self.cleanup_thread.start()
        
        # Load persistent cache if enabled
        if self.enable_persistent_cache:
            self._load_persistent_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache with multi-level lookup
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Check L1 cache first (hot cache)
                value = self.l1_cache.get(key)
                if value is not None:
                    self.performance_metrics["cache_hits"] += 1
                    self.performance_metrics["l1_hits"] += 1
                    self._update_access_pattern(key)
                    return self._deserialize_value(value)
                
                # Check L2 cache
                value = self.l2_cache.get(key)
                if value is not None:
                    self.performance_metrics["cache_hits"] += 1
                    self.performance_metrics["l2_hits"] += 1
                    self._update_access_pattern(key)
                    
                    # Promote to L1 if hot
                    if self._is_hot_key(key):
                        self.l1_cache.put(key, value)
                    
                    return self._deserialize_value(value)
                
                # Cache miss
                self.performance_metrics["cache_misses"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Cache get error: {e}", key=key)
            return None
        
        finally:
            # Record access time
            access_time = (time.time() - start_time) * 1000
            self.performance_metrics["access_times"].append(access_time)
            
            # Keep only recent access times
            if len(self.performance_metrics["access_times"]) > 1000:
                self.performance_metrics["access_times"] = self.performance_metrics["access_times"][-1000:]
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Put item in cache with intelligent placement
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live override
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Serialize value
                serialized_value = self._serialize_value(value)
                
                # Calculate size
                size_bytes = len(serialized_value) if isinstance(serialized_value, bytes) else len(str(serialized_value))
                
                # Check memory constraints
                if not self._check_memory_constraints(size_bytes):
                    self._evict_items()
                
                # Create cache entry metadata
                entry = CacheEntry(
                    key=key,
                    value=serialized_value,
                    timestamp=datetime.now(),
                    access_count=1,
                    last_access=datetime.now(),
                    ttl=ttl or timedelta(seconds=self.default_ttl_seconds),
                    size_bytes=size_bytes,
                    metadata={}
                )
                
                # Store in appropriate cache level
                if self._is_hot_key(key) or self._predict_hot_key(key):
                    self.l1_cache.put(key, serialized_value)
                
                self.l2_cache.put(key, serialized_value)
                self.cache_metadata[key] = entry
                
                # Update metrics
                self.performance_metrics["memory_usage_bytes"] += size_bytes
                
                # Update access pattern
                self._update_access_pattern(key)
                
        except Exception as e:
            self.logger.error(f"Cache put error: {e}", key=key)
        
        finally:
            put_time = (time.time() - start_time) * 1000
            if put_time > 1.0:  # Log slow puts
                self.logger.warning(f"Slow cache put: {put_time:.2f}ms", key=key)
    
    def remove(self, key: str) -> bool:
        """
        Remove item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed, False if not found
        """
        with self.lock:
            removed = False
            
            # Remove from L1 cache
            if self.l1_cache.remove(key):
                removed = True
            
            # Remove from L2 cache
            if self.l2_cache.remove(key):
                removed = True
            
            # Remove metadata
            if key in self.cache_metadata:
                entry = self.cache_metadata.pop(key)
                self.performance_metrics["memory_usage_bytes"] -= entry.size_bytes
                removed = True
            
            # Remove from access patterns
            if key in self.access_patterns:
                del self.access_patterns[key]
            
            # Remove from hot keys
            self.hot_keys.discard(key)
            
            return removed
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.cache_metadata.clear()
            self.access_patterns.clear()
            self.hot_keys.clear()
            self.performance_metrics["memory_usage_bytes"] = 0
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for storage"""
        if self.enable_compression:
            try:
                import zlib
                pickled = pickle.dumps(value)
                compressed = zlib.compress(pickled)
                
                # Only use compression if it actually saves space
                if len(compressed) < len(pickled):
                    self.performance_metrics["compression_ratio"] = len(compressed) / len(pickled)
                    return compressed
                else:
                    return pickled
            except Exception:
                return value
        else:
            return value
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize value from storage"""
        if self.enable_compression and isinstance(value, bytes):
            try:
                import zlib
                # Try to decompress
                decompressed = zlib.decompress(value)
                return pickle.loads(decompressed)
            except Exception:
                try:
                    # Maybe it's just pickled
                    return pickle.loads(value)
                except Exception:
                    return value
        else:
            return value
    
    def _check_memory_constraints(self, new_size_bytes: int) -> bool:
        """Check if adding new item would exceed memory limits"""
        
        current_memory_mb = self.performance_metrics["memory_usage_bytes"] / (1024 * 1024)
        new_memory_mb = (self.performance_metrics["memory_usage_bytes"] + new_size_bytes) / (1024 * 1024)
        
        return new_memory_mb <= self.max_memory_mb
    
    def _evict_items(self) -> None:
        """Evict items based on eviction policy"""
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            self._evict_lru()
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            self._evict_lfu()
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            self._evict_ttl()
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            self._evict_adaptive()
        
        self.performance_metrics["evictions"] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used items"""
        
        # Sort by last access time
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].last_access
        )
        
        # Evict oldest 10%
        num_to_evict = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:num_to_evict]:
            self.remove(key)
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used items"""
        
        # Sort by access count
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].access_count
        )
        
        # Evict least accessed 10%
        num_to_evict = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:num_to_evict]:
            self.remove(key)
    
    def _evict_ttl(self) -> None:
        """Evict expired items"""
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache_metadata.items():
            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.remove(key)
    
    def _evict_adaptive(self) -> None:
        """Adaptive eviction based on usage patterns"""
        
        # First, evict expired items
        self._evict_ttl()
        
        # Then, evict based on combined score
        if len(self.cache_metadata) > self.max_size * 0.9:
            current_time = datetime.now()
            
            # Calculate eviction scores
            eviction_scores = []
            for key, entry in self.cache_metadata.items():
                # Combine multiple factors
                time_factor = (current_time - entry.last_access).total_seconds() / 3600  # hours
                frequency_factor = 1.0 / max(1, entry.access_count)
                size_factor = entry.size_bytes / (1024 * 1024)  # MB
                
                score = time_factor * 0.4 + frequency_factor * 0.4 + size_factor * 0.2
                eviction_scores.append((key, score))
            
            # Sort by score (higher score = more likely to evict)
            eviction_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Evict top 20%
            num_to_evict = max(1, len(eviction_scores) // 5)
            for key, _ in eviction_scores[:num_to_evict]:
                self.remove(key)
    
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for a key"""
        
        current_time = datetime.now()
        
        # Update metadata
        if key in self.cache_metadata:
            entry = self.cache_metadata[key]
            entry.access_count += 1
            entry.last_access = current_time
        
        # Update access patterns
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
        # Update hot keys
        if len(self.access_patterns[key]) > 5:  # Accessed more than 5 times in last hour
            self.hot_keys.add(key)
        elif len(self.access_patterns[key]) < 2:
            self.hot_keys.discard(key)
    
    def _is_hot_key(self, key: str) -> bool:
        """Check if key is considered hot"""
        return key in self.hot_keys
    
    def _predict_hot_key(self, key: str) -> bool:
        """Predict if key will become hot based on patterns"""
        
        # Simple prediction based on key patterns
        if "agent" in key and "strategic" in key:
            return True
        if "risk" in key:
            return True
        if "correlation" in key:
            return True
        
        return False
    
    def _background_cleanup(self) -> None:
        """Background cleanup thread"""
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for cleanup interval
                if self.shutdown_event.wait(self.background_cleanup_interval):
                    break
                
                # Perform cleanup
                self._cleanup_expired_entries()
                self._optimize_memory_usage()
                
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries"""
        
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache_metadata.items():
                if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.remove(key)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage"""
        
        # Force garbage collection
        gc.collect()
        
        # Check if we need to evict items
        current_memory_mb = self.performance_metrics["memory_usage_bytes"] / (1024 * 1024)
        
        if current_memory_mb > self.max_memory_mb * 0.8:
            self._evict_items()
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk"""
        
        if self.enable_persistent_cache:
            try:
                # Implementation would load from disk
                pass
            except Exception as e:
                self.logger.warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self) -> None:
        """Save persistent cache to disk"""
        
        if self.enable_persistent_cache:
            try:
                # Implementation would save to disk
                pass
            except Exception as e:
                self.logger.warning(f"Failed to save persistent cache: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        with self.lock:
            total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
            
            metrics = {
                "cache_hit_rate": self.performance_metrics["cache_hits"] / max(1, total_requests),
                "l1_hit_rate": self.performance_metrics["l1_hits"] / max(1, total_requests),
                "l2_hit_rate": self.performance_metrics["l2_hits"] / max(1, total_requests),
                "total_entries": len(self.cache_metadata),
                "l1_entries": self.l1_cache.size(),
                "l2_entries": self.l2_cache.size(),
                "memory_usage_mb": self.performance_metrics["memory_usage_bytes"] / (1024 * 1024),
                "hot_keys_count": len(self.hot_keys),
                "evictions": self.performance_metrics["evictions"],
                "compression_ratio": self.performance_metrics["compression_ratio"]
            }
            
            # Average access time
            if self.performance_metrics["access_times"]:
                metrics["avg_access_time_ms"] = sum(self.performance_metrics["access_times"]) / len(self.performance_metrics["access_times"])
                metrics["max_access_time_ms"] = max(self.performance_metrics["access_times"])
            
            return metrics
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        
        with self.lock:
            return {
                "total_entries": len(self.cache_metadata),
                "memory_usage_bytes": self.performance_metrics["memory_usage_bytes"],
                "hot_keys": list(self.hot_keys),
                "access_patterns_count": len(self.access_patterns),
                "eviction_policy": self.eviction_policy.value,
                "cache_levels": {
                    "l1_size": self.l1_cache.size(),
                    "l2_size": self.l2_cache.size()
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        
        self.shutdown_event.set()
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Save persistent cache
        self._save_persistent_cache()
        
        # Clear all caches
        self.clear()
        
        self.logger.info("Observation Cache shutdown complete")