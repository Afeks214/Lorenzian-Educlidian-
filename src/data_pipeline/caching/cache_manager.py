"""
Advanced caching manager for frequently accessed data
"""

import time
import threading
import hashlib
import pickle
import lzma
import gzip
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import logging
import psutil
import weakref
import sqlite3
from contextlib import contextmanager
import mmap
import tempfile
import json

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataCachingException
from ..core.data_loader import DataChunk

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache manager"""
    max_size_gb: float = 10.0
    cache_directory: str = "/tmp/data_pipeline_cache"
    enable_compression: bool = True
    compression_level: int = 6
    enable_persistence: bool = True
    ttl_seconds: int = 86400  # 24 hours
    
    # Cache strategies
    eviction_policy: str = "lru"  # lru, lfu, ttl, adaptive
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    memory_cache_size_mb: float = 1024.0
    
    # Performance options
    enable_async_writes: bool = True
    enable_prefetching: bool = True
    prefetch_threads: int = 2
    enable_deduplication: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    stats_interval: int = 300  # 5 minutes

@dataclass
class CacheItem:
    """Item stored in cache"""
    key: str
    data: Any
    size_bytes: int
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if item is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()

class CacheManager:
    """
    Advanced cache manager with multiple storage tiers and intelligent eviction
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize storage
        self.memory_cache: Dict[str, CacheItem] = {}
        self.disk_cache_index: Dict[str, str] = {}  # key -> file_path
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background threads
        self._cleanup_thread = None
        self._monitoring_thread = None
        self._prefetch_thread = None
        
        # Setup cache directory
        self._setup_cache_directory()
        
        # Initialize persistence
        if self.config.enable_persistence:
            self._setup_persistence()
        
        # Start background threads
        self._start_background_threads()
    
    def _setup_cache_directory(self):
        """Setup cache directory structure"""
        cache_path = Path(self.config.cache_directory)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (cache_path / "memory").mkdir(exist_ok=True)
        (cache_path / "disk").mkdir(exist_ok=True)
        (cache_path / "temp").mkdir(exist_ok=True)
        (cache_path / "metadata").mkdir(exist_ok=True)
    
    def _setup_persistence(self):
        """Setup persistence using SQLite"""
        db_path = Path(self.config.cache_directory) / "metadata" / "cache.db"
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_items (
                    key TEXT PRIMARY KEY,
                    size_bytes INTEGER,
                    timestamp REAL,
                    access_count INTEGER,
                    last_access REAL,
                    ttl REAL,
                    compressed INTEGER,
                    file_path TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON cache_items(last_access)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON cache_items(timestamp)
            """)
    
    def _start_background_threads(self):
        """Start background maintenance threads"""
        # Cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        # Monitoring thread
        if self.config.enable_monitoring:
            self._monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self._monitoring_thread.start()
        
        # Prefetch thread
        if self.config.enable_prefetching:
            self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._prefetch_thread.start()
    
    def put(self, key: str, data: Any, ttl: Optional[float] = None) -> bool:
        """
        Store data in cache
        """
        with self._lock:
            try:
                # Calculate data size
                size_bytes = self._calculate_size(data)
                
                # Check if item should be compressed
                should_compress = (
                    self.config.enable_compression and 
                    size_bytes > 1024  # Only compress items > 1KB
                )
                
                # Prepare cache item
                cache_item = CacheItem(
                    key=key,
                    data=data,
                    size_bytes=size_bytes,
                    timestamp=time.time(),
                    ttl=ttl or self.config.ttl_seconds,
                    compressed=should_compress
                )
                
                # Determine storage tier
                if self._should_store_in_memory(cache_item):
                    return self._store_in_memory(cache_item)
                else:
                    return self._store_on_disk(cache_item)
                
            except Exception as e:
                logger.error(f"Error storing item in cache: {str(e)}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache
        """
        with self._lock:
            try:
                # Check memory cache first
                if key in self.memory_cache:
                    item = self.memory_cache[key]
                    if item.is_expired():
                        self._remove_from_memory(key)
                        self.stats.misses += 1
                        return None
                    
                    item.update_access()
                    self.stats.hits += 1
                    self.stats.memory_hits += 1
                    return item.data
                
                # Check disk cache
                if key in self.disk_cache_index:
                    item = self._load_from_disk(key)
                    if item is not None and not item.is_expired():
                        item.update_access()
                        self.stats.hits += 1
                        self.stats.disk_hits += 1
                        
                        # Promote to memory if frequently accessed
                        if self._should_promote_to_memory(item):
                            self._promote_to_memory(item)
                        
                        return item.data
                    else:
                        self._remove_from_disk(key)
                
                self.stats.misses += 1
                return None
                
            except Exception as e:
                logger.error(f"Error retrieving item from cache: {str(e)}")
                self.stats.misses += 1
                return None
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            return key in self.memory_cache or key in self.disk_cache_index
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self._lock:
            removed = False
            
            if key in self.memory_cache:
                self._remove_from_memory(key)
                removed = True
            
            if key in self.disk_cache_index:
                self._remove_from_disk(key)
                removed = True
            
            return removed
    
    def clear(self):
        """Clear all cache contents"""
        with self._lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear disk cache
            self.disk_cache_index.clear()
            
            # Remove disk files
            disk_path = Path(self.config.cache_directory) / "disk"
            if disk_path.exists():
                shutil.rmtree(disk_path)
                disk_path.mkdir(exist_ok=True)
            
            # Reset stats
            self.stats.reset()
    
    def _should_store_in_memory(self, item: CacheItem) -> bool:
        """Determine if item should be stored in memory"""
        if not self.config.enable_memory_cache:
            return False
        
        # Check memory usage
        current_memory_mb = self._get_memory_usage_mb()
        if current_memory_mb + (item.size_bytes / 1024 / 1024) > self.config.memory_cache_size_mb:
            return False
        
        # Store frequently accessed items in memory
        return True
    
    def _store_in_memory(self, item: CacheItem) -> bool:
        """Store item in memory cache"""
        try:
            # Make room if necessary
            while self._get_memory_usage_mb() + (item.size_bytes / 1024 / 1024) > self.config.memory_cache_size_mb:
                if not self._evict_from_memory():
                    return False
            
            # Store item
            self.memory_cache[item.key] = item
            self.stats.memory_items += 1
            self.stats.memory_bytes += item.size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing item in memory: {str(e)}")
            return False
    
    def _store_on_disk(self, item: CacheItem) -> bool:
        """Store item on disk"""
        try:
            # Generate file path
            file_path = self._generate_file_path(item.key)
            
            # Compress data if enabled
            data_to_store = item.data
            if item.compressed:
                data_to_store = self._compress_data(data_to_store)
            
            # Store data
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_store, f)
            
            # Update index
            self.disk_cache_index[item.key] = file_path
            
            # Update persistence
            if self.config.enable_persistence:
                self._persist_item_metadata(item, file_path)
            
            self.stats.disk_items += 1
            self.stats.disk_bytes += item.size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing item on disk: {str(e)}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[CacheItem]:
        """Load item from disk"""
        try:
            file_path = self.disk_cache_index.get(key)
            if not file_path or not os.path.exists(file_path):
                return None
            
            # Load data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load metadata
            metadata = self._load_item_metadata(key)
            if metadata is None:
                return None
            
            # Decompress if necessary
            if metadata.get('compressed', False):
                data = self._decompress_data(data)
            
            # Create cache item
            item = CacheItem(
                key=key,
                data=data,
                size_bytes=metadata.get('size_bytes', 0),
                timestamp=metadata.get('timestamp', time.time()),
                access_count=metadata.get('access_count', 0),
                last_access=metadata.get('last_access', time.time()),
                ttl=metadata.get('ttl'),
                compressed=metadata.get('compressed', False)
            )
            
            return item
            
        except Exception as e:
            logger.error(f"Error loading item from disk: {str(e)}")
            return None
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using configured compression"""
        serialized = pickle.dumps(data)
        return lzma.compress(serialized, preset=self.config.compression_level)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data"""
        decompressed = lzma.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _generate_file_path(self, key: str) -> str:
        """Generate file path for cache item"""
        # Create hash of key for filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Create nested directory structure
        dir_path = Path(self.config.cache_directory) / "disk" / key_hash[:2] / key_hash[2:4]
        dir_path.mkdir(parents=True, exist_ok=True)
        
        return str(dir_path / f"{key_hash}.cache")
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data"""
        try:
            if hasattr(data, 'memory_usage'):
                return int(data.memory_usage(deep=True).sum())
            else:
                return len(pickle.dumps(data))
        except Exception:
            return 1024  # Default size
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory cache usage in MB"""
        return self.stats.memory_bytes / 1024 / 1024
    
    def _evict_from_memory(self) -> bool:
        """Evict item from memory cache"""
        if not self.memory_cache:
            return False
        
        # Choose eviction strategy
        if self.config.eviction_policy == "lru":
            key_to_evict = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].last_access)
        elif self.config.eviction_policy == "lfu":
            key_to_evict = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].access_count)
        elif self.config.eviction_policy == "ttl":
            # Evict oldest item
            key_to_evict = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].timestamp)
        else:
            # Default to LRU
            key_to_evict = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].last_access)
        
        # Move to disk before evicting
        item = self.memory_cache[key_to_evict]
        if self.config.enable_disk_cache:
            self._store_on_disk(item)
        
        # Remove from memory
        self._remove_from_memory(key_to_evict)
        
        return True
    
    def _remove_from_memory(self, key: str):
        """Remove item from memory cache"""
        if key in self.memory_cache:
            item = self.memory_cache[key]
            del self.memory_cache[key]
            self.stats.memory_items -= 1
            self.stats.memory_bytes -= item.size_bytes
    
    def _remove_from_disk(self, key: str):
        """Remove item from disk cache"""
        if key in self.disk_cache_index:
            file_path = self.disk_cache_index[key]
            del self.disk_cache_index[key]
            
            # Remove file
            try:
                os.remove(file_path)
            except OSError:
                pass
            
            # Remove from persistence
            if self.config.enable_persistence:
                self._remove_item_metadata(key)
            
            self.stats.disk_items -= 1
    
    def _should_promote_to_memory(self, item: CacheItem) -> bool:
        """Determine if item should be promoted to memory"""
        if not self.config.enable_memory_cache:
            return False
        
        # Promote frequently accessed items
        return item.access_count > 5
    
    def _promote_to_memory(self, item: CacheItem):
        """Promote item from disk to memory"""
        if self._should_store_in_memory(item):
            self._store_in_memory(item)
    
    def _persist_item_metadata(self, item: CacheItem, file_path: str):
        """Persist item metadata to database"""
        try:
            db_path = Path(self.config.cache_directory) / "metadata" / "cache.db"
            
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_items 
                    (key, size_bytes, timestamp, access_count, last_access, ttl, compressed, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.key,
                    item.size_bytes,
                    item.timestamp,
                    item.access_count,
                    item.last_access,
                    item.ttl,
                    int(item.compressed),
                    file_path
                ))
        except Exception as e:
            logger.error(f"Error persisting item metadata: {str(e)}")
    
    def _load_item_metadata(self, key: str) -> Optional[Dict]:
        """Load item metadata from database"""
        try:
            db_path = Path(self.config.cache_directory) / "metadata" / "cache.db"
            
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("""
                    SELECT size_bytes, timestamp, access_count, last_access, ttl, compressed, file_path
                    FROM cache_items WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'size_bytes': row[0],
                        'timestamp': row[1],
                        'access_count': row[2],
                        'last_access': row[3],
                        'ttl': row[4],
                        'compressed': bool(row[5]),
                        'file_path': row[6]
                    }
                
                return None
        except Exception as e:
            logger.error(f"Error loading item metadata: {str(e)}")
            return None
    
    def _remove_item_metadata(self, key: str):
        """Remove item metadata from database"""
        try:
            db_path = Path(self.config.cache_directory) / "metadata" / "cache.db"
            
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
        except Exception as e:
            logger.error(f"Error removing item metadata: {str(e)}")
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                self._cleanup_expired_items()
                self._cleanup_disk_space()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}")
                time.sleep(60)
    
    def _cleanup_expired_items(self):
        """Clean up expired items"""
        with self._lock:
            # Clean memory cache
            expired_keys = []
            for key, item in self.memory_cache.items():
                if item.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_memory(key)
            
            # Clean disk cache
            expired_keys = []
            for key in self.disk_cache_index:
                metadata = self._load_item_metadata(key)
                if metadata:
                    ttl = metadata.get('ttl')
                    timestamp = metadata.get('timestamp', 0)
                    if ttl and time.time() - timestamp > ttl:
                        expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_disk(key)
    
    def _cleanup_disk_space(self):
        """Clean up disk space if over limit"""
        cache_path = Path(self.config.cache_directory)
        
        # Calculate current disk usage
        total_size = sum(
            f.stat().st_size 
            for f in cache_path.rglob('*') 
            if f.is_file()
        )
        
        max_size_bytes = self.config.max_size_gb * 1024 * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Remove oldest items until under limit
            items_to_remove = []
            
            if self.config.enable_persistence:
                db_path = cache_path / "metadata" / "cache.db"
                
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute("""
                        SELECT key, size_bytes, last_access 
                        FROM cache_items 
                        ORDER BY last_access ASC
                    """)
                    
                    for row in cursor:
                        items_to_remove.append(row[0])
                        total_size -= row[1]
                        
                        if total_size <= max_size_bytes * 0.8:  # Clean to 80% of limit
                            break
            
            for key in items_to_remove:
                self._remove_from_disk(key)
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while True:
            try:
                self._log_cache_stats()
                time.sleep(self.config.stats_interval)
            except Exception as e:
                logger.error(f"Error in monitoring worker: {str(e)}")
                time.sleep(60)
    
    def _log_cache_stats(self):
        """Log cache statistics"""
        stats = self.get_stats()
        logger.info(f"Cache Stats: {stats}")
    
    def _prefetch_worker(self):
        """Background prefetching worker"""
        # Implementation would depend on specific prefetching strategy
        while True:
            try:
                # Prefetch logic would go here
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in prefetch worker: {str(e)}")
                time.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = self.stats.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': hit_rate,
                'memory_hits': self.stats.memory_hits,
                'disk_hits': self.stats.disk_hits,
                'memory_items': self.stats.memory_items,
                'disk_items': self.stats.disk_items,
                'memory_bytes': self.stats.memory_bytes,
                'disk_bytes': self.stats.disk_bytes,
                'memory_usage_mb': self._get_memory_usage_mb(),
                'evictions': self.stats.evictions
            }
    
    def shutdown(self):
        """Shutdown cache manager"""
        logger.info("Cache manager shutting down")


class CacheStats:
    """Cache statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.memory_items = 0
        self.disk_items = 0
        self.memory_bytes = 0
        self.disk_bytes = 0
        self.evictions = 0
    
    def reset(self):
        """Reset statistics"""
        self.hits = 0
        self.misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.memory_items = 0
        self.disk_items = 0
        self.memory_bytes = 0
        self.disk_bytes = 0
        self.evictions = 0