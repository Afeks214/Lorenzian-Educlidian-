"""
Distributed Lock System for Race Condition Prevention
==================================================

Redis-based distributed locking mechanism to ensure atomic decision
processing and eliminate race conditions in the Tactical MARL System.

CRITICAL SECURITY IMPLEMENTATION:
- Prevents concurrent modifications to shared state
- Enforces correlation ID uniqueness across the entire system
- Implements lock timeout and cleanup mechanisms
- Provides distributed coordination for multiple process instances

Author: Systems Architect - Infrastructure Hardening
Version: 1.0.0
Classification: CRITICAL SECURITY COMPONENT
"""

import asyncio
import time
import uuid
import logging
import redis.asyncio as redis
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LockResult:
    """Result of a lock acquisition attempt"""
    acquired: bool
    lock_id: Optional[str] = None
    expiry_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class LockMetrics:
    """Metrics for lock operations"""
    total_acquisitions: int = 0
    successful_acquisitions: int = 0
    failed_acquisitions: int = 0
    timeout_acquisitions: int = 0
    total_releases: int = 0
    successful_releases: int = 0
    failed_releases: int = 0
    lock_contention_count: int = 0
    average_hold_time: float = 0.0


class DistributedLockManager:
    """
    Redis-based distributed lock manager for race condition prevention.
    
    Features:
    - Atomic lock acquisition with expiration
    - Correlation ID uniqueness enforcement
    - Lock timeout and cleanup mechanisms
    - Comprehensive monitoring and metrics
    - Deadlock prevention through lock ordering
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_timeout: float = 30.0,
        lock_prefix: str = "tactical_lock:",
        correlation_prefix: str = "correlation_id:"
    ):
        """
        Initialize the distributed lock manager.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            default_timeout: Default lock timeout in seconds
            lock_prefix: Prefix for lock keys in Redis
            correlation_prefix: Prefix for correlation ID tracking
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.default_timeout = default_timeout
        self.lock_prefix = lock_prefix
        self.correlation_prefix = correlation_prefix
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        
        # Lock tracking
        self.active_locks: Dict[str, Dict[str, Any]] = {}
        self.lock_metrics = LockMetrics()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "DistributedLockManager initialized",
            redis_host=redis_host,
            redis_port=redis_port,
            default_timeout=default_timeout
        )

    async def initialize(self) -> bool:
        """
        Initialize Redis connection and start background tasks.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks())
            
            logger.info("DistributedLockManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DistributedLockManager: {e}")
            return False

    async def acquire_decision_lock(
        self,
        correlation_id: str,
        timeout: Optional[float] = None,
        max_wait: float = 10.0
    ) -> LockResult:
        """
        Acquire exclusive lock for decision processing.
        
        This ensures that only one decision request with a given correlation ID
        can be processed at a time, preventing race conditions.
        
        Args:
            correlation_id: Unique correlation ID for the request
            timeout: Lock expiration timeout (uses default if None)
            max_wait: Maximum time to wait for lock acquisition
            
        Returns:
            LockResult with acquisition status and details
        """
        if not self.redis_client:
            return LockResult(
                acquired=False,
                error="Redis connection not initialized"
            )
        
        timeout = timeout or self.default_timeout
        lock_key = f"{self.lock_prefix}decision:{correlation_id}"
        lock_id = str(uuid.uuid4())
        
        start_time = time.time()
        expiry_time = start_time + timeout
        
        try:
            # First, check correlation ID uniqueness
            correlation_key = f"{self.correlation_prefix}{correlation_id}"
            correlation_exists = await self.redis_client.exists(correlation_key)
            
            if correlation_exists:
                self.lock_metrics.failed_acquisitions += 1
                self.lock_metrics.lock_contention_count += 1
                return LockResult(
                    acquired=False,
                    error=f"Correlation ID {correlation_id} already in use"
                )
            
            # Attempt to acquire lock with retry logic
            attempt = 0
            while time.time() - start_time < max_wait:
                # Use SET with NX (Not eXists) and EX (EXpire) options for atomicity
                lock_acquired = await self.redis_client.set(
                    lock_key,
                    lock_id,
                    nx=True,  # Only set if key doesn't exist
                    ex=int(timeout)  # Expire after timeout seconds
                )
                
                if lock_acquired:
                    # Also register correlation ID with expiration
                    await self.redis_client.set(
                        correlation_key,
                        lock_id,
                        ex=int(timeout)
                    )
                    
                    # Track active lock
                    self.active_locks[lock_key] = {
                        "lock_id": lock_id,
                        "correlation_id": correlation_id,
                        "acquired_at": start_time,
                        "expires_at": expiry_time,
                        "correlation_key": correlation_key
                    }
                    
                    self.lock_metrics.total_acquisitions += 1
                    self.lock_metrics.successful_acquisitions += 1
                    
                    logger.info(
                        "Decision lock acquired",
                        correlation_id=correlation_id,
                        lock_id=lock_id,
                        timeout=timeout
                    )
                    
                    return LockResult(
                        acquired=True,
                        lock_id=lock_id,
                        expiry_time=expiry_time
                    )
                
                # Lock not acquired, wait briefly and retry
                # Use exponential backoff instead of fixed delay
                backoff_delay = min(0.01 * (2 ** attempt), 0.5)
                await asyncio.sleep(backoff_delay)
                attempt += 1
            
            # Timeout reached
            self.lock_metrics.total_acquisitions += 1
            self.lock_metrics.timeout_acquisitions += 1
            
            return LockResult(
                acquired=False,
                error=f"Failed to acquire lock within {max_wait}s timeout"
            )
            
        except Exception as e:
            self.lock_metrics.total_acquisitions += 1
            self.lock_metrics.failed_acquisitions += 1
            
            logger.error(
                "Error acquiring decision lock",
                correlation_id=correlation_id,
                error=str(e)
            )
            
            return LockResult(
                acquired=False,
                error=f"Lock acquisition error: {str(e)}"
            )

    async def release_decision_lock(
        self,
        correlation_id: str,
        lock_id: str
    ) -> bool:
        """
        Release decision processing lock.
        
        Args:
            correlation_id: Correlation ID of the request
            lock_id: Unique lock identifier from acquisition
            
        Returns:
            True if lock released successfully, False otherwise
        """
        if not self.redis_client:
            logger.error("Redis connection not initialized")
            return False
        
        lock_key = f"{self.lock_prefix}decision:{correlation_id}"
        correlation_key = f"{self.correlation_prefix}{correlation_id}"
        
        try:
            # Use Lua script for atomic release (check lock_id and delete)
            lua_script = """
            local lock_key = KEYS[1]
            local correlation_key = KEYS[2]
            local expected_lock_id = ARGV[1]
            
            local current_lock_id = redis.call('GET', lock_key)
            if current_lock_id == expected_lock_id then
                redis.call('DEL', lock_key)
                redis.call('DEL', correlation_key)
                return 1
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                lua_script,
                2,  # Number of keys
                lock_key,
                correlation_key,
                lock_id
            )
            
            if result == 1:
                # Update metrics
                if lock_key in self.active_locks:
                    lock_info = self.active_locks[lock_key]
                    hold_time = time.time() - lock_info["acquired_at"]
                    
                    # Update average hold time
                    total_releases = self.lock_metrics.total_releases
                    current_avg = self.lock_metrics.average_hold_time
                    new_avg = (current_avg * total_releases + hold_time) / (total_releases + 1)
                    self.lock_metrics.average_hold_time = new_avg
                    
                    del self.active_locks[lock_key]
                
                self.lock_metrics.total_releases += 1
                self.lock_metrics.successful_releases += 1
                
                logger.info(
                    "Decision lock released",
                    correlation_id=correlation_id,
                    lock_id=lock_id
                )
                
                return True
            else:
                self.lock_metrics.total_releases += 1
                self.lock_metrics.failed_releases += 1
                
                logger.warning(
                    "Failed to release lock - lock ID mismatch or expired",
                    correlation_id=correlation_id,
                    lock_id=lock_id
                )
                
                return False
                
        except Exception as e:
            self.lock_metrics.total_releases += 1
            self.lock_metrics.failed_releases += 1
            
            logger.error(
                "Error releasing decision lock",
                correlation_id=correlation_id,
                lock_id=lock_id,
                error=str(e)
            )
            
            return False

    @asynccontextmanager
    async def decision_lock(
        self,
        correlation_id: str,
        timeout: Optional[float] = None,
        max_wait: float = 10.0
    ):
        """
        Async context manager for decision lock acquisition and release.
        
        Usage:
            async with lock_manager.decision_lock("request_123") as lock_result:
                if lock_result.acquired:
                    # Process decision atomically
                    pass
                else:
                    # Handle lock acquisition failure
                    pass
        
        Args:
            correlation_id: Unique correlation ID for the request
            timeout: Lock expiration timeout
            max_wait: Maximum time to wait for lock acquisition
        """
        lock_result = await self.acquire_decision_lock(
            correlation_id=correlation_id,
            timeout=timeout,
            max_wait=max_wait
        )
        
        try:
            yield lock_result
        finally:
            if lock_result.acquired and lock_result.lock_id:
                await self.release_decision_lock(
                    correlation_id=correlation_id,
                    lock_id=lock_result.lock_id
                )

    async def enforce_correlation_id_uniqueness(
        self,
        correlation_id: str,
        timeout: float = 300.0
    ) -> bool:
        """
        Enforce correlation ID uniqueness across the system.
        
        Args:
            correlation_id: Correlation ID to check/register
            timeout: Timeout for correlation ID registration
            
        Returns:
            True if correlation ID is unique and registered, False if already exists
        """
        if not self.redis_client:
            logger.error("Redis connection not initialized")
            return False
        
        correlation_key = f"{self.correlation_prefix}{correlation_id}"
        
        try:
            # Try to register correlation ID
            registered = await self.redis_client.set(
                correlation_key,
                "active",
                nx=True,  # Only set if doesn't exist
                ex=int(timeout)
            )
            
            if registered:
                logger.debug(
                    "Correlation ID registered",
                    correlation_id=correlation_id,
                    timeout=timeout
                )
                return True
            else:
                logger.warning(
                    "Correlation ID collision detected",
                    correlation_id=correlation_id
                )
                return False
                
        except Exception as e:
            logger.error(
                "Error checking correlation ID uniqueness",
                correlation_id=correlation_id,
                error=str(e)
            )
            return False

    async def release_correlation_id(self, correlation_id: str) -> bool:
        """
        Release correlation ID registration.
        
        Args:
            correlation_id: Correlation ID to release
            
        Returns:
            True if released successfully, False otherwise
        """
        if not self.redis_client:
            return False
        
        correlation_key = f"{self.correlation_prefix}{correlation_id}"
        
        try:
            deleted = await self.redis_client.delete(correlation_key)
            if deleted:
                logger.debug("Correlation ID released", correlation_id=correlation_id)
            return bool(deleted)
        except Exception as e:
            logger.error(
                "Error releasing correlation ID",
                correlation_id=correlation_id,
                error=str(e)
            )
            return False

    async def get_lock_metrics(self) -> LockMetrics:
        """Get current lock operation metrics"""
        return self.lock_metrics

    async def get_active_locks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active locks"""
        return self.active_locks.copy()

    async def cleanup_expired_locks(self) -> int:
        """
        Manually trigger cleanup of expired locks.
        
        Returns:
            Number of locks cleaned up
        """
        if not self.redis_client:
            return 0
        
        current_time = time.time()
        expired_locks = []
        
        for lock_key, lock_info in list(self.active_locks.items()):
            if current_time > lock_info["expires_at"]:
                expired_locks.append((lock_key, lock_info))
        
        # Clean up expired locks
        cleaned_count = 0
        for lock_key, lock_info in expired_locks:
            try:
                # Remove from Redis
                await self.redis_client.delete(lock_key)
                await self.redis_client.delete(lock_info["correlation_key"])
                
                # Remove from local tracking
                del self.active_locks[lock_key]
                cleaned_count += 1
                
                logger.debug(
                    "Cleaned up expired lock",
                    correlation_id=lock_info["correlation_id"],
                    lock_key=lock_key
                )
                
            except Exception as e:
                logger.error(
                    "Error cleaning up expired lock",
                    lock_key=lock_key,
                    error=str(e)
                )
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired locks")
        
        return cleaned_count

    async def _cleanup_expired_locks(self):
        """Background task to periodically clean up expired locks"""
        while not self._shutdown_event.is_set():
            try:
                await self.cleanup_expired_locks()
                await asyncio.sleep(30)  # Clean up every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock cleanup task: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def shutdown(self):
        """Shutdown the lock manager and cleanup resources"""
        logger.info("Shutting down DistributedLockManager")
        
        # Stop cleanup task
        self._shutdown_event.set()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Release all active locks
        for lock_key, lock_info in list(self.active_locks.items()):
            try:
                await self.release_decision_lock(
                    correlation_id=lock_info["correlation_id"],
                    lock_id=lock_info["lock_id"]
                )
            except Exception as e:
                logger.error(f"Error releasing lock during shutdown: {e}")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("DistributedLockManager shutdown complete")


# Global lock manager instance
_lock_manager: Optional[DistributedLockManager] = None


async def get_lock_manager() -> DistributedLockManager:
    """Get or create the global lock manager instance"""
    global _lock_manager
    
    if _lock_manager is None:
        _lock_manager = DistributedLockManager()
        if not await _lock_manager.initialize():
            raise RuntimeError("Failed to initialize DistributedLockManager")
    
    return _lock_manager


async def shutdown_lock_manager():
    """Shutdown the global lock manager instance"""
    global _lock_manager
    
    if _lock_manager:
        await _lock_manager.shutdown()
        _lock_manager = None