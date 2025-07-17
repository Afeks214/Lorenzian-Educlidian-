"""
Distributed Lock Manager for Multi-Instance Race Condition Prevention
=====================================================================

This module provides distributed locking mechanisms using Redis and etcd
for ensuring atomicity across multiple application instances.

Features:
- Redis-based distributed locks with lua scripts for atomicity
- etcd-based distributed locks with lease mechanisms
- Consensus-based locking for critical operations
- Leader election for single-writer scenarios
- Automatic failover and recovery

Author: Agent Beta - Race Condition Elimination Specialist
"""

import asyncio
import time
import uuid
import json
import hashlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
import structlog

# Redis imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# etcd imports
try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class DistributedLockResult:
    """Result of distributed lock operation"""
    success: bool
    lock_id: str
    token: Optional[str] = None
    expires_at: Optional[float] = None
    error: Optional[str] = None
    retry_after: Optional[float] = None


@dataclass
class LeaderElectionResult:
    """Result of leader election"""
    is_leader: bool
    leader_id: str
    term: int
    expires_at: Optional[float] = None


class DistributedLock(ABC):
    """Abstract base class for distributed locks"""
    
    @abstractmethod
    async def acquire(self, timeout: Optional[float] = None) -> DistributedLockResult:
        """Acquire the distributed lock"""
        pass
        
    @abstractmethod
    async def release(self, token: str) -> bool:
        """Release the distributed lock"""
        pass
        
    @abstractmethod
    async def renew(self, token: str, ttl: float) -> bool:
        """Renew the lock lease"""
        pass
        
    @abstractmethod
    async def is_locked(self) -> bool:
        """Check if the lock is currently held"""
        pass


class RedisDistributedLock(DistributedLock):
    """
    Redis-based distributed lock implementation using Lua scripts
    for atomic operations
    """
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 lock_name: str,
                 ttl: float = 30.0,
                 retry_delay: float = 0.1,
                 max_retries: int = 100):
        self.redis_client = redis_client
        self.lock_name = lock_name
        self.ttl = ttl
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.lock_key = f"lock:{lock_name}"
        
        # Lua script for atomic lock acquisition
        self.acquire_script = """
        local lock_key = KEYS[1]
        local token = ARGV[1]
        local ttl = tonumber(ARGV[2])
        
        if redis.call('exists', lock_key) == 0 then
            redis.call('setex', lock_key, ttl, token)
            return 1
        else
            return 0
        end
        """
        
        # Lua script for atomic lock release
        self.release_script = """
        local lock_key = KEYS[1]
        local token = ARGV[1]
        
        if redis.call('get', lock_key) == token then
            return redis.call('del', lock_key)
        else
            return 0
        end
        """
        
        # Lua script for lock renewal
        self.renew_script = """
        local lock_key = KEYS[1]
        local token = ARGV[1]
        local ttl = tonumber(ARGV[2])
        
        if redis.call('get', lock_key) == token then
            redis.call('expire', lock_key, ttl)
            return 1
        else
            return 0
        end
        """
        
    async def acquire(self, timeout: Optional[float] = None) -> DistributedLockResult:
        """Acquire the Redis distributed lock"""
        token = str(uuid.uuid4())
        start_time = time.time()
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                # Try to acquire lock using Lua script
                result = await self.redis_client.eval(
                    self.acquire_script,
                    1,
                    self.lock_key,
                    token,
                    int(self.ttl)
                )
                
                if result == 1:
                    expires_at = time.time() + self.ttl
                    logger.debug("Redis lock acquired", 
                               lock_name=self.lock_name, 
                               token=token[:8])
                    return DistributedLockResult(
                        success=True,
                        lock_id=self.lock_name,
                        token=token,
                        expires_at=expires_at
                    )
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    return DistributedLockResult(
                        success=False,
                        lock_id=self.lock_name,
                        error="Timeout waiting for lock acquisition"
                    )
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay)
                attempts += 1
                
            except Exception as e:
                logger.error("Error acquiring Redis lock", 
                           lock_name=self.lock_name, 
                           error=str(e))
                return DistributedLockResult(
                    success=False,
                    lock_id=self.lock_name,
                    error=str(e)
                )
        
        return DistributedLockResult(
            success=False,
            lock_id=self.lock_name,
            error="Max retries exceeded"
        )
        
    async def release(self, token: str) -> bool:
        """Release the Redis distributed lock"""
        try:
            result = await self.redis_client.eval(
                self.release_script,
                1,
                self.lock_key,
                token
            )
            
            if result == 1:
                logger.debug("Redis lock released", 
                           lock_name=self.lock_name, 
                           token=token[:8])
                return True
            else:
                logger.warning("Failed to release Redis lock - token mismatch",
                             lock_name=self.lock_name,
                             token=token[:8])
                return False
                
        except Exception as e:
            logger.error("Error releasing Redis lock", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False
            
    async def renew(self, token: str, ttl: float) -> bool:
        """Renew the Redis lock lease"""
        try:
            result = await self.redis_client.eval(
                self.renew_script,
                1,
                self.lock_key,
                token,
                int(ttl)
            )
            
            if result == 1:
                logger.debug("Redis lock renewed", 
                           lock_name=self.lock_name, 
                           token=token[:8])
                return True
            else:
                logger.warning("Failed to renew Redis lock - token mismatch",
                             lock_name=self.lock_name,
                             token=token[:8])
                return False
                
        except Exception as e:
            logger.error("Error renewing Redis lock", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False
            
    async def is_locked(self) -> bool:
        """Check if the Redis lock is currently held"""
        try:
            result = await self.redis_client.exists(self.lock_key)
            return result == 1
        except Exception as e:
            logger.error("Error checking Redis lock status", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False


class EtcdDistributedLock(DistributedLock):
    """
    etcd-based distributed lock implementation using lease mechanism
    """
    
    def __init__(self, 
                 etcd_client: etcd3.Etcd3Client,
                 lock_name: str,
                 ttl: float = 30.0):
        self.etcd_client = etcd_client
        self.lock_name = lock_name
        self.ttl = ttl
        self.lock_key = f"/locks/{lock_name}"
        self.lease = None
        
    async def acquire(self, timeout: Optional[float] = None) -> DistributedLockResult:
        """Acquire the etcd distributed lock"""
        try:
            # Create lease
            self.lease = self.etcd_client.lease(int(self.ttl))
            token = str(uuid.uuid4())
            
            # Try to acquire lock
            success = self.etcd_client.transaction(
                compare=[
                    self.etcd_client.transactions.create(self.lock_key) == 0
                ],
                success=[
                    self.etcd_client.transactions.put(
                        self.lock_key, 
                        token, 
                        lease=self.lease
                    )
                ],
                failure=[]
            )
            
            if success:
                expires_at = time.time() + self.ttl
                logger.debug("etcd lock acquired", 
                           lock_name=self.lock_name, 
                           token=token[:8])
                return DistributedLockResult(
                    success=True,
                    lock_id=self.lock_name,
                    token=token,
                    expires_at=expires_at
                )
            else:
                return DistributedLockResult(
                    success=False,
                    lock_id=self.lock_name,
                    error="Lock already held"
                )
                
        except Exception as e:
            logger.error("Error acquiring etcd lock", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return DistributedLockResult(
                success=False,
                lock_id=self.lock_name,
                error=str(e)
            )
            
    async def release(self, token: str) -> bool:
        """Release the etcd distributed lock"""
        try:
            # Delete the lock key
            self.etcd_client.delete(self.lock_key)
            
            # Revoke lease
            if self.lease:
                self.etcd_client.revoke_lease(self.lease.id)
                self.lease = None
                
            logger.debug("etcd lock released", 
                       lock_name=self.lock_name, 
                       token=token[:8])
            return True
            
        except Exception as e:
            logger.error("Error releasing etcd lock", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False
            
    async def renew(self, token: str, ttl: float) -> bool:
        """Renew the etcd lock lease"""
        try:
            if self.lease:
                self.etcd_client.refresh_lease(self.lease.id)
                logger.debug("etcd lock renewed", 
                           lock_name=self.lock_name, 
                           token=token[:8])
                return True
            else:
                logger.warning("No active lease to renew",
                             lock_name=self.lock_name)
                return False
                
        except Exception as e:
            logger.error("Error renewing etcd lock", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False
            
    async def is_locked(self) -> bool:
        """Check if the etcd lock is currently held"""
        try:
            value, _ = self.etcd_client.get(self.lock_key)
            return value is not None
        except Exception as e:
            logger.error("Error checking etcd lock status", 
                       lock_name=self.lock_name, 
                       error=str(e))
            return False


class ConsensusLock:
    """
    Consensus-based distributed lock using multiple backends
    """
    
    def __init__(self, 
                 locks: List[DistributedLock],
                 quorum_size: Optional[int] = None):
        self.locks = locks
        self.quorum_size = quorum_size or (len(locks) // 2 + 1)
        self.acquired_tokens: Dict[str, str] = {}
        
    async def acquire(self, timeout: Optional[float] = None) -> DistributedLockResult:
        """Acquire consensus lock from majority of backends"""
        start_time = time.time()
        acquired_locks = []
        
        try:
            # Try to acquire from all backends
            tasks = []
            for lock in self.locks:
                task = asyncio.create_task(lock.acquire(timeout))
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if we got quorum
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, DistributedLockResult) and result.success:
                    successful_results.append((i, result))
                    acquired_locks.append(i)
                    
            if len(successful_results) >= self.quorum_size:
                # Store tokens for release
                for i, result in successful_results:
                    self.acquired_tokens[str(i)] = result.token
                    
                logger.info("Consensus lock acquired", 
                          acquired_count=len(successful_results),
                          quorum_size=self.quorum_size)
                          
                return DistributedLockResult(
                    success=True,
                    lock_id="consensus",
                    token=json.dumps(self.acquired_tokens)
                )
            else:
                # Release any acquired locks
                await self._release_acquired_locks(acquired_locks)
                
                return DistributedLockResult(
                    success=False,
                    lock_id="consensus",
                    error=f"Failed to acquire quorum ({len(successful_results)}/{self.quorum_size})"
                )
                
        except Exception as e:
            # Release any acquired locks
            await self._release_acquired_locks(acquired_locks)
            
            logger.error("Error acquiring consensus lock", error=str(e))
            return DistributedLockResult(
                success=False,
                lock_id="consensus",
                error=str(e)
            )
            
    async def release(self, token: str) -> bool:
        """Release consensus lock from all backends"""
        try:
            tokens = json.loads(token)
            release_tasks = []
            
            for i, backend_token in tokens.items():
                lock_index = int(i)
                if lock_index < len(self.locks):
                    task = asyncio.create_task(
                        self.locks[lock_index].release(backend_token)
                    )
                    release_tasks.append(task)
                    
            results = await asyncio.gather(*release_tasks, return_exceptions=True)
            
            # Count successful releases
            successful_releases = sum(1 for r in results if r is True)
            
            logger.info("Consensus lock released", 
                      released_count=successful_releases,
                      total_locks=len(release_tasks))
                      
            return successful_releases > 0
            
        except Exception as e:
            logger.error("Error releasing consensus lock", error=str(e))
            return False
            
    async def _release_acquired_locks(self, acquired_indices: List[int]):
        """Release locks that were successfully acquired"""
        for i in acquired_indices:
            try:
                if str(i) in self.acquired_tokens:
                    await self.locks[i].release(self.acquired_tokens[str(i)])
            except Exception as e:
                logger.error("Error releasing lock during cleanup", 
                           index=i, error=str(e))


class LeaderElection:
    """
    Leader election using distributed locks
    """
    
    def __init__(self, 
                 distributed_lock: DistributedLock,
                 node_id: str,
                 heartbeat_interval: float = 10.0):
        self.distributed_lock = distributed_lock
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.is_leader = False
        self.current_term = 0
        self.leader_token: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
    async def start_election(self) -> LeaderElectionResult:
        """Start leader election process"""
        try:
            result = await self.distributed_lock.acquire()
            
            if result.success:
                self.is_leader = True
                self.current_term += 1
                self.leader_token = result.token
                
                # Start heartbeat to maintain leadership
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                logger.info("Leader election won", 
                          node_id=self.node_id,
                          term=self.current_term)
                          
                return LeaderElectionResult(
                    is_leader=True,
                    leader_id=self.node_id,
                    term=self.current_term,
                    expires_at=result.expires_at
                )
            else:
                logger.info("Leader election lost", 
                          node_id=self.node_id,
                          error=result.error)
                          
                return LeaderElectionResult(
                    is_leader=False,
                    leader_id="unknown",
                    term=self.current_term
                )
                
        except Exception as e:
            logger.error("Error in leader election", 
                       node_id=self.node_id, 
                       error=str(e))
            return LeaderElectionResult(
                is_leader=False,
                leader_id="unknown",
                term=self.current_term
            )
            
    async def _heartbeat_loop(self):
        """Heartbeat loop to maintain leadership"""
        while not self._shutdown_event.is_set() and self.is_leader:
            try:
                if self.leader_token:
                    renewed = await self.distributed_lock.renew(
                        self.leader_token, 
                        self.heartbeat_interval * 2
                    )
                    
                    if not renewed:
                        logger.warning("Lost leadership - failed to renew", 
                                     node_id=self.node_id)
                        self.is_leader = False
                        break
                        
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in heartbeat loop", 
                           node_id=self.node_id, 
                           error=str(e))
                await asyncio.sleep(self.heartbeat_interval)
                
    async def resign_leadership(self) -> bool:
        """Resign from leadership"""
        if not self.is_leader:
            return True
            
        try:
            self._shutdown_event.set()
            
            # Cancel heartbeat
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                    
            # Release leadership lock
            if self.leader_token:
                released = await self.distributed_lock.release(self.leader_token)
                self.leader_token = None
            else:
                released = True
                
            self.is_leader = False
            
            logger.info("Leadership resigned", 
                      node_id=self.node_id,
                      term=self.current_term)
                      
            return released
            
        except Exception as e:
            logger.error("Error resigning leadership", 
                       node_id=self.node_id, 
                       error=str(e))
            return False


class DistributedLockManager:
    """
    High-level distributed lock manager
    """
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 etcd_client: Optional[etcd3.Etcd3Client] = None,
                 default_ttl: float = 30.0):
        self.redis_client = redis_client
        self.etcd_client = etcd_client
        self.default_ttl = default_ttl
        self.locks: Dict[str, DistributedLock] = {}
        
    def create_redis_lock(self, name: str, ttl: Optional[float] = None) -> RedisDistributedLock:
        """Create a Redis-based distributed lock"""
        if not self.redis_client:
            raise RuntimeError("Redis client not available")
            
        ttl = ttl or self.default_ttl
        lock = RedisDistributedLock(self.redis_client, name, ttl)
        self.locks[name] = lock
        return lock
        
    def create_etcd_lock(self, name: str, ttl: Optional[float] = None) -> EtcdDistributedLock:
        """Create an etcd-based distributed lock"""
        if not self.etcd_client:
            raise RuntimeError("etcd client not available")
            
        ttl = ttl or self.default_ttl
        lock = EtcdDistributedLock(self.etcd_client, name, ttl)
        self.locks[name] = lock
        return lock
        
    def create_consensus_lock(self, name: str, ttl: Optional[float] = None) -> ConsensusLock:
        """Create a consensus-based distributed lock"""
        ttl = ttl or self.default_ttl
        backend_locks = []
        
        if self.redis_client:
            backend_locks.append(RedisDistributedLock(self.redis_client, name, ttl))
            
        if self.etcd_client:
            backend_locks.append(EtcdDistributedLock(self.etcd_client, name, ttl))
            
        if not backend_locks:
            raise RuntimeError("No distributed lock backends available")
            
        consensus_lock = ConsensusLock(backend_locks)
        self.locks[name] = consensus_lock
        return consensus_lock
        
    @asynccontextmanager
    async def distributed_lock(self, name: str, 
                             backend: str = "redis",
                             ttl: Optional[float] = None,
                             timeout: Optional[float] = None):
        """Context manager for distributed lock acquisition"""
        if backend == "redis":
            lock = self.create_redis_lock(name, ttl)
        elif backend == "etcd":
            lock = self.create_etcd_lock(name, ttl)
        elif backend == "consensus":
            lock = self.create_consensus_lock(name, ttl)
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        result = await lock.acquire(timeout)
        
        if not result.success:
            raise RuntimeError(f"Failed to acquire distributed lock {name}: {result.error}")
            
        try:
            yield result
        finally:
            if result.token:
                await lock.release(result.token)
            
    async def cleanup_expired_locks(self) -> int:
        """Cleanup expired locks"""
        cleaned_count = 0
        
        for name, lock in list(self.locks.items()):
            try:
                if not await lock.is_locked():
                    del self.locks[name]
                    cleaned_count += 1
            except Exception as e:
                logger.error("Error checking lock status", 
                           lock_name=name, 
                           error=str(e))
                
        return cleaned_count
        
    async def get_lock_status(self) -> Dict[str, Any]:
        """Get status of all distributed locks"""
        status = {}
        
        for name, lock in self.locks.items():
            try:
                is_locked = await lock.is_locked()
                status[name] = {
                    'locked': is_locked,
                    'type': type(lock).__name__
                }
            except Exception as e:
                status[name] = {
                    'locked': False,
                    'error': str(e),
                    'type': type(lock).__name__
                }
                
        return status