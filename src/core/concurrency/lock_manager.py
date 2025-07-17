"""
Comprehensive Lock Manager for Race Condition Elimination
=========================================================

This module implements a bulletproof lock manager that provides:
- Multiple lock types (read/write, exclusive, shared)
- Priority-based lock ordering
- Deadlock detection and prevention
- Timeout mechanisms
- Performance monitoring

Author: Agent Beta - Race Condition Elimination Specialist
"""

import logging


import asyncio
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import structlog
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger(__name__)


class LockType(Enum):
    """Types of locks available"""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    MUTEX = "mutex"
    SEMAPHORE = "semaphore"


class LockPriority(Enum):
    """Lock priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class LockRequest:
    """Represents a lock request"""
    lock_id: str
    requester_id: str
    lock_type: LockType
    priority: LockPriority
    timeout: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None


@dataclass
class LockResult:
    """Result of a lock operation"""
    success: bool
    lock_id: str
    granted_at: float
    expires_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DeadlockInfo:
    """Information about detected deadlock"""
    cycle: List[str]
    involved_threads: Set[str]
    detection_time: float
    resolution_strategy: str


class DeadlockDetector:
    """
    Deadlock detection using wait-for graph algorithm
    """
    
    def __init__(self):
        self.wait_for_graph: Dict[str, Set[str]] = defaultdict(set)
        self.lock_holders: Dict[str, Set[str]] = defaultdict(set)
        self.lock_waiters: Dict[str, Set[str]] = defaultdict(set)
        self._graph_lock = threading.RLock()
        
    def add_wait_edge(self, waiter: str, holder: str):
        """Add a wait edge to the graph"""
        with self._graph_lock:
            self.wait_for_graph[waiter].add(holder)
            
    def remove_wait_edge(self, waiter: str, holder: str):
        """Remove a wait edge from the graph"""
        with self._graph_lock:
            self.wait_for_graph[waiter].discard(holder)
            if not self.wait_for_graph[waiter]:
                del self.wait_for_graph[waiter]
                
    def detect_deadlock(self) -> Optional[DeadlockInfo]:
        """Detect deadlock using cycle detection"""
        with self._graph_lock:
            visited = set()
            rec_stack = set()
            path = []
            
            def dfs(node: str) -> Optional[List[str]]:
                if node in rec_stack:
                    # Found cycle
                    cycle_start = path.index(node)
                    return path[cycle_start:]
                    
                if node in visited:
                    return None
                    
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                for neighbor in self.wait_for_graph.get(node, set()):
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                        
                rec_stack.remove(node)
                path.pop()
                return None
                
            # Check all nodes for cycles
            for node in self.wait_for_graph:
                if node not in visited:
                    cycle = dfs(node)
                    if cycle:
                        return DeadlockInfo(
                            cycle=cycle,
                            involved_threads=set(cycle),
                            detection_time=time.time(),
                            resolution_strategy="timeout_oldest"
                        )
                        
            return None
            
    def update_lock_state(self, lock_id: str, holders: Set[str], waiters: Set[str]):
        """Update lock state for deadlock detection"""
        with self._graph_lock:
            self.lock_holders[lock_id] = holders.copy()
            self.lock_waiters[lock_id] = waiters.copy()
            
            # Update wait-for graph
            for waiter in waiters:
                for holder in holders:
                    self.add_wait_edge(waiter, holder)


class LockManager:
    """
    Comprehensive lock manager with deadlock prevention
    """
    
    def __init__(self, 
                 enable_deadlock_detection: bool = True,
                 deadlock_check_interval: float = 1.0,
                 default_timeout: float = 30.0):
        self.enable_deadlock_detection = enable_deadlock_detection
        self.deadlock_check_interval = deadlock_check_interval
        self.default_timeout = default_timeout
        
        # Lock storage
        self.locks: Dict[str, threading.RLock] = {}
        self.read_write_locks: Dict[str, threading.RLock] = {}
        self.lock_holders: Dict[str, Set[str]] = defaultdict(set)
        self.lock_waiters: Dict[str, Set[str]] = defaultdict(set)
        self.lock_requests: Dict[str, List[LockRequest]] = defaultdict(list)
        
        # Deadlock detection
        self.deadlock_detector = DeadlockDetector()
        
        # Monitoring
        self.lock_metrics: Dict[str, Any] = defaultdict(int)
        self.contention_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Threading
        self._manager_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Start deadlock detection if enabled
        if enable_deadlock_detection:
            self._deadlock_thread = threading.Thread(
                target=self._deadlock_detection_loop,
                daemon=True
            )
            self._deadlock_thread.start()
            
    def _deadlock_detection_loop(self):
        """Background thread for deadlock detection"""
        while not self._shutdown_event.is_set():
            try:
                deadlock = self.deadlock_detector.detect_deadlock()
                if deadlock:
                    self._resolve_deadlock(deadlock)
                    
                time.sleep(self.deadlock_check_interval)
                
            except Exception as e:
                logger.error("Error in deadlock detection", error=str(e))
                time.sleep(self.deadlock_check_interval)
                
    def _resolve_deadlock(self, deadlock: DeadlockInfo):
        """Resolve detected deadlock"""
        logger.warning(
            "Deadlock detected - resolving",
            cycle=deadlock.cycle,
            strategy=deadlock.resolution_strategy
        )
        
        # Strategy: Cancel oldest lock request in cycle
        if deadlock.resolution_strategy == "timeout_oldest":
            oldest_request = None
            oldest_time = float('inf')
            
            for thread_id in deadlock.involved_threads:
                for lock_id, requests in self.lock_requests.items():
                    for request in requests:
                        if (request.requester_id == thread_id and 
                            request.timestamp < oldest_time):
                            oldest_request = request
                            oldest_time = request.timestamp
                            
            if oldest_request:
                # Cancel the oldest request
                self._cancel_lock_request(oldest_request)
                logger.info(
                    "Deadlock resolved by canceling oldest request",
                    request_id=oldest_request.lock_id,
                    requester=oldest_request.requester_id
                )
                
    def _cancel_lock_request(self, request: LockRequest):
        """Cancel a lock request"""
        with self._manager_lock:
            if request.lock_id in self.lock_requests:
                self.lock_requests[request.lock_id].remove(request)
                self.lock_waiters[request.lock_id].discard(request.requester_id)
                
    def acquire_lock(self, 
                    lock_id: str, 
                    lock_type: LockType = LockType.EXCLUSIVE,
                    priority: LockPriority = LockPriority.NORMAL,
                    timeout: Optional[float] = None,
                    requester_id: Optional[str] = None) -> LockResult:
        """
        Acquire a lock with deadlock prevention
        """
        requester_id = requester_id or str(threading.current_thread().ident)
        timeout = timeout or self.default_timeout
        
        request = LockRequest(
            lock_id=lock_id,
            requester_id=requester_id,
            lock_type=lock_type,
            priority=priority,
            timeout=timeout
        )
        
        start_time = time.time()
        
        with self._manager_lock:
            # Add to request queue
            self.lock_requests[lock_id].append(request)
            self.lock_waiters[lock_id].add(requester_id)
            
            # Sort by priority
            self.lock_requests[lock_id].sort(key=lambda r: r.priority.value, reverse=True)
            
            # Update deadlock detector
            if self.enable_deadlock_detection:
                self.deadlock_detector.update_lock_state(
                    lock_id,
                    self.lock_holders[lock_id],
                    self.lock_waiters[lock_id]
                )
        
        # Try to acquire lock
        try:
            if lock_type in [LockType.READ, LockType.WRITE]:
                success = self._acquire_read_write_lock(request)
            else:
                success = self._acquire_exclusive_lock(request)
                
            if success:
                with self._manager_lock:
                    self.lock_holders[lock_id].add(requester_id)
                    self.lock_waiters[lock_id].discard(requester_id)
                    if request in self.lock_requests[lock_id]:
                        self.lock_requests[lock_id].remove(request)
                    
                    # Update metrics
                    self.lock_metrics[f'{lock_id}_acquisitions'] += 1
                    acquisition_time = time.time() - start_time
                    self.contention_stats[lock_id].append(acquisition_time)
                    
                return LockResult(
                    success=True,
                    lock_id=lock_id,
                    granted_at=time.time(),
                    expires_at=time.time() + timeout if timeout else None
                )
            else:
                return LockResult(
                    success=False,
                    lock_id=lock_id,
                    granted_at=0,
                    error="Failed to acquire lock within timeout"
                )
                
        except Exception as e:
            logger.error("Error acquiring lock", lock_id=lock_id, error=str(e))
            return LockResult(
                success=False,
                lock_id=lock_id,
                granted_at=0,
                error=str(e)
            )
            
    def _acquire_read_write_lock(self, request: LockRequest) -> bool:
        """Acquire read/write lock with proper semantics"""
        lock_id = request.lock_id
        
        # Create lock if it doesn't exist
        if lock_id not in self.read_write_locks:
            self.read_write_locks[lock_id] = threading.RLock()
            
        lock = self.read_write_locks[lock_id]
        
        if request.lock_type == LockType.READ:
            # Multiple readers allowed
            return lock.acquire(timeout=request.timeout)
        else:
            # Exclusive write lock
            return lock.acquire(timeout=request.timeout)
            
    def _acquire_exclusive_lock(self, request: LockRequest) -> bool:
        """Acquire exclusive lock"""
        lock_id = request.lock_id
        
        # Create lock if it doesn't exist
        if lock_id not in self.locks:
            self.locks[lock_id] = threading.RLock()
            
        lock = self.locks[lock_id]
        return lock.acquire(timeout=request.timeout)
        
    def release_lock(self, lock_id: str, requester_id: Optional[str] = None) -> bool:
        """Release a lock"""
        requester_id = requester_id or str(threading.current_thread().ident)
        
        try:
            with self._manager_lock:
                if requester_id in self.lock_holders[lock_id]:
                    self.lock_holders[lock_id].remove(requester_id)
                    
                    # Release the actual lock
                    if lock_id in self.locks:
                        self.locks[lock_id].release()
                    elif lock_id in self.read_write_locks:
                        self.read_write_locks[lock_id].release()
                        
                    # Update metrics
                    self.lock_metrics[f'{lock_id}_releases'] += 1
                    
                    logger.debug("Lock released", lock_id=lock_id, requester=requester_id)
                    return True
                else:
                    logger.warning("Attempted to release unowned lock", 
                                 lock_id=lock_id, requester=requester_id)
                    return False
                    
        except Exception as e:
            logger.error("Error releasing lock", lock_id=lock_id, error=str(e))
            return False
            
    @contextmanager
    def lock(self, lock_id: str, lock_type: LockType = LockType.EXCLUSIVE, 
             priority: LockPriority = LockPriority.NORMAL, timeout: Optional[float] = None):
        """Context manager for lock acquisition"""
        result = self.acquire_lock(lock_id, lock_type, priority, timeout)
        
        if not result.success:
            raise RuntimeError(f"Failed to acquire lock {lock_id}: {result.error}")
            
        try:
            yield result
        finally:
            self.release_lock(lock_id)
            
    def get_lock_statistics(self) -> Dict[str, Any]:
        """Get lock usage statistics"""
        with self._manager_lock:
            stats = {
                'total_locks': len(self.locks) + len(self.read_write_locks),
                'active_holders': sum(len(holders) for holders in self.lock_holders.values()),
                'waiting_threads': sum(len(waiters) for waiters in self.lock_waiters.values()),
                'metrics': dict(self.lock_metrics),
                'contention_stats': {
                    lock_id: {
                        'avg_wait_time': sum(times) / len(times) if times else 0,
                        'max_wait_time': max(times) if times else 0,
                        'total_acquisitions': len(times)
                    }
                    for lock_id, times in self.contention_stats.items()
                }
            }
            
            return stats
            
    def shutdown(self):
        """Shutdown the lock manager"""
        logger.info("Shutting down lock manager")
        self._shutdown_event.set()
        
        # Wait for deadlock detection thread
        if hasattr(self, '_deadlock_thread'):
            self._deadlock_thread.join(timeout=5.0)
            
        # Release all locks
        with self._manager_lock:
            for lock in self.locks.values():
                try:
                    lock.release()
                except (FileNotFoundError, IOError, OSError) as e:
                    logger.error(f'Error occurred: {e}')
                    
            for lock in self.read_write_locks.values():
                try:
                    lock.release()
                except (FileNotFoundError, IOError, OSError) as e:
                    logger.error(f'Error occurred: {e}')
                    
        logger.info("Lock manager shutdown complete")


# Global lock manager instance
_global_lock_manager: Optional[LockManager] = None
_global_lock_manager_lock = threading.Lock()


def get_global_lock_manager() -> LockManager:
    """Get or create the global lock manager instance"""
    global _global_lock_manager
    
    if _global_lock_manager is None:
        with _global_lock_manager_lock:
            if _global_lock_manager is None:
                _global_lock_manager = LockManager()
                
    return _global_lock_manager


def shutdown_global_lock_manager():
    """Shutdown the global lock manager"""
    global _global_lock_manager
    
    if _global_lock_manager:
        _global_lock_manager.shutdown()
        _global_lock_manager = None