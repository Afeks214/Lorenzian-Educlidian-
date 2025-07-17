"""
Comprehensive Concurrency Framework for Race Condition Elimination
=================================================================

This module provides a bulletproof concurrency framework designed to eliminate
ALL race conditions, deadlocks, and concurrency issues in the GrandModel system.

Key Components:
- Distributed locking with Redis/etcd
- Atomic operations and lock-free data structures
- Deadlock detection and prevention
- Priority-based locking with timeout mechanisms
- Comprehensive testing framework

Author: Agent Beta - Race Condition Elimination Specialist
Classification: CRITICAL SYSTEM COMPONENT
"""

from .lock_manager import (
    LockManager,
    LockType,
    LockPriority,
    LockResult,
    DeadlockDetector,
    get_global_lock_manager
)

from .atomic_operations import (
    AtomicCounter,
    AtomicReference,
    AtomicBoolean,
    AtomicDict,
    AtomicList,
    CompareAndSwap,
    LockFreeQueue,
    LockFreeStack
)

from .distributed_locks import (
    DistributedLockManager,
    RedisDistributedLock,
    EtcdDistributedLock,
    ConsensusLock,
    LeaderElection
)

from .sync_primitives import (
    ReadWriteLock,
    Semaphore,
    Barrier,
    CountDownLatch,
    CyclicBarrier,
    ReentrantLock,
    ConditionVariable
)

from .concurrent_collections import (
    ConcurrentHashMap,
    ConcurrentLinkedQueue,
    ConcurrentSkipList,
    LockFreeHashMap,
    ThreadSafeCache,
    BoundedConcurrentQueue
)

from .deadlock_prevention import (
    DeadlockPreventionManager,
    ResourceOrderingManager,
    TimeoutBasedDeadlockPrevention,
    DeadlockDetectionAlgorithm,
    WaitForGraph
)

from .performance_monitoring import (
    ConcurrencyMetrics,
    LockContentionMonitor,
    ThroughputAnalyzer,
    PerformanceBenchmarker
)

__all__ = [
    # Core Lock Management
    'LockManager',
    'LockType',
    'LockPriority', 
    'LockResult',
    'DeadlockDetector',
    'get_global_lock_manager',
    
    # Atomic Operations
    'AtomicCounter',
    'AtomicReference',
    'AtomicBoolean',
    'AtomicDict',
    'AtomicList',
    'CompareAndSwap',
    'LockFreeQueue',
    'LockFreeStack',
    
    # Distributed Locking
    'DistributedLockManager',
    'RedisDistributedLock',
    'EtcdDistributedLock',
    'ConsensusLock',
    'LeaderElection',
    
    # Synchronization Primitives
    'ReadWriteLock',
    'Semaphore',
    'Barrier',
    'CountDownLatch',
    'CyclicBarrier',
    'ReentrantLock',
    'ConditionVariable',
    
    # Concurrent Collections
    'ConcurrentHashMap',
    'ConcurrentLinkedQueue',
    'ConcurrentSkipList',
    'LockFreeHashMap',
    'ThreadSafeCache',
    'BoundedConcurrentQueue',
    
    # Deadlock Prevention
    'DeadlockPreventionManager',
    'ResourceOrderingManager',
    'TimeoutBasedDeadlockPrevention',
    'DeadlockDetectionAlgorithm',
    'WaitForGraph',
    
    # Performance Monitoring
    'ConcurrencyMetrics',
    'LockContentionMonitor',
    'ThroughputAnalyzer',
    'PerformanceBenchmarker'
]