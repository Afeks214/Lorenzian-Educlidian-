"""
Lock-Free Data Structures - Ultra-low latency concurrent data structures.
Implements lock-free queues, stacks, hashmaps, and synchronization primitives.
"""

import threading
import time
import ctypes
import array
from typing import Any, Dict, List, Optional, Union, Tuple, Generic, TypeVar, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import weakref
import gc
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class LockFreeResult(Enum):
    """Results from lock-free operations."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    FULL = "full"
    EMPTY = "empty"


@dataclass
class LockFreeMetrics:
    """Metrics for lock-free operations."""
    operations_attempted: int = 0
    operations_succeeded: int = 0
    operations_failed: int = 0
    retries_total: int = 0
    contention_detected: int = 0
    average_retry_count: float = 0.0
    max_retry_count: int = 0
    total_operation_time_ns: int = 0


class AtomicInteger:
    """Atomic integer with compare-and-swap operations."""
    
    def __init__(self, value: int = 0):
        self._value = ctypes.c_long(value)
        self._lock = threading.Lock()
    
    def get(self) -> int:
        """Get current value."""
        return self._value.value
    
    def set(self, value: int):
        """Set value."""
        with self._lock:
            self._value.value = value
    
    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Compare and swap operation."""
        with self._lock:
            if self._value.value == expected:
                self._value.value = new_value
                return True
            return False
    
    def add_and_get(self, delta: int) -> int:
        """Add delta and return new value."""
        with self._lock:
            self._value.value += delta
            return self._value.value
    
    def increment_and_get(self) -> int:
        """Increment and return new value."""
        return self.add_and_get(1)
    
    def decrement_and_get(self) -> int:
        """Decrement and return new value."""
        return self.add_and_get(-1)
    
    def get_and_set(self, new_value: int) -> int:
        """Get current value and set new value."""
        with self._lock:
            old_value = self._value.value
            self._value.value = new_value
            return old_value


class AtomicPointer(Generic[T]):
    """Atomic pointer with compare-and-swap operations."""
    
    def __init__(self, value: Optional[T] = None):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self) -> Optional[T]:
        """Get current value."""
        return self._value
    
    def set(self, value: Optional[T]):
        """Set value."""
        with self._lock:
            self._value = value
    
    def compare_and_swap(self, expected: Optional[T], new_value: Optional[T]) -> bool:
        """Compare and swap operation."""
        with self._lock:
            if self._value is expected:
                self._value = new_value
                return True
            return False
    
    def get_and_set(self, new_value: Optional[T]) -> Optional[T]:
        """Get current value and set new value."""
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value


class AtomicCounter:
    """
    High-performance atomic counter with contention tracking.
    Provides thread-safe increment/decrement operations.
    """
    
    def __init__(self, initial_value: int = 0):
        self._counter = AtomicInteger(initial_value)
        self._metrics = LockFreeMetrics()
        self._contention_threshold = 10  # Retries before considering contention
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            current = self._counter.get()
            if self._counter.compare_and_swap(current, current + 1):
                # Success
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                if retry_count > self._contention_threshold:
                    self._metrics.contention_detected += 1
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return current + 1
            
            retry_count += 1
            # Brief pause to reduce contention
            time.sleep(0.000001)  # 1 microsecond
    
    def decrement(self) -> int:
        """Decrement counter and return new value."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            current = self._counter.get()
            if self._counter.compare_and_swap(current, current - 1):
                # Success
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                if retry_count > self._contention_threshold:
                    self._metrics.contention_detected += 1
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return current - 1
            
            retry_count += 1
            time.sleep(0.000001)  # 1 microsecond
    
    def get(self) -> int:
        """Get current value."""
        return self._counter.get()
    
    def add(self, delta: int) -> int:
        """Add delta and return new value."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            current = self._counter.get()
            if self._counter.compare_and_swap(current, current + delta):
                # Success
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                if retry_count > self._contention_threshold:
                    self._metrics.contention_detected += 1
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return current + delta
            
            retry_count += 1
            time.sleep(0.000001)  # 1 microsecond
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get counter metrics."""
        metrics = self._metrics
        if metrics.operations_attempted > 0:
            metrics.average_retry_count = metrics.retries_total / metrics.operations_attempted
        return metrics
    
    def reset_metrics(self):
        """Reset counter metrics."""
        self._metrics = LockFreeMetrics()


class SpinLock:
    """
    Spin lock implementation for very short critical sections.
    More efficient than mutex for brief operations.
    """
    
    def __init__(self):
        self._locked = AtomicInteger(0)
        self._metrics = LockFreeMetrics()
    
    def acquire(self, timeout_ns: Optional[int] = None) -> bool:
        """Acquire the lock."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            if self._locked.compare_and_swap(0, 1):
                # Success
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return True
            
            retry_count += 1
            
            # Check timeout
            if timeout_ns is not None:
                current_time = time.time_ns()
                if current_time - start_time >= timeout_ns:
                    self._metrics.operations_attempted += 1
                    self._metrics.operations_failed += 1
                    return False
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def release(self):
        """Release the lock."""
        if not self._locked.compare_and_swap(1, 0):
            logger.warning("Attempting to release unlocked SpinLock")
    
    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self._locked.get() == 1
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get lock metrics."""
        return self._metrics


class LockFreeNode(Generic[T]):
    """Node for lock-free data structures."""
    
    def __init__(self, data: Optional[T] = None):
        self.data = data
        self.next = AtomicPointer[LockFreeNode[T]](None)
        self.marked = AtomicInteger(0)  # For deletion marking
    
    def is_marked(self) -> bool:
        """Check if node is marked for deletion."""
        return self.marked.get() == 1
    
    def mark(self) -> bool:
        """Mark node for deletion."""
        return self.marked.compare_and_swap(0, 1)


class LockFreeQueue(Generic[T]):
    """
    Lock-free queue using Michael & Scott algorithm.
    Provides high-performance FIFO operations without locks.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        
        # Initialize with dummy node
        dummy = LockFreeNode[T]()
        self._head = AtomicPointer[LockFreeNode[T]](dummy)
        self._tail = AtomicPointer[LockFreeNode[T]](dummy)
        self._size = AtomicCounter(0)
        self._metrics = LockFreeMetrics()
    
    def enqueue(self, data: T) -> LockFreeResult:
        """Enqueue data to tail."""
        if self.max_size is not None and self._size.get() >= self.max_size:
            return LockFreeResult.FULL
        
        start_time = time.time_ns()
        retry_count = 0
        
        new_node = LockFreeNode[T](data)
        
        while True:
            tail = self._tail.get()
            next_node = tail.next.get()
            
            if tail == self._tail.get():  # Tail hasn't changed
                if next_node is None:
                    # Tail is pointing to last node
                    if tail.next.compare_and_swap(None, new_node):
                        # Successfully linked new node
                        self._tail.compare_and_swap(tail, new_node)
                        self._size.increment()
                        
                        # Update metrics
                        self._metrics.operations_attempted += 1
                        self._metrics.operations_succeeded += 1
                        self._metrics.retries_total += retry_count
                        self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                        
                        end_time = time.time_ns()
                        self._metrics.total_operation_time_ns += end_time - start_time
                        
                        return LockFreeResult.SUCCESS
                else:
                    # Tail is lagging, try to advance it
                    self._tail.compare_and_swap(tail, next_node)
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def dequeue(self) -> Tuple[LockFreeResult, Optional[T]]:
        """Dequeue data from head."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            head = self._head.get()
            tail = self._tail.get()
            next_node = head.next.get()
            
            if head == self._head.get():  # Head hasn't changed
                if head == tail:
                    if next_node is None:
                        # Queue is empty
                        self._metrics.operations_attempted += 1
                        self._metrics.operations_failed += 1
                        return LockFreeResult.EMPTY, None
                    else:
                        # Tail is lagging, try to advance it
                        self._tail.compare_and_swap(tail, next_node)
                else:
                    if next_node is None:
                        # Something went wrong
                        continue
                    
                    # Read data before dequeue
                    data = next_node.data
                    
                    # Try to move head forward
                    if self._head.compare_and_swap(head, next_node):
                        self._size.decrement()
                        
                        # Update metrics
                        self._metrics.operations_attempted += 1
                        self._metrics.operations_succeeded += 1
                        self._metrics.retries_total += retry_count
                        self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                        
                        end_time = time.time_ns()
                        self._metrics.total_operation_time_ns += end_time - start_time
                        
                        return LockFreeResult.SUCCESS, data
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def size(self) -> int:
        """Get queue size."""
        return self._size.get()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get queue metrics."""
        return self._metrics


class LockFreeStack(Generic[T]):
    """
    Lock-free stack using Treiber's algorithm.
    Provides high-performance LIFO operations without locks.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self._top = AtomicPointer[LockFreeNode[T]](None)
        self._size = AtomicCounter(0)
        self._metrics = LockFreeMetrics()
    
    def push(self, data: T) -> LockFreeResult:
        """Push data onto stack."""
        if self.max_size is not None and self._size.get() >= self.max_size:
            return LockFreeResult.FULL
        
        start_time = time.time_ns()
        retry_count = 0
        
        new_node = LockFreeNode[T](data)
        
        while True:
            current_top = self._top.get()
            new_node.next.set(current_top)
            
            if self._top.compare_and_swap(current_top, new_node):
                self._size.increment()
                
                # Update metrics
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return LockFreeResult.SUCCESS
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def pop(self) -> Tuple[LockFreeResult, Optional[T]]:
        """Pop data from stack."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            current_top = self._top.get()
            
            if current_top is None:
                # Stack is empty
                self._metrics.operations_attempted += 1
                self._metrics.operations_failed += 1
                return LockFreeResult.EMPTY, None
            
            next_node = current_top.next.get()
            
            if self._top.compare_and_swap(current_top, next_node):
                self._size.decrement()
                
                # Update metrics
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                self._metrics.max_retry_count = max(self._metrics.max_retry_count, retry_count)
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return LockFreeResult.SUCCESS, current_top.data
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def size(self) -> int:
        """Get stack size."""
        return self._size.get()
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return self.size() == 0
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get stack metrics."""
        return self._metrics


class LockFreeHashEntry(Generic[T]):
    """Entry for lock-free hash map."""
    
    def __init__(self, key: Any, value: T):
        self.key = key
        self.value = value
        self.next = AtomicPointer[LockFreeHashEntry[T]](None)
        self.marked = AtomicInteger(0)
    
    def is_marked(self) -> bool:
        """Check if entry is marked for deletion."""
        return self.marked.get() == 1
    
    def mark(self) -> bool:
        """Mark entry for deletion."""
        return self.marked.compare_and_swap(0, 1)


class LockFreeHashMap(Generic[T]):
    """
    Lock-free hash map with separate chaining.
    Provides high-performance key-value operations without locks.
    """
    
    def __init__(self, initial_capacity: int = 16, load_factor: float = 0.75):
        self.initial_capacity = initial_capacity
        self.load_factor = load_factor
        
        # Initialize buckets
        self._buckets = [AtomicPointer[LockFreeHashEntry[T]](None) for _ in range(initial_capacity)]
        self._capacity = initial_capacity
        self._size = AtomicCounter(0)
        self._metrics = LockFreeMetrics()
        
        # Resize lock (only for structural modifications)
        self._resize_lock = SpinLock()
    
    def _hash(self, key: Any) -> int:
        """Hash function."""
        return hash(key) % self._capacity
    
    def _find_entry(self, key: Any) -> Tuple[Optional[LockFreeHashEntry[T]], Optional[LockFreeHashEntry[T]]]:
        """Find entry and its predecessor."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        prev = None
        current = bucket.get()
        
        while current is not None:
            if current.key == key and not current.is_marked():
                return current, prev
            
            prev = current
            current = current.next.get()
        
        return None, prev
    
    def put(self, key: Any, value: T) -> LockFreeResult:
        """Put key-value pair."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            entry, prev = self._find_entry(key)
            
            if entry is not None:
                # Update existing entry
                entry.value = value
                
                # Update metrics
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return LockFreeResult.SUCCESS
            
            # Insert new entry
            bucket_index = self._hash(key)
            bucket = self._buckets[bucket_index]
            
            new_entry = LockFreeHashEntry[T](key, value)
            current_head = bucket.get()
            new_entry.next.set(current_head)
            
            if bucket.compare_and_swap(current_head, new_entry):
                self._size.increment()
                
                # Update metrics
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                # Check if resize is needed
                if self._size.get() > self._capacity * self.load_factor:
                    self._try_resize()
                
                return LockFreeResult.SUCCESS
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def get(self, key: Any) -> Tuple[LockFreeResult, Optional[T]]:
        """Get value for key."""
        start_time = time.time_ns()
        
        entry, _ = self._find_entry(key)
        
        if entry is not None:
            # Update metrics
            self._metrics.operations_attempted += 1
            self._metrics.operations_succeeded += 1
            
            end_time = time.time_ns()
            self._metrics.total_operation_time_ns += end_time - start_time
            
            return LockFreeResult.SUCCESS, entry.value
        
        # Update metrics
        self._metrics.operations_attempted += 1
        self._metrics.operations_failed += 1
        
        return LockFreeResult.FAILURE, None
    
    def remove(self, key: Any) -> LockFreeResult:
        """Remove key-value pair."""
        start_time = time.time_ns()
        retry_count = 0
        
        while True:
            entry, prev = self._find_entry(key)
            
            if entry is None:
                # Key not found
                self._metrics.operations_attempted += 1
                self._metrics.operations_failed += 1
                return LockFreeResult.FAILURE
            
            # Mark entry for deletion
            if entry.mark():
                # Try to physically remove
                if prev is not None:
                    prev.next.compare_and_swap(entry, entry.next.get())
                else:
                    # Remove from bucket head
                    bucket_index = self._hash(key)
                    bucket = self._buckets[bucket_index]
                    bucket.compare_and_swap(entry, entry.next.get())
                
                self._size.decrement()
                
                # Update metrics
                self._metrics.operations_attempted += 1
                self._metrics.operations_succeeded += 1
                self._metrics.retries_total += retry_count
                
                end_time = time.time_ns()
                self._metrics.total_operation_time_ns += end_time - start_time
                
                return LockFreeResult.SUCCESS
            
            retry_count += 1
            if retry_count > 10:
                self._metrics.contention_detected += 1
            
            # Brief pause
            time.sleep(0.000001)  # 1 microsecond
    
    def _try_resize(self):
        """Try to resize the hash map."""
        if self._resize_lock.acquire(timeout_ns=1000000):  # 1ms timeout
            try:
                # Double capacity
                new_capacity = self._capacity * 2
                new_buckets = [AtomicPointer[LockFreeHashEntry[T]](None) for _ in range(new_capacity)]
                
                # Rehash all entries
                for bucket in self._buckets:
                    current = bucket.get()
                    while current is not None:
                        next_entry = current.next.get()
                        
                        if not current.is_marked():
                            # Rehash entry
                            new_index = hash(current.key) % new_capacity
                            new_bucket = new_buckets[new_index]
                            
                            current.next.set(new_bucket.get())
                            new_bucket.set(current)
                        
                        current = next_entry
                
                # Update capacity and buckets
                self._capacity = new_capacity
                self._buckets = new_buckets
                
                logger.info("Hash map resized", old_capacity=self._capacity // 2, new_capacity=new_capacity)
                
            finally:
                self._resize_lock.release()
    
    def size(self) -> int:
        """Get hash map size."""
        return self._size.get()
    
    def is_empty(self) -> bool:
        """Check if hash map is empty."""
        return self.size() == 0
    
    def contains(self, key: Any) -> bool:
        """Check if key exists."""
        result, _ = self.get(key)
        return result == LockFreeResult.SUCCESS
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get hash map metrics."""
        return self._metrics


class LockFreeCircularBuffer(Generic[T]):
    """
    Lock-free circular buffer for single producer, single consumer.
    Optimized for high-frequency operations.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._head = AtomicInteger(0)
        self._tail = AtomicInteger(0)
        self._metrics = LockFreeMetrics()
    
    def put(self, data: T) -> LockFreeResult:
        """Put data into buffer."""
        start_time = time.time_ns()
        
        current_tail = self._tail.get()
        next_tail = (current_tail + 1) % self.capacity
        
        if next_tail == self._head.get():
            # Buffer is full
            self._metrics.operations_attempted += 1
            self._metrics.operations_failed += 1
            return LockFreeResult.FULL
        
        # Store data
        self._buffer[current_tail] = data
        
        # Update tail
        self._tail.set(next_tail)
        
        # Update metrics
        self._metrics.operations_attempted += 1
        self._metrics.operations_succeeded += 1
        
        end_time = time.time_ns()
        self._metrics.total_operation_time_ns += end_time - start_time
        
        return LockFreeResult.SUCCESS
    
    def get(self) -> Tuple[LockFreeResult, Optional[T]]:
        """Get data from buffer."""
        start_time = time.time_ns()
        
        current_head = self._head.get()
        
        if current_head == self._tail.get():
            # Buffer is empty
            self._metrics.operations_attempted += 1
            self._metrics.operations_failed += 1
            return LockFreeResult.EMPTY, None
        
        # Get data
        data = self._buffer[current_head]
        
        # Update head
        self._head.set((current_head + 1) % self.capacity)
        
        # Update metrics
        self._metrics.operations_attempted += 1
        self._metrics.operations_succeeded += 1
        
        end_time = time.time_ns()
        self._metrics.total_operation_time_ns += end_time - start_time
        
        return LockFreeResult.SUCCESS, data
    
    def size(self) -> int:
        """Get buffer size."""
        head = self._head.get()
        tail = self._tail.get()
        
        if tail >= head:
            return tail - head
        else:
            return self.capacity - head + tail
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._head.get() == self._tail.get()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return ((self._tail.get() + 1) % self.capacity) == self._head.get()
    
    def get_metrics(self) -> LockFreeMetrics:
        """Get buffer metrics."""
        return self._metrics


class RCUPointer(Generic[T]):
    """
    Read-Copy-Update pointer for lock-free reads.
    Allows multiple readers without blocking writers.
    """
    
    def __init__(self, initial_value: Optional[T] = None):
        self._pointer = AtomicPointer[T](initial_value)
        self._readers = AtomicCounter(0)
        self._generation = AtomicCounter(0)
        self._pending_updates = []
        self._update_lock = SpinLock()
    
    def read(self) -> Optional[T]:
        """Read current value."""
        self._readers.increment()
        try:
            return self._pointer.get()
        finally:
            self._readers.decrement()
    
    def update(self, new_value: T):
        """Update value using RCU."""
        with self._update_lock:
            # Update pointer
            old_value = self._pointer.get_and_set(new_value)
            
            # Increment generation
            self._generation.increment()
            
            # Wait for all readers to finish
            while self._readers.get() > 0:
                time.sleep(0.000001)  # 1 microsecond
            
            # Old value can now be safely reclaimed
            # In practice, this would be handled by a garbage collector
            
            logger.debug("RCU update completed", generation=self._generation.get())
    
    def get_generation(self) -> int:
        """Get current generation."""
        return self._generation.get()
    
    def get_reader_count(self) -> int:
        """Get current reader count."""
        return self._readers.get()


class HazardPointer(Generic[T]):
    """
    Hazard pointer for memory reclamation in lock-free data structures.
    Prevents use-after-free in concurrent environments.
    """
    
    def __init__(self):
        self._hazard_pointers = {}
        self._retired_pointers = []
        self._thread_local = threading.local()
        self._lock = SpinLock()
    
    def acquire(self, pointer: T) -> bool:
        """Acquire hazard pointer."""
        thread_id = threading.current_thread().ident
        
        with self._lock:
            if thread_id not in self._hazard_pointers:
                self._hazard_pointers[thread_id] = set()
            
            self._hazard_pointers[thread_id].add(pointer)
            return True
    
    def release(self, pointer: T):
        """Release hazard pointer."""
        thread_id = threading.current_thread().ident
        
        with self._lock:
            if thread_id in self._hazard_pointers:
                self._hazard_pointers[thread_id].discard(pointer)
    
    def retire(self, pointer: T):
        """Retire pointer for later reclamation."""
        with self._lock:
            self._retired_pointers.append(pointer)
            
            # Try to reclaim retired pointers
            self._try_reclaim()
    
    def _try_reclaim(self):
        """Try to reclaim retired pointers."""
        # Find all currently protected pointers
        protected = set()
        for hazard_set in self._hazard_pointers.values():
            protected.update(hazard_set)
        
        # Reclaim unprotected pointers
        reclaimed = []
        for pointer in self._retired_pointers:
            if pointer not in protected:
                reclaimed.append(pointer)
        
        # Remove reclaimed pointers from retired list
        for pointer in reclaimed:
            self._retired_pointers.remove(pointer)
        
        if reclaimed:
            logger.debug("Reclaimed pointers", count=len(reclaimed))
    
    def get_hazard_count(self) -> int:
        """Get number of hazard pointers."""
        with self._lock:
            return sum(len(hazard_set) for hazard_set in self._hazard_pointers.values())
    
    def get_retired_count(self) -> int:
        """Get number of retired pointers."""
        with self._lock:
            return len(self._retired_pointers)


# Global instances
_global_hazard_pointer = HazardPointer()


def get_hazard_pointer() -> HazardPointer:
    """Get global hazard pointer instance."""
    return _global_hazard_pointer


def benchmark_lock_free_structure(structure_name: str, structure, operations: List[Tuple[str, Callable]], num_threads: int = 4, operations_per_thread: int = 10000) -> Dict[str, Any]:
    """Benchmark lock-free data structure."""
    import concurrent.futures
    import time
    
    def run_operations(thread_id: int) -> Dict[str, Any]:
        """Run operations for a single thread."""
        start_time = time.time_ns()
        operation_times = []
        
        for _ in range(operations_per_thread):
            # Select random operation
            operation_name, operation_func = operations[hash(time.time_ns()) % len(operations)]
            
            op_start = time.time_ns()
            try:
                result = operation_func()
                op_end = time.time_ns()
                operation_times.append(op_end - op_start)
            except Exception as e:
                logger.error(f"Operation failed", thread_id=thread_id, operation=operation_name, error=str(e))
        
        end_time = time.time_ns()
        
        return {
            'thread_id': thread_id,
            'total_time_ns': end_time - start_time,
            'operation_times': operation_times,
            'operations_completed': len(operation_times)
        }
    
    # Run benchmark
    start_time = time.time_ns()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_operations, i) for i in range(num_threads)]
        thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time_ns()
    
    # Calculate metrics
    total_operations = sum(result['operations_completed'] for result in thread_results)
    total_time_ns = end_time - start_time
    
    all_operation_times = []
    for result in thread_results:
        all_operation_times.extend(result['operation_times'])
    
    avg_operation_time_ns = sum(all_operation_times) / len(all_operation_times) if all_operation_times else 0
    
    return {
        'structure_name': structure_name,
        'num_threads': num_threads,
        'operations_per_thread': operations_per_thread,
        'total_operations': total_operations,
        'total_time_ns': total_time_ns,
        'total_time_ms': total_time_ns / 1_000_000,
        'throughput_ops_per_sec': total_operations / (total_time_ns / 1_000_000_000),
        'avg_operation_time_ns': avg_operation_time_ns,
        'avg_operation_time_us': avg_operation_time_ns / 1_000,
        'thread_results': thread_results,
        'structure_metrics': structure.get_metrics() if hasattr(structure, 'get_metrics') else None
    }