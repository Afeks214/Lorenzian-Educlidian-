"""
Advanced Synchronization Primitives
===================================

This module provides advanced synchronization primitives for fine-grained
concurrency control beyond basic locks.

Features:
- Read-write locks with fairness
- Semaphores with priority
- Barriers and latches
- Condition variables
- Reentrant locks with debugging

Author: Agent Beta - Race Condition Elimination Specialist
"""

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import structlog

logger = structlog.get_logger(__name__)


class FairnessPolicy(Enum):
    """Fairness policies for locks"""
    FIFO = "fifo"
    PRIORITY = "priority"
    UNFAIR = "unfair"


@dataclass
class WaitingThread:
    """Represents a waiting thread"""
    thread_id: str
    wait_type: str
    priority: int
    timestamp: float
    condition: threading.Condition


class ReadWriteLock:
    """
    Advanced read-write lock with fairness and priority support
    """
    
    def __init__(self, fairness_policy: FairnessPolicy = FairnessPolicy.FIFO):
        self.fairness_policy = fairness_policy
        self._lock = threading.RLock()
        self._read_condition = threading.Condition(self._lock)
        self._write_condition = threading.Condition(self._lock)
        
        # State
        self._readers = 0
        self._writers = 0
        self._waiting_writers = 0
        self._waiting_readers = 0
        
        # Fairness tracking
        self._waiting_queue: deque[WaitingThread] = deque()
        self._reader_threads: Set[str] = set()
        self._writer_thread: Optional[str] = None
        
        # Statistics
        self._stats = {
            'reads_acquired': 0,
            'writes_acquired': 0,
            'reads_waiting': 0,
            'writes_waiting': 0,
            'total_wait_time': 0.0
        }
        
    def acquire_read(self, priority: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire read lock"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        with self._lock:
            # Check if already have read lock
            if thread_id in self._reader_threads:
                return True
                
            # Add to waiting queue for fairness
            if self.fairness_policy != FairnessPolicy.UNFAIR:
                waiting_thread = WaitingThread(
                    thread_id=thread_id,
                    wait_type="read",
                    priority=priority,
                    timestamp=start_time,
                    condition=self._read_condition
                )
                self._add_to_waiting_queue(waiting_thread)
                
            self._waiting_readers += 1
            self._stats['reads_waiting'] += 1
            
            try:
                # Wait for read access
                while not self._can_acquire_read(thread_id):
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            return False
                        if not self._read_condition.wait(timeout=remaining_time):
                            return False
                    else:
                        self._read_condition.wait()
                        
                # Acquire read lock
                self._readers += 1
                self._reader_threads.add(thread_id)
                self._waiting_readers -= 1
                self._stats['reads_acquired'] += 1
                self._stats['total_wait_time'] += time.time() - start_time
                
                logger.debug("Read lock acquired", thread_id=thread_id)
                return True
                
            finally:
                if self.fairness_policy != FairnessPolicy.UNFAIR:
                    self._remove_from_waiting_queue(thread_id)
                    
    def acquire_write(self, priority: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire write lock"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        with self._lock:
            # Check if already have write lock (reentrant)
            if self._writer_thread == thread_id:
                self._writers += 1
                return True
                
            # Add to waiting queue for fairness
            if self.fairness_policy != FairnessPolicy.UNFAIR:
                waiting_thread = WaitingThread(
                    thread_id=thread_id,
                    wait_type="write",
                    priority=priority,
                    timestamp=start_time,
                    condition=self._write_condition
                )
                self._add_to_waiting_queue(waiting_thread)
                
            self._waiting_writers += 1
            self._stats['writes_waiting'] += 1
            
            try:
                # Wait for write access
                while not self._can_acquire_write(thread_id):
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            return False
                        if not self._write_condition.wait(timeout=remaining_time):
                            return False
                    else:
                        self._write_condition.wait()
                        
                # Acquire write lock
                self._writers = 1
                self._writer_thread = thread_id
                self._waiting_writers -= 1
                self._stats['writes_acquired'] += 1
                self._stats['total_wait_time'] += time.time() - start_time
                
                logger.debug("Write lock acquired", thread_id=thread_id)
                return True
                
            finally:
                if self.fairness_policy != FairnessPolicy.UNFAIR:
                    self._remove_from_waiting_queue(thread_id)
                    
    def release_read(self):
        """Release read lock"""
        thread_id = str(threading.current_thread().ident)
        
        with self._lock:
            if thread_id not in self._reader_threads:
                logger.warning("Attempt to release unowned read lock", thread_id=thread_id)
                return
                
            self._readers -= 1
            self._reader_threads.remove(thread_id)
            
            logger.debug("Read lock released", thread_id=thread_id)
            
            # Notify waiting threads
            if self._readers == 0:
                self._write_condition.notify_all()
            else:
                self._read_condition.notify_all()
                
    def release_write(self):
        """Release write lock"""
        thread_id = str(threading.current_thread().ident)
        
        with self._lock:
            if self._writer_thread != thread_id:
                logger.warning("Attempt to release unowned write lock", thread_id=thread_id)
                return
                
            self._writers -= 1
            
            if self._writers == 0:
                self._writer_thread = None
                logger.debug("Write lock released", thread_id=thread_id)
                
                # Notify all waiting threads
                self._write_condition.notify_all()
                self._read_condition.notify_all()
                
    def _can_acquire_read(self, thread_id: str) -> bool:
        """Check if thread can acquire read lock"""
        # Can't read if there are active writers
        if self._writers > 0:
            return False
            
        # Fairness check
        if self.fairness_policy == FairnessPolicy.FIFO:
            # Check if any writers are waiting before this reader
            for waiting in self._waiting_queue:
                if waiting.wait_type == "write":
                    return False
                if waiting.thread_id == thread_id:
                    break
                    
        elif self.fairness_policy == FairnessPolicy.PRIORITY:
            # Check if any higher priority writers are waiting
            for waiting in self._waiting_queue:
                if (waiting.wait_type == "write" and 
                    waiting.priority > self._get_thread_priority(thread_id)):
                    return False
                    
        return True
        
    def _can_acquire_write(self, thread_id: str) -> bool:
        """Check if thread can acquire write lock"""
        # Can't write if there are active readers or writers
        if self._readers > 0 or self._writers > 0:
            return False
            
        # Fairness check
        if self.fairness_policy == FairnessPolicy.FIFO:
            # Check if this is the first writer in queue
            for waiting in self._waiting_queue:
                if waiting.wait_type == "write":
                    return waiting.thread_id == thread_id
                    
        elif self.fairness_policy == FairnessPolicy.PRIORITY:
            # Check if this is the highest priority writer
            thread_priority = self._get_thread_priority(thread_id)
            for waiting in self._waiting_queue:
                if (waiting.wait_type == "write" and 
                    waiting.priority > thread_priority):
                    return False
                    
        return True
        
    def _add_to_waiting_queue(self, waiting_thread: WaitingThread):
        """Add thread to waiting queue"""
        if self.fairness_policy == FairnessPolicy.FIFO:
            self._waiting_queue.append(waiting_thread)
        elif self.fairness_policy == FairnessPolicy.PRIORITY:
            # Insert in priority order
            inserted = False
            for i, existing in enumerate(self._waiting_queue):
                if waiting_thread.priority > existing.priority:
                    self._waiting_queue.insert(i, waiting_thread)
                    inserted = True
                    break
            if not inserted:
                self._waiting_queue.append(waiting_thread)
                
    def _remove_from_waiting_queue(self, thread_id: str):
        """Remove thread from waiting queue"""
        self._waiting_queue = deque(
            waiting for waiting in self._waiting_queue 
            if waiting.thread_id != thread_id
        )
        
    def _get_thread_priority(self, thread_id: str) -> int:
        """Get priority for thread"""
        for waiting in self._waiting_queue:
            if waiting.thread_id == thread_id:
                return waiting.priority
        return 1  # Default priority
        
    @contextmanager
    def read_lock(self, priority: int = 1, timeout: Optional[float] = None):
        """Context manager for read lock"""
        if self.acquire_read(priority, timeout):
            try:
                yield
            finally:
                self.release_read()
        else:
            raise RuntimeError("Failed to acquire read lock")
            
    @contextmanager
    def write_lock(self, priority: int = 1, timeout: Optional[float] = None):
        """Context manager for write lock"""
        if self.acquire_write(priority, timeout):
            try:
                yield
            finally:
                self.release_write()
        else:
            raise RuntimeError("Failed to acquire write lock")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get lock statistics"""
        with self._lock:
            return {
                'active_readers': self._readers,
                'active_writers': self._writers,
                'waiting_readers': self._waiting_readers,
                'waiting_writers': self._waiting_writers,
                'stats': dict(self._stats),
                'fairness_policy': self.fairness_policy.value
            }


class Semaphore:
    """
    Advanced semaphore with priority support
    """
    
    def __init__(self, initial_permits: int, fairness_policy: FairnessPolicy = FairnessPolicy.FIFO):
        self.initial_permits = initial_permits
        self.fairness_policy = fairness_policy
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._permits = initial_permits
        self._waiting_queue: deque[WaitingThread] = deque()
        
    def acquire(self, permits: int = 1, priority: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore permits"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        with self._lock:
            # Add to waiting queue for fairness
            if self.fairness_policy != FairnessPolicy.UNFAIR:
                waiting_thread = WaitingThread(
                    thread_id=thread_id,
                    wait_type="acquire",
                    priority=priority,
                    timestamp=start_time,
                    condition=self._condition
                )
                self._add_to_waiting_queue(waiting_thread)
                
            try:
                # Wait for permits
                while not self._can_acquire(thread_id, permits):
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            return False
                        if not self._condition.wait(timeout=remaining_time):
                            return False
                    else:
                        self._condition.wait()
                        
                # Acquire permits
                self._permits -= permits
                logger.debug("Semaphore acquired", thread_id=thread_id, permits=permits)
                return True
                
            finally:
                if self.fairness_policy != FairnessPolicy.UNFAIR:
                    self._remove_from_waiting_queue(thread_id)
                    
    def release(self, permits: int = 1):
        """Release semaphore permits"""
        thread_id = str(threading.current_thread().ident)
        
        with self._lock:
            self._permits += permits
            logger.debug("Semaphore released", thread_id=thread_id, permits=permits)
            self._condition.notify_all()
            
    def _can_acquire(self, thread_id: str, permits: int) -> bool:
        """Check if thread can acquire permits"""
        if self._permits < permits:
            return False
            
        # Fairness check
        if self.fairness_policy == FairnessPolicy.FIFO:
            # Check if this is the first thread in queue
            if self._waiting_queue:
                return self._waiting_queue[0].thread_id == thread_id
                
        elif self.fairness_policy == FairnessPolicy.PRIORITY:
            # Check if this is the highest priority thread
            thread_priority = self._get_thread_priority(thread_id)
            for waiting in self._waiting_queue:
                if waiting.priority > thread_priority:
                    return False
                    
        return True
        
    def _add_to_waiting_queue(self, waiting_thread: WaitingThread):
        """Add thread to waiting queue"""
        if self.fairness_policy == FairnessPolicy.FIFO:
            self._waiting_queue.append(waiting_thread)
        elif self.fairness_policy == FairnessPolicy.PRIORITY:
            # Insert in priority order
            inserted = False
            for i, existing in enumerate(self._waiting_queue):
                if waiting_thread.priority > existing.priority:
                    self._waiting_queue.insert(i, waiting_thread)
                    inserted = True
                    break
            if not inserted:
                self._waiting_queue.append(waiting_thread)
                
    def _remove_from_waiting_queue(self, thread_id: str):
        """Remove thread from waiting queue"""
        self._waiting_queue = deque(
            waiting for waiting in self._waiting_queue 
            if waiting.thread_id != thread_id
        )
        
    def _get_thread_priority(self, thread_id: str) -> int:
        """Get priority for thread"""
        for waiting in self._waiting_queue:
            if waiting.thread_id == thread_id:
                return waiting.priority
        return 1  # Default priority
        
    @contextmanager
    def acquire_context(self, permits: int = 1, priority: int = 1, timeout: Optional[float] = None):
        """Context manager for semaphore"""
        if self.acquire(permits, priority, timeout):
            try:
                yield
            finally:
                self.release(permits)
        else:
            raise RuntimeError("Failed to acquire semaphore")
            
    def available_permits(self) -> int:
        """Get number of available permits"""
        with self._lock:
            return self._permits
            
    def waiting_threads(self) -> int:
        """Get number of waiting threads"""
        with self._lock:
            return len(self._waiting_queue)


class Barrier:
    """
    Synchronization barrier for coordinating multiple threads
    """
    
    def __init__(self, num_threads: int, action: Optional[Callable] = None):
        self.num_threads = num_threads
        self.action = action
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiting = 0
        self._broken = False
        self._generation = 0
        
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait at the barrier"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        with self._lock:
            if self._broken:
                raise RuntimeError("Barrier is broken")
                
            generation = self._generation
            self._waiting += 1
            
            try:
                # Check if we're the last thread
                if self._waiting == self.num_threads:
                    # All threads have arrived
                    self._waiting = 0
                    self._generation += 1
                    
                    # Execute barrier action if provided
                    if self.action:
                        try:
                            self.action()
                        except Exception as e:
                            logger.error("Error in barrier action", error=str(e))
                            self._broken = True
                            self._condition.notify_all()
                            raise
                            
                    # Wake up all waiting threads
                    self._condition.notify_all()
                    logger.debug("Barrier completed", thread_id=thread_id, generation=generation)
                    return True
                    
                # Wait for other threads
                while self._waiting < self.num_threads and generation == self._generation:
                    if self._broken:
                        raise RuntimeError("Barrier is broken")
                        
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self._broken = True
                            self._condition.notify_all()
                            return False
                        if not self._condition.wait(timeout=remaining_time):
                            self._broken = True
                            self._condition.notify_all()
                            return False
                    else:
                        self._condition.wait()
                        
                logger.debug("Barrier wait completed", thread_id=thread_id, generation=generation)
                return True
                
            except Exception:
                self._broken = True
                self._condition.notify_all()
                raise
                
    def reset(self):
        """Reset the barrier"""
        with self._lock:
            self._broken = False
            self._waiting = 0
            self._generation += 1
            self._condition.notify_all()
            
    def is_broken(self) -> bool:
        """Check if barrier is broken"""
        with self._lock:
            return self._broken
            
    def waiting_threads(self) -> int:
        """Get number of waiting threads"""
        with self._lock:
            return self._waiting


class CountDownLatch:
    """
    Synchronization latch that blocks until count reaches zero
    """
    
    def __init__(self, count: int):
        self.initial_count = count
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._count = count
        
    def await_latch(self, timeout: Optional[float] = None) -> bool:
        """Wait for the latch to open"""
        start_time = time.time()
        
        with self._lock:
            while self._count > 0:
                if timeout:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        return False
                    if not self._condition.wait(timeout=remaining_time):
                        return False
                else:
                    self._condition.wait()
                    
            return True
            
    def count_down(self):
        """Decrement the count"""
        with self._lock:
            if self._count > 0:
                self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()
                    
    def get_count(self) -> int:
        """Get current count"""
        with self._lock:
            return self._count


class CyclicBarrier:
    """
    Reusable barrier that resets after each cycle
    """
    
    def __init__(self, num_threads: int, action: Optional[Callable] = None):
        self.num_threads = num_threads
        self.action = action
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiting = 0
        self._broken = False
        self._generation = 0
        
    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait at the barrier and return arrival index"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        with self._lock:
            if self._broken:
                raise RuntimeError("Barrier is broken")
                
            generation = self._generation
            arrival_index = self._waiting
            self._waiting += 1
            
            try:
                # Check if we're the last thread
                if self._waiting == self.num_threads:
                    # All threads have arrived
                    self._waiting = 0
                    self._generation += 1
                    
                    # Execute barrier action if provided
                    if self.action:
                        try:
                            self.action()
                        except Exception as e:
                            logger.error("Error in cyclic barrier action", error=str(e))
                            self._broken = True
                            self._condition.notify_all()
                            raise
                            
                    # Wake up all waiting threads
                    self._condition.notify_all()
                    logger.debug("Cyclic barrier completed", thread_id=thread_id, generation=generation)
                    return arrival_index
                    
                # Wait for other threads
                while self._waiting < self.num_threads and generation == self._generation:
                    if self._broken:
                        raise RuntimeError("Barrier is broken")
                        
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self._broken = True
                            self._condition.notify_all()
                            raise RuntimeError("Barrier timeout")
                        if not self._condition.wait(timeout=remaining_time):
                            self._broken = True
                            self._condition.notify_all()
                            raise RuntimeError("Barrier timeout")
                    else:
                        self._condition.wait()
                        
                logger.debug("Cyclic barrier wait completed", thread_id=thread_id, generation=generation)
                return arrival_index
                
            except Exception:
                self._broken = True
                self._condition.notify_all()
                raise
                
    def reset(self):
        """Reset the barrier"""
        with self._lock:
            self._broken = False
            self._waiting = 0
            self._generation += 1
            self._condition.notify_all()
            
    def is_broken(self) -> bool:
        """Check if barrier is broken"""
        with self._lock:
            return self._broken
            
    def waiting_threads(self) -> int:
        """Get number of waiting threads"""
        with self._lock:
            return self._waiting


class ReentrantLock:
    """
    Reentrant lock with debugging and monitoring
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._lock = threading.RLock()
        self._holder: Optional[str] = None
        self._hold_count = 0
        self._acquisition_times: List[float] = []
        self._total_acquisitions = 0
        self._total_wait_time = 0.0
        
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the lock"""
        thread_id = str(threading.current_thread().ident)
        start_time = time.time()
        
        success = self._lock.acquire(timeout=timeout)
        
        if success:
            self._holder = thread_id
            self._hold_count += 1
            self._acquisition_times.append(start_time)
            self._total_acquisitions += 1
            self._total_wait_time += time.time() - start_time
            
            logger.debug("Reentrant lock acquired", 
                        name=self.name, 
                        thread_id=thread_id,
                        hold_count=self._hold_count)
                        
        return success
        
    def release(self):
        """Release the lock"""
        thread_id = str(threading.current_thread().ident)
        
        if self._holder != thread_id:
            logger.warning("Attempt to release lock by non-holder", 
                          name=self.name, 
                          thread_id=thread_id,
                          holder=self._holder)
            return
            
        self._hold_count -= 1
        if self._hold_count == 0:
            self._holder = None
            
        self._lock.release()
        
        logger.debug("Reentrant lock released", 
                    name=self.name, 
                    thread_id=thread_id,
                    hold_count=self._hold_count)
                    
    @contextmanager
    def lock(self, timeout: Optional[float] = None):
        """Context manager for lock"""
        if self.acquire(timeout):
            try:
                yield
            finally:
                self.release()
        else:
            raise RuntimeError(f"Failed to acquire lock {self.name}")
            
    def is_held_by_current_thread(self) -> bool:
        """Check if lock is held by current thread"""
        thread_id = str(threading.current_thread().ident)
        return self._holder == thread_id
        
    def get_hold_count(self) -> int:
        """Get number of holds by current thread"""
        thread_id = str(threading.current_thread().ident)
        if self._holder == thread_id:
            return self._hold_count
        return 0
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get lock statistics"""
        return {
            'name': self.name,
            'current_holder': self._holder,
            'hold_count': self._hold_count,
            'total_acquisitions': self._total_acquisitions,
            'average_wait_time': self._total_wait_time / max(1, self._total_acquisitions),
            'total_wait_time': self._total_wait_time
        }


class ConditionVariable:
    """
    Condition variable for thread coordination
    """
    
    def __init__(self, lock: Optional[threading.Lock] = None):
        self._lock = lock or threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiters = 0
        
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for condition"""
        thread_id = str(threading.current_thread().ident)
        self._waiters += 1
        
        try:
            result = self._condition.wait(timeout=timeout)
            logger.debug("Condition wait completed", 
                        thread_id=thread_id, 
                        result=result)
            return result
        finally:
            self._waiters -= 1
            
    def notify(self):
        """Notify one waiting thread"""
        self._condition.notify()
        logger.debug("Condition notified")
        
    def notify_all(self):
        """Notify all waiting threads"""
        self._condition.notify_all()
        logger.debug("Condition notify_all")
        
    def waiting_threads(self) -> int:
        """Get number of waiting threads"""
        return self._waiters
        
    @contextmanager
    def condition_lock(self):
        """Context manager for condition lock"""
        with self._lock:
            yield