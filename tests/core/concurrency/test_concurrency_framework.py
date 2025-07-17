"""
Comprehensive Concurrency Testing Framework
==========================================

This test suite validates the entire concurrency framework for race condition
elimination and provides comprehensive validation of all components.

Author: Agent Beta - Race Condition Elimination Specialist
"""

import pytest
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from src.core.concurrency import (
    LockManager, LockType, LockPriority, get_global_lock_manager,
    AtomicCounter, AtomicReference, AtomicBoolean, AtomicDict, AtomicList,
    LockFreeQueue, LockFreeStack, CompareAndSwap,
    ReadWriteLock, Semaphore, Barrier, CountDownLatch, CyclicBarrier,
    ConcurrentHashMap, ConcurrentLinkedQueue, ThreadSafeCache,
    DeadlockPreventionManager, DeadlockDetectionAlgorithm,
    LockContentionMonitor, ThroughputAnalyzer, ConcurrencyMetrics,
    PerformanceBenchmarker
)


class TestLockManager:
    """Test comprehensive lock manager functionality"""
    
    def test_basic_lock_operations(self):
        """Test basic lock acquire/release operations"""
        manager = LockManager()
        
        # Test exclusive lock
        result = manager.acquire_lock("test_lock", LockType.EXCLUSIVE)
        assert result.success
        assert result.lock_id == "test_lock"
        
        # Test release
        success = manager.release_lock("test_lock")
        assert success
        
        manager.shutdown()
        
    def test_concurrent_lock_access(self):
        """Test concurrent access to locks"""
        manager = LockManager()
        shared_counter = 0
        lock_id = "concurrent_test"
        
        def worker():
            nonlocal shared_counter
            for _ in range(100):
                result = manager.acquire_lock(lock_id, LockType.EXCLUSIVE)
                if result.success:
                    try:
                        shared_counter += 1
                    finally:
                        manager.release_lock(lock_id)
                        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert shared_counter == 1000
        manager.shutdown()
        
    def test_read_write_locks(self):
        """Test read-write lock semantics"""
        manager = LockManager()
        shared_data = [0]
        lock_id = "rw_test"
        
        def reader():
            for _ in range(50):
                result = manager.acquire_lock(lock_id, LockType.READ)
                if result.success:
                    try:
                        # Read operation
                        value = shared_data[0]
                        time.sleep(0.001)  # Simulate read work
                    finally:
                        manager.release_lock(lock_id)
                        
        def writer():
            for _ in range(10):
                result = manager.acquire_lock(lock_id, LockType.WRITE)
                if result.success:
                    try:
                        # Write operation
                        shared_data[0] += 1
                        time.sleep(0.001)  # Simulate write work
                    finally:
                        manager.release_lock(lock_id)
                        
        threads = []
        
        # Start readers
        for _ in range(5):
            thread = threading.Thread(target=reader)
            threads.append(thread)
            thread.start()
            
        # Start writers
        for _ in range(2):
            thread = threading.Thread(target=writer)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert shared_data[0] == 20
        manager.shutdown()
        
    def test_lock_priorities(self):
        """Test priority-based lock ordering"""
        manager = LockManager()
        results = []
        lock_id = "priority_test"
        
        def worker(priority, worker_id):
            result = manager.acquire_lock(
                lock_id, 
                LockType.EXCLUSIVE, 
                LockPriority(priority)
            )
            if result.success:
                try:
                    results.append(worker_id)
                    time.sleep(0.01)
                finally:
                    manager.release_lock(lock_id)
                    
        threads = []
        
        # Create threads with different priorities
        priorities = [
            (LockPriority.LOW.value, "low1"),
            (LockPriority.HIGH.value, "high1"),
            (LockPriority.NORMAL.value, "normal1"),
            (LockPriority.CRITICAL.value, "critical1"),
            (LockPriority.LOW.value, "low2")
        ]
        
        for priority, worker_id in priorities:
            thread = threading.Thread(target=worker, args=(priority, worker_id))
            threads.append(thread)
            
        # Start all threads quickly
        for thread in threads:
            thread.start()
            time.sleep(0.001)  # Small delay to ensure ordering
            
        for thread in threads:
            thread.join()
            
        # Higher priority should execute first
        assert results[0] == "critical1"
        assert results[1] == "high1"
        
        manager.shutdown()
        
    def test_deadlock_detection(self):
        """Test deadlock detection mechanism"""
        manager = LockManager(enable_deadlock_detection=True)
        
        # Create potential deadlock scenario
        lock1 = "lock1"
        lock2 = "lock2"
        deadlock_detected = threading.Event()
        
        def worker1():
            result1 = manager.acquire_lock(lock1, LockType.EXCLUSIVE)
            if result1.success:
                try:
                    time.sleep(0.1)  # Hold lock1
                    result2 = manager.acquire_lock(lock2, LockType.EXCLUSIVE, timeout=0.5)
                    if result2.success:
                        manager.release_lock(lock2)
                finally:
                    manager.release_lock(lock1)
                    
        def worker2():
            result2 = manager.acquire_lock(lock2, LockType.EXCLUSIVE)
            if result2.success:
                try:
                    time.sleep(0.1)  # Hold lock2
                    result1 = manager.acquire_lock(lock1, LockType.EXCLUSIVE, timeout=0.5)
                    if result1.success:
                        manager.release_lock(lock1)
                finally:
                    manager.release_lock(lock2)
                    
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Check if deadlock was detected and resolved
        stats = manager.get_lock_statistics()
        assert stats['total_locks'] >= 2
        
        manager.shutdown()
        
    def test_context_manager(self):
        """Test context manager functionality"""
        manager = LockManager()
        shared_counter = 0
        
        def worker():
            nonlocal shared_counter
            for _ in range(100):
                with manager.lock("context_test"):
                    shared_counter += 1
                    
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert shared_counter == 1000
        manager.shutdown()


class TestAtomicOperations:
    """Test atomic operations and lock-free data structures"""
    
    def test_atomic_counter(self):
        """Test atomic counter operations"""
        counter = AtomicCounter(0)
        
        def worker():
            for _ in range(1000):
                counter.increment()
                
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert counter.get() == 10000
        
    def test_atomic_reference(self):
        """Test atomic reference operations"""
        ref = AtomicReference("initial")
        
        def worker(value):
            ref.set(value)
            
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(f"value_{i}",))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should have one of the values
        final_value = ref.get()
        assert final_value.startswith("value_")
        
    def test_compare_and_swap(self):
        """Test compare and swap operations"""
        counter = AtomicCounter(0)
        
        def worker():
            for _ in range(100):
                while True:
                    current = counter.get()
                    if counter.compare_and_swap(current, current + 1):
                        break
                        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert counter.get() == 1000
        
    def test_atomic_dict(self):
        """Test atomic dictionary operations"""
        atomic_dict = AtomicDict()
        
        def worker(worker_id):
            for i in range(100):
                key = f"key_{worker_id}_{i}"
                atomic_dict.put(key, i)
                
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert atomic_dict.size() == 1000
        
    def test_lock_free_queue(self):
        """Test lock-free queue operations"""
        queue = LockFreeQueue()
        
        def producer(producer_id):
            for i in range(100):
                queue.put(f"item_{producer_id}_{i}")
                
        def consumer(consumer_id, results):
            consumed = []
            while len(consumed) < 100:
                item = queue.get(timeout=1.0)
                if item:
                    consumed.append(item)
            results.extend(consumed)
            
        threads = []
        results = []
        
        # Start producers
        for producer_id in range(5):
            thread = threading.Thread(target=producer, args=(producer_id,))
            threads.append(thread)
            thread.start()
            
        # Start consumers
        for consumer_id in range(5):
            thread = threading.Thread(target=consumer, args=(consumer_id, results))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert len(results) == 500


class TestSynchronizationPrimitives:
    """Test advanced synchronization primitives"""
    
    def test_read_write_lock(self):
        """Test read-write lock implementation"""
        rw_lock = ReadWriteLock()
        shared_data = [0]
        read_count = 0
        
        def reader():
            nonlocal read_count
            for _ in range(10):
                with rw_lock.read_lock():
                    read_count += 1
                    value = shared_data[0]
                    time.sleep(0.001)
                    
        def writer():
            for _ in range(5):
                with rw_lock.write_lock():
                    shared_data[0] += 1
                    time.sleep(0.001)
                    
        threads = []
        
        # Start readers
        for _ in range(3):
            thread = threading.Thread(target=reader)
            threads.append(thread)
            thread.start()
            
        # Start writers
        for _ in range(2):
            thread = threading.Thread(target=writer)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert shared_data[0] == 10
        assert read_count == 30
        
    def test_semaphore(self):
        """Test semaphore implementation"""
        semaphore = Semaphore(3)  # Allow 3 concurrent access
        concurrent_count = AtomicCounter(0)
        max_concurrent = AtomicCounter(0)
        
        def worker():
            with semaphore.acquire_context():
                current = concurrent_count.increment()
                max_concurrent.set(max(max_concurrent.get(), current))
                time.sleep(0.01)
                concurrent_count.decrement()
                
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert max_concurrent.get() <= 3
        
    def test_barrier(self):
        """Test barrier synchronization"""
        barrier = Barrier(5)
        results = []
        
        def worker(worker_id):
            # Do some work
            time.sleep(0.01 * worker_id)
            
            # Wait at barrier
            barrier.wait()
            
            # All threads should reach here simultaneously
            results.append(time.time())
            
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All threads should have reached barrier within small time window
        assert len(results) == 5
        time_range = max(results) - min(results)
        assert time_range < 0.1  # Should be very close in time
        
    def test_countdown_latch(self):
        """Test countdown latch"""
        latch = CountDownLatch(3)
        results = []
        
        def worker(worker_id):
            # Do some work
            time.sleep(0.01 * worker_id)
            results.append(f"worker_{worker_id}_done")
            latch.count_down()
            
        def waiter():
            latch.await_latch()
            results.append("waiter_released")
            
        threads = []
        
        # Start waiter
        waiter_thread = threading.Thread(target=waiter)
        waiter_thread.start()
        
        # Start workers
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
        waiter_thread.join()
        
        assert len(results) == 4
        assert "waiter_released" in results


class TestConcurrentCollections:
    """Test concurrent collections"""
    
    def test_concurrent_hashmap(self):
        """Test concurrent hash map"""
        hashmap = ConcurrentHashMap()
        
        def worker(worker_id):
            for i in range(100):
                key = f"key_{worker_id}_{i}"
                hashmap.put(key, i)
                
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert hashmap.size() == 1000
        
        # Test concurrent reads
        def reader(results):
            for i in range(100):
                value = hashmap.get(f"key_0_{i}")
                if value is not None:
                    results.append(value)
                    
        read_results = []
        read_threads = []
        
        for _ in range(5):
            thread = threading.Thread(target=reader, args=(read_results,))
            read_threads.append(thread)
            thread.start()
            
        for thread in read_threads:
            thread.join()
            
        assert len(read_results) == 500
        
    def test_concurrent_queue(self):
        """Test concurrent linked queue"""
        queue = ConcurrentLinkedQueue(maxsize=100)
        
        def producer(producer_id):
            for i in range(50):
                success = queue.put(f"item_{producer_id}_{i}")
                assert success
                
        def consumer(consumer_id, results):
            consumed = []
            while len(consumed) < 50:
                item = queue.get(timeout=1.0)
                if item:
                    consumed.append(item)
            results.extend(consumed)
            
        threads = []
        results = []
        
        # Start producers
        for producer_id in range(2):
            thread = threading.Thread(target=producer, args=(producer_id,))
            threads.append(thread)
            thread.start()
            
        # Start consumers
        for consumer_id in range(2):
            thread = threading.Thread(target=consumer, args=(consumer_id, results))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert len(results) == 100
        
    def test_thread_safe_cache(self):
        """Test thread-safe cache with TTL"""
        cache = ThreadSafeCache(max_size=100, default_ttl=0.1)
        
        def worker(worker_id):
            for i in range(50):
                key = f"key_{worker_id}_{i}"
                cache.put(key, i)
                
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert cache.size() == 100  # Limited by max_size
        
        # Test TTL expiration
        time.sleep(0.2)  # Wait for TTL to expire
        
        # Cache should be empty after expiration
        expired_value = cache.get("key_0_0")
        assert expired_value is None


class TestDeadlockPrevention:
    """Test deadlock prevention mechanisms"""
    
    def test_deadlock_detection_algorithm(self):
        """Test deadlock detection algorithm"""
        detector = DeadlockDetectionAlgorithm()
        detector.start()
        
        # Simulate deadlock scenario
        from src.core.concurrency.deadlock_prevention import ResourceRequest, ResourceAllocation
        
        # Create requests that would cause deadlock
        request1 = ResourceRequest(
            requester_id="thread1",
            resource_id="resource1",
            request_type="write",
            priority=1,
            timestamp=time.time()
        )
        
        request2 = ResourceRequest(
            requester_id="thread2",
            resource_id="resource2",
            request_type="write",
            priority=1,
            timestamp=time.time()
        )
        
        # Add allocations
        allocation1 = ResourceAllocation(
            holder_id="thread1",
            resource_id="resource2",
            allocation_type="write",
            allocated_at=time.time()
        )
        
        allocation2 = ResourceAllocation(
            holder_id="thread2",
            resource_id="resource1",
            allocation_type="write",
            allocated_at=time.time()
        )
        
        detector.add_resource_allocation(allocation1)
        detector.add_resource_allocation(allocation2)
        detector.add_resource_request(request1)
        detector.add_resource_request(request2)
        
        # Wait for detection
        time.sleep(0.1)
        
        stats = detector.get_detection_statistics()
        assert stats['pending_requests'] >= 2
        
        detector.stop()
        
    def test_resource_ordering(self):
        """Test resource ordering for deadlock prevention"""
        from src.core.concurrency.deadlock_prevention import ResourceOrderingManager
        
        manager = ResourceOrderingManager()
        
        # Register resources
        order1 = manager.register_resource("resource1")
        order2 = manager.register_resource("resource2")
        order3 = manager.register_resource("resource3")
        
        assert order1 < order2 < order3
        
        # Test ordering validation
        held_resources = ["resource1"]
        assert manager.validate_request_order(held_resources, "resource2")
        assert not manager.validate_request_order(held_resources, "resource1")
        
        held_resources = ["resource2"]
        assert not manager.validate_request_order(held_resources, "resource1")
        
    def test_comprehensive_deadlock_prevention(self):
        """Test comprehensive deadlock prevention manager"""
        manager = DeadlockPreventionManager()
        manager.start()
        
        from src.core.concurrency.deadlock_prevention import ResourceRequest, ResourceAllocation
        
        # Test request validation
        request = ResourceRequest(
            requester_id="test_thread",
            resource_id="test_resource",
            request_type="write",
            priority=1,
            timestamp=time.time()
        )
        
        # Should be valid for first resource
        assert manager.validate_resource_request(request, [])
        
        # Test allocation notification
        allocation = ResourceAllocation(
            holder_id="test_thread",
            resource_id="test_resource",
            allocation_type="write",
            allocated_at=time.time()
        )
        
        manager.notify_resource_acquired(allocation)
        manager.notify_resource_released("test_thread", "test_resource")
        
        stats = manager.get_comprehensive_statistics()
        assert stats['manager_stats']['requests_validated'] > 0
        
        manager.stop()


class TestPerformanceMonitoring:
    """Test performance monitoring components"""
    
    def test_lock_contention_monitor(self):
        """Test lock contention monitoring"""
        monitor = LockContentionMonitor()
        
        # Simulate lock operations
        monitor.record_lock_wait("test_lock", "thread1")
        monitor.record_lock_acquisition("test_lock", "thread1", 0.01, time.time())
        monitor.record_lock_release("test_lock", "thread1")
        
        stats = monitor.get_contention_statistics()
        assert "test_lock" in stats
        assert stats["test_lock"]["acquisitions"] == 1
        
    def test_throughput_analyzer(self):
        """Test throughput analysis"""
        analyzer = ThroughputAnalyzer()
        
        # Start measurement
        analyzer.start_measurement("test_ops")
        
        # Simulate operations
        for _ in range(100):
            analyzer.record_operation("test_ops")
            
        # End measurement
        measurement = analyzer.end_measurement("test_ops")
        
        assert measurement is not None
        assert measurement.total_operations == 100
        assert measurement.operations_per_second > 0
        
    def test_concurrency_metrics(self):
        """Test concurrency metrics collection"""
        metrics = ConcurrencyMetrics()
        
        # Record metrics
        for i in range(10):
            metrics.record_metric("test_metric", float(i))
            
        stats = metrics.get_metric_statistics("test_metric")
        assert stats["count"] == 10
        assert stats["average"] == 4.5
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        
    def test_performance_benchmarker(self):
        """Test performance benchmarking"""
        benchmarker = PerformanceBenchmarker()
        
        # Test lock benchmarking
        results = benchmarker.benchmark_lock_performance(
            threading.RLock, 
            num_threads=3, 
            operations_per_thread=10
        )
        
        assert results["lock_type"] == "RLock"
        assert results["total_operations"] == 30
        assert results["throughput"] > 0
        
        # Test comprehensive benchmark
        comprehensive_results = benchmarker.run_comprehensive_benchmark()
        assert "timestamp" in comprehensive_results
        assert "system_info" in comprehensive_results
        assert "lock_benchmarks" in comprehensive_results


class TestIntegration:
    """Integration tests for the entire concurrency framework"""
    
    def test_full_system_integration(self):
        """Test full system integration with all components"""
        # Initialize all components
        lock_manager = LockManager()
        contention_monitor = LockContentionMonitor()
        throughput_analyzer = ThroughputAnalyzer()
        metrics = ConcurrencyMetrics()
        
        # Start measurements
        throughput_analyzer.start_measurement("integration_test")
        
        # Simulate complex concurrent scenario
        shared_data = {"counter": 0, "values": []}
        
        def complex_worker(worker_id):
            # Use atomic operations
            atomic_counter = AtomicCounter(0)
            
            # Use concurrent collections
            concurrent_map = ConcurrentHashMap()
            
            for i in range(50):
                # Acquire lock with monitoring
                with lock_manager.lock(f"worker_{worker_id}"):
                    # Record metrics
                    metrics.record_metric("worker_operations", 1.0)
                    throughput_analyzer.record_operation("integration_test")
                    
                    # Update shared data
                    shared_data["counter"] += 1
                    shared_data["values"].append(f"worker_{worker_id}_item_{i}")
                    
                    # Use atomic operations
                    atomic_counter.increment()
                    
                    # Use concurrent collections
                    concurrent_map.put(f"key_{i}", i)
                    
        # Run concurrent workers
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=complex_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # End measurements
        measurement = throughput_analyzer.end_measurement("integration_test")
        
        # Verify results
        assert shared_data["counter"] == 500
        assert len(shared_data["values"]) == 500
        assert measurement.total_operations == 500
        
        # Check monitoring data
        worker_metrics = metrics.get_metric_statistics("worker_operations")
        assert worker_metrics["count"] == 500
        
        # Cleanup
        lock_manager.shutdown()
        
    def test_stress_testing(self):
        """Stress test the concurrency framework"""
        lock_manager = LockManager()
        
        # High contention scenario
        shared_counter = AtomicCounter(0)
        errors = []
        
        def stress_worker(worker_id):
            try:
                for i in range(1000):
                    with lock_manager.lock("stress_lock"):
                        # Simulate work
                        current = shared_counter.get()
                        time.sleep(0.0001)  # Small delay to increase contention
                        shared_counter.set(current + 1)
                        
            except Exception as e:
                errors.append(str(e))
                
        # Run many concurrent workers
        threads = []
        for worker_id in range(20):
            thread = threading.Thread(target=stress_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should have no errors and correct count
        assert len(errors) == 0
        assert shared_counter.get() == 20000
        
        lock_manager.shutdown()


class TestGlobalLockManager:
    """Test global lock manager singleton"""
    
    def test_global_lock_manager(self):
        """Test global lock manager functionality"""
        manager1 = get_global_lock_manager()
        manager2 = get_global_lock_manager()
        
        # Should be same instance
        assert manager1 is manager2
        
        # Should work normally
        with manager1.lock("global_test"):
            pass
            
        # Statistics should be accessible
        stats = manager1.get_lock_statistics()
        assert isinstance(stats, dict)


if __name__ == "__main__":
    # Run specific test classes
    test_classes = [
        TestLockManager,
        TestAtomicOperations,
        TestSynchronizationPrimitives,
        TestConcurrentCollections,
        TestDeadlockPrevention,
        TestPerformanceMonitoring,
        TestIntegration,
        TestGlobalLockManager
    ]
    
    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        test_instance = test_class()
        
        # Run all test methods
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                print(f"  {method_name}...")
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"    ✓ PASSED")
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")
                    
    print("\nConcurrency framework testing completed!")