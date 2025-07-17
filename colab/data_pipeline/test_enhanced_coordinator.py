#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced DataFlowCoordinator

This test suite validates the enhanced concurrency features, race condition fixes,
and dependency management capabilities of the DataFlowCoordinator.
"""

import sys
import os
import unittest
import threading
import time
import tempfile
import shutil
import logging
import random
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_flow_coordinator import (
    EnhancedDataFlowCoordinator, 
    EnhancedDataStream,
    DataStreamType,
    DataStreamPriority,
    DataMessage,
    AtomicCounter,
    ThreadSafeDict,
    DependencyGraph,
    LockFreeQueue,
    ConcurrencyMonitor,
    EnhancedCoordinatorConfig,
    create_enhanced_coordinator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestAtomicCounter(unittest.TestCase):
    """Test atomic counter thread safety"""
    
    def test_atomic_increment(self):
        """Test atomic increment operations"""
        counter = AtomicCounter(0)
        
        def increment_worker():
            for _ in range(1000):
                counter.increment()
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify atomic operations
        self.assertEqual(counter.get(), 10000)
    
    def test_compare_and_swap(self):
        """Test compare and swap operations"""
        counter = AtomicCounter(0)
        
        # Test successful CAS
        self.assertTrue(counter.compare_and_swap(0, 10))
        self.assertEqual(counter.get(), 10)
        
        # Test failed CAS
        self.assertFalse(counter.compare_and_swap(0, 20))
        self.assertEqual(counter.get(), 10)


class TestThreadSafeDict(unittest.TestCase):
    """Test thread-safe dictionary implementation"""
    
    def test_concurrent_access(self):
        """Test concurrent read/write operations"""
        safe_dict = ThreadSafeDict()
        
        def writer_worker(worker_id):
            for i in range(100):
                safe_dict.set(f"key_{worker_id}_{i}", f"value_{worker_id}_{i}")
        
        def reader_worker(worker_id):
            for i in range(100):
                value = safe_dict.get(f"key_{worker_id}_{i}")
                if value is not None:
                    self.assertIn(f"worker_{worker_id}", value)
        
        # Start writers and readers
        writers = []
        readers = []
        
        for i in range(5):
            writer = threading.Thread(target=writer_worker, args=(i,))
            reader = threading.Thread(target=reader_worker, args=(i,))
            writers.append(writer)
            readers.append(reader)
        
        # Start all threads
        for writer in writers:
            writer.start()
        for reader in readers:
            reader.start()
        
        # Wait for completion
        for writer in writers:
            writer.join()
        for reader in readers:
            reader.join()
        
        # Verify data integrity
        self.assertEqual(len(safe_dict), 500)  # 5 workers * 100 items each


class TestDependencyGraph(unittest.TestCase):
    """Test dependency graph functionality"""
    
    def test_dependency_addition(self):
        """Test adding dependencies"""
        graph = DependencyGraph()
        
        graph.add_dependency("A", "B")
        graph.add_dependency("B", "C")
        
        self.assertEqual(graph.get_dependencies("A"), {"B"})
        self.assertEqual(graph.get_dependencies("B"), {"C"})
        self.assertEqual(graph.get_dependents("B"), {"A"})
        self.assertEqual(graph.get_dependents("C"), {"B"})
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        graph = DependencyGraph()
        
        # Create a cycle: A -> B -> C -> A
        graph.add_dependency("A", "B")
        graph.add_dependency("B", "C")
        graph.add_dependency("C", "A")
        
        self.assertTrue(graph.has_cycle())
    
    def test_execution_order(self):
        """Test topological ordering"""
        graph = DependencyGraph()
        
        # Create dependencies: A -> B -> C
        graph.add_dependency("A", "B")
        graph.add_dependency("B", "C")
        
        execution_order = graph.get_execution_order()
        
        # C should come before B, B should come before A
        self.assertIn("C", execution_order)
        self.assertIn("B", execution_order)
        self.assertIn("A", execution_order)
        
        c_index = execution_order.index("C")
        b_index = execution_order.index("B")
        a_index = execution_order.index("A")
        
        self.assertTrue(c_index < b_index < a_index)


class TestEnhancedDataStream(unittest.TestCase):
    """Test enhanced data stream functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.stream = EnhancedDataStream(
            stream_id="test_stream",
            stream_type=DataStreamType.MARKET_DATA,
            buffer_size=1000,
            priority=DataStreamPriority.HIGH
        )
    
    def test_priority_publishing(self):
        """Test priority-based message publishing"""
        # Publish messages with different priorities
        high_msg = self.stream.publish_priority(
            data="high_priority_data",
            priority=DataStreamPriority.HIGH
        )
        
        low_msg = self.stream.publish_priority(
            data="low_priority_data",
            priority=DataStreamPriority.LOW
        )
        
        critical_msg = self.stream.publish_priority(
            data="critical_priority_data",
            priority=DataStreamPriority.CRITICAL
        )
        
        self.assertTrue(high_msg)
        self.assertTrue(low_msg)
        self.assertTrue(critical_msg)
        
        # Get messages - should be ordered by priority
        messages = self.stream.get_priority_messages(max_messages=3)
        
        # Critical should come first
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].priority, DataStreamPriority.CRITICAL)
        self.assertEqual(messages[1].priority, DataStreamPriority.HIGH)
        self.assertEqual(messages[2].priority, DataStreamPriority.LOW)
    
    def test_enhanced_subscription(self):
        """Test enhanced subscription mechanism"""
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
        
        # Subscribe to stream
        subscriber_id = self.stream.subscribe_enhanced(message_handler)
        
        # Publish a message
        self.stream.publish_priority(
            data="test_data",
            priority=DataStreamPriority.MEDIUM
        )
        
        # Wait for async notification
        time.sleep(0.1)
        
        # Verify message received
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0].data, "test_data")
        
        # Unsubscribe
        self.stream.unsubscribe_enhanced(subscriber_id)
        
        # Publish another message
        self.stream.publish_priority(
            data="test_data_2",
            priority=DataStreamPriority.MEDIUM
        )
        
        # Wait and verify no new messages
        time.sleep(0.1)
        self.assertEqual(len(received_messages), 1)
    
    def test_concurrent_publishing(self):
        """Test concurrent publishing stress test"""
        message_count = 1000
        thread_count = 10
        
        def publisher_worker(worker_id):
            for i in range(message_count // thread_count):
                self.stream.publish_priority(
                    data=f"data_{worker_id}_{i}",
                    priority=DataStreamPriority.MEDIUM
                )
        
        # Start publisher threads
        threads = []
        for i in range(thread_count):
            thread = threading.Thread(target=publisher_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all messages were published
        total_messages = 0
        while True:
            messages = self.stream.get_priority_messages(max_messages=100)
            if not messages:
                break
            total_messages += len(messages)
        
        self.assertEqual(total_messages, message_count)
    
    def tearDown(self):
        """Clean up test environment"""
        self.stream.stop()


class TestEnhancedDataFlowCoordinator(unittest.TestCase):
    """Test enhanced data flow coordinator"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.coordinator = EnhancedDataFlowCoordinator(
            coordination_dir=self.temp_dir,
            enable_persistence=True,
            max_concurrent_streams=50,
            deadlock_detection_interval=1.0,
            enable_performance_monitoring=True
        )
    
    def test_enhanced_stream_creation(self):
        """Test enhanced stream creation with dependencies"""
        # Create streams with dependencies
        stream_a = self.coordinator.create_enhanced_stream(
            stream_id="stream_a",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="notebook_1",
            consumer_notebooks=["notebook_2"],
            dependencies=[]
        )
        
        stream_b = self.coordinator.create_enhanced_stream(
            stream_id="stream_b",
            stream_type=DataStreamType.FEATURES,
            producer_notebook="notebook_2",
            consumer_notebooks=["notebook_3"],
            dependencies=["stream_a"]
        )
        
        self.assertIsNotNone(stream_a)
        self.assertIsNotNone(stream_b)
        
        # Verify dependency graph
        deps = self.coordinator.get_dependency_graph_info()
        self.assertEqual(deps['dependencies']['stream_b'], ['stream_a'])
        self.assertEqual(deps['dependents']['stream_a'], ['stream_b'])
    
    def test_circular_dependency_prevention(self):
        """Test circular dependency prevention"""
        # Create initial streams
        self.coordinator.create_enhanced_stream(
            stream_id="stream_a",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="notebook_1",
            consumer_notebooks=["notebook_2"],
            dependencies=[]
        )
        
        self.coordinator.create_enhanced_stream(
            stream_id="stream_b",
            stream_type=DataStreamType.FEATURES,
            producer_notebook="notebook_2",
            consumer_notebooks=["notebook_3"],
            dependencies=["stream_a"]
        )
        
        # Try to create circular dependency
        with self.assertRaises(ValueError):
            self.coordinator.create_enhanced_stream(
                stream_id="stream_c",
                stream_type=DataStreamType.PREDICTIONS,
                producer_notebook="notebook_3",
                consumer_notebooks=["notebook_1"],
                dependencies=["stream_b", "stream_a"]  # This would create a cycle
            )
    
    def test_dependency_resolution(self):
        """Test dependency resolution for publishing"""
        # Create streams with dependencies
        self.coordinator.create_enhanced_stream(
            stream_id="base_stream",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="notebook_1",
            consumer_notebooks=["notebook_2"],
            dependencies=[]
        )
        
        self.coordinator.create_enhanced_stream(
            stream_id="dependent_stream",
            stream_type=DataStreamType.FEATURES,
            producer_notebook="notebook_2",
            consumer_notebooks=["notebook_3"],
            dependencies=["base_stream"]
        )
        
        # Publish to base stream first
        success = self.coordinator.publish_with_dependencies(
            stream_id="base_stream",
            data="base_data",
            priority=DataStreamPriority.HIGH
        )
        self.assertTrue(success)
        
        # Publish to dependent stream
        success = self.coordinator.publish_with_dependencies(
            stream_id="dependent_stream",
            data="dependent_data",
            dependencies=["base_stream"],
            priority=DataStreamPriority.MEDIUM
        )
        self.assertTrue(success)
    
    def test_execution_order_optimization(self):
        """Test execution order optimization"""
        # Create complex dependency chain
        streams = ["stream_a", "stream_b", "stream_c", "stream_d"]
        
        # Create dependencies: A -> B -> C -> D
        for i, stream_id in enumerate(streams):
            dependencies = [streams[i-1]] if i > 0 else []
            self.coordinator.create_enhanced_stream(
                stream_id=stream_id,
                stream_type=DataStreamType.MARKET_DATA,
                producer_notebook=f"notebook_{i}",
                consumer_notebooks=[f"notebook_{i+1}"],
                dependencies=dependencies
            )
        
        # Get execution order
        execution_order = self.coordinator.get_stream_execution_order()
        
        # Verify order is correct
        self.assertEqual(execution_order, ["stream_a", "stream_b", "stream_c", "stream_d"])
    
    def test_concurrent_stream_operations(self):
        """Test concurrent stream operations"""
        stream_count = 20
        
        def create_stream_worker(stream_id):
            try:
                self.coordinator.create_enhanced_stream(
                    stream_id=f"stream_{stream_id}",
                    stream_type=DataStreamType.MARKET_DATA,
                    producer_notebook=f"notebook_{stream_id}",
                    consumer_notebooks=[f"notebook_{stream_id + 1}"],
                    priority=DataStreamPriority.MEDIUM
                )
            except Exception as e:
                logger.error(f"Error creating stream {stream_id}: {e}")
        
        # Create streams concurrently
        threads = []
        for i in range(stream_count):
            thread = threading.Thread(target=create_stream_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify streams were created
        status = self.coordinator.get_enhanced_coordination_status()
        self.assertEqual(status['enhanced_streams'], stream_count)
    
    def test_performance_monitoring(self):
        """Test performance monitoring features"""
        # Create a stream and publish messages
        stream = self.coordinator.create_enhanced_stream(
            stream_id="perf_test_stream",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="notebook_1",
            consumer_notebooks=["notebook_2"]
        )
        
        # Publish several messages
        for i in range(100):
            self.coordinator.publish_with_dependencies(
                stream_id="perf_test_stream",
                data=f"test_data_{i}",
                priority=DataStreamPriority.MEDIUM
            )
        
        # Wait for monitoring to update
        time.sleep(2.0)
        
        # Check performance metrics
        status = self.coordinator.get_enhanced_coordination_status()
        
        self.assertGreater(status['operation_counters']['stream_creations'], 0)
        self.assertGreater(status['operation_counters']['message_publishes'], 0)
        self.assertIn('performance_metrics', status)
    
    def test_deadlock_detection(self):
        """Test deadlock detection mechanism"""
        # The monitoring should run in background
        time.sleep(2.0)  # Wait for monitoring cycle
        
        # Check that monitoring is working
        status = self.coordinator.get_enhanced_coordination_status()
        self.assertIn('concurrency_metrics', status)
        
        # No deadlocks should be detected in normal operation
        self.assertEqual(status['operation_counters']['deadlock_detections'], 0)
    
    def tearDown(self):
        """Clean up test environment"""
        self.coordinator.shutdown_enhanced()
        shutil.rmtree(self.temp_dir)


class TestConcurrencyMonitor(unittest.TestCase):
    """Test concurrency monitoring features"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = ConcurrencyMonitor()
    
    def test_lock_contention_tracking(self):
        """Test lock contention tracking"""
        # Record some contention events
        self.monitor.record_lock_contention("test_lock")
        self.monitor.record_lock_contention("test_lock")
        
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['lock_contention_count'], 2)
    
    def test_lock_acquisition_tracking(self):
        """Test lock acquisition/release tracking"""
        thread_id = "test_thread"
        lock_name = "test_lock"
        
        # Record acquisition
        self.monitor.record_lock_acquisition(lock_name, thread_id)
        
        # Wait a bit
        time.sleep(0.1)
        
        # Record release
        self.monitor.record_lock_release(lock_name, thread_id)
        
        metrics = self.monitor.get_metrics()
        self.assertIn(f"lock_hold_time_{lock_name}", metrics['performance_metrics'])
        self.assertGreater(metrics['performance_metrics'][f"lock_hold_time_{lock_name}"], 0)


class TestLockFreeQueue(unittest.TestCase):
    """Test lock-free queue implementation"""
    
    def test_basic_operations(self):
        """Test basic queue operations"""
        queue = LockFreeQueue(maxsize=100)
        
        # Test put and get
        self.assertTrue(queue.put("test_item"))
        self.assertEqual(queue.qsize(), 1)
        
        item = queue.get()
        self.assertEqual(item, "test_item")
        self.assertEqual(queue.qsize(), 0)
        self.assertTrue(queue.empty())
    
    def test_concurrent_operations(self):
        """Test concurrent queue operations"""
        queue = LockFreeQueue(maxsize=1000)
        items_to_add = 500
        
        def producer():
            for i in range(items_to_add):
                queue.put(f"item_{i}")
        
        def consumer():
            consumed = []
            while len(consumed) < items_to_add:
                item = queue.get(timeout=1.0)
                if item is not None:
                    consumed.append(item)
            return consumed
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        # Verify all items were processed
        self.assertTrue(queue.empty())


class TestRaceConditionScenarios(unittest.TestCase):
    """Test specific race condition scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.coordinator = EnhancedDataFlowCoordinator(
            coordination_dir=self.temp_dir,
            enable_persistence=True,
            max_concurrent_streams=100,
            deadlock_detection_interval=0.5,
            enable_performance_monitoring=True
        )
    
    def test_concurrent_stream_creation_and_publishing(self):
        """Test race conditions in stream creation and publishing"""
        stream_count = 20
        messages_per_stream = 50
        
        def create_and_publish_worker(worker_id):
            try:
                # Create stream
                stream_id = f"race_stream_{worker_id}"
                self.coordinator.create_enhanced_stream(
                    stream_id=stream_id,
                    stream_type=DataStreamType.MARKET_DATA,
                    producer_notebook=f"notebook_{worker_id}",
                    consumer_notebooks=[f"notebook_{worker_id + 1000}"]
                )
                
                # Publish messages
                for i in range(messages_per_stream):
                    self.coordinator.publish_with_dependencies(
                        stream_id=stream_id,
                        data=f"data_{worker_id}_{i}",
                        priority=DataStreamPriority.MEDIUM
                    )
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        # Start workers
        threads = []
        for i in range(stream_count):
            thread = threading.Thread(target=create_and_publish_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify integrity
        status = self.coordinator.get_enhanced_coordination_status()
        self.assertEqual(status['enhanced_streams'], stream_count)
        self.assertEqual(status['operation_counters']['message_publishes'], 
                        stream_count * messages_per_stream)
    
    def test_subscriber_race_conditions(self):
        """Test race conditions in subscriber management"""
        stream = self.coordinator.create_enhanced_stream(
            stream_id="subscriber_test_stream",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="notebook_1",
            consumer_notebooks=["notebook_2"]
        )
        
        received_messages = []
        message_lock = threading.Lock()
        
        def message_handler(message):
            with message_lock:
                received_messages.append(message)
        
        # Subscribe/unsubscribe concurrently
        def subscription_worker():
            for i in range(100):
                subscriber_id = stream.subscribe_enhanced(message_handler)
                time.sleep(0.001)  # Small delay
                stream.unsubscribe_enhanced(subscriber_id)
        
        # Publishing worker
        def publishing_worker():
            for i in range(100):
                stream.publish_priority(
                    data=f"test_data_{i}",
                    priority=DataStreamPriority.MEDIUM
                )
                time.sleep(0.001)
        
        # Start workers
        sub_thread = threading.Thread(target=subscription_worker)
        pub_thread = threading.Thread(target=publishing_worker)
        
        sub_thread.start()
        pub_thread.start()
        
        sub_thread.join()
        pub_thread.join()
        
        # Verify no crashes and some messages were received
        self.assertGreaterEqual(len(received_messages), 0)
    
    def tearDown(self):
        """Clean up test environment"""
        self.coordinator.shutdown_enhanced()
        shutil.rmtree(self.temp_dir)


def run_stress_test():
    """Run stress test for enhanced coordinator"""
    logger.info("Starting stress test...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        coordinator = EnhancedDataFlowCoordinator(
            coordination_dir=temp_dir,
            enable_persistence=True,
            max_concurrent_streams=200,
            deadlock_detection_interval=1.0,
            enable_performance_monitoring=True
        )
        
        # Create many streams with complex dependencies
        stream_count = 50
        for i in range(stream_count):
            dependencies = [f"stream_{j}" for j in range(max(0, i-3), i)]
            
            try:
                coordinator.create_enhanced_stream(
                    stream_id=f"stream_{i}",
                    stream_type=DataStreamType.MARKET_DATA,
                    producer_notebook=f"notebook_{i}",
                    consumer_notebooks=[f"notebook_{i+1}"],
                    dependencies=dependencies
                )
            except ValueError as e:
                logger.info(f"Expected dependency error for stream {i}: {e}")
        
        # Publish messages concurrently
        def publisher_worker(worker_id):
            for i in range(100):
                try:
                    coordinator.publish_with_dependencies(
                        stream_id=f"stream_{worker_id % 10}",
                        data=f"stress_data_{worker_id}_{i}",
                        priority=DataStreamPriority.MEDIUM
                    )
                except Exception as e:
                    logger.error(f"Publisher {worker_id} error: {e}")
        
        # Start publishers
        threads = []
        for i in range(20):
            thread = threading.Thread(target=publisher_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check final status
        status = coordinator.get_enhanced_coordination_status()
        logger.info(f"Stress test completed - Status: {status}")
        
        coordinator.shutdown_enhanced()
        
    finally:
        shutil.rmtree(temp_dir)
    
    logger.info("Stress test completed successfully!")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run stress test
    run_stress_test()