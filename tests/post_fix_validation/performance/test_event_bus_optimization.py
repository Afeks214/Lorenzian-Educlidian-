"""
Test suite for event bus optimization validation.
Validates that async event processing and batching improvements work correctly.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.performance.async_event_bus import AsyncEventBus, BatchingStrategy, EventBatch
from src.core.events import Event, EventType
from datetime import datetime


class TestEventBusOptimization:
    """Test event bus optimization fixes"""

    @pytest.fixture
    def event_bus(self):
        """Create test event bus instance"""
        return AsyncEventBus(
            max_workers=4,
            batch_window_ms=10,
            max_batch_size=100,
            strategy=BatchingStrategy.HYBRID,
            enable_monitoring=True
        )

    @pytest.fixture
    def test_events(self):
        """Create test events"""
        events = []
        for i in range(100):
            event = Event(
                event_type=EventType.NEW_TICK,
                timestamp=datetime.now(),
                payload={'tick_id': i, 'price': 100.0 + i},
                source=f"test_source_{i}"
            )
            events.append(event)
        return events

    def test_async_event_processing_performance(self, event_bus, test_events):
        """Test that async event processing achieves target performance"""
        async def run_performance_test():
            await event_bus.start()
            
            # Set up callback to track processed events
            processed_events = []
            
            def test_callback(event: Event):
                processed_events.append(event)
            
            # Subscribe to events
            event_bus.subscribe(EventType.NEW_TICK, test_callback)
            
            # Measure processing time
            start_time = time.perf_counter()
            
            # Publish events
            for event in test_events:
                event_bus.publish(event)
            
            # Wait for processing to complete
            await asyncio.sleep(0.1)  # Allow time for batching and processing
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            await event_bus.stop()
            
            # Verify performance targets
            assert processing_time < 0.1  # Should process 100 events in under 100ms
            assert len(processed_events) == len(test_events)
            
            # Verify 80-90% latency reduction target
            # (compared to synchronous processing baseline)
            expected_sync_time = len(test_events) * 0.001  # 1ms per event synchronously
            latency_reduction = (expected_sync_time - processing_time) / expected_sync_time
            assert latency_reduction >= 0.8  # 80% reduction minimum

        asyncio.run(run_performance_test())

    def test_batch_processing_efficiency(self, event_bus, test_events):
        """Test event batching efficiency"""
        async def run_batch_test():
            await event_bus.start()
            
            # Publish events rapidly
            for event in test_events:
                event_bus.publish(event)
            
            # Wait for batching
            await asyncio.sleep(0.05)
            
            # Get metrics
            metrics = event_bus.get_metrics()
            
            await event_bus.stop()
            
            # Verify batch efficiency
            assert metrics['batch_efficiency'] > 0.8  # 80% efficiency
            assert metrics['avg_batch_size'] > 10  # Should batch multiple events
            assert metrics['batches_processed'] > 0

        asyncio.run(run_batch_test())

    def test_concurrent_event_processing(self, event_bus):
        """Test concurrent event processing without race conditions"""
        async def run_concurrency_test():
            await event_bus.start()
            
            # Track processed events with thread safety
            processed_events = []
            lock = threading.Lock()
            
            def thread_safe_callback(event: Event):
                with lock:
                    processed_events.append(event)
            
            # Subscribe to multiple event types
            event_bus.subscribe(EventType.NEW_TICK, thread_safe_callback)
            event_bus.subscribe(EventType.NEW_5MIN_BAR, thread_safe_callback)
            
            # Publish events from multiple threads
            def publish_events(event_type: EventType, count: int):
                for i in range(count):
                    event = Event(
                        event_type=event_type,
                        timestamp=datetime.now(),
                        payload={'id': i},
                        source=f"thread_{threading.current_thread().ident}"
                    )
                    event_bus.publish(event)
            
            # Start concurrent publishing
            threads = []
            for event_type in [EventType.NEW_TICK, EventType.NEW_5MIN_BAR]:
                thread = threading.Thread(
                    target=publish_events,
                    args=(event_type, 50)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            await event_bus.stop()
            
            # Verify all events were processed
            assert len(processed_events) == 100  # 50 per event type
            
            # Verify no race conditions (no duplicate processing)
            event_ids = [(e.event_type, e.payload['id']) for e in processed_events]
            assert len(set(event_ids)) == len(event_ids)

        asyncio.run(run_concurrency_test())

    def test_memory_usage_optimization(self, event_bus):
        """Test memory usage optimization"""
        async def run_memory_test():
            await event_bus.start()
            
            # Get initial memory usage
            initial_metrics = event_bus.get_metrics()
            
            # Process large number of events
            for i in range(1000):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'large_data': 'x' * 1000},  # 1KB per event
                    source=f"memory_test_{i}"
                )
                event_bus.publish(event)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Get final metrics
            final_metrics = event_bus.get_metrics()
            
            await event_bus.stop()
            
            # Verify memory usage remains reasonable
            assert final_metrics['queue_depth'] < 100  # Queue should be processed
            assert final_metrics['events_processed'] == 1000

        asyncio.run(run_memory_test())

    def test_emergency_event_bypass(self, event_bus):
        """Test emergency event bypass mechanism"""
        async def run_emergency_test():
            await event_bus.start()
            
            # Track processed events with timestamps
            processed_events = []
            
            def emergency_callback(event: Event):
                processed_events.append({
                    'event': event,
                    'processed_at': time.perf_counter()
                })
            
            # Subscribe to emergency events
            event_bus.subscribe(EventType.EMERGENCY_STOP, emergency_callback)
            
            # Publish emergency event
            emergency_event = Event(
                event_type=EventType.EMERGENCY_STOP,
                timestamp=datetime.now(),
                payload={'reason': 'test_emergency'},
                source='emergency_test'
            )
            
            publish_time = time.perf_counter()
            event_bus.publish(emergency_event)
            
            # Wait briefly for processing
            await asyncio.sleep(0.01)
            
            await event_bus.stop()
            
            # Verify emergency event was processed immediately
            assert len(processed_events) == 1
            processing_latency = processed_events[0]['processed_at'] - publish_time
            assert processing_latency < 0.001  # Under 1ms for emergency events

        asyncio.run(run_emergency_test())

    def test_event_deduplication(self, event_bus):
        """Test high-frequency event deduplication"""
        async def run_deduplication_test():
            await event_bus.start()
            
            processed_events = []
            
            def dedup_callback(event: Event):
                processed_events.append(event)
            
            # Subscribe to high-frequency events
            event_bus.subscribe(EventType.NEW_TICK, dedup_callback)
            
            # Publish duplicate events rapidly
            for i in range(10):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'price': 100.0},  # Same price
                    source='dedup_test'
                )
                event_bus.publish(event)
            
            # Wait for processing
            await asyncio.sleep(0.05)
            
            await event_bus.stop()
            
            # Verify deduplication occurred
            # Should process fewer events due to deduplication
            assert len(processed_events) < 10

        asyncio.run(run_deduplication_test())

    def test_batching_strategies(self):
        """Test different batching strategies"""
        strategies = [
            BatchingStrategy.TIME_BASED,
            BatchingStrategy.SIZE_BASED,
            BatchingStrategy.HYBRID,
            BatchingStrategy.PRIORITY_BASED
        ]
        
        for strategy in strategies:
            async def run_strategy_test():
                event_bus = AsyncEventBus(
                    max_workers=4,
                    batch_window_ms=20,
                    max_batch_size=50,
                    strategy=strategy
                )
                
                await event_bus.start()
                
                # Publish events with different priorities
                for i in range(100):
                    event_type = EventType.NEW_TICK if i % 2 == 0 else EventType.RISK_BREACH
                    event = Event(
                        event_type=event_type,
                        timestamp=datetime.now(),
                        payload={'id': i},
                        source=f'strategy_test_{i}'
                    )
                    event_bus.publish(event)
                
                # Wait for processing
                await asyncio.sleep(0.1)
                
                # Get metrics
                metrics = event_bus.get_metrics()
                
                await event_bus.stop()
                
                # Verify strategy worked
                assert metrics['events_processed'] == 100
                assert metrics['batches_processed'] > 0
                
                # Strategy-specific assertions
                if strategy == BatchingStrategy.PRIORITY_BASED:
                    # Priority-based should process high-priority events first
                    assert metrics['avg_processing_time_ms'] < 10
                elif strategy == BatchingStrategy.SIZE_BASED:
                    # Size-based should have consistent batch sizes
                    assert metrics['avg_batch_size'] > 10
                elif strategy == BatchingStrategy.TIME_BASED:
                    # Time-based should batch based on time windows
                    assert metrics['avg_processing_time_ms'] < 50

            asyncio.run(run_strategy_test())

    def test_error_handling_in_batches(self, event_bus):
        """Test error handling in batch processing"""
        async def run_error_test():
            await event_bus.start()
            
            # Create callback that fails for certain events
            def error_callback(event: Event):
                if event.payload.get('should_fail', False):
                    raise ValueError("Test error")
                # Success case
                return True
            
            # Subscribe to events
            event_bus.subscribe(EventType.NEW_TICK, error_callback)
            
            # Publish mix of good and bad events
            for i in range(10):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'id': i, 'should_fail': i % 3 == 0},  # Every 3rd event fails
                    source=f'error_test_{i}'
                )
                event_bus.publish(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Get metrics
            metrics = event_bus.get_metrics()
            
            await event_bus.stop()
            
            # Verify error handling
            assert metrics['callback_errors'] > 0  # Should have recorded errors
            assert metrics['events_processed'] == 10  # Should have processed all events

        asyncio.run(run_error_test())

    @pytest.mark.integration
    def test_performance_under_load(self):
        """Test performance under high load"""
        async def run_load_test():
            event_bus = AsyncEventBus(
                max_workers=8,
                batch_window_ms=5,
                max_batch_size=200,
                strategy=BatchingStrategy.HYBRID
            )
            
            await event_bus.start()
            
            # High load test
            start_time = time.perf_counter()
            
            # Publish 10,000 events
            for i in range(10000):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'id': i, 'price': 100.0 + (i % 100)},
                    source=f'load_test_{i}'
                )
                event_bus.publish(event)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Get final metrics
            metrics = event_bus.get_metrics()
            
            await event_bus.stop()
            
            # Verify performance targets
            assert total_time < 2.0  # Should process 10K events in under 2 seconds
            assert metrics['events_processed'] == 10000
            assert metrics['throughput_eps'] > 5000  # 5K events per second minimum

        asyncio.run(run_load_test())

    def test_graceful_shutdown(self, event_bus):
        """Test graceful shutdown with pending events"""
        async def run_shutdown_test():
            await event_bus.start()
            
            # Publish events
            for i in range(100):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'id': i},
                    source=f'shutdown_test_{i}'
                )
                event_bus.publish(event)
            
            # Immediate shutdown
            await event_bus.stop()
            
            # Verify no errors during shutdown
            # This test passes if no exceptions are raised

        asyncio.run(run_shutdown_test())

    def test_metrics_accuracy(self, event_bus):
        """Test metrics accuracy and reporting"""
        async def run_metrics_test():
            await event_bus.start()
            
            # Clear metrics
            event_bus.reset_metrics()
            
            # Publish known number of events
            event_count = 50
            for i in range(event_count):
                event = Event(
                    event_type=EventType.NEW_TICK,
                    timestamp=datetime.now(),
                    payload={'id': i},
                    source=f'metrics_test_{i}'
                )
                event_bus.publish(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Get metrics
            metrics = event_bus.get_metrics()
            
            await event_bus.stop()
            
            # Verify metrics accuracy
            assert metrics['events_processed'] == event_count
            assert metrics['batches_processed'] > 0
            assert metrics['avg_batch_size'] > 0
            assert metrics['avg_processing_time_ms'] > 0
            assert metrics['throughput_eps'] > 0

        asyncio.run(run_metrics_test())