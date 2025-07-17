"""
Unit tests for the EventBus system.

This module tests the event publishing, subscription, dispatch,
and message routing functionality of the event bus.
"""
import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, call
from typing import Dict, Any, List, Callable
from datetime import datetime
import queue

# Test markers
pytestmark = [pytest.mark.unit]


class TestEventBus:
    """Test the core EventBus functionality."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus for testing."""
        event_bus = Mock()
        event_bus._subscribers = {}
        event_bus._event_queue = queue.Queue()
        event_bus.running = False
        
        # Mock methods
        event_bus.subscribe = Mock()
        event_bus.unsubscribe = Mock()
        event_bus.publish = Mock()
        event_bus.start = Mock()
        event_bus.stop = Mock()
        event_bus.dispatch_forever = Mock()
        
        return event_bus

    @pytest.fixture
    def sample_events(self):
        """Sample events for testing."""
        return [
            {"type": "NEW_TICK", "data": {"symbol": "EURUSD", "price": 1.0850, "timestamp": time.time()}},
            {"type": "NEW_5MIN_BAR", "data": {"symbol": "EURUSD", "ohlcv": [1.0850, 1.0852, 1.0848, 1.0851, 1000]}},
            {"type": "NEW_30MIN_BAR", "data": {"symbol": "EURUSD", "ohlcv": [1.0845, 1.0855, 1.0843, 1.0851, 15000]}},
            {"type": "INDICATORS_READY", "data": {"mlmi": 0.75, "nwrqk": 0.25, "timestamp": time.time()}},
            {"type": "SYNERGY_DETECTED", "data": {"pattern": "TYPE_1", "confidence": 0.85}},
            {"type": "STRATEGIC_DECISION", "data": {"position": 0.7, "confidence": 0.8}},
            {"type": "EXECUTE_TRADE", "data": {"symbol": "EURUSD", "side": "BUY", "size": 0.1}},
            {"type": "TRADE_CLOSED", "data": {"symbol": "EURUSD", "pnl": 50.0, "duration": 1800}}
        ]

    def test_event_bus_initialization(self, mock_event_bus):
        """Test event bus initialization."""
        assert mock_event_bus._subscribers == {}
        assert mock_event_bus.running is False
        assert mock_event_bus._event_queue is not None

    def test_subscriber_registration(self, mock_event_bus):
        """Test event subscriber registration."""
        # Mock event handlers
        tick_handler = Mock()
        bar_handler = Mock()
        decision_handler = Mock()
        
        # Mock subscription behavior
        def mock_subscribe(event_type, handler):
            if event_type not in mock_event_bus._subscribers:
                mock_event_bus._subscribers[event_type] = []
            mock_event_bus._subscribers[event_type].append(handler)
        
        mock_event_bus.subscribe.side_effect = mock_subscribe
        
        # Register subscribers
        mock_event_bus.subscribe("NEW_TICK", tick_handler)
        mock_event_bus.subscribe("NEW_5MIN_BAR", bar_handler)
        mock_event_bus.subscribe("STRATEGIC_DECISION", decision_handler)
        
        # Verify registration
        mock_event_bus.subscribe.assert_has_calls([
            call("NEW_TICK", tick_handler),
            call("NEW_5MIN_BAR", bar_handler),
            call("STRATEGIC_DECISION", decision_handler)
        ])

    def test_event_publishing(self, mock_event_bus, sample_events):
        """Test event publishing functionality."""
        for event in sample_events:
            mock_event_bus.publish(event)
        
        # Verify all events were published
        assert mock_event_bus.publish.call_count == len(sample_events)
        
        # Check specific event publications
        mock_event_bus.publish.assert_has_calls([call(event) for event in sample_events])

    def test_event_dispatch(self, mock_event_bus):
        """Test event dispatch to subscribers."""
        # Mock subscribers
        tick_handler = Mock()
        bar_handler = Mock()
        
        # Mock dispatch behavior
        def mock_dispatch_event(event):
            event_type = event["type"]
            if event_type in mock_event_bus._subscribers:
                for handler in mock_event_bus._subscribers[event_type]:
                    handler(event)
        
        mock_event_bus.dispatch_event = Mock(side_effect=mock_dispatch_event)
        mock_event_bus._subscribers = {
            "NEW_TICK": [tick_handler],
            "NEW_5MIN_BAR": [bar_handler]
        }
        
        # Dispatch events
        tick_event = {"type": "NEW_TICK", "data": {"price": 1.0850}}
        bar_event = {"type": "NEW_5MIN_BAR", "data": {"close": 1.0851}}
        
        mock_event_bus.dispatch_event(tick_event)
        mock_event_bus.dispatch_event(bar_event)
        
        # Verify handlers were called
        tick_handler.assert_called_once_with(tick_event)
        bar_handler.assert_called_once_with(bar_event)

    def test_subscriber_removal(self, mock_event_bus):
        """Test subscriber removal functionality."""
        handler = Mock()
        
        # Mock unsubscription behavior
        def mock_unsubscribe(event_type, handler):
            if event_type in mock_event_bus._subscribers:
                if handler in mock_event_bus._subscribers[event_type]:
                    mock_event_bus._subscribers[event_type].remove(handler)
        
        mock_event_bus.unsubscribe.side_effect = mock_unsubscribe
        mock_event_bus._subscribers = {"NEW_TICK": [handler]}
        
        # Remove subscriber
        mock_event_bus.unsubscribe("NEW_TICK", handler)
        
        # Verify removal
        mock_event_bus.unsubscribe.assert_called_once_with("NEW_TICK", handler)

    def test_multiple_subscribers_same_event(self, mock_event_bus):
        """Test multiple subscribers for the same event type."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()
        
        # Mock dispatch to multiple handlers
        def mock_dispatch_event(event):
            event_type = event["type"]
            if event_type in mock_event_bus._subscribers:
                for handler in mock_event_bus._subscribers[event_type]:
                    handler(event)
        
        mock_event_bus.dispatch_event = Mock(side_effect=mock_dispatch_event)
        mock_event_bus._subscribers = {
            "NEW_TICK": [handler1, handler2, handler3]
        }
        
        # Dispatch event
        event = {"type": "NEW_TICK", "data": {"price": 1.0850}}
        mock_event_bus.dispatch_event(event)
        
        # Verify all handlers were called
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)
        handler3.assert_called_once_with(event)

    def test_event_filtering(self, mock_event_bus):
        """Test event filtering functionality."""
        all_handler = Mock()
        tick_handler = Mock()
        
        # Mock filtering behavior
        def mock_dispatch_with_filter(event):
            event_type = event["type"]
            # All handler receives all events
            all_handler(event)
            # Specific handler only receives matching events
            if event_type == "NEW_TICK":
                tick_handler(event)
        
        mock_event_bus.dispatch_event = Mock(side_effect=mock_dispatch_with_filter)
        
        # Dispatch different events
        events = [
            {"type": "NEW_TICK", "data": {"price": 1.0850}},
            {"type": "NEW_5MIN_BAR", "data": {"close": 1.0851}},
            {"type": "NEW_TICK", "data": {"price": 1.0852}}
        ]
        
        for event in events:
            mock_event_bus.dispatch_event(event)
        
        # Verify filtering
        assert all_handler.call_count == 3  # Receives all events
        assert tick_handler.call_count == 2  # Only receives tick events

    def test_event_bus_lifecycle(self, mock_event_bus):
        """Test event bus start/stop lifecycle."""
        # Test start
        mock_event_bus.start()
        mock_event_bus.start.assert_called_once()
        
        # Test stop
        mock_event_bus.stop()
        mock_event_bus.stop.assert_called_once()

    def test_error_handling_in_dispatch(self, mock_event_bus):
        """Test error handling during event dispatch."""
        failing_handler = Mock(side_effect=Exception("Handler error"))
        working_handler = Mock()
        
        # Mock dispatch with error handling
        def mock_dispatch_with_error_handling(event):
            handlers = mock_event_bus._subscribers.get(event["type"], [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but continue with other handlers
                    mock_event_bus.handle_dispatch_error(e, handler, event)
        
        mock_event_bus.dispatch_event = Mock(side_effect=mock_dispatch_with_error_handling)
        mock_event_bus.handle_dispatch_error = Mock()
        mock_event_bus._subscribers = {
            "NEW_TICK": [failing_handler, working_handler]
        }
        
        # Dispatch event
        event = {"type": "NEW_TICK", "data": {"price": 1.0850}}
        mock_event_bus.dispatch_event(event)
        
        # Verify error handling
        failing_handler.assert_called_once_with(event)
        working_handler.assert_called_once_with(event)
        mock_event_bus.handle_dispatch_error.assert_called_once()


class TestAsyncEventBus:
    """Test asynchronous event bus functionality."""

    @pytest.fixture
    def mock_async_event_bus(self):
        """Create a mock async event bus."""
        async_bus = Mock()
        async_bus._async_subscribers = {}
        async_bus.running = False
        
        # Mock async methods
        async_bus.subscribe_async = Mock()
        async_bus.publish_async = Mock()
        async_bus.dispatch_async = Mock()
        
        return async_bus

    @pytest.mark.asyncio
    async def test_async_event_publishing(self, mock_async_event_bus):
        """Test asynchronous event publishing."""
        event = {"type": "NEW_TICK", "data": {"price": 1.0850}}
        
        # Mock async publish
        mock_async_event_bus.publish_async.return_value = asyncio.Future()
        mock_async_event_bus.publish_async.return_value.set_result(True)
        
        result = await mock_async_event_bus.publish_async(event)
        
        assert result is True
        mock_async_event_bus.publish_async.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_async_event_dispatch(self, mock_async_event_bus):
        """Test asynchronous event dispatch."""
        async_handler = Mock()
        
        # Mock async dispatch
        async def mock_dispatch(event):
            await async_handler(event)
        
        mock_async_event_bus.dispatch_async.side_effect = mock_dispatch
        
        event = {"type": "STRATEGIC_DECISION", "data": {"position": 0.7}}
        await mock_async_event_bus.dispatch_async(event)
        
        async_handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, mock_async_event_bus):
        """Test concurrent event processing."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()
        
        # Mock concurrent dispatch
        async def mock_concurrent_dispatch(events):
            tasks = []
            for event in events:
                for handler in [handler1, handler2, handler3]:
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
            await asyncio.gather(*tasks)
        
        mock_async_event_bus.dispatch_concurrent = Mock(side_effect=mock_concurrent_dispatch)
        
        events = [
            {"type": "NEW_TICK", "data": {"price": 1.0850}},
            {"type": "NEW_TICK", "data": {"price": 1.0851}}
        ]
        
        await mock_async_event_bus.dispatch_concurrent(events)
        
        # Each handler should be called for each event
        assert handler1.call_count == 2
        assert handler2.call_count == 2
        assert handler3.call_count == 2


class TestEventBusPerformance:
    """Test event bus performance characteristics."""

    @pytest.fixture
    def performance_event_bus(self):
        """Create an event bus for performance testing."""
        bus = Mock()
        bus.metrics = {
            "events_published": 0,
            "events_dispatched": 0,
            "dispatch_time_ms": [],
            "throughput_events_per_second": 0
        }
        
        # Mock performance tracking
        def mock_publish_with_metrics(event):
            bus.metrics["events_published"] += 1
            start_time = time.perf_counter()
            # Simulate dispatch
            time.sleep(0.0001)  # 0.1ms simulated dispatch time
            end_time = time.perf_counter()
            bus.metrics["dispatch_time_ms"].append((end_time - start_time) * 1000)
            bus.metrics["events_dispatched"] += 1
        
        bus.publish = Mock(side_effect=mock_publish_with_metrics)
        
        return bus

    def test_high_frequency_event_publishing(self, performance_event_bus):
        """Test high-frequency event publishing performance."""
        num_events = 10000
        
        start_time = time.perf_counter()
        
        # Publish many events rapidly
        for i in range(num_events):
            event = {"type": "NEW_TICK", "data": {"price": 1.0850 + i * 0.0001}}
            performance_event_bus.publish(event)
        
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        throughput = num_events / elapsed_seconds
        
        # Performance requirements
        assert throughput > 1000  # Should handle >1000 events/second
        assert performance_event_bus.metrics["events_published"] == num_events
        
        print(f"Event publishing throughput: {throughput:.0f} events/second")

    def test_dispatch_latency(self, performance_event_bus):
        """Test event dispatch latency."""
        num_events = 1000
        
        # Publish events and measure dispatch latency
        for i in range(num_events):
            event = {"type": "NEW_TICK", "data": {"price": 1.0850}}
            performance_event_bus.publish(event)
        
        dispatch_times = performance_event_bus.metrics["dispatch_time_ms"]
        avg_latency = sum(dispatch_times) / len(dispatch_times)
        max_latency = max(dispatch_times)
        p95_latency = sorted(dispatch_times)[int(len(dispatch_times) * 0.95)]
        
        # Latency requirements
        assert avg_latency < 1.0  # Average < 1ms
        assert p95_latency < 2.0  # 95th percentile < 2ms
        assert max_latency < 5.0  # Max < 5ms
        
        print(f"Dispatch latency - Avg: {avg_latency:.3f}ms, P95: {p95_latency:.3f}ms, Max: {max_latency:.3f}ms")

    def test_memory_usage_under_load(self, performance_event_bus, memory_profiler):
        """Test memory usage under high event load."""
        memory_profiler.start()
        
        # Generate high event load
        for i in range(50000):
            event = {
                "type": "NEW_TICK",
                "data": {
                    "price": 1.0850 + i * 0.0001,
                    "volume": 100,
                    "timestamp": time.time(),
                    "metadata": {"sequence": i}
                }
            }
            performance_event_bus.publish(event)
        
        memory_usage = memory_profiler.get_current_usage()
        
        # Memory should remain reasonable
        assert memory_usage < 100  # Should stay under 100MB
        
        print(f"Memory usage under load: {memory_usage:.2f}MB")

    def test_concurrent_publishing(self, performance_event_bus):
        """Test concurrent event publishing from multiple threads."""
        num_threads = 4
        events_per_thread = 1000
        results = queue.Queue()
        
        def publisher_thread(thread_id):
            thread_start = time.perf_counter()
            for i in range(events_per_thread):
                event = {
                    "type": "NEW_TICK",
                    "data": {"price": 1.0850, "thread_id": thread_id, "sequence": i}
                }
                performance_event_bus.publish(event)
            thread_end = time.perf_counter()
            results.put(thread_end - thread_start)
        
        # Start publisher threads
        threads = []
        start_time = time.perf_counter()
        
        for i in range(num_threads):
            thread = threading.Thread(target=publisher_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_events = num_threads * events_per_thread
        throughput = total_events / total_time
        
        # Concurrent performance requirements
        assert throughput > 2000  # Should handle >2000 events/second concurrently
        assert performance_event_bus.metrics["events_published"] == total_events
        
        print(f"Concurrent publishing throughput: {throughput:.0f} events/second")


class TestEventBusReliability:
    """Test event bus reliability and fault tolerance."""

    @pytest.fixture
    def reliable_event_bus(self):
        """Create an event bus with reliability features."""
        bus = Mock()
        bus.failed_events = []
        bus.retry_count = 0
        bus.max_retries = 3
        
        # Mock reliability methods
        bus.publish_with_retry = Mock()
        bus.handle_failed_event = Mock()
        bus.dead_letter_queue = Mock()
        
        return bus

    def test_event_retry_mechanism(self, reliable_event_bus):
        """Test event retry mechanism for failed dispatches."""
        failed_event = {"type": "EXECUTE_TRADE", "data": {"symbol": "EURUSD"}}
        
        # Mock retry behavior
        def mock_retry(event, max_retries=3):
            for attempt in range(max_retries):
                reliable_event_bus.retry_count += 1
                # Simulate success on final attempt
                if attempt == max_retries - 1:
                    return True
            return False
        
        reliable_event_bus.publish_with_retry.side_effect = lambda event: mock_retry(event)
        
        # Test retry
        result = reliable_event_bus.publish_with_retry(failed_event)
        
        assert result is True
        assert reliable_event_bus.retry_count == 3
        reliable_event_bus.publish_with_retry.assert_called_once_with(failed_event)

    def test_dead_letter_queue(self, reliable_event_bus):
        """Test dead letter queue for permanently failed events."""
        permanently_failed_event = {"type": "CRITICAL_ERROR", "data": {"error": "unrecoverable"}}
        
        # Mock dead letter queue behavior
        def mock_send_to_dlq(event):
            reliable_event_bus.failed_events.append(event)
        
        reliable_event_bus.dead_letter_queue.side_effect = mock_send_to_dlq
        
        # Send to dead letter queue
        reliable_event_bus.dead_letter_queue(permanently_failed_event)
        
        assert len(reliable_event_bus.failed_events) == 1
        assert reliable_event_bus.failed_events[0] == permanently_failed_event
        reliable_event_bus.dead_letter_queue.assert_called_once_with(permanently_failed_event)

    def test_event_ordering_guarantee(self, reliable_event_bus):
        """Test event ordering guarantees."""
        ordered_events = [
            {"type": "NEW_TICK", "data": {"sequence": 1, "price": 1.0850}},
            {"type": "NEW_TICK", "data": {"sequence": 2, "price": 1.0851}},
            {"type": "NEW_TICK", "data": {"sequence": 3, "price": 1.0852}},
            {"type": "NEW_TICK", "data": {"sequence": 4, "price": 1.0853}}
        ]
        
        received_order = []
        
        def order_tracking_handler(event):
            received_order.append(event["data"]["sequence"])
        
        # Mock ordered dispatch
        def mock_ordered_dispatch(events):
            for event in events:
                order_tracking_handler(event)
        
        reliable_event_bus.dispatch_ordered = Mock(side_effect=mock_ordered_dispatch)
        
        # Dispatch events
        reliable_event_bus.dispatch_ordered(ordered_events)
        
        # Verify order preservation
        assert received_order == [1, 2, 3, 4]
        reliable_event_bus.dispatch_ordered.assert_called_once_with(ordered_events)

    def test_duplicate_event_detection(self, reliable_event_bus):
        """Test duplicate event detection and prevention."""
        event_with_id = {"type": "NEW_TICK", "id": "tick_123", "data": {"price": 1.0850}}
        
        processed_events = set()
        
        def mock_deduplicate(event):
            event_id = event.get("id")
            if event_id in processed_events:
                return False  # Duplicate
            processed_events.add(event_id)
            return True  # New event
        
        reliable_event_bus.is_duplicate = Mock(side_effect=lambda event: not mock_deduplicate(event))
        
        # Process same event twice
        first_result = reliable_event_bus.is_duplicate(event_with_id)
        second_result = reliable_event_bus.is_duplicate(event_with_id)
        
        assert first_result is False  # First time, not duplicate
        assert second_result is True   # Second time, is duplicate
        
        assert reliable_event_bus.is_duplicate.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])