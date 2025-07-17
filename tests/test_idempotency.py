"""
Test Idempotency Functionality

Tests that duplicate events are properly handled and don't result in duplicate trades.
"""

import pytest
import asyncio
import time
import redis.asyncio as redis
from unittest.mock import Mock, patch
from datetime import datetime

from src.tactical.controller import TacticalMARLController, SynergyEvent


class TestIdempotency:
    """Test idempotency functionality."""
    
    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for testing."""
        client = redis.from_url("redis://localhost:6379/3")  # Use test DB
        await client.ping()
        
        # Clear test database
        await client.flushdb()
        
        yield client
        
        # Cleanup
        await client.flushdb()
        await client.close()
    
    @pytest.fixture
    async def controller(self, redis_client):
        """Create controller for testing."""
        controller = TacticalMARLController(redis_url="redis://localhost:6379/3")
        await controller.initialize()
        yield controller
        await controller.cleanup()
    
    @pytest.fixture
    def sample_synergy_event(self):
        """Create sample synergy event."""
        return SynergyEvent(
            synergy_type="TYPE_1",
            direction=1,
            confidence=0.75,
            signal_sequence=[],
            market_context={"test": True},
            correlation_id="test-correlation-123",
            timestamp=time.time()
        )
    
    @pytest.mark.asyncio
    async def test_duplicate_event_detection(self, controller, sample_synergy_event):
        """Test that duplicate events are detected."""
        # First event should not be duplicate
        is_duplicate = await controller._is_duplicate_event(sample_synergy_event)
        assert not is_duplicate
        
        # Mark event as processing
        await controller._mark_event_processing(sample_synergy_event)
        
        # Same event should now be detected as duplicate
        is_duplicate = await controller._is_duplicate_event(sample_synergy_event)
        assert is_duplicate
    
    @pytest.mark.asyncio
    async def test_completed_event_detection(self, controller, sample_synergy_event):
        """Test that completed events are detected."""
        # Create mock decision
        mock_decision = Mock()
        mock_decision.action = "long"
        mock_decision.confidence = 0.8
        mock_decision.execution_command = {"action": "long"}
        mock_decision.correlation_id = sample_synergy_event.correlation_id
        
        # Mark event as completed
        await controller._mark_event_completed(sample_synergy_event, mock_decision)
        
        # Event should be detected as duplicate (already completed)
        is_duplicate = await controller._is_duplicate_event(sample_synergy_event)
        assert is_duplicate
    
    @pytest.mark.asyncio
    async def test_duplicate_event_response(self, controller, sample_synergy_event):
        """Test response to duplicate events."""
        # Create duplicate event response
        response = controller._create_duplicate_event_response(sample_synergy_event)
        
        assert response.action == "hold"
        assert response.confidence == 0.0
        assert response.correlation_id == sample_synergy_event.correlation_id
        assert "duplicate_event" in response.consensus_breakdown
    
    @pytest.mark.asyncio
    async def test_event_processing_lifecycle(self, controller, sample_synergy_event):
        """Test full event processing lifecycle."""
        # Initially not duplicate
        assert not await controller._is_duplicate_event(sample_synergy_event)
        
        # Mark as processing
        await controller._mark_event_processing(sample_synergy_event)
        
        # Should be duplicate while processing
        assert await controller._is_duplicate_event(sample_synergy_event)
        
        # Create mock decision
        mock_decision = Mock()
        mock_decision.action = "long"
        mock_decision.confidence = 0.8
        mock_decision.execution_command = {"action": "long"}
        mock_decision.correlation_id = sample_synergy_event.correlation_id
        
        # Mark as completed
        await controller._mark_event_completed(sample_synergy_event, mock_decision)
        
        # Should still be duplicate (now completed)
        assert await controller._is_duplicate_event(sample_synergy_event)
    
    @pytest.mark.asyncio
    async def test_event_without_correlation_id(self, controller):
        """Test event without correlation ID."""
        event_without_id = SynergyEvent(
            synergy_type="TYPE_1",
            direction=1,
            confidence=0.75,
            signal_sequence=[],
            market_context={"test": True},
            correlation_id="",  # Empty correlation ID
            timestamp=time.time()
        )
        
        # Should not be considered duplicate
        assert not await controller._is_duplicate_event(event_without_id)
        
        # Should handle marking gracefully
        await controller._mark_event_processing(event_without_id)
        assert not await controller._is_duplicate_event(event_without_id)
    
    @pytest.mark.asyncio
    async def test_redis_failure_handling(self, controller, sample_synergy_event):
        """Test handling of Redis failures."""
        # Mock Redis client to raise exception
        with patch.object(controller.redis_client, 'exists', side_effect=Exception("Redis error")):
            # Should assume not duplicate on error
            is_duplicate = await controller._is_duplicate_event(sample_synergy_event)
            assert not is_duplicate
    
    @pytest.mark.asyncio
    async def test_event_ttl_expiration(self, controller, sample_synergy_event, redis_client):
        """Test that events expire after TTL."""
        # Mark event as processing
        await controller._mark_event_processing(sample_synergy_event)
        
        # Should be duplicate immediately
        assert await controller._is_duplicate_event(sample_synergy_event)
        
        # Check TTL is set
        processing_key = f"tactical:processing:{sample_synergy_event.correlation_id}"
        ttl = await redis_client.ttl(processing_key)
        assert ttl > 0
        assert ttl <= 300  # 5 minutes
    
    @pytest.mark.asyncio
    async def test_multiple_events_isolation(self, controller):
        """Test that multiple events are isolated."""
        event1 = SynergyEvent(
            synergy_type="TYPE_1",
            direction=1,
            confidence=0.75,
            signal_sequence=[],
            market_context={"test": True},
            correlation_id="test-correlation-1",
            timestamp=time.time()
        )
        
        event2 = SynergyEvent(
            synergy_type="TYPE_2",
            direction=-1,
            confidence=0.65,
            signal_sequence=[],
            market_context={"test": True},
            correlation_id="test-correlation-2",
            timestamp=time.time()
        )
        
        # Both should initially not be duplicates
        assert not await controller._is_duplicate_event(event1)
        assert not await controller._is_duplicate_event(event2)
        
        # Mark event1 as processing
        await controller._mark_event_processing(event1)
        
        # Only event1 should be duplicate
        assert await controller._is_duplicate_event(event1)
        assert not await controller._is_duplicate_event(event2)
    
    @pytest.mark.asyncio
    async def test_event_details_storage(self, controller, sample_synergy_event, redis_client):
        """Test that event details are stored correctly."""
        await controller._mark_event_processing(sample_synergy_event)
        
        # Check event details are stored
        details_key = f"tactical:event_details:{sample_synergy_event.correlation_id}"
        details_raw = await redis_client.get(details_key)
        assert details_raw is not None
        
        import json
        details = json.loads(details_raw)
        assert details["synergy_type"] == sample_synergy_event.synergy_type
        assert details["direction"] == sample_synergy_event.direction
        assert details["confidence"] == sample_synergy_event.confidence
        assert "processing_start" in details
    
    @pytest.mark.asyncio
    async def test_completion_data_storage(self, controller, sample_synergy_event, redis_client):
        """Test that completion data is stored correctly."""
        # Create mock decision
        mock_decision = Mock()
        mock_decision.action = "long"
        mock_decision.confidence = 0.8
        mock_decision.execution_command = {"action": "long", "size": 0.1}
        mock_decision.correlation_id = sample_synergy_event.correlation_id
        
        await controller._mark_event_completed(sample_synergy_event, mock_decision)
        
        # Check completion data is stored
        completed_key = f"tactical:completed:{sample_synergy_event.correlation_id}"
        completion_raw = await redis_client.get(completed_key)
        assert completion_raw is not None
        
        import json
        completion_data = json.loads(completion_raw)
        assert completion_data["action"] == "long"
        assert completion_data["confidence"] == 0.8
        assert completion_data["correlation_id"] == sample_synergy_event.correlation_id
        assert "completion_time" in completion_data
    
    @pytest.mark.asyncio
    async def test_processing_key_cleanup(self, controller, sample_synergy_event, redis_client):
        """Test that processing keys are cleaned up on completion."""
        # Mark as processing
        await controller._mark_event_processing(sample_synergy_event)
        
        processing_key = f"tactical:processing:{sample_synergy_event.correlation_id}"
        assert await redis_client.exists(processing_key)
        
        # Mark as completed
        mock_decision = Mock()
        mock_decision.action = "hold"
        mock_decision.confidence = 0.5
        mock_decision.execution_command = {"action": "hold"}
        mock_decision.correlation_id = sample_synergy_event.correlation_id
        
        await controller._mark_event_completed(sample_synergy_event, mock_decision)
        
        # Processing key should be removed
        assert not await redis_client.exists(processing_key)
        
        # But completed key should exist
        completed_key = f"tactical:completed:{sample_synergy_event.correlation_id}"
        assert await redis_client.exists(completed_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])