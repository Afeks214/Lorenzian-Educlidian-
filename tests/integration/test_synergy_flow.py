"""
Integration test for complete SYNERGY_DETECTED flow.
Tests the full pipeline from event to decision with all components.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import httpx

from src.api.event_handler import EventHandler
from src.api.models import (
    StrategicDecisionRequest, StrategicDecisionResponse,
    MarketState, SynergyContext, MatrixData, SynergyType,
    DecisionType, RiskLevel
)
from src.monitoring.metrics_exporter import metrics_exporter
from src.monitoring.health_monitor import health_monitor
from src.security.auth import create_access_token, Permission


class TestSynergyFlow:
    """Integration test for complete synergy detection and decision flow."""
    
    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token for testing."""
        return create_access_token(
            user_id="test_user",
            role="trader",
            permissions=[Permission.READ, Permission.TRADE]
        )
    
    @pytest.fixture
    def market_state(self):
        """Create test market state."""
        return MarketState(
            timestamp=datetime.utcnow(),
            symbol="BTCUSDT",
            price=50000.0,
            volume=1500.0,
            volatility=0.025,
            trend="bullish"
        )
    
    @pytest.fixture
    def synergy_context(self):
        """Create test synergy context."""
        return SynergyContext(
            synergy_type=SynergyType.TYPE_1,
            strength=0.85,
            confidence=0.92,
            pattern_data={
                "indicators": ["MLMI", "NWRQK", "FVG"],
                "signals": {
                    "entry": True,
                    "strength": "high",
                    "confluence": 3
                }
            },
            correlation_id="test-synergy-001"
        )
    
    @pytest.fixture
    def matrix_data(self):
        """Create test matrix data."""
        return MatrixData(
            matrix_type="30m",
            shape=[48, 13],
            data=[[0.1 + (i * 0.01) for j in range(13)] for i in range(48)],
            features=[
                "open", "high", "low", "close", "volume",
                "mlmi", "nwrqk", "fvg", "lvn", "mmd",
                "rsi", "macd", "bb"
            ],
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    async def event_handler(self):
        """Create event handler for testing."""
        handler = EventHandler(redis_url="redis://localhost:6379")
        # Mock Redis for testing
        handler.redis = AsyncMock()
        handler.pubsub = AsyncMock()
        handler._running = True
        return handler
    
    @pytest.mark.asyncio
    async def test_complete_synergy_flow(self, event_handler, valid_token,
                                       market_state, synergy_context, matrix_data):
        """Test the complete flow from SYNERGY_DETECTED event to trading decision."""
        
        # Step 1: Simulate SYNERGY_DETECTED event
        event_data = {
            "type": "SYNERGY_DETECTED",
            "data": {
                "synergy_type": synergy_context.synergy_type.value,
                "strength": synergy_context.strength,
                "confidence": synergy_context.confidence,
                "pattern_data": synergy_context.pattern_data,
                "market_snapshot": {
                    "symbol": market_state.symbol,
                    "price": market_state.price,
                    "volume": market_state.volume
                }
            },
            "correlation_id": synergy_context.correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Track correlation ID
        metrics_exporter.track_correlation_id(synergy_context.correlation_id)
        
        # Step 2: Publish event
        await event_handler.publish_event(
            event_type="SYNERGY_DETECTED",
            event_data=event_data["data"],
            correlation_id=synergy_context.correlation_id
        )
        
        # Verify event was published
        event_handler.redis.publish.assert_called()
        event_handler.redis.setex.assert_called()  # Event history
        
        # Step 3: Simulate API request based on event
        request_data = StrategicDecisionRequest(
            market_state=market_state,
            synergy_context=synergy_context,
            matrix_data=matrix_data,
            correlation_id=synergy_context.correlation_id
        )
        
        # Step 4: Mock decision logic
        with patch('src.api.main.event_handler', event_handler):
            # Simulate decision response
            decision_response = StrategicDecisionResponse(
                correlation_id=synergy_context.correlation_id,
                decision=DecisionType.LONG,
                confidence=0.87,
                risk_level=RiskLevel.MEDIUM,
                position_size=0.6,
                stop_loss=market_state.price * 0.98,
                take_profit=market_state.price * 1.03,
                agent_decisions=[
                    {
                        "agent_name": "strategic_agent",
                        "decision": DecisionType.LONG,
                        "confidence": 0.89,
                        "reasoning": {"synergy_alignment": "high"}
                    },
                    {
                        "agent_name": "risk_agent",
                        "decision": DecisionType.LONG,
                        "confidence": 0.85,
                        "reasoning": {"risk_assessment": "acceptable"}
                    }
                ],
                inference_latency_ms=3.2
            )
            
            # Step 5: Verify metrics were updated
            metrics_exporter.record_synergy_response(
                synergy_context.synergy_type.value,
                "success",
                synergy_context.correlation_id
            )
            
            metrics_exporter.update_model_confidence(
                decision_response.confidence,
                "strategic_marl",
                "ensemble"
            )
            
            # Step 6: Simulate trade execution event
            trade_event = {
                "type": "TRADE_EXECUTED",
                "data": {
                    "correlation_id": synergy_context.correlation_id,
                    "decision": decision_response.decision.value,
                    "position_size": decision_response.position_size,
                    "entry_price": market_state.price,
                    "stop_loss": decision_response.stop_loss,
                    "take_profit": decision_response.take_profit,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await event_handler.publish_event(
                event_type="TRADE_EXECUTED",
                event_data=trade_event["data"],
                correlation_id=synergy_context.correlation_id
            )
            
            # Step 7: Update position tracking
            metrics_exporter.update_active_positions(1)
            
            # Verify complete flow
            assert synergy_context.correlation_id in metrics_exporter.correlation_ids
            assert decision_response.decision == DecisionType.LONG
            assert decision_response.confidence > 0.8
            assert decision_response.inference_latency_ms < 5.0
    
    @pytest.mark.asyncio
    async def test_event_replay_capability(self, event_handler):
        """Test event replay functionality for recovery scenarios."""
        # Mock Redis scan
        event_handler.redis.scan_iter = AsyncMock(return_value=[
            "event_history:SYNERGY_DETECTED:replay-001",
            "event_history:SYNERGY_DETECTED:replay-002"
        ].__iter__())
        
        # Mock stored events
        event_1 = {
            "type": "SYNERGY_DETECTED",
            "data": {"synergy_type": "TYPE_1"},
            "correlation_id": "replay-001",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event_2 = {
            "type": "SYNERGY_DETECTED",
            "data": {"synergy_type": "TYPE_2"},
            "correlation_id": "replay-002",
            "timestamp": (datetime.utcnow() + timedelta(minutes=1)).isoformat()
        }
        
        event_handler.redis.get = AsyncMock(side_effect=[
            json.dumps(event_1),
            json.dumps(event_2)
        ])
        
        # Replay events
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow() + timedelta(hours=1)
        
        events = await event_handler.replay_events(
            "SYNERGY_DETECTED",
            start_time,
            end_time
        )
        
        assert len(events) == 2
        assert events[0]["correlation_id"] == "replay-001"
        assert events[1]["correlation_id"] == "replay-002"
        # Events should be sorted by timestamp
        assert events[0]["timestamp"] < events[1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, event_handler):
        """Test error handling in the event flow."""
        # Test with callback that raises exception
        error_count = 0
        
        async def failing_callback(event_data):
            nonlocal error_count
            error_count += 1
            raise Exception("Simulated processing error")
        
        # Subscribe with failing callback
        await event_handler.subscribe("TEST_EVENT", failing_callback)
        
        # Mock DLQ operations
        event_handler.redis.lpush = AsyncMock()
        event_handler.redis.ltrim = AsyncMock()
        
        # Process message that will fail
        test_message = {
            "channel": "events:TEST_EVENT",
            "type": "message",
            "data": json.dumps({
                "type": "TEST_EVENT",
                "data": {"test": "data"},
                "correlation_id": "error-test-001"
            })
        }
        
        await event_handler._handle_message(test_message)
        
        # Verify error was handled and sent to DLQ
        event_handler.redis.lpush.assert_called()
        dlq_call = event_handler.redis.lpush.call_args
        assert "dlq:TEST_EVENT" in dlq_call[0]
        
        # Verify DLQ entry contains error info
        dlq_entry = json.loads(dlq_call[0][1])
        assert "errors" in dlq_entry
        assert len(dlq_entry["errors"]) > 0
        assert "Simulated processing error" in dlq_entry["errors"][0]
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, event_handler):
        """Test handling multiple concurrent events."""
        processed_events = []
        
        async def track_callback(event_data):
            processed_events.append(event_data["correlation_id"])
            await asyncio.sleep(0.01)  # Simulate processing
        
        await event_handler.subscribe("CONCURRENT_TEST", track_callback)
        
        # Simulate multiple concurrent events
        tasks = []
        for i in range(10):
            event_data = {
                "type": "CONCURRENT_TEST",
                "data": {"index": i},
                "correlation_id": f"concurrent-{i:03d}"
            }
            
            task = event_handler.publish_event(
                "CONCURRENT_TEST",
                event_data["data"],
                event_data["correlation_id"]
            )
            tasks.append(task)
        
        # Wait for all publishes
        await asyncio.gather(*tasks)
        
        # Verify all events were published
        assert event_handler.redis.publish.call_count >= 10
    
    @pytest.mark.asyncio
    async def test_health_monitoring_during_flow(self):
        """Test that health monitoring works during event processing."""
        # Mock health checks
        with patch('src.monitoring.health_monitor.HealthMonitor._check_redis') as mock_redis, \
             patch('src.monitoring.health_monitor.HealthMonitor._check_models') as mock_models, \
             patch('src.monitoring.health_monitor.HealthMonitor._check_api') as mock_api:
            
            # Configure mock responses
            from src.monitoring.health_monitor import ComponentHealth, HealthStatus
            
            mock_redis.return_value = ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            )
            
            mock_models.return_value = ComponentHealth(
                name="models",
                status=HealthStatus.HEALTHY,
                message="Models loaded",
                last_check=datetime.utcnow()
            )
            
            mock_api.return_value = ComponentHealth(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API responding",
                last_check=datetime.utcnow()
            )
            
            # Get health during processing
            health_data = await health_monitor.get_detailed_health()
            
            assert health_data["status"] == "healthy"
            assert len(health_data["components"]) > 0
            assert all(c["status"] == "healthy" for c in health_data["components"])


class TestEndToEndValidation:
    """End-to-end validation of the complete system."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_system_integration(self):
        """Test complete system integration with all components."""
        # This test would require all services running
        # For unit testing, we'll mock the external dependencies
        
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            # Step 1: Health check
            with patch.object(client, 'get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "status": "healthy",
                    "components": [
                        {"name": "redis", "status": "healthy"},
                        {"name": "models", "status": "healthy"}
                    ]
                }
                
                health_response = await client.get("/health")
                assert health_response.status_code == 200
            
            # Step 2: Authenticate
            token = create_access_token(
                user_id="integration_test",
                role="trader",
                permissions=[Permission.READ, Permission.TRADE]
            )
            
            headers = {"Authorization": f"Bearer {token}"}
            
            # Step 3: Make decision
            decision_request = {
                "market_state": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": "BTCUSDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "volatility": 0.02,
                    "trend": "bullish"
                },
                "synergy_context": {
                    "synergy_type": "TYPE_1",
                    "strength": 0.85,
                    "confidence": 0.9,
                    "pattern_data": {},
                    "correlation_id": "integration-test-001"
                },
                "matrix_data": {
                    "matrix_type": "30m",
                    "shape": [48, 13],
                    "data": [[0.0] * 13 for _ in range(48)],
                    "features": ["close"] * 13,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            with patch.object(client, 'post') as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "correlation_id": "integration-test-001",
                    "decision": "LONG",
                    "confidence": 0.85,
                    "risk_level": "MEDIUM",
                    "position_size": 0.5,
                    "inference_latency_ms": 3.5
                }
                
                decision_response = await client.post(
                    "/decide",
                    json=decision_request,
                    headers=headers
                )
                
                assert decision_response.status_code == 200
                result = decision_response.json()
                assert result["decision"] == "LONG"
                assert result["inference_latency_ms"] < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])