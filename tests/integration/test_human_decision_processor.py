"""
Tests for Human Decision Processing Engine
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.integration.human_decision_processor import (
    HumanDecisionProcessor,
    TradeDecision,
    HumanDecisionResult,
    DecisionPriority,
    MockExecutionCallback
)
from src.core.event_bus import EventBus


@pytest.fixture
def event_bus():
    """Create test event bus."""
    return EventBus()


@pytest.fixture
async def decision_processor(event_bus):
    """Create test decision processor."""
    processor = HumanDecisionProcessor(
        redis_url="redis://localhost:6379",
        event_bus=event_bus
    )
    
    # Mock Redis for testing
    processor.redis_client = AsyncMock()
    processor.redis_client.setex.return_value = True
    processor.redis_client.get.return_value = None
    processor.redis_client.delete.return_value = True
    
    await processor.start()
    yield processor
    await processor.stop()


@pytest.fixture
def sample_trade_decision():
    """Create sample trade decision."""
    return TradeDecision(
        decision_id="test_decision_123",
        trade_id="trade_456",
        symbol="BTCUSD",
        direction="LONG",
        quantity=1.5,
        entry_price=45000.0,
        risk_score=0.65,
        priority=DecisionPriority.HIGH,
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        flagged_reason="High volatility detected",
        failure_probability=0.35,
        agent_recommendations=[
            {
                "agent_name": "risk_agent",
                "recommendation": "CAUTION",
                "confidence": 0.78,
                "reasoning": {"volatility": "elevated"}
            }
        ]
    )


class TestDecisionSubmission:
    """Test trade decision submission."""
    
    async def test_submit_for_review(self, decision_processor, sample_trade_decision):
        """Test submitting a trade for human review."""
        decision_id = await decision_processor.submit_for_review(
            sample_trade_decision,
            timeout_seconds=300
        )
        
        assert decision_id == sample_trade_decision.decision_id
        assert decision_id in decision_processor.pending_decisions
        
        pending_decision = decision_processor.pending_decisions[decision_id]
        assert pending_decision.trade_id == sample_trade_decision.trade_id
        assert pending_decision.symbol == sample_trade_decision.symbol
    
    async def test_submit_multiple_decisions(self, decision_processor):
        """Test submitting multiple decisions."""
        decisions = []
        for i in range(3):
            decision = TradeDecision(
                decision_id=f"decision_{i}",
                trade_id=f"trade_{i}",
                symbol="BTCUSD",
                direction="LONG", 
                quantity=1.0,
                entry_price=45000.0,
                risk_score=0.5,
                priority=DecisionPriority.MEDIUM,
                expires_at=datetime.utcnow() + timedelta(minutes=5),
                flagged_reason="Test decision",
                failure_probability=0.3
            )
            decisions.append(decision)
            await decision_processor.submit_for_review(decision)
        
        assert len(decision_processor.pending_decisions) == 3
        
        pending_decisions = await decision_processor.get_pending_decisions()
        assert len(pending_decisions) == 3


class TestDecisionProcessing:
    """Test human decision processing."""
    
    async def test_approve_decision(self, decision_processor, sample_trade_decision):
        """Test approving a trade decision."""
        # Add mock execution callback
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        # Submit decision for review
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        # Process approval
        result = await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Market conditions are favorable and risk is acceptable",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        assert result.decision == "APPROVE"
        assert result.trade_id == sample_trade_decision.trade_id
        assert result.user_id == "operator_123"
        assert result.execution_confirmed is True
        assert result.processing_time_ms > 0
        
        # Check that trade was executed
        assert len(mock_callback.executed_trades) == 1
        executed_trade = mock_callback.executed_trades[0]
        assert executed_trade["trade_id"] == sample_trade_decision.trade_id
        assert executed_trade["approved"] is True
        
        # Check that agents were notified
        assert len(mock_callback.notified_agents) == 1
        notification = mock_callback.notified_agents[0]
        assert notification["decision_id"] == decision_id
        assert notification["decision"] == "APPROVE"
        
        # Decision should be removed from pending
        assert decision_id not in decision_processor.pending_decisions
    
    async def test_reject_decision(self, decision_processor, sample_trade_decision):
        """Test rejecting a trade decision."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        result = await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="REJECT",
            reasoning="Risk levels are too high for current market conditions",
            user_id="manager_456",
            user_role="risk_manager"
        )
        
        assert result.decision == "REJECT"
        assert result.execution_confirmed is True
        
        # Check that trade was executed (rejection)
        executed_trade = mock_callback.executed_trades[0]
        assert executed_trade["approved"] is False
    
    async def test_invalid_decision_value(self, decision_processor, sample_trade_decision):
        """Test processing invalid decision value."""
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        with pytest.raises(ValueError, match="Decision must be APPROVE or REJECT"):
            await decision_processor.process_human_decision(
                decision_id=decision_id,
                decision="MAYBE",
                reasoning="Not sure about this one",
                user_id="operator_123",
                user_role="risk_operator"
            )
    
    async def test_nonexistent_decision(self, decision_processor):
        """Test processing decision that doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            await decision_processor.process_human_decision(
                decision_id="nonexistent_decision",
                decision="APPROVE",
                reasoning="This should fail",
                user_id="operator_123",
                user_role="risk_operator"
            )


class TestDecisionExpiration:
    """Test decision expiration handling."""
    
    async def test_expired_decision_rejection(self, decision_processor):
        """Test that expired decisions are automatically rejected."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        # Create decision that expires immediately
        expired_decision = TradeDecision(
            decision_id="expired_decision",
            trade_id="expired_trade",
            symbol="BTCUSD",
            direction="LONG",
            quantity=1.0,
            entry_price=45000.0,
            risk_score=0.5,
            priority=DecisionPriority.LOW,
            expires_at=datetime.utcnow() - timedelta(seconds=1),  # Already expired
            flagged_reason="Test expiration",
            failure_probability=0.3
        )
        
        decision_id = await decision_processor.submit_for_review(expired_decision)
        
        # Try to process expired decision
        with pytest.raises(ValueError, match="expired"):
            await decision_processor.process_human_decision(
                decision_id=decision_id,
                decision="APPROVE",
                reasoning="This should fail",
                user_id="operator_123",
                user_role="risk_operator"
            )
    
    async def test_automatic_expiration_cleanup(self, decision_processor):
        """Test automatic cleanup of expired decisions."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        # Create decision that expires quickly
        short_lived_decision = TradeDecision(
            decision_id="short_lived",
            trade_id="short_trade",
            symbol="BTCUSD",
            direction="LONG",
            quantity=1.0,
            entry_price=45000.0,
            risk_score=0.5,
            priority=DecisionPriority.LOW,
            expires_at=datetime.utcnow() + timedelta(milliseconds=100),
            flagged_reason="Test cleanup",
            failure_probability=0.3
        )
        
        decision_id = await decision_processor.submit_for_review(short_lived_decision)
        assert decision_id in decision_processor.pending_decisions
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Trigger cleanup manually (since background task may not run in test)
        await decision_processor._handle_expired_decision(short_lived_decision)
        
        # Decision should be removed from pending
        assert decision_id not in decision_processor.pending_decisions
        
        # Should have been auto-rejected
        assert len(mock_callback.executed_trades) == 1
        executed_trade = mock_callback.executed_trades[0]
        assert executed_trade["approved"] is False


class TestPerformanceMetrics:
    """Test performance metrics tracking."""
    
    async def test_metrics_update(self, decision_processor, sample_trade_decision):
        """Test that performance metrics are updated."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        initial_metrics = await decision_processor.get_performance_metrics()
        initial_count = initial_metrics["total_decisions"]
        
        # Process a decision
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Good trade opportunity",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        # Check metrics update
        updated_metrics = await decision_processor.get_performance_metrics()
        assert updated_metrics["total_decisions"] == initial_count + 1
        assert updated_metrics["average_processing_time_ms"] > 0
        assert updated_metrics["approval_rate"] > 0
    
    async def test_approval_rate_calculation(self, decision_processor):
        """Test approval rate calculation."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        # Process mix of approvals and rejections
        decisions = []
        for i in range(4):
            decision = TradeDecision(
                decision_id=f"decision_{i}",
                trade_id=f"trade_{i}",
                symbol="BTCUSD",
                direction="LONG",
                quantity=1.0,
                entry_price=45000.0,
                risk_score=0.5,
                priority=DecisionPriority.MEDIUM,
                expires_at=datetime.utcnow() + timedelta(minutes=5),
                flagged_reason="Test decision",
                failure_probability=0.3
            )
            decisions.append(decision)
            await decision_processor.submit_for_review(decision)
        
        # Approve 3, reject 1
        for i in range(3):
            await decision_processor.process_human_decision(
                decision_id=f"decision_{i}",
                decision="APPROVE",
                reasoning="Approved",
                user_id="operator_123",
                user_role="risk_operator"
            )
        
        await decision_processor.process_human_decision(
            decision_id="decision_3",
            decision="REJECT",
            reasoning="Rejected",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        metrics = await decision_processor.get_performance_metrics()
        assert metrics["approval_rate"] == 0.75  # 3/4 = 75%


class TestAuditTrail:
    """Test audit trail functionality."""
    
    async def test_audit_log_creation(self, decision_processor, sample_trade_decision):
        """Test that audit logs are created for all events."""
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        # Mock Redis to capture audit logs
        audit_logs = []
        
        async def mock_setex(key, ttl, value):
            if key.startswith("audit_log:"):
                audit_logs.append({"key": key, "value": value})
            return True
        
        decision_processor.redis_client.setex = mock_setex
        
        # Process decision
        await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Test audit trail",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        # Should have audit logs for submission and decision
        assert len(audit_logs) >= 2
        
        # Check that audit entries contain required fields
        for log in audit_logs:
            assert "audit_log:" in log["key"]
    
    async def test_decision_history(self, decision_processor, sample_trade_decision):
        """Test decision history tracking."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Test history tracking",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        history = await decision_processor.get_decision_history(limit=10)
        assert len(history) == 1
        
        decision_entry = history[0]
        assert decision_entry.decision_id == decision_id
        assert decision_entry.trade_id == sample_trade_decision.trade_id
        assert decision_entry.decision == "APPROVE"
        assert decision_entry.user_id == "operator_123"


class TestConcurrency:
    """Test concurrent decision processing."""
    
    async def test_concurrent_decision_processing(self, decision_processor):
        """Test processing multiple decisions concurrently."""
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        # Create multiple decisions
        decisions = []
        for i in range(5):
            decision = TradeDecision(
                decision_id=f"concurrent_{i}",
                trade_id=f"trade_{i}",
                symbol="BTCUSD",
                direction="LONG",
                quantity=1.0,
                entry_price=45000.0,
                risk_score=0.5,
                priority=DecisionPriority.MEDIUM,
                expires_at=datetime.utcnow() + timedelta(minutes=5),
                flagged_reason="Concurrent test",
                failure_probability=0.3
            )
            decisions.append(decision)
            await decision_processor.submit_for_review(decision)
        
        # Process decisions concurrently
        tasks = []
        for i, decision in enumerate(decisions):
            task = decision_processor.process_human_decision(
                decision_id=decision.decision_id,
                decision="APPROVE" if i % 2 == 0 else "REJECT",
                reasoning=f"Concurrent decision {i}",
                user_id=f"operator_{i}",
                user_role="risk_operator"
            )
            tasks.append(task)
        
        # Wait for all decisions to complete
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result.execution_confirmed for result in results)
        
        # All decisions should be processed
        assert len(decision_processor.pending_decisions) == 0
        assert len(mock_callback.executed_trades) == 5


class TestErrorHandling:
    """Test error handling scenarios."""
    
    async def test_execution_callback_failure(self, decision_processor, sample_trade_decision):
        """Test handling execution callback failures."""
        # Create callback that fails
        class FailingCallback:
            async def execute_trade(self, decision, approved):
                raise Exception("Execution failed")
            
            async def notify_agents(self, decision_result):
                raise Exception("Notification failed")
        
        failing_callback = FailingCallback()
        decision_processor.add_execution_callback(failing_callback)
        
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        
        # Should handle callback failure gracefully
        result = await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Test error handling",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        assert result.execution_confirmed is False  # Execution failed
    
    async def test_redis_failure_resilience(self, decision_processor, sample_trade_decision):
        """Test resilience to Redis failures."""
        # Make Redis operations fail
        decision_processor.redis_client.setex.side_effect = Exception("Redis failed")
        
        # Should still be able to submit and process decisions
        decision_id = await decision_processor.submit_for_review(sample_trade_decision)
        assert decision_id in decision_processor.pending_decisions
        
        mock_callback = MockExecutionCallback()
        decision_processor.add_execution_callback(mock_callback)
        
        result = await decision_processor.process_human_decision(
            decision_id=decision_id,
            decision="APPROVE",
            reasoning="Test Redis failure resilience",
            user_id="operator_123",
            user_role="risk_operator"
        )
        
        assert result.execution_confirmed is True
        assert len(mock_callback.executed_trades) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])