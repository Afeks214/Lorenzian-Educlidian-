"""
Human Decision Processing Engine
Handles APPROVE/REJECT decisions with immediate system integration and complete audit trails.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import redis.asyncio as redis

from src.monitoring.logger_config import get_logger
from src.core.event_bus import EventBus, Event

logger = get_logger(__name__)

class DecisionStatus(str, Enum):
    """Decision processing status."""
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    EXPIRED = "expired"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DecisionPriority(str, Enum):
    """Decision priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradeDecision(BaseModel):
    """Trade decision model."""
    decision_id: str = Field(..., description="Unique decision identifier")
    trade_id: str = Field(..., description="Trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Trade direction")
    quantity: float = Field(..., description="Trade quantity")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    risk_score: float = Field(..., description="Risk score (0-1)")
    priority: DecisionPriority = Field(..., description="Decision priority")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Decision expiration")
    flagged_reason: str = Field(..., description="Reason for human review")
    agent_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    failure_probability: float = Field(..., description="Monte Carlo failure probability")

class HumanDecisionResult(BaseModel):
    """Human decision result."""
    decision_id: str = Field(..., description="Decision identifier")
    trade_id: str = Field(..., description="Trade identifier")
    decision: str = Field(..., description="APPROVE or REJECT")
    reasoning: str = Field(..., description="Decision reasoning")
    user_id: str = Field(..., description="Decision maker")
    user_role: str = Field(..., description="User role")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time")
    execution_confirmed: bool = Field(default=False, description="Execution confirmed")

class DecisionAuditLog(BaseModel):
    """Comprehensive audit log entry."""
    audit_id: str = Field(..., description="Audit entry ID")
    decision_id: str = Field(..., description="Decision ID")
    trade_id: str = Field(..., description="Trade ID")
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(..., description="Event data")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    system_state: Dict[str, Any] = Field(default_factory=dict, description="System state snapshot")

class ExecutionCallback:
    """Callback interface for trade execution."""
    
    async def execute_trade(self, decision: TradeDecision, approved: bool) -> Dict[str, Any]:
        """Execute or cancel trade based on decision."""
        raise NotImplementedError
    
    async def notify_agents(self, decision_result: HumanDecisionResult) -> bool:
        """Notify all agents of the decision."""
        raise NotImplementedError

class HumanDecisionProcessor:
    """
    Human Decision Processing Engine.
    Manages the complete lifecycle of human decisions with immediate system integration.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        event_bus: Optional[EventBus] = None
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.event_bus = event_bus or EventBus()
        
        # Decision storage
        self.pending_decisions: Dict[str, TradeDecision] = {}
        self.decision_history: List[HumanDecisionResult] = []
        
        # Execution callbacks
        self.execution_callbacks: List[ExecutionCallback] = []
        
        # Performance metrics
        self.decision_count = 0
        self.average_processing_time = 0.0
        self.approval_rate = 0.0
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the decision processor."""
        # Initialize Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        
        # Start background cleanup task
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_decisions())
        
        logger.info("Human Decision Processor started")
    
    async def stop(self) -> None:
        """Stop the decision processor."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Human Decision Processor stopped")
    
    def add_execution_callback(self, callback: ExecutionCallback) -> None:
        """Add execution callback."""
        self.execution_callbacks.append(callback)
        logger.info(f"Added execution callback: {callback.__class__.__name__}")
    
    async def submit_for_review(
        self,
        trade_decision: TradeDecision,
        timeout_seconds: int = 300
    ) -> str:
        """
        Submit a trade for human review.
        Returns the decision ID for tracking.
        """
        # Set expiration time
        trade_decision.expires_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
        
        # Store in memory and Redis
        self.pending_decisions[trade_decision.decision_id] = trade_decision
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"pending_decision:{trade_decision.decision_id}",
                    timeout_seconds,
                    trade_decision.json()
                )
            except Exception as e:
                logger.error(f"Failed to store decision in Redis: {e}")
        
        # Create audit log
        await self._create_audit_log(
            trade_decision.decision_id,
            trade_decision.trade_id,
            "DECISION_SUBMITTED",
            {
                "trade_data": trade_decision.dict(),
                "timeout_seconds": timeout_seconds
            }
        )
        
        # Publish event for dashboard
        await self.event_bus.publish(Event(
            type="TRADE_FLAGGED_FOR_REVIEW",
            data={
                "decision_id": trade_decision.decision_id,
                "trade_id": trade_decision.trade_id,
                "symbol": trade_decision.symbol,
                "priority": trade_decision.priority.value,
                "risk_score": trade_decision.risk_score,
                "expires_at": trade_decision.expires_at.isoformat()
            }
        ))
        
        logger.info(
            f"Trade submitted for review: {trade_decision.trade_id} "
            f"(decision_id: {trade_decision.decision_id})"
        )
        
        return trade_decision.decision_id
    
    async def process_human_decision(
        self,
        decision_id: str,
        decision: str,
        reasoning: str,
        user_id: str,
        user_role: str
    ) -> HumanDecisionResult:
        """
        Process a human decision with immediate execution.
        """
        start_time = time.time()
        
        # Validate decision
        if decision not in ["APPROVE", "REJECT"]:
            raise ValueError("Decision must be APPROVE or REJECT")
        
        # Get pending decision
        trade_decision = self.pending_decisions.get(decision_id)
        if not trade_decision:
            # Try to load from Redis
            if self.redis_client:
                try:
                    decision_data = await self.redis_client.get(f"pending_decision:{decision_id}")
                    if decision_data:
                        trade_decision = TradeDecision.parse_raw(decision_data)
                except Exception as e:
                    logger.error(f"Failed to load decision from Redis: {e}")
        
        if not trade_decision:
            raise ValueError(f"Decision {decision_id} not found or expired")
        
        # Check if decision has expired
        if datetime.utcnow() > trade_decision.expires_at:
            await self._handle_expired_decision(trade_decision)
            raise ValueError(f"Decision {decision_id} has expired")
        
        # Create decision result
        processing_time_ms = (time.time() - start_time) * 1000
        
        decision_result = HumanDecisionResult(
            decision_id=decision_id,
            trade_id=trade_decision.trade_id,
            decision=decision,
            reasoning=reasoning,
            user_id=user_id,
            user_role=user_role,
            processing_time_ms=processing_time_ms
        )
        
        # Create audit log for decision
        await self._create_audit_log(
            decision_id,
            trade_decision.trade_id,
            "HUMAN_DECISION_MADE",
            {
                "decision": decision,
                "reasoning": reasoning,
                "user_id": user_id,
                "user_role": user_role,
                "processing_time_ms": processing_time_ms
            },
            user_id=user_id
        )
        
        # Execute decision immediately
        execution_success = await self._execute_decision(trade_decision, decision_result)
        decision_result.execution_confirmed = execution_success
        
        if not execution_success:
            logger.error(f"Failed to execute decision {decision_id}")
            await self._create_audit_log(
                decision_id,
                trade_decision.trade_id,
                "EXECUTION_FAILED",
                {"decision": decision, "user_id": user_id},
                user_id=user_id
            )
        
        # Remove from pending decisions
        if decision_id in self.pending_decisions:
            del self.pending_decisions[decision_id]
        
        if self.redis_client:
            try:
                await self.redis_client.delete(f"pending_decision:{decision_id}")
            except Exception as e:
                logger.error(f"Failed to remove decision from Redis: {e}")
        
        # Store decision result
        self.decision_history.append(decision_result)
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"decision_result:{decision_id}",
                    86400 * 30,  # 30 days retention
                    decision_result.json()
                )
            except Exception as e:
                logger.error(f"Failed to store decision result: {e}")
        
        # Update performance metrics
        self._update_performance_metrics(decision_result)
        
        # Publish completion event
        await self.event_bus.publish(Event(
            type="HUMAN_DECISION_COMPLETED",
            data={
                "decision_id": decision_id,
                "trade_id": trade_decision.trade_id,
                "decision": decision,
                "user_id": user_id,
                "execution_confirmed": execution_success,
                "processing_time_ms": processing_time_ms
            }
        ))
        
        logger.info(
            f"Human decision processed: {decision_id} -> {decision} "
            f"by {user_id} (execution: {'success' if execution_success else 'failed'})"
        )
        
        return decision_result
    
    async def _execute_decision(
        self,
        trade_decision: TradeDecision,
        decision_result: HumanDecisionResult
    ) -> bool:
        """Execute the decision through callbacks."""
        approved = decision_result.decision == "APPROVE"
        execution_results = []
        
        # Execute through all callbacks
        for callback in self.execution_callbacks:
            try:
                result = await callback.execute_trade(trade_decision, approved)
                execution_results.append(result)
                
                # Notify agents of the decision
                await callback.notify_agents(decision_result)
                
            except Exception as e:
                logger.error(f"Execution callback failed: {e}")
                execution_results.append({"success": False, "error": str(e)})
        
        # Log execution results
        await self._create_audit_log(
            decision_result.decision_id,
            trade_decision.trade_id,
            "EXECUTION_ATTEMPT",
            {
                "approved": approved,
                "execution_results": execution_results
            },
            user_id=decision_result.user_id
        )
        
        # Return success if any callback succeeded
        return any(result.get("success", False) for result in execution_results)
    
    async def _handle_expired_decision(self, trade_decision: TradeDecision) -> None:
        """Handle expired decision - automatically reject."""
        logger.warning(f"Decision expired: {trade_decision.decision_id}")
        
        # Create automatic rejection
        decision_result = HumanDecisionResult(
            decision_id=trade_decision.decision_id,
            trade_id=trade_decision.trade_id,
            decision="REJECT",
            reasoning="Decision expired - automatic rejection",
            user_id="system",
            user_role="system",
            processing_time_ms=0.0,
            execution_confirmed=True  # Auto-rejection is always confirmed
        )
        
        # Execute rejection
        await self._execute_decision(trade_decision, decision_result)
        
        # Create audit log
        await self._create_audit_log(
            trade_decision.decision_id,
            trade_decision.trade_id,
            "DECISION_EXPIRED",
            {"auto_rejected": True}
        )
        
        # Remove from pending
        if trade_decision.decision_id in self.pending_decisions:
            del self.pending_decisions[trade_decision.decision_id]
    
    async def _cleanup_expired_decisions(self) -> None:
        """Background task to clean up expired decisions."""
        while self._running:
            try:
                now = datetime.utcnow()
                expired_decisions = []
                
                for decision_id, trade_decision in self.pending_decisions.items():
                    if now > trade_decision.expires_at:
                        expired_decisions.append(trade_decision)
                
                # Handle expired decisions
                for trade_decision in expired_decisions:
                    await self._handle_expired_decision(trade_decision)
                
                # Sleep for 5 seconds before next check
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(10)
    
    async def _create_audit_log(
        self,
        decision_id: str,
        trade_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """Create comprehensive audit log entry."""
        audit_id = f"audit_{int(time.time() * 1000000)}"
        
        # Capture system state
        system_state = {
            "pending_decisions_count": len(self.pending_decisions),
            "total_decisions_processed": self.decision_count,
            "average_processing_time_ms": self.average_processing_time,
            "approval_rate": self.approval_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        audit_entry = DecisionAuditLog(
            audit_id=audit_id,
            decision_id=decision_id,
            trade_id=trade_id,
            event_type=event_type,
            event_data=event_data,
            user_id=user_id,
            system_state=system_state
        )
        
        # Store in Redis with 1 year retention
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"audit_log:{audit_id}",
                    365 * 24 * 3600,
                    audit_entry.json()
                )
            except Exception as e:
                logger.error(f"Failed to store audit log: {e}")
        
        # Always log to application logger
        logger.info(f"AUDIT [{event_type}]: {audit_entry.dict()}")
    
    def _update_performance_metrics(self, decision_result: HumanDecisionResult) -> None:
        """Update performance metrics."""
        self.decision_count += 1
        
        # Update average processing time
        total_time = self.average_processing_time * (self.decision_count - 1)
        total_time += decision_result.processing_time_ms
        self.average_processing_time = total_time / self.decision_count
        
        # Update approval rate
        approvals = sum(1 for d in self.decision_history if d.decision == "APPROVE")
        self.approval_rate = approvals / len(self.decision_history) if self.decision_history else 0.0
    
    async def get_pending_decisions(self) -> List[TradeDecision]:
        """Get all pending decisions."""
        return list(self.pending_decisions.values())
    
    async def get_decision_history(
        self,
        limit: int = 100,
        user_id: Optional[str] = None
    ) -> List[HumanDecisionResult]:
        """Get decision history with optional user filter."""
        history = self.decision_history
        
        if user_id:
            history = [d for d in history if d.user_id == user_id]
        
        return history[-limit:] if limit else history
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get decision processing performance metrics."""
        return {
            "total_decisions": self.decision_count,
            "pending_decisions": len(self.pending_decisions),
            "average_processing_time_ms": self.average_processing_time,
            "approval_rate": self.approval_rate,
            "decision_history_size": len(self.decision_history)
        }

# Mock execution callback for testing
class MockExecutionCallback(ExecutionCallback):
    """Mock execution callback for testing."""
    
    def __init__(self):
        self.executed_trades = []
        self.notified_agents = []
    
    async def execute_trade(self, decision: TradeDecision, approved: bool) -> Dict[str, Any]:
        """Mock trade execution."""
        self.executed_trades.append({
            "trade_id": decision.trade_id,
            "approved": approved,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate execution latency
        await asyncio.sleep(0.01)  # 10ms
        
        return {"success": True, "execution_id": f"exec_{int(time.time())}"}
    
    async def notify_agents(self, decision_result: HumanDecisionResult) -> bool:
        """Mock agent notification."""
        self.notified_agents.append({
            "decision_id": decision_result.decision_id,
            "decision": decision_result.decision,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True