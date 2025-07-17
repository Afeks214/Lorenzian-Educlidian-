"""
Human Feedback API for Expert Trading Decisions

This module provides secure endpoints for capturing expert trader preferences
and integrating them into the MARL training process through RLHF.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import redis
import structlog
from pydantic import BaseModel, Field

from ..core.event_bus import EventBus, Event, EventType
from ..security.auth import SecurityManager

logger = structlog.get_logger()


class DecisionComplexity(Enum):
    """Classification of decision complexity requiring expert input"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StrategyType(Enum):
    """Types of trading strategies"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"


@dataclass
class MarketContext:
    """Current market conditions context"""
    symbol: str
    price: float
    volatility: float
    volume: float
    trend_strength: float
    support_level: float
    resistance_level: float
    time_of_day: str
    market_regime: str
    correlation_shock: bool = False


@dataclass
class TradingStrategy:
    """Represents a trading strategy option"""
    strategy_id: str
    strategy_type: StrategyType
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    time_horizon: int  # minutes
    risk_reward_ratio: float
    confidence_score: float
    reasoning: str
    expected_pnl: float
    max_drawdown: float


@dataclass
class DecisionPoint:
    """A complex trading decision requiring expert input"""
    decision_id: str
    timestamp: datetime
    context: MarketContext
    complexity: DecisionComplexity
    strategies: List[TradingStrategy]
    current_position: Optional[Dict[str, Any]]
    expert_deadline: datetime
    model_recommendation: str
    confidence_threshold: float


@dataclass
class ExpertChoice:
    """Expert's decision with full context"""
    decision_id: str
    chosen_strategy_id: str
    expert_id: str
    timestamp: datetime
    confidence: float
    reasoning: str
    alternative_considered: Optional[str]
    market_view: str
    risk_assessment: str


class ExpertCredentials(BaseModel):
    """Expert authentication credentials"""
    expert_id: str
    password: str


class StrategyComparison(BaseModel):
    """Strategy comparison for UI display"""
    strategy_a: TradingStrategy
    strategy_b: TradingStrategy
    context: MarketContext
    decision_id: str


class ExpertFeedback(BaseModel):
    """Expert feedback submission"""
    decision_id: str = Field(..., description="Unique decision identifier")
    chosen_strategy_id: str = Field(..., description="Selected strategy ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Expert confidence (0-1)")
    reasoning: str = Field(..., min_length=10, description="Detailed reasoning")
    market_view: str = Field(..., description="Overall market view")
    risk_assessment: str = Field(..., description="Risk assessment")


class FeedbackAPI:
    """Secure API for expert trading feedback collection"""

    def __init__(self, event_bus: EventBus, redis_client: redis.Redis):
        self.event_bus = event_bus
        self.redis_client = redis_client
        self.security_manager = SecurityManager()
        self.app = FastAPI(title="Expert Trading Feedback API", version="1.0.0")
        self.security = HTTPBearer()
        
        # Active WebSocket connections for real-time updates
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Decision storage (in production, use proper database)
        self.pending_decisions: Dict[str, DecisionPoint] = {}
        self.expert_choices: List[ExpertChoice] = []
        
        # Expert authentication storage
        self.expert_credentials: Dict[str, str] = {}  # expert_id -> hashed_password
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("Expert Feedback API initialized")

    def _setup_middleware(self):
        """Setup CORS and security middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://trading-dashboard.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/auth/login")
        async def login(credentials: ExpertCredentials):
            """Authenticate expert and return JWT token"""
            try:
                if not self._verify_expert_credentials(credentials.expert_id, credentials.password):
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                token = self._generate_jwt_token(credentials.expert_id)
                
                logger.info("Expert authenticated", expert_id=credentials.expert_id)
                return {"access_token": token, "token_type": "bearer"}
                
            except Exception as e:
                logger.error("Authentication failed", error=str(e))
                raise HTTPException(status_code=500, detail="Authentication failed")

        @self.app.get("/decisions/pending")
        async def get_pending_decisions(expert: dict = Depends(self._get_current_expert)):
            """Get all pending decisions requiring expert input"""
            try:
                decisions = []
                for decision in self.pending_decisions.values():
                    if decision.expert_deadline > datetime.now():
                        decisions.append({
                            "decision_id": decision.decision_id,
                            "timestamp": decision.timestamp.isoformat(),
                            "complexity": decision.complexity.value,
                            "symbol": decision.context.symbol,
                            "deadline": decision.expert_deadline.isoformat(),
                            "strategies_count": len(decision.strategies)
                        })
                
                return {"decisions": decisions, "count": len(decisions)}
                
            except Exception as e:
                logger.error("Failed to fetch pending decisions", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to fetch decisions")

        @self.app.get("/decisions/{decision_id}")
        async def get_decision_details(decision_id: str, expert: dict = Depends(self._get_current_expert)):
            """Get detailed decision information"""
            try:
                if decision_id not in self.pending_decisions:
                    raise HTTPException(status_code=404, detail="Decision not found")
                
                decision = self.pending_decisions[decision_id]
                
                # Check if decision is still valid
                if decision.expert_deadline < datetime.now():
                    raise HTTPException(status_code=410, detail="Decision deadline expired")
                
                return {
                    "decision": asdict(decision),
                    "time_remaining": (decision.expert_deadline - datetime.now()).total_seconds()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to fetch decision details", error=str(e), decision_id=decision_id)
                raise HTTPException(status_code=500, detail="Failed to fetch decision")

        @self.app.post("/decisions/{decision_id}/feedback")
        async def submit_feedback(
            decision_id: str, 
            feedback: ExpertFeedback,
            expert: dict = Depends(self._get_current_expert)
        ):
            """Submit expert feedback for a decision"""
            try:
                if decision_id not in self.pending_decisions:
                    raise HTTPException(status_code=404, detail="Decision not found")
                
                decision = self.pending_decisions[decision_id]
                
                # Validate strategy choice
                valid_strategy_ids = [s.strategy_id for s in decision.strategies]
                if feedback.chosen_strategy_id not in valid_strategy_ids:
                    raise HTTPException(status_code=400, detail="Invalid strategy selection")
                
                # Create expert choice record
                expert_choice = ExpertChoice(
                    decision_id=decision_id,
                    chosen_strategy_id=feedback.chosen_strategy_id,
                    expert_id=expert["expert_id"],
                    timestamp=datetime.now(),
                    confidence=feedback.confidence,
                    reasoning=feedback.reasoning,
                    market_view=feedback.market_view,
                    risk_assessment=feedback.risk_assessment,
                    alternative_considered=None  # Could be enhanced to capture this
                )
                
                # Store choice
                self.expert_choices.append(expert_choice)
                
                # Remove from pending decisions
                del self.pending_decisions[decision_id]
                
                # Notify RLHF training system
                await self._notify_rlhf_system(expert_choice, decision)
                
                # Broadcast to connected clients
                await self._broadcast_decision_update(decision_id, "completed")
                
                logger.info(
                    "Expert feedback received", 
                    decision_id=decision_id,
                    expert_id=expert["expert_id"],
                    chosen_strategy=feedback.chosen_strategy_id
                )
                
                return {"status": "success", "message": "Feedback recorded successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to submit feedback", error=str(e), decision_id=decision_id)
                raise HTTPException(status_code=500, detail="Failed to submit feedback")

        @self.app.get("/analytics/expert/{expert_id}")
        async def get_expert_analytics(expert_id: str, expert: dict = Depends(self._get_current_expert)):
            """Get analytics for expert decisions"""
            try:
                # Only allow experts to view their own analytics or admin access
                if expert["expert_id"] != expert_id and not expert.get("is_admin", False):
                    raise HTTPException(status_code=403, detail="Access denied")
                
                expert_decisions = [choice for choice in self.expert_choices if choice.expert_id == expert_id]
                
                if not expert_decisions:
                    return {"decisions_count": 0, "average_confidence": 0, "success_rate": 0}
                
                avg_confidence = sum(choice.confidence for choice in expert_decisions) / len(expert_decisions)
                
                # Calculate success rate (would need actual PnL data in production)
                success_rate = 0.75  # Mock value
                
                return {
                    "decisions_count": len(expert_decisions),
                    "average_confidence": round(avg_confidence, 3),
                    "success_rate": round(success_rate, 3),
                    "recent_decisions": [
                        {
                            "decision_id": choice.decision_id,
                            "timestamp": choice.timestamp.isoformat(),
                            "confidence": choice.confidence,
                            "strategy": choice.chosen_strategy_id
                        }
                        for choice in expert_decisions[-10:]  # Last 10 decisions
                    ]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to fetch expert analytics", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to fetch analytics")

        @self.app.websocket("/ws/{expert_id}")
        async def websocket_endpoint(websocket: WebSocket, expert_id: str):
            """WebSocket connection for real-time updates"""
            await websocket.accept()
            self.active_connections[expert_id] = websocket
            
            logger.info("WebSocket connection established", expert_id=expert_id)
            
            try:
                while True:
                    # Keep connection alive and handle any incoming messages
                    data = await websocket.receive_text()
                    # Echo back or handle specific commands
                    await websocket.send_text(f"Received: {data}")
                    
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed", expert_id=expert_id)
                if expert_id in self.active_connections:
                    del self.active_connections[expert_id]

    def _verify_expert_credentials(self, expert_id: str, password: str) -> bool:
        """Verify expert credentials against stored hashes"""
        if expert_id not in self.expert_credentials:
            return False
        
        stored_hash = self.expert_credentials[expert_id].encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)

    def _generate_jwt_token(self, expert_id: str) -> str:
        """Generate JWT token for authenticated expert"""
        payload = {
            "expert_id": expert_id,
            "exp": datetime.utcnow() + timedelta(hours=8),  # 8-hour expiry
            "iat": datetime.utcnow()
        }
        
        # In production, use proper secret management
        secret_key = self.security_manager.get_jwt_secret()
        return jwt.encode(payload, secret_key, algorithm="HS256")

    async def _get_current_expert(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Extract and validate expert from JWT token"""
        try:
            secret_key = self.security_manager.get_jwt_secret()
            payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
            expert_id = payload.get("expert_id")
            
            if expert_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            return {"expert_id": expert_id}
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def _notify_rlhf_system(self, expert_choice: ExpertChoice, decision: DecisionPoint):
        """Notify RLHF training system of new expert feedback"""
        event = self.event_bus.create_event(
            event_type=EventType.STRATEGIC_DECISION,  # Custom event type for RLHF
            payload={
                "type": "expert_feedback",
                "expert_choice": asdict(expert_choice),
                "decision_context": asdict(decision)
            },
            source="human_feedback_api"
        )
        
        self.event_bus.publish(event)
        logger.info("RLHF system notified", decision_id=expert_choice.decision_id)

    async def _broadcast_decision_update(self, decision_id: str, status: str):
        """Broadcast decision updates to connected WebSocket clients"""
        message = json.dumps({
            "type": "decision_update",
            "decision_id": decision_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        
        disconnected = []
        for expert_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to expert {expert_id}: {e}")
                disconnected.append(expert_id)
        
        # Clean up disconnected clients
        for expert_id in disconnected:
            del self.active_connections[expert_id]

    def add_expert(self, expert_id: str, password: str) -> bool:
        """Add a new expert to the system"""
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            self.expert_credentials[expert_id] = hashed_password.decode('utf-8')
            
            logger.info("Expert added to system", expert_id=expert_id)
            return True
            
        except Exception as e:
            logger.error("Failed to add expert", error=str(e))
            return False

    async def submit_decision_for_expert_input(self, decision: DecisionPoint) -> bool:
        """Submit a complex decision for expert input"""
        try:
            self.pending_decisions[decision.decision_id] = decision
            
            # Broadcast to connected experts
            await self._broadcast_new_decision(decision)
            
            logger.info(
                "Decision submitted for expert input",
                decision_id=decision.decision_id,
                complexity=decision.complexity.value,
                strategies_count=len(decision.strategies)
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to submit decision", error=str(e))
            return False

    async def _broadcast_new_decision(self, decision: DecisionPoint):
        """Broadcast new decision to all connected experts"""
        message = json.dumps({
            "type": "new_decision",
            "decision_id": decision.decision_id,
            "complexity": decision.complexity.value,
            "symbol": decision.context.symbol,
            "deadline": decision.expert_deadline.isoformat(),
            "strategies_count": len(decision.strategies)
        })
        
        disconnected = []
        for expert_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to expert {expert_id}: {e}")
                disconnected.append(expert_id)
        
        # Clean up disconnected clients
        for expert_id in disconnected:
            del self.active_connections[expert_id]

    def get_expert_choices_for_training(self, limit: Optional[int] = None) -> List[ExpertChoice]:
        """Get expert choices for RLHF training"""
        choices = self.expert_choices.copy()
        if limit:
            choices = choices[-limit:]
        return choices

    def get_api_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app