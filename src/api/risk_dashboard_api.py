"""
Human-in-the-Loop Risk Dashboard API
Provides real-time risk data exposure with WebSocket streaming for the validation dashboard.
"""

import os
import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.authentication import verify_token, UserInfo, RolePermission
from src.monitoring.logger_config import get_logger
from src.risk.agents.real_time_risk_assessor import RealTimeRiskAssessor
from src.risk.core.var_calculator import VaRCalculator
from src.risk.core.correlation_tracker import CorrelationTracker

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_permissions: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, permissions: List[str]):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_permissions[user_id] = permissions
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            del self.user_permissions[user_id]
            logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast(self, message: Dict[str, Any], required_permission: str = None):
        """Broadcast a message to all connected users with the required permission."""
        disconnected_users = []
        
        for user_id, websocket in self.active_connections.items():
            # Check permission if required
            if required_permission and required_permission not in self.user_permissions.get(user_id, []):
                continue
                
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)

# Pydantic models for dashboard API
class RiskMetrics(BaseModel):
    """Current portfolio risk metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_var: float = Field(..., description="Portfolio VaR")
    correlation_shock_level: float = Field(..., description="Current correlation shock level")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Current Sharpe ratio")
    volatility: float = Field(..., description="Portfolio volatility")
    leverage: float = Field(..., description="Current leverage")
    liquidity_risk: float = Field(..., description="Liquidity risk score")

class AgentStatus(BaseModel):
    """Individual MARL agent status."""
    agent_name: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Agent status (active/inactive/error)")
    last_update: datetime = Field(..., description="Last update timestamp")
    performance_score: float = Field(..., description="Performance score (0-1)")
    current_recommendation: str = Field(..., description="Current recommendation")
    confidence: float = Field(..., description="Recommendation confidence")

class FlaggedTrade(BaseModel):
    """Trade flagged for human review."""
    trade_id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Trade direction (LONG/SHORT)")
    quantity: float = Field(..., description="Trade quantity")
    entry_price: float = Field(..., description="Proposed entry price")
    risk_score: float = Field(..., description="Risk score (0-1)")
    failure_probability: float = Field(..., description="Monte Carlo failure probability")
    agent_recommendations: List[Dict[str, Any]] = Field(..., description="Agent recommendations")
    flagged_at: datetime = Field(default_factory=datetime.utcnow)
    flagged_reason: str = Field(..., description="Reason for flagging")
    expires_at: datetime = Field(..., description="Decision expiration time")

class CrisisAlert(BaseModel):
    """Crisis detection alert."""
    alert_id: str = Field(..., description="Alert identifier")
    severity: str = Field(..., description="Alert severity (LOW/MEDIUM/HIGH/CRITICAL)")
    alert_type: str = Field(..., description="Type of crisis detected")
    message: str = Field(..., description="Alert message")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = Field(default_factory=dict, description="Associated metrics")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")

class HumanDecision(BaseModel):
    """Human decision on flagged trade."""
    trade_id: str = Field(..., description="Trade identifier")
    decision: str = Field(..., description="APPROVE or REJECT")
    reasoning: str = Field(..., description="Decision reasoning")
    user_id: str = Field(..., description="Decision maker user ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DashboardData(BaseModel):
    """Complete dashboard data."""
    risk_metrics: RiskMetrics
    agent_statuses: List[AgentStatus]
    flagged_trades: List[FlaggedTrade]
    crisis_alerts: List[CrisisAlert]
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Global instances
connection_manager = ConnectionManager()
redis_client: Optional[redis.Redis] = None
risk_assessor: Optional[RealTimeRiskAssessor] = None
var_calculator: Optional[VaRCalculator] = None
correlation_tracker: Optional[CorrelationTracker] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global redis_client, risk_assessor, var_calculator, correlation_tracker
    
    logger.info("Starting Risk Dashboard API")
    
    # Initialize Redis connection
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    
    # Initialize risk components
    risk_assessor = RealTimeRiskAssessor()
    var_calculator = VaRCalculator()
    correlation_tracker = CorrelationTracker()
    
    # Start background tasks
    asyncio.create_task(risk_data_broadcaster())
    
    logger.info("Risk Dashboard API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Risk Dashboard API")
    
    if redis_client:
        await redis_client.close()
    
    logger.info("Risk Dashboard API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Human-in-the-Loop Risk Dashboard API",
    description="Real-time risk data API for human validation dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dashboard.grandmodel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization"],
)

async def risk_data_broadcaster():
    """Background task to broadcast real-time risk data."""
    while True:
        try:
            # Generate current dashboard data
            dashboard_data = await get_current_dashboard_data()
            
            # Broadcast to all connected users
            message = {
                "type": "dashboard_update",
                "data": dashboard_data.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await connection_manager.broadcast(message, required_permission="dashboard_read")
            
            # Sleep for 100ms to achieve <1s updates
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in risk data broadcaster: {e}")
            await asyncio.sleep(1)

async def get_current_dashboard_data() -> DashboardData:
    """Generate current dashboard data."""
    # Mock data for now - in production this would query real systems
    
    risk_metrics = RiskMetrics(
        portfolio_var=0.025,
        correlation_shock_level=0.3,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        volatility=0.15,
        leverage=2.5,
        liquidity_risk=0.2
    )
    
    agent_statuses = [
        AgentStatus(
            agent_name="strategic_mlmi",
            status="active",
            last_update=datetime.utcnow(),
            performance_score=0.92,
            current_recommendation="LONG",
            confidence=0.85
        ),
        AgentStatus(
            agent_name="tactical_fvg",
            status="active", 
            last_update=datetime.utcnow(),
            performance_score=0.88,
            current_recommendation="HOLD",
            confidence=0.67
        ),
        AgentStatus(
            agent_name="risk_monitor",
            status="active",
            last_update=datetime.utcnow(),
            performance_score=0.95,
            current_recommendation="REDUCE_EXPOSURE", 
            confidence=0.78
        ),
        AgentStatus(
            agent_name="portfolio_optimizer",
            status="active",
            last_update=datetime.utcnow(),
            performance_score=0.90,
            current_recommendation="REBALANCE",
            confidence=0.72
        )
    ]
    
    # Check for flagged trades from Redis
    flagged_trades = []
    if redis_client:
        try:
            flagged_keys = await redis_client.keys("flagged_trade:*")
            for key in flagged_keys:
                trade_data = await redis_client.get(key)
                if trade_data:
                    trade = FlaggedTrade.parse_raw(trade_data)
                    flagged_trades.append(trade)
        except Exception as e:
            logger.error(f"Error fetching flagged trades: {e}")
    
    # Check for crisis alerts
    crisis_alerts = []
    if redis_client:
        try:
            alert_keys = await redis_client.keys("crisis_alert:*")
            for key in alert_keys:
                alert_data = await redis_client.get(key)
                if alert_data:
                    alert = CrisisAlert.parse_raw(alert_data)
                    crisis_alerts.append(alert)
        except Exception as e:
            logger.error(f"Error fetching crisis alerts: {e}")
    
    return DashboardData(
        risk_metrics=risk_metrics,
        agent_statuses=agent_statuses,
        flagged_trades=flagged_trades,
        crisis_alerts=crisis_alerts
    )

@app.get("/api/dashboard/data")
@limiter.limit("60/minute")
async def get_dashboard_data(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> DashboardData:
    """Get current dashboard data."""
    # Check permissions
    if RolePermission.DASHBOARD_READ not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return await get_current_dashboard_data()

@app.get("/api/dashboard/risk-metrics")
@limiter.limit("100/minute")
async def get_risk_metrics(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> RiskMetrics:
    """Get current risk metrics."""
    if RolePermission.DASHBOARD_READ not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    dashboard_data = await get_current_dashboard_data()
    return dashboard_data.risk_metrics

@app.get("/api/dashboard/flagged-trades")
@limiter.limit("100/minute")
async def get_flagged_trades(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> List[FlaggedTrade]:
    """Get trades requiring human review."""
    if RolePermission.TRADE_REVIEW not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    dashboard_data = await get_current_dashboard_data()
    return dashboard_data.flagged_trades

@app.post("/api/dashboard/decide")
@limiter.limit("30/minute")
async def make_human_decision(
    decision: HumanDecision,
    user: UserInfo = Depends(verify_token)
) -> Dict[str, str]:
    """Process human decision on flagged trade."""
    if RolePermission.TRADE_APPROVE not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Validate decision
    if decision.decision not in ["APPROVE", "REJECT"]:
        raise HTTPException(status_code=400, detail="Decision must be APPROVE or REJECT")
    
    if len(decision.reasoning.strip()) < 10:
        raise HTTPException(status_code=400, detail="Reasoning must be at least 10 characters")
    
    # Store decision in audit trail
    audit_record = {
        **decision.dict(),
        "decision_timestamp": datetime.utcnow().isoformat(),
        "user_role": user.role,
        "ip_address": "unknown"  # Would be extracted from request in production
    }
    
    if redis_client:
        try:
            # Store audit record
            await redis_client.setex(
                f"decision_audit:{decision.trade_id}:{int(time.time())}",
                3600 * 24 * 30,  # 30 days retention
                json.dumps(audit_record)
            )
            
            # Remove from flagged trades
            await redis_client.delete(f"flagged_trade:{decision.trade_id}")
            
            # Notify all connected users
            notification = {
                "type": "decision_made",
                "data": {
                    "trade_id": decision.trade_id,
                    "decision": decision.decision,
                    "user_id": user.user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            await connection_manager.broadcast(notification, required_permission="dashboard_read")
            
        except Exception as e:
            logger.error(f"Error processing decision: {e}")
            raise HTTPException(status_code=500, detail="Failed to process decision")
    
    logger.info(f"Human decision processed: {decision.trade_id} -> {decision.decision} by {user.user_id}")
    
    return {"status": "success", "message": "Decision processed successfully"}

@app.websocket("/ws/dashboard")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str
):
    """WebSocket endpoint for real-time dashboard updates."""
    try:
        # Verify token for WebSocket connection
        # In production, this would use proper JWT verification
        if not token or len(token) < 32:
            await websocket.close(code=4001, reason="Invalid token")
            return
        
        # Mock user verification
        user_id = "dashboard_user"
        permissions = [RolePermission.DASHBOARD_READ, RolePermission.TRADE_REVIEW]
        
        await connection_manager.connect(websocket, user_id, permissions)
        
        # Send initial data
        dashboard_data = await get_current_dashboard_data()
        initial_message = {
            "type": "initial_data",
            "data": dashboard_data.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_json(initial_message)
        
        # Keep connection alive
        while True:
            # Receive ping messages to keep connection alive
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if message == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send ping to client
                await websocket.send_text("ping")
            
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(user_id)

@app.post("/api/dashboard/flag-trade")
async def flag_trade_for_review(
    trade: FlaggedTrade,
    user: UserInfo = Depends(verify_token)
) -> Dict[str, str]:
    """Flag a trade for human review (called by Pre-Mortem Agent)."""
    if RolePermission.SYSTEM_INTEGRATION not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if redis_client:
        try:
            # Store flagged trade
            await redis_client.setex(
                f"flagged_trade:{trade.trade_id}",
                300,  # 5 minutes expiration
                trade.json()
            )
            
            # Broadcast alert to dashboard users
            alert_message = {
                "type": "trade_flagged",
                "data": trade.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            await connection_manager.broadcast(alert_message, required_permission="trade_review")
            
        except Exception as e:
            logger.error(f"Error flagging trade: {e}")
            raise HTTPException(status_code=500, detail="Failed to flag trade")
    
    logger.info(f"Trade flagged for review: {trade.trade_id}")
    
    return {"status": "success", "message": "Trade flagged for review"}

@app.post("/api/dashboard/crisis-alert")
async def create_crisis_alert(
    alert: CrisisAlert,
    user: UserInfo = Depends(verify_token)
) -> Dict[str, str]:
    """Create a crisis alert (called by Meta-Learning Agent)."""
    if RolePermission.SYSTEM_INTEGRATION not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if redis_client:
        try:
            # Store crisis alert
            await redis_client.setex(
                f"crisis_alert:{alert.alert_id}",
                3600,  # 1 hour expiration
                alert.json()
            )
            
            # Broadcast crisis alert
            alert_message = {
                "type": "crisis_alert",
                "data": alert.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            await connection_manager.broadcast(alert_message)
            
        except Exception as e:
            logger.error(f"Error creating crisis alert: {e}")
            raise HTTPException(status_code=500, detail="Failed to create alert")
    
    logger.info(f"Crisis alert created: {alert.alert_id} - {alert.severity}")
    
    return {"status": "success", "message": "Crisis alert created"}

@app.get("/api/dashboard/health")
async def dashboard_health() -> Dict[str, Any]:
    """Dashboard API health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "redis": "unknown",
            "websockets": len(connection_manager.active_connections),
            "risk_assessor": "active" if risk_assessor else "inactive"
        }
    }
    
    # Check Redis connection
    if redis_client:
        try:
            await redis_client.ping()
            health_status["components"]["redis"] = "healthy"
        except Exception:
            health_status["components"]["redis"] = "unhealthy"
            health_status["status"] = "degraded"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.risk_dashboard_api:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )