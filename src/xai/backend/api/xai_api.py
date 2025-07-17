"""
XAI Trading Explanations API

Provides comprehensive API endpoints for AI trading explanations,
analytics, and real-time communication.

Features:
- REST API for explanations and analytics
- WebSocket support for real-time updates
- Natural language processing for queries
- Performance metrics and insights
- Export capabilities
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import uuid
import os
import tempfile
import pandas as pd

# Import the existing XAI engine
import sys
sys.path.append('/home/QuantNova/GrandModel/src')
from tactical.xai_engine import TacticalXAIEngine, ExplanationType, ExplanationAudience

logger = logging.getLogger(__name__)

# Pydantic models for API
class TimeFrame(str, Enum):
    ONE_HOUR = '1h'
    FOUR_HOURS = '4h'
    ONE_DAY = '1d'
    ONE_WEEK = '1w'
    ONE_MONTH = '1M'
    THREE_MONTHS = '3M'
    SIX_MONTHS = '6M'
    ONE_YEAR = '1Y'
    ALL_TIME = 'ALL'

class AssetClass(str, Enum):
    EQUITIES = 'EQUITIES'
    FUTURES = 'FUTURES'
    FOREX = 'FOREX'
    CRYPTO = 'CRYPTO'
    BONDS = 'BONDS'
    COMMODITIES = 'COMMODITIES'

class ActionType(int, Enum):
    HOLD = 0
    DECREASE_LONG = 1
    INCREASE_LONG = 2
    MARKET_BUY = 3
    MARKET_SELL = 4
    LIMIT_BUY = 5
    LIMIT_SELL = 6
    INCREASE_SHORT = 7
    DECREASE_SHORT = 8

class ExplanationRequest(BaseModel):
    decision_id: str
    audience: str = ExplanationAudience.TRADER.value
    include_alternatives: bool = False
    include_feature_details: bool = True

class ExplanationResponse(BaseModel):
    id: str
    decision_id: str
    type: str
    audience: str
    summary: str
    detailed_reasoning: str
    feature_importance: Dict[str, float]
    top_factors: List[Dict[str, Any]]
    confidence: float
    generation_time: float
    created_at: datetime

class DecisionRequest(BaseModel):
    page: int = 1
    page_size: int = 20
    symbols: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    asset_classes: Optional[List[AssetClass]] = None
    sort_by: str = 'timestamp'
    sort_order: str = 'desc'

class AnalyticsRequest(BaseModel):
    type: str
    symbols: Optional[List[str]] = None
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    filters: Optional[Dict[str, Any]] = None
    include_charts: bool = True

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    response_format: str = 'text'

class ChatResponse(BaseModel):
    id: str
    response: str
    conversation_id: str
    attachments: Optional[List[Dict[str, Any]]] = None
    suggested_questions: Optional[List[str]] = None
    processing_time: float
    confidence: float

class NaturalLanguageQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict, message_type: str = None):
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                # Check if client is subscribed to this message type
                if message_type and client_id in self.subscriptions:
                    subscriptions = self.subscriptions[client_id].get('types', [])
                    if message_type not in subscriptions:
                        continue
                
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def subscribe(self, client_id: str, subscription_data: dict):
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = {}
        self.subscriptions[client_id].update(subscription_data)

# Initialize FastAPI app
app = FastAPI(
    title="XAI Trading API",
    description="AI-Powered Trading Explanations and Analytics API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
xai_engine = TacticalXAIEngine()
connection_manager = ConnectionManager()

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple auth - replace with proper authentication"""
    if not credentials:
        return None
    # In production, verify the token and return user data
    return {"id": "user123", "username": "trader", "role": "trader"}

# Mock data storage (replace with proper database)
decisions_db = []
explanations_db = []
conversations_db = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "xai_engine": "running",
            "websocket": "running"
        }
    }

# Explanations endpoints
@app.post("/api/explanations/generate", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user)
):
    """Generate explanation for a trading decision"""
    try:
        # Mock decision data (replace with actual database lookup)
        decision_data = {
            "id": request.decision_id,
            "symbol": "NQ",
            "action": ActionType.INCREASE_LONG,
            "confidence": 0.85,
            "timestamp": datetime.now()
        }
        
        # Generate explanation using XAI engine
        # This is simplified - in production, you'd retrieve the actual decision snapshot
        explanation_result = ExplanationResponse(
            id=str(uuid.uuid4()),
            decision_id=request.decision_id,
            type=ExplanationType.FEATURE_IMPORTANCE.value,
            audience=request.audience,
            summary=f"AI decision to {ActionType.INCREASE_LONG.name} NQ with 85% confidence",
            detailed_reasoning="The AI system identified strong bullish momentum indicators including positive price action, increased volume, and favorable technical patterns. The decision was supported by convergence across multiple agents with high consensus agreement.",
            feature_importance={
                "price_momentum": 0.35,
                "volume_ratio": 0.25,
                "technical_indicators": 0.20,
                "market_sentiment": 0.15,
                "volatility": 0.05
            },
            top_factors=[
                {"name": "price_momentum", "importance": 0.35, "direction": "positive"},
                {"name": "volume_ratio", "importance": 0.25, "direction": "positive"},
                {"name": "technical_indicators", "importance": 0.20, "direction": "positive"}
            ],
            confidence=0.85,
            generation_time=150.5,
            created_at=datetime.now()
        )
        
        # Store explanation
        explanations_db.append(explanation_result.dict())
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(
            connection_manager.broadcast,
            {
                "type": "EXPLANATION_READY",
                "payload": explanation_result.dict(),
                "timestamp": datetime.now().isoformat()
            },
            "EXPLANATION_READY"
        )
        
        return explanation_result
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explanations")
async def get_explanations(
    page: int = 1,
    page_size: int = 20,
    user = Depends(get_current_user)
):
    """Get list of explanations with pagination"""
    try:
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        explanations = explanations_db[start_idx:end_idx]
        total_count = len(explanations_db)
        
        return {
            "success": True,
            "data": {
                "explanations": explanations,
                "pagination": {
                    "page": page,
                    "pageSize": page_size,
                    "totalItems": total_count,
                    "totalPages": (total_count + page_size - 1) // page_size,
                    "hasNext": end_idx < total_count,
                    "hasPrevious": page > 1
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Decisions endpoints
@app.get("/api/decisions")
async def get_decisions(
    request: DecisionRequest = Depends(),
    user = Depends(get_current_user)
):
    """Get trading decisions with filtering and pagination"""
    try:
        # Mock decision data (replace with actual database query)
        mock_decisions = [
            {
                "id": str(uuid.uuid4()),
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                "symbol": "NQ" if i % 2 == 0 else "ES",
                "assetClass": AssetClass.FUTURES.value,
                "action": ActionType.INCREASE_LONG.value,
                "confidence": 0.8 + (i % 3) * 0.05,
                "size": 1.0,
                "reasoning": f"Decision {i}: Strong bullish momentum detected",
                "agentContributions": [],
                "riskMetrics": {
                    "varRisk": 0.02,
                    "positionRisk": 0.05,
                    "correlationRisk": 0.03,
                    "volatility": 0.15
                },
                "marketContext": {
                    "volatility": 0.15,
                    "trend": "BULLISH",
                    "volume": 1500000,
                    "liquidity": 0.95
                },
                "executionStatus": "EXECUTED"
            }
            for i in range(50)  # Generate 50 mock decisions
        ]
        
        # Apply filters
        filtered_decisions = mock_decisions
        if request.symbols:
            filtered_decisions = [d for d in filtered_decisions if d["symbol"] in request.symbols]
        
        # Apply pagination
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        paginated_decisions = filtered_decisions[start_idx:end_idx]
        
        return {
            "success": True,
            "data": {
                "decisions": paginated_decisions,
                "pagination": {
                    "page": request.page,
                    "pageSize": request.page_size,
                    "totalItems": len(filtered_decisions),
                    "totalPages": (len(filtered_decisions) + request.page_size - 1) // request.page_size,
                    "hasNext": end_idx < len(filtered_decisions),
                    "hasPrevious": request.page > 1
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/decisions/{decision_id}")
async def get_decision_detail(
    decision_id: str,
    user = Depends(get_current_user)
):
    """Get detailed decision information with explanation"""
    try:
        # Mock decision detail (replace with actual database lookup)
        decision = {
            "id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": "NQ",
            "assetClass": AssetClass.FUTURES.value,
            "action": ActionType.INCREASE_LONG.value,
            "confidence": 0.85,
            "size": 1.0,
            "reasoning": "Strong bullish momentum with high agent consensus",
            "agentContributions": [
                {
                    "agentId": "fvg_agent",
                    "agentName": "Fair Value Gap Agent",
                    "action": ActionType.INCREASE_LONG.value,
                    "confidence": 0.88,
                    "weight": 0.3,
                    "keyFactors": ["price_gap", "volume_confirmation"]
                },
                {
                    "agentId": "momentum_agent",
                    "agentName": "Momentum Agent",
                    "action": ActionType.INCREASE_LONG.value,
                    "confidence": 0.82,
                    "weight": 0.25,
                    "keyFactors": ["ma_crossover", "rsi_bullish"]
                }
            ],
            "riskMetrics": {
                "varRisk": 0.02,
                "positionRisk": 0.05,
                "correlationRisk": 0.03,
                "volatility": 0.15
            },
            "marketContext": {
                "volatility": 0.15,
                "trend": "BULLISH",
                "volume": 1500000,
                "liquidity": 0.95
            },
            "executionStatus": "EXECUTED"
        }
        
        # Mock explanation
        explanation = {
            "id": str(uuid.uuid4()),
            "decisionId": decision_id,
            "type": ExplanationType.FEATURE_IMPORTANCE.value,
            "audience": ExplanationAudience.TRADER.value,
            "summary": "AI recommended INCREASE_LONG for NQ with 85% confidence",
            "detailedReasoning": "The AI system identified strong bullish momentum indicators...",
            "featureImportance": {
                "price_momentum": 0.35,
                "volume_ratio": 0.25,
                "technical_indicators": 0.20
            },
            "topFactors": [
                {"name": "price_momentum", "importance": 0.35, "direction": "positive"}
            ],
            "confidence": 0.85,
            "generationTime": 150.5
        }
        
        return {
            "success": True,
            "data": {
                "decision": decision,
                "explanation": explanation,
                "relatedDecisions": []
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get decision detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.post("/api/analytics/query")
async def query_analytics(
    request: AnalyticsRequest,
    user = Depends(get_current_user)
):
    """Query analytics data with flexible parameters"""
    try:
        # Mock analytics data
        metrics = {
            "totalReturns": 0.15,
            "annualizedReturns": 0.18,
            "volatility": 0.12,
            "sharpeRatio": 1.5,
            "maxDrawdown": -0.08,
            "winRate": 0.65,
            "profitFactor": 1.8,
            "tradesCount": 150,
            "avgTradeSize": 1.2,
            "periodStart": request.start_date.isoformat(),
            "periodEnd": request.end_date.isoformat()
        }
        
        charts = {}
        if request.include_charts:
            # Mock chart data
            charts = {
                "performanceChart": {
                    "type": "line",
                    "data": {
                        "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                        "datasets": [{
                            "label": "Portfolio Returns",
                            "data": [0, 2.5, 5.2, 3.8, 7.1]
                        }]
                    }
                },
                "drawdownChart": {
                    "type": "line",
                    "data": {
                        "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                        "datasets": [{
                            "label": "Drawdown",
                            "data": [0, -1.2, -3.5, -2.1, -0.8]
                        }]
                    }
                }
            }
        
        insights = [
            {
                "type": "performance",
                "title": "Strong Performance Trend",
                "description": "Portfolio showing consistent upward trend with low volatility",
                "severity": "info",
                "actionable": False
            }
        ]
        
        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "charts": charts,
                "insights": insights
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to query analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance")
async def get_performance_metrics(
    timeframe: TimeFrame = TimeFrame.ONE_DAY,
    user = Depends(get_current_user)
):
    """Get performance metrics for specified timeframe"""
    try:
        # Mock performance metrics
        metrics = {
            "totalReturns": 0.15,
            "annualizedReturns": 0.18,
            "volatility": 0.12,
            "sharpeRatio": 1.5,
            "maxDrawdown": -0.08,
            "winRate": 0.65,
            "profitFactor": 1.8,
            "tradesCount": 150,
            "avgTradeSize": 1.2,
            "periodStart": (datetime.now() - timedelta(days=30)).isoformat(),
            "periodEnd": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/api/chat/message", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user)
):
    """Send chat message and get AI response"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Mock AI response generation
        response_text = f"Thank you for your question: '{request.message}'. Based on the current market conditions and recent trading decisions, I can provide the following insights..."
        
        # Create response
        response = ChatResponse(
            id=str(uuid.uuid4()),
            response=response_text,
            conversation_id=conversation_id,
            attachments=[],
            suggested_questions=[
                "What were the key factors in the last decision?",
                "How is the portfolio performing today?",
                "What are the current risk levels?"
            ],
            processing_time=250.0,
            confidence=0.9
        )
        
        # Store conversation (in production, use proper database)
        if conversation_id not in conversations_db:
            conversations_db[conversation_id] = {
                "id": conversation_id,
                "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
                "createdAt": datetime.now().isoformat(),
                "messages": []
            }
        
        # Add messages to conversation
        conversations_db[conversation_id]["messages"].extend([
            {
                "id": str(uuid.uuid4()),
                "type": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": response.id,
                "type": "assistant",
                "content": response.response,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "confidence": response.confidence,
                    "processingTime": response.processing_time
                }
            }
        ])
        
        conversations_db[conversation_id]["updatedAt"] = datetime.now().isoformat()
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(
            connection_manager.broadcast,
            {
                "type": "CHAT_RESPONSE",
                "payload": response.dict(),
                "timestamp": datetime.now().isoformat()
            },
            "CHAT_RESPONSE"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to process chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations(user = Depends(get_current_user)):
    """Get list of conversations"""
    try:
        conversations = []
        for conv_id, conv_data in conversations_db.items():
            conversations.append({
                "id": conv_id,
                "title": conv_data["title"],
                "lastMessage": conv_data["messages"][-1]["content"] if conv_data["messages"] else "",
                "timestamp": conv_data["updatedAt"],
                "messageCount": len(conv_data["messages"])
            })
        
        return {
            "success": True,
            "data": {
                "conversations": conversations
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation(
    request: dict,
    user = Depends(get_current_user)
):
    """Create new conversation"""
    try:
        conversation_id = str(uuid.uuid4())
        title = request.get("title", "New Conversation")
        
        conversation = {
            "id": conversation_id,
            "title": title,
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            "messages": []
        }
        
        conversations_db[conversation_id] = conversation
        
        return {
            "success": True,
            "data": {
                "conversation": conversation
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Natural Language Processing
@app.post("/api/nlp/query")
async def process_natural_language_query(
    request: NaturalLanguageQueryRequest,
    user = Depends(get_current_user)
):
    """Process natural language query and return structured response"""
    try:
        # Mock NLP processing
        query_lower = request.query.lower()
        
        response_data = {}
        
        if "performance" in query_lower:
            response_data = {
                "type": "performance",
                "data": {
                    "metrics": {
                        "totalReturns": 0.15,
                        "sharpeRatio": 1.5,
                        "maxDrawdown": -0.08
                    },
                    "summary": "Portfolio performance has been strong with 15% total returns and a Sharpe ratio of 1.5"
                }
            }
        elif "risk" in query_lower:
            response_data = {
                "type": "risk",
                "data": {
                    "currentVaR": 0.02,
                    "portfolioExposure": 0.75,
                    "summary": "Current VaR is 2% with moderate portfolio exposure"
                }
            }
        else:
            response_data = {
                "type": "general",
                "data": {
                    "summary": f"I understand you're asking about: {request.query}. Let me provide relevant trading insights..."
                }
            }
        
        return {
            "success": True,
            "data": response_data
        }
        
    except Exception as e:
        logger.error(f"Failed to process NLP query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "SUBSCRIBE":
                # Handle subscription
                connection_manager.subscribe(client_id, {
                    "types": message.get("payload", {}).get("messageType", [])
                })
                await connection_manager.send_personal_message({
                    "type": "SUBSCRIPTION_CONFIRMED",
                    "payload": {"subscriptionId": message.get("payload", {}).get("subscriptionId")},
                    "timestamp": datetime.now().isoformat()
                }, client_id)
            
            elif message.get("type") == "PING":
                # Handle ping/pong
                await connection_manager.send_personal_message({
                    "type": "PONG",
                    "payload": message.get("payload", {}),
                    "timestamp": datetime.now().isoformat()
                }, client_id)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)

# System status
@app.get("/api/system/status")
async def get_system_status():
    """Get system status information"""
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "uptime": 3600,
            "services": [
                {
                    "name": "XAI Engine",
                    "status": "running",
                    "health": "healthy",
                    "version": "1.0.0",
                    "lastCheck": datetime.now().isoformat()
                },
                {
                    "name": "WebSocket Server",
                    "status": "running",
                    "health": "healthy",
                    "version": "1.0.0",
                    "lastCheck": datetime.now().isoformat()
                }
            ],
            "performance": {
                "avgResponseTime": 150,
                "throughput": 1000,
                "errorRate": 0.001
            },
            "lastUpdate": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")