"""
Main FastAPI application for Strategic MARL 30m System.
Implements secure, high-performance API with comprehensive monitoring.
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.models import (
    HealthCheckResponse, StrategicDecisionRequest, StrategicDecisionResponse,
    ErrorResponse, WebSocketMessage, DecisionType, RiskLevel
)
from src.api.event_handler import EventHandler
from src.monitoring.health_monitor import health_monitor
from src.monitoring.metrics_exporter import metrics_exporter, get_metrics, get_metrics_content_type
from src.monitoring.logger_config import (
    get_logger, set_correlation_id, get_correlation_id,
    with_correlation_id, log_request
)

# Initialize logger
logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Event handler (initialized in lifespan)
event_handler: Optional[EventHandler] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    """
    # Startup
    logger.info("Starting Strategic MARL API server")
    
    # Initialize event handler
    global event_handler
    event_handler = EventHandler(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
    await event_handler.start()
    
    # Update system info metrics
    metrics_exporter.update_system_info({
        "version": "1.0.0",
        "environment": os.getenv("APP_ENV", "production"),
        "service": "strategic-marl"
    })
    
    logger.info("API server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Strategic MARL API server")
    
    if event_handler:
        await event_handler.stop()
    
    logger.info("API server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="GrandModel Strategic MARL API",
    description="Production-ready API for Strategic MARL 30m Trading System",
    version="1.0.0",
    docs_url="/docs" if os.getenv("APP_ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("APP_ENV") != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "https://grandmodel.app").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization"],
    expose_headers=["X-Correlation-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["grandmodel.app", "*.grandmodel.app", "localhost"]
)

# Add custom middleware for logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with correlation ID."""
    return await log_request(request, call_next)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token or API key.
    """
    token = credentials.credentials
    
    # In production, this would verify against actual auth service
    # For now, basic validation
    if not token or len(token) < 32:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    # Return user info (in production, decode JWT)
    return {
        "user_id": "system",
        "permissions": ["read", "write", "trade"]
    }

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
@limiter.limit("10/minute")
async def health_check(request: Request) -> HealthCheckResponse:
    """
    Comprehensive health check endpoint.
    Returns detailed system health status.
    """
    correlation_id = get_correlation_id() or set_correlation_id()
    
    with metrics_exporter.track_http_request("GET", "/health"):
        logger.info("Health check requested", correlation_id=correlation_id)
        
        health_data = await health_monitor.get_detailed_health()
        
        return HealthCheckResponse(**health_data)

# Metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def prometheus_metrics(request: Request) -> PlainTextResponse:
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    metrics_data = get_metrics()
    return PlainTextResponse(
        content=metrics_data,
        media_type=get_metrics_content_type()
    )

# Strategic decision endpoint
@app.post("/decide", response_model=StrategicDecisionResponse, tags=["Trading"])
@limiter.limit("100/minute")
async def make_strategic_decision(
    request: StrategicDecisionRequest,
    user: Dict[str, Any] = Depends(verify_token)
) -> StrategicDecisionResponse:
    """
    Make a strategic trading decision based on market state and synergy detection.
    
    This endpoint processes the SYNERGY_DETECTED event and returns a strategic
    decision with position sizing and risk management parameters.
    """
    correlation_id = request.correlation_id or set_correlation_id()
    
    logger.info(
        "Strategic decision requested",
        correlation_id=correlation_id,
        synergy_type=request.synergy_context.synergy_type,
        user_id=user["user_id"]
    )
    
    # Track metrics
    metrics_exporter.track_correlation_id(correlation_id)
    
    try:
        async with metrics_exporter.measure_inference_latency(
            model_type="strategic_marl",
            agent_name="ensemble",
            correlation_id=correlation_id
        ):
            # In production, this would call the actual MARL model
            # For now, simulated decision logic
            start_time = time.time()
            
            # Simulate model inference
            await asyncio.sleep(0.003)  # 3ms simulated inference
            
            # Mock decision based on synergy strength
            if request.synergy_context.strength > 0.8:
                decision = DecisionType.LONG
                confidence = 0.85
                risk_level = RiskLevel.MEDIUM
                position_size = 0.7
            elif request.synergy_context.strength > 0.6:
                decision = DecisionType.LONG
                confidence = 0.65
                risk_level = RiskLevel.LOW
                position_size = 0.4
            else:
                decision = DecisionType.HOLD
                confidence = 0.5
                risk_level = RiskLevel.LOW
                position_size = 0.0
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            metrics_exporter.update_model_confidence(confidence, "strategic_marl", "ensemble")
            metrics_exporter.record_synergy_response(
                request.synergy_context.synergy_type.value,
                "success",
                correlation_id
            )
            
            response = StrategicDecisionResponse(
                correlation_id=correlation_id,
                decision=decision,
                confidence=confidence,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=request.market_state.price * 0.98 if decision != DecisionType.HOLD else None,
                take_profit=request.market_state.price * 1.03 if decision != DecisionType.HOLD else None,
                agent_decisions=[
                    {
                        "agent_name": "strategic_agent",
                        "decision": decision,
                        "confidence": confidence + 0.05,
                        "reasoning": {"synergy_alignment": "high"}
                    },
                    {
                        "agent_name": "risk_agent",
                        "decision": decision,
                        "confidence": confidence - 0.05,
                        "reasoning": {"risk_assessment": risk_level.value}
                    }
                ],
                inference_latency_ms=inference_time_ms
            )
            
            logger.info(
                "Strategic decision completed",
                correlation_id=correlation_id,
                decision=decision.value,
                confidence=confidence,
                latency_ms=inference_time_ms
            )
            
            return response
            
    except Exception as e:
        logger.error(
            "Strategic decision failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        metrics_exporter.record_error("decision_error", "api", "error")
        metrics_exporter.record_synergy_response(
            request.synergy_context.synergy_type.value,
            "failure",
            correlation_id
        )
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time events
@app.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    """
    await websocket.accept()
    correlation_id = set_correlation_id()
    
    logger.info("WebSocket connection established", correlation_id=correlation_id)
    
    try:
        # Subscribe to events
        async def event_callback(event_data: Dict[str, Any]):
            message = WebSocketMessage(
                event_type=event_data.get("type", "unknown"),
                data=event_data,
                correlation_id=correlation_id
            )
            await websocket.send_json(message.dict())
        
        if event_handler:
            await event_handler.subscribe("SYNERGY_DETECTED", event_callback)
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", correlation_id=correlation_id)
    except Exception as e:
        logger.error("WebSocket error", correlation_id=correlation_id, error=str(e))
    finally:
        if event_handler:
            await event_handler.unsubscribe("SYNERGY_DETECTED", event_callback)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    correlation_id = get_correlation_id()
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            correlation_id=correlation_id
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured response."""
    correlation_id = get_correlation_id()
    
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal error occurred",
            correlation_id=correlation_id
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        log_config="configs/logging.yaml"
    )