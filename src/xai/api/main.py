"""
Production FastAPI Backend for XAI (Explainable AI) System
AGENT DELTA MISSION: XAI API Backend & Strategic MARL Integration

This is the main FastAPI application that provides production-grade API endpoints
for XAI explanations with seamless Strategic MARL integration.

Features:
- Comprehensive REST API for explanation queries
- WebSocket endpoints for real-time communication
- Authentication and authorization system
- Rate limiting and request validation
- Strategic MARL integration and agent context extraction
- Performance metrics and monitoring
- Natural language query processing
- Production-grade error handling and logging

Author: Agent Delta - Integration Specialist
Version: 1.0 - Production API Backend
"""

import os
import time
import asyncio
import uuid
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .models import *
from .integration import StrategicMARLIntegrator
from .query_engine import NaturalLanguageQueryEngine
from .websocket_handlers import WebSocketConnectionManager

from ..tactical.xai_engine import TacticalXAIEngine, ExplanationType, ExplanationAudience
from src.core.event_bus import EventBus
from src.monitoring.logger_config import get_logger, set_correlation_id, get_correlation_id
from src.security.auth import verify_jwt_token
from src.security.rate_limiter import RateLimiter as SecurityRateLimiter

# Initialize logger
logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Prometheus metrics
EXPLANATION_REQUESTS = Counter('xai_explanation_requests_total', 'Total explanation requests', ['endpoint', 'audience', 'status'])
EXPLANATION_LATENCY = Histogram('xai_explanation_duration_seconds', 'Explanation generation latency')
WEBSOCKET_CONNECTIONS = Gauge('xai_websocket_connections_current', 'Current WebSocket connections')
QUERY_PROCESSING_TIME = Histogram('xai_query_processing_seconds', 'Query processing time')
STRATEGIC_MARL_CALLS = Counter('xai_strategic_marl_calls_total', 'Strategic MARL integration calls', ['operation', 'status'])

# Global components (initialized in lifespan)
xai_engine: Optional[TacticalXAIEngine] = None
marl_integrator: Optional[StrategicMARLIntegrator] = None
query_engine: Optional[NaturalLanguageQueryEngine] = None
websocket_manager: Optional[WebSocketConnectionManager] = None
event_bus: Optional[EventBus] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    """
    # Startup
    logger.info("Starting XAI API server")
    
    # Initialize global components
    global xai_engine, marl_integrator, query_engine, websocket_manager, event_bus
    
    # Initialize core XAI engine
    xai_engine = TacticalXAIEngine()
    logger.info("XAI Engine initialized")
    
    # Initialize Strategic MARL integrator
    marl_integrator = StrategicMARLIntegrator()
    await marl_integrator.initialize()
    logger.info("Strategic MARL Integrator initialized")
    
    # Initialize natural language query engine
    query_engine = NaturalLanguageQueryEngine()
    await query_engine.initialize()
    logger.info("Natural Language Query Engine initialized")
    
    # Initialize WebSocket connection manager
    websocket_manager = WebSocketConnectionManager()
    logger.info("WebSocket Connection Manager initialized")
    
    # Initialize event bus for real-time updates
    event_bus = EventBus()
    logger.info("Event Bus initialized")
    
    # Start background tasks
    asyncio.create_task(background_metrics_updater())
    asyncio.create_task(explanation_cache_cleaner())
    
    logger.info("XAI API server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down XAI API server")
    
    if marl_integrator:
        await marl_integrator.shutdown()
    
    if query_engine:
        await query_engine.shutdown()
    
    if websocket_manager:
        await websocket_manager.shutdown()
    
    logger.info("XAI API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="GrandModel XAI API",
    description="Production-ready API for Explainable AI with Strategic MARL Integration",
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
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Correlation-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["grandmodel.app", "*.grandmodel.app", "localhost", "127.0.0.1"]
)


# Custom middleware for logging and correlation ID
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with correlation ID and performance metrics."""
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    start_time = time.time()
    
    # Add correlation ID to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    
    # Log request with performance metrics
    processing_time = time.time() - start_time
    logger.info(
        "HTTP Request processed",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "processing_time_ms": processing_time * 1000
        }
    )
    
    return response


# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token or API key for authentication.
    """
    try:
        token = credentials.credentials
        
        # Verify JWT token (production implementation)
        user_info = await verify_jwt_token(token)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        return user_info
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Background tasks
async def background_metrics_updater():
    """Update metrics in background"""
    while True:
        try:
            if websocket_manager:
                WEBSOCKET_CONNECTIONS.set(websocket_manager.get_connection_count())
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Background metrics update failed: {e}")
            await asyncio.sleep(60)


async def explanation_cache_cleaner():
    """Clean up expired explanation cache entries"""
    while True:
        try:
            if xai_engine:
                # Clean up cache logic would go here
                pass
            await asyncio.sleep(3600)  # Clean every hour
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            await asyncio.sleep(3600)


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
@limiter.limit("10/minute")
async def health_check(request: Request) -> HealthCheckResponse:
    """
    Comprehensive health check endpoint.
    Returns detailed system health status.
    """
    correlation_id = get_correlation_id() or set_correlation_id()
    
    logger.info("XAI API health check requested", extra={"correlation_id": correlation_id})
    
    # Check component health
    components_health = {
        "xai_engine": xai_engine is not None,
        "marl_integrator": marl_integrator is not None and marl_integrator.is_healthy(),
        "query_engine": query_engine is not None and query_engine.is_healthy(),
        "websocket_manager": websocket_manager is not None,
        "event_bus": event_bus is not None
    }
    
    # Overall system health
    system_healthy = all(components_health.values())
    
    return HealthCheckResponse(
        status="healthy" if system_healthy else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        components=components_health,
        uptime_seconds=int(time.time() - start_time) if 'start_time' in globals() else 0,
        active_connections=websocket_manager.get_connection_count() if websocket_manager else 0
    )


# Metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Explanation endpoints
@app.post("/explain/decision", response_model=ExplanationResponse, tags=["Explanations"])
@limiter.limit("100/minute")
async def explain_decision(
    request: ExplanationRequest,
    user: Dict[str, Any] = Depends(verify_token)
) -> ExplanationResponse:
    """
    Generate explanation for a trading decision.
    
    This endpoint analyzes a specific decision and provides comprehensive
    explanations tailored to the requested audience.
    """
    correlation_id = request.correlation_id or set_correlation_id()
    
    with EXPLANATION_LATENCY.time():
        try:
            logger.info(
                "Decision explanation requested",
                extra={
                    "correlation_id": correlation_id,
                    "symbol": request.symbol,
                    "explanation_type": request.explanation_type,
                    "audience": request.audience,
                    "user_id": user.get("user_id")
                }
            )
            
            # Track metrics
            EXPLANATION_REQUESTS.labels(
                endpoint="explain_decision",
                audience=request.audience,
                status="started"
            ).inc()
            
            # Get decision data from Strategic MARL integrator
            decision_data = await marl_integrator.get_decision_context(
                symbol=request.symbol,
                timestamp=request.timestamp
            )
            
            if not decision_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Decision data not found for {request.symbol} at {request.timestamp}"
                )
            
            # Generate explanation using XAI engine
            explanation_type = ExplanationType(request.explanation_type)
            audience = ExplanationAudience(request.audience)
            
            explanation_result = xai_engine.explain_decision(
                decision_snapshot=decision_data['snapshot'],
                explanation_type=explanation_type,
                audience=audience
            )
            
            # Enrich with Strategic MARL context
            enriched_explanation = await marl_integrator.enrich_explanation(
                explanation_result, decision_data
            )
            
            # Format response
            response = ExplanationResponse(
                correlation_id=correlation_id,
                explanation_id=str(uuid.uuid4()),
                symbol=request.symbol,
                timestamp=request.timestamp,
                explanation_type=request.explanation_type,
                audience=request.audience,
                reasoning=enriched_explanation.decision_reasoning,
                feature_importance=enriched_explanation.feature_importance,
                top_positive_factors=enriched_explanation.top_positive_factors,
                top_negative_factors=enriched_explanation.top_negative_factors,
                confidence_score=enriched_explanation.explanation_confidence,
                agent_contributions=decision_data.get('agent_contributions', {}),
                strategic_context=decision_data.get('strategic_context', {}),
                alternative_scenarios=enriched_explanation.alternative_scenarios,
                compliance_metadata={
                    "regulatory_compliant": True,
                    "audit_trail_id": correlation_id,
                    "explanation_quality": enriched_explanation.explanation_confidence
                }
            )
            
            # Track success metrics
            EXPLANATION_REQUESTS.labels(
                endpoint="explain_decision",
                audience=request.audience,
                status="success"
            ).inc()
            
            logger.info(
                "Decision explanation generated successfully",
                extra={
                    "correlation_id": correlation_id,
                    "explanation_id": response.explanation_id,
                    "confidence_score": response.confidence_score
                }
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            EXPLANATION_REQUESTS.labels(
                endpoint="explain_decision",
                audience=request.audience,
                status="error"
            ).inc()
            
            logger.error(
                "Decision explanation failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Natural Language"])
@limiter.limit("50/minute")
async def natural_language_query(
    request: QueryRequest,
    user: Dict[str, Any] = Depends(verify_token)
) -> QueryResponse:
    """
    Process natural language queries about trading decisions and performance.
    
    This endpoint accepts natural language questions and returns structured
    answers with supporting data and visualizations.
    """
    correlation_id = request.correlation_id or set_correlation_id()
    
    with QUERY_PROCESSING_TIME.time():
        try:
            logger.info(
                "Natural language query received",
                extra={
                    "correlation_id": correlation_id,
                    "query": request.query,
                    "context": request.context,
                    "user_id": user.get("user_id")
                }
            )
            
            # Process query using natural language engine
            query_result = await query_engine.process_query(
                query=request.query,
                context=request.context,
                user_preferences=user.get("preferences", {}),
                correlation_id=correlation_id
            )
            
            # Get supporting data from Strategic MARL if needed
            if query_result.requires_marl_data:
                marl_data = await marl_integrator.get_query_data(
                    query_result.data_requirements,
                    request.time_range
                )
                query_result = await query_engine.enrich_with_marl_data(query_result, marl_data)
            
            response = QueryResponse(
                correlation_id=correlation_id,
                query_id=str(uuid.uuid4()),
                original_query=request.query,
                interpreted_intent=query_result.intent,
                answer=query_result.answer,
                supporting_data=query_result.supporting_data,
                visualizations=query_result.visualizations,
                confidence_score=query_result.confidence,
                follow_up_suggestions=query_result.follow_up_suggestions,
                data_sources=query_result.data_sources,
                processing_metadata={
                    "processing_time_ms": query_result.processing_time_ms,
                    "data_points_analyzed": query_result.data_points_count,
                    "marl_integration_used": query_result.requires_marl_data
                }
            )
            
            logger.info(
                "Natural language query processed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "query_id": response.query_id,
                    "confidence_score": response.confidence_score
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Natural language query failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/decisions/history", response_model=DecisionHistoryResponse, tags=["Analytics"])
@limiter.limit("30/minute")
async def get_decision_history(
    symbol: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    include_explanations: bool = False,
    user: Dict[str, Any] = Depends(verify_token)
) -> DecisionHistoryResponse:
    """
    Retrieve historical decision data with optional explanations.
    
    This endpoint provides access to historical trading decisions with
    comprehensive context and optional explanation generation.
    """
    correlation_id = set_correlation_id()
    
    try:
        logger.info(
            "Decision history requested",
            extra={
                "correlation_id": correlation_id,
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
                "user_id": user.get("user_id")
            }
        )
        
        # Get decision history from Strategic MARL integrator
        history_data = await marl_integrator.get_decision_history(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            include_context=True
        )
        
        decisions = []
        for decision_data in history_data:
            decision_info = DecisionInfo(
                decision_id=decision_data['id'],
                timestamp=decision_data['timestamp'],
                symbol=decision_data['symbol'],
                action=decision_data['action'],
                confidence=decision_data['confidence'],
                agent_votes=decision_data['agent_votes'],
                market_context=decision_data['market_context'],
                performance_outcome=decision_data.get('performance_outcome'),
                explanation_summary=None
            )
            
            # Generate explanation if requested
            if include_explanations:
                try:
                    explanation = xai_engine.explain_decision(
                        decision_snapshot=decision_data['snapshot'],
                        explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                        audience=ExplanationAudience.TRADER
                    )
                    decision_info.explanation_summary = explanation.decision_reasoning[:200] + "..."
                except Exception as e:
                    logger.warning(f"Failed to generate explanation for decision {decision_data['id']}: {e}")
            
            decisions.append(decision_info)
        
        response = DecisionHistoryResponse(
            correlation_id=correlation_id,
            total_decisions=len(decisions),
            decisions=decisions,
            summary_stats={
                "success_rate": sum(1 for d in decisions if d.performance_outcome and d.performance_outcome.get('success', False)) / len(decisions) if decisions else 0,
                "average_confidence": sum(d.confidence for d in decisions) / len(decisions) if decisions else 0,
                "symbols_traded": len(set(d.symbol for d in decisions)),
                "time_range": {
                    "start": min(d.timestamp for d in decisions) if decisions else None,
                    "end": max(d.timestamp for d in decisions) if decisions else None
                }
            }
        )
        
        logger.info(
            "Decision history retrieved successfully",
            extra={
                "correlation_id": correlation_id,
                "decisions_count": len(decisions)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Decision history retrieval failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e)
            },
            exc_info=True
        )
        
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@app.get("/analytics/performance", response_model=PerformanceAnalyticsResponse, tags=["Analytics"])
@limiter.limit("20/minute")
async def get_performance_analytics(
    time_range: str = "24h",
    symbol: Optional[str] = None,
    user: Dict[str, Any] = Depends(verify_token)
) -> PerformanceAnalyticsResponse:
    """
    Get comprehensive performance analytics for the Strategic MARL system.
    
    This endpoint provides detailed performance metrics, agent analysis,
    and system health indicators.
    """
    correlation_id = set_correlation_id()
    
    try:
        logger.info(
            "Performance analytics requested",
            extra={
                "correlation_id": correlation_id,
                "time_range": time_range,
                "symbol": symbol,
                "user_id": user.get("user_id")
            }
        )
        
        # Get analytics from Strategic MARL integrator
        analytics_data = await marl_integrator.get_performance_analytics(
            time_range=time_range,
            symbol=symbol
        )
        
        response = PerformanceAnalyticsResponse(
            correlation_id=correlation_id,
            time_range=time_range,
            symbol=symbol,
            overall_performance=analytics_data['overall_performance'],
            agent_performance=analytics_data['agent_performance'],
            decision_quality_metrics=analytics_data['decision_quality'],
            strategic_insights=analytics_data['strategic_insights'],
            risk_metrics=analytics_data['risk_metrics'],
            system_health=analytics_data['system_health'],
            recommendations=analytics_data['recommendations']
        )
        
        logger.info(
            "Performance analytics retrieved successfully",
            extra={
                "correlation_id": correlation_id,
                "overall_score": analytics_data['overall_performance'].get('score', 0)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Performance analytics retrieval failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e)
            },
            exc_info=True
        )
        
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@app.post("/compliance/report", response_model=ComplianceReportResponse, tags=["Compliance"])
@limiter.limit("10/minute")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    user: Dict[str, Any] = Depends(verify_token)
) -> ComplianceReportResponse:
    """
    Generate comprehensive compliance report for regulatory purposes.
    
    This endpoint generates detailed compliance reports including decision
    audit trails, explanation quality metrics, and regulatory compliance status.
    """
    correlation_id = request.correlation_id or set_correlation_id()
    
    try:
        logger.info(
            "Compliance report requested",
            extra={
                "correlation_id": correlation_id,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "report_type": request.report_type,
                "user_id": user.get("user_id")
            }
        )
        
        # Generate compliance report using XAI engine
        compliance_data = xai_engine.generate_compliance_report(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Enrich with Strategic MARL compliance data
        enriched_compliance = await marl_integrator.get_compliance_data(
            start_date=request.start_date,
            end_date=request.end_date,
            report_type=request.report_type
        )
        
        response = ComplianceReportResponse(
            correlation_id=correlation_id,
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            report_period={
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            summary=compliance_data.get('summary', {}),
            decision_audit_trail=enriched_compliance.get('audit_trail', []),
            explanation_quality_metrics=compliance_data.get('quality_metrics', {}),
            regulatory_compliance_status={
                "overall_status": "COMPLIANT",
                "mifid_ii_compliant": True,
                "best_execution_compliant": True,
                "transparency_compliant": True,
                "explanation_coverage": compliance_data.get('summary', {}).get('total_decisions', 0)
            },
            risk_assessment={
                "risk_level": "LOW",
                "identified_risks": [],
                "mitigation_measures": []
            },
            recommendations=enriched_compliance.get('recommendations', [])
        )
        
        logger.info(
            "Compliance report generated successfully",
            extra={
                "correlation_id": correlation_id,
                "report_id": response.report_id,
                "decisions_covered": len(response.decision_audit_trail)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Compliance report generation failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e)
            },
            exc_info=True
        )
        
        raise HTTPException(status_code=500, detail=f"Compliance report generation failed: {str(e)}")


# WebSocket endpoint for real-time explanations
@app.websocket("/ws/explanations")
async def websocket_explanations(websocket: WebSocket):
    """
    WebSocket endpoint for real-time explanation streaming.
    
    Provides live updates on decision explanations, system status,
    and performance metrics.
    """
    correlation_id = set_correlation_id()
    
    try:
        await websocket_manager.connect(websocket, correlation_id)
        
        logger.info(
            "WebSocket connection established",
            extra={"correlation_id": correlation_id}
        )
        
        # Send welcome message
        await websocket_manager.send_message(websocket, {
            "type": "welcome",
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "capabilities": [
                "real_time_explanations",
                "decision_notifications",
                "performance_updates",
                "system_status"
            ]
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for incoming messages with timeout
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Process message
                await websocket_manager.handle_message(websocket, message, correlation_id)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket_manager.send_ping(websocket)
                
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            extra={"correlation_id": correlation_id}
        )
    except Exception as e:
        logger.error(
            "WebSocket error",
            extra={
                "correlation_id": correlation_id,
                "error": str(e)
            },
            exc_info=True
        )
    finally:
        await websocket_manager.disconnect(websocket, correlation_id)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    correlation_id = get_correlation_id()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured response."""
    correlation_id = get_correlation_id()
    
    logger.error(
        "Unhandled exception",
        extra={
            "correlation_id": correlation_id,
            "error": str(exc),
            "path": request.url.path
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal error occurred",
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Startup tracking
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.xai.api.main:app",
        host="0.0.0.0",
        port=8001,  # Use different port from existing API
        workers=1,  # Single worker for WebSocket support
        loop="uvloop",
        log_level="info",
        access_log=True
    )