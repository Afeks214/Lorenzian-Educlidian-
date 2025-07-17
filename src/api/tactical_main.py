"""
Tactical MARL System - FastAPI Server

High-performance FastAPI server for tactical 5-minute MARL system
with sub-100ms latency requirements and Redis Streams integration.
"""

import asyncio
import time
import logging
import signal
import sys
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.monitoring.tactical_metrics import tactical_metrics, get_tactical_metrics, get_tactical_metrics_content_type
from src.monitoring.tactical_health import tactical_health_monitor
from src.monitoring.service_health_state_machine import ServiceHealthStateMachine
from src.tactical.controller import TacticalMARLController
from src.tactical.environment import TacticalEnvironment
from src.tactical.aggregator import TacticalDecisionAggregator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Pydantic models for API
class SynergyEventData(BaseModel):
    """SYNERGY_DETECTED event data structure."""
    synergy_type: str = Field(..., description="Type of synergy detected (TYPE_1, TYPE_2, etc.)")
    direction: int = Field(..., description="Direction: 1 for long, -1 for short")
    confidence: float = Field(..., description="Confidence level (0-1)")
    signal_sequence: List[Dict[str, Any]] = Field(default_factory=list, description="Signal sequence data")
    market_context: Dict[str, Any] = Field(default_factory=dict, description="Market context information")
    correlation_id: str = Field(..., description="Correlation ID for tracking")
    timestamp: float = Field(..., description="Event timestamp")

class DecisionRequest(BaseModel):
    """Manual decision request structure."""
    matrix_state: List[List[float]] = Field(..., description="60x7 matrix state")
    synergy_context: Optional[SynergyEventData] = Field(None, description="Synergy context")
    override_params: Dict[str, Any] = Field(default_factory=dict, description="Parameter overrides")
    correlation_id: str = Field(..., description="Correlation ID for tracking")

class DecisionResponse(BaseModel):
    """Decision response structure."""
    decision: Dict[str, Any] = Field(..., description="Final decision")
    agent_breakdown: Dict[str, Any] = Field(..., description="Individual agent decisions")
    timing: Dict[str, float] = Field(..., description="Timing information")
    correlation_id: str = Field(..., description="Correlation ID")

class HealthResponse(BaseModel):
    """Health check response structure."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: List[Dict[str, Any]] = Field(..., description="Component health details")
    performance_summary: Dict[str, Any] = Field(..., description="Performance summary")
    recommendations: List[str] = Field(default_factory=list, description="Health recommendations")

# Global application state
app_state = {
    "tactical_controller": None,
    "tactical_environment": None,
    "decision_aggregator": None,
    "health_state_machine": None,
    "startup_time": time.time(),
    "requests_processed": 0,
    "shutdown_initiated": False
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting Tactical MARL System")
    
    try:
        # Initialize tactical components
        app_state["tactical_controller"] = TacticalMARLController()
        app_state["tactical_environment"] = TacticalEnvironment()
        app_state["decision_aggregator"] = TacticalDecisionAggregator()
        
        # Initialize health state machine
        app_state["health_state_machine"] = ServiceHealthStateMachine("tactical-marl")
        await app_state["health_state_machine"].initialize()
        await app_state["health_state_machine"].start_monitoring()
        
        # Start metrics server
        tactical_metrics.start_metrics_server()
        
        # Start background tasks
        asyncio.create_task(background_health_monitoring())
        asyncio.create_task(background_metrics_collection())
        
        # Update system info
        tactical_metrics.update_system_info({
            "version": "1.0.0",
            "service": "tactical-marl",
            "startup_time": str(time.time()),
            "python_version": sys.version,
            "environment": "production"
        })
        
        logger.info("✅ Tactical MARL System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start Tactical MARL System: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Tactical MARL System")
        app_state["shutdown_initiated"] = True
        
        # Cleanup resources
        if app_state["tactical_controller"]:
            await app_state["tactical_controller"].cleanup()
        
        if app_state["health_state_machine"]:
            await app_state["health_state_machine"].cleanup()
        
        logger.info("✅ Tactical MARL System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Tactical MARL System",
    description="High-frequency tactical trading system with sub-100ms latency",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# SECURITY: Secure CORS configuration with trusted domains only
TRUSTED_ORIGINS = [
    "http://localhost:3000",  # Development frontend
    "http://localhost:8000",  # API documentation
    "http://localhost:8001",  # Tactical API
    "https://grandmodel.trading",  # Production domain (example)
    "https://api.grandmodel.trading",  # Production API domain (example)
]

# Get additional trusted origins from environment
if additional_origins := os.getenv("TRUSTED_ORIGINS"):
    TRUSTED_ORIGINS.extend(additional_origins.split(","))

# Add secure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=TRUSTED_ORIGINS,  # SECURITY: No wildcards allowed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specific methods only
    allow_headers=[
        "Authorization",
        "Content-Type", 
        "X-Correlation-ID",
        "X-Request-ID",
        "Accept",
        "Origin",
        "User-Agent"
    ],  # Specific headers only
    expose_headers=[
        "X-Process-Time",
        "X-Correlation-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset"
    ]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# SECURITY: Add comprehensive security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add comprehensive security headers to all responses."""
    response = await call_next(request)
    
    # Security headers for production-grade security
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # HSTS for HTTPS (only add if using HTTPS)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    
    # Content Security Policy (CSP)
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # Remove server information disclosure
    response.headers.pop("Server", None)
    response.headers.pop("X-Powered-By", None)
    
    return response

# SECURITY: Add trusted host middleware
TRUSTED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "grandmodel.trading",  # Production domain
    "api.grandmodel.trading",  # API domain
]

# Get additional trusted hosts from environment
if additional_hosts := os.getenv("TRUSTED_HOSTS"):
    TRUSTED_HOSTS.extend(additional_hosts.split(","))

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=TRUSTED_HOSTS
)

# SECURITY: Initialize rate limiter
from src.security.rate_limiter import get_rate_limiter

# Request timing and security middleware
@app.middleware("http")
async def timing_and_security_middleware(request: Request, call_next):
    """Add request timing, correlation ID tracking, and security checks."""
    start_time = time.perf_counter()
    correlation_id = request.headers.get("X-Correlation-ID", f"req-{time.time()}")
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    try:
        # SECURITY: Apply rate limiting
        rate_limiter = await get_rate_limiter()
        if rate_limiter:
            # Check endpoint-specific rate limits
            endpoint = request.url.path.strip("/").split("/")[0] or "root"
            allowed, metadata = await rate_limiter.check_rate_limit(endpoint, request)
            
            if not allowed:
                return rate_limiter._rate_limit_exceeded_response(metadata)
        
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Correlation-ID"] = correlation_id
        
        # SECURITY: Add rate limit headers if available
        if 'metadata' in locals() and metadata:
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", ""))
        
        # Record metrics
        tactical_metrics.tactical_events_processed.labels(
            event_type="http_request",
            source="api",
            status="success"
        ).inc()
        
        # Update request counter
        app_state["requests_processed"] += 1
        
        return response
        
    except Exception as e:
        process_time = time.perf_counter() - start_time
        
        # Record error metrics
        tactical_metrics.record_error(
            error_type="http_request_failed",
            component="api",
            severity="error"
        )
        
        logger.error(f"Request failed: {e}", correlation_id=correlation_id)
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for tactical system."""
    try:
        async with tactical_metrics.measure_pipeline_component("health_check", "full_check"):
            health_data = await tactical_health_monitor.get_detailed_health()
            
            return HealthResponse(
                status=health_data["status"],
                timestamp=health_data["timestamp"],
                components=health_data["components"],
                performance_summary=health_data["performance_summary"],
                recommendations=health_data.get("recommendations", [])
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/health/state")
async def health_state():
    """Get advanced health state from state machine."""
    try:
        if not app_state["health_state_machine"]:
            raise HTTPException(status_code=503, detail="Health state machine not initialized")
        
        health_status = await app_state["health_state_machine"].get_health_status()
        return health_status
        
    except Exception as e:
        logger.error(f"Health state check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health state check failed: {str(e)}")

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    metrics_data = get_tactical_metrics()
    return PlainTextResponse(
        content=metrics_data,
        media_type=get_tactical_metrics_content_type()
    )

# Manual decision endpoint
@app.post("/decide", response_model=DecisionResponse)
async def make_decision(request: DecisionRequest):
    """Make tactical trading decision with atomic processing and race condition protection."""
    start_time = time.perf_counter()
    
    # Import distributed lock manager
    from src.tactical.distributed_lock import get_lock_manager
    
    try:
        # Get lock manager
        lock_manager = await get_lock_manager()
        
        # Validate request before acquiring lock
        if not request.matrix_state or len(request.matrix_state) != 60:
            raise HTTPException(
                status_code=400,
                detail="Invalid matrix state: must be 60x7 matrix"
            )
        
        if any(len(row) != 7 for row in request.matrix_state):
            raise HTTPException(
                status_code=400,
                detail="Invalid matrix state: each row must have 7 features"
            )
        
        # Validate correlation ID format and uniqueness
        if not request.correlation_id or len(request.correlation_id.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="correlation_id is required and cannot be empty"
            )
        
        # Atomic decision processing with distributed locking
        async with lock_manager.decision_lock(
            correlation_id=request.correlation_id,
            timeout=30.0,  # 30 second lock timeout
            max_wait=10.0   # Maximum 10 seconds to wait for lock
        ) as lock_result:
            
            if not lock_result.acquired:
                # Lock acquisition failed - return appropriate error
                if "already in use" in (lock_result.error or ""):
                    raise HTTPException(
                        status_code=409,  # Conflict
                        detail=f"Request with correlation_id '{request.correlation_id}' is already being processed"
                    )
                else:
                    raise HTTPException(
                        status_code=503,  # Service Unavailable
                        detail=f"Unable to acquire processing lock: {lock_result.error}"
                    )
            
            # Lock acquired successfully - process decision atomically
            logger.info(
                "Processing decision with acquired lock",
                correlation_id=request.correlation_id,
                lock_id=lock_result.lock_id
            )
            
            # Process decision within the lock
            async with tactical_metrics.measure_decision_latency(
                synergy_type=request.synergy_context.synergy_type if request.synergy_context else "manual",
                decision_type="manual"
            ):
                decision_result = await app_state["tactical_controller"].process_decision_request(
                    matrix_state=request.matrix_state,
                    synergy_context=request.synergy_context.dict() if request.synergy_context else None,
                    override_params=request.override_params,
                    correlation_id=request.correlation_id
                )
            
            # Calculate timing
            total_time = time.perf_counter() - start_time
            
            # Record metrics
            tactical_metrics.record_synergy_response(
                synergy_type=request.synergy_context.synergy_type if request.synergy_context else "manual",
                response_action=decision_result["decision"]["action"],
                confidence=decision_result["decision"]["confidence"]
            )
            
            logger.info(
                "Decision processing completed successfully",
                correlation_id=request.correlation_id,
                processing_time_ms=total_time * 1000,
                action=decision_result["decision"]["action"]
            )
            
            return DecisionResponse(
                decision=decision_result["decision"],
                agent_breakdown=decision_result["agent_breakdown"],
                timing={
                    "total_ms": total_time * 1000,
                    "inference_ms": decision_result["timing"]["inference_ms"],
                    "aggregation_ms": decision_result["timing"]["aggregation_ms"],
                    "lock_acquisition_ms": (lock_result.expiry_time - start_time) * 1000 if lock_result.expiry_time else 0
                },
                correlation_id=request.correlation_id
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Decision processing failed",
            correlation_id=request.correlation_id,
            error=str(e),
            error_type=type(e).__name__
        )
        tactical_metrics.record_error(
            error_type="decision_processing_failed",
            component="controller",
            severity="error"
        )
        raise HTTPException(
            status_code=500, 
            detail="Decision processing failed due to internal error"
        )

# System status endpoint
@app.get("/status")
async def system_status():
    """Get system status information."""
    uptime = time.time() - app_state["startup_time"]
    
    return {
        "service": "tactical-marl",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "requests_processed": app_state["requests_processed"],
        "components": {
            "tactical_controller": app_state["tactical_controller"] is not None,
            "tactical_environment": app_state["tactical_environment"] is not None,
            "decision_aggregator": app_state["decision_aggregator"] is not None,
        },
        "performance": tactical_metrics.get_performance_summary(),
        "timestamp": time.time()
    }

# Performance endpoint
@app.get("/performance")
async def performance_metrics():
    """Get detailed performance metrics."""
    return {
        "latency_summary": tactical_metrics.get_performance_summary(),
        "system_metrics": {
            "uptime": time.time() - app_state["startup_time"],
            "requests_processed": app_state["requests_processed"],
            "memory_usage": "N/A",  # Would be implemented with actual monitoring
        },
        "timestamp": time.time()
    }

# Background tasks
async def background_health_monitoring():
    """Background task for continuous health monitoring."""
    while not app_state["shutdown_initiated"]:
        try:
            await tactical_health_monitor.check_all_components()
            await asyncio.sleep(10)  # Check every 10 seconds
        except Exception as e:
            logger.error(f"Background health monitoring failed: {e}")
            await asyncio.sleep(30)  # Retry after 30 seconds on error

async def background_metrics_collection():
    """Background task for metrics collection."""
    while not app_state["shutdown_initiated"]:
        try:
            # Update throughput metrics
            tactical_metrics.update_throughput()
            
            # Update system metrics
            tactical_metrics.update_system_info({
                "requests_processed": str(app_state["requests_processed"]),
                "uptime_seconds": str(time.time() - app_state["startup_time"])
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Background metrics collection failed: {e}")
            await asyncio.sleep(15)  # Retry after 15 seconds on error

# Signal handlers
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    app_state["shutdown_initiated"] = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main entry point
if __name__ == "__main__":
    # Configure uvicorn for high performance
    uvicorn.run(
        "src.api.tactical_main:app",
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for consistency
        loop="asyncio",
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False,
        # Performance optimizations
        backlog=2048,
        max_requests=10000,
        max_requests_jitter=1000,
        # Timeout settings for high-frequency trading
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )