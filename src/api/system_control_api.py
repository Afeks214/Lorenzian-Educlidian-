"""
System Control API for GrandModel
=================================

REST API endpoints for system control operations including:
- System ON/OFF controls
- Emergency stop functionality
- Status monitoring
- Health checks
- Activity logging
- Authentication and authorization

This module provides secure, authenticated endpoints for controlling
the GrandModel trading system with comprehensive audit logging.
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# System state management
class SystemState(str, Enum):
    """System state enumeration"""
    OFF = "OFF"
    ON = "ON"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MAINTENANCE = "MAINTENANCE"

class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Pydantic models
class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: SystemState
    timestamp: datetime
    uptime_seconds: Optional[int] = None
    version: str
    components: Dict[str, Any]
    performance_metrics: Dict[str, float]
    last_operation: Optional[str] = None
    last_operation_time: Optional[datetime] = None
    message: Optional[str] = None

class SystemControlRequest(BaseModel):
    """System control request model"""
    action: str = Field(..., description="Action to perform (on/off/emergency)")
    reason: Optional[str] = Field(None, description="Reason for the action")
    force: bool = Field(False, description="Force the action even if unsafe")
    timeout_seconds: Optional[int] = Field(30, description="Timeout for the operation")

class SystemControlResponse(BaseModel):
    """System control response model"""
    success: bool
    message: str
    previous_state: SystemState
    new_state: SystemState
    timestamp: datetime
    operation_id: str
    estimated_completion_time: Optional[datetime] = None

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: HealthStatus
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    overall_score: float
    issues: List[str]
    recommendations: List[str]

class ActivityLogEntry(BaseModel):
    """Activity log entry model"""
    timestamp: datetime
    level: LogLevel
    component: str
    operation: str
    user: str
    ip_address: str
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None

class ActivityLogResponse(BaseModel):
    """Activity log response model"""
    entries: List[ActivityLogEntry]
    total_count: int
    page: int
    page_size: int
    has_more: bool

class SystemControlManager:
    """
    System control manager for handling system operations
    """
    
    def __init__(self):
        self.current_state = SystemState.OFF
        self.state_transition_time = datetime.now()
        self.system_start_time = None
        self.activity_log = []
        self.component_status = {}
        self.performance_metrics = {}
        self.active_operations = {}
        self.emergency_stop_active = False
        
        # Initialize component status
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize component status tracking"""
        self.component_status = {
            "marl_engine": {"status": "stopped", "health": "unknown", "last_check": datetime.now()},
            "risk_manager": {"status": "stopped", "health": "unknown", "last_check": datetime.now()},
            "execution_engine": {"status": "stopped", "health": "unknown", "last_check": datetime.now()},
            "data_pipeline": {"status": "stopped", "health": "unknown", "last_check": datetime.now()},
            "monitoring_system": {"status": "stopped", "health": "unknown", "last_check": datetime.now()},
            "database": {"status": "unknown", "health": "unknown", "last_check": datetime.now()},
            "redis_cache": {"status": "unknown", "health": "unknown", "last_check": datetime.now()},
            "event_bus": {"status": "unknown", "health": "unknown", "last_check": datetime.now()},
        }
        
    def _log_activity(self, level: LogLevel, component: str, operation: str, 
                     user: str, ip_address: str, message: str, 
                     details: Optional[Dict[str, Any]] = None,
                     duration_ms: Optional[float] = None):
        """Log system activity"""
        entry = ActivityLogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            operation=operation,
            user=user,
            ip_address=ip_address,
            message=message,
            details=details,
            duration_ms=duration_ms
        )
        
        self.activity_log.append(entry)
        
        # Keep only last 10000 entries
        if len(self.activity_log) > 10000:
            self.activity_log = self.activity_log[-10000:]
            
        # Log to standard logger
        log_method = getattr(logger, level.lower())
        log_method(f"[{component}] {operation}: {message}", extra={"user": user, "ip": ip_address})
        
    async def get_system_status(self) -> SystemStatusResponse:
        """Get current system status"""
        uptime = None
        if self.system_start_time:
            uptime = int((datetime.now() - self.system_start_time).total_seconds())
            
        # Update performance metrics
        self._update_performance_metrics()
        
        return SystemStatusResponse(
            status=self.current_state,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            version="1.0.0",
            components=self.component_status,
            performance_metrics=self.performance_metrics,
            last_operation=self.activity_log[-1].operation if self.activity_log else None,
            last_operation_time=self.activity_log[-1].timestamp if self.activity_log else None,
            message=self._get_status_message()
        )
        
    async def turn_system_on(self, request: SystemControlRequest, user: str, ip_address: str) -> SystemControlResponse:
        """Turn system ON"""
        operation_id = f"turn_on_{int(time.time())}"
        start_time = time.time()
        
        try:
            if self.current_state == SystemState.ON:
                return SystemControlResponse(
                    success=False,
                    message="System is already ON",
                    previous_state=self.current_state,
                    new_state=self.current_state,
                    timestamp=datetime.now(),
                    operation_id=operation_id
                )
                
            if self.emergency_stop_active and not request.force:
                return SystemControlResponse(
                    success=False,
                    message="Emergency stop is active. Use force=true to override.",
                    previous_state=self.current_state,
                    new_state=self.current_state,
                    timestamp=datetime.now(),
                    operation_id=operation_id
                )
                
            self._log_activity(
                LogLevel.INFO, "system_control", "turn_on_start", 
                user, ip_address, f"Starting system turn-on sequence. Reason: {request.reason}"
            )
            
            previous_state = self.current_state
            self.current_state = SystemState.STARTING
            self.state_transition_time = datetime.now()
            
            # Simulate system startup sequence
            await self._startup_sequence(request.timeout_seconds or 30)
            
            self.current_state = SystemState.ON
            self.system_start_time = datetime.now()
            self.emergency_stop_active = False
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.INFO, "system_control", "turn_on_complete", 
                user, ip_address, "System successfully turned ON",
                details={"reason": request.reason, "force": request.force},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=True,
                message="System successfully turned ON",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
        except Exception as e:
            self.current_state = SystemState.OFF
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.ERROR, "system_control", "turn_on_failed", 
                user, ip_address, f"Failed to turn system ON: {str(e)}",
                details={"error": str(e), "reason": request.reason},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=False,
                message=f"Failed to turn system ON: {str(e)}",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
    async def turn_system_off(self, request: SystemControlRequest, user: str, ip_address: str) -> SystemControlResponse:
        """Turn system OFF"""
        operation_id = f"turn_off_{int(time.time())}"
        start_time = time.time()
        
        try:
            if self.current_state == SystemState.OFF:
                return SystemControlResponse(
                    success=False,
                    message="System is already OFF",
                    previous_state=self.current_state,
                    new_state=self.current_state,
                    timestamp=datetime.now(),
                    operation_id=operation_id
                )
                
            self._log_activity(
                LogLevel.INFO, "system_control", "turn_off_start", 
                user, ip_address, f"Starting system turn-off sequence. Reason: {request.reason}"
            )
            
            previous_state = self.current_state
            self.current_state = SystemState.STOPPING
            self.state_transition_time = datetime.now()
            
            # Simulate system shutdown sequence
            await self._shutdown_sequence(request.timeout_seconds or 30)
            
            self.current_state = SystemState.OFF
            self.system_start_time = None
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.INFO, "system_control", "turn_off_complete", 
                user, ip_address, "System successfully turned OFF",
                details={"reason": request.reason, "force": request.force},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=True,
                message="System successfully turned OFF",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.ERROR, "system_control", "turn_off_failed", 
                user, ip_address, f"Failed to turn system OFF: {str(e)}",
                details={"error": str(e), "reason": request.reason},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=False,
                message=f"Failed to turn system OFF: {str(e)}",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
    async def emergency_stop(self, request: SystemControlRequest, user: str, ip_address: str) -> SystemControlResponse:
        """Emergency stop system"""
        operation_id = f"emergency_stop_{int(time.time())}"
        start_time = time.time()
        
        try:
            self._log_activity(
                LogLevel.CRITICAL, "system_control", "emergency_stop_start", 
                user, ip_address, f"EMERGENCY STOP initiated. Reason: {request.reason}"
            )
            
            previous_state = self.current_state
            self.current_state = SystemState.EMERGENCY_STOP
            self.state_transition_time = datetime.now()
            self.emergency_stop_active = True
            
            # Immediately stop all operations
            await self._emergency_shutdown_sequence()
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.CRITICAL, "system_control", "emergency_stop_complete", 
                user, ip_address, "EMERGENCY STOP completed",
                details={"reason": request.reason, "previous_state": previous_state.value},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=True,
                message="EMERGENCY STOP completed successfully",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self._log_activity(
                LogLevel.CRITICAL, "system_control", "emergency_stop_failed", 
                user, ip_address, f"EMERGENCY STOP failed: {str(e)}",
                details={"error": str(e), "reason": request.reason},
                duration_ms=duration_ms
            )
            
            return SystemControlResponse(
                success=False,
                message=f"EMERGENCY STOP failed: {str(e)}",
                previous_state=previous_state,
                new_state=self.current_state,
                timestamp=datetime.now(),
                operation_id=operation_id
            )
            
    async def get_health_status(self) -> HealthCheckResponse:
        """Get system health status"""
        components = {}
        issues = []
        recommendations = []
        overall_score = 0.0
        
        # Check each component
        for component, status in self.component_status.items():
            component_health = await self._check_component_health(component)
            components[component] = component_health
            
            if component_health["status"] == "unhealthy":
                issues.append(f"{component} is unhealthy: {component_health.get('message', 'Unknown issue')}")
            elif component_health["status"] == "degraded":
                issues.append(f"{component} is degraded: {component_health.get('message', 'Performance issues')}")
                
        # Calculate overall score
        healthy_components = sum(1 for c in components.values() if c["status"] == "healthy")
        overall_score = (healthy_components / len(components)) * 100 if components else 0
        
        # Determine overall health
        if overall_score >= 90:
            overall_health = HealthStatus.HEALTHY
        elif overall_score >= 70:
            overall_health = HealthStatus.DEGRADED
        elif overall_score >= 50:
            overall_health = HealthStatus.UNHEALTHY
        else:
            overall_health = HealthStatus.UNKNOWN
            
        # Generate recommendations
        if overall_score < 90:
            recommendations.append("Consider reviewing component health and performance metrics")
        if len(issues) > 0:
            recommendations.append("Address the identified issues to improve system health")
            
        return HealthCheckResponse(
            status=overall_health,
            timestamp=datetime.now(),
            components=components,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
        
    def get_activity_logs(self, page: int = 1, page_size: int = 100, 
                         level: Optional[LogLevel] = None,
                         component: Optional[str] = None,
                         operation: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> ActivityLogResponse:
        """Get activity logs with filtering"""
        
        # Filter logs
        filtered_logs = self.activity_log
        
        if level:
            filtered_logs = [log for log in filtered_logs if log.level == level]
        if component:
            filtered_logs = [log for log in filtered_logs if log.component == component]
        if operation:
            filtered_logs = [log for log in filtered_logs if log.operation == operation]
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
            
        # Sort by timestamp (newest first)
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_logs = filtered_logs[start_idx:end_idx]
        
        return ActivityLogResponse(
            entries=page_logs,
            total_count=len(filtered_logs),
            page=page,
            page_size=page_size,
            has_more=end_idx < len(filtered_logs)
        )
        
    async def _startup_sequence(self, timeout_seconds: int):
        """Simulate system startup sequence"""
        components = list(self.component_status.keys())
        
        for component in components:
            self.component_status[component]["status"] = "starting"
            await asyncio.sleep(0.5)  # Simulate startup time
            self.component_status[component]["status"] = "running"
            self.component_status[component]["health"] = "healthy"
            self.component_status[component]["last_check"] = datetime.now()
            
    async def _shutdown_sequence(self, timeout_seconds: int):
        """Simulate system shutdown sequence"""
        components = list(self.component_status.keys())
        
        for component in reversed(components):
            self.component_status[component]["status"] = "stopping"
            await asyncio.sleep(0.3)  # Simulate shutdown time
            self.component_status[component]["status"] = "stopped"
            self.component_status[component]["health"] = "unknown"
            self.component_status[component]["last_check"] = datetime.now()
            
    async def _emergency_shutdown_sequence(self):
        """Simulate emergency shutdown sequence"""
        for component in self.component_status:
            self.component_status[component]["status"] = "emergency_stop"
            self.component_status[component]["health"] = "unknown"
            self.component_status[component]["last_check"] = datetime.now()
            
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component"""
        status = self.component_status.get(component, {})
        
        # Simulate health check
        if status.get("status") == "running":
            return {
                "status": "healthy",
                "message": "Component is running normally",
                "last_check": datetime.now(),
                "metrics": {
                    "cpu_usage": 25.5,
                    "memory_usage": 45.2,
                    "response_time_ms": 12.3
                }
            }
        elif status.get("status") == "starting":
            return {
                "status": "degraded",
                "message": "Component is starting up",
                "last_check": datetime.now(),
                "metrics": {}
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Component is not running",
                "last_check": datetime.now(),
                "metrics": {}
            }
            
    def _update_performance_metrics(self):
        """Update system performance metrics"""
        self.performance_metrics = {
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 67.8,
            "disk_usage_percent": 32.1,
            "network_io_mbps": 156.7,
            "active_connections": 245,
            "requests_per_second": 1250,
            "avg_response_time_ms": 18.5,
            "error_rate_percent": 0.12
        }
        
    def _get_status_message(self) -> str:
        """Get human-readable status message"""
        if self.current_state == SystemState.ON:
            return "System is running normally"
        elif self.current_state == SystemState.OFF:
            return "System is offline"
        elif self.current_state == SystemState.STARTING:
            return "System is starting up..."
        elif self.current_state == SystemState.STOPPING:
            return "System is shutting down..."
        elif self.current_state == SystemState.EMERGENCY_STOP:
            return "EMERGENCY STOP is active - system is locked"
        elif self.current_state == SystemState.MAINTENANCE:
            return "System is in maintenance mode"
        else:
            return "System status unknown"

# Global system control manager
system_control = SystemControlManager()

# FastAPI app
app = FastAPI(
    title="GrandModel System Control API",
    description="System control API for GrandModel trading system",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify authentication token and return user info
    """
    # In production, this should verify JWT token against auth service
    token = credentials.credentials
    
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
        
    # Mock user for demonstration
    return {
        "user_id": "admin",
        "username": "admin",
        "permissions": ["system_control", "read", "write"],
        "ip_address": "127.0.0.1"
    }

# API Endpoints

@app.get("/api/system/status", response_model=SystemStatusResponse, tags=["System Control"])
@limiter.limit("30/minute")
async def get_system_status(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current system status
    """
    return await system_control.get_system_status()

@app.post("/api/system/on", response_model=SystemControlResponse, tags=["System Control"])
@limiter.limit("10/minute")
async def turn_system_on(
    control_request: SystemControlRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Turn system ON
    """
    if "system_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    ip_address = request.client.host
    return await system_control.turn_system_on(control_request, user["username"], ip_address)

@app.post("/api/system/off", response_model=SystemControlResponse, tags=["System Control"])
@limiter.limit("10/minute")
async def turn_system_off(
    control_request: SystemControlRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Turn system OFF
    """
    if "system_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    ip_address = request.client.host
    return await system_control.turn_system_off(control_request, user["username"], ip_address)

@app.post("/api/system/emergency", response_model=SystemControlResponse, tags=["System Control"])
@limiter.limit("5/minute")
async def emergency_stop(
    control_request: SystemControlRequest,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Emergency stop system
    """
    if "system_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    ip_address = request.client.host
    return await system_control.emergency_stop(control_request, user["username"], ip_address)

@app.get("/api/system/health", response_model=HealthCheckResponse, tags=["System Control"])
@limiter.limit("60/minute")
async def get_health_status(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get system health status
    """
    return await system_control.get_health_status()

@app.get("/api/system/logs", response_model=ActivityLogResponse, tags=["System Control"])
@limiter.limit("30/minute")
async def get_activity_logs(
    request: Request,
    page: int = 1,
    page_size: int = 100,
    level: Optional[LogLevel] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get recent activity logs
    """
    if page_size > 1000:
        raise HTTPException(status_code=400, detail="Page size cannot exceed 1000")
        
    return system_control.get_activity_logs(
        page=page,
        page_size=page_size,
        level=level,
        component=component,
        operation=operation
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)