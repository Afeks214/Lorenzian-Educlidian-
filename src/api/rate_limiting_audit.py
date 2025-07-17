"""
Advanced Rate Limiting and Audit Logging System
================================================

Comprehensive rate limiting and audit logging system for system control API including:
- Multi-tier rate limiting (per-user, per-IP, per-endpoint)
- Adaptive rate limiting based on system load
- Distributed rate limiting with Redis
- Comprehensive audit logging
- Real-time monitoring and alerting
- Rate limit bypass for emergency situations
- Detailed analytics and reporting

This module provides enterprise-grade rate limiting and audit capabilities
for secure system control operations.
"""

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import secrets

from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import redis.asyncio as redis
from pydantic import BaseModel, Field
import psutil
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.api.authentication import UserInfo, verify_token
from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/grandmodel")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Rate limiting configuration
@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    
    # Per-user limits
    user_requests_per_minute: int = 60
    user_requests_per_hour: int = 1000
    user_requests_per_day: int = 10000
    
    # Per-IP limits
    ip_requests_per_minute: int = 100
    ip_requests_per_hour: int = 5000
    
    # Per-endpoint limits
    endpoint_requests_per_minute: Dict[str, int] = None
    
    # Emergency bypass
    emergency_bypass_enabled: bool = True
    emergency_bypass_users: List[str] = None
    
    # Adaptive rate limiting
    adaptive_enabled: bool = True
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    
    # Penalties
    violation_penalty_minutes: int = 5
    repeated_violation_multiplier: float = 2.0
    
    def __post_init__(self):
        if self.endpoint_requests_per_minute is None:
            self.endpoint_requests_per_minute = {
                "/api/system/on": 5,
                "/api/system/off": 5,
                "/api/system/emergency": 10,
                "/api/system/status": 120,
                "/api/system/health": 120,
                "/api/system/logs": 30
            }
        
        if self.emergency_bypass_users is None:
            self.emergency_bypass_users = ["admin", "emergency_user"]

class RateLimitType(str, Enum):
    """Rate limit types"""
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    GLOBAL = "global"

class AuditEventType(str, Enum):
    """Audit event types"""
    LOGIN = "login"
    LOGOUT = "logout"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    EMERGENCY_STOP = "emergency_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    PERMISSION_CHANGE = "permission_change"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"

class AuditSeverity(str, Enum):
    """Audit severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Database models
class RateLimitViolation(Base):
    """Rate limit violation record"""
    __tablename__ = "rate_limit_violations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=True)
    ip_address = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    limit_type = Column(String, nullable=False)
    limit_value = Column(Integer, nullable=False)
    actual_value = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_agent = Column(String, nullable=True)
    penalty_applied = Column(Boolean, default=False)
    penalty_duration = Column(Integer, nullable=True)

class AuditLog(Base):
    """Audit log record"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=True)
    username = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    action = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON, nullable=True)
    request_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    outcome = Column(String, nullable=False)  # success, failure, error
    response_time_ms = Column(Float, nullable=True)
    
class SystemMetrics(Base):
    """System metrics for adaptive rate limiting"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cpu_usage = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=False)
    disk_usage = Column(Float, nullable=False)
    network_io = Column(Float, nullable=False)
    active_connections = Column(Integer, nullable=False)
    request_rate = Column(Float, nullable=False)
    error_rate = Column(Float, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit_type: RateLimitType
    identifier: str
    limit_value: int
    current_value: int
    window_start: datetime
    window_end: datetime
    penalty_until: Optional[datetime] = None
    
class AuditEntry(BaseModel):
    """Audit entry model"""
    user_id: Optional[str]
    username: Optional[str]
    event_type: AuditEventType
    resource: str
    action: str
    severity: AuditSeverity
    ip_address: str
    user_agent: Optional[str]
    timestamp: datetime
    details: Optional[Dict[str, Any]]
    request_id: Optional[str]
    session_id: Optional[str]
    outcome: str
    response_time_ms: Optional[float]

class RateLimitingAuditSystem:
    """
    Advanced rate limiting and audit logging system
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.system_metrics_cache: Dict[str, float] = {}
        self.emergency_mode = False
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
    
    def get_db(self) -> Session:
        """Get database session"""
        db = SessionLocal()
        try:
            return db
        finally:
            pass  # Don't close here, let caller handle it
    
    async def check_rate_limit(self, 
                              request: Request,
                              user: Optional[UserInfo] = None,
                              endpoint: Optional[str] = None) -> Tuple[bool, Optional[RateLimitInfo]]:
        """
        Check if request is within rate limits
        
        Returns:
            (allowed, rate_limit_info)
        """
        await self.init_redis()
        
        if not self.redis_client:
            logger.warning("Redis not available, allowing request")
            return True, None
        
        # Check emergency bypass
        if self.emergency_mode or (user and user.username in self.config.emergency_bypass_users):
            return True, None
        
        # Get identifiers
        ip_address = request.client.host
        user_id = user.user_id if user else None
        endpoint_path = endpoint or str(request.url.path)
        
        # Check system load for adaptive rate limiting
        if self.config.adaptive_enabled:
            await self._update_system_metrics()
            if await self._should_apply_adaptive_limits():
                # Reduce limits by 50% during high load
                return await self._check_adaptive_rate_limit(request, user, endpoint_path)
        
        # Check per-user limits
        if user_id:
            user_allowed, user_info = await self._check_user_rate_limit(user_id)
            if not user_allowed:
                await self._record_violation(user_id, ip_address, endpoint_path, RateLimitType.USER, user_info)
                return False, user_info
        
        # Check per-IP limits
        ip_allowed, ip_info = await self._check_ip_rate_limit(ip_address)
        if not ip_allowed:
            await self._record_violation(user_id, ip_address, endpoint_path, RateLimitType.IP, ip_info)
            return False, ip_info
        
        # Check per-endpoint limits
        endpoint_allowed, endpoint_info = await self._check_endpoint_rate_limit(endpoint_path, ip_address)
        if not endpoint_allowed:
            await self._record_violation(user_id, ip_address, endpoint_path, RateLimitType.ENDPOINT, endpoint_info)
            return False, endpoint_info
        
        # Record successful request
        await self._record_successful_request(user_id, ip_address, endpoint_path)
        
        return True, None
    
    async def _check_user_rate_limit(self, user_id: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check user-specific rate limits"""
        current_time = time.time()
        
        # Check minute limit
        minute_key = f"rate_limit:user:{user_id}:minute:{int(current_time // 60)}"
        minute_count = await self.redis_client.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0
        
        if minute_count >= self.config.user_requests_per_minute:
            return False, RateLimitInfo(
                limit_type=RateLimitType.USER,
                identifier=user_id,
                limit_value=self.config.user_requests_per_minute,
                current_value=minute_count,
                window_start=datetime.fromtimestamp(int(current_time // 60) * 60),
                window_end=datetime.fromtimestamp((int(current_time // 60) + 1) * 60)
            )
        
        # Check hour limit
        hour_key = f"rate_limit:user:{user_id}:hour:{int(current_time // 3600)}"
        hour_count = await self.redis_client.get(hour_key)
        hour_count = int(hour_count) if hour_count else 0
        
        if hour_count >= self.config.user_requests_per_hour:
            return False, RateLimitInfo(
                limit_type=RateLimitType.USER,
                identifier=user_id,
                limit_value=self.config.user_requests_per_hour,
                current_value=hour_count,
                window_start=datetime.fromtimestamp(int(current_time // 3600) * 3600),
                window_end=datetime.fromtimestamp((int(current_time // 3600) + 1) * 3600)
            )
        
        # Increment counters
        await self.redis_client.incr(minute_key)
        await self.redis_client.expire(minute_key, 60)
        await self.redis_client.incr(hour_key)
        await self.redis_client.expire(hour_key, 3600)
        
        return True, None
    
    async def _check_ip_rate_limit(self, ip_address: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check IP-specific rate limits"""
        current_time = time.time()
        
        # Check minute limit
        minute_key = f"rate_limit:ip:{ip_address}:minute:{int(current_time // 60)}"
        minute_count = await self.redis_client.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0
        
        if minute_count >= self.config.ip_requests_per_minute:
            return False, RateLimitInfo(
                limit_type=RateLimitType.IP,
                identifier=ip_address,
                limit_value=self.config.ip_requests_per_minute,
                current_value=minute_count,
                window_start=datetime.fromtimestamp(int(current_time // 60) * 60),
                window_end=datetime.fromtimestamp((int(current_time // 60) + 1) * 60)
            )
        
        # Check hour limit
        hour_key = f"rate_limit:ip:{ip_address}:hour:{int(current_time // 3600)}"
        hour_count = await self.redis_client.get(hour_key)
        hour_count = int(hour_count) if hour_count else 0
        
        if hour_count >= self.config.ip_requests_per_hour:
            return False, RateLimitInfo(
                limit_type=RateLimitType.IP,
                identifier=ip_address,
                limit_value=self.config.ip_requests_per_hour,
                current_value=hour_count,
                window_start=datetime.fromtimestamp(int(current_time // 3600) * 3600),
                window_end=datetime.fromtimestamp((int(current_time // 3600) + 1) * 3600)
            )
        
        # Increment counters
        await self.redis_client.incr(minute_key)
        await self.redis_client.expire(minute_key, 60)
        await self.redis_client.incr(hour_key)
        await self.redis_client.expire(hour_key, 3600)
        
        return True, None
    
    async def _check_endpoint_rate_limit(self, endpoint: str, ip_address: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check endpoint-specific rate limits"""
        endpoint_limit = self.config.endpoint_requests_per_minute.get(endpoint, 100)
        current_time = time.time()
        
        # Use IP+endpoint combination for endpoint limiting
        minute_key = f"rate_limit:endpoint:{endpoint}:{ip_address}:minute:{int(current_time // 60)}"
        minute_count = await self.redis_client.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0
        
        if minute_count >= endpoint_limit:
            return False, RateLimitInfo(
                limit_type=RateLimitType.ENDPOINT,
                identifier=f"{endpoint}:{ip_address}",
                limit_value=endpoint_limit,
                current_value=minute_count,
                window_start=datetime.fromtimestamp(int(current_time // 60) * 60),
                window_end=datetime.fromtimestamp((int(current_time // 60) + 1) * 60)
            )
        
        # Increment counter
        await self.redis_client.incr(minute_key)
        await self.redis_client.expire(minute_key, 60)
        
        return True, None
    
    async def _check_adaptive_rate_limit(self, request: Request, user: Optional[UserInfo], endpoint: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check rate limits with adaptive scaling"""
        # Reduce limits by 50% during high load
        original_config = self.config
        
        # Create temporary config with reduced limits
        reduced_config = RateLimitConfig(
            user_requests_per_minute=original_config.user_requests_per_minute // 2,
            user_requests_per_hour=original_config.user_requests_per_hour // 2,
            ip_requests_per_minute=original_config.ip_requests_per_minute // 2,
            ip_requests_per_hour=original_config.ip_requests_per_hour // 2,
            endpoint_requests_per_minute={
                k: v // 2 for k, v in original_config.endpoint_requests_per_minute.items()
            }
        )
        
        # Temporarily use reduced config
        self.config = reduced_config
        
        try:
            return await self.check_rate_limit(request, user, endpoint)
        finally:
            # Restore original config
            self.config = original_config
    
    async def _update_system_metrics(self):
        """Update system metrics for adaptive rate limiting"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            self.system_metrics_cache = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "timestamp": time.time()
            }
            
            # Store in database
            db = self.get_db()
            try:
                metrics = SystemMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory_percent,
                    disk_usage=disk_percent,
                    network_io=0.0,  # Placeholder
                    active_connections=0,  # Placeholder
                    request_rate=0.0,  # Placeholder
                    error_rate=0.0  # Placeholder
                )
                db.add(metrics)
                db.commit()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def _should_apply_adaptive_limits(self) -> bool:
        """Check if adaptive rate limiting should be applied"""
        metrics = self.system_metrics_cache
        
        if not metrics or time.time() - metrics["timestamp"] > 60:
            return False
        
        return (
            metrics["cpu_usage"] > self.config.cpu_threshold or
            metrics["memory_usage"] > self.config.memory_threshold
        )
    
    async def _record_violation(self, user_id: Optional[str], ip_address: str, 
                               endpoint: str, limit_type: RateLimitType, 
                               limit_info: RateLimitInfo):
        """Record rate limit violation"""
        
        # Store in database
        db = self.get_db()
        try:
            violation = RateLimitViolation(
                user_id=user_id,
                ip_address=ip_address,
                endpoint=endpoint,
                limit_type=limit_type.value,
                limit_value=limit_info.limit_value,
                actual_value=limit_info.current_value,
                user_agent="",  # Would be populated from request
                penalty_applied=False
            )
            db.add(violation)
            db.commit()
        finally:
            db.close()
        
        # Log violation
        logger.warning(f"Rate limit violation: {limit_type.value} for {limit_info.identifier}")
        
        # Apply penalty if configured
        await self._apply_penalty(limit_info.identifier, limit_type)
    
    async def _apply_penalty(self, identifier: str, limit_type: RateLimitType):
        """Apply penalty for rate limit violation"""
        if not self.redis_client:
            return
        
        # Increase violation count
        violation_key = f"violations:{limit_type.value}:{identifier}"
        violation_count = await self.redis_client.incr(violation_key)
        await self.redis_client.expire(violation_key, 3600)  # 1 hour
        
        # Calculate penalty duration
        base_penalty = self.config.violation_penalty_minutes
        penalty_duration = int(base_penalty * (self.config.repeated_violation_multiplier ** (violation_count - 1)))
        
        # Apply penalty
        penalty_key = f"penalty:{limit_type.value}:{identifier}"
        await self.redis_client.setex(penalty_key, penalty_duration * 60, "1")
        
        logger.info(f"Applied {penalty_duration} minute penalty to {identifier}")
    
    async def _record_successful_request(self, user_id: Optional[str], ip_address: str, endpoint: str):
        """Record successful request for analytics"""
        if not self.redis_client:
            return
        
        # Update request counters for analytics
        current_time = time.time()
        analytics_key = f"analytics:requests:{int(current_time // 60)}"
        
        analytics_data = {
            "user_id": user_id,
            "ip_address": ip_address,
            "endpoint": endpoint,
            "timestamp": current_time
        }
        
        await self.redis_client.lpush(analytics_key, json.dumps(analytics_data))
        await self.redis_client.expire(analytics_key, 3600)  # 1 hour
    
    async def log_audit_event(self, 
                             request: Request,
                             user: Optional[UserInfo],
                             event_type: AuditEventType,
                             resource: str,
                             action: str,
                             severity: AuditSeverity,
                             outcome: str,
                             details: Optional[Dict[str, Any]] = None,
                             response_time_ms: Optional[float] = None):
        """Log audit event"""
        
        # Create audit entry
        audit_entry = AuditEntry(
            user_id=user.user_id if user else None,
            username=user.username if user else None,
            event_type=event_type,
            resource=resource,
            action=action,
            severity=severity,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            timestamp=datetime.utcnow(),
            details=details,
            request_id=request.headers.get("x-request-id"),
            session_id=user.session_id if user else None,
            outcome=outcome,
            response_time_ms=response_time_ms
        )
        
        # Store in database
        db = self.get_db()
        try:
            audit_log = AuditLog(**audit_entry.dict())
            db.add(audit_log)
            db.commit()
        finally:
            db.close()
        
        # Log to application logger
        logger.info(f"AUDIT: {event_type.value} - {action} - {outcome} - User: {user.username if user else 'anonymous'}")
        
        # Store in Redis for real-time monitoring
        await self.init_redis()
        if self.redis_client:
            await self.redis_client.lpush(
                "audit_stream", 
                json.dumps(audit_entry.dict(), default=str)
            )
            await self.redis_client.ltrim("audit_stream", 0, 1000)  # Keep last 1000 entries
    
    async def get_rate_limit_status(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        await self.init_redis()
        
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        current_time = time.time()
        status = {}
        
        # User limits
        user_minute_key = f"rate_limit:user:{user_id}:minute:{int(current_time // 60)}"
        user_hour_key = f"rate_limit:user:{user_id}:hour:{int(current_time // 3600)}"
        
        user_minute_count = await self.redis_client.get(user_minute_key)
        user_hour_count = await self.redis_client.get(user_hour_key)
        
        status["user"] = {
            "minute": {
                "limit": self.config.user_requests_per_minute,
                "current": int(user_minute_count) if user_minute_count else 0,
                "remaining": self.config.user_requests_per_minute - (int(user_minute_count) if user_minute_count else 0)
            },
            "hour": {
                "limit": self.config.user_requests_per_hour,
                "current": int(user_hour_count) if user_hour_count else 0,
                "remaining": self.config.user_requests_per_hour - (int(user_hour_count) if user_hour_count else 0)
            }
        }
        
        # IP limits
        ip_minute_key = f"rate_limit:ip:{ip_address}:minute:{int(current_time // 60)}"
        ip_hour_key = f"rate_limit:ip:{ip_address}:hour:{int(current_time // 3600)}"
        
        ip_minute_count = await self.redis_client.get(ip_minute_key)
        ip_hour_count = await self.redis_client.get(ip_hour_key)
        
        status["ip"] = {
            "minute": {
                "limit": self.config.ip_requests_per_minute,
                "current": int(ip_minute_count) if ip_minute_count else 0,
                "remaining": self.config.ip_requests_per_minute - (int(ip_minute_count) if ip_minute_count else 0)
            },
            "hour": {
                "limit": self.config.ip_requests_per_hour,
                "current": int(ip_hour_count) if ip_hour_count else 0,
                "remaining": self.config.ip_requests_per_hour - (int(ip_hour_count) if ip_hour_count else 0)
            }
        }
        
        return status
    
    async def get_audit_logs(self, 
                           user_id: Optional[str] = None,
                           event_type: Optional[AuditEventType] = None,
                           severity: Optional[AuditSeverity] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs with filtering"""
        
        db = self.get_db()
        try:
            query = db.query(AuditLog)
            
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if event_type:
                query = query.filter(AuditLog.event_type == event_type.value)
            if severity:
                query = query.filter(AuditLog.severity == severity.value)
            if start_time:
                query = query.filter(AuditLog.timestamp >= start_time)
            if end_time:
                query = query.filter(AuditLog.timestamp <= end_time)
            
            logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": str(log.id),
                    "user_id": log.user_id,
                    "username": log.username,
                    "event_type": log.event_type,
                    "resource": log.resource,
                    "action": log.action,
                    "severity": log.severity,
                    "ip_address": log.ip_address,
                    "timestamp": log.timestamp.isoformat(),
                    "details": log.details,
                    "outcome": log.outcome,
                    "response_time_ms": log.response_time_ms
                }
                for log in logs
            ]
            
        finally:
            db.close()
    
    async def get_rate_limit_violations(self, 
                                       user_id: Optional[str] = None,
                                       ip_address: Optional[str] = None,
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get rate limit violations"""
        
        db = self.get_db()
        try:
            query = db.query(RateLimitViolation)
            
            if user_id:
                query = query.filter(RateLimitViolation.user_id == user_id)
            if ip_address:
                query = query.filter(RateLimitViolation.ip_address == ip_address)
            
            violations = query.order_by(RateLimitViolation.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": str(violation.id),
                    "user_id": violation.user_id,
                    "ip_address": violation.ip_address,
                    "endpoint": violation.endpoint,
                    "limit_type": violation.limit_type,
                    "limit_value": violation.limit_value,
                    "actual_value": violation.actual_value,
                    "timestamp": violation.timestamp.isoformat(),
                    "penalty_applied": violation.penalty_applied
                }
                for violation in violations
            ]
            
        finally:
            db.close()
    
    async def enable_emergency_mode(self, user_id: str, reason: str):
        """Enable emergency mode to bypass rate limits"""
        self.emergency_mode = True
        
        logger.critical(f"Emergency mode enabled by {user_id}: {reason}")
        
        # Log to audit
        audit_entry = AuditEntry(
            user_id=user_id,
            username=None,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            resource="rate_limiting",
            action="emergency_mode_enabled",
            severity=AuditSeverity.CRITICAL,
            ip_address="system",
            user_agent="system",
            timestamp=datetime.utcnow(),
            details={"reason": reason},
            request_id=None,
            session_id=None,
            outcome="success"
        )
        
        db = self.get_db()
        try:
            audit_log = AuditLog(**audit_entry.dict())
            db.add(audit_log)
            db.commit()
        finally:
            db.close()
    
    async def disable_emergency_mode(self, user_id: str):
        """Disable emergency mode"""
        self.emergency_mode = False
        
        logger.info(f"Emergency mode disabled by {user_id}")
        
        # Log to audit
        audit_entry = AuditEntry(
            user_id=user_id,
            username=None,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            resource="rate_limiting",
            action="emergency_mode_disabled",
            severity=AuditSeverity.HIGH,
            ip_address="system",
            user_agent="system",
            timestamp=datetime.utcnow(),
            details={},
            request_id=None,
            session_id=None,
            outcome="success"
        )
        
        db = self.get_db()
        try:
            audit_log = AuditLog(**audit_entry.dict())
            db.add(audit_log)
            db.commit()
        finally:
            db.close()

# Global rate limiting and audit system
rate_limit_audit_system = RateLimitingAuditSystem()

# Dependency for rate limiting
async def check_rate_limit_dependency(request: Request, user: UserInfo = Depends(verify_token)):
    """Rate limiting dependency"""
    allowed, limit_info = await rate_limit_audit_system.check_rate_limit(request, user)
    
    if not allowed:
        # Add rate limit headers
        response = HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limit_info.limit_value),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(limit_info.window_end.timestamp())),
                "Retry-After": str(int((limit_info.window_end - datetime.utcnow()).total_seconds()))
            }
        )
        
        # Log rate limit violation
        await rate_limit_audit_system.log_audit_event(
            request=request,
            user=user,
            event_type=AuditEventType.RATE_LIMIT_VIOLATION,
            resource="rate_limiting",
            action="rate_limit_exceeded",
            severity=AuditSeverity.MEDIUM,
            outcome="blocked",
            details={
                "limit_type": limit_info.limit_type.value,
                "limit_value": limit_info.limit_value,
                "current_value": limit_info.current_value
            }
        )
        
        raise response
    
    return user

# Audit logging dependency
async def audit_log_dependency(request: Request, user: UserInfo = Depends(verify_token)):
    """Audit logging dependency"""
    
    # This would be called after the request is processed
    # For now, just return the user
    return user