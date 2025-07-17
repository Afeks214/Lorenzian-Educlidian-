"""
Operational Controls for Operations System

This module provides operational control capabilities including
circuit breakers, rate limiting, system controls, and emergency procedures.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class ControlStatus(Enum):
    """Control status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    DISABLED = "disabled"


class ControlType(Enum):
    """Control types"""
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMITER = "rate_limiter"
    THROTTLE = "throttle"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE_MODE = "maintenance_mode"
    FEATURE_FLAG = "feature_flag"


class ActionType(Enum):
    """Action types"""
    BLOCK = "block"
    DELAY = "delay"
    REDIRECT = "redirect"
    ALERT = "alert"
    SHUTDOWN = "shutdown"
    THROTTLE = "throttle"


@dataclass
class ControlAction:
    """Control action definition"""
    action_id: str
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    name: str
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    last_success: Optional[datetime] = None
    total_calls: int = 0
    successful_calls: int = 0
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    def should_trip(self) -> bool:
        """Check if circuit breaker should trip"""
        return self.failure_count >= self.failure_threshold
    
    def should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.state != "OPEN":
            return False
        
        if self.last_failure is None:
            return True
        
        return (datetime.now() - self.last_failure).total_seconds() >= self.recovery_timeout


@dataclass
class RateLimiterState:
    """Rate limiter state"""
    name: str
    requests: deque = field(default_factory=deque)
    max_requests: int = 100
    time_window: int = 60  # seconds
    blocked_count: int = 0
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        self.blocked_count += 1
        return False


class OperationalControls:
    """Comprehensive operational controls system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.is_running = False
        self.monitoring_task = None
        
        # Control states
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.rate_limiters: Dict[str, RateLimiterState] = {}
        self.control_actions: Dict[str, ControlAction] = {}
        
        # System state
        self.system_status = "normal"
        self.maintenance_mode = False
        self.emergency_stop = False
        self.feature_flags: Dict[str, bool] = {}
        
        # Throttling
        self.throttle_settings: Dict[str, Dict[str, Any]] = {}
        self.active_throttles: Dict[str, datetime] = {}
        
        # Statistics
        self.total_control_actions = 0
        self.blocked_requests = 0
        self.circuit_breaker_trips = 0
        self.rate_limit_violations = 0
        
        # Event history
        self.control_events = deque(maxlen=1000)
        
        # Initialize default controls
        self._initialize_default_controls()
        
        logger.info("Operational Controls initialized")
    
    def _initialize_default_controls(self):
        """Initialize default operational controls"""
        # Default circuit breakers
        self.circuit_breakers["api_calls"] = CircuitBreakerState(
            name="api_calls",
            failure_threshold=5,
            recovery_timeout=60
        )
        
        self.circuit_breakers["database"] = CircuitBreakerState(
            name="database",
            failure_threshold=3,
            recovery_timeout=30
        )
        
        self.circuit_breakers["external_service"] = CircuitBreakerState(
            name="external_service",
            failure_threshold=10,
            recovery_timeout=120
        )
        
        # Default rate limiters
        self.rate_limiters["api_requests"] = RateLimiterState(
            name="api_requests",
            max_requests=1000,
            time_window=60
        )
        
        self.rate_limiters["user_actions"] = RateLimiterState(
            name="user_actions",
            max_requests=100,
            time_window=60
        )
        
        # Default control actions
        self.control_actions["emergency_shutdown"] = ControlAction(
            action_id="emergency_shutdown",
            action_type=ActionType.SHUTDOWN,
            description="Emergency system shutdown",
            parameters={"graceful": True, "timeout": 30}
        )
        
        self.control_actions["throttle_requests"] = ControlAction(
            action_id="throttle_requests",
            action_type=ActionType.THROTTLE,
            description="Throttle incoming requests",
            parameters={"factor": 0.5, "duration": 300}
        )
        
        self.control_actions["maintenance_mode"] = ControlAction(
            action_id="maintenance_mode",
            action_type=ActionType.BLOCK,
            description="Enable maintenance mode",
            parameters={"message": "System under maintenance"}
        )
        
        # Default feature flags
        self.feature_flags["new_algorithm"] = False
        self.feature_flags["experimental_features"] = False
        self.feature_flags["debug_mode"] = False
    
    async def start_monitoring(self):
        """Start control monitoring"""
        if self.is_running:
            logger.warning("Control monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Control monitoring started")
    
    async def stop_monitoring(self):
        """Stop control monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Control monitoring stopped")
    
    async def _monitoring_loop(self):
        """Control monitoring loop"""
        while self.is_running:
            try:
                # Update circuit breaker states
                await self._update_circuit_breakers()
                
                # Check throttle expirations
                await self._check_throttle_expirations()
                
                # Cleanup old events
                await self._cleanup_old_events()
                
                # Publish control metrics
                await self._publish_control_metrics()
                
            except Exception as e:
                logger.error("Error in control monitoring loop", error=str(e))
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _update_circuit_breakers(self):
        """Update circuit breaker states"""
        for breaker in self.circuit_breakers.values():
            if breaker.state == "OPEN" and breaker.should_attempt_reset():
                breaker.state = "HALF_OPEN"
                logger.info("Circuit breaker reset attempted", name=breaker.name)
    
    async def _check_throttle_expirations(self):
        """Check for expired throttles"""
        now = datetime.now()
        expired_throttles = []
        
        for throttle_name, expiry_time in self.active_throttles.items():
            if now >= expiry_time:
                expired_throttles.append(throttle_name)
        
        for throttle_name in expired_throttles:
            del self.active_throttles[throttle_name]
            logger.info("Throttle expired", name=throttle_name)
    
    async def _cleanup_old_events(self):
        """Clean up old control events"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        while (self.control_events and 
               self.control_events[0].get("timestamp", datetime.now()) < cutoff_time):
            self.control_events.popleft()
    
    async def _publish_control_metrics(self):
        """Publish control metrics"""
        metrics = {
            "circuit_breakers": {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "success_rate": breaker.success_rate,
                    "total_calls": breaker.total_calls
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "rate_limiters": {
                name: {
                    "current_requests": len(limiter.requests),
                    "max_requests": limiter.max_requests,
                    "blocked_count": limiter.blocked_count
                }
                for name, limiter in self.rate_limiters.items()
            },
            "system_status": self.system_status,
            "maintenance_mode": self.maintenance_mode,
            "emergency_stop": self.emergency_stop,
            "active_throttles": len(self.active_throttles),
            "total_control_actions": self.total_control_actions,
            "blocked_requests": self.blocked_requests,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "rate_limit_violations": self.rate_limit_violations
        }
        
        # Publish metrics event
        metrics_event = Event(
            type=EventType.METRICS,
            payload={
                "source": "operational_controls",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(metrics_event)
    
    @asynccontextmanager
    async def circuit_breaker(self, name: str):
        """Circuit breaker context manager"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerState(name=name)
        
        breaker = self.circuit_breakers[name]
        
        # Check if circuit is open
        if breaker.state == "OPEN":
            if not breaker.should_attempt_reset():
                raise Exception(f"Circuit breaker {name} is OPEN")
            else:
                breaker.state = "HALF_OPEN"
        
        breaker.total_calls += 1
        
        try:
            yield
            
            # Success
            breaker.successful_calls += 1
            breaker.last_success = datetime.now()
            
            if breaker.state == "HALF_OPEN":
                breaker.state = "CLOSED"
                breaker.failure_count = 0
                logger.info("Circuit breaker reset", name=name)
                
        except Exception as e:
            # Failure
            breaker.failure_count += 1
            breaker.last_failure = datetime.now()
            
            if breaker.should_trip():
                breaker.state = "OPEN"
                self.circuit_breaker_trips += 1
                logger.error("Circuit breaker tripped", name=name, failure_count=breaker.failure_count)
                
                # Trigger alert
                await self._trigger_circuit_breaker_alert(name, breaker)
            
            raise e
    
    async def _trigger_circuit_breaker_alert(self, name: str, breaker: CircuitBreakerState):
        """Trigger circuit breaker alert"""
        alert_event = Event(
            type=EventType.ALERT,
            payload={
                "source": "circuit_breaker",
                "severity": "error",
                "title": f"Circuit Breaker Tripped: {name}",
                "message": f"Circuit breaker {name} has tripped after {breaker.failure_count} failures",
                "metadata": {
                    "circuit_breaker": name,
                    "failure_count": breaker.failure_count,
                    "success_rate": breaker.success_rate,
                    "total_calls": breaker.total_calls
                }
            }
        )
        
        await self.event_bus.publish(alert_event)
    
    async def check_rate_limit(self, name: str) -> bool:
        """Check rate limit"""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = RateLimiterState(name=name)
        
        limiter = self.rate_limiters[name]
        
        if limiter.is_allowed():
            return True
        else:
            self.rate_limit_violations += 1
            self.blocked_requests += 1
            
            # Log rate limit violation
            logger.warning("Rate limit exceeded", name=name, blocked_count=limiter.blocked_count)
            
            return False
    
    async def execute_control_action(self, action_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a control action"""
        if action_id not in self.control_actions:
            logger.error("Control action not found", action_id=action_id)
            return False
        
        action = self.control_actions[action_id]
        
        if not action.enabled:
            logger.warning("Control action is disabled", action_id=action_id)
            return False
        
        try:
            # Execute action based on type
            if action.action_type == ActionType.SHUTDOWN:
                await self._execute_shutdown_action(action, context)
            elif action.action_type == ActionType.THROTTLE:
                await self._execute_throttle_action(action, context)
            elif action.action_type == ActionType.BLOCK:
                await self._execute_block_action(action, context)
            elif action.action_type == ActionType.ALERT:
                await self._execute_alert_action(action, context)
            else:
                logger.warning("Unknown action type", action_type=action.action_type)
                return False
            
            # Update action statistics
            action.execution_count += 1
            action.last_executed = datetime.now()
            self.total_control_actions += 1
            
            # Log control event
            await self._log_control_event(action_id, action.action_type, context)
            
            logger.info("Control action executed", action_id=action_id, action_type=action.action_type.value)
            return True
            
        except Exception as e:
            logger.error("Error executing control action", action_id=action_id, error=str(e))
            return False
    
    async def _execute_shutdown_action(self, action: ControlAction, context: Optional[Dict[str, Any]]):
        """Execute shutdown action"""
        graceful = action.parameters.get("graceful", True)
        timeout = action.parameters.get("timeout", 30)
        
        self.emergency_stop = True
        
        # Publish shutdown event
        shutdown_event = Event(
            type=EventType.SYSTEM_SHUTDOWN,
            payload={
                "graceful": graceful,
                "timeout": timeout,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(shutdown_event)
        
        if graceful:
            logger.info("Graceful shutdown initiated", timeout=timeout)
        else:
            logger.warning("Emergency shutdown initiated")
    
    async def _execute_throttle_action(self, action: ControlAction, context: Optional[Dict[str, Any]]):
        """Execute throttle action"""
        factor = action.parameters.get("factor", 0.5)
        duration = action.parameters.get("duration", 300)
        
        # Apply throttle
        throttle_name = context.get("throttle_name", "default") if context else "default"
        expiry_time = datetime.now() + timedelta(seconds=duration)
        self.active_throttles[throttle_name] = expiry_time
        
        # Store throttle settings
        self.throttle_settings[throttle_name] = {
            "factor": factor,
            "duration": duration,
            "started": datetime.now().isoformat()
        }
        
        logger.info("Throttle applied", name=throttle_name, factor=factor, duration=duration)
    
    async def _execute_block_action(self, action: ControlAction, context: Optional[Dict[str, Any]]):
        """Execute block action"""
        message = action.parameters.get("message", "Service temporarily unavailable")
        
        if action.action_id == "maintenance_mode":
            self.maintenance_mode = True
            logger.info("Maintenance mode enabled", message=message)
        else:
            logger.info("Block action executed", message=message)
    
    async def _execute_alert_action(self, action: ControlAction, context: Optional[Dict[str, Any]]):
        """Execute alert action"""
        alert_event = Event(
            type=EventType.ALERT,
            payload={
                "source": "operational_controls",
                "severity": action.parameters.get("severity", "warning"),
                "title": action.parameters.get("title", "Operational Control Alert"),
                "message": action.parameters.get("message", action.description),
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(alert_event)
    
    async def _log_control_event(self, action_id: str, action_type: ActionType, context: Optional[Dict[str, Any]]):
        """Log control event"""
        event = {
            "timestamp": datetime.now(),
            "action_id": action_id,
            "action_type": action_type.value,
            "context": context or {}
        }
        
        self.control_events.append(event)
    
    def set_feature_flag(self, flag_name: str, enabled: bool):
        """Set feature flag"""
        self.feature_flags[flag_name] = enabled
        logger.info("Feature flag updated", flag_name=flag_name, enabled=enabled)
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag"""
        return self.feature_flags.get(flag_name, default)
    
    def is_throttled(self, throttle_name: str = "default") -> bool:
        """Check if throttle is active"""
        return throttle_name in self.active_throttles
    
    def get_throttle_factor(self, throttle_name: str = "default") -> float:
        """Get throttle factor"""
        if throttle_name in self.throttle_settings:
            return self.throttle_settings[throttle_name]["factor"]
        return 1.0
    
    def enable_maintenance_mode(self, message: str = "System under maintenance"):
        """Enable maintenance mode"""
        self.maintenance_mode = True
        logger.info("Maintenance mode enabled", message=message)
    
    def disable_maintenance_mode(self):
        """Disable maintenance mode"""
        self.maintenance_mode = False
        logger.info("Maintenance mode disabled")
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.system_status = "emergency"
        logger.critical("Emergency stop triggered")
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        self.system_status = "normal"
        logger.info("Emergency stop reset")
    
    def add_circuit_breaker(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Add circuit breaker"""
        self.circuit_breakers[name] = CircuitBreakerState(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info("Circuit breaker added", name=name)
    
    def add_rate_limiter(self, name: str, max_requests: int = 100, time_window: int = 60):
        """Add rate limiter"""
        self.rate_limiters[name] = RateLimiterState(
            name=name,
            max_requests=max_requests,
            time_window=time_window
        )
        logger.info("Rate limiter added", name=name)
    
    def add_control_action(self, action: ControlAction):
        """Add control action"""
        self.control_actions[action.action_id] = action
        logger.info("Control action added", action_id=action.action_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "system_status": self.system_status,
            "maintenance_mode": self.maintenance_mode,
            "emergency_stop": self.emergency_stop,
            "circuit_breakers": {
                name: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "success_rate": breaker.success_rate
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "rate_limiters": {
                name: {
                    "current_requests": len(limiter.requests),
                    "blocked_count": limiter.blocked_count
                }
                for name, limiter in self.rate_limiters.items()
            },
            "active_throttles": list(self.active_throttles.keys()),
            "feature_flags": self.feature_flags,
            "statistics": {
                "total_control_actions": self.total_control_actions,
                "blocked_requests": self.blocked_requests,
                "circuit_breaker_trips": self.circuit_breaker_trips,
                "rate_limit_violations": self.rate_limit_violations
            }
        }
    
    def get_control_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get control events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            {
                **event,
                "timestamp": event["timestamp"].isoformat()
            }
            for event in self.control_events
            if event["timestamp"] >= cutoff_time
        ]