"""
Risk Error Handler for Live Execution

This module provides comprehensive error handling and risk failure management
for live trading execution, replacing fallback mechanisms with proper trade
rejection and risk control enforcement.

Key Features:
- Proper trade rejection instead of fallback execution
- Risk failure escalation protocols
- Comprehensive error propagation
- Fail-safe mechanisms for critical errors
- Audit trail for all error events
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import traceback

from src.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories"""
    VALIDATION = "validation"
    EXECUTION = "execution"
    RISK_CONTROL = "risk_control"
    SYSTEM = "system"
    NETWORK = "network"
    DATA = "data"
    BROKER = "broker"


@dataclass
class ErrorEvent:
    """Error event record"""
    timestamp: datetime
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    resolution_attempted: bool = False
    resolution_successful: bool = False
    escalated: bool = False


class RiskErrorHandler:
    """
    Risk Error Handler for Live Execution
    
    Handles all error conditions in live trading with proper risk controls
    and trade rejection mechanisms instead of fallback execution.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Error handling configuration
        self.strict_mode = config.get("strict_error_handling", True)
        self.auto_recovery = config.get("auto_recovery_enabled", True)
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        self.escalation_threshold = config.get("escalation_threshold", 5)
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[datetime]] = {}
        
        # Emergency protocols
        self.emergency_callbacks: List[Callable] = []
        self.system_shutdown_callbacks: List[Callable] = []
        
        # State management
        self.error_rate_limit = config.get("error_rate_limit", 10)  # errors per minute
        self.system_health_degraded = False
        self.trading_halted = False
        
        logger.info("Risk Error Handler initialized")
    
    async def handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle validation errors with proper trade rejection
        
        Args:
            error: The validation error
            context: Error context including order details
            
        Returns:
            Error response with rejection details
        """
        error_id = self._generate_error_id()
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message=str(error),
            details={"original_error": str(error)},
            source="validation_system",
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        self._update_error_counts("validation")
        
        # Log error
        logger.error(f"❌ Validation error {error_id}: {error}")
        
        # Publish error event
        await self._publish_error_event(error_event)
        
        # In strict mode, reject trade completely
        if self.strict_mode:
            return {
                "status": "rejected",
                "error_id": error_id,
                "reason": "validation_failed",
                "message": str(error),
                "action": "trade_rejected",
                "retry_allowed": False
            }
        
        # Check if error is recoverable
        if self._is_recoverable_error(error):
            return {
                "status": "rejected",
                "error_id": error_id,
                "reason": "validation_failed",
                "message": str(error),
                "action": "trade_rejected",
                "retry_allowed": True,
                "retry_after_seconds": 1
            }
        
        return {
            "status": "rejected",
            "error_id": error_id,
            "reason": "validation_failed",
            "message": str(error),
            "action": "trade_rejected",
            "retry_allowed": False
        }
    
    async def handle_execution_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle execution errors with proper error propagation
        
        Args:
            error: The execution error
            context: Error context including order details
            
        Returns:
            Error response with appropriate action
        """
        error_id = self._generate_error_id()
        
        # Determine severity based on error type
        severity = self._determine_severity(error)
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=ErrorCategory.EXECUTION,
            severity=severity,
            message=str(error),
            details={"original_error": str(error)},
            source="execution_system",
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        self._update_error_counts("execution")
        
        # Log error
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 Critical execution error {error_id}: {error}")
        else:
            logger.error(f"❌ Execution error {error_id}: {error}")
        
        # Publish error event
        await self._publish_error_event(error_event)
        
        # Handle critical errors
        if severity == ErrorSeverity.CRITICAL:
            await self._handle_critical_error(error_event)
            return {
                "status": "failed",
                "error_id": error_id,
                "reason": "critical_execution_error",
                "message": str(error),
                "action": "emergency_stop_triggered",
                "retry_allowed": False
            }
        
        # Handle recoverable errors
        if self._is_recoverable_error(error):
            return {
                "status": "failed",
                "error_id": error_id,
                "reason": "execution_error",
                "message": str(error),
                "action": "trade_rejected",
                "retry_allowed": True,
                "retry_after_seconds": 2
            }
        
        return {
            "status": "failed",
            "error_id": error_id,
            "reason": "execution_error",
            "message": str(error),
            "action": "trade_rejected",
            "retry_allowed": False
        }
    
    async def handle_risk_control_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle risk control errors with immediate escalation
        
        Args:
            error: The risk control error
            context: Error context including risk details
            
        Returns:
            Error response with risk control action
        """
        error_id = self._generate_error_id()
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=ErrorCategory.RISK_CONTROL,
            severity=ErrorSeverity.CRITICAL,
            message=str(error),
            details={"original_error": str(error)},
            source="risk_control_system",
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        self._update_error_counts("risk_control")
        
        # Log critical error\n        logger.critical(f"🚨 Risk control error {error_id}: {error}")
        
        # Publish error event
        await self._publish_error_event(error_event)
        
        # Immediate escalation for risk control failures
        await self._escalate_error(error_event)
        
        # Risk control errors are never retryable
        return {
            "status": "failed",
            "error_id": error_id,
            "reason": "risk_control_failure",
            "message": str(error),
            "action": "emergency_protocols_activated",
            "retry_allowed": False
        }
    
    async def handle_system_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle system errors with fail-safe mechanisms
        
        Args:
            error: The system error
            context: Error context including system details
            
        Returns:
            Error response with system action
        """
        error_id = self._generate_error_id()
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.FATAL,
            message=str(error),
            details={"original_error": str(error)},
            source="system",
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        self._update_error_counts("system")
        
        # Log fatal error
        logger.critical(f"💀 System error {error_id}: {error}")
        
        # Publish error event
        await self._publish_error_event(error_event)
        
        # Trigger fail-safe mechanisms
        await self._trigger_fail_safe(error_event)
        
        return {
            "status": "system_failure",
            "error_id": error_id,
            "reason": "system_error",
            "message": str(error),
            "action": "system_shutdown_initiated",
            "retry_allowed": False
        }
    
    async def handle_broker_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle broker connection and API errors
        
        Args:
            error: The broker error
            context: Error context including broker details
            
        Returns:
            Error response with broker action
        """
        error_id = self._generate_error_id()
        
        # Determine if this is a connectivity issue
        is_connectivity_error = any(keyword in str(error).lower() for keyword in 
                                  ["connection", "timeout", "network", "unreachable"])
        
        severity = ErrorSeverity.WARNING if is_connectivity_error else ErrorSeverity.ERROR
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=ErrorCategory.BROKER,
            severity=severity,
            message=str(error),
            details={"original_error": str(error), "is_connectivity": is_connectivity_error},
            source="broker_system",
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        self._update_error_counts("broker")
        
        # Log error
        logger.error(f"❌ Broker error {error_id}: {error}")
        
        # Publish error event
        await self._publish_error_event(error_event)
        
        # Handle connectivity errors with retry
        if is_connectivity_error:
            return {
                "status": "failed",
                "error_id": error_id,
                "reason": "broker_connectivity_error",
                "message": str(error),
                "action": "trade_rejected",
                "retry_allowed": True,
                "retry_after_seconds": 5
            }
        
        # Handle API errors
        return {
            "status": "failed",
            "error_id": error_id,
            "reason": "broker_api_error",
            "message": str(error),
            "action": "trade_rejected",
            "retry_allowed": False
        }
    
    async def _handle_critical_error(self, error_event: ErrorEvent):
        """Handle critical errors with immediate action"""
        try:
            # Trigger emergency protocols
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        "reason": f"Critical error: {error_event.message}",
                        "error_id": error_event.error_id,
                        "automatic": True
                    },
                    "RiskErrorHandler"
                )
            )
            
            # Execute emergency callbacks
            for callback in self.emergency_callbacks:
                try:
                    await callback(error_event)
                except Exception as e:
                    logger.error(f"Emergency callback failed: {e}")
            
            # Set system health degraded
            self.system_health_degraded = True
            
            logger.critical(f"🚨 Critical error protocols activated for {error_event.error_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle critical error: {e}")
    
    async def _escalate_error(self, error_event: ErrorEvent):
        """Escalate error to higher level systems"""
        try:
            error_event.escalated = True
            
            # Publish escalation event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_BREACH,
                    {
                        "type": "error_escalation",
                        "error_id": error_event.error_id,
                        "category": error_event.category.value,
                        "severity": error_event.severity.value,
                        "message": error_event.message
                    },
                    "RiskErrorHandler"
                )
            )
            
            logger.warning(f"⬆️ Error escalated: {error_event.error_id}")
            
        except Exception as e:
            logger.error(f"Failed to escalate error: {e}")
    
    async def _trigger_fail_safe(self, error_event: ErrorEvent):
        """Trigger fail-safe mechanisms for fatal errors"""
        try:
            # Halt trading
            self.trading_halted = True
            
            # Execute system shutdown callbacks
            for callback in self.system_shutdown_callbacks:
                try:
                    await callback(error_event)
                except Exception as e:
                    logger.error(f"System shutdown callback failed: {e}")
            
            # Emergency stop all positions
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        "reason": f"System failure: {error_event.message}",
                        "error_id": error_event.error_id,
                        "automatic": True,
                        "fail_safe": True
                    },
                    "RiskErrorHandler"
                )
            )
            
            logger.critical(f"💀 Fail-safe mechanisms triggered for {error_event.error_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger fail-safe: {e}")
    
    async def _publish_error_event(self, error_event: ErrorEvent):
        """Publish error event to event bus"""
        try:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.ERROR,
                    {
                        "error_id": error_event.error_id,
                        "category": error_event.category.value,
                        "severity": error_event.severity.value,
                        "message": error_event.message,
                        "source": error_event.source,
                        "context": error_event.context
                    },
                    "RiskErrorHandler"
                )
            )
        except Exception as e:
            logger.error(f"Failed to publish error event: {e}")
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"ERR_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and message"""
        error_str = str(error).lower()
        
        # Critical errors
        if any(keyword in error_str for keyword in ["critical", "fatal", "emergency", "system"]):
            return ErrorSeverity.CRITICAL
        
        # Connection errors are usually warnings
        if any(keyword in error_str for keyword in ["connection", "timeout", "network"]):
            return ErrorSeverity.WARNING
        
        # Default to error
        return ErrorSeverity.ERROR
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Check if error is recoverable"""
        error_str = str(error).lower()
        
        # Recoverable errors
        recoverable_keywords = ["timeout", "connection", "network", "temporary", "retry"]
        if any(keyword in error_str for keyword in recoverable_keywords):
            return True
        
        # Non-recoverable errors
        non_recoverable_keywords = ["invalid", "unauthorized", "forbidden", "limit", "breach"]
        if any(keyword in error_str for keyword in non_recoverable_keywords):
            return False
        
        return False
    
    def _update_error_counts(self, error_type: str):
        """Update error counts and check for patterns"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Track error patterns
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = []
        
        self.error_patterns[error_type].append(datetime.now())
        
        # Remove old entries (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.error_patterns[error_type] = [
            timestamp for timestamp in self.error_patterns[error_type]
            if timestamp > cutoff_time
        ]
        
        # Check for error rate limits
        if len(self.error_patterns[error_type]) > self.error_rate_limit:
            logger.warning(f"⚠️ Error rate limit exceeded for {error_type}")
            self.system_health_degraded = True
    
    def register_emergency_callback(self, callback: Callable):
        """Register emergency callback"""
        self.emergency_callbacks.append(callback)
    
    def register_shutdown_callback(self, callback: Callable):
        """Register system shutdown callback"""
        self.system_shutdown_callbacks.append(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        recent_errors = [
            error for error in self.error_events
            if (datetime.now() - error.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_events),
            "recent_errors": len(recent_errors),
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "error_rate": len(recent_errors) / 60,  # errors per minute
            "system_health_degraded": self.system_health_degraded,
            "trading_halted": self.trading_halted
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error events"""
        recent_errors = sorted(self.error_events, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "error_id": error.error_id,
                "timestamp": error.timestamp.isoformat(),
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "source": error.source,
                "escalated": error.escalated,
                "resolution_attempted": error.resolution_attempted
            }
            for error in recent_errors
        ]
    
    def reset_system_health(self):
        """Reset system health status"""
        self.system_health_degraded = False
        self.trading_halted = False
        logger.info("✅ System health status reset")
    
    def clear_old_errors(self, hours: int = 24):
        """Clear old error events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        original_count = len(self.error_events)
        self.error_events = [
            error for error in self.error_events
            if error.timestamp > cutoff_time
        ]
        
        cleared_count = original_count - len(self.error_events)
        if cleared_count > 0:
            logger.info(f"🧹 Cleared {cleared_count} old error events")