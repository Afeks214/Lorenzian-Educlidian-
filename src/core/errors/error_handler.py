"""
AGENT 7: Enhanced Error Handler with Silent Failure Prevention
Comprehensive error handling with automatic retry, circuit breaker patterns,
silent failure elimination, and trading-specific error recovery.
"""

import asyncio
import functools
import logging
import time
import traceback
import uuid
from typing import Dict, Any, Optional, Callable, Type, List, Union, Awaitable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone, timedelta

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    CircuitBreakerError, TimeoutError, NetworkError, DependencyError,
    RecoverableError, NonRecoverableError, SystemError, ValidationError,
    DataError, ResourceError, AuthorizationError
)

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retriable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        NetworkError, TimeoutError, DependencyError, RecoverableError
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exception: Type[Exception] = Exception
    name: str = "default"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerManager:
    """Manages circuit breaker state and behavior."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.next_attempt_time = 0.0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            now = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if now >= self.next_attempt_time:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self, exception: Exception):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                
                logger.warning(
                    f"Circuit breaker opened for {self.config.name}",
                    extra={
                        "circuit_name": self.config.name,
                        "failure_count": self.failure_count,
                        "failure_threshold": self.config.failure_threshold,
                        "recovery_timeout": self.config.recovery_timeout
                    }
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "next_attempt_time": self.next_attempt_time
            }


class RetryManager:
    """Manages retry logic with various strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return min(delay, self.config.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if retry should be attempted."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is retriable
        for exc_type in self.config.retriable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False


class FallbackManager:
    """Manages fallback mechanisms."""
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
    
    def register_fallback(self, name: str, fallback: Callable):
        """Register a fallback function."""
        self.fallbacks[name] = fallback
    
    def execute_fallback(self, name: str, *args, **kwargs):
        """Execute fallback function."""
        if name in self.fallbacks:
            return self.fallbacks[name](*args, **kwargs)
        return None


@dataclass
class SilentFailureConfig:
    """Configuration for silent failure prevention."""
    track_mandatory_responses: bool = True
    response_timeout_seconds: float = 30.0
    null_response_is_failure: bool = True
    empty_response_is_failure: bool = True
    track_function_calls: bool = True
    alert_on_silent_failure: bool = True


@dataclass
class ErrorStatistics:
    """Error statistics tracking."""
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    silent_failures: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    circuit_breaker_trips: int = 0
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.total_errors += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def record_silent_failure(self):
        """Record a silent failure."""
        self.silent_failures += 1
    
    def record_recovery_attempt(self, success: bool):
        """Record a recovery attempt."""
        self.recovery_attempts += 1
        if success:
            self.recovery_successes += 1
    
    def record_circuit_breaker_trip(self):
        """Record a circuit breaker trip."""
        self.circuit_breaker_trips += 1


class ErrorHandler:
    """
    Comprehensive error handler with retry, circuit breaker, fallback support,
    and silent failure prevention.
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_manager: Optional[FallbackManager] = None,
        silent_failure_config: Optional[SilentFailureConfig] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config
        self.fallback_manager = fallback_manager or FallbackManager()
        self.silent_failure_config = silent_failure_config or SilentFailureConfig()
        
        self.retry_manager = RetryManager(self.retry_config)
        self.circuit_breaker = None
        if circuit_breaker_config:
            self.circuit_breaker = CircuitBreakerManager(circuit_breaker_config)
        
        # Silent failure prevention
        self.mandatory_response_functions: Set[str] = set()
        self.function_call_tracking: Dict[str, List[datetime]] = {}
        self.response_validators: Dict[str, Callable] = {}
        self.silent_failure_alerts: List[Dict[str, Any]] = []
        
        # Statistics tracking
        self.statistics = ErrorStatistics()
        
        # Error correlation tracking
        self.error_correlation_window = timedelta(minutes=5)
        self.correlated_errors: List[Dict[str, Any]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("Enhanced ErrorHandler initialized with silent failure prevention")
    
    def register_mandatory_response_function(self, function_name: str, validator: Optional[Callable] = None):
        """Register a function that must return a valid response."""
        with self._lock:
            self.mandatory_response_functions.add(function_name)
            if validator:
                self.response_validators[function_name] = validator
            logger.info(f"Registered mandatory response function: {function_name}")
    
    def track_function_call(self, function_name: str):
        """Track function call for silent failure detection."""
        if not self.silent_failure_config.track_function_calls:
            return
        
        with self._lock:
            if function_name not in self.function_call_tracking:
                self.function_call_tracking[function_name] = []
            
            self.function_call_tracking[function_name].append(datetime.now(timezone.utc))
            
            # Keep only recent calls
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self.function_call_tracking[function_name] = [
                call_time for call_time in self.function_call_tracking[function_name]
                if call_time >= cutoff_time
            ]
    
    def validate_response(self, function_name: str, response: Any) -> bool:
        """Validate function response to detect silent failures."""
        if function_name not in self.mandatory_response_functions:
            return True
        
        # Check for null response
        if self.silent_failure_config.null_response_is_failure and response is None:
            self._record_silent_failure(function_name, "null_response", response)
            return False
        
        # Check for empty response
        if self.silent_failure_config.empty_response_is_failure:
            if hasattr(response, '__len__') and len(response) == 0:
                self._record_silent_failure(function_name, "empty_response", response)
                return False
        
        # Use custom validator if available
        if function_name in self.response_validators:
            try:
                validator = self.response_validators[function_name]
                if not validator(response):
                    self._record_silent_failure(function_name, "custom_validation_failed", response)
                    return False
            except Exception as e:
                logger.warning(f"Response validator failed for {function_name}: {e}")
                return False
        
        return True
    
    def _record_silent_failure(self, function_name: str, failure_type: str, response: Any):
        """Record a silent failure occurrence."""
        with self._lock:
            self.statistics.record_silent_failure()
            
            failure_record = {
                "function_name": function_name,
                "failure_type": failure_type,
                "response": str(response),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            self.silent_failure_alerts.append(failure_record)
            
            # Keep only recent alerts
            if len(self.silent_failure_alerts) > 1000:
                self.silent_failure_alerts = self.silent_failure_alerts[-1000:]
            
            logger.error(
                f"Silent failure detected in {function_name}: {failure_type}",
                extra=failure_record
            )
            
            # Trigger alert if enabled
            if self.silent_failure_config.alert_on_silent_failure:
                self._trigger_silent_failure_alert(failure_record)
    
    def _trigger_silent_failure_alert(self, failure_record: Dict[str, Any]):
        """Trigger alert for silent failure."""
        alert_message = (
            f"SILENT FAILURE ALERT: {failure_record['function_name']} "
            f"failed with {failure_record['failure_type']}"
        )
        
        logger.critical(alert_message, extra=failure_record)
        
        # Additional alerting mechanisms can be added here
        # (e.g., send to monitoring systems, notify operators, etc.)
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        fallback_name: Optional[str] = None,
        function_name: Optional[str] = None,
        **fallback_kwargs
    ) -> Any:
        """Handle exception with comprehensive recovery mechanisms."""
        
        with self._lock:
            # Record error statistics
            self.statistics.record_error(type(exception).__name__)
            
            # Convert generic exceptions to GrandModel exceptions
            if not isinstance(exception, BaseGrandModelError):
                exception = self._convert_exception(exception, context)
            
            # Enhance context with correlation information
            if context is None:
                context = ErrorContext()
            
            # Add error to correlation tracking
            self._add_correlated_error(exception, context, function_name)
            
            # Log the error with enhanced context
            logger.error(
                f"Error handled: {exception}",
                extra={
                    "error_type": type(exception).__name__,
                    "error_message": str(exception),
                    "correlation_id": context.correlation_id,
                    "function_name": function_name,
                    "severity": exception.severity.value if hasattr(exception, 'severity') else 'unknown',
                    "recoverable": getattr(exception, 'recoverable', True),
                    "stack_trace": traceback.format_exc()
                }
            )
            
            # Try fallback if available
            if fallback_name:
                try:
                    self.statistics.record_recovery_attempt(True)
                    result = self.fallback_manager.execute_fallback(fallback_name, **fallback_kwargs)
                    
                    # Validate fallback response
                    if function_name and not self.validate_response(function_name, result):
                        logger.warning(f"Fallback response validation failed for {function_name}")
                        self.statistics.record_recovery_attempt(False)
                        return None
                    
                    return result
                except Exception as fallback_error:
                    self.statistics.record_recovery_attempt(False)
                    logger.error(f"Fallback failed: {fallback_error}")
            
            # Re-raise if not recoverable
            if not getattr(exception, 'recoverable', True):
                raise exception
            
            # Return None for graceful degradation
            return None
    
    def _add_correlated_error(self, exception: Exception, context: ErrorContext, function_name: Optional[str]):
        """Add error to correlation tracking."""
        error_record = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "function_name": function_name,
            "correlation_id": context.correlation_id,
            "timestamp": datetime.now(timezone.utc),
            "severity": getattr(exception, 'severity', ErrorSeverity.MEDIUM).value
        }
        
        self.correlated_errors.append(error_record)
        
        # Keep only recent errors
        cutoff_time = datetime.now(timezone.utc) - self.error_correlation_window
        self.correlated_errors = [
            error for error in self.correlated_errors
            if error["timestamp"] >= cutoff_time
        ]
        
        # Check for error patterns
        self._detect_error_patterns()
    
    def _detect_error_patterns(self):
        """Detect error patterns and correlations."""
        if len(self.correlated_errors) < 3:
            return
        
        # Group errors by type and function
        error_groups = {}
        for error in self.correlated_errors:
            key = f"{error['error_type']}:{error['function_name']}"
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(error)
        
        # Check for patterns (3+ errors of same type in window)
        for key, errors in error_groups.items():
            if len(errors) >= 3:
                logger.warning(
                    f"Error pattern detected: {key} occurred {len(errors)} times in {self.error_correlation_window}",
                    extra={
                        "pattern_key": key,
                        "error_count": len(errors),
                        "correlation_ids": [error["correlation_id"] for error in errors]
                    }
                )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            return {
                "total_errors": self.statistics.total_errors,
                "error_types": dict(self.statistics.error_types),
                "silent_failures": self.statistics.silent_failures,
                "recovery_attempts": self.statistics.recovery_attempts,
                "recovery_successes": self.statistics.recovery_successes,
                "recovery_success_rate": (
                    self.statistics.recovery_successes / self.statistics.recovery_attempts
                    if self.statistics.recovery_attempts > 0 else 0.0
                ),
                "circuit_breaker_trips": self.statistics.circuit_breaker_trips,
                "recent_silent_failures": len(self.silent_failure_alerts),
                "correlated_errors": len(self.correlated_errors),
                "mandatory_response_functions": len(self.mandatory_response_functions),
                "function_call_tracking": {
                    func: len(calls) for func, calls in self.function_call_tracking.items()
                }
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get health report of error handling system."""
        stats = self.get_error_statistics()
        
        # Calculate health score
        health_score = 100
        
        # Deduct for silent failures
        if stats["silent_failures"] > 0:
            health_score -= min(stats["silent_failures"] * 5, 30)
        
        # Deduct for low recovery rate
        if stats["recovery_success_rate"] < 0.8:
            health_score -= (0.8 - stats["recovery_success_rate"]) * 50
        
        # Deduct for high error rate
        if stats["total_errors"] > 100:
            health_score -= min((stats["total_errors"] - 100) * 0.1, 20)
        
        health_score = max(0, min(100, health_score))
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
            "statistics": stats,
            "recommendations": self._generate_health_recommendations(stats)
        }
    
    def _generate_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on statistics."""
        recommendations = []
        
        if stats["silent_failures"] > 0:
            recommendations.append(
                f"Address {stats['silent_failures']} silent failures by implementing proper response validation"
            )
        
        if stats["recovery_success_rate"] < 0.8:
            recommendations.append(
                f"Improve error recovery mechanisms - current success rate: {stats['recovery_success_rate']:.1%}"
            )
        
        if stats["total_errors"] > 100:
            recommendations.append(
                f"Investigate high error rate - {stats['total_errors']} total errors recorded"
            )
        
        if len(stats["error_types"]) > 10:
            recommendations.append(
                "Consider consolidating error types - too many different error types may indicate inconsistent error handling"
            )
        
        return recommendations
    
    def _convert_exception(self, exception: Exception, context: Optional[ErrorContext] = None) -> BaseGrandModelError:
        """Convert generic exception to GrandModel exception."""
        error_map = {
            ConnectionError: NetworkError,
            TimeoutError: TimeoutError,
            ValueError: ValidationError,
            KeyError: DataError,
            FileNotFoundError: ResourceError,
            PermissionError: AuthorizationError,
            ImportError: DependencyError,
            MemoryError: ResourceError,
            OSError: SystemError
        }
        
        for exc_type, grandmodel_exc_type in error_map.items():
            if isinstance(exception, exc_type):
                return grandmodel_exc_type(
                    message=str(exception),
                    context=context,
                    cause=exception
                )
        
        # Default to generic error
        return BaseGrandModelError(
            message=str(exception),
            context=context,
            cause=exception
        )
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                # Check circuit breaker
                if self.circuit_breaker and not self.circuit_breaker.can_execute():
                    raise CircuitBreakerError(
                        message=f"Circuit breaker is open for {self.circuit_breaker.config.name}",
                        circuit_name=self.circuit_breaker.config.name,
                        failure_count=self.circuit_breaker.failure_count
                    )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(e)
                
                # Check if should retry
                if not self.retry_manager.should_retry(e, attempt):
                    break
                
                # Calculate delay
                delay = self.retry_manager.calculate_delay(attempt)
                
                logger.warning(
                    f"Retry attempt {attempt}/{self.retry_config.max_attempts} after {delay:.2f}s",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.retry_config.max_attempts,
                        "delay": delay,
                        "error": str(e)
                    }
                )
                
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception


class GlobalErrorHandler:
    """Global singleton error handler."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.handlers: Dict[str, ErrorHandler] = {}
        self.default_handler = ErrorHandler()
        self._initialized = True
    
    def register_handler(self, name: str, handler: ErrorHandler):
        """Register named error handler."""
        self.handlers[name] = handler
    
    def get_handler(self, name: str = "default") -> ErrorHandler:
        """Get error handler by name."""
        if name == "default":
            return self.default_handler
        return self.handlers.get(name, self.default_handler)
    
    def handle_exception(self, exception: Exception, handler_name: str = "default", **kwargs) -> Any:
        """Handle exception using named handler."""
        handler = self.get_handler(handler_name)
        return handler.handle_exception(exception, **kwargs)


# Global error handler instance
global_error_handler = GlobalErrorHandler()


def handle_exception(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **fallback_kwargs
) -> Any:
    """Handle exception using global error handler."""
    return global_error_handler.handle_exception(
        exception, 
        handler_name=handler_name,
        context=context,
        fallback_name=fallback_name,
        **fallback_kwargs
    )


async def handle_async_exception(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **fallback_kwargs
) -> Any:
    """Handle exception asynchronously using global error handler."""
    # Run in thread pool for CPU-bound error handling
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        handle_exception,
        exception,
        context,
        fallback_name,
        handler_name,
        fallback_kwargs
    )


def with_error_handling(
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **handler_kwargs
):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle_exception(
                    e,
                    fallback_name=fallback_name,
                    handler_name=handler_name,
                    **handler_kwargs
                )
        return wrapper
    return decorator


def async_with_error_handling(
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **handler_kwargs
):
    """Async decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return await handle_async_exception(
                    e,
                    fallback_name=fallback_name,
                    handler_name=handler_name,
                    **handler_kwargs
                )
        return wrapper
    return decorator


@contextmanager
def error_context(
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **handler_kwargs
):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        handle_exception(
            e,
            fallback_name=fallback_name,
            handler_name=handler_name,
            **handler_kwargs
        )


@asynccontextmanager
async def async_error_context(
    fallback_name: Optional[str] = None,
    handler_name: str = "default",
    **handler_kwargs
):
    """Async context manager for error handling."""
    try:
        yield
    except Exception as e:
        await handle_async_exception(
            e,
            fallback_name=fallback_name,
            handler_name=handler_name,
            **handler_kwargs
        )