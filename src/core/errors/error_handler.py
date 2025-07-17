"""
Error Handler with Recovery, Retry, and Circuit Breaker Mechanisms

Provides comprehensive error handling with automatic retry, circuit breaker patterns,
and fallback mechanisms to replace bare except clauses.
"""

import asyncio
import functools
import logging
import time
from typing import Dict, Any, Optional, Callable, Type, List, Union, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager, asynccontextmanager

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    CircuitBreakerError, TimeoutError, NetworkError, DependencyError,
    RecoverableError, NonRecoverableError
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


class ErrorHandler:
    """Comprehensive error handler with retry, circuit breaker, and fallback support."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_manager: Optional[FallbackManager] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config
        self.fallback_manager = fallback_manager or FallbackManager()
        
        self.retry_manager = RetryManager(self.retry_config)
        self.circuit_breaker = None
        if circuit_breaker_config:
            self.circuit_breaker = CircuitBreakerManager(circuit_breaker_config)
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        fallback_name: Optional[str] = None,
        **fallback_kwargs
    ) -> Any:
        """Handle exception with recovery mechanisms."""
        
        # Convert generic exceptions to GrandModel exceptions
        if not isinstance(exception, BaseGrandModelError):
            exception = self._convert_exception(exception, context)
        
        # Log the error
        logger.error(
            f"Error handled: {exception}",
            extra={
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "correlation_id": context.correlation_id if context else None
            }
        )
        
        # Try fallback if available
        if fallback_name:
            try:
                return self.fallback_manager.execute_fallback(fallback_name, **fallback_kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
        
        # Re-raise if not recoverable
        if not getattr(exception, 'recoverable', True):
            raise exception
        
        # Return None for graceful degradation
        return None
    
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