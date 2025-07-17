"""
Base Exception Classes for GrandModel

Provides a comprehensive hierarchy of exception classes to replace bare except clauses
with specific, actionable error handling.
"""

import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and routing."""
    SYSTEM = "system"
    CONFIG = "config"
    DATA = "data"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for error tracking and debugging."""
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    environment: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class BaseGrandModelError(Exception):
    """
    Base exception class for all GrandModel exceptions.
    
    Provides structured error information and eliminates the need for bare except clauses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base GrandModel error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            severity: Error severity level
            category: Error category for classification
            context: Error context for tracking
            cause: Root cause exception
            recoverable: Whether error is recoverable
            retry_after: Suggested retry delay in seconds
            error_details: Additional error details
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or f"{category.value}_error"
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.error_details = error_details or {}
        
        # Auto-log all errors
        self._log_error()
    
    def _log_error(self):
        """Log error with structured information."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(self.severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"[{self.error_code}] {self.message}",
            extra={
                "error_code": self.error_code,
                "severity": self.severity.value,
                "category": self.category.value,
                "correlation_id": self.context.correlation_id,
                "recoverable": self.recoverable,
                "retry_after": self.retry_after,
                "error_details": self.error_details,
                "cause": str(self.cause) if self.cause else None
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": {
                "timestamp": self.context.timestamp,
                "correlation_id": self.context.correlation_id,
                "request_id": self.context.request_id,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "service_name": self.context.service_name,
                "service_version": self.context.service_version,
                "environment": self.context.environment,
                "additional_data": self.context.additional_data
            },
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "error_details": self.error_details,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}', severity={self.severity})"


class SystemError(BaseGrandModelError):
    """System-level errors (hardware, OS, runtime)."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.SYSTEM,
            recoverable=kwargs.get('recoverable', False),
            **kwargs
        )


class ConfigurationError(BaseGrandModelError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.CONFIG,
            recoverable=kwargs.get('recoverable', False),
            error_details={"config_key": config_key},
            **kwargs
        )


class DataError(BaseGrandModelError):
    """Data-related errors (validation, corruption, format)."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.DATA,
            recoverable=kwargs.get('recoverable', True),
            error_details={"data_source": data_source},
            **kwargs
        )


class NetworkError(BaseGrandModelError):
    """Network-related errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.NETWORK,
            recoverable=kwargs.get('recoverable', True),
            retry_after=kwargs.get('retry_after', 5.0),
            error_details={"endpoint": endpoint, "status_code": status_code},
            **kwargs
        )


class SecurityError(BaseGrandModelError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.CRITICAL),
            category=ErrorCategory.SECURITY,
            recoverable=kwargs.get('recoverable', False),
            **kwargs
        )


class PerformanceError(BaseGrandModelError):
    """Performance-related errors."""
    
    def __init__(self, message: str, metric: Optional[str] = None, threshold: Optional[float] = None, actual: Optional[float] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.PERFORMANCE,
            recoverable=kwargs.get('recoverable', True),
            error_details={"metric": metric, "threshold": threshold, "actual": actual},
            **kwargs
        )


class ValidationError(BaseGrandModelError):
    """Validation-related errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.VALIDATION,
            recoverable=kwargs.get('recoverable', True),
            error_details={"field": field, "value": str(value) if value is not None else None},
            **kwargs
        )


class AuthenticationError(BaseGrandModelError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.AUTHENTICATION,
            recoverable=kwargs.get('recoverable', False),
            **kwargs
        )


class AuthorizationError(BaseGrandModelError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.AUTHORIZATION,
            recoverable=kwargs.get('recoverable', False),
            error_details={"required_permission": required_permission},
            **kwargs
        )


class RateLimitError(BaseGrandModelError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, limit: Optional[int] = None, window: Optional[float] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.RATE_LIMIT,
            recoverable=kwargs.get('recoverable', True),
            retry_after=kwargs.get('retry_after', window),
            error_details={"limit": limit, "window": window},
            **kwargs
        )


class CircuitBreakerError(BaseGrandModelError):
    """Circuit breaker errors."""
    
    def __init__(self, message: str, circuit_name: str, failure_count: int = 0, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.CIRCUIT_BREAKER,
            recoverable=kwargs.get('recoverable', True),
            retry_after=kwargs.get('retry_after', 30.0),
            error_details={"circuit_name": circuit_name, "failure_count": failure_count},
            **kwargs
        )


class TimeoutError(BaseGrandModelError):
    """Timeout-related errors."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            category=ErrorCategory.TIMEOUT,
            recoverable=kwargs.get('recoverable', True),
            retry_after=kwargs.get('retry_after', 1.0),
            error_details={"timeout_duration": timeout_duration},
            **kwargs
        )


class ResourceError(BaseGrandModelError):
    """Resource-related errors (memory, disk, CPU)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.RESOURCE,
            recoverable=kwargs.get('recoverable', True),
            error_details={"resource_type": resource_type},
            **kwargs
        )


class DependencyError(BaseGrandModelError):
    """Dependency-related errors."""
    
    def __init__(self, message: str, dependency_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            category=ErrorCategory.DEPENDENCY,
            recoverable=kwargs.get('recoverable', True),
            retry_after=kwargs.get('retry_after', 10.0),
            error_details={"dependency_name": dependency_name},
            **kwargs
        )


class CriticalError(BaseGrandModelError):
    """Critical errors that require immediate attention."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            recoverable=kwargs.get('recoverable', False),
            **kwargs
        )


class RecoverableError(BaseGrandModelError):
    """Base class for recoverable errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            recoverable=True,
            **kwargs
        )


class NonRecoverableError(BaseGrandModelError):
    """Base class for non-recoverable errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            recoverable=False,
            **kwargs
        )


# Compatibility with existing tactical exceptions
class CriticalDependencyError(DependencyError):
    """Critical dependency failure (backward compatibility)."""
    
    def __init__(self, dependency: str, error_message: str, **kwargs):
        super().__init__(
            message=f"Critical dependency failure [{dependency}]: {error_message}",
            dependency_name=dependency,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class MatrixValidationError(ValidationError):
    """Matrix validation failure (backward compatibility)."""
    
    def __init__(self, validation_type: str, error_message: str, matrix_shape: Optional[tuple] = None, **kwargs):
        super().__init__(
            message=f"Matrix validation failure [{validation_type}]: {error_message}",
            field="matrix",
            value=f"shape={matrix_shape}" if matrix_shape else None,
            error_details={"validation_type": validation_type, "matrix_shape": matrix_shape},
            **kwargs
        )


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open (backward compatibility)."""
    
    def __init__(self, circuit_name: str, failure_count: int, failure_threshold: int, **kwargs):
        super().__init__(
            message=f"Circuit breaker open [{circuit_name}]: {failure_count}/{failure_threshold} failures",
            circuit_name=circuit_name,
            failure_count=failure_count,
            error_details={"failure_threshold": failure_threshold},
            **kwargs
        )