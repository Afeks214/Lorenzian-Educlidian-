"""
Comprehensive Error Handling Framework for GrandModel

This module provides a unified error handling system that:
- Eliminates bare except clauses
- Provides structured error classification
- Implements error recovery mechanisms
- Enables comprehensive error monitoring
- Supports graceful degradation
"""

from .base_exceptions import (
    BaseGrandModelError,
    SystemError,
    ConfigurationError,
    DataError,
    NetworkError,
    SecurityError,
    PerformanceError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    CircuitBreakerError,
    TimeoutError,
    ResourceError,
    DependencyError,
    CriticalError,
    RecoverableError,
    NonRecoverableError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext
)

from .error_handler import (
    ErrorHandler,
    GlobalErrorHandler,
    handle_exception,
    handle_async_exception,
    with_error_handling,
    async_with_error_handling,
    CircuitBreakerManager,
    RetryManager,
    FallbackManager
)

from .error_logger import (
    ErrorLogger,
    StructuredErrorLogger,
    ErrorReport,
    ErrorMetrics,
    ErrorAggregator
)

from .error_monitor import (
    ErrorMonitor,
    AlertManager,
    ErrorThresholdMonitor,
    ErrorTrendAnalyzer,
    ErrorNotifier
)

from .error_recovery import (
    ErrorRecoveryManager,
    RecoveryStrategy,
    GracefulDegradationStrategy,
    SystemHealthChecker,
    ServiceHealthTracker
)

__all__ = [
    # Base exceptions
    'BaseGrandModelError',
    'SystemError',
    'ConfigurationError',
    'DataError',
    'NetworkError',
    'SecurityError',
    'PerformanceError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'CircuitBreakerError',
    'TimeoutError',
    'ResourceError',
    'DependencyError',
    'CriticalError',
    'RecoverableError',
    'NonRecoverableError',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    
    # Error handling
    'ErrorHandler',
    'GlobalErrorHandler',
    'handle_exception',
    'handle_async_exception',
    'with_error_handling',
    'async_with_error_handling',
    'CircuitBreakerManager',
    'RetryManager',
    'FallbackManager',
    
    # Error logging
    'ErrorLogger',
    'StructuredErrorLogger',
    'ErrorReport',
    'ErrorMetrics',
    'ErrorAggregator',
    
    # Error monitoring
    'ErrorMonitor',
    'AlertManager',
    'ErrorThresholdMonitor',
    'ErrorTrendAnalyzer',
    'ErrorNotifier',
    
    # Error recovery
    'ErrorRecoveryManager',
    'RecoveryStrategy',
    'GracefulDegradationStrategy',
    'SystemHealthChecker',
    'ServiceHealthTracker'
]