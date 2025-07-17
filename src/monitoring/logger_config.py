"""
Structured logging configuration with correlation ID support.
Provides JSON-formatted logs for production observability.
"""

import logging
import sys
import uuid
import time
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
from functools import wraps
import structlog
from pythonjsonlogger import jsonlogger

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class CorrelationIdProcessor:
    """Add correlation ID to all log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to the event dict."""
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        return event_dict

class PerformanceProcessor:
    """Add performance metrics to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add timestamp and performance data."""
        event_dict['timestamp'] = time.time()
        event_dict['timestamp_human'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        return event_dict

class SecurityProcessor:
    """Sanitize sensitive data from logs."""
    
    SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'api_key', 'authorization',
        'credit_card', 'ssn', 'private_key'
    }
    
    def __call__(self, logger, method_name, event_dict):
        """Mask sensitive fields in logs."""
        for key in list(event_dict.keys()):
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                event_dict[key] = '***REDACTED***'
        return event_dict

def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    service_name: str = "grandmodel",
    environment: str = "production"
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "console")
        service_name: Name of the service for log identification
        environment: Environment name (production, staging, development)
    """
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        CorrelationIdProcessor(),
        PerformanceProcessor(),
        SecurityProcessor(),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            json_default=str,
            json_encoder=None
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Add service metadata to all logs
    logging.LoggerAdapter(root_logger, {
        'service': service_name,
        'environment': environment
    })
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)

def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())

def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    If no ID is provided, generates a new one.
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    correlation_id_var.set(correlation_id)
    return correlation_id

def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()

def with_correlation_id(func):
    """
    Decorator to ensure a correlation ID exists for the function execution.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        correlation_id = get_correlation_id()
        if not correlation_id:
            correlation_id = set_correlation_id()
        
        logger = get_logger(func.__module__)
        logger.info(
            "Function execution started",
            function=func.__name__,
            correlation_id=correlation_id
        )
        
        try:
            result = await func(*args, **kwargs)
            logger.info(
                "Function execution completed",
                function=func.__name__,
                correlation_id=correlation_id
            )
            return result
        except Exception as e:
            logger.error(
                "Function execution failed",
                function=func.__name__,
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        correlation_id = get_correlation_id()
        if not correlation_id:
            correlation_id = set_correlation_id()
        
        logger = get_logger(func.__module__)
        logger.info(
            "Function execution started",
            function=func.__name__,
            correlation_id=correlation_id
        )
        
        try:
            result = func(*args, **kwargs)
            logger.info(
                "Function execution completed",
                function=func.__name__,
                correlation_id=correlation_id
            )
            return result
        except Exception as e:
            logger.error(
                "Function execution failed",
                function=func.__name__,
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class LogContext:
    """Context manager for temporary log context."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.previous_context = {}
        
    def __enter__(self):
        """Enter context and bind values."""
        logger = structlog.get_logger()
        for key, value in self.context.items():
            self.previous_context[key] = logger._context.get(key)
            logger.bind(**{key: value})
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous values."""
        logger = structlog.get_logger()
        for key, value in self.previous_context.items():
            if value is None:
                logger.unbind(key)
            else:
                logger.bind(**{key: value})

# Performance logging utilities
class PerformanceLogger:
    """Log performance metrics for operations."""
    
    def __init__(self, operation_name: str, logger: Optional[structlog.BoundLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.operation_name}",
            operation=self.operation_name
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log operation completion with duration."""
        duration = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                f"Failed {self.operation_name}",
                operation=self.operation_name,
                duration_seconds=duration,
                error=str(exc_val)
            )
        else:
            self.logger.info(
                f"Completed {self.operation_name}",
                operation=self.operation_name,
                duration_seconds=duration
            )

# Request logging middleware
async def log_request(request, call_next):
    """
    Middleware to log HTTP requests with correlation ID.
    """
    # Extract or generate correlation ID
    correlation_id = request.headers.get('X-Correlation-ID')
    if not correlation_id:
        correlation_id = generate_correlation_id()
    
    set_correlation_id(correlation_id)
    
    logger = get_logger("api.request")
    
    # Log request
    logger.info(
        "Request received",
        method=request.method,
        path=request.url.path,
        client_host=request.client.host if request.client else None,
        correlation_id=correlation_id
    )
    
    # Time the request
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration,
            correlation_id=correlation_id
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            duration_seconds=duration,
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise

# Configure logging on module import
configure_logging()