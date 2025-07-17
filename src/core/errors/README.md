# GrandModel Error Handling Framework

## Overview

The GrandModel Error Handling Framework is a comprehensive system designed to eliminate bare except clauses and provide robust, structured error handling throughout the codebase. This framework ensures reliable error management, automatic recovery, and comprehensive monitoring.

## Key Features

### 🎯 **Zero Bare Except Clauses**
- Systematic elimination of all bare `except:` clauses
- Replaced with specific exception handling
- Comprehensive error classification and routing

### 🔄 **Automatic Error Recovery**
- Retry mechanisms with exponential backoff
- Circuit breaker patterns for failing services
- Graceful degradation strategies
- Fallback mechanisms for critical operations

### 📊 **Comprehensive Monitoring**
- Real-time error tracking and metrics
- Threshold-based alerting
- Trend analysis and anomaly detection
- Health monitoring and reporting

### 🔒 **Production-Ready Reliability**
- Structured error logging with correlation IDs
- Error aggregation and deduplication
- Performance impact minimization
- Thread-safe and async-compatible

## Architecture

```
src/core/errors/
├── __init__.py                 # Main imports and exports
├── base_exceptions.py          # Exception hierarchy
├── error_handler.py           # Core error handling logic
├── error_logger.py           # Structured logging system
├── error_monitor.py          # Monitoring and alerting
├── error_recovery.py         # Recovery mechanisms
└── README.md                 # This documentation
```

## Core Components

### 1. Base Exception Hierarchy

```python
from src.core.errors import BaseGrandModelError, ErrorSeverity, ErrorCategory

# Custom error with context
error = BaseGrandModelError(
    message="Database connection failed",
    severity=ErrorSeverity.HIGH,
    category=ErrorCategory.DEPENDENCY,
    error_code="DB_CONNECTION_FAILED"
)
```

**Available Exception Types:**
- `BaseGrandModelError` - Base class for all custom exceptions
- `SystemError` - System-level errors
- `ConfigurationError` - Configuration-related errors
- `DataError` - Data validation/corruption errors
- `NetworkError` - Network connectivity errors
- `SecurityError` - Security-related errors
- `PerformanceError` - Performance threshold violations
- `ValidationError` - Input validation errors
- `AuthenticationError` - Authentication failures
- `AuthorizationError` - Authorization failures
- `CircuitBreakerError` - Circuit breaker activation
- `TimeoutError` - Operation timeouts
- `ResourceError` - Resource exhaustion
- `DependencyError` - External dependency failures

### 2. Error Handler with Recovery

```python
from src.core.errors import ErrorHandler, RetryConfig, CircuitBreakerConfig

# Configure error handler
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL
)

circuit_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0
)

handler = ErrorHandler(
    retry_config=retry_config,
    circuit_breaker_config=circuit_config
)

# Use with retry logic
result = handler.execute_with_retry(risky_operation, *args, **kwargs)
```

### 3. Structured Error Logging

```python
from src.core.errors import ErrorLogger, ErrorSeverity, ErrorCategory

logger = ErrorLogger()

# Log error with context
correlation_id = logger.log_error(
    error=my_error,
    additional_context={"user_id": "123", "request_id": "abc"},
    include_stack_trace=True
)

# Get error metrics
metrics = logger.get_metrics()
print(f"Total errors: {metrics['total_errors']}")
print(f"Error rate: {metrics['error_rate']}")
```

### 4. Error Monitoring and Alerting

```python
from src.core.errors import ErrorMonitor, AlertRule, AlertType

monitor = ErrorMonitor()

# Add custom alert rule
rule = AlertRule(
    name="critical_errors",
    alert_type=AlertType.CRITICAL_ERROR,
    threshold=1,
    window_size=60,
    severity_filter={ErrorSeverity.CRITICAL}
)

monitor.add_alert_rule(rule)

# Monitor an error
monitor.monitor_error(error_report)
```

### 5. Recovery Strategies

```python
from src.core.errors import ErrorRecoveryManager, attempt_recovery

# Attempt automatic recovery
recovery_manager = ErrorRecoveryManager()
success, result = recovery_manager.attempt_recovery(error, context)

if success:
    print(f"Recovery successful: {result}")
else:
    print("Recovery failed, escalating...")
```

## Usage Patterns

### 1. Replacing Bare Except Clauses

**❌ Before (Bare Except):**
```python
try:
    risky_operation()
except:
    pass
```

**✅ After (Specific Exceptions):**
```python
from src.core.errors import handle_exception, NetworkError, TimeoutError

try:
    risky_operation()
except (NetworkError, TimeoutError) as e:
    handle_exception(e, fallback_name="network_fallback")
except Exception as e:
    handle_exception(e, context=error_context)
```

### 2. Using Decorators

```python
from src.core.errors import with_error_handling, async_with_error_handling

@with_error_handling(fallback_name="default_response")
def api_endpoint():
    # Your code here
    return process_request()

@async_with_error_handling(handler_name="async_handler")
async def async_operation():
    # Your async code here
    return await process_async_request()
```

### 3. Context Manager Usage

```python
from src.core.errors import error_context, async_error_context

with error_context(fallback_name="file_fallback"):
    with open("file.txt", "r") as f:
        data = f.read()

async with async_error_context(handler_name="async_handler"):
    async with aiofiles.open("file.txt", "r") as f:
        data = await f.read()
```

### 4. Custom Recovery Strategies

```python
from src.core.errors import RecoveryStrategy, ErrorRecoveryManager

class CustomRecoveryStrategy(RecoveryStrategy):
    def can_recover(self, error):
        return isinstance(error, MyCustomError)
    
    def recover(self, error, context):
        # Custom recovery logic
        return {"action": "custom_recovery", "result": "success"}
    
    def get_priority(self):
        return 1  # High priority

# Register custom strategy
recovery_manager = ErrorRecoveryManager()
recovery_manager.add_strategy(CustomRecoveryStrategy())
```

## Error Classification

### Severity Levels

- **LOW**: Minor issues that don't affect functionality
- **MEDIUM**: Issues that may affect some functionality
- **HIGH**: Significant issues that affect core functionality
- **CRITICAL**: Critical failures that may cause system outage

### Category Types

- **SYSTEM**: Hardware, OS, runtime issues
- **CONFIG**: Configuration problems
- **DATA**: Data validation, corruption, format issues
- **NETWORK**: Network connectivity problems
- **SECURITY**: Security-related issues
- **PERFORMANCE**: Performance threshold violations
- **VALIDATION**: Input validation failures
- **AUTHENTICATION**: Authentication problems
- **AUTHORIZATION**: Authorization failures
- **DEPENDENCY**: External service failures

## Monitoring and Alerting

### Alert Types

- **THRESHOLD_EXCEEDED**: Error count exceeds threshold
- **ERROR_RATE_HIGH**: Error rate exceeds threshold
- **CRITICAL_ERROR**: Critical error occurred
- **REPEATED_ERROR**: Same error repeated multiple times
- **TREND_ANOMALY**: Unusual error patterns detected

### Notification Channels

- **Email**: SMTP-based email notifications
- **Webhook**: HTTP webhook notifications
- **Logging**: Structured log messages
- **Custom**: Extensible notification system

## Configuration

### Global Configuration

```python
from src.core.errors import ErrorLogger, ErrorMonitor, ErrorRecoveryManager

# Configure logging
logger_config = {
    "log_level": "INFO",
    "log_file": "/var/log/grandmodel/errors.log",
    "enable_structured_logging": True,
    "enable_metrics": True
}

# Configure monitoring
monitor_config = {
    "notifications": {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.company.com",
            "from": "alerts@company.com",
            "to": ["dev-team@company.com"]
        }
    }
}

# Configure recovery
recovery_config = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "circuit_breaker_threshold": 5,
    "enable_graceful_degradation": True
}
```

### Environment Variables

```bash
# Error handling configuration
GRANDMODEL_ERROR_LOG_LEVEL=INFO
GRANDMODEL_ERROR_LOG_FILE=/var/log/grandmodel/errors.log
GRANDMODEL_ERROR_MONITORING_ENABLED=true
GRANDMODEL_ERROR_RECOVERY_ENABLED=true

# Alerting configuration
GRANDMODEL_ALERT_EMAIL_ENABLED=true
GRANDMODEL_ALERT_EMAIL_SMTP_SERVER=smtp.company.com
GRANDMODEL_ALERT_EMAIL_FROM=alerts@company.com
GRANDMODEL_ALERT_EMAIL_TO=dev-team@company.com
```

## Performance Considerations

### Minimal Overhead

- **Lazy Loading**: Components loaded only when needed
- **Efficient Logging**: Structured logging with minimal serialization
- **Async Support**: Full async/await compatibility
- **Memory Management**: Automatic cleanup and resource management

### Monitoring Impact

- **Background Processing**: Monitoring runs in background threads
- **Batch Operations**: Efficient batch processing of errors
- **Caching**: Intelligent caching of error patterns
- **Resource Limits**: Configurable resource limits

## Testing

### Unit Tests

```python
import pytest
from src.core.errors import BaseGrandModelError, ErrorHandler

def test_error_handling():
    handler = ErrorHandler()
    
    def failing_function():
        raise ValueError("Test error")
    
    # Test retry mechanism
    with pytest.raises(ValueError):
        handler.execute_with_retry(failing_function)
    
    # Test fallback
    result = handler.handle_exception(
        ValueError("Test error"),
        fallback_name="test_fallback"
    )
    assert result is not None
```

### Integration Tests

```python
from src.core.errors import ErrorMonitor, ErrorLogger

def test_end_to_end_error_handling():
    monitor = ErrorMonitor()
    logger = ErrorLogger()
    
    # Create test error
    error = BaseGrandModelError("Test error")
    
    # Log and monitor
    correlation_id = logger.log_error(error)
    monitor.monitor_error(logger.get_recent_errors()[0])
    
    # Verify metrics
    metrics = logger.get_metrics()
    assert metrics["total_errors"] > 0
```

## Best Practices

### 1. **Always Use Specific Exceptions**
```python
# ✅ Good
try:
    operation()
except (ValueError, TypeError) as e:
    handle_exception(e)

# ❌ Bad
try:
    operation()
except:
    pass
```

### 2. **Include Context Information**
```python
from src.core.errors import ErrorContext

context = ErrorContext(
    user_id="123",
    request_id="abc",
    additional_data={"operation": "user_login"}
)

error = BaseGrandModelError(
    message="Login failed",
    context=context
)
```

### 3. **Use Correlation IDs**
```python
# Always include correlation IDs for tracking
correlation_id = logger.log_error(error)
print(f"Error logged with ID: {correlation_id}")
```

### 4. **Implement Fallback Strategies**
```python
@with_error_handling(fallback_name="cache_fallback")
def get_user_data(user_id):
    # Primary data source
    return fetch_from_database(user_id)

# Register fallback
def cache_fallback(error, context):
    # Return cached data
    return get_cached_user_data(context.get("user_id"))
```

### 5. **Monitor Error Trends**
```python
# Set up monitoring for error trends
monitor = ErrorMonitor()
monitor.start()

# Check dashboard regularly
dashboard_data = monitor.get_dashboard_data()
```

## Migration Guide

### From Bare Except to Structured Handling

1. **Identify Bare Except Clauses**
   ```bash
   grep -r "except:" src/
   ```

2. **Replace with Specific Exceptions**
   ```python
   # Old
   try:
       risky_operation()
   except:
       pass
   
   # New
   try:
       risky_operation()
   except (SpecificError, AnotherError) as e:
       handle_exception(e)
   ```

3. **Add Error Context**
   ```python
   from src.core.errors import ErrorContext
   
   context = ErrorContext(
       service_name="my_service",
       additional_data={"operation": "specific_operation"}
   )
   ```

4. **Implement Recovery**
   ```python
   from src.core.errors import attempt_recovery
   
   try:
       risky_operation()
   except Exception as e:
       success, result = attempt_recovery(e, context)
       if not success:
           # Escalate or fail gracefully
           pass
   ```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure proper imports
   from src.core.errors import BaseGrandModelError
   ```

2. **Configuration Issues**
   ```python
   # Check configuration
   from src.core.errors import get_global_error_logger
   logger = get_global_error_logger()
   ```

3. **Performance Issues**
   ```python
   # Disable features if needed
   config = ErrorLoggerConfig(
       enable_metrics=False,
       enable_structured_logging=False
   )
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
from src.core.errors import ErrorLogger
logger = ErrorLogger()
logger.config.log_level = logging.DEBUG
```

## Contributing

### Adding New Exception Types

1. **Extend Base Exception**
   ```python
   class MyCustomError(BaseGrandModelError):
       def __init__(self, message, custom_field=None):
           super().__init__(
               message=message,
               category=ErrorCategory.CUSTOM,
               error_details={"custom_field": custom_field}
           )
   ```

2. **Add to Exports**
   ```python
   # In __init__.py
   from .base_exceptions import MyCustomError
   
   __all__ = [
       # ... existing exports
       'MyCustomError'
   ]
   ```

### Adding Recovery Strategies

1. **Implement Recovery Strategy**
   ```python
   class MyRecoveryStrategy(RecoveryStrategy):
       def can_recover(self, error):
           return isinstance(error, MyCustomError)
       
       def recover(self, error, context):
           # Recovery logic
           return {"action": "recovered"}
       
       def get_priority(self):
           return 1
   ```

2. **Register Strategy**
   ```python
   from src.core.errors import get_global_recovery_manager
   
   recovery_manager = get_global_recovery_manager()
   recovery_manager.add_strategy(MyRecoveryStrategy())
   ```

## License

This error handling framework is part of the GrandModel project and follows the same licensing terms.

---

**For more information, see the individual module documentation and examples in the test suite.**