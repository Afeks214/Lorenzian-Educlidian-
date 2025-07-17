#!/usr/bin/env python3
"""
AGENT 13: Structured Logging with Correlation IDs and Centralized Aggregation
Comprehensive logging system with structured format, correlation tracking, and centralized aggregation
"""

import json
import logging
import time
import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
import sys
import os
import queue
import gzip
import traceback
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import socket
import redis
import elasticsearch
from prometheus_client import Counter, Histogram, Gauge

# Logging metrics
LOG_MESSAGES_TOTAL = Counter(
    'log_messages_total',
    'Total number of log messages',
    ['level', 'service', 'component']
)

LOG_PROCESSING_TIME = Histogram(
    'log_processing_time_seconds',
    'Log processing time in seconds',
    ['handler_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf')]
)

LOG_ERRORS_TOTAL = Counter(
    'log_errors_total',
    'Total number of logging errors',
    ['error_type', 'handler']
)

LOG_BUFFER_SIZE = Gauge(
    'log_buffer_size',
    'Current size of log buffer',
    ['buffer_type']
)

class LogLevel(Enum):
    """Custom log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class LogComponent(Enum):
    """System components for logging."""
    TRADING_ENGINE = "trading_engine"
    MARL_AGENTS = "marl_agents"
    RISK_MANAGEMENT = "risk_management"
    DATA_PIPELINE = "data_pipeline"
    EXECUTION_ENGINE = "execution_engine"
    MONITORING = "monitoring"
    API = "api"
    AUTHENTICATION = "authentication"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    SYSTEM = "system"

@dataclass
class LogContext:
    """Logging context with correlation information."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trade_id: Optional[str] = None
    strategy_id: Optional[str] = None
    component: Optional[LogComponent] = None
    service: str = "grandmodel"
    version: str = "1.0.0"
    environment: str = "production"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enum to string
        if self.component:
            data['component'] = self.component.value
        return data

@dataclass
class StructuredLogRecord:
    """Structured log record with all metadata."""
    timestamp: datetime
    level: str
    message: str
    context: LogContext
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    hostname: str
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    business_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'context': self.context.to_dict(),
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'hostname': self.hostname,
            'extra_fields': self.extra_fields
        }
        
        if self.exception_info:
            data['exception_info'] = self.exception_info
        if self.performance_metrics:
            data['performance_metrics'] = self.performance_metrics
        if self.business_metrics:
            data['business_metrics'] = self.business_metrics
            
        return data
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class CorrelationContextManager:
    """Thread-local context manager for correlation IDs."""
    
    def __init__(self):
        self._context = threading.local()
        
    def set_context(self, context: LogContext):
        """Set logging context for current thread."""
        self._context.log_context = context
        
    def get_context(self) -> LogContext:
        """Get logging context for current thread."""
        if not hasattr(self._context, 'log_context'):
            self._context.log_context = LogContext()
        return self._context.log_context
        
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, 'log_context'):
            del self._context.log_context
            
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context updates."""
        old_context = self.get_context()
        new_context = LogContext(**{**asdict(old_context), **kwargs})
        
        self.set_context(new_context)
        try:
            yield new_context
        finally:
            self.set_context(old_context)

# Global correlation context manager
correlation_context = CorrelationContextManager()

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Get correlation context
        context = correlation_context.get_context()
        
        # Create structured record
        structured_record = StructuredLogRecord(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc),
            level=record.levelname,
            message=record.getMessage(),
            context=context,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            hostname=socket.gethostname(),
            extra_fields=getattr(record, 'extra_fields', {})
        )
        
        # Add exception info if present
        if record.exc_info:
            structured_record.exception_info = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add performance metrics if present
        if hasattr(record, 'performance_metrics'):
            structured_record.performance_metrics = record.performance_metrics
            
        # Add business metrics if present
        if hasattr(record, 'business_metrics'):
            structured_record.business_metrics = record.business_metrics
            
        return structured_record.to_json()

class ElasticsearchHandler(logging.Handler):
    """Handler for sending logs to Elasticsearch."""
    
    def __init__(self, elasticsearch_client: elasticsearch.Elasticsearch, index_pattern: str = "grandmodel-logs"):
        super().__init__()
        self.es_client = elasticsearch_client
        self.index_pattern = index_pattern
        self.buffer = queue.Queue(maxsize=1000)
        self.batch_size = 100
        self.flush_interval = 5  # seconds
        
        # Start background thread for batch processing
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        
    def emit(self, record: logging.LogRecord):
        """Add log record to buffer."""
        try:
            if not self.buffer.full():
                self.buffer.put(record, block=False)
            else:
                LOG_ERRORS_TOTAL.labels(error_type="buffer_full", handler="elasticsearch").inc()
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type="emit_error", handler="elasticsearch").inc()
            self.handleError(record)
            
    def _batch_processor(self):
        """Background thread for batch processing logs."""
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Try to get record with timeout
                try:
                    record = self.buffer.get(timeout=1)
                    batch.append(record)
                except queue.Empty:
                    pass
                    
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_flush >= self.flush_interval)
                )
                
                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = current_time
                    
                # Update buffer size metric
                LOG_BUFFER_SIZE.labels(buffer_type="elasticsearch").set(self.buffer.qsize())
                
            except Exception as e:
                LOG_ERRORS_TOTAL.labels(error_type="batch_processing", handler="elasticsearch").inc()
                batch = []  # Clear batch on error
                time.sleep(1)
                
    def _flush_batch(self, batch: List[logging.LogRecord]):
        """Flush batch of log records to Elasticsearch."""
        if not batch:
            return
            
        start_time = time.time()
        
        try:
            # Prepare bulk index operations
            actions = []
            
            for record in batch:
                # Format record
                formatted_record = self.format(record)
                log_data = json.loads(formatted_record)
                
                # Generate index name with date
                index_name = f"{self.index_pattern}-{datetime.utcnow().strftime('%Y.%m.%d')}"
                
                # Add to bulk actions
                actions.append({
                    '_index': index_name,
                    '_source': log_data
                })
                
            # Bulk index to Elasticsearch
            self.es_client.bulk(body=actions)
            
            # Record metrics
            processing_time = time.time() - start_time
            LOG_PROCESSING_TIME.labels(handler_type="elasticsearch").observe(processing_time)
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type="flush_error", handler="elasticsearch").inc()
            # In production, you might want to retry or save to fallback storage
            
class RedisHandler(logging.Handler):
    """Handler for sending logs to Redis streams."""
    
    def __init__(self, redis_client: redis.Redis, stream_name: str = "grandmodel:logs"):
        super().__init__()
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.max_stream_length = 10000  # Maximum entries in stream
        
    def emit(self, record: logging.LogRecord):
        """Send log record to Redis stream."""
        start_time = time.time()
        
        try:
            # Format record
            formatted_record = self.format(record)
            log_data = json.loads(formatted_record)
            
            # Add to Redis stream
            self.redis_client.xadd(
                self.stream_name,
                log_data,
                maxlen=self.max_stream_length,
                approximate=True
            )
            
            # Record metrics
            processing_time = time.time() - start_time
            LOG_PROCESSING_TIME.labels(handler_type="redis").observe(processing_time)
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type="emit_error", handler="redis").inc()
            self.handleError(record)

class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.name = name
        
        # Add structured formatter
        self.formatter = StructuredFormatter()
        
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured data."""
        # Create log record
        record = self.logger.makeRecord(
            name=self.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=kwargs.get('exc_info')
        )
        
        # Add extra fields
        record.extra_fields = kwargs.get('extra_fields', {})
        record.performance_metrics = kwargs.get('performance_metrics')
        record.business_metrics = kwargs.get('business_metrics')
        
        # Handle the record
        self.logger.handle(record)
        
        # Record metrics
        context = correlation_context.get_context()
        LOG_MESSAGES_TOTAL.labels(
            level=logging.getLevelName(level),
            service=context.service,
            component=context.component.value if context.component else "unknown"
        ).inc()
        
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE.value, message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG.value, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO.value, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING.value, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR.value, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL.value, message, **kwargs)
        
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            performance_metrics={
                'operation': operation,
                'duration_seconds': duration,
                **metrics
            }
        )
        
    def log_business_event(self, event_type: str, **data):
        """Log business event."""
        self.info(
            f"Business event: {event_type}",
            business_metrics={
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                **data
            }
        )
        
    def log_trade_event(self, trade_id: str, event_type: str, **data):
        """Log trade-specific event."""
        with correlation_context.context(trade_id=trade_id):
            self.log_business_event(event_type, trade_id=trade_id, **data)
            
    def log_error_with_context(self, error: Exception, context_data: Dict[str, Any]):
        """Log error with additional context."""
        self.error(
            f"Error occurred: {str(error)}",
            exc_info=True,
            extra_fields={
                'error_type': type(error).__name__,
                'error_context': context_data
            }
        )
        
    @contextmanager
    def performance_context(self, operation: str, **metadata):
        """Context manager for performance logging."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            self.log_performance(operation, duration, status="error", error=str(e), **metadata)
            raise
        else:
            duration = time.time() - start_time
            self.log_performance(operation, duration, status="success", **metadata)

class LoggingConfiguration:
    """Configuration for structured logging system."""
    
    def __init__(self):
        self.console_enabled = True
        self.file_enabled = True
        self.elasticsearch_enabled = False
        self.redis_enabled = False
        self.log_level = logging.INFO
        self.log_file_path = "logs/grandmodel.log"
        self.log_file_max_size = 100 * 1024 * 1024  # 100MB
        self.log_file_backup_count = 10
        self.elasticsearch_hosts = ["localhost:9200"]
        self.elasticsearch_index_pattern = "grandmodel-logs"
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_db = 0
        self.redis_stream_name = "grandmodel:logs"
        
    @classmethod
    def from_env(cls) -> 'LoggingConfiguration':
        """Create configuration from environment variables."""
        config = cls()
        
        config.console_enabled = os.getenv('LOG_CONSOLE_ENABLED', 'true').lower() == 'true'
        config.file_enabled = os.getenv('LOG_FILE_ENABLED', 'true').lower() == 'true'
        config.elasticsearch_enabled = os.getenv('LOG_ELASTICSEARCH_ENABLED', 'false').lower() == 'true'
        config.redis_enabled = os.getenv('LOG_REDIS_ENABLED', 'false').lower() == 'true'
        
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        config.log_level = getattr(logging, log_level_str, logging.INFO)
        
        config.log_file_path = os.getenv('LOG_FILE_PATH', config.log_file_path)
        config.elasticsearch_hosts = os.getenv('ELASTICSEARCH_HOSTS', 'localhost:9200').split(',')
        config.redis_host = os.getenv('REDIS_HOST', config.redis_host)
        config.redis_port = int(os.getenv('REDIS_PORT', str(config.redis_port)))
        
        return config

class StructuredLoggingSystem:
    """Main structured logging system."""
    
    def __init__(self, config: LoggingConfiguration):
        self.config = config
        self.loggers = {}
        self.handlers = []
        self.queue = queue.Queue()
        self.queue_listener = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging handlers and configuration."""
        # Clear existing handlers
        logging.getLogger().handlers = []
        
        # Create handlers
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
            self.handlers.append(console_handler)
            
        if self.config.file_enabled:
            # Ensure log directory exists
            log_dir = Path(self.config.log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.config.log_file_path,
                maxBytes=self.config.log_file_max_size,
                backupCount=self.config.log_file_backup_count
            )
            file_handler.setFormatter(StructuredFormatter())
            self.handlers.append(file_handler)
            
        if self.config.elasticsearch_enabled:
            try:
                es_client = elasticsearch.Elasticsearch(
                    hosts=self.config.elasticsearch_hosts
                )
                es_handler = ElasticsearchHandler(
                    es_client,
                    self.config.elasticsearch_index_pattern
                )
                es_handler.setFormatter(StructuredFormatter())
                self.handlers.append(es_handler)
            except Exception as e:
                print(f"Failed to setup Elasticsearch handler: {e}")
                
        if self.config.redis_enabled:
            try:
                redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db
                )
                redis_handler = RedisHandler(
                    redis_client,
                    self.config.redis_stream_name
                )
                redis_handler.setFormatter(StructuredFormatter())
                self.handlers.append(redis_handler)
            except Exception as e:
                print(f"Failed to setup Redis handler: {e}")
                
        # Setup queue listener for asynchronous logging
        if self.handlers:
            queue_handler = QueueHandler(self.queue)
            self.queue_listener = QueueListener(
                self.queue,
                *self.handlers,
                respect_handler_level=True
            )
            self.queue_listener.start()
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(queue_handler)
            root_logger.setLevel(self.config.log_level)
            
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create structured logger."""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, self.config.log_level)
        return self.loggers[name]
        
    def shutdown(self):
        """Shutdown logging system."""
        if self.queue_listener:
            self.queue_listener.stop()
            
        for handler in self.handlers:
            handler.close()

# Global logging system instance
_logging_system = None

def setup_logging(config: Optional[LoggingConfiguration] = None) -> StructuredLoggingSystem:
    """Setup global logging system."""
    global _logging_system
    
    if config is None:
        config = LoggingConfiguration.from_env()
        
    _logging_system = StructuredLoggingSystem(config)
    return _logging_system

def get_logger(name: str) -> StructuredLogger:
    """Get logger from global logging system."""
    if _logging_system is None:
        setup_logging()
    return _logging_system.get_logger(name)

def shutdown_logging():
    """Shutdown global logging system."""
    global _logging_system
    if _logging_system:
        _logging_system.shutdown()
        _logging_system = None

# Example usage and demonstration
if __name__ == "__main__":
    # Setup logging
    config = LoggingConfiguration.from_env()
    logging_system = setup_logging(config)
    
    # Get logger
    logger = get_logger("trading_engine")
    
    # Set correlation context
    with correlation_context.context(
        component=LogComponent.TRADING_ENGINE,
        request_id="req_123",
        user_id="user_456"
    ):
        # Log various message types
        logger.info("Trading engine started")
        
        # Log with performance context
        with logger.performance_context("order_processing", order_id="order_789"):
            time.sleep(0.1)  # Simulate processing
            
        # Log business event
        logger.log_business_event(
            "order_placed",
            order_id="order_789",
            symbol="BTCUSD",
            quantity=1.5,
            price=50000.0
        )
        
        # Log trade event
        logger.log_trade_event(
            "trade_123",
            "trade_executed",
            symbol="BTCUSD",
            quantity=1.0,
            price=50100.0,
            pnl=150.0
        )
        
        # Log error with context
        try:
            raise ValueError("Example error")
        except Exception as e:
            logger.log_error_with_context(
                e,
                {"order_id": "order_789", "symbol": "BTCUSD"}
            )
            
    print("Structured logging example completed")
    
    # Shutdown logging
    shutdown_logging()
