#!/usr/bin/env python3
"""
Centralized Logging System for GrandModel MARL Trading System
High-performance structured logging with real-time analysis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import traceback
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import redis
import elasticsearch
from elasticsearch import AsyncElasticsearch
import structlog
from prometheus_client import Counter, Histogram, Gauge
import fluentd
from fluentd import sender

# Logging metrics
LOG_MESSAGES_TOTAL = Counter('log_messages_total', 'Total log messages', ['level', 'service', 'component'])
LOG_PROCESSING_DURATION = Histogram('log_processing_duration_seconds', 'Log processing time', ['destination'])
LOG_ERRORS_TOTAL = Counter('log_errors_total', 'Log processing errors', ['error_type', 'destination'])
LOG_BUFFER_SIZE = Gauge('log_buffer_size', 'Current log buffer size', ['destination'])

class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class LogCategory(Enum):
    """Log categories for classification."""
    SYSTEM = "system"
    TRADING = "trading"
    MARL = "marl"
    RISK = "risk"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    AUDIT = "audit"

@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: datetime
    level: LogLevel
    service: str
    component: str
    category: LogCategory
    message: str
    correlation_id: str = None
    user_id: str = None
    session_id: str = None
    trace_id: str = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate correlation ID if not provided."""
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'level_value': self.level.value,
            'service': self.service,
            'component': self.component,
            'category': self.category.value,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'trace_id': self.trace_id,
            **self.extra_fields
        }

class LogDestination:
    """Base class for log destinations."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        self.flush_interval = config.get('flush_interval', 5)
        
    async def send(self, log_event: LogEvent) -> bool:
        """Send log event to destination."""
        raise NotImplementedError("Subclasses must implement send method")
    
    async def flush(self) -> bool:
        """Flush buffered log events."""
        raise NotImplementedError("Subclasses must implement flush method")

class ElasticsearchLogDestination(LogDestination):
    """Elasticsearch log destination."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.client = AsyncElasticsearch(
            hosts=config.get('hosts', ['localhost:9200']),
            http_auth=(config.get('username'), config.get('password')) if config.get('username') else None,
            use_ssl=config.get('use_ssl', False),
            verify_certs=config.get('verify_certs', True)
        )
        self.index_prefix = config.get('index_prefix', 'grandmodel-logs')
        
    async def send(self, log_event: LogEvent) -> bool:
        """Send log event to Elasticsearch."""
        start_time = time.time()
        
        try:
            # Generate index name with date
            index_name = f"{self.index_prefix}-{log_event.timestamp.strftime('%Y.%m.%d')}"
            
            # Prepare document
            doc = log_event.to_dict()
            
            # Add Elasticsearch-specific fields
            doc['@timestamp'] = log_event.timestamp.isoformat()
            
            # Send to Elasticsearch
            await self.client.index(
                index=index_name,
                body=doc,
                doc_type='_doc'
            )
            
            LOG_PROCESSING_DURATION.labels(destination=self.name).observe(time.time() - start_time)
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='elasticsearch', destination=self.name).inc()
            print(f"Elasticsearch logging error: {e}")
            return False
    
    async def flush(self) -> bool:
        """Flush buffered events to Elasticsearch."""
        if not self.buffer:
            return True
            
        with self.buffer_lock:
            events_to_send = self.buffer.copy()
            self.buffer.clear()
        
        if not events_to_send:
            return True
        
        try:
            # Prepare bulk request
            bulk_body = []
            for event in events_to_send:
                index_name = f"{self.index_prefix}-{event.timestamp.strftime('%Y.%m.%d')}"
                
                # Index action
                bulk_body.append({
                    'index': {
                        '_index': index_name,
                        '_type': '_doc'
                    }
                })
                
                # Document
                doc = event.to_dict()
                doc['@timestamp'] = event.timestamp.isoformat()
                bulk_body.append(doc)
            
            # Send bulk request
            response = await self.client.bulk(body=bulk_body)
            
            # Check for errors
            if response.get('errors'):
                error_count = sum(1 for item in response['items'] if 'error' in item.get('index', {}))
                LOG_ERRORS_TOTAL.labels(error_type='elasticsearch_bulk', destination=self.name).inc(error_count)
                return False
            
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='elasticsearch_bulk', destination=self.name).inc()
            print(f"Elasticsearch bulk logging error: {e}")
            return False

class RedisLogDestination(LogDestination):
    """Redis log destination."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.client = redis.Redis(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6379),
            db=config.get('db', 0),
            decode_responses=True
        )
        self.stream_name = config.get('stream_name', 'grandmodel:logs')
        self.max_length = config.get('max_length', 10000)
        
    async def send(self, log_event: LogEvent) -> bool:
        """Send log event to Redis stream."""
        start_time = time.time()
        
        try:
            # Prepare fields for Redis stream
            fields = log_event.to_dict()
            
            # Add to Redis stream
            self.client.xadd(
                self.stream_name,
                fields,
                maxlen=self.max_length
            )
            
            LOG_PROCESSING_DURATION.labels(destination=self.name).observe(time.time() - start_time)
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='redis', destination=self.name).inc()
            print(f"Redis logging error: {e}")
            return False
    
    async def flush(self) -> bool:
        """Flush buffered events to Redis."""
        if not self.buffer:
            return True
            
        with self.buffer_lock:
            events_to_send = self.buffer.copy()
            self.buffer.clear()
        
        try:
            # Use pipeline for bulk operations
            pipeline = self.client.pipeline()
            
            for event in events_to_send:
                fields = event.to_dict()
                pipeline.xadd(
                    self.stream_name,
                    fields,
                    maxlen=self.max_length
                )
            
            # Execute pipeline
            pipeline.execute()
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='redis_bulk', destination=self.name).inc()
            print(f"Redis bulk logging error: {e}")
            return False

class FluentdLogDestination(LogDestination):
    """Fluentd log destination."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sender = sender.FluentSender(
            config.get('tag', 'grandmodel'),
            host=config.get('host', 'localhost'),
            port=config.get('port', 24224)
        )
        
    async def send(self, log_event: LogEvent) -> bool:
        """Send log event to Fluentd."""
        start_time = time.time()
        
        try:
            # Send to Fluentd
            success = self.sender.emit(
                f"{log_event.service}.{log_event.component}",
                log_event.to_dict()
            )
            
            if success:
                LOG_PROCESSING_DURATION.labels(destination=self.name).observe(time.time() - start_time)
                return True
            else:
                LOG_ERRORS_TOTAL.labels(error_type='fluentd', destination=self.name).inc()
                return False
                
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='fluentd', destination=self.name).inc()
            print(f"Fluentd logging error: {e}")
            return False
    
    async def flush(self) -> bool:
        """Flush buffered events to Fluentd."""
        if not self.buffer:
            return True
            
        with self.buffer_lock:
            events_to_send = self.buffer.copy()
            self.buffer.clear()
        
        try:
            for event in events_to_send:
                self.sender.emit(
                    f"{event.service}.{event.component}",
                    event.to_dict()
                )
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='fluentd_bulk', destination=self.name).inc()
            print(f"Fluentd bulk logging error: {e}")
            return False

class FileLogDestination(LogDestination):
    """File log destination."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.file_path = config.get('file_path', '/var/log/grandmodel/grandmodel.log')
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.backup_count = config.get('backup_count', 5)
        
    async def send(self, log_event: LogEvent) -> bool:
        """Send log event to file."""
        start_time = time.time()
        
        try:
            # Format log message
            log_line = json.dumps(log_event.to_dict()) + '\n'
            
            # Write to file
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(log_line)
                f.flush()
            
            LOG_PROCESSING_DURATION.labels(destination=self.name).observe(time.time() - start_time)
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='file', destination=self.name).inc()
            print(f"File logging error: {e}")
            return False
    
    async def flush(self) -> bool:
        """Flush buffered events to file."""
        if not self.buffer:
            return True
            
        with self.buffer_lock:
            events_to_send = self.buffer.copy()
            self.buffer.clear()
        
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                for event in events_to_send:
                    log_line = json.dumps(event.to_dict()) + '\n'
                    f.write(log_line)
                f.flush()
            return True
            
        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type='file_bulk', destination=self.name).inc()
            print(f"File bulk logging error: {e}")
            return False

class CentralizedLoggingSystem:
    """Centralized logging system for GrandModel."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.destinations = {}
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        self.flush_task = None
        self.context_storage = threading.local()
        
        # Initialize destinations
        self._initialize_destinations()
        
        # Configure structured logging
        self._configure_structured_logging()
        
        # Start processing
        self._start_processing()
    
    def _initialize_destinations(self):
        """Initialize log destinations."""
        destinations_config = self.config.get('destinations', {})
        
        for dest_name, dest_config in destinations_config.items():
            dest_type = dest_config.get('type')
            
            if dest_type == 'elasticsearch':
                self.destinations[dest_name] = ElasticsearchLogDestination(dest_name, dest_config)
            elif dest_type == 'redis':
                self.destinations[dest_name] = RedisLogDestination(dest_name, dest_config)
            elif dest_type == 'fluentd':
                self.destinations[dest_name] = FluentdLogDestination(dest_name, dest_config)
            elif dest_type == 'file':
                self.destinations[dest_name] = FileLogDestination(dest_name, dest_config)
            else:
                print(f"Unknown destination type: {dest_type}")
    
    def _configure_structured_logging(self):
        """Configure structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_context_processor,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _add_context_processor(self, logger, method_name, event_dict):
        """Add context information to log events."""
        # Add correlation ID if available
        correlation_id = getattr(self.context_storage, 'correlation_id', None)
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        
        # Add user ID if available
        user_id = getattr(self.context_storage, 'user_id', None)
        if user_id:
            event_dict['user_id'] = user_id
        
        # Add session ID if available
        session_id = getattr(self.context_storage, 'session_id', None)
        if session_id:
            event_dict['session_id'] = session_id
        
        # Add trace ID if available
        trace_id = getattr(self.context_storage, 'trace_id', None)
        if trace_id:
            event_dict['trace_id'] = trace_id
        
        return event_dict
    
    def _start_processing(self):
        """Start log processing tasks."""
        self.processing_task = asyncio.create_task(self._process_logs())
        self.flush_task = asyncio.create_task(self._flush_logs())
    
    async def _process_logs(self):
        """Process logs from queue."""
        while True:
            try:
                log_event = await self.log_queue.get()
                
                # Send to all destinations
                tasks = []
                for destination in self.destinations.values():
                    task = asyncio.create_task(destination.send(log_event))
                    tasks.append(task)
                
                # Wait for all destinations to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update metrics
                LOG_MESSAGES_TOTAL.labels(
                    level=log_event.level.name,
                    service=log_event.service,
                    component=log_event.component
                ).inc()
                
            except Exception as e:
                print(f"Error processing log: {e}")
    
    async def _flush_logs(self):
        """Periodically flush log destinations."""
        while True:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                
                # Flush all destinations
                tasks = []
                for destination in self.destinations.values():
                    task = asyncio.create_task(destination.flush())
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                print(f"Error flushing logs: {e}")
    
    async def log(self, level: LogLevel, service: str, component: str, 
                  category: LogCategory, message: str, **kwargs):
        """Log a message."""
        
        # Create log event
        log_event = LogEvent(
            timestamp=datetime.utcnow(),
            level=level,
            service=service,
            component=component,
            category=category,
            message=message,
            correlation_id=kwargs.get('correlation_id'),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            trace_id=kwargs.get('trace_id'),
            extra_fields={k: v for k, v in kwargs.items() if k not in ['correlation_id', 'user_id', 'session_id', 'trace_id']}
        )
        
        # Add to queue
        try:
            await self.log_queue.put(log_event)
        except asyncio.QueueFull:
            print(f"Log queue full, dropping message: {message}")
    
    @contextmanager
    def context(self, correlation_id: str = None, user_id: str = None, 
                session_id: str = None, trace_id: str = None):
        """Context manager for setting log context."""
        
        # Store current context
        old_correlation_id = getattr(self.context_storage, 'correlation_id', None)
        old_user_id = getattr(self.context_storage, 'user_id', None)
        old_session_id = getattr(self.context_storage, 'session_id', None)
        old_trace_id = getattr(self.context_storage, 'trace_id', None)
        
        # Set new context
        if correlation_id:
            self.context_storage.correlation_id = correlation_id
        if user_id:
            self.context_storage.user_id = user_id
        if session_id:
            self.context_storage.session_id = session_id
        if trace_id:
            self.context_storage.trace_id = trace_id
        
        try:
            yield
        finally:
            # Restore previous context
            self.context_storage.correlation_id = old_correlation_id
            self.context_storage.user_id = old_user_id
            self.context_storage.session_id = old_session_id
            self.context_storage.trace_id = old_trace_id
    
    def create_logger(self, service: str, component: str) -> 'ComponentLogger':
        """Create a component-specific logger."""
        return ComponentLogger(self, service, component)
    
    async def search_logs(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs (if Elasticsearch is configured)."""
        
        # Find Elasticsearch destination
        es_destination = None
        for dest in self.destinations.values():
            if isinstance(dest, ElasticsearchLogDestination):
                es_destination = dest
                break
        
        if not es_destination:
            return []
        
        try:
            # Build search query
            search_body = {
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
            
            # Search
            response = await es_destination.client.search(
                index=f"{es_destination.index_prefix}-*",
                body=search_body
            )
            
            # Return results
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            print(f"Error searching logs: {e}")
            return []
    
    async def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'queue_size': self.log_queue.qsize(),
            'destinations': {},
            'total_messages': 0,
            'total_errors': 0
        }
        
        # Get destination-specific stats
        for dest_name, destination in self.destinations.items():
            buffer_size = len(destination.buffer)
            stats['destinations'][dest_name] = {
                'buffer_size': buffer_size,
                'type': destination.__class__.__name__
            }
            LOG_BUFFER_SIZE.labels(destination=dest_name).set(buffer_size)
        
        return stats
    
    async def shutdown(self):
        """Shutdown the logging system."""
        print("Shutting down logging system...")
        
        # Cancel processing tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.flush_task:
            self.flush_task.cancel()
        
        # Final flush
        tasks = []
        for destination in self.destinations.values():
            task = asyncio.create_task(destination.flush())
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close Elasticsearch connections
        for destination in self.destinations.values():
            if isinstance(destination, ElasticsearchLogDestination):
                await destination.client.close()

class ComponentLogger:
    """Component-specific logger."""
    
    def __init__(self, logging_system: CentralizedLoggingSystem, service: str, component: str):
        self.logging_system = logging_system
        self.service = service
        self.component = component
    
    async def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message."""
        await self.logging_system.log(LogLevel.DEBUG, self.service, self.component, category, message, **kwargs)
    
    async def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message."""
        await self.logging_system.log(LogLevel.INFO, self.service, self.component, category, message, **kwargs)
    
    async def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message."""
        await self.logging_system.log(LogLevel.WARNING, self.service, self.component, category, message, **kwargs)
    
    async def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log error message."""
        await self.logging_system.log(LogLevel.ERROR, self.service, self.component, category, message, **kwargs)
    
    async def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log critical message."""
        await self.logging_system.log(LogLevel.CRITICAL, self.service, self.component, category, message, **kwargs)

# Factory function
def create_logging_system(config: Dict[str, Any]) -> CentralizedLoggingSystem:
    """Create centralized logging system."""
    return CentralizedLoggingSystem(config)

# Example configuration
EXAMPLE_CONFIG = {
    "destinations": {
        "elasticsearch": {
            "type": "elasticsearch",
            "hosts": ["localhost:9200"],
            "index_prefix": "grandmodel-logs",
            "max_buffer_size": 1000,
            "flush_interval": 5
        },
        "redis": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "stream_name": "grandmodel:logs",
            "max_length": 10000
        },
        "file": {
            "type": "file",
            "file_path": "/var/log/grandmodel/grandmodel.log",
            "max_file_size": 104857600,  # 100MB
            "backup_count": 5
        }
    }
}

# Example usage
async def main():
    """Example usage of centralized logging."""
    config = EXAMPLE_CONFIG
    logging_system = create_logging_system(config)
    
    # Create component logger
    logger = logging_system.create_logger("trading_engine", "execution_handler")
    
    # Log some messages
    await logger.info("Trading engine started", LogCategory.TRADING, startup_time=time.time())
    await logger.warning("High latency detected", LogCategory.PERFORMANCE, latency_ms=150)
    await logger.error("Trade execution failed", LogCategory.TRADING, trade_id="T123", error_code="TIMEOUT")
    
    # Use context
    with logging_system.context(correlation_id="req-123", user_id="user-456"):
        await logger.info("Processing user request", LogCategory.BUSINESS, action="place_order")
    
    # Get stats
    stats = await logging_system.get_log_stats()
    print(f"Logging stats: {stats}")
    
    # Shutdown
    await logging_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())