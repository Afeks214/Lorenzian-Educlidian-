"""
Test adapters for hexagonal architecture.

These adapters provide test-specific implementations of the port interfaces,
enabling complete control over dependencies in tests.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from datetime import datetime, timedelta
from collections import defaultdict
from unittest.mock import Mock, MagicMock
import json

from .ports import (
    EventBusPort, KernelPort, HealthMonitorPort, DataStoragePort, 
    TimeServicePort, LoggingPort, ConfigurationPort, NetworkPort, 
    FileSystemPort, MetricsPort, Event, EventType, HealthStatus,
    ComponentHealth, LogLevel
)


class TestEventBusAdapter(EventBusPort):
    """Test adapter for event bus with full tracking capabilities."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._published_events: List[Event] = []
        self._event_history: Dict[EventType, List[Event]] = defaultdict(list)
        self._enabled = True
        
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
    
    def publish(self, event: Event) -> None:
        """Publish an event."""
        if not self._enabled:
            return
            
        # Track event
        self._published_events.append(event)
        self._event_history[event.event_type].append(event)
        
        # Notify subscribers
        for callback in self._subscribers[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                # In test environment, we might want to track errors
                pass
    
    def create_event(self, event_type: EventType, payload: Any, source: str) -> Event:
        """Create a new event."""
        return Event(event_type, payload, source)
    
    # Test-specific methods
    def get_published_events(self) -> List[Event]:
        """Get all published events."""
        return self._published_events.copy()
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events of a specific type."""
        return self._event_history[event_type].copy()
    
    def get_last_event(self, event_type: Optional[EventType] = None) -> Optional[Event]:
        """Get the last published event."""
        if event_type:
            events = self._event_history[event_type]
            return events[-1] if events else None
        return self._published_events[-1] if self._published_events else None
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._published_events.clear()
        self._event_history.clear()
    
    def disable(self) -> None:
        """Disable event publishing."""
        self._enabled = False
    
    def enable(self) -> None:
        """Enable event publishing."""
        self._enabled = True
    
    def assert_event_published(self, event_type: EventType, count: Optional[int] = None) -> bool:
        """Assert that an event was published."""
        events = self.get_events_by_type(event_type)
        if not events:
            raise AssertionError(f"No events of type {event_type} were published")
        if count is not None and len(events) != count:
            raise AssertionError(f"Expected {count} events of type {event_type}, got {len(events)}")
        return True


class TestKernelAdapter(KernelPort):
    """Test adapter for kernel with configurable components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._components: Dict[str, Any] = {}
        self._config = config or {}
        self._event_bus = TestEventBusAdapter()
        self._running = False
        self._shutdown_called = False
        
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        return self._components.get(name)
    
    def get_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self._config.copy()
    
    def get_event_bus(self) -> EventBusPort:
        """Get the event bus."""
        return self._event_bus
    
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._running
    
    def shutdown(self) -> None:
        """Shutdown the system."""
        self._running = False
        self._shutdown_called = True
    
    # Test-specific methods
    def add_component(self, name: str, component: Any) -> None:
        """Add a component for testing."""
        self._components[name] = component
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration for testing."""
        self._config = config
    
    def start(self) -> None:
        """Start the system (for testing)."""
        self._running = True
        self._shutdown_called = False
    
    def was_shutdown_called(self) -> bool:
        """Check if shutdown was called."""
        return self._shutdown_called


class TestHealthMonitorAdapter(HealthMonitorPort):
    """Test adapter for health monitoring with configurable responses."""
    
    def __init__(self):
        self._component_health: Dict[str, ComponentHealth] = {}
        self._system_health = HealthStatus.HEALTHY
        self._check_count = 0
        
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage."""
        self._check_count += 1
        return ComponentHealth(
            name="system_resources",
            status=HealthStatus.HEALTHY,
            message="Test system resources healthy",
            details={"cpu_percent": 25.0, "memory_mb": 256}
        )
    
    async def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        if component_name in self._component_health:
            return self._component_health[component_name]
        return ComponentHealth(
            name=component_name,
            status=HealthStatus.HEALTHY,
            message=f"Test component {component_name} healthy"
        )
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        return self._system_health
    
    # Test-specific methods
    def set_component_health(self, component_name: str, health: ComponentHealth) -> None:
        """Set component health for testing."""
        self._component_health[component_name] = health
    
    def set_system_health(self, health: HealthStatus) -> None:
        """Set system health for testing."""
        self._system_health = health
    
    def get_check_count(self) -> int:
        """Get number of checks performed."""
        return self._check_count


class TestDataStorageAdapter(DataStoragePort):
    """Test adapter for data storage with in-memory storage."""
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._operations: List[Dict[str, Any]] = []
        
    async def store(self, key: str, value: Any) -> None:
        """Store data."""
        self._storage[key] = value
        self._operations.append({"operation": "store", "key": key, "value": value})
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data."""
        self._operations.append({"operation": "retrieve", "key": key})
        return self._storage.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete data."""
        self._operations.append({"operation": "delete", "key": key})
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        self._operations.append({"operation": "exists", "key": key})
        return key in self._storage
    
    # Test-specific methods
    def get_storage(self) -> Dict[str, Any]:
        """Get current storage state."""
        return self._storage.copy()
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get all operations performed."""
        return self._operations.copy()
    
    def clear(self) -> None:
        """Clear storage and operations."""
        self._storage.clear()
        self._operations.clear()


class TestTimeServiceAdapter(TimeServicePort):
    """Test adapter for time service with controllable time."""
    
    def __init__(self, initial_time: Optional[datetime] = None):
        self._current_time = initial_time or datetime.now()
        self._time_calls: List[datetime] = []
        self._sleep_calls: List[float] = []
        
    def now(self) -> datetime:
        """Get current time."""
        self._time_calls.append(self._current_time)
        return self._current_time
    
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        self._sleep_calls.append(seconds)
        # In test, we can either actually sleep or just advance time
        self._current_time += timedelta(seconds=seconds)
    
    def set_time(self, time: datetime) -> None:
        """Set current time."""
        self._current_time = time
    
    # Test-specific methods
    def advance_time(self, seconds: float) -> None:
        """Advance time by specified seconds."""
        self._current_time += timedelta(seconds=seconds)
    
    def get_time_calls(self) -> List[datetime]:
        """Get all time calls."""
        return self._time_calls.copy()
    
    def get_sleep_calls(self) -> List[float]:
        """Get all sleep calls."""
        return self._sleep_calls.copy()


class TestLoggingAdapter(LoggingPort):
    """Test adapter for logging with full log capture."""
    
    def __init__(self):
        self._logs: List[Dict[str, Any]] = []
        
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log a message."""
        self._logs.append({
            "level": level.value,
            "message": message,
            "timestamp": datetime.now(),
            "kwargs": kwargs
        })
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get captured logs."""
        return self._logs.copy()
    
    # Test-specific methods
    def get_logs_by_level(self, level: LogLevel) -> List[Dict[str, Any]]:
        """Get logs by level."""
        return [log for log in self._logs if log["level"] == level.value]
    
    def clear_logs(self) -> None:
        """Clear all logs."""
        self._logs.clear()
    
    def assert_log_contains(self, message: str, level: Optional[LogLevel] = None) -> bool:
        """Assert that logs contain a message."""
        logs = self.get_logs_by_level(level) if level else self._logs
        for log in logs:
            if message in log["message"]:
                return True
        raise AssertionError(f"No log found containing '{message}'")


class TestConfigurationAdapter(ConfigurationPort):
    """Test adapter for configuration with mutable config."""
    
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._config = initial_config or {}
        self._get_calls: List[str] = []
        self._set_calls: List[Dict[str, Any]] = []
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        self._get_calls.append(key)
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._set_calls.append({"key": key, "value": value})
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration."""
        # In test, this is a no-op
        pass
    
    # Test-specific methods
    def get_get_calls(self) -> List[str]:
        """Get all get calls."""
        return self._get_calls.copy()
    
    def get_set_calls(self) -> List[Dict[str, Any]]:
        """Get all set calls."""
        return self._set_calls.copy()


class TestNetworkAdapter(NetworkPort):
    """Test adapter for network operations with configurable responses."""
    
    def __init__(self):
        self._http_responses: Dict[str, Dict[str, Any]] = {}
        self._websocket_messages: Dict[str, List[Dict[str, Any]]] = {}
        self._connected = True
        self._requests: List[Dict[str, Any]] = []
        
    async def http_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make HTTP GET request."""
        self._requests.append({"method": "GET", "url": url, "headers": headers})
        return self._http_responses.get(url, {"status": 200, "data": {}})
    
    async def http_post(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make HTTP POST request."""
        self._requests.append({"method": "POST", "url": url, "data": data, "headers": headers})
        return self._http_responses.get(url, {"status": 200, "data": {}})
    
    async def websocket_connect(self, url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Connect to websocket."""
        messages = self._websocket_messages.get(url, [])
        for message in messages:
            yield message
    
    def is_connected(self) -> bool:
        """Check if network is connected."""
        return self._connected
    
    # Test-specific methods
    def set_http_response(self, url: str, response: Dict[str, Any]) -> None:
        """Set HTTP response for URL."""
        self._http_responses[url] = response
    
    def set_websocket_messages(self, url: str, messages: List[Dict[str, Any]]) -> None:
        """Set websocket messages for URL."""
        self._websocket_messages[url] = messages
    
    def set_connected(self, connected: bool) -> None:
        """Set connection status."""
        self._connected = connected
    
    def get_requests(self) -> List[Dict[str, Any]]:
        """Get all requests made."""
        return self._requests.copy()


class TestFileSystemAdapter(FileSystemPort):
    """Test adapter for file system operations with in-memory files."""
    
    def __init__(self):
        self._files: Dict[str, str] = {}
        self._operations: List[Dict[str, Any]] = []
        
    def read_file(self, path: str) -> str:
        """Read file content."""
        self._operations.append({"operation": "read", "path": path})
        if path not in self._files:
            raise FileNotFoundError(f"File {path} not found")
        return self._files[path]
    
    def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        self._operations.append({"operation": "write", "path": path, "content": content})
        self._files[path] = content
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        self._operations.append({"operation": "exists", "path": path})
        return path in self._files
    
    def delete_file(self, path: str) -> bool:
        """Delete file."""
        self._operations.append({"operation": "delete", "path": path})
        if path in self._files:
            del self._files[path]
            return True
        return False
    
    def list_files(self, directory: str) -> List[str]:
        """List files in directory."""
        self._operations.append({"operation": "list", "directory": directory})
        # Simple implementation - return files that start with directory path
        return [path for path in self._files.keys() if path.startswith(directory)]
    
    # Test-specific methods
    def get_files(self) -> Dict[str, str]:
        """Get current file system state."""
        return self._files.copy()
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get all operations performed."""
        return self._operations.copy()


class TestMetricsAdapter(MetricsPort):
    """Test adapter for metrics with full metric capture."""
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._operations: List[Dict[str, Any]] = []
        
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self._counters[name] += value
        self._operations.append({
            "type": "counter",
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": datetime.now()
        })
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self._gauges[name] = value
        self._operations.append({
            "type": "gauge",
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": datetime.now()
        })
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram value."""
        self._histograms[name].append(value)
        self._operations.append({
            "type": "histogram",
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": datetime.now()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": dict(self._histograms)
        }
    
    # Test-specific methods
    def get_counter_value(self, name: str) -> int:
        """Get counter value."""
        return self._counters[name]
    
    def get_gauge_value(self, name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(name)
    
    def get_histogram_values(self, name: str) -> List[float]:
        """Get histogram values."""
        return self._histograms[name].copy()
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get all operations."""
        return self._operations.copy()
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._operations.clear()