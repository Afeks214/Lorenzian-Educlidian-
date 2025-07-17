"""
Port interfaces for hexagonal architecture testing.

These abstract interfaces define the contracts that all components must adhere to,
enabling complete dependency injection and isolation in tests.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Standard event types for testing."""
    NEW_TICK = "NEW_TICK"
    NEW_BAR = "NEW_BAR"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    TRADE_QUALIFIED = "TRADE_QUALIFIED"
    RISK_BREACH = "RISK_BREACH"


class Event:
    """Standard event structure for testing."""
    def __init__(self, event_type: EventType, payload: Any, source: str, timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.timestamp = timestamp or datetime.now()


class EventBusPort(ABC):
    """Port for event bus communication."""
    
    @abstractmethod
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type."""
        pass
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event."""
        pass
    
    @abstractmethod
    def create_event(self, event_type: EventType, payload: Any, source: str) -> Event:
        """Create a new event."""
        pass


class KernelPort(ABC):
    """Port for kernel/orchestrator communication."""
    
    @abstractmethod
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        pass
    
    @abstractmethod
    def get_event_bus(self) -> EventBusPort:
        """Get the event bus."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if system is running."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the system."""
        pass


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Component health information."""
    def __init__(self, name: str, status: HealthStatus, message: str = "", details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.last_check = datetime.now()


class HealthMonitorPort(ABC):
    """Port for health monitoring."""
    
    @abstractmethod
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage."""
        pass
    
    @abstractmethod
    async def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        pass
    
    @abstractmethod
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        pass


class DataStoragePort(ABC):
    """Port for data storage operations."""
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> None:
        """Store data."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class TimeServicePort(ABC):
    """Port for time-related operations."""
    
    @abstractmethod
    def now(self) -> datetime:
        """Get current time."""
        pass
    
    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        pass
    
    @abstractmethod
    def set_time(self, time: datetime) -> None:
        """Set current time (for testing)."""
        pass


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LoggingPort(ABC):
    """Port for logging operations."""
    
    @abstractmethod
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log a message."""
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get captured logs (for testing)."""
        pass


class ConfigurationPort(ABC):
    """Port for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration."""
        pass


class NetworkPort(ABC):
    """Port for network operations."""
    
    @abstractmethod
    async def http_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make HTTP GET request."""
        pass
    
    @abstractmethod
    async def http_post(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make HTTP POST request."""
        pass
    
    @abstractmethod
    async def websocket_connect(self, url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Connect to websocket."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if network is connected."""
        pass


class FileSystemPort(ABC):
    """Port for file system operations."""
    
    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read file content."""
        pass
    
    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        pass
    
    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    def list_files(self, directory: str) -> List[str]:
        """List files in directory."""
        pass


class MetricsPort(ABC):
    """Port for metrics collection."""
    
    @abstractmethod
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram value."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics (for testing)."""
        pass