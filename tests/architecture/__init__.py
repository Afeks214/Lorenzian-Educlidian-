"""
Hexagonal Architecture Framework for Testing

This module provides the foundational architecture for test isolation using
the ports-and-adapters pattern. It enables complete dependency injection
and mock control for comprehensive testing.
"""

from .ports import (
    EventBusPort,
    KernelPort,
    HealthMonitorPort,
    DataStoragePort,
    TimeServicePort,
    LoggingPort,
    ConfigurationPort,
    NetworkPort,
    FileSystemPort,
    MetricsPort
)

from .adapters import (
    TestEventBusAdapter,
    TestKernelAdapter,
    TestHealthMonitorAdapter,
    TestDataStorageAdapter,
    TestTimeServiceAdapter,
    TestLoggingAdapter,
    TestConfigurationAdapter,
    TestNetworkAdapter,
    TestFileSystemAdapter,
    TestMetricsAdapter
)

from .container import (
    DependencyContainer,
    TestContainer
)

from .fixtures import (
    hexagonal_test_setup,
    isolated_component_test,
    integration_test_setup
)

__all__ = [
    # Ports
    "EventBusPort",
    "KernelPort", 
    "HealthMonitorPort",
    "DataStoragePort",
    "TimeServicePort",
    "LoggingPort",
    "ConfigurationPort",
    "NetworkPort",
    "FileSystemPort",
    "MetricsPort",
    
    # Adapters
    "TestEventBusAdapter",
    "TestKernelAdapter",
    "TestHealthMonitorAdapter", 
    "TestDataStorageAdapter",
    "TestTimeServiceAdapter",
    "TestLoggingAdapter",
    "TestConfigurationAdapter",
    "TestNetworkAdapter",
    "TestFileSystemAdapter",
    "TestMetricsAdapter",
    
    # Container
    "DependencyContainer",
    "TestContainer",
    
    # Fixtures
    "hexagonal_test_setup",
    "isolated_component_test",
    "integration_test_setup"
]