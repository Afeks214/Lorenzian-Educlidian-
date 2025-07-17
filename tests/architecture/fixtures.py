"""
Pytest fixtures for hexagonal architecture testing.

This module provides pytest fixtures that integrate the hexagonal architecture
framework with pytest, enabling easy setup of isolated component tests.
"""

import pytest
from typing import Dict, Any, Optional, Type, TypeVar, Generator
from unittest.mock import Mock, patch
from datetime import datetime

from .container import TestContainer, ComponentTestContext, create_test_container
from .ports import (
    EventBusPort, KernelPort, HealthMonitorPort, DataStoragePort,
    TimeServicePort, LoggingPort, ConfigurationPort, NetworkPort,
    FileSystemPort, MetricsPort, HealthStatus
)
from .adapters import (
    TestEventBusAdapter, TestKernelAdapter, TestHealthMonitorAdapter,
    TestDataStorageAdapter, TestTimeServiceAdapter, TestLoggingAdapter,
    TestConfigurationAdapter, TestNetworkAdapter, TestFileSystemAdapter,
    TestMetricsAdapter
)

T = TypeVar('T')


@pytest.fixture
def test_container() -> Generator[TestContainer, None, None]:
    """
    Provide a fresh test container for each test.
    
    The container is reset to a clean state before each test and
    cleaned up after each test.
    """
    container = create_test_container()
    container.reset_test_state()
    yield container
    container.clear_all()


@pytest.fixture
def isolated_test_context() -> Generator[ComponentTestContext, None, None]:
    """
    Provide an isolated test context for component testing.
    
    Each test gets a completely isolated environment with fresh
    dependencies that don't interfere with other tests.
    """
    with ComponentTestContext() as context:
        yield context


@pytest.fixture
def event_bus(test_container: TestContainer) -> TestEventBusAdapter:
    """Provide a test event bus."""
    return test_container.get_event_bus()


@pytest.fixture
def kernel(test_container: TestContainer) -> TestKernelAdapter:
    """Provide a test kernel."""
    return test_container.get_kernel()


@pytest.fixture
def health_monitor(test_container: TestContainer) -> TestHealthMonitorAdapter:
    """Provide a test health monitor."""
    return test_container.get_health_monitor()


@pytest.fixture
def data_storage(test_container: TestContainer) -> TestDataStorageAdapter:
    """Provide a test data storage."""
    return test_container.get_data_storage()


@pytest.fixture
def time_service(test_container: TestContainer) -> TestTimeServiceAdapter:
    """Provide a test time service."""
    return test_container.get_time_service()


@pytest.fixture
def logging_service(test_container: TestContainer) -> TestLoggingAdapter:
    """Provide a test logging service."""
    return test_container.get_logging()


@pytest.fixture
def configuration(test_container: TestContainer) -> TestConfigurationAdapter:
    """Provide a test configuration."""
    return test_container.get_configuration()


@pytest.fixture
def network(test_container: TestContainer) -> TestNetworkAdapter:
    """Provide a test network."""
    return test_container.get_network()


@pytest.fixture
def file_system(test_container: TestContainer) -> TestFileSystemAdapter:
    """Provide a test file system."""
    return test_container.get_file_system()


@pytest.fixture
def metrics(test_container: TestContainer) -> TestMetricsAdapter:
    """Provide a test metrics service."""
    return test_container.get_metrics()


@pytest.fixture
def mock_current_time():
    """
    Mock the current time to a fixed value.
    
    Useful for tests that need predictable time behavior.
    """
    fixed_time = datetime(2023, 1, 1, 12, 0, 0)
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.utcnow.return_value = fixed_time
        yield fixed_time


def hexagonal_test_setup(
    component_class: Type[T],
    test_container: Optional[TestContainer] = None,
    **component_kwargs
) -> T:
    """
    Set up a component for hexagonal testing.
    
    This function instantiates a component with all its dependencies
    injected from the test container, enabling isolated testing.
    
    Args:
        component_class: The component class to instantiate
        test_container: Optional test container (creates new one if None)
        **component_kwargs: Additional keyword arguments for component
        
    Returns:
        The instantiated component with test dependencies
    """
    if test_container is None:
        test_container = create_test_container()
    
    # Try to resolve the component using dependency injection
    if test_container.is_registered(component_class):
        return test_container.resolve(component_class)
    
    # If not registered, try to create with constructor injection
    try:
        return test_container._create_with_injection(component_class)
    except Exception:
        # Fall back to manual construction with provided kwargs
        return component_class(**component_kwargs)


def isolated_component_test(
    component_class: Type[T],
    test_scenario: Optional[str] = None,
    **component_kwargs
) -> Generator[T, None, None]:
    """
    Create an isolated test environment for a component.
    
    This context manager provides a completely isolated test environment
    with fresh dependencies that don't interfere with other tests.
    
    Args:
        component_class: The component class to test
        test_scenario: Optional scenario name for pre-configuration
        **component_kwargs: Additional keyword arguments for component
        
    Yields:
        The component instance ready for testing
    """
    with ComponentTestContext() as context:
        if test_scenario:
            context.configure_test_scenario(test_scenario)
        
        component = hexagonal_test_setup(component_class, context, **component_kwargs)
        yield component


def integration_test_setup(
    components: Dict[str, Type],
    test_container: Optional[TestContainer] = None,
    shared_dependencies: bool = True
) -> Dict[str, Any]:
    """
    Set up multiple components for integration testing.
    
    This function creates multiple components that can share dependencies
    (like event bus) for integration testing scenarios.
    
    Args:
        components: Dictionary of component name to component class
        test_container: Optional test container
        shared_dependencies: Whether components share dependency instances
        
    Returns:
        Dictionary of component name to instantiated component
    """
    if test_container is None:
        test_container = create_test_container()
    
    instantiated_components = {}
    
    for name, component_class in components.items():
        try:
            component = test_container.resolve(component_class)
        except ValueError:
            # If not registered, try constructor injection
            try:
                component = test_container._create_with_injection(component_class)
            except Exception:
                # Fall back to basic instantiation
                component = component_class()
        
        instantiated_components[name] = component
        
        # Register component in container for cross-references
        test_container.register_instance(component_class, component)
    
    return instantiated_components


# Scenario-specific fixtures

@pytest.fixture
def offline_test_scenario(test_container: TestContainer) -> TestContainer:
    """Configure test container for offline testing."""
    test_container.configure_test_scenario("offline_mode")
    return test_container


@pytest.fixture
def unhealthy_system_scenario(test_container: TestContainer) -> TestContainer:
    """Configure test container for unhealthy system testing."""
    test_container.configure_test_scenario("unhealthy_system")
    return test_container


@pytest.fixture
def slow_network_scenario(test_container: TestContainer) -> TestContainer:
    """Configure test container for slow network testing."""
    test_container.configure_test_scenario("slow_network")
    return test_container


@pytest.fixture
def limited_storage_scenario(test_container: TestContainer) -> TestContainer:
    """Configure test container for limited storage testing."""
    test_container.configure_test_scenario("limited_storage")
    return test_container


# Performance testing fixtures

@pytest.fixture
def performance_test_setup(test_container: TestContainer) -> Dict[str, Any]:
    """Set up performance testing environment."""
    # Configure for performance testing
    time_service = test_container.get_time_service()
    metrics = test_container.get_metrics()
    
    # Set up performance tracking
    performance_context = {
        "start_time": time_service.now(),
        "metrics": metrics,
        "measurements": []
    }
    
    return performance_context


@pytest.fixture
def stress_test_setup(test_container: TestContainer) -> Dict[str, Any]:
    """Set up stress testing environment."""
    # Configure for stress testing
    test_container.configure_test_scenario("slow_network")
    test_container.configure_test_scenario("limited_storage")
    
    return {
        "container": test_container,
        "max_iterations": 1000,
        "timeout_seconds": 30
    }


# Utility fixtures

@pytest.fixture
def sample_test_data() -> Dict[str, Any]:
    """Provide sample test data for components."""
    return {
        "market_data": {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000,
            "timestamp": "2023-01-01T12:00:00"
        },
        "trade_signal": {
            "signal": "BUY",
            "confidence": 0.85,
            "position_size": 0.1
        },
        "risk_metrics": {
            "var_95": 0.02,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.2
        }
    }


@pytest.fixture
def component_configuration() -> Dict[str, Any]:
    """Provide default component configuration."""
    return {
        "strategic_marl": {
            "enabled": True,
            "n_agents": 3,
            "learning_rate": 0.001,
            "window_size": 48
        },
        "tactical_marl": {
            "enabled": True,
            "n_agents": 2,
            "learning_rate": 0.0005,
            "window_size": 60
        },
        "risk_management": {
            "enabled": True,
            "max_position_size": 0.1,
            "stop_loss_threshold": 0.02
        }
    }


# Assertion helpers

def assert_component_health(health_monitor: TestHealthMonitorAdapter, expected_status: HealthStatus):
    """Assert component health status."""
    import asyncio
    actual_status = asyncio.run(health_monitor.get_overall_health())
    assert actual_status == expected_status, f"Expected {expected_status}, got {actual_status}"


def assert_events_published(event_bus: TestEventBusAdapter, expected_events: Dict[str, int]):
    """Assert specific events were published."""
    from .ports import EventType
    
    for event_type_str, expected_count in expected_events.items():
        event_type = EventType(event_type_str)
        actual_count = len(event_bus.get_events_by_type(event_type))
        assert actual_count == expected_count, f"Expected {expected_count} {event_type_str} events, got {actual_count}"


def assert_metrics_recorded(metrics: TestMetricsAdapter, expected_metrics: Dict[str, Any]):
    """Assert specific metrics were recorded."""
    for metric_name, expected_value in expected_metrics.items():
        if isinstance(expected_value, int):
            actual_value = metrics.get_counter_value(metric_name)
        elif isinstance(expected_value, float):
            actual_value = metrics.get_gauge_value(metric_name)
        else:
            actual_value = metrics.get_histogram_values(metric_name)
        
        assert actual_value == expected_value, f"Expected {metric_name}={expected_value}, got {actual_value}"


def assert_configuration_accessed(config: TestConfigurationAdapter, expected_keys: List[str]):
    """Assert specific configuration keys were accessed."""
    accessed_keys = config.get_get_calls()
    for key in expected_keys:
        assert key in accessed_keys, f"Expected configuration key '{key}' to be accessed"


def assert_storage_operations(storage: TestDataStorageAdapter, expected_operations: List[str]):
    """Assert specific storage operations occurred."""
    operations = storage.get_operations()
    operation_types = [op["operation"] for op in operations]
    
    for expected_op in expected_operations:
        assert expected_op in operation_types, f"Expected storage operation '{expected_op}' not found"