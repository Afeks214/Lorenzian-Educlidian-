"""
Dependency injection container for hexagonal architecture testing.

This module provides a flexible dependency injection framework that allows
complete control over component dependencies during testing.
"""

from typing import Dict, Any, Type, TypeVar, Optional, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import inspect

from .ports import (
    EventBusPort, KernelPort, HealthMonitorPort, DataStoragePort,
    TimeServicePort, LoggingPort, ConfigurationPort, NetworkPort,
    FileSystemPort, MetricsPort
)

from .adapters import (
    TestEventBusAdapter, TestKernelAdapter, TestHealthMonitorAdapter,
    TestDataStorageAdapter, TestTimeServiceAdapter, TestLoggingAdapter,
    TestConfigurationAdapter, TestNetworkAdapter, TestFileSystemAdapter,
    TestMetricsAdapter
)

T = TypeVar('T')


class DependencyScope(Enum):
    """Dependency lifecycle scopes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class DependencyBinding:
    """Dependency binding configuration."""
    interface: Type
    implementation: Type
    scope: DependencyScope = DependencyScope.TRANSIENT
    factory: Optional[Callable] = None
    instance: Optional[Any] = None


class DependencyContainer:
    """
    Dependency injection container for hexagonal architecture.
    
    Provides registration, resolution, and lifecycle management for dependencies.
    """
    
    def __init__(self):
        self._bindings: Dict[Type, DependencyBinding] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._resolution_stack: List[Type] = []
        
    def register(self, 
                 interface: Type[T], 
                 implementation: Type[T],
                 scope: DependencyScope = DependencyScope.TRANSIENT) -> 'DependencyContainer':
        """Register a dependency binding."""
        self._bindings[interface] = DependencyBinding(
            interface=interface,
            implementation=implementation,
            scope=scope
        )
        return self
        
    def register_factory(self,
                        interface: Type[T],
                        factory: Callable[[], T],
                        scope: DependencyScope = DependencyScope.TRANSIENT) -> 'DependencyContainer':
        """Register a factory for dependency creation."""
        self._bindings[interface] = DependencyBinding(
            interface=interface,
            implementation=None,
            scope=scope,
            factory=factory
        )
        return self
        
    def register_instance(self, interface: Type[T], instance: T) -> 'DependencyContainer':
        """Register a specific instance."""
        self._bindings[interface] = DependencyBinding(
            interface=interface,
            implementation=type(instance),
            scope=DependencyScope.SINGLETON,
            instance=instance
        )
        self._singletons[interface] = instance
        return self
        
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a dependency."""
        # Check for circular dependencies
        if interface in self._resolution_stack:
            raise ValueError(f"Circular dependency detected: {' -> '.join(str(t) for t in self._resolution_stack)} -> {interface}")
        
        # Check if interface is registered
        if interface not in self._bindings:
            raise ValueError(f"No binding found for {interface}")
        
        binding = self._bindings[interface]
        
        # Handle singleton scope
        if binding.scope == DependencyScope.SINGLETON:
            if interface in self._singletons:
                return self._singletons[interface]
            instance = self._create_instance(binding)
            self._singletons[interface] = instance
            return instance
        
        # Handle scoped instances
        if binding.scope == DependencyScope.SCOPED:
            if interface in self._scoped_instances:
                return self._scoped_instances[interface]
            instance = self._create_instance(binding)
            self._scoped_instances[interface] = instance
            return instance
        
        # Handle transient scope
        return self._create_instance(binding)
    
    def _create_instance(self, binding: DependencyBinding) -> Any:
        """Create an instance based on binding configuration."""
        # Use existing instance if available
        if binding.instance is not None:
            return binding.instance
        
        # Use factory if available
        if binding.factory is not None:
            return binding.factory()
        
        # Create instance using constructor injection
        self._resolution_stack.append(binding.interface)
        try:
            instance = self._create_with_injection(binding.implementation)
            return instance
        finally:
            self._resolution_stack.pop()
    
    def _create_with_injection(self, implementation: Type) -> Any:
        """Create instance with constructor dependency injection."""
        # Get constructor signature
        signature = inspect.signature(implementation.__init__)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to resolve by type annotation
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # If not found and has default, use default
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency {param.annotation} for {implementation}")
            elif param.default != inspect.Parameter.empty:
                kwargs[param_name] = param.default
        
        return implementation(**kwargs)
    
    def clear_scoped(self) -> None:
        """Clear scoped instances."""
        self._scoped_instances.clear()
    
    def clear_all(self) -> None:
        """Clear all cached instances."""
        self._singletons.clear()
        self._scoped_instances.clear()
    
    def is_registered(self, interface: Type) -> bool:
        """Check if interface is registered."""
        return interface in self._bindings
    
    def get_registrations(self) -> Dict[Type, DependencyBinding]:
        """Get all registrations."""
        return self._bindings.copy()


class TestContainer(DependencyContainer):
    """
    Test-specific dependency container with pre-configured test adapters.
    
    This container comes pre-configured with test adapters for all standard
    ports, making it easy to set up isolated tests.
    """
    
    def __init__(self):
        super().__init__()
        self._configure_default_test_adapters()
        
    def _configure_default_test_adapters(self):
        """Configure default test adapters for all ports."""
        # Register all test adapters as singletons for consistent test state
        self.register(EventBusPort, TestEventBusAdapter, DependencyScope.SINGLETON)
        self.register(KernelPort, TestKernelAdapter, DependencyScope.SINGLETON)
        self.register(HealthMonitorPort, TestHealthMonitorAdapter, DependencyScope.SINGLETON)
        self.register(DataStoragePort, TestDataStorageAdapter, DependencyScope.SINGLETON)
        self.register(TimeServicePort, TestTimeServiceAdapter, DependencyScope.SINGLETON)
        self.register(LoggingPort, TestLoggingAdapter, DependencyScope.SINGLETON)
        self.register(ConfigurationPort, TestConfigurationAdapter, DependencyScope.SINGLETON)
        self.register(NetworkPort, TestNetworkAdapter, DependencyScope.SINGLETON)
        self.register(FileSystemPort, TestFileSystemAdapter, DependencyScope.SINGLETON)
        self.register(MetricsPort, TestMetricsAdapter, DependencyScope.SINGLETON)
    
    def get_event_bus(self) -> TestEventBusAdapter:
        """Get the test event bus."""
        return self.resolve(EventBusPort)
    
    def get_kernel(self) -> TestKernelAdapter:
        """Get the test kernel."""
        return self.resolve(KernelPort)
    
    def get_health_monitor(self) -> TestHealthMonitorAdapter:
        """Get the test health monitor."""
        return self.resolve(HealthMonitorPort)
    
    def get_data_storage(self) -> TestDataStorageAdapter:
        """Get the test data storage."""
        return self.resolve(DataStoragePort)
    
    def get_time_service(self) -> TestTimeServiceAdapter:
        """Get the test time service."""
        return self.resolve(TimeServicePort)
    
    def get_logging(self) -> TestLoggingAdapter:
        """Get the test logging."""
        return self.resolve(LoggingPort)
    
    def get_configuration(self) -> TestConfigurationAdapter:
        """Get the test configuration."""
        return self.resolve(ConfigurationPort)
    
    def get_network(self) -> TestNetworkAdapter:
        """Get the test network."""
        return self.resolve(NetworkPort)
    
    def get_file_system(self) -> TestFileSystemAdapter:
        """Get the test file system."""
        return self.resolve(FileSystemPort)
    
    def get_metrics(self) -> TestMetricsAdapter:
        """Get the test metrics."""
        return self.resolve(MetricsPort)
    
    def configure_test_scenario(self, scenario_name: str) -> None:
        """Configure common test scenarios."""
        if scenario_name == "offline_mode":
            # Configure for offline testing
            network = self.get_network()
            network.set_connected(False)
            
        elif scenario_name == "unhealthy_system":
            # Configure for unhealthy system testing
            from .ports import HealthStatus
            health_monitor = self.get_health_monitor()
            health_monitor.set_system_health(HealthStatus.UNHEALTHY)
            
        elif scenario_name == "slow_network":
            # Configure for slow network testing
            network = self.get_network()
            network.set_http_response("*", {"status": 200, "latency": 5000})
            
        elif scenario_name == "limited_storage":
            # Configure for limited storage testing
            storage = self.get_data_storage()
            # Mock storage full condition
            original_store = storage.store
            
            async def limited_store(key: str, value: Any):
                if len(storage.get_storage()) >= 10:
                    raise Exception("Storage full")
                await original_store(key, value)
            
            storage.store = limited_store
    
    def reset_test_state(self) -> None:
        """Reset all test adapters to clean state."""
        # Clear all cached instances and recreate adapters
        self.clear_all()
        
        # Reset specific adapter states
        event_bus = self.get_event_bus()
        event_bus.clear_history()
        event_bus.enable()
        
        kernel = self.get_kernel()
        kernel.start()
        
        health_monitor = self.get_health_monitor()
        from .ports import HealthStatus
        health_monitor.set_system_health(HealthStatus.HEALTHY)
        
        data_storage = self.get_data_storage()
        data_storage.clear()
        
        time_service = self.get_time_service()
        from datetime import datetime
        time_service.set_time(datetime.now())
        
        logging = self.get_logging()
        logging.clear_logs()
        
        configuration = self.get_configuration()
        configuration.set("test_mode", True)
        
        network = self.get_network()
        network.set_connected(True)
        
        file_system = self.get_file_system()
        file_system.get_files().clear()
        
        metrics = self.get_metrics()
        metrics.clear_metrics()


class ComponentTestContext:
    """
    Context manager for isolated component testing.
    
    Provides a clean dependency injection environment for each test.
    """
    
    def __init__(self, container: Optional[TestContainer] = None):
        self.container = container or TestContainer()
        self._original_state = None
        
    def __enter__(self) -> TestContainer:
        """Enter test context."""
        self._original_state = self.container.get_registrations()
        self.container.reset_test_state()
        return self.container
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit test context."""
        # Reset to clean state
        self.container.reset_test_state()
        return False  # Don't suppress exceptions


def create_test_container() -> TestContainer:
    """Create a new test container with default configuration."""
    return TestContainer()


def create_isolated_test_context() -> ComponentTestContext:
    """Create an isolated test context."""
    return ComponentTestContext()