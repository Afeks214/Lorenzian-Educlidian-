"""
pytest configuration for system control tests.

This module provides fixtures and configuration for the comprehensive
testing framework of the master switch system.
"""

import pytest
import tempfile
import os
import logging
import time
from unittest.mock import Mock, patch

from src.core.trading_system_controller import TradingSystemController


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test markers for different test categories
pytest_markers = [
    "unit: Unit tests for core functionality",
    "integration: Integration tests with other components",
    "safety: Safety mechanism and failsafe tests",
    "performance: Performance and load tests",
    "slow: Tests that take longer to run",
    "stress: Stress tests that push system limits"
]


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="trading_system_test_")
    yield temp_dir
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def temp_state_file(temp_dir):
    """Create a temporary state file for testing."""
    state_file = os.path.join(temp_dir, "test_state.json")
    yield state_file
    
    # Cleanup
    try:
        os.unlink(state_file)
    except FileNotFoundError:
        pass


@pytest.fixture
def controller_factory(temp_dir):
    """Factory for creating controller instances with test configuration."""
    controllers = []
    
    def create_controller(**kwargs):
        # Default test configuration
        default_config = {
            "max_concurrent_operations": 10,
            "heartbeat_timeout": 2.0,
            "state_persistence_path": os.path.join(temp_dir, f"controller_state_{len(controllers)}.json")
        }
        
        # Merge with provided kwargs
        config = {**default_config, **kwargs}
        
        controller = TradingSystemController(**config)
        controllers.append(controller)
        return controller
    
    yield create_controller
    
    # Cleanup all controllers
    for controller in controllers:
        try:
            controller.shutdown(timeout=5.0)
        except Exception:
            pass


@pytest.fixture
def basic_controller(controller_factory):
    """Basic controller instance for simple tests."""
    return controller_factory()


@pytest.fixture
def performance_controller(controller_factory):
    """Controller configured for performance testing."""
    return controller_factory(
        max_concurrent_operations=100,
        heartbeat_timeout=5.0
    )


@pytest.fixture
def mock_components():
    """Mock components for testing."""
    components = {
        "data_handler": Mock(),
        "risk_manager": Mock(),
        "execution_engine": Mock(),
        "tactical_controller": Mock(),
        "kill_switch": Mock()
    }
    
    # Configure mocks
    for component_name, component_mock in components.items():
        component_mock.name = component_name
        component_mock.start.return_value = True
        component_mock.stop.return_value = True
        component_mock.is_healthy.return_value = True
        component_mock.get_status.return_value = {"status": "healthy"}
    
    return components


@pytest.fixture
def mock_safety_checks():
    """Mock safety checks for testing."""
    checks = {
        "market_connectivity": Mock(return_value=True),
        "risk_limits": Mock(return_value=True),
        "data_quality": Mock(return_value=True),
        "broker_connection": Mock(return_value=True)
    }
    
    return checks


@pytest.fixture
def event_logger():
    """Event logger for tracking system events during tests."""
    events = []
    
    def log_event(event_type, data):
        events.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })
    
    log_event.events = events
    log_event.clear = lambda: events.clear()
    
    return log_event


@pytest.fixture
def performance_monitor():
    """Performance monitor for tracking test performance."""
    metrics = {
        "start_times": [],
        "stop_times": [],
        "emergency_stop_times": [],
        "component_update_times": [],
        "health_check_times": []
    }
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = metrics
        
        def time_operation(self, operation_name, operation_func):
            start_time = time.perf_counter()
            result = operation_func()
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(duration)
            
            return result
        
        def get_metrics(self):
            return self.metrics.copy()
        
        def clear_metrics(self):
            for key in self.metrics:
                self.metrics[key].clear()
    
    return PerformanceMonitor()


# Test configuration hooks
def pytest_configure(config):
    """Configure pytest for system control tests."""
    # Register custom markers
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    # Add markers based on test paths and names
    for item in items:
        # Add markers based on test file names
        if "test_master_switch" in item.fspath.basename:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "test_safety" in item.fspath.basename:
            item.add_marker(pytest.mark.safety)
        elif "test_performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that take long
        if any(keyword in item.name for keyword in ["stress", "scalability", "memory_leak"]):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.stress)
        
        # Add performance marker for performance tests
        if any(keyword in item.name for keyword in ["performance", "latency", "throughput"]):
            item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Setup hook for individual test runs."""
    # Skip stress tests unless explicitly requested
    if item.get_closest_marker("stress") and not item.config.getoption("--stress"):
        pytest.skip("Stress tests skipped (use --stress to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--stress",
        action="store_true",
        default=False,
        help="Run stress tests"
    )
    
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests"
    )


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment."""
    # Environment setup
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup
    os.environ.clear()
    os.environ.update(original_env)


# Custom assertions for system control tests
class SystemControlAssertions:
    """Custom assertions for system control testing."""
    
    @staticmethod
    def assert_valid_state_transition(controller, expected_state, timeout=5.0):
        """Assert that controller reaches expected state within timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if controller.get_state() == expected_state:
                return True
            time.sleep(0.1)
        
        current_state = controller.get_state()
        raise AssertionError(f"Expected state {expected_state}, got {current_state}")
    
    @staticmethod
    def assert_performance_threshold(times, threshold, operation_name):
        """Assert that operation times are within performance threshold."""
        if not times:
            raise AssertionError(f"No timing data for {operation_name}")
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        if avg_time > threshold:
            raise AssertionError(f"{operation_name} average time {avg_time:.4f}s exceeds threshold {threshold:.4f}s")
        
        if max_time > threshold * 2:
            raise AssertionError(f"{operation_name} maximum time {max_time:.4f}s exceeds threshold {threshold * 2:.4f}s")
    
    @staticmethod
    def assert_component_health(controller, component_name, expected_status):
        """Assert that component has expected health status."""
        component = controller.get_component_status(component_name)
        if component is None:
            raise AssertionError(f"Component {component_name} not found")
        
        if component.status != expected_status:
            raise AssertionError(f"Component {component_name} has status {component.status}, expected {expected_status}")
    
    @staticmethod
    def assert_no_memory_leak(initial_memory, final_memory, threshold_mb=10):
        """Assert that memory usage didn't increase significantly."""
        memory_increase = final_memory - initial_memory
        if memory_increase > threshold_mb:
            raise AssertionError(f"Memory leak detected: {memory_increase:.2f}MB increase")


@pytest.fixture
def assert_system():
    """Fixture providing custom assertions."""
    return SystemControlAssertions()


# Error handling for test failures
@pytest.fixture(autouse=True)
def handle_test_failures(request):
    """Handle test failures and cleanup."""
    yield
    
    # Cleanup on test failure
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        # Log additional debug information on failure
        print(f"\n--- Test {request.node.name} failed ---")
        print(f"Test file: {request.node.fspath}")
        print(f"Test function: {request.node.function.__name__}")
        
        # Try to get controller state if available
        if hasattr(request.node.function, '__self__'):
            controller = getattr(request.node.function.__self__, 'controller', None)
            if controller:
                try:
                    state = controller.get_state()
                    components = controller.get_all_components()
                    print(f"Controller state: {state}")
                    print(f"Component count: {len(components)}")
                except Exception as e:
                    print(f"Could not get controller state: {e}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test report available for failure handling."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)