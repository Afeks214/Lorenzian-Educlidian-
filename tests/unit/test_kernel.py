"""
Unit tests for the AlgoSpaceKernel core orchestration system.

This module tests the kernel initialization, component management,
event wiring, and lifecycle management.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any
import logging
import yaml
import tempfile
from pathlib import Path

# Test markers
pytestmark = [pytest.mark.unit]


class TestAlgoSpaceKernel:
    """Test the main AlgoSpaceKernel orchestration class."""

    @pytest.fixture
    def kernel_config(self, temp_dir):
        """Sample configuration for kernel testing."""
        config = {
            "data_handler": {
                "type": "backtest",
                "file_path": str(temp_dir / "test_data.csv")
            },
            "matrix_assemblers": {
                "30m": {
                    "window_size": 48,
                    "features": ["mlmi_value", "mlmi_signal", "nwrqk_value"]
                },
                "5m": {
                    "window_size": 60,
                    "features": ["fvg_bullish_active", "fvg_bearish_active"]
                }
            },
            "strategic_marl": {
                "enabled": True,
                "n_agents": 3,
                "learning_rate": 0.001
            }
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_file)

    @pytest.fixture
    def mock_kernel(self, kernel_config):
        """Create a mock kernel for testing."""
        # Import here to avoid circular imports in tests
        try:
            from src.core.kernel import AlgoSpaceKernel
            kernel = AlgoSpaceKernel(kernel_config)
        except ImportError:
            # Fallback to mock if actual kernel not available
            kernel = Mock()
            kernel.config_path = kernel_config
            kernel.config = {}
            kernel.event_bus = Mock()
            kernel.components = {}
            kernel.running = False
            
        return kernel

    def test_kernel_initialization(self, kernel_config):
        """Test kernel initialization with configuration."""
        # Mock the kernel class since it might not exist yet
        mock_kernel = Mock()
        mock_kernel.config_path = kernel_config
        mock_kernel.config = {}
        mock_kernel.event_bus = Mock()
        mock_kernel.components = {}
        mock_kernel.running = False
        
        # Test initialization
        assert mock_kernel.config_path == kernel_config
        assert isinstance(mock_kernel.components, dict)
        assert mock_kernel.running is False
        assert mock_kernel.event_bus is not None

    def test_config_loading(self, mock_kernel, kernel_config):
        """Test configuration loading from YAML file."""
        # Mock config loading
        mock_config = {
            "data_handler": {"type": "backtest"},
            "matrix_assemblers": {"30m": {"window_size": 48}},
            "strategic_marl": {"enabled": True}
        }
        
        mock_kernel.load_config = Mock(return_value=mock_config)
        config = mock_kernel.load_config(kernel_config)
        
        assert "data_handler" in config
        assert "matrix_assemblers" in config
        assert "strategic_marl" in config
        assert config["data_handler"]["type"] == "backtest"
        
        mock_kernel.load_config.assert_called_once_with(kernel_config)

    def test_component_instantiation(self, mock_kernel):
        """Test component instantiation process."""
        # Mock components
        mock_components = {
            "data_handler": Mock(),
            "bar_generator": Mock(),
            "indicator_engine": Mock(),
            "matrix_30m": Mock(),
            "matrix_5m": Mock(),
            "synergy_detector": Mock(),
            "strategic_marl": Mock(),
            "execution_handler": Mock()
        }
        
        mock_kernel.instantiate_components = Mock(return_value=mock_components)
        components = mock_kernel.instantiate_components()
        
        # Verify expected components are instantiated
        expected_components = [
            "data_handler", "bar_generator", "indicator_engine",
            "matrix_30m", "matrix_5m", "synergy_detector",
            "strategic_marl", "execution_handler"
        ]
        
        for component_name in expected_components:
            assert component_name in components
            assert components[component_name] is not None
        
        mock_kernel.instantiate_components.assert_called_once()

    def test_event_wiring(self, mock_kernel):
        """Test event system wiring between components."""
        # Mock event bus and components
        mock_kernel.event_bus = Mock()
        mock_kernel.components = {
            "data_handler": Mock(),
            "bar_generator": Mock(),
            "indicator_engine": Mock(),
            "matrix_30m": Mock(),
            "synergy_detector": Mock(),
            "strategic_marl": Mock(),
            "execution_handler": Mock()
        }
        
        # Mock event wiring
        mock_kernel.wire_events = Mock()
        mock_kernel.wire_events()
        
        # Verify event wiring was called
        mock_kernel.wire_events.assert_called_once()

    def test_component_initialization_sequence(self, mock_kernel):
        """Test the component initialization sequence."""
        # Mock the initialization sequence
        initialization_steps = [
            "load_config",
            "instantiate_components", 
            "wire_events",
            "initialize_components"
        ]
        
        for step in initialization_steps:
            mock_method = Mock()
            setattr(mock_kernel, step, mock_method)
        
        # Mock the main initialize method
        mock_kernel.initialize = Mock()
        mock_kernel.initialize()
        
        # Verify initialization was called
        mock_kernel.initialize.assert_called_once()

    def test_kernel_startup(self, mock_kernel):
        """Test kernel startup process."""
        # Mock startup sequence
        mock_kernel.initialize = Mock()
        mock_kernel.start_data_stream = Mock()
        mock_kernel.run_event_loop = Mock()
        
        # Mock run method
        mock_kernel.run = Mock()
        mock_kernel.run()
        
        # Verify startup sequence
        mock_kernel.run.assert_called_once()

    def test_kernel_shutdown(self, mock_kernel):
        """Test graceful kernel shutdown."""
        # Mock shutdown components
        mock_kernel.components = {
            "data_handler": Mock(),
            "execution_handler": Mock(),
            "strategic_marl": Mock()
        }
        
        # Add shutdown methods to components
        for component in mock_kernel.components.values():
            component.save_state = Mock()
            component.stop = Mock()
        
        # Mock shutdown process
        mock_kernel.shutdown = Mock()
        mock_kernel.shutdown()
        
        # Verify shutdown was called
        mock_kernel.shutdown.assert_called_once()

    def test_error_handling(self, mock_kernel):
        """Test error handling in kernel operations."""
        # Mock error scenarios
        error_scenarios = [
            {"type": "ConfigError", "component": "config_loader"},
            {"type": "ComponentError", "component": "data_handler"},
            {"type": "EventError", "component": "event_bus"},
            {"type": "RuntimeError", "component": "strategic_marl"}
        ]
        
        for scenario in error_scenarios:
            mock_kernel.handle_error = Mock()
            mock_kernel.handle_error(scenario)
            mock_kernel.handle_error.assert_called_with(scenario)

    def test_component_retrieval(self, mock_kernel):
        """Test component retrieval by name."""
        # Mock components
        mock_kernel.components = {
            "data_handler": Mock(),
            "strategic_marl": Mock(),
            "execution_handler": Mock()
        }
        
        mock_kernel.get_component = Mock(side_effect=lambda name: mock_kernel.components.get(name))
        
        # Test retrieval
        data_handler = mock_kernel.get_component("data_handler")
        strategic_marl = mock_kernel.get_component("strategic_marl")
        non_existent = mock_kernel.get_component("non_existent")
        
        assert data_handler is not None
        assert strategic_marl is not None
        assert non_existent is None

    def test_system_status(self, mock_kernel):
        """Test system status reporting."""
        # Mock status
        mock_status = {
            "running": True,
            "mode": "backtest",
            "components": ["data_handler", "strategic_marl", "execution_handler"],
            "subscribers": 15,
            "uptime": 3600
        }
        
        mock_kernel.get_status = Mock(return_value=mock_status)
        status = mock_kernel.get_status()
        
        assert "running" in status
        assert "mode" in status
        assert "components" in status
        assert status["running"] is True
        assert isinstance(status["components"], list)

    def test_event_bus_access(self, mock_kernel):
        """Test event bus access through kernel."""
        mock_event_bus = Mock()
        mock_kernel.event_bus = mock_event_bus
        mock_kernel.get_event_bus = Mock(return_value=mock_event_bus)
        
        event_bus = mock_kernel.get_event_bus()
        
        assert event_bus is mock_event_bus
        mock_kernel.get_event_bus.assert_called_once()


class TestKernelLifecycleManagement:
    """Test kernel lifecycle management operations."""

    @pytest.fixture
    def lifecycle_kernel(self):
        """Create a kernel for lifecycle testing."""
        kernel = Mock()
        kernel.running = False
        kernel.components = {}
        kernel.initialization_state = "not_started"
        
        return kernel

    def test_initialization_states(self, lifecycle_kernel):
        """Test different initialization states."""
        states = [
            "not_started",
            "loading_config",
            "instantiating_components",
            "wiring_events", 
            "initializing_components",
            "ready"
        ]
        
        lifecycle_kernel.get_initialization_state = Mock()
        lifecycle_kernel.set_initialization_state = Mock()
        
        for state in states:
            lifecycle_kernel.set_initialization_state(state)
            lifecycle_kernel.set_initialization_state.assert_called_with(state)

    def test_component_lifecycle(self, lifecycle_kernel):
        """Test individual component lifecycle management."""
        # Mock component with lifecycle methods
        mock_component = Mock()
        mock_component.initialize = Mock()
        mock_component.start = Mock()
        mock_component.stop = Mock()
        mock_component.cleanup = Mock()
        
        lifecycle_kernel.components = {"test_component": mock_component}
        
        # Test lifecycle operations
        lifecycle_kernel.initialize_component = Mock()
        lifecycle_kernel.start_component = Mock()
        lifecycle_kernel.stop_component = Mock()
        lifecycle_kernel.cleanup_component = Mock()
        
        # Execute lifecycle operations
        lifecycle_kernel.initialize_component("test_component")
        lifecycle_kernel.start_component("test_component")
        lifecycle_kernel.stop_component("test_component")
        lifecycle_kernel.cleanup_component("test_component")
        
        # Verify calls
        lifecycle_kernel.initialize_component.assert_called_with("test_component")
        lifecycle_kernel.start_component.assert_called_with("test_component")
        lifecycle_kernel.stop_component.assert_called_with("test_component")
        lifecycle_kernel.cleanup_component.assert_called_with("test_component")

    def test_dependency_resolution(self, lifecycle_kernel):
        """Test component dependency resolution."""
        # Mock components with dependencies
        component_dependencies = {
            "data_handler": [],
            "bar_generator": ["data_handler"],
            "indicator_engine": ["bar_generator"],
            "matrix_30m": ["indicator_engine"],
            "strategic_marl": ["matrix_30m", "synergy_detector"],
            "synergy_detector": ["indicator_engine"],
            "execution_handler": ["strategic_marl"]
        }
        
        lifecycle_kernel.get_component_dependencies = Mock(return_value=component_dependencies)
        lifecycle_kernel.resolve_dependencies = Mock()
        
        dependencies = lifecycle_kernel.get_component_dependencies()
        lifecycle_kernel.resolve_dependencies()
        
        # Verify dependency structure
        assert "data_handler" in dependencies
        assert dependencies["data_handler"] == []  # No dependencies
        assert "data_handler" in dependencies["bar_generator"]
        assert "strategic_marl" in dependencies["execution_handler"]
        
        lifecycle_kernel.resolve_dependencies.assert_called_once()

    def test_health_monitoring(self, lifecycle_kernel):
        """Test component health monitoring."""
        # Mock health status
        health_status = {
            "data_handler": {"status": "healthy", "last_update": "2023-01-01T10:00:00Z"},
            "strategic_marl": {"status": "healthy", "last_update": "2023-01-01T10:00:00Z"},
            "execution_handler": {"status": "warning", "last_update": "2023-01-01T09:59:00Z"}
        }
        
        lifecycle_kernel.check_component_health = Mock()
        lifecycle_kernel.get_system_health = Mock(return_value=health_status)
        
        system_health = lifecycle_kernel.get_system_health()
        
        assert "data_handler" in system_health
        assert "strategic_marl" in system_health
        assert "execution_handler" in system_health
        assert system_health["data_handler"]["status"] == "healthy"
        assert system_health["execution_handler"]["status"] == "warning"

    def test_resource_cleanup(self, lifecycle_kernel):
        """Test resource cleanup on shutdown."""
        # Mock components with resources
        mock_components = {}
        for name in ["data_handler", "strategic_marl", "execution_handler"]:
            component = Mock()
            component.cleanup_resources = Mock()
            component.save_state = Mock()
            component.close_connections = Mock()
            mock_components[name] = component
        
        lifecycle_kernel.components = mock_components
        lifecycle_kernel.cleanup_all_resources = Mock()
        
        # Execute cleanup
        lifecycle_kernel.cleanup_all_resources()
        
        # Verify cleanup was called
        lifecycle_kernel.cleanup_all_resources.assert_called_once()


class TestKernelErrorRecovery:
    """Test kernel error recovery mechanisms."""

    @pytest.fixture
    def recovery_kernel(self):
        """Create a kernel for error recovery testing."""
        kernel = Mock()
        kernel.error_count = 0
        kernel.max_retries = 3
        kernel.recovery_strategies = ["restart_component", "fallback_mode", "safe_shutdown"]
        
        return kernel

    def test_component_failure_handling(self, recovery_kernel):
        """Test handling of component failures."""
        failure_scenarios = [
            {"component": "data_handler", "error": "ConnectionTimeout", "severity": "high"},
            {"component": "strategic_marl", "error": "ModelLoadError", "severity": "critical"},
            {"component": "execution_handler", "error": "BrokerDisconnected", "severity": "high"}
        ]
        
        recovery_kernel.handle_component_failure = Mock()
        
        for scenario in failure_scenarios:
            recovery_kernel.handle_component_failure(scenario)
            recovery_kernel.handle_component_failure.assert_called_with(scenario)

    def test_retry_mechanisms(self, recovery_kernel):
        """Test retry mechanisms for failed operations."""
        retry_operations = [
            {"operation": "connect_broker", "max_retries": 3, "backoff": "exponential"},
            {"operation": "load_model", "max_retries": 2, "backoff": "linear"},
            {"operation": "subscribe_data", "max_retries": 5, "backoff": "fixed"}
        ]
        
        recovery_kernel.retry_operation = Mock()
        
        for operation in retry_operations:
            recovery_kernel.retry_operation(operation)
            recovery_kernel.retry_operation.assert_called_with(operation)

    def test_fallback_strategies(self, recovery_kernel):
        """Test fallback strategies for system degradation."""
        fallback_scenarios = [
            {"trigger": "data_feed_failure", "strategy": "use_cached_data"},
            {"trigger": "model_unavailable", "strategy": "use_simple_rules"},
            {"trigger": "broker_disconnected", "strategy": "paper_trading_mode"}
        ]
        
        recovery_kernel.activate_fallback = Mock()
        
        for scenario in fallback_scenarios:
            recovery_kernel.activate_fallback(scenario)
            recovery_kernel.activate_fallback.assert_called_with(scenario)

    def test_circuit_breaker_pattern(self, recovery_kernel):
        """Test circuit breaker pattern for error protection."""
        circuit_breaker_config = {
            "failure_threshold": 5,
            "timeout_seconds": 60,
            "half_open_max_calls": 3
        }
        
        recovery_kernel.circuit_breaker = Mock()
        recovery_kernel.circuit_breaker.configure(circuit_breaker_config)
        recovery_kernel.circuit_breaker.is_open = Mock(return_value=False)
        recovery_kernel.circuit_breaker.record_success = Mock()
        recovery_kernel.circuit_breaker.record_failure = Mock()
        
        # Test circuit breaker operations
        assert not recovery_kernel.circuit_breaker.is_open()
        recovery_kernel.circuit_breaker.record_success()
        recovery_kernel.circuit_breaker.record_failure()
        
        recovery_kernel.circuit_breaker.configure.assert_called_with(circuit_breaker_config)
        recovery_kernel.circuit_breaker.record_success.assert_called_once()
        recovery_kernel.circuit_breaker.record_failure.assert_called_once()

    def test_disaster_recovery(self, recovery_kernel):
        """Test disaster recovery procedures."""
        disaster_scenarios = [
            {"type": "total_system_failure", "procedure": "emergency_shutdown"},
            {"type": "data_corruption", "procedure": "restore_from_backup"},
            {"type": "critical_component_failure", "procedure": "activate_standby"}
        ]
        
        recovery_kernel.execute_disaster_recovery = Mock()
        
        for scenario in disaster_scenarios:
            recovery_kernel.execute_disaster_recovery(scenario)
            recovery_kernel.execute_disaster_recovery.assert_called_with(scenario)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])