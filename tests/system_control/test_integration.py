"""
Integration tests for the Trading System Controller.

This module tests the integration between the master switch system and other
system components, including the kill switch, risk management, and tactical controller.
"""

import pytest
import asyncio
import threading
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone
import tempfile
import os

from src.core.trading_system_controller import (
    TradingSystemController,
    SystemState,
    ComponentStatus
)
from src.safety.kill_switch import TradingSystemKillSwitch, initialize_kill_switch
from src.tactical.controller import TacticalMARLController


class TestSystemIntegration:
    """Test integration between master switch and other system components."""
    
    @pytest.fixture
    def temp_state_file(self):
        """Create temporary state file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def controller(self, temp_state_file):
        """Create controller instance for integration testing."""
        controller = TradingSystemController(
            max_concurrent_operations=10,
            heartbeat_timeout=2.0,
            state_persistence_path=temp_state_file
        )
        yield controller
        controller.shutdown(timeout=5.0)
    
    @pytest.fixture
    def mock_kill_switch(self):
        """Create mock kill switch for testing."""
        with patch('src.safety.kill_switch.TradingSystemKillSwitch') as mock_class:
            mock_instance = Mock()
            mock_instance.is_active.return_value = False
            mock_instance.get_status.return_value = {'shutdown_active': False}
            mock_class.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_tactical_controller(self):
        """Create mock tactical controller for testing."""
        with patch('src.tactical.controller.TacticalMARLController') as mock_class:
            mock_instance = Mock()
            mock_instance.running = False
            mock_instance.get_stats.return_value = {
                'decisions_processed': 0,
                'running': False,
                'inference_pool': {'available': False}
            }
            mock_class.return_value = mock_instance
            yield mock_instance
    
    def test_kill_switch_integration(self, controller, mock_kill_switch):
        """Test integration with kill switch system."""
        # Register kill switch as component
        controller.register_component("kill_switch", health_check_interval=1.0)
        
        # Add kill switch monitoring
        def kill_switch_health_check():
            return not mock_kill_switch.is_active()
        
        controller.add_safety_check(kill_switch_health_check)
        
        # Start system successfully
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Simulate kill switch activation
        mock_kill_switch.is_active.return_value = True
        
        # Safety check should now fail
        assert not controller._run_safety_checks()
        
        # Emergency stop should be triggered
        controller.emergency_stop(reason="kill_switch_activated")
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_tactical_controller_integration(self, controller, mock_tactical_controller):
        """Test integration with tactical controller."""
        # Register tactical controller as component
        controller.register_component("tactical_controller", health_check_interval=1.0)
        
        # Mock tactical controller initialization
        mock_tactical_controller.initialize = AsyncMock()
        mock_tactical_controller.start_event_listener = AsyncMock()
        mock_tactical_controller.stop_event_listener = AsyncMock()
        mock_tactical_controller.cleanup = AsyncMock()
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update tactical controller status
        controller.update_component_status("tactical_controller", ComponentStatus.HEALTHY)
        
        # Test system health with tactical controller
        assert controller.is_healthy()
        
        # Simulate tactical controller failure
        controller.update_component_status("tactical_controller", ComponentStatus.FAILED)
        assert not controller.is_healthy()
        
        # Emergency stop should handle tactical controller cleanup
        controller.emergency_stop(reason="tactical_controller_failed")
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_risk_management_integration(self, controller):
        """Test integration with risk management system."""
        # Register risk management components
        risk_components = [
            "var_calculator",
            "correlation_tracker",
            "position_sizing_agent",
            "risk_monitor"
        ]
        
        for component in risk_components:
            controller.register_component(component, health_check_interval=2.0)
        
        # Add risk management safety checks
        def risk_limits_check():
            # Simulate risk limits validation
            return True
        
        def var_calculation_check():
            # Simulate VaR calculation validation
            return True
        
        controller.add_safety_check(risk_limits_check)
        controller.add_safety_check(var_calculation_check)
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update all risk components to healthy
        for component in risk_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        assert controller.is_healthy()
        
        # Simulate risk breach
        controller.update_component_status("var_calculator", ComponentStatus.FAILED)
        assert not controller.is_healthy()
        
        # Test risk-based emergency stop
        controller.emergency_stop(reason="var_calculation_failed")
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_data_pipeline_integration(self, controller):
        """Test integration with data pipeline components."""
        # Register data pipeline components
        data_components = [
            "data_handler",
            "bar_generator",
            "matrix_assembler_5m",
            "matrix_assembler_30m",
            "indicator_engine"
        ]
        
        for component in data_components:
            controller.register_component(component, health_check_interval=1.0)
        
        # Add data quality safety checks
        def data_quality_check():
            # Simulate data quality validation
            return True
        
        def market_data_check():
            # Simulate market data connectivity check
            return True
        
        controller.add_safety_check(data_quality_check)
        controller.add_safety_check(market_data_check)
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update data components to healthy
        for component in data_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        assert controller.is_healthy()
        
        # Simulate data pipeline failure
        controller.update_component_status("data_handler", ComponentStatus.FAILED)
        controller.update_component_status("matrix_assembler_5m", ComponentStatus.DEGRADED)
        
        assert not controller.is_healthy()
    
    def test_execution_engine_integration(self, controller):
        """Test integration with execution engine."""
        # Register execution components
        execution_components = [
            "execution_engine",
            "order_manager",
            "broker_interface",
            "position_manager"
        ]
        
        for component in execution_components:
            controller.register_component(component, health_check_interval=1.5)
        
        # Add execution safety checks
        def broker_connectivity_check():
            # Simulate broker connectivity check
            return True
        
        def position_limits_check():
            # Simulate position limits check
            return True
        
        controller.add_safety_check(broker_connectivity_check)
        controller.add_safety_check(position_limits_check)
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update execution components to healthy
        for component in execution_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        assert controller.is_healthy()
        
        # Simulate execution engine failure
        controller.update_component_status("execution_engine", ComponentStatus.FAILED)
        assert not controller.is_healthy()
        
        # Test execution-based emergency stop
        controller.emergency_stop(reason="execution_engine_failed")
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_monitoring_system_integration(self, controller):
        """Test integration with monitoring and alerting system."""
        # Register monitoring components
        monitoring_components = [
            "prometheus_metrics",
            "health_monitor",
            "alert_manager",
            "performance_monitor"
        ]
        
        for component in monitoring_components:
            controller.register_component(component, health_check_interval=3.0)
        
        # Mock monitoring callbacks
        alert_callback = Mock()
        health_callback = Mock()
        
        # Add event handlers for monitoring
        controller.add_event_handler("system_started", alert_callback)
        controller.add_event_handler("system_stopped", alert_callback)
        controller.add_event_handler("emergency_stop", alert_callback)
        controller.add_event_handler("component_health_timeout", health_callback)
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Check monitoring event was fired
        alert_callback.assert_called()
        
        # Update monitoring components
        for component in monitoring_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # Test component health timeout event
        controller.update_component_status("health_monitor", ComponentStatus.FAILED)
        
        # Manually trigger health check to fire event
        controller._check_component_health()
        
        # Stop system
        controller.stop_system(timeout=5.0)
        
        # Check stop event was fired
        assert alert_callback.call_count >= 2  # Start and stop events
    
    def test_full_system_lifecycle_integration(self, controller):
        """Test complete system lifecycle with all components."""
        # Register all major system components
        all_components = [
            # Core components
            "kernel",
            "config_manager",
            "event_bus",
            
            # Data components
            "data_handler",
            "bar_generator",
            "matrix_assembler_5m",
            "matrix_assembler_30m",
            "indicator_engine",
            
            # AI/ML components
            "tactical_controller",
            "strategic_controller",
            "synergy_detector",
            "regime_detector",
            
            # Risk management
            "var_calculator",
            "correlation_tracker",
            "position_sizing_agent",
            "risk_monitor",
            
            # Execution
            "execution_engine",
            "order_manager",
            "broker_interface",
            
            # Monitoring
            "health_monitor",
            "alert_manager",
            "performance_monitor"
        ]
        
        for component in all_components:
            controller.register_component(component, health_check_interval=2.0)
        
        # Add comprehensive safety checks
        def comprehensive_safety_check():
            # Simulate comprehensive system safety validation
            return True
        
        controller.add_safety_check(comprehensive_safety_check)
        
        # Start system
        result = controller.start_system(timeout=10.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Update all components to healthy
        for component in all_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # System should be healthy
        assert controller.is_healthy()
        
        # Test pause/resume with all components
        result = controller.pause_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.PAUSED
        
        result = controller.resume_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Test graceful shutdown with all components
        result = controller.stop_system(timeout=10.0)
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE
        
        # Check state history captures all transitions
        history = controller.get_state_history()
        assert len(history) >= 4  # Start, pause, resume, stop
        
        # Verify all transitions were successful
        for transition in history:
            assert transition.success is True
    
    def test_cascading_failure_handling(self, controller):
        """Test handling of cascading component failures."""
        # Register interdependent components
        components = [
            "data_handler",
            "matrix_assembler",
            "tactical_controller",
            "execution_engine"
        ]
        
        for component in components:
            controller.register_component(component, health_check_interval=1.0)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set all components to healthy
        for component in components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # Simulate cascading failure
        controller.update_component_status("data_handler", ComponentStatus.FAILED)
        controller.update_component_status("matrix_assembler", ComponentStatus.FAILED)
        controller.update_component_status("tactical_controller", ComponentStatus.DEGRADED)
        
        # System should detect unhealthy state
        assert not controller.is_healthy()
        
        # Emergency stop should handle cascading failures
        result = controller.emergency_stop(reason="cascading_failure")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_real_time_component_monitoring(self, controller):
        """Test real-time component monitoring and health updates."""
        # Register components with different health check intervals
        components = [
            ("fast_component", 0.1),
            ("medium_component", 0.5),
            ("slow_component", 1.0)
        ]
        
        for name, interval in components:
            controller.register_component(name, health_check_interval=interval)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set components to healthy
        for name, _ in components:
            controller.update_component_status(name, ComponentStatus.HEALTHY)
        
        # Let monitoring run for a bit
        time.sleep(0.2)
        
        # Check that monitoring is active
        assert controller._monitoring_active
        
        # Stop one component's heartbeat (simulate timeout)
        # Component should be marked as failed after timeout
        time.sleep(0.5)  # Wait for health check
        
        # Manually trigger health check
        controller._check_component_health()
        
        # Stop system
        controller.stop_system(timeout=5.0)
        
        # Monitoring should be stopped
        assert not controller._monitoring_active
    
    def test_event_propagation_across_components(self, controller):
        """Test event propagation across integrated components."""
        # Register components
        components = ["event_producer", "event_consumer", "event_processor"]
        
        for component in components:
            controller.register_component(component, health_check_interval=1.0)
        
        # Create event handlers to track event flow
        event_log = []
        
        def log_event(event_type):
            def handler(data):
                event_log.append((event_type, data))
            return handler
        
        # Add event handlers
        controller.add_event_handler("system_started", log_event("system_started"))
        controller.add_event_handler("system_stopped", log_event("system_stopped"))
        controller.add_event_handler("emergency_stop", log_event("emergency_stop"))
        controller.add_event_handler("component_health_timeout", log_event("component_health_timeout"))
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Update components
        for component in components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # Pause and resume
        controller.pause_system(timeout=5.0)
        controller.resume_system(timeout=5.0)
        
        # Emergency stop
        controller.emergency_stop(reason="test_event_propagation")
        
        # Check events were logged
        assert len(event_log) >= 3
        
        # Check event types were captured
        event_types = [event[0] for event in event_log]
        assert "system_started" in event_types
        assert "emergency_stop" in event_types
    
    def test_cross_component_dependency_validation(self, controller):
        """Test validation of cross-component dependencies."""
        # Register components with dependencies
        controller.register_component("database", health_check_interval=2.0)
        controller.register_component("cache", health_check_interval=1.0)
        controller.register_component("api_service", health_check_interval=1.0)
        
        # Add dependency validation
        def database_dependency_check():
            # API service depends on database
            db_component = controller.get_component_status("database")
            if db_component and db_component.status == ComponentStatus.FAILED:
                return False
            return True
        
        def cache_dependency_check():
            # API service depends on cache
            cache_component = controller.get_component_status("cache")
            if cache_component and cache_component.status == ComponentStatus.FAILED:
                return False
            return True
        
        controller.add_safety_check(database_dependency_check)
        controller.add_safety_check(cache_dependency_check)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set components to healthy
        controller.update_component_status("database", ComponentStatus.HEALTHY)
        controller.update_component_status("cache", ComponentStatus.HEALTHY)
        controller.update_component_status("api_service", ComponentStatus.HEALTHY)
        
        # System should be healthy
        assert controller.is_healthy()
        
        # Fail database component
        controller.update_component_status("database", ComponentStatus.FAILED)
        
        # System should still be running but dependency check will fail
        assert not controller._run_safety_checks()
        
        # Try to start system again (should fail due to dependencies)
        controller.stop_system(timeout=5.0)
        result = controller.start_system(timeout=5.0)
        assert result is False


class TestAsyncIntegration:
    """Test integration with async components."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for async testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_async_component_lifecycle(self, controller):
        """Test integration with async components."""
        # Register async components
        async_components = [
            "async_data_streamer",
            "async_market_connector",
            "async_trade_executor"
        ]
        
        for component in async_components:
            controller.register_component(component, health_check_interval=1.0)
        
        # Mock async component initialization
        async def mock_async_init():
            await asyncio.sleep(0.1)  # Simulate async initialization
            return True
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update async components to healthy
        for component in async_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # System should be healthy
        assert controller.is_healthy()
        
        # Test async shutdown
        result = controller.stop_system(timeout=5.0)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_event_handling(self, controller):
        """Test async event handling integration."""
        # Register async event handler
        async_events = []
        
        def async_event_handler(data):
            async_events.append(data)
        
        controller.add_event_handler("system_started", async_event_handler)
        controller.add_event_handler("system_stopped", async_event_handler)
        
        # Start and stop system
        controller.start_system(timeout=5.0)
        controller.stop_system(timeout=5.0)
        
        # Check async events were handled
        assert len(async_events) >= 2


class TestFailureRecovery:
    """Test failure recovery and resilience."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for failure recovery testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=5.0)
    
    def test_component_failure_recovery(self, controller):
        """Test recovery from component failures."""
        # Register components
        components = ["service_a", "service_b", "service_c"]
        
        for component in components:
            controller.register_component(component, health_check_interval=0.5)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set components to healthy
        for component in components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # Simulate component failure
        controller.update_component_status("service_a", ComponentStatus.FAILED)
        
        # System should detect failure
        assert not controller.is_healthy()
        
        # Recover component
        controller.update_component_status("service_a", ComponentStatus.HEALTHY)
        
        # System should be healthy again
        assert controller.is_healthy()
    
    def test_system_recovery_from_emergency_stop(self, controller):
        """Test system recovery from emergency stop."""
        # Register component
        controller.register_component("test_service", health_check_interval=1.0)
        
        # Start system
        controller.start_system(timeout=5.0)
        controller.update_component_status("test_service", ComponentStatus.HEALTHY)
        
        # Emergency stop
        controller.emergency_stop(reason="test_recovery")
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # Reset failsafe
        controller.reset_failsafe(force=True)
        
        # Should be able to start system again
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
    
    def test_partial_system_recovery(self, controller):
        """Test partial system recovery scenarios."""
        # Register multiple components
        critical_components = ["critical_service_1", "critical_service_2"]
        optional_components = ["optional_service_1", "optional_service_2"]
        
        for component in critical_components + optional_components:
            controller.register_component(component, health_check_interval=1.0)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set all components to healthy
        for component in critical_components + optional_components:
            controller.update_component_status(component, ComponentStatus.HEALTHY)
        
        # Fail optional components
        for component in optional_components:
            controller.update_component_status(component, ComponentStatus.FAILED)
        
        # System should still be considered healthy if only optional components fail
        # (This would require additional logic in the controller for component priorities)
        
        # Fail critical components
        for component in critical_components:
            controller.update_component_status(component, ComponentStatus.FAILED)
        
        # System should now be unhealthy
        assert not controller.is_healthy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])