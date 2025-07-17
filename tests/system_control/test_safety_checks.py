"""
Safety mechanism tests for the Trading System Controller.

This module tests all safety mechanisms including failsafe operations,
emergency protocols, validation checks, and recovery procedures.
"""

import pytest
import threading
import time
import signal
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, Future

from src.core.trading_system_controller import (
    TradingSystemController,
    SystemState,
    ComponentStatus,
    StateTransition
)


class TestSafetyMechanisms:
    """Test suite for safety mechanisms and failsafe operations."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for safety testing."""
        controller = TradingSystemController(
            max_concurrent_operations=5,
            heartbeat_timeout=1.0
        )
        yield controller
        controller.shutdown(timeout=5.0)
    
    def test_safety_check_validation(self, controller):
        """Test safety check validation during startup."""
        # Add multiple safety checks
        checks = {
            "market_connectivity": Mock(return_value=True),
            "risk_limits": Mock(return_value=True),
            "broker_connection": Mock(return_value=True),
            "data_quality": Mock(return_value=True)
        }
        
        for name, check in checks.items():
            controller.add_safety_check(check)
        
        # Register component
        controller.register_component("test_component")
        
        # Start system - should succeed with all checks passing
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Verify all checks were called
        for check in checks.values():
            check.assert_called_once()
    
    def test_safety_check_failure_blocks_startup(self, controller):
        """Test that failed safety checks block system startup."""
        # Add passing and failing checks
        passing_check = Mock(return_value=True)
        failing_check = Mock(return_value=False)
        
        controller.add_safety_check(passing_check)
        controller.add_safety_check(failing_check)
        
        # Register component
        controller.register_component("test_component")
        
        # Start system - should fail due to failing check
        result = controller.start_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
        
        # Both checks should have been called
        passing_check.assert_called_once()
        failing_check.assert_called_once()
    
    def test_safety_check_exception_handling(self, controller):
        """Test handling of exceptions in safety checks."""
        # Add check that raises exception
        exception_check = Mock(side_effect=Exception("Safety check failed"))
        normal_check = Mock(return_value=True)
        
        controller.add_safety_check(exception_check)
        controller.add_safety_check(normal_check)
        
        # Register component
        controller.register_component("test_component")
        
        # Start system - should fail due to exception
        result = controller.start_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
        
        # Exception check should have been called
        exception_check.assert_called_once()
        # Normal check should not be called after exception
        normal_check.assert_not_called()
    
    def test_force_start_bypasses_safety_checks(self, controller):
        """Test force start bypasses safety checks."""
        # Add failing safety check
        failing_check = Mock(return_value=False)
        controller.add_safety_check(failing_check)
        
        # Register component
        controller.register_component("test_component")
        
        # Force start should succeed despite failing check
        result = controller.start_system(timeout=5.0, force=True)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Safety check should not have been called
        failing_check.assert_not_called()
    
    def test_emergency_stop_callbacks(self, controller):
        """Test emergency stop callbacks execution."""
        # Add emergency callbacks
        callbacks = [Mock(), Mock(), Mock()]
        
        for callback in callbacks:
            controller.add_emergency_callback(callback)
        
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Trigger emergency stop
        result = controller.emergency_stop(reason="test_emergency")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # All callbacks should have been called
        for callback in callbacks:
            callback.assert_called_once()
    
    def test_emergency_stop_callback_exceptions(self, controller):
        """Test emergency stop continues even if callbacks fail."""
        # Add callbacks, some that fail
        good_callback = Mock()
        bad_callback = Mock(side_effect=Exception("Callback failed"))
        another_good_callback = Mock()
        
        controller.add_emergency_callback(good_callback)
        controller.add_emergency_callback(bad_callback)
        controller.add_emergency_callback(another_good_callback)
        
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Trigger emergency stop
        result = controller.emergency_stop(reason="test_emergency")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # All callbacks should have been called
        good_callback.assert_called_once()
        bad_callback.assert_called_once()
        another_good_callback.assert_called_once()
    
    def test_failsafe_activation_and_reset(self, controller):
        """Test failsafe activation and reset mechanism."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Initially failsafe should be inactive
        assert controller._failsafe_active is False
        
        # Emergency stop should activate failsafe
        controller.emergency_stop(reason="test_failsafe")
        assert controller._failsafe_active is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # Try to reset failsafe without force (should fail)
        result = controller.reset_failsafe(force=False)
        assert result is False
        assert controller._failsafe_active is True
        
        # Reset failsafe with force
        result = controller.reset_failsafe(force=True)
        assert result is True
        assert controller._failsafe_active is False
    
    def test_component_health_timeout_detection(self, controller):
        """Test component health timeout detection."""
        # Register component with short timeout
        controller.register_component("timeout_test", health_check_interval=0.1)
        controller.start_system(timeout=5.0)
        
        # Set component to healthy
        controller.update_component_status("timeout_test", ComponentStatus.HEALTHY)
        
        # Wait for timeout
        time.sleep(0.3)  # Wait 3x the heartbeat interval
        
        # Manually trigger health check
        controller._check_component_health()
        
        # Component should be marked as failed
        component = controller.get_component_status("timeout_test")
        assert component.status == ComponentStatus.FAILED
        
        # System should be unhealthy
        assert not controller.is_healthy()
    
    def test_component_health_timeout_event(self, controller):
        """Test component health timeout events are fired."""
        # Register event handler
        timeout_events = []
        
        def timeout_handler(data):
            timeout_events.append(data)
        
        controller.add_event_handler("component_health_timeout", timeout_handler)
        
        # Register component with short timeout
        controller.register_component("timeout_test", health_check_interval=0.1)
        controller.start_system(timeout=5.0)
        
        # Set component to healthy
        controller.update_component_status("timeout_test", ComponentStatus.HEALTHY)
        
        # Wait for timeout
        time.sleep(0.3)
        
        # Manually trigger health check
        controller._check_component_health()
        
        # Timeout event should have been fired
        assert len(timeout_events) == 1
        assert timeout_events[0]["component"] == "timeout_test"
        assert "elapsed" in timeout_events[0]
        assert "timeout" in timeout_events[0]
    
    def test_concurrent_emergency_stops(self, controller):
        """Test concurrent emergency stops are handled safely."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Create multiple threads calling emergency stop
        results = []
        threads = []
        
        def emergency_stop_thread(reason):
            result = controller.emergency_stop(reason=f"emergency_{reason}")
            results.append(result)
        
        # Start multiple emergency stop threads
        for i in range(5):
            thread = threading.Thread(target=emergency_stop_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Only one emergency stop should succeed
        successful_stops = sum(1 for result in results if result is True)
        assert successful_stops == 1
        
        # System should be in emergency stopped state
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_operation_cancellation_on_emergency_stop(self, controller):
        """Test that all operations are cancelled on emergency stop."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Add mock operations
        mock_future1 = Mock(spec=Future)
        mock_future2 = Mock(spec=Future)
        
        controller._active_operations["op1"] = mock_future1
        controller._active_operations["op2"] = mock_future2
        
        # Emergency stop
        controller.emergency_stop(reason="test_cancellation")
        
        # All operations should be cancelled
        mock_future1.cancel.assert_called_once()
        mock_future2.cancel.assert_called_once()
        
        # Operations list should be empty
        assert len(controller._active_operations) == 0
    
    def test_state_persistence_on_emergency_stop(self, controller):
        """Test state persistence during emergency stop."""
        # Create temporary state file
        fd, state_file = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        
        try:
            # Configure controller with state file
            controller._state_persistence_path = state_file
            
            # Register component and start system
            controller.register_component("test_component")
            controller.start_system(timeout=5.0)
            
            # Emergency stop
            controller.emergency_stop(reason="test_persistence")
            
            # Check state file was created and contains emergency stop state
            assert os.path.exists(state_file)
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            assert state_data["state"] == "emergency_stopped"
            assert state_data["failsafe_active"] is True
            
        finally:
            try:
                os.unlink(state_file)
            except FileNotFoundError:
                pass
    
    def test_system_health_monitoring(self, controller):
        """Test continuous system health monitoring."""
        # Register components
        components = ["comp1", "comp2", "comp3"]
        for comp in components:
            controller.register_component(comp, health_check_interval=0.2)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Set components to healthy
        for comp in components:
            controller.update_component_status(comp, ComponentStatus.HEALTHY)
        
        # System should be healthy
        assert controller.is_healthy()
        
        # Fail one component
        controller.update_component_status("comp1", ComponentStatus.FAILED)
        
        # System should be unhealthy
        assert not controller.is_healthy()
        
        # Degrade another component
        controller.update_component_status("comp2", ComponentStatus.DEGRADED)
        
        # System should still be unhealthy
        assert not controller.is_healthy()
        
        # Fix failed component
        controller.update_component_status("comp1", ComponentStatus.HEALTHY)
        
        # System should still be unhealthy due to degraded component
        assert not controller.is_healthy()
        
        # Fix degraded component
        controller.update_component_status("comp2", ComponentStatus.HEALTHY)
        
        # System should be healthy again
        assert controller.is_healthy()
    
    def test_graceful_shutdown_with_pending_operations(self, controller):
        """Test graceful shutdown waits for pending operations."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Add mock pending operation
        mock_future = Mock(spec=Future)
        mock_future.done.return_value = False
        
        controller._active_operations["pending_op"] = mock_future
        
        # Start shutdown in separate thread
        shutdown_result = []
        
        def shutdown_thread():
            result = controller.stop_system(timeout=2.0)
            shutdown_result.append(result)
        
        thread = threading.Thread(target=shutdown_thread)
        thread.start()
        
        # Wait a bit, then complete the operation
        time.sleep(0.5)
        mock_future.done.return_value = True
        controller._active_operations.clear()
        
        # Wait for shutdown to complete
        thread.join(timeout=5.0)
        
        # Shutdown should have succeeded
        assert len(shutdown_result) == 1
        assert shutdown_result[0] is True
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_forced_shutdown_cancels_operations(self, controller):
        """Test forced shutdown cancels pending operations."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Add mock pending operation
        mock_future = Mock(spec=Future)
        mock_future.done.return_value = False
        
        controller._active_operations["pending_op"] = mock_future
        
        # Force shutdown
        result = controller.stop_system(timeout=2.0, force=True)
        
        # Shutdown should succeed
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE
        
        # Operation should be cancelled
        mock_future.cancel.assert_called_once()
    
    def test_safety_check_with_component_dependencies(self, controller):
        """Test safety checks that depend on component states."""
        # Register interdependent components
        controller.register_component("database", health_check_interval=1.0)
        controller.register_component("cache", health_check_interval=1.0)
        controller.register_component("api_service", health_check_interval=1.0)
        
        # Add dependency safety check
        def dependency_check():
            # API service depends on database and cache
            db_component = controller.get_component_status("database")
            cache_component = controller.get_component_status("cache")
            
            if db_component and db_component.status == ComponentStatus.FAILED:
                return False
            if cache_component and cache_component.status == ComponentStatus.FAILED:
                return False
            
            return True
        
        controller.add_safety_check(dependency_check)
        
        # Set components to healthy and start system
        controller.update_component_status("database", ComponentStatus.HEALTHY)
        controller.update_component_status("cache", ComponentStatus.HEALTHY)
        controller.update_component_status("api_service", ComponentStatus.HEALTHY)
        
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Stop system and fail database
        controller.stop_system(timeout=5.0)
        controller.update_component_status("database", ComponentStatus.FAILED)
        
        # Try to start system again - should fail due to dependency
        result = controller.start_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_emergency_stop_from_different_states(self, controller):
        """Test emergency stop works from all system states."""
        # Register component
        controller.register_component("test_component")
        
        # Test emergency stop from INACTIVE
        result = controller.emergency_stop(reason="from_inactive")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # Reset and test from ACTIVE
        controller.reset_failsafe(force=True)
        controller.start_system(timeout=5.0)
        
        result = controller.emergency_stop(reason="from_active")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        
        # Reset and test from PAUSED
        controller.reset_failsafe(force=True)
        controller.start_system(timeout=5.0)
        controller.pause_system(timeout=5.0)
        
        result = controller.emergency_stop(reason="from_paused")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
    
    def test_monitoring_thread_safety(self, controller):
        """Test monitoring thread safety during concurrent operations."""
        # Register component with short heartbeat
        controller.register_component("test_component", health_check_interval=0.1)
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Continuously update component status from multiple threads
        def update_component_status():
            for i in range(10):
                controller.update_component_status("test_component", ComponentStatus.HEALTHY)
                time.sleep(0.05)
        
        # Start multiple update threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_component_status)
            threads.append(thread)
            thread.start()
        
        # Let monitoring run while updates happen
        time.sleep(0.5)
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # System should still be functional
        assert controller.is_healthy()
        
        # Stop system
        controller.stop_system(timeout=5.0)


class TestFailureScenarios:
    """Test various failure scenarios and recovery mechanisms."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for failure testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=5.0)
    
    def test_component_startup_failure(self, controller):
        """Test handling of component startup failures."""
        # Mock component startup to fail
        with patch.object(controller, '_start_components', side_effect=Exception("Component startup failed")):
            controller.register_component("test_component")
            
            result = controller.start_system(timeout=5.0)
            assert result is False
            assert controller.get_state() == SystemState.ERROR
            
            # Check error is recorded in state history
            history = controller.get_state_history()
            error_transition = next(t for t in history if t.to_state == SystemState.ERROR)
            assert error_transition.success is False
            assert error_transition.metadata is not None
    
    def test_component_shutdown_failure(self, controller):
        """Test handling of component shutdown failures."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Mock component shutdown to fail
        with patch.object(controller, '_stop_components', side_effect=Exception("Component shutdown failed")):
            result = controller.stop_system(timeout=5.0)
            assert result is False
            assert controller.get_state() == SystemState.ERROR
    
    def test_state_persistence_failure(self, controller):
        """Test handling of state persistence failures."""
        # Configure invalid state file path
        controller._state_persistence_path = "/invalid/path/state.json"
        
        # Register component and start system
        controller.register_component("test_component")
        
        # System should still start despite persistence failure
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
    
    def test_event_handler_failure(self, controller):
        """Test system continues despite event handler failures."""
        # Add event handler that fails
        def failing_handler(data):
            raise Exception("Event handler failed")
        
        controller.add_event_handler("system_started", failing_handler)
        
        # Register component and start system
        controller.register_component("test_component")
        
        # System should start despite handler failure
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
    
    def test_monitoring_thread_failure_recovery(self, controller):
        """Test monitoring thread failure and recovery."""
        # Mock monitoring loop to fail initially
        original_check = controller._check_component_health
        
        def failing_check():
            # Fail first time, then work normally
            if not hasattr(failing_check, 'called'):
                failing_check.called = True
                raise Exception("Monitoring failed")
            return original_check()
        
        controller._check_component_health = failing_check
        
        # Register component and start system
        controller.register_component("test_component", health_check_interval=0.1)
        controller.start_system(timeout=5.0)
        
        # Let monitoring run and recover
        time.sleep(0.5)
        
        # System should still be functional
        assert controller._monitoring_active
        
        # Stop system
        controller.stop_system(timeout=5.0)
    
    def test_memory_exhaustion_protection(self, controller):
        """Test protection against memory exhaustion."""
        # Register many components to simulate memory pressure
        for i in range(1000):
            controller.register_component(f"component_{i}", health_check_interval=60.0)
        
        # System should still function
        result = controller.start_system(timeout=5.0)
        assert result is True
        
        # Update all components (simulate memory pressure)
        for i in range(1000):
            controller.update_component_status(f"component_{i}", ComponentStatus.HEALTHY)
        
        # System should still be healthy
        assert controller.is_healthy()
        
        # Stop system
        controller.stop_system(timeout=5.0)
    
    def test_concurrent_state_corruption_protection(self, controller):
        """Test protection against concurrent state corruption."""
        # Register component
        controller.register_component("test_component")
        
        # Start system
        controller.start_system(timeout=5.0)
        
        # Attempt to corrupt state from multiple threads
        def corrupt_state():
            for _ in range(100):
                try:
                    # Try various state manipulations
                    controller._state = SystemState.ERROR
                    controller._state = SystemState.ACTIVE
                    controller._failsafe_active = True
                    controller._failsafe_active = False
                except Exception:
                    pass  # Ignore any exceptions
        
        # Start multiple corruption threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=corrupt_state)
            threads.append(thread)
            thread.start()
        
        # Let corruption attempts run
        time.sleep(0.5)
        
        # Stop system normally
        result = controller.stop_system(timeout=5.0)
        
        # Wait for corruption threads
        for thread in threads:
            thread.join(timeout=1.0)
        
        # System should have stopped normally
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE


class TestValidationMechanisms:
    """Test validation mechanisms and input sanitization."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for validation testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=5.0)
    
    def test_component_name_validation(self, controller):
        """Test component name validation."""
        # Test valid names
        valid_names = ["component1", "data_handler", "risk-manager", "api.service"]
        for name in valid_names:
            result = controller.register_component(name)
            assert result is True
        
        # Test invalid names (if validation is implemented)
        # This would require additional validation logic in the controller
        pass
    
    def test_timeout_validation(self, controller):
        """Test timeout parameter validation."""
        # Register component
        controller.register_component("test_component")
        
        # Test with various timeout values
        valid_timeouts = [0.1, 1.0, 5.0, 30.0]
        for timeout in valid_timeouts:
            result = controller.start_system(timeout=timeout)
            assert result is True
            
            result = controller.stop_system(timeout=timeout)
            assert result is True
    
    def test_metadata_validation(self, controller):
        """Test metadata validation for components."""
        # Test various metadata types
        metadata_tests = [
            {"simple": "value"},
            {"number": 42},
            {"boolean": True},
            {"list": [1, 2, 3]},
            {"nested": {"key": "value"}},
            {}  # Empty metadata
        ]
        
        for i, metadata in enumerate(metadata_tests):
            result = controller.register_component(f"test_component_{i}", metadata=metadata)
            assert result is True
            
            # Verify metadata was stored
            component = controller.get_component_status(f"test_component_{i}")
            assert component.metadata == metadata
    
    def test_state_transition_validation(self, controller):
        """Test state transition validation."""
        # Register component
        controller.register_component("test_component")
        
        # Test valid state transitions
        assert controller.get_state() == SystemState.INACTIVE
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Pause system
        result = controller.pause_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.PAUSED
        
        # Resume system
        result = controller.resume_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Stop system
        result = controller.stop_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_concurrent_access_validation(self, controller):
        """Test validation of concurrent access patterns."""
        # Register component
        controller.register_component("test_component")
        
        # Test concurrent component registration
        results = []
        
        def register_components():
            for i in range(10):
                result = controller.register_component(f"concurrent_comp_{i}")
                results.append(result)
        
        # Start multiple registration threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=register_components)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # All registrations should succeed
        assert all(results)
        
        # Check all components were registered
        all_components = controller.get_all_components()
        assert len(all_components) >= 1  # At least the initial component


if __name__ == "__main__":
    pytest.main([__file__, "-v"])