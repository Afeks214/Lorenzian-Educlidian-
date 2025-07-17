"""
Core functionality tests for the Trading System Controller (Master Switch).

This module tests the fundamental operations of the master switch system,
including state transitions, component management, and basic safety mechanisms.
"""

import pytest
import threading
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

from src.core.trading_system_controller import (
    TradingSystemController,
    SystemState,
    ComponentStatus,
    StateTransition,
    ComponentInfo
)

class TestTradingSystemController:
    """Test suite for TradingSystemController core functionality."""
    
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
        """Create controller instance for testing."""
        controller = TradingSystemController(
            max_concurrent_operations=5,
            heartbeat_timeout=1.0,
            state_persistence_path=temp_state_file
        )
        yield controller
        controller.shutdown(timeout=2.0)
    
    def test_controller_initialization(self, controller):
        """Test controller initialization."""
        assert controller.get_state() == SystemState.INACTIVE
        assert controller.get_all_components() == {}
        assert controller.get_performance_metrics()["state"] == "inactive"
        assert not controller.is_healthy()  # No components registered
    
    def test_start_system_from_inactive(self, controller):
        """Test starting system from inactive state."""
        # Register a test component
        assert controller.register_component("test_component", health_check_interval=0.5)
        
        # Start system
        result = controller.start_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Check state history
        history = controller.get_state_history()
        assert len(history) >= 1
        assert history[-1].to_state == SystemState.ACTIVE
        assert history[-1].success is True
    
    def test_start_system_already_active(self, controller):
        """Test starting system when already active."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Try to start again
        result = controller.start_system(timeout=5.0)
        assert result is True  # Should return True but not restart
        assert controller.get_state() == SystemState.ACTIVE
    
    def test_start_system_invalid_state(self, controller):
        """Test starting system from invalid state."""
        # Force controller into stopping state
        controller._state = SystemState.STOPPING
        
        result = controller.start_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.STOPPING
    
    def test_stop_system_graceful(self, controller):
        """Test graceful system shutdown."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Stop system
        result = controller.stop_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE
        
        # Check state history
        history = controller.get_state_history()
        stop_transition = next(t for t in history if t.to_state == SystemState.INACTIVE)
        assert stop_transition.success is True
    
    def test_stop_system_already_inactive(self, controller):
        """Test stopping system when already inactive."""
        result = controller.stop_system(timeout=5.0)
        assert result is True  # Should return True but not stop again
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_stop_system_force(self, controller):
        """Test force stopping system."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Simulate pending operations
        controller._active_operations["test_op"] = Mock()
        
        # Force stop
        result = controller.stop_system(timeout=5.0, force=True)
        assert result is True
        assert controller.get_state() == SystemState.INACTIVE
        assert len(controller._active_operations) == 0
    
    def test_emergency_stop(self, controller):
        """Test emergency stop functionality."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Add emergency callback
        emergency_callback = Mock()
        controller.add_emergency_callback(emergency_callback)
        
        # Emergency stop
        result = controller.emergency_stop(reason="test_emergency", initiator="test_user")
        assert result is True
        assert controller.get_state() == SystemState.EMERGENCY_STOPPED
        assert controller._failsafe_active is True
        
        # Check emergency callback was called
        emergency_callback.assert_called_once()
        
        # Check state history
        history = controller.get_state_history()
        emergency_transition = next(t for t in history if t.to_state == SystemState.EMERGENCY_STOPPED)
        assert emergency_transition.success is True
        assert emergency_transition.initiator == "test_user"
        assert "test_emergency" in emergency_transition.reason
    
    def test_pause_resume_system(self, controller):
        """Test pausing and resuming system."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Pause system
        result = controller.pause_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.PAUSED
        
        # Resume system
        result = controller.resume_system(timeout=5.0)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Check state history
        history = controller.get_state_history()
        pause_transition = next(t for t in history if t.to_state == SystemState.PAUSED)
        resume_transition = next(t for t in history if t.to_state == SystemState.ACTIVE and t.reason == "manual_resume")
        
        assert pause_transition.success is True
        assert resume_transition.success is True
    
    def test_pause_system_invalid_state(self, controller):
        """Test pausing system from invalid state."""
        result = controller.pause_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_resume_system_invalid_state(self, controller):
        """Test resuming system from invalid state."""
        result = controller.resume_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
    
    def test_component_registration(self, controller):
        """Test component registration and management."""
        # Register component
        result = controller.register_component(
            "test_component",
            health_check_interval=2.0,
            metadata={"version": "1.0.0"}
        )
        assert result is True
        
        # Check component info
        component = controller.get_component_status("test_component")
        assert component is not None
        assert component.name == "test_component"
        assert component.status == ComponentStatus.UNKNOWN
        assert component.health_check_interval == 2.0
        assert component.metadata["version"] == "1.0.0"
        
        # Test duplicate registration
        result = controller.register_component("test_component")
        assert result is False
        
        # Get all components
        all_components = controller.get_all_components()
        assert len(all_components) == 1
        assert "test_component" in all_components
    
    def test_component_unregistration(self, controller):
        """Test component unregistration."""
        # Register and then unregister component
        controller.register_component("test_component")
        
        result = controller.unregister_component("test_component")
        assert result is True
        
        # Check component is gone
        component = controller.get_component_status("test_component")
        assert component is None
        
        # Test unregistering non-existent component
        result = controller.unregister_component("nonexistent")
        assert result is False
    
    def test_component_status_updates(self, controller):
        """Test component status updates."""
        # Register component
        controller.register_component("test_component")
        
        # Update status
        result = controller.update_component_status(
            "test_component",
            ComponentStatus.HEALTHY,
            metadata={"last_operation": "test_op"}
        )
        assert result is True
        
        # Check updated status
        component = controller.get_component_status("test_component")
        assert component.status == ComponentStatus.HEALTHY
        assert component.metadata["last_operation"] == "test_op"
        
        # Test updating non-existent component
        result = controller.update_component_status("nonexistent", ComponentStatus.HEALTHY)
        assert result is False
    
    def test_event_handling(self, controller):
        """Test event handling system."""
        # Add event handler
        event_handler = Mock()
        result = controller.add_event_handler("test_event", event_handler)
        assert result is True
        
        # Emit event
        controller._emit_event("test_event", {"message": "test"})
        event_handler.assert_called_once_with({"message": "test"})
        
        # Remove event handler
        result = controller.remove_event_handler("test_event", event_handler)
        assert result is True
        
        # Test removing non-existent handler
        result = controller.remove_event_handler("test_event", event_handler)
        assert result is False
    
    def test_safety_checks(self, controller):
        """Test safety check system."""
        # Add safety check that passes
        passing_check = Mock(return_value=True)
        controller.add_safety_check(passing_check)
        
        # Add safety check that fails
        failing_check = Mock(return_value=False)
        controller.add_safety_check(failing_check)
        
        # Register component
        controller.register_component("test_component")
        
        # Try to start system (should fail due to failing check)
        result = controller.start_system(timeout=5.0)
        assert result is False
        assert controller.get_state() == SystemState.INACTIVE
        
        # Both checks should have been called
        passing_check.assert_called_once()
        failing_check.assert_called_once()
    
    def test_safety_checks_force_start(self, controller):
        """Test force starting system bypasses safety checks."""
        # Add safety check that fails
        failing_check = Mock(return_value=False)
        controller.add_safety_check(failing_check)
        
        # Register component
        controller.register_component("test_component")
        
        # Force start system (should succeed despite failing check)
        result = controller.start_system(timeout=5.0, force=True)
        assert result is True
        assert controller.get_state() == SystemState.ACTIVE
        
        # Safety check should not have been called
        failing_check.assert_not_called()
    
    def test_performance_metrics(self, controller):
        """Test performance metrics collection."""
        # Register component
        controller.register_component("test_component")
        
        # Get initial metrics
        metrics = controller.get_performance_metrics()
        assert metrics["state"] == "inactive"
        assert metrics["component_count"] == 1
        assert metrics["active_operations"] == 0
        assert metrics["failsafe_active"] is False
        
        # Start system and check metrics
        controller.start_system(timeout=5.0)
        metrics = controller.get_performance_metrics()
        assert metrics["state"] == "active"
        assert metrics["uptime"] >= 0
    
    def test_health_check(self, controller):
        """Test system health checking."""
        # System should be unhealthy when inactive
        assert not controller.is_healthy()
        
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Update component to healthy
        controller.update_component_status("test_component", ComponentStatus.HEALTHY)
        assert controller.is_healthy()
        
        # Update component to failed
        controller.update_component_status("test_component", ComponentStatus.FAILED)
        assert not controller.is_healthy()
    
    def test_state_persistence(self, controller, temp_state_file):
        """Test state persistence to disk."""
        # Register component and start system
        controller.register_component("test_component", metadata={"version": "1.0.0"})
        controller.start_system(timeout=5.0)
        
        # Force state save
        controller._persist_state()
        
        # Check file exists and has content
        assert os.path.exists(temp_state_file)
        with open(temp_state_file, 'r') as f:
            content = f.read()
            assert "active" in content
            assert "test_component" in content
    
    def test_state_loading(self, controller, temp_state_file):
        """Test loading state from disk."""
        # Create state file with test data
        import json
        state_data = {
            "state": "paused",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "test_component": {
                    "status": "healthy",
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                    "health_check_interval": 5.0,
                    "metadata": {"version": "1.0.0"}
                }
            },
            "failsafe_active": False
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(state_data, f)
        
        # Load state
        result = controller.load_state()
        assert result is True
        assert controller.get_state() == SystemState.PAUSED
        
        # Check component was loaded
        component = controller.get_component_status("test_component")
        assert component is not None
        assert component.status == ComponentStatus.HEALTHY
        assert component.metadata["version"] == "1.0.0"
    
    def test_failsafe_reset(self, controller):
        """Test failsafe reset functionality."""
        # Activate failsafe
        controller._failsafe_active = True
        
        # Reset failsafe
        result = controller.reset_failsafe()
        assert result is True
        assert controller._failsafe_active is False
        
        # Test reset from emergency stop (should fail without force)
        controller._state = SystemState.EMERGENCY_STOPPED
        controller._failsafe_active = True
        
        result = controller.reset_failsafe()
        assert result is False
        assert controller._failsafe_active is True
        
        # Test force reset
        result = controller.reset_failsafe(force=True)
        assert result is True
        assert controller._failsafe_active is False
    
    def test_concurrent_state_transitions(self, controller):
        """Test concurrent state transitions are handled safely."""
        # Register component
        controller.register_component("test_component")
        
        # Start system first
        controller.start_system(timeout=5.0)
        
        # Create multiple threads trying to change state
        results = []
        threads = []
        
        def try_pause():
            result = controller.pause_system(timeout=2.0)
            results.append(('pause', result))
        
        def try_stop():
            result = controller.stop_system(timeout=2.0)
            results.append(('stop', result))
        
        def try_emergency():
            result = controller.emergency_stop(reason="test")
            results.append(('emergency', result))
        
        # Start threads
        for func in [try_pause, try_stop, try_emergency]:
            thread = threading.Thread(target=func)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results - only one should succeed
        successful_results = [r for r in results if r[1] is True]
        assert len(successful_results) == 1
        
        # System should be in a valid final state
        final_state = controller.get_state()
        assert final_state in [SystemState.INACTIVE, SystemState.PAUSED, SystemState.EMERGENCY_STOPPED]
    
    def test_component_health_monitoring(self, controller):
        """Test component health monitoring."""
        # Register component with short heartbeat interval
        controller.register_component("test_component", health_check_interval=0.1)
        controller.start_system(timeout=5.0)
        
        # Update component to healthy
        controller.update_component_status("test_component", ComponentStatus.HEALTHY)
        
        # Wait for health check timeout
        time.sleep(0.3)  # Wait longer than 2 * health_check_interval
        
        # Manually trigger health check
        controller._check_component_health()
        
        # Component should now be marked as failed
        component = controller.get_component_status("test_component")
        assert component.status == ComponentStatus.FAILED
    
    def test_state_history_tracking(self, controller):
        """Test state history tracking."""
        # Register component
        controller.register_component("test_component")
        
        # Perform multiple state transitions
        controller.start_system(timeout=5.0)
        controller.pause_system(timeout=5.0)
        controller.resume_system(timeout=5.0)
        controller.stop_system(timeout=5.0)
        
        # Check state history
        history = controller.get_state_history()
        assert len(history) >= 4
        
        # Check state transition sequence
        states = [t.to_state for t in history]
        assert SystemState.ACTIVE in states
        assert SystemState.PAUSED in states
        assert SystemState.INACTIVE in states
        
        # Check all transitions have required fields
        for transition in history:
            assert transition.id is not None
            assert transition.timestamp is not None
            assert transition.initiator is not None
            assert transition.reason is not None
            assert isinstance(transition.success, bool)
    
    def test_error_handling_in_state_transitions(self, controller):
        """Test error handling during state transitions."""
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
            assert "error" in error_transition.metadata
    
    def test_cleanup_on_shutdown(self, controller):
        """Test proper cleanup on controller shutdown."""
        # Register component and start system
        controller.register_component("test_component")
        controller.start_system(timeout=5.0)
        
        # Add event handler
        event_handler = Mock()
        controller.add_event_handler("test_event", event_handler)
        
        # Shutdown controller
        controller.shutdown(timeout=5.0)
        
        # Check system is inactive
        assert controller.get_state() == SystemState.INACTIVE
        
        # Check event handlers are cleared
        assert len(controller._event_handlers) == 0
        
        # Check operations are cleared
        assert len(controller._active_operations) == 0


class TestStateTransitions:
    """Test state transition logic and validation."""
    
    def test_valid_state_transitions(self):
        """Test all valid state transitions are allowed."""
        # Define valid transitions
        valid_transitions = {
            SystemState.INACTIVE: [SystemState.STARTING],
            SystemState.STARTING: [SystemState.ACTIVE, SystemState.ERROR],
            SystemState.ACTIVE: [SystemState.PAUSING, SystemState.STOPPING, SystemState.EMERGENCY_STOPPED],
            SystemState.PAUSING: [SystemState.PAUSED, SystemState.ERROR],
            SystemState.PAUSED: [SystemState.STARTING, SystemState.STOPPING],
            SystemState.STOPPING: [SystemState.INACTIVE, SystemState.ERROR],
            SystemState.EMERGENCY_STOPPED: [SystemState.INACTIVE],  # Only with reset
            SystemState.ERROR: [SystemState.INACTIVE]  # Only with reset
        }
        
        # Test each transition
        for from_state, to_states in valid_transitions.items():
            for to_state in to_states:
                # Create transition
                transition = StateTransition(
                    id=str(uuid.uuid4()),
                    from_state=from_state,
                    to_state=to_state,
                    timestamp=datetime.now(timezone.utc),
                    initiator="test",
                    reason="test_transition",
                    success=True
                )
                
                # Validate transition is valid
                assert transition.from_state == from_state
                assert transition.to_state == to_state
                assert transition.success is True
    
    def test_state_transition_metadata(self):
        """Test state transition metadata handling."""
        transition = StateTransition(
            id=str(uuid.uuid4()),
            from_state=SystemState.INACTIVE,
            to_state=SystemState.ACTIVE,
            timestamp=datetime.now(timezone.utc),
            initiator="test_user",
            reason="test_reason",
            success=True,
            duration_ms=150.5,
            metadata={"custom_field": "custom_value"}
        )
        
        assert transition.duration_ms == 150.5
        assert transition.metadata["custom_field"] == "custom_value"
        
        # Test serialization
        transition_dict = asdict(transition)
        assert "duration_ms" in transition_dict
        assert "metadata" in transition_dict


class TestComponentManagement:
    """Test component registration and management."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for component testing."""
        controller = TradingSystemController()
        yield controller
        controller.shutdown(timeout=2.0)
    
    def test_component_lifecycle(self, controller):
        """Test complete component lifecycle."""
        # Register component
        result = controller.register_component(
            "lifecycle_test",
            health_check_interval=1.0,
            metadata={"type": "test_component"}
        )
        assert result is True
        
        # Check initial state
        component = controller.get_component_status("lifecycle_test")
        assert component.status == ComponentStatus.UNKNOWN
        assert component.metadata["type"] == "test_component"
        
        # Update to healthy
        controller.update_component_status("lifecycle_test", ComponentStatus.HEALTHY)
        component = controller.get_component_status("lifecycle_test")
        assert component.status == ComponentStatus.HEALTHY
        
        # Update to degraded with additional metadata
        controller.update_component_status(
            "lifecycle_test",
            ComponentStatus.DEGRADED,
            metadata={"warning": "performance_degraded"}
        )
        component = controller.get_component_status("lifecycle_test")
        assert component.status == ComponentStatus.DEGRADED
        assert component.metadata["warning"] == "performance_degraded"
        assert component.metadata["type"] == "test_component"  # Original metadata preserved
        
        # Update to failed
        controller.update_component_status("lifecycle_test", ComponentStatus.FAILED)
        component = controller.get_component_status("lifecycle_test")
        assert component.status == ComponentStatus.FAILED
        
        # Unregister component
        result = controller.unregister_component("lifecycle_test")
        assert result is True
        
        # Check component is gone
        component = controller.get_component_status("lifecycle_test")
        assert component is None
    
    def test_multiple_components(self, controller):
        """Test managing multiple components."""
        # Register multiple components
        components = ["comp1", "comp2", "comp3"]
        for comp in components:
            controller.register_component(comp, health_check_interval=0.5)
        
        # Check all components are registered
        all_components = controller.get_all_components()
        assert len(all_components) == 3
        for comp in components:
            assert comp in all_components
        
        # Update each component to different status
        controller.update_component_status("comp1", ComponentStatus.HEALTHY)
        controller.update_component_status("comp2", ComponentStatus.DEGRADED)
        controller.update_component_status("comp3", ComponentStatus.FAILED)
        
        # Check individual statuses
        assert controller.get_component_status("comp1").status == ComponentStatus.HEALTHY
        assert controller.get_component_status("comp2").status == ComponentStatus.DEGRADED
        assert controller.get_component_status("comp3").status == ComponentStatus.FAILED
    
    def test_component_heartbeat_tracking(self, controller):
        """Test component heartbeat tracking."""
        # Register component
        controller.register_component("heartbeat_test", health_check_interval=0.1)
        
        # Get initial heartbeat
        component = controller.get_component_status("heartbeat_test")
        initial_heartbeat = component.last_heartbeat
        
        # Wait a bit and update status
        time.sleep(0.05)
        controller.update_component_status("heartbeat_test", ComponentStatus.HEALTHY)
        
        # Check heartbeat was updated
        component = controller.get_component_status("heartbeat_test")
        assert component.last_heartbeat > initial_heartbeat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])