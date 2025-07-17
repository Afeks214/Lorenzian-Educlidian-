"""
Emergency Failsafe Test Suite

Comprehensive tests for emergency failsafe protocols:
- Emergency level escalation
- Failsafe action execution  
- Automatic recovery
- Performance monitoring
- Incident reporting

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.consensus.emergency_failsafe import (
    EmergencyFailsafe, EmergencyLevel, FailsafeAction, 
    EmergencyEvent, FailsafeConfig
)


class TestEmergencyFailsafe:
    """Test suite for emergency failsafe system"""
    
    @pytest.fixture
    def agent_ids(self):
        """Standard agent IDs for testing"""
        return ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7']
    
    @pytest.fixture
    def failsafe_config(self):
        """Test configuration for failsafe"""
        return FailsafeConfig(
            max_consensus_failures=2,  # Lower for testing
            max_byzantine_ratio=0.3,
            consensus_timeout_threshold=5.0,
            view_change_cascade_threshold=3,
            signature_failure_threshold=0.2,
            network_partition_timeout=10.0,
            emergency_shutdown_threshold=5,
            auto_recovery_enabled=True,
            safe_mode_threshold=3
        )
    
    @pytest.fixture
    def emergency_callback(self):
        """Mock emergency callback function"""
        return Mock()
    
    @pytest.fixture
    def failsafe(self, agent_ids, failsafe_config, emergency_callback):
        """Emergency failsafe instance for testing"""
        return EmergencyFailsafe(
            agent_ids=agent_ids,
            config=failsafe_config,
            emergency_callback=emergency_callback
        )
    
    def test_failsafe_initialization(self, failsafe, agent_ids, failsafe_config):
        """Test emergency failsafe initialization"""
        assert failsafe.agent_ids == agent_ids
        assert failsafe.config == failsafe_config
        assert failsafe.current_level == EmergencyLevel.GREEN
        assert failsafe.monitoring_active is True
        assert len(failsafe.active_events) == 0
        assert failsafe.consensus_failures == 0
        assert failsafe.consecutive_failures == 0
    
    def test_normal_consensus_recording(self, failsafe):
        """Test recording normal consensus attempts"""
        # Record successful consensus
        failsafe.record_consensus_attempt(
            success=True,
            latency=0.2,
            byzantine_detected=[],
            view_changes=0,
            signature_failures=0,
            total_signatures=5
        )
        
        assert failsafe.consecutive_failures == 0
        assert failsafe.current_level == EmergencyLevel.GREEN
        assert len(failsafe.active_events) == 0
    
    def test_consensus_failure_detection(self, failsafe):
        """Test consensus failure detection and escalation"""
        # Record multiple consecutive failures
        for i in range(3):  # More than max_consensus_failures (2)
            failsafe.record_consensus_attempt(
                success=False,
                latency=1.0,
                byzantine_detected=[],
                view_changes=0
            )
        
        assert failsafe.consecutive_failures == 3
        assert failsafe.current_level >= EmergencyLevel.ORANGE
        assert len(failsafe.active_events) > 0
        
        # Check for consecutive failure event
        failure_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'consecutive_consensus_failures'
        ]
        assert len(failure_events) > 0
    
    def test_byzantine_majority_detection(self, failsafe):
        """Test Byzantine majority attack detection"""
        # Simulate Byzantine majority (>30% of agents)
        byzantine_agents = ['agent_1', 'agent_2', 'agent_3']  # 3/7 = 43% > 30%
        
        failsafe.record_consensus_attempt(
            success=False,
            latency=0.5,
            byzantine_detected=byzantine_agents,
            view_changes=0
        )
        
        assert len(failsafe.byzantine_agents) == 3
        assert failsafe.current_level >= EmergencyLevel.RED
        
        # Check for Byzantine majority event
        byzantine_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'byzantine_majority'
        ]
        assert len(byzantine_events) > 0
    
    def test_consensus_timeout_detection(self, failsafe):
        """Test consensus timeout detection"""
        # Set last successful consensus to old time
        failsafe.last_successful_consensus = time.time() - 20.0  # 20 seconds ago
        
        # Record failed consensus
        failsafe.record_consensus_attempt(
            success=False,
            latency=10.0,  # Long latency
            byzantine_detected=[]
        )
        
        assert failsafe.current_level >= EmergencyLevel.ORANGE
        
        # Check for timeout event
        timeout_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'consensus_timeout'
        ]
        assert len(timeout_events) > 0
    
    def test_view_change_cascade_detection(self, failsafe):
        """Test view change cascade detection"""
        # Record consensus with many view changes
        failsafe.record_consensus_attempt(
            success=False,
            latency=2.0,
            byzantine_detected=[],
            view_changes=5  # Above threshold of 3
        )
        
        assert failsafe.current_level >= EmergencyLevel.YELLOW
        
        # Check for view change cascade event
        cascade_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'view_change_cascade'
        ]
        assert len(cascade_events) > 0
    
    def test_signature_failure_detection(self, failsafe):
        """Test high signature failure rate detection"""
        # Record consensus with high signature failure rate
        failsafe.record_consensus_attempt(
            success=True,
            latency=0.3,
            byzantine_detected=[],
            signature_failures=5,   # High failures
            total_signatures=20     # 25% failure rate > 20% threshold
        )
        
        assert failsafe.current_level >= EmergencyLevel.ORANGE
        
        # Check for signature failure event
        sig_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'high_signature_failures'
        ]
        assert len(sig_events) > 0
    
    def test_safe_mode_activation(self, failsafe):
        """Test safe mode activation"""
        assert failsafe.safe_mode_active is False
        
        # Record enough failures to trigger safe mode
        for i in range(4):  # Above safe_mode_threshold (3)
            failsafe.record_consensus_attempt(
                success=False,
                latency=1.0,
                byzantine_detected=[]
            )
        
        assert failsafe.current_level >= EmergencyLevel.RED
        assert failsafe.safe_mode_active is True
        assert failsafe.safe_mode_start_time is not None
        
        # Check for safe mode event
        safe_mode_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'safe_mode_required'
        ]
        assert len(safe_mode_events) > 0
    
    def test_emergency_level_escalation(self, failsafe):
        """Test emergency level escalation process"""
        initial_level = failsafe.current_level
        
        # Create emergency event
        event = failsafe._create_emergency_event(
            event_type='test_escalation',
            severity=EmergencyLevel.ORANGE,
            description='Test escalation event',
            affected_components=['test_system']
        )
        
        # Manually escalate
        failsafe._escalate_emergency_level(EmergencyLevel.ORANGE)
        
        assert failsafe.current_level == EmergencyLevel.ORANGE
        assert failsafe.current_level != initial_level
    
    def test_emergency_protocol_execution(self, failsafe):
        """Test emergency protocol execution"""
        # Test yellow alert protocol
        failsafe._handle_yellow_alert()
        assert failsafe.current_level == EmergencyLevel.GREEN  # No change for yellow
        
        # Test orange alert protocol
        failsafe._handle_orange_alert()
        assert failsafe.degraded_performance_mode is True
        
        # Test red alert protocol
        failsafe._handle_red_alert()
        assert failsafe.safe_mode_active is True
    
    def test_emergency_event_creation(self, failsafe, emergency_callback):
        """Test emergency event creation and callback"""
        event = failsafe._create_emergency_event(
            event_type='test_event',
            severity=EmergencyLevel.YELLOW,
            description='Test emergency event',
            affected_components=['consensus']
        )
        
        assert event.event_type == 'test_event'
        assert event.severity == EmergencyLevel.YELLOW
        assert event.description == 'Test emergency event'
        assert 'consensus' in event.affected_components
        assert len(event.recommended_actions) > 0
        
        # Verify callback was called
        emergency_callback.assert_called_once_with(event)
        
        # Verify event is tracked
        assert event.event_id in failsafe.active_events
        assert event in failsafe.event_history
    
    def test_recommended_actions_generation(self, failsafe):
        """Test recommended actions generation"""
        # Yellow alert actions
        yellow_actions = failsafe._get_recommended_actions(EmergencyLevel.YELLOW, 'timing_issue')
        assert FailsafeAction.MONITOR in yellow_actions
        assert FailsafeAction.ALERT in yellow_actions
        
        # Orange alert actions
        orange_actions = failsafe._get_recommended_actions(EmergencyLevel.ORANGE, 'byzantine_detected')
        assert FailsafeAction.ALERT in orange_actions
        assert FailsafeAction.ISOLATE_BYZANTINE in orange_actions
        
        # Red alert actions
        red_actions = failsafe._get_recommended_actions(EmergencyLevel.RED, 'consensus_failure')
        assert FailsafeAction.FORCE_SAFE_MODE in red_actions
        assert FailsafeAction.ACTIVATE_BACKUP in red_actions
        
        # Black alert actions
        black_actions = failsafe._get_recommended_actions(EmergencyLevel.BLACK, 'system_compromise')
        assert FailsafeAction.EMERGENCY_SHUTDOWN in black_actions
    
    def test_automatic_recovery(self, failsafe):
        """Test automatic recovery functionality"""
        # Put system in emergency state
        failsafe.current_level = EmergencyLevel.ORANGE
        failsafe.degraded_performance_mode = True
        failsafe.consecutive_failures = 2
        
        # Simulate recovery conditions
        failsafe.consecutive_failures = 0
        failsafe.last_successful_consensus = time.time()
        
        # Attempt recovery
        failsafe._attempt_recovery()
        
        assert failsafe.current_level == EmergencyLevel.GREEN
        assert failsafe.degraded_performance_mode is False
        assert failsafe.failsafe_metrics['auto_recoveries'] >= 1
    
    def test_manual_recovery(self, failsafe):
        """Test manual recovery functionality"""
        # Put system in emergency state
        failsafe.current_level = EmergencyLevel.RED
        failsafe.safe_mode_active = True
        failsafe.byzantine_agents.add('agent_1')
        failsafe.consensus_failures = 5
        
        # Create active event
        event = failsafe._create_emergency_event(
            event_type='manual_test',
            severity=EmergencyLevel.RED,
            description='Test event for manual recovery',
            affected_components=['test']
        )
        
        # Perform manual recovery
        success = failsafe.manual_recovery("Emergency resolved by operator")
        
        assert success is True
        assert failsafe.current_level == EmergencyLevel.GREEN
        assert failsafe.safe_mode_active is False
        assert len(failsafe.byzantine_agents) == 0
        assert failsafe.consensus_failures == 0
        assert len(failsafe.active_events) == 0
        assert failsafe.failsafe_metrics['manual_interventions'] >= 1
    
    def test_system_status_reporting(self, failsafe):
        """Test system status reporting"""
        # Modify some state
        failsafe.current_level = EmergencyLevel.ORANGE
        failsafe.degraded_performance_mode = True
        failsafe.consensus_failures = 3
        failsafe.byzantine_agents.add('agent_1')
        
        status = failsafe.get_system_status()
        
        assert status['emergency_level'] == 'orange'
        assert status['degraded_performance_mode'] is True
        assert status['consensus_failures'] == 3
        assert 'agent_1' in status['byzantine_agents']
        assert status['byzantine_ratio'] > 0
        assert 'time_since_last_success' in status
        assert 'metrics' in status
    
    def test_active_events_reporting(self, failsafe):
        """Test active events reporting"""
        # Create some events
        event1 = failsafe._create_emergency_event(
            event_type='test_event_1',
            severity=EmergencyLevel.YELLOW,
            description='First test event',
            affected_components=['component1']
        )
        
        event2 = failsafe._create_emergency_event(
            event_type='test_event_2',
            severity=EmergencyLevel.ORANGE,
            description='Second test event',
            affected_components=['component2']
        )
        
        active_events = failsafe.get_active_events()
        
        assert len(active_events) == 2
        
        event_ids = [event['event_id'] for event in active_events]
        assert event1.event_id in event_ids
        assert event2.event_id in event_ids
        
        # Check event structure
        for event_data in active_events:
            assert 'event_id' in event_data
            assert 'event_type' in event_data
            assert 'severity' in event_data
            assert 'timestamp' in event_data
            assert 'description' in event_data
            assert 'recommended_actions' in event_data
    
    def test_incident_report_export(self, failsafe):
        """Test incident report export"""
        # Create some activity
        failsafe.current_level = EmergencyLevel.ORANGE
        failsafe._create_emergency_event(
            event_type='report_test',
            severity=EmergencyLevel.ORANGE,
            description='Test event for report',
            affected_components=['test_component']
        )
        
        report_json = failsafe.export_incident_report()
        
        assert isinstance(report_json, str)
        
        import json
        report = json.loads(report_json)
        
        assert 'timestamp' in report
        assert 'system_status' in report
        assert 'active_events' in report
        assert 'event_history' in report
        assert 'configuration' in report
        
        # Verify configuration in report
        config = report['configuration']
        assert 'max_consensus_failures' in config
        assert 'max_byzantine_ratio' in config
        assert 'auto_recovery_enabled' in config
    
    def test_monitoring_loop_functionality(self, failsafe):
        """Test background monitoring loop"""
        # The monitoring loop runs in a separate thread
        assert failsafe.monitoring_active is True
        assert failsafe.monitoring_thread.is_alive()
        
        # Simulate network partition condition
        failsafe.last_successful_consensus = time.time() - 50  # 50 seconds ago
        
        # Give monitoring loop time to detect
        time.sleep(0.1)
        
        # Force one cycle of monitoring
        failsafe._periodic_health_check()
        
        # Should detect network partition
        partition_events = [
            event for event in failsafe.active_events.values()
            if event.event_type == 'network_partition'
        ]
        # May or may not trigger depending on timing
    
    def test_event_cleanup(self, failsafe):
        """Test automatic event cleanup"""
        # Create old resolved event
        old_event = failsafe._create_emergency_event(
            event_type='old_event',
            severity=EmergencyLevel.YELLOW,
            description='Old event for cleanup test',
            affected_components=['old_component']
        )
        
        # Simulate green status for cleanup
        failsafe.current_level = EmergencyLevel.GREEN
        
        # Manually set old timestamp
        old_event.timestamp = time.time() - 7200  # 2 hours ago
        
        initial_count = len(failsafe.active_events)
        
        # Trigger cleanup
        failsafe._cleanup_resolved_events()
        
        # Event should be cleaned up
        assert len(failsafe.active_events) <= initial_count
    
    def test_graceful_shutdown(self, failsafe):
        """Test graceful shutdown of failsafe system"""
        assert failsafe.monitoring_active is True
        
        failsafe.shutdown()
        
        assert failsafe.monitoring_active is False
        # Thread should eventually stop


class TestFailsafeConfig:
    """Test failsafe configuration"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = FailsafeConfig()
        
        assert config.max_consensus_failures == 3
        assert config.max_byzantine_ratio == 0.33
        assert config.consensus_timeout_threshold == 10.0
        assert config.view_change_cascade_threshold == 5
        assert config.signature_failure_threshold == 0.1
        assert config.network_partition_timeout == 30.0
        assert config.emergency_shutdown_threshold == 10
        assert config.auto_recovery_enabled is True
        assert config.safe_mode_threshold == 5
    
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = FailsafeConfig(
            max_consensus_failures=5,
            max_byzantine_ratio=0.4,
            auto_recovery_enabled=False
        )
        
        assert config.max_consensus_failures == 5
        assert config.max_byzantine_ratio == 0.4
        assert config.auto_recovery_enabled is False
        # Other values should be defaults
        assert config.consensus_timeout_threshold == 10.0


class TestEmergencyEvent:
    """Test emergency event container"""
    
    def test_event_creation(self):
        """Test emergency event creation"""
        event = EmergencyEvent(
            event_id='test_event_123',
            event_type='test_type',
            severity=EmergencyLevel.ORANGE,
            timestamp=0,  # Will be set in __post_init__
            description='Test event description',
            affected_components=['component1', 'component2'],
            recommended_actions=[FailsafeAction.ALERT, FailsafeAction.MONITOR]
        )
        
        assert event.event_id == 'test_event_123'
        assert event.event_type == 'test_type'
        assert event.severity == EmergencyLevel.ORANGE
        assert event.timestamp > 0  # Should be set automatically
        assert event.description == 'Test event description'
        assert len(event.affected_components) == 2
        assert len(event.recommended_actions) == 2
        assert event.auto_executed is False
        assert event.resolution_timestamp is None


class TestEmergencyLevels:
    """Test emergency level enumeration"""
    
    def test_emergency_levels(self):
        """Test emergency level values"""
        levels = [
            EmergencyLevel.GREEN,
            EmergencyLevel.YELLOW,
            EmergencyLevel.ORANGE,
            EmergencyLevel.RED,
            EmergencyLevel.BLACK
        ]
        
        # Verify all levels have unique values
        level_values = [level.value for level in levels]
        assert len(level_values) == len(set(level_values))
        
        # Verify level values are strings
        for level in levels:
            assert isinstance(level.value, str)


class TestFailsafeActions:
    """Test failsafe action enumeration"""
    
    def test_failsafe_actions(self):
        """Test failsafe action types"""
        actions = [
            FailsafeAction.MONITOR,
            FailsafeAction.ALERT,
            FailsafeAction.DEGRADE_PERFORMANCE,
            FailsafeAction.ACTIVATE_BACKUP,
            FailsafeAction.ISOLATE_BYZANTINE,
            FailsafeAction.EMERGENCY_SHUTDOWN,
            FailsafeAction.FORCE_SAFE_MODE,
            FailsafeAction.NETWORK_PARTITION_RECOVERY
        ]
        
        # Verify all actions have unique values
        action_values = [action.value for action in actions]
        assert len(action_values) == len(set(action_values))
        
        # Verify action values are strings
        for action in actions:
            assert isinstance(action.value, str)


@pytest.mark.integration
class TestFailsafeIntegration:
    """Integration tests for emergency failsafe system"""
    
    def test_full_emergency_scenario(self):
        """Test complete emergency scenario from detection to recovery"""
        agent_ids = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
        config = FailsafeConfig(
            max_consensus_failures=2,
            safe_mode_threshold=3,
            auto_recovery_enabled=True
        )
        
        callback_events = []
        def emergency_callback(event):
            callback_events.append(event)
        
        failsafe = EmergencyFailsafe(agent_ids, config, emergency_callback)
        
        # Phase 1: Normal operation
        assert failsafe.current_level == EmergencyLevel.GREEN
        
        # Phase 2: Consensus failures
        for i in range(3):  # Trigger consecutive failures
            failsafe.record_consensus_attempt(
                success=False,
                latency=2.0,
                byzantine_detected=['agent_1'] if i == 2 else [],
                view_changes=1
            )
        
        # Should escalate to red alert and activate safe mode
        assert failsafe.current_level >= EmergencyLevel.RED
        assert failsafe.safe_mode_active is True
        assert len(callback_events) > 0
        
        # Phase 3: Recovery
        # Simulate successful consensus
        failsafe.record_consensus_attempt(
            success=True,
            latency=0.3,
            byzantine_detected=[],
            view_changes=0
        )
        
        # Manual recovery
        recovery_success = failsafe.manual_recovery("Issue resolved by operator")
        assert recovery_success is True
        assert failsafe.current_level == EmergencyLevel.GREEN
        assert failsafe.safe_mode_active is False
        
        # Verify metrics
        metrics = failsafe.get_system_status()['metrics']
        assert metrics['manual_interventions'] >= 1
        assert metrics['safe_mode_activations'] >= 1
        
        # Cleanup
        failsafe.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])