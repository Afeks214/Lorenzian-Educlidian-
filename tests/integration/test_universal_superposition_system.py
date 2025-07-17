"""
Comprehensive Integration Test Suite for Universal Superposition System.

This test suite validates the entire universal superposition framework by testing:
- End-to-end superposition validation and monitoring
- Performance monitoring system integration
- Cascade integrity checking across MARL systems
- Quality metrics for superposition outputs
- System integration under various conditions
- Error handling and recovery mechanisms

The tests ensure that all components work together correctly and maintain
performance targets under production conditions.
"""

import pytest
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import system components
from src.validation.universal_superposition_validator import (
    UniversalSuperpositionValidator,
    SuperpositionState,
    SuperpositionProperty,
    ValidationLevel,
    ValidationResult
)
from src.monitoring.superposition_performance_monitor import (
    SuperpositionPerformanceMonitor,
    PerformanceLevel,
    MonitoringMode,
    PerformanceMetric
)
from src.validation.cascade_integrity_checker import (
    CascadeIntegrityChecker,
    CascadeLevel,
    IntegrityStatus,
    CommunicationChannel,
    AgentNode
)
from src.monitoring.superposition_quality_metrics import (
    SuperpositionQualityMetrics,
    SuperpositionOutput,
    QualityDimension,
    QualityLevel
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUniversalSuperpositionSystem:
    """Comprehensive test suite for the universal superposition system."""
    
    @pytest.fixture
    def validator(self):
        """Create a superposition validator for testing."""
        return UniversalSuperpositionValidator(
            tolerance=1e-6,
            validation_level=ValidationLevel.STANDARD,
            performance_target_ms=5.0
        )
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        return SuperpositionPerformanceMonitor(
            target_latency_ms=5.0,
            monitoring_mode=MonitoringMode.REALTIME,
            alert_threshold_ms=4.0,
            critical_threshold_ms=8.0
        )
    
    @pytest.fixture
    def cascade_checker(self):
        """Create a cascade integrity checker for testing."""
        return CascadeIntegrityChecker(
            max_cascade_latency_ms=10.0,
            heartbeat_interval_ms=1000.0,
            dependency_timeout_ms=5000.0
        )
    
    @pytest.fixture
    def quality_metrics(self):
        """Create a quality metrics system for testing."""
        return SuperpositionQualityMetrics(
            history_size=100,
            consistency_threshold=0.8,
            coherence_threshold=0.7,
            calibration_threshold=0.75
        )
    
    @pytest.fixture
    def sample_superposition_state(self):
        """Create a sample superposition state for testing."""
        n_agents = 3
        amplitudes = np.array([0.6, 0.3, 0.1])
        phases = np.array([0.0, np.pi/4, np.pi/2])
        
        return SuperpositionState(
            amplitudes=amplitudes,
            phases=phases,
            agent_contributions={'agent1': 0.6, 'agent2': 0.3, 'agent3': 0.1},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7, 'agent3': 0.6},
            coherence_matrix=np.array([
                [1.0, 0.2, 0.1],
                [0.2, 1.0, 0.3],
                [0.1, 0.3, 1.0]
            ])
        )
    
    @pytest.fixture
    def sample_superposition_output(self):
        """Create a sample superposition output for testing."""
        return SuperpositionOutput(
            timestamp=datetime.now(),
            decision_probabilities=np.array([0.4, 0.35, 0.25]),
            agent_contributions={'agent1': 0.5, 'agent2': 0.3, 'agent3': 0.2},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7, 'agent3': 0.6},
            ensemble_confidence=0.75,
            decision_value=0.65
        )
    
    def test_validator_initialization(self, validator):
        """Test that the validator initializes correctly."""
        assert validator.tolerance == 1e-6
        assert validator.validation_level == ValidationLevel.STANDARD
        assert validator.performance_target_ms == 5.0
        assert len(validator.validation_history) == 0
        assert validator.total_tests == 0
        assert validator.passed_tests == 0
    
    def test_validator_superposition_state_validation(self, validator, sample_superposition_state):
        """Test comprehensive superposition state validation."""
        # Test validation
        results = validator.validate_superposition_state(sample_superposition_state)
        
        # Check that all required validations were performed
        expected_validations = [
            'Coherence Validation',
            'Normalization Validation',
            'Orthogonality Validation',
            'Unitarity Validation',
            'Linearity Validation'
        ]
        
        for validation in expected_validations:
            assert validation in [r.test_name for r in results.values()]
        
        # Check that results are properly structured
        for result in results.values():
            assert isinstance(result, ValidationResult)
            assert result.test_name is not None
            assert result.property_tested is not None
            assert isinstance(result.passed, bool)
            assert isinstance(result.score, float)
            assert result.performance_impact is not None
    
    def test_validator_performance_requirements(self, validator, sample_superposition_state):
        """Test that validator meets performance requirements."""
        # Perform validation and measure time
        start_time = time.perf_counter()
        results = validator.validate_superposition_state(sample_superposition_state)
        end_time = time.perf_counter()
        
        validation_time_ms = (end_time - start_time) * 1000
        
        # Should complete within performance target
        assert validation_time_ms < validator.performance_target_ms * 2  # Allow 2x for test overhead
        
        # Check that performance was tracked
        for result in results.values():
            if result.performance_impact is not None:
                assert result.performance_impact < validator.performance_target_ms
    
    def test_validator_mathematical_accuracy(self, validator):
        """Test mathematical accuracy of validation algorithms."""
        # Create a perfect superposition state
        perfect_state = SuperpositionState(
            amplitudes=np.array([1.0, 0.0, 0.0]),
            phases=np.array([0.0, 0.0, 0.0]),
            agent_contributions={'agent1': 1.0, 'agent2': 0.0, 'agent3': 0.0},
            confidence_scores={'agent1': 1.0, 'agent2': 0.0, 'agent3': 0.0},
            coherence_matrix=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
        )
        
        results = validator.validate_superposition_state(perfect_state)
        
        # Most validations should pass for perfect state
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        assert passed_count >= total_count * 0.8  # At least 80% should pass
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.target_latency_ms == 5.0
        assert performance_monitor.monitoring_mode == MonitoringMode.REALTIME
        assert performance_monitor.alert_threshold_ms == 4.0
        assert performance_monitor.critical_threshold_ms == 8.0
        assert not performance_monitor.monitoring_active
    
    def test_performance_monitor_component_registration(self, performance_monitor):
        """Test component registration in performance monitor."""
        # Register components
        performance_monitor.register_component(
            "test_component",
            expected_latency_ms=3.0,
            critical_path=True
        )
        
        # Check registration
        assert "test_component" in performance_monitor.registered_components
        comp_info = performance_monitor.registered_components["test_component"]
        assert comp_info['expected_latency_ms'] == 3.0
        assert comp_info['critical_path'] is True
    
    def test_performance_monitor_measurement(self, performance_monitor):
        """Test performance measurement functionality."""
        # Register component
        performance_monitor.register_component("test_component", 3.0)
        
        # Test measurement context manager
        with performance_monitor.measure_performance("test_component", "test_operation"):
            time.sleep(0.001)  # Simulate work
        
        # Check that metric was recorded
        assert len(performance_monitor.metrics_history) > 0
        metric = performance_monitor.metrics_history[-1]
        assert metric.component == "test_component"
        assert metric.name == "test_component_test_operation"
        assert metric.value > 0  # Should have measured some latency
    
    def test_performance_monitor_alerting(self, performance_monitor):
        """Test performance alerting system."""
        # Register component
        performance_monitor.register_component("slow_component", 2.0)
        
        # Record a slow metric
        performance_monitor.record_metric(
            component_name="slow_component",
            operation_name="slow_operation",
            latency_ms=10.0  # Exceeds critical threshold
        )
        
        # Check alerts were generated
        assert len(performance_monitor.alert_history) > 0
        
        # Check alert content
        alert = performance_monitor.alert_history[-1]
        assert "CRITICAL" in alert['message']
        assert "slow_component" in alert['message']
    
    def test_performance_monitor_real_time_monitoring(self, performance_monitor):
        """Test real-time monitoring functionality."""
        # Register component
        performance_monitor.register_component("monitored_component", 3.0)
        
        # Start monitoring
        performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        
        # Record some metrics
        for i in range(5):
            performance_monitor.record_metric(
                component_name="monitored_component",
                operation_name="operation",
                latency_ms=2.0 + i * 0.5
            )
            time.sleep(0.01)
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
        
        # Check metrics were processed
        assert len(performance_monitor.metrics_history) >= 5
    
    def test_cascade_checker_initialization(self, cascade_checker):
        """Test cascade checker initialization."""
        assert cascade_checker.max_cascade_latency_ms == 10.0
        assert cascade_checker.heartbeat_interval_ms == 1000.0
        assert cascade_checker.dependency_timeout_ms == 5000.0
        assert len(cascade_checker.agents) == 0
    
    def test_cascade_checker_agent_registration(self, cascade_checker):
        """Test agent registration in cascade checker."""
        # Register agents
        cascade_checker.register_agent(
            "strategic_agent",
            CascadeLevel.STRATEGIC,
            expected_latency_ms=5.0,
            dependencies=[]
        )
        
        cascade_checker.register_agent(
            "tactical_agent",
            CascadeLevel.TACTICAL,
            expected_latency_ms=3.0,
            dependencies=["strategic_agent"]
        )
        
        # Check registration
        assert len(cascade_checker.agents) == 2
        assert "strategic_agent" in cascade_checker.agents
        assert "tactical_agent" in cascade_checker.agents
        
        # Check dependencies
        tactical_agent = cascade_checker.agents["tactical_agent"]
        assert "strategic_agent" in tactical_agent.dependencies
        
        strategic_agent = cascade_checker.agents["strategic_agent"]
        assert "tactical_agent" in strategic_agent.dependents
    
    def test_cascade_checker_integrity_check(self, cascade_checker):
        """Test cascade integrity checking."""
        # Register agents with dependencies
        cascade_checker.register_agent(
            "agent1", CascadeLevel.STRATEGIC, 5.0, []
        )
        cascade_checker.register_agent(
            "agent2", CascadeLevel.TACTICAL, 3.0, ["agent1"]
        )
        cascade_checker.register_agent(
            "agent3", CascadeLevel.EXECUTION, 2.0, ["agent2"]
        )
        
        # Update heartbeats
        current_time = datetime.now()
        for agent_id in cascade_checker.agents:
            cascade_checker.update_agent_heartbeat(agent_id, current_time)
        
        # Perform integrity check
        report = cascade_checker.check_cascade_integrity()
        
        # Check report structure
        assert report.overall_status is not None
        assert isinstance(report.agent_statuses, dict)
        assert isinstance(report.communication_health, dict)
        assert isinstance(report.dependency_violations, list)
        assert isinstance(report.performance_issues, list)
        assert isinstance(report.recovery_suggestions, list)
        assert report.cascade_latency_ms >= 0
        assert 0 <= report.error_propagation_risk <= 1
    
    def test_cascade_checker_communication_logging(self, cascade_checker):
        """Test communication logging in cascade checker."""
        # Register agents
        cascade_checker.register_agent("agent1", CascadeLevel.STRATEGIC, 5.0)
        cascade_checker.register_agent("agent2", CascadeLevel.TACTICAL, 3.0)
        
        # Log communication
        cascade_checker.log_communication(
            from_agent="agent1",
            to_agent="agent2",
            channel=CommunicationChannel.DIRECT,
            latency_ms=2.5,
            success=True
        )
        
        # Check logging
        assert len(cascade_checker.communication_log) > 0
        
        # Check statistics
        comm_key = "agent1->agent2"
        assert comm_key in cascade_checker.communication_stats
        stats = cascade_checker.communication_stats[comm_key]
        assert stats['total_attempts'] == 1
        assert stats['successful_attempts'] == 1
        assert stats['success_rate'] == 1.0
    
    def test_quality_metrics_initialization(self, quality_metrics):
        """Test quality metrics initialization."""
        assert quality_metrics.history_size == 100
        assert quality_metrics.consistency_threshold == 0.8
        assert quality_metrics.coherence_threshold == 0.7
        assert quality_metrics.calibration_threshold == 0.75
        assert len(quality_metrics.output_history) == 0
    
    def test_quality_metrics_output_recording(self, quality_metrics, sample_superposition_output):
        """Test output recording in quality metrics."""
        # Record output
        quality_metrics.record_output(sample_superposition_output)
        
        # Check recording
        assert len(quality_metrics.output_history) == 1
        recorded_output = quality_metrics.output_history[0]
        assert recorded_output.timestamp == sample_superposition_output.timestamp
        assert np.array_equal(recorded_output.decision_probabilities, 
                             sample_superposition_output.decision_probabilities)
    
    def test_quality_metrics_assessment(self, quality_metrics):
        """Test quality assessment functionality."""
        # Record multiple outputs
        for i in range(10):
            output = SuperpositionOutput(
                timestamp=datetime.now(),
                decision_probabilities=np.array([0.4 + i*0.01, 0.35, 0.25 - i*0.01]),
                agent_contributions={'agent1': 0.5, 'agent2': 0.3, 'agent3': 0.2},
                confidence_scores={'agent1': 0.8, 'agent2': 0.7, 'agent3': 0.6},
                ensemble_confidence=0.75,
                decision_value=0.65 + i*0.01
            )
            quality_metrics.record_output(output)
        
        # Perform assessment
        report = quality_metrics.assess_quality()
        
        # Check report structure
        assert report.overall_quality is not None
        assert isinstance(report.dimension_scores, dict)
        assert isinstance(report.metrics, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.trend_analysis, dict)
        
        # Check that all dimensions were assessed
        for dimension in QualityDimension:
            assert dimension in report.dimension_scores
    
    def test_system_integration_basic(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test basic system integration."""
        # Register components in performance monitor
        performance_monitor.register_component("validator", 5.0)
        performance_monitor.register_component("cascade_checker", 10.0)
        performance_monitor.register_component("quality_metrics", 3.0)
        
        # Register agents in cascade checker
        cascade_checker.register_agent("agent1", CascadeLevel.STRATEGIC, 5.0)
        cascade_checker.register_agent("agent2", CascadeLevel.TACTICAL, 3.0, ["agent1"])
        
        # Create superposition state
        state = SuperpositionState(
            amplitudes=np.array([0.7, 0.3]),
            phases=np.array([0.0, np.pi/4]),
            agent_contributions={'agent1': 0.7, 'agent2': 0.3},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            coherence_matrix=np.array([[1.0, 0.2], [0.2, 1.0]])
        )
        
        # Test integrated workflow
        with performance_monitor.measure_performance("validator", "validation"):
            validation_results = validator.validate_superposition_state(state)
        
        with performance_monitor.measure_performance("cascade_checker", "integrity_check"):
            integrity_report = cascade_checker.check_cascade_integrity()
        
        # Create quality output
        output = SuperpositionOutput(
            timestamp=datetime.now(),
            decision_probabilities=np.array([0.7, 0.3]),
            agent_contributions={'agent1': 0.7, 'agent2': 0.3},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            ensemble_confidence=0.75,
            decision_value=0.65
        )
        
        with performance_monitor.measure_performance("quality_metrics", "assessment"):
            quality_metrics.record_output(output)
        
        # Check that all components worked
        assert len(validation_results) > 0
        assert integrity_report.overall_status is not None
        assert len(quality_metrics.output_history) == 1
        assert len(performance_monitor.metrics_history) >= 3
    
    def test_system_integration_performance_targets(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test that integrated system meets performance targets."""
        # Register components
        performance_monitor.register_component("integrated_system", 15.0)  # Total target
        
        # Create test data
        state = SuperpositionState(
            amplitudes=np.array([0.6, 0.4]),
            phases=np.array([0.0, np.pi/3]),
            agent_contributions={'agent1': 0.6, 'agent2': 0.4},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            coherence_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
        )
        
        # Register agents
        cascade_checker.register_agent("agent1", CascadeLevel.STRATEGIC, 5.0)
        cascade_checker.register_agent("agent2", CascadeLevel.TACTICAL, 3.0, ["agent1"])
        
        # Measure integrated performance
        with performance_monitor.measure_performance("integrated_system", "full_workflow"):
            # Validation
            validation_results = validator.validate_superposition_state(state)
            
            # Cascade integrity
            integrity_report = cascade_checker.check_cascade_integrity()
            
            # Quality assessment
            output = SuperpositionOutput(
                timestamp=datetime.now(),
                decision_probabilities=np.array([0.6, 0.4]),
                agent_contributions={'agent1': 0.6, 'agent2': 0.4},
                confidence_scores={'agent1': 0.8, 'agent2': 0.7},
                ensemble_confidence=0.75,
                decision_value=0.65
            )
            quality_metrics.record_output(output)
        
        # Check performance
        latest_metric = performance_monitor.metrics_history[-1]
        assert latest_metric.value < 15.0  # Should meet target
    
    def test_system_integration_error_handling(self, validator, cascade_checker):
        """Test error handling in integrated system."""
        # Test with invalid superposition state
        invalid_state = SuperpositionState(
            amplitudes=np.array([]),  # Empty array
            phases=np.array([0.0]),   # Mismatched size
            agent_contributions={'agent1': 0.5, 'agent2': 0.5},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            coherence_matrix=np.array([[1.0, 0.2], [0.2, 1.0]])
        )
        
        # Validator should handle gracefully
        results = validator.validate_superposition_state(invalid_state)
        
        # Should have structure validation failure
        assert 'structure' in results
        assert not results['structure'].passed
        
        # Test cascade checker with missing agents
        cascade_checker.register_agent("agent1", CascadeLevel.STRATEGIC, 5.0, ["nonexistent_agent"])
        
        # Should handle gracefully
        report = cascade_checker.check_cascade_integrity()
        assert len(report.dependency_violations) > 0
    
    def test_system_integration_concurrent_operations(self, validator, performance_monitor, cascade_checker):
        """Test concurrent operations in integrated system."""
        # Register components
        performance_monitor.register_component("concurrent_validator", 5.0)
        performance_monitor.register_component("concurrent_cascade", 10.0)
        
        # Register agents
        for i in range(5):
            cascade_checker.register_agent(f"agent{i}", CascadeLevel.TACTICAL, 3.0)
        
        # Create test states
        states = []
        for i in range(10):
            state = SuperpositionState(
                amplitudes=np.array([0.5 + i*0.02, 0.5 - i*0.02]),
                phases=np.array([0.0, np.pi/4 + i*0.1]),
                agent_contributions={'agent1': 0.5 + i*0.02, 'agent2': 0.5 - i*0.02},
                confidence_scores={'agent1': 0.8, 'agent2': 0.7},
                coherence_matrix=np.array([[1.0, 0.2], [0.2, 1.0]])
            )
            states.append(state)
        
        # Test concurrent validation
        def validate_state(state):
            with performance_monitor.measure_performance("concurrent_validator", "validation"):
                return validator.validate_superposition_state(state)
        
        def check_integrity():
            with performance_monitor.measure_performance("concurrent_cascade", "integrity_check"):
                return cascade_checker.check_cascade_integrity()
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit validation tasks
            validation_futures = [executor.submit(validate_state, state) for state in states]
            
            # Submit integrity checks
            integrity_futures = [executor.submit(check_integrity) for _ in range(5)]
            
            # Wait for completion
            validation_results = [future.result() for future in validation_futures]
            integrity_results = [future.result() for future in integrity_futures]
        
        # Check that all operations completed successfully
        assert len(validation_results) == 10
        assert len(integrity_results) == 5
        
        # Check that all validations have results
        for result in validation_results:
            assert len(result) > 0
        
        # Check that all integrity checks have status
        for result in integrity_results:
            assert result.overall_status is not None
    
    def test_system_integration_monitoring_workflow(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test complete monitoring workflow integration."""
        # Start monitoring
        performance_monitor.start_monitoring()
        cascade_checker.start_monitoring()
        
        try:
            # Register components and agents
            performance_monitor.register_component("workflow_validator", 5.0)
            performance_monitor.register_component("workflow_cascade", 10.0)
            performance_monitor.register_component("workflow_quality", 3.0)
            
            cascade_checker.register_agent("strategic", CascadeLevel.STRATEGIC, 5.0)
            cascade_checker.register_agent("tactical", CascadeLevel.TACTICAL, 3.0, ["strategic"])
            cascade_checker.register_agent("execution", CascadeLevel.EXECUTION, 2.0, ["tactical"])
            
            # Update heartbeats
            current_time = datetime.now()
            for agent_id in cascade_checker.agents:
                cascade_checker.update_agent_heartbeat(agent_id, current_time)
            
            # Simulate workflow over time
            for i in range(5):
                # Create state
                state = SuperpositionState(
                    amplitudes=np.array([0.6, 0.25, 0.15]),
                    phases=np.array([0.0, np.pi/4, np.pi/2]),
                    agent_contributions={'strategic': 0.6, 'tactical': 0.25, 'execution': 0.15},
                    confidence_scores={'strategic': 0.8, 'tactical': 0.7, 'execution': 0.6},
                    coherence_matrix=np.array([
                        [1.0, 0.3, 0.2],
                        [0.3, 1.0, 0.4],
                        [0.2, 0.4, 1.0]
                    ])
                )
                
                # Validate
                with performance_monitor.measure_performance("workflow_validator", "validation"):
                    validation_results = validator.validate_superposition_state(state)
                
                # Record cascade performance
                cascade_checker.record_agent_performance("strategic", 4.5)
                cascade_checker.record_agent_performance("tactical", 2.8)
                cascade_checker.record_agent_performance("execution", 1.9)
                
                # Create quality output
                output = SuperpositionOutput(
                    timestamp=datetime.now(),
                    decision_probabilities=np.array([0.6, 0.25, 0.15]),
                    agent_contributions={'strategic': 0.6, 'tactical': 0.25, 'execution': 0.15},
                    confidence_scores={'strategic': 0.8, 'tactical': 0.7, 'execution': 0.6},
                    ensemble_confidence=0.73,
                    decision_value=0.62
                )
                
                with performance_monitor.measure_performance("workflow_quality", "assessment"):
                    quality_metrics.record_output(output)
                
                # Log some communication
                cascade_checker.log_communication(
                    from_agent="strategic",
                    to_agent="tactical",
                    channel=CommunicationChannel.HIERARCHICAL,
                    latency_ms=2.0,
                    success=True
                )
                
                time.sleep(0.1)  # Simulate time passage
            
            # Allow monitoring to process
            time.sleep(0.2)
            
            # Check results
            assert len(performance_monitor.metrics_history) > 0
            assert len(cascade_checker.communication_log) > 0
            assert len(quality_metrics.output_history) == 5
            
            # Get summaries
            perf_summary = performance_monitor.get_performance_summary()
            cascade_topology = cascade_checker.get_cascade_topology()
            quality_summary = quality_metrics.get_quality_summary()
            
            assert perf_summary['monitoring_active'] is True
            assert cascade_topology['nodes'] == 3
            assert quality_summary['status'] == 'active'
            
        finally:
            # Stop monitoring
            performance_monitor.stop_monitoring()
            cascade_checker.stop_monitoring()
    
    def test_system_integration_stress_test(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test system under stress conditions."""
        # Register many components
        for i in range(20):
            performance_monitor.register_component(f"stress_component_{i}", 5.0)
            cascade_checker.register_agent(f"stress_agent_{i}", CascadeLevel.TACTICAL, 3.0)
        
        # Create many states
        states = []
        for i in range(100):
            state = SuperpositionState(
                amplitudes=np.random.dirichlet([1, 1, 1]),
                phases=np.random.uniform(0, 2*np.pi, 3),
                agent_contributions={f'agent_{j}': np.random.random()/3 for j in range(3)},
                confidence_scores={f'agent_{j}': np.random.uniform(0.5, 1.0) for j in range(3)},
                coherence_matrix=np.random.random((3, 3)) * 0.5 + np.eye(3) * 0.5
            )
            states.append(state)
        
        # Process all states
        start_time = time.perf_counter()
        
        for i, state in enumerate(states):
            with performance_monitor.measure_performance(f"stress_component_{i%20}", "stress_test"):
                # Validation
                validation_results = validator.validate_superposition_state(state)
                
                # Quality assessment
                output = SuperpositionOutput(
                    timestamp=datetime.now(),
                    decision_probabilities=state.amplitudes,
                    agent_contributions=state.agent_contributions,
                    confidence_scores=state.confidence_scores,
                    ensemble_confidence=np.mean(list(state.confidence_scores.values())),
                    decision_value=np.mean(state.amplitudes)
                )
                quality_metrics.record_output(output)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Check performance under stress
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(performance_monitor.metrics_history) == 100
        assert len(quality_metrics.output_history) == 100
        
        # Check that system maintained quality
        perf_summary = performance_monitor.get_performance_summary()
        assert perf_summary['alert_summary']['critical_alerts_last_hour'] < 10  # Limited critical alerts
    
    def test_system_integration_data_export(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test data export functionality across all components."""
        # Set up system
        performance_monitor.register_component("export_test", 5.0)
        cascade_checker.register_agent("export_agent", CascadeLevel.TACTICAL, 3.0)
        
        # Generate some data
        state = SuperpositionState(
            amplitudes=np.array([0.7, 0.3]),
            phases=np.array([0.0, np.pi/4]),
            agent_contributions={'agent1': 0.7, 'agent2': 0.3},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            coherence_matrix=np.array([[1.0, 0.2], [0.2, 1.0]])
        )
        
        # Run operations
        validation_results = validator.validate_superposition_state(state)
        
        with performance_monitor.measure_performance("export_test", "operation"):
            time.sleep(0.001)
        
        output = SuperpositionOutput(
            timestamp=datetime.now(),
            decision_probabilities=np.array([0.7, 0.3]),
            agent_contributions={'agent1': 0.7, 'agent2': 0.3},
            confidence_scores={'agent1': 0.8, 'agent2': 0.7},
            ensemble_confidence=0.75,
            decision_value=0.65
        )
        quality_metrics.record_output(output)
        
        # Test exports
        validator_data = validator.export_validation_data()
        performance_data = performance_monitor.export_metrics()
        cascade_data = cascade_checker.export_cascade_data()
        quality_data = quality_metrics.export_quality_data()
        
        # Check export structure
        assert 'summary' in validator_data
        assert 'history' in validator_data
        assert 'configuration' in validator_data
        
        assert 'summary' in performance_data
        assert 'system_snapshot' in performance_data
        
        assert 'agents' in cascade_data
        assert 'cascade_topology' in cascade_data
        
        assert 'quality_reports' in quality_data
        assert 'configuration' in quality_data
        
        # Check data integrity
        assert len(validator_data['history']) > 0
        assert performance_data['summary']['total_metrics'] > 0
        assert cascade_data['cascade_topology']['nodes'] == 1
        assert len(quality_data['quality_reports']) > 0
    
    def test_system_integration_full_workflow_validation(self, validator, performance_monitor, cascade_checker, quality_metrics):
        """Test complete workflow validation with all components."""
        # This is the ultimate integration test
        
        # Setup complete system
        performance_monitor.register_component("full_validator", 5.0, critical_path=True)
        performance_monitor.register_component("full_cascade", 10.0, critical_path=True)
        performance_monitor.register_component("full_quality", 3.0)
        
        cascade_checker.register_agent("strategic", CascadeLevel.STRATEGIC, 5.0)
        cascade_checker.register_agent("tactical", CascadeLevel.TACTICAL, 3.0, ["strategic"])
        cascade_checker.register_agent("execution", CascadeLevel.EXECUTION, 2.0, ["tactical"])
        cascade_checker.register_agent("risk", CascadeLevel.RISK_MANAGEMENT, 4.0, ["strategic", "tactical"])
        
        # Start monitoring
        performance_monitor.start_monitoring()
        cascade_checker.start_monitoring()
        
        try:
            # Create realistic superposition state
            state = SuperpositionState(
                amplitudes=np.array([0.5, 0.3, 0.15, 0.05]),
                phases=np.array([0.0, np.pi/6, np.pi/3, np.pi/2]),
                agent_contributions={
                    'strategic': 0.5,
                    'tactical': 0.3,
                    'execution': 0.15,
                    'risk': 0.05
                },
                confidence_scores={
                    'strategic': 0.85,
                    'tactical': 0.75,
                    'execution': 0.65,
                    'risk': 0.90
                },
                coherence_matrix=np.array([
                    [1.0, 0.4, 0.2, 0.3],
                    [0.4, 1.0, 0.5, 0.2],
                    [0.2, 0.5, 1.0, 0.1],
                    [0.3, 0.2, 0.1, 1.0]
                ])
            )
            
            # Update heartbeats
            current_time = datetime.now()
            for agent_id in cascade_checker.agents:
                cascade_checker.update_agent_heartbeat(agent_id, current_time)
            
            # Full workflow execution
            workflow_start = time.perf_counter()
            
            # Step 1: Validate superposition
            with performance_monitor.measure_performance("full_validator", "validation"):
                validation_results = validator.validate_superposition_state(state, ValidationLevel.COMPREHENSIVE)
            
            # Step 2: Check cascade integrity
            with performance_monitor.measure_performance("full_cascade", "integrity_check"):
                integrity_report = cascade_checker.check_cascade_integrity()
            
            # Step 3: Assess quality
            output = SuperpositionOutput(
                timestamp=datetime.now(),
                decision_probabilities=state.amplitudes,
                agent_contributions=state.agent_contributions,
                confidence_scores=state.confidence_scores,
                ensemble_confidence=np.mean(list(state.confidence_scores.values())),
                decision_value=np.sum(state.amplitudes * np.array([0.8, 0.6, 0.4, 0.9]))
            )
            
            with performance_monitor.measure_performance("full_quality", "assessment"):
                quality_metrics.record_output(output)
                quality_report = quality_metrics.assess_quality()
            
            workflow_end = time.perf_counter()
            workflow_time = (workflow_end - workflow_start) * 1000
            
            # Validate complete workflow
            
            # 1. Performance requirements
            assert workflow_time < 20.0  # Complete workflow under 20ms
            
            # 2. Validation results
            assert len(validation_results) > 0
            validation_passed = sum(1 for r in validation_results.values() if r.passed)
            assert validation_passed >= len(validation_results) * 0.7  # 70% validation pass rate
            
            # 3. Cascade integrity
            assert integrity_report.overall_status in [IntegrityStatus.HEALTHY, IntegrityStatus.DEGRADED]
            assert integrity_report.cascade_latency_ms < cascade_checker.max_cascade_latency_ms
            
            # 4. Quality assessment
            assert quality_report.overall_quality in [QualityLevel.ACCEPTABLE, QualityLevel.GOOD, QualityLevel.EXCELLENT]
            assert len(quality_report.metrics) > 0
            
            # 5. System monitoring
            perf_summary = performance_monitor.get_performance_summary()
            assert perf_summary['monitoring_active'] is True
            assert len(perf_summary['component_statistics']) == 3
            
            # 6. Integration consistency
            # Check that all components report consistent timing
            validator_metrics = [m for m in performance_monitor.metrics_history if m.component == "full_validator"]
            cascade_metrics = [m for m in performance_monitor.metrics_history if m.component == "full_cascade"]
            quality_metrics_perf = [m for m in performance_monitor.metrics_history if m.component == "full_quality"]
            
            assert len(validator_metrics) > 0
            assert len(cascade_metrics) > 0
            assert len(quality_metrics_perf) > 0
            
            # All individual components should meet their targets
            assert all(m.value < 6.0 for m in validator_metrics)  # Validator target + margin
            assert all(m.value < 11.0 for m in cascade_metrics)  # Cascade target + margin
            assert all(m.value < 4.0 for m in quality_metrics_perf)  # Quality target + margin
            
            # Generate final report
            final_report = {
                'workflow_time_ms': workflow_time,
                'validation_results': len(validation_results),
                'validation_pass_rate': validation_passed / len(validation_results),
                'cascade_status': integrity_report.overall_status.value,
                'cascade_latency_ms': integrity_report.cascade_latency_ms,
                'quality_level': quality_report.overall_quality.value,
                'performance_summary': perf_summary,
                'success': True
            }
            
            logger.info(f"Full workflow validation completed successfully: {final_report}")
            
        finally:
            # Clean up
            performance_monitor.stop_monitoring()
            cascade_checker.stop_monitoring()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])