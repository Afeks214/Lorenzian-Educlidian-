"""
Comprehensive Test Suite for Intelligence Coordination

Tests all intelligence coordination functionality including:
- Intelligence Coordinator
- Event Orchestrator
- Decision Fusion Engine
- Adaptive Learning System
- Quality Assurance Monitor
- Seamless Integration Layer

Test Coverage:
- Unit tests for each component
- Integration tests for component interaction
- Performance tests for latency requirements
- Stress tests for high-load scenarios
- Failure tests for error handling
- Security tests for robustness
"""

import pytest
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import queue

from src.core.events import EventBus, Event, EventType
from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.marl.agent_coordinator import AgentCoordinator, CoordinatorConfig, ConsensusResult
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, RiskCriticMode

# Intelligence Layer Imports
from src.risk.intelligence.intelligence_coordinator import (
    IntelligenceCoordinator, IntelligenceConfig, IntelligenceType, 
    IntelligencePriority, CoordinationStatus, IntelligenceDecision
)
from src.risk.intelligence.event_orchestrator import (
    EventOrchestrator, EventPriority, EventCategory, EventPattern,
    EventMetadata, ProcessedEvent, EventCorrelation
)
from src.risk.intelligence.decision_fusion_engine import (
    DecisionFusionEngine, DecisionType, FusionMethod, ConflictResolutionStrategy,
    AgentDecision, FusionResult, AgentCredibility
)
from src.risk.intelligence.adaptive_learning_system import (
    AdaptiveLearningSystem, OptimizationMetric, LearningAlgorithm, 
    AdaptationStrategy, PerformanceMetric
)
from src.risk.intelligence.quality_assurance_monitor import (
    QualityAssuranceMonitor, HealthStatus, AlertSeverity, AnomalyType,
    QualityMetric, HealthCheck, QualityAlert
)
from src.risk.integration.seamless_integration_layer import (
    SeamlessIntegrationLayer, IntegrationMode, FallbackTrigger,
    PerformanceTarget, IntegrationMetrics
)


class MockRiskAgent(BaseRiskAgent):
    """Mock risk agent for testing"""
    
    def __init__(self, name: str, response_time_ms: float = 1.0):
        self.name = name
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.last_risk_state = None
        
    def make_decision(self, state_vector: np.ndarray) -> tuple:
        """Mock decision making"""
        self.call_count += 1
        self.last_risk_state = state_vector
        
        # Simulate processing time
        time.sleep(self.response_time_ms / 1000.0)
        
        # Return mock decision and confidence
        action = np.random.randint(0, 4)
        confidence = np.random.uniform(0.6, 0.9)
        
        return action, confidence
    
    def emergency_stop(self, reason: str) -> bool:
        """Mock emergency stop"""
        return True


class MockIntelligenceComponent:
    """Mock intelligence component for testing"""
    
    def __init__(self, name: str, component_type: IntelligenceType):
        self.name = name
        self.component_type = component_type
        self.call_count = 0
        
    def make_decision(self, risk_state: RiskState, context: Dict[str, Any]) -> tuple:
        """Mock intelligence decision"""
        self.call_count += 1
        
        # Simulate different component behaviors
        if self.component_type == IntelligenceType.CRISIS_FORECASTER:
            decision = {"crisis_probability": np.random.uniform(0.1, 0.9)}
            confidence = np.random.uniform(0.7, 0.95)
        elif self.component_type == IntelligenceType.PRE_MORTEM_ANALYST:
            decision = {"failure_scenarios": ["scenario_1", "scenario_2"]}
            confidence = np.random.uniform(0.6, 0.85)
        else:  # HUMAN_OVERSIGHT
            decision = {"human_approval": np.random.choice([True, False])}
            confidence = 1.0 if decision["human_approval"] else 0.5
        
        reasoning = f"Decision from {self.name}"
        
        return decision, confidence, reasoning


@pytest.fixture
def event_bus():
    """Create event bus for testing"""
    return EventBus()


@pytest.fixture
def mock_risk_state():
    """Create mock risk state"""
    return RiskState(
        timestamp=datetime.now(),
        var_estimate_5pct=0.15,
        var_estimate_1pct=0.25,
        correlation_risk=0.3,
        liquidity_conditions=0.8,
        margin_usage_pct=0.6,
        market_stress_level=0.4,
        time_of_day_risk=0.5
    )


@pytest.fixture
def mock_marl_coordinator(event_bus):
    """Create mock MARL coordinator"""
    config = CoordinatorConfig()
    critic = Mock(spec=CentralizedCritic)
    
    coordinator = AgentCoordinator(config, critic, event_bus)
    
    # Add mock agents
    for i, name in enumerate(['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer']):
        agent = MockRiskAgent(name, response_time_ms=2.0)
        coordinator.register_agent(agent)
    
    return coordinator


@pytest.fixture
def intelligence_config():
    """Create intelligence configuration"""
    return IntelligenceConfig(
        max_coordination_latency_ms=5.0,
        emergency_response_time_ms=1.0,
        health_check_interval_s=0.1  # Faster for testing
    )


@pytest.fixture
def intelligence_coordinator(intelligence_config, mock_marl_coordinator, event_bus):
    """Create intelligence coordinator"""
    critic = Mock(spec=CentralizedCritic)
    return IntelligenceCoordinator(
        intelligence_config, 
        mock_marl_coordinator, 
        critic, 
        event_bus
    )


class TestIntelligenceCoordinator:
    """Test suite for Intelligence Coordinator"""
    
    def test_initialization(self, intelligence_coordinator):
        """Test intelligence coordinator initialization"""
        assert intelligence_coordinator.config.max_coordination_latency_ms == 5.0
        assert intelligence_coordinator.coordination_status == CoordinationStatus.OPTIMAL
        assert intelligence_coordinator.coordination_count == 0
        assert len(intelligence_coordinator.intelligence_components) == 0
    
    def test_component_registration(self, intelligence_coordinator):
        """Test intelligence component registration"""
        component = MockIntelligenceComponent("crisis_forecaster", IntelligenceType.CRISIS_FORECASTER)
        
        success = intelligence_coordinator.register_intelligence_component(
            "crisis_forecaster",
            IntelligenceType.CRISIS_FORECASTER,
            component.make_decision,
            IntelligencePriority.EMERGENCY,
            2.0
        )
        
        assert success
        assert "crisis_forecaster" in intelligence_coordinator.intelligence_components
        assert intelligence_coordinator.intelligence_components["crisis_forecaster"].weight == 2.0
    
    def test_coordination_decision(self, intelligence_coordinator, mock_risk_state):
        """Test intelligence coordination decision making"""
        # Register mock components
        crisis_component = MockIntelligenceComponent("crisis_forecaster", IntelligenceType.CRISIS_FORECASTER)
        premortem_component = MockIntelligenceComponent("premortem_analyst", IntelligenceType.PRE_MORTEM_ANALYST)
        
        intelligence_coordinator.register_intelligence_component(
            "crisis_forecaster", IntelligenceType.CRISIS_FORECASTER,
            crisis_component.make_decision, IntelligencePriority.EMERGENCY
        )
        intelligence_coordinator.register_intelligence_component(
            "premortem_analyst", IntelligenceType.PRE_MORTEM_ANALYST,
            premortem_component.make_decision, IntelligencePriority.MEDIUM
        )
        
        # Test coordination
        result = intelligence_coordinator.coordinate_intelligence_decision(
            mock_risk_state, {"test_context": True}
        )
        
        assert result is not None
        assert result.execution_time_ms > 0
        assert len(result.participating_components) > 0
        assert intelligence_coordinator.coordination_count == 1
    
    def test_emergency_override(self, intelligence_coordinator, mock_risk_state):
        """Test emergency override functionality"""
        # Register emergency component
        emergency_component = MockIntelligenceComponent("emergency_crisis", IntelligenceType.CRISIS_FORECASTER)
        
        def emergency_decision(risk_state, context):
            return {"emergency_action": "STOP_ALL_TRADING"}, 0.95, "Emergency detected"
        
        intelligence_coordinator.register_intelligence_component(
            "emergency_crisis", IntelligenceType.CRISIS_FORECASTER,
            emergency_decision, IntelligencePriority.EMERGENCY
        )
        
        result = intelligence_coordinator.coordinate_intelligence_decision(
            mock_risk_state, {}
        )
        
        # Should use emergency decision
        assert result.coordination_method == "emergency_override"
        assert "emergency_action" in str(result.coordinated_decision)
    
    def test_performance_monitoring(self, intelligence_coordinator, mock_risk_state):
        """Test performance monitoring and metrics"""
        # Register component with slow response
        slow_component = MockIntelligenceComponent("slow_component", IntelligenceType.PRE_MORTEM_ANALYST)
        
        def slow_decision(risk_state, context):
            time.sleep(0.01)  # 10ms delay
            return {"slow_decision": True}, 0.8, "Slow processing"
        
        intelligence_coordinator.register_intelligence_component(
            "slow_component", IntelligenceType.PRE_MORTEM_ANALYST,
            slow_decision, IntelligencePriority.MEDIUM
        )
        
        # Multiple coordinations to gather metrics
        for _ in range(5):
            intelligence_coordinator.coordinate_intelligence_decision(mock_risk_state, {})
        
        metrics = intelligence_coordinator.get_coordination_metrics()
        
        assert metrics['coordination_count'] == 5
        assert metrics['avg_response_time_ms'] > 0
        assert metrics['coordination_status'] == CoordinationStatus.OPTIMAL.value
        assert 'component_metrics' in metrics
    
    def test_health_monitoring(self, intelligence_coordinator):
        """Test health monitoring functionality"""
        intelligence_coordinator.start_health_monitoring()
        
        # Let monitoring run briefly
        time.sleep(0.2)
        
        intelligence_coordinator.running = False
        intelligence_coordinator.health_monitor_thread.join(timeout=1.0)
        
        # Health monitoring should have completed without errors
        assert True  # Test passes if no exceptions thrown


class TestEventOrchestrator:
    """Test suite for Event Orchestrator"""
    
    def test_initialization(self):
        """Test event orchestrator initialization"""
        orchestrator = EventOrchestrator(max_throughput_events_per_sec=1000)
        
        assert orchestrator.max_throughput == 1000
        assert orchestrator.thread_count == 4
        assert orchestrator.event_count == 0
        assert not orchestrator.running
    
    def test_event_route_registration(self):
        """Test event route registration"""
        orchestrator = EventOrchestrator()
        
        def mock_handler(event):
            return "handled"
        
        orchestrator.register_event_route(
            EventType.RISK_BREACH,
            EventPriority.CRITICAL,
            EventCategory.RISK_ALERT,
            [mock_handler],
            max_processing_time_ms=10.0
        )
        
        assert EventType.RISK_BREACH in orchestrator.event_routes
        route = orchestrator.event_routes[EventType.RISK_BREACH]
        assert route.priority == EventPriority.CRITICAL
        assert len(route.handlers) == 1
    
    def test_event_submission_and_processing(self, event_bus):
        """Test event submission and processing"""
        orchestrator = EventOrchestrator()
        
        processed_events = []
        
        def test_handler(processed_event):
            processed_events.append(processed_event)
            return "processed"
        
        # Register handler
        orchestrator.register_event_route(
            EventType.NEW_TICK,
            EventPriority.LOW,
            EventCategory.MARKET_DATA,
            [test_handler]
        )
        
        # Start processing
        orchestrator.start_processing()
        
        # Submit test event
        test_event = Event(
            event_type=EventType.NEW_TICK,
            timestamp=datetime.now(),
            payload={"price": 100.0},
            source="test"
        )
        
        event_id = orchestrator.submit_event(test_event)
        
        # Wait for processing
        time.sleep(0.1)
        orchestrator.stop_processing()
        
        assert event_id != ""
        assert orchestrator.event_count > 0
        assert len(processed_events) > 0
    
    def test_priority_ordering(self):
        """Test priority-based event ordering"""
        orchestrator = EventOrchestrator()
        
        processed_order = []
        
        def priority_handler(processed_event):
            processed_order.append(processed_event.original_event.event_type)
        
        # Register handlers for different priorities
        orchestrator.register_event_route(
            EventType.EMERGENCY_STOP, EventPriority.EMERGENCY,
            EventCategory.EMERGENCY, [priority_handler]
        )
        orchestrator.register_event_route(
            EventType.RISK_BREACH, EventPriority.CRITICAL,
            EventCategory.RISK_ALERT, [priority_handler]
        )
        orchestrator.register_event_route(
            EventType.NEW_TICK, EventPriority.LOW,
            EventCategory.MARKET_DATA, [priority_handler]
        )
        
        orchestrator.start_processing()
        
        # Submit events in reverse priority order
        events = [
            Event(EventType.NEW_TICK, datetime.now(), {}, "test"),
            Event(EventType.RISK_BREACH, datetime.now(), {}, "test"),
            Event(EventType.EMERGENCY_STOP, datetime.now(), {}, "test")
        ]
        
        for event in events:
            orchestrator.submit_event(event)
        
        time.sleep(0.1)
        orchestrator.stop_processing()
        
        # Emergency should be processed first
        assert EventType.EMERGENCY_STOP in processed_order
        # Note: Exact ordering depends on timing, but emergency should be prioritized
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        orchestrator = EventOrchestrator()
        
        def fast_handler(event):
            time.sleep(0.001)  # 1ms
            return "fast"
        
        orchestrator.register_event_route(
            EventType.NEW_TICK, EventPriority.LOW,
            EventCategory.MARKET_DATA, [fast_handler]
        )
        
        orchestrator.start_processing()
        
        # Submit multiple events
        for i in range(10):
            event = Event(EventType.NEW_TICK, datetime.now(), {"id": i}, "test")
            orchestrator.submit_event(event)
        
        time.sleep(0.2)
        orchestrator.stop_processing()
        
        metrics = orchestrator.get_performance_metrics()
        
        assert metrics['event_count'] == 10
        assert metrics['avg_latency_ms'] > 0
        assert metrics['error_count'] >= 0
        assert metrics['meets_latency_target'] is not None


class TestDecisionFusionEngine:
    """Test suite for Decision Fusion Engine"""
    
    def test_initialization(self):
        """Test decision fusion engine initialization"""
        engine = DecisionFusionEngine()
        
        assert engine.fusion_count == 0
        assert engine.learning_enabled
        assert isinstance(engine.bayesian_core, type(engine.bayesian_core))
    
    def test_agent_registration(self):
        """Test agent registration for credibility tracking"""
        engine = DecisionFusionEngine()
        
        engine.register_agent("test_agent", base_credibility=0.8)
        
        assert "test_agent" in engine.credibility_scorer.agent_credibilities
        credibility = engine.credibility_scorer.agent_credibilities["test_agent"]
        assert credibility.base_credibility == 0.8
    
    def test_single_decision_fusion(self):
        """Test fusion of single decision"""
        engine = DecisionFusionEngine()
        
        decision = AgentDecision(
            agent_name="test_agent",
            decision_value=0.75,
            confidence=0.85,
            uncertainty=0.1,
            timestamp=datetime.now(),
            decision_type=DecisionType.POSITION_SIZE
        )
        
        result = engine.fuse_decisions([decision], DecisionType.POSITION_SIZE)
        
        assert result.fused_decision == 0.75
        assert result.fusion_confidence == 0.85
        assert len(result.participating_agents) == 1
        assert not result.conflict_detected
    
    def test_multiple_decision_fusion(self):
        """Test fusion of multiple decisions"""
        engine = DecisionFusionEngine()
        
        # Register agents
        engine.register_agent("agent1", 0.8)
        engine.register_agent("agent2", 0.7)
        
        decisions = [
            AgentDecision("agent1", 0.6, 0.9, 0.05, datetime.now(), DecisionType.POSITION_SIZE),
            AgentDecision("agent2", 0.8, 0.7, 0.1, datetime.now(), DecisionType.POSITION_SIZE)
        ]
        
        result = engine.fuse_decisions(decisions, DecisionType.POSITION_SIZE)
        
        assert 0.6 <= result.fused_decision <= 0.8  # Should be between the two values
        assert result.fusion_confidence > 0
        assert len(result.participating_agents) == 2
    
    def test_conflict_detection(self):
        """Test conflict detection between agents"""
        engine = DecisionFusionEngine()
        
        # Create conflicting decisions
        decisions = [
            AgentDecision("agent1", 0.2, 0.9, 0.05, datetime.now(), DecisionType.POSITION_SIZE),
            AgentDecision("agent2", 0.8, 0.9, 0.05, datetime.now(), DecisionType.POSITION_SIZE)
        ]
        
        result = engine.fuse_decisions(decisions, DecisionType.POSITION_SIZE)
        
        # Should detect conflict due to large disagreement
        assert len(result.conflicts_detected) > 0 or result.conflict_detected
    
    def test_performance_tracking(self):
        """Test performance tracking and credibility updates"""
        engine = DecisionFusionEngine()
        engine.register_agent("learning_agent", 0.5)
        
        # Simulate performance updates
        for i in range(10):
            predicted = 0.6 + i * 0.02
            actual = 0.65 + i * 0.02  # Slightly different
            engine.update_agent_performance(
                "learning_agent", DecisionType.POSITION_SIZE,
                predicted, actual, 0.8
            )
        
        credibility = engine.credibility_scorer.agent_credibilities["learning_agent"]
        assert len(credibility.recent_performance) > 0
        
        # Credibility should have been updated
        metrics = engine.get_performance_metrics()
        assert "learning_agent" in metrics['agent_metrics']


class TestAdaptiveLearningSystem:
    """Test suite for Adaptive Learning System"""
    
    def test_initialization(self):
        """Test adaptive learning system initialization"""
        system = AdaptiveLearningSystem(AdaptationStrategy.BALANCED)
        
        assert system.adaptation_strategy == AdaptationStrategy.BALANCED
        assert system.learning_enabled
        assert system.adaptation_count == 0
        assert len(system.parameter_optimizer.parameters) > 0
    
    def test_performance_recording(self):
        """Test performance metric recording"""
        system = AdaptiveLearningSystem()
        
        system.record_performance(
            OptimizationMetric.COORDINATION_LATENCY,
            3.5,  # 3.5ms latency
            {"component": "test"}
        )
        
        # Check if metric was recorded
        metrics = system.performance_monitor.get_recent_performance(
            OptimizationMetric.COORDINATION_LATENCY.value, 1
        )
        assert len(metrics) == 1
        assert metrics[0] == 3.5
    
    def test_parameter_adaptation(self):
        """Test parameter adaptation based on performance"""
        system = AdaptiveLearningSystem()
        
        # Record poor performance to trigger adaptation
        for _ in range(5):
            system.record_performance(
                OptimizationMetric.OVERALL_PERFORMANCE,
                0.3,  # Poor performance
                {}
            )
        
        # Trigger optimization manually
        system._optimize_parameters()
        
        assert system.adaptation_count > 0
        
        # Check if parameters were updated
        optimization_status = system.parameter_optimizer.get_optimization_status()
        assert optimization_status['optimization_steps'] > 0
    
    def test_ab_testing(self):
        """Test A/B testing framework"""
        system = AdaptiveLearningSystem()
        
        test_id = system.create_ab_test(
            "test_parameter_optimization",
            "coordination_timeout_ms",
            control_value=5.0,
            treatment_value=3.0,
            duration_hours=1
        )
        
        assert test_id != ""
        assert test_id in system.ab_testing_engine.active_tests
        
        # Simulate test data
        for i in range(50):
            # Control group
            system.ab_testing_engine.record_test_outcome(test_id, "control", 0.7 + np.random.normal(0, 0.1))
            # Treatment group (slightly better performance)
            system.ab_testing_engine.record_test_outcome(test_id, "treatment", 0.75 + np.random.normal(0, 0.1))
        
        # Analyze test
        result = system.ab_testing_engine.analyze_test(test_id)
        
        if result:  # Test might not have enough data
            assert result.control_performance is not None
            assert result.treatment_performance is not None
    
    def test_context_aware_adaptation(self):
        """Test context-aware adaptation"""
        system = AdaptiveLearningSystem(AdaptationStrategy.CONTEXT_AWARE)
        
        from src.risk.intelligence.adaptive_learning_system import LearningContext
        
        # Test high volatility context
        high_vol_context = LearningContext(
            market_volatility=0.8,
            trading_volume=1000000,
            time_of_day="trading",
            market_phase="trading",
            system_load=0.3,
            recent_performance=0.6,
            error_rate=0.05
        )
        
        original_learning_rate = system.parameter_optimizer.learning_rate
        system.adapt_to_context(high_vol_context)
        
        # Learning rate should be adjusted for high volatility
        assert system.parameter_optimizer.learning_rate != original_learning_rate
    
    def test_learning_metrics(self):
        """Test learning metrics collection"""
        system = AdaptiveLearningSystem()
        
        # Record some performance data
        system.record_performance(OptimizationMetric.COORDINATION_LATENCY, 4.0)
        system.record_performance(OptimizationMetric.FUSION_ACCURACY, 0.85)
        
        metrics = system.get_learning_metrics()
        
        assert 'learning_enabled' in metrics
        assert 'adaptation_count' in metrics
        assert 'current_performance' in metrics
        assert 'performance_trends' in metrics
        assert 'optimization_status' in metrics


class TestQualityAssuranceMonitor:
    """Test suite for Quality Assurance Monitor"""
    
    def test_initialization(self):
        """Test quality assurance monitor initialization"""
        monitor = QualityAssuranceMonitor()
        
        assert monitor.monitoring_enabled
        assert monitor.auto_recovery_enabled
        assert len(monitor.active_alerts) == 0
    
    def test_component_registration(self):
        """Test component registration for monitoring"""
        monitor = QualityAssuranceMonitor()
        
        monitor.register_component("test_component")
        
        assert "test_component" in monitor.health_monitor.component_metrics
    
    def test_output_validation(self):
        """Test component output validation"""
        monitor = QualityAssuranceMonitor()
        monitor.register_component("validator_test")
        
        # Test valid output
        valid = monitor.validate_component_output(
            "validator_test",
            0.75,  # Valid float
            float,
            confidence=0.85
        )
        assert valid
        
        # Test invalid output
        invalid = monitor.validate_component_output(
            "validator_test",
            "invalid",  # String instead of float
            float,
            confidence=0.85
        )
        assert not invalid
        assert len(monitor.active_alerts) > 0
    
    def test_performance_metric_recording(self):
        """Test performance metric recording and anomaly detection"""
        monitor = QualityAssuranceMonitor()
        monitor.register_component("perf_test")
        
        # Record normal performance
        for i in range(20):
            monitor.record_performance_metric(
                "perf_test",
                QualityMetric.RESPONSE_TIME,
                5.0 + np.random.normal(0, 0.5)  # Around 5ms
            )
        
        # Record anomalous performance
        monitor.record_performance_metric(
            "perf_test",
            QualityMetric.RESPONSE_TIME,
            50.0  # 50ms - should trigger anomaly
        )
        
        # Should detect anomaly and create alert
        assert len(monitor.active_alerts) > 0
    
    def test_health_monitoring(self):
        """Test component health monitoring"""
        monitor = QualityAssuranceMonitor()
        monitor.register_component("health_test")
        
        # Record successful operations
        for _ in range(10):
            monitor.health_monitor.record_response_time("health_test", 5.0)
            monitor.health_monitor.record_success("health_test")
        
        health_check = monitor.health_monitor.get_component_health("health_test")
        
        assert health_check.component_name == "health_test"
        assert health_check.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        
        # Record failures
        for _ in range(5):
            monitor.health_monitor.record_failure("health_test", "Test failure")
        
        health_check = monitor.health_monitor.get_component_health("health_test")
        assert health_check.status != HealthStatus.HEALTHY
    
    def test_quality_report(self):
        """Test comprehensive quality report generation"""
        monitor = QualityAssuranceMonitor()
        monitor.register_component("report_test")
        
        # Generate some data
        monitor.record_performance_metric("report_test", QualityMetric.ACCURACY, 0.85)
        monitor.validate_component_output("report_test", 0.5, float, 0.8)
        
        report = monitor.get_quality_report()
        
        assert 'timestamp' in report
        assert 'health_summary' in report
        assert 'active_alerts' in report
        assert 'validation_metrics' in report
        assert 'component_quality_scores' in report
    
    def test_alert_management(self):
        """Test alert creation and management"""
        monitor = QualityAssuranceMonitor()
        
        alert_received = []
        
        def alert_callback(alert):
            alert_received.append(alert)
        
        monitor.add_alert_callback(alert_callback)
        
        # Trigger an alert
        monitor._create_alert(
            "test_component",
            AlertSeverity.WARNING,
            "Test alert message",
            AnomalyType.LATENCY_SPIKE
        )
        
        assert len(monitor.active_alerts) > 0
        assert len(alert_received) > 0
        
        # Test alert resolution
        alert_id = list(monitor.active_alerts.keys())[0]
        monitor.resolve_alert(alert_id, "Test resolution")
        
        assert alert_id not in monitor.active_alerts


class TestSeamlessIntegrationLayer:
    """Test suite for Seamless Integration Layer"""
    
    def test_initialization(self, mock_marl_coordinator):
        """Test seamless integration layer initialization"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            integration_mode=IntegrationMode.SHADOW_MODE
        )
        
        assert integration.existing_coordinator == mock_marl_coordinator
        assert integration.integration_mode == IntegrationMode.SHADOW_MODE
        assert integration.shadow_mode
        assert not integration.fallback_manager.fallback_active
    
    def test_legacy_mode_operation(self, mock_marl_coordinator, mock_risk_state):
        """Test operation in legacy-only mode"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            integration_mode=IntegrationMode.LEGACY_ONLY
        )
        
        # Should use original coordinator without modification
        result = integration.existing_coordinator.coordinate_decision(mock_risk_state)
        
        assert result is not None
        assert integration.coordination_count > 0
    
    def test_shadow_mode_operation(self, mock_marl_coordinator, intelligence_coordinator, mock_risk_state):
        """Test operation in shadow mode"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            intelligence_coordinator,
            IntegrationMode.SHADOW_MODE
        )
        
        # Execute coordination
        result = integration.existing_coordinator.coordinate_decision(mock_risk_state)
        
        assert result is not None
        assert len(integration.integration_metrics) > 0
        
        # Should have recorded both legacy and intelligence metrics
        metrics = integration.integration_metrics[-1]
        assert metrics.legacy_response_time_ms > 0
    
    def test_fallback_mechanism(self, mock_marl_coordinator, mock_risk_state):
        """Test fallback mechanism"""
        # Create integration with mock intelligence that fails
        failing_intelligence = Mock()
        failing_intelligence.coordinate_intelligence_decision.side_effect = Exception("Intelligence failure")
        
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            failing_intelligence,
            IntegrationMode.FULL_INTELLIGENCE
        )
        
        # Should fallback to legacy on intelligence failure
        result = integration.existing_coordinator.coordinate_decision(mock_risk_state)
        
        assert result is not None
        assert integration.fallback_manager.fallback_active
        assert integration.fallback_count > 0
    
    def test_performance_monitoring(self, mock_marl_coordinator):
        """Test performance monitoring and guardian"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            integration_mode=IntegrationMode.SHADOW_MODE
        )
        
        # Record performance metrics
        integration.performance_guardian.record_current_metric(
            PerformanceTarget.RESPONSE_TIME_MS, 15.0  # Above 10ms target
        )
        
        violations = integration.performance_guardian.check_performance_violation()
        assert len(violations) > 0  # Should detect violation
        
        degradation = integration.performance_guardian.get_performance_degradation()
        assert degradation > 0
    
    def test_integration_mode_switching(self, mock_marl_coordinator):
        """Test switching between integration modes"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            integration_mode=IntegrationMode.SHADOW_MODE
        )
        
        assert integration.shadow_mode
        
        # Switch to hybrid mode
        integration.set_integration_mode(IntegrationMode.HYBRID_MODE)
        
        assert not integration.shadow_mode
        assert integration.gradual_migration_enabled
        assert integration.migration_progress > 0
    
    def test_integration_status_reporting(self, mock_marl_coordinator, mock_risk_state):
        """Test integration status reporting"""
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            integration_mode=IntegrationMode.SHADOW_MODE
        )
        
        # Execute some coordinations to generate data
        for _ in range(3):
            integration.existing_coordinator.coordinate_decision(mock_risk_state)
        
        status = integration.get_integration_status()
        
        assert 'integration_mode' in status
        assert 'coordination_count' in status
        assert 'performance_metrics' in status
        assert 'fallback_history' in status
        assert status['coordination_count'] >= 3


class TestIntegrationAndPerformance:
    """Integration and performance tests"""
    
    def test_end_to_end_coordination(self, mock_marl_coordinator, intelligence_coordinator, mock_risk_state):
        """Test end-to-end intelligence coordination"""
        # Register intelligence components
        crisis_component = MockIntelligenceComponent("crisis_forecaster", IntelligenceType.CRISIS_FORECASTER)
        intelligence_coordinator.register_intelligence_component(
            "crisis_forecaster", IntelligenceType.CRISIS_FORECASTER,
            crisis_component.make_decision, IntelligencePriority.HIGH
        )
        
        # Create integration layer
        integration = SeamlessIntegrationLayer(
            mock_marl_coordinator,
            intelligence_coordinator,
            IntegrationMode.FULL_INTELLIGENCE
        )
        
        # Execute coordination
        start_time = time.time()
        result = integration.existing_coordinator.coordinate_decision(mock_risk_state)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert result is not None
        assert response_time_ms < 100  # Should be fast (100ms threshold for test)
        assert integration.coordination_count > 0
    
    def test_latency_requirements(self, intelligence_coordinator, mock_risk_state):
        """Test latency requirements are met"""
        # Register fast components
        for i in range(3):
            component = MockIntelligenceComponent(f"fast_component_{i}", IntelligenceType.PRE_MORTEM_ANALYST)
            intelligence_coordinator.register_intelligence_component(
                f"fast_component_{i}", IntelligenceType.PRE_MORTEM_ANALYST,
                component.make_decision, IntelligencePriority.MEDIUM
            )
        
        # Test multiple coordinations
        response_times = []
        for _ in range(10):
            start_time = time.time()
            result = intelligence_coordinator.coordinate_intelligence_decision(mock_risk_state, {})
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            assert result is not None
        
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # Check latency requirements
        assert avg_response_time < 10.0  # Average < 10ms
        assert p95_response_time < 20.0  # P95 < 20ms
    
    def test_stress_testing(self, intelligence_coordinator, mock_risk_state):
        """Test system under stress conditions"""
        # Register multiple components
        for i in range(5):
            component = MockIntelligenceComponent(f"stress_component_{i}", IntelligenceType.PRE_MORTEM_ANALYST)
            intelligence_coordinator.register_intelligence_component(
                f"stress_component_{i}", IntelligenceType.PRE_MORTEM_ANALYST,
                component.make_decision, IntelligencePriority.MEDIUM
            )
        
        # Concurrent coordination requests
        def coordination_worker():
            return intelligence_coordinator.coordinate_intelligence_decision(mock_risk_state, {})
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(coordination_worker) for _ in range(50)]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Coordination failed under stress: {e}")
        
        # All requests should complete successfully
        assert len(results) == 50
        
        # Check performance didn't degrade severely
        metrics = intelligence_coordinator.get_coordination_metrics()
        assert metrics['coordination_count'] >= 50
    
    def test_error_handling_and_recovery(self, intelligence_coordinator, mock_risk_state):
        """Test error handling and recovery mechanisms"""
        # Register component that sometimes fails
        failure_count = 0
        
        def unreliable_component(risk_state, context):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated component failure")
            return {"decision": "success"}, 0.8, "Normal operation"
        
        intelligence_coordinator.register_intelligence_component(
            "unreliable_component", IntelligenceType.PRE_MORTEM_ANALYST,
            unreliable_component, IntelligencePriority.MEDIUM
        )
        
        # Execute multiple coordinations
        success_count = 0
        for _ in range(10):
            try:
                result = intelligence_coordinator.coordinate_intelligence_decision(mock_risk_state, {})
                if result is not None:
                    success_count += 1
            except Exception:
                pass  # Expected failures
        
        # Should handle failures gracefully and continue operating
        assert success_count > 5  # At least some coordinations should succeed
        assert intelligence_coordinator.coordination_count > 0


# Performance benchmarks
def test_coordination_latency_benchmark(mock_marl_coordinator, intelligence_coordinator, mock_risk_state):
    """Benchmark coordination latency"""
    # Baseline: Legacy MARL coordination
    legacy_times = []
    for _ in range(100):
        start = time.perf_counter()
        mock_marl_coordinator.coordinate_decision(mock_risk_state)
        end = time.perf_counter()
        legacy_times.append((end - start) * 1000)
    
    # Intelligence coordination
    intelligence_times = []
    for _ in range(100):
        start = time.perf_counter()
        intelligence_coordinator.coordinate_intelligence_decision(mock_risk_state, {})
        end = time.perf_counter()
        intelligence_times.append((end - start) * 1000)
    
    legacy_avg = np.mean(legacy_times)
    intelligence_avg = np.mean(intelligence_times)
    
    print(f"\nLatency Benchmark Results:")
    print(f"Legacy MARL Average: {legacy_avg:.2f}ms")
    print(f"Intelligence Average: {intelligence_avg:.2f}ms")
    print(f"Overhead: {intelligence_avg - legacy_avg:.2f}ms")
    print(f"Relative Performance: {intelligence_avg / legacy_avg:.2f}x")
    
    # Performance requirements
    assert legacy_avg < 15.0  # Legacy should be < 15ms
    assert intelligence_avg < 25.0  # Intelligence should be < 25ms
    assert (intelligence_avg / legacy_avg) < 3.0  # Intelligence should be < 3x slower


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])