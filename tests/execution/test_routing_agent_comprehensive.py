"""
Comprehensive Test Suite for Routing Agent (π₅) - The Arbitrageur
================================================================

Complete validation and testing framework for the 5th MARL agent including:
- Unit tests for core functionality
- Integration tests with broker performance tracker
- QoE calculation and feedback system validation
- MARL training loop testing
- Performance benchmarking and load testing
- Edge case and failure scenario testing

Author: Agent 1 - The Arbitrageur Implementation
Date: 2025-07-13
"""

import asyncio
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

from src.execution.agents.routing_agent import (
    RoutingAgent, RoutingState, RoutingAction, BrokerType, QoEMetrics,
    BrokerPerformanceMonitor, RoutingNetwork, create_routing_agent, DEFAULT_ROUTING_CONFIG
)
from src.execution.brokers.broker_performance_tracker import (
    BrokerPerformanceTracker, ExecutionRecord, BrokerPerformanceSummary
)
from src.execution.analytics.qoe_calculator import (
    QoECalculator, QoEMeasurement, QoEComponents, QoEGrade
)
from src.execution.analytics.routing_analytics import (
    RoutingAnalytics, RoutingDecisionLog, RoutingSessionAnalytics
)
from src.core.events import EventBus, Event, EventType


class TestRoutingAgentCore:
    """Test core routing agent functionality"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def routing_config(self):
        return DEFAULT_ROUTING_CONFIG.copy()
    
    @pytest.fixture
    def routing_agent(self, routing_config, event_bus):
        return RoutingAgent(routing_config, event_bus)
    
    @pytest.fixture
    def sample_routing_state(self):
        """Create a sample routing state for testing"""
        return RoutingState(
            order_size=1000,
            order_value=50000.0,
            order_urgency=0.7,
            order_side=0.0,  # BUY
            order_type=0.0,  # MARKET
            
            volatility=0.15,
            spread_bps=5.0,
            volume_ratio=1.2,
            time_of_day=0.5,
            market_stress=0.3,
            
            broker_latencies=np.array([0.8, 0.9, 0.7, 0.6]),  # Normalized scores
            broker_fill_rates=np.array([0.995, 0.998, 0.992, 0.996]),
            broker_costs=np.array([0.8, 0.7, 0.9, 0.85]),
            broker_reliabilities=np.array([0.99, 0.98, 0.995, 0.97]),
            broker_availabilities=np.array([1.0, 1.0, 1.0, 1.0]),
            
            portfolio_value=100000.0,
            position_concentration=0.15,
            risk_budget_used=0.3,
            recent_pnl=0.05,
            correlation_risk=0.2
        )
    
    def test_routing_agent_initialization(self, routing_agent):
        """Test routing agent initialization"""
        assert routing_agent is not None
        assert len(routing_agent.broker_ids) == 4
        assert routing_agent.num_brokers == 4
        assert routing_agent.network is not None
        assert routing_agent.performance_monitor is not None
        assert routing_agent.target_qoe_score == 0.85
        assert routing_agent.max_routing_latency_us == 100.0
    
    def test_routing_state_creation(self, sample_routing_state):
        """Test routing state creation and tensor conversion"""
        assert sample_routing_state.dimension == 55
        
        tensor = sample_routing_state.to_tensor()
        assert tensor.shape == (55,)
        assert torch.is_tensor(tensor)
        assert tensor.dtype == torch.float32
    
    def test_routing_network_forward_pass(self, routing_agent, sample_routing_state):
        """Test neural network forward pass"""
        state_tensor = sample_routing_state.to_tensor()
        
        # Test forward pass
        action_logits, state_value = routing_agent.network(state_tensor)
        
        assert action_logits.shape == (routing_agent.num_brokers,)
        assert state_value.shape == (1,)
        
        # Test action probabilities
        action_probs = routing_agent.network.get_action_probabilities(state_tensor)
        assert action_probs.shape == (routing_agent.num_brokers,)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_routing_action_execution(self, routing_agent, sample_routing_state):
        """Test routing action execution"""
        start_time = time.perf_counter()
        
        # Test deterministic action
        action = await routing_agent.act(sample_routing_state, deterministic=True)
        
        end_time = time.perf_counter()
        execution_time_us = (end_time - start_time) * 1_000_000
        
        # Validate action
        assert isinstance(action, RoutingAction)
        assert action.broker_id in routing_agent.broker_ids
        assert isinstance(action.broker_type, BrokerType)
        assert 0.0 <= action.confidence <= 1.0
        assert 0.0 <= action.expected_qoe <= 1.0
        assert action.reasoning is not None
        
        # Validate performance requirement
        assert execution_time_us <= routing_agent.max_routing_latency_us * 2  # Allow some buffer for testing
    
    @pytest.mark.asyncio
    async def test_routing_action_exploration(self, routing_agent, sample_routing_state):
        """Test exploration vs exploitation in routing decisions"""
        # Set high exploration
        routing_agent.exploration_epsilon = 0.9
        
        actions = []
        for _ in range(20):
            action = await routing_agent.act(sample_routing_state, deterministic=False)
            actions.append(action.broker_id)
        
        # Should see some variety in exploration mode
        unique_brokers = set(actions)
        assert len(unique_brokers) >= 2, "Exploration should lead to variety in broker selection"
        
        # Set low exploration
        routing_agent.exploration_epsilon = 0.0
        
        deterministic_actions = []
        for _ in range(10):
            action = await routing_agent.act(sample_routing_state, deterministic=True)
            deterministic_actions.append(action.broker_id)
        
        # Should be consistent in deterministic mode
        assert len(set(deterministic_actions)) == 1, "Deterministic mode should be consistent"
    
    def test_expected_qoe_calculation(self, routing_agent, sample_routing_state):
        """Test expected QoE calculation"""
        for broker_id in routing_agent.broker_ids:
            expected_qoe = routing_agent._calculate_expected_qoe(broker_id, sample_routing_state)
            assert 0.0 <= expected_qoe <= 1.0
        
        # Test with market stress
        stressed_state = sample_routing_state
        stressed_state.market_stress = 0.8
        stressed_qoe = routing_agent._calculate_expected_qoe(routing_agent.broker_ids[0], stressed_state)
        
        normal_qoe = routing_agent._calculate_expected_qoe(routing_agent.broker_ids[0], sample_routing_state)
        assert stressed_qoe <= normal_qoe, "High market stress should reduce expected QoE"
    
    def test_qoe_reward_calculation(self, routing_agent):
        """Test QoE reward calculation for MARL training"""
        # Create sample QoE metrics
        qoe_metrics = QoEMetrics(
            execution_id="test_exec_001",
            broker_id="IB",
            fill_rate=0.995,
            slippage_bps=2.5,
            commission_cost=0.005,
            latency_ms=15.0
        )
        
        reward = routing_agent._calculate_qoe_reward(qoe_metrics)
        
        assert isinstance(reward, float)
        assert reward >= 0.0  # Rewards should be non-negative
        
        # Test excellent performance
        excellent_metrics = QoEMetrics(
            execution_id="test_exec_002",
            broker_id="IB",
            fill_rate=1.0,
            slippage_bps=0.5,
            commission_cost=0.001,
            latency_ms=5.0
        )
        excellent_metrics.qoe_score = 0.95
        
        excellent_reward = routing_agent._calculate_qoe_reward(excellent_metrics)
        assert excellent_reward > reward, "Excellent performance should yield higher reward"
    
    def test_broker_state_vector_creation(self, routing_agent):
        """Test broker state vector creation for network input"""
        # Create mock broker data
        order_data = {
            'quantity': 1000,
            'notional_value': 50000,
            'urgency': 0.7,
            'side': 'BUY',
            'order_type_code': 0
        }
        
        market_data = {
            'volatility': 0.15,
            'spread_bps': 5.0,
            'volume_ratio': 1.2,
            'time_of_day_normalized': 0.5,
            'stress_indicator': 0.3
        }
        
        portfolio_data = {
            'portfolio_value': 100000,
            'concentration': 0.15,
            'risk_budget_used': 0.3,
            'recent_pnl': 0.05,
            'correlation_risk': 0.2
        }
        
        state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
        
        assert isinstance(state, RoutingState)
        assert state.dimension == 55
        
        # Test tensor conversion
        tensor = state.to_tensor()
        assert tensor.shape == (55,)
        assert not torch.isnan(tensor).any(), "State tensor should not contain NaN values"
        assert torch.isfinite(tensor).all(), "State tensor should not contain infinite values"


class TestBrokerPerformanceMonitor:
    """Test broker performance monitoring system"""
    
    @pytest.fixture
    def performance_monitor(self):
        return BrokerPerformanceMonitor(window_minutes=60)
    
    @pytest.fixture
    def sample_execution(self):
        """Create sample execution for testing"""
        from src.execution.brokers.base_broker import BrokerExecution
        return BrokerExecution(
            execution_id="exec_001",
            broker_order_id="order_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.25,
            timestamp=datetime.now(),
            commission=0.50,
            fees=0.10
        )
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization"""
        assert performance_monitor is not None
        assert performance_monitor.window_minutes == 60
        assert len(performance_monitor.metrics) == 0
    
    def test_execution_recording(self, performance_monitor, sample_execution):
        """Test execution recording and metrics calculation"""
        broker_id = "IB"
        latency_ms = 25.0
        slippage_bps = 2.5
        commission = 0.50
        
        performance_monitor.add_execution(
            broker_id, sample_execution, latency_ms, slippage_bps, commission
        )
        
        # Check data storage
        assert broker_id in performance_monitor.execution_history
        assert len(performance_monitor.execution_history[broker_id]) == 1
        assert len(performance_monitor.latency_history[broker_id]) == 1
        
        # Trigger metrics update
        performance_monitor._update_broker_metrics(broker_id)
        
        # Check metrics calculation
        assert broker_id in performance_monitor.metrics
        metrics = performance_monitor.metrics[broker_id]
        assert metrics.avg_latency_ms == latency_ms
        assert metrics.fill_rate == 1.0  # Assuming fully filled
        assert metrics.avg_slippage_bps == slippage_bps
    
    def test_state_vector_generation(self, performance_monitor, sample_execution):
        """Test broker state vector generation for neural network"""
        broker_ids = ["IB", "ALPACA", "TDA", "SCHWAB"]
        
        # Add some sample data
        for broker_id in broker_ids[:2]:
            performance_monitor.add_execution(
                broker_id, sample_execution, 20.0, 1.5, 0.40
            )
            performance_monitor._update_broker_metrics(broker_id)
        
        # Get state vectors
        latencies, fill_rates, costs, reliabilities, availabilities = (
            performance_monitor.get_broker_state_vector(broker_ids)
        )
        
        # Validate dimensions
        assert len(latencies) == len(broker_ids)
        assert len(fill_rates) == len(broker_ids)
        assert len(costs) == len(broker_ids)
        assert len(reliabilities) == len(broker_ids)
        assert len(availabilities) == len(broker_ids)
        
        # Validate value ranges
        assert all(0.0 <= val <= 1.0 for val in latencies)
        assert all(0.0 <= val <= 1.0 for val in fill_rates)
        assert all(0.0 <= val <= 1.0 for val in costs)
        assert all(0.0 <= val <= 1.0 for val in reliabilities)
        assert all(0.0 <= val <= 1.0 for val in availabilities)
    
    def test_quality_score_calculation(self, performance_monitor):
        """Test quality score calculation"""
        from src.execution.brokers.broker_performance_tracker import BrokerPerformanceMetrics
        
        # Create metrics with known values
        metrics = BrokerPerformanceMetrics(
            broker_id="TEST",
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            avg_latency_ms=50.0,
            fill_rate=0.995,
            avg_slippage_bps=2.0,
            commission_per_share=0.005,
            uptime_percentage=0.99,
            error_rate=0.01
        )
        
        quality_score = performance_monitor._calculate_quality_score(metrics)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5, "Good metrics should yield good quality score"
        
        # Test poor metrics
        poor_metrics = BrokerPerformanceMetrics(
            broker_id="POOR",
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            avg_latency_ms=500.0,
            fill_rate=0.8,
            avg_slippage_bps=20.0,
            commission_per_share=0.05,
            uptime_percentage=0.9,
            error_rate=0.1
        )
        
        poor_quality_score = performance_monitor._calculate_quality_score(poor_metrics)
        assert poor_quality_score < quality_score, "Poor metrics should yield lower quality score"


class TestQoECalculator:
    """Test Quality of Execution calculator"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def qoe_calculator(self, event_bus):
        config = {'measurement_window_hours': 24, 'min_measurements_for_benchmark': 10}
        return QoECalculator(config, event_bus)
    
    @pytest.fixture
    def sample_execution_data(self):
        """Create sample execution data for QoE calculation"""
        return {
            'execution_id': 'exec_qoe_001',
            'broker_id': 'IB',
            'symbol': 'AAPL',
            'timestamp': datetime.now(),
            'order_quantity': 100,
            'filled_quantity': 100,
            'execution_price': 150.25,
            'expected_price': 150.20,
            'commission': 0.50,
            'latency_ms': 25.0,
            'market_price_at_order': 150.22,
            'market_price_at_execution': 150.24,
            'spread_at_execution': 0.04,
            'volatility': 0.15,
            'market_stress_level': 0.2,
            'order_urgency': 0.5,
            'order_complexity': 0.3
        }
    
    def test_qoe_calculator_initialization(self, qoe_calculator):
        """Test QoE calculator initialization"""
        assert qoe_calculator is not None
        assert qoe_calculator.measurement_window_hours == 24
        assert qoe_calculator.min_measurements_for_benchmark == 10
        assert qoe_calculator.total_measurements == 0
    
    def test_qoe_measurement_calculation(self, qoe_calculator, sample_execution_data):
        """Test QoE measurement calculation"""
        measurement = qoe_calculator.calculate_qoe(sample_execution_data)
        
        # Validate measurement object
        assert isinstance(measurement, QoEMeasurement)
        assert measurement.execution_id == sample_execution_data['execution_id']
        assert measurement.broker_id == sample_execution_data['broker_id']
        
        # Validate calculated metrics
        assert measurement.fill_rate == 1.0  # Fully filled
        assert measurement.slippage_bps > 0  # Should have some slippage
        assert 0.0 <= measurement.qoe_score <= 1.0
        assert isinstance(measurement.qoe_grade, QoEGrade)
        
        # Validate components
        assert isinstance(measurement.components, QoEComponents)
        assert 0.0 <= measurement.components.fill_rate_score <= 1.0
        assert 0.0 <= measurement.components.slippage_score <= 1.0
        assert 0.0 <= measurement.components.commission_score <= 1.0
        assert 0.0 <= measurement.components.latency_score <= 1.0
    
    def test_qoe_grade_assignment(self, qoe_calculator):
        """Test QoE grade assignment based on scores"""
        test_cases = [
            (0.97, QoEGrade.EXCELLENT),
            (0.92, QoEGrade.VERY_GOOD),
            (0.87, QoEGrade.GOOD),
            (0.80, QoEGrade.AVERAGE),
            (0.70, QoEGrade.BELOW_AVERAGE),
            (0.60, QoEGrade.POOR),
            (0.50, QoEGrade.VERY_POOR),
            (0.40, QoEGrade.UNACCEPTABLE)
        ]
        
        for score, expected_grade in test_cases:
            measurement = QoEMeasurement(
                execution_id="test",
                broker_id="TEST",
                symbol="TEST",
                timestamp=datetime.now(),
                order_quantity=100,
                filled_quantity=100,
                execution_price=100.0,
                expected_price=100.0,
                commission=0.0,
                latency_ms=10.0,
                market_price_at_order=100.0,
                market_price_at_execution=100.0,
                spread_at_execution=0.01,
                volatility=0.1,
                market_stress_level=0.0,
                order_urgency=0.5,
                order_size_category="SMALL",
                order_complexity=0.1
            )
            
            # Override QoE score for testing
            measurement.qoe_score = score
            measurement.qoe_grade = measurement._determine_grade(score)
            
            assert measurement.qoe_grade == expected_grade
    
    def test_qoe_feedback_generation(self, qoe_calculator, sample_execution_data):
        """Test QoE feedback generation for MARL training"""
        measurement = qoe_calculator.calculate_qoe(sample_execution_data)
        feedback = qoe_calculator.create_qoe_feedback(measurement)
        
        # Validate feedback structure
        required_keys = [
            'total_reward', 'fill_rate_reward', 'slippage_reward',
            'commission_reward', 'latency_reward', 'qoe_reward',
            'peer_reward', 'improvement_reward'
        ]
        
        for key in required_keys:
            assert key in feedback
            assert isinstance(feedback[key], float)
            assert feedback[key] >= 0.0  # All rewards should be non-negative
        
        # Validate total reward is reasonable
        assert 0.0 <= feedback['total_reward'] <= 2.0  # Should be roughly in this range


class TestRoutingAnalytics:
    """Test routing analytics and logging"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def routing_analytics(self, event_bus):
        config = {
            'log_retention_days': 7,
            'analytics_window_hours': 24,
            'enable_detailed_logging': True,
            'enable_performance_tracking': True
        }
        return RoutingAnalytics(config, event_bus)
    
    @pytest.fixture
    def sample_routing_action(self):
        return RoutingAction(
            broker_id="IB",
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            confidence=0.85,
            expected_qoe=0.82,
            reasoning="Test routing decision"
        )
    
    @pytest.fixture
    def sample_decision_context(self):
        return {
            'symbol': 'AAPL',
            'order_size': 1000,
            'order_value': 150000.0,
            'order_urgency': 0.7,
            'order_side': 'BUY',
            'order_type': 'MARKET',
            'market_volatility': 0.15,
            'market_stress': 0.3,
            'spread_bps': 5.0,
            'volume_ratio': 1.2,
            'time_of_day': 0.5,
            'decision_latency_us': 75.0,
            'routing_strategy': 'PERFORMANCE_BASED',
            'primary_factors': ['latency', 'qoe_score'],
            'state_vector_hash': 'abc123',
            'action_probabilities': {'IB': 0.4, 'ALPACA': 0.3, 'TDA': 0.2, 'SCHWAB': 0.1},
            'exploration_factor': 0.1
        }
    
    @pytest.fixture
    def sample_broker_states(self):
        return {
            'IB': {'latency': 20.0, 'fill_rate': 0.998, 'cost': 0.005, 'qoe_score': 0.85},
            'ALPACA': {'latency': 30.0, 'fill_rate': 0.995, 'cost': 0.003, 'qoe_score': 0.82},
            'TDA': {'latency': 25.0, 'fill_rate': 0.996, 'cost': 0.007, 'qoe_score': 0.78},
            'SCHWAB': {'latency': 35.0, 'fill_rate': 0.992, 'cost': 0.006, 'qoe_score': 0.75}
        }
    
    def test_routing_analytics_initialization(self, routing_analytics):
        """Test routing analytics initialization"""
        assert routing_analytics is not None
        assert routing_analytics.log_retention_days == 7
        assert routing_analytics.analytics_window_hours == 24
        assert routing_analytics.enable_detailed_logging is True
        assert routing_analytics.total_decisions_logged == 0
        assert routing_analytics.current_session_id is not None
    
    def test_decision_logging(self, routing_analytics, sample_routing_action, 
                            sample_decision_context, sample_broker_states):
        """Test routing decision logging"""
        decision_log = routing_analytics.log_routing_decision(
            sample_routing_action, sample_decision_context, sample_broker_states
        )
        
        # Validate log entry
        assert isinstance(decision_log, RoutingDecisionLog)
        assert decision_log.decision_id is not None
        assert decision_log.selected_broker == sample_routing_action.broker_id
        assert decision_log.confidence == sample_routing_action.confidence
        assert decision_log.expected_qoe == sample_routing_action.expected_qoe
        
        # Validate context data
        assert decision_log.symbol == sample_decision_context['symbol']
        assert decision_log.order_size == sample_decision_context['order_size']
        assert decision_log.market_volatility == sample_decision_context['market_volatility']
        
        # Validate broker states
        assert len(decision_log.available_brokers) == len(sample_broker_states)
        assert decision_log.broker_latencies['IB'] == sample_broker_states['IB']['latency']
        
        # Validate tracking
        assert routing_analytics.total_decisions_logged == 1
        assert len(routing_analytics.decision_logs) == 1
    
    def test_outcome_updating(self, routing_analytics, sample_routing_action,
                            sample_decision_context, sample_broker_states):
        """Test updating decision logs with outcomes"""
        # Log a decision
        decision_log = routing_analytics.log_routing_decision(
            sample_routing_action, sample_decision_context, sample_broker_states
        )
        
        # Create outcome data
        qoe_metrics = QoEMetrics(
            execution_id="exec_001",
            broker_id=sample_routing_action.broker_id,
            fill_rate=0.995,
            slippage_bps=2.5,
            commission_cost=0.005,
            latency_ms=22.0
        )
        qoe_metrics.qoe_score = 0.83
        
        # Update outcome
        routing_analytics.update_decision_outcome(decision_log.decision_id, qoe_metrics)
        
        # Validate outcome update
        assert decision_log.actual_qoe == qoe_metrics.qoe_score
        assert decision_log.actual_latency_ms == qoe_metrics.latency_ms
        assert decision_log.actual_slippage_bps == qoe_metrics.slippage_bps
        assert decision_log.actual_fill_rate == qoe_metrics.fill_rate
        assert decision_log.prediction_error is not None
    
    def test_analytics_summary_generation(self, routing_analytics, sample_routing_action,
                                        sample_decision_context, sample_broker_states):
        """Test analytics summary generation"""
        # Log multiple decisions
        for i in range(5):
            context = sample_decision_context.copy()
            context['order_size'] = 1000 + i * 100
            routing_analytics.log_routing_decision(
                sample_routing_action, context, sample_broker_states
            )
        
        # Generate summary
        summary = routing_analytics.get_routing_analytics_summary(time_period_hours=1)
        
        # Validate summary structure
        assert 'total_decisions' in summary
        assert 'unique_brokers_used' in summary
        assert 'avg_confidence' in summary
        assert 'avg_expected_qoe' in summary
        assert 'broker_usage' in summary
        assert 'current_session' in summary
        
        # Validate values
        assert summary['total_decisions'] == 5
        assert summary['unique_brokers_used'] == 1  # All same broker
        assert 0.0 <= summary['avg_confidence'] <= 1.0
        assert 0.0 <= summary['avg_expected_qoe'] <= 1.0
    
    def test_broker_specific_analytics(self, routing_analytics, sample_routing_action,
                                     sample_decision_context, sample_broker_states):
        """Test broker-specific analytics"""
        # Log decisions for specific broker
        for i in range(3):
            routing_analytics.log_routing_decision(
                sample_routing_action, sample_decision_context, sample_broker_states
            )
        
        # Get broker analytics
        broker_analytics = routing_analytics.get_broker_analytics(
            sample_routing_action.broker_id, time_period_hours=1
        )
        
        # Validate analytics
        assert 'broker_id' in broker_analytics
        assert 'total_decisions' in broker_analytics
        assert 'usage_percentage' in broker_analytics
        assert 'avg_confidence' in broker_analytics
        assert broker_analytics['broker_id'] == sample_routing_action.broker_id
        assert broker_analytics['total_decisions'] == 3


class TestIntegration:
    """Integration tests for the complete routing system"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def complete_routing_system(self, event_bus):
        """Create complete routing system with all components"""
        routing_config = DEFAULT_ROUTING_CONFIG.copy()
        routing_agent = RoutingAgent(routing_config, event_bus)
        
        qoe_config = {'measurement_window_hours': 24, 'min_measurements_for_benchmark': 5}
        qoe_calculator = QoECalculator(qoe_config, event_bus)
        
        analytics_config = {'log_retention_days': 7, 'analytics_window_hours': 24}
        routing_analytics = RoutingAnalytics(analytics_config, event_bus)
        
        return {
            'routing_agent': routing_agent,
            'qoe_calculator': qoe_calculator,
            'routing_analytics': routing_analytics,
            'event_bus': event_bus
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing_flow(self, complete_routing_system):
        """Test complete end-to-end routing flow"""
        routing_agent = complete_routing_system['routing_agent']
        qoe_calculator = complete_routing_system['qoe_calculator']
        routing_analytics = complete_routing_system['routing_analytics']
        
        # Create routing state
        order_data = {
            'quantity': 1000, 'notional_value': 150000, 'urgency': 0.7,
            'side': 'BUY', 'order_type_code': 0
        }
        market_data = {
            'volatility': 0.15, 'spread_bps': 5.0, 'volume_ratio': 1.2,
            'time_of_day_normalized': 0.5, 'stress_indicator': 0.3
        }
        portfolio_data = {
            'portfolio_value': 100000, 'concentration': 0.15,
            'risk_budget_used': 0.3, 'recent_pnl': 0.05, 'correlation_risk': 0.2
        }
        
        routing_state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
        
        # Execute routing decision
        routing_action = await routing_agent.act(routing_state)
        
        # Log routing decision
        decision_context = {**order_data, **market_data, 'decision_latency_us': 85.0}
        broker_states = {
            'IB': {'latency': 20.0, 'fill_rate': 0.998, 'cost': 0.005, 'qoe_score': 0.85},
            'ALPACA': {'latency': 30.0, 'fill_rate': 0.995, 'cost': 0.003, 'qoe_score': 0.82}
        }
        
        decision_log = routing_analytics.log_routing_decision(
            routing_action, decision_context, broker_states
        )
        
        # Simulate execution outcome
        execution_data = {
            'execution_id': 'exec_integration_001',
            'broker_id': routing_action.broker_id,
            'symbol': 'AAPL',
            'order_quantity': 1000,
            'filled_quantity': 995,  # Partial fill
            'execution_price': 150.25,
            'expected_price': 150.20,
            'commission': 2.50,
            'latency_ms': 22.0,
            'market_stress_level': 0.3,
            'order_urgency': 0.7
        }
        
        # Calculate QoE
        qoe_measurement = qoe_calculator.calculate_qoe(execution_data)
        
        # Update routing analytics with outcome
        routing_analytics.update_decision_outcome(decision_log.decision_id, qoe_measurement)
        
        # Add execution feedback to routing agent
        qoe_metrics = QoEMetrics(
            execution_id=execution_data['execution_id'],
            broker_id=routing_action.broker_id,
            fill_rate=qoe_measurement.fill_rate,
            slippage_bps=qoe_measurement.slippage_bps,
            commission_cost=qoe_measurement.commission,
            latency_ms=qoe_measurement.latency_ms
        )
        qoe_metrics.qoe_score = qoe_measurement.qoe_score
        
        returned_qoe_metrics = routing_agent.add_execution_feedback(
            routing_action.broker_id, 
            Mock(execution_id=execution_data['execution_id']),
            qoe_measurement.latency_ms,
            qoe_measurement.slippage_bps,
            qoe_measurement.commission
        )
        
        # Validate complete flow
        assert routing_action.broker_id in routing_agent.broker_ids
        assert qoe_measurement.qoe_score > 0.0
        assert decision_log.actual_qoe is not None
        assert returned_qoe_metrics.qoe_score > 0.0
        assert routing_agent.routing_stats['total_routes'] == 1
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, complete_routing_system):
        """Test that performance requirements are met"""
        routing_agent = complete_routing_system['routing_agent']
        
        # Create multiple routing states for testing
        test_states = []
        for i in range(10):
            order_data = {
                'quantity': 1000 + i * 100, 'notional_value': 150000 + i * 10000,
                'urgency': 0.5 + i * 0.05, 'side': 'BUY', 'order_type_code': 0
            }
            market_data = {
                'volatility': 0.15 + i * 0.01, 'spread_bps': 5.0 + i * 0.5,
                'volume_ratio': 1.0 + i * 0.1, 'time_of_day_normalized': 0.5,
                'stress_indicator': 0.1 + i * 0.05
            }
            portfolio_data = {
                'portfolio_value': 100000, 'concentration': 0.15,
                'risk_budget_used': 0.3, 'recent_pnl': 0.05, 'correlation_risk': 0.2
            }
            
            state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
            test_states.append(state)
        
        # Test routing performance
        routing_times = []
        for state in test_states:
            start_time = time.perf_counter()
            await routing_agent.act(state, deterministic=True)
            end_time = time.perf_counter()
            
            routing_time_us = (end_time - start_time) * 1_000_000
            routing_times.append(routing_time_us)
        
        # Validate performance requirements
        avg_routing_time = np.mean(routing_times)
        p95_routing_time = np.percentile(routing_times, 95)
        
        assert avg_routing_time <= routing_agent.max_routing_latency_us * 3  # Allow buffer for testing
        assert p95_routing_time <= routing_agent.max_routing_latency_us * 5  # More generous for p95
        
        print(f"Routing performance: avg={avg_routing_time:.1f}μs, p95={p95_routing_time:.1f}μs")
    
    @pytest.mark.asyncio
    async def test_learning_and_adaptation(self, complete_routing_system):
        """Test learning and adaptation capabilities"""
        routing_agent = complete_routing_system['routing_agent']
        
        # Enable training mode
        routing_agent.training_mode = True
        routing_agent.exploration_epsilon = 0.2
        
        # Create consistent routing state
        order_data = {'quantity': 1000, 'notional_value': 150000, 'urgency': 0.7}
        market_data = {'volatility': 0.15, 'spread_bps': 5.0, 'stress_indicator': 0.3}
        portfolio_data = {'portfolio_value': 100000, 'concentration': 0.15}
        
        routing_state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
        
        # Record initial broker preferences
        initial_actions = []
        for _ in range(20):
            action = await routing_agent.act(routing_state)
            initial_actions.append(action.broker_id)
        
        initial_distribution = {broker: initial_actions.count(broker) for broker in routing_agent.broker_ids}
        
        # Simulate positive feedback for one broker
        best_broker = "IB"
        for _ in range(10):
            qoe_metrics = QoEMetrics(
                execution_id=f"training_{_}",
                broker_id=best_broker,
                fill_rate=0.999,
                slippage_bps=0.5,
                commission_cost=0.001,
                latency_ms=8.0
            )
            qoe_metrics.qoe_score = 0.95
            
            routing_agent.add_execution_feedback(
                best_broker, Mock(execution_id=f"training_{_}"),
                8.0, 0.5, 0.001
            )
        
        # Record new broker preferences after feedback
        final_actions = []
        for _ in range(20):
            action = await routing_agent.act(routing_state)
            final_actions.append(action.broker_id)
        
        final_distribution = {broker: final_actions.count(broker) for broker in routing_agent.broker_ids}
        
        # Validate learning effect (should show some preference change)
        # Note: This is a simplified test - full MARL training would require more sophisticated validation
        assert routing_agent.routing_stats['total_routes'] > 0
        assert len(routing_agent.qoe_history) > 0
        assert routing_agent.routing_stats['avg_qoe_score'] > 0.0
        
        print(f"Initial distribution: {initial_distribution}")
        print(f"Final distribution: {final_distribution}")
        print(f"Average QoE: {routing_agent.routing_stats['avg_qoe_score']:.3f}")


class TestLoadAndStress:
    """Load testing and stress testing for routing system"""
    
    @pytest.fixture
    def routing_agent(self):
        config = DEFAULT_ROUTING_CONFIG.copy()
        config['max_routing_latency_us'] = 50.0  # Stricter requirement for load testing
        return RoutingAgent(config, EventBus())
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_decisions(self, routing_agent):
        """Test concurrent routing decisions under load"""
        # Create multiple routing states
        states = []
        for i in range(50):
            order_data = {'quantity': 1000 + i, 'notional_value': 150000}
            market_data = {'volatility': 0.15, 'stress_indicator': 0.2}
            portfolio_data = {'portfolio_value': 100000}
            
            state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
            states.append(state)
        
        # Execute concurrent decisions
        start_time = time.perf_counter()
        
        tasks = [routing_agent.act(state, deterministic=True) for state in states]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        
        # Validate results
        successful_results = [r for r in results if isinstance(r, RoutingAction)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) >= 45  # Allow some failures under load
        assert len(failed_results) <= 5
        
        # Validate performance
        total_time_s = end_time - start_time
        decisions_per_second = len(successful_results) / total_time_s
        
        print(f"Concurrent performance: {decisions_per_second:.1f} decisions/second")
        print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
        
        assert decisions_per_second >= 50.0  # Should handle at least 50 decisions/second
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, routing_agent):
        """Test memory usage doesn't grow excessively under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute many routing decisions
        order_data = {'quantity': 1000, 'notional_value': 150000}
        market_data = {'volatility': 0.15, 'stress_indicator': 0.2}
        portfolio_data = {'portfolio_value': 100000}
        
        routing_state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
        
        for i in range(1000):
            await routing_agent.act(routing_state, deterministic=True)
            
            # Add some feedback to test memory management
            if i % 10 == 0:
                qoe_metrics = QoEMetrics(
                    execution_id=f"memory_test_{i}",
                    broker_id="IB",
                    fill_rate=0.99,
                    slippage_bps=2.0,
                    commission_cost=0.005,
                    latency_ms=15.0
                )
                qoe_metrics.qoe_score = 0.85
                
                routing_agent.add_execution_feedback("IB", Mock(), 15.0, 2.0, 0.005)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 100MB for 1000 decisions)
        assert memory_increase < 100.0, f"Excessive memory usage: {memory_increase:.1f}MB"


@pytest.mark.asyncio
async def test_routing_agent_factory_function():
    """Test factory function for creating routing agent"""
    config = DEFAULT_ROUTING_CONFIG.copy()
    event_bus = EventBus()
    
    routing_agent = create_routing_agent(config, event_bus)
    
    assert isinstance(routing_agent, RoutingAgent)
    assert routing_agent.config == config
    assert routing_agent.event_bus == event_bus


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])