"""
Test Suite for Stop/Target Agent (π₂)

Comprehensive testing of the Stop/Target Agent including:
- Basic functionality and action space validation
- ATR-based calculation accuracy
- Volatility regime adaptation
- Trailing stop mechanism
- Time decay adjustments
- Performance requirements (<10ms response time)
- Integration with BaseRiskAgent
- Emergency stop protocols

Author: Agent 3 - Stop/Target Agent Developer
Version: 1.0
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import time

from src.risk.agents.stop_target_agent import (
    StopTargetAgent, PositionContext, StopTargetLevels,
    VolatilityRegime, TrendMode
)
from src.risk.agents.base_risk_agent import RiskState
from src.core.events import EventBus, Event, EventType


class TestStopTargetAgent:
    """Test suite for Stop/Target Agent"""
    
    @pytest.fixture
    def config(self):
        """Default agent configuration"""
        return {
            'name': 'test_stop_target_agent',
            'atr_period': 14,
            'min_stop_multiplier': 0.5,
            'max_stop_multiplier': 3.0,
            'min_target_multiplier': 0.5,
            'max_target_multiplier': 3.0,
            'enable_trailing_stops': True,
            'trailing_activation_pct': 1.0,
            'trailing_step_pct': 0.5,
            'enable_time_decay': True,
            'max_hold_time_minutes': 240,
            'time_decay_factor': 0.1,
            'max_response_time_ms': 10.0
        }
    
    @pytest.fixture
    def event_bus(self):
        """Mock event bus"""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def agent(self, config, event_bus):
        """Stop/Target agent instance"""
        return StopTargetAgent(config, event_bus)
    
    @pytest.fixture
    def sample_risk_state(self):
        """Sample risk state for testing"""
        return RiskState(
            account_equity_normalized=1.0,
            open_positions_count=2,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.05,
            margin_usage_pct=0.4,
            time_of_day_risk=0.2,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
    
    @pytest.fixture
    def sample_position_context(self):
        """Sample position context for testing"""
        return PositionContext(
            entry_price=100.0,
            current_price=102.0,
            position_size=1000.0,
            time_in_trade_minutes=30,
            unrealized_pnl_pct=2.0,
            avg_true_range=1.5,
            price_velocity=0.005,
            volume_profile=0.8
        )
    
    def test_agent_initialization(self, config, event_bus):
        """Test agent initialization"""
        agent = StopTargetAgent(config, event_bus)
        
        assert agent.name == 'test_stop_target_agent'
        assert agent.atr_period == 14
        assert agent.enable_trailing_stops is True
        assert agent.enable_time_decay is True
        assert agent.action_space.low.tolist() == [0.5, 0.5]
        assert agent.action_space.high.tolist() == [3.0, 3.0]
        assert agent.max_response_time_ms == 10.0
    
    def test_action_space_validation(self, agent):
        """Test action space constraints"""
        # Test valid action
        valid_action = np.array([1.5, 2.0])
        assert agent.action_space.contains(valid_action)
        
        # Test invalid actions
        invalid_actions = [
            np.array([0.3, 2.0]),  # Below minimum
            np.array([1.5, 3.5]),  # Above maximum
            np.array([4.0, 2.0]),  # Both above maximum
        ]
        
        for action in invalid_actions:
            assert not agent.action_space.contains(action)
    
    def test_observation_space_validation(self, agent):
        """Test extended observation space (16D)"""
        # Valid 16D observation
        valid_obs = np.random.randn(16)
        assert agent.validate_observation(valid_obs) is True
        
        # Invalid observations
        invalid_obs = [
            np.random.randn(10),  # Wrong dimension
            np.array([np.nan] * 16),  # Contains NaN
            np.array([np.inf] * 16),  # Contains Inf
        ]
        
        for obs in invalid_obs:
            assert agent.validate_observation(obs) is False
    
    def test_atr_calculation(self, agent):
        """Test ATR calculation accuracy"""
        # Test data: high, low, close prices
        high_prices = [105.0, 107.0, 106.0, 108.0, 110.0]
        low_prices = [103.0, 104.0, 104.5, 106.0, 108.0]
        close_prices = [104.0, 106.0, 105.0, 107.0, 109.0]
        
        atr = agent.calculate_atr(high_prices, low_prices, close_prices)
        
        assert atr > 0
        assert atr < 10  # Reasonable range
        assert isinstance(atr, float)
    
    def test_volatility_regime_detection(self, agent, sample_risk_state):
        """Test volatility regime classification"""
        # Low volatility
        sample_risk_state.volatility_regime = 0.1
        regime = agent.detect_volatility_regime(sample_risk_state)
        assert regime == VolatilityRegime.LOW
        
        # Medium volatility
        sample_risk_state.volatility_regime = 0.4
        regime = agent.detect_volatility_regime(sample_risk_state)
        assert regime == VolatilityRegime.MEDIUM
        
        # High volatility
        sample_risk_state.volatility_regime = 0.7
        regime = agent.detect_volatility_regime(sample_risk_state)
        assert regime == VolatilityRegime.HIGH
        
        # Extreme volatility
        sample_risk_state.volatility_regime = 0.9
        regime = agent.detect_volatility_regime(sample_risk_state)
        assert regime == VolatilityRegime.EXTREME
    
    def test_trend_mode_detection(self, agent, sample_position_context):
        """Test trend mode classification"""
        # Ranging market
        sample_position_context.price_velocity = 0.0005
        mode = agent.detect_trend_mode(sample_position_context)
        assert mode == TrendMode.RANGING
        
        # Trending market
        sample_position_context.price_velocity = 0.015
        mode = agent.detect_trend_mode(sample_position_context)
        assert mode == TrendMode.TRENDING
        
        # Reversal market
        sample_position_context.price_velocity = 0.005
        mode = agent.detect_trend_mode(sample_position_context)
        assert mode == TrendMode.REVERSAL
    
    def test_volatility_adjustment_calculation(self, agent):
        """Test volatility-based adjustments"""
        # Low volatility - tighter stops, wider targets
        stop_adj, target_adj = agent.calculate_volatility_adjustment(VolatilityRegime.LOW)
        assert stop_adj < 1.0
        assert target_adj > 1.0
        
        # Medium volatility - neutral
        stop_adj, target_adj = agent.calculate_volatility_adjustment(VolatilityRegime.MEDIUM)
        assert stop_adj == 1.0
        assert target_adj == 1.0
        
        # High volatility - wider stops, tighter targets
        stop_adj, target_adj = agent.calculate_volatility_adjustment(VolatilityRegime.HIGH)
        assert stop_adj > 1.0
        assert target_adj < 1.0
        
        # Extreme volatility - much wider stops, much tighter targets
        stop_adj, target_adj = agent.calculate_volatility_adjustment(VolatilityRegime.EXTREME)
        assert stop_adj > 1.5
        assert target_adj < 0.8
    
    def test_time_decay_adjustment(self, agent):
        """Test time decay adjustment"""
        # Early in trade - no decay
        decay = agent.calculate_time_decay_adjustment(10)
        assert decay == 1.0 or decay > 0.9
        
        # Mid trade - some decay
        decay = agent.calculate_time_decay_adjustment(120)
        assert 0.7 < decay < 1.0
        
        # Late in trade - more decay
        decay = agent.calculate_time_decay_adjustment(220)
        assert 0.3 <= decay < 0.8
        
        # Maximum time - maximum decay
        decay = agent.calculate_time_decay_adjustment(300)
        assert decay >= 0.3  # Never more than 70% tightening
    
    def test_trailing_stop_calculation(self, agent, sample_position_context):
        """Test trailing stop mechanism"""
        # Not profitable enough - no trailing
        sample_position_context.unrealized_pnl_pct = 0.5
        new_stop, activated = agent.calculate_trailing_stop(sample_position_context, 99.0)
        assert new_stop == 99.0
        assert activated is False
        
        # Profitable long position - should trail up
        sample_position_context.unrealized_pnl_pct = 2.0
        sample_position_context.position_size = 1000.0  # Long
        sample_position_context.current_price = 105.0
        
        new_stop, activated = agent.calculate_trailing_stop(sample_position_context, 99.0)
        assert new_stop > 99.0
        assert activated is True
        
        # Profitable short position - should trail down
        sample_position_context.position_size = -1000.0  # Short
        sample_position_context.current_price = 95.0
        
        new_stop, activated = agent.calculate_trailing_stop(sample_position_context, 101.0)
        assert new_stop < 101.0
        assert activated is True
    
    def test_calculate_risk_action(self, agent, sample_risk_state):
        """Test risk action calculation"""
        action, confidence = agent.calculate_risk_action(sample_risk_state)
        
        assert isinstance(action, np.ndarray)
        assert len(action) == 2
        assert 0.5 <= action[0] <= 3.0  # Stop multiplier in range
        assert 0.5 <= action[1] <= 3.0  # Target multiplier in range
        assert 0.0 <= confidence <= 1.0
    
    def test_stop_target_level_calculation(self, agent, sample_position_context):
        """Test stop/target level calculation"""
        action = np.array([1.5, 2.0])
        levels = agent.calculate_stop_target_levels(sample_position_context, action)
        
        assert isinstance(levels, StopTargetLevels)
        assert levels.stop_loss_price > 0
        assert levels.take_profit_price > 0
        assert levels.stop_multiplier > 0
        assert levels.target_multiplier > 0
        assert isinstance(levels.last_update_time, datetime)
        
        # For long position
        if sample_position_context.position_size > 0:
            assert levels.stop_loss_price < sample_position_context.entry_price
            assert levels.take_profit_price > sample_position_context.entry_price
    
    def test_risk_constraint_validation(self, agent, sample_risk_state):
        """Test risk constraint validation"""
        # Normal conditions - should pass
        assert agent.validate_risk_constraints(sample_risk_state) is True
        
        # High VaR - should fail
        sample_risk_state.var_estimate_5pct = 0.2
        assert agent.validate_risk_constraints(sample_risk_state) is False
        
        # Reset and test drawdown
        sample_risk_state.var_estimate_5pct = 0.02
        sample_risk_state.current_drawdown_pct = 0.25
        assert agent.validate_risk_constraints(sample_risk_state) is False
        
        # Reset and test market stress
        sample_risk_state.current_drawdown_pct = 0.05
        sample_risk_state.market_stress_level = 0.95
        assert agent.validate_risk_constraints(sample_risk_state) is False
    
    def test_response_time_requirement(self, agent, sample_risk_state):
        """Test <10ms response time requirement"""
        start_time = time.time()
        
        # Perform 100 calculations
        for _ in range(100):
            action, confidence = agent.calculate_risk_action(sample_risk_state)
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_time_per_calc = total_time / 100
        
        assert avg_time_per_calc < 10.0, f"Average calculation time: {avg_time_per_calc:.2f}ms"
    
    def test_position_step_function(self, agent, sample_position_context):
        """Test position-specific step function"""
        risk_vector = np.random.randn(10)
        
        levels, confidence = agent.step_position(risk_vector, sample_position_context)
        
        assert isinstance(levels, StopTargetLevels)
        assert 0.0 <= confidence <= 1.0
        assert agent.current_levels == levels
        assert len(agent.position_history) > 0
    
    def test_feature_extraction(self, agent):
        """Test feature extraction from extended observation"""
        # Test 16D observation
        obs_16d = np.random.randn(16)
        features = agent.extract_features(obs_16d)
        assert len(features) == 16
        
        # Test matrix input
        obs_matrix = np.random.randn(2, 8)
        features = agent.extract_features(obs_matrix)
        assert len(features) == 16
        
        # Test short observation - should be padded
        obs_short = np.random.randn(12)
        features = agent.extract_features(obs_short)
        assert len(features) == 16
    
    def test_emergency_stop_protocol(self, agent, event_bus):
        """Test emergency stop functionality"""
        result = agent.emergency_stop_all_positions("Test emergency")
        
        assert result is True
        event_bus.publish.assert_called()
        
        # Check that emergency event was published
        calls = event_bus.publish.call_args_list
        assert len(calls) >= 1
    
    def test_event_handling(self, agent, event_bus):
        """Test event handling capabilities"""
        # Test position update event
        position_event = Mock(spec=Event)
        position_event.data = {'stop_triggered': True, 'position_id': 'test_pos'}
        
        initial_stops = agent.stops_triggered
        agent._handle_position_update(position_event)
        assert agent.stops_triggered == initial_stops + 1
        
        # Test target hit event
        target_event = Mock(spec=Event)
        target_event.data = {'target_hit': True, 'position_id': 'test_pos'}
        
        initial_targets = agent.targets_hit
        agent._handle_position_update(target_event)
        assert agent.targets_hit == initial_targets + 1
    
    def test_risk_metrics_calculation(self, agent):
        """Test risk metrics calculation"""
        # Generate some activity
        agent.stops_triggered = 5
        agent.targets_hit = 8
        agent.total_stop_adjustments = 100
        agent.response_times = [5.0, 7.0, 3.0, 8.0, 6.0]
        
        metrics = agent.get_risk_metrics()
        
        assert metrics.total_risk_decisions == 100
        assert metrics.risk_events_detected == 13  # 5 + 8
        assert metrics.avg_response_time_ms == 5.8  # Average of response times
    
    def test_agent_reset(self, agent):
        """Test agent reset functionality"""
        # Set some state
        agent.stops_triggered = 10
        agent.targets_hit = 5
        agent.current_levels = Mock()
        agent.position_history = [Mock(), Mock()]
        
        agent.reset()
        
        assert agent.stops_triggered == 0
        assert agent.targets_hit == 0
        assert agent.current_levels is None
        assert len(agent.position_history) == 0
        assert len(agent.atr_values) == 0
    
    def test_safe_action_fallback(self, agent):
        """Test safe action fallback"""
        safe_action = agent._get_safe_action()
        
        assert isinstance(safe_action, np.ndarray)
        assert len(safe_action) == 2
        assert 0.5 <= safe_action[0] <= 3.0
        assert 0.5 <= safe_action[1] <= 3.0
    
    def test_string_representations(self, agent):
        """Test string representation methods"""
        agent.stops_triggered = 3
        agent.targets_hit = 7
        agent.total_stop_adjustments = 50
        
        str_repr = str(agent)
        assert "stops=3" in str_repr
        assert "targets=7" in str_repr
        
        repr_str = repr(agent)
        assert "StopTargetAgent" in repr_str
        assert "stops_triggered=3" in repr_str
        assert "targets_hit=7" in repr_str


class TestStopTargetAgentIntegration:
    """Integration tests for Stop/Target Agent"""
    
    def test_integration_with_base_risk_agent(self):
        """Test integration with BaseRiskAgent"""
        config = {
            'name': 'integration_test_agent',
            'max_response_time_ms': 10.0
        }
        event_bus = Mock(spec=EventBus)
        
        agent = StopTargetAgent(config, event_bus)
        
        # Test that it properly inherits from BaseRiskAgent
        assert hasattr(agent, 'risk_tolerance')
        assert hasattr(agent, 'enable_emergency_stop')
        assert hasattr(agent, 'response_times')
        assert agent.max_response_time_ms == 10.0
    
    def test_full_workflow_simulation(self):
        """Test complete workflow simulation"""
        config = {
            'name': 'workflow_test_agent',
            'enable_trailing_stops': True,
            'enable_time_decay': True
        }
        event_bus = Mock(spec=EventBus)
        agent = StopTargetAgent(config, event_bus)
        
        # Simulate market data and position
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=1,
            volatility_regime=0.6,
            correlation_risk=0.4,
            var_estimate_5pct=0.03,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.3,
            market_stress_level=0.4,
            liquidity_conditions=0.7
        )
        
        position_context = PositionContext(
            entry_price=100.0,
            current_price=103.0,
            position_size=1000.0,
            time_in_trade_minutes=60,
            unrealized_pnl_pct=3.0,
            avg_true_range=2.0,
            price_velocity=0.008,
            volume_profile=0.9
        )
        
        # Execute workflow
        levels, confidence = agent.step_position(risk_state.to_vector(), position_context)
        
        # Validate results
        assert isinstance(levels, StopTargetLevels)
        assert 0.0 <= confidence <= 1.0
        assert levels.stop_loss_price < position_context.entry_price  # Long position
        assert levels.take_profit_price > position_context.entry_price
        assert agent.total_stop_adjustments > 0
    
    def test_performance_under_stress(self):
        """Test performance under stress conditions"""
        config = {'name': 'stress_test_agent'}
        event_bus = Mock(spec=EventBus)
        agent = StopTargetAgent(config, event_bus)
        
        # Extreme stress conditions
        stress_risk_state = RiskState(
            account_equity_normalized=0.7,
            open_positions_count=10,
            volatility_regime=0.95,
            correlation_risk=0.9,
            var_estimate_5pct=0.12,
            current_drawdown_pct=0.18,
            margin_usage_pct=0.95,
            time_of_day_risk=0.8,
            market_stress_level=0.95,
            liquidity_conditions=0.1
        )
        
        action, confidence = agent.calculate_risk_action(stress_risk_state)
        
        # Under stress, should use wider stops and conservative targets
        assert action[0] > 1.5  # Wider stops
        assert confidence < 0.8  # Lower confidence under stress
        
        # Should fail risk constraint validation
        assert agent.validate_risk_constraints(stress_risk_state) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])