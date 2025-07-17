"""
Integration test suite for Strategic-Tactical Bridge.

This module tests the critical bridge between the 30-minute strategic
decision layer and the 5-minute tactical execution layer.
"""
import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.strategic, pytest.mark.tactical]


class TestStrategicTacticalBridge:
    """Test the bridge between strategic and tactical layers."""

    @pytest.fixture
    def bridge_config(self):
        """Configuration for strategic-tactical bridge."""
        return {
            "strategic_horizon": "30m",
            "tactical_horizon": "5m",
            "decision_translation": {
                "enabled": True,
                "confidence_threshold": 0.6,
                "position_scaling": True,
                "risk_adjustment": True
            },
            "execution_planning": {
                "slice_size": 0.1,
                "max_slices": 10,
                "timing_optimization": True,
                "fvg_targeting": True
            },
            "feedback_loop": {
                "enabled": True,
                "update_frequency": "5m",
                "learning_rate": 0.01
            }
        }

    @pytest.fixture
    def mock_bridge(self, bridge_config):
        """Create a mock strategic-tactical bridge."""
        bridge = Mock()
        bridge.config = bridge_config
        
        # Mock components
        bridge.strategic_interface = Mock()
        bridge.tactical_interface = Mock()
        bridge.decision_translator = Mock()
        bridge.execution_planner = Mock()
        bridge.feedback_processor = Mock()
        
        # Mock methods
        bridge.translate_decision = Mock()
        bridge.create_execution_plan = Mock()
        bridge.monitor_execution = Mock()
        bridge.update_strategic_feedback = Mock()
        bridge.synchronize_timeframes = Mock()
        
        return bridge

    @pytest.fixture
    def sample_strategic_decision(self):
        """Sample strategic decision from 30m layer."""
        return {
            "timestamp": datetime.now(),
            "timeframe": "30m",
            "position": 0.7,
            "confidence": 0.85,
            "pattern": "TYPE_1",
            "risk_budget": 0.02,
            "expected_duration": 6,  # bars (3 hours)
            "metadata": {
                "mlmi_signal": 0.8,
                "nwrqk_trend": 0.7,
                "mmd_regime": "TRENDING",
                "synergy_strength": 0.9
            }
        }

    @pytest.fixture
    def sample_tactical_state(self):
        """Sample tactical state from 5m layer."""
        return {
            "timestamp": datetime.now(),
            "current_position": 0.3,
            "pending_orders": 2,
            "filled_quantity": 0.3,
            "remaining_quantity": 0.4,
            "average_fill_price": 1.0847,
            "slippage": 0.0003,
            "execution_time_elapsed": 15,  # minutes
            "fvg_opportunities": [
                {"level": 1.0845, "type": "bearish", "strength": 0.8},
                {"level": 1.0842, "type": "bearish", "strength": 0.6}
            ]
        }

    def test_decision_translation(self, mock_bridge, sample_strategic_decision):
        """Test translation of strategic decisions to tactical instructions."""
        # Mock translation result
        tactical_instructions = {
            "target_position": 0.7,
            "execution_method": "ICEBERG",
            "slice_size": 0.1,
            "max_slippage": 0.0015,
            "urgency": 0.6,
            "fvg_targeting": True,
            "time_limit": 180,  # minutes
            "risk_limits": {
                "max_drawdown": 0.02,
                "stop_loss": 0.015
            }
        }
        
        mock_bridge.translate_decision = Mock(return_value=tactical_instructions)
        result = mock_bridge.translate_decision(sample_strategic_decision)
        
        assert result["target_position"] == 0.7
        assert result["execution_method"] == "ICEBERG"
        assert result["fvg_targeting"] is True
        assert "risk_limits" in result
        
        mock_bridge.translate_decision.assert_called_once_with(sample_strategic_decision)

    def test_execution_plan_creation(self, mock_bridge, sample_strategic_decision):
        """Test creation of detailed execution plan."""
        execution_plan = {
            "total_quantity": 0.7,
            "slices": [
                {"quantity": 0.1, "target_price": 1.0845, "timing": "immediate"},
                {"quantity": 0.1, "target_price": 1.0842, "timing": "fvg_opportunity"},
                {"quantity": 0.15, "target_price": 1.0840, "timing": "passive"},
                {"quantity": 0.15, "target_price": 1.0838, "timing": "passive"},
                {"quantity": 0.2, "target_price": 1.0835, "timing": "contingent"}
            ],
            "total_slices": 5,
            "estimated_duration": 120,  # minutes
            "contingency_plans": {
                "adverse_move": "reduce_position",
                "low_liquidity": "extend_timeline",
                "high_volatility": "increase_patience"
            }
        }
        
        mock_bridge.create_execution_plan = Mock(return_value=execution_plan)
        plan = mock_bridge.create_execution_plan(sample_strategic_decision)
        
        assert plan["total_quantity"] == 0.7
        assert len(plan["slices"]) == 5
        assert "contingency_plans" in plan
        
        # Validate slice quantities sum to total
        total_slice_quantity = sum(slice_info["quantity"] for slice_info in plan["slices"])
        assert np.isclose(total_slice_quantity, plan["total_quantity"])

    def test_execution_monitoring(self, mock_bridge, sample_tactical_state):
        """Test real-time execution monitoring."""
        monitoring_result = {
            "progress": 0.43,  # 43% complete
            "performance": {
                "slippage": 0.0003,
                "time_efficiency": 0.85,
                "fvg_utilization": 0.6
            },
            "adjustments": {
                "speed_up": False,
                "slow_down": True,
                "change_method": False
            },
            "estimated_completion": 45  # minutes remaining
        }
        
        mock_bridge.monitor_execution = Mock(return_value=monitoring_result)
        result = mock_bridge.monitor_execution(sample_tactical_state)
        
        assert "progress" in result
        assert "performance" in result
        assert "adjustments" in result
        assert 0 <= result["progress"] <= 1
        
        mock_bridge.monitor_execution.assert_called_once_with(sample_tactical_state)

    def test_strategic_feedback_loop(self, mock_bridge):
        """Test feedback from tactical execution to strategic layer."""
        execution_feedback = {
            "actual_fill_price": 1.0846,
            "total_slippage": 0.0004,
            "execution_time": 95,  # minutes
            "partial_fills": 3,
            "market_impact": 0.0002,
            "fvg_effectiveness": 0.75,
            "pattern_validation": {
                "TYPE_1": True,
                "confidence_realized": 0.8
            }
        }
        
        strategic_update = {
            "model_update": True,
            "confidence_adjustment": -0.05,
            "execution_cost_estimate": 0.0006,
            "pattern_validation": True
        }
        
        mock_bridge.update_strategic_feedback = Mock(return_value=strategic_update)
        result = mock_bridge.update_strategic_feedback(execution_feedback)
        
        assert "model_update" in result
        assert "confidence_adjustment" in result
        assert isinstance(result["pattern_validation"], bool)
        
        mock_bridge.update_strategic_feedback.assert_called_once_with(execution_feedback)

    def test_timeframe_synchronization(self, mock_bridge):
        """Test synchronization between 30m and 5m timeframes."""
        sync_data = {
            "strategic_timestamp": datetime(2023, 1, 1, 14, 30),  # 2:30 PM
            "tactical_timestamps": [
                datetime(2023, 1, 1, 14, 30),
                datetime(2023, 1, 1, 14, 35),
                datetime(2023, 1, 1, 14, 40),
                datetime(2023, 1, 1, 14, 45),
                datetime(2023, 1, 1, 14, 50),
                datetime(2023, 1, 1, 14, 55)
            ]
        }
        
        sync_result = {
            "aligned": True,
            "tactical_bars_per_strategic": 6,
            "next_strategic_bar": datetime(2023, 1, 1, 15, 0),
            "time_to_next_strategic": 600  # seconds
        }
        
        mock_bridge.synchronize_timeframes = Mock(return_value=sync_result)
        result = mock_bridge.synchronize_timeframes(sync_data)
        
        assert result["aligned"] is True
        assert result["tactical_bars_per_strategic"] == 6
        assert isinstance(result["next_strategic_bar"], datetime)

    @pytest.mark.performance
    def test_bridge_latency(self, mock_bridge, sample_strategic_decision, performance_timer):
        """Test bridge latency for decision translation."""
        performance_timer.start()
        
        # Translate decision
        mock_bridge.translate_decision(sample_strategic_decision)
        
        performance_timer.stop()
        
        # Requirement: Bridge latency < 10ms
        assert performance_timer.elapsed_ms() < 10

    @pytest.mark.asyncio
    async def test_async_bridge_operations(self, mock_bridge):
        """Test asynchronous bridge operations."""
        # Mock async methods
        mock_bridge.translate_async = AsyncMock(return_value={"target_position": 0.7})
        mock_bridge.monitor_async = AsyncMock(return_value={"progress": 0.5})
        mock_bridge.feedback_async = AsyncMock(return_value={"update": "success"})
        
        # Run async operations
        translate_result = await mock_bridge.translate_async()
        monitor_result = await mock_bridge.monitor_async()
        feedback_result = await mock_bridge.feedback_async()
        
        assert translate_result["target_position"] == 0.7
        assert monitor_result["progress"] == 0.5
        assert feedback_result["update"] == "success"


class TestPositionSizingBridge:
    """Test position sizing coordination between strategic and tactical layers."""

    @pytest.fixture
    def sizing_bridge_config(self):
        """Configuration for position sizing bridge."""
        return {
            "strategic_allocation": 0.8,
            "tactical_max_slice": 0.1,
            "risk_scaling": True,
            "kelly_optimization": True,
            "volatility_adjustment": True
        }

    @pytest.fixture
    def mock_sizing_bridge(self, sizing_bridge_config):
        """Create mock position sizing bridge."""
        bridge = Mock()
        bridge.config = sizing_bridge_config
        
        # Mock methods
        bridge.calculate_slice_sizes = Mock()
        bridge.apply_risk_scaling = Mock()
        bridge.optimize_kelly_fraction = Mock()
        bridge.adjust_for_volatility = Mock()
        
        return bridge

    def test_slice_size_calculation(self, mock_sizing_bridge):
        """Test calculation of optimal slice sizes."""
        strategic_position = 0.8
        market_conditions = {
            "volatility": 0.15,
            "liquidity": 0.7,
            "spread": 0.0008
        }
        
        slice_plan = {
            "slice_sizes": [0.1, 0.1, 0.15, 0.15, 0.3],
            "timing_intervals": [0, 5, 10, 20, 30],  # minutes
            "adaptive_sizing": True
        }
        
        mock_sizing_bridge.calculate_slice_sizes = Mock(return_value=slice_plan)
        result = mock_sizing_bridge.calculate_slice_sizes(strategic_position, market_conditions)
        
        assert "slice_sizes" in result
        assert "timing_intervals" in result
        assert len(result["slice_sizes"]) == len(result["timing_intervals"])
        
        # Verify slices sum to strategic position
        total_size = sum(result["slice_sizes"])
        assert np.isclose(total_size, strategic_position)

    def test_risk_scaling_application(self, mock_sizing_bridge):
        """Test application of risk scaling to position sizes."""
        base_position = 0.8
        risk_factors = {
            "var_estimate": 0.02,
            "max_drawdown": 0.05,
            "correlation_risk": 0.3,
            "regime_uncertainty": 0.2
        }
        
        scaled_position = 0.6  # Reduced due to risk
        
        mock_sizing_bridge.apply_risk_scaling = Mock(return_value=scaled_position)
        result = mock_sizing_bridge.apply_risk_scaling(base_position, risk_factors)
        
        assert result <= base_position  # Should reduce position
        assert result >= 0
        
        mock_sizing_bridge.apply_risk_scaling.assert_called_once_with(base_position, risk_factors)

    def test_kelly_optimization(self, mock_sizing_bridge):
        """Test Kelly criterion optimization for position sizing."""
        historical_performance = {
            "win_rate": 0.58,
            "avg_win": 0.025,
            "avg_loss": 0.018,
            "num_trades": 150
        }
        
        kelly_fraction = 0.35
        
        mock_sizing_bridge.optimize_kelly_fraction = Mock(return_value=kelly_fraction)
        result = mock_sizing_bridge.optimize_kelly_fraction(historical_performance)
        
        assert 0 <= result <= 1  # Kelly fraction should be between 0 and 1
        mock_sizing_bridge.optimize_kelly_fraction.assert_called_once_with(historical_performance)

    def test_volatility_adjustment(self, mock_sizing_bridge):
        """Test volatility-based position adjustment."""
        base_position = 0.7
        volatility_data = {
            "current_vol": 0.18,
            "target_vol": 0.12,
            "vol_forecast": 0.15
        }
        
        adjusted_position = 0.55  # Reduced due to high volatility
        
        mock_sizing_bridge.adjust_for_volatility = Mock(return_value=adjusted_position)
        result = mock_sizing_bridge.adjust_for_volatility(base_position, volatility_data)
        
        assert result <= base_position  # Should reduce for higher volatility
        mock_sizing_bridge.adjust_for_volatility.assert_called_once_with(base_position, volatility_data)


class TestTimingBridge:
    """Test timing coordination between strategic and tactical layers."""

    @pytest.fixture
    def timing_bridge_config(self):
        """Configuration for timing bridge."""
        return {
            "strategic_patience": 180,  # minutes
            "tactical_urgency_levels": [0.2, 0.5, 0.8, 1.0],
            "fvg_wait_time": 30,  # minutes
            "market_hours_adjustment": True
        }

    @pytest.fixture
    def mock_timing_bridge(self, timing_bridge_config):
        """Create mock timing bridge."""
        bridge = Mock()
        bridge.config = timing_bridge_config
        
        # Mock methods
        bridge.calculate_urgency = Mock()
        bridge.optimize_entry_timing = Mock()
        bridge.coordinate_timeframes = Mock()
        bridge.handle_time_decay = Mock()
        
        return bridge

    def test_urgency_calculation(self, mock_timing_bridge):
        """Test calculation of tactical urgency from strategic timeline."""
        strategic_timeline = {
            "total_duration": 180,  # minutes
            "elapsed_time": 45,
            "remaining_quantity": 0.6,
            "market_conditions": "normal"
        }
        
        urgency_level = 0.4  # Moderate urgency
        
        mock_timing_bridge.calculate_urgency = Mock(return_value=urgency_level)
        result = mock_timing_bridge.calculate_urgency(strategic_timeline)
        
        assert 0 <= result <= 1
        mock_timing_bridge.calculate_urgency.assert_called_once_with(strategic_timeline)

    def test_entry_timing_optimization(self, mock_timing_bridge):
        """Test optimization of entry timing."""
        timing_context = {
            "fvg_levels": [1.0845, 1.0842],
            "support_resistance": [1.0840, 1.0850],
            "volatility_windows": ["low", "medium", "low"],
            "session_overlap": True
        }
        
        optimal_timing = {
            "immediate_entry": 0.3,
            "fvg_wait": 0.4,
            "support_wait": 0.3,
            "recommended_action": "fvg_wait"
        }
        
        mock_timing_bridge.optimize_entry_timing = Mock(return_value=optimal_timing)
        result = mock_timing_bridge.optimize_entry_timing(timing_context)
        
        assert "recommended_action" in result
        assert result["recommended_action"] in ["immediate_entry", "fvg_wait", "support_wait"]
        
        # Probabilities should sum to 1
        prob_sum = result["immediate_entry"] + result["fvg_wait"] + result["support_wait"]
        assert np.isclose(prob_sum, 1.0)

    def test_timeframe_coordination(self, mock_timing_bridge):
        """Test coordination between different timeframes."""
        timeframe_data = {
            "strategic_bar_start": datetime(2023, 1, 1, 14, 30),
            "strategic_bar_end": datetime(2023, 1, 1, 15, 0),
            "current_time": datetime(2023, 1, 1, 14, 45),
            "tactical_bars_remaining": 3
        }
        
        coordination_result = {
            "time_pressure": 0.5,
            "bars_remaining": 3,
            "recommended_pace": "moderate",
            "deadline_approach": False
        }
        
        mock_timing_bridge.coordinate_timeframes = Mock(return_value=coordination_result)
        result = mock_timing_bridge.coordinate_timeframes(timeframe_data)
        
        assert "time_pressure" in result
        assert "recommended_pace" in result
        assert 0 <= result["time_pressure"] <= 1

    def test_time_decay_handling(self, mock_timing_bridge):
        """Test handling of time decay in strategic decisions."""
        decay_context = {
            "decision_age": 25,  # minutes
            "original_confidence": 0.85,
            "market_change": 0.02,
            "pattern_persistence": 0.8
        }
        
        decay_adjustment = {
            "adjusted_confidence": 0.80,
            "urgency_increase": 0.1,
            "position_scaling": 0.95
        }
        
        mock_timing_bridge.handle_time_decay = Mock(return_value=decay_adjustment)
        result = mock_timing_bridge.handle_time_decay(decay_context)
        
        assert "adjusted_confidence" in result
        assert "urgency_increase" in result
        assert result["adjusted_confidence"] <= decay_context["original_confidence"]


class TestErrorRecoveryBridge:
    """Test error recovery and failover mechanisms in the bridge."""

    @pytest.fixture
    def recovery_bridge_config(self):
        """Configuration for error recovery bridge."""
        return {
            "max_retries": 3,
            "timeout_seconds": 30,
            "fallback_strategies": ["reduce_position", "market_orders", "cancel_execution"],
            "circuit_breaker": {
                "enabled": True,
                "error_threshold": 5,
                "cooldown_minutes": 10
            }
        }

    @pytest.fixture
    def mock_recovery_bridge(self, recovery_bridge_config):
        """Create mock error recovery bridge."""
        bridge = Mock()
        bridge.config = recovery_bridge_config
        bridge.error_count = 0
        bridge.circuit_breaker_active = False
        
        # Mock methods
        bridge.handle_communication_failure = Mock()
        bridge.fallback_to_market_orders = Mock()
        bridge.activate_circuit_breaker = Mock()
        bridge.recover_from_error = Mock()
        
        return bridge

    def test_communication_failure_handling(self, mock_recovery_bridge):
        """Test handling of communication failures between layers."""
        failure_context = {
            "error_type": "timeout",
            "component": "strategic_interface",
            "retry_count": 1,
            "last_successful_contact": datetime.now() - timedelta(minutes=2)
        }
        
        recovery_action = {
            "action": "retry_with_backoff",
            "backoff_seconds": 5,
            "fallback_ready": True
        }
        
        mock_recovery_bridge.handle_communication_failure = Mock(return_value=recovery_action)
        result = mock_recovery_bridge.handle_communication_failure(failure_context)
        
        assert "action" in result
        assert "fallback_ready" in result
        mock_recovery_bridge.handle_communication_failure.assert_called_once_with(failure_context)

    def test_fallback_to_market_orders(self, mock_recovery_bridge):
        """Test fallback to market orders when sophisticated execution fails."""
        fallback_context = {
            "original_plan": "ICEBERG",
            "remaining_quantity": 0.4,
            "urgency": 0.9,
            "max_slippage_acceptable": 0.002
        }
        
        fallback_plan = {
            "execution_method": "MARKET",
            "slice_immediately": True,
            "estimated_slippage": 0.0015
        }
        
        mock_recovery_bridge.fallback_to_market_orders = Mock(return_value=fallback_plan)
        result = mock_recovery_bridge.fallback_to_market_orders(fallback_context)
        
        assert result["execution_method"] == "MARKET"
        assert "estimated_slippage" in result
        mock_recovery_bridge.fallback_to_market_orders.assert_called_once_with(fallback_context)

    def test_circuit_breaker_activation(self, mock_recovery_bridge):
        """Test circuit breaker activation under error conditions."""
        error_history = [
            {"timestamp": datetime.now() - timedelta(minutes=1), "type": "timeout"},
            {"timestamp": datetime.now() - timedelta(minutes=2), "type": "execution_error"},
            {"timestamp": datetime.now() - timedelta(minutes=3), "type": "communication_failure"},
            {"timestamp": datetime.now() - timedelta(minutes=4), "type": "data_error"},
            {"timestamp": datetime.now() - timedelta(minutes=5), "type": "timeout"}
        ]
        
        circuit_breaker_result = {
            "activated": True,
            "cooldown_until": datetime.now() + timedelta(minutes=10),
            "safe_mode": True
        }
        
        mock_recovery_bridge.activate_circuit_breaker = Mock(return_value=circuit_breaker_result)
        result = mock_recovery_bridge.activate_circuit_breaker(error_history)
        
        assert result["activated"] is True
        assert "cooldown_until" in result
        mock_recovery_bridge.activate_circuit_breaker.assert_called_once_with(error_history)

    def test_error_recovery_process(self, mock_recovery_bridge):
        """Test complete error recovery process."""
        recovery_context = {
            "error_type": "execution_stall",
            "affected_components": ["tactical_executor", "position_tracker"],
            "recovery_priority": "high",
            "data_consistency": "partial"
        }
        
        recovery_result = {
            "recovery_successful": True,
            "components_restored": ["tactical_executor", "position_tracker"],
            "data_reconciled": True,
            "execution_resumed": True
        }
        
        mock_recovery_bridge.recover_from_error = Mock(return_value=recovery_result)
        result = mock_recovery_bridge.recover_from_error(recovery_context)
        
        assert result["recovery_successful"] is True
        assert "components_restored" in result
        mock_recovery_bridge.recover_from_error.assert_called_once_with(recovery_context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])