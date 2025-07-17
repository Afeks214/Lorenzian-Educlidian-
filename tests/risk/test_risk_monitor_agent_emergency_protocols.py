"""
Comprehensive Test Suite for Risk Monitor Agent Emergency Protocols

Tests all critical emergency response capabilities including:
- Emergency action execution
- Response time validation
- Market stress detection
- Real-time risk assessment
- Integration framework
- Performance optimization
"""

import pytest
import asyncio
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

from src.risk.agents.risk_monitor_agent import (
    RiskMonitorAgent, EmergencyAction, RiskSeverity, MarketStressIndicator
)
from src.risk.agents.emergency_action_system import (
    EmergencyActionExecutor, ActionPriority, ExecutionStatus
)
from src.risk.agents.real_time_risk_assessor import (
    RealTimeRiskAssessor, RiskBreach, BreachSeverity, RiskMetricType
)
from src.risk.agents.market_stress_detector import (
    MarketStressDetector, MarketRegime, StressSignal, FlashCrashDetector
)
from src.risk.agents.performance_optimizer import (
    PerformanceOptimizer, PerformanceCache, fast_var_calculation
)
from src.risk.agents.integration_framework import (
    RiskMonitorIntegration, VaRCalculatorBridge, CorrelationTrackerBridge
)
from src.risk.agents.base_risk_agent import RiskState, RiskMetrics
from src.risk.core.var_calculator import VaRCalculator, VaRResult
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.core.events import EventBus, Event, EventType


class TestRiskMonitorAgentEmergencyProtocols:
    """Test suite for Risk Monitor Agent emergency protocols"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def mock_var_calculator(self):
        """Create mock VaR calculator"""
        mock_calc = Mock(spec=VaRCalculator)
        mock_calc.portfolio_value = 1000000.0
        mock_calc.positions = {
            'AAPL': Mock(market_value=300000, symbol='AAPL'),
            'GOOGL': Mock(market_value=400000, symbol='GOOGL'),
            'MSFT': Mock(market_value=300000, symbol='MSFT')
        }
        
        # Mock VaR result
        mock_var_result = Mock(spec=VaRResult)
        mock_var_result.portfolio_var = 20000.0  # 2% VaR
        mock_var_result.confidence_level = 0.95
        mock_var_result.timestamp = datetime.now()
        mock_var_result.component_vars = {'AAPL': 8000, 'GOOGL': 7000, 'MSFT': 5000}
        
        mock_calc.get_latest_var.return_value = mock_var_result
        mock_calc.calculate_var.return_value = mock_var_result
        
        return mock_calc
    
    @pytest.fixture
    def mock_correlation_tracker(self):
        """Create mock correlation tracker"""
        mock_tracker = Mock(spec=CorrelationTracker)
        mock_tracker.current_regime = CorrelationRegime.NORMAL
        mock_tracker.get_correlation_matrix.return_value = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        return mock_tracker
    
    @pytest.fixture
    def risk_monitor_config(self):
        """Configuration for Risk Monitor Agent"""
        return {
            'var_breach_threshold': 0.02,
            'correlation_shock_threshold': 0.7,
            'drawdown_emergency_threshold': 0.10,
            'margin_critical_threshold': 0.90,
            'position_reduction_pct': 0.4,
            'hedge_ratio': 0.8,
            'max_response_time_ms': 10.0,
            'action_cooldown_seconds': 5
        }
    
    @pytest.fixture
    def risk_monitor_agent(self, risk_monitor_config, mock_var_calculator, 
                          mock_correlation_tracker, event_bus):
        """Create Risk Monitor Agent for testing"""
        return RiskMonitorAgent(
            config=risk_monitor_config,
            var_calculator=mock_var_calculator,
            correlation_tracker=mock_correlation_tracker,
            event_bus=event_bus
        )
    
    def test_risk_monitor_agent_initialization(self, risk_monitor_agent, risk_monitor_config):
        """Test Risk Monitor Agent initialization"""
        assert risk_monitor_agent is not None
        assert risk_monitor_agent.var_breach_threshold == risk_monitor_config['var_breach_threshold']
        assert risk_monitor_agent.max_response_time_ms == risk_monitor_config['max_response_time_ms']
        assert risk_monitor_agent.current_risk_severity == RiskSeverity.LOW
        assert len(risk_monitor_agent.emergency_actions_taken) == 0
    
    def test_critical_condition_detection(self, risk_monitor_agent):
        """Test detection of critical risk conditions"""
        # Test emergency drawdown threshold
        risk_state = RiskState(
            account_equity_normalized=0.8,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.15,  # Above emergency threshold
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        critical_action = risk_monitor_agent._check_critical_conditions(risk_state)
        assert critical_action == EmergencyAction.CLOSE_ALL.value
        
        # Test critical margin usage
        risk_state.current_drawdown_pct = 0.05  # Normal
        risk_state.margin_usage_pct = 0.95  # Critical
        
        critical_action = risk_monitor_agent._check_critical_conditions(risk_state)
        assert critical_action == EmergencyAction.REDUCE_POSITION.value
    
    def test_var_breach_detection(self, risk_monitor_agent):
        """Test VaR breach detection logic"""
        # Minor breach
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.025,  # 1.25x threshold
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        action = risk_monitor_agent._check_var_breach(risk_state)
        assert action is None  # Below 1.5x threshold
        
        # Major breach
        risk_state.var_estimate_5pct = 0.045  # 2.25x threshold
        action = risk_monitor_agent._check_var_breach(risk_state)
        assert action == EmergencyAction.CLOSE_ALL.value
    
    def test_correlation_shock_detection(self, risk_monitor_agent):
        """Test correlation shock detection"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.8,  # Above shock threshold
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        action = risk_monitor_agent._check_correlation_shock(risk_state)
        assert action == EmergencyAction.HEDGE.value
    
    def test_response_time_compliance(self, risk_monitor_agent):
        """Test that risk calculations meet <10ms response time"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        # Test multiple calculations for consistency
        response_times = []
        for _ in range(100):
            start_time = time.time()
            action, confidence = risk_monitor_agent.calculate_risk_action(risk_state)
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
        
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        assert avg_response_time < 10.0, f"Average response time {avg_response_time:.2f}ms exceeds 10ms"
        assert max_response_time < 15.0, f"Max response time {max_response_time:.2f}ms too high"
        assert risk_monitor_agent.response_time_violations == 0
    
    def test_market_stress_assessment(self, risk_monitor_agent):
        """Test market stress indicator calculation"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.8,  # High volatility
            correlation_risk=0.6,
            var_estimate_5pct=0.03,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.7,
            time_of_day_risk=0.8,
            market_stress_level=0.7,  # High stress
            liquidity_conditions=0.3  # Poor liquidity
        )
        
        stress_indicator = risk_monitor_agent._assess_market_stress(risk_state)
        
        assert isinstance(stress_indicator, MarketStressIndicator)
        assert stress_indicator.volatility_spike > 0.5
        assert stress_indicator.liquidity_drought > 0.5
        
        stress_level = stress_indicator.get_stress_level()
        assert stress_level in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_emergency_action_execution(self, risk_monitor_agent):
        """Test emergency action execution"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        # Test position reduction
        result = await risk_monitor_agent.execute_emergency_action(
            EmergencyAction.REDUCE_POSITION,
            risk_state,
            "Test reduction"
        )
        
        assert result.success is True
        assert result.action == EmergencyAction.REDUCE_POSITION
        assert result.execution_time_ms > 0
        assert result.risk_reduction_pct > 0
        
        # Test emergency liquidation
        result = await risk_monitor_agent.execute_emergency_action(
            EmergencyAction.CLOSE_ALL,
            risk_state,
            "Test liquidation"
        )
        
        assert result.success is True
        assert result.action == EmergencyAction.CLOSE_ALL
        assert result.risk_reduction_pct == 1.0  # 100% liquidation
    
    def test_action_cooldown_mechanism(self, risk_monitor_agent):
        """Test action cooldown to prevent over-trading"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.8,
            correlation_risk=0.8,  # High correlation
            var_estimate_5pct=0.04,  # High VaR
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        # First action should trigger
        action1, confidence1 = risk_monitor_agent.calculate_risk_action(risk_state)
        assert action1 != EmergencyAction.NO_ACTION.value
        
        # Immediate second action should be on cooldown
        action2, confidence2 = risk_monitor_agent.calculate_risk_action(risk_state)
        assert action2 == EmergencyAction.NO_ACTION.value
    
    def test_risk_metrics_tracking(self, risk_monitor_agent):
        """Test risk metrics tracking and reporting"""
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        # Generate some activity
        for _ in range(10):
            risk_monitor_agent.calculate_risk_action(risk_state)
        
        metrics = risk_monitor_agent.get_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_risk_decisions == 10
        assert metrics.avg_response_time_ms > 0
        
        # Test emergency action summary
        summary = risk_monitor_agent.get_emergency_action_summary()
        assert 'total_actions' in summary
        assert 'response_time_compliance' in summary


class TestEmergencyActionSystem:
    """Test suite for Emergency Action Execution System"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def action_executor_config(self):
        return {
            'max_execution_time_ms': 5000,
            'slippage_tolerance': 0.001,
            'minimum_position_size': 100,
            'emergency_liquidation_enabled': True,
            'hedge_instruments': ['SPY', 'VIX'],
            'max_hedge_notional': 1000000
        }
    
    @pytest.fixture
    def action_executor(self, event_bus, action_executor_config):
        return EmergencyActionExecutor(event_bus, action_executor_config)
    
    @pytest.mark.asyncio
    async def test_position_reduction_execution(self, action_executor):
        """Test position reduction execution"""
        # Setup mock positions
        action_executor.positions = {
            'AAPL': Mock(quantity=1000, market_value=150000, current_price=150.0, risk_contribution=0.3),
            'GOOGL': Mock(quantity=500, market_value=150000, current_price=300.0, risk_contribution=0.4)
        }
        action_executor.portfolio_value = 300000
        
        result = await action_executor.execute_reduce_position(0.5)  # 50% reduction
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.positions_processed == 2
        assert result.success_rate == 1.0
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, action_executor):
        """Test emergency liquidation execution"""
        # Setup mock positions
        action_executor.positions = {
            'AAPL': Mock(quantity=1000, market_value=150000, current_price=150.0, risk_contribution=0.5),
            'GOOGL': Mock(quantity=500, market_value=150000, current_price=300.0, risk_contribution=0.3)
        }
        action_executor.portfolio_value = 300000
        
        result = await action_executor.execute_close_all()
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.positions_processed == 2
        assert result.success_rate == 1.0
        
        # Verify positions are cleared
        for position in action_executor.positions.values():
            assert position.quantity == 0
            assert position.market_value == 0
    
    @pytest.mark.asyncio
    async def test_hedge_creation(self, action_executor):
        """Test hedge position creation"""
        action_executor.portfolio_value = 1000000
        
        result = await action_executor.execute_hedge(0.8)  # 80% hedge
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.positions_processed > 0
        assert result.volume_executed > 0
        
        # Verify hedge positions created
        hedge_positions = [symbol for symbol in action_executor.positions.keys() 
                          if '_HEDGE' in symbol]
        assert len(hedge_positions) > 0
    
    def test_execution_performance_tracking(self, action_executor):
        """Test execution performance tracking"""
        # Simulate some executions
        action_executor.execution_history = [
            Mock(status=ExecutionStatus.COMPLETED, execution_time_ms=3.5, volume_executed=100000),
            Mock(status=ExecutionStatus.COMPLETED, execution_time_ms=4.2, volume_executed=150000),
            Mock(status=ExecutionStatus.FAILED, execution_time_ms=2.1, volume_executed=0)
        ]
        action_executor.execution_times = [3.5, 4.2, 2.1]
        action_executor.total_executions = 3
        action_executor.successful_executions = 2
        
        stats = action_executor.get_execution_stats()
        
        assert stats['total_executions'] == 3
        assert stats['success_rate'] == 2/3
        assert stats['avg_execution_time_ms'] == pytest.approx(3.27, rel=0.1)
        assert stats['total_volume_executed'] == 250000


class TestMarketStressDetector:
    """Test suite for Market Stress Detector"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def stress_detector_config(self):
        return {
            'flash_crash': {
                'price_drop_threshold': 0.05,
                'time_window_seconds': 300,
                'volume_spike_threshold': 3.0
            },
            'volatility': {
                'volatility_lookback': 100,
                'volatility_spike_threshold': 3.0,
                'ewma_lambda': 0.94
            },
            'liquidity': {
                'spread_threshold': 0.001,
                'depth_threshold': 0.5
            }
        }
    
    @pytest.fixture
    def stress_detector(self, event_bus, stress_detector_config):
        return MarketStressDetector(event_bus, stress_detector_config)
    
    def test_flash_crash_detection(self, stress_detector):
        """Test flash crash detection algorithm"""
        from src.risk.agents.market_stress_detector import MarketDataPoint
        
        # Simulate flash crash scenario
        market_data = []
        base_price = 100.0
        
        # Normal market data
        for i in range(10):
            market_data.append(MarketDataPoint(
                timestamp=datetime.now() + timedelta(seconds=i),
                symbol='SPY',
                price=base_price + np.random.normal(0, 0.5),
                volume=1000000,
                bid=base_price - 0.01,
                ask=base_price + 0.01,
                spread=0.02,
                volatility=0.15
            ))
        
        # Flash crash - 8% drop
        for i in range(10, 15):
            crash_price = base_price * (1 - 0.08 * (i - 9) / 5)
            market_data.append(MarketDataPoint(
                timestamp=datetime.now() + timedelta(seconds=i),
                symbol='SPY',
                price=crash_price,
                volume=5000000,  # High volume
                bid=crash_price - 0.05,
                ask=crash_price + 0.05,
                spread=0.1,
                volatility=0.8
            ))
        
        flash_event = stress_detector.flash_crash_detector.analyze_price_movement(market_data)
        
        assert flash_event is not None
        assert flash_event.signal_type == StressSignal.PRICE_GAP
        assert flash_event.severity > 0.8
        assert flash_event.confidence > 0.6
        assert flash_event.market_regime == MarketRegime.FLASH_CRASH
    
    def test_volatility_spike_detection(self, stress_detector):
        """Test volatility spike detection"""
        # Simulate normal returns
        normal_returns = np.random.normal(0, 0.01, 100)
        
        # Add to detector
        for ret in normal_returns:
            stress_detector.volatility_analyzer.volatility_history.append(abs(ret) * np.sqrt(252))
        
        # Simulate volatility spike
        spike_returns = np.random.normal(0, 0.05, 10)  # 5x higher volatility
        
        vol_event = stress_detector.volatility_analyzer.update_volatility(spike_returns.tolist())
        
        assert vol_event is not None
        assert vol_event.signal_type == StressSignal.VOLATILITY_SPIKE
        assert vol_event.severity > 0.5
    
    def test_market_regime_transitions(self, stress_detector):
        """Test market regime transition logic"""
        # Start in normal regime
        assert stress_detector.current_regime == MarketRegime.NORMAL
        
        # Simulate stress events
        from src.risk.agents.market_stress_detector import StressEvent
        
        stress_event = StressEvent(
            event_id="test_stress",
            signal_type=StressSignal.VOLATILITY_SPIKE,
            severity=0.8,
            confidence=0.9,
            detection_time=datetime.now(),
            affected_symbols=['SPY'],
            market_regime=MarketRegime.STRESSED,
            predicted_duration=300.0,
            recommended_action="REDUCE_RISK",
            raw_data={}
        )
        
        stress_detector._update_market_regime([stress_event])
        
        assert stress_detector.current_regime == MarketRegime.STRESSED
    
    def test_stress_level_calculation(self, stress_detector):
        """Test current stress level calculation"""
        # Add some stress events
        from src.risk.agents.market_stress_detector import StressEvent
        
        recent_event = StressEvent(
            event_id="recent_stress",
            signal_type=StressSignal.CORRELATION_SHOCK,
            severity=0.7,
            confidence=0.8,
            detection_time=datetime.now() - timedelta(seconds=30),
            affected_symbols=['SPY', 'AAPL'],
            market_regime=MarketRegime.ELEVATED,
            predicted_duration=180.0,
            recommended_action="MONITOR_CLOSELY",
            raw_data={}
        )
        
        stress_detector.stress_events.append(recent_event)
        
        stress_level = stress_detector.get_current_stress_level()
        
        assert stress_level['stress_score'] > 0
        assert stress_level['recent_events'] == 1
        assert 'flash_crash_probability' in stress_level
        assert 'recommended_action' in stress_level


class TestPerformanceOptimization:
    """Test suite for Performance Optimization"""
    
    @pytest.fixture
    def performance_config(self):
        return {
            'response_time_target_ms': 10.0,
            'cache_size': 1000,
            'cache_ttl_seconds': 60,
            'memory_pool_size': 100,
            'max_workers': 4
        }
    
    @pytest.fixture
    def performance_optimizer(self, performance_config):
        return PerformanceOptimizer(performance_config)
    
    def test_jit_compilation_performance(self, performance_optimizer):
        """Test JIT-compiled function performance"""
        weights = np.array([0.4, 0.3, 0.3])
        volatilities = np.array([0.2, 0.25, 0.3])
        correlation_matrix = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        
        # Test multiple calls for performance
        start_time = time.time()
        for _ in range(1000):
            result = fast_var_calculation(weights, volatilities, correlation_matrix, 1.645)
        execution_time = (time.time() - start_time) * 1000
        
        assert execution_time < 100  # Should be very fast due to JIT
        assert result > 0  # Sanity check
    
    def test_performance_cache(self, performance_optimizer):
        """Test performance caching system"""
        cache = performance_optimizer.cache
        
        # Test cache operations
        cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        assert cache.hit_rate == 1.0
        
        # Test cache miss
        assert cache.get("nonexistent_key") is None
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss
    
    def test_optimized_var_calculation(self, performance_optimizer):
        """Test optimized VaR calculation with caching"""
        weights = np.array([0.5, 0.3, 0.2])
        volatilities = np.array([0.2, 0.25, 0.3])
        correlation_matrix = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        
        # First call - should cache result
        start_time = time.time()
        result1 = performance_optimizer.optimized_var_calculation(
            weights, volatilities, correlation_matrix, 0.95
        )
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call - should use cache
        start_time = time.time()
        result2 = performance_optimizer.optimized_var_calculation(
            weights, volatilities, correlation_matrix, 0.95
        )
        second_call_time = (time.time() - start_time) * 1000
        
        assert result1 == result2
        assert second_call_time < first_call_time  # Cache should be faster
        assert first_call_time < 10  # Within target
    
    def test_performance_benchmarking(self, performance_optimizer):
        """Test performance benchmarking"""
        benchmark_results = performance_optimizer.benchmark_performance(100)
        
        assert 'var_calculation_ms' in benchmark_results
        assert 'correlation_update_ms' in benchmark_results
        assert 'stress_calculation_ms' in benchmark_results
        
        # All calculations should be fast
        for metric, time_ms in benchmark_results.items():
            assert time_ms < 5.0, f"{metric} took {time_ms:.2f}ms, exceeds 5ms target"
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, performance_optimizer):
        """Test parallel risk calculation execution"""
        # Create mock calculations
        def mock_calc1():
            time.sleep(0.002)  # 2ms
            return "result1"
        
        def mock_calc2():
            time.sleep(0.003)  # 3ms
            return "result2"
        
        def mock_calc3():
            time.sleep(0.001)  # 1ms
            return "result3"
        
        calculations = [mock_calc1, mock_calc2, mock_calc3]
        
        start_time = time.time()
        results = await performance_optimizer.parallel_risk_calculation(calculations)
        execution_time = (time.time() - start_time) * 1000
        
        assert len(results) == 3
        assert execution_time < 10  # Parallel should be faster than sequential
        assert all(result is not None for result in results)


class TestIntegrationFramework:
    """Test suite for Integration Framework"""
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def mock_var_calculator(self):
        mock_calc = Mock(spec=VaRCalculator)
        mock_calc.portfolio_value = 1000000.0
        mock_calc.positions = {'AAPL': Mock(), 'GOOGL': Mock()}
        mock_calc.get_latest_var.return_value = Mock(portfolio_var=20000.0)
        return mock_calc
    
    @pytest.fixture
    def mock_correlation_tracker(self):
        mock_tracker = Mock(spec=CorrelationTracker)
        mock_tracker.current_regime = CorrelationRegime.NORMAL
        mock_tracker.get_correlation_matrix.return_value = np.eye(3)
        return mock_tracker
    
    @pytest.fixture
    def integration_config(self):
        return {
            'agent': {
                'var_breach_threshold': 0.02,
                'max_response_time_ms': 10.0
            },
            'performance': {
                'response_time_target_ms': 10.0,
                'cache_size': 100
            },
            'actions': {
                'max_execution_time_ms': 5000
            },
            'assessment': {
                'assessment_interval_ms': 100
            },
            'stress_detection': {}
        }
    
    @pytest.fixture
    def integration_framework(self, mock_var_calculator, mock_correlation_tracker, 
                            event_bus, integration_config):
        return RiskMonitorIntegration(
            var_calculator=mock_var_calculator,
            correlation_tracker=mock_correlation_tracker,
            event_bus=event_bus,
            config=integration_config
        )
    
    @pytest.mark.asyncio
    async def test_integration_startup(self, integration_framework):
        """Test integration framework startup"""
        success = await integration_framework.start_integration()
        
        assert success is True
        assert integration_framework.integration_active is True
        
        # Cleanup
        await integration_framework.stop_integration()
    
    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, integration_framework):
        """Test component health monitoring"""
        await integration_framework.start_integration()
        
        # Wait for health check
        await asyncio.sleep(0.1)
        
        status = integration_framework.get_integration_status()
        
        assert 'component_health' in status
        assert 'var_calculator' in status['component_health']
        assert 'correlation_tracker' in status['component_health']
        
        await integration_framework.stop_integration()
    
    @pytest.mark.asyncio
    async def test_end_to_end_risk_processing(self, integration_framework):
        """Test complete end-to-end risk processing"""
        await integration_framework.start_integration()
        
        risk_state = RiskState(
            account_equity_normalized=1.0,
            open_positions_count=5,
            volatility_regime=0.5,
            correlation_risk=0.3,
            var_estimate_5pct=0.02,
            current_drawdown_pct=0.02,
            margin_usage_pct=0.5,
            time_of_day_risk=0.5,
            market_stress_level=0.3,
            liquidity_conditions=0.8
        )
        
        action, confidence = await integration_framework.process_risk_state(risk_state)
        
        assert action is not None
        assert 0 <= confidence <= 1.0
        
        await integration_framework.stop_integration()
    
    @pytest.mark.asyncio
    async def test_integration_test_suite(self, integration_framework):
        """Test comprehensive integration test"""
        await integration_framework.start_integration()
        
        test_results = await integration_framework.run_integration_test()
        
        assert 'var_integration' in test_results
        assert 'correlation_integration' in test_results
        assert 'risk_processing' in test_results
        assert 'performance_optimization' in test_results
        
        # Most tests should pass
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        assert passed_tests / total_tests >= 0.75  # At least 75% pass rate
        
        await integration_framework.stop_integration()
    
    def test_system_diagnostics(self, integration_framework):
        """Test system diagnostics reporting"""
        diagnostics = integration_framework.get_system_diagnostics()
        
        assert 'var_calculator_summary' in diagnostics
        assert 'correlation_regime' in diagnostics
        assert 'recent_breaches' in diagnostics
        assert 'recent_stress_events' in diagnostics
        assert 'emergency_action_summary' in diagnostics


@pytest.mark.asyncio
async def test_complete_emergency_scenario():
    """
    Integration test for complete emergency scenario
    
    Tests the entire emergency response pipeline from detection to action.
    """
    # Setup
    event_bus = EventBus()
    
    mock_var_calculator = Mock(spec=VaRCalculator)
    mock_var_calculator.portfolio_value = 1000000.0
    mock_var_calculator.positions = {
        'AAPL': Mock(market_value=400000, quantity=2000, symbol='AAPL'),
        'GOOGL': Mock(market_value=600000, quantity=1000, symbol='GOOGL')
    }
    
    mock_correlation_tracker = Mock(spec=CorrelationTracker)
    mock_correlation_tracker.current_regime = CorrelationRegime.CRISIS
    
    config = {
        'var_breach_threshold': 0.02,
        'max_response_time_ms': 10.0,
        'position_reduction_pct': 0.5
    }
    
    # Create agent
    agent = RiskMonitorAgent(
        config=config,
        var_calculator=mock_var_calculator,
        correlation_tracker=mock_correlation_tracker,
        event_bus=event_bus
    )
    
    # Create crisis scenario
    crisis_risk_state = RiskState(
        account_equity_normalized=0.7,  # Portfolio down 30%
        open_positions_count=2,
        volatility_regime=0.9,  # Extreme volatility
        correlation_risk=0.95,  # Correlation shock
        var_estimate_5pct=0.08,  # 8% VaR - 4x normal
        current_drawdown_pct=0.25,  # 25% drawdown - emergency level
        margin_usage_pct=0.95,  # Critical margin
        time_of_day_risk=0.8,
        market_stress_level=0.95,  # Extreme stress
        liquidity_conditions=0.1  # Liquidity crisis
    )
    
    # Test emergency response
    start_time = time.time()
    action, confidence = agent.calculate_risk_action(crisis_risk_state)
    response_time = (time.time() - start_time) * 1000
    
    # Verify emergency response
    assert action == EmergencyAction.CLOSE_ALL.value  # Should trigger emergency liquidation
    assert confidence > 0.8  # High confidence in decision
    assert response_time < 10.0  # Within response time target
    
    # Test action execution
    execution_result = await agent.execute_emergency_action(
        EmergencyAction.CLOSE_ALL,
        crisis_risk_state,
        "Crisis emergency liquidation"
    )
    
    assert execution_result.success is True
    assert execution_result.risk_reduction_pct == 1.0  # 100% liquidation
    assert execution_result.execution_time_ms < 5000  # Within execution time limit
    
    # Verify agent state
    assert len(agent.emergency_actions_taken) == 1
    assert agent.current_risk_severity == RiskSeverity.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])