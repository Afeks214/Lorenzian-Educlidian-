"""
Black Swan Simulation Test Suite for VaR Model

This comprehensive test suite validates the enhanced VaR model's ability to:
1. Detect extreme correlation regime changes
2. Respond appropriately to correlation shocks
3. Maintain performance under stress conditions
4. Execute automated risk reduction protocols

Test Scenarios:
- Market crash with instant correlation spike to 0.95
- Gradual correlation buildup over time
- Flash correlation events
- Performance under extreme conditions
- Recovery and reset procedures
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio
from typing import List, Dict

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime, CorrelationShock
from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData


class TestBlackSwanScenarios:
    """Test suite for black swan correlation scenarios"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def correlation_tracker(self, event_bus):
        """Create correlation tracker with test configuration"""
        tracker = CorrelationTracker(
            event_bus=event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.5,
            shock_window_minutes=10,
            max_correlation_history=100
        )
        
        # Initialize with test assets
        test_assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        tracker.initialize_assets(test_assets)
        
        return tracker
    
    @pytest.fixture
    def var_calculator(self, correlation_tracker, event_bus):
        """Create VaR calculator for testing"""
        return VaRCalculator(
            correlation_tracker=correlation_tracker,
            event_bus=event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 10]
        )
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample portfolio positions"""
        return {
            'AAPL': PositionData('AAPL', 1000, 180000, 180.0, 0.25),
            'GOOGL': PositionData('GOOGL', 500, 150000, 300.0, 0.30),
            'MSFT': PositionData('MSFT', 800, 240000, 300.0, 0.22),
            'TSLA': PositionData('TSLA', 200, 50000, 250.0, 0.45),
            'NVDA': PositionData('NVDA', 300, 120000, 400.0, 0.40)
        }
    
    def _setup_portfolio(self, var_calculator, positions):
        """Helper to setup portfolio in VaR calculator"""
        var_calculator.positions = positions
        var_calculator.portfolio_value = sum(pos.market_value for pos in positions.values())
    
    def _generate_normal_returns(self, correlation_tracker, n_periods=60):
        """Generate normal market returns for baseline"""
        assets = correlation_tracker.assets
        
        for period in range(n_periods):
            timestamp = datetime.now() - timedelta(minutes=n_periods-period)
            
            for asset in assets:
                # Generate normal returns (low correlation)
                base_return = np.random.normal(0, 0.01)  # 1% daily vol
                
                # Add some realistic correlation structure
                if asset in ['AAPL', 'MSFT']:  # Tech correlation
                    correlation_factor = np.random.normal(0, 0.003)
                    return_value = base_return + correlation_factor
                else:
                    return_value = base_return
                
                # Simulate price update
                if len(correlation_tracker.asset_returns[asset]) > 0:
                    last_price = correlation_tracker.asset_returns[asset][-1][1]
                    new_price = last_price * (1 + return_value)
                else:
                    new_price = 100.0 * (1 + return_value)
                
                # Create mock bar data
                bar_data = Mock()
                bar_data.symbol = asset
                bar_data.timestamp = timestamp
                bar_data.close = new_price
                
                # Send event
                event = Event(
                    event_type=EventType.NEW_5MIN_BAR,
                    timestamp=timestamp,
                    payload=bar_data,
                    source='TestGenerator'
                )
                
                correlation_tracker._handle_price_update(event)
    
    def test_correlation_shock_detection(self, correlation_tracker):
        """Test detection of sudden correlation spike"""
        
        # Generate normal market conditions
        self._generate_normal_returns(correlation_tracker, 50)
        
        # Verify normal regime
        assert correlation_tracker.current_regime == CorrelationRegime.NORMAL
        baseline_corr = correlation_tracker._calculate_average_correlation()
        assert baseline_corr < 0.3, f"Baseline correlation too high: {baseline_corr}"
        
        # Simulate black swan event - all correlations spike to 0.95
        original_matrix = correlation_tracker.simulate_correlation_shock(0.95)
        
        # Verify shock detection
        assert len(correlation_tracker.shock_alerts) > 0, "Correlation shock not detected"
        
        latest_shock = correlation_tracker.shock_alerts[-1]
        assert latest_shock.severity in ["HIGH", "CRITICAL"]
        assert latest_shock.correlation_change > 0.5
        assert correlation_tracker.current_regime in [CorrelationRegime.CRISIS, CorrelationRegime.ELEVATED]
        
        # Verify average correlation increased dramatically
        shocked_corr = correlation_tracker._calculate_average_correlation()
        assert shocked_corr > 0.8, f"Shocked correlation not high enough: {shocked_corr}"
        
        print(f"✓ Black swan detected: {baseline_corr:.3f} -> {shocked_corr:.3f}")
        print(f"✓ Shock severity: {latest_shock.severity}")
        print(f"✓ Regime changed to: {correlation_tracker.current_regime.value}")
    
    def test_automated_leverage_reduction(self, correlation_tracker):
        """Test automated leverage reduction during correlation shock"""
        
        # Setup leverage callback
        leverage_actions = []
        def leverage_callback(new_leverage):
            leverage_actions.append(new_leverage)
        
        correlation_tracker.register_leverage_callback(leverage_callback)
        correlation_tracker.current_leverage = 4.0  # High leverage
        
        # Generate normal conditions
        self._generate_normal_returns(correlation_tracker, 30)
        
        # Trigger correlation shock
        correlation_tracker.simulate_correlation_shock(0.90)
        
        # Verify leverage reduction was triggered
        assert len(leverage_actions) > 0, "Leverage reduction not triggered"
        assert len(correlation_tracker.risk_actions) > 0, "Risk action not recorded"
        
        latest_action = correlation_tracker.risk_actions[-1]
        assert latest_action.action_type == "LEVERAGE_REDUCTION"
        assert latest_action.leverage_after < latest_action.leverage_before
        assert latest_action.manual_reset_required is True
        
        # Verify 50% reduction
        expected_leverage = 4.0 * 0.5
        assert abs(latest_action.leverage_after - expected_leverage) < 0.1
        
        print(f"✓ Leverage reduced: {latest_action.leverage_before} -> {latest_action.leverage_after}")
        print(f"✓ Manual reset required: {latest_action.manual_reset_required}")
    
    @pytest.mark.asyncio
    async def test_var_regime_adjustment(self, correlation_tracker, var_calculator, sample_positions):
        """Test VaR adjustment during correlation regimes"""
        
        # Setup portfolio
        self._setup_portfolio(var_calculator, sample_positions)
        
        # Generate normal returns
        self._generate_normal_returns(correlation_tracker, 50)
        
        # Calculate baseline VaR
        normal_var = await var_calculator.calculate_var(confidence_level=0.95, time_horizon=1)
        assert normal_var is not None, "Failed to calculate normal VaR"
        baseline_var = normal_var.portfolio_var
        
        # Trigger correlation shock
        correlation_tracker.simulate_correlation_shock(0.95)
        
        # Calculate shocked VaR
        shocked_var = await var_calculator.calculate_var(confidence_level=0.95, time_horizon=1)
        assert shocked_var is not None, "Failed to calculate shocked VaR"
        
        # Verify VaR increased significantly
        var_increase = shocked_var.portfolio_var / baseline_var
        assert var_increase > 1.3, f"VaR increase insufficient: {var_increase:.2f}x"
        assert shocked_var.correlation_regime in ["CRISIS", "ELEVATED"]
        
        print(f"✓ VaR increased {var_increase:.2f}x during correlation shock")
        print(f"✓ Baseline VaR: ${baseline_var:,.0f}")
        print(f"✓ Shocked VaR: ${shocked_var.portfolio_var:,.0f}")
    
    def test_gradual_correlation_buildup(self, correlation_tracker):
        """Test detection of gradual correlation increases"""
        
        # Generate baseline
        self._generate_normal_returns(correlation_tracker, 30)
        baseline_corr = correlation_tracker._calculate_average_correlation()
        
        # Gradually increase correlations
        for step in range(10):
            correlation_level = 0.3 + (step * 0.07)  # 0.3 to 0.93
            correlation_tracker.simulate_correlation_shock(correlation_level)
            
            if step >= 7:  # Should trigger shock detection at ~0.8 correlation
                if len(correlation_tracker.shock_alerts) > 0:
                    break
        
        # Verify gradual buildup was detected
        assert len(correlation_tracker.shock_alerts) > 0, "Gradual correlation buildup not detected"
        final_corr = correlation_tracker._calculate_average_correlation()
        assert final_corr > 0.7, f"Final correlation not high enough: {final_corr}"
        
        print(f"✓ Gradual buildup detected: {baseline_corr:.3f} -> {final_corr:.3f}")
    
    def test_flash_correlation_event(self, correlation_tracker):
        """Test detection of brief correlation spikes"""
        
        # Generate normal conditions
        self._generate_normal_returns(correlation_tracker, 40)
        
        # Flash spike - high correlation for short period
        correlation_tracker.simulate_correlation_shock(0.92)
        
        # Verify immediate detection
        assert len(correlation_tracker.shock_alerts) > 0, "Flash correlation event not detected"
        
        latest_shock = correlation_tracker.shock_alerts[-1]
        time_to_detection = (datetime.now() - latest_shock.timestamp).total_seconds()
        assert time_to_detection < 1.0, f"Detection too slow: {time_to_detection:.2f}s"
        
        print(f"✓ Flash event detected in {time_to_detection:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_performance_under_stress(self, correlation_tracker, var_calculator, sample_positions):
        """Test performance during extreme correlation conditions"""
        
        # Setup portfolio
        self._setup_portfolio(var_calculator, sample_positions)
        
        # Generate data
        self._generate_normal_returns(correlation_tracker, 100)
        
        # Performance test under stress
        performance_times = []
        
        for i in range(10):
            # Alternate between normal and shocked correlations
            if i % 2 == 0:
                correlation_tracker.simulate_correlation_shock(0.95)
            else:
                correlation_tracker.simulate_correlation_shock(0.2)
            
            # Measure VaR calculation time
            start_time = datetime.now()
            var_result = await var_calculator.calculate_var()
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_times.append(calc_time)
        
        # Verify performance target met
        avg_time = np.mean(performance_times)
        max_time = np.max(performance_times)
        
        assert avg_time < 5.0, f"Average calculation time exceeded target: {avg_time:.2f}ms"
        assert max_time < 10.0, f"Maximum calculation time excessive: {max_time:.2f}ms"
        
        print(f"✓ Performance under stress - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
    
    def test_manual_reset_procedure(self, correlation_tracker):
        """Test manual reset of risk controls"""
        
        # Generate shock and trigger risk reduction
        self._generate_normal_returns(correlation_tracker, 30)
        correlation_tracker.simulate_correlation_shock(0.95)
        
        # Verify manual reset required
        assert correlation_tracker.manual_reset_required is True
        assert correlation_tracker.current_regime != CorrelationRegime.NORMAL
        
        # Perform manual reset
        reset_success = correlation_tracker.manual_reset_risk_controls(
            operator_id="test_operator",
            reason="Post-investigation reset after false alarm"
        )
        
        # Verify reset
        assert reset_success is True
        assert correlation_tracker.manual_reset_required is False
        assert correlation_tracker.current_regime == CorrelationRegime.NORMAL
        assert correlation_tracker.max_allowed_leverage == 4.0
        
        print("✓ Manual reset procedure successful")
    
    def test_multiple_concurrent_shocks(self, correlation_tracker):
        """Test handling of multiple correlation shocks in sequence"""
        
        # Generate baseline
        self._generate_normal_returns(correlation_tracker, 30)
        
        # Trigger multiple shocks rapidly
        shock_levels = [0.85, 0.90, 0.95, 0.88, 0.92]
        
        for level in shock_levels:
            correlation_tracker.simulate_correlation_shock(level)
        
        # Verify all shocks detected and system remains stable
        assert len(correlation_tracker.shock_alerts) >= len(shock_levels)
        
        # Verify system didn't crash
        regime_status = correlation_tracker.get_regime_status()
        assert regime_status['current_regime'] in ['CRISIS', 'ELEVATED']
        assert regime_status['recent_shocks'] >= len(shock_levels)
        
        print(f"✓ Handled {len(shock_levels)} concurrent shocks")
        print(f"✓ Total shocks detected: {regime_status['recent_shocks']}")
    
    @pytest.mark.asyncio
    async def test_component_var_accuracy(self, correlation_tracker, var_calculator, sample_positions):
        """Test accuracy of component VaR calculations during shocks"""
        
        # Setup portfolio
        self._setup_portfolio(var_calculator, sample_positions)
        self._generate_normal_returns(correlation_tracker, 50)
        
        # Calculate VaR before shock
        normal_var = await var_calculator.calculate_var()
        
        # Trigger shock and recalculate
        correlation_tracker.simulate_correlation_shock(0.90)
        shocked_var = await var_calculator.calculate_var()
        
        # Verify component VaRs sum approximately to total
        component_sum = sum(abs(var) for var in shocked_var.component_vars.values())
        portfolio_var = abs(shocked_var.portfolio_var)
        
        # Allow for some diversification benefit
        ratio = component_sum / portfolio_var
        assert 1.0 <= ratio <= 2.0, f"Component VaR sum ratio unrealistic: {ratio:.2f}"
        
        # Verify all positions have non-zero component VaR
        for symbol, component_var in shocked_var.component_vars.items():
            assert component_var != 0, f"Zero component VaR for {symbol}"
        
        print(f"✓ Component VaR accuracy validated (ratio: {ratio:.2f})")
    
    def test_event_bus_integration(self, correlation_tracker, event_bus):
        """Test proper event publishing during correlation shocks"""
        
        # Setup event listener
        events_received = []
        
        def event_listener(event):
            events_received.append(event)
        
        event_bus.subscribe(EventType.RISK_BREACH, event_listener)
        event_bus.subscribe(EventType.VAR_UPDATE, event_listener)
        
        # Generate normal conditions then shock
        self._generate_normal_returns(correlation_tracker, 30)
        correlation_tracker.simulate_correlation_shock(0.95)
        
        # Verify events were published
        risk_breach_events = [e for e in events_received if e.event_type == EventType.RISK_BREACH]
        var_update_events = [e for e in events_received if e.event_type == EventType.VAR_UPDATE]
        
        assert len(risk_breach_events) > 0, "Risk breach event not published"
        assert len(var_update_events) > 0, "VaR update events not published"
        
        # Verify event content
        latest_breach = risk_breach_events[-1]
        assert latest_breach.payload['type'] == 'CORRELATION_SHOCK'
        assert 'severity' in latest_breach.payload
        assert 'manual_reset_required' in latest_breach.payload
        
        print(f"✓ Event integration validated ({len(events_received)} events)")


class TestPerformanceBenchmarks:
    """Performance benchmark tests for correlation tracking"""
    
    @pytest.fixture
    def large_universe_tracker(self, event_bus):
        """Create tracker with large asset universe"""
        tracker = CorrelationTracker(
            event_bus=event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.5,
            shock_window_minutes=10
        )
        
        # Large universe (S&P 500 simulation)
        large_assets = [f"STOCK_{i:03d}" for i in range(100)]
        tracker.initialize_assets(large_assets)
        
        return tracker
    
    def test_large_universe_performance(self, large_universe_tracker):
        """Test performance with large asset universe"""
        
        tracker = large_universe_tracker
        
        # Generate returns for large universe
        performance_times = []
        
        for period in range(50):
            start_time = datetime.now()
            
            # Generate correlated returns for all assets
            timestamp = datetime.now() - timedelta(minutes=50-period)
            
            for asset in tracker.assets:
                # Simple correlated return generation
                base_return = np.random.normal(0, 0.01)
                correlation_factor = np.random.normal(0, 0.005)
                return_value = base_return + correlation_factor
                
                # Simulate price update
                if len(tracker.asset_returns[asset]) > 0:
                    last_price = tracker.asset_returns[asset][-1][1]
                    new_price = last_price * (1 + return_value)
                else:
                    new_price = 100.0 * (1 + return_value)
                
                bar_data = Mock()
                bar_data.symbol = asset
                bar_data.timestamp = timestamp
                bar_data.close = new_price
                
                event = Event(
                    event_type=EventType.NEW_5MIN_BAR,
                    timestamp=timestamp,
                    payload=bar_data,
                    source='PerfTest'
                )
                
                tracker._handle_price_update(event)
            
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_times.append(calc_time)
        
        # Verify performance
        avg_time = np.mean(performance_times)
        max_time = np.max(performance_times)
        
        # More lenient targets for large universe
        assert avg_time < 50.0, f"Large universe avg time too slow: {avg_time:.2f}ms"
        assert max_time < 100.0, f"Large universe max time too slow: {max_time:.2f}ms"
        
        print(f"✓ Large universe performance - Assets: {len(tracker.assets)}")
        print(f"✓ Avg time: {avg_time:.2f}ms, Max time: {max_time:.2f}ms")


if __name__ == "__main__":
    """Run black swan tests directly"""
    
    print("🦢 Starting Black Swan VaR Model Tests...")
    print("=" * 50)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])