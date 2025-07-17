"""
CL Risk Management Integration Tests
===================================

Comprehensive integration tests for the CL crude oil risk management system.
Tests all components working together including risk manager, position sizing,
portfolio management, execution controls, and monitoring dashboard.

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the CL risk management components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.cl_risk_manager import CLRiskManager, CLRiskMetrics, CLPosition, CLRiskAlert, RiskLevel
from execution.cl_position_sizing import CLPositionSizer, SizingMethod, SizingParameters
from execution.cl_risk_controls import CLRiskControlSystem, RiskControl, RiskControlType, ControlTrigger
from execution.cl_market_considerations import CLMarketAnalyzer, InventoryData, GeopoliticalEvent
from execution.cl_portfolio_manager import CLPortfolioManager, PortfolioPosition, AssetClass
from execution.cl_execution_controls import CLExecutionEngine, ExecutionOrder, OrderType, ExecutionAlgorithm
from execution.cl_risk_dashboard import CLRiskDashboard, DashboardMetric, DashboardAlert, AlertSeverity

class TestCLRiskManagementIntegration:
    """Integration tests for CL risk management system"""
    
    @pytest.fixture
    def risk_config(self):
        """Risk management configuration"""
        return {
            'initial_capital': 1000000,
            'max_risk_per_trade': 0.02,
            'max_position_size': 0.10,
            'max_drawdown': 0.20,
            'daily_loss_limit': 0.05,
            'leverage_limit': 3.0,
            'cl_contract_size': 1000,
            'cl_tick_size': 0.01,
            'cl_tick_value': 10.0,
            'volatility_lookback': 20,
            'kelly_criterion': {
                'enabled': True,
                'max_kelly_fraction': 0.25,
                'lookback_period': 100,
                'safety_factor': 0.5
            },
            'market_condition_adjustments': {
                'trending_market_multiplier': 1.2,
                'ranging_market_multiplier': 0.8,
                'high_volatility_multiplier': 0.7,
                'low_volatility_multiplier': 1.1
            },
            'stop_loss': {
                'enabled': True,
                'default_percent': 0.02,
                'volatility_adjusted': True
            },
            'take_profit': {
                'enabled': True,
                'risk_reward_ratio': 1.5
            },
            'drawdown_protection': {
                'enabled': True,
                'max_drawdown_percent': 0.20
            },
            'circuit_breakers': {
                'level_1': 0.10,
                'level_2': 0.15,
                'level_3': 0.20
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            'symbol': 'CL_202501',
            'close': 75.50,
            'open': 75.30,
            'high': 75.80,
            'low': 75.20,
            'volume': 500000,
            'timestamp': datetime.now().isoformat(),
            'atr_20': 1.50,
            'atr_50': 1.40,
            'volatility': 0.025,
            'bid': 75.48,
            'ask': 75.52,
            'spread': 0.04,
            'prices': [
                {'open': 75.0, 'high': 75.5, 'low': 74.8, 'close': 75.2, 'volume': 1000},
                {'open': 75.2, 'high': 75.6, 'low': 75.0, 'close': 75.4, 'volume': 1200},
                {'open': 75.4, 'high': 75.8, 'low': 75.2, 'close': 75.5, 'volume': 1100}
            ]
        }
    
    @pytest.fixture
    def sample_signal_data(self):
        """Sample signal data for testing"""
        return {
            'symbol': 'CL_202501',
            'direction': 'long',
            'confidence': 0.75,
            'strength': 0.8,
            'win_rate': 0.6,
            'avg_win': 0.025,
            'avg_loss': 0.015,
            'expected_return': 0.018
        }
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for testing"""
        return {
            'total_value': 1000000,
            'cash_balance': 500000,
            'positions': {
                'CL_202501': {
                    'symbol': 'CL_202501',
                    'quantity': 10,
                    'entry_price': 75.0,
                    'current_price': 75.5,
                    'side': 'long',
                    'value': 755000,
                    'unrealized_pnl': 5000
                }
            },
            'daily_pnl': 5000,
            'current_drawdown': 0.01,
            'total_exposure': 755000,
            'returns_history': [0.01, -0.005, 0.02, 0.015, -0.01]
        }
    
    @pytest.fixture
    async def integrated_system(self, risk_config):
        """Create integrated risk management system"""
        # Initialize components
        risk_manager = CLRiskManager(risk_config)
        position_sizer = CLPositionSizer(risk_config)
        risk_controls = CLRiskControlSystem(risk_config)
        market_analyzer = CLMarketAnalyzer(risk_config)
        portfolio_manager = CLPortfolioManager(risk_config)
        execution_engine = CLExecutionEngine(risk_config)
        dashboard = CLRiskDashboard(risk_config)
        
        # Connect components
        dashboard.connect_components(
            risk_manager=risk_manager,
            portfolio_manager=portfolio_manager,
            execution_engine=execution_engine,
            market_analyzer=market_analyzer
        )
        
        return {
            'risk_manager': risk_manager,
            'position_sizer': position_sizer,
            'risk_controls': risk_controls,
            'market_analyzer': market_analyzer,
            'portfolio_manager': portfolio_manager,
            'execution_engine': execution_engine,
            'dashboard': dashboard
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_trade_flow(self, integrated_system, sample_market_data, sample_signal_data, sample_portfolio_data):
        """Test complete end-to-end trade flow"""
        # Get components
        risk_manager = integrated_system['risk_manager']
        position_sizer = integrated_system['position_sizer']
        risk_controls = integrated_system['risk_controls']
        portfolio_manager = integrated_system['portfolio_manager']
        execution_engine = integrated_system['execution_engine']
        
        # Step 1: Calculate position size
        sizing_result = await position_sizer.calculate_position_size(
            sample_signal_data, sample_market_data, sample_portfolio_data
        )
        
        assert sizing_result['recommended_size'] > 0
        assert 'risk_metrics' in sizing_result
        assert sizing_result['confidence_score'] > 0
        
        # Step 2: Validate trade with risk manager
        trade_data = {
            'symbol': 'CL_202501',
            'size': sizing_result['recommended_size'],
            'price': sample_market_data['close'],
            'side': 'long',
            'stop_loss': sizing_result['risk_metrics']['stop_loss'],
            'take_profit': sizing_result['risk_metrics']['take_profit']
        }
        
        validation_result = await risk_manager.validate_trade(trade_data)
        assert validation_result['approved'] is True
        
        # Step 3: Check risk controls
        position_data = {
            'symbol': 'CL_202501',
            'size': trade_data['size'],
            'entry_price': trade_data['price'],
            'side': trade_data['side'],
            'stop_loss': trade_data['stop_loss'],
            'take_profit': trade_data['take_profit']
        }
        
        controls_result = await risk_controls.evaluate_risk_controls(
            position_data, sample_market_data, sample_portfolio_data
        )
        assert controls_result['controls_evaluated'] > 0
        
        # Step 4: Add position to portfolio
        portfolio_result = await portfolio_manager.add_position({
            'symbol': 'CL_202501',
            'quantity': trade_data['size'],
            'entry_price': trade_data['price'],
            'side': trade_data['side']
        })
        assert portfolio_result['success'] is True
        
        # Step 5: Execute trade
        execution_order = ExecutionOrder(
            order_id=f"test_order_{int(datetime.now().timestamp())}",
            symbol='CL_202501',
            side='buy',
            quantity=trade_data['size'],
            order_type=OrderType.MARKET,
            algorithm=ExecutionAlgorithm.BALANCED
        )
        
        execution_result = await execution_engine.submit_order(execution_order)
        assert execution_result['success'] is True
        
        # Verify complete flow
        assert sizing_result['recommended_size'] == trade_data['size']
        assert validation_result['approved'] is True
        assert portfolio_result['success'] is True
        assert execution_result['success'] is True
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self, integrated_system, sample_market_data, sample_portfolio_data):
        """Test risk limit enforcement across components"""
        risk_manager = integrated_system['risk_manager']
        position_sizer = integrated_system['position_sizer']
        
        # Create large position signal
        large_signal = {
            'symbol': 'CL_202501',
            'direction': 'long',
            'confidence': 0.9,
            'strength': 1.0,
            'win_rate': 0.8,
            'avg_win': 0.05,
            'avg_loss': 0.01
        }
        
        # Test position sizing with risk limits
        sizing_result = await position_sizer.calculate_position_size(
            large_signal, sample_market_data, sample_portfolio_data
        )
        
        # Should be capped by risk limits
        max_risk_amount = sample_portfolio_data['total_value'] * 0.02  # 2% max risk
        assert sizing_result['risk_metrics']['risk_amount'] <= max_risk_amount * 1.1  # Allow small tolerance
        
        # Test validation with excessive size
        oversized_trade = {
            'symbol': 'CL_202501',
            'size': 1000,  # Very large size
            'price': sample_market_data['close'],
            'side': 'long',
            'stop_loss': 73.0,
            'take_profit': 78.0
        }
        
        validation_result = await risk_manager.validate_trade(oversized_trade)
        assert validation_result['approved'] is False
        assert len(validation_result['warnings']) > 0
    
    @pytest.mark.asyncio
    async def test_market_condition_adjustments(self, integrated_system, sample_market_data, sample_signal_data, sample_portfolio_data):
        """Test market condition adjustments"""
        position_sizer = integrated_system['position_sizer']
        market_analyzer = integrated_system['market_analyzer']
        
        # Test different market conditions
        market_conditions = ['trending', 'ranging', 'high_volatility', 'low_volatility']
        
        for condition in market_conditions:
            # Mock market condition
            with patch.object(market_analyzer, '_assess_market_condition', return_value=condition):
                sizing_result = await position_sizer.calculate_position_size(
                    sample_signal_data, sample_market_data, sample_portfolio_data
                )
                
                # Verify adjustment was applied
                assert 'market_condition' in sizing_result['adjustments_applied']
                assert sizing_result['adjustments_applied']['market_condition'] == condition
                assert sizing_result['recommended_size'] > 0
    
    @pytest.mark.asyncio
    async def test_inventory_report_impact(self, integrated_system, sample_market_data):
        """Test inventory report impact on risk decisions"""
        market_analyzer = integrated_system['market_analyzer']
        
        # Create inventory data
        inventory_data = InventoryData(
            report_type='eia_crude',
            actual_change=-2000000,  # 2M barrel draw
            expected_change=-1000000,  # 1M barrel expected
            previous_change=-500000,
            surprise=-1000000,  # Bullish surprise
            timestamp=datetime.now()
        )
        
        # Analyze inventory impact
        impact_result = await market_analyzer.analyze_inventory_impact(inventory_data)
        
        assert impact_result['significance'] > 0
        assert impact_result['trading_recommendation']['action'] == 'bullish'
        assert impact_result['trading_recommendation']['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_geopolitical_risk_assessment(self, integrated_system):
        """Test geopolitical risk assessment"""
        market_analyzer = integrated_system['market_analyzer']
        
        # Mock news data
        news_data = [
            {
                'headline': 'Iran threatens to close strait of hormuz',
                'content': 'Geopolitical tensions rise in middle east',
                'sentiment': -0.8
            },
            {
                'headline': 'OPEC considers production cuts',
                'content': 'Oil cartel meeting scheduled',
                'sentiment': 0.6
            }
        ]
        
        market_data = {'volatility': 0.03}
        
        # Assess geopolitical risk
        risk_result = await market_analyzer.assess_geopolitical_risk(news_data, market_data)
        
        assert risk_result['overall_risk_score'] > 0
        assert 'risk_level' in risk_result
        assert len(risk_result['risk_components']) > 0
        assert len(risk_result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, integrated_system, sample_market_data, sample_portfolio_data):
        """Test circuit breaker activation"""
        risk_controls = integrated_system['risk_controls']
        
        # Create high drawdown scenario
        high_drawdown_portfolio = sample_portfolio_data.copy()
        high_drawdown_portfolio['current_drawdown'] = 0.25  # 25% drawdown
        
        # Check circuit breakers
        circuit_result = await risk_controls._check_circuit_breakers(high_drawdown_portfolio)
        
        assert circuit_result['triggered'] is True
        assert circuit_result['level'] == 'level_3'  # Should trigger level 3
        assert circuit_result['action'] == 'stop_all_trading'
    
    @pytest.mark.asyncio
    async def test_execution_cost_estimation(self, integrated_system, sample_market_data):
        """Test execution cost estimation"""
        execution_engine = integrated_system['execution_engine']
        
        # Create test order
        order = ExecutionOrder(
            order_id="test_cost_estimation",
            symbol='CL_202501',
            side='buy',
            quantity=50,
            order_type=OrderType.MARKET,
            algorithm=ExecutionAlgorithm.BALANCED
        )
        
        # Calculate execution parameters
        execution_params = await execution_engine._calculate_execution_parameters(order)
        
        # Estimate costs
        cost_estimate = await execution_engine._estimate_execution_cost(order, execution_params)
        
        assert cost_estimate['total_cost'] > 0
        assert cost_estimate['cost_bps'] > 0
        assert 'market_impact' in cost_estimate
        assert 'slippage' in cost_estimate
        assert 'commission' in cost_estimate
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, integrated_system, sample_portfolio_data):
        """Test portfolio rebalancing"""
        portfolio_manager = integrated_system['portfolio_manager']
        
        # Add some positions to trigger rebalancing
        await portfolio_manager.add_position({
            'symbol': 'CL_202501',
            'quantity': 100,
            'entry_price': 75.0,
            'side': 'long'
        })
        
        # Check if rebalancing is needed
        rebalance_check = await portfolio_manager._check_rebalancing_needed()
        
        if rebalance_check['rebalance_needed']:
            # Perform rebalancing
            rebalance_result = await portfolio_manager.rebalance_portfolio()
            
            assert rebalance_result['success'] is True
            assert rebalance_result['trades_planned'] >= 0
            assert rebalance_result['trades_executed'] >= 0
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, integrated_system, sample_market_data):
        """Test real-time monitoring and alerting"""
        dashboard = integrated_system['dashboard']
        
        # Start dashboard
        await dashboard.start_dashboard()
        
        # Wait for initial update
        await asyncio.sleep(0.1)
        
        # Get dashboard state
        dashboard_state = dashboard.get_dashboard_state()
        
        assert dashboard_state['system_status']['dashboard_running'] is True
        assert len(dashboard_state['metrics']) > 0
        assert 'alerts' in dashboard_state
        
        # Stop dashboard
        await dashboard.stop_dashboard()
        
        # Verify stopped
        final_state = dashboard.get_dashboard_state()
        assert final_state['system_status']['dashboard_running'] is False
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, integrated_system):
        """Test risk alert generation"""
        risk_manager = integrated_system['risk_manager']
        dashboard = integrated_system['dashboard']
        
        # Create high-risk scenario
        high_risk_data = {
            'symbol': 'CL_202501',
            'size': 200,  # Large size
            'price': 75.0,
            'side': 'long',
            'stop_loss': 70.0,  # Wide stop
            'take_profit': 80.0
        }
        
        # This should generate risk alerts
        validation_result = await risk_manager.validate_trade(high_risk_data)
        
        if not validation_result['approved']:
            # Should have warnings
            assert len(validation_result['warnings']) > 0
        
        # Check if dashboard generates alerts
        await dashboard._update_dashboard()
        
        dashboard_state = dashboard.get_dashboard_state()
        # May or may not have alerts depending on exact conditions
        assert 'alerts' in dashboard_state
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, integrated_system):
        """Test performance degradation detection"""
        execution_engine = integrated_system['execution_engine']
        
        # Simulate multiple orders with degrading performance
        orders = []
        for i in range(5):
            order = ExecutionOrder(
                order_id=f"perf_test_{i}",
                symbol='CL_202501',
                side='buy',
                quantity=10,
                order_type=OrderType.MARKET,
                algorithm=ExecutionAlgorithm.BALANCED
            )
            
            result = await execution_engine.submit_order(order)
            orders.append(result)
        
        # Check execution summary
        summary = execution_engine.get_execution_summary()
        
        assert summary['total_orders'] == 5
        assert summary['total_fills'] > 0
        assert summary['fill_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_stress_scenario_handling(self, integrated_system, sample_market_data):
        """Test system behavior under stress scenarios"""
        risk_manager = integrated_system['risk_manager']
        portfolio_manager = integrated_system['portfolio_manager']
        
        # Create stress scenario
        stress_market_data = sample_market_data.copy()
        stress_market_data['volatility'] = 0.10  # 10% volatility
        stress_market_data['spread'] = 0.20  # Wide spread
        stress_market_data['volume'] = 50000  # Low volume
        
        # Test position sizing under stress
        sizing_result = await integrated_system['position_sizer'].calculate_position_size(
            {
                'symbol': 'CL_202501',
                'direction': 'long',
                'confidence': 0.8,
                'strength': 0.7,
                'win_rate': 0.6,
                'avg_win': 0.03,
                'avg_loss': 0.02
            },
            stress_market_data,
            {'total_value': 1000000, 'positions': {}}
        )
        
        # Should reduce position size under stress
        assert sizing_result['recommended_size'] > 0
        assert 'volatility_adjustment' in sizing_result['adjustments_applied']
    
    @pytest.mark.asyncio
    async def test_correlation_risk_management(self, integrated_system):
        """Test correlation risk management"""
        portfolio_manager = integrated_system['portfolio_manager']
        
        # Add multiple correlated positions
        positions = [
            {'symbol': 'CL_202501', 'quantity': 50, 'entry_price': 75.0, 'side': 'long'},
            {'symbol': 'BZ_202501', 'quantity': 30, 'entry_price': 78.0, 'side': 'long'},
            {'symbol': 'HO_202501', 'quantity': 20, 'entry_price': 2.10, 'side': 'long'}
        ]
        
        for position in positions:
            result = await portfolio_manager.add_position(position)
            assert result['success'] is True
        
        # Check correlation risk
        risk_metrics = portfolio_manager.get_risk_metrics()
        
        assert 'correlation_risk' in risk_metrics
        assert risk_metrics['correlation_risk'] >= 0
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, integrated_system):
        """Test system recovery after component failure"""
        dashboard = integrated_system['dashboard']
        
        # Start dashboard
        await dashboard.start_dashboard()
        
        # Simulate component failure
        original_risk_manager = dashboard.risk_manager
        dashboard.risk_manager = None
        
        # Update should handle missing component gracefully
        await dashboard._update_dashboard()
        
        dashboard_state = dashboard.get_dashboard_state()
        assert dashboard_state['system_status']['connected_components']['risk_manager'] is False
        
        # Restore component
        dashboard.risk_manager = original_risk_manager
        
        # Should recover
        await dashboard._update_dashboard()
        
        recovered_state = dashboard.get_dashboard_state()
        assert recovered_state['system_status']['connected_components']['risk_manager'] is True
        
        await dashboard.stop_dashboard()
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, integrated_system, sample_market_data, sample_portfolio_data):
        """Test data consistency across all components"""
        # Update market data in market analyzer
        market_analyzer = integrated_system['market_analyzer']
        
        # Simulate market data update
        await market_analyzer.analyze_session_liquidity()
        
        # Update portfolio in portfolio manager
        portfolio_manager = integrated_system['portfolio_manager']
        await portfolio_manager.add_position({
            'symbol': 'CL_202501',
            'quantity': 10,
            'entry_price': 75.0,
            'side': 'long'
        })
        
        # Get data from different components
        market_summary = market_analyzer.get_market_summary()
        portfolio_summary = portfolio_manager.get_portfolio_summary()
        
        # Verify consistency
        assert market_summary['timestamp'] is not None
        assert portfolio_summary['timestamp'] is not None
        assert portfolio_summary['num_positions'] > 0
    
    def test_configuration_validation(self, risk_config):
        """Test configuration validation"""
        # Test with valid config
        risk_manager = CLRiskManager(risk_config)
        assert risk_manager.max_risk_per_trade == 0.02
        
        # Test with invalid config
        invalid_config = risk_config.copy()
        invalid_config['max_risk_per_trade'] = -0.1  # Invalid negative risk
        
        # Should handle gracefully or raise appropriate error
        try:
            CLRiskManager(invalid_config)
        except Exception as e:
            assert isinstance(e, (ValueError, AssertionError))
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, integrated_system):
        """Test performance benchmarking"""
        execution_engine = integrated_system['execution_engine']
        
        # Measure execution performance
        start_time = datetime.now()
        
        # Execute multiple orders
        orders = []
        for i in range(10):
            order = ExecutionOrder(
                order_id=f"bench_{i}",
                symbol='CL_202501',
                side='buy',
                quantity=5,
                order_type=OrderType.MARKET,
                algorithm=ExecutionAlgorithm.BALANCED
            )
            
            result = await execution_engine.submit_order(order)
            orders.append(result)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Performance should be reasonable
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Check execution metrics
        summary = execution_engine.get_execution_summary()
        assert summary['total_orders'] == 10
        assert summary['fill_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, integrated_system):
        """Test memory usage optimization"""
        dashboard = integrated_system['dashboard']
        
        # Generate lots of data
        for i in range(1000):
            await dashboard._update_metrics()
        
        # Check memory usage is reasonable
        metrics_history_size = sum(len(history) for history in dashboard.metrics_history.values())
        assert metrics_history_size <= 50000  # Should be bounded
        
        # Check alerts are cleaned up
        assert len(dashboard.alerts_history) <= 1000  # Should be bounded
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self, integrated_system):
        """Test edge case handling"""
        position_sizer = integrated_system['position_sizer']
        
        # Test with zero volatility
        zero_vol_data = {
            'symbol': 'CL_202501',
            'close': 75.0,
            'atr_20': 0.0,
            'atr_50': 0.0,
            'volatility': 0.0,
            'prices': []
        }
        
        signal_data = {
            'symbol': 'CL_202501',
            'direction': 'long',
            'confidence': 0.5,
            'strength': 0.5,
            'win_rate': 0.5,
            'avg_win': 0.02,
            'avg_loss': 0.01
        }
        
        portfolio_data = {'total_value': 1000000, 'positions': {}}
        
        # Should handle gracefully
        result = await position_sizer.calculate_position_size(
            signal_data, zero_vol_data, portfolio_data
        )
        
        assert result['recommended_size'] >= 0
        assert 'error' not in result or result['error'] is None


@pytest.mark.asyncio
async def test_full_system_integration():
    """Test full system integration with all components"""
    # This is a comprehensive integration test
    config = {
        'initial_capital': 1000000,
        'max_risk_per_trade': 0.02,
        'max_position_size': 0.10,
        'max_drawdown': 0.20,
        'daily_loss_limit': 0.05,
        'leverage_limit': 3.0,
        'cl_contract_size': 1000,
        'cl_tick_size': 0.01,
        'cl_tick_value': 10.0,
        'refresh_interval': 1,
        'kelly_criterion': {'enabled': True, 'max_kelly_fraction': 0.25}
    }
    
    # Initialize all components
    risk_manager = CLRiskManager(config)
    position_sizer = CLPositionSizer(config)
    risk_controls = CLRiskControlSystem(config)
    market_analyzer = CLMarketAnalyzer(config)
    portfolio_manager = CLPortfolioManager(config)
    execution_engine = CLExecutionEngine(config)
    dashboard = CLRiskDashboard(config)
    
    # Connect dashboard
    dashboard.connect_components(
        risk_manager=risk_manager,
        portfolio_manager=portfolio_manager,
        execution_engine=execution_engine,
        market_analyzer=market_analyzer
    )
    
    # Start monitoring
    await dashboard.start_dashboard()
    
    # Simulate trading session
    market_data = {
        'symbol': 'CL_202501',
        'close': 75.0,
        'atr_20': 1.5,
        'volatility': 0.02,
        'volume': 500000,
        'prices': [
            {'open': 74.5, 'high': 75.5, 'low': 74.0, 'close': 75.0, 'volume': 1000}
        ]
    }
    
    signal_data = {
        'symbol': 'CL_202501',
        'direction': 'long',
        'confidence': 0.75,
        'strength': 0.8,
        'win_rate': 0.65,
        'avg_win': 0.025,
        'avg_loss': 0.015
    }
    
    portfolio_data = {'total_value': 1000000, 'positions': {}}
    
    # Execute complete trading workflow
    # 1. Size position
    sizing_result = await position_sizer.calculate_position_size(
        signal_data, market_data, portfolio_data
    )
    assert sizing_result['recommended_size'] > 0
    
    # 2. Validate trade
    trade_data = {
        'symbol': 'CL_202501',
        'size': sizing_result['recommended_size'],
        'price': 75.0,
        'side': 'long',
        'stop_loss': 73.0,
        'take_profit': 77.0
    }
    
    validation_result = await risk_manager.validate_trade(trade_data)
    assert validation_result['approved'] is True
    
    # 3. Add to portfolio
    portfolio_result = await portfolio_manager.add_position({
        'symbol': 'CL_202501',
        'quantity': trade_data['size'],
        'entry_price': trade_data['price'],
        'side': 'long'
    })
    assert portfolio_result['success'] is True
    
    # 4. Execute trade
    order = ExecutionOrder(
        order_id="integration_test_order",
        symbol='CL_202501',
        side='buy',
        quantity=trade_data['size'],
        order_type=OrderType.MARKET,
        algorithm=ExecutionAlgorithm.BALANCED
    )
    
    execution_result = await execution_engine.submit_order(order)
    assert execution_result['success'] is True
    
    # 5. Monitor position
    position_data = {
        'symbol': 'CL_202501',
        'size': trade_data['size'],
        'entry_price': trade_data['price'],
        'side': 'long',
        'stop_loss': trade_data['stop_loss'],
        'take_profit': trade_data['take_profit']
    }
    
    controls_result = await risk_controls.evaluate_risk_controls(
        position_data, market_data, portfolio_data
    )
    assert controls_result['controls_evaluated'] > 0
    
    # 6. Check dashboard
    await asyncio.sleep(0.1)  # Allow dashboard to update
    dashboard_state = dashboard.get_dashboard_state()
    assert dashboard_state['system_status']['dashboard_running'] is True
    
    # 7. Stop system
    await dashboard.stop_dashboard()
    
    # Verify final state
    final_state = dashboard.get_dashboard_state()
    assert final_state['system_status']['dashboard_running'] is False
    
    # All tests passed - system integration successful
    print("✅ Full system integration test passed!")


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(test_full_system_integration())