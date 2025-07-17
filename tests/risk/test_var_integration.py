"""
Enhanced VaR Model Integration Testing Suite

This comprehensive test suite validates VaR model integration across multiple 
asset classes, scenario-based testing, and stress testing scenarios.

Key Test Areas:
1. Multi-asset class VaR calculation (Equities, Fixed Income, Commodities, FX)
2. Scenario-based VaR testing (Market crashes, regime changes)
3. Stress testing integration (Historical scenarios, Monte Carlo)
4. Real-time VaR updates and position sensitivity
5. Portfolio optimization integration
6. Performance validation under various market conditions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData
from src.risk.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
from src.risk.agents.real_time_risk_assessor import RealTimeRiskAssessor


class AssetClass(Enum):
    """Asset class definitions"""
    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    FX = "FX"
    CRYPTO = "CRYPTO"


@dataclass
class MultiAssetPosition:
    """Enhanced position data with asset class information"""
    symbol: str
    asset_class: AssetClass
    quantity: float
    market_value: float
    price: float
    volatility: float
    duration: Optional[float] = None  # For fixed income
    beta: Optional[float] = None      # For equities
    delta: Optional[float] = None     # For derivatives


class TestVaRIntegration:
    """Comprehensive VaR integration testing suite"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def correlation_tracker(self, event_bus):
        """Create correlation tracker with multi-asset configuration"""
        tracker = CorrelationTracker(
            event_bus=event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.4,  # More sensitive for integration testing
            shock_window_minutes=15,
            max_correlation_history=200
        )
        
        # Multi-asset universe
        assets = [
            # Equities
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY',
            # Fixed Income
            'TLT', 'IEF', 'LQD', 'HYG', 'TIP',
            # Commodities
            'GLD', 'SLV', 'USO', 'UNG', 'DBA',
            # FX
            'UUP', 'FXE', 'FXY', 'FXC', 'EWZ',
            # Crypto (proxy)
            'GBTC', 'ETHE'
        ]
        
        tracker.initialize_assets(assets)
        return tracker
    
    @pytest.fixture
    def var_calculator(self, correlation_tracker, event_bus):
        """Create VaR calculator with enhanced configuration"""
        return VaRCalculator(
            correlation_tracker=correlation_tracker,
            event_bus=event_bus,
            confidence_levels=[0.90, 0.95, 0.99],
            time_horizons=[1, 5, 10, 22],  # Daily, weekly, bi-weekly, monthly
            default_method="parametric"
        )
    
    @pytest.fixture
    def portfolio_optimizer(self, event_bus):
        """Create portfolio optimizer agent"""
        return PortfolioOptimizerAgent(event_bus=event_bus)
    
    @pytest.fixture
    def multi_asset_portfolio(self):
        """Create diversified multi-asset portfolio"""
        return {
            # Equities (60% allocation)
            'AAPL': MultiAssetPosition('AAPL', AssetClass.EQUITY, 1000, 180000, 180.0, 0.25, beta=1.2),
            'GOOGL': MultiAssetPosition('GOOGL', AssetClass.EQUITY, 500, 150000, 300.0, 0.30, beta=1.1),
            'MSFT': MultiAssetPosition('MSFT', AssetClass.EQUITY, 800, 240000, 300.0, 0.22, beta=0.9),
            'SPY': MultiAssetPosition('SPY', AssetClass.EQUITY, 2000, 800000, 400.0, 0.16, beta=1.0),
            
            # Fixed Income (25% allocation)
            'TLT': MultiAssetPosition('TLT', AssetClass.FIXED_INCOME, 3000, 360000, 120.0, 0.12, duration=17.5),
            'IEF': MultiAssetPosition('IEF', AssetClass.FIXED_INCOME, 2000, 220000, 110.0, 0.08, duration=7.2),
            'LQD': MultiAssetPosition('LQD', AssetClass.FIXED_INCOME, 1000, 130000, 130.0, 0.10, duration=8.5),
            
            # Commodities (10% allocation)
            'GLD': MultiAssetPosition('GLD', AssetClass.COMMODITY, 2000, 320000, 160.0, 0.18),
            'USO': MultiAssetPosition('USO', AssetClass.COMMODITY, 1000, 80000, 80.0, 0.35),
            
            # FX (5% allocation)
            'UUP': MultiAssetPosition('UUP', AssetClass.FX, 5000, 150000, 30.0, 0.08),
            'FXE': MultiAssetPosition('FXE', AssetClass.FX, 1000, 110000, 110.0, 0.12)
        }
    
    def _setup_multi_asset_portfolio(self, var_calculator, portfolio):
        """Helper to setup multi-asset portfolio in VaR calculator"""
        # Convert MultiAssetPosition to PositionData
        position_data = {}
        for symbol, pos in portfolio.items():
            position_data[symbol] = PositionData(
                symbol=pos.symbol,
                quantity=pos.quantity,
                market_value=pos.market_value,
                price=pos.price,
                volatility=pos.volatility
            )
        
        var_calculator.positions = position_data
        var_calculator.portfolio_value = sum(pos.market_value for pos in portfolio.values())
        
        return position_data
    
    def _generate_multi_asset_returns(self, correlation_tracker, n_periods=100):
        """Generate realistic multi-asset returns with cross-asset correlations"""
        assets = correlation_tracker.assets
        
        # Define asset class correlations
        equity_assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY']
        bond_assets = ['TLT', 'IEF', 'LQD', 'HYG', 'TIP']
        commodity_assets = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']
        fx_assets = ['UUP', 'FXE', 'FXY', 'FXC', 'EWZ']
        crypto_assets = ['GBTC', 'ETHE']
        
        # Generate market factor (common systematic risk)
        market_factor = np.random.normal(0, 0.015, n_periods)
        
        for period in range(n_periods):
            timestamp = datetime.now() - timedelta(minutes=n_periods-period)
            
            # Generate asset-specific returns
            for asset in assets:
                # Base idiosyncratic return
                idiosyncratic_return = np.random.normal(0, 0.008)
                
                # Apply asset class factors
                if asset in equity_assets:
                    # Equities: high correlation with market factor
                    beta = np.random.uniform(0.8, 1.5)
                    sector_factor = np.random.normal(0, 0.005)
                    return_value = beta * market_factor[period] + sector_factor + idiosyncratic_return
                    
                elif asset in bond_assets:
                    # Bonds: negative correlation with equities, duration risk
                    duration_factor = np.random.normal(0, 0.003)
                    credit_factor = np.random.normal(0, 0.002)
                    return_value = -0.3 * market_factor[period] + duration_factor + credit_factor + idiosyncratic_return
                    
                elif asset in commodity_assets:
                    # Commodities: inflation hedge, some correlation with market
                    inflation_factor = np.random.normal(0, 0.006)
                    return_value = 0.2 * market_factor[period] + inflation_factor + idiosyncratic_return
                    
                elif asset in fx_assets:
                    # FX: currency factors, flight to quality
                    currency_factor = np.random.normal(0, 0.004)
                    return_value = -0.1 * market_factor[period] + currency_factor + idiosyncratic_return
                    
                elif asset in crypto_assets:
                    # Crypto: high volatility, some correlation with risk-on sentiment
                    crypto_factor = np.random.normal(0, 0.020)
                    return_value = 0.5 * market_factor[period] + crypto_factor + idiosyncratic_return
                    
                else:
                    # Default case
                    return_value = 0.3 * market_factor[period] + idiosyncratic_return
                
                # Add volatility clustering
                if period > 0:
                    # GARCH-like effect
                    prev_return = 0.01 if asset not in correlation_tracker.asset_returns else (
                        correlation_tracker.asset_returns[asset][-1][1] if 
                        len(correlation_tracker.asset_returns[asset]) > 0 else 0.01
                    )
                    volatility_multiplier = 1.0 + 0.1 * abs(prev_return)
                    return_value *= volatility_multiplier
                
                # Simulate price update
                if asset in correlation_tracker.asset_returns and len(correlation_tracker.asset_returns[asset]) > 0:
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
                    source='MultiAssetGenerator'
                )
                
                correlation_tracker._handle_price_update(event)
    
    @pytest.mark.asyncio
    async def test_multi_asset_var_calculation(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test VaR calculation across multiple asset classes"""
        
        # Setup portfolio
        position_data = self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        
        # Generate multi-asset returns
        self._generate_multi_asset_returns(correlation_tracker, 120)
        
        # Calculate VaR for different confidence levels
        var_results = {}
        for confidence_level in [0.95, 0.99]:
            var_result = await var_calculator.calculate_var(
                confidence_level=confidence_level,
                time_horizon=1,
                method="parametric"
            )
            
            assert var_result is not None, f"VaR calculation failed for {confidence_level}"
            var_results[confidence_level] = var_result
        
        # Verify VaR increases with confidence level
        assert var_results[0.99].portfolio_var > var_results[0.95].portfolio_var
        
        # Verify component VaRs are reasonable
        for confidence_level, var_result in var_results.items():
            # Check that all positions have component VaR
            assert len(var_result.component_vars) == len(position_data)
            
            # Verify component VaR magnitudes are reasonable
            for symbol, component_var in var_result.component_vars.items():
                position_value = position_data[symbol].market_value
                component_pct = abs(component_var) / position_value
                assert component_pct < 0.20, f"Component VaR too high for {symbol}: {component_pct:.2%}"
        
        # Verify diversification benefit
        component_sum = sum(abs(var) for var in var_results[0.95].component_vars.values())
        portfolio_var = abs(var_results[0.95].portfolio_var)
        diversification_ratio = component_sum / portfolio_var
        assert diversification_ratio > 1.1, f"Insufficient diversification benefit: {diversification_ratio:.2f}"
        
        print(f"✓ Multi-asset VaR calculation successful")
        print(f"✓ 95% VaR: ${var_results[0.95].portfolio_var:,.0f}")
        print(f"✓ 99% VaR: ${var_results[0.99].portfolio_var:,.0f}")
        print(f"✓ Diversification ratio: {diversification_ratio:.2f}")
    
    @pytest.mark.asyncio
    async def test_time_horizon_scaling(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test VaR scaling across different time horizons"""
        
        # Setup portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 100)
        
        # Calculate VaR for different time horizons
        time_horizons = [1, 5, 10, 22]  # Daily, weekly, bi-weekly, monthly
        var_results = {}
        
        for horizon in time_horizons:
            var_result = await var_calculator.calculate_var(
                confidence_level=0.95,
                time_horizon=horizon,
                method="parametric"
            )
            
            assert var_result is not None, f"VaR calculation failed for {horizon} days"
            var_results[horizon] = var_result
        
        # Verify VaR scaling follows square root of time (approximately)
        daily_var = var_results[1].portfolio_var
        
        for horizon in [5, 10, 22]:
            expected_var = daily_var * np.sqrt(horizon)
            actual_var = var_results[horizon].portfolio_var
            scaling_ratio = actual_var / expected_var
            
            # Allow for some deviation from perfect square root scaling
            assert 0.8 <= scaling_ratio <= 1.3, f"Time scaling incorrect for {horizon} days: {scaling_ratio:.2f}"
        
        print(f"✓ Time horizon scaling validated")
        print(f"✓ 1-day VaR: ${var_results[1].portfolio_var:,.0f}")
        print(f"✓ 5-day VaR: ${var_results[5].portfolio_var:,.0f}")
        print(f"✓ 22-day VaR: ${var_results[22].portfolio_var:,.0f}")
    
    @pytest.mark.asyncio
    async def test_scenario_based_var(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test VaR calculation under specific market scenarios"""
        
        # Setup portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 80)
        
        # Calculate baseline VaR
        baseline_var = await var_calculator.calculate_var(confidence_level=0.95, method="parametric")
        
        # Scenario 1: Market crash (high correlations)
        original_matrix = correlation_tracker.simulate_correlation_shock(0.85)
        crash_var = await var_calculator.calculate_var(confidence_level=0.95, method="parametric")
        
        # Scenario 2: Interest rate shock (affects bonds more)
        correlation_tracker.correlation_matrix = original_matrix.copy()
        # Simulate bond correlation increase
        bond_indices = [correlation_tracker.asset_index.get(asset, -1) for asset in ['TLT', 'IEF', 'LQD'] if asset in correlation_tracker.asset_index]
        bond_indices = [i for i in bond_indices if i >= 0]
        
        if len(bond_indices) >= 2:
            for i in bond_indices:
                for j in bond_indices:
                    if i != j:
                        correlation_tracker.correlation_matrix[i, j] = 0.90
        
        rate_shock_var = await var_calculator.calculate_var(confidence_level=0.95, method="parametric")
        
        # Verify scenario impacts
        assert crash_var.portfolio_var > baseline_var.portfolio_var, "Market crash scenario should increase VaR"
        assert rate_shock_var.portfolio_var > baseline_var.portfolio_var, "Rate shock scenario should increase VaR"
        
        # Verify different scenarios affect different components
        equity_symbols = ['AAPL', 'GOOGL', 'MSFT', 'SPY']
        bond_symbols = ['TLT', 'IEF', 'LQD']
        
        # Market crash should affect equities more
        equity_crash_impact = sum(abs(crash_var.component_vars.get(s, 0)) for s in equity_symbols)
        equity_baseline_impact = sum(abs(baseline_var.component_vars.get(s, 0)) for s in equity_symbols)
        
        if equity_baseline_impact > 0:
            equity_impact_ratio = equity_crash_impact / equity_baseline_impact
            assert equity_impact_ratio > 1.1, f"Market crash should increase equity VaR more: {equity_impact_ratio:.2f}"
        
        print(f"✓ Scenario-based VaR testing successful")
        print(f"✓ Baseline VaR: ${baseline_var.portfolio_var:,.0f}")
        print(f"✓ Market crash VaR: ${crash_var.portfolio_var:,.0f}")
        print(f"✓ Rate shock VaR: ${rate_shock_var.portfolio_var:,.0f}")
    
    @pytest.mark.asyncio
    async def test_var_method_comparison(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test consistency across different VaR calculation methods"""
        
        # Setup portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 150)
        
        # Calculate VaR using different methods
        methods = ["parametric", "historical", "monte_carlo"]
        var_results = {}
        
        for method in methods:
            var_result = await var_calculator.calculate_var(
                confidence_level=0.95,
                time_horizon=1,
                method=method
            )
            
            if var_result is not None:
                var_results[method] = var_result
        
        # Verify we got results for multiple methods
        assert len(var_results) >= 2, "Should have results from multiple VaR methods"
        
        # Compare results - they should be in same order of magnitude
        var_values = [result.portfolio_var for result in var_results.values()]
        min_var = min(var_values)
        max_var = max(var_values)
        
        # VaR estimates should be within reasonable range of each other
        ratio = max_var / min_var
        assert ratio < 3.0, f"VaR methods show excessive variation: {ratio:.2f}"
        
        print(f"✓ VaR method comparison successful")
        for method, result in var_results.items():
            print(f"✓ {method.capitalize()} VaR: ${result.portfolio_var:,.0f}")
    
    @pytest.mark.asyncio
    async def test_real_time_var_updates(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test real-time VaR updates as positions change"""
        
        # Setup initial portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 80)
        
        # Calculate initial VaR
        initial_var = await var_calculator.calculate_var(confidence_level=0.95)
        
        # Simulate position changes
        position_changes = [
            # Increase equity exposure
            ('AAPL', 2000, 360000),  # Double AAPL position
            # Reduce fixed income
            ('TLT', 1500, 180000),   # Reduce TLT position
            # Add new position
            ('NVDA', 500, 200000)    # Add NVDA position
        ]
        
        var_after_changes = []
        
        for symbol, new_quantity, new_market_value in position_changes:
            # Update position
            if symbol in var_calculator.positions:
                var_calculator.positions[symbol].quantity = new_quantity
                var_calculator.positions[symbol].market_value = new_market_value
            else:
                # Add new position
                var_calculator.positions[symbol] = PositionData(
                    symbol=symbol,
                    quantity=new_quantity,
                    market_value=new_market_value,
                    price=new_market_value/new_quantity,
                    volatility=0.30  # Assume high volatility for new position
                )
            
            # Update portfolio value
            var_calculator.portfolio_value = sum(
                pos.market_value for pos in var_calculator.positions.values()
            )
            
            # Recalculate VaR
            updated_var = await var_calculator.calculate_var(confidence_level=0.95)
            var_after_changes.append(updated_var)
        
        # Verify VaR responds to position changes
        final_var = var_after_changes[-1]
        assert final_var.portfolio_var != initial_var.portfolio_var, "VaR should change with position updates"
        
        # Verify portfolio value increased (more risk)
        assert final_var.portfolio_var > initial_var.portfolio_var, "VaR should increase with higher exposure"
        
        print(f"✓ Real-time VaR updates successful")
        print(f"✓ Initial VaR: ${initial_var.portfolio_var:,.0f}")
        print(f"✓ Final VaR: ${final_var.portfolio_var:,.0f}")
    
    @pytest.mark.asyncio
    async def test_stress_testing_integration(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test integration with stress testing scenarios"""
        
        # Setup portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 100)
        
        # Define stress scenarios
        stress_scenarios = [
            {
                'name': '2008 Financial Crisis',
                'equity_shock': -0.30,
                'bond_shock': -0.05,
                'correlation_shock': 0.80
            },
            {
                'name': 'COVID-19 Pandemic',
                'equity_shock': -0.25,
                'bond_shock': 0.10,
                'correlation_shock': 0.85
            },
            {
                'name': 'Interest Rate Shock',
                'equity_shock': -0.10,
                'bond_shock': -0.15,
                'correlation_shock': 0.60
            }
        ]
        
        baseline_var = await var_calculator.calculate_var(confidence_level=0.95)
        stress_results = {}
        
        for scenario in stress_scenarios:
            # Apply correlation shock
            correlation_tracker.simulate_correlation_shock(scenario['correlation_shock'])
            
            # Calculate stressed VaR
            stressed_var = await var_calculator.calculate_var(confidence_level=0.95)
            stress_results[scenario['name']] = stressed_var
        
        # Verify stress scenarios increase VaR
        for scenario_name, stressed_var in stress_results.items():
            assert stressed_var.portfolio_var > baseline_var.portfolio_var, \
                f"Stress scenario {scenario_name} should increase VaR"
        
        # Find worst-case scenario
        worst_scenario = max(stress_results.items(), key=lambda x: x[1].portfolio_var)
        
        print(f"✓ Stress testing integration successful")
        print(f"✓ Baseline VaR: ${baseline_var.portfolio_var:,.0f}")
        print(f"✓ Worst scenario ({worst_scenario[0]}): ${worst_scenario[1].portfolio_var:,.0f}")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test VaR calculation performance under high load"""
        
        # Setup portfolio
        self._setup_multi_asset_portfolio(var_calculator, multi_asset_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 200)
        
        # Performance test - multiple rapid calculations
        calculation_times = []
        var_results = []
        
        for i in range(50):
            start_time = datetime.now()
            
            # Vary confidence levels and methods
            confidence_level = 0.95 if i % 2 == 0 else 0.99
            method = "parametric" if i % 3 == 0 else "monte_carlo"
            
            var_result = await var_calculator.calculate_var(
                confidence_level=confidence_level,
                time_horizon=1,
                method=method
            )
            
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            calculation_times.append(calc_time)
            
            if var_result:
                var_results.append(var_result)
        
        # Verify performance targets
        avg_time = np.mean(calculation_times)
        max_time = np.max(calculation_times)
        p95_time = np.percentile(calculation_times, 95)
        
        assert avg_time < 10.0, f"Average calculation time too slow: {avg_time:.2f}ms"
        assert p95_time < 20.0, f"95th percentile time too slow: {p95_time:.2f}ms"
        assert max_time < 50.0, f"Maximum calculation time too slow: {max_time:.2f}ms"
        
        # Verify calculation stability
        var_values = [result.portfolio_var for result in var_results]
        var_std = np.std(var_values)
        var_mean = np.mean(var_values)
        cv = var_std / var_mean  # Coefficient of variation
        
        assert cv < 0.50, f"VaR calculations too unstable: CV = {cv:.2f}"
        
        print(f"✓ Performance under load validated")
        print(f"✓ Average calculation time: {avg_time:.2f}ms")
        print(f"✓ 95th percentile time: {p95_time:.2f}ms")
        print(f"✓ VaR stability (CV): {cv:.2f}")
    
    @pytest.mark.asyncio
    async def test_copula_modeling_integration(self, correlation_tracker, var_calculator, multi_asset_portfolio):
        """Test integration with copula modeling for tail risk"""
        
        # Setup portfolio (reduce size for copula testing)
        simplified_portfolio = {
            'AAPL': multi_asset_portfolio['AAPL'],
            'TLT': multi_asset_portfolio['TLT'],
            'GLD': multi_asset_portfolio['GLD']
        }
        
        self._setup_multi_asset_portfolio(var_calculator, simplified_portfolio)
        self._generate_multi_asset_returns(correlation_tracker, 150)
        
        # Enable copula modeling
        var_calculator.enable_copula_modeling = True
        
        # Calculate VaR with copula enhancement
        copula_var = await var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon=1,
            method="parametric"
        )
        
        # Calculate traditional VaR for comparison
        var_calculator.enable_copula_modeling = False
        traditional_var = await var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon=1,
            method="parametric"
        )
        
        # Verify copula modeling affects results
        if copula_var and traditional_var:
            assert copula_var.portfolio_var != traditional_var.portfolio_var, \
                "Copula modeling should affect VaR calculation"
            
            # Method should indicate copula enhancement
            assert "copula" in copula_var.calculation_method.lower(), \
                "Calculation method should indicate copula enhancement"
        
        # Test tail risk metrics
        tail_metrics = await var_calculator.calculate_tail_risk_metrics()
        
        if tail_metrics:
            assert 'tail_dependency_lower' in tail_metrics, "Should have lower tail dependency"
            assert 'tail_dependency_upper' in tail_metrics, "Should have upper tail dependency"
            assert 'copula_type' in tail_metrics, "Should identify copula type"
        
        print(f"✓ Copula modeling integration successful")
        if copula_var and traditional_var:
            print(f"✓ Traditional VaR: ${traditional_var.portfolio_var:,.0f}")
            print(f"✓ Copula-enhanced VaR: ${copula_var.portfolio_var:,.0f}")
        if tail_metrics:
            print(f"✓ Tail metrics calculated: {len(tail_metrics)} metrics")


if __name__ == "__main__":
    """Run VaR integration tests directly"""
    
    print("📊 Starting VaR Integration Tests...")
    print("=" * 50)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])