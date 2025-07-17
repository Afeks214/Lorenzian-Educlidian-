"""
Enhanced Correlation Regime Detection Testing Suite

This comprehensive test suite validates correlation regime detection,
shock identification, and dynamic correlation adjustment mechanisms.

Key Test Areas:
1. Correlation regime classification and transitions
2. Shock detection algorithms and thresholds
3. Dynamic correlation adjustment mechanisms
4. Regime persistence and stability testing
5. Cross-asset correlation analysis
6. Performance under various market conditions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime, CorrelationShock
from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData
from src.risk.intelligence.crisis_fingerprint_engine import CrisisFingerprintEngine
from src.risk.intelligence.maml_crisis_detector import MAMLCrisisDetector


class MarketRegimeType(Enum):
    """Market regime types for testing"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class RegimeTestScenario:
    """Test scenario for regime detection"""
    name: str
    description: str
    regime_type: MarketRegimeType
    correlation_level: float
    volatility_multiplier: float
    duration_periods: int
    expected_detection_time: float  # Seconds
    expected_regime: CorrelationRegime


@dataclass
class CorrelationPattern:
    """Correlation pattern for testing"""
    asset_pairs: List[Tuple[str, str]]
    correlation_values: List[float]
    timestamp: datetime
    regime_type: MarketRegimeType
    stability_score: float


class TestCorrelationRegimeDetection:
    """Comprehensive correlation regime detection test suite"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def correlation_tracker(self, event_bus):
        """Create correlation tracker with enhanced configuration"""
        tracker = CorrelationTracker(
            event_bus=event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.4,  # More sensitive for testing
            shock_window_minutes=5,
            max_correlation_history=500
        )
        
        # Extended asset universe for regime testing
        assets = [
            # Equity indices
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI',
            # Individual stocks
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA',
            # Bonds
            'TLT', 'IEF', 'HYG', 'LQD', 'TIP', 'EMB',
            # Commodities
            'GLD', 'SLV', 'USO', 'DBA', 'UNG', 'PDBC',
            # Currencies
            'UUP', 'FXE', 'FXY', 'FXC', 'FXA', 'FXS',
            # Volatility
            'VIX', 'VXX', 'UVXY'
        ]
        
        tracker.initialize_assets(assets)
        return tracker
    
    @pytest.fixture
    def var_calculator(self, correlation_tracker, event_bus):
        """Create VaR calculator for testing"""
        return VaRCalculator(
            correlation_tracker=correlation_tracker,
            event_bus=event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 5, 10]
        )
    
    @pytest.fixture
    def crisis_detector(self, event_bus):
        """Create crisis detection system"""
        return MAMLCrisisDetector(
            event_bus=event_bus,
            adaptation_lr=0.01,
            meta_lr=0.001
        )
    
    @pytest.fixture
    def fingerprint_engine(self, event_bus):
        """Create crisis fingerprint engine"""
        return CrisisFingerprintEngine(
            event_bus=event_bus,
            lookback_window=252,
            min_similarity=0.7
        )
    
    @pytest.fixture
    def test_scenarios(self):
        """Create test scenarios for regime detection"""
        return [
            RegimeTestScenario(
                name="Normal Market",
                description="Typical market conditions with moderate correlations",
                regime_type=MarketRegimeType.BULL_MARKET,
                correlation_level=0.3,
                volatility_multiplier=1.0,
                duration_periods=100,
                expected_detection_time=5.0,
                expected_regime=CorrelationRegime.NORMAL
            ),
            RegimeTestScenario(
                name="Elevated Stress",
                description="Increased market stress with higher correlations",
                regime_type=MarketRegimeType.VOLATILE,
                correlation_level=0.6,
                volatility_multiplier=1.5,
                duration_periods=50,
                expected_detection_time=3.0,
                expected_regime=CorrelationRegime.ELEVATED
            ),
            RegimeTestScenario(
                name="Crisis Mode",
                description="Market crisis with extreme correlations",
                regime_type=MarketRegimeType.CRISIS,
                correlation_level=0.85,
                volatility_multiplier=2.5,
                duration_periods=30,
                expected_detection_time=2.0,
                expected_regime=CorrelationRegime.CRISIS
            ),
            RegimeTestScenario(
                name="Correlation Shock",
                description="Sudden correlation spike from external event",
                regime_type=MarketRegimeType.CRISIS,
                correlation_level=0.95,
                volatility_multiplier=3.0,
                duration_periods=20,
                expected_detection_time=1.0,
                expected_regime=CorrelationRegime.SHOCK
            )
        ]
    
    def _generate_regime_specific_returns(
        self, 
        correlation_tracker, 
        scenario: RegimeTestScenario
    ) -> List[np.ndarray]:
        """Generate returns that follow specific correlation regime"""
        
        assets = correlation_tracker.assets
        n_assets = len(assets)
        n_periods = scenario.duration_periods
        
        # Create target correlation matrix
        target_corr = np.full((n_assets, n_assets), scenario.correlation_level)
        np.fill_diagonal(target_corr, 1.0)
        
        # Add asset-specific correlation patterns
        if scenario.regime_type == MarketRegimeType.CRISIS:
            # During crisis, all assets become highly correlated
            target_corr = np.full((n_assets, n_assets), 0.9)
            np.fill_diagonal(target_corr, 1.0)
            
        elif scenario.regime_type == MarketRegimeType.BULL_MARKET:
            # Bull market: equities correlated, bonds anti-correlated
            equity_indices = [i for i, asset in enumerate(assets) 
                            if asset in ['SPY', 'QQQ', 'IWM', 'AAPL', 'GOOGL', 'MSFT']]
            bond_indices = [i for i, asset in enumerate(assets) 
                           if asset in ['TLT', 'IEF', 'HYG', 'LQD']]
            
            # High equity-equity correlation
            for i in equity_indices:
                for j in equity_indices:
                    if i != j:
                        target_corr[i, j] = 0.7
            
            # Negative equity-bond correlation
            for i in equity_indices:
                for j in bond_indices:
                    target_corr[i, j] = -0.3
                    target_corr[j, i] = -0.3
        
        elif scenario.regime_type == MarketRegimeType.VOLATILE:
            # Volatile market: increased but unstable correlations
            target_corr *= np.random.uniform(0.8, 1.2, (n_assets, n_assets))
            np.fill_diagonal(target_corr, 1.0)
            # Ensure correlation matrix is valid
            target_corr = np.clip(target_corr, -0.99, 0.99)
        
        # Generate correlated returns
        mean_returns = np.zeros(n_assets)
        volatilities = np.full(n_assets, 0.01 * scenario.volatility_multiplier)
        
        # Create covariance matrix
        vol_matrix = np.outer(volatilities, volatilities)
        cov_matrix = target_corr * vol_matrix
        
        # Generate returns
        np.random.seed(42)  # For reproducible tests
        returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_periods
        )
        
        # Apply regime-specific shocks
        if scenario.regime_type == MarketRegimeType.CRISIS:
            # Add fat-tail events
            shock_periods = np.random.choice(n_periods, size=max(1, n_periods//10), replace=False)
            for period in shock_periods:
                shock_magnitude = np.random.uniform(3, 5)  # 3-5 sigma events
                shock_direction = np.random.choice([-1, 1])
                returns[period] *= shock_magnitude * shock_direction
        
        return returns
    
    def _simulate_regime_transition(
        self, 
        correlation_tracker, 
        scenarios: List[RegimeTestScenario]
    ):
        """Simulate market regime transitions"""
        
        assets = correlation_tracker.assets
        
        for scenario in scenarios:
            print(f"Simulating regime: {scenario.name}")
            
            # Generate returns for this regime
            returns = self._generate_regime_specific_returns(correlation_tracker, scenario)
            
            # Feed returns to correlation tracker
            for period in range(len(returns)):
                timestamp = datetime.now() - timedelta(
                    minutes=len(returns) - period
                )
                
                for i, asset in enumerate(assets):
                    return_value = returns[period, i]
                    
                    # Calculate price from return
                    if asset in correlation_tracker.asset_returns and \
                       len(correlation_tracker.asset_returns[asset]) > 0:
                        last_price = correlation_tracker.asset_returns[asset][-1][1]
                        new_price = last_price * (1 + return_value)
                    else:
                        new_price = 100.0 * (1 + return_value)
                    
                    # Create price update event
                    bar_data = Mock()
                    bar_data.symbol = asset
                    bar_data.timestamp = timestamp
                    bar_data.close = new_price
                    
                    event = Event(
                        event_type=EventType.NEW_5MIN_BAR,
                        timestamp=timestamp,
                        payload=bar_data,
                        source=f'RegimeSimulator_{scenario.name}'
                    )
                    
                    correlation_tracker._handle_price_update(event)
    
    def test_regime_classification(self, correlation_tracker, test_scenarios):
        """Test correlation regime classification accuracy"""
        
        # Test each scenario
        for scenario in test_scenarios:
            print(f"Testing regime classification for: {scenario.name}")
            
            # Generate data for this regime
            returns = self._generate_regime_specific_returns(correlation_tracker, scenario)
            
            # Feed data to tracker
            assets = correlation_tracker.assets
            for period in range(len(returns)):
                timestamp = datetime.now() - timedelta(minutes=len(returns) - period)
                
                for i, asset in enumerate(assets):
                    return_value = returns[period, i]
                    
                    if asset in correlation_tracker.asset_returns and \
                       len(correlation_tracker.asset_returns[asset]) > 0:
                        last_price = correlation_tracker.asset_returns[asset][-1][1]
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
                        source='RegimeTest'
                    )
                    
                    correlation_tracker._handle_price_update(event)
            
            # Check regime detection
            detected_regime = correlation_tracker.current_regime
            avg_correlation = correlation_tracker._calculate_average_correlation()
            
            print(f"  Expected regime: {scenario.expected_regime.value}")
            print(f"  Detected regime: {detected_regime.value}")
            print(f"  Average correlation: {avg_correlation:.3f}")
            print(f"  Target correlation: {scenario.correlation_level:.3f}")
            
            # Verify regime detection is reasonable
            if scenario.expected_regime == CorrelationRegime.NORMAL:
                assert avg_correlation < 0.5, f"Normal regime correlation too high: {avg_correlation:.3f}"
            elif scenario.expected_regime == CorrelationRegime.ELEVATED:
                assert 0.4 < avg_correlation < 0.8, f"Elevated regime correlation out of range: {avg_correlation:.3f}"
            elif scenario.expected_regime in [CorrelationRegime.CRISIS, CorrelationRegime.SHOCK]:
                assert avg_correlation > 0.7, f"Crisis regime correlation too low: {avg_correlation:.3f}"
        
        print("✓ Regime classification test completed")
    
    def test_shock_detection_sensitivity(self, correlation_tracker):
        """Test correlation shock detection sensitivity and timing"""
        
        # Generate normal market conditions
        normal_scenario = RegimeTestScenario(
            name="Normal Baseline",
            description="Normal market for shock testing",
            regime_type=MarketRegimeType.BULL_MARKET,
            correlation_level=0.25,
            volatility_multiplier=1.0,
            duration_periods=100,
            expected_detection_time=5.0,
            expected_regime=CorrelationRegime.NORMAL
        )
        
        self._simulate_regime_transition(correlation_tracker, [normal_scenario])
        
        # Record baseline state
        baseline_regime = correlation_tracker.current_regime
        baseline_correlation = correlation_tracker._calculate_average_correlation()
        baseline_alerts = len(correlation_tracker.shock_alerts)
        
        # Test different shock magnitudes
        shock_levels = [0.6, 0.75, 0.9, 0.95]
        
        for shock_level in shock_levels:
            print(f"Testing shock detection at correlation level: {shock_level}")
            
            # Record time before shock
            pre_shock_time = datetime.now()
            
            # Trigger correlation shock
            correlation_tracker.simulate_correlation_shock(shock_level)
            
            # Record time after shock
            post_shock_time = datetime.now()
            detection_time = (post_shock_time - pre_shock_time).total_seconds()
            
            # Check shock detection
            new_alerts = len(correlation_tracker.shock_alerts)
            shock_detected = new_alerts > baseline_alerts
            
            # Verify shock was detected
            if shock_level > correlation_tracker.shock_threshold:
                assert shock_detected, f"Shock not detected at level {shock_level}"
                
                # Check detection timing
                assert detection_time < 2.0, f"Shock detection too slow: {detection_time:.3f}s"
                
                # Verify alert details
                latest_alert = correlation_tracker.shock_alerts[-1]
                assert latest_alert.correlation_change > 0.3, "Shock magnitude not recorded correctly"
                assert latest_alert.current_avg_corr > baseline_correlation, "Shock correlation not higher"
                
                # Check severity classification
                expected_severity = "CRITICAL" if shock_level > 0.85 else "HIGH" if shock_level > 0.7 else "MODERATE"
                assert latest_alert.severity == expected_severity, f"Incorrect severity: {latest_alert.severity}"
            
            print(f"  Shock detected: {shock_detected}")
            print(f"  Detection time: {detection_time:.3f}s")
            print(f"  New correlation: {correlation_tracker._calculate_average_correlation():.3f}")
            
            # Reset for next test
            baseline_alerts = new_alerts
        
        print("✓ Shock detection sensitivity test completed")
    
    def test_regime_persistence_and_stability(self, correlation_tracker):
        """Test regime persistence and stability over time"""
        
        # Test regime stability - regime should persist through minor fluctuations
        stable_scenario = RegimeTestScenario(
            name="Stable Elevated",
            description="Stable elevated correlation regime",
            regime_type=MarketRegimeType.VOLATILE,
            correlation_level=0.65,
            volatility_multiplier=1.3,
            duration_periods=150,
            expected_detection_time=3.0,
            expected_regime=CorrelationRegime.ELEVATED
        )
        
        self._simulate_regime_transition(correlation_tracker, [stable_scenario])
        
        # Record regime states over time
        regime_history = []
        correlation_history = []
        
        # Monitor regime for additional periods with minor fluctuations
        for period in range(50):
            # Add small random fluctuations
            fluctuation = np.random.uniform(0.9, 1.1)
            current_matrix = correlation_tracker.get_correlation_matrix()
            
            if current_matrix is not None:
                # Apply small fluctuation to correlation matrix
                n_assets = current_matrix.shape[0]
                noise = np.random.normal(0, 0.05, (n_assets, n_assets))
                noise = (noise + noise.T) / 2  # Make symmetric
                np.fill_diagonal(noise, 0)  # Keep diagonal as 1
                
                fluctuated_matrix = current_matrix + noise
                fluctuated_matrix = np.clip(fluctuated_matrix, -0.99, 0.99)
                np.fill_diagonal(fluctuated_matrix, 1.0)
                
                # Update tracker's correlation matrix
                correlation_tracker.correlation_matrix = fluctuated_matrix
                
                # Force regime check
                correlation_tracker._check_correlation_shock()
            
            # Record current state
            regime_history.append(correlation_tracker.current_regime)
            correlation_history.append(correlation_tracker._calculate_average_correlation())
        
        # Analyze regime stability
        regime_changes = sum(1 for i in range(1, len(regime_history)) 
                           if regime_history[i] != regime_history[i-1])
        
        correlation_std = np.std(correlation_history)
        correlation_mean = np.mean(correlation_history)
        
        # Verify regime stability
        assert regime_changes < 5, f"Too many regime changes: {regime_changes}"
        assert correlation_std < 0.15, f"Correlation too unstable: {correlation_std:.3f}"
        assert 0.5 < correlation_mean < 0.8, f"Correlation drifted from target: {correlation_mean:.3f}"
        
        print("✓ Regime persistence and stability test completed")
        print(f"  Regime changes: {regime_changes}")
        print(f"  Correlation stability (std): {correlation_std:.3f}")
        print(f"  Average correlation: {correlation_mean:.3f}")
    
    def test_cross_asset_correlation_analysis(self, correlation_tracker):
        """Test cross-asset correlation analysis and patterns"""
        
        # Generate multi-asset regime with specific cross-correlations
        assets = correlation_tracker.assets
        n_assets = len(assets)
        
        # Create sector-specific correlation patterns
        equity_assets = [asset for asset in assets if asset in ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT']]
        bond_assets = [asset for asset in assets if asset in ['TLT', 'IEF', 'HYG', 'LQD']]
        commodity_assets = [asset for asset in assets if asset in ['GLD', 'SLV', 'USO']]
        
        # Generate sector-specific returns
        n_periods = 100
        
        # Market factor
        market_factor = np.random.normal(0, 0.02, n_periods)
        
        for period in range(n_periods):
            timestamp = datetime.now() - timedelta(minutes=n_periods - period)
            
            for asset in assets:
                # Asset-specific return based on sector
                if asset in equity_assets:
                    beta = np.random.uniform(0.8, 1.5)
                    sector_factor = np.random.normal(0, 0.01)
                    return_value = beta * market_factor[period] + sector_factor
                    
                elif asset in bond_assets:
                    # Bonds: negative correlation with equities
                    bond_factor = np.random.normal(0, 0.005)
                    return_value = -0.4 * market_factor[period] + bond_factor
                    
                elif asset in commodity_assets:
                    # Commodities: some correlation with inflation/market
                    commodity_factor = np.random.normal(0, 0.015)
                    return_value = 0.3 * market_factor[period] + commodity_factor
                    
                else:
                    # Other assets
                    return_value = np.random.normal(0, 0.01)
                
                # Calculate price
                if asset in correlation_tracker.asset_returns and \
                   len(correlation_tracker.asset_returns[asset]) > 0:
                    last_price = correlation_tracker.asset_returns[asset][-1][1]
                    new_price = last_price * (1 + return_value)
                else:
                    new_price = 100.0 * (1 + return_value)
                
                # Send price update
                bar_data = Mock()
                bar_data.symbol = asset
                bar_data.timestamp = timestamp
                bar_data.close = new_price
                
                event = Event(
                    event_type=EventType.NEW_5MIN_BAR,
                    timestamp=timestamp,
                    payload=bar_data,
                    source='CrossAssetTest'
                )
                
                correlation_tracker._handle_price_update(event)
        
        # Analyze cross-asset correlations
        correlation_matrix = correlation_tracker.get_correlation_matrix()
        assert correlation_matrix is not None, "Correlation matrix not generated"
        
        # Test specific correlation patterns
        asset_indices = correlation_tracker.asset_index
        
        # Check equity-equity correlations (should be positive)
        equity_indices = [asset_indices[asset] for asset in equity_assets if asset in asset_indices]
        if len(equity_indices) >= 2:
            equity_corr_sum = 0
            equity_pairs = 0
            for i in range(len(equity_indices)):
                for j in range(i+1, len(equity_indices)):
                    idx1, idx2 = equity_indices[i], equity_indices[j]
                    equity_corr_sum += correlation_matrix[idx1, idx2]
                    equity_pairs += 1
            
            avg_equity_corr = equity_corr_sum / equity_pairs if equity_pairs > 0 else 0
            assert avg_equity_corr > 0.1, f"Equity-equity correlation too low: {avg_equity_corr:.3f}"
        
        # Check equity-bond correlations (should be negative or low)
        bond_indices = [asset_indices[asset] for asset in bond_assets if asset in asset_indices]
        if len(equity_indices) >= 1 and len(bond_indices) >= 1:
            equity_bond_corr_sum = 0
            equity_bond_pairs = 0
            for eq_idx in equity_indices:
                for bond_idx in bond_indices:
                    equity_bond_corr_sum += correlation_matrix[eq_idx, bond_idx]
                    equity_bond_pairs += 1
            
            avg_equity_bond_corr = equity_bond_corr_sum / equity_bond_pairs if equity_bond_pairs > 0 else 0
            assert avg_equity_bond_corr < 0.3, f"Equity-bond correlation too high: {avg_equity_bond_corr:.3f}"
        
        print("✓ Cross-asset correlation analysis completed")
        print(f"  Correlation matrix shape: {correlation_matrix.shape}")
        if 'avg_equity_corr' in locals():
            print(f"  Average equity-equity correlation: {avg_equity_corr:.3f}")
        if 'avg_equity_bond_corr' in locals():
            print(f"  Average equity-bond correlation: {avg_equity_bond_corr:.3f}")
    
    def test_regime_transition_dynamics(self, correlation_tracker, test_scenarios):
        """Test dynamics of regime transitions"""
        
        # Test smooth transitions between regimes
        transition_scenarios = [
            test_scenarios[0],  # Normal
            test_scenarios[1],  # Elevated
            test_scenarios[2],  # Crisis
            test_scenarios[1],  # Back to Elevated
            test_scenarios[0]   # Back to Normal
        ]
        
        regime_transitions = []
        correlation_path = []
        
        for i, scenario in enumerate(transition_scenarios):
            print(f"Transitioning to regime {i+1}: {scenario.name}")
            
            # Record pre-transition state
            pre_regime = correlation_tracker.current_regime
            pre_correlation = correlation_tracker._calculate_average_correlation()
            
            # Simulate regime
            self._simulate_regime_transition(correlation_tracker, [scenario])
            
            # Record post-transition state
            post_regime = correlation_tracker.current_regime
            post_correlation = correlation_tracker._calculate_average_correlation()
            
            # Record transition
            regime_transitions.append({
                'from_regime': pre_regime,
                'to_regime': post_regime,
                'from_correlation': pre_correlation,
                'to_correlation': post_correlation,
                'scenario': scenario.name
            })
            
            correlation_path.append(post_correlation)
        
        # Analyze transition dynamics
        successful_transitions = sum(1 for trans in regime_transitions 
                                   if trans['from_regime'] != trans['to_regime'])
        
        # Check that correlations moved in expected direction
        correlation_increases = sum(1 for trans in regime_transitions 
                                  if trans['to_correlation'] > trans['from_correlation'])
        
        # Verify transition behavior
        assert successful_transitions >= 2, f"Insufficient regime transitions: {successful_transitions}"
        
        # Check correlation path makes sense
        max_correlation = max(correlation_path)
        min_correlation = min(correlation_path)
        
        assert max_correlation > 0.7, f"Max correlation too low: {max_correlation:.3f}"
        assert min_correlation < 0.5, f"Min correlation too high: {min_correlation:.3f}"
        
        print("✓ Regime transition dynamics test completed")
        print(f"  Successful transitions: {successful_transitions}")
        print(f"  Correlation range: {min_correlation:.3f} - {max_correlation:.3f}")
        
        for i, trans in enumerate(regime_transitions):
            print(f"  Transition {i+1}: {trans['from_regime'].value} -> {trans['to_regime'].value}")
    
    @pytest.mark.asyncio
    async def test_var_regime_integration(self, correlation_tracker, var_calculator, test_scenarios):
        """Test VaR calculation integration with regime detection"""
        
        # Create test portfolio
        portfolio = {
            'SPY': PositionData('SPY', 1000, 400000, 400.0, 0.16),
            'QQQ': PositionData('QQQ', 800, 320000, 400.0, 0.20),
            'TLT': PositionData('TLT', 2000, 240000, 120.0, 0.12),
            'GLD': PositionData('GLD', 1500, 240000, 160.0, 0.18)
        }
        
        var_calculator.positions = portfolio
        var_calculator.portfolio_value = sum(pos.market_value for pos in portfolio.values())
        
        # Test VaR calculation in different regimes
        var_by_regime = {}
        
        for scenario in test_scenarios[:3]:  # Test first 3 scenarios
            print(f"Testing VaR in regime: {scenario.name}")
            
            # Simulate regime
            self._simulate_regime_transition(correlation_tracker, [scenario])
            
            # Calculate VaR
            var_result = await var_calculator.calculate_var(
                confidence_level=0.95,
                time_horizon=1,
                method="parametric"
            )
            
            assert var_result is not None, f"VaR calculation failed for {scenario.name}"
            
            var_by_regime[scenario.name] = var_result
            
            # Verify regime is reflected in VaR
            assert scenario.expected_regime.value in var_result.correlation_regime or \
                   correlation_tracker.current_regime == scenario.expected_regime, \
                   f"Regime mismatch in VaR result for {scenario.name}"
        
        # Compare VaR across regimes
        normal_var = var_by_regime["Normal Market"].portfolio_var
        elevated_var = var_by_regime["Elevated Stress"].portfolio_var
        crisis_var = var_by_regime["Crisis Mode"].portfolio_var
        
        # VaR should increase with regime severity
        assert elevated_var > normal_var, "VaR should increase from normal to elevated regime"
        assert crisis_var > elevated_var, "VaR should increase from elevated to crisis regime"
        
        # Check reasonable magnitude increases
        elevated_ratio = elevated_var / normal_var
        crisis_ratio = crisis_var / normal_var
        
        assert 1.1 < elevated_ratio < 3.0, f"Elevated VaR ratio unreasonable: {elevated_ratio:.2f}"
        assert 1.5 < crisis_ratio < 5.0, f"Crisis VaR ratio unreasonable: {crisis_ratio:.2f}"
        
        print("✓ VaR regime integration test completed")
        print(f"  Normal VaR: ${normal_var:,.0f}")
        print(f"  Elevated VaR: ${elevated_var:,.0f} ({elevated_ratio:.2f}x)")
        print(f"  Crisis VaR: ${crisis_var:,.0f} ({crisis_ratio:.2f}x)")
    
    def test_performance_under_regime_changes(self, correlation_tracker):
        """Test performance during rapid regime changes"""
        
        # Test rapid regime switching
        rapid_scenarios = [
            RegimeTestScenario(
                name=f"Rapid_{i}",
                description=f"Rapid regime change {i}",
                regime_type=MarketRegimeType.VOLATILE,
                correlation_level=np.random.uniform(0.3, 0.9),
                volatility_multiplier=np.random.uniform(1.0, 2.0),
                duration_periods=20,  # Short duration for rapid changes
                expected_detection_time=1.0,
                expected_regime=CorrelationRegime.ELEVATED
            ) for i in range(10)
        ]
        
        # Measure performance during rapid changes
        performance_times = []
        
        for scenario in rapid_scenarios:
            start_time = datetime.now()
            
            # Simulate rapid regime change
            self._simulate_regime_transition(correlation_tracker, [scenario])
            
            # Measure processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            performance_times.append(processing_time)
        
        # Analyze performance
        avg_processing_time = np.mean(performance_times)
        max_processing_time = np.max(performance_times)
        p95_processing_time = np.percentile(performance_times, 95)
        
        # Verify performance targets
        assert avg_processing_time < 0.1, f"Average processing time too slow: {avg_processing_time:.3f}s"
        assert max_processing_time < 0.5, f"Max processing time too slow: {max_processing_time:.3f}s"
        assert p95_processing_time < 0.2, f"95th percentile too slow: {p95_processing_time:.3f}s"
        
        # Check that system remained stable
        final_correlation = correlation_tracker._calculate_average_correlation()
        assert 0.0 < final_correlation < 1.0, f"Invalid final correlation: {final_correlation}"
        
        print("✓ Performance under regime changes test completed")
        print(f"  Average processing time: {avg_processing_time:.3f}s")
        print(f"  Max processing time: {max_processing_time:.3f}s")
        print(f"  95th percentile: {p95_processing_time:.3f}s")
        print(f"  Final correlation: {final_correlation:.3f}")


if __name__ == "__main__":
    """Run correlation regime detection tests directly"""
    
    print("📈 Starting Correlation Regime Detection Tests...")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])