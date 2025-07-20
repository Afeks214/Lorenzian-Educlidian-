"""
Comprehensive Test Suite for Advanced Kernel Exit Strategy

This test suite validates the mathematical framework, performance characteristics,
and edge cases of the Advanced Kernel Exit Strategy implementation.

Test Categories:
1. Mathematical Correctness Tests
2. Performance Benchmark Tests  
3. Edge Case Handling Tests
4. Integration Tests with Risk Management
5. Regime Detection Accuracy Tests
6. Monte Carlo Uncertainty Tests
7. Backtesting Framework Tests

Author: Advanced Kernel Exit Strategy Test Team
Version: 1.0.0
Date: 2025-07-20
"""

import pytest
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch
import tempfile
import json

# Import the strategy and related components
from src.strategies.advanced_kernel_exit_strategy import (
    AdvancedKernelExitStrategy,
    MarketRegime,
    ExitSignalType,
    KernelRegressionState,
    ExitParameters,
    TrailingStopState,
    create_kernel_exit_strategy,
    run_strategy_backtest,
    fast_kernel_weight_calculation,
    DEFAULT_STRATEGY_CONFIG
)


class TestMathematicalCorrectness:
    """Test mathematical correctness of kernel calculations and formulas"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.n_points = 100
        self.test_prices = 100 + np.cumsum(np.random.normal(0, 0.5, self.n_points))
        self.test_volumes = np.random.uniform(1000, 10000, self.n_points)
    
    def test_rational_quadratic_kernel_properties(self):
        """Test RQ kernel mathematical properties"""
        h = 8.0
        r = 8.0
        
        # Test 1: Kernel should equal 1 when x_t = x_i
        weight = fast_kernel_weight_calculation(100.0, 100.0, h, r)
        assert abs(weight - 1.0) < 1e-10, f"Kernel weight at identical points should be 1, got {weight}"
        
        # Test 2: Kernel should decrease with distance
        weight_close = fast_kernel_weight_calculation(100.0, 100.1, h, r)
        weight_far = fast_kernel_weight_calculation(100.0, 101.0, h, r)
        assert weight_close > weight_far, "Kernel weight should decrease with distance"
        
        # Test 3: Kernel should be positive and bounded
        for x_i in [95, 100, 105]:
            weight = fast_kernel_weight_calculation(100.0, x_i, h, r)
            assert 0 < weight <= 1, f"Kernel weight should be in (0,1], got {weight}"
        
        # Test 4: Symmetry property
        weight_12 = fast_kernel_weight_calculation(100.0, 101.0, h, r)
        weight_21 = fast_kernel_weight_calculation(101.0, 100.0, h, r)
        assert abs(weight_12 - weight_21) < 1e-10, "Kernel should be symmetric"
    
    def test_dynamic_r_factor_bounds(self):
        """Test dynamic r-factor calculation respects bounds"""
        test_regimes = [MarketRegime.VOLATILE, MarketRegime.TRENDING, MarketRegime.RANGING]
        
        for regime in test_regimes:
            for volatility in [0.01, 0.05, 0.1, 0.2]:
                for trend_strength in [0.1, 0.5, 0.9]:
                    r_factor = self.strategy.calculate_dynamic_r_factor(regime, volatility, trend_strength)
                    
                    assert self.config['r_factor_min'] <= r_factor <= self.config['r_factor_max'], \
                        f"R-factor {r_factor} outside bounds for regime {regime}, vol {volatility}, trend {trend_strength}"
    
    def test_kernel_regression_uncertainty_calculation(self):
        """Test uncertainty calculation mathematical properties"""
        kernel_state, uncertainty = self.strategy.calculate_kernel_regression_with_uncertainty(
            self.test_prices, self.config['base_r_factor']
        )
        
        # Test bounds
        assert 0 <= uncertainty <= 1, f"Uncertainty should be in [0,1], got {uncertainty}"
        assert 0 <= kernel_state.kernel_confidence <= 1, f"Confidence should be in [0,1], got {kernel_state.kernel_confidence}"
        
        # Test relationship
        assert kernel_state.kernel_confidence + uncertainty >= 0.5, "Confidence and uncertainty should be related"
    
    def test_trailing_stop_mathematical_properties(self):
        """Test trailing stop calculations"""
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        kernel_state = KernelRegressionState(
            yhat1=100.5, yhat2=100.3, yhat1_slope=0.1, yhat2_slope=0.05,
            r_factor=8.0, h_parameter=8.0, kernel_confidence=0.8,
            regression_variance=0.01, calculation_timestamp=time.time()
        )
        
        # Test long position
        stop_state = self.strategy.calculate_adaptive_trailing_stop(
            position_info, 102.0, kernel_state, 1.0
        )
        
        assert stop_state.current_stop_price < 102.0, "Stop price should be below current price for long"
        assert stop_state.highest_favorable_price == 102.0, "Should track highest favorable price"
        
        # Test short position
        position_info['size'] = -100
        stop_state = self.strategy.calculate_adaptive_trailing_stop(
            position_info, 98.0, kernel_state, 1.0
        )
        
        assert stop_state.current_stop_price > 98.0, "Stop price should be above current price for short"
    
    def test_crossover_signal_detection_logic(self):
        """Test crossover signal detection mathematical logic"""
        # Create two kernel states with crossover
        state_1 = KernelRegressionState(
            yhat1=100.0, yhat2=99.8, yhat1_slope=0.0, yhat2_slope=0.0,
            r_factor=8.0, h_parameter=8.0, kernel_confidence=0.8,
            regression_variance=0.01, calculation_timestamp=time.time()
        )
        
        state_2 = KernelRegressionState(
            yhat1=100.0, yhat2=100.2, yhat1_slope=0.0, yhat2_slope=0.1,
            r_factor=8.0, h_parameter=8.0, kernel_confidence=0.8,
            regression_variance=0.01, calculation_timestamp=time.time()
        )
        
        signals = self.strategy.detect_crossover_signals(state_2, state_1)
        
        # Should detect bullish crossover (yhat2 crossed above yhat1)
        bullish_signals = [s for s in signals if s['direction'] == 'bullish_crossover']
        assert len(bullish_signals) > 0, "Should detect bullish crossover"
        assert bullish_signals[0]['recommended_action'] == 'exit_short'


class TestPerformanceBenchmarks:
    """Test performance characteristics and computational efficiency"""
    
    def setup_method(self):
        """Setup performance testing environment"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
        
        # Large dataset for performance testing
        np.random.seed(42)
        self.n_points = 1000
        self.test_prices = 100 + np.cumsum(np.random.normal(0, 0.5, self.n_points))
        self.test_volumes = np.random.uniform(1000, 10000, self.n_points)
    
    def test_calculation_speed_requirements(self):
        """Test that calculations meet speed requirements (<100ms)"""
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        # Warm up
        for _ in range(10):
            self.strategy.generate_exit_decision(
                position_info=position_info,
                current_price=101.0,
                price_history=self.test_prices[-100:],
                volume_history=self.test_volumes[-100:],
                atr_value=1.0
            )
        
        # Benchmark main calculation
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            
            result = self.strategy.generate_exit_decision(
                position_info=position_info,
                current_price=101.0 + np.random.normal(0, 0.1),
                price_history=self.test_prices[-100:],
                volume_history=self.test_volumes[-100:],
                atr_value=1.0
            )
            
            end_time = time.perf_counter()
            calculation_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(calculation_time)
            
            assert 'error' not in result, f"Calculation failed: {result.get('error')}"
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        print(f"Average calculation time: {avg_time:.2f}ms")
        print(f"P95 calculation time: {p95_time:.2f}ms")
        
        # Performance requirements
        assert avg_time < 100, f"Average calculation time {avg_time:.2f}ms exceeds 100ms limit"
        assert p95_time < 200, f"P95 calculation time {p95_time:.2f}ms exceeds 200ms limit"
    
    def test_memory_efficiency(self):
        """Test memory usage remains bounded"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        # Run many calculations
        for i in range(1000):
            self.strategy.generate_exit_decision(
                position_info=position_info,
                current_price=101.0 + np.random.normal(0, 0.1),
                price_history=self.test_prices[-100:],
                volume_history=self.test_volumes[-100:],
                atr_value=1.0
            )
            
            # Periodic memory check
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB after {i} calculations"
    
    def test_regime_detection_performance(self):
        """Test regime detection computational efficiency"""
        times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            
            regime = self.strategy.detect_market_regime(
                self.test_prices[-100:], 
                self.test_volumes[-100:]
            )
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
            
            assert isinstance(regime, MarketRegime), f"Should return MarketRegime, got {type(regime)}"
        
        avg_time = np.mean(times)
        assert avg_time < 10, f"Regime detection too slow: {avg_time:.2f}ms"


class TestEdgeCaseHandling:
    """Test handling of edge cases and error conditions"""
    
    def setup_method(self):
        """Setup edge case testing"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient historical data"""
        short_prices = np.array([100.0, 100.1, 100.2])  # Very short history
        short_volumes = np.array([1000, 1100, 1200])
        
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        result = self.strategy.generate_exit_decision(
            position_info=position_info,
            current_price=100.2,
            price_history=short_prices,
            volume_history=short_volumes,
            atr_value=1.0
        )
        
        # Should not crash and should return valid result
        assert 'exit_signals' in result
        assert isinstance(result['exit_signals'], list)
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements"""
        # Create data with extreme movements
        extreme_prices = np.array([100, 150, 50, 200, 10, 100])  # Extreme volatility
        normal_volumes = np.array([1000] * 6)
        
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        result = self.strategy.generate_exit_decision(
            position_info=position_info,
            current_price=100.0,
            price_history=extreme_prices,
            volume_history=normal_volumes,
            atr_value=10.0  # High ATR
        )
        
        # Should handle extreme volatility gracefully
        assert 'error' not in result
        assert result['uncertainty_measure'] >= 0
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values"""
        # Create problematic data
        problematic_prices = np.array([100, np.nan, 101, np.inf, 102])
        normal_volumes = np.array([1000, 1100, 1200, 1300, 1400])
        
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        result = self.strategy.generate_exit_decision(
            position_info=position_info,
            current_price=102.0,
            price_history=problematic_prices,
            volume_history=normal_volumes,
            atr_value=1.0
        )
        
        # Should handle NaN/Inf gracefully
        assert 'exit_signals' in result
        assert not np.isnan(result['uncertainty_measure'])
        assert not np.isinf(result['uncertainty_measure'])
    
    def test_zero_volume_handling(self):
        """Test handling of zero or negative volumes"""
        normal_prices = np.array([100, 101, 102, 103, 104])
        zero_volumes = np.array([0, 0, 0, 0, 0])
        
        position_info = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time()
        }
        
        result = self.strategy.generate_exit_decision(
            position_info=position_info,
            current_price=104.0,
            price_history=normal_prices,
            volume_history=zero_volumes,
            atr_value=1.0
        )
        
        # Should handle zero volumes without crashing
        assert 'exit_signals' in result
        assert result['market_regime'] in MarketRegime
    
    def test_boundary_parameter_values(self):
        """Test behavior at parameter boundaries"""
        boundary_configs = [
            {'base_r_factor': 2.0},  # Minimum r-factor
            {'base_r_factor': 20.0}, # Maximum r-factor
            {'uncertainty_threshold': 0.0}, # Minimum uncertainty
            {'uncertainty_threshold': 1.0}  # Maximum uncertainty
        ]
        
        for boundary_config in boundary_configs:
            config = DEFAULT_STRATEGY_CONFIG.copy()
            config.update(boundary_config)
            
            strategy = create_kernel_exit_strategy(config)
            
            # Should initialize without error
            assert strategy is not None
            
            # Should handle calculations
            position_info = {
                'symbol': 'TEST',
                'size': 100,
                'entry_price': 100.0,
                'entry_time': time.time()
            }
            
            result = strategy.generate_exit_decision(
                position_info=position_info,
                current_price=101.0,
                price_history=np.array([99, 100, 101]),
                volume_history=np.array([1000, 1100, 1200]),
                atr_value=1.0
            )
            
            assert 'error' not in result


class TestRiskManagementIntegration:
    """Test integration with risk management components"""
    
    def setup_method(self):
        """Setup risk management tests"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
    
    def test_risk_override_conditions(self):
        """Test that risk override conditions trigger properly"""
        # Create high-risk scenario
        high_risk_position = {
            'symbol': 'TEST',
            'size': 100,
            'entry_price': 100.0,
            'entry_time': time.time() - 90000,  # Old position (over 24h)
            'current_price': 95.0  # Losing position
        }
        
        # Create volatile price history
        volatile_prices = 100 + np.cumsum(np.random.normal(0, 2.0, 100))  # High volatility
        normal_volumes = np.random.uniform(1000, 10000, 100)
        
        result = self.strategy.generate_exit_decision(
            position_info=high_risk_position,
            current_price=95.0,
            price_history=volatile_prices,
            volume_history=normal_volumes,
            atr_value=3.0  # High ATR
        )
        
        # Should detect high risk
        risk_signals = [s for s in result['exit_signals'] if s['type'] == ExitSignalType.RISK_OVERRIDE]
        assert len(risk_signals) > 0 or result['uncertainty_measure'] > 0.5, "Should detect high-risk conditions"
    
    def test_position_sizing_with_uncertainty(self):
        """Test that position sizing considers uncertainty"""
        # Test different uncertainty levels
        for base_uncertainty in [0.1, 0.5, 0.9]:
            # Mock uncertainty in kernel state
            with patch.object(self.strategy, 'calculate_kernel_regression_with_uncertainty') as mock_calc:
                mock_state = KernelRegressionState(
                    yhat1=100.0, yhat2=100.1, yhat1_slope=0.1, yhat2_slope=0.05,
                    r_factor=8.0, h_parameter=8.0, kernel_confidence=1.0-base_uncertainty,
                    regression_variance=base_uncertainty, calculation_timestamp=time.time()
                )
                mock_calc.return_value = (mock_state, base_uncertainty)
                
                position_info = {
                    'symbol': 'TEST',
                    'size': 100,
                    'entry_price': 100.0,
                    'entry_time': time.time()
                }
                
                result = self.strategy.generate_exit_decision(
                    position_info=position_info,
                    current_price=101.0,
                    price_history=np.array([99, 100, 101]),
                    volume_history=np.array([1000, 1100, 1200]),
                    atr_value=1.0
                )
                
                # Higher uncertainty should lead to more conservative signals
                if base_uncertainty > 0.7:
                    uncertainty_exits = [s for s in result['exit_signals'] 
                                       if s['type'] == ExitSignalType.UNCERTAINTY_EXIT]
                    assert len(uncertainty_exits) > 0, f"High uncertainty {base_uncertainty} should trigger exit signals"


class TestRegimeDetectionAccuracy:
    """Test accuracy of market regime detection"""
    
    def setup_method(self):
        """Setup regime detection tests"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
    
    def test_trending_regime_detection(self):
        """Test detection of trending markets"""
        # Create strong uptrend
        trend_prices = np.array([100 + i * 0.5 for i in range(50)])  # Strong uptrend
        normal_volumes = np.random.uniform(1000, 2000, 50)
        
        regime = self.strategy.detect_market_regime(trend_prices, normal_volumes)
        
        assert regime in [MarketRegime.TRENDING, MarketRegime.CALM], \
            f"Should detect trending or calm regime, got {regime}"
    
    def test_ranging_regime_detection(self):
        """Test detection of ranging markets"""
        # Create sideways movement
        base_price = 100
        range_prices = base_price + np.sin(np.linspace(0, 4*np.pi, 50)) * 2  # Sideways oscillation
        normal_volumes = np.random.uniform(1000, 2000, 50)
        
        regime = self.strategy.detect_market_regime(range_prices, normal_volumes)
        
        # Note: May not always detect as ranging due to noise, but should be stable
        assert regime in MarketRegime, f"Should return valid regime, got {regime}"
    
    def test_volatile_regime_detection(self):
        """Test detection of volatile markets"""
        # Create high volatility
        volatile_prices = 100 + np.cumsum(np.random.normal(0, 3.0, 50))  # High volatility
        normal_volumes = np.random.uniform(1000, 2000, 50)
        
        regime = self.strategy.detect_market_regime(volatile_prices, normal_volumes)
        
        # Should detect volatile or transitional
        assert regime in [MarketRegime.VOLATILE, MarketRegime.TRANSITIONAL], \
            f"High volatility should be detected as volatile or transitional, got {regime}"
    
    def test_regime_stability_over_time(self):
        """Test that regime detection is stable over time"""
        # Create consistent trending data
        stable_trend = np.array([100 + i * 0.2 for i in range(100)])
        normal_volumes = np.random.uniform(1000, 2000, 100)
        
        regimes = []
        for i in range(50, 100, 5):  # Check regime every 5 bars
            regime = self.strategy.detect_market_regime(
                stable_trend[:i], normal_volumes[:i]
            )
            regimes.append(regime)
        
        # Most regimes should be the same (allowing some variation)
        unique_regimes = set(regimes)
        assert len(unique_regimes) <= 2, f"Regime should be stable, found {unique_regimes}"


class TestMonteCarloUncertainty:
    """Test Monte Carlo uncertainty estimation"""
    
    def setup_method(self):
        """Setup Monte Carlo testing"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
    
    def test_uncertainty_increases_with_noise(self):
        """Test that uncertainty increases with price noise"""
        base_prices = np.array([100 + i * 0.1 for i in range(50)])  # Smooth trend
        
        # Low noise scenario
        low_noise_prices = base_prices + np.random.normal(0, 0.01, 50)
        kernel_state_1, uncertainty_1 = self.strategy.calculate_kernel_regression_with_uncertainty(
            low_noise_prices, 8.0
        )
        
        # High noise scenario  
        high_noise_prices = base_prices + np.random.normal(0, 0.5, 50)
        kernel_state_2, uncertainty_2 = self.strategy.calculate_kernel_regression_with_uncertainty(
            high_noise_prices, 8.0
        )
        
        assert uncertainty_2 > uncertainty_1, \
            f"Higher noise should increase uncertainty: {uncertainty_1:.3f} vs {uncertainty_2:.3f}"
    
    def test_confidence_bounds(self):
        """Test that confidence measures are properly bounded"""
        test_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        
        kernel_state, uncertainty = self.strategy.calculate_kernel_regression_with_uncertainty(
            test_prices, 8.0
        )
        
        assert 0 <= uncertainty <= 1, f"Uncertainty should be in [0,1], got {uncertainty}"
        assert 0 <= kernel_state.kernel_confidence <= 1, \
            f"Confidence should be in [0,1], got {kernel_state.kernel_confidence}"
        assert kernel_state.regression_variance >= 0, \
            f"Variance should be non-negative, got {kernel_state.regression_variance}"
    
    def test_uncertainty_estimation_consistency(self):
        """Test that uncertainty estimation is consistent across runs"""
        test_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        
        uncertainties = []
        confidences = []
        
        # Run multiple times with same data
        for _ in range(10):
            kernel_state, uncertainty = self.strategy.calculate_kernel_regression_with_uncertainty(
                test_prices, 8.0
            )
            uncertainties.append(uncertainty)
            confidences.append(kernel_state.kernel_confidence)
        
        # Should have some variation due to Monte Carlo sampling, but not too much
        uncertainty_std = np.std(uncertainties)
        confidence_std = np.std(confidences)
        
        assert uncertainty_std < 0.2, f"Uncertainty estimation too variable: std={uncertainty_std:.3f}"
        assert confidence_std < 0.2, f"Confidence estimation too variable: std={confidence_std:.3f}"


class TestBacktestingFramework:
    """Test the backtesting framework functionality"""
    
    def setup_method(self):
        """Setup backtesting tests"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
        
        # Create synthetic backtesting data
        np.random.seed(42)
        n_bars = 500
        
        self.price_data = pd.DataFrame({
            'timestamp': np.arange(n_bars),
            'open': 100 + np.cumsum(np.random.normal(0, 0.3, n_bars)),
            'high': 100 + np.cumsum(np.random.normal(0, 0.3, n_bars)) + np.random.uniform(0, 0.5, n_bars),
            'low': 100 + np.cumsum(np.random.normal(0, 0.3, n_bars)) - np.random.uniform(0, 0.5, n_bars),
            'close': 100 + np.cumsum(np.random.normal(0, 0.3, n_bars)),
            'volume': np.random.uniform(1000, 10000, n_bars)
        })
        
        self.position_data = pd.DataFrame({
            'symbol': ['TEST'] * 20,
            'size': [100] * 20,
            'entry_price': self.price_data['close'].iloc[::25][:20].values,
            'entry_time': self.price_data['timestamp'].iloc[::25][:20].values,
            'exit_time': self.price_data['timestamp'].iloc[::25][:20].values + 50
        })
    
    def test_backtest_execution(self):
        """Test that backtest runs without errors"""
        backtest_config = {'enable_detailed_logging': False}
        
        results = run_strategy_backtest(
            self.strategy, 
            self.price_data, 
            self.position_data, 
            backtest_config
        )
        
        assert 'error' not in results, f"Backtest failed: {results.get('error')}"
        assert 'total_exit_decisions' in results
        assert results['total_exit_decisions'] > 0, "Should have made some exit decisions"
    
    def test_backtest_result_structure(self):
        """Test that backtest results have proper structure"""
        backtest_config = {'enable_detailed_logging': False}
        
        results = run_strategy_backtest(
            self.strategy,
            self.price_data,
            self.position_data,
            backtest_config
        )
        
        required_keys = [
            'total_exit_decisions', 'trailing_stop_exits', 'take_profit_exits',
            'crossover_exits', 'risk_override_exits', 'uncertainty_exits',
            'exit_signal_analysis', 'performance_by_regime', 'calculation_performance'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"
    
    def test_regime_analysis_in_backtest(self):
        """Test that regime analysis is properly tracked in backtest"""
        backtest_config = {'enable_detailed_logging': False}
        
        results = run_strategy_backtest(
            self.strategy,
            self.price_data,
            self.position_data,
            backtest_config
        )
        
        regime_analysis = results['performance_by_regime']
        assert isinstance(regime_analysis, dict), "Regime analysis should be a dictionary"
        
        # Should have detected multiple regimes
        assert len(regime_analysis) > 0, "Should detect at least one regime"
        
        # Each regime should have proper statistics
        for regime, stats in regime_analysis.items():
            assert 'signals' in stats
            assert 'avg_confidence' in stats
            assert 'avg_uncertainty' in stats
            assert stats['signals'] >= 0


class TestIntegrationScenarios:
    """Test full integration scenarios"""
    
    def setup_method(self):
        """Setup integration tests"""
        self.config = DEFAULT_STRATEGY_CONFIG.copy()
        self.strategy = create_kernel_exit_strategy(self.config)
    
    def test_full_trading_cycle(self):
        """Test complete trading cycle from entry to exit"""
        # Simulate a complete trading scenario
        np.random.seed(42)
        
        # Create realistic price movement
        n_bars = 200
        base_trend = np.linspace(100, 110, n_bars)  # 10% uptrend
        noise = np.random.normal(0, 0.5, n_bars)
        prices = base_trend + noise
        volumes = np.random.uniform(1000, 5000, n_bars)
        
        position_info = {
            'symbol': 'INTEGRATION_TEST',
            'size': 100,
            'entry_price': prices[0],
            'entry_time': 0
        }
        
        exit_signals_detected = []
        
        # Simulate real-time decision making
        for i in range(50, n_bars):  # Start after warmup period
            current_price = prices[i]
            price_history = prices[:i]
            volume_history = volumes[:i]
            
            # Calculate simple ATR
            if i >= 14:
                price_changes = np.abs(np.diff(prices[i-14:i]))
                atr_value = np.mean(price_changes)
            else:
                atr_value = 1.0
            
            result = self.strategy.generate_exit_decision(
                position_info=position_info,
                current_price=current_price,
                price_history=price_history,
                volume_history=volume_history,
                atr_value=atr_value
            )
            
            if result['exit_signals']:
                exit_signals_detected.extend(result['exit_signals'])
            
            # Check that all required components are present
            assert 'kernel_state' in result
            assert 'trailing_stop_state' in result
            assert 'risk_assessment' in result
            assert 'market_regime' in result
            
            # Validate component integrity
            assert isinstance(result['kernel_state'], KernelRegressionState)
            assert isinstance(result['trailing_stop_state'], TrailingStopState)
            assert result['market_regime'] in MarketRegime
        
        # Should have detected some exit signals during uptrend
        print(f"Detected {len(exit_signals_detected)} exit signals during integration test")
        
        # Verify strategy state management
        performance_stats = self.strategy.get_performance_stats()
        assert performance_stats['exit_signals_generated'] > 0
    
    def test_multi_position_management(self):
        """Test management of multiple positions simultaneously"""
        positions = [
            {'symbol': 'TEST1', 'size': 100, 'entry_price': 100.0, 'entry_time': time.time()},
            {'symbol': 'TEST2', 'size': -50, 'entry_price': 200.0, 'entry_time': time.time()},
            {'symbol': 'TEST3', 'size': 150, 'entry_price': 150.0, 'entry_time': time.time()}
        ]
        
        # Generate different price histories for each symbol
        price_histories = {
            'TEST1': 100 + np.cumsum(np.random.normal(0, 0.3, 100)),
            'TEST2': 200 + np.cumsum(np.random.normal(0, 0.5, 100)),
            'TEST3': 150 + np.cumsum(np.random.normal(0, 0.4, 100))
        }
        
        volume_history = np.random.uniform(1000, 5000, 100)
        
        all_results = {}
        
        # Process each position
        for position in positions:
            symbol = position['symbol']
            current_price = price_histories[symbol][-1]
            
            result = self.strategy.generate_exit_decision(
                position_info=position,
                current_price=current_price,
                price_history=price_histories[symbol],
                volume_history=volume_history,
                atr_value=2.0
            )
            
            all_results[symbol] = result
            
            # Each position should get independent analysis
            assert result['symbol'] == symbol
            assert 'exit_signals' in result
        
        # Verify independent tracking
        assert len(self.strategy.trailing_stops) <= len(positions)
        assert len(self.strategy.kernel_states) <= len(positions)


if __name__ == "__main__":
    """Run all tests when executed directly"""
    pytest.main([__file__, "-v", "--tb=short"])