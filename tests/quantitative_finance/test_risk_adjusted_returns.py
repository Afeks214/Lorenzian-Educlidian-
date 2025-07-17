"""
Risk-Adjusted Returns Testing Suite

This module provides comprehensive testing for risk-adjusted returns including:
- Sortino ratio, Calmar ratio, and maximum drawdown
- Value at Risk (VaR) and Expected Shortfall (ES) metrics
- Regime-based performance analysis
- Downside risk measurements
- Advanced risk-adjusted performance metrics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.risk.core.var_calculator import VaRCalculator
from src.risk.agents.performance_attribution import PerformanceAttributionEngine
from src.core.events import EventBus


class TestRiskAdjustedReturns:
    """Test suite for risk-adjusted return calculations"""
    
    @pytest.fixture
    def return_data(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        
        # Generate returns for different market conditions
        n_days = 252
        
        # Base returns with time-varying volatility
        base_returns = np.random.normal(0.0008, 0.012, n_days)
        
        # Add volatility clustering
        volatility_process = np.random.normal(0.012, 0.003, n_days)
        volatility_process = np.abs(volatility_process)  # Ensure positive volatility
        
        # Generate returns with volatility clustering
        returns = np.random.normal(0.0008, 1.0, n_days) * volatility_process
        
        # Risk-free rate
        rf_rate = 0.02 / 252  # Daily risk-free rate
        
        # Benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.01, n_days)
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        return {
            'returns': returns,
            'benchmark_returns': benchmark_returns,
            'rf_rate': rf_rate,
            'timestamps': timestamps,
            'volatility_process': volatility_process
        }
    
    def test_sortino_ratio_calculation(self, return_data):
        """Test Sortino ratio calculation"""
        returns = return_data['returns']
        rf_rate = return_data['rf_rate']
        
        # Calculate Sortino ratio
        excess_returns = returns - rf_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            sortino_ratio = np.mean(excess_returns) / downside_deviation
        else:
            sortino_ratio = np.inf
        
        # Annualized Sortino ratio
        annualized_sortino = sortino_ratio * np.sqrt(252)
        
        # Verify calculation
        assert isinstance(sortino_ratio, float)
        assert isinstance(annualized_sortino, float)
        assert not np.isnan(sortino_ratio)
        
        # Test that Sortino ratio is reasonable
        if not np.isinf(sortino_ratio):
            assert -5.0 <= annualized_sortino <= 5.0
    
    def test_calmar_ratio_calculation(self, return_data):
        """Test Calmar ratio calculation"""
        returns = return_data['returns']
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate annualized return
        total_return = cumulative_returns[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = np.inf
        
        # Verify calculation
        assert isinstance(max_drawdown, float)
        assert isinstance(calmar_ratio, float)
        assert max_drawdown <= 0
        assert not np.isnan(max_drawdown)
        
        # Test reasonable ranges
        if not np.isinf(calmar_ratio):
            assert -10.0 <= calmar_ratio <= 10.0
    
    def test_maximum_drawdown_calculation(self, return_data):
        """Test maximum drawdown calculation"""
        returns = return_data['returns']
        
        # Calculate cumulative wealth
        cumulative_wealth = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_wealth)
        
        # Calculate drawdown series
        drawdown = (cumulative_wealth - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Drawdown duration
        in_drawdown = drawdown < -0.01  # 1% drawdown threshold
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        # Average drawdown duration
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Verify calculations
        assert isinstance(max_drawdown, float)
        assert isinstance(avg_drawdown_duration, float)
        assert max_drawdown <= 0
        assert avg_drawdown_duration >= 0
        assert len(drawdown) == len(returns)
        assert all(dd <= 0 for dd in drawdown)
    
    def test_value_at_risk_calculation(self, return_data):
        """Test Value at Risk (VaR) calculation"""
        returns = return_data['returns']
        
        # Test different confidence levels
        confidence_levels = [0.95, 0.99, 0.999]
        
        for confidence in confidence_levels:
            # Historical VaR
            var_historical = np.percentile(returns, (1 - confidence) * 100)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            var_parametric = mean_return + std_return * stats.norm.ppf(1 - confidence)
            
            # Verify VaR calculations
            assert isinstance(var_historical, float)
            assert isinstance(var_parametric, float)
            assert var_historical <= 0  # VaR should be negative for losses
            assert var_parametric <= 0
            
            # Test that higher confidence gives more negative VaR
            if confidence > 0.95:
                var_95 = np.percentile(returns, 5)
                assert var_historical <= var_95
    
    def test_expected_shortfall_calculation(self, return_data):
        """Test Expected Shortfall (Conditional VaR) calculation"""
        returns = return_data['returns']
        
        # Test different confidence levels
        confidence_levels = [0.95, 0.99, 0.999]
        
        for confidence in confidence_levels:
            # Calculate VaR first
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            
            # Expected Shortfall: average of returns below VaR
            tail_returns = returns[returns <= var_threshold]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
            
            # Verify ES calculation
            assert isinstance(expected_shortfall, float)
            assert expected_shortfall <= var_threshold  # ES should be worse than VaR
            assert expected_shortfall <= 0  # Should be negative for losses
    
    def test_regime_based_performance_analysis(self, return_data):
        """Test regime-based performance analysis"""
        returns = return_data['returns']
        volatility_process = return_data['volatility_process']
        
        # Define regimes based on volatility
        vol_median = np.median(volatility_process)
        high_vol_regime = volatility_process > vol_median
        low_vol_regime = volatility_process <= vol_median
        
        # Analyze performance in each regime
        high_vol_returns = returns[high_vol_regime]
        low_vol_returns = returns[low_vol_regime]
        
        # Calculate regime-specific metrics
        regimes = {
            'High_Vol': high_vol_returns,
            'Low_Vol': low_vol_returns
        }
        
        regime_metrics = {}
        for regime_name, regime_returns in regimes.items():
            if len(regime_returns) > 10:  # Ensure sufficient data
                regime_metrics[regime_name] = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0,
                    'skewness': stats.skew(regime_returns),
                    'kurtosis': stats.kurtosis(regime_returns),
                    'var_95': np.percentile(regime_returns, 5),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns)
                }
        
        # Verify regime analysis
        for regime_name, metrics in regime_metrics.items():
            assert isinstance(metrics['mean_return'], float)
            assert isinstance(metrics['volatility'], float)
            assert isinstance(metrics['sharpe_ratio'], float)
            assert isinstance(metrics['skewness'], float)
            assert isinstance(metrics['kurtosis'], float)
            assert isinstance(metrics['var_95'], float)
            assert isinstance(metrics['max_drawdown'], float)
            
            # Test that metrics are reasonable
            assert metrics['volatility'] > 0
            assert metrics['var_95'] <= 0
            assert metrics['max_drawdown'] <= 0
    
    def test_downside_risk_measurements(self, return_data):
        """Test downside risk measurements"""
        returns = return_data['returns']
        rf_rate = return_data['rf_rate']
        
        # Downside deviation
        excess_returns = returns - rf_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Lower partial moments
        def lower_partial_moment(returns, threshold, order):
            """Calculate lower partial moment"""
            deviations = np.maximum(threshold - returns, 0)
            return np.mean(deviations ** order)
        
        # LPM with different orders
        lpm_1 = lower_partial_moment(returns, 0, 1)  # Mean downside deviation
        lpm_2 = lower_partial_moment(returns, 0, 2)  # Variance of downside
        
        # Omega ratio
        threshold = 0.0
        upside_returns = returns[returns > threshold]
        downside_returns = returns[returns <= threshold]
        
        upside_potential = np.mean(upside_returns - threshold) if len(upside_returns) > 0 else 0
        downside_risk = np.mean(threshold - downside_returns) if len(downside_returns) > 0 else 0
        
        omega_ratio = upside_potential / downside_risk if downside_risk > 0 else np.inf
        
        # Verify downside risk measurements
        assert isinstance(downside_deviation, float)
        assert isinstance(lpm_1, float)
        assert isinstance(lpm_2, float)
        assert isinstance(omega_ratio, float)
        
        assert downside_deviation >= 0
        assert lpm_1 >= 0
        assert lpm_2 >= 0
        assert omega_ratio >= 0
    
    def test_advanced_risk_adjusted_metrics(self, return_data):
        """Test advanced risk-adjusted performance metrics"""
        returns = return_data['returns']
        benchmark_returns = return_data['benchmark_returns']
        rf_rate = return_data['rf_rate']
        
        # Information ratio
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio
        excess_returns = returns - rf_rate
        market_excess = benchmark_returns - rf_rate
        
        if len(market_excess) > 1 and np.var(market_excess) > 0:
            beta = np.cov(excess_returns, market_excess)[0, 1] / np.var(market_excess)
            treynor_ratio = np.mean(excess_returns) / beta if beta != 0 else np.inf
        else:
            treynor_ratio = 0
        
        # Jensen's alpha
        if len(market_excess) > 1 and np.var(market_excess) > 0:
            beta = np.cov(excess_returns, market_excess)[0, 1] / np.var(market_excess)
            jensen_alpha = np.mean(excess_returns) - beta * np.mean(market_excess)
        else:
            jensen_alpha = 0
        
        # Appraisal ratio
        residual_returns = excess_returns - beta * market_excess if 'beta' in locals() else excess_returns
        appraisal_ratio = np.mean(residual_returns) / np.std(residual_returns) if np.std(residual_returns) > 0 else 0
        
        # Verify advanced metrics
        assert isinstance(information_ratio, float)
        assert isinstance(treynor_ratio, float)
        assert isinstance(jensen_alpha, float)
        assert isinstance(appraisal_ratio, float)
        
        # Test reasonable ranges
        if not np.isinf(treynor_ratio):
            assert -10.0 <= treynor_ratio <= 10.0
        assert -5.0 <= information_ratio <= 5.0
        assert -0.5 <= jensen_alpha <= 0.5  # Daily alpha should be small
        assert -5.0 <= appraisal_ratio <= 5.0
    
    def test_tail_risk_metrics(self, return_data):
        """Test tail risk metrics"""
        returns = return_data['returns']
        
        # Tail ratio
        def tail_ratio(returns, percentile=5):
            """Calculate tail ratio (absolute value of left tail to right tail)"""
            left_tail = abs(np.percentile(returns, percentile))
            right_tail = np.percentile(returns, 100 - percentile)
            return left_tail / right_tail if right_tail > 0 else np.inf
        
        tail_ratio_5 = tail_ratio(returns, 5)
        tail_ratio_1 = tail_ratio(returns, 1)
        
        # Gain-to-pain ratio
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        gain_to_pain = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 else np.inf
        
        # Pain index
        pain_index = np.mean(np.abs(returns[returns < 0])) if len(losses) > 0 else 0
        
        # Ulcer index
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        # Verify tail risk metrics
        assert isinstance(tail_ratio_5, float)
        assert isinstance(tail_ratio_1, float)
        assert isinstance(gain_to_pain, float)
        assert isinstance(pain_index, float)
        assert isinstance(ulcer_index, float)
        
        assert tail_ratio_5 > 0
        assert tail_ratio_1 > 0
        assert gain_to_pain >= 0
        assert pain_index >= 0
        assert ulcer_index >= 0
    
    def test_rolling_risk_adjusted_metrics(self, return_data):
        """Test rolling risk-adjusted metrics"""
        returns = return_data['returns']
        rf_rate = return_data['rf_rate']
        
        # Rolling window parameters
        window_size = 30
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = []
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            excess_returns = window_returns - rf_rate
            sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            rolling_sharpe.append(sharpe)
        
        # Calculate rolling Sortino ratio
        rolling_sortino = []
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            excess_returns = window_returns - rf_rate
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                sortino = np.mean(excess_returns) / downside_deviation
            else:
                sortino = np.inf
            
            rolling_sortino.append(sortino)
        
        # Calculate rolling maximum drawdown
        rolling_max_dd = []
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            max_dd = self._calculate_max_drawdown(window_returns)
            rolling_max_dd.append(max_dd)
        
        # Verify rolling metrics
        assert len(rolling_sharpe) == len(returns) - window_size
        assert len(rolling_sortino) == len(returns) - window_size
        assert len(rolling_max_dd) == len(returns) - window_size
        
        # Test that rolling metrics are reasonable
        for sharpe in rolling_sharpe:
            assert isinstance(sharpe, float)
            assert not np.isnan(sharpe)
        
        for sortino in rolling_sortino:
            assert isinstance(sortino, float)
            assert not np.isnan(sortino)
        
        for max_dd in rolling_max_dd:
            assert isinstance(max_dd, float)
            assert max_dd <= 0
    
    def test_stress_testing_metrics(self, return_data):
        """Test stress testing metrics"""
        returns = return_data['returns']
        
        # Stress test scenarios
        scenarios = {
            'market_crash': -0.20,  # 20% single-day crash
            'volatility_spike': 0.05,  # 5% volatility spike
            'correlation_shock': 0.03   # 3% correlation shock
        }
        
        # Calculate stressed returns
        stressed_metrics = {}
        for scenario_name, shock_size in scenarios.items():
            if 'crash' in scenario_name:
                # Apply negative shock
                stressed_returns = returns.copy()
                stressed_returns[0] = shock_size  # Apply shock to first day
            else:
                # Apply volatility shock
                stressed_returns = returns + np.random.normal(0, abs(shock_size), len(returns))
            
            # Calculate metrics under stress
            stressed_metrics[scenario_name] = {
                'mean_return': np.mean(stressed_returns),
                'volatility': np.std(stressed_returns),
                'var_95': np.percentile(stressed_returns, 5),
                'max_drawdown': self._calculate_max_drawdown(stressed_returns)
            }
        
        # Verify stress testing
        for scenario_name, metrics in stressed_metrics.items():
            assert isinstance(metrics['mean_return'], float)
            assert isinstance(metrics['volatility'], float)
            assert isinstance(metrics['var_95'], float)
            assert isinstance(metrics['max_drawdown'], float)
            
            # Stress scenarios should generally worsen risk metrics
            assert metrics['volatility'] > 0
            assert metrics['var_95'] <= 0
            assert metrics['max_drawdown'] <= 0
    
    def test_var_calculator_integration(self, return_data):
        """Test integration with VaR calculator"""
        returns = return_data['returns']
        
        # Create VaR calculator
        var_calculator = VaRCalculator(
            confidence_level=0.95,
            lookback_window=100,
            n_simulations=1000
        )
        
        # Calculate VaR using different methods
        historical_var = var_calculator.calculate_historical_var(returns)
        parametric_var = var_calculator.calculate_parametric_var(returns)
        
        # Verify VaR calculations
        assert isinstance(historical_var, float)
        assert isinstance(parametric_var, float)
        assert historical_var <= 0
        assert parametric_var <= 0
    
    def test_performance_attribution_risk_metrics(self, return_data):
        """Test risk metrics in performance attribution context"""
        returns = return_data['returns']
        
        # Create attribution engine
        event_bus = EventBus()
        attribution_engine = PerformanceAttributionEngine(
            event_bus=event_bus,
            n_strategies=3,
            benchmark_return=0.08,
            risk_free_rate=0.02
        )
        
        # Test strategy performance metrics calculation
        timestamps = return_data['timestamps'][:100]  # Use subset for speed
        
        # Create mock strategy data
        strategy_returns = {
            'Strategy_0': returns[:100],
            'Strategy_1': returns[:100] + np.random.normal(0, 0.001, 100),
            'Strategy_2': returns[:100] + np.random.normal(0, 0.002, 100)
        }
        
        weights = np.random.dirichlet([1, 1, 1], 100)
        portfolio_returns = np.array([
            np.sum(weights[i] * np.array([strategy_returns[f'Strategy_{j}'][i] for j in range(3)]))
            for i in range(100)
        ])
        
        # Update attribution engine
        for i, timestamp in enumerate(timestamps):
            strategy_returns_dict = {
                strategy_id: returns[i] 
                for strategy_id, returns in strategy_returns.items()
            }
            
            attribution_engine.update_performance_data(
                timestamp=timestamp,
                strategy_returns=strategy_returns_dict,
                portfolio_return=portfolio_returns[i],
                weights=weights[i]
            )
        
        # Calculate strategy performance metrics
        metrics = attribution_engine.calculate_strategy_performance_metrics('Strategy_0')
        
        if metrics is not None:
            # Verify risk-adjusted metrics
            assert isinstance(metrics.sharpe_ratio, float)
            assert isinstance(metrics.sortino_ratio, float)
            assert isinstance(metrics.calmar_ratio, float)
            assert isinstance(metrics.max_drawdown, float)
            assert isinstance(metrics.var_95, float)
            assert isinstance(metrics.cvar_95, float)
            
            # Test ranges
            assert -1.0 <= metrics.max_drawdown <= 0.0
            assert metrics.var_95 <= 0
            assert metrics.cvar_95 <= metrics.var_95
    
    def _calculate_max_drawdown(self, returns):
        """Helper method to calculate maximum drawdown"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)


class TestAdvancedRiskMetrics:
    """Test suite for advanced risk metrics"""
    
    def test_conditional_drawdown_at_risk(self):
        """Test Conditional Drawdown at Risk (CDaR)"""
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.015, 252)
        
        # Calculate drawdown series
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        # CDaR calculation
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        # Sort drawdowns (most negative first)
        sorted_drawdowns = np.sort(drawdown)
        
        # CDaR is the average of the worst alpha% drawdowns
        n_tail = int(alpha * len(sorted_drawdowns))
        if n_tail > 0:
            cdar = np.mean(sorted_drawdowns[:n_tail])
        else:
            cdar = sorted_drawdowns[0]
        
        # Verify CDaR calculation
        assert isinstance(cdar, float)
        assert cdar <= 0
        assert cdar <= np.min(drawdown)  # CDaR should be worse than maximum drawdown
    
    def test_expected_maximum_drawdown(self):
        """Test Expected Maximum Drawdown calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.015, 252)
        
        # Monte Carlo simulation for expected maximum drawdown
        n_simulations = 100  # Reduced for testing speed
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Generate random returns with same characteristics
            sim_returns = np.random.normal(np.mean(returns), np.std(returns), len(returns))
            
            # Calculate maximum drawdown for this simulation
            cumulative_returns = np.cumprod(1 + sim_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = np.min(drawdown)
            max_drawdowns.append(max_dd)
        
        # Expected maximum drawdown
        expected_max_drawdown = np.mean(max_drawdowns)
        
        # Verify calculation
        assert isinstance(expected_max_drawdown, float)
        assert expected_max_drawdown <= 0
        assert len(max_drawdowns) == n_simulations
    
    def test_omega_ratio(self):
        """Test Omega ratio calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.015, 252)
        
        # Test different thresholds
        thresholds = [0.0, 0.0005, 0.001]
        
        for threshold in thresholds:
            # Calculate gains and losses relative to threshold
            gains = returns - threshold
            upside_gains = gains[gains > 0]
            downside_losses = gains[gains <= 0]
            
            # Omega ratio
            if len(upside_gains) > 0 and len(downside_losses) > 0:
                omega = np.mean(upside_gains) / abs(np.mean(downside_losses))
            else:
                omega = np.inf
            
            # Verify Omega ratio
            assert isinstance(omega, float)
            assert omega >= 0
            
            # Higher threshold should generally give lower Omega ratio
            if threshold > 0:
                omega_zero = self._calculate_omega_ratio(returns, 0.0)
                if not np.isinf(omega) and not np.isinf(omega_zero):
                    assert omega <= omega_zero or abs(omega - omega_zero) < 0.1
    
    def test_kappa_ratio(self):
        """Test Kappa ratio calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.015, 252)
        rf_rate = 0.02 / 252
        
        # Test different orders
        orders = [1, 2, 3]
        
        for order in orders:
            # Calculate lower partial moment
            excess_returns = returns - rf_rate
            threshold = 0.0
            
            downside_deviations = np.maximum(threshold - excess_returns, 0)
            lpm = np.mean(downside_deviations ** order)
            
            # Kappa ratio
            if lpm > 0:
                kappa = np.mean(excess_returns) / (lpm ** (1/order))
            else:
                kappa = np.inf
            
            # Verify Kappa ratio
            assert isinstance(kappa, float)
            assert not np.isnan(kappa)
    
    def _calculate_omega_ratio(self, returns, threshold):
        """Helper method to calculate Omega ratio"""
        gains = returns - threshold
        upside_gains = gains[gains > 0]
        downside_losses = gains[gains <= 0]
        
        if len(upside_gains) > 0 and len(downside_losses) > 0:
            return np.mean(upside_gains) / abs(np.mean(downside_losses))
        else:
            return np.inf


if __name__ == "__main__":
    pytest.main([__file__])