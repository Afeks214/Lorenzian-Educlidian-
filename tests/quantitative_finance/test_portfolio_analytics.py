"""
Portfolio Analytics Testing Suite

This module provides comprehensive testing for portfolio analytics including:
- Information ratio, Sharpe ratio, and alpha calculations
- Tracking error and active risk measurements
- Portfolio turnover and transaction cost analysis
- Performance attribution and risk decomposition
- Portfolio optimization metrics validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import unittest.mock as mock

from src.risk.agents.portfolio_performance_monitor import (
    PortfolioPerformanceMonitor,
    PerformanceMetrics,
    PerformanceAlert,
    PerformanceAlertLevel,
    OptimizationRecommendation
)
from src.risk.agents.performance_attribution import (
    PerformanceAttributionEngine,
    AttributionMethod,
    PerformanceMetric
)
from src.core.events import EventBus, Event, EventType


class TestPortfolioAnalytics:
    """Test suite for comprehensive portfolio analytics"""
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Generate sample portfolio data for testing"""
        np.random.seed(42)
        
        # Generate sample data for 252 trading days
        n_days = 252
        n_assets = 10
        
        # Asset returns with realistic parameters
        asset_returns = np.random.multivariate_normal(
            mean=np.random.normal(0.0008, 0.0002, n_assets),
            cov=self._generate_realistic_covariance_matrix(n_assets),
            size=n_days
        )
        
        # Portfolio weights (time-varying)
        weights = np.random.dirichlet(np.ones(n_assets), n_days)
        
        # Portfolio returns
        portfolio_returns = np.sum(asset_returns * weights, axis=1)
        
        # Benchmark returns (market index)
        benchmark_returns = np.random.normal(0.0005, 0.012, n_days)
        
        # Risk-free rate
        rf_rate = 0.02 / 252  # Daily risk-free rate
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        return {
            'timestamps': timestamps,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'asset_returns': asset_returns,
            'weights': weights,
            'rf_rate': rf_rate,
            'n_assets': n_assets,
            'n_days': n_days
        }
    
    def _generate_realistic_covariance_matrix(self, n_assets: int) -> np.ndarray:
        """Generate realistic covariance matrix for assets"""
        # Generate correlation matrix
        correlations = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlations = (correlations + correlations.T) / 2  # Make symmetric
        np.fill_diagonal(correlations, 1.0)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlations)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate volatilities
        volatilities = np.random.uniform(0.008, 0.025, n_assets)
        
        # Convert to covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        return cov_matrix
    
    def test_information_ratio_calculation(self, sample_portfolio_data):
        """Test information ratio calculation"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        # Calculate information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
        
        # Annualized information ratio
        annualized_ir = information_ratio * np.sqrt(252)
        
        # Verify calculation
        assert isinstance(information_ratio, float)
        assert isinstance(annualized_ir, float)
        assert not np.isnan(information_ratio)
        assert not np.isnan(annualized_ir)
        
        # Test typical ranges
        assert -5.0 <= annualized_ir <= 5.0  # Typical range for information ratio
    
    def test_sharpe_ratio_calculation(self, sample_portfolio_data):
        """Test Sharpe ratio calculation"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        rf_rate = sample_portfolio_data['rf_rate']
        
        # Calculate Sharpe ratio
        excess_returns = portfolio_returns - rf_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Annualized Sharpe ratio
        annualized_sharpe = sharpe_ratio * np.sqrt(252)
        
        # Verify calculation
        assert isinstance(sharpe_ratio, float)
        assert isinstance(annualized_sharpe, float)
        assert not np.isnan(sharpe_ratio)
        assert not np.isnan(annualized_sharpe)
        
        # Test typical ranges
        assert -3.0 <= annualized_sharpe <= 3.0  # Typical range for Sharpe ratio
    
    def test_alpha_calculation(self, sample_portfolio_data):
        """Test alpha calculation (CAPM)"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        rf_rate = sample_portfolio_data['rf_rate']
        
        # Calculate beta
        portfolio_excess = portfolio_returns - rf_rate
        market_excess = benchmark_returns - rf_rate
        
        covariance = np.cov(portfolio_excess, market_excess)[0, 1]
        market_variance = np.var(market_excess)
        beta = covariance / market_variance if market_variance > 0 else 0
        
        # Calculate alpha
        portfolio_mean = np.mean(portfolio_excess)
        market_mean = np.mean(market_excess)
        alpha = portfolio_mean - beta * market_mean
        
        # Annualized alpha
        annualized_alpha = alpha * 252
        
        # Verify calculation
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert isinstance(annualized_alpha, float)
        assert not np.isnan(alpha)
        assert not np.isnan(beta)
        
        # Test typical ranges
        assert -2.0 <= beta <= 2.0  # Typical range for beta
        assert -0.5 <= annualized_alpha <= 0.5  # Typical range for alpha
    
    def test_tracking_error_calculation(self, sample_portfolio_data):
        """Test tracking error calculation"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        # Calculate tracking error
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        # Annualized tracking error
        annualized_te = tracking_error * np.sqrt(252)
        
        # Verify calculation
        assert isinstance(tracking_error, float)
        assert isinstance(annualized_te, float)
        assert tracking_error >= 0
        assert annualized_te >= 0
        assert not np.isnan(tracking_error)
        
        # Test typical ranges
        assert 0.0 <= annualized_te <= 0.3  # Typical range for tracking error
    
    def test_active_risk_measurement(self, sample_portfolio_data):
        """Test active risk measurement"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        # Active risk = standard deviation of active returns
        active_returns = portfolio_returns - benchmark_returns
        active_risk = np.std(active_returns)
        
        # Annualized active risk
        annualized_active_risk = active_risk * np.sqrt(252)
        
        # Active risk decomposition
        upside_active_risk = np.std(active_returns[active_returns > 0])
        downside_active_risk = np.std(active_returns[active_returns < 0])
        
        # Verify calculations
        assert isinstance(active_risk, float)
        assert active_risk >= 0
        assert not np.isnan(active_risk)
        
        # Test components
        assert isinstance(upside_active_risk, float)
        assert isinstance(downside_active_risk, float)
        assert upside_active_risk >= 0
        assert downside_active_risk >= 0
    
    def test_portfolio_turnover_calculation(self, sample_portfolio_data):
        """Test portfolio turnover calculation"""
        weights = sample_portfolio_data['weights']
        n_days = sample_portfolio_data['n_days']
        
        # Calculate turnover
        turnover_rates = []
        for i in range(1, n_days):
            weight_change = np.abs(weights[i] - weights[i-1])
            turnover = np.sum(weight_change) / 2  # Divided by 2 to avoid double counting
            turnover_rates.append(turnover)
        
        # Average turnover
        avg_turnover = np.mean(turnover_rates)
        
        # Annualized turnover
        annualized_turnover = avg_turnover * 252
        
        # Verify calculation
        assert isinstance(avg_turnover, float)
        assert isinstance(annualized_turnover, float)
        assert 0.0 <= avg_turnover <= 2.0  # Theoretical maximum is 2.0
        assert 0.0 <= annualized_turnover <= 500.0  # Reasonable upper bound
    
    def test_transaction_cost_analysis(self, sample_portfolio_data):
        """Test transaction cost analysis"""
        weights = sample_portfolio_data['weights']
        n_days = sample_portfolio_data['n_days']
        
        # Simulate transaction costs
        bid_ask_spread = 0.001  # 10 basis points
        market_impact = 0.0005  # 5 basis points
        commission = 0.0001  # 1 basis point
        
        total_transaction_cost = bid_ask_spread + market_impact + commission
        
        # Calculate transaction costs
        transaction_costs = []
        for i in range(1, n_days):
            weight_change = np.abs(weights[i] - weights[i-1])
            daily_cost = np.sum(weight_change) * total_transaction_cost
            transaction_costs.append(daily_cost)
        
        # Average transaction cost
        avg_transaction_cost = np.mean(transaction_costs)
        
        # Annualized transaction cost
        annualized_transaction_cost = avg_transaction_cost * 252
        
        # Verify calculation
        assert isinstance(avg_transaction_cost, float)
        assert isinstance(annualized_transaction_cost, float)
        assert avg_transaction_cost >= 0
        assert annualized_transaction_cost >= 0
        
        # Test reasonable ranges
        assert 0.0 <= annualized_transaction_cost <= 0.1  # Should be less than 10%
    
    def test_performance_attribution_integration(self, sample_portfolio_data):
        """Test integration with performance attribution"""
        # Create attribution engine
        event_bus = EventBus()
        attribution_engine = PerformanceAttributionEngine(
            event_bus=event_bus,
            n_strategies=5,
            benchmark_return=0.08,
            risk_free_rate=0.02
        )
        
        # Load sample data
        timestamps = sample_portfolio_data['timestamps'][:100]  # Use subset for speed
        portfolio_returns = sample_portfolio_data['portfolio_returns'][:100]
        
        # Create mock strategy returns
        n_strategies = 5
        strategy_returns = {}
        for i in range(n_strategies):
            strategy_returns[f"Strategy_{i}"] = portfolio_returns + np.random.normal(0, 0.001, len(portfolio_returns))
        
        # Create mock weights
        weights = np.random.dirichlet(np.ones(n_strategies), len(timestamps))
        
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
        
        # Test attribution calculation
        attribution = attribution_engine.calculate_portfolio_attribution()
        
        assert attribution is not None
        assert len(attribution.strategy_contributions) == n_strategies
        assert isinstance(attribution.information_ratio, float)
    
    def test_risk_decomposition_analysis(self, sample_portfolio_data):
        """Test risk decomposition analysis"""
        asset_returns = sample_portfolio_data['asset_returns']
        weights = sample_portfolio_data['weights']
        
        # Calculate portfolio variance decomposition
        n_assets = sample_portfolio_data['n_assets']
        avg_weights = np.mean(weights, axis=0)
        
        # Covariance matrix
        cov_matrix = np.cov(asset_returns.T)
        
        # Portfolio variance
        portfolio_variance = avg_weights.T @ cov_matrix @ avg_weights
        
        # Risk contributions
        marginal_contributions = cov_matrix @ avg_weights
        risk_contributions = avg_weights * marginal_contributions / portfolio_variance
        
        # Verify risk decomposition
        assert isinstance(portfolio_variance, float)
        assert portfolio_variance >= 0
        assert len(risk_contributions) == n_assets
        assert abs(np.sum(risk_contributions) - 1.0) < 1e-10  # Should sum to 1
        
        # Test individual contributions
        for contrib in risk_contributions:
            assert isinstance(contrib, float)
            assert not np.isnan(contrib)
    
    def test_performance_metrics_validation(self, sample_portfolio_data):
        """Test comprehensive performance metrics validation"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        rf_rate = sample_portfolio_data['rf_rate']
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            portfolio_returns, benchmark_returns, rf_rate
        )
        
        # Validate all metrics
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['annualized_return'], float)
        assert isinstance(metrics['volatility'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['information_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)
        assert isinstance(metrics['sortino_ratio'], float)
        assert isinstance(metrics['calmar_ratio'], float)
        
        # Test ranges
        assert -1.0 <= metrics['max_drawdown'] <= 0.0
        assert metrics['volatility'] >= 0
        assert 0.0 <= metrics['hit_ratio'] <= 1.0
        
        # Test that metrics are not NaN
        for metric_name, metric_value in metrics.items():
            assert not np.isnan(metric_value), f"Metric {metric_name} is NaN"
    
    def test_rolling_analytics(self, sample_portfolio_data):
        """Test rolling analytics calculation"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        # Rolling window parameters
        window_size = 30
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = []
        for i in range(window_size, len(portfolio_returns)):
            window_returns = portfolio_returns[i-window_size:i]
            window_sharpe = np.mean(window_returns) / np.std(window_returns) if np.std(window_returns) > 0 else 0
            rolling_sharpe.append(window_sharpe)
        
        # Calculate rolling information ratio
        rolling_ir = []
        for i in range(window_size, len(portfolio_returns)):
            window_portfolio = portfolio_returns[i-window_size:i]
            window_benchmark = benchmark_returns[i-window_size:i]
            excess_returns = window_portfolio - window_benchmark
            window_ir = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            rolling_ir.append(window_ir)
        
        # Verify calculations
        assert len(rolling_sharpe) == len(portfolio_returns) - window_size
        assert len(rolling_ir) == len(portfolio_returns) - window_size
        
        # Test that rolling metrics are reasonable
        for sharpe in rolling_sharpe:
            assert isinstance(sharpe, float)
            assert not np.isnan(sharpe)
            assert -5.0 <= sharpe <= 5.0
        
        for ir in rolling_ir:
            assert isinstance(ir, float)
            assert not np.isnan(ir)
            assert -5.0 <= ir <= 5.0
    
    def test_regime_based_analytics(self, sample_portfolio_data):
        """Test regime-based analytics"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        # Define regimes based on volatility
        rolling_vol = pd.Series(portfolio_returns).rolling(20).std()
        vol_threshold = rolling_vol.median()
        
        # High volatility regime
        high_vol_mask = rolling_vol > vol_threshold
        high_vol_returns = portfolio_returns[high_vol_mask.fillna(False)]
        
        # Low volatility regime
        low_vol_mask = rolling_vol <= vol_threshold
        low_vol_returns = portfolio_returns[low_vol_mask.fillna(False)]
        
        # Calculate regime-specific metrics
        if len(high_vol_returns) > 10:
            high_vol_sharpe = np.mean(high_vol_returns) / np.std(high_vol_returns)
        else:
            high_vol_sharpe = 0
        
        if len(low_vol_returns) > 10:
            low_vol_sharpe = np.mean(low_vol_returns) / np.std(low_vol_returns)
        else:
            low_vol_sharpe = 0
        
        # Verify regime analysis
        assert isinstance(high_vol_sharpe, float)
        assert isinstance(low_vol_sharpe, float)
        assert not np.isnan(high_vol_sharpe)
        assert not np.isnan(low_vol_sharpe)
    
    def test_drawdown_analysis(self, sample_portfolio_data):
        """Test comprehensive drawdown analysis"""
        portfolio_returns = sample_portfolio_data['portfolio_returns']
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Drawdown statistics
        max_drawdown = np.min(drawdown)
        avg_drawdown = np.mean(drawdown[drawdown < 0])
        
        # Drawdown duration
        in_drawdown = drawdown < -0.01  # 1% drawdown threshold
        drawdown_periods = []
        current_period = 0
        
        for is_in_dd in in_drawdown:
            if is_in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        # Verify drawdown analysis
        assert isinstance(max_drawdown, float)
        assert max_drawdown <= 0  # Should be negative
        assert isinstance(avg_drawdown, float)
        assert len(drawdown_periods) >= 0
        
        if len(drawdown_periods) > 0:
            avg_drawdown_duration = np.mean(drawdown_periods)
            assert isinstance(avg_drawdown_duration, float)
            assert avg_drawdown_duration > 0
    
    def _calculate_comprehensive_metrics(self, portfolio_returns, benchmark_returns, rf_rate):
        """Helper method to calculate comprehensive performance metrics"""
        # Return metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Excess returns
        excess_returns = portfolio_returns - rf_rate
        benchmark_excess = benchmark_returns - rf_rate
        active_returns = portfolio_returns - benchmark_returns
        
        # Risk-adjusted metrics
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0
        
        # Downside metrics
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Hit ratio
        hit_ratio = np.mean(active_returns > 0)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'hit_ratio': hit_ratio
        }


class TestPortfolioOptimizationMetrics:
    """Test suite for portfolio optimization metrics"""
    
    def test_mean_variance_optimization_metrics(self):
        """Test mean-variance optimization metrics"""
        # Sample expected returns and covariance matrix
        n_assets = 5
        expected_returns = np.array([0.08, 0.10, 0.12, 0.09, 0.11])
        
        # Generate covariance matrix
        correlations = np.array([
            [1.00, 0.30, 0.20, 0.10, 0.15],
            [0.30, 1.00, 0.25, 0.15, 0.20],
            [0.20, 0.25, 1.00, 0.30, 0.25],
            [0.10, 0.15, 0.30, 1.00, 0.20],
            [0.15, 0.20, 0.25, 0.20, 1.00]
        ])
        
        volatilities = np.array([0.15, 0.18, 0.22, 0.16, 0.20])
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Equal weight portfolio
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        rf_rate = 0.02
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
        
        # Verify calculations
        assert isinstance(portfolio_return, float)
        assert isinstance(portfolio_volatility, float)
        assert isinstance(sharpe_ratio, float)
        assert 0.0 <= portfolio_return <= 0.5
        assert 0.0 <= portfolio_volatility <= 1.0
        assert -2.0 <= sharpe_ratio <= 2.0
    
    def test_risk_parity_metrics(self):
        """Test risk parity optimization metrics"""
        # Sample data
        n_assets = 4
        volatilities = np.array([0.15, 0.18, 0.20, 0.16])
        correlations = np.array([
            [1.00, 0.30, 0.20, 0.25],
            [0.30, 1.00, 0.35, 0.20],
            [0.20, 0.35, 1.00, 0.30],
            [0.25, 0.20, 0.30, 1.00]
        ])
        
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Equal risk contribution target
        target_risk_contributions = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Initial equal weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Calculate actual risk contributions
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_contributions = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance
        
        # Risk parity error
        risk_parity_error = np.sum(np.abs(risk_contributions - target_risk_contributions))
        
        # Verify calculations
        assert len(risk_contributions) == n_assets
        assert abs(np.sum(risk_contributions) - 1.0) < 1e-10
        assert isinstance(risk_parity_error, float)
        assert risk_parity_error >= 0
    
    def test_black_litterman_metrics(self):
        """Test Black-Litterman optimization metrics"""
        # This tests the framework for Black-Litterman implementation
        
        # Market capitalization weights (equilibrium)
        market_caps = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Implied equilibrium returns
        risk_aversion = 3.0
        cov_matrix = np.array([
            [0.040, 0.012, 0.008, 0.004],
            [0.012, 0.050, 0.015, 0.006],
            [0.008, 0.015, 0.060, 0.010],
            [0.004, 0.006, 0.010, 0.030]
        ])
        
        # Calculate implied returns
        implied_returns = risk_aversion * np.dot(cov_matrix, market_caps)
        
        # Verify structure
        assert len(implied_returns) == len(market_caps)
        assert isinstance(implied_returns, np.ndarray)
        assert all(isinstance(ret, float) for ret in implied_returns)


if __name__ == "__main__":
    pytest.main([__file__])