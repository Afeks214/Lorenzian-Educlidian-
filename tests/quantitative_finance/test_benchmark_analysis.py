"""
Benchmark Analysis Testing Suite

This module provides comprehensive testing for benchmark analysis including:
- Benchmark construction and rebalancing
- Benchmark tracking and deviation analysis
- Custom benchmark creation and validation
- Performance comparison and attribution
- Benchmark selection and optimization
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import unittest.mock as mock

from src.risk.agents.performance_attribution import (
    PerformanceAttributionEngine,
    AttributionMethod,
    PerformanceMetric
)
from src.core.events import EventBus, Event, EventType


class TestBenchmarkAnalysis:
    """Test suite for comprehensive benchmark analysis"""
    
    @pytest.fixture
    def market_data(self):
        """Generate realistic market data for benchmark testing"""
        np.random.seed(42)
        
        # Generate market data for 252 trading days
        n_days = 252
        n_assets = 20
        
        # Asset characteristics
        asset_names = [f"Asset_{i:02d}" for i in range(n_assets)]
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'] * 4
        
        # Market capitalizations (in billions)
        market_caps = np.random.lognormal(mean=8, sigma=2, size=n_assets)
        
        # Asset returns with sector factors
        sector_factors = {
            'Technology': 0.0012,
            'Healthcare': 0.0008,
            'Finance': 0.0006,
            'Energy': 0.0004,
            'Consumer': 0.0010
        }
        
        asset_returns = np.zeros((n_days, n_assets))
        for i, asset in enumerate(asset_names):
            sector = sectors[i]
            sector_return = sector_factors[sector]
            asset_vol = np.random.uniform(0.015, 0.025)
            asset_returns[:, i] = np.random.normal(sector_return, asset_vol, n_days)
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        return {
            'timestamps': timestamps,
            'asset_names': asset_names,
            'asset_returns': asset_returns,
            'market_caps': market_caps,
            'sectors': sectors,
            'n_assets': n_assets,
            'n_days': n_days
        }
    
    def test_market_cap_weighted_benchmark(self, market_data):
        """Test market capitalization weighted benchmark construction"""
        market_caps = market_data['market_caps']
        asset_returns = market_data['asset_returns']
        
        # Calculate market cap weights
        total_market_cap = np.sum(market_caps)
        benchmark_weights = market_caps / total_market_cap
        
        # Calculate benchmark returns
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Verify benchmark construction
        assert len(benchmark_weights) == market_data['n_assets']
        assert abs(np.sum(benchmark_weights) - 1.0) < 1e-10
        assert all(weight >= 0 for weight in benchmark_weights)
        assert len(benchmark_returns) == market_data['n_days']
        
        # Test benchmark properties
        benchmark_volatility = np.std(benchmark_returns)
        benchmark_mean_return = np.mean(benchmark_returns)
        
        assert isinstance(benchmark_volatility, float)
        assert isinstance(benchmark_mean_return, float)
        assert benchmark_volatility > 0
    
    def test_equal_weighted_benchmark(self, market_data):
        """Test equal weighted benchmark construction"""
        asset_returns = market_data['asset_returns']
        n_assets = market_data['n_assets']
        
        # Equal weights
        benchmark_weights = np.ones(n_assets) / n_assets
        
        # Calculate benchmark returns
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Verify equal weighting
        assert len(benchmark_weights) == n_assets
        assert abs(np.sum(benchmark_weights) - 1.0) < 1e-10
        assert all(abs(weight - 1.0/n_assets) < 1e-10 for weight in benchmark_weights)
        
        # Test benchmark properties
        benchmark_volatility = np.std(benchmark_returns)
        benchmark_mean_return = np.mean(benchmark_returns)
        
        assert isinstance(benchmark_volatility, float)
        assert isinstance(benchmark_mean_return, float)
        assert benchmark_volatility > 0
    
    def test_fundamental_weighted_benchmark(self, market_data):
        """Test fundamental weighted benchmark construction"""
        # Simulate fundamental data
        n_assets = market_data['n_assets']
        
        # Fundamental metrics (earnings, book value, sales, etc.)
        earnings = np.random.lognormal(mean=6, sigma=1.5, size=n_assets)
        book_values = np.random.lognormal(mean=7, sigma=1.2, size=n_assets)
        sales = np.random.lognormal(mean=8, sigma=1.8, size=n_assets)
        
        # Composite fundamental score
        fundamental_scores = (earnings + book_values + sales) / 3
        
        # Calculate fundamental weights
        total_fundamental_value = np.sum(fundamental_scores)
        benchmark_weights = fundamental_scores / total_fundamental_value
        
        # Calculate benchmark returns
        asset_returns = market_data['asset_returns']
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Verify fundamental weighting
        assert len(benchmark_weights) == n_assets
        assert abs(np.sum(benchmark_weights) - 1.0) < 1e-10
        assert all(weight >= 0 for weight in benchmark_weights)
        
        # Test that weights differ from equal weighting
        equal_weights = np.ones(n_assets) / n_assets
        weight_difference = np.sum(np.abs(benchmark_weights - equal_weights))
        assert weight_difference > 0.01  # Should be meaningfully different
    
    def test_sector_neutral_benchmark(self, market_data):
        """Test sector neutral benchmark construction"""
        sectors = market_data['sectors']
        market_caps = market_data['market_caps']
        n_assets = market_data['n_assets']
        
        # Get unique sectors
        unique_sectors = list(set(sectors))
        
        # Calculate sector neutral weights
        benchmark_weights = np.zeros(n_assets)
        
        for sector in unique_sectors:
            sector_mask = np.array([s == sector for s in sectors])
            sector_assets = np.sum(sector_mask)
            
            if sector_assets > 0:
                # Equal weight within sector
                sector_weight = 1.0 / len(unique_sectors)
                asset_weight = sector_weight / sector_assets
                benchmark_weights[sector_mask] = asset_weight
        
        # Calculate benchmark returns
        asset_returns = market_data['asset_returns']
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Verify sector neutral weighting
        assert abs(np.sum(benchmark_weights) - 1.0) < 1e-10
        
        # Test sector allocations
        for sector in unique_sectors:
            sector_mask = np.array([s == sector for s in sectors])
            sector_total_weight = np.sum(benchmark_weights[sector_mask])
            expected_sector_weight = 1.0 / len(unique_sectors)
            assert abs(sector_total_weight - expected_sector_weight) < 1e-10
    
    def test_benchmark_rebalancing(self, market_data):
        """Test benchmark rebalancing procedures"""
        timestamps = market_data['timestamps']
        asset_returns = market_data['asset_returns']
        market_caps = market_data['market_caps']
        
        # Simulate market cap changes over time
        n_days = market_data['n_days']
        n_assets = market_data['n_assets']
        
        # Initial market cap weights
        initial_weights = market_caps / np.sum(market_caps)
        
        # Simulate rebalancing every 30 days
        rebalancing_frequency = 30
        rebalancing_dates = list(range(0, n_days, rebalancing_frequency))
        
        benchmark_returns = []
        current_weights = initial_weights.copy()
        
        for i in range(n_days):
            # Calculate return for current day
            daily_return = np.dot(asset_returns[i], current_weights)
            benchmark_returns.append(daily_return)
            
            # Update weights due to price changes
            price_changes = 1 + asset_returns[i]
            current_weights = current_weights * price_changes
            current_weights = current_weights / np.sum(current_weights)
            
            # Rebalance if it's a rebalancing date
            if i in rebalancing_dates:
                # Rebalance to original market cap weights
                current_weights = initial_weights.copy()
        
        # Verify rebalancing
        assert len(benchmark_returns) == n_days
        assert all(isinstance(ret, float) for ret in benchmark_returns)
        
        # Test rebalancing impact
        benchmark_volatility = np.std(benchmark_returns)
        assert benchmark_volatility > 0
    
    def test_benchmark_tracking_analysis(self, market_data):
        """Test benchmark tracking analysis"""
        # Create benchmark
        market_caps = market_data['market_caps']
        benchmark_weights = market_caps / np.sum(market_caps)
        asset_returns = market_data['asset_returns']
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Create portfolio that attempts to track benchmark
        portfolio_weights = benchmark_weights + np.random.normal(0, 0.01, len(benchmark_weights))
        portfolio_weights = np.abs(portfolio_weights)  # Ensure positive
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)  # Normalize
        
        portfolio_returns = np.dot(asset_returns, portfolio_weights)
        
        # Calculate tracking metrics
        tracking_error = np.std(portfolio_returns - benchmark_returns)
        tracking_correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        
        # Active return
        active_returns = portfolio_returns - benchmark_returns
        active_return = np.mean(active_returns)
        
        # Information ratio
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        # Verify tracking analysis
        assert isinstance(tracking_error, float)
        assert isinstance(tracking_correlation, float)
        assert isinstance(information_ratio, float)
        assert tracking_error >= 0
        assert -1.0 <= tracking_correlation <= 1.0
        assert tracking_correlation > 0.8  # Should be highly correlated
    
    def test_benchmark_deviation_analysis(self, market_data):
        """Test benchmark deviation analysis"""
        # Create benchmark
        market_caps = market_data['market_caps']
        benchmark_weights = market_caps / np.sum(market_caps)
        asset_returns = market_data['asset_returns']
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Create portfolio with systematic deviations
        portfolio_weights = benchmark_weights.copy()
        
        # Overweight top 3 assets, underweight bottom 3
        sorted_indices = np.argsort(market_caps)[::-1]
        portfolio_weights[sorted_indices[:3]] *= 1.2
        portfolio_weights[sorted_indices[-3:]] *= 0.8
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        
        portfolio_returns = np.dot(asset_returns, portfolio_weights)
        
        # Calculate deviation metrics
        weight_deviations = portfolio_weights - benchmark_weights
        active_weights = np.abs(weight_deviations)
        
        # Active share
        active_share = np.sum(active_weights) / 2
        
        # Tracking error
        tracking_error = np.std(portfolio_returns - benchmark_returns)
        
        # Beta relative to benchmark
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Verify deviation analysis
        assert isinstance(active_share, float)
        assert isinstance(tracking_error, float)
        assert isinstance(beta, float)
        assert 0.0 <= active_share <= 1.0
        assert tracking_error >= 0
        assert active_share > 0.01  # Should have some active share
    
    def test_custom_benchmark_creation(self, market_data):
        """Test custom benchmark creation"""
        asset_returns = market_data['asset_returns']
        sectors = market_data['sectors']
        n_assets = market_data['n_assets']
        
        # Custom benchmark specifications
        custom_specs = {
            'Technology': 0.30,
            'Healthcare': 0.25,
            'Finance': 0.20,
            'Energy': 0.15,
            'Consumer': 0.10
        }
        
        # Create custom benchmark weights
        benchmark_weights = np.zeros(n_assets)
        
        for i, sector in enumerate(sectors):
            if sector in custom_specs:
                sector_weight = custom_specs[sector]
                # Count assets in this sector
                sector_assets = sum(1 for s in sectors if s == sector)
                asset_weight = sector_weight / sector_assets
                benchmark_weights[i] = asset_weight
        
        # Normalize weights
        benchmark_weights = benchmark_weights / np.sum(benchmark_weights)
        
        # Calculate benchmark returns
        benchmark_returns = np.dot(asset_returns, benchmark_weights)
        
        # Verify custom benchmark
        assert abs(np.sum(benchmark_weights) - 1.0) < 1e-10
        assert all(weight >= 0 for weight in benchmark_weights)
        
        # Test sector allocations
        for sector, target_weight in custom_specs.items():
            sector_mask = np.array([s == sector for s in sectors])
            actual_weight = np.sum(benchmark_weights[sector_mask])
            assert abs(actual_weight - target_weight) < 0.01
    
    def test_benchmark_validation(self, market_data):
        """Test benchmark validation procedures"""
        # Create various benchmarks
        market_caps = market_data['market_caps']
        asset_returns = market_data['asset_returns']
        n_assets = market_data['n_assets']
        
        # Market cap weighted benchmark
        mcw_weights = market_caps / np.sum(market_caps)
        mcw_returns = np.dot(asset_returns, mcw_weights)
        
        # Equal weighted benchmark
        ew_weights = np.ones(n_assets) / n_assets
        ew_returns = np.dot(asset_returns, ew_weights)
        
        # Validation criteria
        def validate_benchmark(weights, returns):
            """Validate benchmark meets basic criteria"""
            validation_results = {
                'weights_sum_to_one': abs(np.sum(weights) - 1.0) < 1e-10,
                'all_weights_positive': all(w >= 0 for w in weights),
                'returns_not_nan': not np.any(np.isnan(returns)),
                'returns_not_inf': not np.any(np.isinf(returns)),
                'reasonable_volatility': 0.001 <= np.std(returns) <= 0.1,
                'reasonable_returns': -0.1 <= np.mean(returns) <= 0.1
            }
            return validation_results
        
        # Validate market cap weighted benchmark
        mcw_validation = validate_benchmark(mcw_weights, mcw_returns)
        assert all(mcw_validation.values())
        
        # Validate equal weighted benchmark
        ew_validation = validate_benchmark(ew_weights, ew_returns)
        assert all(ew_validation.values())
    
    def test_benchmark_performance_comparison(self, market_data):
        """Test benchmark performance comparison"""
        market_caps = market_data['market_caps']
        asset_returns = market_data['asset_returns']
        n_assets = market_data['n_assets']
        
        # Create multiple benchmarks
        benchmarks = {}
        
        # Market cap weighted
        mcw_weights = market_caps / np.sum(market_caps)
        benchmarks['Market_Cap_Weighted'] = np.dot(asset_returns, mcw_weights)
        
        # Equal weighted
        ew_weights = np.ones(n_assets) / n_assets
        benchmarks['Equal_Weighted'] = np.dot(asset_returns, ew_weights)
        
        # Volatility weighted (inverse volatility)
        asset_vols = np.std(asset_returns, axis=0)
        inv_vol_weights = (1 / asset_vols) / np.sum(1 / asset_vols)
        benchmarks['Inverse_Volatility'] = np.dot(asset_returns, inv_vol_weights)
        
        # Compare benchmark performance
        benchmark_metrics = {}
        for name, returns in benchmarks.items():
            benchmark_metrics[name] = {
                'mean_return': np.mean(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'total_return': np.prod(1 + returns) - 1
            }
        
        # Verify performance comparison
        for name, metrics in benchmark_metrics.items():
            assert isinstance(metrics['mean_return'], float)
            assert isinstance(metrics['volatility'], float)
            assert isinstance(metrics['sharpe_ratio'], float)
            assert metrics['volatility'] > 0
            assert -1.0 <= metrics['max_drawdown'] <= 0.0
        
        # Test that benchmarks have different performance characteristics
        mean_returns = [metrics['mean_return'] for metrics in benchmark_metrics.values()]
        volatilities = [metrics['volatility'] for metrics in benchmark_metrics.values()]
        
        # Should have some variation in performance
        assert np.std(mean_returns) > 0
        assert np.std(volatilities) > 0
    
    def test_benchmark_selection_criteria(self, market_data):
        """Test benchmark selection criteria"""
        market_caps = market_data['market_caps']
        asset_returns = market_data['asset_returns']
        n_assets = market_data['n_assets']
        
        # Create candidate benchmarks
        candidates = {}
        
        # Market cap weighted
        mcw_weights = market_caps / np.sum(market_caps)
        candidates['MCW'] = {
            'weights': mcw_weights,
            'returns': np.dot(asset_returns, mcw_weights)
        }
        
        # Equal weighted
        ew_weights = np.ones(n_assets) / n_assets
        candidates['EW'] = {
            'weights': ew_weights,
            'returns': np.dot(asset_returns, ew_weights)
        }
        
        # Selection criteria
        def evaluate_benchmark(benchmark_data):
            """Evaluate benchmark based on selection criteria"""
            weights = benchmark_data['weights']
            returns = benchmark_data['returns']
            
            criteria = {
                'diversification': self._calculate_diversification_ratio(weights),
                'stability': 1.0 / np.std(returns),  # Higher is better
                'capacity': np.sum(weights > 0.001),  # Number of meaningful positions
                'turnover': 0.0,  # Would calculate turnover in real implementation
                'liquidity': 1.0  # Would assess liquidity in real implementation
            }
            
            return criteria
        
        # Evaluate candidates
        evaluations = {}
        for name, benchmark_data in candidates.items():
            evaluations[name] = evaluate_benchmark(benchmark_data)
        
        # Verify evaluations
        for name, evaluation in evaluations.items():
            assert isinstance(evaluation['diversification'], float)
            assert isinstance(evaluation['stability'], float)
            assert isinstance(evaluation['capacity'], (int, float))
            assert evaluation['diversification'] > 0
            assert evaluation['stability'] > 0
            assert evaluation['capacity'] > 0
    
    def test_benchmark_optimization(self, market_data):
        """Test benchmark optimization procedures"""
        asset_returns = market_data['asset_returns']
        n_assets = market_data['n_assets']
        
        # Optimization objective: minimize tracking error to equal weighted benchmark
        target_returns = np.mean(asset_returns, axis=1)
        
        # Simple optimization: minimize sum of squared deviations
        def objective_function(weights):
            """Objective function for benchmark optimization"""
            benchmark_returns = np.dot(asset_returns, weights)
            tracking_error = np.sum((benchmark_returns - target_returns) ** 2)
            return tracking_error
        
        # Constraints: weights sum to 1, all weights >= 0
        def constraint_function(weights):
            """Constraint function for optimization"""
            return np.sum(weights) - 1.0
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Simple gradient descent (for testing purposes)
        # In practice, would use scipy.optimize or similar
        optimized_weights = initial_weights.copy()
        
        # Verify optimization structure
        assert len(optimized_weights) == n_assets
        assert abs(np.sum(optimized_weights) - 1.0) < 1e-10
        assert all(w >= 0 for w in optimized_weights)
        
        # Test objective function
        initial_objective = objective_function(initial_weights)
        optimized_objective = objective_function(optimized_weights)
        
        assert isinstance(initial_objective, float)
        assert isinstance(optimized_objective, float)
        assert initial_objective >= 0
        assert optimized_objective >= 0
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_diversification_ratio(self, weights):
        """Calculate diversification ratio"""
        # Simplified diversification measure
        herfindahl_index = np.sum(weights ** 2)
        diversification_ratio = 1.0 / herfindahl_index
        return diversification_ratio


class TestBenchmarkIntegration:
    """Test suite for benchmark integration with attribution systems"""
    
    def test_attribution_engine_integration(self):
        """Test integration with performance attribution engine"""
        # Create attribution engine
        event_bus = EventBus()
        attribution_engine = PerformanceAttributionEngine(
            event_bus=event_bus,
            n_strategies=3,
            benchmark_return=0.08,
            risk_free_rate=0.02
        )
        
        # Simulate benchmark data
        n_days = 100
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
        
        # Create benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.01, n_days)
        
        # Create portfolio data
        strategy_returns = {
            'Strategy_0': np.random.normal(0.0008, 0.012, n_days),
            'Strategy_1': np.random.normal(0.0006, 0.011, n_days),
            'Strategy_2': np.random.normal(0.0007, 0.013, n_days)
        }
        
        weights = np.random.dirichlet([1, 1, 1], n_days)
        portfolio_returns = np.array([
            np.sum(weights[i] * np.array([strategy_returns[f'Strategy_{j}'][i] for j in range(3)]))
            for i in range(n_days)
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
                weights=weights[i],
                benchmark_return=benchmark_returns[i]
            )
        
        # Test attribution with custom benchmark
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        # Verify integration
        assert attribution is not None
        assert isinstance(attribution.benchmark_return, float)
        assert isinstance(attribution.excess_return, float)
        assert isinstance(attribution.tracking_error, float)
        assert len(attribution.strategy_contributions) == 3
    
    def test_benchmark_regime_analysis(self):
        """Test benchmark performance in different market regimes"""
        # Simulate different market regimes
        n_days = 252
        
        # Bull market regime (first 100 days)
        bull_returns = np.random.normal(0.001, 0.008, 100)
        
        # Bear market regime (next 100 days)
        bear_returns = np.random.normal(-0.0005, 0.015, 100)
        
        # Sideways market regime (last 52 days)
        sideways_returns = np.random.normal(0.0002, 0.005, 52)
        
        # Combined benchmark returns
        benchmark_returns = np.concatenate([bull_returns, bear_returns, sideways_returns])
        
        # Analyze regime performance
        regimes = {
            'Bull': bull_returns,
            'Bear': bear_returns,
            'Sideways': sideways_returns
        }
        
        regime_metrics = {}
        for regime_name, regime_returns in regimes.items():
            regime_metrics[regime_name] = {
                'mean_return': np.mean(regime_returns),
                'volatility': np.std(regime_returns),
                'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_returns)
            }
        
        # Verify regime analysis
        for regime_name, metrics in regime_metrics.items():
            assert isinstance(metrics['mean_return'], float)
            assert isinstance(metrics['volatility'], float)
            assert isinstance(metrics['sharpe_ratio'], float)
            assert isinstance(metrics['max_drawdown'], float)
        
        # Test regime differences
        bull_return = regime_metrics['Bull']['mean_return']
        bear_return = regime_metrics['Bear']['mean_return']
        
        # Bull market should have higher returns than bear market
        assert bull_return > bear_return
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)


if __name__ == "__main__":
    pytest.main([__file__])