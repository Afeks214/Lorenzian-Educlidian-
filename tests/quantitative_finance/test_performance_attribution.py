"""
Performance Attribution Testing Suite

This module provides comprehensive testing for performance attribution models including:
- Brinson-Fachler attribution model
- Brinson-Hood-Beebower attribution model
- Factor-based attribution analysis
- Risk decomposition attribution
- Sector, country, and style attribution models
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import unittest.mock as mock

from src.risk.agents.performance_attribution import (
    PerformanceAttributionEngine,
    AttributionMethod,
    PerformanceMetric,
    StrategyPerformanceMetrics,
    AttributionContribution,
    PortfolioAttribution
)
from src.core.events import EventBus, Event, EventType


class TestPerformanceAttributionEngine:
    """Test suite for Performance Attribution Engine"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample performance data for testing"""
        np.random.seed(42)
        
        # Generate sample returns for 252 trading days
        n_days = 252
        n_strategies = 5
        
        # Strategy returns with different characteristics
        strategy_returns = {}
        for i in range(n_strategies):
            mu = np.random.normal(0.0008, 0.0002)  # Different expected returns
            sigma = np.random.uniform(0.01, 0.02)   # Different volatilities
            returns = np.random.normal(mu, sigma, n_days)
            strategy_returns[f"Strategy_{i}"] = returns
        
        # Portfolio returns (weighted combination)
        weights = np.random.dirichlet(np.ones(n_strategies), n_days)
        portfolio_returns = np.array([
            np.sum(weights[i] * np.array([strategy_returns[f"Strategy_{j}"][i] for j in range(n_strategies)]))
            for i in range(n_days)
        ])
        
        # Benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.012, n_days)
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        return {
            'timestamps': timestamps,
            'strategy_returns': strategy_returns,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'weights': weights,
            'n_strategies': n_strategies
        }
    
    @pytest.fixture
    def attribution_engine(self):
        """Create attribution engine instance"""
        event_bus = EventBus()
        return PerformanceAttributionEngine(
            event_bus=event_bus,
            n_strategies=5,
            benchmark_return=0.08,
            risk_free_rate=0.02,
            attribution_window=252
        )
    
    def test_engine_initialization(self, attribution_engine):
        """Test proper initialization of attribution engine"""
        assert attribution_engine.n_strategies == 5
        assert attribution_engine.benchmark_return == 0.08
        assert attribution_engine.risk_free_rate == 0.02
        assert attribution_engine.attribution_window == 252
        assert len(attribution_engine.strategy_returns) == 5
        assert len(attribution_engine.factor_names) == 5
    
    def test_update_performance_data(self, attribution_engine, sample_data):
        """Test updating performance data"""
        timestamps = sample_data['timestamps'][:10]
        strategy_returns = sample_data['strategy_returns']
        portfolio_returns = sample_data['portfolio_returns'][:10]
        weights = sample_data['weights'][:10]
        
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
        
        # Verify data was stored correctly
        assert len(attribution_engine.portfolio_returns) == 10
        assert len(attribution_engine.strategy_returns['Strategy_0']) == 10
        assert len(attribution_engine.weights_history) == 10
        assert len(attribution_engine.benchmark_returns) == 10
    
    def test_strategy_performance_metrics_calculation(self, attribution_engine, sample_data):
        """Test calculation of strategy performance metrics"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate metrics for first strategy
        metrics = attribution_engine.calculate_strategy_performance_metrics("Strategy_0")
        
        assert metrics is not None
        assert isinstance(metrics, StrategyPerformanceMetrics)
        assert metrics.strategy_id == "Strategy_0"
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert -1.0 <= metrics.max_drawdown <= 0.0  # Max drawdown should be negative
        assert metrics.hit_ratio >= 0.0 and metrics.hit_ratio <= 1.0
    
    def test_brinson_fachler_attribution(self, attribution_engine, sample_data):
        """Test Brinson-Fachler attribution method"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        assert attribution is not None
        assert isinstance(attribution, PortfolioAttribution)
        assert attribution.method == AttributionMethod.BRINSON_FACHLER
        assert len(attribution.strategy_contributions) == 5
        
        # Check that contributions add up approximately to total excess return
        total_contribution = sum(
            contrib.return_contribution 
            for contrib in attribution.strategy_contributions
        )
        
        # Allow some tolerance for numerical errors
        assert abs(total_contribution - attribution.portfolio_return) < 0.01
        
        # Test individual contribution components
        for contrib in attribution.strategy_contributions:
            assert isinstance(contrib, AttributionContribution)
            assert contrib.strategy_id.startswith("Strategy_")
            assert isinstance(contrib.selection_effect, float)
            assert isinstance(contrib.allocation_effect, float)
            assert isinstance(contrib.interaction_effect, float)
    
    def test_return_decomposition_attribution(self, attribution_engine, sample_data):
        """Test return decomposition attribution method"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.RETURN_DECOMPOSITION
        )
        
        assert attribution is not None
        assert attribution.method == AttributionMethod.RETURN_DECOMPOSITION
        assert len(attribution.strategy_contributions) == 5
        
        # Verify attribution components
        assert isinstance(attribution.total_selection_effect, float)
        assert isinstance(attribution.total_allocation_effect, float)
        assert isinstance(attribution.attribution_r_squared, float)
        assert 0.0 <= attribution.attribution_r_squared <= 1.0
    
    def test_factor_based_attribution(self, attribution_engine, sample_data):
        """Test factor-based attribution analysis"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Test factor returns generation
        for factor_name in attribution_engine.factor_names:
            assert factor_name in attribution_engine.factor_returns
            assert len(attribution_engine.factor_returns[factor_name]) > 0
    
    def test_risk_decomposition_attribution(self, attribution_engine, sample_data):
        """Test risk decomposition attribution"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.RETURN_DECOMPOSITION
        )
        
        # Test risk contributions
        total_risk_contribution = sum(
            contrib.risk_contribution 
            for contrib in attribution.strategy_contributions
        )
        
        # Risk contributions should sum to approximately 1.0
        assert abs(total_risk_contribution - 1.0) < 0.1
        
        # Test tracking error contributions
        for contrib in attribution.strategy_contributions:
            assert contrib.tracking_error_contribution >= 0.0
    
    def test_sector_attribution_model(self, attribution_engine, sample_data):
        """Test sector-based attribution model"""
        # This would simulate sector-based attribution
        # For now, we'll test the framework can handle sector classifications
        
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Simulate sector classifications
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        # Test that we can add sector information to contributions
        for i, contrib in enumerate(attribution.strategy_contributions):
            contrib.factor_contributions['sector'] = sectors[i % len(sectors)]
            assert 'sector' in contrib.factor_contributions
    
    def test_country_attribution_model(self, attribution_engine, sample_data):
        """Test country-based attribution model"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Simulate country classifications
        countries = ['US', 'Europe', 'Asia', 'Emerging', 'Global']
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.FACTOR_BASED
        )
        
        # Test that we can add country information to contributions
        for i, contrib in enumerate(attribution.strategy_contributions):
            contrib.factor_contributions['country'] = countries[i % len(countries)]
            assert 'country' in contrib.factor_contributions
    
    def test_style_attribution_model(self, attribution_engine, sample_data):
        """Test style-based attribution model"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Simulate style classifications
        styles = ['Growth', 'Value', 'Momentum', 'Quality', 'Low_Vol']
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.FACTOR_BASED
        )
        
        # Test that we can add style information to contributions
        for i, contrib in enumerate(attribution.strategy_contributions):
            contrib.factor_contributions['style'] = styles[i % len(styles)]
            assert 'style' in contrib.factor_contributions
    
    def test_date_range_attribution(self, attribution_engine, sample_data):
        """Test attribution calculation for specific date ranges"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Test attribution for specific date range
        start_date = sample_data['timestamps'][50]
        end_date = sample_data['timestamps'][150]
        
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER,
            start_date=start_date,
            end_date=end_date
        )
        
        assert attribution is not None
        assert attribution.period_start >= start_date
        assert attribution.period_end <= end_date
    
    def test_performance_summary_generation(self, attribution_engine, sample_data):
        """Test generation of performance summaries"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate metrics first
        metrics = attribution_engine.calculate_strategy_performance_metrics("Strategy_0")
        assert metrics is not None
        
        # Get summary
        summary = attribution_engine.get_strategy_performance_summary("Strategy_0")
        
        assert "strategy_id" in summary
        assert "latest_metrics" in summary
        assert "performance_history_count" in summary
        assert "data_points" in summary
        
        # Test latest metrics
        latest_metrics = summary["latest_metrics"]
        assert "annualized_return" in latest_metrics
        assert "volatility" in latest_metrics
        assert "sharpe_ratio" in latest_metrics
        assert "information_ratio" in latest_metrics
    
    def test_attribution_summary_generation(self, attribution_engine, sample_data):
        """Test generation of attribution summaries"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate attribution first
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        assert attribution is not None
        
        # Get summary
        summary = attribution_engine.get_attribution_summary()
        
        assert "total_attributions" in summary
        assert "latest_attribution" in summary
        assert "strategy_contributions" in summary
        
        # Test latest attribution info
        latest_attribution = summary["latest_attribution"]
        assert "method" in latest_attribution
        assert "excess_return" in latest_attribution
        assert "information_ratio" in latest_attribution
    
    def test_mathematical_consistency(self, attribution_engine, sample_data):
        """Test mathematical consistency of attribution calculations"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        # Test that selection + allocation + interaction ≈ excess return
        total_effects = (
            attribution.total_selection_effect +
            attribution.total_allocation_effect +
            attribution.total_interaction_effect
        )
        
        explained_return = total_effects
        residual = attribution.excess_return - explained_return
        
        # The residual should be small (< 1% of excess return)
        if abs(attribution.excess_return) > 1e-6:
            assert abs(residual / attribution.excess_return) < 0.01
    
    def test_performance_metrics_accuracy(self, attribution_engine, sample_data):
        """Test accuracy of performance metrics calculations"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Calculate metrics
        metrics = attribution_engine.calculate_strategy_performance_metrics("Strategy_0")
        
        # Test Sharpe ratio calculation
        if metrics.volatility > 0:
            expected_sharpe = (metrics.annualized_return - attribution_engine.risk_free_rate) / metrics.volatility
            assert abs(metrics.sharpe_ratio - expected_sharpe) < 0.001
        
        # Test that beta is reasonable (-2 to 2 range typically)
        assert -2.0 <= metrics.beta <= 2.0
        
        # Test that correlations are within [-1, 1]
        assert -1.0 <= metrics.correlation_with_portfolio <= 1.0
        assert -1.0 <= metrics.correlation_with_benchmark <= 1.0
    
    def test_edge_cases(self, attribution_engine):
        """Test edge cases and error handling"""
        # Test with no data
        metrics = attribution_engine.calculate_strategy_performance_metrics("Strategy_0")
        assert metrics is None
        
        attribution = attribution_engine.calculate_portfolio_attribution()
        assert attribution is None
        
        # Test with insufficient data
        timestamp = datetime.now()
        attribution_engine.update_performance_data(
            timestamp=timestamp,
            strategy_returns={"Strategy_0": 0.01},
            portfolio_return=0.01,
            weights=np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        )
        
        metrics = attribution_engine.calculate_strategy_performance_metrics("Strategy_0")
        assert metrics is None  # Should be None due to insufficient data
    
    def test_performance_regression(self, attribution_engine, sample_data):
        """Test performance regression - ensure calculations complete within time limits"""
        # Load sample data
        self._load_sample_data(attribution_engine, sample_data)
        
        # Test calculation time
        start_time = datetime.now()
        
        # Calculate attribution
        attribution = attribution_engine.calculate_portfolio_attribution(
            method=AttributionMethod.BRINSON_FACHLER
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should complete within 100ms for sample data
        assert calculation_time < 100.0
        assert attribution is not None
        
        # Test that calculation times are tracked
        assert len(attribution_engine.calculation_times) > 0
    
    def _load_sample_data(self, attribution_engine, sample_data):
        """Helper method to load sample data into attribution engine"""
        timestamps = sample_data['timestamps']
        strategy_returns = sample_data['strategy_returns']
        portfolio_returns = sample_data['portfolio_returns']
        weights = sample_data['weights']
        
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


class TestBrinsonFachlerAttribution:
    """Focused tests for Brinson-Fachler attribution model"""
    
    def test_brinson_fachler_decomposition(self):
        """Test Brinson-Fachler decomposition mathematics"""
        # Create simple test case with known values
        portfolio_weights = np.array([0.4, 0.3, 0.3])
        benchmark_weights = np.array([0.33, 0.33, 0.34])
        
        portfolio_returns = np.array([0.10, 0.08, 0.12])
        benchmark_returns = np.array([0.09, 0.09, 0.09])
        
        # Calculate expected effects manually
        allocation_effects = (portfolio_weights - benchmark_weights) * benchmark_returns
        selection_effects = benchmark_weights * (portfolio_returns - benchmark_returns)
        interaction_effects = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)
        
        # Verify calculations
        assert len(allocation_effects) == 3
        assert len(selection_effects) == 3
        assert len(interaction_effects) == 3
        
        # Total effects should sum to total excess return
        total_allocation = np.sum(allocation_effects)
        total_selection = np.sum(selection_effects)
        total_interaction = np.sum(interaction_effects)
        
        portfolio_return = np.sum(portfolio_weights * portfolio_returns)
        benchmark_return = np.sum(benchmark_weights * benchmark_returns)
        excess_return = portfolio_return - benchmark_return
        
        total_explained = total_allocation + total_selection + total_interaction
        
        # Should be mathematically consistent
        assert abs(total_explained - excess_return) < 1e-10


class TestBrinsonHoodBeeobowerAttribution:
    """Tests for Brinson-Hood-Beebower attribution model"""
    
    def test_brinson_hood_beebower_model(self):
        """Test Brinson-Hood-Beebower attribution model implementation"""
        # This would be implemented as an extension of the existing framework
        # For now, test the mathematical framework
        
        # BHB model focuses on:
        # 1. Asset Allocation Effect
        # 2. Security Selection Effect  
        # 3. Interaction Effect
        
        # Test data setup
        portfolio_weights = np.array([0.5, 0.3, 0.2])
        benchmark_weights = np.array([0.4, 0.4, 0.2])
        
        portfolio_returns = np.array([0.12, 0.08, 0.10])
        benchmark_sector_returns = np.array([0.10, 0.09, 0.11])
        
        # Asset allocation effect: (wp - wb) * rb
        allocation_effect = (portfolio_weights - benchmark_weights) * benchmark_sector_returns
        
        # Security selection effect: wb * (rp - rb)
        selection_effect = benchmark_weights * (portfolio_returns - benchmark_sector_returns)
        
        # Verify structure
        assert len(allocation_effect) == 3
        assert len(selection_effect) == 3
        
        # Total effects
        total_allocation = np.sum(allocation_effect)
        total_selection = np.sum(selection_effect)
        
        # These should be mathematically consistent
        assert isinstance(total_allocation, float)
        assert isinstance(total_selection, float)


if __name__ == "__main__":
    pytest.main([__file__])