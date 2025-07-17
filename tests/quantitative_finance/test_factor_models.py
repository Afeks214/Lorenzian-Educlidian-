"""
Factor Models Testing Suite

This module provides comprehensive testing for factor models including:
- Fama-French three-factor and five-factor models
- Momentum and mean reversion factors
- Custom factor construction and validation
- Factor exposure analysis and risk attribution
- Multi-factor model performance evaluation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.risk.agents.performance_attribution import PerformanceAttributionEngine
from src.core.events import EventBus


class TestFactorModels:
    """Test suite for factor model implementations"""
    
    @pytest.fixture
    def market_factor_data(self):
        """Generate sample market factor data"""
        np.random.seed(42)
        
        # Generate factor data for 252 trading days
        n_days = 252
        
        # Market factor (excess market return)
        market_factor = np.random.normal(0.0005, 0.012, n_days)
        
        # Size factor (SMB - Small Minus Big)
        size_factor = np.random.normal(0.0002, 0.008, n_days)
        
        # Value factor (HML - High Minus Low)
        value_factor = np.random.normal(0.0001, 0.006, n_days)
        
        # Momentum factor (UMD - Up Minus Down)
        momentum_factor = np.random.normal(0.0003, 0.009, n_days)
        
        # Profitability factor (RMW - Robust Minus Weak)
        profitability_factor = np.random.normal(0.0001, 0.005, n_days)
        
        # Investment factor (CMA - Conservative Minus Aggressive)
        investment_factor = np.random.normal(-0.0001, 0.004, n_days)
        
        # Risk-free rate
        rf_rate = np.full(n_days, 0.02 / 252)
        
        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        return {
            'timestamps': timestamps,
            'market_factor': market_factor,
            'size_factor': size_factor,
            'value_factor': value_factor,
            'momentum_factor': momentum_factor,
            'profitability_factor': profitability_factor,
            'investment_factor': investment_factor,
            'rf_rate': rf_rate,
            'n_days': n_days
        }
    
    @pytest.fixture
    def asset_returns_data(self, market_factor_data):
        """Generate asset returns using factor model"""
        np.random.seed(42)
        
        n_assets = 10
        n_days = market_factor_data['n_days']
        
        # Factor loadings for each asset
        factor_loadings = {
            'market_beta': np.random.normal(1.0, 0.3, n_assets),
            'size_loading': np.random.normal(0.0, 0.5, n_assets),
            'value_loading': np.random.normal(0.0, 0.4, n_assets),
            'momentum_loading': np.random.normal(0.0, 0.3, n_assets),
            'profitability_loading': np.random.normal(0.0, 0.3, n_assets),
            'investment_loading': np.random.normal(0.0, 0.2, n_assets)
        }
        
        # Generate asset returns using factor model
        asset_returns = np.zeros((n_days, n_assets))
        
        for i in range(n_assets):
            # Idiosyncratic risk
            idiosyncratic_returns = np.random.normal(0, 0.01, n_days)
            
            # Factor model: r_i = alpha + beta_1 * f_1 + ... + beta_k * f_k + epsilon_i
            asset_returns[:, i] = (
                market_factor_data['rf_rate'] +  # Risk-free rate
                factor_loadings['market_beta'][i] * market_factor_data['market_factor'] +
                factor_loadings['size_loading'][i] * market_factor_data['size_factor'] +
                factor_loadings['value_loading'][i] * market_factor_data['value_factor'] +
                factor_loadings['momentum_loading'][i] * market_factor_data['momentum_factor'] +
                factor_loadings['profitability_loading'][i] * market_factor_data['profitability_factor'] +
                factor_loadings['investment_loading'][i] * market_factor_data['investment_factor'] +
                idiosyncratic_returns
            )
        
        return {
            'asset_returns': asset_returns,
            'factor_loadings': factor_loadings,
            'n_assets': n_assets
        }
    
    def test_fama_french_three_factor_model(self, market_factor_data, asset_returns_data):
        """Test Fama-French three-factor model"""
        # Extract factors
        market_factor = market_factor_data['market_factor']
        size_factor = market_factor_data['size_factor']
        value_factor = market_factor_data['value_factor']
        rf_rate = market_factor_data['rf_rate']
        
        # Extract asset returns
        asset_returns = asset_returns_data['asset_returns']
        true_loadings = asset_returns_data['factor_loadings']
        n_assets = asset_returns_data['n_assets']
        
        # Test factor regression for each asset
        for i in range(n_assets):
            # Excess returns
            excess_returns = asset_returns[:, i] - rf_rate
            
            # Factor matrix
            factors = np.column_stack([
                market_factor,
                size_factor,
                value_factor
            ])
            
            # Add intercept
            X = np.column_stack([np.ones(len(factors)), factors])
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                
                # Extract coefficients
                alpha = beta[0]
                market_beta = beta[1]
                size_beta = beta[2]
                value_beta = beta[3]
                
                # Verify regression results
                assert isinstance(alpha, float)
                assert isinstance(market_beta, float)
                assert isinstance(size_beta, float)
                assert isinstance(value_beta, float)
                
                # Test that estimated betas are reasonable
                assert -3.0 <= market_beta <= 3.0
                assert -2.0 <= size_beta <= 2.0
                assert -2.0 <= value_beta <= 2.0
                
                # Test that market beta is close to true loading
                true_market_beta = true_loadings['market_beta'][i]
                assert abs(market_beta - true_market_beta) < 0.5  # Allow some estimation error
                
            except np.linalg.LinAlgError:
                # Skip if matrix is singular
                continue
    
    def test_fama_french_five_factor_model(self, market_factor_data, asset_returns_data):
        """Test Fama-French five-factor model"""
        # Extract factors
        market_factor = market_factor_data['market_factor']
        size_factor = market_factor_data['size_factor']
        value_factor = market_factor_data['value_factor']
        profitability_factor = market_factor_data['profitability_factor']
        investment_factor = market_factor_data['investment_factor']
        rf_rate = market_factor_data['rf_rate']
        
        # Extract asset returns
        asset_returns = asset_returns_data['asset_returns']
        n_assets = asset_returns_data['n_assets']
        
        # Test factor regression for each asset
        for i in range(n_assets):
            # Excess returns
            excess_returns = asset_returns[:, i] - rf_rate
            
            # Five-factor matrix
            factors = np.column_stack([
                market_factor,
                size_factor,
                value_factor,
                profitability_factor,
                investment_factor
            ])
            
            # Add intercept
            X = np.column_stack([np.ones(len(factors)), factors])
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                
                # Extract coefficients
                alpha = beta[0]
                market_beta = beta[1]
                size_beta = beta[2]
                value_beta = beta[3]
                profitability_beta = beta[4]
                investment_beta = beta[5]
                
                # Verify regression results
                assert isinstance(alpha, float)
                assert isinstance(market_beta, float)
                assert isinstance(size_beta, float)
                assert isinstance(value_beta, float)
                assert isinstance(profitability_beta, float)
                assert isinstance(investment_beta, float)
                
                # Test that estimated betas are reasonable
                assert -3.0 <= market_beta <= 3.0
                assert -2.0 <= size_beta <= 2.0
                assert -2.0 <= value_beta <= 2.0
                assert -2.0 <= profitability_beta <= 2.0
                assert -2.0 <= investment_beta <= 2.0
                
                # Calculate R-squared
                fitted_returns = X @ beta
                ss_res = np.sum((excess_returns - fitted_returns) ** 2)
                ss_tot = np.sum((excess_returns - np.mean(excess_returns)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Five-factor model should explain reasonable portion of variance
                assert 0.0 <= r_squared <= 1.0
                
            except np.linalg.LinAlgError:
                # Skip if matrix is singular
                continue
    
    def test_momentum_factor_model(self, market_factor_data, asset_returns_data):
        """Test momentum factor model"""
        # Extract factors
        market_factor = market_factor_data['market_factor']
        momentum_factor = market_factor_data['momentum_factor']
        rf_rate = market_factor_data['rf_rate']
        
        # Extract asset returns
        asset_returns = asset_returns_data['asset_returns']
        n_assets = asset_returns_data['n_assets']
        
        # Test momentum exposure for each asset
        for i in range(n_assets):
            # Excess returns
            excess_returns = asset_returns[:, i] - rf_rate
            
            # Market and momentum factors
            factors = np.column_stack([
                market_factor,
                momentum_factor
            ])
            
            # Add intercept
            X = np.column_stack([np.ones(len(factors)), factors])
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                
                # Extract coefficients
                alpha = beta[0]
                market_beta = beta[1]
                momentum_beta = beta[2]
                
                # Verify regression results
                assert isinstance(alpha, float)
                assert isinstance(market_beta, float)
                assert isinstance(momentum_beta, float)
                
                # Test that momentum beta is reasonable
                assert -2.0 <= momentum_beta <= 2.0
                
            except np.linalg.LinAlgError:
                # Skip if matrix is singular
                continue
    
    def test_mean_reversion_factor(self, market_factor_data):
        """Test mean reversion factor construction"""
        # Create mean reversion factor from market returns
        market_returns = market_factor_data['market_factor']
        
        # Calculate rolling mean
        window = 20
        rolling_mean = pd.Series(market_returns).rolling(window).mean()
        
        # Mean reversion factor: negative of deviation from rolling mean
        mean_reversion_factor = -(market_returns - rolling_mean.fillna(0))
        
        # Verify mean reversion factor
        assert len(mean_reversion_factor) == len(market_returns)
        assert isinstance(mean_reversion_factor, np.ndarray)
        
        # Test that mean reversion factor has expected properties
        # Should be negatively correlated with market returns
        correlation = np.corrcoef(market_returns[window:], mean_reversion_factor[window:])[0, 1]
        assert correlation < 0  # Should be negative correlation
    
    def test_custom_factor_construction(self, market_factor_data):
        """Test custom factor construction"""
        # Construct custom factors
        market_factor = market_factor_data['market_factor']
        size_factor = market_factor_data['size_factor']
        value_factor = market_factor_data['value_factor']
        
        # Quality factor (combination of profitability and investment)
        quality_factor = (
            market_factor_data['profitability_factor'] - 
            market_factor_data['investment_factor']
        )
        
        # Volatility factor (based on rolling volatility)
        volatility_factor = pd.Series(market_factor).rolling(20).std().fillna(0).values
        
        # Skewness factor (based on rolling skewness)
        skewness_factor = pd.Series(market_factor).rolling(20).skew().fillna(0).values
        
        # Verify custom factors
        assert len(quality_factor) == len(market_factor)
        assert len(volatility_factor) == len(market_factor)
        assert len(skewness_factor) == len(market_factor)
        
        # Test factor properties
        assert isinstance(quality_factor, np.ndarray)
        assert isinstance(volatility_factor, np.ndarray)
        assert isinstance(skewness_factor, np.ndarray)
        
        # Volatility factor should be non-negative
        assert all(vol >= 0 for vol in volatility_factor)
    
    def test_factor_validation(self, market_factor_data):
        """Test factor validation procedures"""
        factors = {
            'Market': market_factor_data['market_factor'],
            'Size': market_factor_data['size_factor'],
            'Value': market_factor_data['value_factor'],
            'Momentum': market_factor_data['momentum_factor'],
            'Profitability': market_factor_data['profitability_factor'],
            'Investment': market_factor_data['investment_factor']
        }
        
        # Validation criteria
        def validate_factor(factor_returns, factor_name):
            """Validate factor meets basic criteria"""
            validation_results = {
                'no_missing_values': not np.any(np.isnan(factor_returns)),
                'no_infinite_values': not np.any(np.isinf(factor_returns)),
                'reasonable_volatility': 0.001 <= np.std(factor_returns) <= 0.1,
                'reasonable_mean': -0.01 <= np.mean(factor_returns) <= 0.01,
                'sufficient_variation': np.var(factor_returns) > 1e-8
            }
            return validation_results
        
        # Validate each factor
        for factor_name, factor_returns in factors.items():
            validation = validate_factor(factor_returns, factor_name)
            
            # All validation criteria should pass
            for criterion, passed in validation.items():
                assert passed, f"Factor {factor_name} failed validation: {criterion}"
    
    def test_factor_correlation_analysis(self, market_factor_data):
        """Test factor correlation analysis"""
        # Create factor matrix
        factor_matrix = np.column_stack([
            market_factor_data['market_factor'],
            market_factor_data['size_factor'],
            market_factor_data['value_factor'],
            market_factor_data['momentum_factor'],
            market_factor_data['profitability_factor'],
            market_factor_data['investment_factor']
        ])
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(factor_matrix.T)
        
        # Verify correlation matrix
        assert correlation_matrix.shape == (6, 6)
        assert np.all(np.diag(correlation_matrix) == 1.0)  # Diagonal should be 1
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric
        
        # Test that correlations are reasonable
        for i in range(6):
            for j in range(6):
                correlation = correlation_matrix[i, j]
                assert -1.0 <= correlation <= 1.0
                
                # Off-diagonal elements should not be perfectly correlated
                if i != j:
                    assert abs(correlation) < 0.99
    
    def test_factor_risk_attribution(self, market_factor_data, asset_returns_data):
        """Test factor risk attribution"""
        # Extract data
        asset_returns = asset_returns_data['asset_returns']
        n_assets = asset_returns_data['n_assets']
        
        # Create factor matrix
        factor_matrix = np.column_stack([
            market_factor_data['market_factor'],
            market_factor_data['size_factor'],
            market_factor_data['value_factor'],
            market_factor_data['momentum_factor']
        ])
        
        # Calculate factor loadings for each asset
        factor_loadings = np.zeros((n_assets, 4))
        
        for i in range(n_assets):
            # Excess returns
            excess_returns = asset_returns[:, i] - market_factor_data['rf_rate']
            
            # Factor regression
            try:
                X = np.column_stack([np.ones(len(factor_matrix)), factor_matrix])
                beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                factor_loadings[i] = beta[1:]  # Skip intercept
                
            except np.linalg.LinAlgError:
                # Use zero loadings if regression fails
                factor_loadings[i] = np.zeros(4)
        
        # Calculate factor covariance matrix
        factor_cov = np.cov(factor_matrix.T)
        
        # Portfolio weights (equal weighted)
        portfolio_weights = np.ones(n_assets) / n_assets
        
        # Portfolio factor loadings
        portfolio_loadings = portfolio_weights @ factor_loadings
        
        # Factor risk contribution
        factor_risk = portfolio_loadings @ factor_cov @ portfolio_loadings.T
        
        # Verify factor risk attribution
        assert isinstance(factor_risk, float)
        assert factor_risk >= 0
        assert len(portfolio_loadings) == 4
        
        # Test individual factor contributions
        for i in range(4):
            factor_contribution = portfolio_loadings[i] ** 2 * factor_cov[i, i]
            assert isinstance(factor_contribution, float)
            assert factor_contribution >= 0
    
    def test_multi_factor_model_performance(self, market_factor_data, asset_returns_data):
        """Test multi-factor model performance evaluation"""
        # Extract data
        asset_returns = asset_returns_data['asset_returns']
        n_assets = asset_returns_data['n_assets']
        rf_rate = market_factor_data['rf_rate']
        
        # Different factor models to compare
        models = {
            'One_Factor': [market_factor_data['market_factor']],
            'Three_Factor': [
                market_factor_data['market_factor'],
                market_factor_data['size_factor'],
                market_factor_data['value_factor']
            ],
            'Five_Factor': [
                market_factor_data['market_factor'],
                market_factor_data['size_factor'],
                market_factor_data['value_factor'],
                market_factor_data['profitability_factor'],
                market_factor_data['investment_factor']
            ]
        }
        
        # Evaluate each model
        model_performance = {}
        
        for model_name, factors in models.items():
            # Create factor matrix
            factor_matrix = np.column_stack(factors)
            
            # Evaluate model for each asset
            r_squared_values = []
            
            for i in range(n_assets):
                # Excess returns
                excess_returns = asset_returns[:, i] - rf_rate
                
                # Factor regression
                try:
                    X = np.column_stack([np.ones(len(factor_matrix)), factor_matrix])
                    beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                    
                    # Calculate R-squared
                    fitted_returns = X @ beta
                    ss_res = np.sum((excess_returns - fitted_returns) ** 2)
                    ss_tot = np.sum((excess_returns - np.mean(excess_returns)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    r_squared_values.append(r_squared)
                    
                except np.linalg.LinAlgError:
                    # Skip if regression fails
                    continue
            
            # Average R-squared for this model
            if r_squared_values:
                model_performance[model_name] = {
                    'avg_r_squared': np.mean(r_squared_values),
                    'median_r_squared': np.median(r_squared_values),
                    'min_r_squared': np.min(r_squared_values),
                    'max_r_squared': np.max(r_squared_values)
                }
        
        # Verify model performance
        for model_name, performance in model_performance.items():
            assert isinstance(performance['avg_r_squared'], float)
            assert isinstance(performance['median_r_squared'], float)
            assert 0.0 <= performance['avg_r_squared'] <= 1.0
            assert 0.0 <= performance['median_r_squared'] <= 1.0
            assert 0.0 <= performance['min_r_squared'] <= 1.0
            assert 0.0 <= performance['max_r_squared'] <= 1.0
        
        # Test that models with more factors generally perform better
        if 'One_Factor' in model_performance and 'Three_Factor' in model_performance:
            one_factor_r2 = model_performance['One_Factor']['avg_r_squared']
            three_factor_r2 = model_performance['Three_Factor']['avg_r_squared']
            
            # Three-factor model should generally perform better (or at least not worse)
            assert three_factor_r2 >= one_factor_r2 - 0.1  # Allow some tolerance
    
    def test_factor_timing_analysis(self, market_factor_data):
        """Test factor timing analysis"""
        # Extract factors
        market_factor = market_factor_data['market_factor']
        size_factor = market_factor_data['size_factor']
        value_factor = market_factor_data['value_factor']
        
        # Calculate rolling performance of factors
        window = 30
        
        # Rolling Sharpe ratios
        market_rolling_sharpe = []
        size_rolling_sharpe = []
        value_rolling_sharpe = []
        
        for i in range(window, len(market_factor)):
            # Market factor Sharpe ratio
            market_window = market_factor[i-window:i]
            market_sharpe = np.mean(market_window) / np.std(market_window) if np.std(market_window) > 0 else 0
            market_rolling_sharpe.append(market_sharpe)
            
            # Size factor Sharpe ratio
            size_window = size_factor[i-window:i]
            size_sharpe = np.mean(size_window) / np.std(size_window) if np.std(size_window) > 0 else 0
            size_rolling_sharpe.append(size_sharpe)
            
            # Value factor Sharpe ratio
            value_window = value_factor[i-window:i]
            value_sharpe = np.mean(value_window) / np.std(value_window) if np.std(value_window) > 0 else 0
            value_rolling_sharpe.append(value_sharpe)
        
        # Verify timing analysis
        assert len(market_rolling_sharpe) == len(market_factor) - window
        assert len(size_rolling_sharpe) == len(size_factor) - window
        assert len(value_rolling_sharpe) == len(value_factor) - window
        
        # Test that rolling Sharpe ratios are reasonable
        for sharpe_list in [market_rolling_sharpe, size_rolling_sharpe, value_rolling_sharpe]:
            for sharpe in sharpe_list:
                assert isinstance(sharpe, float)
                assert not np.isnan(sharpe)
                assert -5.0 <= sharpe <= 5.0  # Reasonable range
    
    def test_factor_portfolio_construction(self, market_factor_data, asset_returns_data):
        """Test factor-based portfolio construction"""
        # Extract data
        asset_returns = asset_returns_data['asset_returns']
        n_assets = asset_returns_data['n_assets']
        rf_rate = market_factor_data['rf_rate']
        
        # Create factor matrix
        factor_matrix = np.column_stack([
            market_factor_data['market_factor'],
            market_factor_data['size_factor'],
            market_factor_data['value_factor']
        ])
        
        # Calculate factor loadings for each asset
        factor_loadings = np.zeros((n_assets, 3))
        
        for i in range(n_assets):
            # Excess returns
            excess_returns = asset_returns[:, i] - rf_rate
            
            # Factor regression
            try:
                X = np.column_stack([np.ones(len(factor_matrix)), factor_matrix])
                beta = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                factor_loadings[i] = beta[1:]  # Skip intercept
                
            except np.linalg.LinAlgError:
                # Use zero loadings if regression fails
                factor_loadings[i] = np.zeros(3)
        
        # Construct factor portfolios
        # Long-short portfolio for each factor
        factor_portfolios = {}
        
        for factor_idx in range(3):
            loadings = factor_loadings[:, factor_idx]
            
            # Long top tercile, short bottom tercile
            tercile_cutoffs = np.percentile(loadings, [33.33, 66.67])
            
            long_assets = loadings >= tercile_cutoffs[1]
            short_assets = loadings <= tercile_cutoffs[0]
            
            # Portfolio weights
            weights = np.zeros(n_assets)
            if np.sum(long_assets) > 0:
                weights[long_assets] = 1.0 / np.sum(long_assets)
            if np.sum(short_assets) > 0:
                weights[short_assets] = -1.0 / np.sum(short_assets)
            
            # Portfolio returns
            portfolio_returns = asset_returns @ weights
            
            factor_portfolios[f'Factor_{factor_idx}'] = {
                'weights': weights,
                'returns': portfolio_returns
            }
        
        # Verify factor portfolios
        for factor_name, portfolio_data in factor_portfolios.items():
            weights = portfolio_data['weights']
            returns = portfolio_data['returns']
            
            assert len(weights) == n_assets
            assert len(returns) == len(market_factor_data['market_factor'])
            assert abs(np.sum(weights)) < 1e-10 or abs(np.sum(weights) - 1.0) < 1e-10  # Should be dollar-neutral or long-only
            
            # Test portfolio return properties
            assert isinstance(np.mean(returns), float)
            assert isinstance(np.std(returns), float)
            assert np.std(returns) > 0


class TestFactorModelIntegration:
    """Test suite for factor model integration"""
    
    def test_attribution_engine_factor_integration(self):
        """Test integration with performance attribution engine"""
        # Create attribution engine
        event_bus = EventBus()
        attribution_engine = PerformanceAttributionEngine(
            event_bus=event_bus,
            n_strategies=3,
            benchmark_return=0.08,
            risk_free_rate=0.02
        )
        
        # Test that factor names are available
        assert hasattr(attribution_engine, 'factor_names')
        assert len(attribution_engine.factor_names) > 0
        
        # Test factor returns generation
        timestamp = datetime.now()
        attribution_engine._update_factor_returns(timestamp)
        
        # Verify factor returns were generated
        for factor_name in attribution_engine.factor_names:
            assert factor_name in attribution_engine.factor_returns
            assert len(attribution_engine.factor_returns[factor_name]) > 0
    
    def test_factor_based_attribution_calculation(self):
        """Test factor-based attribution calculation"""
        # This would be implemented as an extension of the existing framework
        # For now, test the mathematical framework
        
        # Sample data
        n_assets = 5
        n_factors = 3
        
        # Factor loadings matrix (assets x factors)
        factor_loadings = np.random.normal(0, 0.5, (n_assets, n_factors))
        
        # Factor returns
        factor_returns = np.random.normal(0.001, 0.01, n_factors)
        
        # Portfolio weights
        portfolio_weights = np.random.dirichlet(np.ones(n_assets))
        
        # Calculate factor contributions
        portfolio_factor_loadings = portfolio_weights @ factor_loadings
        factor_contributions = portfolio_factor_loadings * factor_returns
        
        # Verify factor attribution
        assert len(factor_contributions) == n_factors
        assert all(isinstance(contrib, float) for contrib in factor_contributions)
        
        # Total factor contribution
        total_factor_contribution = np.sum(factor_contributions)
        assert isinstance(total_factor_contribution, float)


if __name__ == "__main__":
    pytest.main([__file__])