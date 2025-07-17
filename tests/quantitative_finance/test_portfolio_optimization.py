"""
Portfolio Optimization Testing Suite

Comprehensive testing for portfolio optimization models including:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Model
- Risk Parity Strategies
- Minimum Variance Portfolios
- Factor Models and Risk Attribution
- Performance Attribution Analysis
- Rebalancing Strategies
- Constraint-based Optimization
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize, linprog
from scipy.linalg import inv, pinv
import warnings

from tests.quantitative_finance import (
    TOLERANCE, BENCHMARK_TOLERANCE, PERFORMANCE_BENCHMARKS,
    TestDataSets, assert_close, create_correlation_matrix
)


class MeanVarianceOptimizer:
    """Mean-Variance Optimization (Markowitz) implementation"""
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
        self.expected_returns = None
        self.covariance_matrix = None
        self.optimal_weights = None
        self.efficient_frontier = None
        self.fitted = False
    
    def fit(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> Dict[str, Any]:
        """Fit mean-variance optimizer"""
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        
        n_assets = len(expected_returns)
        
        # Solve for optimal weights: w = (1/λ) * Σ^(-1) * μ
        # where λ is risk aversion, Σ is covariance matrix, μ is expected returns
        try:
            inv_cov = inv(covariance_matrix)
            self.optimal_weights = (1 / self.risk_aversion) * inv_cov @ expected_returns
            
            # Normalize weights to sum to 1
            self.optimal_weights = self.optimal_weights / np.sum(self.optimal_weights)
            
            self.fitted = True
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(self.optimal_weights, expected_returns)
            portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'sharpe_ratio': portfolio_return / np.sqrt(portfolio_variance),
                'weights': self.optimal_weights.copy()
            }
            
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            pinv_cov = pinv(covariance_matrix)
            self.optimal_weights = (1 / self.risk_aversion) * pinv_cov @ expected_returns
            self.optimal_weights = self.optimal_weights / np.sum(self.optimal_weights)
            self.fitted = True
            
            portfolio_return = np.dot(self.optimal_weights, expected_returns)
            portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'sharpe_ratio': portfolio_return / np.sqrt(portfolio_variance),
                'weights': self.optimal_weights.copy(),
                'warning': 'Used pseudo-inverse due to singular covariance matrix'
            }
    
    def optimize_with_constraints(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 min_weights: np.ndarray = None, max_weights: np.ndarray = None) -> Dict[str, Any]:
        """Optimize with weight constraints"""
        n_assets = len(expected_returns)
        
        if min_weights is None:
            min_weights = np.zeros(n_assets)
        if max_weights is None:
            max_weights = np.ones(n_assets)
        
        # Objective function: maximize utility = μ'w - (λ/2) * w'Σw
        def objective(weights):
            return -(np.dot(weights, expected_returns) - 
                    0.5 * self.risk_aversion * np.dot(weights, covariance_matrix @ weights))
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: min_weights <= w <= max_weights
        bounds = [(min_weights[i], max_weights[i]) for i in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.optimal_weights = result.x
            self.fitted = True
            
            portfolio_return = np.dot(self.optimal_weights, expected_returns)
            portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'sharpe_ratio': portfolio_return / np.sqrt(portfolio_variance),
                'weights': self.optimal_weights.copy(),
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'error': result.message
            }
    
    def generate_efficient_frontier(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                   n_portfolios: int = 50) -> Dict[str, np.ndarray]:
        """Generate efficient frontier"""
        n_assets = len(expected_returns)
        
        # Calculate minimum variance portfolio
        ones = np.ones(n_assets)
        inv_cov = inv(covariance_matrix)
        
        # Global minimum variance portfolio
        min_var_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        min_var_return = np.dot(min_var_weights, expected_returns)
        
        # Maximum return portfolio (100% in highest return asset)
        max_return_idx = np.argmax(expected_returns)
        max_return = expected_returns[max_return_idx]
        
        # Generate target returns
        target_returns = np.linspace(min_var_return, max_return, n_portfolios)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Solve for minimum variance portfolio with target return
            def objective(weights):
                return np.dot(weights, covariance_matrix @ weights)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = [(0, 1) for _ in range(n_assets)]
            initial_weights = np.ones(n_assets) / n_assets
            
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, covariance_matrix @ weights)
                
                frontier_returns.append(portfolio_return)
                frontier_volatilities.append(np.sqrt(portfolio_variance))
                frontier_weights.append(weights)
        
        self.efficient_frontier = {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'weights': np.array(frontier_weights)
        }
        
        return self.efficient_frontier


class BlackLittermanModel:
    """Black-Litterman model implementation"""
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        self.risk_aversion = risk_aversion
        self.tau = tau  # Scales uncertainty of prior
        self.fitted = False
        self.posterior_returns = None
        self.posterior_covariance = None
    
    def fit(self, market_caps: np.ndarray, covariance_matrix: np.ndarray,
            views_matrix: np.ndarray = None, views_returns: np.ndarray = None,
            views_uncertainty: np.ndarray = None) -> Dict[str, Any]:
        """Fit Black-Litterman model"""
        
        # Step 1: Calculate implied equilibrium returns (reverse optimization)
        # π = λ * Σ * w_market
        market_weights = market_caps / np.sum(market_caps)
        implied_returns = self.risk_aversion * covariance_matrix @ market_weights
        
        # Step 2: If no views provided, use equilibrium returns
        if views_matrix is None:
            self.posterior_returns = implied_returns
            self.posterior_covariance = covariance_matrix
            self.fitted = True
            
            return {
                'implied_returns': implied_returns,
                'posterior_returns': self.posterior_returns,
                'market_weights': market_weights,
                'views_incorporated': False
            }
        
        # Step 3: Incorporate views using Black-Litterman formula
        # Posterior mean: μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
        # Posterior covariance: Σ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        
        n_assets = len(market_caps)
        
        # Prior precision matrix
        prior_precision = inv(self.tau * covariance_matrix)
        
        # Views precision matrix
        if views_uncertainty is None:
            # Default uncertainty: diagonal matrix with variances
            views_uncertainty = np.eye(len(views_returns)) * 0.01
        
        views_precision = inv(views_uncertainty)
        
        # Posterior precision matrix
        posterior_precision = prior_precision + views_matrix.T @ views_precision @ views_matrix
        
        # Posterior covariance matrix
        self.posterior_covariance = inv(posterior_precision)
        
        # Posterior mean
        self.posterior_returns = self.posterior_covariance @ (
            prior_precision @ implied_returns + 
            views_matrix.T @ views_precision @ views_returns
        )
        
        self.fitted = True
        
        return {
            'implied_returns': implied_returns,
            'posterior_returns': self.posterior_returns,
            'posterior_covariance': self.posterior_covariance,
            'market_weights': market_weights,
            'views_incorporated': True,
            'n_views': len(views_returns)
        }
    
    def optimize_portfolio(self) -> Dict[str, Any]:
        """Optimize portfolio using Black-Litterman inputs"""
        if not self.fitted:
            raise ValueError("Model must be fitted before optimization")
        
        # Use mean-variance optimization with Black-Litterman inputs
        optimizer = MeanVarianceOptimizer(risk_aversion=self.risk_aversion)
        return optimizer.fit(self.posterior_returns, self.posterior_covariance)


class RiskParityOptimizer:
    """Risk Parity portfolio optimization"""
    
    def __init__(self, method: str = "equal_risk_contribution"):
        self.method = method
        self.optimal_weights = None
        self.fitted = False
    
    def fit(self, covariance_matrix: np.ndarray, target_risk_contributions: np.ndarray = None) -> Dict[str, Any]:
        """Fit risk parity optimizer"""
        n_assets = covariance_matrix.shape[0]
        
        if target_risk_contributions is None:
            # Equal risk contribution
            target_risk_contributions = np.ones(n_assets) / n_assets
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, covariance_matrix @ weights)
            
            # Calculate marginal contributions to risk
            marginal_contributions = covariance_matrix @ weights
            
            # Calculate risk contributions
            risk_contributions = weights * marginal_contributions / portfolio_variance
            
            # Minimize squared deviations from target
            return np.sum((risk_contributions - target_risk_contributions)**2)
        
        # Constraints: weights sum to 1, weights >= 0
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.001, 0.999) for _ in range(n_assets)]  # Small bounds to avoid numerical issues
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.optimal_weights = result.x
            self.fitted = True
            
            # Calculate portfolio metrics
            portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
            marginal_contributions = covariance_matrix @ self.optimal_weights
            risk_contributions = self.optimal_weights * marginal_contributions / portfolio_variance
            
            return {
                'weights': self.optimal_weights.copy(),
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'risk_contributions': risk_contributions,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'error': result.message
            }
    
    def equal_weight_portfolio(self, n_assets: int) -> np.ndarray:
        """Create equal weight portfolio"""
        return np.ones(n_assets) / n_assets
    
    def inverse_volatility_portfolio(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Create inverse volatility weighted portfolio"""
        volatilities = np.sqrt(np.diag(covariance_matrix))
        weights = 1 / volatilities
        return weights / np.sum(weights)


class MinimumVarianceOptimizer:
    """Minimum Variance portfolio optimization"""
    
    def __init__(self):
        self.optimal_weights = None
        self.fitted = False
    
    def fit(self, covariance_matrix: np.ndarray, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fit minimum variance optimizer"""
        n_assets = covariance_matrix.shape[0]
        
        if constraints is None:
            # Analytical solution for unconstrained minimum variance
            ones = np.ones(n_assets)
            inv_cov = inv(covariance_matrix)
            self.optimal_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            self.fitted = True
            
            portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
            
            return {
                'weights': self.optimal_weights.copy(),
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'optimization_success': True
            }
        else:
            # Numerical optimization with constraints
            def objective(weights):
                return np.dot(weights, covariance_matrix @ weights)
            
            # Default constraints: weights sum to 1
            constraint_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Add custom constraints
            if 'min_weights' in constraints:
                min_weights = constraints['min_weights']
                bounds = [(min_weights[i], 1.0) for i in range(n_assets)]
            else:
                bounds = [(0, 1) for _ in range(n_assets)]
            
            if 'max_weights' in constraints:
                max_weights = constraints['max_weights']
                bounds = [(bounds[i][0], max_weights[i]) for i in range(n_assets)]
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraint_list)
            
            if result.success:
                self.optimal_weights = result.x
                self.fitted = True
                
                portfolio_variance = np.dot(self.optimal_weights, covariance_matrix @ self.optimal_weights)
                
                return {
                    'weights': self.optimal_weights.copy(),
                    'portfolio_variance': portfolio_variance,
                    'portfolio_volatility': np.sqrt(portfolio_variance),
                    'optimization_success': True
                }
            else:
                return {
                    'optimization_success': False,
                    'error': result.message
                }


class FactorModel:
    """Factor model for risk attribution"""
    
    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
        self.specific_returns = None
        self.fitted = False
    
    def fit(self, returns: np.ndarray, factor_returns: np.ndarray = None) -> Dict[str, Any]:
        """Fit factor model"""
        n_assets, n_periods = returns.shape
        
        if factor_returns is None:
            # Use PCA to extract factors
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_factors)
            self.factor_returns = pca.fit_transform(returns.T).T
            self.factor_loadings = pca.components_.T
        else:
            # Use provided factor returns
            self.factor_returns = factor_returns
            
            # Estimate factor loadings using regression
            self.factor_loadings = np.zeros((n_assets, self.n_factors))
            
            for i in range(n_assets):
                # Regression: r_i = α + β * f + ε
                X = np.column_stack([np.ones(n_periods), factor_returns.T])
                y = returns[i, :]
                
                # Solve using least squares
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
                self.factor_loadings[i, :] = betas[1:]  # Exclude intercept
        
        # Calculate specific returns (residuals)
        predicted_returns = self.factor_loadings @ self.factor_returns
        self.specific_returns = returns - predicted_returns
        
        self.fitted = True
        
        # Calculate R-squared for each asset
        r_squared = []
        for i in range(n_assets):
            ss_res = np.sum(self.specific_returns[i, :]**2)
            ss_tot = np.sum((returns[i, :] - np.mean(returns[i, :]))**2)
            r_squared.append(1 - ss_res / ss_tot)
        
        return {
            'factor_loadings': self.factor_loadings,
            'factor_returns': self.factor_returns,
            'specific_returns': self.specific_returns,
            'r_squared': r_squared,
            'n_factors': self.n_factors
        }
    
    def risk_attribution(self, portfolio_weights: np.ndarray) -> Dict[str, Any]:
        """Perform risk attribution"""
        if not self.fitted:
            raise ValueError("Model must be fitted before risk attribution")
        
        # Portfolio factor loadings
        portfolio_loadings = portfolio_weights @ self.factor_loadings
        
        # Factor covariance matrix
        factor_cov = np.cov(self.factor_returns)
        
        # Specific risk covariance matrix
        specific_cov = np.diag(np.var(self.specific_returns, axis=1))
        
        # Portfolio variance decomposition
        factor_variance = portfolio_loadings @ factor_cov @ portfolio_loadings.T
        specific_variance = portfolio_weights @ specific_cov @ portfolio_weights.T
        total_variance = factor_variance + specific_variance
        
        # Risk contributions by factor
        factor_contributions = []
        for i in range(self.n_factors):
            contribution = portfolio_loadings[i]**2 * factor_cov[i, i] / total_variance
            factor_contributions.append(contribution)
        
        return {
            'total_variance': total_variance,
            'factor_variance': factor_variance,
            'specific_variance': specific_variance,
            'factor_contributions': factor_contributions,
            'factor_share': factor_variance / total_variance,
            'specific_share': specific_variance / total_variance
        }


class TestMeanVarianceOptimization:
    """Test suite for mean-variance optimization"""
    
    @pytest.fixture
    def sample_data(self):
        """Get sample portfolio data"""
        return TestDataSets.get_portfolio_data()
    
    def test_unconstrained_optimization(self, sample_data):
        """Test unconstrained mean-variance optimization"""
        expected_returns = sample_data['expected_returns']
        covariance_matrix = sample_data['covariance_matrix']
        
        start_time = time.time()
        optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
        results = optimizer.fit(expected_returns, covariance_matrix)
        execution_time = time.time() - start_time
        
        # Check optimization results
        assert 'weights' in results, "Results should contain weights"
        assert 'portfolio_return' in results, "Results should contain portfolio return"
        assert 'portfolio_volatility' in results, "Results should contain portfolio volatility"
        
        # Check weight properties
        weights = results['weights']
        assert len(weights) == len(expected_returns), "Weights length should match assets"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
        
        # Check portfolio metrics
        assert results['portfolio_return'] > 0, "Portfolio return should be positive"
        assert results['portfolio_volatility'] > 0, "Portfolio volatility should be positive"
        assert results['sharpe_ratio'] > 0, "Sharpe ratio should be positive"
        
        # Performance check
        assert execution_time < PERFORMANCE_BENCHMARKS["portfolio_optimization"]
    
    def test_constrained_optimization(self, sample_data):
        """Test constrained mean-variance optimization"""
        expected_returns = sample_data['expected_returns']
        covariance_matrix = sample_data['covariance_matrix']
        n_assets = len(expected_returns)
        
        # Set constraints: no short selling, max 40% per asset
        min_weights = np.zeros(n_assets)
        max_weights = np.ones(n_assets) * 0.4
        
        optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
        results = optimizer.optimize_with_constraints(
            expected_returns, covariance_matrix, min_weights, max_weights
        )
        
        # Check optimization success
        assert results['optimization_success'], "Optimization should succeed"
        
        # Check constraint satisfaction
        weights = results['weights']
        assert np.all(weights >= min_weights - TOLERANCE), "Min weight constraints should be satisfied"
        assert np.all(weights <= max_weights + TOLERANCE), "Max weight constraints should be satisfied"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
    
    def test_efficient_frontier(self, sample_data):
        """Test efficient frontier generation"""
        expected_returns = sample_data['expected_returns']
        covariance_matrix = sample_data['covariance_matrix']
        
        optimizer = MeanVarianceOptimizer()
        frontier = optimizer.generate_efficient_frontier(expected_returns, covariance_matrix, n_portfolios=20)
        
        # Check frontier properties
        assert 'returns' in frontier, "Frontier should contain returns"
        assert 'volatilities' in frontier, "Frontier should contain volatilities"
        assert 'weights' in frontier, "Frontier should contain weights"
        
        returns = frontier['returns']
        volatilities = frontier['volatilities']
        weights = frontier['weights']
        
        # Check dimensions
        assert len(returns) <= 20, "Should have at most 20 portfolios"
        assert len(volatilities) == len(returns), "Volatilities should match returns"
        assert weights.shape[0] == len(returns), "Weights should match portfolios"
        
        # Check monotonicity (returns should increase along frontier)
        for i in range(len(returns) - 1):
            assert returns[i] <= returns[i + 1], "Returns should be non-decreasing along frontier"
        
        # Check weight constraints
        for i in range(len(weights)):
            assert_close(np.sum(weights[i]), 1.0, TOLERANCE), f"Portfolio {i} weights should sum to 1"
    
    def test_portfolio_metrics_calculation(self, sample_data):
        """Test portfolio metrics calculation"""
        expected_returns = sample_data['expected_returns']
        covariance_matrix = sample_data['covariance_matrix']
        
        optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
        results = optimizer.fit(expected_returns, covariance_matrix)
        
        # Verify portfolio metrics manually
        weights = results['weights']
        manual_return = np.dot(weights, expected_returns)
        manual_variance = np.dot(weights, covariance_matrix @ weights)
        manual_volatility = np.sqrt(manual_variance)
        manual_sharpe = manual_return / manual_volatility
        
        assert_close(results['portfolio_return'], manual_return, TOLERANCE)
        assert_close(results['portfolio_variance'], manual_variance, TOLERANCE)
        assert_close(results['portfolio_volatility'], manual_volatility, TOLERANCE)
        assert_close(results['sharpe_ratio'], manual_sharpe, TOLERANCE)
    
    def test_risk_aversion_sensitivity(self, sample_data):
        """Test sensitivity to risk aversion parameter"""
        expected_returns = sample_data['expected_returns']
        covariance_matrix = sample_data['covariance_matrix']
        
        risk_aversions = [0.5, 1.0, 2.0, 5.0]
        results = []
        
        for risk_aversion in risk_aversions:
            optimizer = MeanVarianceOptimizer(risk_aversion=risk_aversion)
            result = optimizer.fit(expected_returns, covariance_matrix)
            results.append(result)
        
        # Higher risk aversion should lead to lower portfolio risk
        for i in range(len(results) - 1):
            assert results[i]['portfolio_volatility'] >= results[i + 1]['portfolio_volatility'], \
                "Portfolio volatility should decrease with higher risk aversion"


class TestBlackLittermanModel:
    """Test suite for Black-Litterman model"""
    
    @pytest.fixture
    def market_data(self):
        """Get market data for Black-Litterman"""
        portfolio_data = TestDataSets.get_portfolio_data()
        
        # Create market capitalizations
        market_caps = np.array([100, 80, 60, 40, 20])  # Billion USD
        
        return {
            'market_caps': market_caps,
            'covariance_matrix': portfolio_data['covariance_matrix'],
            'expected_returns': portfolio_data['expected_returns']
        }
    
    def test_equilibrium_returns(self, market_data):
        """Test equilibrium returns calculation"""
        bl_model = BlackLittermanModel(risk_aversion=3.0)
        
        results = bl_model.fit(
            market_caps=market_data['market_caps'],
            covariance_matrix=market_data['covariance_matrix']
        )
        
        # Check results
        assert 'implied_returns' in results, "Should calculate implied returns"
        assert 'market_weights' in results, "Should calculate market weights"
        assert not results['views_incorporated'], "No views should be incorporated"
        
        # Check market weights
        market_weights = results['market_weights']
        assert_close(np.sum(market_weights), 1.0, TOLERANCE), "Market weights should sum to 1"
        
        # Check implied returns
        implied_returns = results['implied_returns']
        assert len(implied_returns) == len(market_data['market_caps']), "Implied returns should match assets"
        assert np.all(implied_returns > 0), "Implied returns should be positive"
    
    def test_views_incorporation(self, market_data):
        """Test views incorporation in Black-Litterman"""
        bl_model = BlackLittermanModel(risk_aversion=3.0, tau=0.025)
        
        # Create views: Asset 0 will outperform Asset 1 by 2%
        views_matrix = np.array([[1, -1, 0, 0, 0]])  # P matrix
        views_returns = np.array([0.02])  # Q vector
        views_uncertainty = np.array([[0.001]])  # Omega matrix
        
        results = bl_model.fit(
            market_caps=market_data['market_caps'],
            covariance_matrix=market_data['covariance_matrix'],
            views_matrix=views_matrix,
            views_returns=views_returns,
            views_uncertainty=views_uncertainty
        )
        
        # Check results
        assert results['views_incorporated'], "Views should be incorporated"
        assert results['n_views'] == 1, "Should have 1 view"
        assert 'posterior_returns' in results, "Should have posterior returns"
        assert 'posterior_covariance' in results, "Should have posterior covariance"
        
        # Check that views affected the returns
        implied_returns = results['implied_returns']
        posterior_returns = results['posterior_returns']
        
        # Asset 0 should have higher expected return than implied
        assert posterior_returns[0] > implied_returns[0], "Asset 0 return should increase due to positive view"
        
        # Asset 1 should have lower expected return than implied
        assert posterior_returns[1] < implied_returns[1], "Asset 1 return should decrease due to negative view"
    
    def test_portfolio_optimization(self, market_data):
        """Test portfolio optimization with Black-Litterman"""
        bl_model = BlackLittermanModel(risk_aversion=3.0)
        
        # First fit the model
        bl_model.fit(
            market_caps=market_data['market_caps'],
            covariance_matrix=market_data['covariance_matrix']
        )
        
        # Then optimize portfolio
        portfolio_results = bl_model.optimize_portfolio()
        
        # Check optimization results
        assert 'weights' in portfolio_results, "Should have optimal weights"
        assert 'portfolio_return' in portfolio_results, "Should have portfolio return"
        assert 'portfolio_volatility' in portfolio_results, "Should have portfolio volatility"
        
        # Check weight properties
        weights = portfolio_results['weights']
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
        assert portfolio_results['portfolio_return'] > 0, "Portfolio return should be positive"
        assert portfolio_results['portfolio_volatility'] > 0, "Portfolio volatility should be positive"
    
    def test_tau_parameter_sensitivity(self, market_data):
        """Test sensitivity to tau parameter"""
        base_results = []
        tau_values = [0.01, 0.025, 0.05, 0.1]
        
        # Views: Asset 0 will outperform by 3%
        views_matrix = np.array([[1, 0, 0, 0, 0]])
        views_returns = np.array([0.03])
        
        for tau in tau_values:
            bl_model = BlackLittermanModel(risk_aversion=3.0, tau=tau)
            results = bl_model.fit(
                market_caps=market_data['market_caps'],
                covariance_matrix=market_data['covariance_matrix'],
                views_matrix=views_matrix,
                views_returns=views_returns
            )
            base_results.append(results)
        
        # Higher tau should give more weight to views
        for i in range(len(base_results) - 1):
            current_return = base_results[i]['posterior_returns'][0]
            next_return = base_results[i + 1]['posterior_returns'][0]
            
            # Higher tau should move posterior return closer to view
            assert next_return >= current_return, "Higher tau should give more weight to views"


class TestRiskParityOptimization:
    """Test suite for risk parity optimization"""
    
    @pytest.fixture
    def sample_covariance(self):
        """Get sample covariance matrix"""
        return TestDataSets.get_portfolio_data()['covariance_matrix']
    
    def test_equal_risk_contribution(self, sample_covariance):
        """Test equal risk contribution portfolio"""
        optimizer = RiskParityOptimizer(method="equal_risk_contribution")
        results = optimizer.fit(sample_covariance)
        
        # Check optimization success
        assert results['optimization_success'], "Optimization should succeed"
        
        # Check weights
        weights = results['weights']
        assert len(weights) == sample_covariance.shape[0], "Weights should match number of assets"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
        assert np.all(weights > 0), "All weights should be positive"
        
        # Check risk contributions
        risk_contributions = results['risk_contributions']
        target_contribution = 1.0 / len(weights)
        
        for rc in risk_contributions:
            assert_close(rc, target_contribution, 0.05), "Risk contributions should be approximately equal"
    
    def test_custom_risk_targets(self, sample_covariance):
        """Test custom risk contribution targets"""
        n_assets = sample_covariance.shape[0]
        
        # Custom targets: first asset gets 50%, others split remaining 50%
        target_contributions = np.array([0.5, 0.125, 0.125, 0.125, 0.125])
        
        optimizer = RiskParityOptimizer()
        results = optimizer.fit(sample_covariance, target_contributions)
        
        # Check optimization success
        assert results['optimization_success'], "Optimization should succeed"
        
        # Check risk contributions are close to targets
        risk_contributions = results['risk_contributions']
        for i, (actual, target) in enumerate(zip(risk_contributions, target_contributions)):
            assert_close(actual, target, 0.1), f"Risk contribution {i} should be close to target"
    
    def test_inverse_volatility_portfolio(self, sample_covariance):
        """Test inverse volatility weighted portfolio"""
        optimizer = RiskParityOptimizer()
        weights = optimizer.inverse_volatility_portfolio(sample_covariance)
        
        # Check weights properties
        assert len(weights) == sample_covariance.shape[0], "Weights should match number of assets"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
        assert np.all(weights > 0), "All weights should be positive"
        
        # Check inverse relationship with volatility
        volatilities = np.sqrt(np.diag(sample_covariance))
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if volatilities[i] > volatilities[j]:
                    assert weights[i] < weights[j], "Lower volatility should have higher weight"
    
    def test_equal_weight_portfolio(self, sample_covariance):
        """Test equal weight portfolio"""
        n_assets = sample_covariance.shape[0]
        
        optimizer = RiskParityOptimizer()
        weights = optimizer.equal_weight_portfolio(n_assets)
        
        # Check equal weights
        expected_weight = 1.0 / n_assets
        for weight in weights:
            assert_close(weight, expected_weight, TOLERANCE), "All weights should be equal"
    
    def test_risk_parity_vs_equal_weights(self, sample_covariance):
        """Test risk parity vs equal weights"""
        n_assets = sample_covariance.shape[0]
        
        # Risk parity portfolio
        rp_optimizer = RiskParityOptimizer()
        rp_results = rp_optimizer.fit(sample_covariance)
        rp_weights = rp_results['weights']
        
        # Equal weight portfolio
        eq_weights = np.ones(n_assets) / n_assets
        
        # Calculate risk contributions for both
        def calculate_risk_contributions(weights, cov_matrix):
            portfolio_var = np.dot(weights, cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            return weights * marginal_contrib / portfolio_var
        
        rp_risk_contrib = calculate_risk_contributions(rp_weights, sample_covariance)
        eq_risk_contrib = calculate_risk_contributions(eq_weights, sample_covariance)
        
        # Risk parity should have more equal risk contributions
        rp_risk_std = np.std(rp_risk_contrib)
        eq_risk_std = np.std(eq_risk_contrib)
        
        assert rp_risk_std <= eq_risk_std, "Risk parity should have more equal risk contributions"


class TestMinimumVarianceOptimization:
    """Test suite for minimum variance optimization"""
    
    @pytest.fixture
    def sample_covariance(self):
        """Get sample covariance matrix"""
        return TestDataSets.get_portfolio_data()['covariance_matrix']
    
    def test_unconstrained_minimum_variance(self, sample_covariance):
        """Test unconstrained minimum variance optimization"""
        optimizer = MinimumVarianceOptimizer()
        results = optimizer.fit(sample_covariance)
        
        # Check optimization success
        assert results['optimization_success'], "Optimization should succeed"
        
        # Check weights
        weights = results['weights']
        assert len(weights) == sample_covariance.shape[0], "Weights should match number of assets"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
        
        # Check portfolio metrics
        assert results['portfolio_variance'] > 0, "Portfolio variance should be positive"
        assert results['portfolio_volatility'] > 0, "Portfolio volatility should be positive"
        
        # Verify minimum variance property
        # Generate random portfolios and check they have higher variance
        n_assets = len(weights)
        min_variance = results['portfolio_variance']
        
        for _ in range(10):
            random_weights = np.random.random(n_assets)
            random_weights = random_weights / np.sum(random_weights)
            random_variance = np.dot(random_weights, sample_covariance @ random_weights)
            
            assert random_variance >= min_variance - TOLERANCE, "Random portfolio should have higher variance"
    
    def test_constrained_minimum_variance(self, sample_covariance):
        """Test constrained minimum variance optimization"""
        n_assets = sample_covariance.shape[0]
        
        # Constraints: no short selling, max 50% per asset
        constraints = {
            'min_weights': np.zeros(n_assets),
            'max_weights': np.ones(n_assets) * 0.5
        }
        
        optimizer = MinimumVarianceOptimizer()
        results = optimizer.fit(sample_covariance, constraints)
        
        # Check optimization success
        assert results['optimization_success'], "Optimization should succeed"
        
        # Check constraint satisfaction
        weights = results['weights']
        assert np.all(weights >= constraints['min_weights'] - TOLERANCE), "Min constraints should be satisfied"
        assert np.all(weights <= constraints['max_weights'] + TOLERANCE), "Max constraints should be satisfied"
        assert_close(np.sum(weights), 1.0, TOLERANCE), "Weights should sum to 1"
    
    def test_minimum_variance_properties(self, sample_covariance):
        """Test minimum variance portfolio properties"""
        optimizer = MinimumVarianceOptimizer()
        results = optimizer.fit(sample_covariance)
        
        weights = results['weights']
        
        # Check that weights are inversely related to variance
        # (assets with lower variance should have higher weights, all else equal)
        asset_variances = np.diag(sample_covariance)
        
        # For demonstration, check that the lowest variance asset has positive weight
        min_var_asset = np.argmin(asset_variances)
        assert weights[min_var_asset] > 0, "Lowest variance asset should have positive weight"
    
    def test_comparison_with_equal_weights(self, sample_covariance):
        """Test minimum variance vs equal weights"""
        n_assets = sample_covariance.shape[0]
        
        # Minimum variance portfolio
        optimizer = MinimumVarianceOptimizer()
        mv_results = optimizer.fit(sample_covariance)
        mv_variance = mv_results['portfolio_variance']
        
        # Equal weight portfolio
        eq_weights = np.ones(n_assets) / n_assets
        eq_variance = np.dot(eq_weights, sample_covariance @ eq_weights)
        
        # Minimum variance should have lower variance
        assert mv_variance <= eq_variance + TOLERANCE, "Minimum variance portfolio should have lower variance"


class TestFactorModel:
    """Test suite for factor models"""
    
    @pytest.fixture
    def sample_returns(self):
        """Get sample returns data"""
        portfolio_data = TestDataSets.get_portfolio_data()
        return portfolio_data['returns'].T  # Transpose to get assets x time
    
    def test_factor_model_fitting(self, sample_returns):
        """Test factor model fitting"""
        factor_model = FactorModel(n_factors=3)
        results = factor_model.fit(sample_returns)
        
        # Check results
        assert 'factor_loadings' in results, "Should have factor loadings"
        assert 'factor_returns' in results, "Should have factor returns"
        assert 'specific_returns' in results, "Should have specific returns"
        assert 'r_squared' in results, "Should have R-squared values"
        
        # Check dimensions
        n_assets, n_periods = sample_returns.shape
        
        factor_loadings = results['factor_loadings']
        factor_returns = results['factor_returns']
        specific_returns = results['specific_returns']
        r_squared = results['r_squared']
        
        assert factor_loadings.shape == (n_assets, 3), "Factor loadings should have correct dimensions"
        assert factor_returns.shape == (3, n_periods), "Factor returns should have correct dimensions"
        assert specific_returns.shape == (n_assets, n_periods), "Specific returns should have correct dimensions"
        assert len(r_squared) == n_assets, "R-squared should have one value per asset"
        
        # Check R-squared values
        for r2 in r_squared:
            assert 0 <= r2 <= 1, "R-squared should be between 0 and 1"
    
    def test_risk_attribution(self, sample_returns):
        """Test risk attribution using factor model"""
        factor_model = FactorModel(n_factors=2)
        factor_model.fit(sample_returns)
        
        # Create sample portfolio weights
        n_assets = sample_returns.shape[0]
        portfolio_weights = np.ones(n_assets) / n_assets
        
        attribution = factor_model.risk_attribution(portfolio_weights)
        
        # Check attribution results
        assert 'total_variance' in attribution, "Should have total variance"
        assert 'factor_variance' in attribution, "Should have factor variance"
        assert 'specific_variance' in attribution, "Should have specific variance"
        assert 'factor_contributions' in attribution, "Should have factor contributions"
        
        # Check variance decomposition
        total_var = attribution['total_variance']
        factor_var = attribution['factor_variance']
        specific_var = attribution['specific_variance']
        
        assert_close(total_var, factor_var + specific_var, TOLERANCE), "Variance should decompose correctly"
        assert total_var > 0, "Total variance should be positive"
        assert factor_var >= 0, "Factor variance should be non-negative"
        assert specific_var >= 0, "Specific variance should be non-negative"
        
        # Check shares sum to 1
        factor_share = attribution['factor_share']
        specific_share = attribution['specific_share']
        assert_close(factor_share + specific_share, 1.0, TOLERANCE), "Shares should sum to 1"
    
    def test_factor_model_with_external_factors(self, sample_returns):
        """Test factor model with external factor returns"""
        n_assets, n_periods = sample_returns.shape
        
        # Create synthetic factor returns
        external_factors = np.random.normal(0, 0.02, (2, n_periods))
        
        factor_model = FactorModel(n_factors=2)
        results = factor_model.fit(sample_returns, external_factors)
        
        # Check that external factors were used
        assert np.array_equal(results['factor_returns'], external_factors), "Should use external factors"
        
        # Check factor loadings were estimated
        factor_loadings = results['factor_loadings']
        assert factor_loadings.shape == (n_assets, 2), "Factor loadings should have correct dimensions"
    
    def test_factor_model_explained_variance(self, sample_returns):
        """Test factor model explained variance"""
        # Test with different numbers of factors
        n_factors_list = [1, 2, 3, 4]
        explained_variances = []
        
        for n_factors in n_factors_list:
            factor_model = FactorModel(n_factors=n_factors)
            results = factor_model.fit(sample_returns)
            
            # Calculate average R-squared
            avg_r_squared = np.mean(results['r_squared'])
            explained_variances.append(avg_r_squared)
        
        # More factors should explain more variance
        for i in range(len(explained_variances) - 1):
            assert explained_variances[i] <= explained_variances[i + 1], \
                "More factors should explain more variance"
    
    def test_factor_orthogonality(self, sample_returns):
        """Test factor orthogonality (for PCA-based factors)"""
        factor_model = FactorModel(n_factors=3)
        results = factor_model.fit(sample_returns)
        
        factor_returns = results['factor_returns']
        
        # Calculate correlation matrix of factors
        factor_corr = np.corrcoef(factor_returns)
        
        # Off-diagonal elements should be close to zero
        for i in range(factor_corr.shape[0]):
            for j in range(i + 1, factor_corr.shape[1]):
                assert abs(factor_corr[i, j]) < 0.1, "Factors should be approximately orthogonal"


class TestPortfolioOptimizationPerformance:
    """Test performance of portfolio optimization methods"""
    
    def test_mean_variance_performance(self):
        """Test mean-variance optimization performance"""
        # Large portfolio
        n_assets = 50
        expected_returns = np.random.uniform(0.05, 0.15, n_assets)
        correlation_matrix = create_correlation_matrix(n_assets, 0.3)
        volatilities = np.random.uniform(0.1, 0.3, n_assets)
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        start_time = time.time()
        optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
        results = optimizer.fit(expected_returns, covariance_matrix)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < PERFORMANCE_BENCHMARKS["portfolio_optimization"]
        assert 'weights' in results, "Should return optimal weights"
    
    def test_efficient_frontier_performance(self):
        """Test efficient frontier generation performance"""
        portfolio_data = TestDataSets.get_portfolio_data()
        
        start_time = time.time()
        optimizer = MeanVarianceOptimizer()
        frontier = optimizer.generate_efficient_frontier(
            portfolio_data['expected_returns'],
            portfolio_data['covariance_matrix'],
            n_portfolios=25
        )
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0, "Efficient frontier generation should be fast"
        assert len(frontier['returns']) > 0, "Should generate frontier points"
    
    def test_black_litterman_performance(self):
        """Test Black-Litterman performance"""
        portfolio_data = TestDataSets.get_portfolio_data()
        market_caps = np.array([100, 80, 60, 40, 20])
        
        start_time = time.time()
        bl_model = BlackLittermanModel()
        results = bl_model.fit(market_caps, portfolio_data['covariance_matrix'])
        execution_time = time.time() - start_time
        
        # Should be fast
        assert execution_time < 1.0, "Black-Litterman should be fast"
        assert 'implied_returns' in results, "Should calculate implied returns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])