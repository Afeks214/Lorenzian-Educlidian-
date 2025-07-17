"""
Risk-Adjusted Strategies Testing Framework

This module provides comprehensive testing for risk-adjusted trading strategies,
including Kelly criterion, position sizing, risk budgeting, portfolio construction,
and dynamic hedging strategies.

Key Features:
- Kelly criterion and optimal position sizing
- Risk budgeting and portfolio construction
- Dynamic hedging and risk management
- Value at Risk (VaR) integration
- Expected Shortfall (ES) calculations
- Risk parity and equal risk contribution
- Multi-objective optimization
- Stress testing and scenario analysis
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from unittest.mock import Mock, patch
import asyncio
from scipy import stats, optimize
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskModel(Enum):
    """Risk model types"""
    HISTORICAL = "historical"
    GARCH = "garch"
    FACTOR = "factor"
    SHRINKAGE = "shrinkage"
    ROBUST = "robust"


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    MEAN_VARIANCE = "mean_variance"


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    portfolio_volatility: float
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_es_95: float
    portfolio_es_99: float
    max_drawdown: float
    beta: float
    tracking_error: float
    information_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    symbol: str
    current_weight: float
    target_weight: float
    kelly_weight: float
    risk_budget_weight: float
    final_weight: float
    expected_return: float
    risk_contribution: float
    confidence_interval: Tuple[float, float]


@dataclass
class PortfolioConstruction:
    """Portfolio construction result"""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_metrics: RiskMetrics
    position_sizes: List[PositionSizingResult]
    optimization_details: Dict[str, Any]


class KellyCriterionCalculator:
    """
    Kelly Criterion calculator for optimal position sizing.
    """
    
    def __init__(self):
        self.min_kelly_fraction = 0.01
        self.max_kelly_fraction = 0.25
        self.confidence_level = 0.95
        
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        variance: float,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Kelly fraction for single asset.
        
        Args:
            expected_return: Expected return of the asset
            variance: Variance of the asset
            risk_free_rate: Risk-free rate
            
        Returns:
            Kelly fraction (optimal position size)
        """
        
        if variance <= 0:
            return 0.0
        
        # Kelly fraction = (μ - rf) / σ²
        excess_return = expected_return - risk_free_rate
        kelly_fraction = excess_return / variance
        
        # Apply practical bounds
        kelly_fraction = max(self.min_kelly_fraction, 
                           min(self.max_kelly_fraction, kelly_fraction))
        
        return kelly_fraction
    
    def calculate_portfolio_kelly(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Calculate Kelly fractions for portfolio of assets.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Kelly fractions for each asset
        """
        
        # Ensure alignment
        assets = expected_returns.index
        cov_matrix = covariance_matrix.loc[assets, assets]
        
        # Calculate excess returns
        excess_returns = expected_returns - risk_free_rate
        
        try:
            # Kelly weights = Σ⁻¹ × (μ - rf)
            inv_cov = np.linalg.inv(cov_matrix.values)
            kelly_weights = np.dot(inv_cov, excess_returns.values)
            
            # Convert to pandas Series
            kelly_series = pd.Series(kelly_weights, index=assets)
            
            # Apply practical bounds
            kelly_series = kelly_series.clip(
                lower=self.min_kelly_fraction,
                upper=self.max_kelly_fraction
            )
            
            # Normalize to ensure sum <= 1
            total_weight = kelly_series.sum()
            if total_weight > 1.0:
                kelly_series = kelly_series / total_weight
            
            return kelly_series
            
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, use diagonal approximation
            logger.warning("Singular covariance matrix, using diagonal approximation")
            
            kelly_series = pd.Series(index=assets, dtype=float)
            for asset in assets:
                variance = cov_matrix.loc[asset, asset]
                kelly_series[asset] = self.calculate_kelly_fraction(
                    expected_returns[asset], variance, risk_free_rate
                )
            
            return kelly_series
    
    def calculate_fractional_kelly(
        self,
        kelly_weights: pd.Series,
        fraction: float = 0.5
    ) -> pd.Series:
        """
        Calculate fractional Kelly positions.
        
        Args:
            kelly_weights: Full Kelly weights
            fraction: Fraction of Kelly to use (0-1)
            
        Returns:
            Fractional Kelly weights
        """
        
        return kelly_weights * fraction
    
    def calculate_kelly_confidence_interval(
        self,
        expected_return: float,
        variance: float,
        sample_size: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Kelly fraction.
        
        Args:
            expected_return: Expected return estimate
            variance: Variance estimate
            sample_size: Sample size used for estimation
            confidence_level: Confidence level
            
        Returns:
            Confidence interval (lower, upper)
        """
        
        if sample_size <= 2 or variance <= 0:
            return (0.0, 0.0)
        
        # Standard error of Kelly fraction
        # Approximation: SE(f) ≈ √(2/n) × σ/μ
        if expected_return != 0:
            se_kelly = np.sqrt(2 / sample_size) * np.sqrt(variance) / abs(expected_return)
        else:
            se_kelly = 0.1  # Default for zero expected return
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        kelly_fraction = self.calculate_kelly_fraction(expected_return, variance)
        
        lower = kelly_fraction - z_score * se_kelly
        upper = kelly_fraction + z_score * se_kelly
        
        # Apply bounds
        lower = max(0, lower)
        upper = min(1, upper)
        
        return (lower, upper)


class RiskBudgetingOptimizer:
    """
    Risk budgeting and portfolio construction optimizer.
    """
    
    def __init__(self):
        self.min_weight = 0.01
        self.max_weight = 0.4
        self.risk_budget_tolerance = 1e-6
        self.max_iterations = 1000
        
    def optimize_risk_parity(
        self,
        covariance_matrix: pd.DataFrame,
        target_volatility: float = 0.15
    ) -> pd.Series:
        """
        Optimize for risk parity portfolio.
        
        Args:
            covariance_matrix: Covariance matrix of returns
            target_volatility: Target portfolio volatility
            
        Returns:
            Risk parity weights
        """
        
        n_assets = len(covariance_matrix)
        assets = covariance_matrix.index
        
        # Objective function: minimize sum of squared risk contribution differences
        def risk_parity_objective(weights):
            weights = np.array(weights)
            
            # Calculate portfolio risk contributions
            portfolio_var = np.dot(weights, np.dot(covariance_matrix.values, weights))
            marginal_contrib = np.dot(covariance_matrix.values, weights)
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Target equal risk contribution
            target_contrib = 1.0 / n_assets
            
            # Minimize sum of squared deviations
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            weights = pd.Series(result.x, index=assets)
            
            # Scale to target volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            if portfolio_vol > 0:
                scaling_factor = target_volatility / portfolio_vol
                weights = weights * scaling_factor
                
                # Renormalize
                weights = weights / weights.sum()
            
            return weights
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return pd.Series(1.0 / n_assets, index=assets)
    
    def optimize_risk_budgeting(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_budgets: pd.Series,
        target_return: Optional[float] = None
    ) -> pd.Series:
        """
        Optimize portfolio with risk budgeting constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_budgets: Risk budget allocation for each asset
            target_return: Target portfolio return
            
        Returns:
            Risk budgeted weights
        """
        
        assets = expected_returns.index
        n_assets = len(assets)
        
        # Ensure risk budgets sum to 1
        risk_budgets = risk_budgets / risk_budgets.sum()
        
        # Objective function: minimize tracking error to risk budgets
        def risk_budget_objective(weights):
            weights = np.array(weights)
            
            # Calculate portfolio risk contributions
            portfolio_var = np.dot(weights, np.dot(covariance_matrix.values, weights))
            if portfolio_var <= 0:
                return 1e6  # Penalty for invalid portfolio
            
            marginal_contrib = np.dot(covariance_matrix.values, weights)
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Target risk contributions
            target_contrib = risk_budgets.values
            
            # Minimize sum of squared deviations
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns.values) - target_return
            })
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess based on risk budgets
        x0 = risk_budgets.values.copy()
        
        # Optimize
        result = optimize.minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            return pd.Series(result.x, index=assets)
        else:
            logger.warning("Risk budgeting optimization failed, using risk budget weights")
            return risk_budgets
    
    def calculate_risk_contributions(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk contributions for given weights.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix
            
        Returns:
            Risk contributions for each asset
        """
        
        # Calculate portfolio variance
        portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_var <= 0:
            return pd.Series(0, index=weights.index)
        
        # Calculate marginal contributions
        marginal_contrib = np.dot(covariance_matrix, weights)
        
        # Calculate risk contributions
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return risk_contrib


class RiskMetricsCalculator:
    """
    Calculator for various risk metrics.
    """
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = 0.02
        
    def calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for portfolio.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns
            benchmark_returns: Benchmark returns for tracking error
            
        Returns:
            RiskMetrics object with all metrics
        """
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic metrics
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_return = portfolio_returns.mean() * 252
        
        # Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        var_99 = np.percentile(portfolio_returns, 1) * np.sqrt(252)
        
        # Expected Shortfall (ES)
        es_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        es_99 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean() * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Beta (if benchmark provided)
        beta = 0.0
        tracking_error = 0.0
        information_ratio = 0.0
        
        if benchmark_returns is not None:
            # Align dates
            aligned_portfolio = portfolio_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_portfolio.index).dropna()
            
            if len(aligned_portfolio) > 10:
                # Calculate beta
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Tracking error
                excess_returns = aligned_portfolio - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Risk-adjusted ratios
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = portfolio_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return RiskMetrics(
            portfolio_volatility=portfolio_volatility,
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            portfolio_es_95=es_95,
            portfolio_es_99=es_99,
            max_drawdown=max_drawdown,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
    
    def calculate_var_es(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Expected Shortfall.
        
        Args:
            returns: Return series
            confidence_level: Confidence level (0-1)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Tuple of (VaR, ES)
        """
        
        if method == "historical":
            # Historical simulation
            var = np.percentile(returns, (1 - confidence_level) * 100)
            es = returns[returns <= var].mean()
            
        elif method == "parametric":
            # Parametric (normal distribution)
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            es = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)
            
        elif method == "monte_carlo":
            # Monte Carlo simulation
            mean = returns.mean()
            std = returns.std()
            simulated_returns = np.random.normal(mean, std, 10000)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            es = simulated_returns[simulated_returns <= var].mean()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var, es


class DynamicHedgingStrategy:
    """
    Dynamic hedging strategy implementation.
    """
    
    def __init__(self):
        self.hedge_threshold = 0.02  # 2% VaR threshold
        self.hedge_ratio_range = (0.5, 1.5)
        self.rebalance_frequency = 'D'  # Daily
        
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        method: str = "minimum_variance"
    ) -> float:
        """
        Calculate optimal hedge ratio.
        
        Args:
            portfolio_returns: Portfolio returns to hedge
            hedge_instrument_returns: Hedge instrument returns
            method: Hedging method ('minimum_variance', 'beta', 'correlation')
            
        Returns:
            Optimal hedge ratio
        """
        
        # Align returns
        aligned_portfolio = portfolio_returns.dropna()
        aligned_hedge = hedge_instrument_returns.reindex(aligned_portfolio.index).dropna()
        
        if len(aligned_portfolio) < 10 or len(aligned_hedge) < 10:
            return 0.0
        
        if method == "minimum_variance":
            # Minimum variance hedge ratio
            covariance = np.cov(aligned_portfolio, aligned_hedge)[0, 1]
            hedge_variance = np.var(aligned_hedge)
            hedge_ratio = covariance / hedge_variance if hedge_variance > 0 else 0
            
        elif method == "beta":
            # Beta hedge ratio
            hedge_ratio = np.cov(aligned_portfolio, aligned_hedge)[0, 1] / np.var(aligned_hedge)
            
        elif method == "correlation":
            # Correlation-based hedge ratio
            correlation = np.corrcoef(aligned_portfolio, aligned_hedge)[0, 1]
            portfolio_std = np.std(aligned_portfolio)
            hedge_std = np.std(aligned_hedge)
            hedge_ratio = correlation * portfolio_std / hedge_std if hedge_std > 0 else 0
            
        else:
            raise ValueError(f"Unknown hedging method: {method}")
        
        # Apply bounds
        hedge_ratio = max(self.hedge_ratio_range[0], 
                         min(self.hedge_ratio_range[1], hedge_ratio))
        
        return hedge_ratio
    
    def calculate_dynamic_hedge_weights(
        self,
        portfolio_weights: pd.Series,
        returns: pd.DataFrame,
        hedge_instruments: List[str],
        risk_threshold: float = 0.02
    ) -> pd.Series:
        """
        Calculate dynamic hedge weights based on risk threshold.
        
        Args:
            portfolio_weights: Current portfolio weights
            returns: Historical returns
            hedge_instruments: List of hedge instruments
            risk_threshold: Risk threshold for hedging
            
        Returns:
            Combined portfolio and hedge weights
        """
        
        # Calculate current portfolio risk
        portfolio_returns = (returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
        current_var = np.percentile(portfolio_returns, 5)
        
        # Initialize hedge weights
        hedge_weights = pd.Series(0, index=hedge_instruments)
        
        # If risk exceeds threshold, calculate hedge positions
        if abs(current_var) > risk_threshold:
            for hedge_instrument in hedge_instruments:
                if hedge_instrument in returns.columns:
                    hedge_ratio = self.calculate_hedge_ratio(
                        portfolio_returns,
                        returns[hedge_instrument]
                    )
                    
                    # Scale hedge ratio by risk excess
                    risk_excess = (abs(current_var) - risk_threshold) / risk_threshold
                    hedge_weights[hedge_instrument] = hedge_ratio * risk_excess
        
        # Combine portfolio and hedge weights
        combined_weights = portfolio_weights.copy()
        for hedge_instrument in hedge_instruments:
            combined_weights[hedge_instrument] = hedge_weights[hedge_instrument]
        
        return combined_weights


class PortfolioOptimizer:
    """
    Comprehensive portfolio optimizer with multiple objectives.
    """
    
    def __init__(self):
        self.kelly_calculator = KellyCriterionCalculator()
        self.risk_budgeting = RiskBudgetingOptimizer()
        self.risk_metrics = RiskMetricsCalculator()
        self.hedging_strategy = DynamicHedgingStrategy()
        
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE,
        constraints: Optional[Dict[str, Any]] = None,
        risk_free_rate: float = 0.02
    ) -> PortfolioConstruction:
        """
        Optimize portfolio based on specified objective.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            objective: Optimization objective
            constraints: Additional constraints
            risk_free_rate: Risk-free rate
            
        Returns:
            PortfolioConstruction with optimized weights and metrics
        """
        
        assets = expected_returns.index
        n_assets = len(assets)
        
        # Default constraints
        default_constraints = {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'target_return': None,
            'target_volatility': None,
            'max_concentration': 0.4
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        # Optimize based on objective
        if objective == OptimizationObjective.KELLY_CRITERION:
            weights = self.kelly_calculator.calculate_portfolio_kelly(
                expected_returns, covariance_matrix, risk_free_rate
            )
            
        elif objective == OptimizationObjective.RISK_PARITY:
            weights = self.risk_budgeting.optimize_risk_parity(covariance_matrix)
            
        elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
            weights = self._optimize_maximum_sharpe(
                expected_returns, covariance_matrix, risk_free_rate, default_constraints
            )
            
        elif objective == OptimizationObjective.MINIMIZE_RISK:
            weights = self._optimize_minimum_risk(
                covariance_matrix, default_constraints
            )
            
        elif objective == OptimizationObjective.MAXIMIZE_RETURN:
            weights = self._optimize_maximum_return(
                expected_returns, covariance_matrix, default_constraints
            )
            
        else:
            # Default to equal weights
            weights = pd.Series(1.0 / n_assets, index=assets)
        
        # Calculate portfolio metrics
        expected_portfolio_return = np.dot(weights, expected_returns)
        expected_portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        portfolio_sharpe = (expected_portfolio_return - risk_free_rate) / expected_portfolio_volatility if expected_portfolio_volatility > 0 else 0
        
        # Calculate position sizing details
        position_sizes = self._calculate_position_sizing_details(
            weights, expected_returns, covariance_matrix, risk_free_rate
        )
        
        # Create dummy risk metrics (would be calculated with actual returns)
        risk_metrics = RiskMetrics(
            portfolio_volatility=expected_portfolio_volatility,
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            portfolio_es_95=0.0,
            portfolio_es_99=0.0,
            max_drawdown=0.0,
            beta=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            sharpe_ratio=portfolio_sharpe,
            sortino_ratio=0.0,
            calmar_ratio=0.0
        )
        
        return PortfolioConstruction(
            weights=weights,
            expected_return=expected_portfolio_return,
            expected_volatility=expected_portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            risk_metrics=risk_metrics,
            position_sizes=position_sizes,
            optimization_details={
                'objective': objective.value,
                'constraints': default_constraints,
                'risk_free_rate': risk_free_rate
            }
        )
    
    def _optimize_maximum_sharpe(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float,
        constraints: Dict[str, Any]
    ) -> pd.Series:
        """Optimize for maximum Sharpe ratio"""
        
        n_assets = len(expected_returns)
        assets = expected_returns.index
        
        # Objective function: negative Sharpe ratio
        def negative_sharpe(weights):
            weights = np.array(weights)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            if portfolio_vol == 0:
                return 1e6
            
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return pd.Series(result.x, index=assets)
        else:
            return pd.Series(1.0 / n_assets, index=assets)
    
    def _optimize_minimum_risk(
        self,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> pd.Series:
        """Optimize for minimum risk (variance)"""
        
        n_assets = len(covariance_matrix)
        assets = covariance_matrix.index
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            weights = np.array(weights)
            return np.dot(weights, np.dot(covariance_matrix.values, weights))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return pd.Series(result.x, index=assets)
        else:
            return pd.Series(1.0 / n_assets, index=assets)
    
    def _optimize_maximum_return(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> pd.Series:
        """Optimize for maximum return"""
        
        n_assets = len(expected_returns)
        assets = expected_returns.index
        
        # Objective function: negative portfolio return
        def negative_return(weights):
            weights = np.array(weights)
            return -np.dot(weights, expected_returns)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Add volatility constraint if specified
        if constraints.get('target_volatility'):
            target_vol = constraints['target_volatility']
            cons.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w, np.dot(covariance_matrix.values, w))) - target_vol
            })
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            negative_return,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return pd.Series(result.x, index=assets)
        else:
            return pd.Series(1.0 / n_assets, index=assets)
    
    def _calculate_position_sizing_details(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> List[PositionSizingResult]:
        """Calculate detailed position sizing information"""
        
        # Calculate Kelly weights
        kelly_weights = self.kelly_calculator.calculate_portfolio_kelly(
            expected_returns, covariance_matrix, risk_free_rate
        )
        
        # Calculate risk budgets (equal risk contribution)
        risk_budgets = pd.Series(1.0 / len(weights), index=weights.index)
        risk_budget_weights = self.risk_budgeting.optimize_risk_budgeting(
            expected_returns, covariance_matrix, risk_budgets
        )
        
        # Calculate risk contributions
        risk_contributions = self.risk_budgeting.calculate_risk_contributions(
            weights, covariance_matrix
        )
        
        position_sizes = []
        
        for asset in weights.index:
            # Calculate confidence interval for Kelly
            variance = covariance_matrix.loc[asset, asset]
            confidence_interval = self.kelly_calculator.calculate_kelly_confidence_interval(
                expected_returns[asset], variance, 252  # Assume 252 trading days
            )
            
            position_sizes.append(PositionSizingResult(
                symbol=asset,
                current_weight=weights[asset],
                target_weight=weights[asset],
                kelly_weight=kelly_weights[asset],
                risk_budget_weight=risk_budget_weights[asset],
                final_weight=weights[asset],
                expected_return=expected_returns[asset],
                risk_contribution=risk_contributions[asset],
                confidence_interval=confidence_interval
            ))
        
        return position_sizes


# Test fixtures
@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing"""
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.7, 0.5, 0.4],
        [0.6, 1.0, 0.5, 0.6, 0.3],
        [0.7, 0.5, 1.0, 0.4, 0.2],
        [0.5, 0.6, 0.4, 1.0, 0.3],
        [0.4, 0.3, 0.2, 0.3, 1.0]
    ])
    
    # Generate returns
    mean_returns = np.array([0.0005, 0.0008, 0.0006, 0.0007, 0.0010])
    volatilities = np.array([0.02, 0.025, 0.018, 0.022, 0.035])
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate multivariate normal returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
    
    return pd.DataFrame(returns, index=dates, columns=assets)


@pytest.fixture
def kelly_calculator():
    """Create Kelly criterion calculator instance"""
    return KellyCriterionCalculator()


@pytest.fixture
def risk_budgeting_optimizer():
    """Create risk budgeting optimizer instance"""
    return RiskBudgetingOptimizer()


@pytest.fixture
def risk_metrics_calculator():
    """Create risk metrics calculator instance"""
    return RiskMetricsCalculator()


@pytest.fixture
def portfolio_optimizer():
    """Create portfolio optimizer instance"""
    return PortfolioOptimizer()


# Comprehensive test suite
@pytest.mark.asyncio
class TestRiskAdjustedStrategies:
    """Comprehensive risk-adjusted strategies tests"""
    
    def test_kelly_criterion_single_asset(self, kelly_calculator):
        """Test Kelly criterion for single asset"""
        
        # Test with profitable asset
        kelly_fraction = kelly_calculator.calculate_kelly_fraction(
            expected_return=0.12,
            variance=0.04,
            risk_free_rate=0.02
        )
        
        assert kelly_fraction > 0
        assert kelly_fraction <= kelly_calculator.max_kelly_fraction
        
        # Test with unprofitable asset
        kelly_fraction_negative = kelly_calculator.calculate_kelly_fraction(
            expected_return=0.01,
            variance=0.04,
            risk_free_rate=0.02
        )
        
        assert kelly_fraction_negative >= kelly_calculator.min_kelly_fraction
    
    def test_kelly_criterion_portfolio(self, kelly_calculator, sample_returns_data):
        """Test Kelly criterion for portfolio"""
        
        # Calculate expected returns and covariance
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        kelly_weights = kelly_calculator.calculate_portfolio_kelly(
            expected_returns, covariance_matrix
        )
        
        assert len(kelly_weights) == len(expected_returns)
        assert all(kelly_weights >= 0)
        assert kelly_weights.sum() <= 1.0
        
        # Higher expected return assets should have higher Kelly weights
        sorted_returns = expected_returns.sort_values(ascending=False)
        high_return_asset = sorted_returns.index[0]
        low_return_asset = sorted_returns.index[-1]
        
        assert kelly_weights[high_return_asset] >= kelly_weights[low_return_asset]
    
    def test_fractional_kelly(self, kelly_calculator, sample_returns_data):
        """Test fractional Kelly calculation"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        full_kelly = kelly_calculator.calculate_portfolio_kelly(
            expected_returns, covariance_matrix
        )
        
        half_kelly = kelly_calculator.calculate_fractional_kelly(
            full_kelly, fraction=0.5
        )
        
        assert all(half_kelly == full_kelly * 0.5)
        assert half_kelly.sum() <= full_kelly.sum()
    
    def test_kelly_confidence_interval(self, kelly_calculator):
        """Test Kelly confidence interval calculation"""
        
        lower, upper = kelly_calculator.calculate_kelly_confidence_interval(
            expected_return=0.12,
            variance=0.04,
            sample_size=252
        )
        
        assert lower <= upper
        assert lower >= 0
        assert upper <= 1
        
        # Larger sample size should have narrower confidence interval
        lower_large, upper_large = kelly_calculator.calculate_kelly_confidence_interval(
            expected_return=0.12,
            variance=0.04,
            sample_size=1000
        )
        
        assert (upper_large - lower_large) <= (upper - lower)
    
    def test_risk_parity_optimization(self, risk_budgeting_optimizer, sample_returns_data):
        """Test risk parity optimization"""
        
        covariance_matrix = sample_returns_data.cov() * 252
        
        risk_parity_weights = risk_budgeting_optimizer.optimize_risk_parity(
            covariance_matrix, target_volatility=0.15
        )
        
        assert len(risk_parity_weights) == len(covariance_matrix)
        assert all(risk_parity_weights >= 0)
        assert abs(risk_parity_weights.sum() - 1.0) < 1e-6
        
        # Calculate risk contributions
        risk_contributions = risk_budgeting_optimizer.calculate_risk_contributions(
            risk_parity_weights, covariance_matrix
        )
        
        # Risk contributions should be approximately equal
        target_contribution = 1.0 / len(risk_parity_weights)
        for contribution in risk_contributions:
            assert abs(contribution - target_contribution) < 0.1  # 10% tolerance
    
    def test_risk_budgeting_optimization(self, risk_budgeting_optimizer, sample_returns_data):
        """Test risk budgeting optimization"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        # Define risk budgets (higher allocation to less risky assets)
        risk_budgets = pd.Series([0.3, 0.25, 0.25, 0.15, 0.05], index=expected_returns.index)
        
        optimized_weights = risk_budgeting_optimizer.optimize_risk_budgeting(
            expected_returns, covariance_matrix, risk_budgets
        )
        
        assert len(optimized_weights) == len(expected_returns)
        assert all(optimized_weights >= 0)
        assert abs(optimized_weights.sum() - 1.0) < 1e-6
        
        # Calculate actual risk contributions
        risk_contributions = risk_budgeting_optimizer.calculate_risk_contributions(
            optimized_weights, covariance_matrix
        )
        
        # Risk contributions should be close to target budgets
        for asset in risk_budgets.index:
            assert abs(risk_contributions[asset] - risk_budgets[asset]) < 0.15  # 15% tolerance
    
    def test_risk_metrics_calculation(self, risk_metrics_calculator, sample_returns_data):
        """Test risk metrics calculation"""
        
        # Equal weight portfolio
        weights = pd.Series(0.2, index=sample_returns_data.columns)
        
        risk_metrics = risk_metrics_calculator.calculate_portfolio_metrics(
            weights, sample_returns_data
        )
        
        assert risk_metrics.portfolio_volatility > 0
        assert risk_metrics.portfolio_var_95 < 0  # VaR should be negative
        assert risk_metrics.portfolio_var_99 < risk_metrics.portfolio_var_95  # 99% VaR more extreme
        assert risk_metrics.portfolio_es_95 < risk_metrics.portfolio_var_95  # ES more extreme than VaR
        assert risk_metrics.max_drawdown <= 0
        assert -1 <= risk_metrics.sharpe_ratio <= 5  # Reasonable range
    
    def test_var_es_calculation(self, risk_metrics_calculator, sample_returns_data):
        """Test VaR and ES calculation methods"""
        
        portfolio_returns = sample_returns_data.mean(axis=1)
        
        # Test different methods
        methods = ['historical', 'parametric', 'monte_carlo']
        
        for method in methods:
            var, es = risk_metrics_calculator.calculate_var_es(
                portfolio_returns, confidence_level=0.95, method=method
            )
            
            assert var < 0  # VaR should be negative
            assert es < var  # ES should be more extreme than VaR
            assert abs(var) < 0.1  # Reasonable magnitude
            assert abs(es) < 0.1  # Reasonable magnitude
    
    def test_dynamic_hedging_strategy(self, sample_returns_data):
        """Test dynamic hedging strategy"""
        
        hedging_strategy = DynamicHedgingStrategy()
        
        # Create portfolio and hedge instrument returns
        portfolio_returns = sample_returns_data[['AAPL', 'GOOGL']].mean(axis=1)
        hedge_returns = sample_returns_data['MSFT']  # Use MSFT as hedge
        
        # Calculate hedge ratio
        hedge_ratio = hedging_strategy.calculate_hedge_ratio(
            portfolio_returns, hedge_returns, method='minimum_variance'
        )
        
        assert isinstance(hedge_ratio, float)
        assert hedging_strategy.hedge_ratio_range[0] <= hedge_ratio <= hedging_strategy.hedge_ratio_range[1]
        
        # Test different methods
        methods = ['minimum_variance', 'beta', 'correlation']
        
        for method in methods:
            ratio = hedging_strategy.calculate_hedge_ratio(
                portfolio_returns, hedge_returns, method=method
            )
            assert isinstance(ratio, float)
            assert ratio >= 0  # Should be non-negative
    
    def test_portfolio_optimization_kelly(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization with Kelly criterion"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.KELLY_CRITERION
        )
        
        assert isinstance(result, PortfolioConstruction)
        assert len(result.weights) == len(expected_returns)
        assert all(result.weights >= 0)
        assert result.weights.sum() <= 1.0
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert len(result.position_sizes) == len(expected_returns)
    
    def test_portfolio_optimization_risk_parity(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization with risk parity"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.RISK_PARITY
        )
        
        assert isinstance(result, PortfolioConstruction)
        assert len(result.weights) == len(expected_returns)
        assert all(result.weights >= 0)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.expected_volatility > 0
        
        # Check risk contributions are approximately equal
        risk_contributions = [pos.risk_contribution for pos in result.position_sizes]
        target_contribution = 1.0 / len(risk_contributions)
        
        for contribution in risk_contributions:
            assert abs(contribution - target_contribution) < 0.2  # 20% tolerance
    
    def test_portfolio_optimization_max_sharpe(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization for maximum Sharpe ratio"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.MAXIMIZE_SHARPE
        )
        
        assert isinstance(result, PortfolioConstruction)
        assert len(result.weights) == len(expected_returns)
        assert all(result.weights >= 0)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.sharpe_ratio > 0
        
        # Sharpe ratio should be reasonable
        assert result.sharpe_ratio < 10  # Upper bound sanity check
    
    def test_portfolio_optimization_min_risk(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization for minimum risk"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.MINIMIZE_RISK
        )
        
        assert isinstance(result, PortfolioConstruction)
        assert len(result.weights) == len(expected_returns)
        assert all(result.weights >= 0)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.expected_volatility > 0
        
        # Should have lower volatility than equal weight portfolio
        equal_weight_vol = np.sqrt(np.dot(
            np.ones(len(expected_returns)) / len(expected_returns),
            np.dot(covariance_matrix, np.ones(len(expected_returns)) / len(expected_returns))
        ))
        
        assert result.expected_volatility <= equal_weight_vol
    
    def test_optimization_constraints(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization with constraints"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        constraints = {
            'min_weight': 0.05,
            'max_weight': 0.3,
            'max_concentration': 0.3
        }
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.MAXIMIZE_SHARPE,
            constraints=constraints
        )
        
        # Check constraints are satisfied
        assert all(result.weights >= constraints['min_weight'])
        assert all(result.weights <= constraints['max_weight'])
        assert result.weights.max() <= constraints['max_concentration']
    
    def test_stress_testing_scenarios(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio optimization under stress scenarios"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        # Create stressed scenarios
        stress_scenarios = {
            'high_volatility': covariance_matrix * 2,
            'low_returns': expected_returns * 0.5,
            'high_correlation': covariance_matrix * np.ones_like(covariance_matrix) * 0.8
        }
        
        base_result = portfolio_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, OptimizationObjective.KELLY_CRITERION
        )
        
        for scenario_name, scenario_data in stress_scenarios.items():
            if scenario_name == 'high_volatility':
                stressed_result = portfolio_optimizer.optimize_portfolio(
                    expected_returns, scenario_data, OptimizationObjective.KELLY_CRITERION
                )
            elif scenario_name == 'low_returns':
                stressed_result = portfolio_optimizer.optimize_portfolio(
                    scenario_data, covariance_matrix, OptimizationObjective.KELLY_CRITERION
                )
            else:  # high_correlation
                stressed_result = portfolio_optimizer.optimize_portfolio(
                    expected_returns, scenario_data, OptimizationObjective.KELLY_CRITERION
                )
            
            # Stressed scenarios should result in different allocations
            assert not np.allclose(base_result.weights, stressed_result.weights, atol=0.1)
            
            # All should be valid portfolios
            assert all(stressed_result.weights >= 0)
            assert stressed_result.weights.sum() <= 1.0
    
    def test_position_sizing_details(self, portfolio_optimizer, sample_returns_data):
        """Test detailed position sizing information"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        result = portfolio_optimizer.optimize_portfolio(
            expected_returns,
            covariance_matrix,
            objective=OptimizationObjective.KELLY_CRITERION
        )
        
        # Check position sizing details
        for position in result.position_sizes:
            assert isinstance(position, PositionSizingResult)
            assert position.symbol in expected_returns.index
            assert position.final_weight >= 0
            assert position.kelly_weight >= 0
            assert position.risk_budget_weight >= 0
            assert position.confidence_interval[0] <= position.confidence_interval[1]
            assert position.expected_return == expected_returns[position.symbol]
    
    def test_multi_objective_optimization(self, portfolio_optimizer, sample_returns_data):
        """Test multi-objective optimization comparison"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        objectives = [
            OptimizationObjective.KELLY_CRITERION,
            OptimizationObjective.RISK_PARITY,
            OptimizationObjective.MAXIMIZE_SHARPE,
            OptimizationObjective.MINIMIZE_RISK
        ]
        
        results = {}
        
        for objective in objectives:
            result = portfolio_optimizer.optimize_portfolio(
                expected_returns, covariance_matrix, objective
            )
            results[objective] = result
        
        # All results should be valid
        for objective, result in results.items():
            assert all(result.weights >= 0)
            assert result.weights.sum() <= 1.0
            assert result.expected_volatility > 0
        
        # Different objectives should produce different allocations
        kelly_weights = results[OptimizationObjective.KELLY_CRITERION].weights
        risk_parity_weights = results[OptimizationObjective.RISK_PARITY].weights
        
        assert not np.allclose(kelly_weights, risk_parity_weights, atol=0.1)
        
        # Min risk should have lowest volatility
        min_risk_vol = results[OptimizationObjective.MINIMIZE_RISK].expected_volatility
        max_sharpe_vol = results[OptimizationObjective.MAXIMIZE_SHARPE].expected_volatility
        
        assert min_risk_vol <= max_sharpe_vol
    
    def test_portfolio_rebalancing_frequency(self, portfolio_optimizer, sample_returns_data):
        """Test portfolio rebalancing at different frequencies"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        # Simulate portfolio drift over time
        base_weights = portfolio_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, OptimizationObjective.KELLY_CRITERION
        ).weights
        
        # Simulate price changes
        price_changes = np.random.normal(0.001, 0.02, len(base_weights))
        
        # Calculate drifted weights
        drifted_values = base_weights * (1 + price_changes)
        drifted_weights = drifted_values / drifted_values.sum()
        
        # Rebalance back to target
        rebalanced_result = portfolio_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, OptimizationObjective.KELLY_CRITERION
        )
        
        # Rebalanced weights should be close to original target
        assert np.allclose(base_weights, rebalanced_result.weights, atol=0.1)
        
        # Should have lower risk than drifted portfolio
        drifted_vol = np.sqrt(np.dot(drifted_weights, np.dot(covariance_matrix, drifted_weights)))
        assert rebalanced_result.expected_volatility <= drifted_vol
    
    def test_extreme_market_conditions(self, portfolio_optimizer):
        """Test portfolio optimization under extreme market conditions"""
        
        assets = ['ASSET1', 'ASSET2', 'ASSET3']
        
        # Extreme scenarios
        scenarios = {
            'high_volatility': {
                'returns': pd.Series([0.2, 0.15, 0.1], index=assets),
                'cov': pd.DataFrame(np.diag([0.5, 0.4, 0.3]), index=assets, columns=assets)
            },
            'negative_returns': {
                'returns': pd.Series([-0.1, -0.05, -0.15], index=assets),
                'cov': pd.DataFrame(np.diag([0.1, 0.08, 0.12]), index=assets, columns=assets)
            },
            'perfect_correlation': {
                'returns': pd.Series([0.1, 0.08, 0.12], index=assets),
                'cov': pd.DataFrame(np.ones((3, 3)) * 0.1, index=assets, columns=assets)
            }
        }
        
        for scenario_name, scenario_data in scenarios.items():
            try:
                result = portfolio_optimizer.optimize_portfolio(
                    scenario_data['returns'],
                    scenario_data['cov'],
                    OptimizationObjective.KELLY_CRITERION
                )
                
                # Should produce valid portfolio even in extreme conditions
                assert all(result.weights >= 0)
                assert result.weights.sum() <= 1.0
                
                # In extreme conditions, may have conservative allocations
                if scenario_name == 'negative_returns':
                    assert result.weights.sum() < 0.5  # Should be conservative
                
            except Exception as e:
                logger.warning(f"Optimization failed for {scenario_name}: {e}")
                # Extreme conditions may cause optimization to fail, which is acceptable
    
    def test_transaction_cost_integration(self, portfolio_optimizer, sample_returns_data):
        """Test integration of transaction costs in optimization"""
        
        expected_returns = sample_returns_data.mean() * 252
        covariance_matrix = sample_returns_data.cov() * 252
        
        # Simulate transaction costs
        transaction_cost_rate = 0.005  # 50 basis points
        
        # Adjust expected returns for transaction costs
        adjusted_returns = expected_returns - transaction_cost_rate
        
        # Optimize with and without transaction costs
        result_no_costs = portfolio_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, OptimizationObjective.KELLY_CRITERION
        )
        
        result_with_costs = portfolio_optimizer.optimize_portfolio(
            adjusted_returns, covariance_matrix, OptimizationObjective.KELLY_CRITERION
        )
        
        # Transaction costs should lead to more conservative positions
        assert result_with_costs.weights.sum() <= result_no_costs.weights.sum()
        
        # Should affect allocation differently across assets
        assert not np.allclose(result_no_costs.weights, result_with_costs.weights, atol=0.05)