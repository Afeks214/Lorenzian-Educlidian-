"""
Numba JIT Optimized Risk Calculation Engine
===========================================

This module implements high-performance risk calculations using Numba JIT compilation
for institutional-grade speed and efficiency. It provides:

- Ultra-fast VaR and CVaR calculations
- Optimized portfolio risk metrics
- Real-time correlation and covariance calculations
- Efficient Monte Carlo simulations
- Memory-optimized risk tracking
- Parallel processing capabilities

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from numba import jit, njit, prange, cuda
from numba.types import float64, int32, boolean
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Constants for numerical stability
EPSILON = 1e-12
MAX_ITERATIONS = 1000
CONVERGENCE_THRESHOLD = 1e-8


@dataclass
class RiskCalculationConfig:
    """Configuration for risk calculations"""
    use_parallel: bool = True
    use_gpu: bool = False
    max_threads: int = 4
    chunk_size: int = 1000
    precision: str = "float64"
    memory_limit_gb: float = 8.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'use_parallel': self.use_parallel,
            'use_gpu': self.use_gpu,
            'max_threads': self.max_threads,
            'chunk_size': self.chunk_size,
            'precision': self.precision,
            'memory_limit_gb': self.memory_limit_gb
        }


# Core JIT optimized functions
@njit(float64(float64[:], float64), cache=True, fastmath=True)
def fast_quantile(data: np.ndarray, quantile: float) -> float:
    """
    Fast quantile calculation - JIT optimized
    
    Args:
        data: Input data array
        quantile: Quantile level (0.0 to 1.0)
    
    Returns:
        Quantile value
    """
    if len(data) == 0:
        return 0.0
    
    # Sort data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Calculate index
    index = quantile * (n - 1)
    
    # Interpolate if needed
    if index == int(index):
        return sorted_data[int(index)]
    else:
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        
        if upper_idx >= n:
            return sorted_data[n - 1]
        
        weight = index - lower_idx
        return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight


@njit(float64(float64[:], float64), cache=True, fastmath=True)
def fast_var(returns: np.ndarray, confidence_level: float) -> float:
    """
    Fast VaR calculation - JIT optimized
    
    Args:
        returns: Return array
        confidence_level: Confidence level (0.0 to 1.0)
    
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate quantile
    quantile_level = 1.0 - confidence_level
    var_value = fast_quantile(returns, quantile_level)
    
    return abs(var_value)


@njit(float64(float64[:], float64), cache=True, fastmath=True)
def fast_cvar(returns: np.ndarray, confidence_level: float) -> float:
    """
    Fast CVaR (Expected Shortfall) calculation - JIT optimized
    
    Args:
        returns: Return array
        confidence_level: Confidence level (0.0 to 1.0)
    
    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate VaR threshold
    var_threshold = -fast_var(returns, confidence_level)
    
    # Get tail returns
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return abs(var_threshold)
    
    # Calculate expected shortfall
    cvar_value = np.mean(tail_returns)
    
    return abs(cvar_value)


@njit(float64(float64[:], float64[:]), cache=True, fastmath=True)
def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fast correlation calculation - JIT optimized
    
    Args:
        x: First time series
        y: Second time series
    
    Returns:
        Correlation coefficient
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate numerator and denominators
    numerator = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    
    for i in range(len(x)):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        numerator += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    
    # Calculate correlation
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator < EPSILON:
        return 0.0
    
    correlation = numerator / denominator
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, correlation))


@njit(float64[:, :](float64[:, :]), cache=True, fastmath=True)
def fast_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Fast correlation matrix calculation - JIT optimized
    
    Args:
        returns: Return matrix (time x assets)
    
    Returns:
        Correlation matrix
    """
    n_assets = returns.shape[1]
    correlation_matrix = np.eye(n_assets)
    
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr = fast_correlation(returns[:, i], returns[:, j])
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    return correlation_matrix


@njit(float64[:, :](float64[:, :]), cache=True, fastmath=True)
def fast_covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Fast covariance matrix calculation - JIT optimized
    
    Args:
        returns: Return matrix (time x assets)
    
    Returns:
        Covariance matrix
    """
    n_time, n_assets = returns.shape
    covariance_matrix = np.zeros((n_assets, n_assets))
    
    # Calculate means
    means = np.mean(returns, axis=0)
    
    # Calculate covariances
    for i in range(n_assets):
        for j in range(i, n_assets):
            covariance = 0.0
            for t in range(n_time):
                covariance += (returns[t, i] - means[i]) * (returns[t, j] - means[j])
            
            covariance /= (n_time - 1)
            covariance_matrix[i, j] = covariance
            covariance_matrix[j, i] = covariance
    
    return covariance_matrix


@njit(float64(float64[:], float64[:, :], float64[:]), cache=True, fastmath=True)
def fast_portfolio_var(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    returns: np.ndarray
) -> float:
    """
    Fast portfolio VaR calculation - JIT optimized
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        returns: Historical returns matrix
    
    Returns:
        Portfolio VaR
    """
    # Calculate portfolio variance
    portfolio_variance = 0.0
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            portfolio_variance += weights[i] * weights[j] * covariance_matrix[i, j]
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    
    # Calculate VaR
    var_value = fast_var(portfolio_returns, 0.95)
    
    return var_value


@njit(float64[:](float64[:], float64[:, :], float64[:]), cache=True, fastmath=True)
def fast_marginal_var(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    returns: np.ndarray
) -> np.ndarray:
    """
    Fast marginal VaR calculation - JIT optimized
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        returns: Historical returns matrix
    
    Returns:
        Marginal VaR for each asset
    """
    n_assets = len(weights)
    marginal_vars = np.zeros(n_assets)
    
    # Calculate portfolio variance
    portfolio_variance = 0.0
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += weights[i] * weights[j] * covariance_matrix[i, j]
    
    portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))
    
    if portfolio_volatility < EPSILON:
        return marginal_vars
    
    # Calculate marginal contributions
    for i in range(n_assets):
        marginal_contribution = 0.0
        for j in range(n_assets):
            marginal_contribution += weights[j] * covariance_matrix[i, j]
        
        marginal_vars[i] = marginal_contribution / portfolio_volatility
    
    return marginal_vars


@njit(float64[:](float64[:], float64[:, :], float64[:]), cache=True, fastmath=True)
def fast_component_var(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    returns: np.ndarray
) -> np.ndarray:
    """
    Fast component VaR calculation - JIT optimized
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        returns: Historical returns matrix
    
    Returns:
        Component VaR for each asset
    """
    marginal_vars = fast_marginal_var(weights, covariance_matrix, returns)
    component_vars = weights * marginal_vars
    
    return component_vars


@njit(parallel=True, cache=True, fastmath=True)
def fast_rolling_var(
    returns: np.ndarray,
    window: int32,
    confidence_level: float64
) -> np.ndarray:
    """
    Fast rolling VaR calculation - JIT optimized with parallel processing
    
    Args:
        returns: Return array
        window: Rolling window size
        confidence_level: Confidence level
    
    Returns:
        Rolling VaR values
    """
    n = len(returns)
    rolling_vars = np.zeros(n)
    
    for i in prange(window, n):
        window_returns = returns[i - window:i]
        rolling_vars[i] = fast_var(window_returns, confidence_level)
    
    return rolling_vars


@njit(parallel=True, cache=True, fastmath=True)
def fast_rolling_correlation(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    window: int32
) -> np.ndarray:
    """
    Fast rolling correlation calculation - JIT optimized with parallel processing
    
    Args:
        returns_x: First return series
        returns_y: Second return series
        window: Rolling window size
    
    Returns:
        Rolling correlation values
    """
    n = len(returns_x)
    rolling_corr = np.zeros(n)
    
    for i in prange(window, n):
        window_x = returns_x[i - window:i]
        window_y = returns_y[i - window:i]
        rolling_corr[i] = fast_correlation(window_x, window_y)
    
    return rolling_corr


@njit(float64(float64[:], int32), cache=True, fastmath=True)
def fast_maximum_drawdown(equity_curve: np.ndarray, lookback: int32) -> float:
    """
    Fast maximum drawdown calculation - JIT optimized
    
    Args:
        equity_curve: Equity curve
        lookback: Lookback period
    
    Returns:
        Maximum drawdown
    """
    n = len(equity_curve)
    
    if n == 0:
        return 0.0
    
    # Use lookback period if specified
    start_idx = max(0, n - lookback) if lookback > 0 else 0
    
    max_drawdown = 0.0
    running_max = equity_curve[start_idx]
    
    for i in range(start_idx, n):
        # Update running maximum
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]
        
        # Calculate drawdown
        drawdown = (running_max - equity_curve[i]) / running_max
        
        # Update maximum drawdown
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown


@njit(float64(float64[:], float64[:], float64), cache=True, fastmath=True)
def fast_sharpe_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float64
) -> float:
    """
    Fast Sharpe ratio calculation - JIT optimized
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate
    
    # Calculate mean and standard deviation
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess < EPSILON:
        return 0.0
    
    return mean_excess / std_excess


@njit(float64(float64[:], float64), cache=True, fastmath=True)
def fast_sortino_ratio(returns: np.ndarray, risk_free_rate: float64) -> float:
    """
    Fast Sortino ratio calculation - JIT optimized
    
    Args:
        returns: Portfolio returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate
    
    # Calculate mean excess return
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_variance = np.mean(downside_returns ** 2)
    downside_deviation = np.sqrt(downside_variance)
    
    if downside_deviation < EPSILON:
        return 0.0
    
    return mean_excess / downside_deviation


@njit(float64(float64[:], float64[:]), cache=True, fastmath=True)
def fast_beta(portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """
    Fast beta calculation - JIT optimized
    
    Args:
        portfolio_returns: Portfolio returns
        market_returns: Market returns
    
    Returns:
        Beta coefficient
    """
    if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) == 0:
        return 1.0
    
    # Calculate covariance and variance
    portfolio_mean = np.mean(portfolio_returns)
    market_mean = np.mean(market_returns)
    
    covariance = 0.0
    market_variance = 0.0
    
    for i in range(len(portfolio_returns)):
        portfolio_excess = portfolio_returns[i] - portfolio_mean
        market_excess = market_returns[i] - market_mean
        
        covariance += portfolio_excess * market_excess
        market_variance += market_excess ** 2
    
    if market_variance < EPSILON:
        return 1.0
    
    return covariance / market_variance


@njit(parallel=True, cache=True, fastmath=True)
def fast_monte_carlo_var(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: np.ndarray,
    num_simulations: int32,
    confidence_level: float64,
    time_horizon: int32
) -> float:
    """
    Fast Monte Carlo VaR calculation - JIT optimized with parallel processing
    
    Args:
        mean_returns: Expected returns
        covariance_matrix: Covariance matrix
        weights: Portfolio weights
        num_simulations: Number of simulations
        confidence_level: Confidence level
        time_horizon: Time horizon
    
    Returns:
        Monte Carlo VaR
    """
    n_assets = len(weights)
    portfolio_returns = np.zeros(num_simulations)
    
    # Cholesky decomposition for covariance matrix
    cholesky = np.linalg.cholesky(covariance_matrix)
    
    # Run simulations in parallel
    for sim in prange(num_simulations):
        # Generate random normal vector
        random_vector = np.random.randn(n_assets)
        
        # Apply Cholesky decomposition
        correlated_returns = np.dot(cholesky, random_vector)
        
        # Scale for time horizon
        simulated_returns = (mean_returns * time_horizon + 
                           correlated_returns * np.sqrt(time_horizon))
        
        # Calculate portfolio return
        portfolio_returns[sim] = np.dot(simulated_returns, weights)
    
    # Calculate VaR from simulated returns
    var_value = fast_var(portfolio_returns, confidence_level)
    
    return var_value


@njit(float64(float64[:], float64[:], float64), cache=True, fastmath=True)
def fast_tracking_error(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: float64
) -> float:
    """
    Fast tracking error calculation - JIT optimized
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized tracking error
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = portfolio_returns - benchmark_returns
    
    # Calculate standard deviation
    tracking_error = np.std(excess_returns)
    
    # Annualize
    annualized_tracking_error = tracking_error * np.sqrt(periods_per_year)
    
    return annualized_tracking_error


@njit(float64(float64[:], float64[:], float64), cache=True, fastmath=True)
def fast_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: float64
) -> float:
    """
    Fast information ratio calculation - JIT optimized
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
    
    Returns:
        Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = portfolio_returns - benchmark_returns
    
    # Calculate mean excess return
    mean_excess = np.mean(excess_returns)
    
    # Calculate tracking error
    tracking_error = np.std(excess_returns)
    
    if tracking_error < EPSILON:
        return 0.0
    
    # Annualize
    annualized_excess = mean_excess * periods_per_year
    annualized_tracking_error = tracking_error * np.sqrt(periods_per_year)
    
    return annualized_excess / annualized_tracking_error


@njit(float64(float64[:]), cache=True, fastmath=True)
def fast_herfindahl_index(weights: np.ndarray) -> float:
    """
    Fast Herfindahl concentration index calculation - JIT optimized
    
    Args:
        weights: Portfolio weights
    
    Returns:
        Herfindahl index
    """
    if len(weights) == 0:
        return 0.0
    
    # Calculate sum of squared weights
    herfindahl = 0.0
    for i in range(len(weights)):
        herfindahl += weights[i] ** 2
    
    return herfindahl


@njit(float64(float64[:], float64[:, :]), cache=True, fastmath=True)
def fast_portfolio_concentration(
    weights: np.ndarray,
    correlation_matrix: np.ndarray
) -> float:
    """
    Fast portfolio concentration calculation - JIT optimized
    
    Args:
        weights: Portfolio weights
        correlation_matrix: Correlation matrix
    
    Returns:
        Portfolio concentration measure
    """
    if len(weights) == 0:
        return 0.0
    
    # Calculate effective number of independent positions
    n_assets = len(weights)
    effective_positions = 0.0
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                effective_positions += weights[i] ** 2
            else:
                effective_positions += weights[i] * weights[j] * correlation_matrix[i, j]
    
    # Normalize
    if effective_positions > 0:
        return 1.0 / effective_positions
    else:
        return 0.0


class NumbaRiskEngine:
    """
    High-performance risk calculation engine using Numba JIT compilation
    
    This class provides institutional-grade performance for risk calculations
    with optimized memory usage and parallel processing capabilities.
    """
    
    def __init__(self, config: RiskCalculationConfig):
        """
        Initialize the Numba risk engine
        
        Args:
            config: Risk calculation configuration
        """
        self.config = config
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = {}
        self.cache_stats: Dict[str, int] = {'hits': 0, 'misses': 0}
        
        # Memory management
        self.memory_usage: Dict[str, float] = {}
        
        # Warm up JIT functions
        self._warm_up_jit()
        
        logger.info("NumbaRiskEngine initialized",
                   extra={'config': config.to_dict()})
    
    def _warm_up_jit(self) -> None:
        """Warm up JIT functions for better performance"""
        
        # Create dummy data for warm-up
        dummy_returns = np.random.randn(100)
        dummy_matrix = np.random.randn(100, 10)
        dummy_weights = np.random.rand(10)
        dummy_weights /= np.sum(dummy_weights)
        
        # Warm up key functions
        try:
            _ = fast_var(dummy_returns, 0.95)
            _ = fast_cvar(dummy_returns, 0.95)
            _ = fast_correlation_matrix(dummy_matrix)
            _ = fast_covariance_matrix(dummy_matrix)
            _ = fast_portfolio_var(dummy_weights, np.eye(10), dummy_matrix)
            
            logger.info("JIT functions warmed up successfully")
        except Exception as e:
            logger.warning(f"JIT warm-up failed: {e}")
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk with optimized performance
        
        Args:
            returns: Return array
            confidence_level: Confidence level
            method: Calculation method
        
        Returns:
            VaR value
        """
        start_time = datetime.now()
        
        try:
            if method == "historical":
                result = fast_var(returns, confidence_level)
            elif method == "monte_carlo":
                # For single asset Monte Carlo
                mean_return = np.mean(returns)
                volatility = np.std(returns)
                
                # Generate simulated returns
                num_sims = 10000
                simulated_returns = np.random.normal(mean_return, volatility, num_sims)
                result = fast_var(simulated_returns, confidence_level)
            else:
                result = fast_var(returns, confidence_level)
            
            return result
            
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'var' not in self.calculation_times:
                self.calculation_times['var'] = []
            
            self.calculation_times['var'].append(calc_time)
            
            # Keep only recent times
            if len(self.calculation_times['var']) > 1000:
                self.calculation_times['var'] = self.calculation_times['var'][-1000:]
    
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk with optimized performance
        
        Args:
            returns: Return array
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        start_time = datetime.now()
        
        try:
            result = fast_cvar(returns, confidence_level)
            return result
            
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'cvar' not in self.calculation_times:
                self.calculation_times['cvar'] = []
            
            self.calculation_times['cvar'].append(calc_time)
    
    def calculate_portfolio_risk(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            returns: Return matrix (time x assets)
            weights: Portfolio weights
            confidence_level: Confidence level
        
        Returns:
            Dictionary of risk metrics
        """
        start_time = datetime.now()
        
        try:
            # Calculate covariance matrix
            covariance_matrix = fast_covariance_matrix(returns)
            
            # Calculate portfolio VaR
            portfolio_var = fast_portfolio_var(weights, covariance_matrix, returns)
            
            # Calculate component VaR
            component_vars = fast_component_var(weights, covariance_matrix, returns)
            
            # Calculate marginal VaR
            marginal_vars = fast_marginal_var(weights, covariance_matrix, returns)
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns, weights)
            
            # Calculate CVaR
            portfolio_cvar = fast_cvar(portfolio_returns, confidence_level)
            
            # Calculate concentration risk
            concentration = fast_herfindahl_index(weights)
            
            # Calculate maximum drawdown
            equity_curve = np.cumsum(portfolio_returns)
            max_drawdown = fast_maximum_drawdown(equity_curve, 0)
            
            # Calculate performance metrics
            sharpe_ratio = fast_sharpe_ratio(portfolio_returns, np.zeros_like(portfolio_returns), 0.0)
            sortino_ratio = fast_sortino_ratio(portfolio_returns, 0.0)
            
            return {
                'portfolio_var': portfolio_var,
                'portfolio_cvar': portfolio_cvar,
                'component_vars': component_vars,
                'marginal_vars': marginal_vars,
                'concentration': concentration,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'portfolio_risk' not in self.calculation_times:
                self.calculation_times['portfolio_risk'] = []
            
            self.calculation_times['portfolio_risk'].append(calc_time)
    
    def calculate_rolling_metrics(
        self,
        returns: np.ndarray,
        window: int = 252,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling risk metrics
        
        Args:
            returns: Return array
            window: Rolling window size
            confidence_level: Confidence level
        
        Returns:
            Dictionary of rolling metrics
        """
        start_time = datetime.now()
        
        try:
            # Calculate rolling VaR
            rolling_var = fast_rolling_var(returns, window, confidence_level)
            
            # Calculate rolling volatility
            rolling_volatility = np.zeros_like(returns)
            for i in range(window, len(returns)):
                rolling_volatility[i] = np.std(returns[i-window:i])
            
            # Calculate rolling maximum drawdown
            rolling_max_dd = np.zeros_like(returns)
            for i in range(window, len(returns)):
                equity_segment = np.cumsum(returns[i-window:i])
                rolling_max_dd[i] = fast_maximum_drawdown(equity_segment, 0)
            
            return {
                'rolling_var': rolling_var,
                'rolling_volatility': rolling_volatility,
                'rolling_max_drawdown': rolling_max_dd
            }
            
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'rolling_metrics' not in self.calculation_times:
                self.calculation_times['rolling_metrics'] = []
            
            self.calculation_times['rolling_metrics'].append(calc_time)
    
    def calculate_correlation_metrics(
        self,
        returns: np.ndarray,
        window: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate correlation metrics
        
        Args:
            returns: Return matrix (time x assets)
            window: Rolling window size (optional)
        
        Returns:
            Dictionary of correlation metrics
        """
        start_time = datetime.now()
        
        try:
            # Calculate correlation matrix
            correlation_matrix = fast_correlation_matrix(returns)
            
            # Calculate average correlation
            n_assets = correlation_matrix.shape[0]
            total_corr = 0.0
            count = 0
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    total_corr += correlation_matrix[i, j]
                    count += 1
            
            avg_correlation = total_corr / count if count > 0 else 0.0
            
            # Calculate rolling correlations if window specified
            rolling_correlations = {}
            if window is not None and returns.shape[1] >= 2:
                for i in range(returns.shape[1]):
                    for j in range(i + 1, returns.shape[1]):
                        key = f"corr_{i}_{j}"
                        rolling_correlations[key] = fast_rolling_correlation(
                            returns[:, i], returns[:, j], window
                        )
            
            return {
                'correlation_matrix': correlation_matrix,
                'average_correlation': avg_correlation,
                'rolling_correlations': rolling_correlations
            }
            
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'correlation_metrics' not in self.calculation_times:
                self.calculation_times['correlation_metrics'] = []
            
            self.calculation_times['correlation_metrics'].append(calc_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {}
        
        for metric_name, times in self.calculation_times.items():
            if times:
                stats[metric_name] = {
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'std_time_ms': np.std(times),
                    'calculation_count': len(times)
                }
        
        stats['cache_stats'] = self.cache_stats.copy()
        stats['memory_usage'] = self.memory_usage.copy()
        
        return stats
    
    def benchmark_performance(
        self,
        data_size: int = 10000,
        n_assets: int = 100,
        iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark performance with synthetic data
        
        Args:
            data_size: Size of synthetic data
            n_assets: Number of assets
            iterations: Number of iterations
        
        Returns:
            Benchmark results
        """
        
        # Generate synthetic data
        returns = np.random.randn(data_size, n_assets) * 0.02
        weights = np.random.rand(n_assets)
        weights /= np.sum(weights)
        
        # Benchmark VaR calculation
        start_time = datetime.now()
        for _ in range(iterations):
            portfolio_returns = np.dot(returns, weights)
            _ = fast_var(portfolio_returns, 0.95)
        var_time = (datetime.now() - start_time).total_seconds() * 1000 / iterations
        
        # Benchmark CVaR calculation
        start_time = datetime.now()
        for _ in range(iterations):
            portfolio_returns = np.dot(returns, weights)
            _ = fast_cvar(portfolio_returns, 0.95)
        cvar_time = (datetime.now() - start_time).total_seconds() * 1000 / iterations
        
        # Benchmark correlation matrix
        start_time = datetime.now()
        for _ in range(iterations):
            _ = fast_correlation_matrix(returns)
        corr_time = (datetime.now() - start_time).total_seconds() * 1000 / iterations
        
        # Benchmark portfolio risk
        start_time = datetime.now()
        for _ in range(iterations):
            covariance_matrix = fast_covariance_matrix(returns)
            _ = fast_portfolio_var(weights, covariance_matrix, returns)
        portfolio_time = (datetime.now() - start_time).total_seconds() * 1000 / iterations
        
        return {
            'var_time_ms': var_time,
            'cvar_time_ms': cvar_time,
            'correlation_time_ms': corr_time,
            'portfolio_risk_time_ms': portfolio_time,
            'data_size': data_size,
            'n_assets': n_assets,
            'iterations': iterations
        }


# Factory function
def create_numba_risk_engine(config_dict: Optional[Dict[str, Any]] = None) -> NumbaRiskEngine:
    """
    Create a Numba risk engine with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        NumbaRiskEngine instance
    """
    
    if config_dict is None:
        config = RiskCalculationConfig()
    else:
        config = RiskCalculationConfig(**config_dict)
    
    return NumbaRiskEngine(config)