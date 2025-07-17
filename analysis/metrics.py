"""
Performance Metrics for Strategic MARL System

Provides comprehensive metrics for evaluating trading performance
and comparing different agents (baseline vs MARL).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from functools import lru_cache
from numba import jit, njit
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: Optional[float] = None
    jensens_alpha: Optional[float] = None
    treynor_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    tail_ratio: Optional[float] = None
    ulcer_index: Optional[float] = None
    burke_ratio: Optional[float] = None
    sterling_ratio: Optional[float] = None
    pain_index: Optional[float] = None
    gain_to_pain_ratio: Optional[float] = None
    martin_ratio: Optional[float] = None
    conditional_sharpe_ratio: Optional[float] = None
    rachev_ratio: Optional[float] = None
    modified_sharpe_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': self.total_trades,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'information_ratio': self.information_ratio,
            'jensens_alpha': self.jensens_alpha,
            'treynor_ratio': self.treynor_ratio,
            'omega_ratio': self.omega_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio,
            'ulcer_index': self.ulcer_index,
            'burke_ratio': self.burke_ratio,
            'sterling_ratio': self.sterling_ratio,
            'pain_index': self.pain_index,
            'gain_to_pain_ratio': self.gain_to_pain_ratio,
            'martin_ratio': self.martin_ratio,
            'conditional_sharpe_ratio': self.conditional_sharpe_ratio,
            'rachev_ratio': self.rachev_ratio,
            'modified_sharpe_ratio': self.modified_sharpe_ratio
        }
        
    def __str__(self) -> str:
        """String representation"""
        return f"""Performance Metrics:
  Total Return: {self.total_return:.2%}
  Sharpe Ratio: {self.sharpe_ratio:.3f}
  Max Drawdown: {self.max_drawdown:.2%}
  Win Rate: {self.win_rate:.2%}
  Profit Factor: {self.profit_factor:.2f}
  Calmar Ratio: {self.calmar_ratio:.3f}
  Total Trades: {self.total_trades}"""


@njit
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate returns from price series (JIT optimized)
    
    Args:
        prices: Array of prices
        
    Returns:
        Array of returns
    """
    if len(prices) < 2:
<<<<<<< HEAD
        return np.array([0.0])[:0]  # Return empty array with correct type
=======
        return np.array([])
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
        
    returns = np.diff(prices) / prices[:-1]
    return returns


<<<<<<< HEAD
=======
@lru_cache(maxsize=128)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
def calculate_sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio (cached)
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
        
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside
        
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
        
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation
    return sortino


@njit
def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown (JIT optimized)
    
    Args:
        equity_curve: Array of portfolio values
        
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
        
<<<<<<< HEAD
    # Calculate running maximum manually for numba compatibility
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
=======
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find peak before trough
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])
    
    return abs(max_dd), peak_idx, max_dd_idx


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profits / gross losses)
    
    Args:
        returns: Array of returns
        
    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return 0.0
        
    profits = returns[returns > 0]
    losses = abs(returns[returns < 0])
    
    if len(losses) == 0 or sum(losses) == 0:
        return float('inf') if len(profits) > 0 else 0.0
        
    return sum(profits) / sum(losses)


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns)
    
    Args:
        returns: Array of returns
        
    Returns:
        Win rate between 0 and 1
    """
    if len(returns) == 0:
        return 0.0
        
    return np.sum(returns > 0) / len(returns)


def calculate_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Args:
        returns: Array of returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
        
    annual_return = np.mean(returns) * periods_per_year
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    return annual_return / max_dd


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio vs benchmark
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
        
    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate tracking error
    excess_returns = returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
        
    info_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / tracking_error
    return info_ratio


def calculate_jensens_alpha(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    beta: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's Alpha (risk-adjusted excess return)
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        beta: Portfolio beta relative to benchmark
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Jensen's Alpha
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
        
    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Annualize returns
    portfolio_return = np.mean(returns) * periods_per_year
    benchmark_return = np.mean(benchmark_returns) * periods_per_year
    
    # Jensen's Alpha = Rp - [Rf + β(Rm - Rf)]
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    return alpha


def calculate_treynor_ratio(
    returns: np.ndarray,
    beta: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Treynor Ratio (excess return per unit of systematic risk)
    
    Args:
        returns: Array of returns
        beta: Portfolio beta relative to benchmark
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Treynor Ratio
    """
    if len(returns) == 0 or beta == 0:
        return 0.0
        
    # Annualize returns
    portfolio_return = np.mean(returns) * periods_per_year
    
    # Treynor Ratio = (Rp - Rf) / β
    treynor = (portfolio_return - risk_free_rate) / beta
    
    return treynor


def calculate_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega Ratio (probability weighted ratio of gains to losses)
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return threshold
        
    Returns:
        Omega Ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Excess returns above threshold
    excess_returns = returns - threshold
    
    # Gains and losses
    gains = excess_returns[excess_returns > 0]
    losses = abs(excess_returns[excess_returns < 0])
    
    if len(losses) == 0:
        return float('inf') if len(gains) > 0 else 1.0
        
    if len(gains) == 0:
        return 0.0
        
    # Omega = Sum of gains / Sum of losses
    omega = np.sum(gains) / np.sum(losses)
    
    return omega


def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate skewness (measure of asymmetry)
    
    Args:
        returns: Array of returns
        
    Returns:
        Skewness
    """
    if len(returns) < 3:
        return 0.0
        
    return stats.skew(returns)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate kurtosis (measure of tail heaviness)
    
    Args:
        returns: Array of returns
        
    Returns:
        Kurtosis (excess kurtosis)
    """
    if len(returns) < 4:
        return 0.0
        
    return stats.kurtosis(returns)


def calculate_tail_ratio(
    returns: np.ndarray,
    left_percentile: float = 5.0,
    right_percentile: float = 95.0
) -> float:
    """
    Calculate tail ratio (ratio of right tail to left tail)
    
    Args:
        returns: Array of returns
        left_percentile: Left tail percentile
        right_percentile: Right tail percentile
        
    Returns:
        Tail ratio
    """
    if len(returns) == 0:
        return 0.0
        
    left_tail = np.percentile(returns, left_percentile)
    right_tail = np.percentile(returns, right_percentile)
    
    if left_tail == 0:
        return float('inf') if right_tail > 0 else 0.0
        
    return abs(right_tail / left_tail)


@njit
def calculate_ulcer_index(
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Ulcer Index (measure of downside risk) - JIT optimized
    
    Args:
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Ulcer Index
    """
    if len(equity_curve) == 0:
        return 0.0
        
    # Calculate running maximum
<<<<<<< HEAD
    # Calculate running maximum manually for numba compatibility
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
=======
    running_max = np.maximum.accumulate(equity_curve)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
    
    # Calculate drawdown percentage
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    
    # Ulcer Index = sqrt(mean(drawdown^2))
    ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
    
    return ulcer_index


@njit
def calculate_burke_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Burke Ratio (return per unit of drawdown squared) - JIT optimized
    
    Args:
        returns: Array of returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Burke Ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
        
    # Annualize returns
    annual_return = np.mean(returns) * periods_per_year
    
    # Calculate running maximum
<<<<<<< HEAD
    # Calculate running maximum manually for numba compatibility
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
=======
    running_max = np.maximum.accumulate(equity_curve)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
    
    # Calculate drawdown percentage
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    
    # Sum of squared drawdowns
    sum_squared_drawdowns = np.sum(drawdown_pct ** 2)
    
    if sum_squared_drawdowns == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    # Burke Ratio = Annual Return / sqrt(Sum of squared drawdowns)
    burke_ratio = annual_return / np.sqrt(sum_squared_drawdowns)
    
    return burke_ratio


@njit
def calculate_sterling_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sterling Ratio (return per unit of average drawdown) - JIT optimized
    
    Args:
        returns: Array of returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Sterling Ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
        
    # Annualize returns
    annual_return = np.mean(returns) * periods_per_year
    
    # Calculate running maximum
<<<<<<< HEAD
    # Calculate running maximum manually for numba compatibility
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
=======
    running_max = np.maximum.accumulate(equity_curve)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Average drawdown (only negative values)
    negative_drawdowns = drawdown[drawdown < 0]
    
    if len(negative_drawdowns) == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    avg_drawdown = abs(np.mean(negative_drawdowns))
    
    if avg_drawdown == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    # Sterling Ratio = Annual Return / Average Drawdown
    sterling_ratio = annual_return / avg_drawdown
    
    return sterling_ratio


def calculate_beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate beta (systematic risk relative to benchmark)
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Beta coefficient
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 1.0
        
    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate covariance and variance
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return 1.0
        
    beta = covariance / benchmark_variance
    
    return beta


@njit
def calculate_pain_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Pain Index (average drawdown over time)
    
    Args:
        equity_curve: Array of portfolio values
        
    Returns:
        Pain Index
    """
    if len(equity_curve) == 0:
        return 0.0
        
    # Calculate running maximum
<<<<<<< HEAD
    # Calculate running maximum manually for numba compatibility
    running_max = np.zeros_like(equity_curve)
    running_max[0] = equity_curve[0]
    for i in range(1, len(equity_curve)):
        running_max[i] = max(running_max[i-1], equity_curve[i])
=======
    running_max = np.maximum.accumulate(equity_curve)
>>>>>>> 5d16eb51764e3a7b5b5b1da0fee55c73e7d3fce8
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Pain Index = mean of absolute drawdown values
    pain_index = np.mean(np.abs(drawdown))
    
    return pain_index


@njit
def calculate_gain_to_pain_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Gain to Pain Ratio (annualized return / pain index)
    
    Args:
        returns: Array of returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Gain to Pain Ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
        
    annual_return = np.mean(returns) * periods_per_year
    pain_index = calculate_pain_index(equity_curve)
    
    if pain_index == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    return annual_return / pain_index


@njit
def calculate_martin_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Martin Ratio (return per unit of Ulcer Index)
    
    Args:
        returns: Array of returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year
        
    Returns:
        Martin Ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
        
    annual_return = np.mean(returns) * periods_per_year
    ulcer_index = calculate_ulcer_index(equity_curve, periods_per_year)
    
    if ulcer_index == 0:
        return float('inf') if annual_return > 0 else 0.0
        
    return annual_return / ulcer_index


@njit
def calculate_conditional_sharpe_ratio(
    returns: np.ndarray,
    percentile: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Conditional Sharpe Ratio (Sharpe ratio of returns below percentile)
    
    Args:
        returns: Array of returns
        percentile: Percentile threshold (0.95 = 95th percentile)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Conditional Sharpe Ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate threshold
    threshold = np.percentile(returns, percentile * 100)
    
    # Filter returns below threshold
    conditional_returns = returns[returns <= threshold]
    
    if len(conditional_returns) == 0:
        return 0.0
        
    # Calculate conditional Sharpe
    excess_returns = conditional_returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
        
    conditional_sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    return conditional_sharpe


@njit
def calculate_rachev_ratio(
    returns: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05
) -> float:
    """
    Calculate Rachev Ratio (ratio of expected tail loss to expected tail gain)
    
    Args:
        returns: Array of returns
        alpha: Left tail percentile (0.05 = 5%)
        beta: Right tail percentile (0.05 = 5%)
        
    Returns:
        Rachev Ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate tail thresholds
    left_threshold = np.percentile(returns, alpha * 100)
    right_threshold = np.percentile(returns, (1 - beta) * 100)
    
    # Extract tail returns
    left_tail = returns[returns <= left_threshold]
    right_tail = returns[returns >= right_threshold]
    
    if len(left_tail) == 0 or len(right_tail) == 0:
        return 0.0
        
    # Calculate expected tail values
    expected_tail_loss = np.mean(left_tail)
    expected_tail_gain = np.mean(right_tail)
    
    if expected_tail_loss == 0:
        return float('inf') if expected_tail_gain > 0 else 0.0
        
    return abs(expected_tail_gain / expected_tail_loss)


def calculate_modified_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Modified Sharpe Ratio (accounts for skewness and kurtosis)
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Modified Sharpe Ratio
    """
    if len(returns) < 4:
        return 0.0
        
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
        
    # Calculate traditional Sharpe
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(excess_returns)
    kurtosis = stats.kurtosis(excess_returns)
    
    # Modified Sharpe adjustment factor
    adjustment = 1 + (skewness / 6) * sharpe + ((kurtosis - 3) / 24) * (sharpe ** 2)
    
    modified_sharpe = sharpe * adjustment
    
    return modified_sharpe


def calculate_all_metrics(
    equity_curve: np.ndarray,
    returns: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate all performance metrics
    
    Args:
        equity_curve: Array of portfolio values
        returns: Optional array of returns (calculated if not provided)
        benchmark_returns: Optional benchmark returns for information ratio
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        PerformanceMetrics object
    """
    # Calculate returns if not provided
    if returns is None:
        returns = calculate_returns(equity_curve)
        
    # Basic metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0.0
    volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0.0
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(returns, equity_curve, periods_per_year)
    
    # Win/loss metrics
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
    avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0.0
    
    # Information ratio if benchmark provided
    info_ratio = None
    jensens_alpha = None
    treynor_ratio = None
    beta = None
    
    if benchmark_returns is not None:
        info_ratio = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        beta = calculate_beta(returns, benchmark_returns)
        
        if beta is not None:
            jensens_alpha = calculate_jensens_alpha(returns, benchmark_returns, beta, risk_free_rate, periods_per_year)
            treynor_ratio = calculate_treynor_ratio(returns, beta, risk_free_rate, periods_per_year)
    
    # Advanced metrics
    omega_ratio = calculate_omega_ratio(returns)
    skewness = calculate_skewness(returns)
    kurtosis = calculate_kurtosis(returns)
    tail_ratio = calculate_tail_ratio(returns)
    ulcer_index = calculate_ulcer_index(equity_curve, periods_per_year)
    burke_ratio = calculate_burke_ratio(returns, equity_curve, periods_per_year)
    sterling_ratio = calculate_sterling_ratio(returns, equity_curve, periods_per_year)
    
    # Additional advanced metrics
    pain_index = calculate_pain_index(equity_curve)
    gain_to_pain_ratio = calculate_gain_to_pain_ratio(returns, equity_curve, periods_per_year)
    martin_ratio = calculate_martin_ratio(returns, equity_curve, periods_per_year)
    conditional_sharpe_ratio = calculate_conditional_sharpe_ratio(returns, 0.95, risk_free_rate, periods_per_year)
    rachev_ratio = calculate_rachev_ratio(returns)
    modified_sharpe_ratio = calculate_modified_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        
    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=len(returns),
        volatility=volatility,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        information_ratio=info_ratio,
        jensens_alpha=jensens_alpha,
        treynor_ratio=treynor_ratio,
        omega_ratio=omega_ratio,
        skewness=skewness,
        kurtosis=kurtosis,
        tail_ratio=tail_ratio,
        ulcer_index=ulcer_index,
        burke_ratio=burke_ratio,
        sterling_ratio=sterling_ratio,
        pain_index=pain_index,
        gain_to_pain_ratio=gain_to_pain_ratio,
        martin_ratio=martin_ratio,
        conditional_sharpe_ratio=conditional_sharpe_ratio,
        rachev_ratio=rachev_ratio,
        modified_sharpe_ratio=modified_sharpe_ratio
    )


def compare_performance(
    metrics1: PerformanceMetrics,
    metrics2: PerformanceMetrics,
    labels: Tuple[str, str] = ("Strategy 1", "Strategy 2")
) -> pd.DataFrame:
    """
    Compare two sets of performance metrics
    
    Args:
        metrics1: First set of metrics
        metrics2: Second set of metrics
        labels: Labels for the two strategies
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame({
        labels[0]: metrics1.to_dict(),
        labels[1]: metrics2.to_dict()
    }).T
    
    # Add improvement column
    improvement = {}
    for metric in metrics1.to_dict():
        val1 = getattr(metrics1, metric)
        val2 = getattr(metrics2, metric)
        
        if val1 is not None and val2 is not None and val1 != 0:
            if metric == 'max_drawdown':  # Lower is better
                improvement[metric] = (val1 - val2) / abs(val1)
            else:  # Higher is better
                improvement[metric] = (val2 - val1) / abs(val1)
        else:
            improvement[metric] = 0.0
            
    df['Improvement %'] = pd.Series(improvement).values * 100
    
    return df